from __future__ import annotations

import types
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from areal.utils import logging

logger = logging.getLogger("Semantic KV")


def _compute_budget(
    total_tokens: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
) -> int:
    return min(
        int(total_tokens * budget_ratio) + sink_window_size + recent_window_size,
        total_tokens,
    )


def _repeat_kv_tokens(
    tokens: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    if num_key_value_groups == 1:
        return tokens
    return tokens.repeat_interleave(num_key_value_groups, dim=1)


def _semantic_attention_step(
    query_state: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
    dropout_p: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_keys = _repeat_kv_tokens(key_states, num_key_value_groups).transpose(0, 1)
    expanded_values = _repeat_kv_tokens(value_states, num_key_value_groups).transpose(
        0, 1
    )

    attn_logits = (
        torch.einsum("hd,hsd->hs", query_state.float(), expanded_keys.float())
        * scaling
    )
    attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32)
    attn_probs_for_output = attn_probs.to(expanded_values.dtype)
    if training and dropout_p > 0.0:
        attn_probs_for_output = F.dropout(
            attn_probs_for_output,
            p=dropout_p,
            training=True,
        )

    output = torch.einsum("hs,hsd->hd", attn_probs_for_output, expanded_values)
    return output.to(query_state.dtype), attn_probs


def _hard_greedy_select(
    candidate_features: torch.Tensor,
    candidate_scores: torch.Tensor,
    num_select: int,
    retained_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if num_select <= 0 or candidate_features.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=candidate_features.device)

    current_retained = (
        retained_features
        if retained_features is not None
        else candidate_features.new_empty((0, candidate_features.shape[-1]))
    )
    remaining_mask = torch.ones(
        candidate_features.shape[0], dtype=torch.bool, device=candidate_features.device
    )
    chosen: list[int] = []

    target_count = min(num_select, candidate_features.shape[0])
    while len(chosen) < target_count:
        remaining_idx = remaining_mask.nonzero(as_tuple=False).flatten()
        if remaining_idx.numel() == 0:
            break

        remaining_feats = candidate_features[remaining_idx]
        remaining_scores = candidate_scores[remaining_idx]
        if current_retained.numel() == 0:
            logits = torch.log(remaining_scores.clamp_min(1e-6))
        else:
            min_distance = torch.cdist(
                remaining_feats.float(),
                current_retained.float(),
                p=2,
            ).min(dim=-1).values
            logits = torch.log(remaining_scores.clamp_min(1e-6)) + torch.log(
                min_distance.clamp_min(1e-6)
            )
        best_idx = remaining_idx[logits.argmax()]

        chosen.append(best_idx.item())
        remaining_mask[best_idx] = False
        current_retained = torch.cat(
            [current_retained, candidate_features[best_idx : best_idx + 1]],
            dim=0,
        )

    return torch.tensor(chosen, dtype=torch.long, device=candidate_features.device)


class SemanticKVProjectionLayer(nn.Module):
    def __init__(
        self,
        head_dim: int,
        low_rank_dim: int,
        budget_ratio: float,
        sink_window_size: int,
        recent_window_size: int,
        selector_temperature: float,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.low_rank_dim = low_rank_dim
        self.budget_ratio = budget_ratio
        self.sink_window_size = sink_window_size
        self.recent_window_size = recent_window_size
        self.selector_temperature = selector_temperature

        self.weight = nn.Parameter(
            torch.empty(low_rank_dim, head_dim, dtype=params_dtype)
        )
        nn.init.orthogonal_(self.weight)

        self.last_training_state = None
        self.runtime_generation_state = []

    def reset_training_state(self):
        self.last_training_state = {"compression_events": []}

    def reset_runtime_state(self):
        self.runtime_generation_state = []

    def get_runtime_state(self, batch_idx: int):
        if batch_idx >= len(self.runtime_generation_state):
            return None
        return self.runtime_generation_state[batch_idx]

    def set_runtime_state(
        self,
        batch_idx: int,
        state: dict[str, Union[torch.Tensor, int]],
    ):
        while len(self.runtime_generation_state) <= batch_idx:
            self.runtime_generation_state.append(None)
        self.runtime_generation_state[batch_idx] = state

    def trim_runtime_state(self, batch_size: int):
        if len(self.runtime_generation_state) > batch_size:
            self.runtime_generation_state = self.runtime_generation_state[:batch_size]

    def record_compression_event(
        self,
        projected_tokens: torch.Tensor,
        selected_features: torch.Tensor,
        importance: torch.Tensor,
        selected_indices: torch.Tensor,
    ):
        if self.last_training_state is None:
            self.reset_training_state()
        self.last_training_state["compression_events"].append(
            {
                "projected_tokens": projected_tokens,
                "selected_features": selected_features,
                "importance": importance,
                "selected_indices": selected_indices,
            }
        )

    def project_token_keys(self, key_states: torch.Tensor) -> torch.Tensor:
        token_keys = key_states.mean(dim=1).float()
        return F.linear(token_keys, self.weight.float())

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, low_rank_dim={self.low_rank_dim}, "
            f"budget_ratio={self.budget_ratio}, "
            f"sink_window_size={self.sink_window_size}, "
            f"recent_window_size={self.recent_window_size}, "
            f"selector_temperature={self.selector_temperature}"
        )


def _compress_live_cache(
    semantic_kv: SemanticKVProjectionLayer,
    live_keys: torch.Tensor,
    live_values: torch.Tensor,
    live_projected: torch.Tensor,
    live_importance: torch.Tensor,
    total_tokens_seen: int,
    use_straight_through: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    live_len = live_keys.shape[0]
    cache_budget = _compute_budget(
        total_tokens=total_tokens_seen,
        budget_ratio=semantic_kv.budget_ratio,
        sink_window_size=semantic_kv.sink_window_size,
        recent_window_size=semantic_kv.recent_window_size,
    )
    if live_len <= cache_budget:
        return live_keys, live_values, live_projected, live_importance

    keep_sink = min(semantic_kv.sink_window_size, cache_budget, live_len)
    remaining_after_sink = max(cache_budget - keep_sink, 0)
    keep_recent = min(
        semantic_kv.recent_window_size,
        remaining_after_sink,
        live_len - keep_sink,
    )
    middle_end = live_len - keep_recent
    candidate_indices = torch.arange(
        keep_sink,
        middle_end,
        device=live_keys.device,
        dtype=torch.long,
    )
    num_select = max(cache_budget - keep_sink - keep_recent, 0)

    protected_parts = []
    if keep_sink > 0:
        protected_parts.append(
            torch.arange(keep_sink, device=live_keys.device, dtype=torch.long)
        )
    if keep_recent > 0:
        protected_parts.append(
            torch.arange(middle_end, live_len, device=live_keys.device, dtype=torch.long)
        )
    protected_indices = (
        torch.cat(protected_parts, dim=0)
        if protected_parts
        else torch.empty(0, device=live_keys.device, dtype=torch.long)
    )
    protected_features = (
        live_projected[protected_indices] if protected_indices.numel() > 0 else None
    )

    candidate_features = live_projected[candidate_indices]
    candidate_scores = live_importance[candidate_indices]

    if use_straight_through and num_select > 0 and candidate_indices.numel() > 0:
        remaining_mask = torch.ones(
            candidate_indices.shape[0],
            dtype=torch.bool,
            device=live_keys.device,
        )
        current_retained = (
            protected_features
            if protected_features is not None
            else candidate_features.new_empty((0, candidate_features.shape[-1]))
        )

        selected_keys = []
        selected_values = []
        selected_projected = []
        selected_importance = []
        selected_indices = []

        selector_temperature = max(semantic_kv.selector_temperature, 1e-6)
        for _ in range(min(num_select, candidate_indices.numel())):
            remaining_idx = remaining_mask.nonzero(as_tuple=False).flatten()
            if remaining_idx.numel() == 0:
                break

            remaining_feats = candidate_features[remaining_idx]
            remaining_scores = candidate_scores[remaining_idx]
            if current_retained.numel() == 0:
                logits = torch.log(remaining_scores.clamp_min(1e-6))
            else:
                min_distance = torch.cdist(
                    remaining_feats.float(),
                    current_retained.float(),
                    p=2,
                ).min(dim=-1).values
                logits = torch.log(remaining_scores.clamp_min(1e-6)) + torch.log(
                    min_distance.clamp_min(1e-6)
                )

            hard_idx_in_remaining = logits.argmax()
            hard_local_idx = remaining_idx[hard_idx_in_remaining]

            soft_probs = torch.softmax(logits / selector_temperature, dim=0)
            hard_probs = F.one_hot(
                hard_idx_in_remaining,
                num_classes=remaining_idx.numel(),
            ).to(soft_probs.dtype)
            st_weights = hard_probs + soft_probs - soft_probs.detach()

            selected_keys.append(
                torch.einsum(
                    "n,nkh->kh",
                    st_weights.to(live_keys.dtype),
                    live_keys[candidate_indices[remaining_idx]],
                )
            )
            selected_values.append(
                torch.einsum(
                    "n,nkh->kh",
                    st_weights.to(live_values.dtype),
                    live_values[candidate_indices[remaining_idx]],
                )
            )
            selected_projected_feat = torch.einsum(
                "n,nr->r",
                st_weights.to(live_projected.dtype),
                candidate_features[remaining_idx],
            )
            selected_projected.append(selected_projected_feat)
            selected_importance.append(
                torch.sum(st_weights.to(live_importance.dtype) * remaining_scores)
            )
            selected_indices.append(candidate_indices[hard_local_idx])

            current_retained = torch.cat(
                [current_retained, selected_projected_feat.unsqueeze(0)],
                dim=0,
            )
            remaining_mask[hard_local_idx] = False

        if selected_indices:
            selected_indices = torch.stack(selected_indices)
            selected_keys = torch.stack(selected_keys)
            selected_values = torch.stack(selected_values)
            selected_projected = torch.stack(selected_projected)
            selected_importance = torch.stack(selected_importance)

            sort_order = selected_indices.argsort()
            selected_indices = selected_indices[sort_order]
            selected_keys = selected_keys[sort_order]
            selected_values = selected_values[sort_order]
            selected_projected = selected_projected[sort_order]
            selected_importance = selected_importance[sort_order]
        else:
            selected_indices = torch.empty(
                0, dtype=torch.long, device=live_keys.device
            )
            selected_keys = live_keys.new_empty(
                (0, live_keys.shape[1], live_keys.shape[2])
            )
            selected_values = live_values.new_empty(
                (0, live_values.shape[1], live_values.shape[2])
            )
            selected_projected = live_projected.new_empty(
                (0, live_projected.shape[-1])
            )
            selected_importance = live_importance.new_empty((0,))

        selected_features = torch.cat(
            [
                live_projected[:keep_sink],
                selected_projected,
                live_projected[middle_end:],
            ],
            dim=0,
        )
        hard_keep_indices = torch.cat(
            [
                torch.arange(keep_sink, device=live_keys.device, dtype=torch.long),
                selected_indices,
                torch.arange(
                    middle_end,
                    live_len,
                    device=live_keys.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        semantic_kv.record_compression_event(
            projected_tokens=live_projected,
            selected_features=selected_features,
            importance=live_importance,
            selected_indices=hard_keep_indices,
        )

        live_keys = torch.cat(
            [live_keys[:keep_sink], selected_keys, live_keys[middle_end:]],
            dim=0,
        )
        live_values = torch.cat(
            [live_values[:keep_sink], selected_values, live_values[middle_end:]],
            dim=0,
        )
        live_projected = selected_features
        live_importance = torch.cat(
            [
                live_importance[:keep_sink],
                selected_importance,
                live_importance[middle_end:],
            ],
            dim=0,
        )
        return live_keys, live_values, live_projected, live_importance

    selected_middle = _hard_greedy_select(
        candidate_features=candidate_features,
        candidate_scores=candidate_scores,
        num_select=num_select,
        retained_features=protected_features,
    )
    if selected_middle.numel() > 0:
        selected_middle = candidate_indices[selected_middle]
    keep_idx = torch.cat([protected_indices, selected_middle], dim=0).sort().values
    selected_features = live_projected[keep_idx]

    semantic_kv.record_compression_event(
        projected_tokens=live_projected,
        selected_features=selected_features,
        importance=live_importance,
        selected_indices=keep_idx,
    )

    live_keys = live_keys[keep_idx]
    live_values = live_values[keep_idx]
    live_projected = live_projected[keep_idx]
    live_importance = live_importance[keep_idx]
    return live_keys, live_values, live_projected, live_importance


def _build_runtime_state_from_prefix(
    semantic_kv: SemanticKVProjectionLayer,
    prefix_keys: torch.Tensor,
    prefix_values: torch.Tensor,
    logical_len: int,
) -> dict[str, Union[torch.Tensor, int]]:
    if prefix_keys.shape[0] == 0:
        return {
            "live_keys": prefix_keys.new_empty((0, prefix_keys.shape[1], prefix_keys.shape[2])),
            "live_values": prefix_values.new_empty(
                (0, prefix_values.shape[1], prefix_values.shape[2])
            ),
            "live_projected": prefix_keys.new_empty(
                (0, semantic_kv.low_rank_dim),
                dtype=torch.float32,
            ),
            "live_importance": prefix_keys.new_empty((0,), dtype=torch.float32),
            "logical_len": logical_len,
        }

    live_projected = semantic_kv.project_token_keys(prefix_keys)
    live_importance = torch.ones(
        (prefix_keys.shape[0],),
        dtype=torch.float32,
        device=prefix_keys.device,
    )
    live_keys, live_values, live_projected, live_importance = _compress_live_cache(
        semantic_kv=semantic_kv,
        live_keys=prefix_keys,
        live_values=prefix_values,
        live_projected=live_projected,
        live_importance=live_importance,
        total_tokens_seen=logical_len,
        use_straight_through=False,
    )
    return {
        "live_keys": live_keys,
        "live_values": live_values,
        "live_projected": live_projected,
        "live_importance": live_importance,
        "logical_len": logical_len,
    }


def _detach_runtime_state(
    state: dict[str, Union[torch.Tensor, int]],
) -> dict[str, Union[torch.Tensor, int]]:
    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def _run_semantic_kv_sequence(
    semantic_kv: SemanticKVProjectionLayer,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
    dropout_p: float,
    use_straight_through: bool,
    initial_state: Optional[dict[str, Union[torch.Tensor, int]]] = None,
    return_final_state: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, Union[torch.Tensor, int]]]]:
    seq_len = query_states.shape[0]
    output = value_states.new_zeros((seq_len, query_states.shape[1], value_states.shape[-1]))

    if initial_state is None:
        live_keys = key_states.new_empty((0, key_states.shape[1], key_states.shape[2]))
        live_values = value_states.new_empty(
            (0, value_states.shape[1], value_states.shape[2])
        )
        live_projected = key_states.new_empty(
            (0, semantic_kv.low_rank_dim),
            dtype=torch.float32,
        )
        live_importance = key_states.new_empty((0,), dtype=torch.float32)
        total_tokens_seen = 0
    else:
        live_keys = initial_state["live_keys"]
        live_values = initial_state["live_values"]
        live_projected = initial_state["live_projected"]
        live_importance = initial_state["live_importance"]
        total_tokens_seen = int(initial_state["logical_len"])

    for token_idx in range(seq_len):
        live_keys = torch.cat([live_keys, key_states[token_idx : token_idx + 1]], dim=0)
        live_values = torch.cat(
            [live_values, value_states[token_idx : token_idx + 1]],
            dim=0,
        )
        live_projected = torch.cat(
            [
                live_projected,
                semantic_kv.project_token_keys(key_states[token_idx : token_idx + 1]),
            ],
            dim=0,
        )
        live_importance = torch.cat(
            [
                live_importance,
                live_importance.new_zeros((1,), dtype=torch.float32),
            ],
            dim=0,
        )

        token_output, attn_probs = _semantic_attention_step(
            query_state=query_states[token_idx],
            key_states=live_keys,
            value_states=live_values,
            num_key_value_groups=num_key_value_groups,
            scaling=scaling,
            dropout_p=dropout_p,
            training=use_straight_through,
        )
        output[token_idx] = token_output
        live_importance = live_importance + attn_probs.sum(dim=0).to(live_importance.dtype)
        total_tokens_seen += 1

        live_keys, live_values, live_projected, live_importance = _compress_live_cache(
            semantic_kv=semantic_kv,
            live_keys=live_keys,
            live_values=live_values,
            live_projected=live_projected,
            live_importance=live_importance,
            total_tokens_seen=total_tokens_seen,
            use_straight_through=use_straight_through,
        )

    final_state = {
        "live_keys": live_keys,
        "live_values": live_values,
        "live_projected": live_projected,
        "live_importance": live_importance,
        "logical_len": total_tokens_seen,
    }
    if return_final_state:
        return output, final_state
    return output


def _semantic_kv_forward_impl(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    max_seqlen: Optional[int],
    past_key_value: Optional[Cache],
    cache_position: Optional[torch.LongTensor],
    use_qk_norm: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not hasattr(self, "semantic_kv"):
        raise ValueError("semantic_kv module is not attached to the attention layer")
    del max_seqlen

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)
    if use_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_value is not None:
        if cu_seqlens is not None:
            raise NotImplementedError(
                "Semantic-KV cached forward does not support packed cu_seqlens inputs."
            )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )

    semantic_kv = self.semantic_kv
    semantic_kv.reset_training_state()
    use_straight_through = self.training and torch.is_grad_enabled()
    dropout_p = 0.0 if not self.training else self.attention_dropout
    if past_key_value is None or query_states.shape[2] > 1:
        semantic_kv.reset_runtime_state()
    else:
        semantic_kv.trim_runtime_state(query_states.shape[0])

    if query_states.shape[0] == 1 and cu_seqlens is not None:
        total_tokens = query_states.shape[2]
        attn_output = value_states.new_zeros(
            (1, total_tokens, query_states.shape[1], value_states.shape[-1])
        )
        boundaries = cu_seqlens.to(dtype=torch.long)
        num_segments = boundaries.numel() - 1
        if isinstance(attention_mask, dict):
            num_segments = max(num_segments - 1, 0)

        for seq_idx in range(num_segments):
            start = int(boundaries[seq_idx].item())
            end = int(boundaries[seq_idx + 1].item())
            if end <= start:
                continue
            seq_output = _run_semantic_kv_sequence(
                semantic_kv=semantic_kv,
                query_states=query_states[0, :, start:end, :].transpose(0, 1),
                key_states=key_states[0, :, start:end, :].transpose(0, 1),
                value_states=value_states[0, :, start:end, :].transpose(0, 1),
                num_key_value_groups=self.num_key_value_groups,
                scaling=self.scaling,
                dropout_p=dropout_p,
                use_straight_through=use_straight_through,
            )
            attn_output[0, start:end] = seq_output
    else:
        batch_size, _, seq_len, _ = query_states.shape
        attn_output = value_states.new_zeros(
            (batch_size, seq_len, query_states.shape[1], value_states.shape[-1])
        )
        for batch_idx in range(batch_size):
            valid_len = seq_len
            if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
                valid_len = max(int(attention_mask[batch_idx].sum().item()), 1)
            past_len = max(key_states.shape[2] - valid_len, 0) if past_key_value is not None else 0
            initial_state = None
            if past_len > 0:
                initial_state = semantic_kv.get_runtime_state(batch_idx)
                if initial_state is None or int(initial_state["logical_len"]) != past_len:
                    initial_state = _build_runtime_state_from_prefix(
                        semantic_kv=semantic_kv,
                        prefix_keys=key_states[batch_idx, :, :past_len, :].transpose(0, 1),
                        prefix_values=value_states[
                            batch_idx, :, :past_len, :
                        ].transpose(0, 1),
                        logical_len=past_len,
                    )

            current_query = query_states[batch_idx, :, :valid_len, :].transpose(0, 1)
            current_keys = key_states[
                batch_idx, :, past_len : past_len + valid_len, :
            ].transpose(0, 1)
            current_values = value_states[
                batch_idx, :, past_len : past_len + valid_len, :
            ].transpose(0, 1)
            seq_output, final_state = _run_semantic_kv_sequence(
                semantic_kv=semantic_kv,
                query_states=current_query,
                key_states=current_keys,
                value_states=current_values,
                num_key_value_groups=self.num_key_value_groups,
                scaling=self.scaling,
                dropout_p=dropout_p,
                use_straight_through=use_straight_through,
                initial_state=initial_state,
                return_final_state=True,
            )
            attn_output[batch_idx, :valid_len] = seq_output
            if past_key_value is not None or cache_position is not None:
                semantic_kv.set_runtime_state(
                    batch_idx,
                    _detach_runtime_state(final_state),
                )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def llama_semantic_kv_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],
    cu_seq_lens_k: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    max_seqlen: Optional[int] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    del cu_seq_lens_q, cu_seq_lens_k, kwargs
    return _semantic_kv_forward_impl(
        self=self,
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        past_key_value=past_key_value,
        cache_position=cache_position,
        use_qk_norm=False,
    )


def qwen3_semantic_kv_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],
    cu_seq_lens_k: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    max_seqlen: Optional[int] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    del cu_seq_lens_q, cu_seq_lens_k, kwargs
    return _semantic_kv_forward_impl(
        self=self,
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        past_key_value=past_key_value,
        cache_position=cache_position,
        use_qk_norm=True,
    )


def enable_semantic_kv_training(
    model,
    forward_fn,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    selector_temperature: float,
    ulysses_sp_size: int = 1,
):
    if ulysses_sp_size > 1:
        logger.warning(
            "Semantic-KV training uses an explicit packed-sequence compressed "
            "attention loop. Sequence-parallel integration is still unsupported."
        )

    dtype = next(model.parameters()).dtype

    for layer in model.model.layers:
        module = layer.self_attn
        if not hasattr(module, "_semantic_kv_original_forward"):
            module._semantic_kv_original_forward = module.forward
        module.forward = types.MethodType(forward_fn, module)

        if "semantic_kv" not in module._modules:
            module.add_module(
                "semantic_kv",
                SemanticKVProjectionLayer(
                    head_dim=module.head_dim,
                    low_rank_dim=low_rank_dim,
                    budget_ratio=budget_ratio,
                    sink_window_size=sink_window_size,
                    recent_window_size=recent_window_size,
                    selector_temperature=selector_temperature,
                    params_dtype=dtype,
                ),
            )


def enable_llama_semantic_kv_training(
    model: LlamaForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    selector_temperature: float,
    ulysses_sp_size: int = 1,
):
    enable_semantic_kv_training(
        model=model,
        forward_fn=llama_semantic_kv_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        selector_temperature=selector_temperature,
        ulysses_sp_size=ulysses_sp_size,
    )


def enable_qwen2_semantic_kv_training(
    model: Qwen2ForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    selector_temperature: float,
    ulysses_sp_size: int = 1,
):
    enable_semantic_kv_training(
        model=model,
        forward_fn=llama_semantic_kv_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        selector_temperature=selector_temperature,
        ulysses_sp_size=ulysses_sp_size,
    )


def enable_qwen3_semantic_kv_training(
    model: Qwen3ForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    selector_temperature: float,
    ulysses_sp_size: int = 1,
):
    enable_semantic_kv_training(
        model=model,
        forward_fn=qwen3_semantic_kv_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        selector_temperature=selector_temperature,
        ulysses_sp_size=ulysses_sp_size,
    )
