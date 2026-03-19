from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from areal.utils import logging

logger = logging.getLogger("Semantic KV")


def _assign_clusters(
    token_features: torch.Tensor,
    selected_features: torch.Tensor,
) -> torch.Tensor:
    if selected_features.numel() == 0:
        return torch.zeros(
            token_features.shape[:2], dtype=torch.long, device=token_features.device
        )

    distances = torch.cdist(token_features, selected_features, p=2)
    return distances.argmin(dim=-1)


def _greedy_select_batch(
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
    chosen = []

    if current_retained.numel() == 0:
        seed_idx = candidate_scores.argmax()
        chosen.append(seed_idx.item())
        remaining_mask[seed_idx] = False
        current_retained = candidate_features[seed_idx : seed_idx + 1]

    target_count = min(num_select, candidate_features.shape[0])
    while len(chosen) < target_count:
        remaining_idx = remaining_mask.nonzero(as_tuple=False).flatten()
        if remaining_idx.numel() == 0:
            break

        remaining_feats = candidate_features[remaining_idx]
        min_distance = torch.cdist(remaining_feats, current_retained, p=2).min(
            dim=-1
        ).values
        joint_score = candidate_scores[remaining_idx] * min_distance
        best_idx = remaining_idx[joint_score.argmax()]

        chosen.append(best_idx.item())
        remaining_mask[best_idx] = False
        current_retained = torch.cat(
            [current_retained, candidate_features[best_idx : best_idx + 1]], dim=0
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
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.low_rank_dim = low_rank_dim
        self.budget_ratio = budget_ratio
        self.sink_window_size = sink_window_size
        self.recent_window_size = recent_window_size

        self.weight = nn.Parameter(
            torch.empty(low_rank_dim, head_dim, dtype=params_dtype)
        )
        nn.init.orthogonal_(self.weight)

        self.last_training_state = None

    def capture_training_state(
        self,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        token_keys = key_states.mean(dim=1).float()
        projected_tokens = F.linear(token_keys, self.weight.float())

        # TODO: replace this proxy salience with cumulative attention importance from
        # the actual compressed rollout/cache path once semantic-KV forward is wired in.
        importance = token_keys.norm(dim=-1)

        batch_size, seq_len, _ = projected_tokens.shape
        selected_indices = []
        selected_counts = []

        for batch_idx in range(batch_size):
            valid_len = seq_len
            if attention_mask is not None and attention_mask.ndim == 2:
                valid_len = int(attention_mask[batch_idx].sum().item())
                valid_len = max(valid_len, 1)

            budget_tokens = min(
                int(valid_len * self.budget_ratio)
                + self.sink_window_size
                + self.recent_window_size,
                valid_len,
            )
            keep_sink = min(self.sink_window_size, budget_tokens, valid_len)
            remaining_after_sink = max(budget_tokens - keep_sink, 0)
            keep_recent = min(
                self.recent_window_size, remaining_after_sink, valid_len - keep_sink
            )
            middle_end = valid_len - keep_recent
            num_select = max(budget_tokens - keep_sink - keep_recent, 0)

            protected_parts = []
            if keep_sink > 0:
                protected_parts.append(
                    torch.arange(
                        keep_sink,
                        device=projected_tokens.device,
                        dtype=torch.long,
                    )
                )
            if keep_recent > 0:
                protected_parts.append(
                    torch.arange(
                        middle_end,
                        valid_len,
                        device=projected_tokens.device,
                        dtype=torch.long,
                    )
                )

            protected_indices = (
                torch.cat(protected_parts, dim=0)
                if protected_parts
                else torch.empty(0, device=projected_tokens.device, dtype=torch.long)
            )
            protected_features = (
                projected_tokens[batch_idx, protected_indices]
                if protected_indices.numel() > 0
                else None
            )

            candidate_indices = torch.arange(
                keep_sink,
                middle_end,
                device=projected_tokens.device,
                dtype=torch.long,
            )
            local_selected = _greedy_select_batch(
                candidate_features=projected_tokens[batch_idx, candidate_indices],
                candidate_scores=importance[batch_idx, candidate_indices],
                num_select=num_select,
                retained_features=protected_features,
            )
            if local_selected.numel() > 0:
                local_selected = candidate_indices[local_selected]

            merged = (
                torch.cat([protected_indices, local_selected], dim=0)
                if protected_indices.numel() > 0
                else local_selected
            )
            merged = merged.sort().values
            selected_indices.append(merged)
            selected_counts.append(int(merged.numel()))

        max_selected = max(selected_counts) if selected_counts else 0
        padded_selected = torch.zeros(
            batch_size, max_selected, dtype=torch.long, device=projected_tokens.device
        )
        for batch_idx, indices in enumerate(selected_indices):
            if indices.numel() > 0:
                padded_selected[batch_idx, : indices.numel()] = indices

        if max_selected > 0:
            gather_idx = padded_selected.unsqueeze(-1).expand(
                -1, -1, projected_tokens.shape[-1]
            )
            selected_features = projected_tokens.gather(1, gather_idx)
        else:
            selected_features = projected_tokens.new_empty(
                batch_size, 0, projected_tokens.shape[-1]
            )

        cluster_assignment = _assign_clusters(projected_tokens, selected_features)
        self.last_training_state = {
            "projected_tokens": projected_tokens,
            "importance": importance,
            "selected_indices": padded_selected,
            "selected_counts": torch.tensor(
                selected_counts, device=projected_tokens.device, dtype=torch.long
            ),
            "cluster_assignment": cluster_assignment,
            "attention_mask": attention_mask,
        }

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, low_rank_dim={self.low_rank_dim}, "
            f"budget_ratio={self.budget_ratio}, sink_window_size={self.sink_window_size}, "
            f"recent_window_size={self.recent_window_size}"
        )


def llama_semantic_kv_passthrough_forward(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    attention_mask = kwargs.get("attention_mask", None)

    if hidden_states is not None and hasattr(self, "semantic_kv"):
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        self.semantic_kv.capture_training_state(
            key_states=key_states,
            attention_mask=attention_mask,
        )

    # TODO: replace this passthrough with the actual semantic-KV compressed attention
    # path so GRPO gradients can flow into the low-rank projector via policy loss.
    return self._semantic_kv_original_forward(*args, **kwargs)


def qwen3_semantic_kv_passthrough_forward(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    attention_mask = kwargs.get("attention_mask", None)

    if hidden_states is not None and hasattr(self, "semantic_kv"):
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
            1, 2
        )
        self.semantic_kv.capture_training_state(
            key_states=key_states,
            attention_mask=attention_mask,
        )

    # TODO: replace this passthrough with the actual semantic-KV compressed attention
    # path so GRPO gradients can flow into the low-rank projector via policy loss.
    return self._semantic_kv_original_forward(*args, **kwargs)


def enable_semantic_kv_training(
    model,
    forward_fn,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    ulysses_sp_size: int = 1,
):
    if ulysses_sp_size > 1:
        logger.warning(
            "Semantic-KV training currently records packed sequence statistics only. "
            "Per-request cache-aware Ulysses integration is TODO."
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
                    params_dtype=dtype,
                ),
            )


def enable_llama_semantic_kv_training(
    model: LlamaForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    ulysses_sp_size: int = 1,
):
    enable_semantic_kv_training(
        model=model,
        forward_fn=llama_semantic_kv_passthrough_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        ulysses_sp_size=ulysses_sp_size,
    )


def enable_qwen2_semantic_kv_training(
    model: Qwen2ForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    ulysses_sp_size: int = 1,
):
    enable_semantic_kv_training(
        model=model,
        forward_fn=llama_semantic_kv_passthrough_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        ulysses_sp_size=ulysses_sp_size,
    )


def enable_qwen3_semantic_kv_training(
    model: Qwen3ForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    ulysses_sp_size: int = 1,
):
    enable_semantic_kv_training(
        model=model,
        forward_fn=qwen3_semantic_kv_passthrough_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        ulysses_sp_size=ulysses_sp_size,
    )
