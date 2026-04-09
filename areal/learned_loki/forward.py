from __future__ import annotations

import math
import types
from typing import Optional

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

logger = logging.getLogger("Learned Loki")


def _init_orthogonal_param(param: nn.Parameter):
    with torch.no_grad():
        init_weight = torch.empty(
            param.shape,
            device=param.device,
            dtype=torch.float32,
        )
        nn.init.orthogonal_(init_weight)
        param.copy_(init_weight.to(dtype=param.dtype))


def _repeat_kv_tokens(
    tokens: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    if num_key_value_groups == 1:
        return tokens
    return tokens.repeat_interleave(num_key_value_groups, dim=1)


def _split_windows(
    total_tokens: int,
    sink_window_size: int,
    recent_window_size: int,
) -> tuple[int, int]:
    keep_sink = min(sink_window_size, total_tokens)
    keep_recent = min(recent_window_size, max(total_tokens - keep_sink, 0))
    middle_end = total_tokens - keep_recent
    return keep_sink, middle_end


def _compute_projected_scores(
    query_state: torch.Tensor,
    projected_keys: torch.Tensor,
    projector_weight: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    projected_query = F.linear(query_state.float(), projector_weight.float())
    expanded_projected_keys = _repeat_kv_tokens(
        projected_keys,
        num_key_value_groups,
    ).transpose(0, 1)
    approx_scores = (
        torch.einsum(
            "hr,hsr->hs",
            projected_query.float(),
            expanded_projected_keys.float(),
        )
        / math.sqrt(projector_weight.shape[0])
    )
    return approx_scores.mean(dim=0)


def _learned_loki_attention_step(
    query_state: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gates: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    expanded_keys = _repeat_kv_tokens(key_states, num_key_value_groups).transpose(0, 1)
    expanded_values = _repeat_kv_tokens(value_states, num_key_value_groups).transpose(
        0, 1
    )

    attn_logits = (
        torch.einsum("hd,hsd->hs", query_state.float(), expanded_keys.float()) * scaling
    )
    attn_logits = attn_logits + torch.log(gates.clamp_min(1e-6)).unsqueeze(0)
    attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32)
    attn_probs_for_output = attn_probs.to(expanded_values.dtype)
    if training and dropout_p > 0.0:
        attn_probs_for_output = F.dropout(
            attn_probs_for_output,
            p=dropout_p,
            training=True,
        )

    output = torch.einsum("hs,hsd->hd", attn_probs_for_output, expanded_values)
    return output.to(query_state.dtype)


class LearnedLokiProjectionLayer(nn.Module):
    def __init__(
        self,
        head_dim: int,
        low_rank_dim: int,
        budget_ratio: float,
        sink_window_size: int,
        recent_window_size: int,
        gate_temperature: float,
        threshold_init: float,
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
        self.threshold = nn.Parameter(
            torch.full((1,), threshold_init, dtype=params_dtype or torch.float32)
        )
        self.gate_temperature_param = nn.Parameter(
            torch.full((1,), gate_temperature, dtype=params_dtype or torch.float32),
            requires_grad=False,
        )
        _init_orthogonal_param(self.weight)

        self.last_training_state = None

    def reset_training_state(self):
        self.last_training_state = {
            "middle_gate_sum": None,
            "middle_gate_count": 0,
        }

    def record_middle_gates(self, middle_gates: Optional[torch.Tensor]):
        if middle_gates is None or middle_gates.numel() == 0:
            return
        if self.last_training_state is None:
            self.reset_training_state()
        gate_sum = middle_gates.sum()
        if self.last_training_state["middle_gate_sum"] is None:
            self.last_training_state["middle_gate_sum"] = gate_sum
        else:
            self.last_training_state["middle_gate_sum"] = (
                self.last_training_state["middle_gate_sum"] + gate_sum
            )
        self.last_training_state["middle_gate_count"] += int(middle_gates.numel())

    def project_keys(self, key_states: torch.Tensor) -> torch.Tensor:
        return F.linear(key_states.float(), self.weight.float())

    @property
    def gate_temperature(self) -> float:
        return float(self.gate_temperature_param.detach().item())

    @gate_temperature.setter
    def gate_temperature(self, value: float):
        with torch.no_grad():
            self.gate_temperature_param.fill_(float(value))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, low_rank_dim={self.low_rank_dim}, "
            f"budget_ratio={self.budget_ratio}, "
            f"sink_window_size={self.sink_window_size}, "
            f"recent_window_size={self.recent_window_size}, "
            f"gate_temperature={self.gate_temperature}"
        )


def _run_learned_loki_sequence(
    learned_loki: LearnedLokiProjectionLayer,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
    dropout_p: float,
    record_training_state: bool,
    prefix_keys: Optional[torch.Tensor] = None,
    prefix_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_len = query_states.shape[0]
    output = value_states.new_zeros((seq_len, query_states.shape[1], value_states.shape[-1]))

    if prefix_keys is None or prefix_values is None:
        live_keys = key_states.new_empty((0, key_states.shape[1], key_states.shape[2]))
        live_values = value_states.new_empty(
            (0, value_states.shape[1], value_states.shape[2])
        )
        live_projected_keys = key_states.new_empty(
            (0, key_states.shape[1], learned_loki.low_rank_dim),
            dtype=torch.float32,
        )
    else:
        live_keys = prefix_keys
        live_values = prefix_values
        live_projected_keys = learned_loki.project_keys(prefix_keys)

    for token_idx in range(seq_len):
        live_keys = torch.cat([live_keys, key_states[token_idx : token_idx + 1]], dim=0)
        live_values = torch.cat(
            [live_values, value_states[token_idx : token_idx + 1]],
            dim=0,
        )
        live_projected_keys = torch.cat(
            [
                live_projected_keys,
                learned_loki.project_keys(key_states[token_idx : token_idx + 1]),
            ],
            dim=0,
        )

        live_len = live_keys.shape[0]
        keep_sink, middle_end = _split_windows(
            live_len,
            sink_window_size=learned_loki.sink_window_size,
            recent_window_size=learned_loki.recent_window_size,
        )
        gates = live_keys.new_ones((live_len,), dtype=torch.float32)
        middle_gates = None
        if middle_end > keep_sink:
            approx_scores = _compute_projected_scores(
                query_state=query_states[token_idx],
                projected_keys=live_projected_keys,
                projector_weight=learned_loki.weight,
                num_key_value_groups=num_key_value_groups,
            )
            middle_gates = torch.sigmoid(
                (approx_scores[keep_sink:middle_end] - learned_loki.threshold.float())
                / max(learned_loki.gate_temperature, 1e-6)
            )
            gates[keep_sink:middle_end] = middle_gates
            if record_training_state:
                learned_loki.record_middle_gates(middle_gates)

        output[token_idx] = _learned_loki_attention_step(
            query_state=query_states[token_idx],
            key_states=live_keys,
            value_states=live_values,
            gates=gates,
            num_key_value_groups=num_key_value_groups,
            scaling=scaling,
            dropout_p=dropout_p,
            training=record_training_state,
        )

    return output


def _learned_loki_forward_impl(
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
    if not hasattr(self, "learned_loki"):
        raise ValueError("learned_loki module is not attached to the attention layer")
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
                "Learned-Loki cached forward does not support packed cu_seqlens inputs."
            )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )

    learned_loki = self.learned_loki
    learned_loki.reset_training_state()
    record_training_state = self.training and torch.is_grad_enabled()
    dropout_p = 0.0 if not self.training else self.attention_dropout

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
            seq_output = _run_learned_loki_sequence(
                learned_loki=learned_loki,
                query_states=query_states[0, :, start:end, :].transpose(0, 1),
                key_states=key_states[0, :, start:end, :].transpose(0, 1),
                value_states=value_states[0, :, start:end, :].transpose(0, 1),
                num_key_value_groups=self.num_key_value_groups,
                scaling=self.scaling,
                dropout_p=dropout_p,
                record_training_state=record_training_state,
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
            past_len = (
                max(key_states.shape[2] - valid_len, 0) if past_key_value is not None else 0
            )

            prefix_keys = None
            prefix_values = None
            if past_len > 0:
                prefix_keys = key_states[batch_idx, :, :past_len, :].transpose(0, 1)
                prefix_values = value_states[batch_idx, :, :past_len, :].transpose(0, 1)

            current_query = query_states[batch_idx, :, :valid_len, :].transpose(0, 1)
            current_keys = key_states[
                batch_idx, :, past_len : past_len + valid_len, :
            ].transpose(0, 1)
            current_values = value_states[
                batch_idx, :, past_len : past_len + valid_len, :
            ].transpose(0, 1)
            seq_output = _run_learned_loki_sequence(
                learned_loki=learned_loki,
                query_states=current_query,
                key_states=current_keys,
                value_states=current_values,
                num_key_value_groups=self.num_key_value_groups,
                scaling=self.scaling,
                dropout_p=dropout_p,
                record_training_state=record_training_state,
                prefix_keys=prefix_keys,
                prefix_values=prefix_values,
            )
            attn_output[batch_idx, :valid_len] = seq_output

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def llama_learned_loki_forward(
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
    return _learned_loki_forward_impl(
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


def qwen3_learned_loki_forward(
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
    return _learned_loki_forward_impl(
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


def enable_learned_loki_training(
    model,
    forward_fn,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    gate_temperature: float,
    threshold_init: float,
    ulysses_sp_size: int = 1,
):
    if ulysses_sp_size > 1:
        logger.warning(
            "Learned-Loki training uses an explicit packed-sequence exact attention "
            "loop. Sequence-parallel integration is still unsupported."
        )

    dtype = next(model.parameters()).dtype

    for layer in model.model.layers:
        module = layer.self_attn
        if not hasattr(module, "_learned_loki_original_forward"):
            module._learned_loki_original_forward = module.forward
        module.forward = types.MethodType(forward_fn, module)

        if "learned_loki" not in module._modules:
            module.add_module(
                "learned_loki",
                LearnedLokiProjectionLayer(
                    head_dim=module.head_dim,
                    low_rank_dim=low_rank_dim,
                    budget_ratio=budget_ratio,
                    sink_window_size=sink_window_size,
                    recent_window_size=recent_window_size,
                    gate_temperature=gate_temperature,
                    threshold_init=threshold_init,
                    params_dtype=dtype,
                ),
            )


def enable_llama_learned_loki_training(
    model: LlamaForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    gate_temperature: float,
    threshold_init: float,
    ulysses_sp_size: int = 1,
):
    enable_learned_loki_training(
        model=model,
        forward_fn=llama_learned_loki_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        gate_temperature=gate_temperature,
        threshold_init=threshold_init,
        ulysses_sp_size=ulysses_sp_size,
    )


def enable_qwen2_learned_loki_training(
    model: Qwen2ForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    gate_temperature: float,
    threshold_init: float,
    ulysses_sp_size: int = 1,
):
    enable_learned_loki_training(
        model=model,
        forward_fn=llama_learned_loki_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        gate_temperature=gate_temperature,
        threshold_init=threshold_init,
        ulysses_sp_size=ulysses_sp_size,
    )


def enable_qwen3_learned_loki_training(
    model: Qwen3ForCausalLM,
    low_rank_dim: int,
    budget_ratio: float,
    sink_window_size: int,
    recent_window_size: int,
    gate_temperature: float,
    threshold_init: float,
    ulysses_sp_size: int = 1,
):
    enable_learned_loki_training(
        model=model,
        forward_fn=qwen3_learned_loki_forward,
        low_rank_dim=low_rank_dim,
        budget_ratio=budget_ratio,
        sink_window_size=sink_window_size,
        recent_window_size=recent_window_size,
        gate_temperature=gate_temperature,
        threshold_init=threshold_init,
        ulysses_sp_size=ulysses_sp_size,
    )
