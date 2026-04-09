from .forward import (
    enable_llama_learned_loki_training,
    enable_qwen2_learned_loki_training,
    enable_qwen3_learned_loki_training,
)
from .loss import learned_loki_sparse_loss_fn
from .utils import (
    collect_learned_loki_training_states,
    get_learned_loki_modules,
    initialize_learned_loki_from_checkpoint,
    load_learned_loki_weight,
    save_learned_loki_weight,
)


def enable_learned_loki_training(
    model,
    low_rank_dim: int,
    budget_ratio: float = 0.5,
    sink_window_size: int = 16,
    recent_window_size: int = 64,
    gate_temperature: float = 1.0,
    threshold_init: float = 0.0,
    ulysses_sp_size: int = 1,
):
    model_type = model.config.model_type.lower()
    if model_type == "llama":
        enable_llama_learned_loki_training(
            model=model,
            low_rank_dim=low_rank_dim,
            budget_ratio=budget_ratio,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
            gate_temperature=gate_temperature,
            threshold_init=threshold_init,
            ulysses_sp_size=ulysses_sp_size,
        )
    elif model_type == "qwen2":
        enable_qwen2_learned_loki_training(
            model=model,
            low_rank_dim=low_rank_dim,
            budget_ratio=budget_ratio,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
            gate_temperature=gate_temperature,
            threshold_init=threshold_init,
            ulysses_sp_size=ulysses_sp_size,
        )
    elif model_type == "qwen3":
        enable_qwen3_learned_loki_training(
            model=model,
            low_rank_dim=low_rank_dim,
            budget_ratio=budget_ratio,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
            gate_temperature=gate_temperature,
            threshold_init=threshold_init,
            ulysses_sp_size=ulysses_sp_size,
        )
    else:
        raise ValueError(f"Model type {model_type} is not supported.")


__all__ = [
    "collect_learned_loki_training_states",
    "enable_learned_loki_training",
    "get_learned_loki_modules",
    "initialize_learned_loki_from_checkpoint",
    "learned_loki_sparse_loss_fn",
    "load_learned_loki_weight",
    "save_learned_loki_weight",
]
