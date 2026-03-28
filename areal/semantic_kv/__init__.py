from .forward import (
    enable_llama_semantic_kv_training,
    enable_qwen2_semantic_kv_training,
    enable_qwen3_semantic_kv_training,
)
from .loss import semantic_cluster_loss_fn
from .utils import (
    collect_semantic_kv_training_states,
    get_semantic_kv_modules,
    get_semantic_kv_weight,
    load_semantic_kv_weight,
    save_semantic_kv_weight,
)


def enable_semantic_kv_training(
    model,
    low_rank_dim: int = 32,
    budget_ratio: float = 0.5,
    sink_window_size: int = 16,
    recent_window_size: int = 64,
    selector_temperature: float = 1.0,
    ulysses_sp_size: int = 1,
):
    if "llama" in model.config.model_type:
        enable_llama_semantic_kv_training(
            model=model,
            low_rank_dim=low_rank_dim,
            budget_ratio=budget_ratio,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
            selector_temperature=selector_temperature,
            ulysses_sp_size=ulysses_sp_size,
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen2_semantic_kv_training(
            model=model,
            low_rank_dim=low_rank_dim,
            budget_ratio=budget_ratio,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
            selector_temperature=selector_temperature,
            ulysses_sp_size=ulysses_sp_size,
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_semantic_kv_training(
            model=model,
            low_rank_dim=low_rank_dim,
            budget_ratio=budget_ratio,
            sink_window_size=sink_window_size,
            recent_window_size=recent_window_size,
            selector_temperature=selector_temperature,
            ulysses_sp_size=ulysses_sp_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


__all__ = [
    "collect_semantic_kv_training_states",
    "enable_semantic_kv_training",
    "get_semantic_kv_modules",
    "get_semantic_kv_weight",
    "load_semantic_kv_weight",
    "save_semantic_kv_weight",
    "semantic_cluster_loss_fn",
]
