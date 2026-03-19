import os
import re

import torch


def get_semantic_kv_modules(model):
    modules = []
    for layer in model.model.layers:
        module = layer.self_attn
        if not hasattr(module, "semantic_kv") or module.semantic_kv is None:
            continue
        modules.append(module.semantic_kv)

    if len(modules) != len(model.model.layers):
        raise ValueError("Not all layers have semantic_kv modules.")
    return modules


def get_semantic_kv_weight(model):
    return [module.weight for module in get_semantic_kv_modules(model)]


def collect_semantic_kv_training_states(model):
    return [
        module.last_training_state
        for module in get_semantic_kv_modules(model)
        if module.last_training_state is not None
    ]


def save_semantic_kv_weight(model, state_dict, save_dir):
    semantic_kv_state_dict = {}
    projection_weight_dict = {}

    for key, value in state_dict.items():
        if "semantic_kv" not in key:
            continue
        semantic_kv_state_dict[key] = value

        if key.endswith("semantic_kv.weight"):
            match = re.search(r"\.layers\.(\d+)\.", key)
            if match is None:
                raise ValueError(f"Unable to parse layer index from key: {key}")
            layer_idx = int(match.group(1))
            projection_weight_dict[layer_idx] = value.detach().cpu()

    if len(semantic_kv_state_dict) == 0:
        raise ValueError("No semantic_kv weight found")

    modules = get_semantic_kv_modules(model)
    config = {
        "low_rank_dim": modules[0].low_rank_dim,
        "budget_ratio": modules[0].budget_ratio,
        "sink_window_size": modules[0].sink_window_size,
        "recent_window_size": modules[0].recent_window_size,
    }

    torch.save(
        {
            "model_name": model.config._name_or_path,
            "semantic_kv_state_dict": semantic_kv_state_dict,
            "projection_weight_dict": projection_weight_dict,
            "config": config,
        },
        os.path.join(save_dir, "semantic_kv.pt"),
    )


def load_semantic_kv_weight(path):
    if os.path.isdir(path):
        path = os.path.join(path, "semantic_kv.pt")

    ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt.get("model_name", None)
    semantic_kv_state_dict = ckpt.get("semantic_kv_state_dict", None)
    config = ckpt.get("config", None)
    if model_name is None or semantic_kv_state_dict is None or config is None:
        raise ValueError("Invalid semantic KV checkpoint format")
    return model_name, semantic_kv_state_dict, config
