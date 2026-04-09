import os
import re

import torch


def get_learned_loki_modules(model):
    modules = []
    for layer in model.model.layers:
        module = layer.self_attn
        if not hasattr(module, "learned_loki") or module.learned_loki is None:
            continue
        modules.append(module.learned_loki)

    if len(modules) != len(model.model.layers):
        raise ValueError("Not all layers have learned_loki modules.")
    return modules


def collect_learned_loki_training_states(model):
    return [
        module.last_training_state
        for module in get_learned_loki_modules(model)
        if module.last_training_state is not None
    ]


def save_learned_loki_weight(model, state_dict, save_dir):
    learned_loki_state_dict = {}
    projection_weight_dict = {}
    threshold_dict = {}

    for key, value in state_dict.items():
        if "learned_loki" not in key:
            continue
        learned_loki_state_dict[key] = value

        match = re.search(r"\.layers\.(\d+)\.", key)
        if match is None:
            continue
        layer_idx = int(match.group(1))

        if key.endswith("learned_loki.weight"):
            projection_weight_dict[layer_idx] = value.detach().cpu()
        elif key.endswith("learned_loki.threshold"):
            threshold_dict[layer_idx] = value.detach().cpu()

    if len(learned_loki_state_dict) == 0:
        raise ValueError("No learned_loki weight found")

    modules = get_learned_loki_modules(model)
    config = {
        "low_rank_dim": modules[0].low_rank_dim,
        "budget_ratio": modules[0].budget_ratio,
        "sink_window_size": modules[0].sink_window_size,
        "recent_window_size": modules[0].recent_window_size,
        "gate_temperature": modules[0].gate_temperature,
    }

    torch.save(
        {
            "model_name": model.config._name_or_path,
            "learned_loki_state_dict": learned_loki_state_dict,
            "projection_weight_dict": projection_weight_dict,
            "threshold_dict": threshold_dict,
            "config": config,
        },
        os.path.join(save_dir, "learned_loki.pt"),
    )


def load_learned_loki_weight(path):
    if os.path.isdir(path):
        path = os.path.join(path, "learned_loki.pt")

    ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt.get("model_name", None)
    learned_loki_state_dict = ckpt.get("learned_loki_state_dict", None)
    config = ckpt.get("config", None)
    if model_name is None or learned_loki_state_dict is None or config is None:
        raise ValueError("Invalid Learned-Loki checkpoint format")
    return model_name, learned_loki_state_dict, config


def initialize_learned_loki_from_checkpoint(model, path):
    model_name, learned_loki_state_dict, config = load_learned_loki_weight(path)
    if model_name != model.config._name_or_path:
        raise ValueError(
            f"Model name mismatch: {model_name} vs {model.config._name_or_path}"
        )

    missing_keys, unexpected_keys = model.load_state_dict(
        learned_loki_state_dict,
        strict=False,
    )
    missing_keys = [
        key for key in missing_keys if "learned_loki" in key and "gate_temperature_param" not in key
    ]
    unexpected_keys = [key for key in unexpected_keys if "learned_loki" in key]
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Failed to initialize Learned-Loki checkpoint. "
            f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}"
        )
    return config
