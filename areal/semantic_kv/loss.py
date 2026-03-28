from typing import Optional

import torch

from areal.utils import stats_tracker


def _infer_device(training_states) -> torch.device:
    for state in training_states:
        compression_events = state.get("compression_events", None)
        if compression_events:
            return compression_events[0]["projected_tokens"].device
        if "projected_tokens" in state:
            return state["projected_tokens"].device
    return torch.device("cpu")


def _semantic_cluster_loss_from_event(
    projected_tokens: torch.Tensor,
    selected_features: torch.Tensor,
    importance: torch.Tensor,
    temperature: float,
    eps: float,
) -> Optional[torch.Tensor]:
    if projected_tokens.numel() == 0 or selected_features.shape[0] <= 1:
        return None

    feats = projected_tokens.float()
    centers = selected_features.float()
    token_weights = importance.float()
    token_weights = token_weights / token_weights.sum().clamp_min(eps)

    distances = torch.cdist(feats, centers, p=2).pow(2)
    assign = torch.softmax(-distances / max(temperature, eps), dim=-1)
    within_cluster = (token_weights.unsqueeze(-1) * assign * distances).sum()

    center_dist = torch.pdist(centers, p=2).pow(2)
    between_cluster = (
        center_dist.mean() if center_dist.numel() > 0 else feats.new_tensor(eps)
    )
    return within_cluster / (between_cluster + eps)


def semantic_cluster_loss_fn(
    training_states,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    losses = []
    total_selected = 0.0

    for state in training_states:
        compression_events = state.get("compression_events", None)
        if compression_events is not None:
            for event in compression_events:
                loss = _semantic_cluster_loss_from_event(
                    projected_tokens=event["projected_tokens"],
                    selected_features=event["selected_features"],
                    importance=event["importance"],
                    temperature=temperature,
                    eps=eps,
                )
                if loss is None:
                    continue
                losses.append(loss)
                total_selected += float(event["selected_features"].shape[0])
            continue

        projected_tokens = state["projected_tokens"].float()
        selected_indices = state["selected_indices"]
        selected_counts = state["selected_counts"]
        importance = state["importance"].float()
        attention_mask = state.get("attention_mask", None)

        for batch_idx in range(projected_tokens.shape[0]):
            valid_len = projected_tokens.shape[1]
            if attention_mask is not None and attention_mask.ndim == 2:
                valid_len = int(attention_mask[batch_idx].sum().item())
                valid_len = max(valid_len, 1)

            selected_count = int(selected_counts[batch_idx].item())
            if selected_count <= 1:
                continue

            loss = _semantic_cluster_loss_from_event(
                projected_tokens=projected_tokens[batch_idx, :valid_len],
                selected_features=projected_tokens[batch_idx][
                    selected_indices[batch_idx, :selected_count]
                ],
                importance=importance[batch_idx, :valid_len],
                temperature=temperature,
                eps=eps,
            )
            if loss is None:
                continue
            losses.append(loss)
            total_selected += float(selected_count)

    if len(losses) == 0:
        return torch.tensor(0.0, device=_infer_device(training_states))

    loss = torch.stack(losses).mean()
    stats_tracker.scalar(
        semantic_kv_cluster_loss=loss.detach(),
        semantic_kv_avg_selected_tokens=total_selected / max(len(losses), 1),
    )
    return loss
