import torch

from areal.utils import stats_tracker


def semantic_cluster_loss_fn(
    training_states,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    losses = []
    total_selected = 0.0

    for state in training_states:
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

            feats = projected_tokens[batch_idx, :valid_len]
            centers = feats[selected_indices[batch_idx, :selected_count]]
            distances = torch.cdist(feats, centers, p=2).pow(2)

            assign = torch.softmax(-distances / max(temperature, eps), dim=-1)
            token_weights = importance[batch_idx, :valid_len]
            token_weights = token_weights / token_weights.sum().clamp_min(eps)

            within_cluster = (token_weights.unsqueeze(-1) * assign * distances).sum()

            center_dist = torch.pdist(centers, p=2).pow(2)
            between_cluster = center_dist.mean() if center_dist.numel() > 0 else eps
            losses.append(within_cluster / (between_cluster + eps))
            total_selected += float(selected_count)

    if len(losses) == 0:
        return torch.tensor(0.0, device=training_states[0]["projected_tokens"].device)

    loss = torch.stack(losses).mean()
    stats_tracker.scalar(
        semantic_kv_cluster_loss=loss.detach(),
        semantic_kv_avg_selected_tokens=total_selected / max(len(losses), 1),
    )
    return loss
