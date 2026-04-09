import torch

from areal.utils import stats_tracker


def learned_loki_sparse_loss_fn(training_states) -> torch.Tensor:
    total_sum = None
    total_count = 0
    device = torch.device("cpu")

    for state in training_states:
        gate_sum = state.get("middle_gate_sum", None)
        gate_count = int(state.get("middle_gate_count", 0))
        if gate_sum is not None:
            device = gate_sum.device
        if gate_sum is None or gate_count <= 0:
            continue
        total_sum = gate_sum if total_sum is None else total_sum + gate_sum
        total_count += gate_count

    if total_sum is None or total_count == 0:
        return torch.zeros((), device=device)

    loss = total_sum / total_count
    stats_tracker.scalar(
        learned_loki_sparse_loss=loss.detach(),
        learned_loki_avg_middle_gate=loss.detach(),
        learned_loki_middle_gate_count=float(total_count),
    )
    return loss
