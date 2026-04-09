set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

model="${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
trial_name="${2:-semantic_kv_r32}"
config_path="${3:-examples/semantic_kv/grpo.yaml}"

low_rank_dim="${SEMANTIC_KV_RANK:-32}"
budget_ratio="${SEMANTIC_KV_BUDGET_RATIO:-0.5}"
sink_window_size="${SEMANTIC_KV_SINK_SIZE:-16}"
recent_window_size="${SEMANTIC_KV_RECENT_SIZE:-64}"
selector_temperature="${SEMANTIC_KV_SELECTOR_TEMPERATURE:-1.0}"
cluster_loss_scale="${SEMANTIC_KV_CLUSTER_LOSS_SCALE:-0.1}"
cluster_temperature="${SEMANTIC_KV_CLUSTER_TEMPERATURE:-0.1}"
epochs="${SEMANTIC_KV_EPOCHS:-2}"
lr="${SEMANTIC_KV_LR:-1e-2}"

python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config "${config_path}" \
    experiment_name="semantic-kv-grpo" \
    trial_name="${trial_name}" \
    total_train_epochs="${epochs}" \
    ++actor.path="${model}" \
    ++ref.path="${model}" \
    ++sglang.model_path="${model}" \
    ++actor.optimizer.lr="${lr}" \
    ++actor.enable_mixed_attn_training="false" \
    ++actor.enable_semantic_kv_training="true" \
    ++actor.semantic_kv_rank="${low_rank_dim}" \
    ++actor.semantic_kv_budget_ratio="${budget_ratio}" \
    ++actor.semantic_kv_sink_window_size="${sink_window_size}" \
    ++actor.semantic_kv_recent_window_size="${recent_window_size}" \
    ++actor.semantic_kv_selector_temperature="${selector_temperature}" \
    ++actor.semantic_kv_cluster_loss_scale="${cluster_loss_scale}" \
    ++actor.semantic_kv_cluster_temperature="${cluster_temperature}" \
    ++sglang.enable_semantic_kv="true" \
    ++sglang.semantic_kv_rank="${low_rank_dim}" \
    ++sglang.semantic_kv_budget_ratio="${budget_ratio}" \
    ++sglang.semantic_kv_sink_window_size="${sink_window_size}" \
    ++sglang.semantic_kv_recent_window_size="${recent_window_size}" \
    ++sglang.disable_radix_cache="true" \
    ++sglang.disable_overlap_schedule="true" \
    ++sglang.attention_backend="torch_native" \
    ++stats_logger.wandb.mode="${WANDB_MODE}"
