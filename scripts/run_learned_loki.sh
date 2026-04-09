set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
xdg_cache_home="${XDG_CACHE_HOME:-$HOME/.cache}"
export AREAL_LOCAL_CACHE_DIR="${AREAL_LOCAL_CACHE_DIR:-${xdg_cache_home}/areal}"
export AREAL_PORT_LOCKFILE_ROOT="${AREAL_PORT_LOCKFILE_ROOT:-${AREAL_LOCAL_CACHE_DIR}/ports}"

model="${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
trial_name="${2:-learned_loki_r32}"
config_path="${3:-examples/learned_loki/grpo.yaml}"

low_rank_dim="${LEARNED_LOKI_RANK:-32}"
budget_ratio="${LEARNED_LOKI_BUDGET_RATIO:-0.5}"
sink_window_size="${LEARNED_LOKI_SINK_SIZE:-16}"
recent_window_size="${LEARNED_LOKI_RECENT_SIZE:-64}"
threshold_init="${LEARNED_LOKI_THRESHOLD_INIT:-0.0}"
init_mode="${LEARNED_LOKI_INIT_MODE:-orthogonal}"
init_path="${LEARNED_LOKI_INIT_PATH:-}"
gate_temperature_init="${LEARNED_LOKI_GATE_TEMPERATURE_INIT:-1.0}"
gate_temperature_final="${LEARNED_LOKI_GATE_TEMPERATURE_FINAL:-1.0}"
gate_temperature_anneal_steps="${LEARNED_LOKI_GATE_TEMPERATURE_ANNEAL_STEPS:-0}"
sparse_loss_scale="${LEARNED_LOKI_SPARSE_LOSS_SCALE:-0.1}"
reward_scaled_sparse_loss="${LEARNED_LOKI_REWARD_SCALED_SPARSE_LOSS:-false}"
epochs="${LEARNED_LOKI_EPOCHS:-2}"
lr="${LEARNED_LOKI_LR:-1e-2}"
attn_impl="${LEARNED_LOKI_ATTN_IMPL:-flash_attention_2}"
train_batch_size="${LEARNED_LOKI_TRAIN_BATCH_SIZE:-32}"
valid_batch_size="${LEARNED_LOKI_VALID_BATCH_SIZE:-32}"
max_tokens_per_mb="${LEARNED_LOKI_MAX_TOKENS_PER_MB:-7680}"
max_new_tokens="${LEARNED_LOKI_MAX_NEW_TOKENS:-7168}"
context_length="${LEARNED_LOKI_CONTEXT_LENGTH:-7680}"
max_running_requests="${LEARNED_LOKI_MAX_RUNNING_REQUESTS:-128}"
max_concurrent_rollouts="${LEARNED_LOKI_MAX_CONCURRENT_ROLLOUTS:-${train_batch_size}}"
mem_fraction_static="${LEARNED_LOKI_MEM_FRACTION_STATIC:-0.7}"
gradient_checkpointing="${LEARNED_LOKI_GRADIENT_CHECKPOINTING:-false}"
fsdp_offload_params="${LEARNED_LOKI_FSDP_OFFLOAD_PARAMS:-false}"

init_path_override="++actor.learned_loki_init_path=null"
if [ -n "${init_path}" ]; then
    init_path_override="++actor.learned_loki_init_path=${init_path}"
fi

python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config "${config_path}" \
    experiment_name="learned-loki-grpo" \
    trial_name="${trial_name}" \
    total_train_epochs="${epochs}" \
    ++actor.path="${model}" \
    ++ref.path="${model}" \
    ++sglang.model_path="${model}" \
    ++actor.optimizer.lr="${lr}" \
    ++actor.attn_impl="${attn_impl}" \
    ++ref.attn_impl="${attn_impl}" \
    ++actor.gradient_checkpointing="${gradient_checkpointing}" \
    ++actor.fsdp.offload_params="${fsdp_offload_params}" \
    ++actor.mb_spec.max_tokens_per_mb="${max_tokens_per_mb}" \
    ++actor.max_new_tokens="${max_new_tokens}" \
    ++gconfig.max_new_tokens="${max_new_tokens}" \
    ++actor.enable_mixed_attn_training="false" \
    ++actor.enable_semantic_kv_training="false" \
    ++actor.enable_learned_loki_training="true" \
    ++actor.learned_loki_rank="${low_rank_dim}" \
    ++actor.learned_loki_budget_ratio="${budget_ratio}" \
    ++actor.learned_loki_sink_window_size="${sink_window_size}" \
    ++actor.learned_loki_recent_window_size="${recent_window_size}" \
    ++actor.learned_loki_threshold_init="${threshold_init}" \
    ++actor.learned_loki_init_mode="${init_mode}" \
    "${init_path_override}" \
    ++actor.learned_loki_gate_temperature_init="${gate_temperature_init}" \
    ++actor.learned_loki_gate_temperature_final="${gate_temperature_final}" \
    ++actor.learned_loki_gate_temperature_anneal_steps="${gate_temperature_anneal_steps}" \
    ++actor.learned_loki_sparse_loss_scale="${sparse_loss_scale}" \
    ++actor.learned_loki_reward_scaled_sparse_loss="${reward_scaled_sparse_loss}" \
    ++sglang.enable_learned_loki="true" \
    ++sglang.learned_loki_rank="${low_rank_dim}" \
    ++sglang.learned_loki_budget_ratio="${budget_ratio}" \
    ++sglang.learned_loki_sink_window_size="${sink_window_size}" \
    ++sglang.learned_loki_recent_window_size="${recent_window_size}" \
    ++sglang.learned_loki_gate_temperature="${gate_temperature_init}" \
    ++sglang.learned_loki_threshold_init="${threshold_init}" \
    ++sglang.context_length="${context_length}" \
    ++sglang.max_running_requests="${max_running_requests}" \
    ++sglang.mem_fraction_static="${mem_fraction_static}" \
    ++sglang.disable_radix_cache="true" \
    ++sglang.disable_overlap_schedule="true" \
    ++sglang.attention_backend="torch_native" \
    ++train_dataset.batch_size="${train_batch_size}" \
    ++valid_dataset.batch_size="${valid_batch_size}" \
    ++rollout.consumer_batch_size="${train_batch_size}" \
    ++rollout.max_concurrent_rollouts="${max_concurrent_rollouts}" \
    ++stats_logger.wandb.mode="${WANDB_MODE}"
