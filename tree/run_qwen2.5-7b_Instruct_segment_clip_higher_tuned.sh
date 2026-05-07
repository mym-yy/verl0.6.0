set -x
# Save login home before HOME is overridden (otherwise ~/.wandb_api_key would be wrong).
_ORIG_HOME="${HOME}"

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export PYTHONPATH=/root/autodl-tmp/verl0.6.0:/root/autodl-tmp/vllm:${PYTHONPATH}
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VERL_DEBUG_LOG_PATH=/root/autodl-tmp/debug_log
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
HOME=/root/autodl-tmp
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/data"}

TRAIN_FILE=/root/autodl-tmp/data/dapo-math-17k/dapo-math-17k-verl.parquet
TEST_FILE=/root/autodl-tmp/data/aime-2024/aime-2024-verl.parquet

# This script intentionally keeps the same strategy family:
# - high-entropy branch tree rollout
# - grpo_tree_segment_strict
# - segment_clip_higher
#
# The changes are only for stabilization:
# - lighter tree to avoid 20+ leaves/prompt blow-up
# - shorter responses to reduce invalid / verbose outputs
# - aligned overlong penalty so long responses are actually discouraged
# - milder clip-higher window and slightly smaller LR / stronger KL
project_name=verl_grpo_tree_rollout_segment_tuned
experiment_name=qwen2.5_7b_instruct_segment_clip_higher_tuned_v1

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1536}"
ACTOR_LR="${ACTOR_LR:-8e-7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.18}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.22}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.015}"
TREE_ENTROPY_THRESHOLD="${TREE_ENTROPY_THRESHOLD:-1.8}"
TREE_BRANCHING_FACTOR="${TREE_BRANCHING_FACTOR:-2}"
TREE_MAX_DEPTH="${TREE_MAX_DEPTH:-2}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-256}"
OVERLONG_PENALTY="${OVERLONG_PENALTY:-1.0}"
VAL_N="${VAL_N:-4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:--1}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "ERROR: converted train parquet is missing: ${TRAIN_FILE}" >&2
  echo "Run: python3 /root/autodl-tmp/verl0.6.0/tree/prepare_dapo_math_verl.py" >&2
  exit 1
fi

LOG_DIR="${HOME}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Training started at $(date) ===" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "project_name=${project_name}" | tee -a "${LOG_FILE}"
echo "experiment_name=${experiment_name}" | tee -a "${LOG_FILE}"
echo "strategy=tree+grpo_tree_segment_strict+segment_clip_higher" | tee -a "${LOG_FILE}"
echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}" | tee -a "${LOG_FILE}"
echo "TREE_ENTROPY_THRESHOLD=${TREE_ENTROPY_THRESHOLD}" | tee -a "${LOG_FILE}"
echo "TREE_BRANCHING_FACTOR=${TREE_BRANCHING_FACTOR}" | tee -a "${LOG_FILE}"
echo "TREE_MAX_DEPTH=${TREE_MAX_DEPTH}" | tee -a "${LOG_FILE}"
echo "CLIP_RATIO_LOW=${CLIP_RATIO_LOW}" | tee -a "${LOG_FILE}"
echo "CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH}" | tee -a "${LOG_FILE}"
echo "ACTOR_LR=${ACTOR_LR}" | tee -a "${LOG_FILE}"
echo "VAL_N=${VAL_N}" | tee -a "${LOG_FILE}"
echo "VAL_TEMPERATURE=${VAL_TEMPERATURE}" | tee -a "${LOG_FILE}"
echo "VAL_TOP_P=${VAL_TOP_P}" | tee -a "${LOG_FILE}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for _wandb_keyfile in "${_ORIG_HOME}/.wandb_api_key" "${HOME}/.wandb_api_key"; do
    if [[ -f "${_wandb_keyfile}" ]]; then
      export WANDB_API_KEY="$(tr -d ' \n\r\t' < "${_wandb_keyfile}")"
      break
    fi
  done
fi

# Local testing only: put your key here if you do not use env / ~/.wandb_api_key.
# Priority: shell export > key files above > this line (empty = skip).
# Do not commit real keys to shared repos.
_WANDB_API_KEY_INLINE="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"
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -n "${_WANDB_API_KEY_INLINE}" ]]; then
  export WANDB_API_KEY="${_WANDB_API_KEY_INLINE}"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is empty. trainer.logger includes wandb but Ray has no TTY for wandb login." >&2
  echo "Fix: set _WANDB_API_KEY_INLINE in this script, OR export WANDB_API_KEY=..., OR use ${_ORIG_HOME}/.wandb_api_key / ${HOME}/.wandb_api_key" >&2
  exit 1
fi
export WANDB_KEY="${WANDB_API_KEY}"

export WANDB_MODE=online
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_DIR="${HOME}/wandb"
mkdir -p "${WANDB_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_tree_segment_strict \
    algorithm.rollout_is=False \
    algorithm.rollout_is_threshold=null \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=segment_clip_higher \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.tree_search.enable=True \
    actor_rollout_ref.rollout.tree_search.entropy_threshold="${TREE_ENTROPY_THRESHOLD}" \
    actor_rollout_ref.rollout.tree_search.branching_factor="${TREE_BRANCHING_FACTOR}" \
    actor_rollout_ref.rollout.tree_search.max_tree_depth="${TREE_MAX_DEPTH}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}" \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY}" \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=2 \
    trainer.rollout_data_dir="${HOME}/rollout_data/${project_name}/${experiment_name}" \
    trainer.validation_data_dir="${HOME}/validation_data/${project_name}/${experiment_name}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K}" \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
