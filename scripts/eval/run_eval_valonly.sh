#!/bin/bash
# Evaluate using verl val_only mode: 100% reuses the training inference pipeline
#
# Usage:
#   # Evaluate step 300
#   bash scripts/run_eval_valonly.sh checkpoints/grpo-agentic-qwen3-8b-20260323_180259/global_step_300
#
#   # Evaluate step 200
#   bash scripts/run_eval_valonly.sh checkpoints/grpo-agentic-qwen3-8b-20260323_180259/global_step_200
#
#   # Evaluate base model (default Qwen3-8B)
#   bash scripts/run_eval_valonly.sh base
#
#   # Evaluate a specified base model
#   bash scripts/run_eval_valonly.sh base data/eval_test_1000.parquet /path/to/your/model
#
#   # Evaluate a checkpoint from another experiment
#   bash scripts/run_eval_valonly.sh /path/to/global_step_XXX

set -e

# ============ Parameters ============
CKPT_PATH="${1:?Usage: bash scripts/run_eval_valonly.sh <global_step_path|base> [eval_data] [model_path]}"
EVAL_DATA="${2:-data/eval_test_1000.parquet}"
MODEL_PATH_OVERRIDE="${3:-}"

# Determine whether to evaluate the base model
IS_BASE=false
if [[ "$CKPT_PATH" == "base" ]]; then
    IS_BASE=true
    if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then
        STEP_NAME="base_$(basename "$MODEL_PATH_OVERRIDE")"
    else
        STEP_NAME="base_model"
    fi
else
    # Ensure absolute path
    if [[ ! "$CKPT_PATH" = /* ]]; then
        CKPT_PATH="$(pwd)/${CKPT_PATH}"
    fi
    # Extract step name
    STEP_NAME=$(basename "$CKPT_PATH")  # e.g. global_step_300
fi

if [[ ! "$EVAL_DATA" = /* ]]; then
    EVAL_DATA="$(pwd)/${EVAL_DATA}"
fi

# ============ Activate conda environment ============
eval "$(conda shell.bash hook)"
conda activate onepred
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# ============ vLLM V1 engine (same as training) ============
export VLLM_USE_V1=1

# ============ Use LLM Judge V1 (Claude Opus 4.6) for evaluation ============
export USE_LLM_JUDGE_API=1

# ============ Path configuration (same as training) ============
export PROJECT_DIR="${ONEPRED_ROOT:-.}"
export VERL_DIR="${ONEPRED_ROOT:-.}/third_party/verl"
export MODEL_PATH="${MODEL_PATH_OVERRIDE:-/path/to/your/model}"
export REWARD_FN_PATH="${PROJECT_DIR}/onepred/reward_fn.py"
export INTERACTION_CONFIG="${PROJECT_DIR}/configs/interaction_config.yaml"
export AGENT_LOOP_CONFIG="${PROJECT_DIR}/configs/recurrent_agent_loop_config.yaml"

# ============ Training data (val_only also needs train_files to initialize dataloader) ============
export TRAIN_FILE="${TRAIN_FILE:-${PROJECT_DIR}/data/rl_agentic_train.parquet}"

# ============ Output directory ============
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="eval-${STEP_NAME}-${TIMESTAMP}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/eval.log"
VAL_DATA_DIR="${OUTPUT_DIR}/val_traces"
mkdir -p "${OUTPUT_DIR}" "${VAL_DATA_DIR}"

# ============ Structured traces ============
export ONEPRED_TRACE_DIR="${OUTPUT_DIR}/structured_traces"
mkdir -p "${ONEPRED_TRACE_DIR}"
# Print all traces to stdout for easy inspection
export ONEPRED_TRACE_PRINT_RATE="1.0"

# ============ Ensure scripts can be imported ============
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# ============ Training configuration ============
NUM_GPUS=8
TRAIN_BATCH_SIZE=32
ROLLOUT_N=1
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096
LR=5e-7
MICRO_BATCH=2

echo "=========================================="
echo "verl val_only evaluation"
if [[ "$IS_BASE" == "true" ]]; then
    echo "Model: ${MODEL_PATH} (base, untrained)"
else
    echo "Checkpoint: ${CKPT_PATH}"
fi
echo "Eval data: ${EVAL_DATA}"
echo "GPU count: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

cd "${PROJECT_DIR}"

# Build resume arguments (not needed for base model)
RESUME_ARGS=""
if [[ "$IS_BASE" == "false" ]]; then
    RESUME_ARGS="trainer.resume_mode=resume_path trainer.resume_from_path=${CKPT_PATH}"
fi

python3 -m onepred.main_ppo_recurrent \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${EVAL_DATA} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH} \
    actor_rollout_ref.rollout.logprobs_mode=null \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=${INTERACTION_CONFIG} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=recurrent_memory_agent \
    +actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG} \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=onepred.recurrent_generation_manager.RecurrentAgentLoopManager \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path=${REWARD_FN_PATH} \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.balance_batch=False \
    trainer.logger='["console"]' \
    trainer.project_name=OnePred \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${OUTPUT_DIR}/ckpt_tmp \
    trainer.validation_data_dir=${VAL_DATA_DIR} \
    trainer.log_val_generations=1000 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    ${RESUME_ARGS} \
    2>&1 | tee ${LOG_FILE}

# ============ Post-processing: auto-generate summary.json + report.txt ============
echo ""
echo "[Post-processing] Generating summary.json + report.txt ..."
python3 -m tools.summarize_eval \
    "${OUTPUT_DIR}" \
    --step_name "${STEP_NAME}" \
    --eval_data "${EVAL_DATA}"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Log: ${LOG_FILE}"
echo "Validation traces: ${VAL_DATA_DIR}"
echo "Structured traces: ${ONEPRED_TRACE_DIR}"
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo "Report: ${OUTPUT_DIR}/report.txt"
echo "=========================================="
