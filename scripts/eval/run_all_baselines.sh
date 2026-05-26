#!/bin/bash
# Evaluate 3 baselines: current_turn, last_3, full_concat
# Data: v4 test set (5000 JSONL entries)
# Model: Qwen3-8B zero-shot, temperature=0
#
# Usage:
#   bash scripts/run_all_baselines.sh
#
# Requires 3 GPUs (parallel inference)

set -e

# ============ Environment ============
eval "$(conda shell.bash hook)"
conda activate onepred
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# ============ Configuration ============
PROJECT_DIR="${ONEPRED_ROOT}"
MODEL="${MODEL_DIR}/hf/Qwen3-8B"
RERANKER_MODEL="${MODEL_DIR}/hf/Qwen3-Reranker-0.6B"
DATA="${DATA_DIR}/test.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export EMBEDDING_MODEL_PATH="${MODEL_DIR}/hf/bge-m3"
export EMBEDDING_DEVICE="cuda:0"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p outputs

# ============ Logging ============
MASTER_LOG="outputs/run_all_baselines-${TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================="
echo "Baseline evaluation: current_turn / last_3 / full_concat"
echo "Model: ${MODEL}"
echo "Data: ${DATA}"
echo "Time: ${TIMESTAMP}"
echo "=========================================="

# ============ Phase 1: Inference (3 methods in parallel, 1 GPU each) ============
echo ""
echo "[Phase 1] Starting 3 inference tasks..."

METHODS=(current_turn last_3 full_concat)
GPUS=(0 1 2)
PIDS=()

for i in "${!METHODS[@]}"; do
    METHOD=${METHODS[$i]}
    GPU=${GPUS[$i]}
    DIR="outputs/baseline-${METHOD}-${TIMESTAMP}"
    mkdir -p "${DIR}"
    python -m tools.evaluate_baselines \
        --method ${METHOD} \
        --model_path ${MODEL} \
        --data ${DATA} \
        --gpu ${GPU} \
        --output_dir "${DIR}" \
        > "${DIR}/inference.log" 2>&1 &
    PIDS+=($!)
    echo "  [GPU ${GPU}] ${METHOD}  PID=${PIDS[-1]}"
done

echo ""
echo "[Phase 1] Waiting for all inference tasks to complete..."
echo "  Monitor: tail -f outputs/baseline-*-${TIMESTAMP}/eval.log"

FAILED=""
for i in "${!METHODS[@]}"; do
    wait ${PIDS[$i]} || FAILED="${FAILED} ${METHODS[$i]}"
    echo "  ${METHODS[$i]}  done"
done

if [ -n "$FAILED" ]; then
    echo ""
    echo "[Warning] The following inference tasks failed:${FAILED}"
    echo "Please check the corresponding inference.log files"
fi

echo ""
echo "[Phase 1] All inference complete!"

# ============ Phase 2: Compute all metrics (3 baselines in parallel, each on a different GPU) ============
echo ""
echo "[Phase 2] Starting 3 metric computation tasks (parallel)..."

METRIC_GPUS=(3 4 5)
METRIC_PIDS=()
for i in "${!METHODS[@]}"; do
    METHOD=${METHODS[$i]}
    MGPU=${METRIC_GPUS[$i]}
    DIR="outputs/baseline-${METHOD}-${TIMESTAMP}"
    TRACES="${DIR}/structured_traces/traces.jsonl"
    if [ -f "$TRACES" ]; then
        EMBEDDING_DEVICE="cuda:${MGPU}" python -m tools.compute_metrics \
            "${TRACES}" \
            --reranker_model "${RERANKER_MODEL}" \
            --device "cuda:${MGPU}" \
            --llm_judge_workers 30 \
            --output_dir "${DIR}" \
            2>&1 | tee "${DIR}/compute_metrics.log" &
        METRIC_PIDS+=($!)
        echo "  [GPU ${MGPU}] ${METHOD} PID=${METRIC_PIDS[-1]}"
    else
        echo "  [Skipped] ${METHOD}: ${TRACES} does not exist"
    fi
done

echo ""
echo "[Phase 2] Waiting for metric computation to complete..."
for PID in "${METRIC_PIDS[@]}"; do
    wait $PID
done

echo ""
echo "[Phase 2] Metric computation complete!"

# ============ Phase 3: Summarize results ============
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="

for METHOD in "${METHODS[@]}"; do
    F="outputs/baseline-${METHOD}-${TIMESTAMP}/metrics.json"
    if [ -f "$F" ]; then
        echo ""
        echo "--- ${METHOD} ---"
        python3 -c "
import json
d = json.load(open('$F'))
for k, v in sorted(d.items()):
    if isinstance(v, float):
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v}')
"
    else
        echo "  ${METHOD}: metrics.json does not exist"
    fi
done

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo "Output directories:"
for METHOD in "${METHODS[@]}"; do
    echo "  outputs/baseline-${METHOD}-${TIMESTAMP}/"
done
echo "Master log: ${MASTER_LOG}"
echo "=========================================="
