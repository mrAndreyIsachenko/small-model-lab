#!/usr/bin/env bash
# End-to-end M4-M5 GPU experiment runner for RTX 4090 (24 GB) or H100 SXM (80 GB).
#
# Usage:
#   GPU_PROFILE=h100 ./scripts/remote/run_m4_m5_gpu.sh
#   GPU_PROFILE=rtx4090 MAX_DOCS=50000 MAX_STEPS=2000 ./scripts/remote/run_m4_m5_gpu.sh
#   SMOKE_ONLY=1 ./scripts/remote/run_m4_m5_gpu.sh
set -euo pipefail

GPU_PROFILE="${GPU_PROFILE:-h100}"
SHARD_DIR="${SHARD_DIR:-shards/fineweb_edu}"
RESULTS_DIR="${RESULTS_DIR:-results}"
REPORT_PATH="${REPORT_PATH:-reports/sprint1/summary.md}"
START_WATCHDOG="${START_WATCHDOG:-1}"
WATCHDOG_IDLE_MINUTES="${WATCHDOG_IDLE_MINUTES:-15}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

echo "=================================================================="
echo "[sml-gpu] Starting M4-M5 GPU experiment suite (profile=${GPU_PROFILE})"
echo "=================================================================="

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    DEVICE="cuda"
else
    echo "[sml-gpu] WARNING: nvidia-smi not found; falling back to auto device detection."
    DEVICE="auto"
fi

if [ "${START_WATCHDOG}" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "[sml-gpu] Launching idle GPU watchdog (${WATCHDOG_IDLE_MINUTES} min threshold)..."
    nohup ./scripts/remote/idle_watchdog.sh "${WATCHDOG_IDLE_MINUTES}" 5 > watchdog.log 2>&1 &
    WATCHDOG_PID=$!
    trap 'kill "${WATCHDOG_PID}" >/dev/null 2>&1 || true' EXIT
fi

# 1. Verify H1 mathematical & causal equivalence on target hardware
echo "[sml-gpu] [1/5] Verifying H1 GPT-2 124M parity..."
uv run sml verify-h1 \
    --experiment experiments/h1_gpt2_parity.yaml \
    --output-dir "${RESULTS_DIR}" \
    --run-id h1_gpt2_parity

# 2. Run M4 / H2 MFU & Triton fused-kernel throughput waterfall
echo "[sml-gpu] [2/5] Running H2 5-stage MFU throughput waterfall..."
if [ "${SMOKE_ONLY}" = "1" ]; then
    uv run sml profile-h2 \
        --experiment experiments/h2_mfu_waterfall.yaml \
        --output-dir "${RESULTS_DIR}" \
        --run-id h2_mfu_waterfall \
        --device "${DEVICE}" \
        --warmup-steps 1 \
        --measure-steps 2 \
        --seq-len 256 \
        --micro-batch-size 2
elif [ "${GPU_PROFILE}" = "rtx4090" ]; then
    uv run sml profile-h2 \
        --experiment experiments/h2_mfu_waterfall.yaml \
        --output-dir "${RESULTS_DIR}" \
        --run-id h2_mfu_waterfall_rtx4090 \
        --device "${DEVICE}" \
        --warmup-steps 3 \
        --measure-steps 10 \
        --seq-len 1024 \
        --micro-batch-size 8
else
    uv run sml profile-h2 \
        --experiment experiments/h2_mfu_waterfall.yaml \
        --output-dir "${RESULTS_DIR}" \
        --run-id h2_mfu_waterfall \
        --device "${DEVICE}" \
        --warmup-steps 5 \
        --measure-steps 15
fi

# 3. Prepare streaming dataset shards (FineWeb-Edu or synthetic smoke)
echo "[sml-gpu] [3/5] Preparing tokenized uint16 shards in ${SHARD_DIR}..."
if [ -f "${SHARD_DIR}/manifest.json" ]; then
    echo "[sml-gpu] Reusing existing shard manifest at ${SHARD_DIR}/manifest.json"
elif [ "${SMOKE_ONLY}" = "1" ]; then
    uv run sml prepare-data \
        --data-config configs/data/tinystories.yaml \
        --output-dir "${SHARD_DIR}" \
        --max-docs 60 \
        --synthetic
else
    MAX_DOCS_ARG="${MAX_DOCS:-250000}"
    uv run sml prepare-data \
        --data-config configs/data/fineweb_edu_4b.yaml \
        --output-dir "${SHARD_DIR}" \
        --max-docs "${MAX_DOCS_ARG}"
fi

# 4. Run M5 / H3 (WSD + cooldown branch) & H4 (15 log-spaced induction checkpoints)
echo "[sml-gpu] [4/5] Running H3/H4 WSD pretraining with cooldown branching..."
if [ "${SMOKE_ONLY}" = "1" ]; then
    uv run sml train \
        --experiment experiments/h3_wsd_100m.yaml \
        --manifest "${SHARD_DIR}/manifest.json" \
        --output-dir "${RESULTS_DIR}" \
        --run-id h3_wsd_100m_smoke \
        --device "${DEVICE}" \
        --max-steps 6 \
        --seq-len 64 \
        --micro-batch-size 2 \
        --num-checkpoints 4
elif [ "${GPU_PROFILE}" = "rtx4090" ]; then
    STEPS_ARG="${MAX_STEPS:-2000}"
    uv run sml train \
        --experiment experiments/h3_wsd_100m.yaml \
        --manifest "${SHARD_DIR}/manifest.json" \
        --output-dir "${RESULTS_DIR}" \
        --run-id h3_wsd_100m_rtx4090 \
        --device "${DEVICE}" \
        --max-steps "${STEPS_ARG}" \
        --seq-len 1024 \
        --micro-batch-size 8 \
        --num-checkpoints 15
else
    EXTRA_TRAIN_ARGS=()
    if [ -n "${MAX_STEPS:-}" ]; then
        EXTRA_TRAIN_ARGS+=(--max-steps "${MAX_STEPS}")
    fi
    uv run sml train \
        --experiment experiments/h3_wsd_100m.yaml \
        --manifest "${SHARD_DIR}/manifest.json" \
        --output-dir "${RESULTS_DIR}" \
        --run-id h3_wsd_100m \
        --device "${DEVICE}" \
        --num-checkpoints 15 \
        "${EXTRA_TRAIN_ARGS[@]}"
fi

# 5. Compile Markdown report strictly from measured JSON artifacts
echo "[sml-gpu] [5/5] Compiling Sprint 1 report to ${REPORT_PATH}..."
uv run sml report --results-dir "${RESULTS_DIR}" --output "${REPORT_PATH}"

echo "=================================================================="
echo "[sml-gpu] All M4-M5 stages finished. Report written to ${REPORT_PATH}"
echo "=================================================================="
