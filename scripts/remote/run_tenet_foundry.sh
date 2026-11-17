#!/usr/bin/env bash
# Build a Tenet release bundle on a remote foundry box (MPS laptop or H100 node).
#
# Usage:
#   TENET_PROFILE=3b-mps ./scripts/remote/run_tenet_foundry.sh
#
# Environment knobs (all optional):
#   TENET_PROFILE  3b-mps | 8b-h100 | 12b-h100  (default: 3b-mps)
#   MODEL_NAME / RELEASE_ID / DEVICE / BACKEND  override the profile values
#   OUTPUT_DIR     release output root          (default: results/tenet_releases)
#   HF_NAMESPACE   HF Hub namespace             (default: tenet-lab)
#   UPLOAD_TO_HF   1 to upload the bundle       (default: 0)
#   SMOKE_ONLY     1 for a tiny CPU smoke run   (default: 0)
#   DRY_RUN        1 to print the command only  (default: 0)
#   RUN_SUFFIX     run-id suffix for benchmarks (default: per-profile suffix)
#   RESULTS_ROOT   benchmark/results root       (default: results)
#   RUN_STAGE_BENCHMARKS  1 to run verify-h9..h12 first (default: 1, or 0 with SMOKE_ONLY=1)
#   START_WATCHDOG 1 to start idle_watchdog.sh  (default: 0)

set -euo pipefail

TENET_PROFILE="${TENET_PROFILE:-3b-mps}"

case "${TENET_PROFILE}" in
  3b-mps)
    PROFILE_MODEL="Qwen/Qwen2.5-3B-Instruct"
    PROFILE_RELEASE_ID="Tenet-3B-Qwen2.5-Unslop-v1"
    PROFILE_DEVICE="mps"
    PROFILE_BACKEND="llm"
    PROFILE_RUN_SUFFIX="qwen3b_mps"
    ;;
  8b-h100)
    PROFILE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    PROFILE_RELEASE_ID="Tenet-8B-Llama3.1-Unslop-v1"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="llm"
    PROFILE_RUN_SUFFIX="llama8b_h100"
    ;;
  12b-h100)
    PROFILE_MODEL="mistralai/Mistral-Nemo-Instruct-2407"
    PROFILE_RELEASE_ID="Tenet-12B-Nemo-Unslop-v1"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="llm"
    PROFILE_RUN_SUFFIX="nemo12b_h100"
    ;;
  *)
    echo "Unknown TENET_PROFILE '${TENET_PROFILE}' (expected 3b-mps, 8b-h100, or 12b-h100)." >&2
    exit 2
    ;;
esac

MODEL_NAME="${MODEL_NAME:-${PROFILE_MODEL}}"
RELEASE_ID="${RELEASE_ID:-${PROFILE_RELEASE_ID}}"
DEVICE="${DEVICE:-${PROFILE_DEVICE}}"
BACKEND="${BACKEND:-${PROFILE_BACKEND}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/tenet_releases}"
HF_NAMESPACE="${HF_NAMESPACE:-tenet-lab}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-${PROFILE_RUN_SUFFIX}}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
START_WATCHDOG="${START_WATCHDOG:-0}"

GRPO_STEPS="${GRPO_STEPS:-27}"
N_PROMPTS="${N_PROMPTS:-8}"
SEQ_LEN="${SEQ_LEN:-32}"

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  BACKEND="nano"
  DEVICE="cpu"
  GRPO_STEPS=4
  N_PROMPTS=2
  SEQ_LEN=16
fi

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  RUN_STAGE_BENCHMARKS="${RUN_STAGE_BENCHMARKS:-0}"
else
  RUN_STAGE_BENCHMARKS="${RUN_STAGE_BENCHMARKS:-1}"
fi

HF_REPO_ID="${HF_NAMESPACE}/${RELEASE_ID}"
RELEASE_PATH="${OUTPUT_DIR}/${RELEASE_ID}"

# Llama-3.1 weights are gated on the HF Hub; fail fast with a clear message.
if [[ "${BACKEND}" == "llm" && "${DRY_RUN}" == "0" && "${MODEL_NAME}" == *"Llama-3.1"* ]]; then
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: MODEL_NAME '${MODEL_NAME}' requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) to be set." >&2
    exit 1
  fi
fi

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

VERIFY_H9_CMD=(
  uv run sml verify-h9
  --experiment experiments/h9_biprojected_surgery.yaml
  --output-dir "${RESULTS_ROOT}"
  --run-id "h9_${RUN_SUFFIX}"
  --backend "${BACKEND}"
  --model-name "${MODEL_NAME}"
  --device "${DEVICE}"
)
VERIFY_H10_CMD=(
  uv run sml verify-h10
  --experiment experiments/h10_unslop_grpo.yaml
  --output-dir "${RESULTS_ROOT}"
  --run-id "h10_${RUN_SUFFIX}"
  --backend "${BACKEND}"
  --model-name "${MODEL_NAME}"
  --device "${DEVICE}"
)
VERIFY_H11_CMD=(
  uv run sml verify-h11
  --experiment experiments/h11_provider_fp8_readiness.yaml
  --output-dir "${RESULTS_ROOT}"
  --run-id "h11_${RUN_SUFFIX}"
  --backend "${BACKEND}"
  --model-name "${MODEL_NAME}"
  --device "${DEVICE}"
)
VERIFY_H12_CMD=(
  uv run sml verify-h12
  --experiment experiments/h12_subspace_safe_merge.yaml
  --output-dir "${RESULTS_ROOT}"
  --run-id "h12_${RUN_SUFFIX}"
  --backend "${BACKEND}"
  --model-name "${MODEL_NAME}"
  --device "${DEVICE}"
)
EXPORT_HUB_CMD=(
  uv run sml export-hub
  --release-dir "${RELEASE_PATH}"
  --release-id "${RELEASE_ID}"
  --base-model "${MODEL_NAME}"
  --results-root "${RESULTS_ROOT}"
  --hf-namespace "${HF_NAMESPACE}"
)
REPORT_CMD=(
  uv run sml report
  --results-dir "${RESULTS_ROOT}"
  --output reports/sprint1/summary.md
)

BUILD_CMD=(
  uv run sml build-tenet
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --release-id "${RELEASE_ID}"
  --backend "${BACKEND}"
  --device "${DEVICE}"
  --grpo-steps "${GRPO_STEPS}"
  --n-prompts "${N_PROMPTS}"
  --seq-len "${SEQ_LEN}"
  --hf-namespace "${HF_NAMESPACE}"
)

mkdir -p "${RESULTS_ROOT}"

if [[ "${START_WATCHDOG}" == "1" && "${DRY_RUN}" == "0" ]]; then
  nohup ./scripts/remote/idle_watchdog.sh > "${RESULTS_ROOT}/idle_watchdog.log" 2>&1 &
fi

if [[ "${RUN_STAGE_BENCHMARKS}" == "1" ]]; then
  run_or_print "${VERIFY_H9_CMD[@]}"
  run_or_print "${VERIFY_H10_CMD[@]}"
  run_or_print "${VERIFY_H11_CMD[@]}"
  run_or_print "${VERIFY_H12_CMD[@]}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] ${BUILD_CMD[*]}"
  run_or_print "${EXPORT_HUB_CMD[@]}"
  run_or_print "${REPORT_CMD[@]}"
  if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    echo "[dry-run] huggingface-cli upload \"${HF_REPO_ID}\" \"${RELEASE_PATH}\" --repo-type model"
  fi
  exit 0
fi

echo "Building Tenet release '${RELEASE_ID}' (backend=${BACKEND}, device=${DEVICE})."
"${BUILD_CMD[@]}"

"${EXPORT_HUB_CMD[@]}"
"${REPORT_CMD[@]}"

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
  echo "Uploading '${RELEASE_PATH}' to '${HF_REPO_ID}'."
  huggingface-cli upload "${HF_REPO_ID}" "${RELEASE_PATH}" --repo-type model
fi
