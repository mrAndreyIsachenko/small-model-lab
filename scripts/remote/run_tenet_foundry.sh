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

set -euo pipefail

TENET_PROFILE="${TENET_PROFILE:-3b-mps}"

case "${TENET_PROFILE}" in
  3b-mps)
    PROFILE_MODEL="Qwen/Qwen2.5-3B-Instruct"
    PROFILE_RELEASE_ID="Tenet-3B-Qwen2.5-Unslop-v1"
    PROFILE_DEVICE="mps"
    PROFILE_BACKEND="llm"
    ;;
  8b-h100)
    PROFILE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    PROFILE_RELEASE_ID="Tenet-8B-Llama3.1-Unslop-v1"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="llm"
    ;;
  12b-h100)
    PROFILE_MODEL="mistralai/Mistral-Nemo-Instruct-2407"
    PROFILE_RELEASE_ID="Tenet-12B-Nemo-Unslop-v1"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="llm"
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

HF_REPO_ID="${HF_NAMESPACE}/${RELEASE_ID}"
RELEASE_PATH="${OUTPUT_DIR}/${RELEASE_ID}"

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

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] ${BUILD_CMD[*]}"
  if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    echo "[dry-run] huggingface-cli upload \"${HF_REPO_ID}\" \"${RELEASE_PATH}\" --repo-type model"
  fi
  exit 0
fi

echo "Building Tenet release '${RELEASE_ID}' (backend=${BACKEND}, device=${DEVICE})."
"${BUILD_CMD[@]}"

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
  echo "Uploading '${RELEASE_PATH}' to '${HF_REPO_ID}'."
  huggingface-cli upload "${HF_REPO_ID}" "${RELEASE_PATH}" --repo-type model
fi
