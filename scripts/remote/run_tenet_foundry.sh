#!/usr/bin/env bash
# Build a Tenet release bundle on a remote foundry box (MPS laptop or H100 node).
#
# Usage:
#   TENET_PROFILE=3b-mps ./scripts/remote/run_tenet_foundry.sh
#
# Environment knobs (all optional):
#   TENET_PROFILE  3b-mps | 4b-qwen3-v2-h100 | 8b-h100 | 12b-h100 | 32b-h100 | 70b-h100
#                  | 8b-qwen3-v2-h100 | 14b-qwen3-v2-h100 | 32b-qwen3-v2-h100  (default: 3b-mps)
#   MODEL_NAME / RELEASE_ID / DEVICE / BACKEND  override the profile values
#   OUTPUT_DIR     release output root          (default: results/tenet_releases)
#   HF_NAMESPACE   HF Hub namespace             (default: tenet-lab)
#   UPLOAD_TO_HF   1 to upload the bundle       (default: 0)
#   MATERIALIZE    1 to run Tier 3 materialization via `sml materialize-tenet` (default: 0)
#   MATERIALIZE_OUTPUT_DIR  materialized checkpoint output dir (default: results/tenet_materialized)
#   LORA_PATH      optional Tenet LoRA adapter path for materialization
#   SQL_LORA_PATH  optional SQL LoRA adapter to merge during materialization
#   FP8_EXPORT     1 to also export FP8 shards (default: 0)
#   DROP_LOCAL_WEIGHTS  1 to drop local materialized shards after a successful Hub push (default: 0)
#   ALLOW_GATE_FAIL  1 to upload and exit 0 even if release gates fail (default: 0)
#   SMOKE_ONLY     1 for a tiny CPU smoke run   (default: 0)
#   DRY_RUN        1 to print the command only  (default: 0)
#   RUN_SUFFIX     run-id suffix for benchmarks (default: per-profile suffix)
#   RESULTS_ROOT   benchmark/results root       (default: results)
#   RUN_STAGE_BENCHMARKS  1 to run verify-h9..h13 first (default: 1, or 0 with SMOKE_ONLY=1)
#   RUN_EXTERNAL_EVAL  1 to run `sml eval-external` (default: 1 when RUN_STAGE_BENCHMARKS=1, else 0)
#   RUN_VLLM_SMOKE 1 to serve the release with vLLM FP8 and run `sml vllm-smoke` (default: 0)
#   VLLM_MODEL     model path/repo served by vLLM (default: MODEL_NAME)
#   VLLM_HOST      vLLM serve host               (default: 127.0.0.1)
#   VLLM_PORT      vLLM serve port               (default: 8000)
#   VLLM_MAX_MODEL_LEN  vLLM max model len       (default: 8192)
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
  32b-h100)
    PROFILE_MODEL="Qwen/Qwen2.5-32B-Instruct"
    PROFILE_RELEASE_ID="Tenet-32B-Qwen2.5-Unslop-v1"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="llm"
    PROFILE_RUN_SUFFIX="qwen32b_h100"
    ;;
  70b-h100)
    PROFILE_MODEL="meta-llama/Llama-3.3-70B-Instruct"
    PROFILE_RELEASE_ID="Tenet-70B-Llama3.3-Unslop-v1"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="llm"
    PROFILE_RUN_SUFFIX="llama70b_h100"
    ;;
  4b-qwen3-v2-h100)
    PROFILE_MODEL="Qwen/Qwen3-4B"
    PROFILE_RELEASE_ID="Tenet-4B-Qwen3-Unslop-v2"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="v2"
    PROFILE_RUN_SUFFIX="qwen3_4b_v2_h100"
    ;;
  8b-qwen3-v2-h100)
    PROFILE_MODEL="Qwen/Qwen3-8B"
    PROFILE_RELEASE_ID="Tenet-8B-Qwen3-Unslop-v2"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="v2"
    PROFILE_RUN_SUFFIX="qwen3_8b_v2_h100"
    ;;
  14b-qwen3-v2-h100)
    PROFILE_MODEL="Qwen/Qwen3-14B"
    PROFILE_RELEASE_ID="Tenet-14B-Qwen3-Unslop-v2"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="v2"
    PROFILE_RUN_SUFFIX="qwen3_14b_v2_h100"
    ;;
  32b-qwen3-v2-h100)
    PROFILE_MODEL="Qwen/Qwen3-32B"
    PROFILE_RELEASE_ID="Tenet-32B-Qwen3-Unslop-v2"
    PROFILE_DEVICE="cuda"
    PROFILE_BACKEND="v2"
    PROFILE_RUN_SUFFIX="qwen3_32b_v2_h100"
    ;;
  *)
    echo "Unknown TENET_PROFILE '${TENET_PROFILE}' (expected 3b-mps, 4b-qwen3-v2-h100, 8b-qwen3-v2-h100, 14b-qwen3-v2-h100, 32b-qwen3-v2-h100, 8b-h100, 12b-h100, 32b-h100, or 70b-h100)." >&2
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
MATERIALIZE="${MATERIALIZE:-0}"
MATERIALIZE_OUTPUT_DIR="${MATERIALIZE_OUTPUT_DIR:-results/tenet_materialized}"
LORA_PATH="${LORA_PATH:-}"
SQL_LORA_PATH="${SQL_LORA_PATH:-}"
FP8_EXPORT="${FP8_EXPORT:-0}"
DROP_LOCAL_WEIGHTS="${DROP_LOCAL_WEIGHTS:-0}"
ALLOW_GATE_FAIL="${ALLOW_GATE_FAIL:-0}"
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

RUN_VLLM_SMOKE="${RUN_VLLM_SMOKE:-0}"
VLLM_MODEL="${VLLM_MODEL:-${MODEL_NAME}}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
if [[ -z "${RUN_EXTERNAL_EVAL:-}" ]]; then
  if [[ "${RUN_STAGE_BENCHMARKS}" == "1" ]]; then
    RUN_EXTERNAL_EVAL=1
  else
    RUN_EXTERNAL_EVAL=0
  fi
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

# Tenet v2 runs skip the legacy build-tenet pipeline entirely: the v2 foundry
# script owns all four stages (Lancet -> GRPO -> GGUF -> gates) plus Hub upload.
if [[ "${BACKEND}" == "v2" && "${SMOKE_ONLY}" != "1" ]]; then
  V2_CMD=(
    .venv/bin/python scripts/run_tenet_v2_foundry.py
    --model-name "${MODEL_NAME}"
    --release-name "${RELEASE_ID}"
    --releases-dir "${OUTPUT_DIR}"
  )
  if [[ "${UPLOAD_TO_HF}" == "1" || "${DROP_LOCAL_WEIGHTS}" == "1" ]]; then
    V2_CMD+=(
      --upload-to-hf
      --drop-local-weights
      --hf-namespace "${HF_NAMESPACE}"
    )
  fi
  if [[ "${ALLOW_GATE_FAIL}" == "1" ]]; then
    V2_CMD+=(--allow-gate-fail)
  fi
  echo "Delegating Tenet v2 release '${RELEASE_ID}' to scripts/run_tenet_v2_foundry.py."
  run_or_print "${V2_CMD[@]}"
  exit 0
fi

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

if [[ "${MATERIALIZE}" == "1" ]]; then
  MATERIALIZE_CMD=(
    uv run sml materialize-tenet
    --release-id "${RELEASE_ID}"
    --recipe-dir "${RELEASE_PATH}"
    --output-dir "${MATERIALIZE_OUTPUT_DIR}"
    --backend "${BACKEND}"
    --device "${DEVICE}"
    --hf-namespace "${HF_NAMESPACE}"
  )
  if [[ -n "${LORA_PATH}" ]]; then
    MATERIALIZE_CMD+=(--lora-path "${LORA_PATH}")
  fi
  if [[ -n "${SQL_LORA_PATH}" ]]; then
    MATERIALIZE_CMD+=(--sql-lora-path "${SQL_LORA_PATH}")
  fi
  if [[ "${FP8_EXPORT}" == "1" ]]; then
    MATERIALIZE_CMD+=(--fp8-export)
  fi
  if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    MATERIALIZE_CMD+=(--push-to-hub)
  fi
  if [[ "${DROP_LOCAL_WEIGHTS}" == "1" ]]; then
    MATERIALIZE_CMD+=(--drop-local-weights)
  fi
fi

mkdir -p "${RESULTS_ROOT}"

if [[ "${START_WATCHDOG}" == "1" && "${DRY_RUN}" == "0" ]]; then
  nohup ./scripts/remote/idle_watchdog.sh > "${RESULTS_ROOT}/idle_watchdog.log" 2>&1 &
fi

if [[ "${RUN_STAGE_BENCHMARKS}" == "1" ]]; then
  run_or_print "${VERIFY_H9_CMD[@]}"
  run_or_print "${VERIFY_H10_CMD[@]}"
  run_or_print "${VERIFY_H11_CMD[@]}"
  run_or_print "${VERIFY_H12_CMD[@]}"
  run_or_print uv run sml verify-h13 --experiment experiments/h13_streaming_shard_surgery.yaml --output-dir "${RESULTS_ROOT}" --run-id "h13_${RUN_SUFFIX}" --backend "${BACKEND}" --model-name "${MODEL_NAME}" --device "${DEVICE}"
fi

if [[ "${RUN_EXTERNAL_EVAL}" == "1" ]]; then
  run_or_print uv run sml eval-external --output-dir "${RESULTS_ROOT}/external_evals" --run-id "ext_${RUN_SUFFIX}" --release-dir "${RELEASE_PATH}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] ${BUILD_CMD[*]}"
  run_or_print "${EXPORT_HUB_CMD[@]}"
  run_or_print "${REPORT_CMD[@]}"
  if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
    echo "[dry-run] huggingface-cli upload \"${HF_REPO_ID}\" \"${RELEASE_PATH}\" --repo-type model"
  fi
  if [[ "${MATERIALIZE}" == "1" ]]; then
    echo "[dry-run] ${MATERIALIZE_CMD[*]}"
  fi
  if [[ "${RUN_VLLM_SMOKE}" == "1" ]]; then
    echo "[dry-run] vllm serve \"${VLLM_MODEL}\" --quantization fp8 --dtype bfloat16 --host \"${VLLM_HOST}\" --port \"${VLLM_PORT}\" --max-model-len \"${VLLM_MAX_MODEL_LEN}\" --enable-auto-tool-choice"
    echo "[dry-run] uv run sml vllm-smoke --base-url \"http://${VLLM_HOST}:${VLLM_PORT}/v1\" --model-name \"${VLLM_MODEL}\" --output-json \"${RESULTS_ROOT}/vllm_fp8_smoke.json\""
  fi
  exit 0
fi

echo "Building Tenet release '${RELEASE_ID}' (backend=${BACKEND}, device=${DEVICE})."
"${BUILD_CMD[@]}"

"${EXPORT_HUB_CMD[@]}"
"${REPORT_CMD[@]}"

if [[ "${MATERIALIZE}" == "1" ]]; then
  echo "Materializing Tenet checkpoint for '${RELEASE_ID}' into '${MATERIALIZE_OUTPUT_DIR}'."
  "${MATERIALIZE_CMD[@]}"
fi

if [[ "${RUN_VLLM_SMOKE}" == "1" ]]; then
  VLLM_PID=""
  cleanup_vllm() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill "${VLLM_PID}" 2>/dev/null || true
      wait "${VLLM_PID}" 2>/dev/null || true
    fi
  }
  trap cleanup_vllm EXIT

  echo "Starting vLLM FP8 smoke server for '${VLLM_MODEL}' on ${VLLM_HOST}:${VLLM_PORT}."
  vllm serve "${VLLM_MODEL}" \
    --quantization fp8 \
    --dtype bfloat16 \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --enable-auto-tool-choice \
    > "${RESULTS_ROOT}/vllm_serve.log" 2>&1 &
  VLLM_PID=$!

  VLLM_READY=0
  for _ in $(seq 1 120); do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "ERROR: vLLM server exited unexpectedly; see ${RESULTS_ROOT}/vllm_serve.log" >&2
      exit 1
    fi
    if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      VLLM_READY=1
      break
    fi
    sleep 5
  done

  if [[ "${VLLM_READY}" != "1" ]]; then
    echo "ERROR: vLLM server did not become ready within 600s; see ${RESULTS_ROOT}/vllm_serve.log" >&2
    exit 1
  fi

  uv run sml vllm-smoke \
    --base-url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --model-name "${VLLM_MODEL}" \
    --output-json "${RELEASE_PATH}/vllm_fp8_smoke.json"

  cleanup_vllm
  trap - EXIT
fi

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
  echo "Uploading '${RELEASE_PATH}' to '${HF_REPO_ID}'."
  huggingface-cli upload "${HF_REPO_ID}" "${RELEASE_PATH}" --repo-type model
fi
