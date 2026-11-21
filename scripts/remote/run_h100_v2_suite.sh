#!/usr/bin/env bash
# Master H100 runner for Tenet v1/v2 3-way external benchmarks and Tenet v2 14B/32B foundry releases.
#
# Usage:
#   ./scripts/remote/run_h100_v2_suite.sh
#
# Optional environment knobs:
#   HF_NAMESPACE        HF Hub namespace                 (default: mr-andrey-isachenko)
#   BATCH_SIZE          lm-eval batch size on CUDA       (default: 16)
#   LM_EVAL_LIMIT       lm-eval sample limit             (default: 40)
#   SEED                evaluation seed                  (default: 1337)
#   FORCE_RERUN         1 to re-run existing evals       (default: 0)
#   KEEP_LOCAL_WEIGHTS  1 to retain local .safetensors/.gguf after upload (default: 0)
#   SKIP_LLAMA_BUILD    1 to skip llama.cpp build check  (default: 0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

HF_NAMESPACE="${HF_NAMESPACE:-mr-andrey-isachenko}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-40}"
SEED="${SEED:-1337}"
FORCE_RERUN="${FORCE_RERUN:-0}"
KEEP_LOCAL_WEIGHTS="${KEEP_LOCAL_WEIGHTS:-0}"
SKIP_LLAMA_BUILD="${SKIP_LLAMA_BUILD:-0}"

# ---------------------------------------------------------------------------
# 1. Preflight checks: nvidia-smi, .env (HF_TOKEN), idle_watchdog.sh
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script must run on an NVIDIA GPU host (H100)." >&2
  exit 1
fi
echo "=== GPU Preflight ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Add HF_TOKEN=... to ${REPO_ROOT}/.env or export it." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/results"
WATCHDOG_PID=""
cleanup() {
  if [[ -n "${WATCHDOG_PID}" ]] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${WATCHDOG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -x "${REPO_ROOT}/scripts/remote/idle_watchdog.sh" ]]; then
  echo "=== Starting idle_watchdog.sh in background ==="
  nohup "${REPO_ROOT}/scripts/remote/idle_watchdog.sh" > "${REPO_ROOT}/results/idle_watchdog.log" 2>&1 &
  WATCHDOG_PID=$!
fi

# ---------------------------------------------------------------------------
# 2. Environment sync & llama.cpp binary setup
# ---------------------------------------------------------------------------
echo "=== Syncing uv environment ==="
uv sync --frozen --extra data --extra parity --extra telemetry --extra eval

LLAMA_DIR="${REPO_ROOT}/.tools/llama.cpp"
if [[ "${SKIP_LLAMA_BUILD}" != "1" ]]; then
  need_llama=0
  for bin in llama-quantize llama-imatrix llama-perplexity llama-server; do
    if ! command -v "${bin}" >/dev/null 2>&1 && [[ ! -x "${LLAMA_DIR}/${bin}" ]]; then
      need_llama=1
      break
    fi
  done

  if [[ "${need_llama}" == "1" ]]; then
    echo "=== Building llama.cpp (llama-quantize, llama-imatrix, llama-perplexity, llama-server) ==="
    mkdir -p "${REPO_ROOT}/.tools"
    if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
      git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
    fi
    cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF
    cmake --build "${LLAMA_DIR}/build" --target llama-quantize llama-imatrix llama-perplexity llama-server -j"$(nproc)"
    for bin in llama-quantize llama-imatrix llama-perplexity llama-server; do
      if [[ -x "${LLAMA_DIR}/build/bin/${bin}" ]]; then
        ln -sf "${LLAMA_DIR}/build/bin/${bin}" "${LLAMA_DIR}/${bin}"
      fi
    done
  else
    echo "=== llama.cpp binaries already available ==="
  fi
fi
export PATH="${LLAMA_DIR}:${PATH}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_external_eval_if_needed() {
  local release_id="$1"
  local base_model="$2"
  local out_json="${REPO_ROOT}/results/external_evals/${release_id}.json"

  if [[ -f "${out_json}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[skip] ${out_json} already exists (set FORCE_RERUN=1 to rerun)."
    return 0
  fi

  echo "=== Running 3-way external benchmark: ${release_id} (${base_model}) ==="
  uv run python "${REPO_ROOT}/scripts/eval/run_external_benchmark_3way.py" \
    --release-id "${release_id}" \
    --base-model "${base_model}" \
    --tenet-repo "${HF_NAMESPACE}/${release_id}" \
    --limit "${LM_EVAL_LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --seed "${SEED}"
}

run_foundry_v2_and_eval() {
  local profile="$1"
  local base_model="$2"
  local release_id="$3"
  local release_dir="${REPO_ROOT}/results/tenet_releases/${release_id}"

  if [[ ! -f "${release_dir}/foundry_summary.json" || "${FORCE_RERUN}" == "1" ]]; then
    echo "=== Running Tenet Foundry v2: ${profile} (${base_model} -> ${release_id}) ==="
    TENET_PROFILE="${profile}" \
    HF_NAMESPACE="${HF_NAMESPACE}" \
    UPLOAD_TO_HF=1 \
    DROP_LOCAL_WEIGHTS=0 \
    RUN_EXTERNAL_EVAL=0 \
    ALLOW_GATE_FAIL="${ALLOW_GATE_FAIL:-1}" \
      "${REPO_ROOT}/scripts/remote/run_tenet_foundry.sh"
  else
    echo "[skip] ${release_dir}/foundry_summary.json already exists (set FORCE_RERUN=1 to rerun)."
  fi

  run_external_eval_if_needed "${release_id}" "${base_model}"

  if [[ "${KEEP_LOCAL_WEIGHTS}" != "1" ]]; then
    echo "=== Dropping heavy local weights for ${release_id} (uploaded to HF) ==="
    rm -f "${release_dir}"/model*.safetensors "${release_dir}"/*.gguf
    rm -rf "${REPO_ROOT}/results/tenet_materialized/${release_id}"
  fi
}

# ---------------------------------------------------------------------------
# Step A: 3-way External Benchmarks on CUDA for already-released models
# ---------------------------------------------------------------------------
echo "=== Step A: 3-way External Benchmarks (CUDA, batch_size=${BATCH_SIZE}) ==="
run_external_eval_if_needed "Tenet-8B-Qwen3-Unslop-v2" "Qwen/Qwen3-8B"
run_external_eval_if_needed "Tenet-12B-Nemo-Unslop-v1" "mistralai/Mistral-Nemo-Instruct-2407"
run_external_eval_if_needed "Tenet-32B-Qwen2.5-Unslop-v1" "Qwen/Qwen2.5-32B-Instruct"

# ---------------------------------------------------------------------------
# Step B: Foundry v2 (14B & 32B) + HF Upload + 3-way External Eval + Collection
# ---------------------------------------------------------------------------
echo "=== Step B: Foundry v2 (14B & 32B) + Upload + 3-way External Eval ==="
run_foundry_v2_and_eval "14b-qwen3-v2-h100" "Qwen/Qwen3-14B" "Tenet-14B-Qwen3-Unslop-v2"
run_foundry_v2_and_eval "32b-qwen3-v2-h100" "Qwen/Qwen3-32B" "Tenet-32B-Qwen3-Unslop-v2"

echo "=== Updating Hugging Face Tenet v1 & v2 Collections ==="
uv run python "${REPO_ROOT}/scripts/hub/create_tenet_collection.py"

echo "=== H100 v2 Suite Complete ==="
