#!/usr/bin/env bash
# Quantize Tenet-32B-Qwen3-Unslop-v2 into a CKA-aware mixed-precision GGUF.
# estimated_avg_bpw: 4.952
# default_quant: Q4_K_M
# q8_0_layers: 0, 42, 46, 48, 63
# q6_k_layers: 24, 30, 45
# boundary_layers: 0, 63
#
# Usage:
#   LLAMA_QUANTIZE_BIN=/path/to/llama-quantize ./quantize_cka_gguf.sh [--imatrix <path>]
#
# Environment overrides:
#   LLAMA_QUANTIZE_BIN  llama-quantize binary (default: llama-quantize)
#   CONVERT_HF_TO_GGUF  HF->GGUF converter (default: convert_hf_to_gguf.py)
#   INPUT_GGUF          f16 input (default: $SCRIPT_DIR/Tenet-32B-Qwen3-Unslop-v2-f16.gguf)
#   OUTPUT_GGUF         output file (default: $SCRIPT_DIR/Tenet-32B-Qwen3-Unslop-v2-cka-q4_k_m.gguf)
#   IMATRIX_PATH        importance matrix GGUF for llama-quantize
#                       (--imatrix <path> overrides this)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_QUANTIZE_BIN="${LLAMA_QUANTIZE_BIN:-llama-quantize}"
CONVERT_HF_TO_GGUF="${CONVERT_HF_TO_GGUF:-convert_hf_to_gguf.py}"
INPUT_GGUF="${INPUT_GGUF:-$SCRIPT_DIR/Tenet-32B-Qwen3-Unslop-v2-f16.gguf}"
OUTPUT_GGUF="${OUTPUT_GGUF:-$SCRIPT_DIR/Tenet-32B-Qwen3-Unslop-v2-cka-q4_k_m.gguf}"
DEFAULT_QUANT="${DEFAULT_QUANT:-Q4_K_M}"
IMATRIX_PATH="${IMATRIX_PATH:-}"
IMATRIX_ARGS=()

# Command-line arguments; --imatrix overrides the IMATRIX_PATH env var.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --imatrix)
      if [[ $# -lt 2 ]]; then
        echo "error: --imatrix requires a path argument" >&2
        exit 1
      fi
      IMATRIX_PATH="$2"
      shift 2
      ;;
    --imatrix=*)
      IMATRIX_PATH="${1#--imatrix=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $(basename "$0") [--imatrix <path>]"
      echo "  --imatrix <path>  importance matrix GGUF for llama-quantize"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$IMATRIX_PATH" ]]; then
  if [[ ! -f "$IMATRIX_PATH" ]]; then
    echo "error: IMATRIX_PATH not found at $IMATRIX_PATH" >&2
    exit 1
  fi
  IMATRIX_ARGS=(--imatrix "$IMATRIX_PATH")
fi

# Optional auto-conversion: build the f16 GGUF from the HF checkpoint
# shipped next to this script when INPUT_GGUF is missing.
if [[ ! -f "$INPUT_GGUF" ]]; then
  if command -v "$CONVERT_HF_TO_GGUF" >/dev/null 2>&1; then
    echo "INPUT_GGUF not found; converting HF checkpoint in $SCRIPT_DIR"
    (cd "$SCRIPT_DIR" && "$CONVERT_HF_TO_GGUF" "$SCRIPT_DIR" \
      --outfile "$INPUT_GGUF" --outtype f16)
  else
    echo "error: INPUT_GGUF not found at $INPUT_GGUF and" >&2
    echo "error: CONVERT_HF_TO_GGUF ($CONVERT_HF_TO_GGUF) is not in PATH." >&2
    exit 1
  fi
fi

if [[ -n "$IMATRIX_PATH" ]]; then
  echo "Quantizing $INPUT_GGUF -> $OUTPUT_GGUF (default: $DEFAULT_QUANT, imatrix: $IMATRIX_PATH)"
else
  echo "Quantizing $INPUT_GGUF -> $OUTPUT_GGUF (default: $DEFAULT_QUANT)"
fi

"$LLAMA_QUANTIZE_BIN" \
  ${IMATRIX_ARGS[@]+"${IMATRIX_ARGS[@]}"} \
  --tensor-type \
  blk.0.attn_output.weight=q8_0 \
  --tensor-type \
  blk.0.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.24.attn_output.weight=q6_k \
  --tensor-type \
  blk.24.ffn_down.weight=q6_k \
  --tensor-type \
  blk.30.attn_output.weight=q6_k \
  --tensor-type \
  blk.30.ffn_down.weight=q6_k \
  --tensor-type \
  blk.42.attn_output.weight=q8_0 \
  --tensor-type \
  blk.42.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.45.attn_output.weight=q6_k \
  --tensor-type \
  blk.45.ffn_down.weight=q6_k \
  --tensor-type \
  blk.46.attn_output.weight=q8_0 \
  --tensor-type \
  blk.46.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.48.attn_output.weight=q8_0 \
  --tensor-type \
  blk.48.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.63.attn_output.weight=q8_0 \
  --tensor-type \
  blk.63.ffn_down.weight=q8_0 \
  "$INPUT_GGUF" \
  "$OUTPUT_GGUF" \
  "$DEFAULT_QUANT"

echo "Quantized GGUF written to: $OUTPUT_GGUF"
echo ""
echo "Ollama usage:"
echo "  ollama create Tenet-32B-Qwen3-Unslop-v2 -f $SCRIPT_DIR/Modelfile"
echo ""
echo "LM Studio usage:"
echo "  import $OUTPUT_GGUF via the My Models folder (or drag-and-drop)."
