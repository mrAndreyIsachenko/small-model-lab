#!/usr/bin/env bash
# Quantize Tenet-3B-Qwen2.5-Unslop-v1 into a CKA-aware mixed-precision GGUF.
# estimated_avg_bpw: 5.282
# default_quant: Q4_K_M
# q8_0_layers: 0, 5, 6, 7, 8, 9, 10, 12, 13, 14, 35
# q6_k_layers: 15, 19, 20, 21, 22, 23, 24, 25, 26
# boundary_layers: 0, 35
#
# Usage:
#   LLAMA_QUANTIZE_BIN=/path/to/llama-quantize ./quantize_cka_gguf.sh
#
# Environment overrides:
#   LLAMA_QUANTIZE_BIN  llama-quantize binary (default: llama-quantize)
#   CONVERT_HF_TO_GGUF  HF->GGUF converter (default: convert_hf_to_gguf.py)
#   INPUT_GGUF          f16 input (default: $SCRIPT_DIR/Tenet-3B-Qwen2.5-Unslop-v1-f16.gguf)
#   OUTPUT_GGUF         output file (default: $SCRIPT_DIR/Tenet-3B-Qwen2.5-Unslop-v1-cka-q4_k_m.gguf)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_QUANTIZE_BIN="${LLAMA_QUANTIZE_BIN:-llama-quantize}"
CONVERT_HF_TO_GGUF="${CONVERT_HF_TO_GGUF:-convert_hf_to_gguf.py}"
INPUT_GGUF="${INPUT_GGUF:-$SCRIPT_DIR/Tenet-3B-Qwen2.5-Unslop-v1-f16.gguf}"
OUTPUT_GGUF="${OUTPUT_GGUF:-$SCRIPT_DIR/Tenet-3B-Qwen2.5-Unslop-v1-cka-q4_k_m.gguf}"
DEFAULT_QUANT="${DEFAULT_QUANT:-Q4_K_M}"

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

echo "Quantizing $INPUT_GGUF -> $OUTPUT_GGUF (default: $DEFAULT_QUANT)"

"$LLAMA_QUANTIZE_BIN" \
  --tensor-type \
  blk.0.attn_output.weight=q8_0 \
  --tensor-type \
  blk.0.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.5.attn_output.weight=q8_0 \
  --tensor-type \
  blk.5.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.6.attn_output.weight=q8_0 \
  --tensor-type \
  blk.6.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.7.attn_output.weight=q8_0 \
  --tensor-type \
  blk.7.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.8.attn_output.weight=q8_0 \
  --tensor-type \
  blk.8.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.9.attn_output.weight=q8_0 \
  --tensor-type \
  blk.9.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.10.attn_output.weight=q8_0 \
  --tensor-type \
  blk.10.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.12.attn_output.weight=q8_0 \
  --tensor-type \
  blk.12.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.13.attn_output.weight=q8_0 \
  --tensor-type \
  blk.13.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.14.attn_output.weight=q8_0 \
  --tensor-type \
  blk.14.ffn_down.weight=q8_0 \
  --tensor-type \
  blk.15.attn_output.weight=q6_k \
  --tensor-type \
  blk.15.ffn_down.weight=q6_k \
  --tensor-type \
  blk.19.attn_output.weight=q6_k \
  --tensor-type \
  blk.19.ffn_down.weight=q6_k \
  --tensor-type \
  blk.20.attn_output.weight=q6_k \
  --tensor-type \
  blk.20.ffn_down.weight=q6_k \
  --tensor-type \
  blk.21.attn_output.weight=q6_k \
  --tensor-type \
  blk.21.ffn_down.weight=q6_k \
  --tensor-type \
  blk.22.attn_output.weight=q6_k \
  --tensor-type \
  blk.22.ffn_down.weight=q6_k \
  --tensor-type \
  blk.23.attn_output.weight=q6_k \
  --tensor-type \
  blk.23.ffn_down.weight=q6_k \
  --tensor-type \
  blk.24.attn_output.weight=q6_k \
  --tensor-type \
  blk.24.ffn_down.weight=q6_k \
  --tensor-type \
  blk.25.attn_output.weight=q6_k \
  --tensor-type \
  blk.25.ffn_down.weight=q6_k \
  --tensor-type \
  blk.26.attn_output.weight=q6_k \
  --tensor-type \
  blk.26.ffn_down.weight=q6_k \
  --tensor-type \
  blk.35.attn_output.weight=q8_0 \
  --tensor-type \
  blk.35.ffn_down.weight=q8_0 \
  "$INPUT_GGUF" \
  "$OUTPUT_GGUF" \
  "$DEFAULT_QUANT"

echo "Quantized GGUF written to: $OUTPUT_GGUF"
echo ""
echo "Ollama usage:"
echo "  ollama create Tenet-3B-Qwen2.5-Unslop-v1 -f $SCRIPT_DIR/Modelfile"
echo ""
echo "LM Studio usage:"
echo "  import $OUTPUT_GGUF via the My Models folder (or drag-and-drop)."
