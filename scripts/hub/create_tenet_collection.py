"""Idempotent helper to create/update the Tenet v1 Hugging Face Collection."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

COLLECTION_TITLE = "Tenet v1 — Bi-Projected Unslop & FP8-Safe LLM Family"
# HF API enforces <= 150 chars on collection description.
COLLECTION_DESC = (
    "Dual-Anchor Lancet (H9), Gated Unslop GRPO (H10), vLLM FP8 (H11, Crest < 25), "
    "TIES Merge (H12), Streaming 32B/70B Surgery (H13)."
)
NAMESPACE = "mr-andrey-isachenko"

ITEMS: list[tuple[str, str, str]] = [
    (
        "mr-andrey-isachenko/fp8-abliteration-inspector",
        "space",
        "Interactive static Space to audit abliterated & Tenet checkpoints for FP8 E4M3 "
        "re-leakage, Row Crest-Factor < 25.0, and CKA-guided GGUF quantization.",
    ),
    (
        "mr-andrey-isachenko/Tenet-32B-Qwen2.5-Unslop-v1",
        "model",
        "32B Qwen2.5-32B-Instruct: H13 streaming shard surgery (max diff 7.45e-09), "
        "8/8 FP8 provider readiness (Crest 3.68 < 25.0), CKA-guided GGUF recipe.",
    ),
    (
        "mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1",
        "model",
        "12B Mistral-Nemo-Instruct-2407: 99.14% target suppression, 96.72% H9 KL reduction, "
        "0.0239 slop/refusal, 96.78% FP8 retention (Crest 10.65 < 25.0).",
    ),
    (
        "mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1",
        "model",
        "8B Llama-3.1-8B-Instruct: 99.30% target suppression, 89.89% H9 KL reduction, "
        "0.0410 slop/refusal, 99.36% FP8 retention (Crest 6.32 < 25.0).",
    ),
    (
        "mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1-CKA-GGUF",
        "model",
        "8B CKA-guided mixed-precision GGUF (i1-CKA-Q4_K_M & CKA-Q4_K_M, 5.29 bpw): "
        "3.012x subspace amp, 1.17% L2 err, PPL 9.0533, Mean KLD 0.01656, 6/6 SQL.",
    ),
    (
        "mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1",
        "model",
        "3B Qwen2.5-3B-Instruct: 96.94% target suppression, 99.41% H9 KL reduction "
        "(3.76 -> 0.022), 95.41% FP8 retention (Crest 17.27 < 25.0).",
    ),
    (
        "mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1",
        "model",
        "16K-param Nano audit & research artifact repo (renamed from small-model-lab): "
        "H1-H13 experiment manifests, Lancet recipes, and verification tensors.",
    ),
]


def _load_env_token() -> str:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip("\"'"))
    return os.environ["HF_TOKEN"]


def main() -> None:
    api = HfApi(token=_load_env_token())
    existing = [c for c in api.list_collections(owner=NAMESPACE) if c.title == COLLECTION_TITLE]
    if existing:
        col = api.get_collection(existing[0].slug)
    else:
        col = api.create_collection(
            title=COLLECTION_TITLE,
            namespace=NAMESPACE,
            description=COLLECTION_DESC,
            private=False,
        )

    existing_ids = {(it.item_id, it.item_type) for it in col.items}
    for item_id, item_type, note in ITEMS:
        if (item_id, item_type) not in existing_ids:
            api.add_collection_item(
                collection_slug=col.slug,
                item_id=item_id,
                item_type=item_type,
                note=note,
                exists_ok=True,
            )

    col = api.get_collection(col.slug)
    print(f"Collection URL: {col.url}")


if __name__ == "__main__":
    main()
