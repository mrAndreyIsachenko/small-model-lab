"""Idempotent helper to create/update the Tenet v1 & Tenet v2 Hugging Face Collections."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

NAMESPACE = "mr-andrey-isachenko"

V1_TITLE = "Tenet v1 — Bi-Projected Unslop & FP8-Safe LLM Family"
# HF API enforces <= 150 chars on collection description.
V1_DESC = (
    "Dual-Anchor Lancet (H9), Gated Unslop GRPO (H10), vLLM FP8 (H11, Crest < 25), "
    "TIES Merge (H12), Streaming 32B/70B Surgery (H13)."
)

V1_ITEMS: list[tuple[str, str, str]] = [
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
        "Majors Champion (Model Fight Card 16, 50 HP, 24/24 artifacts, 7/7 Peer Eval hits). "
        "8B Llama-3.1-8B: 99.30% suppression, 89.89% H9 KL reduction, "
        "99.36% FP8 retention (Crest 6.32).",
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
        "Minors Champion (Model Fight Card 16, 90 HP, 21/22 escapes, 22/22 artifacts). "
        "3B Qwen2.5-3B: 96.94% suppression, 99.41% H9 KL reduction (3.76->0.022), "
        "95.41% FP8 retention.",
    ),
    (
        "mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1",
        "model",
        "16K-param Nano audit & research artifact repo (renamed from small-model-lab): "
        "H1-H13 experiment manifests, Lancet recipes, and verification tensors.",
    ),
]

V1_REMOVE_IDS: set[tuple[str, str]] = {
    ("mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2", "model"),
    ("mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2", "model"),
}

V2_TITLE = "Tenet v2 — Thinking-Safe Tri-Anchor Unslop & Domain CKA-GGUF"
V2_DESC = (
    "Tri-Anchor CoT Lancet (H15), Anti-Padding GRPO (H16), "
    "24-Seed Fight Card Gate (H17, 100 HP), Domain i1-CKA-Q4_K_M GGUF (H14+)."
)

V2_REQUIRED_ITEMS: list[tuple[str, str, str]] = [
    (
        "mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2",
        "model",
        "Tenet v2 8B (Qwen3-8B, <think> CoT Tri-Anchor Lancet H15 + Length-Calibrated GRPO H16 "
        "+ Domain i1-CKA-Q4_K_M GGUF): 10/10 gates, 100 HP (24/24), 100% CoT integrity, "
        "KLD 0.0131.",
    ),
    (
        "mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2",
        "model",
        "Tenet v2 4B (Qwen3-4B, <think> CoT Tri-Anchor Lancet H15 + Length-Calibrated GRPO H16 "
        "+ Domain i1-CKA-Q4_K_M GGUF): 10/10 gates, 100 HP (24/24), 100% CoT integrity, "
        "0% collapse.",
    ),
    (
        "mr-andrey-isachenko/fp8-abliteration-inspector",
        "space",
        "Interactive static Space to audit abliterated & Tenet checkpoints for FP8 E4M3 "
        "re-leakage, Row Crest-Factor < 25.0, and CKA-guided GGUF quantization.",
    ),
]

V2_OPTIONAL_ITEMS: list[tuple[str, str, str]] = [
    (
        "mr-andrey-isachenko/Tenet-14B-Qwen3-Unslop-v2",
        "model",
        "Tenet v2 14B (Qwen3-14B, <think> CoT Tri-Anchor Lancet H15 + Length-Calibrated GRPO H16 "
        "+ Domain i1-CKA-Q4_K_M GGUF): H100 foundry release.",
    ),
    (
        "mr-andrey-isachenko/Tenet-32B-Qwen3-Unslop-v2",
        "model",
        "Tenet v2 32B (Qwen3-32B, <think> CoT Tri-Anchor Lancet H15 + Length-Calibrated GRPO H16 "
        "+ Domain i1-CKA-Q4_K_M GGUF): H100 streaming shard release.",
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


def _repo_exists(api: HfApi, repo_id: str, repo_type: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return True
    except (RepositoryNotFoundError, HfHubHTTPError):
        return False


def _upsert_collection(
    api: HfApi,
    *,
    title: str,
    description: str,
    items: list[tuple[str, str, str]],
    remove_ids: set[tuple[str, str]] | None = None,
) -> str:
    existing = [c for c in api.list_collections(owner=NAMESPACE) if c.title == title]
    if existing:
        col = api.get_collection(existing[0].slug)
    else:
        col = api.create_collection(
            title=title,
            namespace=NAMESPACE,
            description=description,
            private=False,
        )

    if remove_ids:
        for it in list(col.items):
            if (it.item_id, it.item_type) in remove_ids:
                api.delete_collection_item(
                    collection_slug=col.slug,
                    item_object_id=it.item_object_id,
                )
        col = api.get_collection(col.slug)

    existing_ids = {(it.item_id, it.item_type) for it in col.items}
    for item_id, item_type, note in items:
        if (item_id, item_type) not in existing_ids:
            api.add_collection_item(
                collection_slug=col.slug,
                item_id=item_id,
                item_type=item_type,
                note=note,
                exists_ok=True,
            )
        else:
            it = next(
                item
                for item in col.items
                if item.item_id == item_id and item.item_type == item_type
            )
            api.update_collection_item(
                collection_slug=col.slug,
                item_object_id=it.item_object_id,
                note=note,
            )

    col = api.get_collection(col.slug)
    return str(col.url)


def main() -> None:
    api = HfApi(token=_load_env_token())

    v1_url = _upsert_collection(
        api,
        title=V1_TITLE,
        description=V1_DESC,
        items=V1_ITEMS,
        remove_ids=V1_REMOVE_IDS,
    )
    print(f"Tenet v1 Collection URL: {v1_url}")

    v2_items = list(V2_REQUIRED_ITEMS)
    for item_id, item_type, note in V2_OPTIONAL_ITEMS:
        if _repo_exists(api, item_id, item_type):
            v2_items.append((item_id, item_type, note))

    v2_url = _upsert_collection(
        api,
        title=V2_TITLE,
        description=V2_DESC,
        items=v2_items,
    )
    print(f"Tenet v2 Collection URL: {v2_url}")


if __name__ == "__main__":
    main()
