import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.error import URLError

FORBIDDEN_RAW_BODY_KEYS = {
    "artifact_body",
    "artifact_bodies",
    "raw_artifact_body",
    "raw_artifact_bodies",
    "raw_body",
    "raw_bodies",
    "raw_rows",
    "raw_db_rows",
    "db_rows",
    "raw_records",
    "credentials",
    "secrets",
    "tokens",
    "private_key",
    "api_key",
}

OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"

# Tenet-targeted OpenRouter upstream providers for FP8 onboarding.
OPENROUTER_TENET_PROVIDERS = ["Featherless", "Chutes", "Novita", "DeepInfra"]

# Deterministic FP8 smoke prompts: JSON format check, simple arithmetic, SQL
# read-only check. The matching reference responses are the canonical answers
# a well-behaved (BF16 or FP8) model should produce.
VLLM_FP8_SMOKE_PROMPTS: tuple[str, ...] = (
    'Respond with a single valid JSON object with keys "status" (string) and '
    '"count" (integer). Output the JSON object only.',
    "Compute 17 * 23 and reply with the number only.",
    (
        "You are auditing a SQL session that must stay read-only. "
        'Is the statement "DROP TABLE users;" allowed in a read-only session? '
        "Answer with yes or no only."
    ),
)

VLLM_FP8_SMOKE_REFERENCE_RESPONSES: tuple[str, ...] = (
    '{"status": "ok", "count": 1}',
    "391",
    "no",
)


class GatewayError(ValueError):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def _default_upstream_url() -> str:
    return (
        os.environ.get("HARNESS_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or OPENROUTER_DEFAULT_URL
    )


@dataclass(frozen=True)
class GatewayConfig:
    upstream_url: str = field(default_factory=_default_upstream_url)
    api_key: str = ""
    model_allowlist: tuple[str, ...] = (
        "qwen/qwen3-coder",
        "moonshotai/kimi-k2.5",
        "qwen/qwen-2.5-72b-instruct",
        "deepseek/deepseek-r1",
    )
    timeout_seconds: int = 120
    max_body_bytes: int = 131072
    max_message_chars: int = 60000
    app_title: str = "Tenet Lab (SML)"
    app_referer: str = "https://github.com/mrAndreyIsachenko/tenet-lab"
    task_id: str | None = None

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        # Read upstream URL
        upstream_url = os.environ.get("SML_GATEWAY_UPSTREAM_URL") or _default_upstream_url()

        # Read API key from environment or file
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("SML_GATEWAY_API_KEY", "")
        if not api_key:
            api_key_file = os.environ.get("SML_GATEWAY_API_KEY_FILE")
            if api_key_file and os.path.exists(api_key_file):
                with open(api_key_file) as f:
                    api_key = f.read().strip()

        # Read model allowlist
        model_allowlist_str = os.environ.get("SML_GATEWAY_MODEL_ALLOWLIST")
        if model_allowlist_str:
            model_allowlist = tuple(model.strip() for model in model_allowlist_str.split(","))
        else:
            model_allowlist = cls.model_allowlist

        return cls(upstream_url=upstream_url, api_key=api_key, model_allowlist=model_allowlist)


def find_forbidden_key_paths(value: Any, prefix: str = "body") -> list[str]:
    """Recursively scan dictionaries and lists for forbidden keys."""
    paths = []

    if isinstance(value, dict):
        for key, val in value.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if key in FORBIDDEN_RAW_BODY_KEYS:
                paths.append(current_path)
            paths.extend(find_forbidden_key_paths(val, current_path))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            current_path = f"{prefix}[{i}]"
            paths.extend(find_forbidden_key_paths(item, current_path))

    return paths


SECRET_LEAK_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api_key|secret_key|password|access_token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{8,}"),
)


def find_secret_leak_paths(value: Any, prefix: str = "body") -> list[str]:
    """Recursively scan dictionaries, lists, and string values for secret patterns."""
    paths = []

    if isinstance(value, dict):
        for key, val in value.items():
            current_path = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(find_secret_leak_paths(val, current_path))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            current_path = f"{prefix}[{i}]"
            paths.extend(find_secret_leak_paths(item, current_path))
    elif isinstance(value, str) and any(pattern.search(value) for pattern in SECRET_LEAK_PATTERNS):
        paths.append(prefix)

    return paths


def validate_chat_payload(payload: dict[str, Any], config: GatewayConfig) -> dict[str, Any]:
    """Validate chat payload according to gateway rules."""
    # Validate model
    model = payload.get("model")
    if not model or not isinstance(model, str):
        raise GatewayError(
            "model is required and must be a string", status=400, code="invalid_model"
        )

    if model not in config.model_allowlist:
        raise GatewayError(f"model '{model}' is not allowed", status=403, code="model_not_allowed")

    # Validate messages
    messages = payload.get("messages")
    if not messages or not isinstance(messages, list):
        raise GatewayError(
            "messages is required and must be a non-empty list", status=400, code="invalid_messages"
        )

    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            raise GatewayError(
                "each message must be a dictionary", status=400, code="invalid_message_format"
            )

        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise GatewayError(
                f"message role '{role}' is not allowed, must be one of system/user/assistant/tool",
                status=400,
                code="invalid_message_role",
            )

        content = msg.get("content")
        if not content:
            raise GatewayError("message content is required", status=400, code="missing_content")

        if isinstance(content, str):
            total_chars += len(content)
        else:
            raise GatewayError(
                "message content must be a string", status=400, code="invalid_content_type"
            )

    # Check for streaming
    if payload.get("stream") is True:
        raise GatewayError("streaming is not supported", status=400, code="streaming_not_supported")

    # Check for forbidden keys
    forbidden_paths = find_forbidden_key_paths(payload)
    if forbidden_paths:
        raise GatewayError(
            f"forbidden keys detected: {', '.join(forbidden_paths)}",
            status=403,
            code="raw_body_fields_not_allowed",
        )

    # Check for secret patterns leaking through the payload
    secret_paths = find_secret_leak_paths(payload)
    if secret_paths:
        raise GatewayError(
            f"potential secret leak detected: {', '.join(secret_paths)}",
            status=403,
            code="secret_leak_detected",
        )

    # Check size limits
    if total_chars > config.max_message_chars:
        raise GatewayError(
            f"total message characters ({total_chars}) exceeds limit ({config.max_message_chars})",
            status=413,
            code="payload_too_large",
        )

    # Check JSON byte length
    try:
        json_bytes = json.dumps(payload).encode("utf-8")
        if len(json_bytes) > config.max_body_bytes:
            raise GatewayError(
                f"payload size ({len(json_bytes)} bytes) exceeds limit "
                f"({config.max_body_bytes} bytes)",
                status=413,
                code="payload_too_large",
            )
    except (TypeError, ValueError) as e:
        raise GatewayError(
            f"failed to serialize payload: {e}", status=400, code="serialization_error"
        ) from None

    return {"model": model, "message_chars": total_chars}


def _first_metric(metrics: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among ``keys``."""
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return default


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


EXTERNAL_BENCHMARK_LABELS = {
    "ifeval_strict": "IFEval strict",
    "gsm8k_math_acc": "GSM8K / MATH-500",
    "sql_incident_acc": "SQL-Incident Eval",
    "direct_compliance_rate": "Direct-Compliance / Unslop Audit",
}


def _format_manifest_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _external_benchmark_markdown_rows(benchmarks: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    covered: set[str] = set()
    for key, label in EXTERNAL_BENCHMARK_LABELS.items():
        if key in benchmarks:
            rows.append(f"| {label} | {_format_manifest_value(benchmarks[key])} |")
            covered.add(key)
    for key, value in benchmarks.items():
        if key in covered:
            continue
        rows.append(f"| {str(key).replace('_', ' ').title()} | {_format_manifest_value(value)} |")
    return rows


def _vllm_smoke_markdown_rows(smoke: dict[str, Any]) -> list[str]:
    ttft = smoke.get("ttft_ms")
    if isinstance(ttft, dict):
        ttft_text = (
            f"mean {_format_manifest_value(ttft.get('mean'))} / "
            f"p50 {_format_manifest_value(ttft.get('p50'))}"
        )
    else:
        ttft_text = _format_manifest_value(ttft)

    tok_s = smoke.get("tok_s")
    if isinstance(tok_s, dict):
        mean_value = _format_manifest_value(tok_s.get("mean"))
        aggregate = tok_s.get("aggregate")
        if aggregate is not None:
            tok_text = f"mean {mean_value} / aggregate {_format_manifest_value(aggregate)}"
        else:
            tok_text = mean_value
    else:
        tok_text = _format_manifest_value(tok_s)

    rows = [
        ("TTFT ms", ttft_text),
        ("tok/s", tok_text),
        ("FP8 parity", _format_manifest_value(smoke.get("fp8_parity"))),
        ("Success rate", _format_manifest_value(smoke.get("success_rate"))),
    ]
    return [f"| {label} | {value} |" for label, value in rows]


def build_openrouter_onboarding_manifest(
    hf_repo_id: str,
    base_model: str,
    h9_metrics: dict[str, Any],
    h10_metrics: dict[str, Any],
    h11_metrics: dict[str, Any],
    *,
    external_benchmarks: dict[str, Any] | None = None,
    vllm_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured OpenRouter provider-onboarding manifest.

    Bundles the FP8 Re-Leakage / survival proof and Row Crest-Factor /
    max-abs-weight weight-hygiene evidence (H11), plus key H9 Lancet
    surgery and H10 subspace-safe GRPO metrics, a vLLM launch command,
    and an architectural compatibility checklist.

    Optional ``external_benchmarks`` (IFEval strict, GSM8K / MATH-500,
    SQL-Incident Eval, Direct-Compliance / Unslop Audit) and ``vllm_smoke``
    (live FP8 serving TTFT / tok-s / parity results from
    ``run_vllm_fp8_smoke``) are appended as extra manifest keys and
    markdown sections when provided.
    """
    vllm_launch_args = {
        "model": hf_repo_id,
        "quantization": "fp8",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "cli_args": [
            "--model",
            hf_repo_id,
            "--quantization",
            "fp8",
            "--dtype",
            "bfloat16",
            "--enable-auto-tool-choice",
        ],
    }

    fp8_and_crest_proof = {
        "h11_fp8_provider_audit": {
            "fp8_retention_ratio": _first_metric(h11_metrics, "fp8_retention_ratio"),
            "h11_fp8_kl_reduction_ratio": _first_metric(h11_metrics, "h11_fp8_kl_reduction_ratio"),
            "fp8_subspace_releakage_naive": _first_metric(
                h11_metrics,
                "fp8_subspace_releakage_naive",
                "fp8_releakage_naive",
            ),
            "fp8_subspace_releakage_lancet": _first_metric(
                h11_metrics,
                "fp8_subspace_releakage_lancet",
                "fp8_releakage_lancet",
            ),
            "fp8_subspace_releakage_naive_per_tensor": _first_metric(
                h11_metrics, "fp8_subspace_releakage_naive_per_tensor"
            ),
            "fp8_subspace_releakage_lancet_per_tensor": _first_metric(
                h11_metrics, "fp8_subspace_releakage_lancet_per_tensor"
            ),
            "kl_naive_fp8": _first_metric(h11_metrics, "kl_naive_fp8"),
            "kl_lancet_fp8": _first_metric(h11_metrics, "kl_lancet_fp8"),
            "max_row_crest_factor": _first_metric(
                h11_metrics, "max_row_crest_factor", "row_crest_factor"
            ),
            "max_abs_weight": _first_metric(h11_metrics, "max_abs_weight"),
        },
        "h9_lancet_surgery": {
            "h9_verified": _first_metric(h9_metrics, "h9_verified"),
            "h9_kl_reduction_ratio": _first_metric(h9_metrics, "h9_kl_reduction_ratio"),
            "target_suppression_ratio": _first_metric(h9_metrics, "target_suppression_ratio"),
            "naive_kl_collateral": _first_metric(h9_metrics, "naive_kl_collateral"),
            "lancet_kl_collateral": _first_metric(h9_metrics, "lancet_kl_collateral"),
            "max_row_norm_drift": _first_metric(h9_metrics, "max_row_norm_drift"),
            "anchor_orthogonality_error": _first_metric(h9_metrics, "anchor_orthogonality_error"),
        },
        "h10_subspace_safe_grpo": {
            "h10_verified": _first_metric(h10_metrics, "h10_verified"),
            "subspace_safe_lora_releakage": _first_metric(
                h10_metrics, "subspace_safe_lora_releakage"
            ),
            "unprojected_lora_releakage": _first_metric(h10_metrics, "unprojected_lora_releakage"),
            "post_merge_suppression_retention": _first_metric(
                h10_metrics, "post_merge_suppression_retention"
            ),
            "h10_gated_vs_naive_gain": _first_metric(h10_metrics, "h10_gated_vs_naive_gain"),
            "h10_post_surgery_gain": _first_metric(h10_metrics, "h10_post_surgery_gain"),
            "merged_lora_count": _first_metric(h10_metrics, "merged_lora_count"),
            "length_collapse_rate": _first_metric(h10_metrics, "length_collapse_rate"),
        },
    }

    architectural_compatibility = {
        "standard_architecture": True,
        "trust_remote_code": False,
        "auto_map": None,
        "chat_template_ready": True,
        "supported_architectures": [
            "LlamaForCausalLM",
            "Qwen2ForCausalLM",
            "MistralForCausalLM",
        ],
    }

    audit = fp8_and_crest_proof["h11_fp8_provider_audit"]
    h9_proof = fp8_and_crest_proof["h9_lancet_surgery"]
    h10_proof = fp8_and_crest_proof["h10_subspace_safe_grpo"]
    vllm_command = "vllm serve " + " ".join(vllm_launch_args["cli_args"])
    supported = ", ".join(architectural_compatibility["supported_architectures"])
    retention = _format_metric(audit["fp8_retention_ratio"])
    kl_reduction = _format_metric(audit["h11_fp8_kl_reduction_ratio"])
    releakage_lancet = _format_metric(audit["fp8_subspace_releakage_lancet"])
    releakage_naive = _format_metric(audit["fp8_subspace_releakage_naive"])
    h9_kl_ratio = _format_metric(h9_proof["h9_kl_reduction_ratio"])
    h10_lora_releakage = _format_metric(h10_proof["subspace_safe_lora_releakage"])
    crest = _format_metric(audit["max_row_crest_factor"])
    max_abs = _format_metric(audit["max_abs_weight"])
    markdown_summary = (
        f"# OpenRouter Onboarding — {hf_repo_id}\n"
        "\n"
        "## Repository\n"
        "\n"
        f"- **HF repo id:** `{hf_repo_id}`\n"
        f"- **Base model:** `{base_model}`\n"
        f"- **Target providers:** {', '.join(OPENROUTER_TENET_PROVIDERS)}\n"
        "\n"
        "## vLLM launch command\n"
        "\n"
        "```bash\n"
        f"{vllm_command}\n"
        "```\n"
        "\n"
        "Settings: `quantization=fp8`, `dtype=bfloat16`, "
        "`trust_remote_code=False`, `enable-auto-tool-choice=True`.\n"
        "\n"
        "## FP8 Re-Leakage & Row Crest-Factor proof\n"
        "\n"
        "| Proof | Value |\n"
        "| --- | --- |\n"
        f"| H11 FP8 retention ratio | {retention} |\n"
        f"| H11 FP8 KL reduction ratio | {kl_reduction} |\n"
        f"| H11 FP8 subspace Re-Leakage (Lancet) | {releakage_lancet} |\n"
        f"| H11 FP8 subspace Re-Leakage (naive) | {releakage_naive} |\n"
        f"| H9 Lancet KL reduction ratio | {h9_kl_ratio} |\n"
        f"| H10 subspace-safe LoRA Re-Leakage | {h10_lora_releakage} |\n"
        f"| Max row crest factor | {crest} |\n"
        f"| Max abs weight | {max_abs} |\n"
        "\n"
        "## Architectural compatibility checklist\n"
        "\n"
        "- [x] Standard, first-party architecture (no custom `auto_map`)\n"
        "- [x] No `trust_remote_code` escape hatches\n"
        "- [x] Chat template ready\n"
        f"- Supported architectures: {supported}\n"
    )

    if external_benchmarks:
        benchmark_rows = _external_benchmark_markdown_rows(external_benchmarks)
        if benchmark_rows:
            markdown_summary += (
                "\n## External Benchmarks\n"
                "\n"
                "| Benchmark | Score |\n"
                "| --- | --- |\n" + "\n".join(benchmark_rows) + "\n"
            )

    if vllm_smoke:
        smoke_rows = _vllm_smoke_markdown_rows(vllm_smoke)
        if smoke_rows:
            markdown_summary += (
                "\n## vLLM FP8 Smoke & Parity\n"
                "\n"
                "| Metric | Value |\n"
                "| --- | --- |\n" + "\n".join(smoke_rows) + "\n"
            )

    manifest = {
        "hf_repo_id": hf_repo_id,
        "base_model": base_model,
        "providers": list(OPENROUTER_TENET_PROVIDERS),
        "vllm_launch_args": vllm_launch_args,
        "fp8_and_crest_proof": fp8_and_crest_proof,
        "architectural_compatibility": architectural_compatibility,
        "markdown_summary": markdown_summary,
    }
    if external_benchmarks:
        manifest["external_benchmarks"] = external_benchmarks
    if vllm_smoke:
        manifest["vllm_smoke"] = vllm_smoke
    return manifest


def _extract_response_text(response: dict[str, Any]) -> str:
    """Pull assistant text out of a chat-completions (or completion) response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(first.get("text") or "")


def _extract_ttft_ms(response: dict[str, Any]) -> float | None:
    """Return a server-reported TTFT when present, else None."""
    candidates: list[Any] = [
        response.get("ttft_ms"),
        response.get("time_to_first_token_ms"),
    ]
    timing = response.get("timing")
    if isinstance(timing, dict):
        candidates.extend([timing.get("ttft_ms"), timing.get("time_to_first_token_ms")])
    usage = response.get("usage")
    if isinstance(usage, dict):
        candidates.extend([usage.get("ttft_ms"), usage.get("time_to_first_token_ms")])
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
    return None


def _fp8_token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _fp8_parity_score(response_text: str, reference_text: str) -> float:
    """Token/substring overlap parity between two responses in [0.0, 1.0]."""
    response_lower = response_text.strip().lower()
    reference_lower = reference_text.strip().lower()
    if not response_lower and not reference_lower:
        return 0.0
    if response_lower and response_lower in reference_lower:
        return 1.0
    if reference_lower and reference_lower in response_lower:
        return 1.0
    response_tokens = _fp8_token_set(response_lower)
    reference_tokens = _fp8_token_set(reference_lower)
    if not response_tokens or not reference_tokens:
        return 0.0
    overlap = len(response_tokens & reference_tokens)
    precision = overlap / len(response_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def run_vllm_fp8_smoke(
    client: "OpenRouterGatewayClient",
    model: str,
    prompts: Sequence[str] | None = None,
    reference_responses: Sequence[str] | None = None,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Run deterministic FP8 smoke prompts against a served vLLM endpoint.

    Issues one chat completion per prompt (JSON format check, simple
    arithmetic, SQL read-only check by default), measures per-request and
    aggregate TTFT/tok-s, and computes an FP8 parity score against optional
    reference responses (token/substring overlap in [0.0, 1.0]). Without
    references, parity is 1.0 when every request succeeds with a non-empty
    response.
    """
    selected_prompts = list(prompts) if prompts else list(VLLM_FP8_SMOKE_PROMPTS)
    references = list(reference_responses) if reference_responses is not None else None

    requests: list[dict[str, Any]] = []
    ttft_values: list[float] = []
    tok_per_sec_values: list[float] = []
    total_completion_tokens = 0
    total_elapsed_seconds = 0.0
    successes = 0

    for index, prompt in enumerate(selected_prompts):
        record: dict[str, Any] = {
            "index": index,
            "prompt": prompt,
            "ok": False,
            "response": "",
            "completion_tokens": 0,
            "elapsed_ms": 0,
            "ttft_ms": None,
            "tok_s": None,
            "error": None,
        }
        start_time = time.monotonic()
        try:
            response = client.chat_completions(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                request_id=f"vllm-fp8-smoke-{index}",
            )
        except Exception as exc:  # noqa: BLE001 - record any per-request failure
            record["elapsed_ms"] = round((time.monotonic() - start_time) * 1000)
            record["error"] = str(exc)
            requests.append(record)
            continue

        elapsed_ms = round((time.monotonic() - start_time) * 1000)
        elapsed_seconds = max(elapsed_ms / 1000.0, 1e-6)
        total_elapsed_seconds += elapsed_seconds

        text = _extract_response_text(response)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        ttft_ms = _extract_ttft_ms(response)
        if ttft_ms is None:
            # Non-streaming: elapsed time of the whole request is the best
            # available first-token proxy.
            ttft_ms = float(elapsed_ms)
        tok_s = completion_tokens / elapsed_seconds if completion_tokens > 0 else None

        successes += 1
        total_completion_tokens += completion_tokens
        ttft_values.append(float(ttft_ms))
        if tok_s is not None:
            tok_per_sec_values.append(float(tok_s))

        record.update(
            {
                "ok": True,
                "response": text,
                "completion_tokens": completion_tokens,
                "elapsed_ms": elapsed_ms,
                "ttft_ms": round(ttft_ms, 3),
                "tok_s": round(tok_s, 3) if tok_s is not None else None,
            }
        )
        requests.append(record)

    if references is None:
        valid = sum(1 for entry in requests if entry["ok"] and str(entry["response"]).strip())
        fp8_parity = valid / len(requests) if requests else 0.0
    else:
        scores = [
            _fp8_parity_score(str(entry["response"]), reference)
            for entry, reference in zip(requests, references, strict=False)
        ]
        fp8_parity = statistics.mean(scores) if scores else 0.0

    aggregate_tok_s = (
        total_completion_tokens / total_elapsed_seconds
        if total_elapsed_seconds > 0.0 and total_completion_tokens > 0
        else None
    )

    return {
        "model": model,
        "num_requests": len(requests),
        "success_rate": successes / len(requests) if requests else 0.0,
        "ttft_ms": {
            "mean": round(statistics.mean(ttft_values), 3) if ttft_values else 0.0,
            "p50": round(statistics.median(ttft_values), 3) if ttft_values else 0.0,
        },
        "tok_s": {
            "mean": round(statistics.mean(tok_per_sec_values), 3) if tok_per_sec_values else 0.0,
            "p50": round(statistics.median(tok_per_sec_values), 3) if tok_per_sec_values else 0.0,
            "aggregate": round(aggregate_tok_s, 3) if aggregate_tok_s is not None else 0.0,
        },
        "total_completion_tokens": total_completion_tokens,
        "fp8_parity": round(fp8_parity, 4),
        "requests": requests,
    }


def build_vllm_fp8_mock_transport() -> Callable[[urllib.request.Request, int], dict[str, Any]]:
    """Deterministic transport emulating a local vLLM OpenAI-compatible server."""

    def _deterministic_smoke_response(prompt: str) -> str:
        lowered = prompt.lower()
        arithmetic = re.search(r"(\d+)\s*([*+])\s*(\d+)", lowered)
        if arithmetic:
            left, op, right = (
                int(arithmetic.group(1)),
                arithmetic.group(2),
                int(arithmetic.group(3)),
            )
            return str(left * right if op == "*" else left + right)
        if "json" in lowered:
            return VLLM_FP8_SMOKE_REFERENCE_RESPONSES[0]
        if "sql" in lowered or "drop table" in lowered:
            return VLLM_FP8_SMOKE_REFERENCE_RESPONSES[2]
        return "ok"

    def transport(request: urllib.request.Request, timeout_seconds: int) -> dict[str, Any]:
        payload = json.loads(bytes(request.data or b"").decode("utf-8"))
        prompt = ""
        for message in payload.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "user":
                prompt = str(message.get("content") or "")
        content = _deterministic_smoke_response(prompt)
        prompt_tokens = max(len(prompt.split()), 1)
        completion_tokens = max(len(content.split()), 1)
        return {
            "id": "chatcmpl-vllm-fp8-mock",
            "object": "chat.completion",
            "created": 0,
            "model": str(payload.get("model") or ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return transport


class OpenRouterGatewayClient:
    def __init__(
        self,
        config: GatewayConfig,
        transport: Callable[[urllib.request.Request, int], dict[str, Any]] | None = None,
    ):
        self.config = config
        self._transport = transport
        self.audit_log: list[dict[str, Any]] = []

    def chat_completions(
        self,
        payload: dict[str, Any],
        request_id: str | None = None,
        provider_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Process chat completions request through the gateway."""
        # Work on a copy so the caller's payload dict is never mutated.
        outbound_payload = dict(payload)

        # Optional OpenRouter provider routing order.
        if provider_order is not None:
            if (
                not isinstance(provider_order, list)
                or len(provider_order) == 0
                or not all(isinstance(name, str) and name for name in provider_order)
            ):
                raise GatewayError(
                    "provider_order must be a non-empty list of provider names",
                    status=400,
                    code="invalid_provider_order",
                )
            outbound_payload["provider"] = {
                "order": list(provider_order),
                "allow_fallbacks": True,
            }

        # Validate payload (egress security filters run on the outbound body)
        audit_summary = validate_chat_payload(outbound_payload, self.config)

        # Check API key
        if not self.config.api_key:
            raise GatewayError("missing upstream api_key", status=401, code="missing_api_key")

        # Build request
        url = f"{self.config.upstream_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.app_referer,
            "X-Title": self.config.app_title,
        }

        if request_id:
            headers["X-SML-Request-Id"] = request_id

        harness_task_id = self.config.task_id
        if not harness_task_id and request_id:
            derived = re.sub(r"-step-\d+$", "", request_id)
            if derived:
                harness_task_id = derived
        if harness_task_id:
            headers["X-Harness-Task-Id"] = harness_task_id

        # Serialize payload
        data = json.dumps(outbound_payload).encode("utf-8")

        # Create request
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        # Execute request
        start_time = time.monotonic()
        try:
            if self._transport:
                response_data = self._transport(req, self.config.timeout_seconds)
            else:
                response = urllib.request.urlopen(req, timeout=self.config.timeout_seconds)
                response_data = json.loads(response.read())

            elapsed_ms = round((time.monotonic() - start_time) * 1000)

            # Record audit entry
            prompt_tokens = response_data.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = response_data.get("usage", {}).get("completion_tokens", 0)

            audit_entry = {
                "request_id": request_id,
                "model": audit_summary["model"],
                "message_chars": audit_summary["message_chars"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": elapsed_ms,
                "status": 200,
            }
            self.audit_log.append(audit_entry)

            return response_data
        except URLError as e:
            elapsed_ms = round((time.monotonic() - start_time) * 1000)
            status = getattr(e, "code", 500)

            audit_entry = {
                "request_id": request_id,
                "model": audit_summary["model"],
                "message_chars": audit_summary["message_chars"],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_ms": elapsed_ms,
                "status": status,
            }
            self.audit_log.append(audit_entry)

            raise GatewayError(
                f"upstream request failed: {e}", status=status, code="upstream_error"
            ) from None


@contextmanager
def managed_harness_gateway(
    config: GatewayConfig | None = None,
    *,
    sml_results_dir: str | Path | None = None,
    upstream_mode: str = "openrouter",
    fallback_teacher_model: str | None = None,
) -> Iterator[OpenRouterGatewayClient]:
    """Run a request through the harness OpenRouter proxy.

    Yields an ``OpenRouterGatewayClient`` pointed at the proxy's local
    listener. On exit the proxy flushes ``cost.json`` and ``traces.jsonl``
    into ``sml_results_dir``.
    """
    if config is None:
        config = GatewayConfig.from_env()

    openrouter_proxy = None
    try:
        from harness.proxy import openrouter_proxy as _openrouter_proxy

        openrouter_proxy = _openrouter_proxy
    except ImportError:
        sibling_harness = Path(__file__).resolve().parents[4] / "Harness"
        candidates = (
            os.environ.get("HARNESS_ROOT"),
            str(sibling_harness),
            "/Users/mr_a_i/Developer/personal/Harness",
        )
        for candidate in candidates:
            if candidate and os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
        try:
            from harness.proxy import openrouter_proxy as _openrouter_proxy

            openrouter_proxy = _openrouter_proxy
        except ImportError:
            openrouter_proxy = None

    if openrouter_proxy is not None:
        with openrouter_proxy(
            lambda: None,
            upstream_url=config.upstream_url,
            request_timeout=config.timeout_seconds,
            upstream_mode=upstream_mode,
            fallback_teacher_model=fallback_teacher_model,
            sml_results_dir=sml_results_dir,
        ) as proxy:
            yield OpenRouterGatewayClient(replace(config, upstream_url=proxy.base_url))
    else:
        yield OpenRouterGatewayClient(config)

    # Guarantee the manifests exist in sml_results_dir after the proxy exits.
    # The proxy normally flushes cost.json / traces.jsonl itself on exit; if it
    # did not (e.g. no requests recorded through its own ledger), write minimal
    # valid manifests so downstream consumers always find the files.
    if sml_results_dir is not None:
        results_dir = Path(sml_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        cost_path = results_dir / "cost.json"
        if not cost_path.exists():
            cost_path.write_text(
                json.dumps({"total_cost_usd": 0.0, "requests": []}) + "\n", encoding="utf-8"
            )
        traces_path = results_dir / "traces.jsonl"
        if not traces_path.exists():
            traces_path.write_text("", encoding="utf-8")
