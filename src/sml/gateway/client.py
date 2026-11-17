import json
import os
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
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
    app_title: str = "Small Model Lab"
    app_referer: str = "https://github.com/mrAndreyIsachenko/small-model-lab"
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


def build_openrouter_onboarding_manifest(
    hf_repo_id: str,
    base_model: str,
    h9_metrics: dict[str, Any],
    h10_metrics: dict[str, Any],
    h11_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured OpenRouter provider-onboarding manifest.

    Bundles the FP8 Re-Leakage / survival proof and Row Crest-Factor /
    max-abs-weight weight-hygiene evidence (H11), plus key H9 Lancet
    surgery and H10 subspace-safe GRPO metrics, a vLLM launch command,
    and an architectural compatibility checklist.
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

    return {
        "hf_repo_id": hf_repo_id,
        "base_model": base_model,
        "providers": list(OPENROUTER_TENET_PROVIDERS),
        "vllm_launch_args": vllm_launch_args,
        "fp8_and_crest_proof": fp8_and_crest_proof,
        "architectural_compatibility": architectural_compatibility,
        "markdown_summary": markdown_summary,
    }


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
        if self.config.task_id:
            headers["X-Harness-Task-Id"] = self.config.task_id

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
