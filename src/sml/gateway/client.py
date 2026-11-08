import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError

FORBIDDEN_RAW_BODY_KEYS = {
    "artifact_body",
    "artifact_bodies",
    "raw_artifact_body",
    "raw_artifact_bodies",
    "raw_body",
    "raw_bodies",
    "credentials",
    "secrets",
    "tokens",
    "private_key",
    "api_key",
}

OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"


class GatewayError(ValueError):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class GatewayConfig:
    upstream_url: str = OPENROUTER_DEFAULT_URL
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

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        # Read upstream URL
        upstream_url = os.environ.get("SML_GATEWAY_UPSTREAM_URL", OPENROUTER_DEFAULT_URL)

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
        self, payload: dict[str, Any], request_id: str | None = None
    ) -> dict[str, Any]:
        """Process chat completions request through the gateway."""
        # Validate payload
        audit_summary = validate_chat_payload(payload, self.config)

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

        # Serialize payload
        data = json.dumps(payload).encode("utf-8")

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
