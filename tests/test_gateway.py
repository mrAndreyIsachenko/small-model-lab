from unittest.mock import patch

import pytest

from sml.gateway import (
    GatewayConfig,
    GatewayError,
    OpenRouterGatewayClient,
    find_forbidden_key_paths,
    find_secret_leak_paths,
    validate_chat_payload,
)


def test_find_forbidden_key_paths():
    """Test recursive scanning for forbidden keys."""
    # Test simple case
    payload = {"credentials": "secret"}
    assert find_forbidden_key_paths(payload) == ["body.credentials"]

    # Test nested case
    payload = {"nested": {"secrets": "secret"}}
    assert find_forbidden_key_paths(payload) == ["body.nested.secrets"]

    # Test array case
    payload = {"items": [{"api_key": "key"}]}
    assert find_forbidden_key_paths(payload) == ["body.items[0].api_key"]

    # Test clean case
    payload = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    assert find_forbidden_key_paths(payload) == []


def test_validate_chat_payload_model_allowlist():
    """Test model allowlist enforcement."""
    config = GatewayConfig()

    # Valid model should pass
    valid_payload = {
        "model": "qwen/qwen3-coder",
        "messages": [{"role": "user", "content": "hello"}],
    }
    result = validate_chat_payload(valid_payload, config)
    assert result["model"] == "qwen/qwen3-coder"

    # Invalid model should fail
    invalid_payload = {
        "model": "forbidden/model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(invalid_payload, config)
    assert exc_info.value.code == "model_not_allowed"
    assert exc_info.value.status == 403


def test_validate_chat_payload_forbidden_keys():
    """Test forbidden raw-body/secret key detection."""
    config = GatewayConfig()

    # Payload with credentials should fail
    payload = {
        "model": "qwen/qwen3-coder",
        "messages": [{"role": "user", "content": "hello"}],
        "credentials": {"api_key": "secret"},
    }
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(payload, config)
    assert exc_info.value.code == "raw_body_fields_not_allowed"
    assert exc_info.value.status == 403

    # Payload with secrets should fail
    payload = {
        "model": "qwen/qwen3-coder",
        "messages": [{"role": "user", "content": "hello"}],
        "secrets": ["secret1", "secret2"],
    }
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(payload, config)
    assert exc_info.value.code == "raw_body_fields_not_allowed"
    assert exc_info.value.status == 403


def test_validate_chat_payload_rejects_nested_raw_rows():
    """Test nested raw_db_rows/raw_rows keys are rejected."""
    config = GatewayConfig()

    payload = {
        "model": "qwen/qwen3-coder",
        "messages": [{"role": "user", "content": "hello"}],
        "context": {"raw_db_rows": [{"id": 1}]},
    }
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(payload, config)
    assert exc_info.value.code == "raw_body_fields_not_allowed"
    assert exc_info.value.status == 403

    payload = {
        "model": "qwen/qwen3-coder",
        "messages": [{"role": "user", "content": "hello"}],
        "history": [{"raw_rows": "row-data"}],
    }
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(payload, config)
    assert exc_info.value.code == "raw_body_fields_not_allowed"
    assert exc_info.value.status == 403


def test_validate_chat_payload_rejects_secret_leak_in_messages():
    """Test secret patterns inside nested message strings are rejected."""
    config = GatewayConfig()

    payload = {
        "model": "qwen/qwen3-coder",
        "messages": [
            {"role": "system", "content": "config dump: api_key = abcd1234efgh"},
            {"role": "user", "content": "hello"},
        ],
    }
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(payload, config)
    assert exc_info.value.code == "secret_leak_detected"
    assert exc_info.value.status == 403

    payload = {
        "model": "qwen/qwen3-coder",
        "messages": [
            {"role": "user", "content": "leaked: -----BEGIN PRIVATE KEY-----\nMIIE..."},
        ],
    }
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(payload, config)
    assert exc_info.value.code == "secret_leak_detected"
    assert exc_info.value.status == 403


def test_find_secret_leak_paths():
    """Test recursive secret pattern detection in nested values."""
    # OpenRouter key inside nested message content
    payload = {"messages": [{"role": "user", "content": "key is sk-or-v1-abcd1234efgh5678"}]}
    assert find_secret_leak_paths(payload) == ["body.messages[0].content"]

    # Generic sk- key inside a nested dict
    payload = {"notes": {"text": "sk-AbCdEf0123456789AbCdEf"}}
    assert find_secret_leak_paths(payload) == ["body.notes.text"]

    # Private key header inside a list
    payload = {"blob": ["-----BEGIN RSA PRIVATE KEY-----"]}
    assert find_secret_leak_paths(payload) == ["body.blob[0]"]

    # key=value style secret inside a string
    payload = {"text": "api_key=supersecret1"}
    assert find_secret_leak_paths(payload) == ["body.text"]

    # Clean payload should not be flagged
    payload = {"model": "qwen/qwen3-coder", "messages": [{"role": "user", "content": "hello"}]}
    assert find_secret_leak_paths(payload) == []


def test_validate_chat_payload_size_limits():
    """Test size limit enforcement."""
    config = GatewayConfig(max_message_chars=100)

    # Exceeding character limit should fail
    payload = {"model": "qwen/qwen3-coder", "messages": [{"role": "user", "content": "a" * 150}]}
    with pytest.raises(GatewayError) as exc_info:
        validate_chat_payload(payload, config)
    assert exc_info.value.code == "payload_too_large"
    assert exc_info.value.status == 413


def test_validate_chat_payload_missing_api_key():
    """Test missing API key handling."""
    config = GatewayConfig(api_key="")
    client = OpenRouterGatewayClient(config)

    payload = {"model": "qwen/qwen3-coder", "messages": [{"role": "user", "content": "hello"}]}

    with pytest.raises(GatewayError) as exc_info:
        client.chat_completions(payload)
    assert exc_info.value.code == "missing_api_key"
    assert exc_info.value.status == 401


def test_audit_log_records_token_usage():
    """Test audit log records token usage and timing without storing prompt text."""
    config = GatewayConfig(api_key="test-key")

    # Mock transport that returns a successful response
    def mock_transport(request, timeout):
        return {
            "id": "test-response-id",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello! How can I help you today?"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    client = OpenRouterGatewayClient(config, transport=mock_transport)

    payload = {"model": "qwen/qwen3-coder", "messages": [{"role": "user", "content": "hello"}]}

    # Make request
    response = client.chat_completions(payload, request_id="test-request-1")

    # Check response
    assert response["id"] == "test-response-id"

    # Check audit log
    assert len(client.audit_log) == 1
    audit_entry = client.audit_log[0]
    assert audit_entry["request_id"] == "test-request-1"
    assert audit_entry["model"] == "qwen/qwen3-coder"
    assert audit_entry["message_chars"] == 5  # "hello"
    assert audit_entry["prompt_tokens"] == 10
    assert audit_entry["completion_tokens"] == 5
    assert audit_entry["status"] == 200
    assert "elapsed_ms" in audit_entry
    assert isinstance(audit_entry["elapsed_ms"], int)

    # Ensure no raw prompt text is stored
    assert "hello" not in str(audit_entry)


def test_gateway_config_from_env():
    """Test GatewayConfig.from_env() method."""
    with patch.dict(
        "os.environ",
        {
            "SML_GATEWAY_UPSTREAM_URL": "https://custom.example.com/api",
            "OPENROUTER_API_KEY": "test-api-key",
            "SML_GATEWAY_MODEL_ALLOWLIST": "model1,model2,model3",
        },
    ):
        config = GatewayConfig.from_env()
        assert config.upstream_url == "https://custom.example.com/api"
        assert config.api_key == "test-api-key"
        assert config.model_allowlist == ("model1", "model2", "model3")


def test_gateway_config_harness_base_url_and_task_id():
    """Test GatewayConfig picks up HARNESS_BASE_URL and client sends X-Harness-Task-Id."""
    with patch.dict(
        "os.environ",
        {"HARNESS_BASE_URL": "http://127.0.0.1:41234/api/v1"},
        clear=False,
    ):
        config = GatewayConfig(api_key="test-key", task_id="task-123")
        assert config.upstream_url == "http://127.0.0.1:41234/api/v1"
        assert GatewayConfig.from_env().upstream_url == "http://127.0.0.1:41234/api/v1"

    captured_headers: dict[str, str] = {}

    def mock_transport(request, timeout):
        captured_headers.update({k.lower(): v for k, v in request.header_items()})
        return {
            "id": "resp-1",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    client = OpenRouterGatewayClient(config, transport=mock_transport)
    payload = {"model": "qwen/qwen3-coder", "messages": [{"role": "user", "content": "hi"}]}
    client.chat_completions(payload)
    assert captured_headers.get("x-harness-task-id") == "task-123"


if __name__ == "__main__":
    pytest.main([__file__])
