"""
Test that API keys are properly redacted in Langfuse exception logs.
"""
import os
import sys
import traceback
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.langfuse.langfuse import (
    LangFuseLogger,
    _redact_sensitive_info_from_string,
)


@pytest.fixture
def mock_langfuse():
    """Mock Langfuse client for testing"""
    with patch.dict(
        "os.environ",
        {
            "LANGFUSE_SECRET_KEY": "sk-lf-test-secret-key-12345678901234567890",
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test-public-key-12345678901234567890",
            "LANGFUSE_HOST": "https://test.langfuse.com",
        },
    ):
        mock_langfuse = MagicMock()
        mock_langfuse.version.__version__ = "2.59.7"
        mock_client = MagicMock()
        mock_client.client.projects.get.return_value = MagicMock(
            data=[MagicMock(id="test-project-id")]
        )
        mock_langfuse.Langfuse.return_value = mock_client

        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            logger = LangFuseLogger()
            yield logger, mock_client


def test_langfuse_key_redaction(mock_langfuse):
    """Comprehensive test that API keys are redacted in all exception logging paths"""
    logger, mock_client = mock_langfuse

    # Test 1: Redaction function handles various key patterns
    test_cases = [
        ("Error: API key sk-1234567890123456789012345678901234567890 is invalid", "sk-1234567890123456789012345678901234567890"),
        ("secret_key=sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz", "sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"),
        ('{"api_key": "pk-lf-1234567890123456789012345678901234567890"}', "pk-lf-1234567890123456789012345678901234567890"),
        ("Authorization: Bearer sk-1234567890123456789012345678901234567890", "sk-1234567890123456789012345678901234567890"),
        ("LANGFUSE_SECRET_KEY=sk-lf-1234567890123456789012345678901234567890", "sk-lf-1234567890123456789012345678901234567890"),
    ]
    for input_text, expected_key in test_cases:
        result = _redact_sensitive_info_from_string(input_text)
        assert expected_key not in result, f"Key {expected_key} not redacted in: {input_text}"
        assert "[REDACTED_API_KEY]" in result

    # Test 2: Multiple key patterns in one string
    multi_key_string = """
    Error occurred:
    - OpenAI key: sk-1234567890123456789012345678901234567890
    - Anthropic key: sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
    - Langfuse public: pk-lf-9876543210987654321098765432109876543210
    - Langfuse secret: sk-lf-1111222233334444555566667777888899990000
    - Generic: api_key=secret123456789012345678901234567890
    """
    redacted = _redact_sensitive_info_from_string(multi_key_string)
    keys = [
        "sk-1234567890123456789012345678901234567890",
        "sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz",
        "pk-lf-9876543210987654321098765432109876543210",
        "sk-lf-1111222233334444555566667777888899990000",
        "secret123456789012345678901234567890",
    ]
    for key in keys:
        assert key not in redacted, f"Key {key} not redacted in multi-key string"
    assert "[REDACTED_API_KEY]" in redacted

    # Test 3: Traceback redaction
    api_key = "sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
    def function_with_key():
        key = api_key
        raise ValueError(f"Error with key {key}")

    error_traceback = ""
    try:
        function_with_key()
    except ValueError:
        error_traceback = traceback.format_exc()

    assert api_key in error_traceback, "Original traceback should contain key"
    redacted_traceback = _redact_sensitive_info_from_string(error_traceback)
    assert api_key not in redacted_traceback, "Redacted traceback should not contain key"
    assert "[REDACTED_API_KEY]" in redacted_traceback

    # Test 4: log_event_on_langfuse exception logging
    api_key = "sk-1234567890123456789012345678901234567890"
    with patch("litellm.integrations.langfuse.langfuse.verbose_logger") as mock_logger:
        # Trigger exception in log_event_on_langfuse by making _get_langfuse_input_output_content fail
        with patch.object(logger, '_get_langfuse_input_output_content', side_effect=Exception(f"Failed with key {api_key}")):
            try:
                logger.log_event_on_langfuse(
                    kwargs={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]},
                    response_obj=None,
                )
            except Exception:
                pass

            assert mock_logger.exception.called, "log_event_on_langfuse should log exceptions"
            logged = mock_logger.exception.call_args[0][0]
            assert api_key not in logged, "Exception message should not contain key"
            assert "[REDACTED_API_KEY]" in logged

    # Test 5: _log_langfuse_v2 exception logging
    api_key = "pk-lf-1234567890123456789012345678901234567890"
    with patch("litellm.integrations.langfuse.langfuse.verbose_logger") as mock_logger:
        mock_client.trace.side_effect = Exception(f"Failed to create trace with key {api_key}")
        try:
            logger._log_langfuse_v2(
                user_id="test-user",
                metadata={},
                litellm_params={},
                output="test output",
                start_time=None,
                end_time=None,
                kwargs={"model": "gpt-3.5-turbo"},
                optional_params={},
                input={"messages": [{"role": "user", "content": "test"}]},
                response_obj=None,
                level="DEFAULT",
                litellm_call_id="test-call-id",
            )
        except Exception:
            pass

        assert mock_logger.error.called, "_log_langfuse_v2 should log errors"
        logged_message = mock_logger.error.call_args[0][0]
        assert api_key not in logged_message, "Error log should not contain key"
        assert "[REDACTED_API_KEY]" in logged_message
