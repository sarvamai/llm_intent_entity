"""Tests for llm_api module — parse endpoint detection and fallback behavior.
Run with: pytest tests/test_llm_api.py -v
"""
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# ─── 1. Portable paths ───
REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_API_PATH = REPO_ROOT / "src" / "llm_intent_entity" / "llm_api.py"

# ─── 2. Build mocks for heavy dependencies ───
def build_mocks():
    """Create fresh mocks for each test run."""
    # Don't mock pydantic or pydantic_core - they're pure Python libraries
    # that don't make network calls, and mocking them breaks internal imports
    return {
        "openai": MagicMock(),
        "google": MagicMock(),
        "google.oauth2": MagicMock(),
        "google.oauth2.service_account": MagicMock(),
        "google.auth": MagicMock(),
        "google.auth.transport": MagicMock(),
        "google.auth.transport.requests": MagicMock(),
        "vertexai": MagicMock(),
        "vertexai.generative_models": MagicMock(),
        "jsonlines": MagicMock(),
        "tqdm": MagicMock(),
        "joblib": MagicMock(),
    }


# ─── 3. Load real llm_api.py with isolated mocks ───
def load_llm_api(mocks=None):
    """Load real llm_api.py with controlled mocks."""
    if mocks is None:
        mocks = build_mocks()
    
    # Keep mocks in sys.modules
    for name, mock in mocks.items():
        sys.modules[name] = mock
    
    spec = importlib.util.spec_from_file_location("llm_api_real", LLM_API_PATH)
    llm_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_api)
    return llm_api, mocks


# ─── 4. Test cases ───
class TestParseEndpointDetection:
    """Test parse endpoint detection logic."""

    def test_is_parse_endpoint_supported_uses_beta_endpoint(self):
        """is_parse_endpoint_supported() should call client.beta.chat.completions.parse, not client.chat.completions.parse."""
        mocks = build_mocks()
        mock_client = mocks["openai"].OpenAI.return_value
        
        llm_api, _ = load_llm_api(mocks)
        api = llm_api.ChatCompletionsAPI(
            model_name="test-model",
            api_key="test-key",
            base_url="http://test-url",
            gemini=False,
        )
        
        mock_client.beta.chat.completions.parse.assert_called_once()
        mock_client.chat.completions.parse.assert_not_called()
        assert api.parse_endpoint_supported is True

    def test_is_parse_endpoint_supported_fallback_on_error(self):
        """When beta parse endpoint raises an exception, should fall back to guided_json."""
        mocks = build_mocks()
        mock_client = mocks["openai"].OpenAI.return_value
        mock_client.beta.chat.completions.parse.side_effect = AttributeError("'ChatCompletions' object has no attribute 'parse'")
        
        llm_api, _ = load_llm_api(mocks)
        api = llm_api.ChatCompletionsAPI(
            model_name="test-model",
            api_key="test-key",
            base_url="http://test-url",
            gemini=False,
        )
        
        mock_client.beta.chat.completions.parse.assert_called_once()
        assert api.parse_endpoint_supported is False

    def test_generate_single_response_uses_beta_parse_when_supported(self):
        """When parse_endpoint_supported=True, generate_single_response should use beta.chat.completions.parse."""
        mocks = build_mocks()
        mock_client = mocks["openai"].OpenAI.return_value
        
        # Setup successful parse response
        mock_parsed_response = MagicMock()
        mock_parsed_response.choices = [MagicMock(message=MagicMock(parsed=MagicMock(model_dump=lambda: {"result": "parsed"})))]
        mock_parsed_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.beta.chat.completions.parse.return_value = mock_parsed_response
        
        llm_api, _ = load_llm_api(mocks)
        api = llm_api.ChatCompletionsAPI(
            model_name="test-model",
            api_key="test-key",
            base_url="http://test-url",
            gemini=False,
        )
        
        from pydantic import BaseModel
        class TestSchema(BaseModel):
            result: str
        
        result = api.generate_single_response(
            prompt="Test prompt",
            schema=TestSchema,
        )
        
        assert mock_client.beta.chat.completions.parse.call_count == 2
        mock_client.chat.completions.create.assert_not_called()
        assert result["status"] == "success"
        assert result["response"] == {"result": "parsed"}

    def test_generate_single_response_falls_back_to_guided_json_when_not_supported(self):
        """When parse_endpoint_supported=False, generate_single_response should use guided_json fallback."""
        mocks = build_mocks()
        mock_client = mocks["openai"].OpenAI.return_value
        mock_client.beta.chat.completions.parse.side_effect = AttributeError("not supported")
        
        # Setup fallback create response
        mock_create_response = MagicMock()
        mock_create_response.choices = [MagicMock(message=MagicMock(content='{"result": "fallback"}'))]
        mock_create_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_create_response
        
        llm_api, _ = load_llm_api(mocks)
        api = llm_api.ChatCompletionsAPI(
            model_name="test-model",
            api_key="test-key",
            base_url="http://test-url",
            gemini=False,
        )
        
        from pydantic import BaseModel
        class TestSchema(BaseModel):
            result: str
        
        result = api.generate_single_response(
            prompt="Test prompt",
            schema=TestSchema,
        )
        
        mock_client.chat.completions.create.assert_called_once()
        assert mock_client.beta.chat.completions.parse.call_count == 1
        assert result["status"] == "success"
        assert result["response"] == {"result": "fallback"}

    def test_probe_does_not_leak_exceptions(self):
        """The capability check should handle all exceptions gracefully without leaking."""
        for exception_class in [AttributeError, TypeError, ValueError, RuntimeError, Exception]:
            mocks = build_mocks()
            mock_client = mocks["openai"].OpenAI.return_value
            mock_client.beta.chat.completions.parse.side_effect = exception_class("test error")
            
            llm_api, _ = load_llm_api(mocks)
            api = llm_api.ChatCompletionsAPI(
                model_name="test-model",
                api_key="test-key",
                base_url="http://test-url",
                gemini=False,
            )
            
            assert api.parse_endpoint_supported is False
            mock_client.beta.chat.completions.parse.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])