"""Tests for normalization layer.
Run with: pytest tests/test_normalization.py -v
"""
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# ─── 1. Portable paths ───
REPO_ROOT = Path(__file__).resolve().parent.parent
UTILITIES_PATH = REPO_ROOT / "src" / "llm_intent_entity" / "utilities.py"
MAIN_PATH = REPO_ROOT / "src" / "llm_intent_entity" / "main.py"

# ─── 2. Robust module isolation context manager ───
class IsolatedModules:
    """Temporarily inject mocks into sys.modules and restore on exit.
    Handles nested modules by tracking parent packages and their attributes.
    """
    def __init__(self, mocks: dict[str, object]):
        self.mocks = mocks
        self.original: dict[str, object] = {}
        self.created: set[str] = set()
        self.parent_attrs: dict[str, dict[str, object]] = {}

    def __enter__(self):
        for name, mock in self.mocks.items():
            # Track if module already exists
            if name in sys.modules:
                self.original[name] = sys.modules[name]
            else:
                self.created.add(name)
                # Also track parent package attributes
                parts = name.split('.')
                for i in range(1, len(parts)):
                    parent = '.'.join(parts[:i])
                    if parent in sys.modules:
                        attr = parts[i]
                        if not hasattr(sys.modules[parent], attr):
                            self.parent_attrs.setdefault(parent, {})[attr] = None
            sys.modules[name] = mock
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore in reverse order to handle nesting correctly
        for name in reversed(list(self.mocks.keys())):
            if name in self.created:
                sys.modules.pop(name, None)
                # Clean up parent attributes we added
                parts = name.split('.')
                for i in range(1, len(parts)):
                    parent = '.'.join(parts[:i])
                    if parent in self.parent_attrs:
                        attr = parts[i]
                        if attr in self.parent_attrs[parent]:
                            delattr(sys.modules[parent], attr)
            elif name in self.original:
                sys.modules[name] = self.original[name]
        return False


# ─── 3. Build mocks for heavy dependencies ───
def build_mocks():
    """Create fresh mocks for each test run."""
    # transformers.WhisperProcessor
    mock_transformers = MagicMock()
    mock_whisper_processor = MagicMock()
    mock_whisper_tokenizer = MagicMock()
    mock_whisper_tokenizer.normalize.return_value = "whisper normalized"
    mock_whisper_processor.tokenizer = mock_whisper_tokenizer
    mock_transformers.WhisperProcessor.from_pretrained.return_value = mock_whisper_processor

    # indicnlp.normalize.indic_normalize.IndicNormalizerFactory
    class MockIndicNormalizer:
        def normalize(self, text):
            return f"indic normalized: {text}"

    class MockIndicFactory:
        def get_normalizer(self, lang):
            return MockIndicNormalizer()

    mock_indic_factory = MockIndicFactory()
    mock_indic_factory_tracked = MagicMock(wraps=mock_indic_factory)

    mock_indic_normalize = MagicMock()
    mock_indic_normalize.IndicNormalizerFactory.return_value = mock_indic_factory_tracked

    # joblib.Parallel - run synchronously to avoid pickling issues with mocks
    # Parallel is a class that's instantiated, then called with the iterable
    class MockParallel:
        def __init__(self, n_jobs=-1, **kwargs):
            self.n_jobs = n_jobs

        def __call__(self, iterable):
            # iterable contains tuples of (func, args, kwargs) from delayed()
            results = []
            for item in iterable:
                if isinstance(item, tuple) and len(item) == 3:
                    func, args, kwargs = item
                    results.append(func(*args, **kwargs))
                elif callable(item):
                    results.append(item())
            return results

    # delayed(func) returns a function that, when called with args, returns (func, args, kwargs)
    def mock_delayed(func):
        def delayed_func(*args, **kwargs):
            return (func, args, kwargs)
        return delayed_func

    # tqdm - just pass through the iterable
    mock_tqdm = MagicMock(side_effect=lambda x, *args, **kwargs: x)

    return {
        "transformers": mock_transformers,
        "indicnlp": MagicMock(),
        "indicnlp.normalize": MagicMock(),
        "indicnlp.normalize.indic_normalize": MagicMock(
            IndicNormalizerFactory=MagicMock(
                return_value=MagicMock(
                    get_normalizer=MagicMock(
                        return_value=MagicMock(normalize=MagicMock(side_effect=lambda t: f"indic normalized: {t}"))
                    )
                )
            )
        ),
        "joblib": MagicMock(Parallel=MockParallel, delayed=mock_delayed),
        "tqdm": MagicMock(tqdm=mock_tqdm),
    }


# ─── 4. Load real utilities.py with isolated mocks ───
def load_utilities():
    """Load real utilities.py with controlled mocks."""
    mocks = build_mocks()
    with IsolatedModules(mocks):
        spec = importlib.util.spec_from_file_location("utilities_real", UTILITIES_PATH)
        utilities = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utilities)
        return utilities


# ─── 5. Load main.py with proper package context ───
def load_main_module():
    """Load real main.py with package context so relative imports work."""
    mocks = build_mocks()

    # First load utilities to get real IndicNormalizer
    utilities = load_utilities()

    # Mock llm_api module
    mock_llm_api = MagicMock()
    mock_llm_api.ChatCompletionsAPI = MagicMock()

    # Create package namespace with utilities and mocked llm_api
    pkg = MagicMock()
    pkg.utilities = utilities
    pkg.llm_api = mock_llm_api

    # Mock all heavy dependencies that main.py or its imports might need
    extra_mocks = {
        "llm_intent_entity": pkg,
        "llm_intent_entity.utilities": utilities,
        "llm_intent_entity.llm_api": mock_llm_api,
        "openai": MagicMock(),
        "google": MagicMock(),
        "google.oauth2": MagicMock(),
        "google.oauth2.service_account": MagicMock(),
        "google.auth": MagicMock(),
        "google.auth.transport": MagicMock(),
        "google.auth.transport.requests": MagicMock(),
        "vertexai": MagicMock(),
        "vertexai.generative_models": MagicMock(),
        "pydantic": MagicMock(),
        "pydantic_core": MagicMock(),
        "jsonlines": MagicMock(),
        "tqdm": MagicMock(),
        "joblib": MagicMock(),
    }

    with IsolatedModules({**mocks, **extra_mocks}):
        spec = importlib.util.spec_from_file_location("main_real", MAIN_PATH)
        main_module = importlib.util.module_from_spec(spec)
        # Set __package__ so relative imports resolve correctly
        main_module.__package__ = "llm_intent_entity"
        # Mock Path.read_text() to use utf-8 encoding
        from pathlib import Path
        original_read_text = Path.read_text
        def mock_read_text(self, encoding=None, *args, **kwargs):
            return original_read_text(self, encoding="utf-8", *args, **kwargs)
        Path.read_text = mock_read_text
        try:
            spec.loader.exec_module(main_module)
        finally:
            Path.read_text = original_read_text
        return main_module


# ─── 6. Load utilities and functions ───
utilities = load_utilities()
IndicNormalizer = utilities.IndicNormalizer
lang_to_code = utilities.lang_to_code
indic_langs = utilities.indic_langs

main_module = load_main_module()
normalize_texts_in_dataframe = main_module.normalize_texts_in_dataframe

import pandas as pd


# ─── 7. Tests ───

class TestIndicNormalizerLogic:
    """Test real IndicNormalizer.normalize_text() logic with controlled mocks."""

    def test_empty_string_returns_empty_early(self):
        """Empty string returns immediately without calling normalizers."""
        normalizer = IndicNormalizer()
        result = normalizer.normalize_text("", "hi")
        assert result == ""

    def test_nan_input_returns_nan_early(self):
        """NaN input returns NaN without calling normalizers."""
        normalizer = IndicNormalizer()
        result = normalizer.normalize_text(float("nan"), "hi")
        assert pd.isna(result)

    def test_none_input_returns_none_early(self):
        """None input returns None without calling normalizers."""
        normalizer = IndicNormalizer()
        result = normalizer.normalize_text(None, "hi")
        assert result is None

    def test_hindi_uses_indic_normalizer_factory(self):
        """Hindi text uses IndicNormalizerFactory path."""
        normalizer = IndicNormalizer()
        result = normalizer.normalize_text("namaste duniya", "hi")
        assert result == "indic normalized: namaste duniya"

    def test_bengali_uses_indic_normalizer_factory(self):
        """Bengali text uses IndicNormalizerFactory path."""
        normalizer = IndicNormalizer()
        result = normalizer.normalize_text("নমস্কার", "bn")
        assert result == "indic normalized: নমস্কার"

    def test_english_uses_whisper_tokenizer(self):
        """English text uses Whisper tokenizer path."""
        normalizer = IndicNormalizer()
        result = normalizer.normalize_text("Hello World", "en")
        assert result == "whisper normalized"

    def test_urdu_uses_whisper_tokenizer(self):
        """Urdu text uses Whisper tokenizer (special handling)."""
        normalizer = IndicNormalizer()
        result = normalizer.normalize_text("ہیلو ورلڈ", "ur")
        assert result == "whisper normalized"


class TestNormalizeTextsInDataframe:
    """Test normalize_texts_in_dataframe() with controlled IndicNormalizer."""

    def test_basic_dataframe_normalization(self):
        """DataFrame with ref/pred columns gets normalized columns added."""
        df = pd.DataFrame({
            "transcription": ["namaste", "hello"],
            "prediction": ["namaste", "helo"],
            "language": ["hi", "en"],
        })
        result = normalize_texts_in_dataframe(df, "transcription", "prediction", "language")
        assert "norm_reference" in result.columns
        assert "norm_prediction" in result.columns
        assert len(result) == 2
        # Hindi row used Indic normalizer
        assert result.loc[0, "norm_reference"] == "indic normalized: namaste"
        assert result.loc[0, "norm_prediction"] == "indic normalized: namaste"
        # English row used Whisper
        assert result.loc[1, "norm_reference"] == "whisper normalized"
        assert result.loc[1, "norm_prediction"] == "whisper normalized"

    def test_per_row_language_argument(self):
        """Each row's language column value is passed to normalize_texts()."""
        df = pd.DataFrame({
            "ref": ["a", "b", "c"],
            "pred": ["a", "b", "c"],
            "lang": ["hi", "bn", "en"],
        })
        result = normalize_texts_in_dataframe(df, "ref", "pred", "lang")
        assert "norm_reference" in result.columns
        assert "norm_prediction" in result.columns
        assert len(result) == 3

    def test_missing_values_converted_to_string_nan(self):
        """NaN in text columns becomes string 'nan' after .astype(str)."""
        df = pd.DataFrame({
            "transcription": ["hello", float("nan")],
            "prediction": ["hello", "world"],
            "language": ["en", "en"],
        })
        result = normalize_texts_in_dataframe(df, "transcription", "prediction", "language")
        assert len(result) == 2
        # First row normal
        assert result.loc[0, "norm_reference"] == "whisper normalized"
        # Second row: .astype(str) converts nan -> "nan", then Whisper normalizes it
        assert result.loc[1, "norm_reference"] == "whisper normalized"

    def test_original_columns_preserved(self):
        """Original columns are not modified."""
        df = pd.DataFrame({
            "transcription": ["namaste"],
            "prediction": ["namaste"],
            "language": ["hi"],
        })
        original_ref = df["transcription"].copy()
        original_pred = df["prediction"].copy()
        result = normalize_texts_in_dataframe(df, "transcription", "prediction", "language")
        pd.testing.assert_series_equal(result["transcription"], original_ref)
        pd.testing.assert_series_equal(result["prediction"], original_pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])