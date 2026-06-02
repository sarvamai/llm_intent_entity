"""Stub heavy optional deps for core logic tests."""
import sys
from unittest.mock import MagicMock

_STUBS = [
    "indicnlp", "indicnlp.normalize", "indicnlp.normalize.indic_normalize",
    "transformers",
    "joblib",
    "gspread", "gspread.exceptions",
    "google", "google.auth", "google.oauth2", "google.oauth2.service_account",
    "google.cloud", "google.cloud.aiplatform",
    "vertexai", "vertexai.generative_models",
    "llm_intent_entity.llm_api",  # stub ChatCompletionsAPI
]
for _m in _STUBS:
    if _m not in sys.modules:
        mock = MagicMock()
        mock.__path__ = []
        sys.modules[_m] = mock

sys.modules["indicnlp.normalize.indic_normalize"].IndicNormalizerFactory = MagicMock
