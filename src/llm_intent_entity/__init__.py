"""
LLM Intent Entity Evaluation Package

This package provides tools for evaluating intent and entity extraction accuracy 
in ASR outputs using Large Language Models.
"""

from .main import process_dataset_for_intent_entity_evaluation
from .llm_api import ChatCompletionsAPI
from .utilities import IndicNormalizer, calculate_intent_accuracy, calculate_entity_metrics

__version__ = "0.1.0"
__all__ = [
    "process_dataset_for_intent_entity_evaluation",
    "ChatCompletionsAPI", 
    "IndicNormalizer",
    "calculate_intent_accuracy",
    "calculate_entity_metrics"
]
