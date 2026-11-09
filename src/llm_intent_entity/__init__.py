"""
LLM Intent Entity Evaluation Package

This package provides tools for evaluating intent and entity extraction accuracy 
in ASR outputs using Large Language Models.
"""

from .main import process_dataset_for_intent_entity_evaluation
from .llm_api import ChatCompletionsAPI
from .utilities import IndicNormalizer, calculate_intent_accuracy, calculate_entity_metrics
from .typed_entities import (
    CanonicalEntity,
    EntityMatch,
    EntityType,
    MatchPolicy,
    TypedEntityCandidate,
    TypedEntityScore,
    canonicalize_entity,
    parse_amount,
    parse_identifier,
    score_typed_entities,
)

__version__ = "0.1.0"
__all__ = [
    "process_dataset_for_intent_entity_evaluation",
    "ChatCompletionsAPI", 
    "IndicNormalizer",
    "calculate_intent_accuracy",
    "calculate_entity_metrics",
    "CanonicalEntity",
    "EntityMatch",
    "EntityType",
    "MatchPolicy",
    "TypedEntityCandidate",
    "TypedEntityScore",
    "canonicalize_entity",
    "parse_amount",
    "parse_identifier",
    "score_typed_entities",
]
