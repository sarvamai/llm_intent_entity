"""
Tests for llm_intent_entity core logic — no LLM API, no Google Sheets, no IndicNormalizer.
Covers: calculate_intent_accuracy(), calculate_entity_metrics(), build_prompt(),
load_and_validate_dataset(), prepare_evaluation_items(), process_llm_responses(),
and calculate_metrics() — all deterministic paths.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import pytest
import pandas as pd

# Allow imports from the src package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from llm_intent_entity.utilities import calculate_intent_accuracy, calculate_entity_metrics
from llm_intent_entity.main import (
    build_prompt,
    load_and_validate_dataset,
    prepare_evaluation_items,
    process_llm_responses,
    calculate_metrics,
)


# ─── calculate_intent_accuracy() ─────────────────────────────────────────────

def test_intent_accuracy_all_correct():
    assert calculate_intent_accuracy([1, 1, 1, 1]) == pytest.approx(1.0)


def test_intent_accuracy_all_wrong():
    assert calculate_intent_accuracy([0, 0, 0]) == pytest.approx(0.0)


def test_intent_accuracy_mixed():
    # 3 correct out of 4
    assert calculate_intent_accuracy([1, 1, 0, 1]) == pytest.approx(0.75)


def test_intent_accuracy_empty():
    assert calculate_intent_accuracy([]) == pytest.approx(0.0)


def test_intent_accuracy_single_correct():
    assert calculate_intent_accuracy([1]) == pytest.approx(1.0)


def test_intent_accuracy_single_wrong():
    assert calculate_intent_accuracy([0]) == pytest.approx(0.0)


# ─── calculate_entity_metrics() ──────────────────────────────────────────────

def test_entity_metrics_perfect_scores():
    result = calculate_entity_metrics([1.0, 1.0, 1.0])
    assert result["mean"] == pytest.approx(1.0)
    assert result["median"] == pytest.approx(1.0)
    assert result["std"] == pytest.approx(0.0)


def test_entity_metrics_zero_scores():
    result = calculate_entity_metrics([0.0, 0.0, 0.0])
    assert result["mean"] == pytest.approx(0.0)
    assert result["median"] == pytest.approx(0.0)


def test_entity_metrics_mixed():
    scores = [0.0, 0.5, 1.0]
    result = calculate_entity_metrics(scores)
    assert result["mean"] == pytest.approx(0.5)
    assert result["median"] == pytest.approx(0.5)
    assert result["std"] > 0.0


def test_entity_metrics_empty():
    result = calculate_entity_metrics([])
    assert result == {"mean": 0.0, "median": 0.0, "std": 0.0}


def test_entity_metrics_single_value():
    result = calculate_entity_metrics([0.7])
    assert result["mean"] == pytest.approx(0.7)
    assert result["median"] == pytest.approx(0.7)
    assert result["std"] == pytest.approx(0.0)


def test_entity_metrics_has_required_keys():
    result = calculate_entity_metrics([0.5, 0.8])
    assert set(result.keys()) == {"mean", "median", "std"}


# ─── build_prompt() ──────────────────────────────────────────────────────────

def test_build_prompt_contains_hypothesis():
    item = {
        "index": 0,
        "hypothesis": "Mujhe appointment book karni hai",
        "ground_truth": "Book an appointment for Monday",
        "context": "",
    }
    prompt = build_prompt(item)
    assert "Mujhe appointment book karni hai" in prompt


def test_build_prompt_contains_ground_truth():
    item = {
        "index": 1,
        "hypothesis": "Balance batao",
        "ground_truth": "Check account balance",
        "context": "Banking IVR call",
    }
    prompt = build_prompt(item)
    assert "Check account balance" in prompt


def test_build_prompt_contains_context():
    item = {
        "index": 2,
        "hypothesis": "Mujhe help chahiye",
        "ground_truth": "I need help",
        "context": "Customer support call centre",
    }
    prompt = build_prompt(item)
    assert "Customer support call centre" in prompt


def test_build_prompt_contains_index():
    item = {"index": 42, "hypothesis": "hello", "ground_truth": "hello", "context": ""}
    prompt = build_prompt(item)
    assert "42" in prompt


def test_build_prompt_valid_json_embedded():
    item = {"index": 0, "hypothesis": "test hyp", "ground_truth": "test gt", "context": "ctx"}
    prompt = build_prompt(item)
    # The INPUT section should be valid JSON
    input_section = prompt.split("**INPUT:**")[-1].strip()
    parsed = json.loads(input_section)
    assert parsed["index"] == 0
    assert parsed["hypothesis"] == "test hyp"
    assert parsed["ground_truth"] == "test gt"


def test_build_prompt_missing_context_defaults_empty():
    item = {"index": 0, "hypothesis": "test", "ground_truth": "gt"}
    prompt = build_prompt(item)
    input_section = prompt.split("**INPUT:**")[-1].strip()
    parsed = json.loads(input_section)
    assert parsed["context"] == ""


# ─── load_and_validate_dataset() ─────────────────────────────────────────────

def test_load_csv_with_required_columns():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("transcription,prediction,audio_filepath,language,context\n")
        f.write("ek sau rupye,ek so rupye,/audio/1.wav,hindi,banking\n")
        fname = f.name
    try:
        df = load_and_validate_dataset(
            fname,
            required_cols={"transcription", "prediction", "audio_filepath", "language"},
        )
        assert len(df) == 1
        assert df.loc[0, "transcription"] == "ek sau rupye"
    finally:
        os.unlink(fname)


def test_load_jsonl_with_required_columns():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"transcription": "hello", "prediction": "helo",
                            "audio_filepath": "/a.wav", "language": "english"}) + "\n")
        fname = f.name
    try:
        df = load_and_validate_dataset(fname, {"transcription", "prediction", "audio_filepath", "language"})
        assert len(df) == 1
    finally:
        os.unlink(fname)


def test_load_missing_column_raises_value_error():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("transcription,prediction\nhello,helo\n")
        fname = f.name
    try:
        with pytest.raises(ValueError, match="Missing columns"):
            load_and_validate_dataset(fname, {"transcription", "prediction", "audio_filepath", "language"})
    finally:
        os.unlink(fname)


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_and_validate_dataset("/does/not/exist.csv", {"transcription"})


def test_load_unsupported_format_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
        f.write("data\n")
        fname = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported"):
            load_and_validate_dataset(fname, {"transcription"})
    finally:
        os.unlink(fname)


# ─── prepare_evaluation_items() ──────────────────────────────────────────────

def test_prepare_evaluation_items_basic():
    df = pd.DataFrame({
        "norm_prediction": ["mujhe help chahiye", "balance batao"],
        "norm_reference": ["I need help", "check balance"],
    })
    items = prepare_evaluation_items(df)
    assert len(items) == 2
    assert items[0]["hypothesis"] == "mujhe help chahiye"
    assert items[0]["ground_truth"] == "I need help"
    assert "index" in items[0]


def test_prepare_evaluation_items_includes_context_when_present():
    df = pd.DataFrame({
        "norm_prediction": ["yes"],
        "norm_reference": ["yes"],
        "context": ["Banking IVR"],
    })
    items = prepare_evaluation_items(df)
    assert items[0]["context"] == "Banking IVR"


def test_prepare_evaluation_items_empty_context_when_missing():
    df = pd.DataFrame({
        "norm_prediction": ["hello"],
        "norm_reference": ["hello"],
    })
    items = prepare_evaluation_items(df)
    assert items[0]["context"] == ""


def test_prepare_evaluation_items_index_matches_df_index():
    df = pd.DataFrame({
        "norm_prediction": ["a", "b", "c"],
        "norm_reference": ["a", "b", "c"],
    })
    items = prepare_evaluation_items(df)
    for item, df_idx in zip(items, df.index):
        assert item["index"] == df_idx


# ─── process_llm_responses() ─────────────────────────────────────────────────

def _make_df_with_indices():
    return pd.DataFrame(
        {"norm_prediction": ["pred_0", "pred_1"], "norm_reference": ["ref_0", "ref_1"]},
        index=[0, 1],
    )


def test_process_llm_responses_fills_scores():
    df = _make_df_with_indices()
    successful = [
        {"key": {"index": 0, "hypothesis": "pred_0", "ground_truth": "ref_0", "context": ""},
         "response": {"intent_score": 1, "intent_explanation": "ok", "entity_score": 0.9,
                      "ground_truth_entities": "balance", "preserved_entities": "balance",
                      "missing_entities": "", "entity_explanation": "all present"}},
        {"key": {"index": 1, "hypothesis": "pred_1", "ground_truth": "ref_1", "context": ""},
         "response": {"intent_score": 0, "intent_explanation": "wrong pronoun", "entity_score": 0.5,
                      "ground_truth_entities": "name, amount", "preserved_entities": "name",
                      "missing_entities": "amount", "entity_explanation": "partial"}},
    ]
    result = process_llm_responses(successful, df)
    assert result.loc[0, "intent_score"] == 1
    assert result.loc[0, "entity_score"] == pytest.approx(0.9)
    assert result.loc[1, "intent_score"] == 0
    assert result.loc[1, "entity_score"] == pytest.approx(0.5)


def test_process_llm_responses_missing_index_fills_error():
    df = _make_df_with_indices()
    successful = []  # No responses at all
    result = process_llm_responses(successful, df)
    assert result.loc[0, "intent_score"] == -1
    assert result.loc[0, "entity_score"] == pytest.approx(-1.0)
    assert "ERROR" in result.loc[0, "intent_explanation"]


def test_process_llm_responses_partial_coverage():
    """Only index 0 responded; index 1 gets error sentinel."""
    df = _make_df_with_indices()
    successful = [
        {"key": {"index": 0, "hypothesis": "pred_0", "ground_truth": "ref_0", "context": ""},
         "response": {"intent_score": 1, "intent_explanation": "ok", "entity_score": 1.0,
                      "ground_truth_entities": "x", "preserved_entities": "x",
                      "missing_entities": "", "entity_explanation": "all"}},
    ]
    result = process_llm_responses(successful, df)
    assert result.loc[0, "intent_score"] == 1
    assert result.loc[1, "intent_score"] == -1  # error sentinel


# ─── calculate_metrics() ─────────────────────────────────────────────────────

def test_calculate_metrics_all_valid():
    df = pd.DataFrame({
        "intent_score": [1, 0, 1, 1],
        "entity_score": [1.0, 0.5, 0.8, 1.0],
    })
    result = calculate_metrics(df)
    assert result["total_samples"] == 4
    assert result["valid_samples"] == 4
    assert result["intent_accuracy"] == pytest.approx(0.75)
    assert result["entity_metrics"]["mean"] == pytest.approx(0.825)


def test_calculate_metrics_filters_error_sentinels():
    """Rows with score == -1 (error sentinels) should be excluded from valid_samples."""
    df = pd.DataFrame({
        "intent_score": [1, -1, 0],
        "entity_score": [0.9, -1.0, 0.5],
    })
    result = calculate_metrics(df)
    assert result["total_samples"] == 3
    assert result["valid_samples"] == 2
    assert result["intent_accuracy"] == pytest.approx(0.5)  # 1 out of 2 valid


def test_calculate_metrics_all_errors_returns_zeros():
    df = pd.DataFrame({
        "intent_score": [-1, -1],
        "entity_score": [-1.0, -1.0],
    })
    result = calculate_metrics(df)
    assert result["valid_samples"] == 0
    assert result["intent_accuracy"] == pytest.approx(0.0)
    assert result["entity_metrics"]["mean"] == pytest.approx(0.0)


def test_calculate_metrics_has_required_keys():
    df = pd.DataFrame({"intent_score": [1], "entity_score": [1.0]})
    result = calculate_metrics(df)
    assert "total_samples" in result
    assert "valid_samples" in result
    assert "intent_accuracy" in result
    assert "entity_metrics" in result


# ─── Indic language critical test cases (data-driven) ────────────────────────

@pytest.mark.parametrize("scores,expected_accuracy", [
    # Telugu safety case: allergy reporting — must be 1 (correct intent preserved)
    ([1], 1.0),
    # Tamil emergency escalation — agent booked appointment instead → 0
    ([0], 0.0),
    # Hindi mixed: 2 correct, 1 wrong
    ([1, 1, 0], pytest.approx(2 / 3)),
    # Odia noise case: 3 out of 4 intents preserved despite outdoor noise
    ([1, 0, 1, 1], 0.75),
])
def test_intent_accuracy_per_language_scenario(scores, expected_accuracy):
    assert calculate_intent_accuracy(scores) == expected_accuracy


@pytest.mark.parametrize("hypothesis,ground_truth,has_context", [
    ("Nenu okka allergy report cheyyali", "I need to report an allergy", True),
    ("Ennaku ippo marbaga vali irukku", "I have chest pain right now", False),
    ("En outstanding amount enna", "What is my outstanding amount", True),
    ("Mora account balance kana", "What is my account balance", True),
    ("mujhe appointment book karni hai", "Book an appointment for Monday", True),
])
def test_build_prompt_works_for_indic_utterances(hypothesis, ground_truth, has_context):
    item = {
        "index": 0,
        "hypothesis": hypothesis,
        "ground_truth": ground_truth,
        "context": "Voice AI evaluation" if has_context else "",
    }
    prompt = build_prompt(item)
    assert hypothesis in prompt
    assert ground_truth in prompt
    # Prompt should be non-trivially long
    assert len(prompt) > 100
