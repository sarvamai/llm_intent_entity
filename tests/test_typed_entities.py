import json
from pathlib import Path

import pandas as pd
import pytest

import llm_intent_entity.main as main_module
from llm_intent_entity.main import process_llm_responses
from llm_intent_entity.typed_entities import (
    EntityType,
    MatchPolicy,
    TypedEntityCandidate,
    canonicalize_entity,
    parse_amount,
    parse_identifier,
    score_typed_entities,
)


ISSUE_5_REFERENCE = "fifty-eight seven twenty double teen nau forty-six"
ISSUE_5_HYPOTHESIS = "fifty eight seven twenty double three nine forty six"


def candidate(entity_type: EntityType, raw: str) -> TypedEntityCandidate:
    return TypedEntityCandidate(entity_type=entity_type, raw=raw)


def test_issue_5_code_mixed_phone_is_exactly_preserved() -> None:
    reference = parse_identifier(ISSUE_5_REFERENCE, language="hi")
    hypothesis = parse_identifier(ISSUE_5_HYPOTHESIS, language="hi")

    assert reference.value == "5872033946"
    assert hypothesis.value == reference.value

    score = score_typed_entities(
        [candidate(EntityType.PHONE_NUMBER, ISSUE_5_REFERENCE)],
        [candidate(EntityType.PHONE_NUMBER, ISSUE_5_HYPOTHESIS)],
        language="hi",
    )
    assert score.score == 1.0
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.named_entity_error_rate == 0.0
    assert not score.review_required


def test_one_digit_identifier_difference_is_a_substitution() -> None:
    score = score_typed_entities(
        [candidate(EntityType.ACCOUNT_ID, "१२३४५६७८९०")],
        [candidate(EntityType.ACCOUNT_ID, "1234567891")],
        language="hi",
    )

    assert score.score == 0.0
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0
    assert score.named_entity_error_rate == 1.0
    assert score.substitutions == 1


@pytest.mark.parametrize("raw", ["₹5,000", "पाँच हज़ार", "five thousand"])
def test_amount_variants_canonicalize_to_inr_5000(raw: str) -> None:
    parsed, currency = parse_amount(raw, language="hi")

    assert parsed.value == "5000"
    assert currency == "INR"
    assert not parsed.review_required


@pytest.mark.parametrize(
    ("language", "raw"),
    [
        ("bn", "পাঁচ হাজার"),
        ("ta", "ஐயாயிரம் ரூபாய்"),
        ("te", "ఐదు వేలు"),
        ("ml", "അയ്യായിരം"),
        ("ur", "پاںچ ہزار"),
    ],
)
def test_agglutinative_and_indian_script_amount_words(language: str, raw: str) -> None:
    parsed, currency = parse_amount(raw, language=language)

    assert parsed.value == "5000"
    assert currency == "INR"


def test_indian_scales_and_decimals_are_supported() -> None:
    lakh, _ = parse_amount("बारह लाख", language="hi")
    crore, _ = parse_amount("two crore", language="en")
    decimal, _ = parse_amount("five point zero five", language="en")

    assert lakh.value == "1200000"
    assert crore.value == "20000000"
    assert decimal.value == "5.05"


def test_indian_language_decimal_marker_is_supported() -> None:
    decimal, _ = parse_amount("ஐந்து புள்ளி பூஜ்ஜியம் ஐந்து", language="ta")

    assert decimal.value == "5.05"


def test_entity_types_prevent_identifier_amount_or_date_collisions() -> None:
    amount_vs_identifier = score_typed_entities(
        [candidate(EntityType.AMOUNT, "₹5,000")],
        [candidate(EntityType.ACCOUNT_ID, "5000")],
        language="hi",
    )
    date_vs_identifier = score_typed_entities(
        [candidate(EntityType.DATE, "2026-07-18")],
        [candidate(EntityType.ACCOUNT_ID, "20260718")],
        language="en",
    )

    assert amount_vs_identifier.score == 0.0
    assert date_vs_identifier.score == 0.0
    assert amount_vs_identifier.named_entity_error_rate == 2.0
    assert date_vs_identifier.named_entity_error_rate == 2.0


def test_ambiguous_date_abstains_instead_of_guessing() -> None:
    score = score_typed_entities(
        [candidate(EntityType.DATE, "03/04/2026")],
        [candidate(EntityType.DATE, "3 April 2026")],
        language="en",
    )

    assert score.score is None
    assert score.review_required
    assert "ambiguous day/month order" in score.review_reasons[0]


def test_date_order_can_be_configured_explicitly() -> None:
    score = score_typed_entities(
        [candidate(EntityType.DATE, "03/04/2026")],
        [candidate(EntityType.DATE, "3 April 2026")],
        language="en",
        policy=MatchPolicy(date_order="DMY"),
    )

    assert score.score == 1.0


def test_invalid_repeat_expression_abstains() -> None:
    parsed = parse_identifier("double thirteen nine", language="en")

    assert parsed.value is None
    assert parsed.review_required
    assert "exactly one digit" in parsed.review_reason


def test_name_and_location_matching_are_exact_by_default_and_configurable() -> None:
    exact = score_typed_entities(
        [candidate(EntityType.PERSON, "John Smith")],
        [candidate(EntityType.PERSON, "Jon Smith")],
    )
    fuzzy = score_typed_entities(
        [candidate(EntityType.LOCATION, "Bengaluru")],
        [candidate(EntityType.LOCATION, "Bangaluru")],
        policy=MatchPolicy(location_mode="fuzzy", fuzzy_threshold=0.85),
    )

    assert exact.score == 0.0
    assert fuzzy.score == 1.0


def test_partial_results_report_precision_recall_f1_and_neer() -> None:
    score = score_typed_entities(
        [
            candidate(EntityType.PERSON, "Asha"),
            candidate(EntityType.LOCATION, "Pune"),
        ],
        [
            candidate(EntityType.PERSON, "Asha"),
            candidate(EntityType.LOCATION, "Mumbai"),
            candidate(EntityType.AMOUNT, "₹500"),
        ],
    )

    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == pytest.approx(1 / 2)
    assert score.f1 == pytest.approx(0.4)
    assert score.named_entity_error_rate == 1.0
    assert score.substitutions == 1
    assert score.insertions == 1


def test_phone_length_policy_abstains_on_implausible_span() -> None:
    entity = canonicalize_entity(candidate(EntityType.PHONE_NUMBER, "one two three"))

    assert entity.review_required
    assert "7-15 digits" in entity.review_reason


def test_process_responses_overrides_issue_5_llm_scores_deterministically() -> None:
    dataframe = pd.DataFrame(
        {
            "norm_reference": [ISSUE_5_REFERENCE],
            "norm_prediction": [ISSUE_5_HYPOTHESIS],
        }
    )
    responses = [
        {
            "key": {
                "index": 0,
                "ground_truth": ISSUE_5_REFERENCE,
                "hypothesis": ISSUE_5_HYPOTHESIS,
                "language": "hi",
            },
            "response": {
                "intent_score": 0,
                "intent_explanation": "Surface forms differ.",
                "entity_score": 0.0,
                "ground_truth_entities": ISSUE_5_REFERENCE,
                "preserved_entities": "",
                "missing_entities": ISSUE_5_REFERENCE,
                "entity_explanation": "Surface forms differ.",
                "typed_entities_complete": True,
                "ground_truth_typed_entities": [
                    {"entity_type": "PHONE_NUMBER", "raw": ISSUE_5_REFERENCE}
                ],
                "hypothesis_typed_entities": [
                    {"entity_type": "PHONE_NUMBER", "raw": ISSUE_5_HYPOTHESIS}
                ],
            },
        }
    ]

    result = process_llm_responses(responses, dataframe)

    assert result.loc[0, "intent_score"] == 1
    assert result.loc[0, "entity_score"] == 1.0
    assert result.loc[0, "entity_f1"] == 1.0
    assert result.loc[0, "named_entity_error_rate"] == 0.0
    assert result.loc[0, "entity_scoring_method"] == "typed_deterministic"
    typed_reference = json.loads(result.loc[0, "ground_truth_typed_entities"])
    assert typed_reference[0]["canonical_digits"] == "5872033946"


def test_old_cached_responses_keep_legacy_entity_score() -> None:
    dataframe = pd.DataFrame(
        {"norm_reference": ["hello"], "norm_prediction": ["hello"]}
    )
    responses = [
        {
            "key": {"index": 0, "ground_truth": "hello", "hypothesis": "hello"},
            "response": {
                "intent_score": 1,
                "intent_explanation": "Same intent.",
                "entity_score": 0.75,
                "ground_truth_entities": "legacy",
                "preserved_entities": "legacy",
                "missing_entities": "",
                "entity_explanation": "Cached response.",
            },
        }
    ]

    result = process_llm_responses(responses, dataframe)

    assert result.loc[0, "entity_score"] == 0.75
    assert result.loc[0, "entity_scoring_method"] == "legacy_llm"


def test_malformed_typed_cache_payload_abstains_without_crashing() -> None:
    dataframe = pd.DataFrame({"norm_reference": ["123"], "norm_prediction": ["123"]})
    responses = [
        {
            "key": {"index": 0, "ground_truth": "123", "hypothesis": "123"},
            "response": {
                "intent_score": 1,
                "typed_entities_complete": True,
                "ground_truth_typed_entities": [
                    {"entity_type": "UNSUPPORTED", "raw": "123"}
                ],
                "hypothesis_typed_entities": [],
            },
        }
    ]

    result = process_llm_responses(responses, dataframe)

    assert result.loc[0, "entity_score"] == -1.0
    assert bool(result.loc[0, "entity_review_required"])
    assert "Invalid typed entity payload" in result.loc[0, "entity_review_reasons"]


def test_language_aware_cache_reuses_pre_language_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "outputs" / "cache"
    cache_dir.mkdir(parents=True)
    cached = {
        "key": {
            "index": 0,
            "hypothesis": "hello",
            "ground_truth": "hello",
            "context": "",
        },
        "response": {"entity_score": 1.0},
    }
    (cache_dir / "sample_intent_entity_cache.jsonl").write_text(
        json.dumps(cached) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)

    class ApiThatMustNotBeCalled:
        def append_to_request_queue(self, **_: object) -> None:
            raise AssertionError("legacy cache entry should have been reused")

        def generate_responses_from_queue(self, **_: object) -> tuple[list, list]:
            raise AssertionError("legacy cache entry should have been reused")

    successful, failed = main_module.query_llm_for_intent_entity_evaluation(
        [
            {
                "index": 0,
                "hypothesis": "hello",
                "ground_truth": "hello",
                "context": "",
                "language": "hi",
            }
        ],
        "sample",
        ApiThatMustNotBeCalled(),
    )

    assert successful == [cached]
    assert failed == []
