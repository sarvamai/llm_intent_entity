import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "cache_match",
    Path(__file__).resolve().parents[1] / "src/llm_intent_entity/cache_match.py",
)
_cache_match = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache_match)
responses_needed_from_cache = _cache_match.responses_needed_from_cache


def test_cache_hits_skip_entries_the_dataset_does_not_need():
    cached = {
        ("a", "b", ""): {"key": {"index": 1}},
        ("unused", "x", ""): {"key": {"index": 9}},
        ("c", "d", "ctx"): {"key": {"index": 2}},
    }
    items = [
        {"hypothesis": "c", "ground_truth": "d", "context": "ctx"},
        {"hypothesis": "a", "ground_truth": "b"},
    ]

    got = responses_needed_from_cache(cached, items)

    assert [row["key"]["index"] for row in got] == [1, 2]


def test_missing_context_matches_an_empty_cache_context():
    cached = {("h", "g", ""): {"key": {"index": 4}}}
    items = [{"hypothesis": "h", "ground_truth": "g"}]

    got = responses_needed_from_cache(cached, items)

    assert [row["key"]["index"] for row in got] == [4]


def test_non_hashable_context_does_not_break_matching():
    cached = {("h", "g", ""): {"key": {"index": 4}}}
    items = [
        {"hypothesis": "x", "ground_truth": "y", "context": ["turn 1", "turn 2"]},
        {"hypothesis": "h", "ground_truth": "g"},
    ]

    got = responses_needed_from_cache(cached, items)

    assert [row["key"]["index"] for row in got] == [4]
