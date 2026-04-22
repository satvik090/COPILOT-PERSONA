from __future__ import annotations

from app.classifier import classify_patterns
from app.config import MAX_INJECTION_TOKENS


def _make_chunk(
    naming: str = "snake_case",
    error: str = "return_tuple",
    docstring: str = "google",
    annotation: str = "full",
) -> dict:
    return {
        "naming_convention": naming,
        "error_handling": error,
        "docstring_format": docstring,
        "annotation_density": annotation,
    }


def test_strong_pattern_detected_above_threshold() -> None:
    chunks = [_make_chunk(naming="snake_case") for _ in range(7)] + [
        _make_chunk(naming="camelCase") for _ in range(3)
    ]

    result = classify_patterns(chunks)

    assert result["strong_patterns"]["naming_conventions"] == "snake_case"
    assert result["confidence_scores"]["naming_conventions"] == 0.7


def test_weak_pattern_not_detected_below_threshold() -> None:
    chunks = [_make_chunk(naming="snake_case") for _ in range(5)] + [
        _make_chunk(naming="camelCase") for _ in range(5)
    ]

    result = classify_patterns(chunks)

    assert "naming_conventions" not in result["strong_patterns"]
    assert result["confidence_scores"]["naming_conventions"] == 0.5


def test_empty_chunks_returns_empty_summary() -> None:
    result = classify_patterns([])

    assert result["summary"] == ""
    assert result["strong_patterns"] == {}
    assert result["confidence_scores"] == {}


def test_summary_under_token_limit() -> None:
    chunks = [_make_chunk() for _ in range(10)]

    result = classify_patterns(chunks)
    estimated_tokens = len(result["summary"].split()) * 1.3

    assert estimated_tokens <= MAX_INJECTION_TOKENS


def test_all_dimensions_represented_in_summary() -> None:
    chunks = [_make_chunk() for _ in range(10)]

    result = classify_patterns(chunks)
    summary = result["summary"]

    assert "Uses snake_case naming consistently." in summary
    assert "Always includes type annotations on all arguments." in summary
    assert "Handles errors by returning tuples not exceptions." in summary
    assert "Writes Google-style docstrings with Args/Returns." in summary
