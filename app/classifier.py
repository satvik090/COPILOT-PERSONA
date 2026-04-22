from __future__ import annotations

from collections import Counter

from app.config import MAX_INJECTION_TOKENS, PATTERN_CONFIDENCE_THRESHOLD


def classify_patterns(chunks: list[dict]) -> dict:
    chunk_count = len(chunks)
    if chunk_count == 0:
        return {
            "strong_patterns": {},
            "confidence_scores": {},
            "chunk_count": 0,
            "summary": "",
        }

    feature_counters = {
        "naming_conventions": Counter(
            chunk["naming_convention"] for chunk in chunks if chunk.get("naming_convention")
        ),
        "error_handling_styles": Counter(
            chunk["error_handling"] for chunk in chunks if chunk.get("error_handling")
        ),
        "docstring_formats": Counter(
            chunk["docstring_format"] for chunk in chunks if chunk.get("docstring_format")
        ),
        "annotation_densities": Counter(
            chunk["annotation_density"] for chunk in chunks if chunk.get("annotation_density")
        ),
    }

    strong_patterns: dict[str, str] = {}
    confidence_scores: dict[str, float] = {}

    for dimension, counts in feature_counters.items():
        if not counts:
            continue

        dominant_value, dominant_count = counts.most_common(1)[0]
        confidence = dominant_count / chunk_count
        confidence_scores[dimension] = confidence

        if confidence >= PATTERN_CONFIDENCE_THRESHOLD:
            strong_patterns[dimension] = dominant_value

    summary = build_summary(strong_patterns, confidence_scores)

    return {
        "strong_patterns": strong_patterns,
        "confidence_scores": confidence_scores,
        "chunk_count": chunk_count,
        "summary": summary,
    }


def build_summary(strong_patterns: dict, confidence_scores: dict) -> str:
    sentences: list[str] = []

    sentence_builders = {
        "naming_conventions": {
            "snake_case": "Uses snake_case naming consistently.",
            "camelCase": "Uses camelCase naming consistently.",
            "mixed": "Mixes naming styles across functions.",
        },
        "annotation_densities": {
            "full": "Always includes type annotations on all arguments.",
            "partial": "Uses type annotations on some arguments.",
            "none": "Rarely includes type annotations.",
        },
        "error_handling_styles": {
            "try_except": "Handles errors with try/except blocks.",
            "return_tuple": "Handles errors by returning tuples not exceptions.",
            "return_none": "Handles errors by returning None.",
            "raises": "Handles errors by raising exceptions.",
            "none": "Shows no consistent error-handling style.",
        },
        "docstring_formats": {
            "google": "Writes Google-style docstrings with Args/Returns.",
            "numpy": "Writes NumPy-style docstrings with section underlines.",
            "plain": "Uses plain descriptive docstrings.",
            "none": "Usually omits docstrings.",
        },
    }

    for dimension in (
        "naming_conventions",
        "annotation_densities",
        "error_handling_styles",
        "docstring_formats",
    ):
        confidence = confidence_scores.get(dimension, 0.0)
        if confidence < PATTERN_CONFIDENCE_THRESHOLD:
            continue

        pattern_value = strong_patterns.get(dimension)
        if pattern_value is None:
            continue

        sentence = sentence_builders.get(dimension, {}).get(pattern_value)
        if sentence:
            sentences.append(sentence)

    if not sentences:
        return ""

    selected_sentences = sentences[:]

    while selected_sentences:
        summary = "Developer patterns detected: " + " ".join(selected_sentences)
        estimated_tokens = len(summary.split()) * 1.3
        if estimated_tokens <= MAX_INJECTION_TOKENS:
            return summary
        selected_sentences.pop()

    return ""
