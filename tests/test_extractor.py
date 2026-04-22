from __future__ import annotations

from pathlib import Path

from app.extractor import extract_features, extract_functions


SAMPLE_FUNC_ANNOTATED = """
def calculate_total(items: list[float], 
                    tax_rate: float = 0.1) -> float:
    \"\"\"Calculate total with tax.
    
    Args:
        items: list of item prices
        tax_rate: tax rate as decimal
    
    Returns:
        total price including tax
    \"\"\"
    subtotal = sum(items)
    return subtotal * (1 + tax_rate)
"""

SAMPLE_FUNC_NO_ANNOTATIONS = """
def process_data(data, config):
    try:
        result = data.transform(config)
        return result
    except Exception as e:
        return None
"""

SAMPLE_FUNC_RETURN_NONE = """
def maybe_clean_data(value):
    cleaned_value = value.strip()
    if cleaned_value:
        return cleaned_value
    return None
"""

SAMPLE_DUNDER_AND_NORMAL = """
class Example:
    def __init__(self, value):
        self.value = value
        self.ready = True
        self.count = 1
        return None

def helper_function(data):
    cleaned = data.strip()
    if cleaned:
        return cleaned
    return data
"""

SAMPLE_SHORT_FUNCTION = """
def tiny():
    return 1
"""


def _write_source(tmp_path: Path, source: str, filename: str = "sample.py") -> Path:
    file_path = tmp_path / filename
    file_path.write_text(source, encoding="utf-8")
    return file_path


def _extract_single_function(tmp_path: Path, source: str) -> dict:
    file_path = _write_source(tmp_path, source)
    extracted = extract_functions(file_path)
    assert len(extracted) == 1
    return extracted[0]


def test_extract_returns_list(tmp_path: Path) -> None:
    file_path = _write_source(tmp_path, SAMPLE_FUNC_ANNOTATED)

    extracted = extract_functions(file_path)

    assert isinstance(extracted, list)


def test_skips_dunder_methods(tmp_path: Path) -> None:
    file_path = _write_source(tmp_path, SAMPLE_DUNDER_AND_NORMAL)

    extracted = extract_functions(file_path)
    names = [item["name"] for item in extracted]

    assert "__init__" not in names
    assert names == ["helper_function"]


def test_skips_short_functions(tmp_path: Path) -> None:
    file_path = _write_source(tmp_path, SAMPLE_SHORT_FUNCTION)

    extracted = extract_functions(file_path)

    assert extracted == []


def test_extracts_signature_correctly(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_ANNOTATED)

    assert extracted["signature"].startswith("def calculate_total(")
    assert "items: list[float]" in extracted["signature"]
    assert "tax_rate: float" in extracted["signature"]
    assert extracted["signature"].endswith("-> float")


def test_extracts_docstring(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_ANNOTATED)

    assert extracted["docstring"] is not None
    assert "Calculate total with tax." in extracted["docstring"]
    assert "Args:" in extracted["docstring"]
    assert "Returns:" in extracted["docstring"]


def test_extracts_annotations(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_ANNOTATED)

    assert extracted["args"][0]["name"] == "items"
    assert extracted["args"][0]["annotation"] == "list[float]"
    assert extracted["args"][0]["has_default"] is False
    assert extracted["args"][1]["name"] == "tax_rate"
    assert extracted["args"][1]["annotation"] == "float"
    assert extracted["args"][1]["has_default"] is True
    assert extracted["return_annotation"] == "float"
    assert extracted["has_type_annotations"] is True


def test_annotation_ratio_fully_annotated(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_ANNOTATED)

    assert extracted["annotation_ratio"] > 0.8


def test_annotation_ratio_no_annotations(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_NO_ANNOTATIONS)

    assert extracted["annotation_ratio"] == 0.0


def test_features_naming_snake_case(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_ANNOTATED)

    features = extract_features(extracted)

    assert features["naming_convention"] == "snake_case"


def test_features_error_handling_try_except(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_NO_ANNOTATIONS)

    features = extract_features(extracted)

    assert features["error_handling"] == "try_except"


def test_features_error_handling_return(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_RETURN_NONE)

    features = extract_features(extracted)

    assert features["error_handling"] == "return_none"


def test_features_docstring_google(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_ANNOTATED)

    features = extract_features(extracted)

    assert features["docstring_format"] == "google"


def test_features_docstring_none(tmp_path: Path) -> None:
    extracted = _extract_single_function(tmp_path, SAMPLE_FUNC_NO_ANNOTATIONS)

    features = extract_features(extracted)

    assert features["docstring_format"] == "none"
