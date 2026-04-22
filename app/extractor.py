from __future__ import annotations

import ast
import logging
import textwrap
from pathlib import Path

from app.config import MIN_FUNCTION_LINES


logger = logging.getLogger(__name__)


def extract_functions(file_path: Path) -> list[dict]:
    source_code = file_path.read_text(encoding="utf-8")

    try:
        module = ast.parse(source_code, filename=str(file_path))
    except SyntaxError as error:
        logger.warning("Failed to parse %s: %s", file_path, error)
        return []

    extracted_functions: list[dict] = []

    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_dunder_name(node.name):
            continue

        line_count = _get_line_count(node)
        if line_count < MIN_FUNCTION_LINES:
            continue

        function_record = _build_function_record(
            node=node,
            source_code=source_code,
            file_path=file_path,
            line_count=line_count,
        )
        extracted_functions.append(function_record)

    return extracted_functions


def extract_features(func: dict) -> dict:
    function_source = textwrap.dedent(func["full_source"])
    function_node = ast.parse(function_source).body[0]

    local_variable_names = _collect_local_variable_names(function_node)
    compared_names = [func["name"], *local_variable_names]

    if any(_is_snake_case(name) for name in compared_names) and all(
        _is_snake_case(name) for name in compared_names
    ):
        naming_convention = "snake_case"
    elif any(_is_camel_case(name) for name in compared_names) and all(
        _is_camel_case(name) for name in compared_names
    ):
        naming_convention = "camelCase"
    else:
        naming_convention = "mixed"

    docstring = func["docstring"]
    if docstring is None:
        docstring_format = "none"
    elif "Args:" in docstring or "Returns:" in docstring:
        docstring_format = "google"
    elif "-----" in docstring:
        docstring_format = "numpy"
    else:
        docstring_format = "plain"

    error_handling = "none"
    if any(isinstance(node, ast.Try) for node in ast.walk(function_node)):
        error_handling = "try_except"
    elif any(isinstance(node, ast.Raise) for node in ast.walk(function_node)):
        error_handling = "raises"
    elif any(_is_tuple_return(node) for node in ast.walk(function_node)):
        error_handling = "return_tuple"
    elif any(_is_none_return(node) for node in ast.walk(function_node)):
        error_handling = "return_none"

    annotation_ratio = func["annotation_ratio"]
    if annotation_ratio > 0.8:
        annotation_density = "full"
    elif 0.1 < annotation_ratio <= 0.8:
        annotation_density = "partial"
    else:
        annotation_density = "none"

    import_style = _collect_name_vocabulary(function_node)

    return {
        **func,
        "naming_convention": naming_convention,
        "error_handling": error_handling,
        "docstring_format": docstring_format,
        "import_style": import_style,
        "annotation_density": annotation_density,
    }


def _build_function_record(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_code: str,
    file_path: Path,
    line_count: int,
) -> dict:
    full_source = ast.get_source_segment(source_code, node) or ast.unparse(node)
    arguments = _extract_arguments(node.args)
    return_annotation = ast.unparse(node.returns) if node.returns is not None else None
    annotated_args = sum(1 for argument in arguments if argument["annotation"] is not None)
    total_args = len(arguments)
    annotation_ratio = annotated_args / total_args if total_args else 0.0
    has_type_annotations = return_annotation is not None or annotated_args > 0
    decorators = [ast.unparse(decorator) for decorator in node.decorator_list]

    return {
        "name": node.name,
        "signature": _build_signature(node),
        "docstring": ast.get_docstring(node),
        "full_source": full_source,
        "file_path": str(file_path),
        "line_count": line_count,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "decorators": decorators,
        "args": arguments,
        "return_annotation": return_annotation,
        "has_type_annotations": has_type_annotations,
        "annotation_ratio": annotation_ratio,
    }


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args_repr = ast.unparse(node.args)
    signature = f"{prefix} {node.name}({args_repr})"
    if node.returns is not None:
        signature = f"{signature} -> {ast.unparse(node.returns)}"
    return signature


def _extract_arguments(arguments: ast.arguments) -> list[dict]:
    extracted_arguments: list[dict] = []
    positional_arguments = [*arguments.posonlyargs, *arguments.args]
    positional_default_count = len(arguments.defaults)
    positional_default_start = len(positional_arguments) - positional_default_count

    for index, argument in enumerate(positional_arguments):
        extracted_arguments.append(
            {
                "name": argument.arg,
                "annotation": ast.unparse(argument.annotation)
                if argument.annotation is not None
                else None,
                "has_default": index >= positional_default_start,
            }
        )

    if arguments.vararg is not None:
        extracted_arguments.append(
            {
                "name": argument_name_with_prefix(arguments.vararg, "*"),
                "annotation": ast.unparse(arguments.vararg.annotation)
                if arguments.vararg.annotation is not None
                else None,
                "has_default": False,
            }
        )

    for argument, default_value in zip(arguments.kwonlyargs, arguments.kw_defaults):
        extracted_arguments.append(
            {
                "name": argument.arg,
                "annotation": ast.unparse(argument.annotation)
                if argument.annotation is not None
                else None,
                "has_default": default_value is not None,
            }
        )

    if arguments.kwarg is not None:
        extracted_arguments.append(
            {
                "name": argument_name_with_prefix(arguments.kwarg, "**"),
                "annotation": ast.unparse(arguments.kwarg.annotation)
                if arguments.kwarg.annotation is not None
                else None,
                "has_default": False,
            }
        )

    return extracted_arguments


def argument_name_with_prefix(argument: ast.arg, prefix: str) -> str:
    return f"{prefix}{argument.arg}"


def _get_line_count(node: ast.AST) -> int:
    start_line = getattr(node, "lineno", 0)
    end_line = getattr(node, "end_lineno", start_line)
    return max(0, end_line - start_line + 1)


def _is_dunder_name(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _collect_local_variable_names(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    names: set[str] = set()

    for node in ast.walk(function_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)

    return sorted(names)


def _collect_name_vocabulary(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, int]:
    vocabulary: dict[str, int] = {}

    for node in ast.walk(function_node):
        if isinstance(node, ast.Name):
            vocabulary[node.id] = vocabulary.get(node.id, 0) + 1

    return dict(sorted(vocabulary.items()))


def _is_tuple_return(node: ast.AST) -> bool:
    return isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)


def _is_none_return(node: ast.AST) -> bool:
    if not isinstance(node, ast.Return):
        return False
    if node.value is None:
        return True
    return isinstance(node.value, ast.Constant) and node.value.value is None


def _is_snake_case(name: str) -> bool:
    if not name:
        return False
    stripped = name.replace("_", "")
    if not stripped.isidentifier():
        return False
    return name == name.lower() and "__" not in name


def _is_camel_case(name: str) -> bool:
    if not name or "_" in name or not name.isidentifier():
        return False
    return name[0].islower() and any(character.isupper() for character in name[1:])
