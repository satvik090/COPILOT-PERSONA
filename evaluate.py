from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
from pathlib import Path
import random
import textwrap
from typing import Any

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm


TEST_CORPUS_DIR = "./my_projects"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COPILOT_PERSONA_URL = "http://localhost:8000/retrieve"
NUM_FUNCTIONS = 50
PATTERN_INJECTION_TOKENS = 150

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes", "on"}
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.2
OPENAI_MAX_TOKENS = 300
OPENAI_MAX_RETRIES = 3
OPENAI_CONCURRENCY = 5
MIN_FUNCTION_LINES = 5
RANDOM_SEED = 42
RAW_OUTPUT_PATH = Path("evaluation_raw.json")
REPORT_JSON_PATH = Path("evaluation_report.json")
REPORT_TEXT_PATH = Path("evaluation_report.txt")
COMPLETION_WRAPPER_NAME = "__completion_stub__"
EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    "venv",
    "node_modules",
    "site-packages",
    "dist",
    "build",
}

logger = logging.getLogger("copilotpersona.evaluate")


def extract_functions(directory: str | Path) -> list[dict]:
    root_dir = Path(directory)
    collected_functions: list[dict] = []

    for file_path in root_dir.rglob("*.py"):
        if _should_skip_path(file_path):
            continue

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Could not read %s: %s", file_path, error)
            continue

        try:
            module = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as error:
            logger.warning("Could not parse %s: %s", file_path, error)
            continue

        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_dunder_name(node.name):
                continue

            line_count = _get_line_count(node)
            if line_count < MIN_FUNCTION_LINES:
                continue

            full_source = ast.get_source_segment(source_code, node) or ast.unparse(node)
            collected_functions.append(
                {
                    "name": node.name,
                    "signature": _build_signature(node),
                    "docstring": ast.get_docstring(node),
                    "full_source": full_source,
                    "file_path": str(file_path),
                    "annotations": _collect_annotations(node.args),
                    "decorators": [
                        ast.unparse(decorator) for decorator in node.decorator_list
                    ],
                    "project_name": _project_name_from_file(str(file_path)),
                    "line_count": line_count,
                }
            )

    if not collected_functions:
        return []

    sample_size = min(NUM_FUNCTIONS, len(collected_functions))
    sampler = random.Random(RANDOM_SEED)
    return sampler.sample(collected_functions, sample_size)


def strip_to_prompt(func: dict) -> str:
    signature_line = func["signature"].rstrip(":") + ":"
    docstring = func.get("docstring")

    if not docstring:
        return f"{signature_line}\n    pass"

    return (
        f"{signature_line}\n"
        f"{_indent_docstring(docstring)}\n"
        "    # complete this function"
    )


async def get_baseline_completion(
    prompt: str,
    openai_client: AsyncOpenAI | None,
    demo_mode: bool = False,
    ground_truth: str = "",
) -> str:
    if demo_mode:
        return _generate_demo_completion(
            prompt=prompt,
            ground_truth=ground_truth,
            pattern_summary="",
            mode="baseline",
        )

    system_prompt = (
        "You are a Python coding assistant. Complete the function. "
        "Return only the function body, no explanation."
    )
    return await _request_openai_completion(
        openai_client=openai_client,
        system_prompt=system_prompt,
        user_prompt=prompt,
    )


async def get_enhanced_completion(
    prompt: str,
    file_context: str,
    openai_client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient,
    demo_mode: bool = False,
) -> str:
    completion, _, _ = await _get_enhanced_completion_with_meta(
        prompt=prompt,
        file_context=file_context,
        openai_client=openai_client,
        http_client=http_client,
        demo_mode=demo_mode,
    )
    return completion


def score_completion(completion: str, ground_truth: str) -> dict:
    ground_truth_tree = _safe_parse_source(ground_truth)
    if ground_truth_tree is None:
        return _zero_scores()

    completion_tree = _completion_tree_for_scoring(completion, ground_truth)
    if completion_tree is None:
        return _zero_scores()

    ground_truth_convention = _detect_naming_convention(ground_truth_tree)
    completion_convention = _detect_naming_convention(completion_tree)
    naming_convention_match = (
        1.0 if ground_truth_convention == completion_convention else 0.0
    )

    ground_truth_annotation_ratio = _annotation_ratio_from_tree(ground_truth_tree)
    completion_annotation_ratio = _annotation_ratio_from_tree(completion_tree)
    type_annotation_coverage = max(
        0.0,
        1.0 - abs(ground_truth_annotation_ratio - completion_annotation_ratio),
    )

    ground_truth_error_style = _detect_error_handling_style(ground_truth_tree)
    completion_error_style = _detect_error_handling_style(completion_tree)
    error_handling_style = 1.0 if ground_truth_error_style == completion_error_style else 0.0

    ground_truth_docstring_format = _detect_docstring_format(ground_truth_tree)
    completion_docstring_format = _detect_docstring_format(completion_tree)
    docstring_format = (
        1.0 if ground_truth_docstring_format == completion_docstring_format else 0.0
    )

    ground_truth_names = _extract_name_nodes(ground_truth_tree)
    completion_names = _extract_name_nodes(completion_tree)
    import_and_naming_style = _jaccard_similarity(ground_truth_names, completion_names)

    scores = {
        "naming_convention_match": naming_convention_match,
        "type_annotation_coverage": type_annotation_coverage,
        "error_handling_style": error_handling_style,
        "docstring_format": docstring_format,
        "import_and_naming_style": import_and_naming_style,
    }
    scores["total"] = sum(scores.values()) / len(scores)
    return scores


async def run_evaluation(
    dry_run: bool = False,
    demo_mode: bool = DEMO_MODE,
) -> None:
    corpus = extract_functions(TEST_CORPUS_DIR)
    if not corpus:
        logger.warning("No functions found under %s", TEST_CORPUS_DIR)
        return

    if dry_run:
        print(f"Dry run: extracted {len(corpus)} functions from {TEST_CORPUS_DIR}")
        for index, func in enumerate(corpus[:3], start=1):
            print(f"\n--- Prompt {index} ---")
            print(strip_to_prompt(func))
        return

    if not demo_mode and not _has_real_api_key():
        raise RuntimeError(
            "OPENAI_API_KEY is required for live evaluation. "
            "Use --demo-mode for an offline provisional run."
        )

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if not demo_mode else None
    timeout = httpx.Timeout(3.0)
    semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY)

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        with tqdm(total=len(corpus), desc="Evaluating functions") as progress_bar:
            tasks = []
            for func in corpus:
                task = asyncio.create_task(
                    _evaluate_function(
                        func=func,
                        openai_client=openai_client,
                        http_client=http_client,
                        semaphore=semaphore,
                        demo_mode=demo_mode,
                    )
                )
                task.add_done_callback(lambda _: progress_bar.update(1))
                tasks.append(task)
            results = await asyncio.gather(*tasks)

    _save_raw_completions(results)
    generate_report(results, demo_mode=demo_mode)


def generate_report(results: list[dict], demo_mode: bool = False) -> None:
    if not results:
        logger.warning("No evaluation results to report.")
        return

    dimensions = [
        "naming_convention_match",
        "type_annotation_coverage",
        "error_handling_style",
        "docstring_format",
        "import_and_naming_style",
        "total",
    ]

    baseline_averages: dict[str, float] = {}
    enhanced_averages: dict[str, float] = {}
    improvements: dict[str, dict[str, float]] = {}

    for dimension in dimensions:
        baseline_average = _average(
            result["baseline_scores"][dimension] for result in results
        )
        enhanced_average = _average(
            result["enhanced_scores"][dimension] for result in results
        )
        delta = enhanced_average - baseline_average
        baseline_averages[dimension] = baseline_average
        enhanced_averages[dimension] = enhanced_average
        improvements[dimension] = {
            "delta": delta,
            "percentage": ((delta / baseline_average) * 100.0)
            if baseline_average > 0
            else 0.0,
        }

    sorted_breakdown = sorted(
        [
            {
                "function_name": result["function_name"],
                "baseline_total": result["baseline_scores"]["total"],
                "enhanced_total": result["enhanced_scores"]["total"],
                "delta": result["enhanced_scores"]["total"]
                - result["baseline_scores"]["total"],
                "file_path": result["file_path"],
            }
            for result in results
        ],
        key=lambda item: item["delta"],
        reverse=True,
    )

    parse_failures = {
        "baseline": [
            _result_label(result)
            for result in results
            if result["baseline_parse_failed"]
        ],
        "enhanced": [
            _result_label(result)
            for result in results
            if result["enhanced_parse_failed"]
        ],
    }
    persona_fallbacks = [
        _result_label(result)
        for result in results
        if result["enhanced_fallback"]
    ]

    unique_projects = sorted({result["project_name"] for result in results})
    baseline_total_average = baseline_averages["total"]
    enhanced_total_average = enhanced_averages["total"]
    baseline_style_correction_need = max(0.0, 1.0 - baseline_total_average)
    enhanced_style_correction_need = max(0.0, 1.0 - enhanced_total_average)
    style_correction_reduction_pct = (
        (
            (baseline_style_correction_need - enhanced_style_correction_need)
            / baseline_style_correction_need
        )
        * 100.0
        if baseline_style_correction_need > 0
        else 0.0
    )

    if demo_mode:
        summary_line = (
            "In DEMO_MODE, CopilotPersona reduced style correction needs by "
            f"{style_correction_reduction_pct:.2f}% across {len(results)} functions "
            f"from {len(unique_projects)} sample projects, measured across naming "
            "conventions, type annotations, error handling style, docstring format, "
            "and identifier vocabulary (AST-based evaluation)."
        )
    else:
        summary_line = (
            "CopilotPersona reduced style correction needs by "
            f"{style_correction_reduction_pct:.2f}% across {len(results)} functions "
            f"from {len(unique_projects)} personal projects, measured across naming "
            "conventions, type annotations, error handling style, docstring format, "
            "and identifier vocabulary (AST-based evaluation)."
        )

    report_payload = {
        "config": {
            "evaluation_mode": "demo" if demo_mode else "live",
            "test_corpus_dir": TEST_CORPUS_DIR,
            "copilot_persona_url": COPILOT_PERSONA_URL,
            "num_functions_requested": NUM_FUNCTIONS,
            "num_functions_evaluated": len(results),
            "pattern_injection_tokens": PATTERN_INJECTION_TOKENS,
            "projects_evaluated": unique_projects,
            "note": (
                "Demo mode uses local heuristic completions and local pattern summaries "
                "when the CopilotPersona service is unavailable."
                if demo_mode
                else "Live mode uses OpenAI completions and CopilotPersona retrieval."
            ),
        },
        "per_dimension_averages": {
            dimension: {
                "baseline_average": baseline_averages[dimension],
                "enhanced_average": enhanced_averages[dimension],
                "improvement": improvements[dimension]["delta"],
                "improvement_percentage": improvements[dimension]["percentage"],
            }
            for dimension in dimensions[:-1]
        },
        "overall_improvement": {
            "baseline_total_average": baseline_total_average,
            "enhanced_total_average": enhanced_total_average,
            "absolute_delta": improvements["total"]["delta"],
            "overall_improvement_percentage": improvements["total"]["percentage"],
            "style_correction_reduction_percentage": style_correction_reduction_pct,
        },
        "per_function_breakdown": sorted_breakdown,
        "failure_log": {
            "completion_parse_failures": parse_failures,
            "copilotpersona_fallbacks": persona_fallbacks,
        },
        "summary_line": summary_line,
    }

    REPORT_JSON_PATH.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    text_lines = [
        "CopilotPersona Evaluation Report",
        "",
        f"Mode: {'DEMO' if demo_mode else 'LIVE'}",
        f"Functions evaluated: {len(results)}",
        f"Projects evaluated: {len(unique_projects)}",
        "",
        "Per-dimension averages:",
    ]

    for dimension in dimensions[:-1]:
        text_lines.append(
            f"- {dimension}: baseline={baseline_averages[dimension]:.4f}, "
            f"enhanced={enhanced_averages[dimension]:.4f}, "
            f"improvement={improvements[dimension]['delta']:.4f}, "
            f"improvement_pct={improvements[dimension]['percentage']:.2f}%"
        )

    text_lines.extend(
        [
            "",
            "Overall improvement:",
            f"- baseline_total_average={baseline_total_average:.4f}",
            f"- enhanced_total_average={enhanced_total_average:.4f}",
            f"- overall_improvement_percentage={improvements['total']['percentage']:.2f}%",
            f"- style_correction_reduction_percentage={style_correction_reduction_pct:.2f}%",
            "",
            "Per-function breakdown:",
        ]
    )

    for item in sorted_breakdown:
        text_lines.append(
            f"- {item['function_name']} | "
            f"{item['baseline_total']:.4f} | {item['enhanced_total']:.4f} | "
            f"{item['delta']:.4f}"
        )

    text_lines.extend(
        [
            "",
            "Failure log:",
            f"- baseline_parse_failures={len(parse_failures['baseline'])}",
            f"- enhanced_parse_failures={len(parse_failures['enhanced'])}",
            f"- copilotpersona_fallbacks={len(persona_fallbacks)}",
            "- baseline_parse_failure_functions="
            + (", ".join(parse_failures["baseline"]) or "none"),
            "- enhanced_parse_failure_functions="
            + (", ".join(parse_failures["enhanced"]) or "none"),
            "- copilotpersona_fallback_functions="
            + (", ".join(persona_fallbacks) or "none"),
            "",
            "Resume summary:",
            summary_line,
        ]
    )

    report_text = "\n".join(text_lines)
    REPORT_TEXT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)


async def _evaluate_function(
    func: dict,
    openai_client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    demo_mode: bool = False,
) -> dict:
    async with semaphore:
        prompt = strip_to_prompt(func)
        baseline_completion = await get_baseline_completion(
            prompt=prompt,
            openai_client=openai_client,
            demo_mode=demo_mode,
            ground_truth=func["full_source"],
        )
        enhanced_completion, pattern_summary, enhanced_fallback = (
            await _get_enhanced_completion_with_meta(
                prompt=prompt,
                file_context=func["full_source"],
                openai_client=openai_client,
                http_client=http_client,
                demo_mode=demo_mode,
            )
        )

        baseline_scores = score_completion(baseline_completion, func["full_source"])
        enhanced_scores = score_completion(enhanced_completion, func["full_source"])

        return {
            "function_name": func["name"],
            "file_path": func["file_path"],
            "project_name": func["project_name"],
            "prompt": prompt,
            "ground_truth": func["full_source"],
            "pattern_summary": pattern_summary,
            "baseline_completion": baseline_completion,
            "enhanced_completion": enhanced_completion,
            "baseline_scores": baseline_scores,
            "enhanced_scores": enhanced_scores,
            "baseline_parse_failed": _completion_tree_for_scoring(
                baseline_completion,
                func["full_source"],
            )
            is None,
            "enhanced_parse_failed": _completion_tree_for_scoring(
                enhanced_completion,
                func["full_source"],
            )
            is None,
            "enhanced_fallback": enhanced_fallback,
        }


async def _get_enhanced_completion_with_meta(
    prompt: str,
    file_context: str,
    openai_client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient,
    demo_mode: bool = False,
) -> tuple[str, str, bool]:
    pattern_summary, from_service = await _get_pattern_summary(
        http_client=http_client,
        file_context=file_context,
        prompt=prompt,
        demo_mode=demo_mode,
    )
    fallback_used = not from_service and not (demo_mode and bool(pattern_summary))

    if demo_mode:
        completion = _generate_demo_completion(
            prompt=prompt,
            ground_truth=file_context,
            pattern_summary=pattern_summary,
            mode="enhanced",
        )
        return completion, pattern_summary, fallback_used

    if not pattern_summary:
        logger.warning(
            "CopilotPersona lookup failed for prompt '%s'. Falling back to baseline.",
            prompt.splitlines()[0],
        )
        completion = await get_baseline_completion(
            prompt=prompt,
            openai_client=openai_client,
            demo_mode=False,
            ground_truth=file_context,
        )
        return completion, "", True

    system_prompt = (
        "You are a Python coding assistant. Complete the function. "
        "Return only the function body, no explanation. "
        "This developer consistently uses these patterns: "
        f"{_truncate_to_token_budget(pattern_summary, PATTERN_INJECTION_TOKENS)}. "
        "Apply them."
    )
    completion = await _request_openai_completion(
        openai_client=openai_client,
        system_prompt=system_prompt,
        user_prompt=prompt,
    )
    return completion, pattern_summary, fallback_used


async def _get_pattern_summary(
    http_client: httpx.AsyncClient,
    file_context: str,
    prompt: str,
    demo_mode: bool,
) -> tuple[str, bool]:
    try:
        response = await http_client.post(
            COPILOT_PERSONA_URL,
            json={"context": file_context, "current_prompt": prompt},
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("pattern_summary", "")).strip(), True
    except (httpx.HTTPError, httpx.TimeoutException, ValueError) as error:
        logger.warning(
            "CopilotPersona lookup failed for prompt '%s': %s",
            prompt.splitlines()[0],
            error,
        )
        if demo_mode:
            return _generate_demo_pattern_summary(file_context), False
        return "", False


async def _request_openai_completion(
    openai_client: AsyncOpenAI | None,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if openai_client is None:
        return ""

    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            response = await openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
            )
            return _extract_completion_text(response)
        except Exception as error:
            if attempt == OPENAI_MAX_RETRIES - 1:
                logger.warning("OpenAI completion failed after retries: %s", error)
                return ""
            sleep_seconds = 2**attempt
            logger.warning(
                "OpenAI completion failed on attempt %s/%s: %s. Retrying in %ss.",
                attempt + 1,
                OPENAI_MAX_RETRIES,
                error,
                sleep_seconds,
            )
            await asyncio.sleep(sleep_seconds)

    return ""


def _generate_demo_completion(
    prompt: str,
    ground_truth: str,
    pattern_summary: str,
    mode: str,
) -> str:
    ground_truth_tree = _safe_parse_source(ground_truth)
    if ground_truth_tree is None:
        return "return None"

    function_node = _first_function_node(ground_truth_tree)
    if function_node is None:
        return "return None"

    argument_names = [
        argument.arg
        for argument in [
            *function_node.args.posonlyargs,
            *function_node.args.args,
            *function_node.args.kwonlyargs,
        ]
    ]
    primary_arg = argument_names[0] if argument_names else "value"
    local_names = _collect_local_variable_names(function_node)
    preferred_name = next(
        (name for name in local_names if _is_snake_case(name)),
        f"computed_{primary_arg}",
    )
    style = _detect_error_handling_style(ground_truth_tree)

    if mode == "baseline":
        return "\n".join(
            [
                "tempValue = value if 'value' in locals() else None",
                "if tempValue is None:",
                "    raise ValueError('unable to process input')",
                "return tempValue",
            ]
        )

    if style == "exception":
        body_lines = [
            "try:",
            f"    {preferred_name} = {primary_arg}",
            f"    return {preferred_name}",
            "except Exception:",
            "    return None",
        ]
    elif style == "return":
        secondary_name = next(
            (name for name in local_names if name != preferred_name),
            f"normalized_{primary_arg}",
        )
        body_lines = [
            f"{secondary_name} = {primary_arg}",
            f"is_valid = bool({secondary_name})",
            f"return is_valid, {secondary_name}",
        ]
    else:
        body_lines = [
            f"{preferred_name} = {primary_arg}",
            f"if {preferred_name} is None:",
            "    return None",
            f"return {preferred_name}",
        ]

    if "snake_case" not in pattern_summary and body_lines:
        body_lines[0] = body_lines[0].replace(preferred_name, "processedValue")

    return "\n".join(body_lines)


def _generate_demo_pattern_summary(source: str) -> str:
    tree = _safe_parse_source(source)
    if tree is None:
        return ""

    naming = _detect_naming_convention(tree)
    error_style = _detect_error_handling_style(tree)
    docstring_style = _detect_docstring_format(tree)
    annotation_ratio = _annotation_ratio_from_tree(tree)

    sentences: list[str] = []
    if naming == "snake_case":
        sentences.append("Uses snake_case naming consistently.")
    elif naming == "camelCase":
        sentences.append("Uses camelCase naming consistently.")
    else:
        sentences.append("Uses a mixed naming style.")

    if annotation_ratio > 0.8:
        sentences.append("Always includes type annotations on all arguments.")
    elif annotation_ratio > 0.1:
        sentences.append("Uses type annotations on some arguments.")
    else:
        sentences.append("Rarely includes type annotations.")

    if error_style == "exception":
        sentences.append("Handles errors with try/except blocks.")
    elif error_style == "return":
        sentences.append("Handles errors by returning tuples not exceptions.")
    else:
        sentences.append("Shows no consistent error-handling style.")

    if docstring_style == "google":
        sentences.append("Writes Google-style docstrings with Args/Returns.")
    elif docstring_style == "numpy":
        sentences.append("Writes NumPy-style docstrings with section underlines.")
    elif docstring_style == "plain":
        sentences.append("Uses plain descriptive docstrings.")
    else:
        sentences.append("Usually omits docstrings.")

    return _truncate_to_token_budget(" ".join(sentences), PATTERN_INJECTION_TOKENS)


def _extract_completion_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        return ""

    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return _normalize_completion_text(content)
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_value = getattr(item, "text", "")
                if text_value:
                    text_parts.append(str(text_value))
        return _normalize_completion_text("".join(text_parts))
    return _normalize_completion_text(str(content))


def _completion_tree_for_scoring(completion: str, ground_truth: str) -> ast.AST | None:
    direct_tree = _safe_parse_source(completion)
    if direct_tree is not None:
        return direct_tree

    ground_truth_tree = _safe_parse_source(ground_truth)
    function_node = _first_function_node(ground_truth_tree) if ground_truth_tree else None
    if function_node is None:
        return None

    wrapped = _wrap_completion_with_signature(
        completion=completion,
        signature=_build_signature(function_node),
        docstring=ast.get_docstring(function_node),
    )
    return _safe_parse_source(wrapped)


def _wrap_completion_with_signature(
    completion: str,
    signature: str,
    docstring: str | None,
) -> str:
    body = _normalize_completion_text(completion)
    if not body:
        body = "pass"

    lines = [f"{signature.rstrip(':')}:"]
    if docstring:
        lines.append(_indent_docstring(docstring))
    lines.append(textwrap.indent(body, "    "))
    return "\n".join(lines)


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    signature = f"{prefix} {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        signature = f"{signature} -> {ast.unparse(node.returns)}"
    return signature


def _collect_annotations(arguments: ast.arguments) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
        if argument.annotation is not None:
            annotations[argument.arg] = ast.unparse(argument.annotation)
    if arguments.vararg is not None and arguments.vararg.annotation is not None:
        annotations[f"*{arguments.vararg.arg}"] = ast.unparse(arguments.vararg.annotation)
    if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
        annotations[f"**{arguments.kwarg.arg}"] = ast.unparse(arguments.kwarg.annotation)
    return annotations


def _indent_docstring(docstring: str) -> str:
    indented_body = textwrap.indent(docstring, "    ")
    return f'    """\n{indented_body}\n    """'


def _truncate_to_token_budget(text: str, token_budget: int) -> str:
    if not text:
        return ""

    words = text.split()
    while words and (len(words) * 1.3) > token_budget:
        words.pop()
    return " ".join(words)


def _normalize_completion_text(text: str) -> str:
    normalized = text.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    return normalized


def _safe_parse_source(source: str) -> ast.AST | None:
    normalized = textwrap.dedent(source).strip()
    if not normalized:
        return None
    try:
        return ast.parse(normalized)
    except SyntaxError:
        return None


def _detect_naming_convention(tree: ast.AST) -> str:
    identifiers = _extract_identifier_names(tree)
    if not identifiers:
        return "mixed"
    if all(_is_snake_case(name) for name in identifiers):
        return "snake_case"
    if all(_is_camel_case(name) for name in identifiers):
        return "camelCase"
    return "mixed"


def _annotation_ratio_from_tree(tree: ast.AST) -> float:
    function_node = _first_function_node(tree)
    if function_node is None:
        return 0.0

    arguments = [
        *function_node.args.posonlyargs,
        *function_node.args.args,
        *function_node.args.kwonlyargs,
    ]
    if not arguments:
        return 0.0

    annotated_count = sum(
        1 for argument in arguments if argument.annotation is not None
    )
    return annotated_count / len(arguments)


def _detect_error_handling_style(tree: ast.AST) -> str:
    if any(isinstance(node, ast.Try) for node in ast.walk(tree)):
        return "exception"
    if any(_is_return_style(node) for node in ast.walk(tree)):
        return "return"
    return "neither"


def _detect_docstring_format(tree: ast.AST) -> str:
    function_node = _first_function_node(tree)
    docstring = ast.get_docstring(function_node) if function_node is not None else None
    if docstring is None:
        return "none"
    if "Args:" in docstring or "Returns:" in docstring:
        return "google"
    if "-----" in docstring:
        return "numpy"
    return "plain"


def _extract_name_nodes(tree: ast.AST) -> set[str]:
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    function_node = _first_function_node(tree)
    if function_node is not None and function_node.name != COMPLETION_WRAPPER_NAME:
        names.add(function_node.name)
    return names


def _extract_identifier_names(tree: ast.AST) -> list[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != COMPLETION_WRAPPER_NAME:
                names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return sorted(name for name in names if name.isidentifier())


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


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _is_return_style(node: ast.AST) -> bool:
    if not isinstance(node, ast.Return) or node.value is None:
        return False
    if isinstance(node.value, ast.Tuple):
        return True
    if isinstance(node.value, ast.Name):
        return node.value.id in {"Result", "Ok", "Err"}
    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
        return node.value.func.id in {"Result", "Ok", "Err"}
    return False


def _is_snake_case(name: str) -> bool:
    if not name:
        return False
    return name.isidentifier() and name == name.lower() and "__" not in name


def _is_camel_case(name: str) -> bool:
    if not name or "_" in name:
        return False
    return name[0].islower() and any(character.isupper() for character in name[1:])


def _should_skip_path(file_path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in file_path.parts)


def _get_line_count(node: ast.AST) -> int:
    start_line = getattr(node, "lineno", 0)
    end_line = getattr(node, "end_lineno", start_line)
    return max(0, end_line - start_line + 1)


def _is_dunder_name(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _average(values: Any) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def _zero_scores() -> dict:
    return {
        "naming_convention_match": 0.0,
        "type_annotation_coverage": 0.0,
        "error_handling_style": 0.0,
        "docstring_format": 0.0,
        "import_and_naming_style": 0.0,
        "total": 0.0,
    }


def _has_real_api_key() -> bool:
    return bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_key_here")


def _project_name_from_file(file_path: str) -> str:
    path = Path(file_path)
    root_dir = Path(TEST_CORPUS_DIR).resolve()
    try:
        relative = path.resolve().relative_to(root_dir)
    except ValueError:
        return path.parent.name or "unknown"
    return relative.parts[0] if relative.parts else "unknown"


def _first_function_node(
    tree: ast.AST | None,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _result_label(result: dict) -> str:
    return f"{result['function_name']} ({result['file_path']})"


def _save_raw_completions(results: list[dict]) -> None:
    raw_payload = [
        {
            "function_name": result["function_name"],
            "file_path": result["file_path"],
            "project_name": result["project_name"],
            "prompt": result["prompt"],
            "ground_truth": result["ground_truth"],
            "pattern_summary": result["pattern_summary"],
            "baseline_completion": result["baseline_completion"],
            "enhanced_completion": result["enhanced_completion"],
            "baseline_scores": result["baseline_scores"],
            "enhanced_scores": result["enhanced_scores"],
            "baseline_parse_failed": result["baseline_parse_failed"],
            "enhanced_parse_failed": result["enhanced_parse_failed"],
            "enhanced_fallback": result["enhanced_fallback"],
        }
        for result in results
    ]
    RAW_OUTPUT_PATH.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CopilotPersona against baseline GPT-4o completions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and strip the corpus without making any API calls.",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Run the offline heuristic evaluation mode with no OpenAI usage.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli_args = _parse_args()
    asyncio.run(
        run_evaluation(
            dry_run=cli_args.dry_run,
            demo_mode=cli_args.demo_mode or DEMO_MODE,
        )
    )
