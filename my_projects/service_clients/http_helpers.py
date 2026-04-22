def build_auth_headers(api_key: str, request_id: str) -> dict[str, str]:
    """Build standard authentication headers.

    Args:
        api_key: Secret API key for the upstream service.
        request_id: Request identifier for tracing.

    Returns:
        Dictionary of reusable HTTP headers.
    """
    sanitized_key = api_key.strip()
    return {
        "Authorization": f"Bearer {sanitized_key}",
        "X-Request-ID": request_id,
        "Content-Type": "application/json",
    }


def extract_retry_window(
    status_code: int,
    retry_after_header: str | None,
) -> tuple[bool, int]:
    """Extract retry information for a failed request.

    Args:
        status_code: HTTP response status code.
        retry_after_header: Retry-After header value if present.

    Returns:
        Pair containing whether the request is retryable and retry window seconds.
    """
    retryable_statuses = {408, 429, 500, 502, 503, 504}
    is_retryable = status_code in retryable_statuses
    retry_window_seconds = int(retry_after_header) if retry_after_header else 0
    return is_retryable, retry_window_seconds


def safe_parse_json_response(response_text: str) -> dict[str, object] | None:
    """Safely parse a JSON response payload.

    Args:
        response_text: Raw response body.

    Returns:
        Parsed JSON-like dictionary or None on failure.
    """
    try:
        cleaned_text = response_text.strip()
        if not cleaned_text:
            return None
        import json

        parsed_payload = json.loads(cleaned_text)
        if isinstance(parsed_payload, dict):
            return parsed_payload
        return None
    except Exception:
        return None


def build_query_params(filters: dict[str, object]) -> dict[str, str]:
    """Build string query parameters from mixed filter values.

    Args:
        filters: Raw filter values from a request.

    Returns:
        Stringified query parameter mapping.
    """
    query_params: dict[str, str] = {}
    for key, value in filters.items():
        if value is None:
            continue
        query_params[key] = str(value).strip()
    return query_params
