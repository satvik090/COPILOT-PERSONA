def normalize_user_record(record: dict[str, object]) -> dict[str, object]:
    """Normalize a user record for downstream jobs.

    Args:
        record: Raw user record from ingestion.

    Returns:
        Normalized dictionary with cleaned fields.
    """
    normalized_name = str(record.get("name", "")).strip().title()
    normalized_email = str(record.get("email", "")).strip().lower()
    normalized_city = str(record.get("city", "")).strip().title()
    return {
        "name": normalized_name,
        "email": normalized_email,
        "city": normalized_city,
        "active": bool(record.get("active", False)),
    }


def filter_active_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Filter down to active records only.

    Args:
        records: Incoming records for a pipeline batch.

    Returns:
        List containing active records.
    """
    active_records: list[dict[str, object]] = []
    for record in records:
        if bool(record.get("active", False)):
            active_records.append(record)
    return active_records


def summarize_batch_metrics(records: list[dict[str, object]]) -> dict[str, int]:
    """Summarize pipeline batch metrics.

    Args:
        records: Records processed in a batch.

    Returns:
        Summary counts for monitoring dashboards.
    """
    active_count = 0
    inactive_count = 0
    for record in records:
        if bool(record.get("active", False)):
            active_count += 1
        else:
            inactive_count += 1
    return {
        "total_count": len(records),
        "active_count": active_count,
        "inactive_count": inactive_count,
    }


def safe_transform_payload(payload: dict[str, object]) -> dict[str, object] | None:
    """Safely transform a payload into normalized output.

    Args:
        payload: Raw payload received from ingestion.

    Returns:
        Normalized payload or None on malformed input.
    """
    try:
        normalized_payload = normalize_user_record(payload)
        normalized_payload["source"] = str(payload.get("source", "unknown")).strip()
        return normalized_payload
    except Exception:
        return None
