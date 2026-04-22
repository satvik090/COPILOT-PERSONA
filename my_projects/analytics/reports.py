def build_weekly_summary(
    weekly_sales: list[float],
    returning_customers: int,
) -> dict[str, float]:
    """Build a weekly analytics summary.

    Args:
        weekly_sales: Sales amounts for the week.
        returning_customers: Count of returning customers.

    Returns:
        Summary dictionary with totals and averages.
    """
    total_sales = sum(weekly_sales)
    average_sales = total_sales / len(weekly_sales) if weekly_sales else 0.0
    return {
        "total_sales": round(total_sales, 2),
        "average_sales": round(average_sales, 2),
        "returning_customers": float(returning_customers),
    }


def classify_customer_value(
    lifetime_value: float,
    order_count: int,
) -> tuple[bool, str]:
    """Classify whether a customer is high value.

    Args:
        lifetime_value: Total spend by the customer.
        order_count: Number of completed orders.

    Returns:
        Pair containing high-value flag and label.
    """
    is_high_value = lifetime_value >= 1000 or order_count >= 10
    value_label = "high" if is_high_value else "standard"
    return is_high_value, value_label


def safe_percentage_change(
    current_value: float,
    previous_value: float,
) -> float | None:
    """Safely calculate percentage change.

    Args:
        current_value: Current measured value.
        previous_value: Previous measured value.

    Returns:
        Percentage change or None if the previous value is zero.
    """
    try:
        if previous_value == 0:
            return None
        delta = current_value - previous_value
        return round((delta / previous_value) * 100, 2)
    except Exception:
        return None


def rank_regions_by_sales(region_totals: dict[str, float]) -> list[tuple[str, float]]:
    """Rank sales regions by total revenue.

    Args:
        region_totals: Mapping of region names to revenue.

    Returns:
        Ordered list of region totals in descending order.
    """
    ranked_regions = sorted(
        region_totals.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked_regions
