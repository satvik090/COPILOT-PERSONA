def calculate_invoice_total(
    line_items: list[float],
    tax_rate: float = 0.18,
    discount_rate: float = 0.0,
) -> float:
    """Calculate the final invoice amount.

    Args:
        line_items: Item prices before tax.
        tax_rate: Tax multiplier expressed as a decimal.
        discount_rate: Discount multiplier expressed as a decimal.

    Returns:
        Final invoice amount after discounts and tax.
    """
    subtotal = sum(line_items)
    discounted_subtotal = subtotal * (1 - discount_rate)
    final_total = discounted_subtotal * (1 + tax_rate)
    return round(final_total, 2)


def build_invoice_summary(
    invoice_id: str,
    customer_name: str,
    line_items: list[float],
) -> dict[str, object]:
    """Build a summary payload for an invoice.

    Args:
        invoice_id: Unique invoice identifier.
        customer_name: Customer display name.
        line_items: List of invoice values.

    Returns:
        Summary dictionary used in billing responses.
    """
    total_amount = calculate_invoice_total(line_items)
    item_count = len(line_items)
    average_amount = total_amount / item_count if item_count else 0.0
    return {
        "invoice_id": invoice_id,
        "customer_name": customer_name,
        "item_count": item_count,
        "total_amount": total_amount,
        "average_amount": round(average_amount, 2),
    }


def validate_payment_amount(
    expected_amount: float,
    received_amount: float,
    tolerance: float = 0.01,
) -> tuple[bool, float]:
    """Validate whether a payment amount is acceptable.

    Args:
        expected_amount: The invoice total that should be paid.
        received_amount: The amount actually received.
        tolerance: Allowed rounding difference.

    Returns:
        Pair containing validation status and delta.
    """
    payment_delta = round(received_amount - expected_amount, 2)
    is_valid = abs(payment_delta) <= tolerance
    return is_valid, payment_delta


def parse_currency_amount(raw_amount: str) -> float | None:
    """Parse a currency amount from a billing form.

    Args:
        raw_amount: Currency text entered by a user.

    Returns:
        Parsed float value or None if the text is invalid.
    """
    try:
        normalized_amount = raw_amount.replace("$", "").replace(",", "").strip()
        if not normalized_amount:
            return None
        return float(normalized_amount)
    except ValueError:
        return None
