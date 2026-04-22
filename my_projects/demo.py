
def calculate_total(items: list[float], tax_rate: float = 0.1) -> float:
    """Calculate total with tax.

    Args:
        items: list of item prices
        tax_rate: tax rate as decimal

    Returns:
        total price including tax
    """
    subtotal = sum(items)
    return subtotal * (1 + tax_rate)


def process_data(data, config):
    try:
        result = data.transform(config)
        return result
    except Exception:
        return None
