"""Normal operational and data-quality error injection for banking datasets."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

# Typical retail-bank payment failure mix (South Africa debit-order / EFT rails).
OPERATIONAL_FAILURE_REASONS: list[tuple[str, float]] = [
    ("insufficient_funds", 0.38),
    ("bank_timeout", 0.24),
    ("mandate_limit", 0.14),
    ("account_blocked", 0.12),
    ("daily_limit_exceeded", 0.06),
    ("invalid_account_details", 0.04),
    ("duplicate_debit", 0.02),
]

CANCELLATION_REASONS = [
    "customer_stop_instruction",
    "mandate_cancelled",
    "account_closed",
]

# Subtle ETL / source-system data defects (not fraud).
DATA_ERROR_RATES: dict[str, float] = {
    "truncated_description": 0.012,
    "amount_rounding_error": 0.004,
    "timestamp_shift": 0.006,
    "missing_failure_reason": 0.003,
    "status_inconsistency": 0.002,
    "duplicate_transaction_id": 0.001,
    "whitespace_in_id": 0.002,
}

DEFAULT_STATUS_PROBS = {
    "completed": 0.965,
    "failed": 0.028,
    "cancelled": 0.007,
}


def pick_failure_reason() -> str:
    reasons, weights = zip(*OPERATIONAL_FAILURE_REASONS)
    return str(np.random.choice(reasons, p=weights))


def pick_cancellation_reason() -> str:
    return random.choice(CANCELLATION_REASONS)


def status_probs_for_due_day(due_day: int, is_recovery: bool = False) -> tuple[float, float, float]:
    """Return (p_completed, p_failed, p_cancelled) with mild calendar effects."""
    if is_recovery:
        return 0.75, 0.22, 0.03

    p_completed, p_failed, p_cancelled = (
        DEFAULT_STATUS_PROBS["completed"],
        DEFAULT_STATUS_PROBS["failed"],
        DEFAULT_STATUS_PROBS["cancelled"],
    )

    # End-of-month cashflow stress.
    if due_day >= 25:
        p_completed -= 0.018
        p_failed += 0.016
        p_cancelled += 0.002
    elif due_day <= 3:
        p_completed += 0.008
        p_failed -= 0.007
        p_cancelled -= 0.001

    p_completed = max(0.0, min(0.999, p_completed))
    p_failed = max(0.0, min(0.999, p_failed))
    p_cancelled = max(0.0, min(0.999, p_cancelled))
    total = p_completed + p_failed + p_cancelled
    return p_completed / total, p_failed / total, p_cancelled / total


def payment_status(due_day: int | None = None, is_recovery: bool = False) -> tuple[str, str | None]:
    """Draw Completed / Failed / Cancelled with an optional failure or cancel reason."""
    if due_day is None:
        p_completed, p_failed, p_cancelled = status_probs_for_due_day(15, is_recovery)
    else:
        p_completed, p_failed, p_cancelled = status_probs_for_due_day(due_day, is_recovery)

    draw = np.random.choice(["Completed", "Failed", "Cancelled"], p=[p_completed, p_failed, p_cancelled])
    if draw == "Failed":
        return draw, pick_failure_reason()
    if draw == "Cancelled":
        return draw, pick_cancellation_reason()
    return draw, None


def _truncate(text: str, max_len: int = 28) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _shift_time(tx_time: str) -> str:
    parts = str(tx_time).split(":")
    if len(parts) != 3:
        return tx_time
    hour = (int(parts[0]) + random.choice([-3, -2, 2, 3, 5])) % 24
    return f"{hour:02d}:{parts[1]}:{parts[2]}"


def inject_data_errors(row: dict[str, Any]) -> list[str]:
    """Apply zero or more data-quality defects to a transaction row in place."""
    errors: list[str] = []

    if random.random() < DATA_ERROR_RATES["truncated_description"] and row.get("description"):
        row["description"] = _truncate(str(row["description"]))
        errors.append("truncated_description")

    if random.random() < DATA_ERROR_RATES["amount_rounding_error"] and row.get("amount") is not None:
        amount = float(row["amount"])
        row["amount"] = round(amount + random.choice([-0.01, 0.01]), 2)
        errors.append("amount_rounding_error")

    if random.random() < DATA_ERROR_RATES["timestamp_shift"] and row.get("transaction_time"):
        row["transaction_time"] = _shift_time(str(row["transaction_time"]))
        errors.append("timestamp_shift")

    if random.random() < DATA_ERROR_RATES["missing_failure_reason"] and row.get("status") == "Failed":
        row["failure_reason"] = None
        errors.append("missing_failure_reason")

    if random.random() < DATA_ERROR_RATES["status_inconsistency"]:
        if row.get("status") == "Completed":
            row["status"] = "Failed"
            row["failure_reason"] = pick_failure_reason()
        elif row.get("status") == "Failed":
            row["status"] = "Completed"
            row["failure_reason"] = None
        errors.append("status_inconsistency")

    if random.random() < DATA_ERROR_RATES["whitespace_in_id"]:
        for col in ("transaction_id", "account_id", "customer_id", "loan_id"):
            if row.get(col):
                row[col] = f" {row[col]} "
                errors.append("whitespace_in_id")
                break

    return errors


def induce_operational_failure(row: dict[str, Any]) -> bool:
    """Convert a completed payment to failed (~1.5% when called with outer rate)."""
    if row.get("status") != "Completed":
        return False
    row["status"] = "Failed"
    row["failure_reason"] = pick_failure_reason()
    return True
