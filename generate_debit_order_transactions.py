from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, EasterMonday, GoodFriday, Holiday, nearest_workday
from pandas.tseries.offsets import CustomBusinessDay
from tqdm import tqdm


class SouthAfricanBusinessCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        Holiday("Human Rights Day", month=3, day=21, observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday("Freedom Day", month=4, day=27, observance=nearest_workday),
        Holiday("Workers Day", month=5, day=1, observance=nearest_workday),
        Holiday("Youth Day", month=6, day=16, observance=nearest_workday),
        Holiday("Womens Day", month=8, day=9, observance=nearest_workday),
        Holiday("Heritage Day", month=9, day=24, observance=nearest_workday),
        Holiday("Reconciliation Day", month=12, day=16, observance=nearest_workday),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("Goodwill Day", month=12, day=26, observance=nearest_workday),
    ]


@dataclass
class TxnContext:
    year: int
    month: int
    month_start: pd.Timestamp
    month_end: pd.Timestamp
    business_day: CustomBusinessDay
    calendar: AbstractHolidayCalendar


def get_path_without_ext(base_path: str, year: int, month: int | None, record_type: str) -> str:
    if month is None:
        return os.path.join(base_path, f"{record_type}_{year}")
    month_dir = os.path.join(base_path, str(year), f"{month:02d}")
    return os.path.join(month_dir, f"{record_type}_{year}_{month:02d}")


def load_parquet_or_csv(path_without_ext: str) -> pd.DataFrame:
    parquet_path = f"{path_without_ext}.parquet"
    csv_path = f"{path_without_ext}.csv"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(parquet_path)


def load_debit_orders_for_period(base_path: str, target_year: int, target_month: int, min_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(min_year, target_year + 1):
        months = range(1, 13)
        if year == target_year:
            months = range(1, target_month + 1)

        for month in months:
            monthly_base = get_path_without_ext(base_path, year, month, "debit_orders")
            flat_base = get_path_without_ext(base_path, year, None, "debit_orders")

            loaded = False
            for candidate in [monthly_base, flat_base]:
                try:
                    frame = load_parquet_or_csv(candidate)
                    frame["_source_year"] = year
                    frame["_source_month"] = month
                    frames.append(frame)
                    loaded = True
                    break
                except FileNotFoundError:
                    continue

            if year == target_year and month == target_month and not loaded:
                print(f"Debit order source not found for {target_year}-{target_month:02d}.")

    if not frames:
        return pd.DataFrame()

    debit_orders = pd.concat(frames, ignore_index=True)
    if "debit_order_id" in debit_orders.columns:
        debit_orders = debit_orders.sort_values(["_source_year", "_source_month"]).drop_duplicates(
            subset=["debit_order_id"], keep="last"
        )
    return debit_orders.reset_index(drop=True)


def normalize_status(value: object) -> str:
    return str(value).strip().lower() if pd.notna(value) else ""


def shift_to_business_day(d: pd.Timestamp, bday: CustomBusinessDay, calendar: AbstractHolidayCalendar) -> pd.Timestamp:
    if d.weekday() >= 5:
        return d + bday
    holidays = calendar.holidays(start=d - timedelta(days=1), end=d + timedelta(days=1))
    if d in holidays:
        return d + bday
    return d


def amount_for_event(row: pd.Series) -> float:
    base_amount = float(row.get("amount", 0.0) or 0.0)
    variable_hint = bool(row.get("is_fixed_amount", True) is False)
    do_type = str(row.get("debit_order_type", "")).strip().lower()

    if variable_hint or "utility" in do_type:
        return round(base_amount * random.uniform(0.85, 1.15), 2)
    if "insurance" in do_type:
        return round(base_amount * random.uniform(0.98, 1.03), 2)
    if "loan" in do_type:
        return round(base_amount * random.uniform(0.99, 1.01), 2)
    return round(base_amount, 2)


def status_for_event(row: pd.Series, due_date: pd.Timestamp) -> tuple[str, str | None]:
    do_type = str(row.get("debit_order_type", "")).lower()
    # Default profile for generic consumer mandates.
    p_completed, p_failed, p_cancelled = 0.965, 0.03, 0.005

    # Business mix tuning by mandate type.
    if "insurance" in do_type:
        # Insurance mandates are usually very stable.
        p_completed, p_failed, p_cancelled = 0.989, 0.009, 0.002
    elif "loan" in do_type:
        p_completed, p_failed, p_cancelled = 0.982, 0.014, 0.004
    elif "utility" in do_type:
        p_completed, p_failed, p_cancelled = 0.94, 0.052, 0.008
    elif "salary" in do_type or "payroll" in do_type:
        p_completed, p_failed, p_cancelled = 0.994, 0.005, 0.001

    # End-of-month cashflow stress impacts utilities more than fixed commitments.
    if due_date.day >= 25 and "utility" in do_type:
        p_completed, p_failed = p_completed - 0.045, p_failed + 0.045
    elif due_date.day >= 25:
        p_completed, p_failed = p_completed - 0.015, p_failed + 0.015

    # Early-month rebound improves completion slightly after salary inflows.
    if due_date.day <= 3 and "utility" in do_type:
        p_completed, p_failed = p_completed + 0.012, p_failed - 0.012

    # Mondays show a mild operational/funds friction.
    if due_date.weekday() == 0:
        p_completed, p_failed = p_completed - 0.01, p_failed + 0.01

    # Keep probabilities valid and normalized.
    p_completed = max(0.001, min(0.999, p_completed))
    p_failed = max(0.0, min(0.999, p_failed))
    p_cancelled = max(0.0, min(0.999, p_cancelled))
    total = p_completed + p_failed + p_cancelled
    p_completed, p_failed, p_cancelled = p_completed / total, p_failed / total, p_cancelled / total

    draw = np.random.choice(["Completed", "Failed", "Cancelled"], p=[p_completed, p_failed, p_cancelled])
    if draw == "Failed":
        reason = np.random.choice([
            "insufficient_funds",
            "account_restricted",
            "banking_system_timeout",
            "mandate_validation_failed",
        ])
        return "Failed", str(reason)
    if draw == "Cancelled":
        return "Cancelled", "customer_stop_instruction"
    return "Completed", None


def pick_time_for_event(row: pd.Series) -> str:
    do_type = str(row.get("debit_order_type", "")).lower()
    if "salary" in do_type or "payroll" in do_type:
        hour = random.choice([5, 6])
    elif "insurance" in do_type:
        hour = random.choice([2, 3, 4, 5, 6])
    elif "loan" in do_type:
        hour = random.choice([6, 7, 8])
    elif "utility" in do_type:
        hour = random.choice([7, 8, 9, 12, 16, 19])
    else:
        hour = random.randint(6, 17)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def apply_payday_spike(transaction_time: str, due_date: pd.Timestamp) -> str:
    hour, minute, second = [int(x) for x in transaction_time.split(":")]
    # Payday processing surge around 24th-1st at early hours.
    if due_date.day in (24, 25, 26, 27, 28, 1) and random.random() < 0.45:
        hour = random.choice([0, 1, 2, 3, 4, 5, 6])
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def due_dates_for_month(row: pd.Series, ctx: TxnContext) -> list[pd.Timestamp]:
    start_date = pd.to_datetime(row.get("start_date"), errors="coerce")
    if pd.isna(start_date):
        return []

    end_date = pd.to_datetime(row.get("end_date"), errors="coerce")
    cancellation_date = pd.to_datetime(row.get("cancellation_date"), errors="coerce")
    suspension_date = pd.to_datetime(row.get("suspension_date"), errors="coerce")

    active_end = pd.Timestamp.max.normalize()
    for maybe_end in [end_date, cancellation_date, suspension_date]:
        if pd.notna(maybe_end):
            active_end = min(active_end, maybe_end.normalize() - timedelta(days=1))

    period_start = max(ctx.month_start, start_date.normalize())
    period_end = min(ctx.month_end, active_end)
    if period_start > period_end:
        return []

    freq = str(row.get("frequency", "Monthly")).strip().lower()
    collection_day = int(row.get("collection_day", start_date.day) or start_date.day)
    dates: list[pd.Timestamp] = []

    if freq == "weekly":
        first = period_start
        while first.weekday() != start_date.weekday():
            first += timedelta(days=1)
        d = first
        while d <= period_end:
            dates.append(shift_to_business_day(d, ctx.business_day, ctx.calendar))
            d += timedelta(days=7)
        return sorted(set(dates))

    month_day = min(collection_day, int(ctx.month_end.day))
    candidate = pd.Timestamp(year=ctx.year, month=ctx.month, day=month_day)

    if freq == "quarterly":
        months_delta = (ctx.year - start_date.year) * 12 + (ctx.month - start_date.month)
        if months_delta % 3 != 0:
            return []
    elif freq == "annually":
        if ctx.month != start_date.month:
            return []

    if period_start <= candidate <= period_end:
        dates.append(shift_to_business_day(candidate, ctx.business_day, ctx.calendar))

    return sorted(set(dates))


def generate_month(target_year: int, target_month: int, min_year: int, base_path: str = "banking_data") -> pd.DataFrame:
    seed = target_year * 100 + target_month
    random.seed(seed)
    np.random.seed(seed)

    debit_orders = load_debit_orders_for_period(base_path, target_year, target_month, min_year)
    if debit_orders.empty:
        print("No debit orders found for generation window.")
        return pd.DataFrame()

    if "status" in debit_orders.columns:
        debit_orders = debit_orders[debit_orders["status"].map(normalize_status).isin(["active", "suspended", "pending"])].copy()

    month_start = pd.Timestamp(year=target_year, month=target_month, day=1)
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    sa_calendar = SouthAfricanBusinessCalendar()
    ctx = TxnContext(
        year=target_year,
        month=target_month,
        month_start=month_start,
        month_end=month_end,
        business_day=CustomBusinessDay(calendar=sa_calendar),
        calendar=sa_calendar,
    )

    rows: list[dict] = []
    counter = 1

    for _, row in tqdm(debit_orders.iterrows(), total=len(debit_orders), desc=f"Debit orders {target_year}-{target_month:02d}"):
        for due_date in due_dates_for_month(row, ctx):
            status, failure_reason = status_for_event(row, due_date)
            amount = amount_for_event(row) if status != "Cancelled" else 0.0

            description = str(row.get("description", "Debit Order")).strip() or "Debit Order"
            if status == "Failed" and failure_reason:
                description = f"{description} - Failed: {failure_reason}"

            rows.append(
                {
                    "transaction_id": f"TXN{target_year}{target_month:02d}{counter:08d}",
                    "debit_order_id": row.get("debit_order_id"),
                    "customer_id": row.get("customer_id"),
                    "account_id": row.get("account_id"),
                    "transaction_date": due_date.strftime("%Y-%m-%d"),
                    "transaction_time": apply_payday_spike(pick_time_for_event(row), due_date),
                    "amount": amount,
                    "debit_credit": "Debit",
                    "status": status,
                    "failure_reason": failure_reason,
                    "description": description,
                    "channel": "Automated",
                    "currency": row.get("currency", "ZAR"),
                    "receiving_account": row.get("account_to") or row.get("beneficiary_account"),
                    "beneficiary_name": row.get("beneficiary_name"),
                    "debit_order_type": row.get("debit_order_type"),
                    "frequency": row.get("frequency"),
                    "is_immediate_payment": False,
                    "transaction_cost": 0.0,
                    "record_last_updated_at": pd.Timestamp.now(),
                }
            )
            counter += 1

    transactions = pd.DataFrame(rows)
    output_base = get_path_without_ext(base_path, target_year, target_month, "debit_order_transactions")
    output_dir = os.path.dirname(output_base)
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_base}.parquet"
    try:
        transactions.to_parquet(output_file, index=False)
    except Exception:
        output_file = f"{output_base}.csv"
        transactions.to_csv(output_file, index=False)

    print(f"Generated {len(transactions)} debit order transactions for {target_year}-{target_month:02d}")
    print(f"Saved to {output_file}")
    return transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic monthly debit order transactions")
    parser.add_argument("--year", type=int, required=True, help="Target year")
    parser.add_argument("--month", type=int, help="Target month (1-12). Omit to generate all months.")
    parser.add_argument("--min-year", type=int, help="Oldest debit-order year to include for active mandates")
    args = parser.parse_args()

    if args.month is not None and (args.month < 1 or args.month > 12):
        raise ValueError("month must be in 1..12")

    min_year = args.min_year if args.min_year is not None else args.year

    if args.month is None:
        for month in range(1, 13):
            generate_month(args.year, month, min_year=min_year)
    else:
        generate_month(args.year, args.month, min_year=min_year)


if __name__ == "__main__":
    main()
