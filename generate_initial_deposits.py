"""Generate separate initial_deposits.jsonl tables from account openings.

The files use the same schema as transactions.jsonl, but they are intentionally
kept separate so downstream work can join or union them explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"


def transaction_columns() -> list[str]:
    sample = BANKING_DIR / "2019" / "01" / "transactions.jsonl"
    return list(pd.read_json(sample, lines=True, nrows=1).columns)


def stable_rng(account_id: str) -> random.Random:
    seed = int(hashlib.sha256(f"{account_id}|initial-deposit-table-v1".encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def next_business_day(value: date) -> date:
    if value.weekday() == 5:
        return value + timedelta(days=2)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def deposit_amount(row: pd.Series) -> float:
    if "initial_deposit" in row.index and row.get("initial_deposit") is not None and not pd.isna(row.get("initial_deposit")):
        return round(max(0.0, float(row.get("initial_deposit"))), 2)
    if row.get("expected_amount") is not None and not pd.isna(row.get("expected_amount")):
        return round(max(0.0, float(row.get("expected_amount"))), 2)
    return 100.0


def build_record(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    account_id = str(row["account_id"])
    customer_id = str(row["customer_id"])
    opened = next_business_day(pd.to_datetime(row["opening_date"]).date())
    rng = stable_rng(account_id)
    ts = datetime.combine(opened, time(rng.randint(8, 15), rng.randint(0, 55), rng.randint(0, 59)))
    amount = deposit_amount(row)
    branch_code = row.get("branch_code")
    if branch_code is None or pd.isna(branch_code):
        branch_code = f"{rng.randint(100000, 999999)}"

    record = {col: None for col in columns}
    record.update(
        {
            "transaction_id": f"INIT{opened:%Y%m%d}-{account_id}",
            "batch_id": f"BATCH-{opened:%Y%m}",
            "generation_timestamp": "2026-06-01T12:00:00",
            "transaction_timestamp": ts.isoformat(),
            "transaction_date": opened.isoformat(),
            "transaction_time": ts.time().isoformat(timespec="seconds"),
            "customer_id": customer_id,
            "account_id": account_id,
            "channel": "branch",
            "channel_metadata": {
                "network_type": None,
                "ip_address": None,
                "terminal_id": f"BR-{branch_code}-OPENING",
                "atm_id": None,
                "branch_code": str(branch_code),
                "gps_coordinates": None,
                "session_duration_seconds": None,
            },
            "category": "initial_deposit",
            "amount": amount,
            "debit_credit": "credit",
            "status": "completed",
            "description": "Initial Deposit",
            "merchant_name": None,
            "is_fraudulent": False,
            "fraud_pattern": None,
            "fraud_confidence": 0,
            "fraud_metadata": {},
            "has_error": False,
            "error_types": [],
            "error_metadata": {},
            "network_latency_ms": 0,
            "authorization_time_ms": 0,
            "third_party_timeout": False,
            "stan": rng.randint(100000, 999999),
            "rrn": f"{opened:%Y%m}{rng.randint(100000000, 999999999)}",
            "source_table": "initial_deposits",
            "customer_session_id": f"OPEN-{customer_id}-{opened:%Y%m%d}",
            "customer_device_fingerprint": None,
            "customer_location_state_before": None,
            "customer_location_state_after": None,
            "customer_behavioral_score": 100.0,
            "realism_score": 100.0,
            "temporal_realism_score": 100.0,
            "spatial_realism_score": 100.0,
            "behavioral_realism_score": 100.0,
            "financial_realism_score": 100.0,
            "account_balance_before": 0.0,
            "account_balance_after": amount,
            "daily_transaction_count_so_far": 1,
            "daily_total_amount_so_far": amount,
            "monthly_transaction_count_so_far": 1,
            "external_context": {
                "day_of_week": ts.strftime("%A"),
                "is_public_holiday": False,
                "is_school_holiday": False,
                "is_payday_window": False,
                "weather_condition": None,
                "load_shedding_stage": None,
                "nearby_events": [],
                "is_weekend": False,
            },
        }
    )
    return {col: clean_value(record.get(col)) for col in columns}


def load_accounts(start_year: int, end_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(BANKING_DIR.glob("20*/??/accounts_*.parquet")):
        df = pd.read_parquet(path)
        if df.empty or not {"account_id", "customer_id", "opening_date"}.issubset(df.columns):
            continue
        opened = pd.to_datetime(df["opening_date"], errors="coerce")
        df = df.loc[opened.notna()].copy()
        df["opening_year"] = opened.loc[opened.notna()].dt.year
        df = df.loc[df["opening_year"].between(start_year, end_year)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    accounts = pd.concat(frames, ignore_index=True)
    accounts["opening_dt"] = pd.to_datetime(accounts["opening_date"], errors="coerce")
    accounts = accounts.sort_values(["account_id", "opening_dt"])
    return accounts.drop_duplicates("account_id", keep="first")


def remove_initial_rows_from_transactions(start_year: int, end_year: int) -> int:
    removed = 0
    for path in sorted(BANKING_DIR.glob("20*/??/transactions.jsonl")):
        year = int(path.parts[-3])
        if not start_year <= year <= end_year:
            continue
        tmp = path.with_name("transactions.clean.tmp")
        month_removed = 0
        with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8", newline="\n") as dst:
            for line in src:
                if '"category":"initial_deposit"' in line or '"description":"Initial Deposit"' in line:
                    month_removed += 1
                    continue
                dst.write(line)
        if month_removed:
            tmp.replace(path)
            removed += month_removed
        else:
            tmp.unlink(missing_ok=True)
    return removed


def write_initial_deposits(start_year: int, end_year: int) -> int:
    columns = transaction_columns()
    accounts = load_accounts(start_year, end_year)
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for _, row in accounts.iterrows():
        opened = next_business_day(pd.to_datetime(row["opening_date"]).date())
        out = BANKING_DIR / str(opened.year) / f"{opened.month:02d}" / "initial_deposits.jsonl"
        grouped.setdefault(out, []).append(build_record(row, columns))

    total = 0
    for path, records in sorted(grouped.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        records.sort(key=lambda r: (r["transaction_timestamp"], r["account_id"]))
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        total += len(records)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Create separate initial_deposits.jsonl tables.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    removed = remove_initial_rows_from_transactions(args.start_year, args.end_year)
    written = write_initial_deposits(args.start_year, args.end_year)
    print(f"Removed {removed:,} Initial Deposit rows from transactions.jsonl.")
    print(f"Wrote {written:,} Initial Deposit rows to initial_deposits.jsonl.")


if __name__ == "__main__":
    main()
