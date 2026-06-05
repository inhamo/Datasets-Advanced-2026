"""Generate separate initial_deposits.jsonl tables from account openings."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"

INITIAL_DEPOSIT_COLUMNS = ["account_id", "amount", "channel", "channel_metadata", "transaction_date"]


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
    if isinstance(value, date):
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


def build_record(row: pd.Series) -> dict[str, Any]:
    account_id = str(row["account_id"])
    opened = next_business_day(pd.to_datetime(row["opening_date"]).date())
    rng = stable_rng(account_id)
    amount = deposit_amount(row)
    branch_code = row.get("branch_code")
    if branch_code is None or pd.isna(branch_code):
        branch_code = f"{rng.randint(100000, 999999)}"

    channel = "atm" if rng.random() < 0.15 else "branch"
    if channel == "atm":
        metadata = {
            "atm_id": f"ATM-{str(branch_code)[-3:]}-OPENING",
            "terminal_id": f"ATMTERM-{str(branch_code)[-3:]}",
            "branch_code": str(branch_code),
        }
    else:
        metadata = {
            "branch_code": str(branch_code),
            "terminal_id": f"BR-{branch_code}-OPENING",
            "atm_id": None,
        }

    record = {
        "account_id": account_id,
        "amount": amount,
        "channel": channel,
        "channel_metadata": metadata,
        "transaction_date": opened.isoformat(),
    }
    return {col: clean_value(record.get(col)) for col in INITIAL_DEPOSIT_COLUMNS}


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
    accounts = load_accounts(start_year, end_year)
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for _, row in accounts.iterrows():
        opened = next_business_day(pd.to_datetime(row["opening_date"]).date())
        out = BANKING_DIR / str(opened.year) / f"{opened.month:02d}" / "initial_deposits.jsonl"
        grouped.setdefault(out, []).append(build_record(row))

    total = 0
    for path, records in sorted(grouped.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        records.sort(key=lambda r: (r["transaction_date"], r["account_id"]))
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
