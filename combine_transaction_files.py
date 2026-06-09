"""
Combine monthly debit order and loan payment transaction files into transactions.jsonl.

For each banking_data/YYYY/MM folder:
  - append debit_order_transactions_YYYY_MM.parquet rows to transactions.jsonl
  - append loan_payment_transactions_YYYY_MM.parquet or .csv rows to transactions.jsonl
  - delete the separate debit_order_transactions and loan_payment_transactions files

The appended rows preserve useful banking fields and add a common transaction
timestamp and category.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) and not isinstance(value, (list, dict, tuple)):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return clean_value(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_value(v) for v in value]
    return value


def read_existing_matching_ids(path: Path, wanted_ids: set[str]) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tx_id = obj.get("transaction_id")
            if tx_id and str(tx_id) in wanted_ids:
                ids.add(str(tx_id))
                if len(ids) == len(wanted_ids):
                    break
    return ids


def transaction_timestamp(row: dict[str, Any]) -> str | None:
    date_part = row.get("transaction_date")
    time_part = row.get("transaction_time")
    if date_part is None:
        return None
    date_text = str(date_part)
    if " " in date_text:
        date_text = date_text.split(" ")[0]
    time_text = "00:00:00" if time_part is None else str(time_part)
    if " " in time_text:
        time_text = time_text.split()[-1]
    try:
        return datetime.fromisoformat(f"{date_text}T{time_text}").isoformat()
    except ValueError:
        return f"{date_text}T{time_text}"


def normalize_status(value: Any) -> str:
    text = str(value or "").strip()
    return text[:1].lower() + text[1:] if text else ""


def normalize_debit_credit(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"debit", "dr"}:
        return "debit"
    if text in {"credit", "cr"}:
        return "credit"
    return text


def row_to_json_record(row: dict[str, Any], year: int, month: int, source_table: str) -> dict[str, Any]:
    cleaned = {str(k): clean_value(v) for k, v in row.items()}
    ts = transaction_timestamp(cleaned)
    category = "loan_payment" if source_table == "loan_payment_transactions" else "debit_order"
    excluded = {
        "bank_name",
        "batch_id",
        "generation_timestamp",
        "record_last_updated_at",
        "transaction_date",
        "transaction_time",
        "source_system",
        "has_data_error",
        "data_error_types",
        "ewallet_number",
        "is_immediate_payment",
        "immediate_payment",
    }
    record = {
        **{key: value for key, value in cleaned.items() if key not in excluded},
        "transaction_timestamp": ts,
        "category": category,
        "amount": clean_value(cleaned.get("amount")),
        "debit_credit": normalize_debit_credit(cleaned.get("debit_credit")),
        "status": normalize_status(cleaned.get("status")),
        "merchant_name": cleaned.get("beneficiary_name") or ("Keystone Bank Loans" if category == "loan_payment" else None),
    }
    return record


def read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def append_source(
    *,
    transactions_path: Path,
    source_path: Path,
    year: int,
    month: int,
    source_table: str,
) -> int:
    if not source_path.exists():
        return 0
    df = read_source(source_path)
    if df.empty:
        return 0
    source_ids = {str(v) for v in df.get("transaction_id", pd.Series(dtype=str)).dropna().tolist()}
    existing_ids = read_existing_matching_ids(transactions_path, source_ids)

    appended = 0
    transactions_path.parent.mkdir(parents=True, exist_ok=True)
    with transactions_path.open("a", encoding="utf-8", newline="\n") as f:
        for row in df.to_dict(orient="records"):
            tx_id = row.get("transaction_id")
            if tx_id is None or str(tx_id) in existing_ids:
                continue
            record = row_to_json_record(row, year, month, source_table)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            existing_ids.add(str(tx_id))
            appended += 1
    return appended


def delete_if_exists(path: Path) -> bool:
    if path.exists():
        path.unlink()
        return True
    return False


def combine_month(year: int, month: int) -> dict[str, Any]:
    month_dir = BANKING_DIR / f"{year}" / f"{month:02d}"
    transactions_path = month_dir / "transactions.jsonl"

    debit_path = month_dir / f"debit_order_transactions_{year}_{month:02d}.parquet"
    loan_parquet = month_dir / f"loan_payment_transactions_{year}_{month:02d}.parquet"
    loan_csv = month_dir / f"loan_payment_transactions_{year}_{month:02d}.csv"

    debit_appended = append_source(
        transactions_path=transactions_path,
        source_path=debit_path,
        year=year,
        month=month,
        source_table="debit_order_transactions",
    )
    loan_appended = append_source(
        transactions_path=transactions_path,
        source_path=loan_parquet if loan_parquet.exists() else loan_csv,
        year=year,
        month=month,
        source_table="loan_payment_transactions",
    )

    deleted = []
    for path in [debit_path, loan_parquet, loan_csv]:
        if delete_if_exists(path):
            deleted.append(path.name)

    return {
        "year": year,
        "month": month,
        "debit_appended": debit_appended,
        "loan_appended": loan_appended,
        "deleted_files": deleted,
    }


def main() -> None:
    results = []
    for year_dir in sorted(BANKING_DIR.glob("20*")):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        for month_dir in sorted(year_dir.glob("??")):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            results.append(combine_month(year, int(month_dir.name)))

    total_debit = sum(r["debit_appended"] for r in results)
    total_loan = sum(r["loan_appended"] for r in results)
    total_deleted = sum(len(r["deleted_files"]) for r in results)
    print("Combined debit order and loan payment transactions into transactions.jsonl")
    print(f"  Months processed: {len(results)}")
    print(f"  Debit order rows appended: {total_debit}")
    print(f"  Loan payment rows appended: {total_loan}")
    print(f"  Source files deleted: {total_deleted}")

    with (BANKING_DIR / "transaction_combine_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["year", "month", "debit_appended", "loan_appended", "deleted_files"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "deleted_files": "|".join(row["deleted_files"])})


if __name__ == "__main__":
    main()
