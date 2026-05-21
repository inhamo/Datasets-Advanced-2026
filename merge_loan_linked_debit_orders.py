"""
Merge yearly loan-linked debit order masters into monthly debit order masters.

Reads:
  banking_data/loan_linked_debit_orders_YYYY.parquet

Writes:
  banking_data/YYYY/MM/debit_orders_YYYY_MM.parquet

After a successful merge, the yearly root source files are deleted so debit order
masters live inside the monthly banking_data folders only.
"""

from __future__ import annotations

from calendar import monthrange
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"


def monthly_debit_order_schema() -> list[str]:
    for path in sorted(BANKING_DIR.glob("20*/??/debit_orders_*.parquet")):
        return list(pd.read_parquet(path).columns)
    raise FileNotFoundError("No monthly debit_orders parquet file found to infer schema.")


def prepare_linked_rows(linked: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    prepared = linked.copy()
    defaults = {
        "suspension_reason": "",
        "suspension_initiated_by": "",
        "cancellation_reason": "",
        "can_be_reactivated": True,
        "notification_required": True,
        "notification_days_before": 2,
        "notification_method": "SMS",
        "account_to": "",
        "beneficiary_account_number": "",
        "beneficiary_branch_code": "",
        "beneficiary_bank_name": "",
        "beneficiary_account_type": "",
        "beneficiary_name": "Keystone Bank Loan Collections",
        "creditor_id": "KEYSTONE-LOAN-COLLECTIONS",
        "linked_policy_number": "",
        "linked_subscription_id": "",
        "linked_account_internal": "",
    }
    for column in columns:
        if column not in prepared.columns:
            prepared[column] = defaults.get(column, pd.NA)

    for column in ["suspension_date", "cancellation_date"]:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
    for column in ["start_date", "end_date", "record_last_updated_at"]:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce")

    return prepared[columns]


def active_in_month(rows: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = pd.Timestamp(year=year, month=month, day=monthrange(year, month)[1])
    start_ok = rows["start_date"].isna() | (rows["start_date"] <= month_end)
    end_ok = rows["end_date"].isna() | (rows["end_date"] >= month_start)
    return rows.loc[start_ok & end_ok].copy()


def merge_year(source_path: Path, columns: list[str]) -> dict[str, int]:
    year = int(source_path.stem.rsplit("_", 1)[1])
    linked = pd.read_parquet(source_path)
    linked = linked.sort_values(["debit_order_id", "record_last_updated_at"]).drop_duplicates("debit_order_id", keep="last")
    linked = prepare_linked_rows(linked, columns)

    months_written = 0
    linked_rows_added = 0
    linked_unique_in_months = set()

    for month_dir in sorted((BANKING_DIR / str(year)).glob("??")):
        if not month_dir.is_dir():
            continue
        month = int(month_dir.name)
        target_path = month_dir / f"debit_orders_{year}_{month:02d}.parquet"
        if target_path.exists():
            existing = pd.read_parquet(target_path)
            for column in columns:
                if column not in existing.columns:
                    existing[column] = pd.NA
            existing = existing[columns]
        else:
            existing = pd.DataFrame(columns=columns)

        month_linked = active_in_month(linked, year, month)
        if month_linked.empty and target_path.exists():
            continue

        combined = pd.concat([existing, month_linked], ignore_index=True)
        before = len(existing)
        combined = combined.drop_duplicates("debit_order_id", keep="last")
        combined.to_parquet(target_path, index=False)

        added = max(0, len(combined) - before)
        linked_rows_added += added
        linked_unique_in_months.update(month_linked["debit_order_id"].dropna().astype(str).tolist())
        months_written += 1

    source_path.unlink()
    return {
        "year": year,
        "source_unique_ids": int(linked["debit_order_id"].nunique()),
        "months_written": months_written,
        "linked_rows_added": linked_rows_added,
        "linked_unique_distributed": len(linked_unique_in_months),
    }


def main() -> None:
    columns = monthly_debit_order_schema()
    sources = sorted(BANKING_DIR.glob("loan_linked_debit_orders_*.parquet"))
    if not sources:
        print("No root loan_linked_debit_orders_YYYY.parquet files found.")
        return

    results = [merge_year(path, columns) for path in sources]
    print("Merged loan-linked debit orders into monthly debit order masters:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
