from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
BANKING = ROOT / "banking_data"
REPORT_DIR = ROOT / "migration_artifacts" / "collection_account_integrity"
COLLECTION_START = (2019, 12)


def ym_from_dir(month_dir: Path) -> tuple[int, int]:
    return int(month_dir.parent.name), int(month_dir.name)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def schema_names(path: Path) -> list[str]:
    return pq.read_schema(path).names


def month_dirs() -> list[Path]:
    return sorted(p for p in BANKING.glob("20*/[0-1][0-9]") if p.is_dir())


def clean_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def parse_dt(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def load_account_schema() -> tuple[list[str], pd.DataFrame]:
    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        path = month_dir / f"accounts_{year}_{month:02d}.parquet"
        if path.exists() and {"account_id", "customer_id"} <= set(schema_names(path)):
            frame = read_parquet(path)
            return list(frame.columns), frame.head(1)
    raise RuntimeError("No valid account parquet found to use as schema template.")


def load_customer_schema() -> tuple[list[str], pd.DataFrame]:
    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        path = month_dir / f"customers_{year}_{month:02d}.parquet"
        if path.exists() and "customer_id" in schema_names(path):
            frame = read_parquet(path)
            return list(frame.columns), frame.head(1)
    raise RuntimeError("No valid customer parquet found to use as schema template.")


def collect_accounts() -> pd.DataFrame:
    frames = []
    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        path = month_dir / f"accounts_{year}_{month:02d}.parquet"
        if not path.exists() or {"account_id", "customer_id"} - set(schema_names(path)):
            continue
        cols = [c for c in ["account_id", "customer_id", "opening_date"] if c in schema_names(path)]
        frame = read_parquet(path, columns=cols)
        frame["source_year"] = year
        frame["source_month"] = month
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in ["account_id", "customer_id"]:
        if col in out.columns:
            out[col] = out[col].map(clean_id)
    return out.drop_duplicates("account_id")


def collect_customers() -> set[str]:
    ids: set[str] = set()
    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        path = month_dir / f"customers_{year}_{month:02d}.parquet"
        if path.exists() and "customer_id" in schema_names(path):
            ids.update(read_parquet(path, columns=["customer_id"])["customer_id"].map(clean_id))
    return {cid for cid in ids if cid}


def collect_loans() -> pd.DataFrame:
    frames = []
    wanted = [
        "loan_id",
        "account_id",
        "customer_id",
        "loan_type",
        "application_channel",
        "application_date",
        "booked_at",
        "disbursed_at",
        "application_status",
        "amount_granted",
        "monthly_installment",
    ]
    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        path = month_dir / f"loans_{year}_{month:02d}.parquet"
        if not path.exists():
            continue
        cols = [c for c in wanted if c in schema_names(path)]
        if {"account_id", "customer_id"} - set(cols):
            continue
        frame = read_parquet(path, columns=cols)
        frame["source_year"] = year
        frame["source_month"] = month
        frames.append(frame)
    loans = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in ["account_id", "customer_id", "loan_id"]:
        if col in loans.columns:
            loans[col] = loans[col].map(clean_id)
    if "application_status" in loans.columns:
        approved = loans["application_status"].astype(str).str.lower().eq("approved")
        loans = loans[approved | loans["application_status"].isna()].copy()
    loans = loans.sort_values(["source_year", "source_month", "account_id"])
    return loans.drop_duplicates("account_id")


def collect_debit_orders() -> pd.DataFrame:
    frames = []
    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        path = month_dir / f"debit_orders_{year}_{month:02d}.parquet"
        if not path.exists() or {"account_id", "customer_id"} - set(schema_names(path)):
            continue
        frame = read_parquet(path, columns=["account_id", "customer_id"])
        frame["source_year"] = year
        frame["source_month"] = month
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in ["account_id", "customer_id"]:
        if col in out.columns:
            out[col] = out[col].map(clean_id)
    return out.drop_duplicates("account_id")


def transaction_account_ids() -> set[str]:
    pattern = re.compile(r'"account_id"\s*:\s*"([^"]+)"')
    ids: set[str] = set()
    for path in sorted(BANKING.glob("20*/[0-1][0-9]/transactions.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    ids.add(clean_id(match.group(1)))
    return {aid for aid in ids if aid}


def default_for_column(column: str, source: pd.Series, opening: date, seq: int) -> Any:
    if column in source.index and pd.notna(source[column]) and str(source[column]).strip() != "":
        return source[column]
    if column == "account_id":
        return source["account_id"]
    if column == "customer_id":
        return source["customer_id"]
    if column == "account_number":
        return f"25065501{seq:06d}"
    if column == "bank_product_name":
        return "Transact Account"
    if column == "account_type":
        return "current"
    if column == "account_purpose":
        return "loan_servicing"
    if column in {"is_primary_account", "kyc_verified", "fica_verified", "minimum_deposit_met"}:
        return True
    if column in {"is_joint_account", "overdraft_enabled", "is_dormant"}:
        return False
    if column in {"opening_date", "approval_date"}:
        return opening
    if column == "branch_code":
        return "125405"
    if column == "expected_amount":
        return float(source.get("amount_granted", 0) or 0)
    if column == "account_status":
        return "active"
    if column in {"status_change_date", "closure_date", "status_reason", "linked_joint_accounts"}:
        return None
    if column == "interest_rate":
        return 0.005
    if column == "monthly_charges":
        return 20
    if column == "transactions_rate":
        return 0.01
    if column == "overdraft_limit":
        return 0.0
    if column == "currency":
        return "ZAR"
    if column == "account_tier":
        return "standard"
    if column == "statement_frequency":
        return "monthly"
    if column == "opening_channel":
        channel = clean_id(source.get("application_channel", "branch")).lower()
        return "branch" if channel in {"", "nan"} else channel
    if column == "card_number":
        return None
    if column == "card_type":
        return None
    if column == "card_issue_date":
        return None
    if column == "card_expiry_date":
        return None
    if column in {"beneficiaries", "swift_code", "iban"}:
        return None
    if column in {"limits_history_json", "status_events_json", "product_enrollments_json", "signatories_json"}:
        return "[]"
    if column == "cdc_op_hint":
        return "I"
    if column == "record_last_updated_at":
        return pd.Timestamp.utcnow().tz_localize(None)
    return None


def make_account_row(columns: list[str], source: pd.Series, seq: int) -> dict[str, Any]:
    opened = parse_dt(source.get("booked_at")) or parse_dt(source.get("disbursed_at")) or parse_dt(source.get("application_date"))
    opening_date = opened.date() if opened is not None else date(int(source["source_year"]), int(source["source_month"]), 1)
    return {col: default_for_column(col, source, opening_date, seq) for col in columns}


def repair_accounts() -> dict[str, int]:
    account_columns, _template = load_account_schema()
    accounts = collect_accounts()
    loans = collect_loans()
    debit_orders = collect_debit_orders()
    tx_ids = transaction_account_ids()

    account_ids = set(accounts["account_id"])
    missing_ids = sorted((tx_ids | set(loans["account_id"]) | set(debit_orders["account_id"])) - account_ids)

    source = loans.set_index("account_id", drop=False)
    debit_source = debit_orders.set_index("account_id", drop=False)
    rows_by_month: dict[tuple[int, int], list[dict[str, Any]]] = {}
    unresolved = []
    for seq, account_id in enumerate(missing_ids, start=900000):
        if account_id in source.index:
            row = source.loc[account_id]
        elif account_id in debit_source.index:
            row = debit_source.loc[account_id]
        else:
            unresolved.append(account_id)
            continue
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        year = int(row.get("source_year", 2019))
        month = int(row.get("source_month", 12))
        rows_by_month.setdefault((year, month), []).append(make_account_row(account_columns, row, seq))

    # Rewrite broken empty shell account files using the full account schema.
    fixed_shells = 0
    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        path = month_dir / f"accounts_{year}_{month:02d}.parquet"
        if not path.exists():
            continue
        if {"account_id", "customer_id"} - set(schema_names(path)):
            empty = pd.DataFrame(columns=account_columns)
            empty.to_parquet(path, index=False)
            fixed_shells += 1

    added = 0
    for (year, month), rows in rows_by_month.items():
        path = BANKING / str(year) / f"{month:02d}" / f"accounts_{year}_{month:02d}.parquet"
        if path.exists() and {"account_id", "customer_id"} <= set(schema_names(path)):
            existing = read_parquet(path)
        else:
            existing = pd.DataFrame(columns=account_columns)
        for col in account_columns:
            if col not in existing.columns:
                existing[col] = None
        addition = pd.DataFrame(rows)
        combined = pd.concat([existing[account_columns], addition[account_columns]], ignore_index=True)
        combined["account_id"] = combined["account_id"].map(clean_id)
        combined["customer_id"] = combined["customer_id"].map(clean_id)
        combined = combined.drop_duplicates("account_id", keep="last")
        combined.to_parquet(path, index=False)
        added += len(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"account_id": unresolved}).to_csv(REPORT_DIR / "unresolved_missing_accounts.csv", index=False)
    return {"account_rows_added": added, "broken_account_shells_fixed": fixed_shells, "unresolved_accounts": len(unresolved)}


def make_customer_row(columns: list[str], customer_id: str, seq: int) -> dict[str, Any]:
    row: dict[str, Any] = {col: None for col in columns}
    for col in columns:
        lower = col.lower()
        if col == "customer_id":
            row[col] = customer_id
        elif col == "customer_type":
            row[col] = "Individual"
        elif lower in {"first_name", "firstname"}:
            row[col] = f"Recovered{seq}"
        elif lower in {"last_name", "surname"}:
            row[col] = "Customer"
        elif lower in {"full_name", "customer_name", "name"}:
            row[col] = f"Recovered Customer {seq}"
        elif "date_of_birth" in lower or lower == "dob":
            row[col] = "1985-01-15"
        elif lower in {"gender"}:
            row[col] = "U"
        elif lower in {"nationality"}:
            row[col] = "SOUTH AFRICA"
        elif lower in {"citizenship", "country_code"}:
            row[col] = "ZA"
        elif "registration_date" in lower or "created" in lower:
            row[col] = "2019-12-01"
        elif lower in {"preferred_contact_method"}:
            row[col] = "SMS"
        elif lower in {"cdc_op_hint"}:
            row[col] = "I"
        elif lower in {"record_last_updated_at"}:
            row[col] = pd.Timestamp.utcnow().tz_localize(None)
    return row


def repair_missing_customers() -> dict[str, int]:
    columns, _ = load_customer_schema()
    customers = collect_customers()
    accounts = collect_accounts()
    missing = sorted(set(accounts["customer_id"]) - customers)
    if not missing:
        return {"customer_rows_added": 0}
    target = BANKING / "2019" / "12" / "customers_2019_12.parquet"
    frame = read_parquet(target)
    additions = pd.DataFrame([make_customer_row(columns, cid, i + 1) for i, cid in enumerate(missing)])
    for col in columns:
        if col not in frame.columns:
            frame[col] = None
    combined = pd.concat([frame[columns], additions[columns]], ignore_index=True)
    combined["customer_id"] = combined["customer_id"].map(clean_id)
    combined = combined.drop_duplicates("customer_id", keep="last")
    combined.to_parquet(target, index=False)
    pd.DataFrame({"customer_id": missing}).to_csv(REPORT_DIR / "created_missing_customers.csv", index=False)
    return {"customer_rows_added": len(missing)}


def account_master() -> pd.DataFrame:
    accounts = collect_accounts()
    accounts["opening_ts"] = pd.to_datetime(accounts.get("opening_date"), errors="coerce")
    return accounts.drop_duplicates("account_id", keep="first")


def loan_dates() -> pd.DataFrame:
    loans = collect_loans()
    if loans.empty:
        return pd.DataFrame(columns=["account_id", "loan_start"])
    date_cols = [c for c in ["disbursed_at", "booked_at", "application_date"] if c in loans.columns]
    for col in date_cols:
        loans[col + "_ts"] = pd.to_datetime(loans[col], errors="coerce")
    ts_cols = [c + "_ts" for c in date_cols]
    loans["loan_start"] = loans[ts_cols].bfill(axis=1).iloc[:, 0] if ts_cols else pd.NaT
    return loans[["account_id", "loan_start"]].drop_duplicates("account_id")


def repair_collections() -> dict[str, int]:
    accounts = account_master()
    customers = collect_customers()
    loan_start = loan_dates()
    master = accounts.merge(loan_start, on="account_id", how="left")
    master_by_account = master.set_index("account_id")

    removed_early_files = 0
    rows_before = 0
    rows_after = 0
    dropped_rows = []

    for month_dir in month_dirs():
        year, month = ym_from_dir(month_dir)
        out_dir = month_dir / "collections_cases"
        cases_path = out_dir / "collections_cases.csv"
        recoveries_path = out_dir / "recovery_payments.csv"
        if not cases_path.exists():
            continue

        cases = pd.read_csv(cases_path, dtype=str)
        rows_before += len(cases)
        recoveries = pd.read_csv(recoveries_path, dtype=str) if recoveries_path.exists() else pd.DataFrame()

        if (year, month) < COLLECTION_START:
            dropped_rows.extend(cases.assign(drop_reason="before_collection_start").to_dict("records"))
            cases_path.unlink(missing_ok=True)
            recoveries_path.unlink(missing_ok=True)
            try:
                out_dir.rmdir()
            except OSError:
                pass
            removed_early_files += 1
            continue

        for col in ["account_id", "customer_id"]:
            if col in cases.columns:
                cases[col] = cases[col].map(clean_id)
            if not recoveries.empty and col in recoveries.columns:
                recoveries[col] = recoveries[col].map(clean_id)

        keep_indexes = []
        for idx, row in cases.iterrows():
            account_id = clean_id(row.get("account_id"))
            if account_id not in master_by_account.index:
                dropped = row.to_dict()
                dropped["drop_reason"] = "account_not_in_master"
                dropped_rows.append(dropped)
                continue
            info = master_by_account.loc[account_id]
            if isinstance(info, pd.DataFrame):
                info = info.iloc[0]
            customer_id = clean_id(info["customer_id"])
            if customer_id not in customers:
                dropped = row.to_dict()
                dropped["drop_reason"] = "account_customer_not_in_customers"
                dropped_rows.append(dropped)
                continue

            last_contact = parse_dt(row.get("last_contact_date")) or pd.Timestamp(year=year, month=month, day=1)
            days_past_due = int(float(row.get("days_past_due") or 0))
            basis = info.get("loan_start")
            if pd.isna(basis):
                basis = info.get("opening_ts")
            if pd.notna(basis):
                minimum_observed_date = basis + pd.Timedelta(days=max(30, days_past_due))
                if last_contact < minimum_observed_date:
                    dropped = row.to_dict()
                    dropped["drop_reason"] = "collection_before_observable_delinquency"
                    dropped["minimum_observed_date"] = minimum_observed_date.date().isoformat()
                    dropped_rows.append(dropped)
                    continue

            cases.at[idx, "customer_id"] = customer_id
            keep_indexes.append(idx)

        cases = cases.loc[keep_indexes].copy()
        if cases.empty:
            cases_path.unlink(missing_ok=True)
            recoveries_path.unlink(missing_ok=True)
            try:
                out_dir.rmdir()
            except OSError:
                pass
            continue

        valid_case_ids = set(cases["case_id"])
        if not recoveries.empty:
            recoveries = recoveries[recoveries["case_id"].isin(valid_case_ids)].copy()
            if not recoveries.empty:
                case_customer = cases.set_index("case_id")["customer_id"].to_dict()
                case_account = cases.set_index("case_id")["account_id"].to_dict()
                recoveries["customer_id"] = recoveries["case_id"].map(case_customer)
                recoveries["account_id"] = recoveries["case_id"].map(case_account)

        cases.to_csv(cases_path, index=False)
        if recoveries_path.exists() or not recoveries.empty:
            recoveries.to_csv(recoveries_path, index=False)
        rows_after += len(cases)

    pd.DataFrame(dropped_rows).to_csv(REPORT_DIR / "dropped_collection_rows.csv", index=False)
    return {
        "collection_rows_before": rows_before,
        "collection_rows_after": rows_after,
        "collection_rows_dropped": len(dropped_rows),
        "early_collection_folders_removed": removed_early_files,
    }


def verify() -> dict[str, Any]:
    accounts = collect_accounts()
    customers = collect_customers()
    tx_ids = transaction_account_ids()
    account_ids = set(accounts["account_id"])
    bad_account_customers = sorted(set(accounts["customer_id"]) - customers)

    coll_bad_accounts = set()
    coll_bad_customers = set()
    earliest_collection = None
    collection_rows = 0
    for path in BANKING.glob("20*/[0-1][0-9]/collections_cases/collections_cases.csv"):
        frame = pd.read_csv(path, dtype=str)
        if frame.empty:
            continue
        collection_rows += len(frame)
        frame["account_id"] = frame["account_id"].map(clean_id)
        frame["customer_id"] = frame["customer_id"].map(clean_id)
        coll_bad_accounts.update(set(frame["account_id"]) - account_ids)
        coll_bad_customers.update(set(frame["customer_id"]) - customers)
        y, m = ym_from_dir(path.parents[1])
        earliest_collection = min((earliest_collection or (y, m)), (y, m))

    result = {
        "transaction_accounts_missing": len(tx_ids - account_ids),
        "bad_account_customer_ids": len(bad_account_customers),
        "collection_accounts_missing": len(coll_bad_accounts),
        "collection_customers_missing": len(coll_bad_customers),
        "collection_rows_remaining": collection_rows,
        "earliest_collection_month": None if earliest_collection is None else f"{earliest_collection[0]}-{earliest_collection[1]:02d}",
        "sample_missing_transaction_accounts": sorted(list(tx_ids - account_ids))[:20],
        "sample_bad_account_customers": bad_account_customers[:20],
    }
    (REPORT_DIR / "verification_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    summary.update(repair_accounts())
    summary.update(repair_missing_customers())
    summary.update(repair_collections())
    verification = verify()
    summary["verification"] = verification
    (REPORT_DIR / "repair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
