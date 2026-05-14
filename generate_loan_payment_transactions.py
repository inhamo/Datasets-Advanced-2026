from __future__ import annotations

import argparse
import calendar
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


START_YEAR = 2024
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"


@dataclass
class ColumnMap:
    loan_id: str
    account_id: str
    customer_id: str
    loan_type: str
    status: str
    amount_granted: str
    monthly_installment: str
    term_months: str


def _resolve_input_files(record_type: str, year: int) -> list[Path]:
    """Prefer monthly files for a year; fallback to flat file."""
    year_dir = DATA_DIR / str(year)
    monthly_parquet = sorted(year_dir.glob(f"*/{record_type}_{year}_*.parquet"))
    monthly_csv = sorted(year_dir.glob(f"*/{record_type}_{year}_*.csv"))

    if monthly_parquet:
        return monthly_parquet
    if monthly_csv:
        return monthly_csv

    flat_parquet = DATA_DIR / f"{record_type}_{year}.parquet"
    flat_csv = DATA_DIR / f"{record_type}_{year}.csv"
    if flat_parquet.exists():
        return [flat_parquet]
    if flat_csv.exists():
        return [flat_csv]
    return []


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _pick_col(columns: pd.Index, options: list[str]) -> str | None:
    for c in options:
        if c in columns:
            return c
    return None


def _resolve_loan_columns(loans_df: pd.DataFrame) -> ColumnMap:
    loan_id = _pick_col(loans_df.columns, ["loan_id", "id", "loan_reference"])
    account_id = _pick_col(loans_df.columns, ["account_id", "account_no"])
    customer_id = _pick_col(loans_df.columns, ["customer_id", "customer_no"])
    loan_type = _pick_col(loans_df.columns, ["loan_type", "product_type"])
    status = _pick_col(loans_df.columns, ["workflow_state", "application_status", "status"])
    amount_granted = _pick_col(loans_df.columns, ["amount_granted", "principal_amount", "loan_amount", "amount"])
    monthly_installment = _pick_col(
        loans_df.columns,
        ["monthly_installment", "reduced_installment", "installment_amount", "monthly_payment", "payment_amount"],
    )
    term_months = _pick_col(loans_df.columns, ["term_months", "terms_months", "loan_term", "duration_months"])

    required = {
        "loan_id": loan_id,
        "account_id": account_id,
        "customer_id": customer_id,
        "loan_type": loan_type,
        "status": status,
        "amount_granted": amount_granted,
        "monthly_installment": monthly_installment,
        "term_months": term_months,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Loan file missing required columns: {missing}")

    return ColumnMap(
        loan_id=loan_id,
        account_id=account_id,
        customer_id=customer_id,
        loan_type=loan_type,
        status=status,
        amount_granted=amount_granted,
        monthly_installment=monthly_installment,
        term_months=term_months,
    )


def _normalize_status(value: object) -> str:
    return str(value).strip().lower() if pd.notna(value) else ""


def _booked_mask(loans_df: pd.DataFrame, cm: ColumnMap) -> pd.Series:
    status_col = loans_df[cm.status].map(_normalize_status)
    # Support both schemas:
    # - workflow_state: Booked / Withdrawn / Declined
    # - application_status: Approved / Rejected
    if cm.status == "workflow_state":
        return status_col == "booked"
    return status_col.isin(["approved", "booked"])


def _first_non_null_datetime(row: pd.Series, columns: list[str]) -> pd.Timestamp | pd.NaT:
    for col in columns:
        if col in row.index:
            v = pd.to_datetime(row.get(col), errors="coerce")
            if pd.notna(v):
                return v
    return pd.NaT


def _clip_day(year: int, month: int, payment_day: int) -> int:
    return max(1, min(payment_day, calendar.monthrange(year, month)[1]))


def _loan_start_date(row: pd.Series) -> pd.Timestamp:
    dt = _first_non_null_datetime(
        row,
        ["first_payment_date", "disbursed_at", "disbursement_date", "booked_at", "approval_date", "decision_at", "application_date"],
    )
    if pd.isna(dt):
        return pd.Timestamp(datetime(datetime.now().year, 1, 15))
    return dt


def _build_existing_paid_keys(target_year: int) -> set[tuple[str, str]]:
    """
    Build keys of already-existing loan payments from debit-order transactions.
    Key = (loan_id, YYYY-MM-DD)
    """
    keys: set[tuple[str, str]] = set()

    tx_files = _resolve_input_files("debit_order_transactions", target_year)
    if not tx_files:
        return keys

    # Build debit_order_id -> linked_loan_id map for the same year.
    order_map: dict[str, str] = {}
    order_files = _resolve_input_files("debit_orders", target_year)
    for f in order_files:
        df = _read_table(f)
        if "debit_order_id" in df.columns and "linked_loan_id" in df.columns:
            tmp = df[["debit_order_id", "linked_loan_id"]].dropna()
            for _, r in tmp.iterrows():
                order_map[str(r["debit_order_id"])] = str(r["linked_loan_id"])

    for f in tx_files:
        tx = _read_table(f)
        if tx.empty:
            continue

        tx_date_col = _pick_col(tx.columns, ["transaction_date", "date"])
        if tx_date_col is None:
            continue

        loan_col = _pick_col(tx.columns, ["loan_id"])
        do_col = _pick_col(tx.columns, ["debit_order_id"])

        tx_dates = pd.to_datetime(tx[tx_date_col], errors="coerce")

        if loan_col is not None:
            for loan_id, tx_date in zip(tx[loan_col], tx_dates):
                if pd.notna(loan_id) and pd.notna(tx_date):
                    keys.add((str(loan_id), tx_date.strftime("%Y-%m-%d")))

        if do_col is not None:
            for debit_order_id, tx_date in zip(tx[do_col], tx_dates):
                if pd.isna(debit_order_id) or pd.isna(tx_date):
                    continue
                linked = order_map.get(str(debit_order_id))
                if linked:
                    keys.add((linked, tx_date.strftime("%Y-%m-%d")))

    return keys


def _load_booked_loans(min_year: int, target_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for y in range(min_year, target_year + 1):
        files = _resolve_input_files("loans", y)
        if not files:
            print(f"No loan files found for year {y}.")
            continue
        for f in files:
            try:
                df = _read_table(f)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"Failed reading {f}: {exc}")

    if not frames:
        return pd.DataFrame()

    loans = pd.concat(frames, ignore_index=True)
    cm = _resolve_loan_columns(loans)

    loans = loans[_booked_mask(loans, cm)].copy()
    loans[cm.amount_granted] = pd.to_numeric(loans[cm.amount_granted], errors="coerce").fillna(0.0)
    loans[cm.monthly_installment] = pd.to_numeric(loans[cm.monthly_installment], errors="coerce").fillna(0.0)
    loans[cm.term_months] = pd.to_numeric(loans[cm.term_months], errors="coerce").fillna(0).astype(int)

    loans = loans[(loans[cm.amount_granted] > 0) & (loans[cm.monthly_installment] > 0) & (loans[cm.term_months] > 0)].copy()
    loans = loans.drop_duplicates(subset=[cm.loan_id])

    return loans


def _build_existing_loan_mandates(target_year: int) -> dict[str, str]:
    """Map loan_id -> debit_order_id from existing debit orders."""
    mapping: dict[str, str] = {}
    for f in _resolve_input_files("debit_orders", target_year):
        try:
            df = _read_table(f)
        except Exception:
            continue
        if "debit_order_id" not in df.columns or "linked_loan_id" not in df.columns:
            continue

        tmp = df[["debit_order_id", "linked_loan_id", "status"]].copy() if "status" in df.columns else df[["debit_order_id", "linked_loan_id"]].copy()
        tmp = tmp.dropna(subset=["debit_order_id", "linked_loan_id"])
        if "status" in tmp.columns:
            tmp = tmp[tmp["status"].map(_normalize_status).isin(["active", "pending", "suspended"])].copy()

        for _, r in tmp.iterrows():
            mapping[str(r["linked_loan_id"])] = str(r["debit_order_id"])

    linked_file = DATA_DIR / f"loan_linked_debit_orders_{target_year}.parquet"
    if linked_file.exists():
        linked_df = pd.read_parquet(linked_file)
        if "debit_order_id" in linked_df.columns and "linked_loan_id" in linked_df.columns:
            linked_df = linked_df.dropna(subset=["debit_order_id", "linked_loan_id"])
            for _, r in linked_df.iterrows():
                mapping[str(r["linked_loan_id"])] = str(r["debit_order_id"])

    return mapping


def _generate_new_loan_mandates(loans: pd.DataFrame, cm: ColumnMap, existing_map: dict[str, str], target_year: int) -> pd.DataFrame:
    """
    Create loan-linked debit order mandates for a subset of loans that do not yet have one.
    This gives integration with debit orders while avoiding duplicates.
    """
    rows: list[dict] = []
    counter = 1
    for _, r in loans.iterrows():
        loan_id = str(r[cm.loan_id])
        if loan_id in existing_map:
            continue
        # Not every loan must use debit order rail.
        if random.random() > 0.55:
            continue

        start = _loan_start_date(r)
        payment_day = int(r.get("payment_day", start.day))
        do_id = f"LDO{target_year}{counter:07d}"
        counter += 1

        rows.append(
            {
                "debit_order_id": do_id,
                "account_id": r.get(cm.account_id),
                "customer_id": r.get(cm.customer_id),
                "debit_order_type": "Loan Repayment",
                "amount": round(float(r[cm.monthly_installment]), 2),
                "frequency": "Monthly",
                "collection_day": payment_day,
                "is_fixed_amount": True,
                "start_date": start.normalize(),
                "end_date": pd.NaT,
                "status": "Active",
                "linked_loan_id": loan_id,
                "description": "Loan Repayment - Integrated Mandate",
                "created_in_year": target_year,
                "created_in_month": datetime.now().month,
                "record_last_updated_at": pd.Timestamp.now(),
            }
        )

    mandates_df = pd.DataFrame(rows)
    out = DATA_DIR / f"loan_linked_debit_orders_{target_year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        existing_df = pd.read_parquet(out)
        if not mandates_df.empty:
            combined = pd.concat([existing_df, mandates_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["linked_loan_id"], keep="first")
            combined.to_parquet(out, index=False)
            print(f"Generated {len(mandates_df)} new loan-linked debit orders: {out}")
            return mandates_df
        return pd.DataFrame()

    if mandates_df.empty:
        return mandates_df

    mandates_df.to_parquet(out, index=False)
    print(f"Generated {len(mandates_df)} new loan-linked debit orders: {out}")
    return mandates_df


def _status_for_payment(is_recovery: bool = False) -> tuple[str, str | None]:
    # Completed, Failed, Cancelled
    probs = [0.96, 0.03, 0.01] if not is_recovery else [0.75, 0.22, 0.03]
    draw = np.random.choice(["Completed", "Failed", "Cancelled"], p=probs)
    if draw == "Failed":
        reason = random.choice(["insufficient_funds", "mandate_limit", "bank_timeout", "account_blocked"])
        return draw, reason
    if draw == "Cancelled":
        return draw, "customer_stop_instruction"
    return draw, None


def _payment_variation(base_amount: float, is_recovery: bool = False) -> tuple[float, str]:
    if is_recovery:
        return round(max(0.0, base_amount * random.uniform(0.4, 0.9)), 2), "recovery_payment"

    r = random.random()
    if r < 0.92:
        return round(base_amount, 2), "standard_payment"
    if r < 0.96:
        return round(base_amount * 1.05, 2), "late_payment"
    if r < 0.98:
        return round(base_amount * random.uniform(0.5, 0.85), 2), "partial_payment"
    return round(base_amount * random.uniform(1.1, 1.8), 2), "extra_payment"


def _generate_schedule_rows(
    loan_row: pd.Series,
    cm: ColumnMap,
    target_year: int,
    target_month: int | None,
    debit_order_id: str | None,
    existing_paid_keys: set[tuple[str, str]],
    generated_keys: set[tuple[str, str]],
    start_counter: int,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    counter = start_counter

    loan_id = str(loan_row[cm.loan_id])
    account_id = str(loan_row[cm.account_id])
    customer_id = str(loan_row[cm.customer_id])
    loan_type = str(loan_row[cm.loan_type])
    base_installment = float(loan_row[cm.monthly_installment])
    terms = int(loan_row[cm.term_months])

    start_date = _loan_start_date(loan_row)
    payment_day = int(loan_row.get("payment_day", start_date.day))
    first_month = (start_date + pd.DateOffset(months=1)).to_pydatetime()

    # Mild default/recovery simulation only to enrich transaction realism.
    will_default = random.random() < 0.04
    default_after_n = random.randint(3, min(24, terms)) if will_default else None
    recovery_budget = random.randint(1, 4) if will_default else 0

    for installment_number in range(1, terms + 1):
        due_month = first_month + pd.DateOffset(months=installment_number - 1)
        due_year = int(due_month.year)
        due_m = int(due_month.month)
        due_day = _clip_day(due_year, due_m, payment_day)
        due_date = pd.Timestamp(datetime(due_year, due_m, due_day))

        if due_date.year != target_year:
            continue
        if target_month is not None and due_date.month != target_month:
            continue

        key = (loan_id, due_date.strftime("%Y-%m-%d"))
        if key in existing_paid_keys or key in generated_keys:
            # Hard guard against double payment.
            continue

        is_recovery = False
        if will_default and installment_number >= int(default_after_n):
            if recovery_budget <= 0 or random.random() > 0.35:
                continue
            recovery_budget -= 1
            is_recovery = True

        amount, payment_type = _payment_variation(base_installment, is_recovery=is_recovery)
        status, failure_reason = _status_for_payment(is_recovery=is_recovery)

        # Rail integration: if linked debit order exists, keep channel as Automated.
        if debit_order_id:
            channel = "Automated"
            immediate_payment = False
            tx_cost = round(random.uniform(0.0, 2.0), 2)
        else:
            channel = np.random.choice(["Automated", "Online", "Mobile"], p=[0.84, 0.11, 0.05])
            immediate_payment = channel in ["Online", "Mobile"] and random.random() < 0.07
            tx_cost = round(5.0 * (2.0 if immediate_payment else 0.5), 2)

        tx_hour = random.choice([2, 3, 4]) if channel == "Automated" else random.randint(1, 7)
        tx_time = f"{tx_hour:02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"

        description = f"Loan Payment - {loan_id}"
        if payment_type == "recovery_payment":
            description += " - Recovery"
        elif payment_type == "partial_payment":
            description += " - Partial"
        elif payment_type == "extra_payment":
            description += " - Extra Principal"
        elif payment_type == "late_payment":
            description += " - Late Fee"

        rows.append(
            {
                "transaction_id": f"TXNL{target_year}{counter:08d}",
                "account_id": account_id,
                "customer_id": customer_id,
                "loan_id": loan_id,
                "debit_order_id": debit_order_id,
                "installment_number": installment_number,
                "transaction_date": due_date.strftime("%Y-%m-%d"),
                "transaction_time": tx_time,
                "amount": amount,
                "debit_credit": "Debit",
                "status": status,
                "failure_reason": failure_reason,
                "description": description,
                "immediate_payment": immediate_payment,
                "receiving_account": None,
                "transaction_cost": tx_cost,
                "ewallet_number": None,
                "channel": channel,
                "loan_type": loan_type,
                "payment_type": payment_type,
                "is_recovery_attempt": is_recovery,
                "source_system": "loan_payment_generator",
            }
        )

        generated_keys.add(key)
        counter += 1

    return rows, counter


def generate_loan_payment_transactions_for_year(
    target_year: int,
    min_year: int = START_YEAR,
    target_month: int | None = None,
) -> pd.DataFrame:
    random.seed(target_year)
    np.random.seed(target_year)

    if target_month is not None and (target_month < 1 or target_month > 12):
        raise ValueError("target_month must be in 1..12")

    loans = _load_booked_loans(min_year=min_year, target_year=target_year)
    if loans.empty:
        print(f"No booked loans found from {min_year} to {target_year}.")
        return pd.DataFrame()

    cm = _resolve_loan_columns(loans)

    existing_paid_keys = _build_existing_paid_keys(target_year)
    existing_mandates = _build_existing_loan_mandates(target_year)

    # Optional integrated mandate generation for loans without a mandate.
    new_mandates = _generate_new_loan_mandates(loans, cm, existing_mandates, target_year)
    if not new_mandates.empty:
        for _, r in new_mandates.iterrows():
            existing_mandates[str(r["linked_loan_id"])] = str(r["debit_order_id"])

    generated_keys: set[tuple[str, str]] = set()
    rows: list[dict] = []
    counter = 1

    period_label = f"{target_year}-{target_month:02d}" if target_month is not None else f"{target_year}"
    for _, loan in tqdm(loans.iterrows(), total=len(loans), desc=f"Loan payments {period_label}"):
        loan_id = str(loan[cm.loan_id])
        debit_order_id = existing_mandates.get(loan_id)

        chunk, counter = _generate_schedule_rows(
            loan_row=loan,
            cm=cm,
            target_year=target_year,
            target_month=target_month,
            debit_order_id=debit_order_id,
            existing_paid_keys=existing_paid_keys,
            generated_keys=generated_keys,
            start_counter=counter,
        )
        rows.extend(chunk)

    tx_df = pd.DataFrame(rows)
    if tx_df.empty:
        print(f"No loan payment transactions generated for {period_label}.")
        return tx_df

    # Final hard de-duplication guard: one transaction per loan per date.
    tx_df = tx_df.sort_values(["loan_id", "transaction_date", "transaction_time"]).drop_duplicates(
        subset=["loan_id", "transaction_date"], keep="first"
    )

    if target_month is None:
        output_parquet = DATA_DIR / f"loan_payment_transactions_{target_year}.parquet"
        output_csv = DATA_DIR / f"loan_payment_transactions_{target_year}.csv"
    else:
        month_dir = DATA_DIR / str(target_year) / f"{target_month:02d}"
        output_parquet = month_dir / f"loan_payment_transactions_{target_year}_{target_month:02d}.parquet"
        output_csv = month_dir / f"loan_payment_transactions_{target_year}_{target_month:02d}.csv"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    tx_df.to_parquet(output_parquet, index=False)
    tx_df.to_csv(output_csv, index=False)

    print(f"Generated {len(tx_df)} loan payment transactions for {period_label}")
    print(f"Saved parquet: {output_parquet}")
    print(f"Saved csv: {output_csv}")
    print(f"Duplicate protection keys used: existing={len(existing_paid_keys):,}, generated={len(generated_keys):,}")

    return tx_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate integrated loan payment transactions with no double payments")
    parser.add_argument("--start_year", type=int, default=START_YEAR, help="First year to process")
    parser.add_argument("--end_year", type=int, default=datetime.now().year, help="Last year to process")
    parser.add_argument("--year", type=int, help="Single target year")
    parser.add_argument("--month", type=int, help="Single target month (1-12)")
    args = parser.parse_args()

    if args.month is not None and (args.month < 1 or args.month > 12):
        raise ValueError("month must be in 1..12")

    if args.year is not None:
        if args.month is None:
            for m in range(1, 13):
                generate_loan_payment_transactions_for_year(target_year=args.year, min_year=args.start_year, target_month=m)
        else:
            generate_loan_payment_transactions_for_year(target_year=args.year, min_year=args.start_year, target_month=args.month)
        return

    if args.start_year > args.end_year:
        raise ValueError("start_year cannot be greater than end_year")

    for y in range(args.start_year, args.end_year + 1):
        for m in range(1, 13):
            generate_loan_payment_transactions_for_year(target_year=y, min_year=args.start_year, target_month=m)


if __name__ == "__main__":
    main()
