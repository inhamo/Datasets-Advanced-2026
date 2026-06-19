"""
Generate synthetic collections and recoveries data for a South African retail bank.

Reads loan payment transactions, optional customer master, and monthly signals.
Writes per month:
  banking_data/{year}/{month}/collections_cases/collections_cases.csv
  banking_data/{year}/{month}/collections_cases/recovery_payments.csv

Only churned customers (failed payments with no successful payment in-month) are
prioritised for new collections; a subset of previously churned customers may
reappear in later years.

Collections are intentionally not generated in the bank's first operating
months. A collections case requires an observed account/loan relationship and
enough elapsed time for repayment failure to age into delinquency.
"""

from __future__ import annotations

import argparse
import csv
import random
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"
CORPUS_DIR = BASE_DIR / "corpus_context"
OUTPUT_SUBDIR = "collections_cases"
DEFAULT_COLLECTION_START_YEAR = 2019
DEFAULT_COLLECTION_START_MONTH = 12
MIN_COLLECTION_ACCOUNT_AGE_DAYS = 180

STAGES = [
    ("pre_delinquent", 1, 30, 0.35),
    ("early_collections", 31, 60, 0.25),
    ("late_collections", 61, 90, 0.20),
    ("legal", 91, 120, 0.15),
    ("write_off", 121, 365, 0.05),
]

ARRANGEMENT_PLANS = ["restructured", "payment_holiday", "extended_term", "settlement_offer"]
PAYMENT_METHODS = ["debit_order", "eft", "cash_deposit", "card"]
CASE_STATUSES = ["open", "closed", "resolved"]

COLLECTORS = [
    "Thabo Mokoena",
    "Nomsa Dlamini",
    "Pieter van der Merwe",
    "Lerato Khumalo",
    "Sipho Naidoo",
    "Ayanda Mthembu",
    "Johan Botha",
    "Zanele Ndlovu",
    "David Govender",
    "Fatima Patel",
]

NOTE_TEMPLATES = [
    "Customer called, says employer missed payroll. Promised R{amount} by month-end.",
    "Phone disconnected. Sent SMS, no response. Escalating to legal.",
    "Arrangement agreed: R{amount}/month for 6 months starting {start_day} {month_name}.",
    "Customer visited branch, paid R{amount} cash. Account now current.",
    "Debit order returned NSF for third time. Recommend restructuring.",
    "COVID hardship relief applied. Payment holiday granted until June 2020.",
    "Churned customer — no active transactional account; pursuing outstanding loan balance.",
    "Former churned customer returned; new arrears flagged after failed debit order.",
    "Customer in {province}, low income band — hardship assessment required.",
    "Legal letter sent via registered post. Awaiting response before summons.",
    "Settlement offer of R{amount} communicated; customer requested until month-end.",
    "Promise to pay recorded after outbound call. Follow-up scheduled.",
]

CASE_FIELDS = [
    "case_id",
    "account_id",
    "customer_id",
    "arrears_amount",
    "days_past_due",
    "collection_stage",
    "last_contact_date",
    "last_contact_channel",
    "promise_to_pay_amount",
    "promise_to_pay_date",
    "arrangement_plan",
    "status",
    "assigned_collector",
    "notes",
]

RECOVERY_FIELDS = [
    "payment_id",
    "case_id",
    "account_id",
    "customer_id",
    "payment_date",
    "amount",
    "payment_method",
    "was_promise_kept",
    "arrears_after_payment",
]


@dataclass
class MonthSignal:
    year: int
    month: int
    fail_pct: float = 0.0
    insuf_count: int = 0


@dataclass
class LoanCustomer:
    account_id: str
    customer_id: str
    amount: float
    loan_type: str
    failure_reason: str = ""


def parse_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "nan"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    if value in (None, "", "nan"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_monthly_signals() -> dict[tuple[int, int], MonthSignal]:
    path = CORPUS_DIR / "index" / "monthly_signals.csv"
    signals: dict[tuple[int, int], MonthSignal] = {}
    if not path.exists():
        return signals
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = parse_int(row.get("year"))
            month = parse_int(row.get("month"))
            signals[(year, month)] = MonthSignal(
                year=year,
                month=month,
                fail_pct=parse_float(row.get("fail_pct")),
                insuf_count=parse_int(row.get("insuf_count")),
            )
    return signals


def load_customer_master() -> dict[str, dict[str, str]]:
    path = BANKING_DIR / "customer_master.csv"
    master: dict[str, dict[str, str]] = {}
    if not path.exists():
        return master
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("customer_id")
            if cid:
                master[str(cid)] = row
    return master


def _account_master_path(year: int, month: int) -> Path:
    return BANKING_DIR / str(year) / f"{month:02d}" / f"accounts_{year}_{month:02d}.parquet"


def load_global_account_master() -> dict[str, dict[str, str]]:
    """Load all account/customer links seen up to the generation run.

    Monthly account files are cohort extracts, not full snapshots. Collections
    therefore validate against the global account universe, not just the same
    month's account file.
    """
    master: dict[str, dict[str, str]] = {}
    if pd is None:
        return master
    for path in sorted(BANKING_DIR.glob("20*/[0-1][0-9]/accounts_*.parquet")):
        try:
            import pyarrow.parquet as pq

            columns = set(pq.read_schema(path).names)
        except Exception:
            continue
        if not {"account_id", "customer_id"}.issubset(columns):
            continue
        wanted = [c for c in ["account_id", "customer_id", "opening_date"] if c in columns]
        try:
            frame = pd.read_parquet(path, columns=wanted)
        except Exception:
            continue
        for _, row in frame.iterrows():
            account_id = str(row.get("account_id", "")).strip()
            if not account_id or account_id.lower() == "nan":
                continue
            master.setdefault(
                account_id,
                {
                    "customer_id": str(row.get("customer_id", "")).strip(),
                    "opening_date": str(row.get("opening_date", "")).strip(),
                },
            )
    return master


def _is_failed_status(status: str) -> bool:
    s = status.strip().lower()
    return s in ("failed", "declined", "rejected", "timeout", "unsuccessful")


def _is_success_status(status: str) -> bool:
    s = status.strip().lower()
    return s in ("success", "completed", "successful", "paid")


def _loan_payment_path(year: int, month: int) -> Path | None:
    month_dir = BANKING_DIR / str(year) / f"{month:02d}"
    if not month_dir.exists():
        return None
    for pattern in (
        f"loan_payment_transactions_{year}_{month:02d}.parquet",
        f"loan_payment_transactions_{year}_{month:02d}.csv",
    ):
        candidate = month_dir / pattern
        if candidate.exists():
            return candidate
    matches = sorted(month_dir.glob("loan_payment_transactions_*.parquet"))
    if matches:
        return matches[0]
    matches = sorted(month_dir.glob("loan_payment_transactions_*.csv"))
    return matches[0] if matches else None


def load_loan_payment_customers(year: int, month: int) -> tuple[list[LoanCustomer], list[LoanCustomer], list[LoanCustomer]]:
    """
    Returns (all_candidates, churned_failed, returned_churned_candidates).

    Churned = failed in-month with no successful payment on same account in-month.
    """
    path = _loan_payment_path(year, month)
    if path is None:
        return [], [], []

    if pd is not None and path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return _partition_loan_customers_df(df)

    return _partition_loan_customers_csv(path)


def _partition_loan_customers_df(df: Any) -> tuple[list[LoanCustomer], list[LoanCustomer], list[LoanCustomer]]:
    required = {"account_id", "customer_id", "status"}
    if not required.issubset(df.columns):
        return [], [], []

    status = df["status"].astype(str)
    failed_mask = status.map(_is_failed_status)
    success_mask = status.map(_is_success_status)

    amount_col = "amount" if "amount" in df.columns else None
    loan_type_col = "loan_type" if "loan_type" in df.columns else None
    reason_col = "failure_reason" if "failure_reason" in df.columns else None

    successful_accounts: set[str] = set()
    if success_mask.any():
        successful_accounts = set(df.loc[success_mask, "account_id"].astype(str))

    by_key: dict[tuple[str, str], LoanCustomer] = {}
    churned: dict[tuple[str, str], LoanCustomer] = {}
    failed_only: dict[tuple[str, str], LoanCustomer] = {}

    for idx in df.index:
        account_id = str(df.at[idx, "account_id"])
        customer_id = str(df.at[idx, "customer_id"])
        if not account_id or not customer_id or account_id == "nan":
            continue
        key = (account_id, customer_id)
        amount = parse_float(df.at[idx, amount_col]) if amount_col else random.uniform(400, 8000)
        loan_type = str(df.at[idx, loan_type_col]) if loan_type_col else "Home Loan"
        reason = str(df.at[idx, reason_col]) if reason_col and pd.notna(df.at[idx, reason_col]) else ""

        row_status = str(df.at[idx, "status"])
        if key not in by_key or amount > by_key[key].amount:
            by_key[key] = LoanCustomer(account_id, customer_id, amount, loan_type, reason)

        if _is_failed_status(row_status):
            lc = LoanCustomer(account_id, customer_id, amount, loan_type, reason)
            failed_only[key] = lc
            if account_id not in successful_accounts:
                churned[key] = lc

    return list(by_key.values()), list(churned.values()), list(failed_only.values())


def _partition_loan_customers_csv(path: Path) -> tuple[list[LoanCustomer], list[LoanCustomer], list[LoanCustomer]]:
    by_key: dict[tuple[str, str], LoanCustomer] = {}
    churned: dict[tuple[str, str], LoanCustomer] = {}
    failed_only: dict[tuple[str, str], LoanCustomer] = {}
    successful_accounts: set[str] = set()
    rows_cache: list[dict[str, str]] = []

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "account_id" not in reader.fieldnames:
            return [], [], []
        for row in reader:
            rows_cache.append(row)
            if _is_success_status(str(row.get("status", ""))):
                aid = row.get("account_id")
                if aid:
                    successful_accounts.add(str(aid))

    for row in rows_cache:
        account_id = str(row.get("account_id", "")).strip()
        customer_id = str(row.get("customer_id", "")).strip()
        if not account_id or not customer_id:
            continue
        key = (account_id, customer_id)
        amount = parse_float(row.get("amount"), random.uniform(400, 8000))
        loan_type = str(row.get("loan_type") or "Home Loan")
        reason = str(row.get("failure_reason") or "").strip()
        status = str(row.get("status", ""))

        if key not in by_key or amount > by_key[key].amount:
            by_key[key] = LoanCustomer(account_id, customer_id, amount, loan_type, reason)

        if _is_failed_status(status):
            lc = LoanCustomer(account_id, customer_id, amount, loan_type, reason)
            failed_only[key] = lc
            if account_id not in successful_accounts:
                churned[key] = lc

    return list(by_key.values()), list(churned.values()), list(failed_only.values())


def contact_channels(year: int, month: int) -> list[str]:
    channels = ["phone_call", "letter", "branch_visit"]
    if year > 2020 or (year == 2020 and month >= 4):
        channels.append("sms")
    if year >= 2021:
        channels.append("email")
    if year >= 2022:
        channels.append("app_message")
    return channels


def pick_stage() -> tuple[str, int]:
    roll = random.random()
    cumulative = 0.0
    for stage, dpd_min, dpd_max, weight in STAGES:
        cumulative += weight
        if roll <= cumulative:
            return stage, random.randint(dpd_min, dpd_max)
    stage, dpd_min, dpd_max, _ = STAGES[-1]
    return stage, random.randint(dpd_min, dpd_max)


def month_name(month: int) -> str:
    names = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return names[month]


def format_money(amount: float) -> str:
    if amount >= 1000:
        return f"{amount:,.0f}".replace(",", " ")
    return f"{amount:.0f}"


def build_note(
    *,
    stage: str,
    is_new: bool,
    is_returned: bool,
    promise_amount: float | None,
    arrangement: str | None,
    province: str,
    year: int,
) -> str:
    amount = promise_amount or random.choice([350, 500, 750, 1200, 1500, 2500])
    parts: list[str] = []
    if is_returned:
        parts.append(
            random.choice(
                [
                    "Former churned customer returned; new arrears flagged after failed debit order.",
                    "Win-back customer relapsed — prior case closed, reopened after NSF debit order.",
                ]
            )
        )
    elif is_new:
        parts.append("New case opened from failed loan repayment and churned relationship status.")

    templates = NOTE_TEMPLATES if year == 2020 else [t for t in NOTE_TEMPLATES if "COVID" not in t]
    template = random.choice(templates)
    note = template.format(
        amount=format_money(amount),
        start_day=random.randint(1, 28),
        month_name=month_name(random.randint(1, 12)),
        province=province or random.choice(["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape"]),
    )
    if year == 2020 and random.random() < 0.15:
        note = "COVID hardship relief applied. Payment holiday granted until June 2020."
    if stage == "legal" and "legal" not in note.lower():
        note += " Escalating to legal."
    if arrangement:
        note += f" Arrangement: {arrangement.replace('_', ' ')}."
    parts.append(note)
    return " ".join(parts)


def pick_customer(
    churned: list[LoanCustomer],
    failed: list[LoanCustomer],
    returned_pool: list[LoanCustomer],
    prefer_returned: bool,
) -> LoanCustomer:
    if prefer_returned and returned_pool:
        return random.choice(returned_pool)
    pools: list[list[LoanCustomer]] = []
    weights: list[float] = []
    if churned:
        pools.append(churned)
        weights.append(0.85)
    if failed:
        pools.append(failed)
        weights.append(0.15)
    if not pools:
        raise RuntimeError("No customers available")
    pool = random.choices(pools, weights=weights[: len(pools)], k=1)[0]
    return random.choice(pool)


def open_case_count(signal: MonthSignal) -> int:
    return max(15, round(signal.insuf_count * 0.3 + signal.fail_pct * 20))


def random_date_in_month(year: int, month: int, latest_day: int | None = None) -> date:
    last = monthrange(year, month)[1]
    end = min(last, latest_day) if latest_day else last
    return date(year, month, random.randint(1, max(1, end)))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_month(
    year: int,
    month: int,
    signal: MonthSignal,
    customer_master: dict[str, dict[str, str]],
    churned_ever: set[str],
    winback_eligible: set[str],
    account_master: dict[str, dict[str, str]],
) -> tuple[int, int, set[str]]:
    random.seed(2019 * 100 + year * 100 + month)

    if (year, month) < (DEFAULT_COLLECTION_START_YEAR, DEFAULT_COLLECTION_START_MONTH):
        return 0, 0, churned_ever

    _all_candidates, churned, failed = load_loan_payment_customers(year, month)
    if not churned and not failed:
        return 0, 0, churned_ever

    month_start = date(year, month, 1)
    eligible_accounts: set[str] = set()
    for account_id, account_row in account_master.items():
        opened_raw = account_row.get("opening_date")
        try:
            opened = date.fromisoformat(str(opened_raw)[:10])
        except ValueError:
            opened = date(2019, 1, 1)
        if (month_start - opened).days >= MIN_COLLECTION_ACCOUNT_AGE_DAYS:
            eligible_accounts.add(account_id)

    churned = [lc for lc in churned if lc.account_id in eligible_accounts]
    failed = [lc for lc in failed if lc.account_id in eligible_accounts]
    collection_pool = churned if churned else failed
    if not collection_pool:
        return 0, 0, churned_ever

    returned_pool = [
        LoanCustomer(
            lc.account_id,
            lc.customer_id,
            lc.amount * random.uniform(0.8, 1.2),
            lc.loan_type,
            lc.failure_reason,
        )
        for lc in collection_pool
        if lc.customer_id in winback_eligible and lc.customer_id in churned_ever
    ]

    open_cases = open_case_count(signal)
    new_cases = max(1, round(open_cases * 0.25))
    recovery_count = max(0, round(new_cases * 0.4))
    returned_slots = max(0, round(new_cases * 0.12)) if year >= 2020 else 0

    channels = contact_channels(year, month)
    cases: list[dict[str, Any]] = []
    promise_cases: list[dict[str, Any]] = []
    new_case_ids: list[str] = []

    for seq in range(1, open_cases + 1):
        is_new = seq <= new_cases
        is_returned = is_new and returned_slots > 0 and seq <= returned_slots and bool(returned_pool)
        if is_returned:
            returned_slots -= 1

        lc = pick_customer(
            churned,
            failed,
            returned_pool,
            prefer_returned=is_returned,
        )
        master_account = account_master.get(lc.account_id)
        if not master_account:
            continue
        lc.customer_id = master_account.get("customer_id") or lc.customer_id

        stage, dpd = pick_stage()
        master_row = customer_master.get(lc.customer_id, {})
        province = master_row.get("province", "")
        credit = parse_float(master_row.get("credit_score"), 650)
        arrears = round(max(200.0, lc.amount * random.uniform(1.0, 3.5) * (1.1 - min(credit, 850) / 1000)), 2)

        promise_amount: float | None = None
        promise_date: str | None = None
        if stage != "write_off" and random.random() < 0.40:
            promise_amount = round(arrears * random.uniform(0.15, 0.55), 2)
            pday = random_date_in_month(year, month)
            if pday.day < monthrange(year, month)[1]:
                pday = pday + timedelta(days=random.randint(3, 14))
            promise_date = pday.isoformat()

        arrangement: str | None = None
        if stage in ("early_collections", "late_collections") and random.random() < 0.20:
            arrangement = random.choice(ARRANGEMENT_PLANS)

        if stage == "write_off":
            status = random.choices(CASE_STATUSES, weights=[10, 55, 35])[0]
        elif promise_amount and random.random() < 0.35:
            status = "resolved"
        else:
            status = random.choices(CASE_STATUSES, weights=[55, 15, 30])[0]

        case_id = f"COLL-{year}{month:02d}-{seq:05d}"
        contact_day = random_date_in_month(year, month)
        try:
            opened = date.fromisoformat(str(master_account.get("opening_date", ""))[:10])
        except ValueError:
            opened = date(2019, 1, 1)
        minimum_contact_day = opened + timedelta(days=max(MIN_COLLECTION_ACCOUNT_AGE_DAYS, dpd))
        if contact_day < minimum_contact_day:
            continue
        case_row = {
            "case_id": case_id,
            "account_id": lc.account_id,
            "customer_id": lc.customer_id,
            "arrears_amount": arrears,
            "days_past_due": dpd,
            "collection_stage": stage,
            "last_contact_date": contact_day.isoformat(),
            "last_contact_channel": random.choice(channels),
            "promise_to_pay_amount": "" if promise_amount is None else f"{promise_amount:.2f}",
            "promise_to_pay_date": promise_date or "",
            "arrangement_plan": arrangement or "",
            "status": status,
            "assigned_collector": random.choice(COLLECTORS),
            "notes": build_note(
                stage=stage,
                is_new=is_new,
                is_returned=is_returned,
                promise_amount=promise_amount,
                arrangement=arrangement,
                province=province,
                year=year,
            ),
        }
        cases.append(case_row)
        if is_new:
            new_case_ids.append(case_id)
            churned_ever.add(lc.customer_id)
        if promise_amount is not None:
            promise_cases.append(case_row)

    kept_promises = random.sample(
        promise_cases,
        k=min(len(promise_cases), max(0, round(len(promise_cases) * 0.60))),
    )
    kept_ids = {c["case_id"] for c in kept_promises}

    recovery_rows: list[dict[str, Any]] = []
    payable_cases = [c for c in cases if c["case_id"] in new_case_ids] or cases
    random.shuffle(payable_cases)

    for pay_seq in range(1, recovery_count + 1):
        if not payable_cases:
            break
        case = payable_cases[(pay_seq - 1) % len(payable_cases)]
        arrears = parse_float(case["arrears_amount"])
        promise_amt = parse_float(case["promise_to_pay_amount"], 0.0)
        has_promise = bool(case["promise_to_pay_date"])
        was_kept = has_promise and case["case_id"] in kept_ids

        if was_kept and promise_amt > 0:
            amount = round(promise_amt * random.uniform(0.95, 1.05), 2)
        else:
            amount = round(arrears * random.uniform(0.10, 0.65), 2)

        pay_date = random_date_in_month(year, month)
        if has_promise and case["promise_to_pay_date"]:
            try:
                promised = date.fromisoformat(case["promise_to_pay_date"])
                if was_kept:
                    pay_date = promised
                else:
                    pay_date = promised + timedelta(days=random.randint(5, 20))
            except ValueError:
                pass

        arrears_after = round(max(0.0, arrears - amount), 2)
        recovery_rows.append(
            {
                "payment_id": f"REC-{year}{month:02d}-{pay_seq:05d}",
                "case_id": case["case_id"],
                "account_id": case["account_id"],
                "customer_id": case["customer_id"],
                "payment_date": pay_date.isoformat(),
                "amount": f"{amount:.2f}",
                "payment_method": random.choice(PAYMENT_METHODS),
                "was_promise_kept": str(was_kept),
                "arrears_after_payment": f"{arrears_after:.2f}",
            }
        )

    out_dir = BANKING_DIR / str(year) / f"{month:02d}" / OUTPUT_SUBDIR
    write_csv(out_dir / "collections_cases.csv", cases, CASE_FIELDS)
    write_csv(out_dir / "recovery_payments.csv", recovery_rows, RECOVERY_FIELDS)
    return len(cases), len(recovery_rows), churned_ever


def mark_winback_eligible(churned_ever: set[str], winback_eligible: set[str]) -> None:
    """A fraction of previously churned customers may return in later years."""
    for cid in churned_ever:
        if cid not in winback_eligible and random.random() < 0.08:
            winback_eligible.add(cid)


def iter_year_months(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic collections and recoveries data.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_COLLECTION_START_YEAR)
    parser.add_argument("--start-month", type=int, default=DEFAULT_COLLECTION_START_MONTH)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--end-month", type=int, default=12)
    args = parser.parse_args()

    signals = load_monthly_signals()
    customer_master = load_customer_master()
    account_master = load_global_account_master()
    churned_ever: set[str] = set()
    winback_eligible: set[str] = set()

    total_cases = 0
    total_recoveries = 0
    months_written = 0
    years_seen: set[int] = set()

    for year, month in iter_year_months(
        args.start_year, args.start_month, args.end_year, args.end_month
    ):
        if year not in years_seen:
            years_seen.add(year)
            random.seed(2019 * 100 + year)
            mark_winback_eligible(churned_ever, winback_eligible)

        signal = signals.get((year, month), MonthSignal(year=year, month=month))
        cases, recoveries, churned_ever = generate_month(
            year,
            month,
            signal,
            customer_master,
            churned_ever,
            winback_eligible,
            account_master,
        )
        if cases == 0:
            continue
        months_written += 1
        total_cases += cases
        total_recoveries += recoveries

    print(f"Collections data written under {BANKING_DIR}\\<year>\\<month>\\{OUTPUT_SUBDIR}")
    print(f"  Months: {months_written}")
    print(f"  Cases: {total_cases}")
    print(f"  Recovery payments: {total_recoveries}")


if __name__ == "__main__":
    main()
