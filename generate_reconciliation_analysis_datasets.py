from __future__ import annotations

import argparse
import json
import random
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"
OUT_DIR = DATA_DIR / "reconciliation_analysis"
CORPUS_TEAMS_FILE = BASE_DIR / "commons" / "corpus_teams.json"
DEFAULT_BANK_NAME = "Keystone Retail Bank"
NON_TRANSACTIONAL_TYPES = ["bank_charge", "service_fee", "interest_income", "withholding_tax", "cash_deposit_fee"]
STATEMENT_SOURCE_FORMATS = ["ISO20022_CAMT053_XML", "CSV", "TXT"]


def load_company_bank_name() -> str:
    try:
        payload = json.loads(CORPUS_TEAMS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_BANK_NAME
    bank_name = str(payload.get("bank_name", "")).strip()
    return bank_name or DEFAULT_BANK_NAME


COMPANY_BANK_NAME = load_company_bank_name()


def maybe_remove_description_underscores(text: Any, rng: random.Random, probability: float = 0.65) -> str:
    description = " ".join(str(text or "").strip().split())
    if "_" in description and rng.random() < probability:
        description = description.replace("_", " ")
        description = " ".join(description.split())
    return description


def month_end(year: int, month: int) -> datetime:
    return datetime(year, month, monthrange(year, month)[1], 23, 59, 59)


def month_add(year: int, month: int, months: int) -> tuple[int, int]:
    idx = (year * 12 + month - 1) + months
    return idx // 12, idx % 12 + 1


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def table_file(record_type: str, year: int, month: int) -> Path | None:
    base = DATA_DIR / str(year) / f"{month:02d}" / f"{record_type}_{year}_{month:02d}"
    if base.with_suffix(".parquet").exists():
        return base.with_suffix(".parquet")
    if base.with_suffix(".csv").exists():
        return base.with_suffix(".csv")
    return None


def load_accounts_customers(year: int, month: int) -> pd.DataFrame:
    account_file = table_file("accounts", year, month)
    customer_file = table_file("customers", year, month)
    if account_file is None or customer_file is None:
        raise FileNotFoundError(f"Missing account/customer files for {year}-{month:02d}")
    accounts = read_table(account_file)
    customers = read_table(customer_file)
    if "customer_id" not in customers.columns and "CustomerID" in customers.columns:
        customers = customers.rename(columns={"CustomerID": "customer_id"})
    merged = accounts.merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
    return merged.drop_duplicates(subset=["account_id"]).reset_index(drop=True)


def stream_transactions(year: int, start_month: int, months: int, account_ids: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for offset in range(months):
        y, m = month_add(year, start_month, offset)
        month_dir = DATA_DIR / str(y) / f"{m:02d}"
        parquet_path = month_dir / f"transactions_{y}_{m:02d}.parquet"
        csv_path = month_dir / f"transactions_{y}_{m:02d}.csv"
        jsonl_path = month_dir / "transactions.jsonl"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df["account_id"] = df["account_id"].astype(str)
            filtered = df[df["account_id"].isin(account_ids)].copy()
            if not filtered.empty:
                frames.append(filtered)
        elif csv_path.exists():
            for chunk in pd.read_csv(csv_path, chunksize=50_000):
                chunk["account_id"] = chunk["account_id"].astype(str)
                filtered = chunk[chunk["account_id"].isin(account_ids)].copy()
                if not filtered.empty:
                    frames.append(filtered)
        elif jsonl_path.exists():
            for chunk in pd.read_json(jsonl_path, lines=True, chunksize=50_000):
                chunk["account_id"] = chunk["account_id"].astype(str)
                filtered = chunk[chunk["account_id"].isin(account_ids)].copy()
                if not filtered.empty:
                    frames.append(filtered)
    if not frames:
        return pd.DataFrame()
    tx = pd.concat(frames, ignore_index=True)
    tx["transaction_timestamp"] = pd.to_datetime(tx["transaction_timestamp"], errors="coerce")
    return tx.sort_values("transaction_timestamp").reset_index(drop=True)


def load_period_transactions(year: int, start_month: int, months: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for offset in range(months):
        y, m = month_add(year, start_month, offset)
        month_dir = DATA_DIR / str(y) / f"{m:02d}"
        parquet_path = month_dir / f"transactions_{y}_{m:02d}.parquet"
        csv_path = month_dir / f"transactions_{y}_{m:02d}.csv"
        jsonl_path = month_dir / "transactions.jsonl"
        if parquet_path.exists():
            frames.append(pd.read_parquet(parquet_path))
        elif csv_path.exists():
            frames.append(pd.read_csv(csv_path))
        elif jsonl_path.exists():
            frames.append(pd.read_json(jsonl_path, lines=True))
    if not frames:
        return pd.DataFrame()
    tx = pd.concat(frames, ignore_index=True)
    tx["account_id"] = tx["account_id"].astype(str)
    tx["transaction_timestamp"] = pd.to_datetime(tx["transaction_timestamp"], errors="coerce")
    return tx.sort_values("transaction_timestamp").reset_index(drop=True)


def signed_amount(row: pd.Series) -> float:
    amount = float(pd.to_numeric(row.get("amount", 0), errors="coerce") or 0)
    return -abs(amount) if str(row.get("debit_credit", "debit")).lower() == "debit" else abs(amount)


def account_display_name(row: pd.Series) -> str:
    for first, last in [("first_name", "last_name"), ("name", "surname"), ("customer_name", "")]:
        if first in row and pd.notna(row.get(first)):
            return f"{row.get(first, '')} {row.get(last, '')}".strip()
    return str(row.get("customer_id", "Customer"))


def fmt_money(value: float) -> str:
    return f"R {value:,.2f}"


def statement_pdf(path: Path, account: pd.Series, rows: list[dict[str, Any]], period_start: datetime, period_end: datetime, opening: float, closing: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("StatementTitle", parent=styles["Title"], fontSize=16, leading=20, textColor=colors.HexColor("#1F2937"))
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=10)
    normal = ParagraphStyle("NormalTight", parent=styles["Normal"], fontSize=9, leading=11)

    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.2 * cm, leftMargin=1.2 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    story: list[Any] = []
    bank = COMPANY_BANK_NAME
    story.append(Paragraph(f"{bank} Bank Statement", title))
    story.append(Paragraph(f"Period: {period_start:%d %b %Y} to {period_end:%d %b %Y}", normal))
    story.append(Paragraph(f"Customer: {account_display_name(account)} | Account: {account.get('account_number', account.get('account_id'))}", normal))
    story.append(Paragraph(f"Opening balance: {fmt_money(opening)} | Closing balance: {fmt_money(closing)}", normal))
    story.append(Spacer(1, 0.35 * cm))

    table_rows = [["Date", "Description", "Reference", "Debit", "Credit", "Balance"]]
    balance = opening
    for row in rows[:80]:
        amount = float(row["statement_amount"])
        balance += amount
        table_rows.append(
            [
                pd.to_datetime(row["statement_date"]).strftime("%d %b"),
                str(row["statement_description"])[:42],
                str(row["bank_reference"])[:18],
                fmt_money(abs(amount)) if amount < 0 else "",
                fmt_money(amount) if amount > 0 else "",
                fmt_money(balance),
            ]
        )
    if len(rows) > 80:
        table_rows.append(["", f"{len(rows) - 80} additional transactions omitted from PDF view", "", "", "", ""])

    table = Table(table_rows, colWidths=[1.6 * cm, 6.4 * cm, 3.0 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Synthetic statement generated for reconciliation and analytics practice data.", small))
    doc.build(story)


def statement_source_format(account: pd.Series, rng: random.Random) -> str:
    return rng.choices(STATEMENT_SOURCE_FORMATS, weights=[0.30, 0.50, 0.20], k=1)[0]


def statement_source_suffix(source_format: str) -> str:
    return {
        "ISO20022_CAMT053_XML": ".xml",
        "CSV": ".csv",
        "TXT": ".txt",
    }.get(source_format, ".csv")


def statement_timeframe_months(account: pd.Series, rng: random.Random) -> int:
    frequency = str(account.get("statement_frequency", "")).lower()
    if frequency == "quarterly":
        return 3
    if frequency == "annually":
        return 6
    if frequency == "monthly":
        return rng.choice([3, 6])
    return rng.choice([3, 6])


def lag_days_for_transaction(tx_time: datetime, lag_profile: dict[str, int]) -> int:
    return int(lag_profile.get(tx_time.strftime("%Y-%m-%d"), 0))


def build_lag_profile(period_start: datetime, period_end: datetime, rng: random.Random) -> dict[str, int]:
    days = max(1, (period_end.date() - period_start.date()).days + 1)
    lag_day_count = max(1, int(days * rng.uniform(0.015, 0.035)))
    profile: dict[str, int] = {}
    for _ in range(lag_day_count):
        lag_date = period_start + timedelta(days=rng.randint(0, days - 1))
        profile[lag_date.strftime("%Y-%m-%d")] = rng.choices([1, 2, 3, 4], weights=[0.55, 0.28, 0.13, 0.04], k=1)[0]
    return profile


def write_statement_source_file(
    path: Path,
    source_format: str,
    account: pd.Series,
    rows: list[dict[str, Any]],
    period_start: datetime,
    period_end: datetime,
    opening: float,
    closing: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if source_format == "ISO20022_CAMT053_XML":
        path.write_text(statement_iso20022_xml(account, rows, period_start, period_end, opening, closing), encoding="utf-8")
        return
    if source_format == "TXT":
        path.write_text(statement_txt(account, rows, period_start, period_end, opening, closing), encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    keep_cols = [
        "statement_transaction_id",
        "account_id",
        "statement_date",
        "posted_date",
        "statement_description",
        "bank_reference",
        "statement_amount",
        "entry_type",
        "reconciliation_required",
        "reconciliation_reason",
        "source_system_lag_days",
        "expected_settlement_date",
        "source_transaction_id",
    ]
    df[keep_cols].to_csv(path, index=False)


def statement_iso20022_xml(account: pd.Series, rows: list[dict[str, Any]], period_start: datetime, period_end: datetime, opening: float, closing: float) -> str:
    account_id = escape(str(account.get("account_id", "")))
    account_number = escape(str(account.get("account_number", account_id)))
    bank_name = escape(COMPANY_BANK_NAME)
    statement_id = f"CAMT-{account_id}-{period_start:%Y%m}-{period_end:%Y%m}"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<Document xmlns="urn:iso:std:iso:20022:tech:xsd:camt.053.001.08">',
        "  <BkToCstmrStmt>",
        "    <GrpHdr>",
        f"      <MsgId>{statement_id}</MsgId>",
        f"      <CreDtTm>{datetime.now().isoformat(timespec='seconds')}</CreDtTm>",
        "    </GrpHdr>",
        "    <Stmt>",
        f"      <Id>{statement_id}</Id>",
        f"      <ElctrncSeqNb>{period_end:%Y%m}</ElctrncSeqNb>",
        f"      <CreDtTm>{datetime.now().isoformat(timespec='seconds')}</CreDtTm>",
        "      <Acct>",
        f"        <Id><Othr><Id>{account_number}</Id></Othr></Id>",
        f"        <Nm>{account_id}</Nm>",
        "        <Ccy>ZAR</Ccy>",
        "      </Acct>",
        f"      <Svcr><FinInstnId><Nm>{bank_name}</Nm></FinInstnId></Svcr>",
        f"      <FrToDt><FrDtTm>{period_start.isoformat()}</FrDtTm><ToDtTm>{period_end.isoformat()}</ToDtTm></FrToDt>",
        f"      <Bal><Tp><CdOrPrtry><Cd>OPBD</Cd></CdOrPrtry></Tp><Amt Ccy=\"ZAR\">{abs(opening):.2f}</Amt><CdtDbtInd>{'CRDT' if opening >= 0 else 'DBIT'}</CdtDbtInd><Dt><Dt>{period_start.date()}</Dt></Dt></Bal>",
        f"      <Bal><Tp><CdOrPrtry><Cd>CLBD</Cd></CdOrPrtry></Tp><Amt Ccy=\"ZAR\">{abs(closing):.2f}</Amt><CdtDbtInd>{'CRDT' if closing >= 0 else 'DBIT'}</CdtDbtInd><Dt><Dt>{period_end.date()}</Dt></Dt></Bal>",
    ]
    for row in rows:
        amount = float(row["statement_amount"])
        entry_type = str(row.get("entry_type", "transactional"))
        bank_code = iso_bank_transaction_code(entry_type, amount)
        date_text = pd.to_datetime(row["statement_date"]).date().isoformat()
        reference = escape(str(row.get("bank_reference", "")))
        description = escape(str(row.get("statement_description", "")))
        lines.extend(
            [
                "      <Ntry>",
                f"        <Amt Ccy=\"ZAR\">{abs(amount):.2f}</Amt>",
                f"        <CdtDbtInd>{'CRDT' if amount >= 0 else 'DBIT'}</CdtDbtInd>",
                "        <Sts><Cd>BOOK</Cd></Sts>",
                f"        <BookgDt><Dt>{date_text}</Dt></BookgDt>",
                f"        <ValDt><Dt>{date_text}</Dt></ValDt>",
                f"        <AcctSvcrRef>{reference}</AcctSvcrRef>",
                f"        <BkTxCd><Domn><Cd>{bank_code[0]}</Cd><Fmly><Cd>{bank_code[1]}</Cd><SubFmlyCd>{bank_code[2]}</SubFmlyCd></Fmly></Domn></BkTxCd>",
                "        <NtryDtls><TxDtls>",
                f"          <Refs><EndToEndId>{reference}</EndToEndId></Refs>",
                f"          <RmtInf><Ustrd>{description}</Ustrd></RmtInf>",
                "        </TxDtls></NtryDtls>",
                "      </Ntry>",
            ]
        )
    lines.extend(["    </Stmt>", "  </BkToCstmrStmt>", "</Document>"])
    return "\n".join(lines) + "\n"


def iso_bank_transaction_code(entry_type: str, amount: float) -> tuple[str, str, str]:
    if entry_type in {"bank_charge", "service_fee", "cash_deposit_fee"}:
        return ("PMNT", "RCDT", "CHRG")
    if entry_type == "interest_income":
        return ("CAMT", "MCOP", "INTR")
    if entry_type == "withholding_tax":
        return ("CAMT", "MCOP", "TAXR")
    if entry_type == "bank_only_transaction":
        return ("PMNT", "ICDT" if amount >= 0 else "ICDT", "ESCT")
    return ("PMNT", "RCDT" if amount < 0 else "ICDT", "DMCT" if amount < 0 else "CRDT")


def statement_txt(account: pd.Series, rows: list[dict[str, Any]], period_start: datetime, period_end: datetime, opening: float, closing: float) -> str:
    header = [
        f"BANK STATEMENT|{COMPANY_BANK_NAME}",
        f"ACCOUNT|{account.get('account_number', account.get('account_id'))}",
        f"CUSTOMER|{account_display_name(account)}",
        f"PERIOD|{period_start:%Y-%m-%d}|{period_end:%Y-%m-%d}",
        f"OPENING_BALANCE|{opening:.2f}",
        f"CLOSING_BALANCE|{closing:.2f}",
        "DATE|REFERENCE|DESCRIPTION|AMOUNT|ENTRY_TYPE|SOURCE_TRANSACTION_ID",
    ]
    body = []
    for row in rows:
        body.append(
            "|".join(
                [
                    pd.to_datetime(row["statement_date"]).strftime("%Y-%m-%d"),
                    str(row.get("bank_reference", "")),
                    str(row.get("statement_description", "")),
                    f"{float(row['statement_amount']):.2f}",
                    str(row.get("entry_type", "")),
                    str(row.get("source_transaction_id") or ""),
                ]
            )
        )
    return "\n".join(header + body) + "\n"


def choose_accounts(accounts: pd.DataFrame, target_accounts: int | None, seed: int) -> pd.DataFrame:
    eligible = accounts[~accounts.get("account_status", "").astype(str).str.lower().isin(["closed", "frozen"])].copy()
    if eligible.empty:
        eligible = accounts.copy()
    if target_accounts is not None and len(eligible) > target_accounts:
        eligible = eligible.sample(n=target_accounts, random_state=seed)
    return eligible.reset_index(drop=True)


def build_for_account(account: pd.Series, ledger: pd.DataFrame, period_start: datetime, period_end: datetime, timeframe_months: int, run_id: str, rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    account_id = str(account["account_id"])
    acc_tx = ledger[(ledger["account_id"].astype(str) == account_id) & (ledger["transaction_timestamp"].between(period_start, period_end))].copy()
    opening = float(pd.to_numeric(acc_tx["account_balance_before"], errors="coerce").dropna().iloc[0]) if not acc_tx.empty and "account_balance_before" in acc_tx else rng.uniform(800, 18000)
    lag_profile = build_lag_profile(period_start, period_end, rng)

    statement_rows: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    non_tx_rows: list[dict[str, Any]] = []

    for _, tx in acc_tx.iterrows():
        if str(tx.get("status", "completed")).lower() == "failed":
            continue
        ledger_amount = signed_amount(tx)
        category = "matched"
        bank_amount = ledger_amount
        bank_date = pd.to_datetime(tx["transaction_timestamp"])
        include_bank = True
        system_lag_days = lag_days_for_transaction(bank_date, lag_profile)
        if system_lag_days:
            bank_date = bank_date + timedelta(days=system_lag_days)

        p = rng.random()
        if bank_date > period_end:
            include_bank = False
            category = "outstanding_at_statement_cutoff"
        elif system_lag_days:
            category = "source_system_lag"
        elif p < 0.005:
            include_bank = False
            category = "missing_in_bank_statement"
        elif p < 0.009:
            category = "amount_mismatch"
            bank_amount = round(ledger_amount * rng.uniform(0.92, 1.08), 2)
        elif p < 0.018:
            category = "timing_difference"
            bank_date = min(period_end, bank_date + timedelta(days=rng.choice([1, 2, 3])))

        statement_id = None
        if include_bank:
            statement_id = f"BST-{account_id}-{len(statement_rows) + 1:06d}"
            statement_rows.append(
                {
                    "statement_transaction_id": statement_id,
                    "account_id": account_id,
                    "customer_id": account.get("customer_id"),
                    "bank_name": COMPANY_BANK_NAME,
                    "statement_date": bank_date,
                    "posted_date": bank_date.date().isoformat(),
                    "statement_description": maybe_remove_description_underscores(tx.get("description", "Bank card transaction"), rng),
                    "bank_reference": str(tx.get("rrn", tx.get("transaction_id"))),
                    "statement_amount": bank_amount,
                    "source_transaction_id": tx.get("transaction_id"),
                    "entry_type": "transactional",
                    "reconciliation_required": category != "matched",
                    "reconciliation_reason": None if category == "matched" else category,
                    "source_system_lag_days": system_lag_days,
                    "expected_settlement_date": bank_date.date().isoformat() if category in ["source_system_lag", "outstanding_at_statement_cutoff", "timing_difference"] else None,
                }
            )
            if rng.random() < 0.003:
                dup = statement_rows[-1].copy()
                dup["statement_transaction_id"] = f"BST-{account_id}-{len(statement_rows) + 1:06d}"
                dup["bank_reference"] = f"{dup['bank_reference']}-DUP"
                statement_rows.append(dup)
                category = "duplicate_bank_entry"

        recon_rows.append(
            {
                "reconciliation_run_id": run_id,
                "account_id": account_id,
                "customer_id": account.get("customer_id"),
                "ledger_transaction_id": tx.get("transaction_id"),
                "statement_transaction_id": statement_id,
                "ledger_amount": ledger_amount,
                "statement_amount": bank_amount if include_bank else None,
                "ledger_date": tx.get("transaction_timestamp"),
                "statement_date": bank_date if include_bank else None,
                "match_status": "matched" if category == "matched" else "exception",
                "exception_category": None if category == "matched" else category,
                "root_cause_tag": None if category == "matched" else root_cause_for(category),
                "source_system_lag_days": system_lag_days,
                "expected_settlement_date": bank_date.date().isoformat() if category in ["source_system_lag", "outstanding_at_statement_cutoff", "timing_difference"] else None,
                "confidence_score": round(rng.uniform(0.96, 1.0), 3) if category == "matched" else round(rng.uniform(0.45, 0.82), 3),
            }
        )
        if category != "matched":
            exception_rows.append(recon_rows[-1].copy())

    for month_offset in range(timeframe_months):
        y, m = month_add(period_start.year, period_start.month, month_offset)
        charge_date = datetime(y, m, min(25, monthrange(y, m)[1]), rng.randint(7, 17), rng.randint(0, 59), 0)
        for entry_type in rng.sample(NON_TRANSACTIONAL_TYPES, k=rng.choice([1, 2, 2, 3])):
            amount = non_transactional_amount(entry_type, rng, account)
            statement_id = f"BST-{account_id}-{len(statement_rows) + 1:06d}"
            row = {
                "statement_transaction_id": statement_id,
                "account_id": account_id,
                "customer_id": account.get("customer_id"),
                "bank_name": COMPANY_BANK_NAME,
                "statement_date": charge_date,
                "posted_date": charge_date.date().isoformat(),
                "statement_description": maybe_remove_description_underscores(entry_type.title(), rng),
                "bank_reference": f"FEE-{y}{m:02d}-{rng.randint(10000,99999)}",
                "statement_amount": amount,
                "source_transaction_id": None,
                "entry_type": entry_type,
                "reconciliation_required": True,
                "reconciliation_reason": "bank_generated_non_transactional_entry",
                "source_system_lag_days": 0,
                "expected_settlement_date": None,
            }
            statement_rows.append(row)
            non_tx_rows.append(
                {
                    **row,
                    "reconciliation_run_id": run_id,
                    "exception_category": "non_transactional_bank_entry",
                    "gl_classification": gl_classification(entry_type),
                    "root_cause_tag": "bank_generated_fee_or_interest",
                }
            )
            exception_rows.append(
                {
                    "reconciliation_run_id": run_id,
                    "account_id": account_id,
                    "customer_id": account.get("customer_id"),
                    "ledger_transaction_id": None,
                    "statement_transaction_id": statement_id,
                    "ledger_amount": None,
                    "statement_amount": amount,
                    "ledger_date": None,
                    "statement_date": charge_date,
                    "match_status": "exception",
                    "exception_category": "non_transactional_bank_entry",
                    "root_cause_tag": "bank_generated_fee_or_interest",
                    "confidence_score": round(rng.uniform(0.88, 0.98), 3),
                }
            )

    if rng.random() < 0.12:
        bank_only_date = period_start + timedelta(days=rng.randint(3, max(4, (period_end - period_start).days - 2)))
        statement_id = f"BST-{account_id}-{len(statement_rows) + 1:06d}"
        row = {
            "statement_transaction_id": statement_id,
            "account_id": account_id,
            "customer_id": account.get("customer_id"),
            "bank_name": COMPANY_BANK_NAME,
            "statement_date": bank_only_date,
            "posted_date": bank_only_date.date().isoformat(),
            "statement_description": rng.choice(["Manual correction", "Unallocated deposit", "External EFT credit"]),
            "bank_reference": f"UNALLOC-{rng.randint(100000,999999)}",
            "statement_amount": round(rng.uniform(120, 3800), 2),
            "source_transaction_id": None,
            "entry_type": "bank_only_transaction",
            "reconciliation_required": True,
            "reconciliation_reason": "missing_in_ledger",
            "source_system_lag_days": 0,
            "expected_settlement_date": None,
        }
        statement_rows.append(row)
        exception_rows.append(
            {
                "reconciliation_run_id": run_id,
                "account_id": account_id,
                "customer_id": account.get("customer_id"),
                "ledger_transaction_id": None,
                "statement_transaction_id": statement_id,
                "ledger_amount": None,
                "statement_amount": row["statement_amount"],
                "ledger_date": None,
                "statement_date": bank_only_date,
                "match_status": "exception",
                "exception_category": "missing_in_ledger",
                "root_cause_tag": "ledger_feed_delay_or_unallocated_bank_item",
                "confidence_score": round(rng.uniform(0.50, 0.78), 3),
            }
        )

    statement_rows = sorted(statement_rows, key=lambda x: pd.to_datetime(x["statement_date"]))
    closing = opening + sum(float(r["statement_amount"]) for r in statement_rows)
    return {
        "statement_rows": statement_rows,
        "recon_rows": recon_rows,
        "exception_rows": exception_rows,
        "non_tx_rows": non_tx_rows,
        "manifest": [
            {
                "reconciliation_run_id": run_id,
                "account_id": account_id,
                "customer_id": account.get("customer_id"),
                "bank_name": COMPANY_BANK_NAME,
                "account_number": account.get("account_number"),
                "statement_timeframe_months": timeframe_months,
                "period_start": period_start,
                "period_end": period_end,
                "opening_balance": round(opening, 2),
                "closing_balance": round(closing, 2),
                "statement_transaction_count": len(statement_rows),
                "ledger_transaction_count": int(len(acc_tx)),
            }
        ],
    }


def root_cause_for(category: str) -> str:
    return {
        "missing_in_bank_statement": "ledger_item_not_yet_settled",
        "outstanding_at_statement_cutoff": "bank_processing_after_statement_cutoff",
        "source_system_lag": "source_system_processing_delay",
        "amount_mismatch": "fee_or_fx_value_difference",
        "timing_difference": "posting_date_lag",
        "duplicate_bank_entry": "bank_statement_duplicate",
    }.get(category, "requires_finance_review")


def gl_classification(entry_type: str) -> str:
    return {
        "bank_charge": "GL_BANK_CHARGES",
        "service_fee": "GL_SERVICE_FEES",
        "interest_income": "GL_INTEREST_INCOME",
        "withholding_tax": "GL_WITHHOLDING_TAX",
        "cash_deposit_fee": "GL_CASH_HANDLING_FEES",
    }.get(entry_type, "GL_CLEARING_ACCOUNT")


def non_transactional_amount(entry_type: str, rng: random.Random, account: pd.Series | None = None) -> float:
    if entry_type == "interest_income":
        base = float(pd.to_numeric(account.get("interest_rate", 0.01), errors="coerce") or 0.01) if account is not None else 0.01
        return round(rng.uniform(20, 850) * max(0.002, base), 2)
    if entry_type == "withholding_tax":
        return -round(rng.uniform(1, 65), 2)
    return -round(rng.uniform(4, 160), 2)


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(path.with_suffix(".csv"), index=False)
        return
    df.to_parquet(path.with_suffix(".parquet"), index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)


def dataframe_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.to_dict(orient="records")


def build_realism_expansion(
    accounts: pd.DataFrame,
    ledger: pd.DataFrame,
    recon: pd.DataFrame,
    statement_tx: pd.DataFrame,
    non_tx: pd.DataFrame,
    exceptions: pd.DataFrame,
    year: int,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    chart_of_accounts = pd.DataFrame(
        [
            {"gl_account_id": "GL1000", "gl_account_name": "Cash Settlement", "gl_type": "Asset"},
            {"gl_account_id": "GL2000", "gl_account_name": "Customer Deposits", "gl_type": "Liability"},
            {"gl_account_id": "GL4100", "gl_account_name": "Fee Income", "gl_type": "Revenue"},
            {"gl_account_id": "GL4200", "gl_account_name": "Interest Income", "gl_type": "Revenue"},
            {"gl_account_id": "GL5100", "gl_account_name": "Interest Expense", "gl_type": "Expense"},
            {"gl_account_id": "GL6100", "gl_account_name": "Tax Payable", "gl_type": "Liability"},
            {"gl_account_id": "GL1999", "gl_account_name": "Suspense / Reconciliation", "gl_type": "Asset"},
        ]
    )

    journal_rows: list[dict[str, Any]] = []
    if not recon.empty:
        recon_copy = recon.copy()
        recon_copy["event_ts"] = pd.to_datetime(recon_copy.get("statement_date", recon_copy.get("ledger_date")), errors="coerce")
        for idx, row in recon_copy.iterrows():
            amount = float(pd.to_numeric(row.get("statement_amount", row.get("ledger_amount", 0.0)), errors="coerce") or 0.0)
            if amount == 0:
                continue
            abs_amt = round(abs(amount), 2)
            journal_id = f"JRN-{year}-{idx + 1:08d}"
            event_ts = row.get("event_ts")
            if pd.isna(event_ts):
                event_ts = datetime(year, 1, 1)
            if amount > 0:
                dr_account, cr_account = ("GL1000", "GL2000")
            else:
                dr_account, cr_account = ("GL2000", "GL1000")

            journal_rows.append(
                {
                    "journal_id": journal_id,
                    "line_no": 1,
                    "gl_account_id": dr_account,
                    "dr_cr": "DR",
                    "amount": abs_amt,
                    "currency": "ZAR",
                    "event_type": "customer_transaction",
                    "source_table": "reconciliation_results",
                    "source_id": row.get("ledger_transaction_id"),
                    "event_timestamp": event_ts,
                    "posting_date": pd.to_datetime(event_ts).date().isoformat(),
                    "posting_status": "posted",
                }
            )
            journal_rows.append(
                {
                    "journal_id": journal_id,
                    "line_no": 2,
                    "gl_account_id": cr_account,
                    "dr_cr": "CR",
                    "amount": abs_amt,
                    "currency": "ZAR",
                    "event_type": "customer_transaction",
                    "source_table": "reconciliation_results",
                    "source_id": row.get("ledger_transaction_id"),
                    "event_timestamp": event_ts,
                    "posting_date": pd.to_datetime(event_ts).date().isoformat(),
                    "posting_status": "posted",
                }
            )

    if not non_tx.empty:
        non_tx_copy = non_tx.copy()
        non_tx_copy["event_ts"] = pd.to_datetime(non_tx_copy.get("statement_date"), errors="coerce")
        for idx, row in non_tx_copy.iterrows():
            amount = float(pd.to_numeric(row.get("statement_amount", 0.0), errors="coerce") or 0.0)
            if amount == 0:
                continue
            abs_amt = round(abs(amount), 2)
            journal_id = f"JRN-NTX-{year}-{idx + 1:07d}"
            entry_type = str(row.get("entry_type", ""))
            if entry_type == "interest_income":
                dr_account, cr_account = ("GL5100", "GL2000")
            elif entry_type == "withholding_tax":
                dr_account, cr_account = ("GL2000", "GL6100")
            else:
                dr_account, cr_account = ("GL2000", "GL4100")
            event_ts = row.get("event_ts")
            if pd.isna(event_ts):
                event_ts = datetime(year, 1, 1)

            journal_rows.append(
                {
                    "journal_id": journal_id,
                    "line_no": 1,
                    "gl_account_id": dr_account,
                    "dr_cr": "DR",
                    "amount": abs_amt,
                    "currency": "ZAR",
                    "event_type": "non_transactional_statement_entry",
                    "source_table": "non_transactional_statement_entries",
                    "source_id": row.get("statement_transaction_id"),
                    "event_timestamp": event_ts,
                    "posting_date": pd.to_datetime(event_ts).date().isoformat(),
                    "posting_status": "posted",
                }
            )
            journal_rows.append(
                {
                    "journal_id": journal_id,
                    "line_no": 2,
                    "gl_account_id": cr_account,
                    "dr_cr": "CR",
                    "amount": abs_amt,
                    "currency": "ZAR",
                    "event_type": "non_transactional_statement_entry",
                    "source_table": "non_transactional_statement_entries",
                    "source_id": row.get("statement_transaction_id"),
                    "event_timestamp": event_ts,
                    "posting_date": pd.to_datetime(event_ts).date().isoformat(),
                    "posting_status": "posted",
                }
            )

    payment_events: list[dict[str, Any]] = []
    if not recon.empty:
        rc = recon.copy()
        rc["ledger_ts"] = pd.to_datetime(rc.get("ledger_date"), errors="coerce")
        rc["statement_ts"] = pd.to_datetime(rc.get("statement_date"), errors="coerce")
        for idx, row in rc.iterrows():
            base_ts = row.get("ledger_ts")
            if pd.isna(base_ts):
                base_ts = row.get("statement_ts")
            if pd.isna(base_ts):
                base_ts = datetime(year, 1, 1, 9, 0, 0)

            payment_id = str(row.get("ledger_transaction_id") or row.get("statement_transaction_id") or f"PAY-{idx + 1:08d}")
            match_status = str(row.get("match_status", "matched"))
            category = str(row.get("exception_category", ""))
            is_exception = match_status == "exception"

            timeline = [
                ("initiated", base_ts),
                ("authorized", base_ts + timedelta(seconds=rng.randint(1, 45))),
            ]
            if is_exception and category in {"missing_in_bank_statement", "outstanding_at_statement_cutoff"}:
                timeline.append(("pending_settlement", base_ts + timedelta(hours=rng.randint(2, 20))))
                timeline.append(("reconciliation_exception", base_ts + timedelta(days=rng.randint(1, 4))))
            elif is_exception:
                timeline.append(("posted", base_ts + timedelta(minutes=rng.randint(2, 90))))
                timeline.append(("reconciliation_exception", base_ts + timedelta(hours=rng.randint(2, 48))))
            else:
                timeline.append(("posted", base_ts + timedelta(minutes=rng.randint(1, 60))))
                timeline.append(("settled", base_ts + timedelta(hours=rng.randint(1, 36))))

            for seq, (status, ts_val) in enumerate(timeline, start=1):
                payment_events.append(
                    {
                        "payment_id": payment_id,
                        "event_sequence": seq,
                        "event_status": status,
                        "event_timestamp": ts_val,
                        "account_id": row.get("account_id"),
                        "customer_id": row.get("customer_id"),
                        "exception_category": None if status != "reconciliation_exception" else category,
                        "source": "reconciliation_results",
                    }
                )

    account_dim_rows: list[dict[str, Any]] = []
    customer_dim_rows: list[dict[str, Any]] = []
    if not accounts.empty:
        for _, row in accounts.iterrows():
            account_id = str(row.get("account_id", ""))
            customer_id = str(row.get("customer_id", ""))
            open_date = pd.to_datetime(row.get("opening_date"), errors="coerce")
            if pd.isna(open_date):
                open_date = datetime(year, 1, 1)
            end_open = "9999-12-31"
            account_tier = str(row.get("account_tier", "standard"))
            account_type = str(row.get("account_type", "personal"))

            if rng.random() < 0.22:
                change_date = open_date + timedelta(days=rng.randint(90, 240))
                account_dim_rows.append(
                    {
                        "account_id": account_id,
                        "customer_id": customer_id,
                        "effective_start_date": open_date.date().isoformat(),
                        "effective_end_date": (change_date.date() - timedelta(days=1)).isoformat(),
                        "is_current": False,
                        "account_type": account_type,
                        "account_tier": "standard",
                        "statement_frequency": str(row.get("statement_frequency", "monthly")),
                        "bank_name": COMPANY_BANK_NAME,
                    }
                )
                account_dim_rows.append(
                    {
                        "account_id": account_id,
                        "customer_id": customer_id,
                        "effective_start_date": change_date.date().isoformat(),
                        "effective_end_date": end_open,
                        "is_current": True,
                        "account_type": account_type,
                        "account_tier": account_tier,
                        "statement_frequency": str(row.get("statement_frequency", "monthly")),
                        "bank_name": COMPANY_BANK_NAME,
                    }
                )
            else:
                account_dim_rows.append(
                    {
                        "account_id": account_id,
                        "customer_id": customer_id,
                        "effective_start_date": open_date.date().isoformat(),
                        "effective_end_date": end_open,
                        "is_current": True,
                        "account_type": account_type,
                        "account_tier": account_tier,
                        "statement_frequency": str(row.get("statement_frequency", "monthly")),
                        "bank_name": COMPANY_BANK_NAME,
                    }
                )

        cust_cols = [c for c in ["customer_id", "customer_segment", "risk_band", "province", "city"] if c in accounts.columns]
        if "customer_id" in cust_cols:
            customer_base = accounts[cust_cols].drop_duplicates(subset=["customer_id"])
            for _, row in customer_base.iterrows():
                customer_dim_rows.append(
                    {
                        "customer_id": row.get("customer_id"),
                        "effective_start_date": f"{year}-01-01",
                        "effective_end_date": "9999-12-31",
                        "is_current": True,
                        "customer_segment": row.get("customer_segment", "retail"),
                        "risk_band": row.get("risk_band", rng.choice(["low", "medium", "high"])),
                        "province": row.get("province"),
                        "city": row.get("city"),
                    }
                )

    alerts: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    if not exceptions.empty:
        ex = exceptions.copy()
        ex["statement_ts"] = pd.to_datetime(ex.get("statement_date"), errors="coerce")
        for idx, row in ex.iterrows():
            sev = "medium"
            category = str(row.get("exception_category", ""))
            if category in {"duplicate_bank_entry", "amount_mismatch"}:
                sev = "high"
            elif category in {"missing_in_ledger", "missing_in_bank_statement"}:
                sev = "critical"
            alert_id = f"ALRT-{year}-{idx + 1:07d}"
            created_ts = row.get("statement_ts")
            if pd.isna(created_ts):
                created_ts = datetime(year, 1, 1, 8, 0, 0)
            alerts.append(
                {
                    "alert_id": alert_id,
                    "alert_type": "reconciliation_exception",
                    "severity": sev,
                    "account_id": row.get("account_id"),
                    "customer_id": row.get("customer_id"),
                    "source_reference": row.get("statement_transaction_id"),
                    "rule_name": f"rule_{category or 'unknown'}",
                    "created_at": created_ts,
                    "status": rng.choice(["new", "investigating", "closed"]),
                }
            )
        for idx, group_start in enumerate(range(0, len(alerts), 6), start=1):
            linked = alerts[group_start:group_start + 6]
            cases.append(
                {
                    "case_id": f"CASE-{year}-{idx:06d}",
                    "opened_at": linked[0]["created_at"] if linked else datetime(year, 1, 1),
                    "closed_at": None if rng.random() < 0.28 else linked[-1]["created_at"],
                    "status": "open" if rng.random() < 0.28 else "closed",
                    "priority": rng.choice(["P1", "P2", "P3"]),
                    "linked_alert_count": len(linked),
                    "primary_alert_id": linked[0]["alert_id"] if linked else None,
                    "owner_team": rng.choice(["compliance", "fraud", "finance_recon"]),
                }
            )

    customer_labels: list[dict[str, Any]] = []
    if not ledger.empty:
        tx = ledger.copy()
        tx["transaction_timestamp"] = pd.to_datetime(tx["transaction_timestamp"], errors="coerce")
        tx = tx.dropna(subset=["transaction_timestamp"])
        tx["month"] = tx["transaction_timestamp"].dt.strftime("%Y-%m")
        grouped = (
            tx.groupby(["customer_id", "month"])
            .agg(
                tx_count=("transaction_id", "count"),
                failed_rate=("status", lambda s: float((s.astype(str).str.lower() == "failed").mean())),
                fraud_events=("is_fraudulent", lambda s: int(pd.Series(s).fillna(False).sum())),
            )
            .reset_index()
        )
        grouped["next_month"] = (pd.to_datetime(grouped["month"] + "-01") + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m")
        has_next = set(zip(grouped["customer_id"].astype(str), grouped["month"].astype(str)))
        for _, row in grouped.iterrows():
            churn = (str(row["customer_id"]), str(row["next_month"])) not in has_next
            customer_labels.append(
                {
                    "customer_id": row["customer_id"],
                    "observation_month": row["month"],
                    "label_date": f"{row['month']}-28",
                    "fraud_label_30d": int(row["fraud_events"] > 0),
                    "default_risk_label_90d": int((row["failed_rate"] > 0.22) or (row["tx_count"] < 2)),
                    "churn_label_60d": int(churn),
                    "feature_window_days": 90,
                }
            )

    dq_rows: list[dict[str, Any]] = []
    dq_inputs: dict[str, pd.DataFrame] = {
        "accounts": accounts,
        "ledger": ledger,
        "reconciliation_results": recon,
        "reconciliation_exceptions": exceptions,
        "statement_transactions": statement_tx,
        "non_transactional_entries": non_tx,
    }
    for name, df in dq_inputs.items():
        if df.empty:
            dq_rows.append(
                {
                    "dataset_name": name,
                    "row_count": 0,
                    "column_count": 0,
                    "null_rate": 1.0,
                    "duplicate_rate": 0.0,
                    "quality_status": "empty",
                }
            )
            continue
        row_count = int(len(df))
        col_count = int(len(df.columns))
        null_rate = float(df.isna().sum().sum() / max(1, row_count * col_count))
        duplicate_rate = float(df.duplicated().mean()) if row_count > 1 else 0.0
        status = "good" if null_rate < 0.12 and duplicate_rate < 0.03 else "review"
        dq_rows.append(
            {
                "dataset_name": name,
                "row_count": row_count,
                "column_count": col_count,
                "null_rate": round(null_rate, 5),
                "duplicate_rate": round(duplicate_rate, 5),
                "quality_status": status,
            }
        )

    return {
        "realism_chart_of_accounts": dataframe_rows(chart_of_accounts),
        "realism_journal_entries": journal_rows,
        "realism_payment_lifecycle_events": payment_events,
        "realism_dim_accounts_scd2": account_dim_rows,
        "realism_dim_customers_scd2": customer_dim_rows,
        "realism_compliance_alerts": alerts,
        "realism_compliance_cases": cases,
        "realism_customer_labels": customer_labels,
        "realism_data_quality_scorecard": dq_rows,
    }


def cleanup_statement_outputs(year: int) -> None:
    out = OUT_DIR / str(year)
    for subdir in ["bank_statement_pdfs", "bank_statement_source_files"]:
        target = out / subdir
        if not target.exists():
            continue
        for path in sorted(target.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass


def generate(year: int, start_month: int, target_accounts: int | None, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    np.random.seed(seed)
    accounts = choose_accounts(load_accounts_customers(year, start_month), target_accounts, seed)
    accounts["statement_timeframe_months"] = [statement_timeframe_months(row, rng) for _, row in accounts.iterrows()]
    max_months = int(accounts["statement_timeframe_months"].max())
    ledger = stream_transactions(year, start_month, max_months, set(accounts["account_id"].astype(str)))
    if ledger.empty:
        raise RuntimeError("No transactions found for selected accounts/timeframe.")
    cleanup_statement_outputs(year)

    manifest: list[dict[str, Any]] = []
    statement_rows: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    non_tx_rows: list[dict[str, Any]] = []

    statement_dir = OUT_DIR / str(year) / "bank_statement_pdfs"
    source_dir = OUT_DIR / str(year) / "bank_statement_source_files"
    for _, account in tqdm(accounts.iterrows(), total=len(accounts), desc="Reconciliation datasets"):
        timeframe = int(account["statement_timeframe_months"])
        period_start = datetime(year, start_month, 1)
        end_y, end_m = month_add(year, start_month, timeframe - 1)
        period_end = month_end(end_y, end_m)
        run_id = f"RECON-{year}{start_month:02d}-{account['account_id']}-{timeframe}M"
        built = build_for_account(account, ledger, period_start, period_end, timeframe, run_id, rng)
        source_format = statement_source_format(account, rng)
        manifest.extend(built["manifest"])
        statement_rows.extend(built["statement_rows"])
        recon_rows.extend(built["recon_rows"])
        exception_rows.extend(built["exception_rows"])
        non_tx_rows.extend(built["non_tx_rows"])
        base_name = f"{account['account_id']}_{period_start:%Y%m}_{period_end:%Y%m}_{timeframe}M_statement"
        pdf_path = statement_dir / f"{base_name}.pdf"
        source_path = source_dir / source_format.lower() / f"{base_name}{statement_source_suffix(source_format)}"
        statement_pdf(pdf_path, account, built["statement_rows"], period_start, period_end, built["manifest"][0]["opening_balance"], built["manifest"][0]["closing_balance"])
        write_statement_source_file(source_path, source_format, account, built["statement_rows"], period_start, period_end, built["manifest"][0]["opening_balance"], built["manifest"][0]["closing_balance"])
        manifest[-1]["statement_pdf_path"] = str(pdf_path.relative_to(DATA_DIR))
        manifest[-1]["statement_source_format"] = source_format
        manifest[-1]["statement_source_file_path"] = str(source_path.relative_to(DATA_DIR))

    recon_df = pd.DataFrame(recon_rows)
    exception_df = pd.DataFrame(exception_rows)
    statement_df = pd.DataFrame(statement_rows)
    non_tx_df = pd.DataFrame(non_tx_rows)

    analytics = build_analytics_marts(ledger, recon_df, exception_df, accounts)
    realism = build_realism_expansion(accounts, ledger, recon_df, statement_df, non_tx_df, exception_df, year, rng)
    out = OUT_DIR / str(year)
    write_dataset(out / "bank_statement_manifest", manifest)
    write_dataset(out / "bank_statement_transactions", statement_rows)
    write_dataset(out / "reconciliation_results", recon_rows)
    write_dataset(out / "reconciliation_exceptions", exception_rows)
    write_dataset(out / "non_transactional_statement_entries", non_tx_rows)
    for name, rows in analytics.items():
        write_dataset(out / name, rows)
    for name, rows in realism.items():
        write_dataset(out / "realism" / name, rows)

    return {
        "accounts": int(len(accounts)),
        "statements": int(len(manifest)),
        "source_statement_files": int(len(manifest)),
        "source_formats": pd.Series([m["statement_source_format"] for m in manifest]).value_counts().to_dict() if manifest else {},
        "statement_rows": int(len(statement_rows)),
        "reconciliation_rows": int(len(recon_rows)),
        "exceptions": int(len(exception_rows)),
        "realism_tables": sorted(realism.keys()),
        "output_dir": str(out),
    }


def build_analytics_marts(ledger: pd.DataFrame, recon: pd.DataFrame, exceptions: pd.DataFrame, accounts: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    tx = ledger.copy()
    tx["transaction_timestamp"] = pd.to_datetime(tx["transaction_timestamp"], errors="coerce")
    tx["month"] = tx["transaction_timestamp"].dt.strftime("%Y-%m")
    tx["signed_amount"] = tx.apply(signed_amount, axis=1)

    monthly = (
        tx.groupby(["month", "account_id", "customer_id"])
        .agg(
            transaction_count=("transaction_id", "count"),
            total_debits=("signed_amount", lambda s: round(float(abs(s[s < 0].sum())), 2)),
            total_credits=("signed_amount", lambda s: round(float(s[s > 0].sum()), 2)),
            net_movement=("signed_amount", "sum"),
            avg_transaction_amount=("amount", "mean"),
            failed_count=("status", lambda s: int((s.astype(str).str.lower() == "failed").sum())),
            fraud_count=("is_fraudulent", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
    )
    monthly["net_movement"] = monthly["net_movement"].round(2)
    monthly["avg_transaction_amount"] = monthly["avg_transaction_amount"].round(2)

    channel = (
        tx.groupby(["month", "channel"])
        .agg(
            transaction_count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            unique_customers=("customer_id", "nunique"),
            failed_count=("status", lambda s: int((s.astype(str).str.lower() == "failed").sum())),
            avg_authorization_time_ms=("authorization_time_ms", "mean"),
        )
        .reset_index()
    )
    channel["failure_rate"] = (channel["failed_count"] / channel["transaction_count"]).round(4)
    channel["total_amount"] = channel["total_amount"].round(2)
    channel["avg_authorization_time_ms"] = channel["avg_authorization_time_ms"].round(2)

    if exceptions.empty:
        exception_summary = pd.DataFrame()
    else:
        exception_summary = (
            exceptions.groupby(["exception_category", "root_cause_tag"])
            .agg(exception_count=("match_status", "count"), total_statement_amount=("statement_amount", "sum"), impacted_accounts=("account_id", "nunique"))
            .reset_index()
        )
        exception_summary["total_statement_amount"] = exception_summary["total_statement_amount"].round(2)

    match_summary = []
    if not recon.empty:
        total = len(recon)
        matched = int((recon["match_status"] == "matched").sum())
        total_exception_rows = int(len(exceptions)) if not exceptions.empty else total - matched
        match_summary.append(
            {
                "metric_date": datetime.now().date().isoformat(),
                "transactional_reconciliation_rows": total,
                "matched_rows": matched,
                "transactional_exception_rows": total - matched,
                "total_exception_rows": total_exception_rows,
                "transactional_auto_match_rate": round(matched / max(1, total), 4),
                "target_auto_match_rate": 0.95,
                "meets_target": matched / max(1, total) >= 0.95,
            }
        )

    account_segments = accounts[["account_id", "customer_id", "account_type", "account_tier", "statement_timeframe_months"]].copy()
    account_segments["analysis_segment"] = account_segments["account_tier"].astype(str) + "_" + account_segments["account_type"].astype(str)

    return {
        "analytics_customer_monthly": monthly.to_dict("records"),
        "analytics_channel_kpis": channel.to_dict("records"),
        "analytics_exception_root_causes": exception_summary.to_dict("records") if not exception_summary.empty else [],
        "analytics_reconciliation_scorecard": match_summary,
        "analytics_account_segments": account_segments.to_dict("records"),
    }


# ── Bank statement PDF generation (monthly-foldered, per-account cycle) ──────

FREQ_TO_CYCLE: dict[str, int] = {"monthly": 1, "quarterly": 3, "annually": 12}


def get_account_cycle(account: pd.Series) -> int:
    """Return cycle_months from explicit column or derived from statement_frequency."""
    explicit = account.get("statement_cycle_months")
    if explicit is not None and not pd.isna(explicit):
        try:
            v = int(explicit)
            if v in (1, 2, 3, 6, 12):
                return v
        except (ValueError, TypeError):
            pass
    freq = str(account.get("statement_frequency", "")).lower()
    return FREQ_TO_CYCLE.get(freq, 3)


def get_period_ends_for_cycle(
    cycle: int, start_year: int = 2019, end_year: int = 2025
) -> list[tuple[int, int]]:
    """Return all (year, end_month) period-end dates for a given cycle.
    For cycle=3 the valid end months are 3,6,9,12; for cycle=1 all months."""
    valid_end_months = [m for m in range(1, 13) if m % cycle == 0]
    return [
        (year, m)
        for year in range(start_year, end_year + 1)
        for m in valid_end_months
    ]


def get_period_ends_for_account(
    account: pd.Series, cycle: int, start_year: int = 2019, end_year: int = 2025
) -> list[tuple[int, int]]:
    """Return cycle end months anchored to the account opening month.

    Example: account opened in Jan with cycle=3 -> first statement ends in Mar.
    Account opened in Feb with cycle=3 -> first statement ends in Apr.
    """
    window_start = start_year * 12
    window_end = end_year * 12 + 11
    opened = pd.to_datetime(account.get("opening_date"), errors="coerce")
    if pd.isna(opened):
        open_idx = window_start
    else:
        open_idx = int(opened.year) * 12 + int(opened.month) - 1

    end_idx = open_idx + cycle - 1
    while end_idx < window_start:
        end_idx += cycle

    closure = pd.to_datetime(account.get("closure_date"), errors="coerce")
    closure_idx = window_end if pd.isna(closure) else int(closure.year) * 12 + int(closure.month) - 1
    final_idx = min(window_end, closure_idx)

    out: list[tuple[int, int]] = []
    while end_idx <= final_idx:
        out.append((end_idx // 12, end_idx % 12 + 1))
        end_idx += cycle
    if not pd.isna(closure) and closure_idx >= open_idx:
        closure_period = (closure_idx // 12, closure_idx % 12 + 1)
        if closure_period not in out:
            out.append(closure_period)
            out.sort()
    return out


def period_start_for_end(end_year: int, end_month: int, cycle: int) -> tuple[int, int]:
    """Return (start_year, start_month) given a period end date and cycle length in months."""
    total = end_year * 12 + end_month - 1  # zero-based month index
    start_total = total - (cycle - 1)
    return start_total // 12, start_total % 12 + 1


def load_all_accounts_deduplicated(start_year: int, end_year: int) -> pd.DataFrame:
    """Load all account parquet/csv files from the monthly folder tree, keeping the latest
    record per account_id (so a customer who appears in multiple months is deduplicated)."""
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            f = table_file("accounts", year, month)
            if f is None:
                continue
            try:
                df = read_table(f)
                df["_src_year"] = year
                df["_src_month"] = month
                frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined
        .sort_values(["_src_year", "_src_month"])
        .drop_duplicates(subset=["account_id"], keep="last")
        .drop(columns=["_src_year", "_src_month"])
        .reset_index(drop=True)
    )
    return combined


def load_transactions_for_range(
    account_ids: set[str], period_start: datetime, period_end: datetime
) -> pd.DataFrame:
    """Stream transactions from the monthly folder tree for specific accounts
    within [period_start, period_end]."""
    if not account_ids:
        return pd.DataFrame()
    sy, sm = period_start.year, period_start.month
    ey, em = period_end.year, period_end.month
    n_months = (ey - sy) * 12 + (em - sm) + 1
    return stream_transactions(sy, sm, n_months, account_ids)


def inject_bank_discrepancies(
    acc_tx: pd.DataFrame,
    period_end: datetime,
    rng: random.Random,
    missing_omitted: set,
    account_id: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply three bank-side discrepancy rules to simulate real-world timing differences.

    1. Deposits In Transit (up to 2): deposits in last 5 days of period get a
       bank_posted_date pushed into the next statement period.
    2. Outstanding Debits (up to 2): outgoing transactions near month-end are
       flagged bank_status = 'Pending'.
    3. Missing Transaction (1 per account per year): one small-value transaction
       is omitted from the PDF entirely.
    """
    if acc_tx.empty:
        return acc_tx.copy(), []

    df = acc_tx.copy()
    df["bank_posted_date"] = df["transaction_timestamp"].dt.strftime("%Y-%m-%d")
    df["bank_status"] = "Posted"
    discrepancies: list[dict] = []

    cutoff = period_end - timedelta(days=5)
    is_credit = df["debit_credit"].astype(str).str.lower().isin(["credit", "cr"])
    is_debit = df["debit_credit"].astype(str).str.lower().isin(["debit", "dr"])
    is_last5 = df["transaction_timestamp"] >= cutoff

    # 1. Deposits in transit
    transit_pool = df.index[is_credit & is_last5].tolist()
    for idx in rng.sample(transit_pool, min(2, len(transit_pool))):
        delayed = period_end + timedelta(days=rng.randint(1, 5))
        df.at[idx, "bank_posted_date"] = delayed.strftime("%Y-%m-%d")
        discrepancies.append({
            "discrepancy_type": "deposit_in_transit",
            "transaction_id": str(df.at[idx, "transaction_id"]),
            "original_date": str(df.at[idx, "transaction_timestamp"].date()),
            "bank_posted_date": delayed.strftime("%Y-%m-%d"),
        })

    # 2. Outstanding debits (Pending)
    pending_pool = df.index[is_debit & is_last5].tolist()
    for idx in rng.sample(pending_pool, min(2, len(pending_pool))):
        df.at[idx, "bank_status"] = "Pending"
        discrepancies.append({
            "discrepancy_type": "outstanding_debit",
            "transaction_id": str(df.at[idx, "transaction_id"]),
            "original_date": str(df.at[idx, "transaction_timestamp"].date()),
            "bank_posted_date": None,
        })

    # 3. Missing transaction (once per account per calendar year)
    year_key = (account_id, period_end.year)
    if year_key not in missing_omitted:
        df["_amt"] = pd.to_numeric(df["amount"], errors="coerce").abs()
        low_thresh = df["_amt"].quantile(0.25)
        small = df[df["_amt"] <= low_thresh].dropna(subset=["_amt"])
        df.drop(columns=["_amt"], inplace=True)
        if not small.empty:
            omit_idx = rng.choice(list(small.index))
            discrepancies.append({
                "discrepancy_type": "missing_transaction",
                "transaction_id": str(df.at[omit_idx, "transaction_id"]),
                "amount": float(df.at[omit_idx, "amount"]),
                "original_date": str(df.at[omit_idx, "transaction_timestamp"].date()),
                "bank_posted_date": None,
            })
            df = df.drop(index=omit_idx).reset_index(drop=True)
            missing_omitted.add(year_key)

    return df, discrepancies


def bank_statement_pdf(
    path: Path,
    account: pd.Series,
    df_rows: pd.DataFrame,
    period_start: datetime,
    period_end: datetime,
    discrepancies: list[dict],
) -> None:
    """Generate a spec-compliant bank statement PDF.
    Named {account_id}.pdf and placed in the caller-supplied path.
    Pending rows are amber; transit rows have a blue posted-date column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("BSTitle", parent=styles["Title"], fontSize=15, leading=19, textColor=colors.HexColor("#1F2937"))
    header_style = ParagraphStyle("BSHeader", parent=styles["Normal"], fontSize=9, leading=12)
    small_style = ParagraphStyle("BSSmall", parent=styles["Normal"], fontSize=7.5, leading=10)

    account_id = str(account.get("account_id", ""))
    account_number = str(account.get("account_number", account_id))
    bank = COMPANY_BANK_NAME

    if period_start.month == period_end.month and period_start.year == period_end.year:
        period_label = f"{period_start:%B %Y}"
    else:
        period_label = f"{period_start:%B %Y} \u2013 {period_end:%B %Y}"
    statement_date_label = f"{period_end.day} {period_end:%B %Y}"

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        rightMargin=1.2 * cm, leftMargin=1.2 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
    )
    story: list[Any] = []

    story.append(Paragraph(f"{escape(bank)} \u2013 Bank Statement", title_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"<b>Account ID:</b> {escape(account_id)}&nbsp;&nbsp;&nbsp;"
        f"<b>Account Number:</b> {escape(account_number)}",
        header_style,
    ))
    story.append(Paragraph(f"<b>Customer:</b> {escape(account_display_name(account))}", header_style))
    story.append(Paragraph(f"<b>Statement Period:</b> {escape(period_label)}", header_style))
    story.append(Paragraph(f"<b>Statement Date:</b> {escape(statement_date_label)}", header_style))
    story.append(Spacer(1, 0.3 * cm))

    if df_rows.empty:
        story.append(Paragraph("No transactions recorded in this statement period.", small_style))
        doc.build(story)
        return

    has_bank_posted = "bank_posted_date" in df_rows.columns
    col_headers = ["Date", "Description", "Reference", "Debit (R)", "Credit (R)"]
    if has_bank_posted:
        col_headers.append("Bank Posted")
    col_headers.append("Status")

    table_data: list[list] = [col_headers]
    for _, row in df_rows.iterrows():
        amount = float(pd.to_numeric(row.get("amount", 0), errors="coerce") or 0)
        is_dr = str(row.get("debit_credit", "debit")).lower() in ("debit", "dr")
        ts = row.get("transaction_timestamp")
        date_str = pd.to_datetime(ts).strftime("%d %b %Y") if pd.notna(ts) else ""
        desc = str(row.get("description", row.get("statement_description", "")))[:40]
        ref = str(row.get("rrn", row.get("transaction_id", "")))[:16]
        status_val = str(row.get("bank_status", "Posted"))
        data_row: list = [
            date_str,
            desc,
            ref,
            f"{amount:,.2f}" if is_dr else "",
            f"{amount:,.2f}" if not is_dr else "",
        ]
        if has_bank_posted:
            data_row.append(str(row.get("bank_posted_date", date_str)))
        data_row.append(status_val)
        table_data.append(data_row)

    col_widths = [1.8 * cm, 5.6 * cm, 2.8 * cm, 2.1 * cm, 2.1 * cm]
    if has_bank_posted:
        col_widths.append(2.1 * cm)
    col_widths.append(1.5 * cm)

    pending_rows = [i + 1 for i, r in enumerate(table_data[1:]) if r[-1] == "Pending"]
    transit_rows: list[int] = []
    if has_bank_posted:
        transit_rows = [
            i + 1 for i, r in enumerate(table_data[1:])
            if r[-2] != r[0] and r[-1] != "Pending"
        ]

    style_cmds: list = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (3, 1), (4, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]
    for pr in pending_rows:
        style_cmds += [
            ("TEXTCOLOR", (0, pr), (-1, pr), colors.HexColor("#B45309")),
            ("BACKGROUND", (0, pr), (-1, pr), colors.HexColor("#FEF3C7")),
        ]
    for tr in transit_rows:
        style_cmds.append(("TEXTCOLOR", (0, tr), (-1, tr), colors.HexColor("#1D4ED8")))

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(style_cmds))
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))

    if discrepancies:
        story.append(Paragraph("<b>Reconciliation Notes (Injected Discrepancies):</b>", header_style))
        for d in discrepancies:
            dtype = d["discrepancy_type"].replace("_", " ").title()
            if d["discrepancy_type"] == "deposit_in_transit":
                note = f"{dtype}: txn {d['transaction_id']} dated {d['original_date']} \u2192 bank posted {d['bank_posted_date']} (next period)"
            elif d["discrepancy_type"] == "outstanding_debit":
                note = f"{dtype}: txn {d['transaction_id']} dated {d['original_date']} shown as Pending"
            else:
                note = (
                    f"{dtype}: txn {d['transaction_id']} "
                    f"(R{float(d.get('amount', 0)):,.2f} on {d.get('original_date', '')}) "
                    f"omitted from bank statement"
                )
            story.append(Paragraph(f"\u2022 {escape(note)}", small_style))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "Synthetic bank statement \u2013 for reconciliation practice and analytics training data only.",
        small_style,
    ))
    doc.build(story)


def print_generation_summary(
    summary_df: pd.DataFrame, total_pdfs: int, start_year: int, end_year: int
) -> None:
    print(f"\n{'=' * 80}")
    print("BANK STATEMENT PDF GENERATION PLAN")
    print(f"{'=' * 80}")
    print(f"Date range     : {start_year}/01 -> {end_year}/12")
    print(f"Total accounts : {len(summary_df):,}")
    print(f"Total PDFs     : {total_pdfs:,}")
    print()
    print(f"{'account_id':<16} {'cycle_months':>13} {'total_statements':>17}  folder_months")
    print("-" * 80)
    for _, row in summary_df.head(30).iterrows():
        print(
            f"{str(row['account_id']):<16} {int(row['cycle_months']):>13} "
            f"{int(row['total_statements']):>17}  {row['folder_months']}"
        )
    if len(summary_df) > 30:
        print(f"  ... {len(summary_df) - 30} more accounts not shown")
    print(f"{'=' * 80}\n")


def generate_bank_statement_pdfs(
    start_year: int = 2019,
    end_year: int = 2025,
    seed: int = 20260519,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate per-account monthly-foldered bank statement PDFs for every
    account from start_year/01 through end_year/12 using each account's
    individual statement cycle (monthly=1, quarterly=3, annual=12).

    Folder structure: banking_data/bank_statements/YEAR/MM/{account_id}.pdf
    Only months where at least one account has a cycle end are created.
    """
    rng = random.Random(seed)

    # ── 1. Load and deduplicate all accounts ──────────────────────────────────
    print("Loading account metadata across all years …")
    all_accounts = load_all_accounts_deduplicated(start_year, end_year)
    if all_accounts.empty:
        print("No account data found. Exiting.")
        return {}

    all_accounts["cycle_months"] = all_accounts.apply(get_account_cycle, axis=1)

    # ── 2. Build period map and pre-execution summary ─────────────────────────
    # period_map groups accounts that share the same (cycle, period_end_year, period_end_month)
    # so transactions can be loaded once per group.
    period_map: dict[tuple[int, int, int], list[pd.Series]] = {}
    summary_rows: list[dict] = []

    for _, account in all_accounts.iterrows():
        cycle = int(account["cycle_months"])
        period_ends = get_period_ends_for_account(account, cycle, start_year, end_year)
        folder_months = sorted({m for _, m in period_ends})
        summary_rows.append({
            "account_id": account["account_id"],
            "cycle_months": cycle,
            "total_statements": len(period_ends),
            "folder_months": "all 12 months" if cycle == 1 else ",".join(f"{m:02d}" for m in folder_months),
        })
        for ey, em in period_ends:
            period_map.setdefault((cycle, ey, em), []).append(account)

    summary_df = pd.DataFrame(summary_rows)
    total_pdfs = int(summary_df["total_statements"].sum())
    print_generation_summary(summary_df, total_pdfs, start_year, end_year)

    if dry_run:
        return {"dry_run": True, "total_pdfs": total_pdfs, "accounts": len(all_accounts)}

    # ── 3. Iterate over every period group and generate PDFs ──────────────────
    out_dir = DATA_DIR

    missing_omitted: set[tuple[str, int]] = set()
    discrepancy_log: list[dict] = []
    total_generated = 0

    # Sort chronologically (end_year, end_month) then by cycle for consistent ordering
    sorted_keys = sorted(period_map.keys(), key=lambda k: (k[1], k[2], k[0]))

    for cycle, ey, em in tqdm(sorted_keys, desc="Statement periods"):
        period_end_dt = month_end(ey, em)
        sy, sm = period_start_for_end(ey, em, cycle)
        period_start_dt = datetime(sy, sm, 1)

        accounts_in_group = period_map[(cycle, ey, em)]
        account_ids = {str(a["account_id"]) for a in accounts_in_group}

        # Load transactions once for all accounts sharing this period
        ledger = load_transactions_for_range(account_ids, period_start_dt, period_end_dt)

        folder = out_dir / str(ey) / f"{em:02d}" / "bank_statements"
        folder.mkdir(parents=True, exist_ok=True)

        for account in accounts_in_group:
            account_id = str(account["account_id"])
            closure = pd.to_datetime(account.get("closure_date"), errors="coerce")
            account_period_end = (
                min(period_end_dt, closure.to_pydatetime())
                if not pd.isna(closure)
                else period_end_dt
            )

            if ledger.empty:
                acc_tx = pd.DataFrame()
            else:
                acc_tx = (
                    ledger[ledger["account_id"].astype(str) == account_id]
                    .copy()
                    .sort_values("transaction_timestamp")
                    .reset_index(drop=True)
                )

            acc_tx_bank, discrepancies = inject_bank_discrepancies(
                acc_tx, account_period_end, rng, missing_omitted, account_id
            )

            for d in discrepancies:
                discrepancy_log.append({
                    "account_id": account_id,
                    "statement_year": ey,
                    "statement_month": em,
                    "cycle_months": cycle,
                    **d,
                })

            pdf_path = folder / f"{account_id}.pdf"
            bank_statement_pdf(
                pdf_path, account, acc_tx_bank,
                period_start_dt, account_period_end, discrepancies,
            )
            total_generated += 1

    # ── 4. Save discrepancy log alongside the PDFs ────────────────────────────
    write_dataset(OUT_DIR / f"bank_statement_discrepancy_log_{start_year}_{end_year}", discrepancy_log)

    result: dict[str, Any] = {
        "total_pdfs_generated": total_generated,
        "total_discrepancies_injected": len(discrepancy_log),
        "discrepancy_types": (
            pd.Series([d["discrepancy_type"] for d in discrepancy_log])
            .value_counts().to_dict()
            if discrepancy_log else {}
        ),
        "output_dir": str(DATA_DIR),
        "accounts": len(all_accounts),
    }
    print(json.dumps(result, indent=2))
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reconciliation datasets OR monthly-foldered bank statement PDFs.\n"
            "  reconciliation  : per-account statement source files + recon tables (existing mode)\n"
            "  bank-statements : per-account cycle-aware PDFs from 2019/01 to 2025/12"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["reconciliation", "bank-statements"],
        default="reconciliation",
        help="Which dataset to generate (default: reconciliation).",
    )
    # reconciliation mode args
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--target-accounts", type=int, default=None,
                        help="Optional account cap for testing.")
    # bank-statements mode args
    parser.add_argument("--start-year", type=int, default=2019,
                        help="First year for bank-statements mode (default: 2019).")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="Last year for bank-statements mode (default: 2025).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generation summary then exit without writing files.")
    # shared
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    if args.mode == "bank-statements":
        generate_bank_statement_pdfs(
            start_year=args.start_year,
            end_year=args.end_year,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    else:
        summary = generate(args.year, args.start_month, args.target_accounts, args.seed)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
