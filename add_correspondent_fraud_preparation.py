"""
Add correspondent-banking fraud preparation signals.

Outputs live under banking_data/YYYY/MM/ and are deliberately small:
- news/*.md + rendered PDFs
- emails/*.eml + rendered PDFs
- correspondent_banking/correspondent_payment_alerts.csv
- correspondent_banking/swift_messages.jsonl
- correspondent_banking/nostro_reconciliation_breaks.csv
- a few supporting international-transfer rows in transactions.jsonl

The cases are hints for analysts. They are not broad labels on customers.
"""

from __future__ import annotations

import csv
import email.utils
import json
import math
import random
import re
from datetime import date, datetime, time, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd

from commons.pdf_renderers import render_news_article_pdf, render_outlook_email_pdf


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"
SOURCE_TABLE = "correspondent_fraud_preparation"


CASES = [
    {
        "year": 2023,
        "month": 9,
        "title": "Regional banks query split dollar payments routed through intermediary accounts",
        "outlet": "Business Day",
        "author": "Londiwe Buthelezi",
        "org": "SADC Banking Compliance Forum",
        "department": "AML Operations",
        "project_type": "correspondent_payment_monitoring",
        "pattern": "correspondent_routing_anomaly",
        "risk_theme": "split USD transfers routed through two intermediary banks before reaching a common beneficiary",
        "beneficiary": "Kavango Trade Clearing LLC",
        "beneficiary_country": "AE",
        "intermediary_bank": "Gulf Meridian Bank",
        "intermediary_swift": "GMBLAEAD",
        "currency": "USD",
        "amounts": [18450.0, 19750.0, 21320.0, 22680.0],
        "news_day": 18,
    },
    {
        "year": 2024,
        "month": 11,
        "title": "Trade-finance teams review import invoices after correspondent bank query",
        "outlet": "Financial Mail",
        "author": "Mpho Radebe",
        "org": "Johannesburg Trade Finance Working Group",
        "department": "Trade Finance",
        "project_type": "invoice_overpricing_detection",
        "pattern": "trade_invoice_mismatch",
        "risk_theme": "invoice values that moved ahead of shipping documents and repeated amendments to beneficiary bank details",
        "beneficiary": "Blue Harbour Components Ltd",
        "beneficiary_country": "HK",
        "intermediary_bank": "Pacific Crown Bank",
        "intermediary_swift": "PCBKHKHH",
        "currency": "USD",
        "amounts": [31240.0, 33780.0, 29860.0, 35410.0],
        "news_day": 12,
    },
    {
        "year": 2025,
        "month": 3,
        "title": "Correspondent bank asks SA lenders to review repeated MT103 beneficiary amendments",
        "outlet": "Moneyweb",
        "author": "Thabiso Mochiko",
        "org": "International Payments Compliance Roundtable",
        "department": "Payments Compliance",
        "project_type": "swift_message_exception_review",
        "pattern": "swift_beneficiary_amendment",
        "risk_theme": "MT103 payment instructions where beneficiary names were amended after release but before settlement",
        "beneficiary": "Baltic Mineral Supplies OU",
        "beneficiary_country": "EE",
        "intermediary_bank": "Nordic Baltic Correspondent Bank",
        "intermediary_swift": "NBCBEE22",
        "currency": "EUR",
        "amounts": [16480.0, 17125.0, 18990.0, 17640.0],
        "news_day": 20,
    },
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:70]


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_case_accounts(year: int, month: int, limit: int = 4) -> list[dict[str, Any]]:
    accounts_path = BANKING_DIR / f"{year}" / f"{month:02d}" / f"accounts_{year}_{month:02d}.parquet"
    customers_path = BANKING_DIR / f"{year}" / f"{month:02d}" / f"customers_{year}_{month:02d}.parquet"
    accounts = pd.read_parquet(accounts_path)
    customers = pd.read_parquet(customers_path)[["customer_id", "full_name"]]
    merged = accounts.merge(customers, on="customer_id", how="left")

    pool = merged[
        (merged["account_status"].astype(str).str.lower().isin(["active", "restricted"]))
        & (
            merged["cross_border_enabled"].fillna(False).astype(bool)
            | merged["currency"].astype(str).isin(["USD", "EUR", "GBP"])
        )
    ].copy()
    if len(pool) < limit:
        pool = merged.copy()

    pool = pool.sort_values(["customer_id", "account_id"]).head(max(limit * 3, limit))
    return [
        {
            "customer_id": str(row["customer_id"]),
            "account_id": str(row["account_id"]),
            "account_number": str(row.get("account_number", "")),
            "full_name": str(row.get("full_name") or row["customer_id"]),
            "bank_name": str(row.get("bank_name", "Keystone Retail Bank")),
            "currency": str(row.get("currency", "ZAR")),
            "swift_code": clean(row.get("swift_code")),
            "iban": clean(row.get("iban")),
        }
        for _, row in pool.head(limit).iterrows()
    ]


def remove_existing_preparation_rows(path: Path) -> None:
    if not path.exists():
        return
    kept: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if row.get("source_table") != SOURCE_TABLE:
                kept.append(line)
    path.write_text("".join(kept), encoding="utf-8")


def append_transaction_rows(case: dict[str, Any], accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    year = int(case["year"])
    month = int(case["month"])
    tx_path = BANKING_DIR / f"{year}" / f"{month:02d}" / "transactions.jsonl"
    remove_existing_preparation_rows(tx_path)

    rng = random.Random(year * 100 + month + 7109)
    rows: list[dict[str, Any]] = []
    base_date = date(year, month, max(4, int(case["news_day"]) - 8))
    for idx, account in enumerate(accounts):
        tx_date = base_date + timedelta(days=idx * 2)
        tx_time = time(10 + idx, rng.randint(3, 53), rng.randint(0, 59))
        amount = float(case["amounts"][idx % len(case["amounts"])])
        status = ["completed", "held_for_review", "completed", "reversed"][idx % 4]
        is_flagged = status in {"held_for_review", "reversed"}
        row = {
            "transaction_id": f"CBF{year}{month:02d}{idx + 1:06d}",
            "batch_id": f"BATCH-{year}{month:02d}-CORR",
            "generation_timestamp": datetime(2026, 5, 22, 18, 0, idx).isoformat(),
            "transaction_timestamp": datetime.combine(tx_date, tx_time).isoformat(),
            "transaction_date": tx_date.isoformat(),
            "transaction_time": tx_time.isoformat(),
            "customer_id": account["customer_id"],
            "account_id": account["account_id"],
            "channel": "international_payments",
            "channel_metadata": {
                "payment_network": "SWIFT",
                "message_type": "MT103",
                "originating_bank": account["bank_name"],
                "originating_swift": account["swift_code"] or "KEYSZACC",
                "intermediary_bank": case["intermediary_bank"],
                "intermediary_swift": case["intermediary_swift"],
                "beneficiary_country": case["beneficiary_country"],
                "purpose_code": rng.choice(["goods_import", "supplier_payment", "consulting_services"]),
            },
            "category": "international_transfer",
            "amount": round(amount, 2),
            "debit_credit": "debit",
            "status": status,
            "failure_reason": None if status != "reversed" else "beneficiary_bank_returned",
            "description": f"SWIFT MT103 to {case['beneficiary']} via {case['intermediary_bank']}",
            "merchant_name": case["beneficiary"],
            "currency": case["currency"],
            "receiving_account": f"{case['beneficiary_country']}{rng.randint(10**10, 10**11 - 1)}",
            "beneficiary_name": case["beneficiary"],
            "is_fraudulent": is_flagged,
            "fraud_pattern": case["pattern"] if is_flagged else None,
            "fraud_confidence": round(rng.uniform(0.61, 0.78), 3) if is_flagged else 0.0,
            "fraud_metadata": {
                "typology": case["pattern"],
                "risk_theme": case["risk_theme"],
                "intermediary_bank": case["intermediary_bank"],
                "intermediary_swift": case["intermediary_swift"],
                "beneficiary_country": case["beneficiary_country"],
                "analyst_hint_only": True,
            }
            if is_flagged
            else {},
            "has_error": False,
            "error_types": [],
            "error_metadata": {},
            "network_latency_ms": rng.randint(500, 2200),
            "authorization_time_ms": rng.randint(1200, 3800),
            "third_party_timeout": False,
            "stan": str(rng.randint(100000, 999999)),
            "rrn": f"{year}{month:02d}{rng.randint(10**8, 10**9 - 1)}",
            "source_table": SOURCE_TABLE,
            "customer_session_id": f"SWIFT-{account['customer_id']}-{year}{month:02d}{idx + 1:02d}",
            "customer_device_fingerprint": f"TREASURY-PORTAL-{account['customer_id'][-5:]}",
            "customer_location_state_before": {},
            "customer_location_state_after": {},
            "external_context": {
                "correspondent_bank_review": True,
                "case_title": case["title"],
                "analyst_preparation_signal": True,
            },
        }
        rows.append(row)

    with tx_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def write_supporting_tables(case: dict[str, Any], accounts: list[dict[str, Any]], tx_rows: list[dict[str, Any]]) -> None:
    year = int(case["year"])
    month = int(case["month"])
    out_dir = BANKING_DIR / f"{year}" / f"{month:02d}" / "correspondent_banking"
    out_dir.mkdir(parents=True, exist_ok=True)

    alerts_path = out_dir / "correspondent_payment_alerts.csv"
    with alerts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "alert_id",
                "transaction_id",
                "customer_name",
                "customer_id",
                "account_id",
                "alert_date",
                "typology",
                "currency",
                "amount",
                "intermediary_bank",
                "intermediary_swift",
                "beneficiary_name",
                "beneficiary_country",
                "risk_reason",
                "alert_status",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(tx_rows, start=1):
            account = next(a for a in accounts if a["account_id"] == row["account_id"])
            writer.writerow(
                {
                    "alert_id": f"CB-ALERT-{year}{month:02d}-{idx:03d}",
                    "transaction_id": row["transaction_id"],
                    "customer_name": account["full_name"],
                    "customer_id": row["customer_id"],
                    "account_id": row["account_id"],
                    "alert_date": row["transaction_date"],
                    "typology": case["pattern"],
                    "currency": case["currency"],
                    "amount": row["amount"],
                    "intermediary_bank": case["intermediary_bank"],
                    "intermediary_swift": case["intermediary_swift"],
                    "beneficiary_name": case["beneficiary"],
                    "beneficiary_country": case["beneficiary_country"],
                    "risk_reason": case["risk_theme"],
                    "alert_status": "open" if row["status"] == "held_for_review" else "under_review",
                }
            )

    swift_path = out_dir / "swift_messages.jsonl"
    with swift_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(tx_rows, start=1):
            account = next(a for a in accounts if a["account_id"] == row["account_id"])
            message = {
                "message_id": f"MT103-{year}{month:02d}-{idx:05d}",
                "transaction_id": row["transaction_id"],
                "message_type": "MT103",
                "sender_bic": account["swift_code"] or "KEYSZACC",
                "receiver_bic": case["intermediary_swift"],
                "value_date": row["transaction_date"],
                "currency": case["currency"],
                "amount": row["amount"],
                "ordering_customer": account["full_name"],
                "ordering_account": account["account_number"],
                "beneficiary_name": case["beneficiary"],
                "beneficiary_country": case["beneficiary_country"],
                "field_70_remittance": "supplier settlement per invoice",
                "field_72_sender_to_receiver": "intermediary review requested" if idx % 2 == 0 else "",
                "screening_status": "possible_match" if idx % 2 == 0 else "cleared",
                "screening_reason": case["risk_theme"] if idx % 2 == 0 else "",
            }
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")

    breaks_path = out_dir / "nostro_reconciliation_breaks.csv"
    with breaks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "break_id",
                "transaction_id",
                "nostro_account",
                "currency",
                "expected_settlement_date",
                "actual_settlement_date",
                "break_type",
                "break_amount",
                "owner_team",
                "status",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(tx_rows, start=1):
            if idx % 2 == 1:
                continue
            tx_date = datetime.fromisoformat(row["transaction_timestamp"]).date()
            writer.writerow(
                {
                    "break_id": f"NOSTRO-{year}{month:02d}-{idx:03d}",
                    "transaction_id": row["transaction_id"],
                    "nostro_account": f"NOSTRO-{case['currency']}-{case['intermediary_swift'][:4]}",
                    "currency": case["currency"],
                    "expected_settlement_date": (tx_date + timedelta(days=1)).isoformat(),
                    "actual_settlement_date": (tx_date + timedelta(days=3)).isoformat(),
                    "break_type": "late_correspondent_confirmation",
                    "break_amount": row["amount"],
                    "owner_team": "payments_reconciliation",
                    "status": "investigating",
                }
            )


def write_news(case: dict[str, Any], accounts: list[dict[str, Any]]) -> Path:
    year = int(case["year"])
    month = int(case["month"])
    published = datetime(year, month, int(case["news_day"]), 9, 20)
    out_dir = BANKING_DIR / f"{year}" / f"{month:02d}" / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [a["full_name"] for a in accounts[:3]]

    body = f"""
{case['org']} said several South African banks had been asked to review a small number of cross-border payments after a correspondent bank raised questions about {case['risk_theme']}.

People familiar with the review said names including {", ".join(names[:-1])} and {names[-1]} appeared in internal case notes shared with participating institutions. The concern is not that every payment is fraudulent, but that the payment trail needs to be matched against SWIFT instructions, beneficiary amendments and nostro settlement records.

One payments compliance specialist said banks were increasingly expected to connect trade, sanctions, reconciliation and customer-account data before deciding whether to file a suspicious-transaction report. "The alert often starts outside the bank, but the evidence sits inside operations data," the person said.

The matter is expected to prompt internal analytics teams to prepare exception views across MT103 messages, intermediary-bank routing, beneficiary history and delayed correspondent confirmations.
""".strip()

    front = f"""---
outlet: {case['outlet']}
outlet_id: {slug(case['outlet'])}
author: "{case['author']}"
published: {published.date().isoformat()}
title: "{case['title']}"
tags: [fraud, correspondent_banking, swift, aml, cross_border_payments]
style: "External payments compliance brief"
department: "{case['department']}"
project_type: {case['project_type']}
---

"""
    path = out_dir / f"{published:%Y%m%d}_{slug(case['title'])}.md"
    path.write_text(front + body + "\n", encoding="utf-8")
    render_news_article_pdf(path)
    return path


def write_email(case: dict[str, Any], accounts: list[dict[str, Any]]) -> Path:
    year = int(case["year"])
    month = int(case["month"])
    sent = datetime(year, month, min(int(case["news_day"]) + 1, 25), 8, 40)
    out_dir = BANKING_DIR / f"{year}" / f"{month:02d}" / "emails"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [a["full_name"] for a in accounts[:3]]

    body = f"""Hi team,

Please prepare a view on the correspondent-banking issue mentioned in the press this week.

The external note mentions {", ".join(names[:-1])} and {names[-1]}. Do not assume guilt from the article. I want us ready before Compliance, Legal and Payments ask for evidence.

Please check whether we have:
- MT103 payments involving {case['beneficiary']} or {case['intermediary_bank']}
- beneficiary amendments or sender-to-receiver notes on the SWIFT messages
- nostro breaks or delayed confirmations around those payments
- any linked customer-account activity that supports or weakens a {case['pattern']} concern

The useful output is a prepared exception pack, not a final accusation.

Regards,
Nandi

Nandi Maseko
Head of Financial Crime Analytics
Keystone Retail Bank
"""

    msg = EmailMessage()
    msg["From"] = "Nandi Maseko <nandi.maseko@fraud.keystonebank.co.za>"
    msg["To"] = "Fraud Analytics <fraud.analytics@keystonebank.co.za>"
    msg["Cc"] = "Payments Compliance <payments.compliance@keystonebank.co.za>, Legal Investigations <legal.investigations@keystonebank.co.za>"
    msg["Subject"] = f"Prepare view - correspondent banking alert for {case['intermediary_bank']}"
    msg["Date"] = email.utils.format_datetime(sent)
    msg["Message-ID"] = f"<correspondent-fraud-prep-{year}{month:02d}@keystonebank.co.za>"
    msg.set_content(body)

    path = out_dir / f"{sent:%Y%m%d-%H%M}-correspondent-fraud-prep.eml"
    path.write_bytes(msg.as_bytes())
    render_outlook_email_pdf(path)
    return path


def main() -> None:
    created = 0
    for case in CASES:
        accounts = load_case_accounts(int(case["year"]), int(case["month"]))
        tx_rows = append_transaction_rows(case, accounts)
        write_supporting_tables(case, accounts, tx_rows)
        write_news(case, accounts)
        write_email(case, accounts)
        created += 1
    print(f"Created {created} correspondent-banking fraud preparation cases.")


if __name__ == "__main__":
    main()
