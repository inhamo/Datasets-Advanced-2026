"""
Create fraud investigation hint artifacts before removing direct fraud labels.

Outputs live under:
  banking_data/YYYY/MM/news/
  banking_data/YYYY/MM/emails/

The artifacts intentionally use customer names in narrative text, not customer IDs,
so analysts have to connect evidence back to customer records through normal
entity-resolution work.
"""

from __future__ import annotations

import email.utils
import random
import re
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd

from commons.pdf_renderers import render_news_article_pdf, render_outlook_email_pdf


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"

CASE_THEMES = {
    "Document Forgery Indicator": {
        "headline": "Micro-lender flags forged payslip ring after application review",
        "org": "Cape Town Micro Finance Forum",
        "body_hint": "forged payslips, edited bank statements and repeated employer letters",
        "department": "Fraud Risk",
        "project_type": "document_forgery_model",
    },
    "Account Takeover Suspected": {
        "headline": "Mobile operator assists banks after SIM-swap account takeover arrests",
        "org": "SAPS Commercial Crimes Unit",
        "body_hint": "SIM swaps, password resets and rapid beneficiary changes",
        "department": "Digital Security",
        "project_type": "account_takeover_monitoring",
    },
    "High-risk Device Fingerprint": {
        "headline": "Payment gateway warns lenders about device fingerprint cluster",
        "org": "Payments Association of South Africa",
        "body_hint": "shared devices, repeated browser fingerprints and unusual login locations",
        "department": "Cyber Fraud",
        "project_type": "device_risk_graph",
    },
    "Synthetic Identity Pattern": {
        "headline": "Insurer uncovers synthetic identity pattern across low-value policies",
        "org": "Insurance Crime Bureau",
        "body_hint": "thin-file profiles, recycled addresses and inconsistent identity documents",
        "department": "Financial Crime",
        "project_type": "synthetic_identity_detection",
    },
    "Velocity Cash-out": {
        "headline": "Retail group reports rapid cash-out pattern after card refund abuse",
        "org": "Retail Risk Council",
        "body_hint": "rapid transfers, cash withdrawals and wallet cash-outs after credits land",
        "department": "Transaction Monitoring",
        "project_type": "cashout_velocity_rules",
    },
    "Mule Account Behavior": {
        "headline": "University warns banks about student accounts used as money mule routes",
        "org": "University Financial Aid Office",
        "body_hint": "incoming third-party deposits followed by fast outbound transfers",
        "department": "AML Operations",
        "project_type": "mule_account_detection",
    },
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for path in sorted(BANKING_DIR.glob("20*/??/customers_*.parquet")):
        year = int(path.parts[-3])
        month = int(path.parts[-2])
        df = pd.read_parquet(path)
        if "is_fraudster" not in df.columns or "fraud_type" not in df.columns:
            continue
        fraud = df[df["is_fraudster"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
        fraud = fraud[fraud["fraud_type"].astype(str).isin(CASE_THEMES.keys())]
        for fraud_type, group in fraud.groupby("fraud_type"):
            if fraud_type in seen_types:
                continue
            picked = group.sort_values("full_name").head(3)
            if picked.empty:
                continue
            cases.append(
                {
                    "year": year,
                    "month": month,
                    "fraud_type": fraud_type,
                    "names": [str(x) for x in picked["full_name"].dropna().tolist()],
                }
            )
            seen_types.add(fraud_type)
        if len(seen_types) == len(CASE_THEMES):
            break

    # Spread cases beyond the first months while keeping real sampled names.
    schedule = [(2019, 3), (2020, 7), (2021, 11), (2022, 8), (2024, 2), (2025, 5)]
    for case, (year, month) in zip(cases, schedule):
        case["year"] = year
        case["month"] = month
    return cases


def write_news(case: dict[str, Any], idx: int) -> Path:
    theme = CASE_THEMES[case["fraud_type"]]
    names = case["names"]
    year = int(case["year"])
    month = int(case["month"])
    published = datetime(year, month, min(8 + idx * 2, 24), 9, 30)
    out_dir = BANKING_DIR / f"{year}" / f"{month:02d}" / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    title = theme["headline"]
    body = f"""
{theme['org']} said it had referred a small cluster of suspected retail-banking fraud cases to industry investigators after names surfaced across separate applications and transaction reviews.

People familiar with the referral said {", ".join(names[:-1])} and {names[-1]} appeared in case notes shared with participating financial institutions. The concern centred on {theme['body_hint']}, although investigators cautioned that banks would need to confirm the pattern against their own account and transaction records.

One investigator said the matter was useful because it did not look like a single obvious fraud event. "The signals sit across onboarding, device behaviour, payment velocity and customer contact history. No one table tells the full story," the person said.

The referral is expected to increase pressure on banks to improve internal fraud typologies, evidence packs and exception dashboards for operational teams.
""".strip()

    front = f"""---
outlet: Business Day
outlet_id: business_day
author: "Staff Reporter"
published: {published.date().isoformat()}
title: "{title}"
tags: [fraud, financial_crime, customer_risk]
style: "External fraud brief"
department: "{theme['department']}"
project_type: {theme['project_type']}
---

"""
    path = out_dir / f"{published:%Y%m%d}_{slug(title)}.md"
    path.write_text(front + body + "\n", encoding="utf-8")
    render_news_article_pdf(path)
    return path


def write_email(case: dict[str, Any], idx: int) -> Path:
    theme = CASE_THEMES[case["fraud_type"]]
    names = case["names"]
    year = int(case["year"])
    month = int(case["month"])
    sent = datetime(year, month, min(10 + idx * 2, 25), 8 + idx % 4, 15)
    out_dir = BANKING_DIR / f"{year}" / f"{month:02d}" / "emails"
    out_dir.mkdir(parents=True, exist_ok=True)

    subject = f"Tip-off from {theme['org']} - need a view on similar customers"
    body = f"""Hi team,

We had a tip-off from {theme['org']} and I need a quick view before Legal and the police liaison meeting.

The names mentioned were {", ".join(names[:-1])} and {names[-1]}. Please do not treat this as proof on its own. I want us to check whether our own data shows the same pattern: {theme['body_hint']}.

Can Fraud Analytics pull a view across onboarding, customer contact, device/session behaviour, payments and account movement? Legal will want an evidence trail and SAPS may ask us to explain why these customers were or were not escalated.

Please come back with:
- any matching internal customers by name and contact details
- transaction behaviours that support or weaken the concern
- whether we need a typology or dashboard for {case['fraud_type'].lower()}
- recommended next step for Legal, AML and Customer Operations

Regards,
Nandi

Nandi Maseko
Head of Financial Crime Analytics
Keystone Retail Bank
"""

    msg = EmailMessage()
    msg["From"] = "Nandi Maseko <nandi.maseko@fraud.keystonebank.co.za>"
    msg["To"] = "Fraud Analytics <fraud.analytics@keystonebank.co.za>"
    msg["Cc"] = "Legal Investigations <legal.investigations@keystonebank.co.za>, Police Liaison <police.liaison@keystonebank.co.za>"
    msg["Subject"] = subject
    msg["Date"] = email.utils.format_datetime(sent)
    msg["Message-ID"] = f"<fraud-tipoff-{year}{month:02d}-{idx}@keystonebank.co.za>"
    msg.set_content(body)

    path = out_dir / f"{sent:%Y%m%d-%H%M}-fraud-tipoff-{idx:02d}.eml"
    path.write_bytes(msg.as_bytes())
    render_outlook_email_pdf(path)
    return path


def main() -> None:
    cases = load_cases()
    for idx, case in enumerate(cases, start=1):
        write_news(case, idx)
        write_email(case, idx)
    print(f"Created {len(cases)} fraud hint news articles and {len(cases)} internal tip-off emails.")


if __name__ == "__main__":
    main()
