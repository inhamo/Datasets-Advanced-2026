"""
Generate realistic internal emails, news articles, and macro events for a DE/DA pipeline narrative.

Outputs under corpus_context/:
  emails/{year}/{month}/*.eml
  news/{year}/{month}/*.md
  events/macro_events.jsonl
  index/monthly_signals.csv
  index/corpus_manifest.csv

Anchors narratives to loan_payment_transactions stats when available.
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import json
import random
import re
import uuid
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd

from commons.pdf_renderers import render_news_article_pdf, render_outlook_email_pdf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"
CORPUS_DIR = BASE_DIR / "corpus_context"
COMMONS = BASE_DIR / "commons"

PRIME_BY_YEAR = {2019: 10.0, 2020: 8.25, 2021: 7.0, 2022: 8.75, 2023: 11.75, 2024: 11.75, 2025: 11.0}

MONTH_FILE = re.compile(r"loan_payment_transactions_(\d{4})_(\d{2})\.(csv|parquet)$")

NEWS_IMAGE_ASSETS = {
    "Loan Products": ("assets/news_images/atm-queue.jpg", "Customers using an ATM queue outside a retail banking point"),
    "Finance Reconciliation": ("assets/news_images/finance-documents.jpg", "Finance documents and bank statement checks on a desk"),
    "Retail Credit Risk": ("assets/news_images/business-meeting.jpg", "Credit risk analysts discussing portfolio data"),
    "Regulatory Compliance": ("assets/news_images/finance-documents.jpg", "Compliance evidence and customer fee documentation"),
    "Products & Remittances": ("assets/news_images/mobile-banking.jpg", "Customer using mobile banking on a phone"),
    "Banking Operations": ("assets/news_images/operations-dashboard.jpg", "Operations dashboard showing transaction monitoring"),
    "Executive Office": ("assets/news_images/city-finance.jpg", "Johannesburg financial district skyline"),
    "Data Engineering": ("assets/news_images/server-room.jpg", "Banking data infrastructure and server racks"),
    "Customer Experience": ("assets/news_images/customer-service.jpg", "Customer service team handling banking queries"),
}


@dataclass
class MonthSignal:
    year: int
    month: int
    tx_count: int
    fail_pct: float
    insuf_count: int
    timeout_count: int
    data_error_pct: float
    top_loan_type: str
    yoy_tx_growth_pct: float | None


def load_json(name: str) -> dict[str, Any]:
    return json.loads((COMMONS / name).read_text(encoding="utf-8"))


def load_monthly_signals() -> list[MonthSignal]:
    by_month: dict[tuple[int, int], Path] = {}
    for path in DATA_DIR.rglob("loan_payment_transactions_*.*"):
        m = MONTH_FILE.search(path.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        existing = by_month.get(key)
        if existing is None or path.suffix.lower() == ".parquet":
            by_month[key] = path

    parsed: list[dict[str, Any]] = []
    for (year, month), path in sorted(by_month.items()):
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if df.empty:
            continue
        fail_pct = float((df["status"] == "Failed").mean() * 100) if "status" in df else 0.0
        insuf = int((df.get("failure_reason") == "insufficient_funds").sum())
        timeout = int((df.get("failure_reason") == "bank_timeout").sum())
        de_pct = 0.0
        if "has_data_error" in df.columns:
            de_pct = float(df["has_data_error"].fillna(False).astype(bool).mean() * 100)
        top_loan = ""
        if "loan_type" in df.columns:
            top_loan = str(df["loan_type"].value_counts().idxmax())
        parsed.append(
            {
                "year": year,
                "month": month,
                "tx_count": len(df),
                "fail_pct": round(fail_pct, 2),
                "insuf_count": insuf,
                "timeout_count": timeout,
                "data_error_pct": round(de_pct, 2),
                "top_loan_type": top_loan,
            }
        )

    by_key = {(r["year"], r["month"]): r for r in parsed}
    signals: list[MonthSignal] = []
    for key in sorted(by_key):
        r = by_key[key]
        yoy = None
        prev = by_key.get((r["year"] - 1, r["month"]))
        if prev and prev["tx_count"]:
            yoy = round((r["tx_count"] - prev["tx_count"]) / prev["tx_count"] * 100, 1)
        signals.append(
            MonthSignal(
                year=r["year"],
                month=r["month"],
                tx_count=r["tx_count"],
                fail_pct=r["fail_pct"],
                insuf_count=r["insuf_count"],
                timeout_count=r["timeout_count"],
                data_error_pct=r["data_error_pct"],
                top_loan_type=r["top_loan_type"],
                yoy_tx_growth_pct=yoy,
            )
        )
    return signals


def events_for_month(events: list[dict], year: int, month: int) -> list[dict]:
    out = []
    for ev in events:
        d = date.fromisoformat(ev["date"])
        if d.year == year and d.month == month:
            out.append(ev)
    return out


def person_email(person: dict, teams: dict) -> str:
    domain = teams["email_domain"]
    team = next(t for t in teams["teams"] if t["id"] == person["team"])
    return f"{person['first'].lower()}.{person['last'].replace(' ', '').lower()}@{team['mailbox']}.{domain}"


def team_mailbox(team_id: str, teams: dict) -> str:
    team = next(t for t in teams["teams"] if t["id"] == team_id)
    return f"{team['mailbox']}@{teams['email_domain']}"


def write_eml(path: Path, msg: EmailMessage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(msg.as_bytes())


def cleanup_email_outputs(month_dir: Path) -> None:
    email_dir = month_dir / "emails"
    if not email_dir.exists():
        return
    for path in email_dir.glob("*"):
        if path.is_file() and path.suffix.lower() in {".eml", ".pdf"}:
            path.unlink()


def cleanup_news_outputs(month_dir: Path) -> None:
    news_dir = month_dir / "news"
    if not news_dir.exists():
        return
    for path in news_dir.glob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".pdf", ".json"}:
            path.unlink()


def news_image_for(department: str) -> tuple[str, str]:
    return NEWS_IMAGE_ASSETS.get(
        department,
        ("assets/news_images/city-finance.jpg", "South African banking and finance scene"),
    )


def build_email(
    *,
    subject: str,
    body: str,
    sender: str,
    to: list[str],
    cc: list[str] | None,
    sent: datetime,
    in_reply_to: str | None = None,
    references: list[str] | None = None,
) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject
    msg["Date"] = email.utils.format_datetime(sent)
    mid = f"<{uuid.uuid4().hex}@keystonebank.co.za>"
    msg["Message-ID"] = mid
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = " ".join(references)
    msg.set_content(body.strip() + "\r\n")
    return msg


def project_request_scenarios(
    sig: MonthSignal,
    teams_cfg: dict,
    people: list[dict],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Standing project-style stakeholder asks across business departments."""
    y, m = sig.year, sig.month
    month_label = date(y, m, 1).strftime("%B %Y")
    quarter = (m - 1) // 3 + 1
    project_code = f"KB-{y}{m:02d}"

    def p(team_id: str) -> str:
        matches = [x for x in people if x["team"] == team_id]
        return person_email(rng.choice(matches), teams_cfg)

    def mb(team_id: str) -> str:
        return team_mailbox(team_id, teams_cfg)

    return [
        {
            "day": rng.randint(3, 8),
            "from": p("executive"),
            "to": [mb("analytics"), mb("data_engineering")],
            "cc": [mb("cfo")],
            "department": "Executive Office",
            "project_type": "dashboard_request",
            "subject": f"{project_code} EXCO dashboard request - FinTech portfolio pulse",
            "body": f"""Team,

Please treat this as a project brief for the {month_label} EXCO pack.

Business question:
Which customer, channel and product movements need management attention before the next operating review?

Required dashboard:
1. Active customers and transaction volume by channel
2. Loan repayment failure rate and NSF concentration
3. Reconciliation exception count by root cause
4. Fraud / anomaly trend and operational timeout view
5. Product adoption and dormant-account movement

Data required:
- multichannel transactions
- loan payment transactions
- bank statement reconciliation outputs
- customer/account dimensions

Deliverable:
Power BI dashboard plus a one-page executive summary. Please include a clear red / amber / green status for each KPI.

Success measure:
EXCO must be able to see the top three actions without asking for a spreadsheet export.

Regards,
Executive Office""",
        },
        {
            "day": rng.randint(4, 10),
            "from": p("loan_department"),
            "to": [mb("analytics"), mb("credit_risk")],
            "cc": [mb("collections")],
            "department": "Loan Products",
            "project_type": "loan_portfolio_analysis",
            "subject": f"{project_code} Loan department request - vintage and arrears dashboard",
            "body": f"""Hi Analytics,

The loan department needs a project view for {month_label}. We want to understand whether the current repayment pattern is product risk, customer affordability risk, or collection timing.

Context:
The loan payment mart has {sig.tx_count:,} rows this month with a {sig.fail_pct:.1f}% failure rate. NSF count is {sig.insuf_count:,}; bank timeout count is {sig.timeout_count:,}.

Analysis required:
1. Arrears by loan type, vintage, province and mandate day
2. Roll-forward from successful debit to failed debit
3. Customers with two or more failed payments in 90 days
4. Early warning score using income, product, channel and recent behaviour

Dataset joins:
loans, debit orders, loan payment transactions, customers, accounts and multichannel transactions.

Deliverable:
A loan portfolio dashboard and a scored customer watchlist for Collections.

Deadline:
First working version before the Q{quarter} credit committee prep meeting.

Mandla""",
        },
        {
            "day": rng.randint(6, 12),
            "from": p("finance_recon"),
            "to": [mb("data_engineering"), mb("analytics")],
            "cc": [mb("cfo"), mb("payments")],
            "department": "Finance Reconciliation",
            "project_type": "bank_reconciliation",
            "subject": f"{project_code} Bank reconciliation project - statement exceptions and GL mapping",
            "body": f"""Data Engineering / Analytics,

Finance needs a reconciliation dataset and dashboard for customer bank statements generated from the transaction files.

Please build for each account:
1. Bank statement source ingestion status by format: ISO 20022, CSV, TXT and PDF
2. Matched ledger vs statement transactions
3. Timing differences and source-system lag days
4. Bank charges, service fees, interest income, withholding tax and cash deposit fees
5. Missing-in-ledger and missing-in-bank-statement exceptions
6. GL mapping for reconciled, unreconciled and non-transactional entries

Expected output:
- reconciliation_results
- reconciliation_exceptions
- non_transactional_statement_entries
- journal export template for manual GL upload
- dashboard showing auto-match rate against the 95% target

Please keep bank fees and interest separate from transactional breaks. They are not data errors; they are finance adjustments.

Regards,
Finance Reconciliation""",
        },
        {
            "day": rng.randint(7, 14),
            "from": p("compliance"),
            "to": [mb("analytics"), mb("fraud")],
            "cc": [mb("data_engineering")],
            "department": "Regulatory Compliance",
            "project_type": "regulatory_monitoring",
            "subject": f"{project_code} Compliance monitoring request - customer conduct and exception evidence",
            "body": f"""Team,

Compliance needs a monitoring pack that can be traced back to raw records.

Use cases:
1. Customers repeatedly charged fees after failed collections
2. Retry behaviour around debit order failures
3. High-risk transaction anomalies linked to account takeover patterns
4. Exceptions where statement data and ledger data disagree
5. Customers impacted by late processing or source-system lag

Required controls:
- row-level lineage from dashboard KPI to source file
- exception category and root cause tag
- customer/account identifiers masked in executive views
- evidence extract for audit sampling

Deliverable:
Compliance dashboard, audit extract and data dictionary section.

Michael""",
        },
        {
            "day": rng.randint(8, 15),
            "from": p("product"),
            "to": [mb("analytics"), mb("data_engineering")],
            "cc": [mb("operations")],
            "department": "Products & Remittances",
            "project_type": "product_analytics",
            "subject": f"{project_code} Product analytics request - adoption, remittance and channel behaviour",
            "body": f"""Hi team,

Product wants a project dataset for digital product performance and remittance targeting.

Business questions:
1. Which customers are shifting from branch / ATM to app, USSD and e-wallet?
2. Which segments should receive remittance or wallet prompts?
3. What channels show failed journeys or timeout friction?
4. Which merchant/category patterns predict repeat digital usage?

Requested outputs:
- monthly product adoption mart
- customer segment table
- channel migration dashboard
- campaign target list with opt-out exclusions

Please include customer tenure, account tier, province, channel mix, transaction category mix and failed transaction history.

Thanks,
Products""",
        },
        {
            "day": rng.randint(9, 16),
            "from": p("operations"),
            "to": [mb("data_engineering"), mb("payments")],
            "cc": [mb("analytics"), mb("infrastructure")],
            "department": "Banking Operations",
            "project_type": "operations_sla",
            "subject": f"{project_code} Operations SLA dashboard - processing delays and service failures",
            "body": f"""Ops needs a dashboard for daily service delivery.

Please build:
1. Transaction volume by hour, channel and status
2. Bank timeout trend and affected accounts
3. Statement processing lag by source file and bank
4. Failed debit-order retries by day
5. Load-shedding and infrastructure context where available

The current month has {sig.timeout_count:,} timeout-related loan payment rows. We need this joined to multichannel authorization times and bank statement posting delays.

Success measure:
Ops can identify the worst processing day, affected channels and downstream reconciliation impact within 30 seconds.

Peter""",
        },
        {
            "day": rng.randint(10, 18),
            "from": p("fraud"),
            "to": [mb("analytics"), mb("compliance")],
            "cc": [mb("credit_risk")],
            "department": "Financial Crime",
            "project_type": "fraud_risk_monitoring",
            "subject": f"{project_code} Fraud analytics request - anomaly-to-case monitoring",
            "body": f"""Analytics,

Financial Crime needs a monitoring view that separates suspicious behaviour from normal reconciliation timing issues.

Please include:
1. Account takeover indicators: new device, unusual location, high-risk transfer
2. Travel-time and velocity anomalies
3. Card testing and mule-account behaviour
4. False-positive controls for delayed bank posting
5. Case queue grouped by customer, account and confidence band

Important:
Do not classify normal bank charges, interest, withholding tax or statement lag as fraud. They must remain finance/reconciliation categories.

Deliverable:
Fraud operations dashboard and a scored case extract.

Aisha""",
        },
        {
            "day": rng.randint(11, 19),
            "from": p("customer_experience"),
            "to": [mb("analytics"), mb("product")],
            "cc": [mb("operations")],
            "department": "Customer Experience",
            "project_type": "customer_experience",
            "subject": f"{project_code} CX insight request - fees, failed payments and complaint drivers",
            "body": f"""Hi,

Customer Experience needs a case-style analysis for the next service review.

Problem statement:
Customers complain when they see fees, failed payments or delayed posting without a clear explanation.

Please analyse:
1. Accounts with bank charges after failed debit orders
2. Customers with repeated NSF and retry attempts
3. Transactions posted later than the customer expected
4. Product/channel segments with the highest failed journey rate
5. Impact on balances after fees, interest and reversals

Output:
CX dashboard, top five complaint drivers and recommended customer messaging triggers.

Zanele""",
        },
        {
            "day": rng.randint(12, 22),
            "from": p("credit_risk"),
            "to": [mb("analytics"), mb("loan_department")],
            "cc": [mb("collections")],
            "department": "Retail Credit Risk",
            "project_type": "risk_model",
            "subject": f"{project_code} Credit risk model request - repayment stress indicators",
            "body": f"""Analytics team,

Credit Risk needs a model-ready dataset for repayment stress.

Target definition:
Customer has a failed loan repayment, NSF event or collections retry in the following 30 days.

Features requested:
- prior 90-day transaction count and net movement
- failed payment history
- debit order day vs salary/payday window
- account balance trend
- bank charges and fee pressure
- channel behaviour and digital activity
- previous reconciliation exceptions

Deliverable:
Feature table, model notebook outline and lift chart by decile.

Johan""",
        },
        {
            "day": rng.randint(13, 23),
            "from": p("payments"),
            "to": [mb("data_engineering"), mb("finance_recon")],
            "cc": [mb("operations")],
            "department": "Payments Operations",
            "project_type": "settlement_reconciliation",
            "subject": f"{project_code} Payments request - settlement lag and retry reconciliation",
            "body": f"""Data Engineering,

Payments needs a settlement reconciliation dataset.

Please identify:
1. Debit orders submitted but not yet reflected on the bank statement
2. Bank statement items posted one to four days after ledger date
3. Duplicate settlement records
4. Reversals and failed retry attempts
5. Bank timeout days and downstream reconciliation impact

The current loan-payment failure rate is {sig.fail_pct:.1f}% and NSF count is {sig.insuf_count:,}. Please separate true payment failure from late bank processing.

Deliverable:
Settlement lag dashboard plus exception extract for sponsor-bank follow-up.

Priya""",
        },
        {
            "day": rng.randint(14, 24),
            "from": p("data_engineering"),
            "to": [mb("analytics"), mb("finance_recon"), mb("compliance")],
            "cc": [mb("executive")],
            "department": "Data Engineering",
            "project_type": "lakehouse_build",
            "subject": f"{project_code} Lakehouse project - source-to-dashboard dataset contract",
            "body": f"""Team,

Data Engineering is setting up the lakehouse contract for the next analytics sprint.

Proposed layers:
Bronze:
- raw transactions
- raw bank statement files: ISO 20022 XML, CSV, TXT and PDF
- raw loan payment/debit order files

Silver:
- standardized bank statement entries
- standardized internal ledger transactions
- customer/account dimensions
- reconciliation matches and exceptions

Gold:
- EXCO dashboard mart
- reconciliation scorecard
- loan risk feature table
- product/channel KPI mart
- compliance evidence extract

Please reply with any extra fields needed before we freeze the schema.

Thabo""",
        },
    ]


def project_request_scenarios(
    sig: MonthSignal,
    teams_cfg: dict,
    people: list[dict],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Human, client-like stakeholder asks across departments."""
    y, m = sig.year, sig.month
    month_label = date(y, m, 1).strftime("%B %Y")
    project_code = f"KB {y}{m:02d}"

    def p(team_id: str) -> str:
        matches = [x for x in people if x["team"] == team_id]
        return person_email(rng.choice(matches), teams_cfg)

    def mb(team_id: str) -> str:
        return team_mailbox(team_id, teams_cfg)

    deliverables = ["an Excel file I can filter myself", "a rough dashboard", "a short presentation", "a one-page summary", "a customer list", "an extract we can check manually"]

    return [
        {
            "day": rng.randint(3, 8),
            "from": p("executive"),
            "to": [mb("analytics")],
            "cc": [mb("data_engineering"), mb("cfo")],
            "department": "Executive Office",
            "project_type": "dashboard_request",
            "subject": f"{project_code} quick view for Monday meeting",
            "body": f"""Hi Sarah,

Do we have a simple view of how the business is looking for {month_label}? I do not need a perfect model, just something I can use in Monday's meeting.

The questions will probably be around customer activity, loan payments, failed transactions and whether the bank statement work is showing anything worrying.

Could you start with {rng.choice(deliverables)}? If the numbers are not final, please mark them as draft.

I know the data sits in different places, so use whatever is available and call out the gaps. I mainly need the story and the top issues.

Thanks,
Kabelo""",
        },
        {
            "day": rng.randint(4, 10),
            "from": p("loan_department"),
            "to": [mb("analytics")],
            "cc": [mb("credit_risk"), mb("collections")],
            "department": "Loan Products",
            "project_type": "loan_portfolio_analysis",
            "subject": f"{project_code} help with loan payment questions",
            "body": f"""Hi,

Can someone help us understand what is happening with loan repayments this month?

I am not sure which file you use for this, but we need to see the failed debits, customers who missed more than once, and whether the problem is mostly affordability or just timing. The failure rate I heard was around {sig.fail_pct:.1f}% but I do not know if that is final.

Could we get {rng.choice(deliverables)} by product and province? A simple first version is fine.

Collections also asked if we can flag customers who might need a call before the next debit run.

Regards,
Mandla""",
        },
        {
            "day": rng.randint(5, 11),
            "from": p("finance_recon"),
            "to": [mb("data_engineering")],
            "cc": [mb("analytics"), mb("cfo")],
            "department": "Finance Reconciliation",
            "project_type": "bank_reconciliation",
            "subject": f"{project_code} bank statement matching issue",
            "body": f"""Morning Thabo,

Finance is trying to compare the customer bank statements to what we have in the transaction files, but there are too many small differences to check by hand.

Some are normal things like bank charges, interest, tax and fees. Others look like timing delays where one system posted later than the other. I do not want those mixed together.

Can your team create {rng.choice(deliverables)} that shows what matched, what did not match, what is just a bank fee or interest, and what looks like a real break?

Please keep it understandable for Finance. We do not need technical column names in the front view.

Thanks,
Gugu""",
        },
        {
            "day": rng.randint(6, 12),
            "from": p("compliance"),
            "to": [mb("analytics")],
            "cc": [mb("fraud"), mb("data_engineering")],
            "department": "Regulatory Compliance",
            "project_type": "regulatory_monitoring",
            "subject": f"{project_code} evidence pack for customer impact review",
            "body": f"""Hi team,

We have a conduct review coming up and I need help pulling together the evidence.

The concern is not one specific system. It is more about whether customers are being charged fees after failed payments, whether retries are reasonable, and whether we can explain late postings if someone asks.

Can you prepare {rng.choice(deliverables)} with a few examples and the summary counts? Please mask customer names in anything that goes to the wider group.

I do not need a fraud investigation. I need enough to show that we can trace the numbers back to source records.

Regards,
Michael""",
        },
        {
            "day": rng.randint(7, 14),
            "from": p("product"),
            "to": [mb("analytics")],
            "cc": [mb("operations")],
            "department": "Products & Remittances",
            "project_type": "product_analytics",
            "subject": f"{project_code} product usage view",
            "body": f"""Hi Emma,

We are looking at wallet and remittance prompts again. Before we make campaign decisions, can we see who is actually using digital channels and who is still mostly branch, ATM or card?

I am not sure if this sits with Product or Operations, but we need channel mix, transaction types, customers with failed journeys, customers who may be under fee pressure, and segments that should not be targeted.

Could you make {rng.choice(deliverables)}? It does not need to be beautiful yet. I just need something we can argue with.

Thanks,
Farah""",
        },
        {
            "day": rng.randint(8, 15),
            "from": p("operations"),
            "to": [mb("payments"), mb("data_engineering")],
            "cc": [mb("analytics"), mb("infrastructure")],
            "department": "Banking Operations",
            "project_type": "operations_sla",
            "subject": f"{project_code} where are the delays coming from",
            "body": f"""Hi Priya / Thabo,

Ops is getting questions about processing delays. We can see failures, but we cannot easily tell whether the issue is the payment switch, the bank response, the ledger posting or the statement arriving late.

Can you help us with {rng.choice(deliverables)} for the month?

Please show the worst days, channels affected and whether the delay created reconciliation breaks later. Timeout count I have is {sig.timeout_count:,}, but please check your side because I may be looking at the wrong report.

Peter""",
        },
        {
            "day": rng.randint(9, 17),
            "from": p("fraud"),
            "to": [mb("analytics")],
            "cc": [mb("compliance"), mb("credit_risk")],
            "department": "Financial Crime",
            "project_type": "fraud_risk_monitoring",
            "subject": f"{project_code} suspicious activity view",
            "body": f"""Hi,

Can we get a better view of suspicious activity without mixing it up with normal bank processing delays?

For example, a late statement posting should not become a fraud case. But a new device, strange location, rapid transfer or card testing pattern should be visible.

Could you put together {rng.choice(deliverables)} with high, medium and low priority cases? It would help if we can click from the summary to the transactions behind it.

This does not need to be final for model use. It is for the operations team to review.

Aisha""",
        },
        {
            "day": rng.randint(10, 18),
            "from": p("customer_experience"),
            "to": [mb("analytics"), mb("product")],
            "cc": [mb("operations")],
            "department": "Customer Experience",
            "project_type": "customer_experience",
            "subject": f"{project_code} customer complaints around fees",
            "body": f"""Hi team,

We are seeing complaints where customers say they do not understand why a fee appeared after a failed payment or why a transaction only showed later.

Can someone help us with {rng.choice(deliverables)}?

What I need is quite basic: how many customers had fees after failed payments, whether those customers also had late postings, examples we can use for customer messaging, and any obvious product or channel pattern.

I do not need deep modelling. I need something human enough for service teams to use.

Thanks,
Zanele""",
        },
        {
            "day": rng.randint(11, 19),
            "from": p("credit_risk"),
            "to": [mb("analytics")],
            "cc": [mb("loan_department"), mb("collections")],
            "department": "Retail Credit Risk",
            "project_type": "risk_model",
            "subject": f"{project_code} early warning idea",
            "body": f"""Hi Analytics,

Could we test a simple early warning view for repayment stress?

I am thinking of customers who had failed debits, low balances, fees piling up, or a sudden change in transaction behaviour. It does not have to be a full model yet.

Can you start with {rng.choice(deliverables)} showing the top risk customers and the reason they were flagged?

Please separate actual affordability stress from operational delays. Otherwise Credit will chase the wrong customers.

Johan""",
        },
        {
            "day": rng.randint(12, 20),
            "from": p("payments"),
            "to": [mb("data_engineering")],
            "cc": [mb("finance_recon"), mb("operations")],
            "department": "Payments Operations",
            "project_type": "settlement_reconciliation",
            "subject": f"{project_code} settlement follow up",
            "body": f"""Hi Thabo,

We need to follow up with the sponsor bank, but before we do that I need to know which items are really late and which ones failed properly.

Can you pull {rng.choice(deliverables)} for debit orders that appeared in our system but not on the bank statement, items that posted one to four days later, duplicate settlement-looking records, reversals and retries?

Please keep the wording simple. The bank team will not understand our internal table names.

Thanks,
Priya""",
        },
        {
            "day": rng.randint(13, 22),
            "from": p("data_engineering"),
            "to": [mb("analytics"), mb("finance_recon"), mb("compliance")],
            "cc": [mb("executive")],
            "department": "Data Engineering",
            "project_type": "lakehouse_build",
            "subject": f"{project_code} can we agree the data we need",
            "body": f"""Hi all,

Before we build more dashboards, can we agree what data everyone actually needs?

From what I can tell, Finance needs statement matching, Compliance needs evidence, Product wants customer segments, Ops wants delays, and Credit wants risk flags. These are related but not the same thing.

Can each team confirm the fields they need by Friday? We will create one shared structure with raw files, cleaned transactions, reconciled results and the final reporting tables.

Please avoid sending only screenshots. If you have a sample Excel layout, send that too because it helps us understand the final shape.

Thabo""",
        },
    ]


def clean_email_text(text: str) -> str:
    return (
        text.replace("_", " ")
        .replace("`", "")
        .replace("Parquet", "data file")
        .replace("Airflow", "the scheduled load")
        .replace("Snowflake", "the reporting database")
        .replace("loan payment transactions", "loan payment files")
        .replace("Loan payment transactions mart", "Loan payment reporting file")
        .replace("loan payment transactions mart", "loan payment reporting file")
        .replace("Loan payment mart", "Loan payment reporting file")
        .replace("loan payment mart", "loan payment reporting file")
    )


def email_scenarios(
    sig: MonthSignal,
    month_events: list[dict],
    teams_cfg: dict,
    people: list[dict],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Return list of email specs (subject, body, from_team, to_teams, day)."""
    y, m = sig.year, sig.month
    prime = PRIME_BY_YEAR.get(y, 10.5)
    bank = teams_cfg["bank_name"]
    specs: list[dict[str, Any]] = []

    def p(team_id: str) -> str:
        matches = [x for x in people if x["team"] == team_id]
        return person_email(rng.choice(matches), teams_cfg)

    specs.extend(project_request_scenarios(sig, teams_cfg, people, rng))

    # --- Data-driven collections / payments ---
    if sig.fail_pct >= 6.0:
        specs.append(
            {
                "day": rng.randint(8, 20),
                "from": p("collections"),
                "to": [team_mailbox("payments", teams_cfg), team_mailbox("analytics", teams_cfg)],
                "cc": [team_mailbox("credit_risk", teams_cfg)],
                "subject": f"Debit order failure rate {sig.fail_pct:.1f}% - {y}-{m:02d} loan repayments",
                "body": f"""Hi team,

Our daily collections extract for loan-linked debit orders shows a failure rate of **{sig.fail_pct:.1f}%** for {date(y, m, 1).strftime('%B %Y')} ({sig.tx_count:,} payment rows in the mart).

Breakdown highlights:
• Insufficient funds: {sig.insuf_count:,} postings
• Bank timeout: {sig.timeout_count:,} postings

This is above our 5.5% internal trigger. Can Payments Ops confirm whether NAEDO retries are clearing on T+2 and whether any sponsor bank latency was logged?

@Analytics – please drop the vintage view (2019 vs 2021 originations) into the shared notebook before Thursday’s stand-up.

Regards,
Collections & Recoveries
{bank}""",
            }
        )

    if sig.insuf_count >= 50 and sig.fail_pct < 6.0:
        specs.append(
            {
                "day": rng.randint(5, 18),
                "from": p("payments"),
                "to": [team_mailbox("collections", teams_cfg)],
                "subject": f"RE: Insufficient funds trend - {sig.insuf_count} NSF cases in loan payments",
                "body": f"""Collections,

Noted {sig.insuf_count} insufficient-funds returns on loan repayment debits this month. Overall failure rate is {sig.fail_pct:.1f}%, which is within tolerance, but NSF is concentrated on **{sig.top_loan_type or 'Home Loan'}** mandates.

We are seeing higher incidence on collection day >= 25 (salary-cycle effect). Recommend holding proactive SMS for T-1 where behaviour score < 620.

Thanks,
Payments Operations""",
            }
        )

    # --- Data engineering / analytics pipeline ---
    if sig.yoy_tx_growth_pct is not None and sig.yoy_tx_growth_pct >= 15:
        specs.append(
            {
                "day": rng.randint(3, 12),
                "from": p("data_engineering"),
                "to": [team_mailbox("analytics", teams_cfg)],
                "cc": [team_mailbox("infrastructure", teams_cfg)],
                "subject": f"Loan payment mart volume +{sig.yoy_tx_growth_pct:.0f}% YoY - capacity check",
                "body": f"""Hi Sarah / Emma,

The `loan_payment_transactions` monthly partition for {y}-{m:02d} landed at **{sig.tx_count:,} rows** (~{sig.yoy_tx_growth_pct:+.1f}% vs same month last year).

Airflow SLA is still green, but the Parquet→Snowflake copy breached 42 minutes on the 3rd run (within our 60m threshold). Please confirm downstream dashboards still use the `_v2` view with `has_data_error` filters applied.

Sipho will add file-size alerts before month-end close.

Thabo
Data Engineering""",
            }
        )

    if sig.data_error_pct >= 2.0:
        specs.append(
            {
                "day": rng.randint(10, 22),
                "from": p("analytics"),
                "to": [team_mailbox("data_engineering", teams_cfg), team_mailbox("compliance", teams_cfg)],
                "subject": f"Data quality - {sig.data_error_pct:.1f}% of loan payments flagged in {y}-{m:02d}",
                "body": f"""Team,

Reconciliation on the loan payment feed shows **{sig.data_error_pct:.1f}%** of rows with `has_data_error=true` (duplicate IDs, truncated descriptions, timestamp shifts).

Before we publish the management collections pack, we need:
1. Exception report by `data_error_types`
2. Sign-off that regulatory returns exclude corrupt keys

This is blocking the monthly MI pack.

Regards,
Retail Analytics""",
            }
        )

    # --- Macro / policy threaded mail ---
    for ev in month_events:
        if ev["category"] == "monetary_policy":
            specs.append(
                {
                    "day": min(28, date.fromisoformat(ev["date"]).day + rng.randint(1, 4)),
                    "from": p("cfo"),
                    "to": [team_mailbox("credit_risk", teams_cfg), team_mailbox("collections", teams_cfg)],
                    "subject": f"Prime / repo move - reprice assumptions ({ev['title'][:40]})",
                    "body": f"""All,

Following: {ev['title']} ({ev['date']}).

Our modelled prime for {y} remains **{prime:.2f}%** on booked variable home loans. Please refresh affordability stress in the origination notebook and alert Collections if early arrears move >10% WoW on 2023 vintages.

ALM will circulate revised FTP curves tomorrow.

Lerato Khumalo
CFO Office""",
                }
            )
        elif ev["category"] == "health" and y == 2020:
            specs.append(
                {
                    "day": min(28, max(1, date.fromisoformat(ev["date"]).day + 2)),
                    "from": p("collections"),
                    "to": [team_mailbox("compliance", teams_cfg), team_mailbox("credit_risk", teams_cfg)],
                    "cc": [team_mailbox("payments", teams_cfg)],
                    "subject": "COVID hardship moratorium - loan debit order handling",
                    "body": f"""Compliance / Risk,

With **{ev['title']}** ({ev['date']}), we are pausing new collection litigation and suppressing NAEDO for customers on the approved hardship list.

Payments data still shows failed debits where customers have not enrolled – see failure rate {sig.fail_pct:.1f}% this month. Please confirm NCR guidance on retry limits.

We need a single source of truth flag in the customer mart by Friday.

Nomsa""",
                }
            )
        elif ev["category"] == "infrastructure" and ev["severity"] in ("high", "critical"):
            specs.append(
                {
                    "day": min(28, date.fromisoformat(ev["date"]).day + rng.randint(0, 3)),
                    "from": p("infrastructure"),
                    "to": [team_mailbox("payments", teams_cfg), team_mailbox("fraud", teams_cfg)],
                    "subject": f"Load shedding / rail impact - {ev['title'][:50]}",
                    "body": f"""Ops,

{ev['title']} ({ev['date']}) is affecting UPS runtime at Joburg DC2. Observed `{sig.timeout_count}` bank_timeout failures on loan payments this month – likely correlated.

ATM and POS decline rates are elevated in multichannel (see fraud dashboard). Failover to CPT active.

David Fourie
Infrastructure""",
                }
            )
        elif ev["category"] == "security" and y == 2021:
            specs.append(
                {
                    "day": 14,
                    "from": p("fraud"),
                    "to": [team_mailbox("infrastructure", teams_cfg), team_mailbox("collections", teams_cfg)],
                    "subject": "KZN/Gauteng unrest - branch and CIT contingency",
                    "body": f"""All,

Branch footprint in affected areas moved to reduced hours. Cash deposit delays may push NSF rates up on loan debits – current month failure **{sig.fail_pct:.1f}%**.

Collections to use SMS template UNREST-2021 before dialling.

Aisha""",
                }
            )

    # --- Standing DE/DA collaboration ---
    if m in (3, 6, 9, 12):
        specs.append(
            {
                "day": rng.randint(20, 28),
                "from": p("analytics"),
                "to": [team_mailbox("data_engineering", teams_cfg)],
                "subject": f"Q{(m - 1) // 3 + 1} loan portfolio MI - data request for {y}",
                "body": f"""Thabo / Sipho,

For quarterly board pack we need from `banking_data`:
• Monthly loan payment failure rate (currently {sig.fail_pct:.1f}% for {y}-{m:02d})
• Top failure_reason distribution
• Channel split (Automated vs digital)

Please snapshot the curated Parquet paths under `/{y}/{m:02d}/` and register in the data catalogue.

Thanks,
Sarah""",
            }
        )

    # --- Go-live / onboarding (early book) ---
    if y == 2019 and m == 2:
        specs.append(
            {
                "day": 11,
                "from": p("data_engineering"),
                "to": [team_mailbox("analytics", teams_cfg), team_mailbox("payments", teams_cfg)],
                "subject": "GO-LIVE: Loan payment transactions mart (2019 vintage)",
                "body": f"""Team,

The first production drop of **loan_payment_transactions** is in `banking_data/2019/02/` ({sig.tx_count:,} rows).

Schema includes installment_number, debit_order_id, failure_reason. Please treat as SoR for collections MI until the general ledger bridge is delivered in Q3.

Thabo""",
            }
        )

    # Ensure at least 2 emails per month
    while len(specs) < 2:
        specs.append(
            {
                "day": rng.randint(4, 24),
                "from": p("analytics"),
                "to": [team_mailbox("collections", teams_cfg)],
                "subject": f"Monthly loan payment KPIs - {y}-{m:02d}",
                "body": f"""Hi,

Quick stats from the mart:
• Volume: {sig.tx_count:,}
• Failure rate: {sig.fail_pct:.1f}%
• Dominant product: {sig.top_loan_type or 'N/A'}

Dashboard refreshed in Power BI workspace **Retail Credit > Loan Repayments**.

Emma""",
            }
        )

    cleaned = []
    for spec in specs:
        spec = dict(spec)
        spec["subject"] = clean_email_text(str(spec["subject"]))
        spec["body"] = clean_email_text(str(spec["body"]))
        cleaned.append(spec)
    return sorted(cleaned, key=lambda x: (int(x.get("day", 15)), str(x.get("subject", ""))))


def news_articles(
    sig: MonthSignal,
    month_events: list[dict],
    outlets: dict,
    rng: random.Random,
) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    y, m = sig.year, sig.month
    outlet = rng.choice(outlets["outlets"])
    author = rng.choice(outlets["authors"])
    articles.extend(department_news_articles(sig, outlets, rng))

    for ev in month_events[:2]:
        d = date.fromisoformat(ev["date"])
        articles.append(
            {
                "outlet": outlet["name"],
                "outlet_id": outlet["id"],
                "author": author,
                "published": d.isoformat(),
                "title": _headline_for_event(ev, rng),
                "body": _article_body(ev, sig, rng),
                "tags": [ev["category"], ev["impact"]],
                "related_event_date": ev["date"],
                "department": event_department(ev),
                "project_type": event_project_type(ev),
            }
        )

    if sig.fail_pct >= 5.5 and not any(a.get("tags", [""])[0] == "health" for a in articles):
        pub = date(y, m, rng.randint(5, 25))
        articles.append(
            {
                "outlet": rng.choice(outlets["outlets"])["name"],
                "outlet_id": "fin24",
                "author": rng.choice(outlets["authors"]),
                "published": pub.isoformat(),
                "title": f"Consumers under pressure as debit orders bounce at higher rates",
                "body": f"""South African households are seeing more debit order reversals as living costs bite, bankers and payment associations say.

Industry payment data — including retail loan collections — shows failed instalments running near **{sig.fail_pct:.1f}%** in recent weeks, with insufficient funds the dominant return code. Analysts link the trend to salary stress and timing of month-end deductions.

"Customers who were marginal in 2022 are now failing after multiple rate hikes," said a Johannesburg-based credit economist who asked not to be named.

Banks are expanding hardship programmes while tightening affordability tests on new {sig.top_loan_type or 'home loan'} applications. The South African Reserve Bank's repo path remains the key variable for 2026 forecasts.""",
                "tags": ["collections", "consumer_credit"],
                "related_data_signal": f"fail_pct={sig.fail_pct}",
                "department": "Loan Products",
                "project_type": "loan_portfolio_analysis",
            }
        )

    if sig.tx_count > 10000 and m == 6:
        pub = date(y, 6, rng.randint(8, 20))
        articles.append(
            {
                "outlet": "Business Day",
                "outlet_id": "business_day",
                "author": rng.choice(outlets["authors"]),
                "published": pub.isoformat(),
                "title": "Retail banks scale digital collections as loan books mature",
                "body": f"""Automated loan repayment volumes processed through South African retail banks have grown sharply as 2019-era mortgages and vehicle finance deals enter mid-life vintages.

Internal industry estimates suggest monthly collection files exceeding **{sig.tx_count // 1000}k** transactions for mid-tier players, straining legacy ETL pipelines built before open banking adoption.

Data engineering teams are migrating to columnar storage and monthly partition folders to keep regulatory MI within SLA.""",
                "tags": ["banking_technology", "data"],
                "related_data_signal": f"tx_count={sig.tx_count}",
                "department": "Data Engineering",
                "project_type": "lakehouse_build",
            }
        )

    if len(articles) < 1:
        pub = date(y, m, 15)
        articles.append(
            {
                "outlet": outlet["name"],
                "outlet_id": outlet["id"],
                "author": author,
                "published": pub.isoformat(),
                "title": f"SA banking sector navigates mixed macro signals in {pub.strftime('%B %Y')}",
                "body": f"Markets and consumers adjusted to shifting rates and operational headwinds during {pub.strftime('%B %Y')}, with payment systems reporting steady digital uptake.",
                "tags": ["banking", "macro"],
                "department": "Executive Office",
                "project_type": "dashboard_request",
            }
        )

    return sorted(articles, key=lambda x: (x["published"], x["title"]))


def department_news_articles(sig: MonthSignal, outlets: dict, rng: random.Random) -> list[dict[str, Any]]:
    y, m = sig.year, sig.month
    month_label = date(y, m, 1).strftime("%B %Y")
    day = lambda: rng.randint(3, 25)

    def article(
        *,
        department: str,
        project_type: str,
        title: str,
        body: str,
        tags: list[str],
        outlet_id: str | None = None,
    ) -> dict[str, Any]:
        outlet = rng.choice(outlets["outlets"])
        if outlet_id:
            outlet = next((o for o in outlets["outlets"] if o["id"] == outlet_id), outlet)
        return {
            "outlet": outlet["name"],
            "outlet_id": outlet["id"],
            "author": rng.choice(outlets["authors"]),
            "published": date(y, m, day()).isoformat(),
            "title": title,
            "body": body,
            "tags": tags,
            "department": department,
            "project_type": project_type,
            "related_data_signal": f"loan_tx={sig.tx_count}; fail_pct={sig.fail_pct}; nsf={sig.insuf_count}; timeout={sig.timeout_count}",
        }

    return [
        article(
            department="Executive Office",
            project_type="dashboard_request",
            title=f"Fintech executives look for sharper portfolio signals as data volumes rise",
            tags=["executive", "portfolio", "dashboard"],
            outlet_id="business_day",
            body=f"""South African fintech executives are asking for faster portfolio views as transaction volumes, customer complaints and operational exceptions move at different speeds.

Industry managers say monthly packs are no longer enough. During {month_label}, retail credit files in the market showed failure rates near **{sig.fail_pct:.1f}%**, while digital activity continued to shift between app, USSD and card channels.

Analysts say the next generation of management reporting will combine customer growth, channel reliability, reconciliation exceptions and credit stress in one executive view.

Project trigger:
Banks with fragmented reporting can use this signal to build an EXCO portfolio dashboard with red, amber and green indicators across growth, risk, operations and finance.""",
        ),
        article(
            department="Loan Products",
            project_type="loan_portfolio_analysis",
            title=f"Lenders revisit loan vintage analytics as missed debit orders remain sticky",
            tags=["loans", "collections", "vintage"],
            outlet_id="fin24",
            body=f"""Retail lenders are paying closer attention to loan vintages as debit-order failures remain a persistent indicator of repayment stress.

Market participants say insufficient-funds returns are most useful when analysed with mandate day, loan type, customer tenure and recent account behaviour. This month, comparable loan-payment extracts showed **{sig.insuf_count:,}** NSF cases and a failure rate around **{sig.fail_pct:.1f}%**.

The practical challenge is distinguishing affordability stress from collection timing and processing delays.

Project trigger:
Loan departments can use the pattern to build an arrears and vintage dashboard, including a watchlist for customers with repeated failed repayments.""",
        ),
        article(
            department="Finance Reconciliation",
            project_type="bank_reconciliation",
            title=f"Bank statement automation exposes hidden workload in fees and late postings",
            tags=["reconciliation", "finance", "bank_statements"],
            outlet_id="moneyweb",
            body=f"""Finance teams are finding that automated bank reconciliation is less about exact matching and more about explaining the exceptions.

Common breaks include source-system lag, duplicate settlement entries, bank charges, service fees, cash deposit fees, withholding tax and interest income. These items often appear on bank statements before they are classified correctly in internal ledgers.

Industry advisers say a good reconciliation process separates genuine data breaks from bank-generated non-transactional entries.

Project trigger:
Finance reconciliation teams can build a bank-statement ingestion and exception dashboard using ISO 20022, CSV, TXT and PDF statement sources.""",
        ),
        article(
            department="Regulatory Compliance",
            project_type="regulatory_monitoring",
            title=f"Conduct-risk teams turn to transaction evidence as fee complaints grow",
            tags=["compliance", "conduct", "audit"],
            outlet_id="daily_maverick",
            body=f"""Conduct-risk teams are under pressure to explain fees, failed payments and delayed postings to customers and regulators.

Compliance specialists say customer-impact monitoring should trace each dashboard number back to a raw transaction, statement entry or exception record. The need is strongest where bank charges follow failed collection attempts.

Audit teams increasingly expect lineage, root-cause tags and masked customer views in routine monitoring packs.

Project trigger:
Compliance teams can use these signals to build a customer-impact monitoring dashboard and evidence extract for audit sampling.""",
        ),
        article(
            department="Products & Remittances",
            project_type="product_analytics",
            title=f"Digital wallet and remittance teams chase cleaner signals for product targeting",
            tags=["products", "remittance", "digital_channels"],
            outlet_id="news24",
            body=f"""Product teams are looking beyond headline transaction counts to understand who is ready for wallet, remittance and digital-payment offers.

Analysts say targeting improves when banks combine channel mix, merchant category behaviour, failed journey history, customer tenure and account tier. The same data can highlight customers moving from ATM and branch use into app, USSD and e-wallet channels.

The risk is promoting products to customers already showing failed-payment or fee pressure.

Project trigger:
Product teams can build a campaign-ready customer segment mart and channel migration dashboard.""",
        ),
        article(
            department="Banking Operations",
            project_type="operations_sla",
            title=f"Payment delays put spotlight on operational SLA dashboards",
            tags=["operations", "sla", "payments"],
            outlet_id="business_day",
            body=f"""Operations teams are investing in daily dashboards that show service breaks before they reach finance or customer complaints.

Banking operations managers say timeout spikes, delayed statement postings and retry queues must be viewed together. This month, comparable loan-payment files showed **{sig.timeout_count:,}** timeout-related rows.

The operational question is no longer whether a transaction failed, but where it failed and what downstream process it affected.

Project trigger:
Operations teams can build an SLA dashboard covering transaction status, authorization time, timeout days, processing lag and reconciliation impact.""",
        ),
        article(
            department="Financial Crime",
            project_type="fraud_risk_monitoring",
            title=f"Fraud teams separate suspicious behaviour from ordinary settlement lag",
            tags=["fraud", "financial_crime", "anomalies"],
            outlet_id="moneyweb",
            body=f"""Fraud teams are refining monitoring rules to avoid treating ordinary bank-statement timing differences as suspicious activity.

Financial-crime analysts say true risk signals include new devices, unusual locations, rapid high-risk transfers, card testing, mule-account patterns and repeated failed authentication. By contrast, bank charges, interest and source-system lag should usually remain reconciliation or finance categories.

Better classification reduces false positives and improves case queues.

Project trigger:
Financial Crime can build an anomaly-to-case dashboard with controls for timing differences and normal bank-generated entries.""",
        ),
        article(
            department="Customer Experience",
            project_type="customer_experience",
            title=f"Fee transparency becomes a customer-experience battleground",
            tags=["customer_experience", "fees", "complaints"],
            outlet_id="news24",
            body=f"""Customer-experience teams are paying closer attention to how fees, reversals and delayed postings appear to customers.

Consumers often see a charge or failed debit before they understand the reason. Service teams say complaints rise when bank fees follow failed debit orders or when a transaction posts days after the customer expected it.

The solution is not only better wording, but better data linking fees, balances, transaction status and customer history.

Project trigger:
CX teams can build a complaint-driver dashboard and messaging trigger list for accounts affected by fees, failed payments and late postings.""",
        ),
        article(
            department="Retail Credit Risk",
            project_type="risk_model",
            title=f"Credit teams add transaction behaviour to repayment stress models",
            tags=["credit_risk", "modelling", "repayment_stress"],
            outlet_id="fin24",
            body=f"""Credit-risk teams are adding transaction behaviour to traditional affordability and bureau-driven views.

Signals such as net account movement, repeated failed debits, fee pressure, channel behaviour and payday timing can improve early warning models. Analysts caution that models should separate genuine stress from operational delays or settlement timing.

Banks using richer behavioural features can prioritise proactive support before arrears become entrenched.

Project trigger:
Credit Risk can build a model-ready feature table and repayment-stress scoring notebook.""",
        ),
        article(
            department="Payments Operations",
            project_type="settlement_reconciliation",
            title=f"Settlement lag remains a blind spot in payment operations",
            tags=["payments", "settlement", "retries"],
            outlet_id="moneyweb",
            body=f"""Payment operations teams continue to face settlement breaks where internal ledgers and bank statements do not agree on timing.

Common causes include sponsor-bank delay, retry processing, bank timeout windows, duplicate settlement files and transactions that clear after statement cutoff.

Analysts say the cleanest operational view separates true failure from late posting and provides evidence for bank follow-up.

Project trigger:
Payments teams can build a settlement-lag dashboard and exception extract for sponsor-bank conversations.""",
        ),
        article(
            department="Data Engineering",
            project_type="lakehouse_build",
            title=f"Lakehouse teams standardise raw statements and transaction evidence",
            tags=["data_engineering", "lakehouse", "data_contracts"],
            outlet_id="business_day",
            body=f"""Data-engineering teams in financial services are standardising how raw source files become analytics-ready evidence.

The strongest designs keep raw statement files, internal ledger transactions, reconciliation outputs and dashboard marts in separate layers. ISO 20022 XML, CSV, TXT and PDF bank statements create useful ingestion tests because each format carries the same business event differently.

Data teams say lineage and data contracts matter as much as storage technology.

Project trigger:
Data Engineering can build a bronze-silver-gold lakehouse with statement ingestion, reconciliation outputs and department-specific data marts.""",
        ),
    ]


def event_department(ev: dict) -> str:
    return {
        "monetary_policy": "Retail Credit Risk",
        "health": "Regulatory Compliance",
        "infrastructure": "Banking Operations",
        "security": "Financial Crime",
        "political": "Executive Office",
        "geopolitical": "Executive Office",
        "fiscal": "Finance Reconciliation",
    }.get(ev.get("category", ""), "Executive Office")


def event_project_type(ev: dict) -> str:
    return {
        "monetary_policy": "risk_model",
        "health": "regulatory_monitoring",
        "infrastructure": "operations_sla",
        "security": "fraud_risk_monitoring",
        "political": "dashboard_request",
        "geopolitical": "dashboard_request",
        "fiscal": "bank_reconciliation",
    }.get(ev.get("category", ""), "dashboard_request")


def _headline_for_event(ev: dict, rng: random.Random) -> str:
    templates = {
        "health": [
            "Lockdown rules reshape branch banking and collections",
            "COVID-19 restrictions hit consumer income and loan repayments",
        ],
        "monetary_policy": [
            "Reserve Bank rate move ripples through prime-linked mortgages",
            "SARB decision shifts lending margins for retail banks",
        ],
        "infrastructure": [
            "Load shedding disrupts ATMs and payment switches",
            "Eskom crisis adds cost to digital banking infrastructure",
        ],
        "security": [
            "Unrest disrupts cash logistics in KwaZulu-Natal and Gauteng",
            "Banks activate contingency plans amid widespread looting",
        ],
        "political": [
            "Election week brings operational caution for banks",
            "GNU policy uncertainty keeps business confidence muted",
        ],
        "geopolitical": [
            "Global fuel shock feeds local inflation and credit stress",
        ],
        "fiscal": [
            "Budget outlook puts spotlight on consumer indebtedness",
        ],
    }
    pool = templates.get(ev["category"], [ev["title"]])
    return rng.choice(pool)


def _article_body(ev: dict, sig: MonthSignal, rng: random.Random) -> str:
    return f"""{ev['title']} — reported {ev['date']}.

Retail bankers said the event category (`{ev['category']}`) has direct bearing on payment rails and branch safety protocols. Collections teams are monitoring loan debit outcomes; internal samples this month show **{sig.fail_pct:.1f}%** failed automated repayments across a **{sig.tx_count:,}**-row payments extract.

{ev['impact'].replace('_', ' ').title()} remains the primary operational concern for tier-2 and tier-3 banks with heavy exposure to salaried borrowers in Gauteng and KwaZulu-Natal.

Analysts cautioned that headline macro news often lags portfolio stress by six to eight weeks. Payment association data will be watched for rising return codes on NAEDO and EFT debits."""


def news_articles(
    sig: MonthSignal,
    month_events: list[dict],
    outlets: dict,
    rng: random.Random,
) -> list[dict[str, Any]]:
    y, m = sig.year, sig.month
    month_label = date(y, m, 1).strftime("%B %Y")
    top_loan = sig.top_loan_type or "home loan"
    yoy = sig.yoy_tx_growth_pct if sig.yoy_tx_growth_pct is not None else rng.uniform(8.0, 24.0)
    customer_growth = max(3.0, min(28.0, abs(yoy) * rng.uniform(0.35, 0.75)))
    digital_growth = rng.uniform(9.0, 18.0)
    event = month_events[0] if month_events else None
    event_text = f"{event['title']} on {event['date']}" if event else f"mixed operating conditions in {month_label}"

    def outlet_by_id(outlet_id: str) -> dict[str, Any]:
        return next((o for o in outlets["outlets"] if o["id"] == outlet_id), rng.choice(outlets["outlets"]))

    def make_article(day: int, outlet_id: str, title: str, body: str, tags: list[str], style: str, department: str, project_type: str) -> dict[str, Any]:
        outlet = outlet_by_id(outlet_id)
        image_path, image_alt = news_image_for(department)
        return {
            "outlet": outlet["name"],
            "outlet_id": outlet["id"],
            "author": rng.choice(outlets["authors"]),
            "published": date(y, m, min(max(day, 1), 28)).isoformat(),
            "title": title,
            "body": body.strip(),
            "tags": tags,
            "style": style,
            "department": department,
            "project_type": project_type,
            "image_path": image_path,
            "image_alt": image_alt,
            "related_data_signal": f"tx_count={sig.tx_count}; fail_pct={sig.fail_pct:.1f}; insuf_count={sig.insuf_count}; timeout_count={sig.timeout_count}; top_loan_type={top_loan}",
        }

    articles = [
        make_article(
            rng.randint(3, 6),
            "news24",
            "Banks watch debit-order strain as households adjust spending",
            f"""South African lenders are watching debit-order strain after industry files showed a failure rate of about {sig.fail_pct:.1f}% in {month_label}. Analysts said insufficient funds, recorded at roughly {sig.insuf_count:,} cases in comparable loan payment data, remained the clearest signal of repayment pressure.""",
            ["loans", "consumer_credit", "collections"],
            "Short news brief",
            "Loan Products",
            "loan_portfolio_analysis",
        ),
        make_article(
            rng.randint(5, 9),
            "moneyweb",
            "Statement delays add noise to month-end finance controls",
            """Finance teams say timing differences between internal ledgers and bank statements remain a stubborn month-end control issue. Bank charges, interest, withholding tax and late postings are increasingly being separated from genuine reconciliation breaks.""",
            ["reconciliation", "bank_statements", "finance"],
            "Short news brief",
            "Finance Reconciliation",
            "bank_reconciliation",
        ),
        make_article(
            rng.randint(8, 13),
            "business_day",
            "Retail banks sharpen arrears analytics as payment stress persists",
            f"""Retail banks are revisiting early-warning indicators as debit-order stress remains visible in consumer portfolios.

Comparable industry data for {month_label} showed {sig.tx_count:,} loan-payment records, with a failure rate of {sig.fail_pct:.1f}% and {sig.insuf_count:,} insufficient-funds cases. Risk teams said the figures were not alarming on their own, but became more useful when combined with customer cash-flow and account behaviour.

"The important distinction is between affordability stress and processing noise," said a Johannesburg-based credit-risk consultant. "A failed debit after a salary delay is not the same as a customer whose account balance has been deteriorating for three months."

Analysts said {top_loan.lower()} portfolios remain sensitive to interest-rate expectations, especially where customers have limited savings buffers. Some banks are also using transaction behaviour to decide which customers should receive proactive support before a formal arrears event.""",
            ["credit_risk", "arrears", "modelling"],
            "Standard business article",
            "Retail Credit Risk",
            "risk_model",
        ),
        make_article(
            rng.randint(12, 17),
            "daily_maverick",
            "Conduct teams press for clearer evidence on fees and failed payments",
            """Conduct-risk teams are asking banks to improve the evidence trail behind fees, failed debits and delayed postings.

The concern is not only whether customers were charged correctly, but whether a bank can explain the sequence of events clearly. Compliance specialists say this becomes difficult when payment retries, bank statement postings and customer communication records sit in separate systems.

"Customers do not care which internal platform created the entry," said one compliance adviser. "They want to know why money moved, why a fee appeared, and whether the bank acted fairly."

Industry observers said the next wave of compliance reporting will need to connect source records, customer impact and exception categories in one view, particularly where late processing creates confusion around account balances.""",
            ["conduct", "compliance", "fees"],
            "Standard business article",
            "Regulatory Compliance",
            "regulatory_monitoring",
        ),
        make_article(
            rng.randint(15, 20),
            "fin24",
            "Digital channels reshape banking workloads as customer growth continues",
            f"""South African banks are seeing a steady increase in customers using digital channels, but the shift is also exposing weaknesses in operational reporting.

Industry data reviewed by analysts suggests active digital customers rose by about {customer_growth:.1f}% year on year across mid-tier retail banks, while e-wallet and app-led transactions grew by an estimated {digital_growth:.1f}%. The gains were strongest in urban provinces, but USSD remained important for lower-income and intermittent-data customers.

The operational effect is more complex than the growth headline suggests. A customer who moves from branch to app may still use ATM cash withdrawals, card payments and debit orders in the same month. Product teams therefore need a joined-up view of channel behaviour rather than isolated product reports.

Loan-payment data adds another layer. In {month_label}, comparable files showed a {sig.fail_pct:.1f}% failed-payment rate, suggesting that some customers targeted for digital offers may also be experiencing cash-flow stress.

Consultants say banks should combine product adoption, transaction categories, failed journeys and fee pressure before launching cross-sell campaigns. "Growth is useful only if the bank understands which customers are ready for more digital engagement," said a Cape Town-based fintech analyst.

The same data can support remittance targeting, dormant-account reactivation and customer-experience monitoring, provided teams agree on common definitions for active customer, failed journey and channel migration.""",
            ["digital_banking", "products", "customer_growth"],
            "Long-form industry report",
            "Products & Remittances",
            "product_analytics",
        ),
        make_article(
            rng.randint(18, 23),
            "business_day",
            "Operational resilience returns to the agenda after payment interruptions",
            f"""Payment operations teams are putting renewed emphasis on resilience after {event_text}.

Industry data shows that even modest processing delays can create a chain of downstream work: customer queries, retry decisions, finance reconciliation breaks and compliance evidence requests. In {month_label}, comparable loan-payment files recorded {sig.timeout_count:,} timeout-related rows, a level operations managers say is enough to justify daily monitoring.

Benchmark analysis from regional banks suggests that institutions with near-real-time exception dashboards resolve payment breaks 22% faster than teams relying on monthly reports. The same benchmark found a 15% reduction in repeated customer queries when service teams had access to posting-delay reasons.

The issue is becoming more urgent as customer counts grow. Analysts estimate that active retail banking customers increased by {customer_growth:.1f}% year on year in the segment, increasing the volume of low-value transactions that still require control checks.

Operations specialists say the most useful dashboards show the worst processing day, affected channel, root cause and downstream finance impact. Without that view, banks risk treating payment delays, failed collections and statement timing differences as separate problems.

The priority for many banks is not a new payment platform, but a clearer operating picture across the existing payment, ledger and statement processes.""",
            ["operations", "payments", "resilience"],
            "Long-form industry report",
            "Banking Operations",
            "operations_sla",
        ),
        make_article(
            rng.randint(22, 26),
            "moneyweb",
            "Banking analytics agenda: five signals to watch this month",
            f"""Key banking signals for {month_label}:

- Loan payment volume: {sig.tx_count:,} records in comparable internal files.
- Failed-payment rate: {sig.fail_pct:.1f}%, with insufficient funds still the main pressure point.
- Processing risk: {sig.timeout_count:,} timeout-related rows point to operational follow-up.
- Digital adoption: industry data indicates channel growth of about {digital_growth:.1f}% year on year, driven by app, wallet and USSD activity.
- Customer growth: benchmark estimates show an increase in customers of roughly {customer_growth:.1f}% year on year across comparable retail portfolios.
- Control focus: finance teams are separating normal bank charges and interest from genuine reconciliation breaks.

Implication:
Banks need a single management view that connects customer growth, product usage, repayment stress, payment delays and statement exceptions.""",
            ["strategy", "analytics", "dashboard"],
            "Consulting-style slide summary",
            "Executive Office",
            "dashboard_request",
        ),
    ]
    return sorted(articles, key=lambda x: (x["published"], x["title"]))


def write_news_md(path: Path, article: dict) -> None:
    front = {
        "outlet": article["outlet"],
        "outlet_id": article.get("outlet_id", ""),
        "author": article["author"],
        "published": article["published"],
        "title": article["title"],
        "tags": article.get("tags", []),
        "style": article.get("style", ""),
        "image_path": article.get("image_path", ""),
        "image_alt": article.get("image_alt", ""),
    }
    if "related_event_date" in article:
        front["related_event_date"] = article["related_event_date"]
    if "related_data_signal" in article:
        front["related_data_signal"] = article["related_data_signal"]
    if "department" in article:
        front["department"] = article["department"]
    if "project_type" in article:
        front["project_type"] = article["project_type"]

    yaml_lines = ["---"]
    for k, v in front.items():
        if isinstance(v, list):
            yaml_lines.append(f"{k}: [{', '.join(v)}]")
        else:
            yaml_lines.append(f'{k}: "{v}"' if " " in str(v) else f"{k}: {v}")
    yaml_lines.append("---\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(yaml_lines) + "\n" + article["body"].strip() + "\n", encoding="utf-8")


def generate_corpus(
    start_year: int = 2019,
    end_year: int = 2025,
    seed: int = 2019,
    write_pdf: bool = True,
) -> None:
    teams_cfg = load_json("corpus_teams.json")
    macro = load_json("macro_timeline_sa.json")
    outlets = load_json("news_outlets_sa.json")
    people = teams_cfg["people"]

    signals = [s for s in load_monthly_signals() if start_year <= s.year <= end_year]
    if not signals:
        print("No loan payment signals found — run loan payment generator first.")
        return

    signals_path = CORPUS_DIR / "index" / "monthly_signals.csv"
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    with signals_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "year",
                "month",
                "tx_count",
                "fail_pct",
                "insuf_count",
                "timeout_count",
                "data_error_pct",
                "top_loan_type",
                "yoy_tx_growth_pct",
            ]
        )
        for s in signals:
            w.writerow(
                [
                    s.year,
                    s.month,
                    s.tx_count,
                    s.fail_pct,
                    s.insuf_count,
                    s.timeout_count,
                    s.data_error_pct,
                    s.top_loan_type,
                    s.yoy_tx_growth_pct if s.yoy_tx_growth_pct is not None else "",
                ]
            )

    events_path = CORPUS_DIR / "events" / "macro_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8") as ef:
        for ev in macro["events"]:
            d = date.fromisoformat(ev["date"])
            if start_year <= d.year <= end_year:
                ef.write(json.dumps(ev, ensure_ascii=False) + "\n")

    manifest_rows: list[dict[str, str]] = []

    for sig in signals:
        rng = random.Random(seed + sig.year * 100 + sig.month)
        month_events = events_for_month(macro["events"], sig.year, sig.month)
        y, m = sig.year, sig.month
        out_base = CORPUS_DIR / str(y) / f"{m:02d}"
        cleanup_email_outputs(out_base)
        cleanup_news_outputs(out_base)

        # Emails
        mail_specs = email_scenarios(sig, month_events, teams_cfg, people, rng)
        prev_mid: str | None = None
        refs: list[str] = []

        for i, spec in enumerate(mail_specs):
            sent = datetime(y, m, min(spec["day"], 28), rng.randint(8, 17), rng.randint(0, 59))
            to_list = spec["to"] if isinstance(spec["to"], list) else [spec["to"]]
            msg = build_email(
                subject=spec["subject"],
                body=spec["body"],
                sender=spec["from"],
                to=to_list,
                cc=spec.get("cc"),
                sent=sent,
                in_reply_to=prev_mid if i > 0 and rng.random() < 0.4 else None,
                references=refs if refs else None,
            )
            prev_mid = msg["Message-ID"]
            refs.append(prev_mid)

            safe_subj = re.sub(r"[^A-Za-z0-9\-]+", "-", spec["subject"])[:60].strip("-")
            fname = f"{sent.strftime('%Y%m%d-%H%M')}-{safe_subj}.eml"
            eml_path = out_base / "emails" / fname
            write_eml(eml_path, msg)
            if write_pdf:
                pdf_path = render_outlook_email_pdf(eml_path)
                manifest_rows.append(
                    {
                        "year": str(y),
                        "month": f"{m:02d}",
                        "artifact_type": "email_pdf",
                        "path": str(pdf_path.relative_to(CORPUS_DIR)),
                        "date": sent.date().isoformat(),
                        "title": spec["subject"],
                        "source": spec["from"],
                        "department": spec.get("department", ""),
                        "project_type": spec.get("project_type", ""),
                    }
                )
            manifest_rows.append(
                {
                    "year": str(y),
                    "month": f"{m:02d}",
                    "artifact_type": "email",
                    "path": str(eml_path.relative_to(CORPUS_DIR)),
                    "date": sent.date().isoformat(),
                    "title": spec["subject"],
                    "source": spec["from"],
                    "department": spec.get("department", ""),
                    "project_type": spec.get("project_type", ""),
                }
            )

        # News
        month_articles = news_articles(sig, month_events, outlets, rng)
        news_json_path = out_base / "news" / f"news_articles_{y}_{m:02d}.json"
        news_json_path.parent.mkdir(parents=True, exist_ok=True)
        news_json_path.write_text(json.dumps(month_articles, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_rows.append(
            {
                "year": str(y),
                "month": f"{m:02d}",
                "artifact_type": "news_json",
                "path": str(news_json_path.relative_to(CORPUS_DIR)),
                "date": f"{y}-{m:02d}-01",
                "title": f"News article set {y}-{m:02d}",
                "source": "generated_news",
                "department": "",
                "project_type": "",
                "style": "json_list",
                "image_path": "",
            }
        )

        for j, article in enumerate(month_articles):
            slug = re.sub(r"[^\w]+", "-", article["title"].lower())[:50]
            pub = article["published"].replace("-", "")
            news_path = out_base / "news" / f"{pub}_{slug}.md"
            write_news_md(news_path, article)
            if write_pdf:
                pdf_path = render_news_article_pdf(news_path)
                manifest_rows.append(
                    {
                        "year": str(y),
                        "month": f"{m:02d}",
                        "artifact_type": "news_pdf",
                        "path": str(pdf_path.relative_to(CORPUS_DIR)),
                        "date": article["published"],
                        "title": article["title"],
                        "source": article["outlet"],
                        "department": article.get("department", ""),
                        "project_type": article.get("project_type", ""),
                        "style": article.get("style", ""),
                        "image_path": article.get("image_path", ""),
                    }
                )
            manifest_rows.append(
                {
                    "year": str(y),
                    "month": f"{m:02d}",
                    "artifact_type": "news",
                    "path": str(news_path.relative_to(CORPUS_DIR)),
                    "date": article["published"],
                    "title": article["title"],
                    "source": article["outlet"],
                    "department": article.get("department", ""),
                    "project_type": article.get("project_type", ""),
                    "style": article.get("style", ""),
                    "image_path": article.get("image_path", ""),
                }
            )

        # Month event summary (for DE pipeline)
        if month_events:
            summary_path = out_base / "events" / f"month_context_{y}_{m:02d}.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "year": y,
                        "month": m,
                        "macro_events": month_events,
                        "data_signals": {
                            "tx_count": sig.tx_count,
                            "fail_pct": sig.fail_pct,
                            "insuf_count": sig.insuf_count,
                            "data_error_pct": sig.data_error_pct,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            manifest_rows.append(
                {
                    "year": str(y),
                    "month": f"{m:02d}",
                    "artifact_type": "event_summary",
                    "path": str(summary_path.relative_to(CORPUS_DIR)),
                    "date": f"{y}-{m:02d}-01",
                    "title": f"Month context {y}-{m:02d}",
                    "source": "macro_timeline_sa",
                    "department": "",
                    "project_type": "",
                    "style": "",
                    "image_path": "",
                }
            )

    manifest_path = CORPUS_DIR / "index" / "corpus_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["year", "month", "artifact_type", "path", "date", "title", "source", "department", "project_type", "style", "image_path"],
        )
        w.writeheader()
        w.writerows(manifest_rows)

    email_n = sum(1 for r in manifest_rows if r["artifact_type"] == "email")
    news_n = sum(1 for r in manifest_rows if r["artifact_type"] == "news")
    pdf_e = sum(1 for r in manifest_rows if r["artifact_type"] == "email_pdf")
    pdf_n = sum(1 for r in manifest_rows if r["artifact_type"] == "news_pdf")
    print(f"Corpus written to {CORPUS_DIR}")
    print(f"  Months: {len(signals)} | Emails: {email_n} | News: {news_n} | PDFs: {pdf_e + pdf_n}")
    print(f"  Index: {manifest_path}")
    print(f"  Signals: {signals_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate emails, news, and events corpus (2019-2025).")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation (eml/md only).")
    args = parser.parse_args()
    generate_corpus(args.start_year, args.end_year, args.seed, write_pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
