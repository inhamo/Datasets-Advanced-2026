"""Regenerate realistic internal bank emails under banking_data/YYYY/MM/emails.

The emails are scenario prompts for analysis projects. They are year-aware so an
older folder never refers to products, channels, regulations or years that have
not happened yet in the synthetic timeline.
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import json
import random
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from calendar import monthrange
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd

from commons.pdf_renderers import render_outlook_email_pdf


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"
COMMONS_DIR = BASE_DIR / "commons"
SOURCE_NOTES_PATH = COMMONS_DIR / "email_research_backbone.csv"


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


TRUSTED_SOURCES = [
    ("McKinsey", "Experience-led growth in banking", "https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/five-ways-to-drive-experience-led-growth-in-banking"),
    ("McKinsey", "State of retail banking", "https://www.mckinsey.com/industries/financial-services/our-insights/the-state-of-retail-banking-profitability-and-growth-in-the-era-of-digital-and-ai"),
    ("McKinsey", "Combating payments fraud", "https://www.mckinsey.com/industries/financial-services/our-insights/combating-payments-fraud-and-enhancing-customer-experience"),
    ("McKinsey", "Scaling gen AI in banking", "https://www.mckinsey.com/industries/financial-services/our-insights/scaling-gen-ai-in-banking-choosing-the-best-operating-model"),
    ("Bain", "Customer loyalty in retail banking", "https://www.bain.com/insights/customer-loyalty-in-retail-banking-2015-global/"),
    ("Bain", "Retail banks leak value", "https://www.bain.com/insights/as-retail-banks-leak-value-heres-how-they-can-stop-it/"),
    ("Bain", "Digital banking customer service", "https://www.bain.com/insights/lets-chat-banking-customer-service-is-going-digital-snap-chart/"),
    ("BCG", "Customer value and banking", "https://www.bcg.com/publications/2022/customer-value-and-banking-in-the-digital-age"),
    ("Deloitte", "Banking outlook", "https://www.deloitte.com/us/en/insights/industry/financial-services/financial-services-industry-outlooks/banking-industry-outlook-2024.html"),
    ("KPMG", "Operational resilience", "https://kpmg.com/us/en/articles/2024/emerging-regulatory-focus-operational-resilience-reg-alert.html"),
    ("KPMG", "Banking customer experience", "https://kpmg.com/ng/en/insights/2024/12/2024-kpmg-nigeria-banking-industry-customer-experience-survey.html"),
    ("PwC", "Banking risk perspective", "https://www.pwc.com/us/en/industries/financial-services/library/our-take/occ-risk-perspective-dec-20-2024.html"),
    ("PwC", "Financial crime operations", "https://www.pwc.com/gx/en/about/analyst-relations/2024/everest-peak-matrix-fcc.html"),
    ("World Bank", "Financial consumer protection", "https://digitalfinance.worldbank.org/topics/financial-consumer-protection"),
    ("World Bank", "Digital security and fraud", "https://digitalfinance.worldbank.org/topics/digital-credit/digital-security-and-fraud"),
    ("World Bank", "Financial inclusion survey", "https://www.worldbank.org/en/topic/financialinclusion/brief/ficpsurvey"),
    ("FIC South Africa", "Money laundering typologies", "https://www.fic.gov.za/document/case-studies-money-laundering-typologies-and-indicators/"),
    ("FIC South Africa", "Money mule insights", "https://www.fic.gov.za/wp-content/uploads/2024/06/Financial-Crime-Insights-Money-mules.pdf"),
    ("FIC South Africa", "Compliance and supervision", "https://www.fic.gov.za/Compliance/"),
    ("SARB", "Bank supervision", "https://www.resbank.co.za/en/home/publications/publication-detail-pages/media-releases/2018/8473"),
]

PROBLEM_PATTERNS = [
    ("customer_experience", "Bad or avoidable service contacts caused by unclear fees, late posting or failed journeys"),
    ("digital_adoption", "Customers moving between branch, ATM, card and digital channels need journey-level evidence"),
    ("loan_products", "Affordability pressure, repricing and repayment timing must be separated from operational failure"),
    ("credit_risk", "Early warning features should use cash-flow behaviour, missed debits and balance movement"),
    ("payments", "Payment failures need root-cause split: insufficient funds, bank timeout, retry, reversal or settlement lag"),
    ("financial_crime", "Fraud work must distinguish mule activity, account takeover, social engineering and false positives"),
    ("compliance", "Conduct monitoring needs traceable evidence and masked customer identifiers"),
    ("operations", "Operational resilience requires critical-service monitoring, incident logs and customer impact counts"),
    ("data_engineering", "Source-to-dashboard lineage, schema drift and data sensitivity need explicit controls"),
    ("product", "Product teams need adoption, dormant-account and hidden-defection signals before campaigns"),
]


def write_research_backbone() -> None:
    """Write 100 source-to-problem mappings used as scenario inspiration."""
    SOURCE_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 1
    for source in TRUSTED_SOURCES:
        for pattern in PROBLEM_PATTERNS[:5]:
            rows.append(
                {
                    "note_id": f"SRC-{idx:03d}",
                    "source_org": source[0],
                    "source_title": source[1],
                    "source_url": source[2],
                    "problem_area": pattern[0],
                    "banking_problem_to_model": pattern[1],
                    "dataset_link": "transactions, customers, accounts, loans, debit orders, communications, statements, campaigns",
                }
            )
            idx += 1
    with SOURCE_NOTES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:100])


def load_teams() -> dict[str, Any]:
    return json.loads((COMMONS_DIR / "corpus_teams.json").read_text(encoding="utf-8"))


def person_email(person: dict[str, str], teams: dict[str, Any]) -> str:
    team = next(t for t in teams["teams"] if t["id"] == person["team"])
    return f"{person['first'].lower()}.{person['last'].replace(' ', '').lower()}@{team['mailbox']}.{teams['email_domain']}"


def mailbox(team_id: str, teams: dict[str, Any]) -> str:
    team = next(t for t in teams["teams"] if t["id"] == team_id)
    return f"{team['mailbox']}@{teams['email_domain']}"


def sender(team_id: str, teams: dict[str, Any], rng: random.Random) -> str:
    people = [p for p in teams["people"] if p["team"] == team_id]
    return person_email(rng.choice(people), teams)


def load_monthly_signals() -> dict[tuple[int, int], MonthSignal]:
    path = BASE_DIR / "corpus_context" / "index" / "monthly_signals.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly signals: {path}")
    out: dict[tuple[int, int], MonthSignal] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yoy = row.get("yoy_tx_growth_pct") or ""
            sig = MonthSignal(
                year=int(row["year"]),
                month=int(row["month"]),
                tx_count=int(float(row["tx_count"])),
                fail_pct=float(row["fail_pct"]),
                insuf_count=int(float(row["insuf_count"])),
                timeout_count=int(float(row["timeout_count"])),
                data_error_pct=float(row.get("data_error_pct") or 0),
                top_loan_type=row.get("top_loan_type") or "personal_loan",
                yoy_tx_growth_pct=float(yoy) if yoy else None,
            )
            out[(sig.year, sig.month)] = sig
    return out


def clean_email_dir(year: int, month: int) -> Path:
    out = BANKING_DIR / str(year) / f"{month:02d}" / "emails"
    out.mkdir(parents=True, exist_ok=True)
    for path in out.iterdir():
        if path.is_file() and path.suffix.lower() in {".eml", ".pdf"}:
            path.unlink()
    return out


def load_customer_names(year: int, month: int) -> list[str]:
    path = BANKING_DIR / str(year) / f"{month:02d}" / f"customers_{year}_{month:02d}.parquet"
    if not path.exists():
        return ["a customer from Gauteng", "a customer from KwaZulu-Natal"]
    df = pd.read_parquet(path, columns=["full_name", "customer_type"])
    names = df.loc[df["customer_type"].eq("Individual"), "full_name"].dropna().astype(str).head(200).tolist()
    return names or ["a customer from Gauteng", "a customer from KwaZulu-Natal"]


def year_capabilities(year: int) -> dict[str, str]:
    if year == 2019:
        return {
            "channels": "branch, ATM, card, phone and debit order channels",
            "digital": "mobile-banking roadmap",
            "risk_theme": "SIM-swap, card testing and mule-account warning signs",
            "service_theme": "branch queues, statement printouts and debit-order confusion",
        }
    if year == 2020:
        return {
            "channels": "branch, phone, card, ATM and early in-app service messages",
            "digital": "bank-from-home adoption",
            "risk_theme": "COVID hardship abuse, phishing and remote onboarding checks",
            "service_theme": "payment holidays, call-centre pressure and branch appointment limits",
        }
    if year == 2021:
        return {
            "channels": "branch, phone, card, ATM, app and SMS channels",
            "digital": "self-service and SMS nudges",
            "risk_theme": "mule accounts, unrest-related cash pressure and SIM-swap alerts",
            "service_theme": "failed debit explanations, SMS wording and reduced branch hours",
        }
    if year == 2022:
        return {
            "channels": "branch, phone, card, ATM, app, SMS and email channels",
            "digital": "email servicing and digital onboarding",
            "risk_theme": "social engineering, fee disputes and affordability stress",
            "service_theme": "load-shedding downtime, fee conduct and digital onboarding drop-off",
        }
    if year == 2023:
        return {
            "channels": "branch, phone, card, ATM, app, SMS, email and social media channels",
            "digital": "social listening and digital wallet growth",
            "risk_theme": "account takeover, scam complaints and mule-account rings",
            "service_theme": "social-media complaints, channel outages and campaign mis-targeting",
        }
    if year == 2024:
        return {
            "channels": "branch, phone, card, ATM, app, SMS, email, social media, WhatsApp and chatbot channels",
            "digital": "WhatsApp, chatbot and assisted self-service",
            "risk_theme": "AI-assisted scams, APP fraud, mule accounts and chatbot escalation risks",
            "service_theme": "customer outcome evidence, WhatsApp containment and chatbot handover quality",
        }
    return {
        "channels": "all customer channels including WhatsApp, chatbot, app, social, email, phone, branch, card and ATM",
        "digital": "AI-assisted service, personalization and model monitoring",
        "risk_theme": "AI-enabled fraud, correspondent-banking alerts and cross-channel mule behaviour",
        "service_theme": "personalized retention, vulnerable-customer treatment and model governance",
    }


def spec(
    *,
    day: int,
    from_team: str,
    to: list[str],
    cc: list[str] | None,
    department: str,
    project_type: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    return {
        "day": day,
        "from_team": from_team,
        "to": to,
        "cc": cc or [],
        "department": department,
        "project_type": project_type,
        "subject": subject,
        "body": body,
    }


def monthly_email_specs(sig: MonthSignal, teams: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    y, m = sig.year, sig.month
    caps = year_capabilities(y)
    month_label = date(y, m, 1).strftime("%B %Y")
    names = load_customer_names(y, m)
    customer_a = rng.choice(names)
    customer_b = rng.choice(names)
    top_loan = (sig.top_loan_type or "personal loan").replace("_", " ")
    deliverable = rng.choice(["Excel extract", "Power BI page", "short slide pack", "case list", "one-page note"])
    light_touch_channel = "phone note or branch callback"
    if y >= 2024:
        light_touch_channel = "SMS, WhatsApp or app message"
    elif y >= 2021:
        light_touch_channel = "SMS or app message"
    elif y >= 2020 and m >= 4:
        light_touch_channel = "phone note or in-app service message"

    specs = [
        spec(
            day=rng.randint(3, 6),
            from_team="executive",
            to=[mailbox("analytics", teams)],
            cc=[mailbox("data_engineering", teams)],
            department="Executive Office",
            project_type="business_health_dashboard",
            subject=f"{month_label} business health view before EXCO",
            body=f"""Hi Sarah,

Can you give me a simple business health view for {month_label}? I want it practical, not a perfect model.

Please show customer activity, active accounts, loan repayment pressure, channel movement and the top customer pain points. Use the bank data we already have. If the number is draft, mark it as draft.

The board question is simple: are we growing safely, are customers struggling, and where is the bank creating avoidable work?

I have heard repayment issues may be up, but I do not want us to work from corridor numbers. Please use the source files and tell me what is actually happening.

Thanks,
Kabelo""",
        ),
        spec(
            day=rng.randint(4, 9),
            from_team="loan_department",
            to=[mailbox("analytics", teams)],
            cc=[mailbox("credit_risk", teams), mailbox("collections", teams)],
            department="Loan Products",
            project_type="loan_portfolio_analysis",
            subject=f"{month_label} loan book questions from product",
            body=f"""Hi team,

Loan Products needs help understanding the {top_loan} book for {month_label}.

The question is not only who failed to pay. We need to know whether the pressure is coming from affordability, debit day timing, repricing, branch-captured mandates or a real product issue.

Can you prepare a {deliverable} showing first missed debit, repeat missed debit, balances before collection, province, origination month and current account activity?

Please do not treat every failure as bad behaviour. Some customers look like they paid late or were affected by processing delays.

Regards,
Mandla""",
        ),
        spec(
            day=rng.randint(5, 10),
            from_team="customer_experience",
            to=[mailbox("analytics", teams), mailbox("product", teams)],
            cc=[mailbox("operations", teams)],
            department="Customer Experience",
            project_type="complaint_driver_analysis",
            subject=f"{month_label} why are customers contacting us",
            body=f"""Howzit team,

Can we get a customer contact view for {month_label}? People are not always complaining about the same thing, even when the transaction looks similar in the data.

Please group contacts by fee questions, debit-order confusion, card or ATM declines, late posting, loan questions and digital help. Link it to {caps['channels']} where possible.

I need examples that service teams can understand. A customer like {customer_a} does not know our source-system names; they just know money moved late or a fee appeared.

Output can be a {deliverable}. Please include suggested wording for the top three contact reasons.

Thanks,
Zanele""",
        ),
        spec(
            day=rng.randint(6, 12),
            from_team="product",
            to=[mailbox("analytics", teams)],
            cc=[mailbox("customer_experience", teams)],
            department="Products and Channels",
            project_type="channel_and_product_analytics",
            subject=f"{month_label} channel movement and hidden defection",
            body=f"""Hi Emma,

Product is worried that some customers still keep the account open but move useful activity elsewhere.

Can you check channel movement across {caps['channels']} and show customers whose transaction activity is falling even though the account is active?

Please split routine activity, product purchase activity and service recovery contacts. We also need a list of customers who should not receive campaigns because recent fees, failed debits or complaints make the timing wrong.

The theme for this month is {caps['digital']}. Keep the view grounded in actual transactions and customer behaviour, not just campaign counts.

Farah""",
        ),
        spec(
            day=rng.randint(7, 13),
            from_team="operations",
            to=[mailbox("payments", teams), mailbox("data_engineering", teams)],
            cc=[mailbox("analytics", teams), mailbox("infrastructure", teams)],
            department="Banking Operations",
            project_type="operational_resilience",
            subject=f"{month_label} service failures and customer impact",
            body=f"""Hi Priya and Thabo,

Ops needs a view of where service failed in {month_label}. Please separate true customer error from bank-side delay.

Use the transaction status, channel, posting timestamp, statement date and any timeout indicators. I have heard there were timeout problems, but I want the customer impact, not just a technical count.

Please show worst day, worst channel, affected accounts, repeat incidents and whether the issue created follow-up calls or complaints.

This is for operational resilience, so include the evidence we would need if an incident review is opened.

Peter""",
        ),
        spec(
            day=rng.randint(8, 14),
            from_team="fraud",
            to=[mailbox("analytics", teams)],
            cc=[mailbox("compliance", teams), mailbox("credit_risk", teams)],
            department="Financial Crime",
            project_type="fraud_and_scam_monitoring",
            subject=f"{month_label} suspicious behaviour without overcalling fraud",
            body=f"""Hi,

Financial Crime needs a better triage view. Please do not make it a generic fraud dashboard.

For {month_label}, focus on {caps['risk_theme']}. Show new device or channel, unusual location, payment velocity, repeated small tests, cash-in then rapid transfer, and customers whose behaviour changed suddenly.

Also show why a case might be a false positive, for example late posting, normal fees, payroll timing or a known branch visit.

If there are named external tips, use customer names rather than IDs in the working note. For this month, please include a check around {customer_b} if their activity looks unusual.

Aisha""",
        ),
        spec(
            day=rng.randint(9, 15),
            from_team="compliance",
            to=[mailbox("analytics", teams), mailbox("data_engineering", teams)],
            cc=[mailbox("fraud", teams)],
            department="Regulatory Compliance",
            project_type="conduct_and_evidence_pack",
            subject=f"{month_label} customer outcome evidence pack",
            body=f"""Hi team,

Compliance needs evidence that we can explain customer outcomes for {month_label}.

Please build a pack covering fees after failed payments, repeated retry attempts, vulnerable customers, suspicious and unusual transaction indicators, complaint handling time and masked customer examples.

The important part is traceability. If someone challenges a number, we must get back to the source file, statement line or communication record.

Do not include raw ID numbers or full account numbers in the pack. Use masked fields for the front view and keep restricted data in the evidence folder.

Michael""",
        ),
        spec(
            day=rng.randint(10, 16),
            from_team="credit_risk",
            to=[mailbox("analytics", teams)],
            cc=[mailbox("loan_department", teams), mailbox("collections", teams)],
            department="Retail Credit Risk",
            project_type="early_warning_features",
            subject=f"{month_label} early warning features for repayment stress",
            body=f"""Hi Analytics,

Can we create a model-ready table for repayment stress using only information available up to the observation date?

Target: failed loan debit, repeat NSF, or collections action in the next 30 days.

Candidate features: balance trend, debit day versus salary window, failed payment history, fees, recent cash withdrawals, channel change, complaint flags and product type.

Please avoid future leakage. A row from an older month must not use outcomes from months that had not happened yet. Same principle for every year.

Johan""",
        ),
        spec(
            day=rng.randint(11, 18),
            from_team="payments",
            to=[mailbox("data_engineering", teams), mailbox("operations", teams)],
            cc=[mailbox("finance_recon", teams)],
            department="Payments Operations",
            project_type="payments_root_cause",
            subject=f"{month_label} failed debit root cause split",
            body=f"""Morning,

Before we call customers, Payments needs to know what actually happened.

Please split failed debits into insufficient funds, bank timeout, rejected mandate, duplicate attempt, reversal, late settlement and customer-cancelled items.

For {month_label}, I am hearing different versions of the story from Collections and Ops. Please validate the root-cause split against the transaction file and statement dates.

Output should be simple enough for the sponsor-bank call and detailed enough for the analysts to reproduce.

Priya""",
        ),
        spec(
            day=rng.randint(12, 20),
            from_team="data_engineering",
            to=[mailbox("analytics", teams), mailbox("compliance", teams), mailbox("operations", teams)],
            cc=[mailbox("executive", teams)],
            department="Data Engineering",
            project_type="trusted_data_product",
            subject=f"{month_label} data contract and sensitive fields",
            body=f"""Hi all,

For the {month_label} sprint, please confirm which fields are genuinely needed before we publish the next trusted data product.

We will keep raw data in the landing area, standardise the repeatable fields, mask sensitive identifiers for reporting and keep lineage back to the original files.

Known asks so far: customer activity, account status, loan behaviour, communication history, complaint themes, campaign response, fraud indicators and payment exceptions.

Please stop sending screenshots as requirements. If you need an Excel layout, send the column names and the business rule in plain English.

Thabo""",
        ),
    ]

    if sig.fail_pct >= 6.0:
        specs.append(
            spec(
                day=rng.randint(13, 21),
                from_team="collections",
                to=[mailbox("analytics", teams), mailbox("customer_experience", teams)],
                cc=[mailbox("credit_risk", teams)],
                department="Collections",
                project_type="collections_prioritisation",
                subject=f"{month_label} who needs a call before the next debit run",
                body=f"""Hi,

Collections does not want to phone everyone who missed a debit. That creates noise and makes customers angry.

Please rank customers for the next debit run using recent NSF, balance recovery, hardship flag, complaint history and whether the customer usually fixes the account within a few days.

People are saying missed debits are higher than usual for {month_label}, so we need a priority list rather than a full dialler dump.

Also give us a small group where a {light_touch_channel} is enough. Service tone matters here.

Nomsa""",
            )
        )

    if y == 2020 and m >= 3:
        specs.append(
            spec(
                day=rng.randint(14, 22),
                from_team="collections",
                to=[mailbox("credit_risk", teams), mailbox("compliance", teams)],
                cc=[mailbox("analytics", teams)],
                department="Collections",
                project_type="hardship_monitoring",
                subject=f"{month_label} payment holiday and hardship view",
                body=f"""Hi Risk and Compliance,

With the COVID disruption, we need one view of customers on payment holiday, customers still being debited, and customers asking for help but not yet approved.

Please check if failed debits are happening after a hardship promise was logged. That is the customer outcome we must avoid.

Keep this to {month_label}. Do not bring in later-year assumptions.

Nomsa""",
            )
        )

    if (y, m) in {(2019, 3), (2021, 11), (2023, 9), (2024, 11), (2025, 3)}:
        issue = {
            (2019, 3): "SIM-swap account takeover tip from a mobile operator",
            (2021, 11): "student-account mule activity reported by a university",
            (2023, 9): "correspondent payment beneficiary change tip-off",
            (2024, 11): "trade-payment layering concern from a correspondent bank",
            (2025, 3): "AI-assisted voice scam complaint linked to payment release",
        }[(y, m)]
        specs.append(
            spec(
                day=rng.randint(15, 23),
                from_team="fraud",
                to=[mailbox("analytics", teams), mailbox("compliance", teams)],
                cc=[mailbox("executive", teams)],
                department="Financial Crime",
                project_type="tipoff_case_review",
                subject=f"{month_label} tip-off review needed",
                body=f"""Team,

We received a tip-off: {issue}.

Please prepare a case view using customer names for the working session, then mask identifiers before anything leaves Financial Crime. Start with {customer_a} and {customer_b}; I am not saying they are guilty, only that their names were mentioned in the external note.

We need transaction timeline, channel changes, device or contact changes, linked accounts, cash movements, transfers, failed attempts and any customer communication around the same dates.

Legal may ask for an evidence pack. Police should only receive what Compliance approves.

Aisha""",
            )
        )

    if y >= 2024:
        specs.append(
            spec(
                day=rng.randint(16, 24),
                from_team="customer_experience",
                to=[mailbox("analytics", teams), mailbox("operations", teams)],
                cc=[mailbox("compliance", teams)],
                department="Customer Experience",
                project_type="chatbot_and_whatsapp_quality",
                subject=f"{month_label} chatbot and WhatsApp handover quality",
                body=f"""Hi team,

Now that WhatsApp and chatbot are live, can we measure whether customers are actually getting helped?

Please show containment, agent handover, repeat contact within seven days, complaint after bot contact and topics where the bot gives up too late.

Do not only show volume. A high bot count with poor resolution is not a win.

Zanele""",
            )
        )

    return sorted(specs, key=lambda x: (int(x["day"]), x["subject"]))


def build_email(spec: dict[str, Any], teams: dict[str, Any], sent: datetime, rng: random.Random) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender(spec["from_team"], teams, rng)
    msg["To"] = ", ".join(spec["to"])
    if spec.get("cc"):
        msg["Cc"] = ", ".join(spec["cc"])
    msg["Subject"] = spec["subject"]
    msg["Date"] = email.utils.format_datetime(sent)
    msg["Message-ID"] = f"<{uuid.uuid4().hex}@keystonebank.co.za>"
    msg["X-Keystone-Department"] = spec["department"]
    msg["X-Keystone-Project-Type"] = spec["project_type"]
    msg.set_content(spec["body"].strip() + "\r\n")
    return msg


def safe_filename(text: str) -> str:
    text = text.replace("_", " ")
    return re.sub(r"[^A-Za-z0-9\-]+", "-", text)[:72].strip("-")


def nearest_business_day(year: int, month: int, day: int) -> int:
    """Move weekend dates to a nearby weekday inside the same month."""
    last_day = monthrange(year, month)[1]
    day = max(1, min(day, last_day))
    current = date(year, month, day)
    if current.weekday() < 5:
        return day
    if current.weekday() == 5:  # Saturday
        return day - 1 if day > 1 else min(day + 2, last_day)
    # Sunday
    return day + 1 if day < last_day else max(day - 2, 1)


def validate_no_future_references(path: Path, year: int) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    problems = []
    for match in re.finditer(r"\b(20[0-9]{2})\b", text):
        found = int(match.group(1))
        if found > year:
            problems.append(f"{path}: references future year {found} in {year} folder")
    return problems


def generate(start_year: int, end_year: int, write_pdf: bool = True) -> None:
    write_research_backbone()
    teams = load_teams()
    signals = load_monthly_signals()
    validation_errors: list[str] = []
    email_count = 0
    pdf_count = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            sig = signals.get((year, month))
            if sig is None:
                continue
            rng = random.Random(7100 + year * 100 + month)
            out_dir = clean_email_dir(year, month)
            specs = monthly_email_specs(sig, teams, rng)
            used_days: dict[int, int] = {}
            for spec_item in specs:
                base_day = nearest_business_day(year, month, min(28, int(spec_item["day"])))
                used_days[base_day] = used_days.get(base_day, 0) + 1
                minute_offset = used_days[base_day] * 7
                sent = datetime(year, month, base_day, rng.randint(8, 16), min(59, rng.randint(0, 45) + minute_offset))
                msg = build_email(spec_item, teams, sent, rng)
                name = f"{sent.strftime('%Y%m%d-%H%M')}-{safe_filename(spec_item['subject'])}.eml"
                path = out_dir / name
                path.write_bytes(msg.as_bytes())
                email_count += 1
                validation_errors.extend(validate_no_future_references(path, year))
                if write_pdf:
                    render_outlook_email_pdf(path)
                    pdf_count += 1

    if validation_errors:
        raise RuntimeError("\n".join(validation_errors[:20]))
    print(f"Regenerated {email_count} emails and {pdf_count} PDFs under banking_data.")
    print(f"Wrote research backbone: {SOURCE_NOTES_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Revamp banking_data email scenarios.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    generate(args.start_year, args.end_year, write_pdf=not args.no_pdf)


if __name__ == "__main__":
    main()
