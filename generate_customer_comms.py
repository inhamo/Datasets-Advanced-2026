"""
Generate synthetic customer communication records for a South African bank.

Outputs per month:
  banking_data/{year}/{month}/customer_communications/communications.csv
  banking_data/{year}/{month}/customer_communications/complaints.csv
  banking_data/{year}/{month}/customer_communications/suggestions.csv

The generator reads existing monthly banking data, monthly signals, news articles,
and macro events so the records are linked to real synthetic customers and to the
business context already present in the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - script still supports limited CSV-only runs
    pd = None


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"
CORPUS_DIR = BASE_DIR / "corpus_context"
COMMUNICATIONS_SUBDIR = "customer_communications"

SA_TZ = timezone(timedelta(hours=2))

COMPLAINT_CATEGORIES = [
    "fee_dispute",
    "failed_transaction",
    "service_delay",
    "incorrect_charge",
    "fraud_concern",
    "statement_error",
    "digital_issue",
    "staff_behavior",
    "other",
]

SUGGESTION_CATEGORIES = ["digital_features", "branch_service", "fees", "product_idea", "other"]

COMM_FIELDS = [
    "comm_id",
    "account_id",
    "customer_id",
    "channel",
    "direction",
    "timestamp",
    "subject",
    "body",
    "sentiment",
    "linked_news_title",
    "linked_event_date",
    "is_complaint",
    "complaint_category",
]

COMPLAINT_FIELDS = COMM_FIELDS + ["resolution_status", "resolution_notes"]

SUGGESTION_FIELDS = [
    "suggestion_id",
    "account_id",
    "customer_id",
    "channel",
    "timestamp",
    "suggestion_text",
    "category",
    "status",
]


@dataclass
class MonthSignal:
    year: int
    month: int
    tx_count: int = 0
    fail_pct: float = 0.0
    insuf_count: int = 0
    timeout_count: int = 0
    top_loan_type: str = "Home Loan"
    yoy_tx_growth_pct: float | None = None


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
            yoy_raw = row.get("yoy_tx_growth_pct", "")
            signals[(year, month)] = MonthSignal(
                year=year,
                month=month,
                tx_count=parse_int(row.get("tx_count")),
                fail_pct=parse_float(row.get("fail_pct")),
                insuf_count=parse_int(row.get("insuf_count")),
                timeout_count=parse_int(row.get("timeout_count")),
                top_loan_type=row.get("top_loan_type") or "Home Loan",
                yoy_tx_growth_pct=None if yoy_raw in ("", None) else parse_float(yoy_raw),
            )
    return signals


def parse_front_matter(md_path: Path) -> dict[str, Any]:
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip().strip('"')
                if value.startswith("[") and value.endswith("]"):
                    value = [x.strip().strip('"').strip("'") for x in value[1:-1].split(",") if x.strip()]
                meta[key.strip()] = value
            body = parts[2].strip()

    meta["body"] = body
    meta.setdefault("title", md_path.stem)
    meta.setdefault("published", "")
    meta.setdefault("tags", [])
    return meta


def load_news(year: int, month: int) -> list[dict[str, Any]]:
    news_dir = CORPUS_DIR / f"{year}" / f"{month:02d}" / "news"
    if not news_dir.exists():
        return []
    articles = []
    for md_path in sorted(news_dir.glob("*.md")):
        article = parse_front_matter(md_path)
        article["path"] = str(md_path)
        articles.append(article)
    return articles


def load_macro_events() -> list[dict[str, Any]]:
    path = CORPUS_DIR / "events" / "macro_events.jsonl"
    events = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def events_for_month(events: list[dict[str, Any]], year: int, month: int) -> list[dict[str, Any]]:
    out = []
    for event in events:
        try:
            event_date = date.fromisoformat(str(event.get("date", "")))
        except ValueError:
            continue
        if event_date.year == year and event_date.month == month:
            out.append(event)
    return out


def available_channels(year: int, month: int) -> list[str]:
    channels = ["phone_call", "branch_visit"]
    if year > 2020 or (year == 2020 and month >= 4):
        channels.append("app_message")
    if year >= 2021:
        channels.append("sms")
    if year >= 2022:
        channels.append("email")
    if year >= 2023:
        channels.append("social_media")
    if year >= 2024:
        channels.extend(["whatsapp", "chatbot"])
    return channels


def weighted_choice(items: list[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in items)
    if total <= 0:
        return items[0][0]
    pick = random.uniform(0, total)
    upto = 0.0
    for item, weight in items:
        upto += weight
        if upto >= pick:
            return item
    return items[-1][0]


def channel_weights(channels: list[str], year: int) -> list[tuple[str, float]]:
    weights = {
        "phone_call": 2.0 if year <= 2021 else 1.4,
        "branch_visit": 1.8 if year <= 2020 else 1.0,
        "app_message": 1.8,
        "sms": 1.2,
        "email": 1.0,
        "social_media": 0.7,
        "whatsapp": 1.6,
        "chatbot": 1.4,
    }
    return [(channel, weights.get(channel, 1.0)) for channel in channels]


def read_pair_columns(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if pd is None:
        return read_pair_columns_csv_only(path)

    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            return [], []
    except Exception:
        return [], []

    if "account_id" not in df.columns or "customer_id" not in df.columns:
        return [], []

    core = df[["account_id", "customer_id"]].dropna().astype(str)
    pairs = list(core.drop_duplicates().itertuples(index=False, name=None))

    failed_pairs: list[tuple[str, str]] = []
    fail_mask = None
    if "status" in df.columns:
        fail_mask = df["status"].astype(str).str.lower().isin(["failed", "declined", "rejected", "timeout"])
    if "failure_reason" in df.columns:
        reason_mask = df["failure_reason"].notna() & (df["failure_reason"].astype(str).str.strip() != "")
        fail_mask = reason_mask if fail_mask is None else (fail_mask | reason_mask)
    if "has_data_error" in df.columns:
        error_mask = df["has_data_error"].astype(str).str.lower().isin(["true", "1", "yes"])
        fail_mask = error_mask if fail_mask is None else (fail_mask | error_mask)
    if fail_mask is not None:
        failed = df.loc[fail_mask, ["account_id", "customer_id"]].dropna().astype(str)
        failed_pairs = list(failed.drop_duplicates().itertuples(index=False, name=None))

    return pairs, failed_pairs


def read_pair_columns_csv_only(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if path.suffix.lower() != ".csv":
        return [], []
    pairs = set()
    failed = set()
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if "account_id" not in (reader.fieldnames or []) or "customer_id" not in (reader.fieldnames or []):
                return [], []
            for row in reader:
                account_id = row.get("account_id")
                customer_id = row.get("customer_id")
                if not account_id or not customer_id:
                    continue
                pair = (str(account_id), str(customer_id))
                pairs.add(pair)
                status = str(row.get("status", "")).lower()
                reason = str(row.get("failure_reason", "")).strip()
                has_error = str(row.get("has_data_error", "")).lower() in ("true", "1", "yes")
                if status in ("failed", "declined", "rejected", "timeout") or reason or has_error:
                    failed.add(pair)
    except OSError:
        return [], []
    return list(pairs), list(failed)


def load_customer_pool(year: int, month: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    month_dir = BANKING_DIR / f"{year}" / f"{month:02d}"
    if not month_dir.exists():
        return [], []

    pairs: set[tuple[str, str]] = set()
    failed_pairs: set[tuple[str, str]] = set()
    priority = [
        f"loan_payment_transactions_{year}_{month:02d}.parquet",
        f"debit_order_transactions_{year}_{month:02d}.parquet",
        f"accounts_{year}_{month:02d}.parquet",
        f"customers_{year}_{month:02d}.parquet",
        f"loan_payment_transactions_{year}_{month:02d}.csv",
    ]
    paths = [month_dir / name for name in priority if (month_dir / name).exists()]
    paths.extend(p for p in sorted(month_dir.glob("*.parquet")) if p not in paths)
    paths.extend(p for p in sorted(month_dir.glob("*.csv")) if p not in paths)

    for path in paths:
        found, failed = read_pair_columns(path)
        pairs.update(found)
        failed_pairs.update(failed)
        if len(pairs) >= 1000 and len(failed_pairs) >= 50:
            break

    return sorted(pairs), sorted(failed_pairs)


def random_timestamp(year: int, month: int, channel: str, after_day: int | None = None) -> datetime:
    last_day = monthrange(year, month)[1]
    start_day = min(max((after_day or 1), 1), last_day)
    day = random.randint(start_day, last_day)

    if channel == "branch_visit":
        hour = random.randint(8, 15)
        minute = random.choice([0, 5, 10, 15, 20, 30, 45, 50])
    elif channel == "phone_call":
        hour = random.randint(7, 18)
        minute = random.randint(0, 59)
    elif channel in ("sms", "app_message", "whatsapp", "chatbot"):
        hour = random.choices(range(24), weights=[1, 1, 1, 1, 1, 2, 4, 6, 7, 8, 8, 8, 7, 7, 7, 8, 9, 10, 9, 7, 5, 3, 2, 1])[0]
        minute = random.randint(0, 59)
    else:
        hour = random.randint(6, 22)
        minute = random.randint(0, 59)

    return datetime(year, month, day, hour, minute, random.randint(0, 59), tzinfo=SA_TZ)


def format_timestamp(ts: datetime, channel: str) -> str:
    if channel in ("app_message", "sms", "chatbot"):
        return ts.isoformat(timespec="seconds")
    if channel == "email":
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if channel in ("phone_call", "branch_visit"):
        return ts.strftime("%d/%m/%Y %H:%M")
    if channel == "social_media":
        return ts.strftime("%Y/%m/%d %H:%M:%S")
    if channel == "whatsapp":
        return ts.strftime("%d-%b-%Y %H:%M")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def select_customer(pairs: list[tuple[str, str]], failed_pairs: list[tuple[str, str]], prefer_failed: bool) -> tuple[str, str]:
    pool = failed_pairs if prefer_failed and failed_pairs and random.random() < 0.75 else pairs
    if not pool:
        return "UNKNOWN_ACCOUNT", "UNKNOWN_CUSTOMER"
    return random.choice(pool)


def classify_news(article: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(article.get("title", "")),
            " ".join(article.get("tags", []) if isinstance(article.get("tags"), list) else [str(article.get("tags", ""))]),
            str(article.get("body", ""))[:600],
        ]
    ).lower()
    if any(word in text for word in ["repo", "interest", "rate", "loan", "arrears", "collections"]):
        return "interest_rate"
    if any(word in text for word in ["fee", "charge", "statement", "reconciliation"]):
        return "fees"
    if any(word in text for word in ["fraud", "suspicious", "scam"]):
        return "fraud"
    if any(word in text for word in ["load shedding", "eskom", "atm", "branch", "outage", "resilience", "timeout"]):
        return "load_shedding"
    if any(word in text for word in ["app", "digital", "wallet", "ussd", "channel"]):
        return "digital"
    return "general"


def direction_for_channel(channel: str, is_complaint: bool) -> str:
    if channel == "sms":
        return "outbound" if random.random() < 0.82 else "inbound"
    if channel in ("social_media", "branch_visit", "phone_call"):
        return "inbound"
    if channel in ("app_message", "whatsapp", "chatbot", "email"):
        return "inbound" if is_complaint or random.random() < 0.72 else "outbound"
    return "inbound"


def body_prefix(channel: str) -> str:
    if channel in ("phone_call", "whatsapp", "social_media") and random.random() < 0.08:
        return random.choice(["Howzit, ", "Eish, ", "Hi there, "])
    return ""


def complaint_category_for(topic: str, channel: str) -> str:
    if topic == "interest_rate":
        return random.choice(["failed_transaction", "fee_dispute", "service_delay"])
    if topic == "fees":
        return random.choice(["fee_dispute", "incorrect_charge", "statement_error"])
    if topic == "fraud":
        return "fraud_concern"
    if topic == "load_shedding":
        return random.choice(["service_delay", "failed_transaction", "digital_issue"])
    if topic == "digital" or channel in ("app_message", "chatbot", "whatsapp"):
        return random.choice(["digital_issue", "service_delay", "failed_transaction"])
    return random.choice(COMPLAINT_CATEGORIES)


def choose_general_topic(signal: MonthSignal, channel: str, is_complaint: bool) -> str:
    if is_complaint and signal.fail_pct > 6 and random.random() < 0.45:
        return "failed_transaction"
    options = {
        "phone_call": ["failed_transaction", "fees", "interest_rate", "card_atm", "branch"],
        "branch_visit": ["card_atm", "branch", "statement", "cash_deposit", "service"],
        "app_message": ["balance", "transfer_delay", "digital", "wallet", "fees"],
        "sms": ["failed_transaction", "statement", "fraud", "branch"],
        "email": ["statement", "dispute", "digital", "suggestion", "fees"],
        "social_media": ["digital", "fees", "service", "praise"],
        "whatsapp": ["digital", "fees", "agent_transfer", "password"],
        "chatbot": ["password", "agent_transfer", "fees", "digital"],
    }
    return random.choice(options.get(channel, ["general"]))


def render_comm_content(
    *,
    channel: str,
    direction: str,
    is_complaint: bool,
    signal: MonthSignal,
    news: dict[str, Any] | None,
    event: dict[str, Any] | None,
) -> tuple[str, str, str, str]:
    linked_title = news.get("title", "") if news else ""
    topic = classify_news(news) if news else choose_general_topic(signal, channel, is_complaint)
    prefix = body_prefix(channel)
    loan_type = signal.top_loan_type or "home loan"

    news_line = ""
    if news:
        news_line = f" I saw the article '{linked_title}' and wanted to check how it affects my account."
    event_line = ""
    if event:
        event_line = f" This also seems connected to {event.get('title', 'the recent market update')}."

    templates = {
        "failed_transaction": (
            "Why did my debit order fail?",
            f"{prefix}My debit order did not go through and I need to know whether it was insufficient funds, a retry problem or a bank processing issue. The account should not be charged twice.{news_line}{event_line}",
        ),
        "fees": (
            "Unauthorised bank charge enquiry",
            f"{prefix}Please explain the latest bank charge on my statement. I cannot see why this fee was taken and I want it reversed if it was incorrect.{news_line}{event_line}",
        ),
        "interest_rate": (
            f"{loan_type} rate increase question",
            f"{prefix}I am worried my {loan_type.lower()} instalment may change. Please confirm whether my repayment will be repriced and from which date.{news_line}{event_line}",
        ),
        "fraud": (
            "Suspicious transaction alert",
            f"{prefix}There is activity on my account that I do not recognise. Please block the card if needed and tell me what evidence you require for the dispute.{news_line}{event_line}",
        ),
        "load_shedding": (
            "ATM and branch access issue",
            f"{prefix}I could not use the ATM or branch service properly during the outage. I need help with late fees and a clear explanation of what happened.{news_line}{event_line}",
        ),
        "digital": (
            "App login problem",
            f"{prefix}The app is not behaving properly and I cannot complete what should be a simple transaction. Please check whether there is an issue on my profile.{news_line}{event_line}",
        ),
        "card_atm": (
            "Card blocked at ATM",
            f"{prefix}My card was blocked at the ATM and I need access restored or a replacement card arranged. I have an urgent payment to make.",
        ),
        "branch": (
            "Branch appointment request",
            "I would like to book time at the branch to sort out my account and get printed documents for my records.",
        ),
        "statement": (
            "Statement copy request",
            "Please send me a copy of my latest statement and confirm the closing balance used for the month-end calculation.",
        ),
        "transfer_delay": (
            "Transfer not reflecting",
            f"{prefix}I made a transfer and it is still not reflecting. Please check whether the payment is delayed in your system.",
        ),
        "wallet": (
            "Digital wallet setup help",
            "I need help setting up the digital wallet and linking it to the right account. The app keeps asking me to try again later.",
        ),
        "cash_deposit": (
            "Cash deposit issue",
            "I made a cash deposit but the amount is not showing correctly. Please investigate the branch or ATM record.",
        ),
        "service": (
            "Complaint about service",
            f"{prefix}The service delay is frustrating. I have followed up more than once and still do not have a proper answer.",
        ),
        "praise": (
            "Kudos to branch staff",
            "The branch team helped me quickly and explained the account options clearly. Please pass on my thanks.",
        ),
        "agent_transfer": (
            "Agent transfer request",
            "The automated response is not solving my issue. Please transfer me to a consultant who can check the account.",
        ),
        "password": (
            "Reset password",
            "I need help resetting my password and confirming that my profile has not been locked.",
        ),
        "balance": (
            "Balance check",
            "Please confirm my available balance and whether any debit orders are still pending.",
        ),
        "suggestion": (
            "Suggestion for bank app",
            "I have feedback on the app experience and would like the product team to review it.",
        ),
    }
    subject, body = templates.get(topic, ("Account enquiry", "Please assist with my banking enquiry."))

    if direction == "outbound":
        outbound = {
            "failed_transaction": (
                "Your loan repayment failed due to insufficient funds",
                "Your scheduled loan repayment was not successful. Please fund your account or contact us to discuss payment options.",
            ),
            "statement": (
                "New statement available",
                "Your latest account statement is available. Please review the charges, interest and debit order activity.",
            ),
            "fraud": (
                "Suspicious transaction alert",
                "We detected activity that may require confirmation. Please reply or contact the bank if this transaction was not yours.",
            ),
            "branch": (
                "Branch holiday hours",
                "Please note adjusted branch hours this month. Digital channels remain available for balance checks and transfers.",
            ),
        }
        subject, body = outbound.get(topic, outbound.get("statement"))

    sentiment = "negative" if is_complaint else random.choices(["neutral", "positive", "negative"], weights=[70, 20, 10])[0]
    return subject, body, sentiment, topic


def pick_context(news_articles: list[dict[str, Any]], events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    use_news = news_articles and random.random() < 0.18
    use_event = events and random.random() < 0.12
    article = random.choice(news_articles) if use_news else None
    event = random.choice(events) if use_event else None
    anchor_days = []
    if article and article.get("published"):
        try:
            anchor_days.append(date.fromisoformat(str(article["published"])).day)
        except ValueError:
            pass
    if event and event.get("date"):
        try:
            anchor_days.append(date.fromisoformat(str(event["date"])).day)
        except ValueError:
            pass
    after_day = min(anchor_days) if anchor_days else None
    return article, event, after_day


def generate_communication(
    *,
    idx: int,
    year: int,
    month: int,
    signal: MonthSignal,
    channels: list[str],
    pairs: list[tuple[str, str]],
    failed_pairs: list[tuple[str, str]],
    news_articles: list[dict[str, Any]],
    events: list[dict[str, Any]],
    is_complaint: bool,
) -> dict[str, Any]:
    channel = weighted_choice(channel_weights(channels, year))
    direction = direction_for_channel(channel, is_complaint)
    article, event, after_day = pick_context(news_articles, events)
    account_id, customer_id = select_customer(pairs, failed_pairs, prefer_failed=is_complaint)
    ts = random_timestamp(year, month, channel, after_day=after_day)
    subject, body, sentiment, topic = render_comm_content(
        channel=channel,
        direction=direction,
        is_complaint=is_complaint,
        signal=signal,
        news=article,
        event=event,
    )
    category = complaint_category_for(topic, channel) if is_complaint else ""
    return {
        "comm_id": f"COMM-{year}{month:02d}-{idx:06d}",
        "account_id": account_id,
        "customer_id": customer_id,
        "channel": channel,
        "direction": direction,
        "timestamp": format_timestamp(ts, channel),
        "subject": subject,
        "body": body,
        "sentiment": sentiment,
        "linked_news_title": article.get("title", "") if article else "",
        "linked_event_date": event.get("date", "") if event else "",
        "is_complaint": str(bool(is_complaint)),
        "complaint_category": category,
    }


def resolution_for(category: str) -> tuple[str, str]:
    status = random.choices(["open", "resolved", "escalted"], weights=[22, 66, 12])[0]
    notes = {
        "fee_dispute": "Fee query logged; customer will receive a reversal decision after statement review.",
        "failed_transaction": "Payment operations checking debit-order status, retry timing and available balance.",
        "service_delay": "Service timeline reviewed and customer updated on next action.",
        "incorrect_charge": "Charge sent to finance controls for validation against tariff table.",
        "fraud_concern": "Card monitoring case opened and customer advised on dispute documents.",
        "statement_error": "Statement extract queued for reconciliation review.",
        "digital_issue": "Digital support case created with device and login details.",
        "staff_behavior": "Branch manager to contact customer and review service notes.",
        "other": "Customer care team reviewing the account history.",
    }
    return status, notes.get(category, notes["other"])


def generate_suggestion(
    *,
    idx: int,
    year: int,
    month: int,
    channels: list[str],
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    non_sms = [c for c in channels if c != "sms"] or channels
    channel = weighted_choice(channel_weights(non_sms, year))
    account_id, customer_id = select_customer(pairs, [], prefer_failed=False)
    ts = random_timestamp(year, month, channel)
    category = random.choices(
        SUGGESTION_CATEGORIES,
        weights=[34, 18, 22, 16, 10],
    )[0]
    texts = {
        "digital_features": [
            "Please add a simple way to see pending debit orders before month end.",
            "It would help if the app showed why a transfer is delayed instead of only saying pending.",
            "Please add better card freeze and unfreeze controls in the app.",
        ],
        "branch_service": [
            "Please allow customers to book branch appointments from the app.",
            "The branch queue system should send an SMS when it is nearly your turn.",
        ],
        "fees": [
            "Please show bank charges before the customer confirms a payment.",
            "A monthly fee breakdown would make the statement easier to understand.",
        ],
        "product_idea": [
            "Please offer a small savings pocket linked to loan repayments.",
            "It would be useful to have a low-cost account for family remittances.",
        ],
        "other": [
            "Please make statements easier to download for tax and rental applications.",
            "Please send clearer notifications when account details change.",
        ],
    }
    return {
        "suggestion_id": f"SUG-{year}{month:02d}-{idx:05d}",
        "account_id": account_id,
        "customer_id": customer_id,
        "channel": channel,
        "timestamp": format_timestamp(ts, channel),
        "suggestion_text": random.choice(texts[category]),
        "category": category,
        "status": random.choices(["submitted", "reviewed", "implemented"], weights=[68, 26, 6])[0],
    }


def target_counts(signal: MonthSignal) -> tuple[int, int, int]:
    total = max(10, round(signal.tx_count * 0.005 + signal.fail_pct * 50))
    complaint_rate = 0.30
    if signal.fail_pct > 8:
        complaint_rate += 0.12
    elif signal.fail_pct > 6:
        complaint_rate += 0.07
    complaints = max(1, round(total * complaint_rate))

    suggestion_rate = 0.05
    if signal.yoy_tx_growth_pct is not None and signal.yoy_tx_growth_pct < 0:
        suggestion_rate += 0.03
    suggestions = max(1, round(total * suggestion_rate))
    return total, min(complaints, total), suggestions


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_month(year: int, month: int, signal: MonthSignal, events_all: list[dict[str, Any]]) -> tuple[int, int, int]:
    random.seed(2019 * 100 + year * 100 + month)

    pairs, failed_pairs = load_customer_pool(year, month)
    if not pairs:
        return 0, 0, 0

    news_articles = load_news(year, month)
    month_events = events_for_month(events_all, year, month)
    channels = available_channels(year, month)
    total, complaint_total, suggestion_total = target_counts(signal)

    complaint_slots = set(random.sample(range(total), k=min(complaint_total, total)))
    communications = []
    complaints = []

    for idx in range(1, total + 1):
        row = generate_communication(
            idx=idx,
            year=year,
            month=month,
            signal=signal,
            channels=channels,
            pairs=pairs,
            failed_pairs=failed_pairs,
            news_articles=news_articles,
            events=month_events,
            is_complaint=(idx - 1) in complaint_slots,
        )
        communications.append(row)
        if row["is_complaint"] == "True":
            status, notes = resolution_for(row["complaint_category"])
            complaints.append({**row, "resolution_status": status, "resolution_notes": notes})

    suggestions = [
        generate_suggestion(idx=i, year=year, month=month, channels=channels, pairs=pairs)
        for i in range(1, suggestion_total + 1)
    ]

    out_dir = BANKING_DIR / f"{year}" / f"{month:02d}" / COMMUNICATIONS_SUBDIR
    write_csv(out_dir / "communications.csv", communications, COMM_FIELDS)
    write_csv(out_dir / "complaints.csv", complaints, COMPLAINT_FIELDS)
    write_csv(out_dir / "suggestions.csv", suggestions, SUGGESTION_FIELDS)
    return len(communications), len(complaints), len(suggestions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic customer communication records.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    signals = load_monthly_signals()
    events = load_macro_events()
    total_comms = 0
    total_complaints = 0
    total_suggestions = 0
    months_written = 0

    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            signal = signals.get((year, month), MonthSignal(year=year, month=month))
            comms, complaints, suggestions = generate_month(year, month, signal, events)
            if comms == 0:
                continue
            months_written += 1
            total_comms += comms
            total_complaints += complaints
            total_suggestions += suggestions

    print(f"Customer communications written under {BANKING_DIR}\\<year>\\<month>\\{COMMUNICATIONS_SUBDIR}")
    print(f"  Months: {months_written}")
    print(f"  Communications: {total_comms}")
    print(f"  Complaints: {total_complaints}")
    print(f"  Suggestions: {total_suggestions}")


if __name__ == "__main__":
    main()
