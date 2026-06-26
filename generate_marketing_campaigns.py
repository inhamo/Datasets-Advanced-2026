"""
Generate synthetic marketing campaigns and customer responses for a South African
retail bank across 2019-2025.

Outputs:
  banking_data/{year}/{month}/marketing_campaigns/campaigns.csv
  banking_data/{year}/{month}/marketing_campaigns/campaign_responses.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"
CORPUS_DIR = BASE_DIR / "corpus_context"
MARKETING_SUBDIR = "marketing_campaigns"
FIRST_MARKETING_MONTH = (2019, 7)

CAMPAIGN_FIELDS = [
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "target_segment",
    "channel",
    "product_focus",
    "offer_summary",
    "start_date",
    "end_date",
    "budget_zar",
    "target_customers_count",
    "region",
    "status",
    "success_metric",
]

RESPONSE_FIELDS = [
    "response_id",
    "campaign_id",
    "customer_id",
    "account_id",
    "response_date",
    "response_type",
    "conversion_value_zar",
    "channel_used",
    "notes",
]

CAMPAIGN_TYPES = [
    "acquisition",
    "cross_sell",
    "retention",
    "reactivation",
    "brand_awareness",
    "digital_adoption",
]

TARGET_SEGMENTS = [
    "youth_18_25",
    "salaried_professionals",
    "sme_business",
    "low_income",
    "dormant_accounts",
    "high_net_worth",
    "home_loan_holders",
]

PRODUCT_FOCUS = [
    "digital_wallet",
    "home_loan",
    "vehicle_finance",
    "credit_card",
    "savings_account",
    "personal_loan",
    "funeral_plan",
]

PROVINCES = [
    "National",
    "Gauteng",
    "Western Cape",
    "KwaZulu-Natal",
    "Eastern Cape",
    "Limpopo",
    "Mpumalanga",
    "Free State",
    "North West",
    "Northern Cape",
    "Johannesburg",
    "Cape Town",
    "Durban",
    "Pretoria",
]


@dataclass
class MonthSignal:
    year: int
    month: int
    yoy_tx_growth_pct: float | None = None
    fail_pct: float = 0.0
    top_loan_type: str = "Home Loan"


def parse_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None, "nan"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_monthly_signals() -> dict[tuple[int, int], MonthSignal]:
    path = CORPUS_DIR / "index" / "monthly_signals.csv"
    signals: dict[tuple[int, int], MonthSignal] = {}
    if not path.exists():
        return signals
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = int(row["year"])
            month = int(row["month"])
            yoy_raw = row.get("yoy_tx_growth_pct", "")
            signals[(year, month)] = MonthSignal(
                year=year,
                month=month,
                yoy_tx_growth_pct=None if yoy_raw in ("", None) else parse_float(yoy_raw),
                fail_pct=parse_float(row.get("fail_pct")),
                top_loan_type=row.get("top_loan_type") or "Home Loan",
            )
    return signals


def parse_news_md(md_path: Path) -> dict[str, Any]:
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
    return [parse_news_md(path) for path in sorted(news_dir.glob("*.md"))]


def news_theme(article: dict[str, Any]) -> str:
    tags = article.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]
    text = " ".join([article.get("title", ""), " ".join(tags), article.get("body", "")[:700]]).lower()
    if any(x in text for x in ["repo", "rate", "interest", "home loan", "arrears"]):
        return "rate_hike"
    if any(x in text for x in ["load shedding", "eskom", "outage", "resilience", "atm"]):
        return "load_shedding"
    if any(x in text for x in ["covid", "lockdown", "relief", "payment holiday"]):
        return "covid_relief"
    if any(x in text for x in ["digital", "wallet", "app", "ussd", "channel"]):
        return "digital"
    if any(x in text for x in ["fee", "charge", "statement"]):
        return "fees"
    return "general"


def available_channels(year: int, month: int) -> list[str]:
    channels = ["radio", "tv", "outdoor", "branch_poster", "sms"]
    if year > 2020 or (year == 2020 and month >= 4):
        channels.extend(["email", "app_push"])
    if year >= 2021:
        channels.append("social_media")
    if year >= 2023:
        channels.append("whatsapp")
    if year >= 2024:
        channels.append("chatbot")
    return channels


def campaigns_per_year(year: int, avg_yoy: float | None) -> int:
    ranges = {
        2019: (4, 6),
        2020: (6, 8),
        2021: (8, 10),
        2022: (10, 12),
        2023: (12, 14),
        2024: (14, 16),
        2025: (16, 18),
    }
    low, high = ranges.get(year, (10, 12))
    count = random.randint(low, high)
    if avg_yoy is not None and avg_yoy < 0:
        count = min(high, count + 1)
    elif avg_yoy is not None and avg_yoy > 18:
        count = max(low, count - 1)
    return count


def read_customer_month(year: int, month: int) -> list[dict[str, str]]:
    month_dir = BANKING_DIR / f"{year}" / f"{month:02d}"
    customers_path = month_dir / f"customers_{year}_{month:02d}.parquet"
    accounts_path = month_dir / f"accounts_{year}_{month:02d}.parquet"

    if pd is not None and customers_path.exists():
        try:
            customers = pd.read_parquet(customers_path)
            accounts = pd.read_parquet(accounts_path) if accounts_path.exists() else None
            account_map: dict[str, str] = {}
            if accounts is not None and {"customer_id", "account_id"}.issubset(accounts.columns):
                sample = accounts[["customer_id", "account_id"]].dropna().astype(str)
                account_map = dict(sample.drop_duplicates("customer_id").itertuples(index=False, name=None))
            rows: list[dict[str, str]] = []
            for _, row in customers.iterrows():
                customer_id = str(row.get("customer_id", ""))
                if not customer_id:
                    continue
                rows.append(
                    {
                        "customer_id": customer_id,
                        "account_id": account_map.get(customer_id, ""),
                        "customer_segment": str(row.get("customer_segment", "") or ""),
                        "income_band": str(row.get("income_band", "") or ""),
                        "digital_exposure_level": str(row.get("digital_exposure_level", "") or ""),
                        "birth_date": str(row.get("birth_date", "") or ""),
                        "occupation": str(row.get("occupation", "") or ""),
                    }
                )
            return rows
        except Exception:
            pass

    csv_path = month_dir / f"loan_payment_transactions_{year}_{month:02d}.csv"
    if not csv_path.exists():
        return []
    seen = set()
    rows = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            customer_id = row.get("customer_id", "")
            account_id = row.get("account_id", "")
            if not customer_id or customer_id in seen:
                continue
            seen.add(customer_id)
            rows.append(
                {
                    "customer_id": customer_id,
                    "account_id": account_id,
                    "customer_segment": "",
                    "income_band": "",
                    "digital_exposure_level": "",
                    "birth_date": "",
                    "occupation": "",
                }
            )
    return rows


def segment_customer(customer: dict[str, str], year: int) -> str:
    segment = customer.get("customer_segment", "").lower()
    income = customer.get("income_band", "").lower()
    occupation = customer.get("occupation", "").lower()
    birth_date = customer.get("birth_date", "")

    age = None
    try:
        age = year - date.fromisoformat(birth_date[:10]).year
    except ValueError:
        pass

    if age is not None and 18 <= age <= 25:
        return "youth_18_25"
    if "business" in segment or "owner" in occupation or "director" in occupation:
        return "sme_business"
    if any(x in income for x in ["very low", "low", "lower"]):
        return "low_income"
    if any(x in income for x in ["high", "upper"]):
        return "high_net_worth"
    if any(x in occupation for x in ["teacher", "nurse", "engineer", "manager", "analyst", "accountant"]):
        return "salaried_professionals"
    return random.choice(["salaried_professionals", "low_income", "dormant_accounts"])


def customer_pool_for_segment(customers: list[dict[str, str]], segment: str, year: int) -> list[dict[str, str]]:
    matched = [c for c in customers if segment_customer(c, year) == segment]
    return matched or customers


def infer_product_focus_from_catalogue(year: int, month: int) -> list[str]:
    month_dir = BANKING_DIR / f"{year}" / f"{month:02d}"
    products = set(PRODUCT_FOCUS)
    if pd is None:
        return sorted(products)
    for path in [month_dir / f"accounts_{year}_{month:02d}.parquet", month_dir / f"loans_{year}_{month:02d}.parquet"]:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        for col in ["bank_product_name", "account_type", "loan_type", "product_type"]:
            if col not in df.columns:
                continue
            values = " ".join(df[col].dropna().astype(str).head(200).tolist()).lower()
            if "home" in values:
                products.add("home_loan")
            if "vehicle" in values or "car" in values:
                products.add("vehicle_finance")
            if "credit" in values:
                products.add("credit_card")
            if "saving" in values:
                products.add("savings_account")
            if "personal" in values:
                products.add("personal_loan")
    return sorted(products)


def choose_campaign_template(year: int, segment: str, signal: MonthSignal, article: dict[str, Any] | None) -> dict[str, str]:
    theme = news_theme(article) if article else "general"
    if theme == "rate_hike":
        return {
            "campaign_name": "Fixed-Rate Home Loan - Lock In Now",
            "campaign_type": "retention",
            "target_segment": "home_loan_holders",
            "product_focus": "home_loan",
            "offer_summary": "Preferential fixed-rate quote for qualifying home loan customers",
            "success_metric": "conversion_rate",
        }
    if theme == "load_shedding":
        return {
            "campaign_name": "Solar Finance - Power Your Home",
            "campaign_type": "cross_sell",
            "target_segment": "salaried_professionals",
            "product_focus": "personal_loan",
            "offer_summary": "Discounted initiation fee on backup power and solar finance",
            "success_metric": "new_accounts",
        }
    if theme == "covid_relief" or (year == 2020 and random.random() < 0.25):
        return {
            "campaign_name": "We're Here to Help - Restructure Your Debt",
            "campaign_type": "retention",
            "target_segment": "low_income",
            "product_focus": "personal_loan",
            "offer_summary": "Payment relief and restructuring support for affected customers",
            "success_metric": "conversion_rate",
        }

    templates_by_year = {
        2019: [
            ("Welcome to Keystone - Your Future Starts Here", "brand_awareness", "salaried_professionals", "savings_account", "No monthly account fee for the first three months", "new_accounts"),
            ("Home Loan Season - Fixed Rate Special", "cross_sell", "home_loan_holders", "home_loan", "Fixed-rate quote and bond registration support", "conversion_rate"),
            ("Switch Your Salary and Save", "acquisition", "salaried_professionals", "savings_account", "Reduced monthly fees when salary is paid into the account", "new_accounts"),
            ("Student Account - No Fees, No Worries", "acquisition", "youth_18_25", "savings_account", "Zero monthly account fees for qualifying students", "new_accounts"),
        ],
        2020: [
            ("COVID Relief - Payment Holiday Enquiries", "retention", "low_income", "personal_loan", "Payment holiday assessment for customers under pressure", "conversion_rate"),
            ("Bank From Home - Digital Account Opening", "digital_adoption", "salaried_professionals", "digital_wallet", "Open and verify your account without visiting a branch", "app_downloads"),
            ("Essential Worker Appreciation - Reduced Fees", "retention", "salaried_professionals", "savings_account", "Reduced account fees for qualifying essential workers", "conversion_rate"),
            ("WhatsApp Banking Beta - Join Now", "digital_adoption", "youth_18_25", "digital_wallet", "Early access to simple balance and payment features", "app_downloads"),
        ],
        2021: [
            ("Go Digital, Get Rewarded - R100 eWallet Bonus", "digital_adoption", "youth_18_25", "digital_wallet", "R100 eWallet reward after qualifying digital activity", "app_downloads"),
            ("Funeral Cover - Because Family Matters", "cross_sell", "low_income", "funeral_plan", "First month premium discounted for new policyholders", "conversion_rate"),
            ("Small Business Boost - SME Loan Special", "acquisition", "sme_business", "personal_loan", "Preferential pricing for qualifying SME working capital", "new_accounts"),
            ("Scan to Pay - Tap, Go, Done", "digital_adoption", "youth_18_25", "digital_wallet", "Rewards for first scan-to-pay transactions", "app_downloads"),
        ],
        2022: [
            ("Credit Card Upgrade - Double Your Rewards", "cross_sell", "salaried_professionals", "credit_card", "Double rewards points for three months", "conversion_rate"),
            ("Green Home Loan - Lower Rates for Solar", "cross_sell", "home_loan_holders", "home_loan", "Lower rate review for energy-efficient home improvements", "conversion_rate"),
            ("Youth Month - Open an Account, Get AirTime", "acquisition", "youth_18_25", "savings_account", "Free airtime voucher after account activation", "new_accounts"),
            ("Black Friday Banking - Zero Fees on EFT", "retention", "low_income", "savings_account", "Zero EFT fees on qualifying digital transactions", "conversion_rate"),
        ],
        2023: [
            ("AI-Powered Budgeting - Free for 3 Months", "digital_adoption", "salaried_professionals", "digital_wallet", "Free budgeting insights for app customers", "app_downloads"),
            ("Travel-Ready Card - No Forex Fees Abroad", "cross_sell", "high_net_worth", "credit_card", "No forex fees on qualifying card spend abroad", "conversion_rate"),
            ("Back-to-School Loan - Affordable Repayments", "cross_sell", "low_income", "personal_loan", "Education expense loan with flexible repayment terms", "conversion_rate"),
            ("Digital Wallet - Send Money Like a Message", "digital_adoption", "youth_18_25", "digital_wallet", "Discounted send-money fees for first-time wallet users", "app_downloads"),
        ],
        2024: [
            ("Chatbot Banking - Ask, Pay, Save", "digital_adoption", "salaried_professionals", "digital_wallet", "Faster self-service support through chatbot banking", "app_downloads"),
            ("Load Shedding Special - Backup Power Loan", "cross_sell", "salaried_professionals", "personal_loan", "Reduced initiation fee for backup power finance", "conversion_rate"),
            ("Wealth Builder - Start Investing with R100", "cross_sell", "high_net_worth", "savings_account", "Start a savings and investment plan from R100", "new_accounts"),
            ("Family Account Bundle - One Fee, Four Accounts", "retention", "salaried_professionals", "savings_account", "Bundled family banking for one monthly fee", "conversion_rate"),
        ],
        2025: [
            ("AI Financial Coach - Personalised Money Tips", "digital_adoption", "salaried_professionals", "digital_wallet", "Personalised app insights free for three months", "app_downloads"),
            ("Global Citizen Account - Multi-Currency Wallet", "acquisition", "high_net_worth", "digital_wallet", "Multi-currency wallet with preferential launch pricing", "new_accounts"),
            ("Retire Ready - Retirement Annuity Special", "cross_sell", "salaried_professionals", "savings_account", "Reduced platform fee on retirement annuity applications", "conversion_rate"),
            ("Next-Gen Banking - Everything in One App", "brand_awareness", "youth_18_25", "digital_wallet", "One app for payments, insights and savings pockets", "app_downloads"),
        ],
    }
    name, ctype, default_segment, product, offer, metric = random.choice(templates_by_year.get(year, templates_by_year[2025]))
    return {
        "campaign_name": name,
        "campaign_type": ctype,
        "target_segment": default_segment,
        "product_focus": product,
        "offer_summary": offer,
        "success_metric": metric,
    }


def choose_channels(year: int, month: int, campaign_type: str, target_segment: str) -> str:
    available = available_channels(year, month)
    preferred = {
        "digital_adoption": ["app_push", "sms", "email", "social_media", "whatsapp", "chatbot"],
        "acquisition": ["radio", "social_media", "outdoor", "sms", "tv"],
        "cross_sell": ["sms", "email", "app_push", "whatsapp", "branch_poster"],
        "retention": ["sms", "email", "app_push", "whatsapp", "branch_poster"],
        "reactivation": ["sms", "email", "whatsapp", "branch_poster"],
        "brand_awareness": ["radio", "tv", "outdoor", "social_media", "branch_poster"],
    }
    candidates = [c for c in preferred.get(campaign_type, available) if c in available]
    if target_segment == "youth_18_25":
        candidates = [c for c in ["social_media", "app_push", "whatsapp", "sms", "email"] if c in available] or candidates
    elif target_segment in ("low_income", "dormant_accounts"):
        candidates = [c for c in ["sms", "branch_poster", "radio", "whatsapp"] if c in available] or candidates

    count = min(len(candidates), random.choices([1, 2, 3], weights=[30, 50, 20])[0])
    return ",".join(random.sample(candidates, k=max(1, count)))


def response_rates(channel: str) -> dict[str, float]:
    rates = {
        "email": {"opened": 0.22, "clicked": 0.04, "converted": 0.012},
        "sms": {"opened": 0.31, "clicked": 0.065, "converted": 0.02},
        "app_push": {"opened": 0.13, "clicked": 0.04, "converted": 0.03},
        "social_media": {"opened": 0.02, "clicked": 0.015, "converted": 0.004},
        "whatsapp": {"opened": 0.46, "clicked": 0.10, "converted": 0.035},
        "branch_poster": {"opened": 0.015, "clicked": 0.006, "converted": 0.003},
        "radio": {"opened": 0.006, "clicked": 0.002, "converted": 0.001},
        "tv": {"opened": 0.008, "clicked": 0.003, "converted": 0.0015},
        "outdoor": {"opened": 0.005, "clicked": 0.002, "converted": 0.001},
        "chatbot": {"opened": 0.16, "clicked": 0.055, "converted": 0.025},
    }
    return rates.get(channel, {"opened": 0.04, "clicked": 0.01, "converted": 0.003})


def segment_conversion_multiplier(segment: str, channel: str) -> float:
    if segment == "youth_18_25" and channel in ("app_push", "social_media", "whatsapp", "chatbot"):
        return 1.35
    if segment == "high_net_worth" and channel in ("email", "branch_poster"):
        return 1.25
    if segment == "low_income" and channel in ("sms", "radio", "branch_poster", "whatsapp"):
        return 1.2
    if segment == "sme_business" and channel in ("email", "sms"):
        return 1.15
    return 1.0


def conversion_value(product_focus: str) -> int:
    ranges = {
        "digital_wallet": (20, 350),
        "home_loan": (2500, 25000),
        "vehicle_finance": (1500, 14000),
        "credit_card": (150, 2500),
        "savings_account": (100, 8000),
        "personal_loan": (500, 7000),
        "funeral_plan": (60, 450),
    }
    low, high = ranges.get(product_focus, (50, 1000))
    return random.randint(low, high)


def response_type_for(channel: str, segment: str, product_focus: str) -> str:
    rates = response_rates(channel)
    convert_rate = rates["converted"] * segment_conversion_multiplier(segment, channel)
    complaint_or_optout = random.uniform(0.02, 0.05)
    roll = random.random()
    if roll < convert_rate:
        return "converted"
    if roll < convert_rate + complaint_or_optout:
        return random.choice(["complained", "opted_out"])
    if roll < convert_rate + complaint_or_optout + rates.get("clicked", 0.0):
        return "clicked"
    if roll < convert_rate + complaint_or_optout + rates.get("clicked", 0.0) + rates.get("opened", 0.0):
        return "opened"
    return "ignored"


def campaign_status(start: date, end: date) -> str:
    if random.random() < 0.025:
        return "cancelled"
    today = date(2026, 5, 19)
    if start > today:
        return "planned"
    if start <= today <= end:
        return "active"
    return "completed"


def generate_campaigns(start_year: int, end_year: int, signals: dict[tuple[int, int], MonthSignal]) -> list[dict[str, Any]]:
    campaigns: list[dict[str, Any]] = []
    annual_seen: dict[int, int] = {}

    for year in range(start_year, end_year + 1):
        yoy_values = [s.yoy_tx_growth_pct for (y, _), s in signals.items() if y == year and s.yoy_tx_growth_pct is not None]
        avg_yoy = sum(yoy_values) / len(yoy_values) if yoy_values else None
        count = campaigns_per_year(year, avg_yoy)
        eligible_months = [
            month
            for month in range(1, 13)
            if (year, month) >= FIRST_MARKETING_MONTH
        ]
        if not eligible_months:
            continue
        months = sorted(random.choices(eligible_months, k=count))

        for month in months:
            annual_seen[year] = annual_seen.get(year, 0) + 1
            signal = signals.get((year, month), MonthSignal(year=year, month=month))
            month_news = load_news(year, month)
            article = random.choice(month_news) if month_news and random.random() < 0.22 else None
            segment = random.choice(TARGET_SEGMENTS)
            template = choose_campaign_template(year, segment, signal, article)
            start_day = random.randint(1, min(24, monthrange(year, month)[1]))
            start = date(year, month, start_day)
            duration = random.randint(14, 58)
            end = start + timedelta(days=duration)
            if end.year > year:
                end = date(year, 12, 31)
            channels = choose_channels(year, month, template["campaign_type"], template["target_segment"])
            product_catalogue = infer_product_focus_from_catalogue(year, month)
            product_focus = template["product_focus"] if template["product_focus"] in product_catalogue else random.choice(product_catalogue)

            intensity = 1.0
            if signal.yoy_tx_growth_pct is not None and signal.yoy_tx_growth_pct < 0:
                intensity += 0.28
            elif signal.yoy_tx_growth_pct is not None and signal.yoy_tx_growth_pct > 15:
                intensity -= 0.12
            if signal.fail_pct > 7 and template["campaign_type"] in ("retention", "reactivation"):
                intensity += 0.15

            target_customers = int(random.randint(850, 8500) * intensity)
            budget = int(random.randint(45000, 650000) * max(0.75, intensity))

            if article:
                if news_theme(article) == "rate_hike":
                    template["offer_summary"] += f" after market concern around {article.get('title', 'rate changes')}"
                elif news_theme(article) == "load_shedding":
                    template["offer_summary"] += " during current power disruption pressure"

            campaigns.append(
                {
                    "campaign_id": f"CAMP-{year}-{annual_seen[year]:03d}",
                    "campaign_name": template["campaign_name"],
                    "campaign_type": template["campaign_type"],
                    "target_segment": template["target_segment"],
                    "channel": channels,
                    "product_focus": product_focus,
                    "offer_summary": template["offer_summary"],
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "budget_zar": str(round(budget / 1000) * 1000),
                    "target_customers_count": str(max(100, target_customers)),
                    "region": random.choices(PROVINCES, weights=[42, 12, 10, 9, 5, 5, 4, 3, 3, 2, 2, 2, 1, 1])[0],
                    "status": campaign_status(start, end),
                    "success_metric": template["success_metric"],
                }
            )

    return campaigns


def active_campaigns_for_month(campaigns: list[dict[str, Any]], year: int, month: int) -> list[dict[str, Any]]:
    if (year, month) < FIRST_MARKETING_MONTH:
        return []
    month_start = date(year, month, 1)
    month_end = date(year, month, monthrange(year, month)[1])
    active = []
    for campaign in campaigns:
        start = date.fromisoformat(campaign["start_date"])
        end = date.fromisoformat(campaign["end_date"])
        if start <= month_end and end >= month_start and campaign["status"] != "cancelled":
            active.append(campaign)
    return active


def campaign_starts_for_month(campaigns: list[dict[str, Any]], year: int, month: int) -> list[dict[str, Any]]:
    if (year, month) < FIRST_MARKETING_MONTH:
        return []
    out = []
    for campaign in campaigns:
        start = date.fromisoformat(campaign["start_date"])
        if start.year == year and start.month == month:
            out.append(campaign)
    return out


def response_note(response_type: str, campaign: dict[str, Any], channel: str) -> str:
    if response_type == "converted":
        return f"Customer accepted offer for {campaign['product_focus']} through {channel}."
    if response_type == "clicked":
        return "Customer clicked through to campaign landing page or product information."
    if response_type == "opened":
        return "Customer opened or viewed the campaign message."
    if response_type == "opted_out":
        return "Customer opted out of marketing messages for this channel."
    if response_type == "complained":
        return "Customer complained about frequency, relevance or timing of marketing contact."
    return "No direct action recorded after campaign exposure."


def response_date_for_campaign(campaign: dict[str, Any], year: int, month: int) -> str:
    month_start = date(year, month, 1)
    month_end = date(year, month, monthrange(year, month)[1])
    start = max(date.fromisoformat(campaign["start_date"]), month_start)
    end = min(date.fromisoformat(campaign["end_date"]), month_end)
    if end < start:
        end = start
    return (start + timedelta(days=random.randint(0, (end - start).days))).isoformat()


def generate_month_responses(
    campaigns: list[dict[str, Any]],
    year: int,
    month: int,
) -> list[dict[str, Any]]:
    if (year, month) < FIRST_MARKETING_MONTH:
        return []
    customers = read_customer_month(year, month)
    if not customers:
        return []

    rows: list[dict[str, Any]] = []
    active = active_campaigns_for_month(campaigns, year, month)
    response_idx = 1

    for campaign in active:
        channels = campaign["channel"].split(",")
        segment = campaign["target_segment"]
        pool = customer_pool_for_segment(customers, segment, year)
        target_count = min(len(pool), max(20, int(int(campaign["target_customers_count"]) / 12)))
        direct_rate = max(response_rates(ch)["opened"] + response_rates(ch)["clicked"] + response_rates(ch)["converted"] for ch in channels)
        if any(ch in ("radio", "tv", "outdoor", "branch_poster") for ch in channels):
            direct_rate *= 0.45
        response_count = max(3, min(len(pool), round(target_count * direct_rate)))
        sampled = random.sample(pool, k=min(response_count, len(pool)))

        for customer in sampled:
            channel = random.choice(channels)
            response_type = response_type_for(channel, segment, campaign["product_focus"])
            rows.append(
                {
                    "response_id": f"RESP-{year}{month:02d}-{response_idx:07d}",
                    "campaign_id": campaign["campaign_id"],
                    "customer_id": customer["customer_id"],
                    "account_id": customer.get("account_id", ""),
                    "response_date": response_date_for_campaign(campaign, year, month),
                    "response_type": response_type,
                    "conversion_value_zar": str(conversion_value(campaign["product_focus"])) if response_type == "converted" else "",
                    "channel_used": channel,
                    "notes": response_note(response_type, campaign, channel),
                }
            )
            response_idx += 1

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic marketing campaigns and responses.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    random.seed(2019 * 100)
    signals = load_monthly_signals()
    campaigns = generate_campaigns(args.start_year, args.end_year, signals)

    response_total = 0
    month_count = 0
    campaigns_written = 0
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            if (year, month) < FIRST_MARKETING_MONTH:
                continue
            out_dir = BANKING_DIR / f"{year}" / f"{month:02d}" / MARKETING_SUBDIR
            month_campaigns = active_campaigns_for_month(campaigns, year, month)
            rows = generate_month_responses(campaigns, year, month)
            write_csv(out_dir / "campaigns.csv", month_campaigns, CAMPAIGN_FIELDS)
            write_csv(out_dir / "campaign_responses.csv", rows, RESPONSE_FIELDS)
            response_total += len(rows)
            month_count += 1
            campaigns_written += len(month_campaigns)

    print(f"Marketing campaigns written under {BANKING_DIR}\\<year>\\<month>\\{MARKETING_SUBDIR}")
    print(f"  Unique campaigns generated: {len(campaigns)}")
    print(f"  Monthly campaign rows: {campaigns_written}")
    print(f"  Response months: {month_count}")
    print(f"  Responses: {response_total}")


if __name__ == "__main__":
    main()
