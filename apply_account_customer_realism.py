"""Apply linked realism corrections to existing customer and account datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"
HISTORY_END = date(2025, 12, 31)

COUNTRY_CODES = {
    "South Africa": "ZA",
    "Zimbabwe": "ZW",
    "Mozambique": "MZ",
    "Lesotho": "LS",
    "Botswana": "BW",
    "Namibia": "NA",
    "Kenya": "KE",
    "Malawi": "MW",
    "Zambia": "ZM",
    "Eswatini": "SZ",
    "Nigeria": "NG",
    "Democratic Republic of the Congo": "CD",
    "United Kingdom": "GB",
    "India": "IN",
    "China": "CN",
    "Saudi Arabia": "SA",
}

ADDITIONAL_COUNTRIES = [
    ("Malawi", 0.24),
    ("Zambia", 0.18),
    ("Eswatini", 0.12),
    ("Nigeria", 0.16),
    ("Democratic Republic of the Congo", 0.08),
    ("United Kingdom", 0.08),
    ("India", 0.06),
    ("China", 0.04),
    ("Saudi Arabia", 0.04),
]

COUNTRY_NAMES = {
    "Malawi": (["Chikondi", "Thoko", "Blessings", "Mphatso"], ["Banda", "Phiri", "Mbewe", "Kamanga"]),
    "Zambia": (["Chanda", "Mutinta", "Mwansa", "Natasha"], ["Bwalya", "Mulenga", "Tembo", "Zulu"]),
    "Eswatini": (["Sibusiso", "Nokuthula", "Thandeka", "Mandla"], ["Dlamini", "Mamba", "Simelane", "Shongwe"]),
    "Nigeria": (["Chinedu", "Amina", "Tunde", "Ngozi"], ["Okafor", "Adeyemi", "Bello", "Eze"]),
    "Democratic Republic of the Congo": (["Patrick", "Chantal", "Jean", "Mireille"], ["Kabongo", "Ilunga", "Mbuyi", "Kasongo"]),
    "United Kingdom": (["James", "Olivia", "Daniel", "Sophie"], ["Taylor", "Brown", "Wilson", "Davies"]),
    "India": (["Arjun", "Priya", "Rahul", "Ananya"], ["Patel", "Naidoo", "Singh", "Shah"]),
    "China": (["Wei", "Li", "Mei", "Jun"], ["Wang", "Chen", "Zhang", "Liu"]),
    "Saudi Arabia": (["Faisal", "Omar", "Noura", "Maha"], ["Al-Harbi", "Al-Qahtani", "Al-Saud", "Al-Dosari"]),
}

EMPLOYERS = [
    "Shoprite Checkers",
    "Pick n Pay",
    "Woolworths",
    "Spar Group",
    "Clicks Group",
    "Dis-Chem",
    "Transnet",
    "Eskom",
    "PRASA",
    "South African Airways",
    "Telkom South Africa",
    "MTN South Africa",
    "Vodacom South Africa",
    "City of Johannesburg",
    "City of Cape Town",
    "eThekwini Municipality",
    "Gauteng Department of Health",
    "KwaZulu-Natal Department of Education",
    "Western Cape Government",
    "Netcare",
    "Life Healthcare",
    "Mediclinic Southern Africa",
    "University of Johannesburg",
    "University of Pretoria",
    "University of KwaZulu-Natal",
    "Mokoena Logistics",
    "Dlamini Construction",
    "Naidoo Accounting Services",
    "Mthembu Security Services",
    "Khumalo Transport",
    "Mahlangu Engineering",
    "Pillay Family Pharmacy",
    "Maseko Catering",
    "Ndlovu Agri Supplies",
    "Botha Auto Repairs",
    "Nkosi Electrical",
    "Jacobs Freight Solutions",
    "Mabena Cleaning Services",
    "Sithole Primary School",
    "Madiba Community Clinic",
]

ORGANISATION_NAMES = [
    "Ikhaya Children's Home",
    "Siyakhula Orphanage",
    "Ubuntu Youth Development Centre",
    "Masibambane Community Trust",
    "Hope Haven Child and Youth Care Centre",
    "Thuthukani Disability Support Association",
    "Sisonke Women's Cooperative",
    "Imbokodo Community Foundation",
    "Khanyisa Early Childhood Centre",
    "Bambanani Food Relief Network",
    "New Life Community Church",
    "St Mark's Parish Welfare Fund",
    "Sakhisizwe School Governing Body",
    "Vukani Sports Development Club",
    "Lethabo Burial Society",
    "Siyabonga Education Trust",
    "Mahlasedi Community Advice Office",
    "Philani Hospice Support Trust",
]

BENEFICIARY_NAMES = [
    "Nomsa Dlamini",
    "Thabo Mokoena",
    "Priya Naidoo",
    "Lerato Molefe",
    "Sibusiso Khumalo",
    "Zanele Maseko",
    "Ayesha Patel",
    "Johan van Wyk",
    "Noluthando Ndlovu",
    "Mandla Sithole",
    "Kagiso Mabena",
    "Fatima Khan",
]

DEFAULT_ACCOUNT_CAPABILITIES = {
    "online_banking",
    "debit_card",
    "credit_card",
}


def stable_float(key: str) -> float:
    value = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    return int.from_bytes(value, "big") / float(2**64)


def stable_rng(key: str) -> random.Random:
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def weighted_pick(rng: random.Random, choices: list[tuple[str, float]]) -> str:
    return rng.choices([value for value, _ in choices], weights=[weight for _, weight in choices], k=1)[0]


def valid_parquet(path: Path, required_column: str) -> bool:
    return required_column in pq.read_schema(path).names


def luhn_card(rng: random.Random, card_type: str) -> str:
    prefix = "4" if card_type.startswith("visa") else "5"
    partial = prefix + "".join(str(rng.randint(0, 9)) for _ in range(14))
    total = 0
    for index, digit in enumerate(reversed([int(value) for value in partial])):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return partial + str((10 - total % 10) % 10)


def comma_values(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def date_value(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def aligned_card_history(row: pd.Series) -> tuple[object, object, object, object]:
    numbers = comma_values(row.get("card_number"))
    types = comma_values(row.get("card_type"))
    issues = comma_values(row.get("card_issue_date"))
    expiries = comma_values(row.get("card_expiry_date"))
    if not numbers:
        return None, None, None, None

    card_type = types[0] if types else "visa_debit"
    issue = date_value(issues[0] if issues else row.get("opening_date")) or date_value(row.get("opening_date"))
    expiry = date_value(expiries[0] if expiries else None)
    if issue is None:
        issue = date(2019, 1, 1)
    if expiry is None:
        expiry = date(issue.year + 4, issue.month, 1)

    history = [(numbers[0], card_type, issue, expiry)]
    rng = stable_rng(f"cards:{row.get('account_id')}")
    while history[-1][3] <= HISTORY_END:
        previous = history[-1]
        next_issue = previous[3] - timedelta(days=21)
        next_type = previous[1] if rng.random() < 0.94 else (
            "mastercard_debit" if previous[1] == "visa_debit" else
            "visa_debit" if previous[1] == "mastercard_debit" else previous[1]
        )
        next_expiry = date(next_issue.year + 4, next_issue.month, 1)
        history.append((luhn_card(rng, next_type), next_type, next_issue, next_expiry))

    return (
        ",".join(item[0] for item in history),
        ",".join(item[1] for item in history),
        ",".join(item[2].isoformat() for item in history),
        ",".join(item[3].isoformat() for item in history),
    )


def optional_bundled_products(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    products = [
        item.strip()
        for item in str(value).replace(",", ";").split(";")
        if item.strip() and item.strip() not in DEFAULT_ACCOUNT_CAPABILITIES
    ]
    return ";".join(dict.fromkeys(products)) if products else None


def opening_channel(capture_channel: object, account_id: str, year: int) -> str:
    capture = str(capture_channel or "Branch").strip().lower()
    mapping = {
        "branch": [("branch", 0.91), ("agent", 0.05), ("phone", 0.03), ("online", 0.01)],
        "mobile": [("mobile_app", 0.86), ("online", 0.09), ("branch", 0.05)],
        "online": [("online", 0.86), ("mobile_app", 0.08), ("branch", 0.06)],
        "call center": [("phone", 0.72), ("branch", 0.20), ("online", 0.08)],
    }
    if year == 2019 and capture != "branch":
        mapping[capture] = [("branch", 0.45), *[(value, weight * 0.55) for value, weight in mapping.get(capture, [("branch", 1.0)])]]
    return weighted_pick(stable_rng(f"opening:{account_id}"), mapping.get(capture, mapping["branch"]))


def saved_beneficiaries(row: pd.Series, customer_type: str) -> object:
    age_years = max(0.0, (HISTORY_END - (date_value(row.get("opening_date")) or HISTORY_END)).days / 365.25)
    online = bool(row.get("online_banking_enabled"))
    if customer_type == "Organization":
        probability = 0.38
    elif customer_type == "Company":
        probability = 0.45
    else:
        probability = 0.08 + (0.11 if online else 0.0) + min(0.08, age_years * 0.012)
    if stable_float(f"beneficiary:{row.get('account_id')}") >= probability:
        return None
    rng = stable_rng(f"beneficiary-list:{row.get('account_id')}")
    count = rng.choices([1, 2, 3, 4], weights=[0.58, 0.27, 0.11, 0.04], k=1)[0]
    return ",".join(rng.sample(BENEFICIARY_NAMES, count))


def diversify_customer(row: pd.Series) -> pd.Series:
    customer_id = str(row.get("customer_id"))
    customer_type = str(row.get("customer_type"))
    if customer_type == "Individual":
        nationality = str(row.get("nationality") or "South Africa")
        if nationality != "South Africa" and stable_float(f"country:{customer_id}") < 0.22:
            rng = stable_rng(f"country-choice:{customer_id}")
            nationality = weighted_pick(rng, ADDITIONAL_COUNTRIES)
            first, last = COUNTRY_NAMES[nationality]
            row["full_name"] = f"{rng.choice(first)} {rng.choice(last)}"
        code = COUNTRY_CODES.get(nationality, "ZA")
        citizenships = [code]
        dual_probability = 0.09 if code != "ZA" else 0.006
        if stable_float(f"dual:{customer_id}") < dual_probability:
            second = "ZA" if code != "ZA" else stable_rng(f"dual-code:{customer_id}").choice(["ZW", "GB", "NG", "AU"])
            citizenships = sorted(set([code, second]), key=lambda value: (value != "ZA", value))
        row["nationality"] = nationality
        row["citizenship"] = ",".join(citizenships)

        occupation = str(row.get("occupation") or "")
        if occupation and "unemployed" not in occupation.lower() and occupation.lower() != "student":
            if stable_float(f"employed:{customer_id}") < 0.74:
                row["employer_name"] = EMPLOYERS[int(stable_float(f"employer:{customer_id}") * len(EMPLOYERS)) % len(EMPLOYERS)]
            else:
                row["employer_name"] = None
        else:
            row["employer_name"] = None

        year = int(str(customer_id)[3:5]) + 2000
        rng = stable_rng(f"capture:{customer_id}")
        branch_weight = max(0.36, 0.58 - (year - 2019) * 0.035)
        row["capture_channel"] = rng.choices(
            ["Branch", "Mobile", "Online", "Call Center"],
            weights=[branch_weight, 0.23 + (year - 2019) * 0.018, 0.15 + (year - 2019) * 0.012, 0.04],
            k=1,
        )[0]
    elif customer_type == "Company" and stable_float(f"organisation:{customer_id}") < 0.18:
        row["customer_type"] = "Organization"
        row["full_name"] = ORGANISATION_NAMES[int(stable_float(f"organisation-name:{customer_id}") * len(ORGANISATION_NAMES)) % len(ORGANISATION_NAMES)]
        row["id_type"] = "NPO/Trust Registration Number"
        row["occupation"] = "Non-profit and community services"
        row["source_of_funds"] = stable_rng(f"organisation-funds:{customer_id}").choice(
            ["Donations", "Grant Funding", "Membership Contributions", "Fundraising Proceeds"]
        )
        row["capture_channel"] = stable_rng(f"organisation-channel:{customer_id}").choices(
            ["Branch", "Online", "Call Center"], weights=[0.76, 0.18, 0.06], k=1
        )[0]
        if "annual_income" in row.index:
            row["annual_income"] = None
    else:
        row["citizenship"] = "ZA"
        row["capture_channel"] = stable_rng(f"company-channel:{customer_id}").choices(
            ["Branch", "Online", "Call Center"], weights=[0.62, 0.31, 0.07], k=1
        )[0]
        if "annual_income" in row.index:
            row["annual_income"] = None
    return row


def load_approved_loan_accounts() -> set[str]:
    approved: set[str] = set()
    paths = sorted(DATA_DIR.glob("*/*/loans_*.parquet"))
    for path in tqdm(paths, desc="Reading loan accounts", unit="file"):
        names = pq.read_schema(path).names
        if "account_id" not in names or "application_status" not in names:
            continue
        frame = pd.read_parquet(path, columns=["account_id", "application_status"])
        mask = frame["application_status"].astype(str).str.lower().isin(["approved", "active", "booked", "disbursed"])
        approved.update(frame.loc[mask, "account_id"].dropna().astype(str))
    return approved


def closure_for(row: pd.Series, approved_loan_accounts: set[str]) -> tuple[date | None, object]:
    account_id = str(row.get("account_id"))
    if account_id in approved_loan_accounts or str(row.get("account_status", "")).lower() != "active":
        return None, None
    opened = date_value(row.get("opening_date"))
    if opened is None or opened + timedelta(days=180) >= HISTORY_END:
        return None, None
    exposure_years = (HISTORY_END - (opened + timedelta(days=180))).days / 365.25
    formal_closure_probability = 1.0 - math.exp(-0.018 * exposure_years)
    if stable_float(f"closure:{account_id}") >= formal_closure_probability:
        return None, None
    rng = stable_rng(f"closure-date:{account_id}")
    span = (HISTORY_END - (opened + timedelta(days=180))).days
    closed = opened + timedelta(days=180 + rng.randint(0, max(0, span)))
    reason = None if rng.random() < 0.27 else rng.choices(
        ["customer_request", "prolonged_inactivity", "moved_to_another_bank", "deceased_estate", "compliance_exit"],
        weights=[0.47, 0.25, 0.16, 0.07, 0.05],
        k=1,
    )[0]
    return closed, reason


def update_status_events(value: object, closure_date: date, reason: object) -> str:
    try:
        events = json.loads(value) if value and not pd.isna(value) else []
    except (TypeError, json.JSONDecodeError):
        events = []
    events.append(
        {
            "event_type": "closed",
            "event_date": closure_date.isoformat(),
            "reason": reason,
        }
    )
    return json.dumps(events, ensure_ascii=False)


def active_card_for(account: pd.Series, event_date: date) -> tuple[str | None, str | None, str | None]:
    numbers = comma_values(account.get("card_number"))
    types = comma_values(account.get("card_type"))
    issues = [date_value(value) for value in comma_values(account.get("card_issue_date"))]
    expiries = comma_values(account.get("card_expiry_date"))
    selected = 0
    for index, issue in enumerate(issues):
        if issue is not None and issue <= event_date:
            selected = index
    if not numbers:
        return None, None, None
    selected = min(selected, len(numbers) - 1)
    return (
        numbers[selected],
        types[min(selected, len(types) - 1)] if types else None,
        expiries[min(selected, len(expiries) - 1)] if expiries else None,
    )


def card_mask(number: str | None) -> str | None:
    return None if not number else f"{number[:6]}******{number[-4:]}"


def card_hash(number: str | None) -> str | None:
    return None if not number else hashlib.sha256(number.encode("utf-8")).hexdigest()[:24]


def process_customers() -> dict[str, dict[str, object]]:
    customer_lookup: dict[str, dict[str, object]] = {}
    paths = sorted(DATA_DIR.glob("*/*/customers_*.parquet"))
    for path in tqdm(paths, desc="Updating customers", unit="file"):
        if not valid_parquet(path, "customer_id"):
            continue
        frame = pd.read_parquet(path)
        frame = frame.apply(diversify_customer, axis=1)
        frame.to_parquet(path, index=False)
        for row in frame[["customer_id", "customer_type", "capture_channel"]].to_dict("records"):
            customer_lookup[str(row["customer_id"])] = row
    return customer_lookup


def process_accounts(
    customer_lookup: dict[str, dict[str, object]],
    approved_loan_accounts: set[str],
) -> tuple[dict[str, date], dict[str, dict[str, object]]]:
    closures: dict[str, date] = {}
    account_lookup: dict[str, dict[str, object]] = {}
    paths = sorted(DATA_DIR.glob("*/*/accounts_*.parquet"))
    for path in tqdm(paths, desc="Updating accounts", unit="file"):
        if not valid_parquet(path, "account_id"):
            continue
        frame = pd.read_parquet(path)
        for column in ["bank_name", "account_purpose"]:
            if column in frame.columns:
                frame = frame.drop(columns=column)
        if "bundled_products" in frame.columns:
            frame["bundled_products"] = frame["bundled_products"].map(optional_bundled_products)
        for index, row in frame.iterrows():
            account_id = str(row["account_id"])
            customer = customer_lookup.get(str(row.get("customer_id")), {})
            opened = date_value(row.get("opening_date")) or date(int(path.parts[-3]), int(path.parts[-2]), 1)
            frame.at[index, "opening_channel"] = opening_channel(customer.get("capture_channel"), account_id, opened.year)
            frame.at[index, "beneficiaries"] = saved_beneficiaries(row, str(customer.get("customer_type", "Individual")))
            card_number, card_type, card_issue, card_expiry = aligned_card_history(row)
            frame.at[index, "card_number"] = card_number
            frame.at[index, "card_type"] = card_type
            frame.at[index, "card_issue_date"] = card_issue
            frame.at[index, "card_expiry_date"] = card_expiry

            existing_closure = date_value(row.get("closure_date"))
            if existing_closure is not None:
                closures[account_id] = existing_closure
            closed, reason = closure_for(row, approved_loan_accounts)
            if closed is not None:
                closures[account_id] = closed
                frame.at[index, "account_status"] = "closed"
                frame.at[index, "status_change_date"] = closed
                frame.at[index, "closure_date"] = closed
                frame.at[index, "status_reason"] = reason
                if "status_events_json" in frame.columns:
                    frame.at[index, "status_events_json"] = update_status_events(row.get("status_events_json"), closed, reason)
        frame.to_parquet(path, index=False)
        for row in frame.to_dict("records"):
            account_lookup[str(row["account_id"])] = row
    return closures, account_lookup


def filter_transactions(closures: dict[str, date]) -> int:
    removed = 0
    paths = sorted(DATA_DIR.glob("*/*/transactions.jsonl"))
    for path in tqdm(paths, desc="Filtering transaction files", unit="file"):
        temp = path.with_suffix(".realism.tmp")
        changed = False
        file_size = path.stat().st_size
        with (
            path.open("r", encoding="utf-8") as source,
            temp.open("w", encoding="utf-8") as target,
            tqdm(
                total=file_size,
                desc=f"  {path.parts[-3]}-{path.parts[-2]}",
                unit="B",
                unit_scale=True,
                leave=False,
            ) as byte_progress,
        ):
            for line in source:
                byte_progress.update(len(line.encode("utf-8")))
                if not line.strip():
                    continue
                row = json.loads(line)
                closed = closures.get(str(row.get("account_id")))
                timestamp = date_value(row.get("transaction_timestamp") or row.get("transaction_date"))
                if closed is not None and timestamp is not None and timestamp > closed:
                    removed += 1
                    changed = True
                    continue
                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        if changed:
            temp.replace(path)
        else:
            temp.unlink()
    return removed


def filter_atm_logs(closures: dict[str, date], account_lookup: dict[str, dict[str, object]]) -> int:
    removed = 0
    paths = sorted(DATA_DIR.glob("*/*/atm_logs_*.parquet"))
    for path in tqdm(paths, desc="Filtering ATM logs", unit="file"):
        frame = pd.read_parquet(path)
        event_dates = pd.to_datetime(frame["event_timestamp"], errors="coerce").dt.date
        keep = [
            closures.get(str(account_id)) is None or pd.isna(event_date) or event_date <= closures[str(account_id)]
            for account_id, event_date in zip(frame["account_id"], event_dates)
        ]
        removed += int((~pd.Series(keep)).sum())
        frame = frame.loc[keep].copy()
        for index, row in frame.iterrows():
            account = account_lookup.get(str(row.get("account_id")))
            event_date = date_value(row.get("event_timestamp"))
            if account is None or event_date is None:
                continue
            number, card_type, expiry = active_card_for(pd.Series(account), event_date)
            frame.at[index, "masked_card_number"] = card_mask(number)
            frame.at[index, "card_number_hash"] = card_hash(number)
            frame.at[index, "card_type"] = card_type
            frame.at[index, "card_expiry_date"] = expiry
        frame.to_parquet(path, index=False)
    return removed


def filter_debit_orders(closures: dict[str, date]) -> int:
    removed = 0
    paths = sorted(DATA_DIR.glob("*/*/debit_orders_*.parquet"))
    closure_timestamps = {
        account_id: pd.Timestamp(closed)
        for account_id, closed in closures.items()
    }
    for path in tqdm(paths, desc="Filtering debit orders", unit="file"):
        frame = pd.read_parquet(path)
        closed = frame["account_id"].astype(str).map(closure_timestamps)
        started = pd.to_datetime(frame["start_date"], errors="coerce")
        remove_mask = closed.notna() & started.notna() & (started > closed)
        removed += int(remove_mask.sum())

        frame = frame.loc[~remove_mask].copy()
        closed = closed.loc[~remove_mask]
        cancel_mask = closed.notna()

        for column in ("end_date", "cancellation_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            frame.loc[cancel_mask, column] = closed.loc[cancel_mask]
        frame.loc[cancel_mask, "cancellation_reason"] = "account_closed"
        frame.loc[cancel_mask, "status"] = "Cancelled"
        frame.to_parquet(path, index=False)
    return removed


def filter_csv_interactions(closures: dict[str, date]) -> int:
    removed = 0
    candidates = [
        ("customer_communications/communications.csv", "timestamp"),
        ("customer_communications/complaints.csv", "timestamp"),
        ("customer_communications/suggestions.csv", "timestamp"),
        ("marketing_campaigns/campaign_responses.csv", "response_date"),
    ]
    paths = [
        (path, timestamp_column)
        for relative, timestamp_column in candidates
        for path in sorted(DATA_DIR.glob(f"*/*/{relative}"))
    ]
    for path, timestamp_column in tqdm(paths, desc="Filtering interactions", unit="file"):
        frame = pd.read_csv(path)
        if frame.empty or "account_id" not in frame.columns or timestamp_column not in frame.columns:
            continue
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", dayfirst=True).dt.date
        keep = [
            closures.get(str(account_id)) is None or pd.isna(timestamp) or timestamp <= closures[str(account_id)]
            for account_id, timestamp in zip(frame["account_id"], timestamps)
        ]
        removed += int((~pd.Series(keep)).sum())
        frame.loc[keep].to_csv(path, index=False)
    return removed


def load_customer_lookup() -> dict[str, dict[str, object]]:
    customers: dict[str, dict[str, object]] = {}
    customer_paths = sorted(DATA_DIR.glob("*/*/customers_*.parquet"))
    for path in tqdm(customer_paths, desc="Loading customer lookup", unit="file", leave=False):
        if not valid_parquet(path, "customer_id"):
            continue
        frame = pd.read_parquet(path, columns=["customer_id", "customer_type", "capture_channel"])
        for row in frame.to_dict("records"):
            customers[str(row["customer_id"])] = row
    return customers


def load_account_lookup() -> tuple[dict[str, date], dict[str, dict[str, object]]]:
    accounts: dict[str, dict[str, object]] = {}
    closures: dict[str, date] = {}
    account_paths = sorted(DATA_DIR.glob("*/*/accounts_*.parquet"))
    for path in tqdm(account_paths, desc="Loading account lookup", unit="file", leave=False):
        if not valid_parquet(path, "account_id"):
            continue
        frame = pd.read_parquet(path)
        for row in frame.to_dict("records"):
            account_id = str(row["account_id"])
            accounts[account_id] = row
            closed = date_value(row.get("closure_date"))
            if closed is not None:
                closures[account_id] = closed
    return closures, accounts


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply linked realism corrections in separate stages.")
    parser.add_argument(
        "--stage",
        choices=["all", "customers", "accounts", "transactions", "atm", "debit-orders", "interactions"],
        default="all",
    )
    args = parser.parse_args()

    print(f"\nStarting stage: {args.stage}")
    customer_lookup: dict[str, dict[str, object]] = {}
    closures: dict[str, date] = {}
    account_lookup: dict[str, dict[str, object]] = {}

    if args.stage == "all":
        customer_lookup = process_customers()
        print(f"Customers mapped: {len(customer_lookup):,}")
        approved_loan_accounts = load_approved_loan_accounts()
        closures, account_lookup = process_accounts(customer_lookup, approved_loan_accounts)
        print(f"Accounts mapped: {len(account_lookup):,}")
        print(f"Accounts closed: {len(closures):,}")
        print(f"Transactions removed after closure: {filter_transactions(closures):,}")
        print(f"ATM events removed after closure: {filter_atm_logs(closures, account_lookup):,}")
        print(f"Debit orders removed after closure: {filter_debit_orders(closures):,}")
        print(f"Communication/campaign rows removed after closure: {filter_csv_interactions(closures):,}")
    elif args.stage == "customers":
        customer_lookup = process_customers()
        print(f"Customers mapped: {len(customer_lookup):,}")
    elif args.stage == "accounts":
        customer_lookup = load_customer_lookup()
        approved_loan_accounts = load_approved_loan_accounts()
        closures, account_lookup = process_accounts(customer_lookup, approved_loan_accounts)
        print(f"Accounts mapped: {len(account_lookup):,}")
        print(f"Accounts closed: {len(closures):,}")
    elif args.stage == "transactions":
        closures, _ = load_account_lookup()
        print(f"Transactions removed after closure: {filter_transactions(closures):,}")
    elif args.stage == "atm":
        closures, account_lookup = load_account_lookup()
        print(f"ATM events removed after closure: {filter_atm_logs(closures, account_lookup):,}")
    elif args.stage == "debit-orders":
        closures, _ = load_account_lookup()
        print(f"Debit orders removed after closure: {filter_debit_orders(closures):,}")
    elif args.stage == "interactions":
        closures, _ = load_account_lookup()
        print(f"Communication/campaign rows removed after closure: {filter_csv_interactions(closures):,}")
    print(f"Completed stage: {args.stage}\n")


if __name__ == "__main__":
    main()
