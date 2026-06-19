from __future__ import annotations

import random
import string
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from commons.data_loader import get_branches_data


ROOT = Path(__file__).resolve().parent
BANKING = ROOT / "banking_data"
ARTIFACTS = ROOT / "migration_artifacts" / "collection_account_integrity"
CREATED_CUSTOMERS = ARTIFACTS / "created_missing_customers.csv"
TARGET_CUSTOMERS = BANKING / "2019" / "12" / "customers_2019_12.parquet"

FIRST_NAMES_M = [
    "Thabo",
    "Sibusiso",
    "Kabelo",
    "Mandla",
    "Sipho",
    "Bongani",
    "Themba",
    "Lungile",
    "Tshepo",
    "Mpho",
    "Wandile",
    "Lukhanyo",
]
FIRST_NAMES_F = [
    "Nomsa",
    "Lerato",
    "Ayanda",
    "Busisiwe",
    "Zanele",
    "Naledi",
    "Thandiwe",
    "Nokuthula",
    "Dineo",
    "Palesa",
    "Refilwe",
    "Nandi",
]
SURNAMES = [
    "Mokoena",
    "Dlamini",
    "Ndlovu",
    "Khumalo",
    "Naidoo",
    "Mthembu",
    "Mabena",
    "Sibanda",
    "Nkosi",
    "Mahlangu",
    "Botha",
    "Pillay",
    "Maseko",
    "Molefe",
]
OCCUPATIONS = [
    ("Retail Assistant", "Employment Income", 86000, 210000),
    ("Teacher", "Employment Income", 180000, 420000),
    ("Driver", "Employment Income", 90000, 240000),
    ("Nurse", "Employment Income", 180000, 390000),
    ("Security Officer", "Employment Income", 72000, 180000),
    ("Admin Clerk", "Employment Income", 96000, 260000),
    ("Artisan", "Employment Income", 160000, 360000),
    ("Small Business Owner", "Business Income", 140000, 520000),
    ("Call Centre Agent", "Employment Income", 84000, 220000),
    ("Farm Worker", "Employment Income", 65000, 150000),
]
EMPLOYERS = [
    "Shoprite Checkers",
    "Pick n Pay",
    "Woolworths",
    "Spar Group",
    "Clicks Group",
    "Netcare",
    "Life Healthcare",
    "City of Johannesburg",
    "eThekwini Municipality",
    "Transnet",
    "MTN South Africa",
    "Vodacom South Africa",
    "Mokoena Logistics",
    "Dlamini Construction",
    "Naidoo Accounting Services",
]
COMPANIES = [
    "Masibambane Trading",
    "Ubuntu Foods",
    "Siyakhula Transport",
    "Khanyisa Projects",
    "Mthembu Hardware",
    "Ndlovu Cleaning Services",
]


def clean_id(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def schema_names(path: Path) -> list[str]:
    return pq.read_schema(path).names


def format_date_yymmdd(dt: date) -> str:
    return f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}"


def sa_id_number(birth_date: date, gender: str, citizen: bool = True) -> str:
    sequence_start = 5000 if gender == "M" else 0
    sequence = random.randint(sequence_start, sequence_start + 4999)
    citizenship_digit = "0" if citizen else "1"
    race_digit = "8"
    base = f"{format_date_yymmdd(birth_date)}{sequence:04d}{citizenship_digit}{race_digit}"
    # Lightweight valid-looking checksum digit; the synthetic project does not
    # require Luhn-perfect IDs, but it must not be blank.
    checksum = str(sum(int(d) for d in base) % 10)
    return base + checksum


def phone_number() -> str:
    prefix = random.choice(["060", "061", "071", "072", "073", "076", "078", "079", "081", "082", "083"])
    return "+27 " + prefix[1:] + " " + "".join(random.choice(string.digits) for _ in range(7))


def email_for(name: str, seq: int) -> str:
    base = "".join(ch for ch in name.lower().replace(" ", ".") if ch.isalpha() or ch == ".")
    return f"{base}{seq}@example.com"


def load_min_account_open_dates(customer_ids: set[str]) -> dict[str, date]:
    records = []
    for path in BANKING.glob("20*/[0-1][0-9]/accounts_*.parquet"):
        try:
            cols = set(schema_names(path))
        except Exception:
            continue
        if not {"customer_id", "opening_date"}.issubset(cols):
            continue
        frame = pd.read_parquet(path, columns=["customer_id", "opening_date"])
        frame["customer_id"] = frame["customer_id"].map(clean_id)
        frame = frame[frame["customer_id"].isin(customer_ids)].copy()
        if not frame.empty:
            records.append(frame)
    if not records:
        return {}
    all_rows = pd.concat(records, ignore_index=True)
    all_rows["opening_date"] = pd.to_datetime(all_rows["opening_date"], errors="coerce")
    out = {}
    for cid, group in all_rows.dropna(subset=["opening_date"]).groupby("customer_id"):
        out[cid] = group["opening_date"].min().date()
    return out


def branch_choice(branches: list[dict]) -> dict:
    return random.choice(branches)


def entry_date_for(customer_id: str, open_dates: dict[str, date]) -> date:
    opened = open_dates.get(customer_id, date(2019, 12, 15))
    start = date(2019, 1, 1)
    days_before = random.randint(0, 60)
    return max(start, opened - timedelta(days=days_before))


def residential_address(branch: dict, seq: int) -> str:
    street = random.choice(["Main Road", "Church Street", "Market Street", "Nelson Mandela Drive", "Kerk Street", "Jan Smuts Avenue"])
    return f"{random.randint(10, 899)} {street}, {branch.get('city')}, {branch.get('province')}, South Africa"


def recovered_individual(customer_id: str, seq: int, branches: list[dict], open_dates: dict[str, date]) -> dict:
    gender = random.choice(["M", "F"])
    first = random.choice(FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F)
    surname = random.choice(SURNAMES)
    full_name = f"{first} {surname}"
    branch = branch_choice(branches)
    birth_year = random.randint(1958, 2000)
    birth_date = date(birth_year, random.randint(1, 12), random.randint(1, 28))
    occupation, source, low_income, high_income = random.choice(OCCUPATIONS)
    income = float(random.randint(low_income, high_income))
    entry = entry_date_for(customer_id, open_dates)
    contact = random.choices(["SMS", "EMAIL", "PHONE"], weights=[0.45, 0.30, 0.25], k=1)[0]
    channel = random.choices(["Branch", "Mobile", "Online"], weights=[0.62, 0.25, 0.13], k=1)[0]
    is_pep = random.random() < 0.015
    return {
        "customer_id": customer_id,
        "customer_type": "Individual",
        "full_name": full_name,
        "birth_date": birth_date,
        "citizenship": "ZA",
        "nationality": "South Africa",
        "residential_address": residential_address(branch, seq),
        "residential_postal_code": str(random.randint(1000, 9999)),
        "commercial_address": None,
        "email": email_for(full_name, seq) if contact == "EMAIL" or random.random() < 0.65 else None,
        "phone_number": phone_number(),
        "id_type": "National ID",
        "id_number": sa_id_number(birth_date, gender),
        "expiry_date": None,
        "visa_type": None,
        "visa_expiry_date": None,
        "passport_expired": False,
        "is_pep": is_pep,
        "sanctioned_country": False,
        "risk_score": round(random.uniform(0.12, 0.72) + (0.12 if is_pep else 0), 3),
        "tax_id_number": str(random.randint(10000000, 99999999)) if income > 95000 else None,
        "occupation": occupation,
        "employer_name": random.choice(EMPLOYERS) if source == "Employment Income" else f"{surname} Trading",
        "source_of_funds": source,
        "marital_status": random.choice(["Single", "Married", "Divorced", "Widowed"]),
        "gender": gender,
        "preferred_contact_method": contact,
        "next_of_kin": f"{random.choice(FIRST_NAMES_F + FIRST_NAMES_M)} {random.choice(SURNAMES)}",
        "date_of_entry": entry,
        "annual_income": income,
        "education_level": random.choice(["High School Completed", "Diploma", "Bachelor's Degree", "Some Secondary"]),
        "location_exposure": random.choice(["Branch Assisted", "Mobile First", "Cash Heavy", "Urban Salaried"]),
        "financial_goal": random.choice(["salary_account", "family_support", "home_ownership", "savings", "vehicle_finance"]),
        "device_type": random.choice(["Android", "Feature Phone", "iPhone"]),
        "is_government_official": False,
        "government_role": None,
        "ethnicity": random.choice(["Black", "Coloured", "Indian", "White"]),
        "branch_id": branch.get("branch_id"),
        "branch_name": branch.get("name"),
        "branch_city": branch.get("city"),
        "branch_province": branch.get("province"),
        "capture_channel": channel,
        "is_affidavit": False,
        "company_age": None,
        "company_size": None,
        "number_of_employees": None,
        "annual_turnover": None,
        "directors_count": None,
        "shareholders_count": None,
        "beneficial_owners_count": None,
        "bee_level": None,
        "vat_registered": None,
        "industry_risk_rating": None,
        "guardian_customer_id": None,
    }


def recovered_company(customer_id: str, seq: int, branches: list[dict], open_dates: dict[str, date]) -> dict:
    branch = branch_choice(branches)
    name = f"{random.choice(COMPANIES)} (Pty) Ltd"
    entry = entry_date_for(customer_id, open_dates)
    employees = random.randint(3, 85)
    turnover = float(random.randint(450_000, 18_000_000))
    risk = random.choice(["low", "medium", "medium", "high"])
    return {
        "customer_id": customer_id,
        "customer_type": "Company",
        "full_name": name,
        "birth_date": None,
        "citizenship": "ZA",
        "nationality": "South Africa",
        "residential_address": None,
        "residential_postal_code": str(random.randint(1000, 9999)),
        "commercial_address": residential_address(branch, seq),
        "email": email_for(name.replace('(Pty) Ltd', '').strip(), seq),
        "phone_number": phone_number(),
        "id_type": "Company Registration",
        "id_number": f"2019/{random.randint(100000,999999)}/07",
        "expiry_date": None,
        "visa_type": None,
        "visa_expiry_date": None,
        "passport_expired": False,
        "is_pep": False,
        "sanctioned_country": False,
        "risk_score": round(random.uniform(0.22, 0.78), 3),
        "tax_id_number": str(random.randint(4000000000, 4999999999)),
        "occupation": None,
        "employer_name": None,
        "source_of_funds": "Business Revenue",
        "marital_status": None,
        "gender": None,
        "preferred_contact_method": random.choice(["EMAIL", "PHONE"]),
        "next_of_kin": None,
        "date_of_entry": entry,
        "annual_income": None,
        "education_level": None,
        "location_exposure": random.choice(["Branch Assisted", "Urban SME", "Cash Heavy"]),
        "financial_goal": random.choice(["working_capital", "business_growth", "asset_finance"]),
        "device_type": random.choice(["Android", "iPhone", "Desktop"]),
        "is_government_official": False,
        "government_role": None,
        "ethnicity": None,
        "branch_id": branch.get("branch_id"),
        "branch_name": branch.get("name"),
        "branch_city": branch.get("city"),
        "branch_province": branch.get("province"),
        "capture_channel": random.choices(["Branch", "Online"], weights=[0.75, 0.25], k=1)[0],
        "is_affidavit": False,
        "company_age": float(max(1, 2019 - random.randint(2003, 2018))),
        "company_size": "micro" if employees < 10 else "small" if employees < 50 else "medium",
        "number_of_employees": float(employees),
        "annual_turnover": turnover,
        "directors_count": float(random.randint(1, 4)),
        "shareholders_count": float(random.randint(1, 6)),
        "beneficial_owners_count": float(random.randint(1, 4)),
        "bee_level": float(random.randint(1, 8)),
        "vat_registered": turnover >= 1_000_000,
        "industry_risk_rating": risk,
        "guardian_customer_id": None,
    }


def main() -> None:
    random.seed(20260619)
    ids = pd.read_csv(CREATED_CUSTOMERS, dtype=str)["customer_id"].map(clean_id).tolist()
    id_set = set(ids)
    branches = get_branches_data()
    open_dates = load_min_account_open_dates(id_set)
    customers = pd.read_parquet(TARGET_CUSTOMERS)

    replacements = []
    for seq, customer_id in enumerate(ids, start=1):
        if customer_id.startswith("COM"):
            replacements.append(recovered_company(customer_id, seq, branches, open_dates))
        else:
            replacements.append(recovered_individual(customer_id, seq, branches, open_dates))
    replacement_df = pd.DataFrame(replacements)

    for col in customers.columns:
        if col not in replacement_df.columns:
            replacement_df[col] = None
    replacement_df = replacement_df[customers.columns]

    remaining = customers[~customers["customer_id"].astype(str).isin(id_set)].copy()
    fixed = pd.concat([remaining, replacement_df], ignore_index=True)
    fixed = fixed.drop_duplicates("customer_id", keep="last")
    fixed.to_parquet(TARGET_CUSTOMERS, index=False)

    profile = replacement_df.copy()
    profile["null_field_count"] = profile.isna().sum(axis=1)
    profile[["customer_id", "customer_type", "full_name", "date_of_entry", "annual_income", "annual_turnover", "branch_city", "capture_channel", "null_field_count"]].to_csv(
        ARTIFACTS / "enriched_recovered_customers.csv",
        index=False,
    )
    print(f"Updated {len(replacement_df)} recovered customer records in {TARGET_CUSTOMERS}")


if __name__ == "__main__":
    main()
