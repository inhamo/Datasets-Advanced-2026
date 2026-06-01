from __future__ import annotations

import argparse
import json
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from commons.data_loader import get_branch_codes_by_city_data, get_retail_bank_products_data


# Seasonal monthly opening patterns for retail banking in South Africa.
# Weights sum to 1.0 and intentionally peak around tax/planning and holiday periods.
MONTHLY_OPENING_WEIGHTS = {
    1: 0.09,
    2: 0.075,
    3: 0.08,
    4: 0.115,
    5: 0.09,
    6: 0.08,
    7: 0.07,
    8: 0.065,
    9: 0.07,
    10: 0.08,
    11: 0.085,
    12: 0.10,
}


def calculate_age(birth_date, target_year):
    if pd.isna(birth_date) or birth_date is None:
        return max(18, target_year - 1990)
    return target_year - birth_date.year


def get_income_level(customer_data):
    income = customer_data.get("annual_income", 300000)
    if income < 100000:
        return "low"
    if income < 600000:
        return "medium"
    return "high"


def generate_sa_account_number(branch_code, global_counter):
    branch_num = str(branch_code).zfill(6)
    account_type = random.choice(["01", "02", "03", "27", "28"])
    sequence = str(global_counter % 1000000).zfill(6)
    base = branch_num + account_type + sequence
    check_sum = sum(int(d) * (2 if i % 2 == 0 else 1) for i, d in enumerate(base))
    check_digit = (10 - (check_sum % 10)) % 10
    return base + str(check_digit)


def generate_swift_code(swift_base, branch_code):
    if random.random() < 0.3:
        return f"{swift_base}{str(branch_code)[-3:]}"
    return swift_base


def generate_iban(account_number):
    bank_code = "250655"
    check_digits = str(random.randint(10, 99))
    return f"ZA{check_digits}{bank_code}{account_number[:10].zfill(10)}"


def generate_card_number(account_type):
    if account_type in ["islamic"]:
        return None

    card_types = {
        "visa_debit": "4",
        "mastercard_debit": "5",
        "visa_credit": "4",
        "mastercard_credit": "5",
    }
    is_credit = account_type in ["premium", "gold", "platinum", "business"]
    card_type = random.choice(["visa_credit", "mastercard_credit"] if is_credit else ["visa_debit", "mastercard_debit"])
    card_num = card_types[card_type] + "".join(str(random.randint(0, 9)) for _ in range(14))

    digits = [int(d) for d in card_num]
    check_sum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        check_sum += d
    check_digit = (10 - (check_sum % 10)) % 10

    return card_num + str(check_digit), card_type


def determine_account_purpose(customer_data, account_type):
    if customer_data.get("customer_type") == "Company":
        purposes = ["business_operations", "payroll", "tax_payments", "investment", "trading"]
        weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        return random.choices(purposes, weights=weights)[0]

    occupation = str(customer_data.get("occupation", ""))
    if "Student" in occupation:
        return random.choice(["student_savings", "bursary_account", "pocket_money"])
    if "Unemployed" in occupation:
        return random.choice(["social_grants", "savings", "family_support"])

    if account_type in ["premium", "gold", "platinum"]:
        purposes = ["wealth_management", "investment", "salary", "savings"]
        weights = [0.3, 0.3, 0.25, 0.15]
    else:
        purposes = ["salary", "savings", "daily_transactions", "emergency_fund", "side_income"]
        weights = [0.4, 0.25, 0.2, 0.1, 0.05]

    return random.choices(purposes, weights=weights)[0]


def generate_beneficiaries(customer_data, fake):
    if random.random() < 0.4:
        return None

    num_beneficiaries = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    beneficiaries = []
    for _ in range(num_beneficiaries):
        name = fake.name()
        relationship = random.choice(["Spouse", "Child", "Parent", "Sibling", "Other"])
        percentage = random.randint(10, 100)
        beneficiaries.append(f"{name}|{relationship}|{percentage}%")
    return ";".join(beneficiaries)


def should_reject_application(customer_data, account_type):
    risk_score = customer_data.get("risk_score", 0.5)
    rejection_prob = 0.03
    if risk_score > 0.85:
        rejection_prob += 0.15
    elif risk_score > 0.7:
        rejection_prob += 0.08

    if account_type in ["premium", "gold", "platinum"]:
        income = customer_data.get("annual_income", 0)
        if income < 300000:
            rejection_prob += 0.2

    if customer_data.get("is_pep") or customer_data.get("sanctioned_country"):
        rejection_prob += 0.1

    if pd.isna(customer_data.get("tax_id_number")) and customer_data.get("annual_income", 0) > 500000:
        rejection_prob += 0.15

    if random.random() < rejection_prob:
        rejection_reasons = [
            "high_risk_profile",
            "insufficient_documentation",
            "failed_credit_check",
            "pep_sanctions_concern",
            "incomplete_kyc",
            "affordability_assessment_failed",
            "adverse_credit_history",
            "employment_verification_failed",
        ]
        return True, random.choice(rejection_reasons)

    return False, None


def build_bank_selector(products_payload):
    banks = products_payload.get("banks", [])
    names = [b["name"] for b in banks]
    weights = [b.get("weight", 1.0) for b in banks]
    swift_by_bank = {b["name"]: b.get("swift_base", "WOLZAJJ") for b in banks}
    return names, weights, swift_by_bank


def build_city_matchers(city_branch_codes):
    matchers = []
    for province, cities in city_branch_codes.items():
        for city, codes in cities.items():
            matchers.append((city.lower(), codes))
    return matchers


def get_branch_code(customer_data, city_branch_codes, city_matchers):
    residential_address = str(customer_data.get("residential_address", ""))
    if not residential_address:
        return random.choice(city_branch_codes["Gauteng"]["Johannesburg"])

    address_lower = residential_address.lower()
    for city_lower, codes in city_matchers:
        if city_lower in address_lower:
            return random.choice(codes)

    return random.choice(city_branch_codes["Gauteng"]["Johannesburg"])


class GlobalAccountCounter:
    def __init__(self, data_path, persist_every=500):
        self.data_path = data_path
        self.counter_file = os.path.join(data_path, "account_counter.json")
        self.persist_every = max(1, int(persist_every))
        self.counter = self._load_counter()
        self._last_persisted_counter = self.counter

    def _load_counter(self):
        if os.path.exists(self.counter_file):
            try:
                with open(self.counter_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data.get("counter", 0)
            except Exception:
                pass

        total_accounts = 0
        
        # Scan flat structure (banking_data/accounts_YEAR.{parquet|csv})
        for year in range(2015, 2031):
            for ext in ["parquet", "csv"]:
                file_path = os.path.join(self.data_path, f"accounts_{year}.{ext}")
                if os.path.exists(file_path):
                    try:
                        if ext == "parquet":
                            total_accounts += len(pd.read_parquet(file_path))
                        else:
                            total_accounts += len(pd.read_csv(file_path))
                    except Exception:
                        continue
        
        # Scan monthly structure (banking_data/YEAR/MM/accounts_YEAR_MM.{parquet|csv})
        for year in range(2015, 2031):
            year_dir = os.path.join(self.data_path, str(year))
            if os.path.isdir(year_dir):
                for month in range(1, 13):
                    month_dir = os.path.join(year_dir, f"{month:02d}")
                    if os.path.isdir(month_dir):
                        for ext in ["parquet", "csv"]:
                            file_path = os.path.join(month_dir, f"accounts_{year}_{month:02d}.{ext}")
                            if os.path.exists(file_path):
                                try:
                                    if ext == "parquet":
                                        total_accounts += len(pd.read_parquet(file_path))
                                    else:
                                        total_accounts += len(pd.read_csv(file_path))
                                except Exception:
                                    continue
        
        self._save_counter(total_accounts)
        return total_accounts

    def _save_counter(self, value):
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.counter_file, "w", encoding="utf-8") as handle:
            json.dump({"counter": value, "timestamp": str(date.today())}, handle)

    def get_next(self):
        current = self.counter
        self.counter += 1
        if (self.counter - self._last_persisted_counter) >= self.persist_every:
            self._save_counter(self.counter)
            self._last_persisted_counter = self.counter
        return current

    def peek(self):
        return self.counter

    def flush(self):
        if self.counter != self._last_persisted_counter:
            self._save_counter(self.counter)
            self._last_persisted_counter = self.counter


def random_date(start_date, end_date):
    delta = (end_date - start_date).days
    if delta <= 0:
        return start_date
    return start_date + timedelta(days=np.random.randint(0, delta + 1))


def resolve_application_window(base_start, base_end, year, month=None):
    start_date = base_start
    end_date = base_end
    if month is not None:
        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year, 12, 31)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)
        start_date = max(start_date, month_start)
        end_date = min(end_date, month_end)
    if start_date > end_date:
        return None, None
    return start_date, end_date


def choose_opening_date(base_start, base_end, year, month=None, monthly_weights=None):
    """
    Select opening date using seasonal monthly weights when month is not forced.
    Falls back to explicit month window when month is provided.
    """
    if month is not None:
        start_date, end_date = resolve_application_window(base_start, base_end, year, month)
        if start_date is None:
            return None
        return random_date(start_date, end_date)

    weights = monthly_weights or MONTHLY_OPENING_WEIGHTS
    valid_months = []
    valid_weights = []
    month_windows = {}

    for month_number in range(1, 13):
        start_date, end_date = resolve_application_window(base_start, base_end, year, month_number)
        if start_date is None:
            continue
        valid_months.append(month_number)
        valid_weights.append(weights.get(month_number, 1.0))
        month_windows[month_number] = (start_date, end_date)

    if not valid_months:
        return None

    total = sum(valid_weights)
    normalized = [w / total for w in valid_weights]
    selected_month = random.choices(valid_months, weights=normalized, k=1)[0]
    start_date, end_date = month_windows[selected_month]
    return random_date(start_date, end_date)


def get_output_path(base_path, year, month=None, record_type="accounts"):
    """
    Resolve output path for accounts and history files.
    If month is None, returns base_path/accounts_YEAR.parquet (flat structure).
    If month is specified, returns base_path/YEAR/MM/accounts_YEAR_MM.parquet (monthly structure).
    """
    if month is None:
        return os.path.join(base_path, f"{record_type}_{year}")
    year_month = f"{year}_{month:02d}"
    month_dir = os.path.join(base_path, str(year), f"{month:02d}")
    return os.path.join(month_dir, f"{record_type}_{year_month}")


def select_realistic_account_type(customer_data, customer_type):
    if customer_type == "Individual":
        income = customer_data.get("annual_income", 300000)
        if income < 100000:
            return random.choices(["easy", "savings"], weights=[0.7, 0.3])[0]
        if income < 300000:
            return random.choices(["savings", "current", "cheque"], weights=[0.5, 0.3, 0.2])[0]
        if income < 600000:
            return random.choices(["current", "cheque", "aspire", "gold"], weights=[0.3, 0.2, 0.3, 0.2])[0]
        if income < 1000000:
            return random.choices(["gold", "premium", "current"], weights=[0.4, 0.4, 0.2])[0]
        return random.choices(["platinum", "premium", "gold"], weights=[0.5, 0.3, 0.2])[0]
    return "business"


def determine_account_tier(account_type, income_level):
    if account_type in ["premium", "gold", "platinum"]:
        return "premium"
    if account_type in ["business"]:
        return "standard"
    if income_level == "low":
        return "basic"
    return "standard"


def generate_credit_limit(account_type, income_level, annual_income):
    overdraft_limit = 0.0
    credit_card_limit = 0.0

    if account_type in ["current", "cheque", "premium", "gold", "platinum", "business"] and random.random() < 0.4:
        if income_level == "high":
            overdraft_limit = round(random.uniform(10000, min(annual_income * 0.5, 100000)), 2)
        elif income_level == "medium":
            overdraft_limit = round(random.uniform(5000, min(annual_income * 0.3, 50000)), 2)
        else:
            overdraft_limit = round(random.uniform(1000, 10000), 2)

    if account_type in ["premium", "gold", "platinum"] and random.random() < 0.7:
        if income_level == "high":
            credit_card_limit = round(random.uniform(50000, 200000), 2)
        elif income_level == "medium":
            credit_card_limit = round(random.uniform(10000, 80000), 2)
        else:
            credit_card_limit = round(random.uniform(5000, 30000), 2)
    elif account_type == "business" and random.random() < 0.5:
        credit_card_limit = round(random.uniform(20000, 150000), 2)

    return overdraft_limit, credit_card_limit


def generate_account_requirements(customer_data, account_type):
    requirements = {
        "proof_of_income_provided": False,
        "proof_of_address_provided": True,
        "bank_statements_provided": False,
        "employer_letter_provided": False,
        "business_registration_provided": False,
        "tax_certificate_provided": False,
        "minimum_deposit_met": True,
    }

    if random.random() < 0.2:
        requirements["bank_statements_provided"] = True

    if account_type in ["premium", "gold", "platinum"]:
        requirements["proof_of_income_provided"] = random.random() < 0.9

    if account_type == "business":
        requirements["business_registration_provided"] = True
        requirements["tax_certificate_provided"] = random.random() < 0.8

    occupation = str(customer_data.get("occupation", ""))
    if occupation not in ["Unemployed", "Student", "Self-Employed"]:
        requirements["employer_letter_provided"] = random.random() < 0.6
        requirements["proof_of_income_provided"] = random.random() < 0.8

    return requirements


def determine_account_status(opening_date, customer_data, account_requirements, year):
    today = date.today()
    days_since_opening = max(0, (today - opening_date).days)
    status = "active"
    status_change_date = None
    status_reason = None
    closure_date = None

    if days_since_opening < 30 and not (account_requirements["proof_of_address_provided"] and account_requirements["minimum_deposit_met"]):
        if random.random() < 0.3:
            status = "pending_verification"
            status_reason = random.choice(["incomplete_documents", "address_verification_pending"])

    risk_score = customer_data.get("risk_score", 0.5)
    if risk_score > 0.8:
        if random.random() < 0.15:
            status = "frozen"
            status_change_date = opening_date + timedelta(days=random.randint(1, max(1, days_since_opening)))
            status_reason = "high_risk_suspicion"
        elif random.random() < 0.25:
            status = "restricted"
            status_change_date = opening_date + timedelta(days=random.randint(1, max(1, days_since_opening)))
            status_reason = "risk_monitoring"
    elif risk_score > 0.6 and random.random() < 0.15:
        status = "restricted"
        status_change_date = opening_date + timedelta(days=random.randint(1, max(1, days_since_opening)))
        status_reason = "moderate_risk"

    if opening_date.year < year and days_since_opening > 1095:
        closure_probability = 0.05 + (days_since_opening - 1095) / 10000
        if random.random() < closure_probability:
            status = "closed"
            status_change_date = opening_date + timedelta(days=random.randint(1, max(1, days_since_opening - 1)))
            closure_date = status_change_date
            status_reason = random.choice(["customer_request", "non_activity", "migration_to_other_bank"])

    if status == "active" and random.random() < 0.08 and opening_date.year < year:
        status = "dormant"
        status_change_date = opening_date + timedelta(days=random.randint(365, max(365, days_since_opening)))
        status_reason = "inactivity"

    if status == "active" and random.random() < 0.02:
        status = "suspended"
        status_change_date = opening_date + timedelta(days=random.randint(1, max(1, days_since_opening)))
        status_reason = random.choice(["fraud_suspicion", "overdue_charges"])

    return status, status_change_date, closure_date, status_reason


def generate_bundled_products(account_type, customer_data, bundled_products_available, target_year):
    available_products = bundled_products_available.get(account_type, [])
    if not available_products:
        return None

    num_products = random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
    if num_products == 0:
        return None

    if customer_data.get("customer_type") == "Individual":
        age = calculate_age(customer_data.get("birth_date", date(1990, 1, 1)), target_year)
        if age < 25 and customer_data.get("occupation") == "Student" and "student_card" in available_products:
            return "student_card"

    selected_products = random.sample(available_products, min(num_products, len(available_products)))
    return ";".join(selected_products) if selected_products else None


def determine_opening_channel_and_details():
    channels = ["branch", "online", "mobile_app", "phone", "agent"]
    weights = [0.50, 0.30, 0.15, 0.03, 0.02]
    return {"opening_channel": random.choices(channels, weights=weights)[0]}


def generate_approval_date(opening_date):
    if random.random() < 0.9:
        return opening_date
    return opening_date + timedelta(days=random.randint(1, 7))


def generate_accounts_with_relationships(customer_data, year):
    if customer_data["customer_type"] == "Individual":
        age = calculate_age(customer_data.get("birth_date", date(1990, 1, 1)), year)
        education = str(customer_data.get("education_level", "")).lower()
        income_level = get_income_level(customer_data)

        highly_educated = any(token in education for token in ["master", "phd", "doctor", "postgrad"])
        if income_level == "high" and highly_educated and age >= 30:
            return random.choices([1, 2, 3], weights=[0.45, 0.47, 0.08])[0]
        if income_level in ["medium", "high"] and age >= 25:
            return random.choices([1, 2], weights=[0.78, 0.22])[0]
        return 1
    return 1


def choose_bank_and_product(account_type, products_payload, bank_names, bank_weights):
    bank_name = random.choices(bank_names, weights=bank_weights, k=1)[0]
    options = products_payload.get("account_products_by_type", {}).get(account_type, ["Generic Account"])
    return bank_name, random.choice(options)


def load_customer_file(path_without_ext):
    parquet_path = f"{path_without_ext}.parquet"
    csv_path = f"{path_without_ext}.csv"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(parquet_path)


def resolve_customer_file_base(base_path, year, month=None):
    if month is not None:
        monthly_base = os.path.join(base_path, str(year), f"{month:02d}", f"customers_{year}_{month:02d}")
        if os.path.exists(f"{monthly_base}.parquet") or os.path.exists(f"{monthly_base}.csv"):
            return monthly_base

    flat_base = os.path.join(base_path, f"customers_{year}")
    if os.path.exists(f"{flat_base}.parquet") or os.path.exists(f"{flat_base}.csv"):
        return flat_base

    if month is not None:
        return os.path.join(base_path, str(year), f"{month:02d}", f"customers_{year}_{month:02d}")
    return flat_base


def load_existing_customer_account_counts(base_path, year, month):
    """
    For monthly runs, load previously generated account files in the same year
    to carry forward per-customer account counts.
    """
    counts = {}
    if month is None or month <= 1:
        return counts

    for prev_month in range(1, month):
        month_dir = os.path.join(base_path, str(year), f"{prev_month:02d}")
        parquet_path = os.path.join(month_dir, f"accounts_{year}_{prev_month:02d}.parquet")
        csv_path = os.path.join(month_dir, f"accounts_{year}_{prev_month:02d}.csv")

        try:
            if os.path.exists(parquet_path):
                df_prev = pd.read_parquet(parquet_path, columns=["customer_id"])
            elif os.path.exists(csv_path):
                df_prev = pd.read_csv(csv_path, usecols=["customer_id"])
            else:
                continue
        except Exception:
            continue

        if "customer_id" not in df_prev.columns or df_prev.empty:
            continue

        month_counts = df_prev["customer_id"].value_counts()
        for customer_id, value in month_counts.items():
            counts[customer_id] = counts.get(customer_id, 0) + int(value)

    return counts


def build_history_tables(df_accounts, target_year):
    limits_rows = []
    status_rows = []
    enroll_rows = []
    signatory_rows = []
    now_ts = pd.Timestamp.now().isoformat()

    column_positions = {name: idx for idx, name in enumerate(df_accounts.columns)}

    def value_from_row(row_values, key, default=None):
        idx = column_positions.get(key)
        if idx is None:
            return default
        value = row_values[idx]
        return default if pd.isna(value) else value

    for row_values in df_accounts.itertuples(index=False, name=None):
        account_id = value_from_row(row_values, "account_id")
        opening_date_value = value_from_row(row_values, "opening_date")
        opening_date = pd.to_datetime(opening_date_value).date()
        approval_date_value = value_from_row(row_values, "approval_date", opening_date)
        approval_date = pd.to_datetime(approval_date_value).date() if approval_date_value is not None else opening_date

        limit_events = [
            {
                "event_type": "opening_limits",
                "event_date": str(opening_date),
                "overdraft_limit": float(value_from_row(row_values, "overdraft_limit", 0.0) or 0.0),
                "credit_card_limit": float(value_from_row(row_values, "credit_card_limit", 0.0) or 0.0),
                "change_reason": "account_opened",
            }
        ]
        if random.random() < 0.22:
            change_date = opening_date + timedelta(days=random.randint(30, 420))
            if change_date.year <= target_year:
                factor = random.uniform(0.8, 1.6)
                limit_events.append(
                    {
                        "event_type": "periodic_review",
                        "event_date": str(change_date),
                        "overdraft_limit": round(float(value_from_row(row_values, "overdraft_limit", 0.0) or 0.0) * factor, 2),
                        "credit_card_limit": round(float(value_from_row(row_values, "credit_card_limit", 0.0) or 0.0) * factor, 2),
                        "change_reason": random.choice(["income_update", "risk_review", "customer_request"]),
                    }
                )
        for event in limit_events:
            limits_rows.append({"account_id": account_id, **event, "ingested_at": now_ts})

        status_events = [
            {
                "event_type": "opened",
                "event_date": str(opening_date),
                "new_status": "active",
                "status_reason": "account_opened",
            },
            {
                "event_type": "approved",
                "event_date": str(approval_date),
                "new_status": "active",
                "status_reason": "approved",
            },
        ]
        status_change_date_value = value_from_row(row_values, "status_change_date")
        if status_change_date_value is not None:
            status_events.append(
                {
                    "event_type": "status_changed",
                    "event_date": str(pd.to_datetime(status_change_date_value).date()),
                    "new_status": value_from_row(row_values, "account_status"),
                    "status_reason": value_from_row(row_values, "status_reason", "status_update") or "status_update",
                }
            )
        closure_date_value = value_from_row(row_values, "closure_date")
        if closure_date_value is not None:
            status_events.append(
                {
                    "event_type": "closed",
                    "event_date": str(pd.to_datetime(closure_date_value).date()),
                    "new_status": "closed",
                    "status_reason": value_from_row(row_values, "status_reason", "closed") or "closed",
                }
            )
        for event in status_events:
            status_rows.append({"account_id": account_id, **event, "ingested_at": now_ts})

        products = []
        bundled = value_from_row(row_values, "bundled_products")
        if isinstance(bundled, str) and bundled.strip():
            products = [p.strip() for p in bundled.split(";") if p.strip()]
        for p in products:
            enroll_rows.append(
                {
                    "account_id": account_id,
                    "product_code": p,
                    "enrollment_date": str(opening_date + timedelta(days=random.randint(0, 60))),
                    "enrollment_status": random.choice(["active", "active", "active", "suspended", "cancelled"]),
                    "ingested_at": now_ts,
                }
            )

        signatory_rows.append(
            {
                "account_id": account_id,
                "customer_id": value_from_row(row_values, "customer_id"),
                "signatory_role": "primary_holder",
                "signing_rule": "single",
                "effective_date": str(opening_date),
                "is_active": True,
                "ingested_at": now_ts,
            }
        )
        linked_joint_accounts = value_from_row(row_values, "linked_joint_accounts")
        if isinstance(linked_joint_accounts, str) and linked_joint_accounts.strip():
            for partner in [p.strip() for p in linked_joint_accounts.split(";") if p.strip()]:
                signatory_rows.append(
                    {
                        "account_id": account_id,
                        "customer_id": partner,
                        "signatory_role": "joint_holder",
                        "signing_rule": random.choice(["either_to_sign", "both_to_sign"]),
                        "effective_date": str(opening_date),
                        "is_active": True,
                        "ingested_at": now_ts,
                    }
                )
        if value_from_row(row_values, "account_type") == "business" and random.random() < 0.65:
            for n in range(random.randint(1, 3)):
                signatory_rows.append(
                    {
                        "account_id": account_id,
                        "customer_id": f"AUTH{random.randint(100000, 999999)}",
                        "signatory_role": random.choice(["authorized_signatory", "finance_manager", "director"]),
                        "signing_rule": random.choice(["single", "dual", "tiered"]),
                        "effective_date": str(opening_date + timedelta(days=random.randint(0, 90))),
                        "is_active": random.random() < 0.9,
                        "ingested_at": now_ts,
                    }
                )

    df_limits = pd.DataFrame(limits_rows)
    df_status = pd.DataFrame(status_rows)
    df_enroll = pd.DataFrame(enroll_rows)
    df_signatories = pd.DataFrame(signatory_rows)

    return df_limits, df_status, df_enroll, df_signatories


def add_history_json_to_main(df_accounts, df_limits, df_status, df_enroll, df_signatories):
    if df_accounts.empty or "account_id" not in df_accounts.columns:
        df_accounts = df_accounts.copy()
        df_accounts["limits_history_json"] = pd.Series(dtype="object")
        df_accounts["status_events_json"] = pd.Series(dtype="object")
        df_accounts["product_enrollments_json"] = pd.Series(dtype="object")
        df_accounts["signatories_json"] = pd.Series(dtype="object")
        df_accounts["cdc_op_hint"] = pd.Series(dtype="object")
        df_accounts["record_last_updated_at"] = pd.Series(dtype="datetime64[ns]")
        return df_accounts

    def grouped_json(df, key):
        out = {}
        if df.empty:
            return out
        cols = [c for c in df.columns if c != key]
        for k, g in df.groupby(key):
            out[k] = g[cols].to_dict(orient="records")
        return out

    limits_map = grouped_json(df_limits, "account_id")
    status_map = grouped_json(df_status, "account_id")
    enroll_map = grouped_json(df_enroll, "account_id")
    sign_map = grouped_json(df_signatories, "account_id")

    df_accounts = df_accounts.copy()
    df_accounts["limits_history_json"] = df_accounts["account_id"].map(lambda x: json.dumps(limits_map.get(x, [])))
    df_accounts["status_events_json"] = df_accounts["account_id"].map(lambda x: json.dumps(status_map.get(x, [])))
    df_accounts["product_enrollments_json"] = df_accounts["account_id"].map(lambda x: json.dumps(enroll_map.get(x, [])))
    df_accounts["signatories_json"] = df_accounts["account_id"].map(lambda x: json.dumps(sign_map.get(x, [])))
    df_accounts["cdc_op_hint"] = np.random.choice(["I", "U"], size=len(df_accounts), p=[0.82, 0.18])
    df_accounts["record_last_updated_at"] = pd.Timestamp.now()
    return df_accounts


def generate_accounts(year, month=None, ultra_fast=False, counter_persist_every=500, include_prior_customers=False):
    """
    Generate accounts for a specific year, optionally filtered to a specific month.
    
        If month is None:
            - All accounts for the year are generated and saved to flat structure (banking_data/accounts_YEAR.parquet)
            - Opening dates follow realistic seasonal monthly weights
    
    If month is specified (1-12):
      - Only accounts with opening_date in that month are generated
      - Saved to monthly structure (banking_data/YEAR/MM/accounts_YEAR_MM.parquet)
      - All history tables and rejected applications follow the same structure
    """
    seed_bytes = os.urandom(4)
    seed_int = int.from_bytes(seed_bytes, byteorder="big")
    random.seed(seed_int)
    np.random.seed(seed_int)
    Faker.seed(seed_int)
    fake = Faker("zu_ZA")

    products_payload = get_retail_bank_products_data()
    branch_codes_payload = get_branch_codes_by_city_data()
    city_matchers = build_city_matchers(branch_codes_payload)
    bundled_products_available = products_payload.get("bundled_products_available", {})
    account_charges = products_payload.get("account_pricing", {})
    bank_names, bank_weights, swift_by_bank = build_bank_selector(products_payload)

    github_repo_path = "banking_data"
    customer_file_base = resolve_customer_file_base(github_repo_path, year, month)
    try:
        df_customers = load_customer_file(customer_file_base)
    except FileNotFoundError:
        print(f"Customer file {customer_file_base}.parquet not found. Exiting.")
        return pd.DataFrame()

    if "date_of_entry" in df_customers.columns:
        entry_dates = pd.to_datetime(df_customers["date_of_entry"], errors="coerce")
        df_customers = df_customers[entry_dates.dt.year == year].copy()

        if month is not None:
            month_end = date(year, month, 1)
            if month < 12:
                month_end = date(year, month + 1, 1) - timedelta(days=1)
            else:
                month_end = date(year, 12, 31)
            df_customers = df_customers[entry_dates <= pd.Timestamp(month_end)].copy()

    if df_customers.empty:
        print(f"No customers found with date_of_entry in year {year}. Exiting.")
        return pd.DataFrame()

    if include_prior_customers and not ultra_fast:
        previous_customers = []
        for prev_year in range(max(2015, year - 3), year):
            try:
                prev_df = load_customer_file(os.path.join(github_repo_path, f"customers_{prev_year}"))
                sample_size = max(1, int(len(prev_df) * 0.03))
                sampled_df = prev_df.sample(n=sample_size, random_state=seed_int)
                previous_customers.append(sampled_df)
            except FileNotFoundError:
                continue

        if previous_customers:
            previous_customers = pd.concat(previous_customers).reset_index(drop=True)
            previous_customers = previous_customers.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
        else:
            previous_customers = pd.DataFrame()

        df_customers = pd.concat([df_customers, previous_customers]).reset_index(drop=True)

    counter = GlobalAccountCounter(github_repo_path, persist_every=counter_persist_every)
    print(f"Starting generation for year {year} with counter at: {counter.peek()}")

    if year == 2020:
        opening_start = date(year, 6, 1)
        opening_end = date(year, 12, 31)
    elif year == 2019:
        opening_start = date(year, 1, 1)
        opening_end = date(year, 12, 31)
    else:
        opening_start = date(max(2015, year - 3), 1, 1)
        opening_end = date(year, 12, 31)

    accounts = []
    rejected_applications = []
    customer_account_counts = load_existing_customer_account_counts(github_repo_path, year, month)
    customer_primary_accounts = {}

    if month is not None:
        print(f"Eligible customers up to {year}-{month:02d}: {len(df_customers)}")
        print(f"Customers with prior accounts from earlier months: {len(customer_account_counts)}")

    df_individuals = df_customers[df_customers["customer_type"] == "Individual"].copy()
    df_companies = df_customers[df_customers["customer_type"] == "Company"].copy()
    individual_ids = df_individuals["customer_id"].values
    non_za_citizens_by_id = {
        row.customer_id: (row.citizenship != "ZA")
        for row in df_individuals[["customer_id", "citizenship"]].itertuples(index=False)
    }
    max_partners = min(max(len(individual_ids) - 1, 0), 3)

    for _, row in tqdm(df_individuals.iterrows(), total=len(df_individuals), desc="Generating Individual Accounts"):
        customer_id = row["customer_id"]
        date_of_entry = pd.to_datetime(row["date_of_entry"]).date() if not pd.isna(row["date_of_entry"]) else opening_start
        current_count = customer_account_counts.get(customer_id, 0)
        education = str(row.get("education_level", "")).lower()
        is_highly_educated = any(token in education for token in ["master", "phd", "doctor", "postgrad"])
        per_customer_cap = 3 if (is_highly_educated and get_income_level(row) == "high") else 2
        max_accounts = per_customer_cap - current_count
        if max_accounts <= 0:
            continue

        num_accounts = min(generate_accounts_with_relationships(row, year), max_accounts)
        income_level = get_income_level(row)
        annual_income = row.get("annual_income", 300000)

        for _ in range(num_accounts):
            acc_type = select_realistic_account_type(row, "Individual")
            is_rejected, rejection_reason = should_reject_application(row, acc_type)
            application_date = choose_opening_date(max(opening_start, date_of_entry), opening_end, year, month, MONTHLY_OPENING_WEIGHTS)
            if application_date is None:
                continue

            if is_rejected:
                global_id = counter.get_next()
                rejected_applications.append(
                    {
                        "application_id": f"APP{global_id:07d}",
                        "customer_id": customer_id,
                        "account_type": acc_type,
                        "application_date": application_date,
                        "rejection_reason": rejection_reason,
                        "rejection_date": application_date,
                    }
                )
                continue

            opening_date = application_date
            approval_date = generate_approval_date(opening_date)
            requirements = generate_account_requirements(row, acc_type)
            account_status, status_change_date, closure_date, status_reason = determine_account_status(opening_date, row, requirements, year)
            branch_code = get_branch_code(row, branch_codes_payload, city_matchers)
            charges = account_charges.get(acc_type, account_charges.get("current", {}))
            channel_details = determine_opening_channel_and_details()
            currency = "ZAR" if random.random() < 0.95 else random.choice(["USD", "EUR"])
            account_tier = determine_account_tier(acc_type, income_level)
            overdraft_limit, credit_card_limit = generate_credit_limit(acc_type, income_level, annual_income)
            bank_name, product_name = choose_bank_and_product(acc_type, products_payload, bank_names, bank_weights)

            global_id = counter.get_next()
            account_number = generate_sa_account_number(branch_code, global_id)
            swift_code = generate_swift_code(swift_by_bank.get(bank_name, "WOLZAJJ"), branch_code) if currency != "ZAR" else None
            iban = generate_iban(account_number) if currency != "ZAR" else None
            account_purpose = determine_account_purpose(row, acc_type)
            expected_amount = round(np.random.lognormal(mean=8.5, sigma=1.2), 2)

            is_primary = customer_id not in customer_primary_accounts
            if is_primary:
                customer_primary_accounts[customer_id] = f"ACC{global_id:07d}"

            statement_frequency = random.choice(["monthly", "quarterly", "annually"])
            online_banking_enabled = random.random() < (0.85 if channel_details["opening_channel"] in ["online", "mobile_app"] else 0.65)
            online_banking_activation_date = (
                opening_date
                if online_banking_enabled and channel_details["opening_channel"] in ["online", "mobile_app"]
                else opening_date + timedelta(days=random.randint(0, 30)) if online_banking_enabled else None
            )

            card_info = generate_card_number(acc_type)
            if card_info:
                card_number, card_type = card_info
                card_expiry_date = date(opening_date.year + random.randint(3, 5), opening_date.month, 1)
                card_issue_date = opening_date if random.random() < 0.8 else opening_date + timedelta(days=random.randint(7, 21))
            else:
                card_number, card_type, card_expiry_date, card_issue_date = None, None, None, None

            beneficiaries = generate_beneficiaries(row, fake)
            cross_border_enabled = currency != "ZAR" or random.random() < 0.3

            accounts.append(
                {
                    "account_id": f"ACC{global_id:07d}",
                    "account_number": account_number,
                    "customer_id": customer_id,
                    "bank_name": bank_name,
                    "bank_product_name": product_name,
                    "account_type": acc_type,
                    "account_purpose": account_purpose,
                    "is_primary_account": is_primary,
                    "opening_date": opening_date,
                    "approval_date": approval_date,
                    "branch_code": branch_code,
                    "kyc_verified": True,
                    "fica_verified": row.get("citizenship") != "ZA",
                    "expected_amount": expected_amount,
                    "initial_deposit": expected_amount,
                    "account_status": account_status,
                    "status_change_date": status_change_date,
                    "closure_date": closure_date,
                    "status_reason": status_reason,
                    "linked_joint_accounts": None,
                    "interest_rate": charges.get("interest_rate", 0.0),
                    "monthly_charges": charges.get("monthly_charges", 0.0),
                    "transactions_rate": charges.get("transactions_rate", 0.0),
                    "negative_balance_rate": charges.get("negative_balance_rate", 0.0),
                    "overdraft_limit": overdraft_limit,
                    "credit_card_limit": credit_card_limit,
                    "bundled_products": generate_bundled_products(acc_type, row, bundled_products_available, year),
                    "currency": currency,
                    "swift_code": swift_code,
                    "iban": iban,
                    "account_tier": account_tier,
                    "statement_frequency": statement_frequency,
                    "online_banking_enabled": online_banking_enabled,
                    "online_banking_activation_date": online_banking_activation_date,
                    "card_number": card_number,
                    "card_type": card_type,
                    "card_issue_date": card_issue_date,
                    "card_expiry_date": card_expiry_date,
                    "beneficiaries": beneficiaries,
                    "cross_border_enabled": cross_border_enabled,
                    **requirements,
                    **channel_details,
                }
            )
            customer_account_counts[customer_id] = customer_account_counts.get(customer_id, 0) + 1

        joint_prob = 0.06 if not is_highly_educated else 0.10
        joint_accounts_to_create = 1 if (year != 2020 and random.random() < joint_prob and (max_accounts - num_accounts) > 0) else 0
        for _ in range(joint_accounts_to_create):
            if max_partners <= 0:
                break
            partners = np.random.choice([cid for cid in individual_ids if cid != customer_id], size=min(random.randint(1, 3), max_partners), replace=False)
            opening_date = choose_opening_date(max(opening_start, date_of_entry), opening_end, year, month, MONTHLY_OPENING_WEIGHTS)
            if opening_date is None:
                continue
            approval_date = generate_approval_date(opening_date)
            requirements = generate_account_requirements(row, "joint")
            account_status, status_change_date, closure_date, status_reason = determine_account_status(opening_date, row, requirements, year)
            branch_code = get_branch_code(row, branch_codes_payload, city_matchers)
            charges = account_charges.get("joint", account_charges.get("current", {}))
            channel_details = determine_opening_channel_and_details()
            account_tier = determine_account_tier("joint", income_level)
            overdraft_limit, credit_card_limit = generate_credit_limit("joint", income_level, annual_income)
            bank_name, product_name = choose_bank_and_product("joint", products_payload, bank_names, bank_weights)
            expected_amount = min(round(np.random.lognormal(mean=8.5, sigma=1.2), 2), 100000)

            global_id = counter.get_next()
            account_number = generate_sa_account_number(branch_code, global_id)
            card_info = generate_card_number("joint")
            if card_info:
                card_number, card_type = card_info
                card_expiry_date = date(opening_date.year + random.randint(3, 5), opening_date.month, 1)
                card_issue_date = opening_date if random.random() < 0.8 else opening_date + timedelta(days=random.randint(7, 21))
            else:
                card_number, card_type, card_expiry_date, card_issue_date = None, None, None, None

            accounts.append(
                {
                    "account_id": f"ACC{global_id:07d}",
                    "account_number": account_number,
                    "customer_id": customer_id,
                    "bank_name": bank_name,
                    "bank_product_name": product_name,
                    "account_type": "joint",
                    "account_purpose": "joint_savings",
                    "is_primary_account": False,
                    "opening_date": opening_date,
                    "approval_date": approval_date,
                    "branch_code": branch_code,
                    "kyc_verified": True,
                    "fica_verified": any(non_za_citizens_by_id.get(cid, False) for cid in [customer_id, *list(partners)]),
                    "expected_amount": expected_amount,
                    "initial_deposit": expected_amount,
                    "account_status": account_status,
                    "status_change_date": status_change_date,
                    "closure_date": closure_date,
                    "status_reason": status_reason,
                    "linked_joint_accounts": ";".join(partners),
                    "interest_rate": charges.get("interest_rate", 0.0),
                    "monthly_charges": charges.get("monthly_charges", 0.0),
                    "transactions_rate": charges.get("transactions_rate", 0.0),
                    "negative_balance_rate": charges.get("negative_balance_rate", 0.0),
                    "overdraft_limit": overdraft_limit,
                    "credit_card_limit": credit_card_limit,
                    "bundled_products": generate_bundled_products("joint", row, bundled_products_available, year),
                    "currency": "ZAR",
                    "swift_code": None,
                    "iban": None,
                    "account_tier": account_tier,
                    "statement_frequency": "monthly",
                    "online_banking_enabled": random.random() < 0.75,
                    "online_banking_activation_date": opening_date + timedelta(days=random.randint(0, 30)),
                    "card_number": card_number,
                    "card_type": card_type,
                    "card_issue_date": card_issue_date,
                    "card_expiry_date": card_expiry_date,
                    "beneficiaries": generate_beneficiaries(row, fake),
                    "cross_border_enabled": False,
                    **requirements,
                    **channel_details,
                }
            )
            customer_account_counts[customer_id] = customer_account_counts.get(customer_id, 0) + 1

    for _, row in tqdm(df_companies.iterrows(), total=len(df_companies), desc="Generating Company Accounts"):
        customer_id = row["customer_id"]
        date_of_entry = pd.to_datetime(row["date_of_entry"]).date() if not pd.isna(row["date_of_entry"]) else opening_start
        current_count = customer_account_counts.get(customer_id, 0)
        max_accounts = 2 - current_count
        if max_accounts <= 0:
            continue

        num_accounts = min(random.choices([1, 2], weights=[0.8, 0.2])[0] if year != 2020 else 1, max_accounts)
        income_level = get_income_level(row)
        annual_income = row.get("annual_income", 1000000)

        for _ in range(num_accounts):
            acc_type = "business"
            is_rejected, rejection_reason = should_reject_application(row, acc_type)
            application_date = choose_opening_date(max(opening_start, date_of_entry), opening_end, year, month, MONTHLY_OPENING_WEIGHTS)
            if application_date is None:
                continue
            if is_rejected:
                global_id = counter.get_next()
                rejected_applications.append(
                    {
                        "application_id": f"APP{global_id:07d}",
                        "customer_id": customer_id,
                        "account_type": acc_type,
                        "application_date": application_date,
                        "rejection_reason": rejection_reason,
                        "rejection_date": application_date,
                    }
                )
                continue

            opening_date = application_date
            approval_date = generate_approval_date(opening_date)
            requirements = generate_account_requirements(row, acc_type)
            account_status, status_change_date, closure_date, status_reason = determine_account_status(opening_date, row, requirements, year)
            branch_code = get_branch_code(row, branch_codes_payload, city_matchers)
            charges = account_charges.get(acc_type, account_charges.get("business", {}))
            channel_details = determine_opening_channel_and_details()
            currency = "ZAR" if random.random() < 0.9 else random.choice(["USD", "EUR"])
            account_tier = determine_account_tier(acc_type, income_level)
            overdraft_limit, credit_card_limit = generate_credit_limit(acc_type, income_level, annual_income)
            bank_name, product_name = choose_bank_and_product(acc_type, products_payload, bank_names, bank_weights)

            global_id = counter.get_next()
            account_number = generate_sa_account_number(branch_code, global_id)
            swift_code = generate_swift_code(swift_by_bank.get(bank_name, "WOLZAJJ"), branch_code) if currency != "ZAR" else None
            iban = generate_iban(account_number) if currency != "ZAR" else None
            account_purpose = determine_account_purpose(row, acc_type)
            expected_amount = round(random.uniform(10000, 1000000), 2)

            is_primary = customer_id not in customer_primary_accounts
            if is_primary:
                customer_primary_accounts[customer_id] = f"ACC{global_id:07d}"

            card_info = generate_card_number(acc_type)
            if card_info:
                card_number, card_type = card_info
                card_expiry_date = date(opening_date.year + random.randint(3, 5), opening_date.month, 1)
                card_issue_date = opening_date if random.random() < 0.8 else opening_date + timedelta(days=random.randint(7, 21))
            else:
                card_number, card_type, card_expiry_date, card_issue_date = None, None, None, None

            accounts.append(
                {
                    "account_id": f"ACC{global_id:07d}",
                    "account_number": account_number,
                    "customer_id": customer_id,
                    "bank_name": bank_name,
                    "bank_product_name": product_name,
                    "account_type": acc_type,
                    "account_purpose": account_purpose,
                    "is_primary_account": is_primary,
                    "opening_date": opening_date,
                    "approval_date": approval_date,
                    "branch_code": branch_code,
                    "kyc_verified": True,
                    "fica_verified": None,
                    "expected_amount": expected_amount,
                    "initial_deposit": expected_amount,
                    "account_status": account_status,
                    "status_change_date": status_change_date,
                    "closure_date": closure_date,
                    "status_reason": status_reason,
                    "linked_joint_accounts": None,
                    "interest_rate": charges.get("interest_rate", 0.0),
                    "monthly_charges": charges.get("monthly_charges", 0.0),
                    "transactions_rate": charges.get("transactions_rate", 0.0),
                    "negative_balance_rate": charges.get("negative_balance_rate", 0.0),
                    "overdraft_limit": overdraft_limit,
                    "credit_card_limit": credit_card_limit,
                    "bundled_products": generate_bundled_products(acc_type, row, bundled_products_available, year),
                    "currency": currency,
                    "swift_code": swift_code,
                    "iban": iban,
                    "account_tier": account_tier,
                    "statement_frequency": random.choice(["monthly", "quarterly"]),
                    "online_banking_enabled": random.random() < 0.95,
                    "online_banking_activation_date": opening_date + timedelta(days=random.randint(0, 14)),
                    "card_number": card_number,
                    "card_type": card_type,
                    "card_issue_date": card_issue_date,
                    "card_expiry_date": card_expiry_date,
                    "beneficiaries": None,
                    "cross_border_enabled": currency != "ZAR" or random.random() < 0.5,
                    **requirements,
                    **channel_details,
                }
            )
            customer_account_counts[customer_id] = customer_account_counts.get(customer_id, 0) + 1

    df_accounts = pd.DataFrame(accounts)
    df_rejected = pd.DataFrame(rejected_applications)
    
    # Filter by month if specified
    if month is not None:
        if not df_accounts.empty and "opening_date" in df_accounts.columns:
            df_accounts["opening_month"] = pd.to_datetime(df_accounts["opening_date"]).dt.month
            df_accounts = df_accounts[df_accounts["opening_month"] == month].copy()
            df_accounts.drop("opening_month", axis=1, inplace=True)

        if not df_rejected.empty and "application_date" in df_rejected.columns:
            df_rejected["application_month"] = pd.to_datetime(df_rejected["application_date"]).dt.month
            df_rejected = df_rejected[df_rejected["application_month"] == month].copy()
            df_rejected.drop("application_month", axis=1, inplace=True)

    df_limits, df_status, df_enroll, df_signatories = build_history_tables(df_accounts, year)
    if ultra_fast:
        df_accounts["limits_history_json"] = "[]"
        df_accounts["status_events_json"] = "[]"
        df_accounts["product_enrollments_json"] = "[]"
        df_accounts["signatories_json"] = "[]"
        df_accounts["cdc_op_hint"] = "I"
        df_accounts["record_last_updated_at"] = pd.Timestamp.now()
    else:
        df_accounts = add_history_json_to_main(df_accounts, df_limits, df_status, df_enroll, df_signatories)

    # Resolve output paths (flat if month=None, monthly if month is specified)
    output_base = get_output_path(github_repo_path, year, month, "accounts")
    output_dir = os.path.dirname(output_base) if month is not None else github_repo_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = output_base + ".parquet"
    try:
        df_accounts.to_parquet(output_file, index=False)
    except Exception:
        output_file = output_base + ".csv"
        df_accounts.to_csv(output_file, index=False)

    # Save history tables with same naming convention
    history_mappings = {
        "account_limits_history": df_limits,
        "account_status_events": df_status,
        "account_product_enrollments": df_enroll,
        "account_signatories": df_signatories,
    }
    
    for base_name, table_df in history_mappings.items():
        history_path = get_output_path(github_repo_path, year, month, base_name)
        parquet_path = history_path + ".parquet"
        try:
            table_df.to_parquet(parquet_path, index=False)
        except Exception:
            csv_path = history_path + ".csv"
            table_df.to_csv(csv_path, index=False)

    if len(df_rejected) > 0:
        rejected_path = get_output_path(github_repo_path, year, month, "rejected_applications")
        rejected_file = rejected_path + ".parquet"
        try:
            df_rejected.to_parquet(rejected_file, index=False)
        except Exception:
            rejected_file = rejected_path + ".csv"
            df_rejected.to_csv(rejected_file, index=False)
        month_str = f" (month {month:02d})" if month is not None else ""
        print(f"Generated {len(df_rejected)} rejected applications for year {year}{month_str}.")
        print(f"Saved to {rejected_file}")

    month_str = f" (month {month:02d})" if month is not None else ""
    print(f"Generated {len(df_accounts)} accounts for year {year}{month_str}.")
    print(f"Final counter: {counter.peek()}")
    print(f"Saved to {output_file}")

    counter.flush()

    if len(df_accounts) > 0:
        print("\nAccount Summary:")
        print(f"- Primary accounts: {df_accounts['is_primary_account'].sum()}")
        if "opening_date" in df_accounts.columns:
            month_distribution = pd.to_datetime(df_accounts["opening_date"]).dt.month.value_counts().sort_index()
            print(f"- Monthly opening distribution: {month_distribution.to_dict()}")
        print(f"- Accounts with overdraft: {(df_accounts['overdraft_limit'] > 0).sum()}")
        print(f"- Accounts with credit cards: {(df_accounts['credit_card_limit'] > 0).sum()}")
        print(f"- Foreign currency accounts: {(df_accounts['currency'] != 'ZAR').sum()}")
        print(f"- Online banking enabled: {df_accounts['online_banking_enabled'].sum()}")
        print(f"- Accounts with beneficiaries: {df_accounts['beneficiaries'].notna().sum()}")
        print(f"- Cross-border enabled: {df_accounts['cross_border_enabled'].sum()}")
        print(f"- Closed accounts: {(df_accounts['account_status'] == 'closed').sum()}")
        print(f"- Dormant accounts: {(df_accounts['account_status'] == 'dormant').sum()}")
        print(f"- Limits history rows: {len(df_limits)}")
        print(f"- Status event rows: {len(df_status)}")
        print(f"- Product enrollment rows: {len(df_enroll)}")
        print(f"- Signatory rows: {len(df_signatories)}")
        print("- Main table includes embedded history JSON columns for CDC practice")

    return df_accounts


def prompt_for_year(default_year=2024):
    raw = input(f"Enter year for account generation [{default_year}]: ").strip()
    if not raw:
        return default_year
    try:
        year = int(raw)
    except ValueError:
        print(f"Invalid year '{raw}', using default {default_year}.")
        return default_year
    if year < 2015 or year > 2035:
        print(f"Year {year} out of range, using default {default_year}.")
        return default_year
    return year


def prompt_for_month():
    raw = input("Enter month (1-12) or leave blank for all months: ").strip()
    if not raw:
        return None
    try:
        month = int(raw)
        if 1 <= month <= 12:
            return month
        else:
            print(f"Invalid month '{month}', must be 1-12. Generating for all months.")
            return None
    except ValueError:
        print(f"Invalid month '{raw}'. Generating for all months.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate account data for a specific year and optional month")
    parser.add_argument("--year", type=int, help="Year for account data generation")
    parser.add_argument("--month", type=int, help="Month (1-12) for monthly subfolder organization. Omit for all months.")
    parser.add_argument("--ultra-fast", action="store_true", help="Enable maximum throughput mode (skips previous-year customer blending and heavy JSON history embedding).")
    parser.add_argument(
        "--counter-persist-every",
        type=int,
        default=500,
        help="Persist account counter every N increments (higher is faster, default: 500).",
    )
    parser.add_argument(
        "--include-prior-customers",
        action="store_true",
        help="Include a small sample of prior-year customers (default is off to focus on customers opened in target year).",
    )
    args = parser.parse_args()

    selected_year = args.year if args.year is not None else prompt_for_year(2024)
    selected_month = args.month
    
    if selected_month is None and args.month is None:
        # Only prompt for month if not provided via CLI and year was provided via CLI or manual entry
        selected_month = prompt_for_month()
    
    if selected_month is not None and (selected_month < 1 or selected_month > 12):
        print(f"Invalid month {selected_month}, must be 1-12. Generating for all months.")
        selected_month = None
    
    generate_accounts(
        selected_year,
        selected_month,
        ultra_fast=args.ultra_fast,
        counter_persist_every=args.counter_persist_every,
        include_prior_customers=args.include_prior_customers,
    )
