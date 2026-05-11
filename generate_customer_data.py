from __future__ import annotations

import calendar
import os
import random
import string
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from commons.data_loader import (
    get_branches_data,
    get_locations_data,
    get_location_profiles_data,
    get_names_by_country_data,
    get_names_data,
    get_occupations_data,
    get_phone_rules_data,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "banking_data"
COMMONS_DIR = BASE_DIR / "commons"


def introduce_typo(text, typo_prob=0.1):
    """Introduce typographical errors in a string with a given probability."""
    if text is None or random.random() > typo_prob:
        return text
    text = list(str(text))
    if len(text) <= 1:
        return "".join(text)
    idx = random.randint(0, len(text) - 1)
    action = random.choice(["swap", "delete", "add"])
    if action == "swap" and idx < len(text) - 1:
        text[idx], text[idx + 1] = text[idx + 1], text[idx]
    elif action == "delete":
        text.pop(idx)
    elif action == "add":
        text.insert(idx, random.choice(string.ascii_lowercase))
    return "".join(text)


def format_date_yymmdd(dt):
    """Format date as YYMMDD, handling dates before 1900 on Windows."""
    if dt is None or pd.isna(dt):
        return ""
    try:
        return dt.strftime("%y%m%d")
    except (ValueError, OSError):
        year_str = str(dt.year % 100).zfill(2)
        month_str = str(dt.month).zfill(2)
        day_str = str(dt.day).zfill(2)
        return f"{year_str}{month_str}{day_str}"


def adjust_date_for_year(birth_date, year_change):
    """Adjust the year of a date while keeping the day valid."""
    target_year = birth_date.year + year_change
    month = birth_date.month
    day = birth_date.day
    last_day = calendar.monthrange(target_year, month)[1]
    day = min(day, last_day)
    try:
        return date(target_year, month, day)
    except ValueError:
        return date(target_year, month, last_day)


def generate_sa_id_number(birth_date):
    """Generate a South African ID number with the correct birth date prefix."""
    yymmdd = format_date_yymmdd(birth_date)
    sequence = "".join(str(random.randint(0, 9)) for _ in range(4))
    citizenship = random.choice(["0", "1"])
    gender = random.choice(["0", "1"])
    checksum = str(random.randint(0, 9))
    return yymmdd + sequence + citizenship + gender + checksum


def generate_birth_certificate_number():
    """Generate a birth certificate number."""
    return "".join(str(random.randint(0, 9)) for _ in range(11))


def generate_passport_number():
    """Generate a simple passport-style identifier."""
    return "".join(random.choice(string.ascii_uppercase) for _ in range(2)) + "".join(
        random.choice(string.digits) for _ in range(7)
    )


def load_reference_data():
    occupations_payload = get_occupations_data()
    locations_payload = get_locations_data()
    location_profiles_payload = get_location_profiles_data()
    branches = get_branches_data()
    names_by_ethnicity = get_names_data()
    names_by_country = get_names_by_country_data()
    phone_rules = get_phone_rules_data()

    occupations = [entry["name"] for entry in occupations_payload["occupations"]]
    occupation_lookup = {
        entry["name"]: {
            "required_education": entry["required_education"],
            "range": tuple(entry["income_range"]),
            "weight": entry["weight"],
            "sector": entry.get("sector"),
        }
        for entry in occupations_payload["occupations"]
    }
    occupation_probs = np.array([entry["weight"] for entry in occupations_payload["occupations"]], dtype=float)
    occupation_probs = occupation_probs / occupation_probs.sum()

    education_levels = occupations_payload["education_levels"]
    education_probs = np.array(occupations_payload["education_probabilities"], dtype=float)
    education_probs = education_probs / education_probs.sum()
    education_hierarchy = {edu: idx for idx, edu in enumerate(education_levels)}

    provinces = [entry["name"] for entry in locations_payload["provinces"]]
    province_probs = np.array([entry["weight"] for entry in locations_payload["provinces"]], dtype=float)
    province_probs = province_probs / province_probs.sum()
    cities_by_province = {entry["name"]: entry["cities"] for entry in locations_payload["provinces"]}
    location_profiles = location_profiles_payload.get("provinces", {})

    branches_by_province = defaultdict(list)
    branches_by_city = defaultdict(list)
    for branch in branches:
        branches_by_province[branch["province"]].append(branch)
        branches_by_city[(branch["province"], branch["city"])].append(branch)

    return {
        "occupations": occupations,
        "occupation_lookup": occupation_lookup,
        "occupation_probs": occupation_probs,
        "education_levels": education_levels,
        "education_probs": education_probs,
        "education_hierarchy": education_hierarchy,
        "provinces": provinces,
        "province_probs": province_probs,
        "cities_by_province": cities_by_province,
        "location_profiles": location_profiles,
        "branches": branches,
        "branches_by_province": branches_by_province,
        "branches_by_city": branches_by_city,
        "names_by_ethnicity": names_by_ethnicity,
        "names_by_country": names_by_country,
        "phone_rules": phone_rules,
        "informal_occupations": occupations_payload.get("informal_occupations", []),
    }


def choose_weighted(options, probabilities):
    return random.choices(options, weights=probabilities, k=1)[0]


def pick_from_distribution(distribution):
    labels = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(labels, weights=weights, k=1)[0]


def generate_name_from_profile(country, ethnicity, gender, names_by_country, names_by_ethnicity):
    country_profile = names_by_country.get(country)
    pools = None
    if country_profile:
        if "ethnic_groups" in country_profile:
            pools = country_profile["ethnic_groups"].get(ethnicity)
        if pools is None:
            pools = country_profile
    if pools is None:
        pools = names_by_ethnicity.get(ethnicity) or names_by_ethnicity["Black"]
    if gender == "M":
        first_name = random.choice(pools["male_first_names"])
    else:
        first_name = random.choice(pools["female_first_names"])
    surname = random.choice(pools["surnames"])
    middle_name = random.choice(pools.get("middle_names", [])) if pools.get("middle_names") and random.random() < 0.35 else None
    name_parts = [first_name]
    if middle_name:
        name_parts.append(middle_name)
    name_parts.append(surname)
    return " ".join(name_parts)


def generate_phone_number(country, phone_rules):
    rule = phone_rules.get(country, phone_rules["South Africa"])
    prefix = random.choice(rule["mobile_prefixes"])
    suffix = "".join(random.choice(string.digits) for _ in range(7))
    number = f"{prefix} {suffix}"
    if random.random() < 0.2:
        number = number.replace(" ", "")
    return number


def mutate_phone_number(phone_number):
    if phone_number is None:
        return None
    error_type = random.choice(["drop_country_code", "truncate", "swap_digits", "drop_separator"])
    digits = "".join(ch for ch in phone_number if ch.isdigit())
    if error_type == "drop_country_code" and phone_number.startswith("+"):
        return phone_number.split(" ", 1)[-1]
    if error_type == "truncate":
        return digits[:-2] if len(digits) > 2 else digits
    if error_type == "swap_digits" and len(digits) > 3:
        idx = random.randint(0, len(digits) - 2)
        digit_list = list(digits)
        digit_list[idx], digit_list[idx + 1] = digit_list[idx + 1], digit_list[idx]
        return "".join(digit_list)
    return phone_number.replace(" ", "")


def mutate_email(email):
    if email is None:
        return None
    error_type = random.choice(["remove_at", "truncate_domain", "drop_tld", "swap_chars"])
    if error_type == "remove_at":
        return email.replace("@", "")
    if error_type == "truncate_domain":
        return email.split("@", 1)[0] + "@"
    if error_type == "drop_tld":
        user, _, domain = email.partition("@")
        return f"{user}@{domain.rsplit('.', 1)[0]}"
    if error_type == "swap_chars" and len(email) > 4:
        idx = random.randint(0, len(email) - 2)
        chars = list(email)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    return email


def mutate_id_number(id_type, id_number, birth_date):
    if id_number is None:
        return None
    error_type = random.choice(["length_error", "digit_swap", "date_mismatch", "prefix_error"])
    if id_type == "National ID":
        if error_type == "length_error":
            return id_number[:-1] if len(id_number) > 1 else id_number
        if error_type == "digit_swap" and len(id_number) > 6:
            idx = random.randint(0, len(id_number) - 2)
            chars = list(id_number)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)
        if error_type == "date_mismatch" and birth_date is not None:
            wrong_birth_date = adjust_date_for_year(birth_date, random.choice([-100, -10, 10, 100]))
            return generate_sa_id_number(wrong_birth_date)
        return random.choice([id_number[:6] + "99" + id_number[8:], id_number[:10]])
    if error_type == "length_error":
        return id_number[:-2]
    if error_type == "digit_swap" and len(id_number) > 3:
        idx = random.randint(0, len(id_number) - 2)
        chars = list(id_number)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    if error_type == "prefix_error":
        return "XX" + id_number[2:]
    return id_number


def choose_branch(province, city, reference_data):
    candidates = reference_data["branches_by_city"].get((province, city))
    if not candidates:
        candidates = reference_data["branches_by_province"].get(province, reference_data["branches"])
    return random.choice(candidates)


def generate_address(fake, city, province):
    base_address = f"{introduce_typo(fake.street_address(), 0.08)}, {city}, {province}, South Africa"
    if random.random() < 0.12:
        base_address = f"Informal Settlement, {city}, {province}"
    if random.random() < 0.12:
        base_address = base_address.replace("South Africa", "")
    if random.random() < 0.10:
        base_address = introduce_typo(base_address, 1.0)
    return base_address.strip().replace("  ", " ")


def generate_customer_data(year):
    seed_bytes = os.urandom(4)
    seed_int = int.from_bytes(seed_bytes, byteorder="big")
    random.seed(seed_int)
    np.random.seed(seed_int)
    Faker.seed(seed_int)

    fake = Faker("zu_ZA")
    reference_data = load_reference_data()

    if year == 2020:
        num_individuals = random.randint(15, 40)
        num_companies = random.randint(0, 3)
        print("Note: 2020 year - Reduced registrations due to COVID-19 lockdowns in South Africa.")
    elif year == 2021:
        num_individuals = random.randint(10000, 15000)
        num_companies = random.randint(1, 8)
        print("Note: 2021 year - Recovery phase post-COVID.")
    elif year in (2022, 2023):
        num_individuals = random.randint(18000, 23000)
        num_companies = random.randint(1, 8)
    else:
        num_individuals = random.randint(10000, 20000)
        num_companies = random.randint(1, 8)

    print(f"Starting generation for year {year}...")

    if num_individuals == 0 and num_companies == 0:
        print("No customers generated for this year.")
        df = pd.DataFrame()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"customers_{year}.parquet"
        try:
            df.to_parquet(output_file, index=False)
        except Exception:
            output_file = OUTPUT_DIR / f"customers_{year}.csv"
            df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        return df

    all_customers = []
    used_ids = set()
    quality_counter = Counter()
    batch_size = 500
    individual_batches = (num_individuals + batch_size - 1) // batch_size

    education_levels = reference_data["education_levels"]
    education_probs = reference_data["education_probs"]
    education_hierarchy = reference_data["education_hierarchy"]
    occupations = reference_data["occupations"]
    occupation_lookup = reference_data["occupation_lookup"]
    occupation_probs = reference_data["occupation_probs"]
    provinces = reference_data["provinces"]
    province_probs = reference_data["province_probs"]
    cities_by_province = reference_data["cities_by_province"]
    names_by_ethnicity = reference_data["names_by_ethnicity"]
    names_by_country = reference_data["names_by_country"]
    phone_rules = reference_data["phone_rules"]
    location_profiles = reference_data["location_profiles"]
    informal_occupations = set(reference_data["informal_occupations"])

    ethnicity_options = ["Black", "Coloured", "White", "Indian", "Asian"]
    ethnicity_weights = [0.80, 0.09, 0.08, 0.025, 0.005]
    non_sa_nationalities = ["Zimbabwe", "Lesotho", "Botswana", "Namibia", "Mozambique"]
    non_sa_weights = [0.30, 0.18, 0.15, 0.12, 0.25]
    capture_channels = ["Branch", "Mobile", "Online", "Call Center"]
    capture_weights = [0.42, 0.29, 0.22, 0.07]

    def generate_batch_individuals(batch_size_value, start_idx, used_ids_local=None):
        if used_ids_local is None:
            used_ids_local = set()

        ages = np.random.choice([21, 25, 35, 45, 55, 65, 75], size=batch_size_value, p=[0.35, 0.30, 0.20, 0.10, 0.04, 0.008, 0.002])
        ages = np.maximum(ages, 21)
        genders = np.random.choice(["M", "F"], size=batch_size_value, p=[0.49, 0.51])
        occupations_batch = np.array([None] * batch_size_value, dtype=object)
        provinces_batch = np.random.choice(provinces, size=batch_size_value, p=province_probs)
        ethnicity_batch = np.random.choice(ethnicity_options, size=batch_size_value, p=ethnicity_weights)
        capture_batch = np.random.choice(capture_channels, size=batch_size_value, p=capture_weights)
        education_batch = np.array([
            pick_from_distribution(location_profiles.get(province, {}).get("education", dict(zip(education_levels, education_probs))))
            for province in provinces_batch
        ], dtype=object)

        for i in range(batch_size_value):
            education = education_batch[i]
            valid_occupations = [
                occ
                for occ in occupations
                if education_hierarchy.get(education, 0)
                >= education_hierarchy.get(occupation_lookup[occ]["required_education"], 0)
            ]
            if not valid_occupations:
                valid_occupations = ["Unemployed Unskilled"]

            valid_probs = np.array([occupation_lookup[occ]["weight"] for occ in valid_occupations], dtype=float)
            valid_probs = valid_probs / valid_probs.sum()
            occupations_batch[i] = (
                "Unemployed Unskilled"
                if random.random() < 0.28
                else random.choices(valid_occupations, weights=valid_probs, k=1)[0]
            )

        results = []
        local_idx = start_idx
        for i in range(batch_size_value):
            age = int(ages[i])
            gender = genders[i]
            occupation = occupations_batch[i]
            province = provinces_batch[i]
            ethnicity = ethnicity_batch[i]
            capture_channel = capture_batch[i]
            city = random.choice(cities_by_province[province])
            province_profile = location_profiles.get(province, {})

            nationality = "South Africa" if random.random() < 0.93 else random.choices(non_sa_nationalities, weights=non_sa_weights, k=1)[0]
            citizenship = "ZA" if nationality == "South Africa" else nationality[:2].upper()
            full_name = generate_name_from_profile(nationality, ethnicity, gender, names_by_country, names_by_ethnicity)
            full_name = introduce_typo(full_name, 0.05)

            income_range = occupation_lookup.get(occupation, {"range": (0, 0)})["range"]
            annual_income = int(np.random.uniform(income_range[0], max(income_range[1], income_range[0] + 1)) * (1 + (age - 25) * 0.01))
            base_risk = 0.15
            if annual_income < 120000:
                base_risk += 0.25
            if age < 25:
                base_risk += 0.15
            if occupation in ["Unemployed Unskilled", "Student"]:
                base_risk += 0.20
            if capture_channel == "Branch":
                base_risk += 0.03
            if province_profile.get("exposure", {}).get("cash_preference", 0) > 0.4:
                base_risk += 0.02
            risk_score = min(round(base_risk + np.random.random() * 0.15, 3), 0.99)

            max_birth_date = date(year, 12, 31) - timedelta(days=21 * 365)
            min_birth_date = date(year, 1, 1) - timedelta(days=75 * 365)
            days_range = (max_birth_date - min_birth_date).days
            correct_birth_date = min_birth_date + timedelta(days=random.randint(0, days_range))
            birth_date = correct_birth_date
            issues = []
            if random.random() < 0.05:
                birth_date = adjust_date_for_year(correct_birth_date, random.choice([100, -100, 10, -10]))
                issues.append("birth_date_mismatch")

            id_type = "National ID" if nationality == "South Africa" else "Passport"
            id_number = generate_sa_id_number(correct_birth_date) if id_type == "National ID" else generate_passport_number()
            if random.random() < 0.05:
                id_number = mutate_id_number(id_type, id_number, correct_birth_date)
                issues.append("id_number_error")

            address = generate_address(fake, city, province)
            postal_code = f"{random.randint(1000, 9999)}"
            if random.random() < 0.03:
                postal_code = postal_code[:-1]
                issues.append("postal_code_error")

            if year == 2020 and random.random() < 0.30:
                date_of_entry = fake.date_between(start_date=date(year, 1, 1), end_date=date(year, 3, 26))
            else:
                date_of_entry = date(year, random.randint(1, 12), random.randint(1, 28))

            phone_number = generate_phone_number(nationality, phone_rules)
            if random.random() < 0.15:
                phone_number = mutate_phone_number(phone_number)
                issues.append("phone_format_error")

            email = fake.email() if random.random() < 0.65 else None
            if email and random.random() < 0.12:
                email = mutate_email(email)
                issues.append("email_format_error")

            tax_id_number = (
                "".join(str(random.randint(0, 9)) for _ in range(10))
                if random.random() < (0.35 if occupation and "unemployed" in occupation.lower() or annual_income < 80000 else 0.82)
                else None
            )
            if tax_id_number and random.random() < 0.10:
                tax_id_number = tax_id_number[:8]
                issues.append("tax_id_truncated")

            source_of_funds = (
                random.choice(["Family Support", "Social Grants", "Savings", "Part-time Work"])
                if occupation and "unemployed" in occupation.lower()
                else random.choice(["Family Support", "Student Loan", "Part-time Work"])
                if occupation == "Student"
                else random.choice(["Employment Income", "Business Income", "Savings", "Pension"]) if age > 55 else "Employment Income"
            )

            exposure_profile = pick_from_distribution({
                "Digital First": province_profile.get("exposure", {}).get("digital_onboarding", 0.25),
                "Branch Assisted": province_profile.get("exposure", {}).get("urban", 0.25),
                "Cash Heavy": province_profile.get("exposure", {}).get("cash_preference", 0.25),
                "Mobile First": province_profile.get("exposure", {}).get("mobile_first", 0.25),
                "Cross-Border Frequent": province_profile.get("exposure", {}).get("cross_border_activity", 0.10),
            })
            customer_goal = pick_from_distribution(province_profile.get("goals", {
                "salary_account": 0.25,
                "savings_buffer": 0.20,
                "payments_and_transfers": 0.20,
                "debit_card_usage": 0.15,
                "credit_access": 0.10,
                "business_transactions": 0.10,
            }))
            spending_habit = pick_from_distribution(province_profile.get("spending", {
                "groceries": 0.30,
                "transport": 0.15,
                "housing": 0.20,
                "utilities": 0.10,
                "debt_repayment": 0.10,
                "entertainment": 0.05,
                "mobile_airtime": 0.05,
                "cash_withdrawals": 0.05,
            }))

            branch = choose_branch(province, city, reference_data)
            if random.random() < 0.02:
                branch = random.choice(reference_data["branches"])
                if branch["province"] != province:
                    issues.append("branch_mismatch")

            customer_id = f"IND{year % 100:02d}{local_idx:06d}"
            while customer_id in used_ids_local:
                local_idx += 1
                customer_id = f"IND{year % 100:02d}{local_idx:06d}"
            used_ids_local.add(customer_id)

            visa_type = None
            visa_expiry_date = None
            if nationality != "South Africa":
                visa_type = random.choice(["Work", "Study", "Business", "Residence"])
                visa_expiry_date = fake.date_between(start_date=date(year - 2, 1, 1), end_date=date(year + 2, 1, 1))

            preferred_contact_method = random.choice(["Email", "Phone", "SMS", None])
            next_of_kin = None if random.random() < 0.85 else introduce_typo(generate_name_from_ethnicity(ethnicity, random.choice(["M", "F"]), names_by_ethnicity), 0.05)
            is_pep = random.random() < 0.01
            sanctioned_country = random.random() < 0.005

            customer_data = {
                "customer_id": customer_id,
                "customer_type": "Individual",
                "full_name": full_name,
                "birth_date": birth_date,
                "citizenship": citizenship,
                "nationality": nationality,
                "residential_address": address,
                "residential_postal_code": postal_code,
                "commercial_address": None,
                "email": email,
                "phone_number": phone_number,
                "id_type": id_type,
                "id_number": id_number,
                "expiry_date": None if id_type == "National ID" else fake.future_date(end_date="+3y"),
                "visa_type": visa_type,
                "visa_expiry_date": visa_expiry_date,
                "is_pep": is_pep,
                "sanctioned_country": sanctioned_country,
                "risk_score": risk_score,
                "tax_id_number": tax_id_number,
                "occupation": occupation,
                "employer_name": (
                    introduce_typo(fake.company(), 0.05)
                    if occupation and "unemployed" not in occupation.lower() and random.random() < 0.65
                    else None
                ),
                "source_of_funds": source_of_funds,
                "marital_status": random.choice(["Single", "Married", "Divorced", "Widowed"]),
                "gender": gender,
                "preferred_contact_method": preferred_contact_method,
                "next_of_kin": next_of_kin,
                "date_of_entry": date_of_entry,
                "annual_income": annual_income,
                "education_level": education,
                "location_exposure": exposure_profile,
                "financial_goal": customer_goal,
                "spending_habit": spending_habit,
                "ethnicity": ethnicity,
                "branch_id": branch["branch_id"],
                "branch_name": branch["branch_name"],
                "branch_city": branch["city"],
                "branch_province": branch["province"],
                "capture_channel": capture_channel,
                "source_system": random.choice(["core_banking", "branch_capture", "digital_onboarding", "migration_import"]),
                "customer_segment": random.choice(["Retail", "Mass Market", "Affluent"]),
                "is_affidavit": False,
                "data_issue_flags": ";".join(issues) if issues else None,
            }
            quality_counter.update(issues or ["clean"])
            results.append(customer_data)
            local_idx += 1

        return results, used_ids_local

    def generate_batch_companies(batch_size_value, start_idx, used_ids_local):
        results = []
        for i in range(batch_size_value):
            idx = start_idx + i + 1
            company_name = introduce_typo(fake.company(), 0.05)
            company_age = random.randint(1, 20)
            employees = random.randint(5, 80)
            turnover = random.randint(1000000, 30000000)
            province = random.choices(provinces, weights=province_probs, k=1)[0]
            city = random.choice(cities_by_province[province])
            branch = choose_branch(province, city, reference_data)
            risk_score = round(0.2 + np.random.random() * 0.25, 3)
            date_of_entry = date(year, random.randint(1, 12), random.randint(1, 28))
            phone_number = generate_phone_number("South Africa", phone_rules)
            if random.random() < 0.1:
                phone_number = mutate_phone_number(phone_number)
            customer_id = f"COM{year % 100:02d}{idx:06d}"
            while customer_id in used_ids_local:
                idx += 1
                customer_id = f"COM{year % 100:02d}{idx:06d}"
            used_ids_local.add(customer_id)

            issues = []
            if random.random() < 0.04:
                issues.append("branch_mismatch")
            if random.random() < 0.08:
                issues.append("company_name_typo")

            company_data = {
                "customer_id": customer_id,
                "customer_type": "Company",
                "full_name": company_name,
                "birth_date": None,
                "citizenship": "ZA",
                "nationality": "South Africa",
                "residential_address": None,
                "residential_postal_code": None,
                "commercial_address": f"{introduce_typo(fake.street_address(), 0.08)}, {city}, {province}, South Africa",
                "email": fake.company_email() if random.random() < 0.9 else None,
                "phone_number": phone_number,
                "id_type": "Registration Number",
                "id_number": f"{random.randint(1900, year)}/{random.randint(100000, 999999)}/{random.randint(1, 99)}",
                "expiry_date": None,
                "visa_type": None,
                "visa_expiry_date": None,
                "is_pep": False,
                "sanctioned_country": random.random() < 0.005,
                "risk_score": risk_score,
                "tax_id_number": "".join(str(random.randint(0, 9)) for _ in range(10)) if random.random() < 0.9 else None,
                "occupation": random.choice(["Retail", "Manufacturing", "Finance", "IT", "Services", "Informal Trade"]),
                "employer_name": None,
                "source_of_funds": random.choice(["Business Income", "Investment Income", "Trade Receipts"]),
                "marital_status": None,
                "gender": None,
                "preferred_contact_method": random.choice(["Email", "Phone", None]),
                "next_of_kin": introduce_typo(fake.name(), 0.05) if random.random() < 0.8 else None,
                "date_of_entry": date_of_entry,
                "annual_income": turnover,
                "education_level": None,
                "location_exposure": None,
                "financial_goal": None,
                "spending_habit": None,
                "ethnicity": None,
                "branch_id": branch["branch_id"],
                "branch_name": branch["branch_name"],
                "branch_city": branch["city"],
                "branch_province": branch["province"],
                "capture_channel": random.choice(["Branch", "Online", "Call Center"]),
                "source_system": random.choice(["core_banking", "branch_capture", "digital_onboarding", "migration_import"]),
                "customer_segment": random.choice(["SME", "Commercial", "Corporate"]),
                "company_age": company_age,
                "number_of_employees": employees,
                "annual_turnover": turnover,
                "directors_count": random.randint(1, 3),
                "shareholders_count": random.randint(1, 5),
                "beneficial_owners_count": random.randint(1, 4),
                "bee_level": random.randint(1, 8) if random.random() < 0.7 else None,
                "vat_registered": random.random() < 0.7,
                "industry_risk_rating": random.choice(["Low", "Medium", "High", None]),
                "data_issue_flags": ";".join(issues) if issues else None,
                "is_affidavit": False,
            }
            quality_counter.update(issues or ["clean"])
            results.append(company_data)
        return results, used_ids_local

    current_idx = 0
    print(f"Generating {num_individuals} individuals in {individual_batches} batches...")
    for batch in tqdm(range(individual_batches), desc="Individual batches"):
        remaining = num_individuals - batch * batch_size
        current_batch_size = min(batch_size, remaining)
        batch_customers, used_ids = generate_batch_individuals(current_batch_size, current_idx, used_ids)
        all_customers.extend(batch_customers)
        current_idx += current_batch_size

    if num_companies > 0:
        print(f"Generating {num_companies} companies...")
        company_customers, used_ids = generate_batch_companies(num_companies, 0, used_ids)
        all_customers.extend(company_customers)

    df = pd.DataFrame(all_customers)

    if "reason_for_opening_account" in df.columns:
        df = df.drop(columns=["reason_for_opening_account"])

    duplicate_ids = df[df["customer_id"].duplicated(keep=False)]
    if not duplicate_ids.empty:
        print(f"WARNING: Found {len(duplicate_ids)} duplicate customer_id values.")

    if not df.empty and "customer_type" in df.columns:
        individual_mask = df["customer_type"] == "Individual"
        num_individuals_df = int(individual_mask.sum())
        if num_individuals_df > 0:
            next_of_kin_indices = np.random.choice(
                df[individual_mask].index,
                size=min(int(num_individuals_df * 0.1), num_individuals_df),
                replace=False,
            )
            df.loc[next_of_kin_indices, "next_of_kin"] = [
                introduce_typo(generate_name_from_ethnicity(random.choice(ethnicity_options), random.choice(["M", "F"]), names_by_ethnicity), 0.05)
                for _ in range(len(next_of_kin_indices))
            ]

    df = df.sample(frac=1, random_state=seed_int).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"customers_{year}.parquet"
    try:
        df.to_parquet(output_file, index=False)
    except Exception as exc:
        print(f"Parquet export failed ({exc}); writing CSV fallback.")
        output_file = OUTPUT_DIR / f"customers_{year}.csv"
        df.to_csv(output_file, index=False)

    print(f"Generated {len(df)} customers (Individuals: {num_individuals}, Companies: {num_companies}) for year {year}")
    if not df.empty:
        individual_df = df[df["customer_type"] == "Individual"]
        if len(individual_df) > 0:
            unemployed_with_employer = individual_df[
                (individual_df["occupation"].str.contains("unemployed", case=False, na=False))
                & (individual_df["employer_name"].notna())
            ]
            missing_email_rate = individual_df["email"].isna().mean()
            missing_tax_id_rate = individual_df["tax_id_number"].isna().mean()
            invalid_id_numbers = individual_df[
                (individual_df["id_type"] == "National ID") & (individual_df["id_number"].str.len() != 13)
            ]
            duplicate_ids = df[df["customer_id"].duplicated(keep=False)]
            birth_date_errors = individual_df[
                (individual_df["id_type"] == "National ID")
                & (individual_df["birth_date"].apply(format_date_yymmdd) != individual_df["id_number"].str[:6])
            ]
            flagged_rows = df["data_issue_flags"].notna().sum() if "data_issue_flags" in df.columns else 0

            print("Data Quality Summary:")
            print(f"- Unemployed individuals with employer names: {len(unemployed_with_employer)}")
            print(f"- Individuals without email: {missing_email_rate:.1%}")
            print(f"- Individuals without tax ID: {missing_tax_id_rate:.1%}")
            print(f"- Invalid ID numbers: {len(invalid_id_numbers)}")
            print(f"- Birth date mismatches with ID: {len(birth_date_errors)}")
            print(f"- Duplicate customer_id: {len(duplicate_ids)}")
            print(f"- Rows with explicit issue flags: {flagged_rows}")
            print(f"- Clean rows: {quality_counter.get('clean', 0)}")
            print(f"- Typographical errors introduced: Yes (NAMES, ADDRESSES, EMAILS, PHONES)")
            print(f"- Branch table used: Yes ({len(reference_data['branches'])} branches)")
            print(f"- Location lookup used: Yes ({len(reference_data['provinces'])} provinces)")

    print(f"Saved to {output_file}")
    return df


if __name__ == "__main__":
    year = 2024
    generate_customer_data(year)
