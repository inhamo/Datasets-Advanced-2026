from __future__ import annotations

import argparse
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from commons.data_loader import get_sa_banking_realism_data
from commons.id_factory import make_application_id, make_loan_id


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"
fake = Faker()
REALISM = get_sa_banking_realism_data()

LOAN_OUTPUT_DROP_COLUMNS = {
    "debt_to_income",
    "affordability_pass",
    "gross_annual_income",
    "gross_monthly_incmoe",
    "gross_monthly_income",
    "net_monthly_income",
    "verified_monthly_expenses",
    "existing_monthly_debt",
    "discretionary_income",
    "max_affordable_installment",
    "affordabilit_ratio",
    "affordability_ratio",
    "pricing_basis",
    "ltv_cap",
    "loan_to-value_ratio",
    "loan_to_value_ratio",
    "application_outcome",
    "workflow_state",
}


LOAN_PRODUCTS = {
    "Home Loan": {
        "term_options": [120, 180, 240, 300, 360],
        "ltv_cap": 0.90,
        "product_cap": 8_000_000,
        "secured": True,
        "spread_base": 1.4,
        "variable_default": True,
    },
    "Vehicle Loan": {
        "term_options": [24, 36, 48, 60, 72],
        "ltv_cap": 0.85,
        "product_cap": 1_500_000,
        "secured": True,
        "spread_base": 2.4,
        "variable_default": False,
    },
    "Business Loan": {
        "term_options": [12, 24, 36, 48, 60, 84],
        "ltv_cap": 0.75,
        "product_cap": 15_000_000,
        "secured": True,
        "spread_base": 2.8,
        "variable_default": True,
    },
    "Personal Loan": {
        "term_options": [12, 24, 36, 48, 60],
        "ltv_cap": None,
        "product_cap": 500_000,
        "secured": False,
        "spread_base": 5.5,
        "variable_default": False,
    },
    "Education Loan": {
        "term_options": [12, 24, 36, 48, 60],
        "ltv_cap": None,
        "product_cap": 800_000,
        "secured": False,
        "spread_base": 3.6,
        "variable_default": False,
    },
}


NCR_DECLINE_CODES = {
    "AFFORDABILITY_FAIL": "NCR_AFF_001",
    "CREDIT_SCORE_FAIL": "NCR_CRE_002",
    "COLLATERAL_SHORTFALL": "NCR_COL_003",
    "DOCUMENTATION_FAIL": "NCR_DOC_004",
    "POLICY_RULE_FAIL": "NCR_POL_005",
}


def month_output_base(year: int, month: int, record_type: str = "loans") -> Path:
    return DATA_DIR / str(year) / f"{month:02d}" / f"{record_type}_{year}_{month:02d}"


def flat_output_base(year: int, record_type: str = "loans") -> Path:
    return DATA_DIR / f"{record_type}_{year}"


def read_table(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def get_input_files(record_type: str, year: int, month: int | None) -> list[Path]:
    candidates: list[Path] = []
    if month is not None:
        monthly_parquet = DATA_DIR / str(year) / f"{month:02d}" / f"{record_type}_{year}_{month:02d}.parquet"
        monthly_csv = DATA_DIR / str(year) / f"{month:02d}" / f"{record_type}_{year}_{month:02d}.csv"
        if monthly_parquet.exists():
            candidates.append(monthly_parquet)
        elif monthly_csv.exists():
            candidates.append(monthly_csv)
        return candidates

    monthly_all = sorted((DATA_DIR / str(year)).glob(f"*/{record_type}_{year}_*.parquet"))
    if not monthly_all:
        monthly_all = sorted((DATA_DIR / str(year)).glob(f"*/{record_type}_{year}_*.csv"))
    if monthly_all:
        return monthly_all

    flat_parquet = DATA_DIR / f"{record_type}_{year}.parquet"
    flat_csv = DATA_DIR / f"{record_type}_{year}.csv"
    if flat_parquet.exists():
        candidates.append(flat_parquet)
    elif flat_csv.exists():
        candidates.append(flat_csv)
    return candidates


def choose_prime_rate(application_date: datetime, year: int) -> float:
    yearly_base = {
        2019: 10.00,
        2020: 8.25,
        2021: 7.00,
        2022: 8.75,
        2023: 11.75,
        2024: 11.75,
        2025: 11.00,
        2026: 10.75,
    }
    base = yearly_base.get(year, 10.50)
    seasonality = 0.15 if application_date.month in [11, 12] else 0.0
    return round(base + np.random.normal(0, 0.10) + seasonality, 2)


def infer_credit_score(customer: dict) -> int:
    annual_income = float(customer.get("annual_income", 300000) or 300000)
    occupation = str(customer.get("occupation", "Unknown"))
    age = 40
    birth_date = customer.get("birth_date")
    if pd.notna(birth_date):
        try:
            age = max(18, datetime.now().year - pd.to_datetime(birth_date).year)
        except Exception:
            age = 40

    score = 620
    if annual_income > 1_000_000:
        score += 70
    elif annual_income > 500_000:
        score += 35
    elif annual_income < 120_000:
        score -= 60

    if occupation in ["Doctor", "Lawyer", "Engineer", "Accountant"]:
        score += 35
    if occupation in ["Unemployed", "Student"]:
        score -= 90

    if age < 23:
        score -= 35
    if age > 35:
        score += 15

    score += int(np.random.normal(0, 35))
    return int(max(300, min(850, score)))


def infer_loan_type(customer: dict, month: int) -> str:
    mix_key = "company" if str(customer.get("customer_type", "Individual")).lower() == "company" else "default"
    if month in [11, 12] and mix_key == "default":
        mix_key = "nov_dec"

    product_mix = REALISM["loan_product_mix"][mix_key]
    options = list(product_mix.keys())
    weights = list(product_mix.values())

    if int(customer.get("cnt_children", 0) or 0) == 0:
        filtered = [(option, weight) for option, weight in zip(options, weights) if option != "Education Loan"]
        options = [option for option, _ in filtered]
        weights = [weight for _, weight in filtered]
    return random.choices(options, weights=weights, k=1)[0]


def loan_application_channel(year: int) -> str:
    channel_mix = REALISM["loan_application_channel_weights"].get(
        str(year),
        REALISM["loan_application_channel_weights"]["2025"],
    )
    return random.choices(list(channel_mix.keys()), weights=list(channel_mix.values()), k=1)[0]


def amortization_payment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    if principal <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0:
        return principal / term_months
    return principal * (r * math.pow(1 + r, term_months)) / (math.pow(1 + r, term_months) - 1)


def principal_from_payment(max_payment: float, annual_rate_pct: float, term_months: int) -> float:
    if max_payment <= 0 or term_months <= 0:
        return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0:
        return max_payment * term_months
    return max_payment * (math.pow(1 + r, term_months) - 1) / (r * math.pow(1 + r, term_months))


def generate_collateral(loan_type: str) -> tuple[str, str, float]:
    if loan_type == "Home Loan":
        value = float(np.random.lognormal(mean=13.2, sigma=0.45))
        return "property", "Residential property", max(250_000.0, min(value, 20_000_000.0))
    if loan_type == "Vehicle Loan":
        value = float(np.random.lognormal(mean=12.0, sigma=0.40))
        return "vehicle", "Motor vehicle", max(80_000.0, min(value, 2_000_000.0))
    if loan_type == "Business Loan":
        choice = random.choice(["equipment", "inventory", "accounts_receivable", "none"])
        if choice == "none":
            return "none", "Unsecured business facility", 0.0
        value = float(np.random.lognormal(mean=13.0, sigma=0.60))
        return choice, choice.replace("_", " ").title(), max(150_000.0, min(value, 25_000_000.0))
    return "none", "Unsecured", 0.0


def affordability_assessment(customer: dict, proposed_installment: float) -> dict:
    assumptions = REALISM["affordability_assumptions"]
    gross_annual_income = float(customer.get("annual_income", 300000) or 300000)
    gross_monthly_income = gross_annual_income / 12.0

    net_ratio = float(np.random.normal(assumptions["net_income_ratio_mean"], assumptions["net_income_ratio_sd"]))
    net_ratio = max(0.55, min(0.90, net_ratio))
    net_monthly_income = gross_monthly_income * net_ratio

    expense_ratio = float(np.random.normal(assumptions["expense_ratio_mean"], assumptions["expense_ratio_sd"]))
    expense_ratio = max(0.20, min(0.75, expense_ratio))
    verified_monthly_expenses = net_monthly_income * expense_ratio

    existing_debt_ratio = float(np.random.normal(assumptions["existing_debt_ratio_mean"], assumptions["existing_debt_ratio_sd"]))
    existing_debt_ratio = max(0.0, min(0.45, existing_debt_ratio))
    existing_monthly_debt = net_monthly_income * existing_debt_ratio

    discretionary_income = net_monthly_income - verified_monthly_expenses - existing_monthly_debt
    max_affordable_installment = max(0.0, discretionary_income * assumptions["installment_buffer_on_discretionary_income"])

    affordability_ratio = 1.0
    if max_affordable_installment > 0:
        affordability_ratio = proposed_installment / max_affordable_installment

    passes = max_affordable_installment >= proposed_installment and discretionary_income > 0

    return {
        "gross_annual_income": round(gross_annual_income, 2),
        "gross_monthly_income": round(gross_monthly_income, 2),
        "net_monthly_income": round(net_monthly_income, 2),
        "verified_monthly_expenses": round(verified_monthly_expenses, 2),
        "existing_monthly_debt": round(existing_monthly_debt, 2),
        "discretionary_income": round(discretionary_income, 2),
        "max_affordable_installment": round(max_affordable_installment, 2),
        "affordability_ratio": round(affordability_ratio, 3),
        "affordability_pass": bool(passes),
    }


def pd_12m_from_score(credit_score: int) -> float:
    x = (credit_score - 620) / 70.0
    pd_12m = 1.0 / (1.0 + math.exp(x))
    return max(0.005, min(0.45, pd_12m))


def lgd_from_collateral(loan_type: str, ltv_ratio: float) -> float:
    if loan_type == "Home Loan":
        base = 0.24
    elif loan_type == "Vehicle Loan":
        base = 0.36
    elif loan_type == "Business Loan":
        base = 0.42
    else:
        base = 0.58

    adjusted = base + max(0.0, ltv_ratio - 0.75) * 0.30
    return max(0.10, min(0.85, adjusted))


def ifrs9_snapshot(credit_score: int, debt_to_income: float, amount_granted: float, loan_type: str, ltv_ratio: float) -> dict:
    stage = 1
    if credit_score < 580 or debt_to_income > 0.42:
        stage = 2
    if credit_score < 500:
        stage = 3

    pd_12m = pd_12m_from_score(credit_score)
    lgd = lgd_from_collateral(loan_type, ltv_ratio)
    ead = max(0.0, amount_granted)
    ecl_12m = ead * pd_12m * lgd

    return {
        "ifrs9_stage_at_origination": stage,
        "ifrs9_pd_12m": round(pd_12m, 4),
        "ifrs9_lgd": round(lgd, 4),
        "ifrs9_ead_origination": round(ead, 2),
        "ifrs9_ecl_12m": round(ecl_12m, 2),
    }


def workflow_timestamps(application_date: datetime, approved: bool, booked: bool) -> dict:
    submitted_at = application_date
    under_review_at = submitted_at + timedelta(hours=random.randint(2, 48))
    decision_at = under_review_at + timedelta(days=random.randint(1, 7))

    if not approved:
        return {
            "workflow_state": "Declined",
            "submitted_at": submitted_at,
            "under_review_at": under_review_at,
            "decision_at": decision_at,
            "booked_at": pd.NaT,
            "disbursed_at": pd.NaT,
        }

    if approved and not booked:
        return {
            "workflow_state": "Withdrawn",
            "submitted_at": submitted_at,
            "under_review_at": under_review_at,
            "decision_at": decision_at,
            "booked_at": pd.NaT,
            "disbursed_at": pd.NaT,
        }

    booked_at = decision_at + timedelta(days=random.randint(1, 3))
    disbursed_at = booked_at + timedelta(days=random.randint(1, 5))
    return {
        "workflow_state": "Booked",
        "submitted_at": submitted_at,
        "under_review_at": under_review_at,
        "decision_at": decision_at,
        "booked_at": booked_at,
        "disbursed_at": disbursed_at,
    }


def ncr_decline(reason_key: str) -> tuple[str, str]:
    code = NCR_DECLINE_CODES[reason_key]
    reason_map = {
        "AFFORDABILITY_FAIL": "Affordability assessment failed",
        "CREDIT_SCORE_FAIL": "Credit score below policy minimum",
        "COLLATERAL_SHORTFALL": "Collateral value insufficient for requested exposure",
        "DOCUMENTATION_FAIL": "Required onboarding documents incomplete",
        "POLICY_RULE_FAIL": "Credit policy rule failed",
    }
    return code, reason_map[reason_key]


def generate_loans(year: int, month: int | None = None, target_records: int | None = None) -> pd.DataFrame:
    random.seed(year if month is None else year * 100 + month)
    np.random.seed(year if month is None else year * 100 + month)
    Faker.seed(year if month is None else year * 100 + month)

    customer_files = get_input_files("customers", year, month)
    account_files = get_input_files("accounts", year, month)
    if not customer_files or not account_files:
        print(f"Missing inputs for year={year}, month={month}. Expected customers and accounts files.")
        return pd.DataFrame()

    customers_df = pd.concat([read_table(p) for p in customer_files], ignore_index=True)
    accounts_df = pd.concat([read_table(p) for p in account_files], ignore_index=True)

    if customers_df.empty or accounts_df.empty:
        print("Input files are empty. No loans generated.")
        return pd.DataFrame()

    customers_df = customers_df.drop_duplicates(subset=["customer_id"])
    accounts_df = accounts_df.drop_duplicates(subset=["account_id"])

    if "date_of_entry" in customers_df.columns:
        customers_df["date_of_entry"] = pd.to_datetime(customers_df["date_of_entry"], errors="coerce")

    eligible = customers_df.merge(accounts_df[["account_id", "customer_id"]], on="customer_id", how="inner")
    if eligible.empty:
        print("No customer-account matches found. No loans generated.")
        return pd.DataFrame()

    if month is not None and "date_of_entry" in eligible.columns:
        eligible = eligible[eligible["date_of_entry"].dt.month.fillna(month) == month]

    if eligible.empty:
        print("No eligible records after month filter. No loans generated.")
        return pd.DataFrame()

    if target_records is None:
        target_records = int(min(max(len(eligible) * 0.45, 3000), 45000))

    eligible = eligible.sample(n=min(len(eligible), target_records), random_state=year)

    rows: list[dict] = []
    id_counter = 1

    for _, row in tqdm(eligible.iterrows(), total=len(eligible), desc=f"Loan Applications {year}{'' if month is None else f'-{month:02d}'}"):
        customer = row.to_dict()
        customer_id = str(customer.get("customer_id"))
        account_id = str(customer.get("account_id"))

        if month is None:
            app_start = datetime(year, 1, 1)
            app_end = datetime(year, 12, 20)
        else:
            app_start = datetime(year, month, 1)
            next_month = month + 1
            if next_month == 13:
                app_end = datetime(year, 12, 28)
            else:
                app_end = datetime(year, next_month, 1) - timedelta(days=3)

        application_date = fake.date_time_between_dates(datetime_start=app_start, datetime_end=app_end)
        channel = loan_application_channel(year)

        credit_score = infer_credit_score(customer)
        loan_type = infer_loan_type(customer, application_date.month)
        product = LOAN_PRODUCTS[loan_type]
        term_months = int(random.choice(product["term_options"]))

        prime_rate = choose_prime_rate(application_date, year)
        spread_adj = 0.0
        if credit_score >= 760:
            spread_adj = -0.8
        elif credit_score >= 700:
            spread_adj = -0.2
        elif credit_score < 580:
            spread_adj = 2.2
        elif credit_score < 640:
            spread_adj = 1.1

        spread = max(0.5, product["spread_base"] + spread_adj + np.random.normal(0, 0.25))
        nominal_rate = round(max(6.5, min(30.0, prime_rate + spread)), 2)
        pricing_basis = "PRIME_PLUS_SPREAD"
        rate_type = "VARIABLE" if product["variable_default"] else "FIXED"

        collateral_type, collateral_description, collateral_value = generate_collateral(loan_type)
        ltv_cap = product["ltv_cap"]
        collateral_cap = float("inf")
        if product["secured"]:
            collateral_cap = collateral_value * (ltv_cap or 0.0)

        annual_income = float(customer.get("annual_income", 300000) or 300000)
        annual_income = max(60_000.0, annual_income)
        requested_amount = max(10_000.0, min(product["product_cap"], annual_income * np.random.uniform(0.25, 2.5)))

        provisional_payment = amortization_payment(requested_amount, nominal_rate, term_months)
        aff = affordability_assessment(customer, provisional_payment)
        affordability_cap = principal_from_payment(aff["max_affordable_installment"], nominal_rate, term_months)

        amount_granted = min(requested_amount, product["product_cap"], affordability_cap, collateral_cap)
        amount_granted = max(0.0, round(amount_granted, 2))

        monthly_installment = round(amortization_payment(amount_granted, nominal_rate, term_months), 2)
        debt_to_income = round(monthly_installment / max(1.0, aff["gross_monthly_income"]), 4)

        decline_priors = REALISM["loan_decline_priors"]
        approval_state = "Approved"
        decline_code = None
        decline_reason = None

        if random.random() < decline_priors["documentation_failure_rate"]:
            approval_state = "Rejected"
            decline_code, decline_reason = ncr_decline("DOCUMENTATION_FAIL")
        elif credit_score < decline_priors["credit_score_hard_decline"]:
            approval_state = "Rejected"
            decline_code, decline_reason = ncr_decline("CREDIT_SCORE_FAIL")
        elif amount_granted <= decline_priors["minimum_bookable_amount"]:
            key = "COLLATERAL_SHORTFALL" if product["secured"] else "AFFORDABILITY_FAIL"
            approval_state = "Rejected"
            decline_code, decline_reason = ncr_decline(key)
        elif not aff["affordability_pass"]:
            approval_state = "Rejected"
            decline_code, decline_reason = ncr_decline("AFFORDABILITY_FAIL")
        elif debt_to_income > decline_priors["max_debt_to_income_for_policy"]:
            approval_state = "Rejected"
            decline_code, decline_reason = ncr_decline("POLICY_RULE_FAIL")
        elif random.random() < decline_priors["manual_policy_decline_rate"]:
            approval_state = "Rejected"
            decline_code, decline_reason = ncr_decline(
                random.choices(
                    list(decline_priors["decline_reason_weights"].keys()),
                    weights=list(decline_priors["decline_reason_weights"].values()),
                    k=1,
                )[0]
            )

        booked = approval_state == "Approved" and random.random() > decline_priors["post_approval_withdrawal_rate"]
        flow = workflow_timestamps(application_date, approved=(approval_state == "Approved"), booked=booked)

        if approval_state != "Approved":
            amount_granted = 0.0
            monthly_installment = 0.0

        apr = round(nominal_rate + 0.6 + (60.0 * 12 / max(1.0, amount_granted + 1.0)), 2) if amount_granted > 0 else nominal_rate

        ltv_ratio = 0.0
        if collateral_value > 0 and amount_granted > 0:
            ltv_ratio = round(amount_granted / collateral_value, 4)

        ifrs9 = ifrs9_snapshot(
            credit_score=credit_score,
            debt_to_income=debt_to_income,
            amount_granted=amount_granted if flow["workflow_state"] == "Booked" else 0.0,
            loan_type=loan_type,
            ltv_ratio=ltv_ratio,
        )

        application_id = make_application_id("LOANAPP", year, application_date.month, id_counter)
        loan_id = make_loan_id(year, application_date.month, id_counter) if flow["workflow_state"] == "Booked" else None
        id_counter += 1

        rows.append(
            {
                "application_id": application_id,
                "loan_id": loan_id,
                "customer_id": customer_id,
                "account_id": account_id,
                "loan_type": loan_type,
                "term_months": term_months,
                "application_channel": channel,
                "application_date": application_date,
                "requested_amount": round(requested_amount, 2),
                "amount_granted": round(amount_granted, 2),
                "monthly_installment": monthly_installment,
                "credit_score": credit_score,
                "debt_to_income": debt_to_income,
                "affordability_pass": aff["affordability_pass"],
                "gross_annual_income": aff["gross_annual_income"],
                "gross_monthly_income": aff["gross_monthly_income"],
                "net_monthly_income": aff["net_monthly_income"],
                "verified_monthly_expenses": aff["verified_monthly_expenses"],
                "existing_monthly_debt": aff["existing_monthly_debt"],
                "discretionary_income": aff["discretionary_income"],
                "max_affordable_installment": aff["max_affordable_installment"],
                "affordability_ratio": aff["affordability_ratio"],
                "pricing_basis": pricing_basis,
                "rate_type": rate_type,
                "prime_rate": prime_rate,
                "spread_bps": round(spread * 100, 2),
                "nominal_rate": nominal_rate,
                "apr": apr,
                "collateral_type": collateral_type,
                "collateral_description": collateral_description,
                "collateral_value": round(collateral_value, 2),
                "ltv_cap": ltv_cap,
                "loan_to_value_ratio": ltv_ratio,
                "application_status": approval_state,
                "application_outcome": flow["workflow_state"],
                "ncr_decline_code": decline_code,
                "ncr_decline_reason": decline_reason,
                "workflow_state": flow["workflow_state"],
                "submitted_at": flow["submitted_at"],
                "under_review_at": flow["under_review_at"],
                "decision_at": flow["decision_at"],
                "booked_at": flow["booked_at"],
                "disbursed_at": flow["disbursed_at"],
                "ifrs9_stage_at_origination": ifrs9["ifrs9_stage_at_origination"],
                "ifrs9_pd_12m": ifrs9["ifrs9_pd_12m"],
                "ifrs9_lgd": ifrs9["ifrs9_lgd"],
                "ifrs9_ead_origination": ifrs9["ifrs9_ead_origination"],
                "ifrs9_ecl_12m": ifrs9["ifrs9_ecl_12m"],
            }
        )

    loans_df = pd.DataFrame(rows)

    if month is not None:
        output_base = month_output_base(year, month, "loans")
    else:
        output_base = flat_output_base(year, "loans")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_parquet = output_base.with_suffix(".parquet")
    output_csv = output_base.with_suffix(".csv")
    output_df = loans_df.drop(columns=[col for col in LOAN_OUTPUT_DROP_COLUMNS if col in loans_df.columns])

    try:
        output_df.to_parquet(output_parquet, index=False)
        print(f"Generated {len(output_df)} loan applications and saved to {output_parquet}")
    except Exception as exc:
        print(f"Parquet export failed ({exc}). Falling back to CSV.")
        output_df.to_csv(output_csv, index=False)
        print(f"Generated {len(output_df)} loan applications and saved to {output_csv}")

    if not loans_df.empty:
        booked_count = int((loans_df["workflow_state"] == "Booked").sum())
        rejected_count = int((loans_df["application_status"] == "Rejected").sum())
        withdrawn_count = int((loans_df["workflow_state"] == "Withdrawn").sum())
        print(f"Booked: {booked_count} | Rejected: {rejected_count} | Withdrawn after approval: {withdrawn_count}")

    return output_df


def run_year(year: int, cadence: str, month: int | None, target_records: int | None) -> None:
    if month is not None:
        if month < 1 or month > 12:
            raise ValueError("month must be between 1 and 12")
        generate_loans(year=year, month=month, target_records=target_records)
        return

    if cadence == "monthly":
        print(f"Generating loan application data for all months in {year}...")
        for m in range(1, 13):
            print(f"\n--- Month {m:02d} ---")
            generate_loans(year=year, month=m, target_records=target_records)
    else:
        generate_loans(year=year, month=None, target_records=target_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate booking-focused loan data")
    parser.add_argument("--year", type=int, required=True, help="Year to generate")
    parser.add_argument("--month", type=int, default=None, help="Optional month 1-12")
    parser.add_argument("--cadence", type=str, choices=["monthly", "yearly"], default="monthly", help="Monthly writes to year/month subfolders")
    parser.add_argument("--target-records", type=int, default=None, help="Optional cap on generated records")
    args = parser.parse_args()

    run_year(year=args.year, cadence=args.cadence, month=args.month, target_records=args.target_records)
