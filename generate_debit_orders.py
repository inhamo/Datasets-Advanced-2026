from __future__ import annotations

import argparse
import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm


SA_CREDITORS = {
    "utility bill": {
        "Eskom": {"abbrev": "ESKOM", "creditor_id": "ESK001"},
        "City Power": {"abbrev": "CITYPWR", "creditor_id": "CTP002"},
        "Rand Water": {"abbrev": "RANDWATER", "creditor_id": "RWA003"},
    },
    "insurance premium": {
        "Discovery Life": {"abbrev": "DISCOVERY", "creditor_id": "DSC100"},
        "Old Mutual": {"abbrev": "OLDMUTUAL", "creditor_id": "OM101"},
        "Sanlam": {"abbrev": "SANLAM", "creditor_id": "SNL102"},
    },
    "subscription": {
        "DSTV": {"abbrev": "MULTICHOICE", "creditor_id": "DTV200"},
        "Netflix": {"abbrev": "NETFLIX", "creditor_id": "NFX201"},
        "Showmax": {"abbrev": "SHOWMAX", "creditor_id": "SHM202"},
    },
    "school fees": {
        "Crawford College": {"abbrev": "CRAWFORD", "creditor_id": "CRW300"},
        "Reddam House": {"abbrev": "REDDAM", "creditor_id": "RDM301"},
    },
}


def output_base(base_path: str, year: int, month: int | None, record_type: str) -> str:
    if month is None:
        return os.path.join(base_path, f"{record_type}_{year}")
    month_dir = os.path.join(base_path, str(year), f"{month:02d}")
    return os.path.join(month_dir, f"{record_type}_{year}_{month:02d}")


def load_parquet_or_csv(path_without_ext: str) -> pd.DataFrame:
    parquet_path = f"{path_without_ext}.parquet"
    csv_path = f"{path_without_ext}.csv"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(parquet_path)


def resolve_input(base_path: str, year: int, month: int | None, record_type: str) -> str:
    if month is not None:
        monthly = output_base(base_path, year, month, record_type)
        if os.path.exists(f"{monthly}.parquet") or os.path.exists(f"{monthly}.csv"):
            return monthly
    flat = output_base(base_path, year, None, record_type)
    if os.path.exists(f"{flat}.parquet") or os.path.exists(f"{flat}.csv"):
        return flat
    return output_base(base_path, year, month, record_type) if month is not None else flat


def choose_debit_type(customer_type: str, is_employed: bool, has_default: bool) -> str:
    if customer_type == "Company":
        types = [
            "Payroll",
            "Supplier Payment",
            "Business Loan Repayment",
            "Office Lease",
            "Utility Bill",
            "Insurance Premium",
            "Software Subscription",
        ]
    else:
        types = [
            "Salary Payment",
            "Utility Bill",
            "Loan Repayment",
            "Subscription",
            "Insurance Premium",
            "Mortgage",
            "School Fees",
            "Credit Card Payment",
            "Membership Fee",
            "Donation",
        ]

    weights = {t: 1.0 for t in types}
    if not is_employed and customer_type != "Company":
        for t, factor in {
            "Loan Repayment": 0.6,
            "Mortgage": 0.6,
            "Utility Bill": 1.25,
            "Subscription": 1.2,
        }.items():
            if t in weights:
                weights[t] *= factor

    if has_default:
        for t in ["Loan Repayment", "Business Loan Repayment"]:
            if t in weights:
                weights[t] *= 0.45

    options = list(weights.keys())
    probs = np.array([weights[k] for k in options], dtype=float)
    probs = probs / probs.sum()
    return str(np.random.choice(options, p=probs))


def amount_for_type(debit_type: str, is_business: bool, is_employed: bool, has_default: bool) -> float:
    dt = debit_type.lower()
    if dt in ["salary payment", "payroll"]:
        amount = max(np.random.normal(25000 if is_business else 18000, 7000), 4000)
        if not is_employed and not is_business:
            amount *= 0.55
    elif dt in ["mortgage", "business loan repayment", "loan repayment"]:
        amount = max(np.random.normal(9000, 2800), 1800)
        if not is_employed or has_default:
            amount *= 0.72
    elif dt in ["utility bill", "software subscription", "subscription", "membership fee"]:
        amount = max(np.random.normal(1300, 500), 150)
    elif dt in ["supplier payment", "office lease"]:
        amount = max(np.random.normal(30000, 13000), 5000)
    else:
        amount = max(np.random.exponential(2200), 100)
    return round(float(amount), 2)


def generate_for_month(year: int, month: int, base_path: str = "banking_data") -> pd.DataFrame:
    seed = int.from_bytes(os.urandom(4), byteorder="big")
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    try:
        fake = Faker("en_ZA")
    except Exception:
        fake = Faker()

    customer_base = resolve_input(base_path, year, month, "customers")
    account_base = resolve_input(base_path, year, month, "accounts")
    try:
        customers_df = load_parquet_or_csv(customer_base)
        accounts_df = load_parquet_or_csv(account_base)
    except FileNotFoundError:
        print(f"Input missing for {year}-{month:02d}. Expected customers/accounts for this period.")
        return pd.DataFrame()

    if customers_df.empty or accounts_df.empty:
        print(f"No customers/accounts rows for {year}-{month:02d}.")
        return pd.DataFrame()

    customers_df = customers_df.drop_duplicates(subset=["customer_id"]).copy()
    accounts_df = accounts_df.drop_duplicates(subset=["account_id"]).copy()
    customers = customers_df.set_index("customer_id").to_dict("index")

    account_status_col = "account_status" if "account_status" in accounts_df.columns else None
    if account_status_col:
        accounts_df = accounts_df[~accounts_df[account_status_col].astype(str).str.lower().isin(["closed", "frozen"])].copy()

    if accounts_df.empty:
        print(f"No active accounts for {year}-{month:02d}.")
        return pd.DataFrame()

    # Keep volume modest but realistic.
    target_count = max(300, min(4500, int(len(accounts_df) * random.uniform(0.35, 0.7))))
    sampled_accounts = accounts_df.sample(target_count, replace=True, random_state=seed).reset_index(drop=True)

    frequencies = ["Monthly", "Weekly", "Quarterly", "Annually"]
    freq_weights = [0.72, 0.14, 0.1, 0.04]
    statuses = ["Active", "Suspended", "Cancelled"]

    debit_orders: list[dict] = []
    seen_pairs: set[str] = set()

    period_start = pd.Timestamp(year=year, month=month, day=1)
    period_end = (period_start + pd.offsets.MonthEnd(0)).normalize()

    for i in tqdm(range(len(sampled_accounts)), desc=f"Debit orders {year}-{month:02d}"):
        account = sampled_accounts.iloc[i]
        customer_id = account.get("customer_id")
        account_id = account.get("account_id")
        if pd.isna(customer_id) or pd.isna(account_id) or customer_id not in customers:
            continue

        customer = customers[customer_id]
        customer_type = str(customer.get("customer_type", "Individual"))
        is_business = customer_type == "Company"

        # In the absence of dedicated employment/default tables, use available customer flags conservatively.
        is_employed = not str(customer.get("occupation", "")).lower().startswith("unemployed")
        has_default = bool(customer.get("loan_default_flag", False))

        debit_type = choose_debit_type(customer_type, is_employed, has_default)
        pair_key = f"{customer_id}|{debit_type}"
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        amount = amount_for_type(debit_type, is_business, is_employed, has_default)
        frequency = str(np.random.choice(frequencies, p=freq_weights))

        # Status realism without over-cancelling portfolio.
        if has_default and debit_type in ["Loan Repayment", "Business Loan Repayment"]:
            status_weights = [0.4, 0.3, 0.3]
        elif not is_employed and not is_business:
            status_weights = [0.62, 0.2, 0.18]
        else:
            status_weights = [0.84, 0.09, 0.07]
        status = str(np.random.choice(statuses, p=status_weights))

        opening_date = pd.to_datetime(account.get("opening_date"), errors="coerce")
        if pd.isna(opening_date):
            opening_date = period_start - timedelta(days=90)
        start_lower = max(period_start - pd.offsets.DateOffset(years=2), opening_date)
        start_date = pd.Timestamp(fake.date_between_dates(date_start=start_lower, date_end=period_end))

        end_date = pd.NaT
        cancellation_date = pd.NaT
        cancellation_reason = None
        suspension_date = pd.NaT
        suspension_reason = None
        suspension_initiated_by = None
        can_be_reactivated = True

        if status == "Suspended":
            suspension_date = pd.Timestamp(fake.date_between_dates(date_start=start_date, date_end=period_end))
            suspension_reason = str(np.random.choice(["insufficient_funds", "customer_request", "fraud_suspected", "dispute"]))
            suspension_initiated_by = str(np.random.choice(["customer", "bank", "creditor", "system"]))
        elif status == "Cancelled":
            cancellation_date = pd.Timestamp(fake.date_between_dates(date_start=start_date, date_end=period_end))
            end_date = cancellation_date
            cancellation_reason = str(np.random.choice(["customer_request", "contract_ended", "account_closed", "creditor_request"]))
            can_be_reactivated = cancellation_reason != "account_closed"
        elif random.random() < 0.16:
            end_date = pd.Timestamp(fake.date_between_dates(date_start=start_date, date_end=period_end))

        notification_required = debit_type in ["Utility Bill", "Insurance Premium", "Subscription", "School Fees"]
        notification_days_before = int(np.random.randint(5, 11)) if notification_required else 0
        notification_method = str(np.random.choice(["sms", "email", "app_notification", "none"])) if notification_required else "none"

        account_to = None
        beneficiary_account_number = None
        beneficiary_branch_code = None
        beneficiary_bank_name = None
        beneficiary_account_type = None
        beneficiary_name = None
        creditor_id = None
        linked_account_internal = None
        linked_loan_id = None
        linked_policy_number = None
        linked_subscription_id = None

        # 65% internal, 35% external.
        if random.random() < 0.65:
            pool = accounts_df[accounts_df["account_id"] != account_id]
            if pool.empty:
                continue
            target = pool.sample(1, random_state=seed + i).iloc[0]
            linked_account_internal = target["account_id"]
            account_to = linked_account_internal
            description = f"Internal transfer - {debit_type}"
        else:
            creditor_bucket = SA_CREDITORS.get(debit_type.lower())
            if creditor_bucket:
                creditor_name = random.choice(list(creditor_bucket.keys()))
                creditor_info = creditor_bucket[creditor_name]
                beneficiary_name = creditor_name
                creditor_id = creditor_info["creditor_id"]
                description = f"{debit_type} - {creditor_name}"
                if debit_type == "Insurance Premium":
                    linked_policy_number = f"POL-{year}{month:02d}-{fake.random_int(1000, 9999)}"
                if debit_type == "Subscription":
                    linked_subscription_id = f"SUB-{creditor_info['abbrev']}-{fake.random_int(100, 999)}"
            else:
                beneficiary_name = fake.company()
                description = f"{debit_type} - {beneficiary_name}"

            beneficiary_account_number = str(fake.random_int(1000000000, 9999999999))
            beneficiary_branch_code = str(fake.random_int(100000, 999999))
            beneficiary_bank_name = str(np.random.choice(["Standard Bank", "ABSA", "Nedbank", "FNB", "Capitec"]))
            beneficiary_account_type = str(np.random.choice(["savings", "current", "transmission"]))
            account_to = beneficiary_account_number

        debit_orders.append(
            {
                "debit_order_id": f"DBT{year}{month:02d}{str(len(debit_orders) + 1).zfill(7)}",
                "account_id": account_id,
                "customer_id": customer_id,
                "debit_order_type": debit_type,
                "amount": amount,
                "frequency": frequency,
                "collection_day": int(start_date.day),
                "is_fixed_amount": debit_type not in ["Utility Bill"],
                "start_date": start_date,
                "end_date": end_date,
                "status": status,
                "suspension_date": suspension_date,
                "suspension_reason": suspension_reason,
                "suspension_initiated_by": suspension_initiated_by,
                "cancellation_date": cancellation_date,
                "cancellation_reason": cancellation_reason,
                "can_be_reactivated": can_be_reactivated,
                "notification_required": notification_required,
                "notification_days_before": notification_days_before,
                "notification_method": notification_method,
                "account_to": account_to,
                "beneficiary_account_number": beneficiary_account_number,
                "beneficiary_branch_code": beneficiary_branch_code,
                "beneficiary_bank_name": beneficiary_bank_name,
                "beneficiary_account_type": beneficiary_account_type,
                "beneficiary_name": beneficiary_name,
                "creditor_id": creditor_id,
                "linked_loan_id": linked_loan_id,
                "linked_policy_number": linked_policy_number,
                "linked_subscription_id": linked_subscription_id,
                "linked_account_internal": linked_account_internal,
                "description": description,
                "created_in_year": year,
                "created_in_month": month,
                "record_last_updated_at": pd.Timestamp.now(),
            }
        )

    result = pd.DataFrame(debit_orders)
    out_base = output_base(base_path, year, month, "debit_orders")
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    out_file = f"{out_base}.parquet"
    try:
        result.to_parquet(out_file, index=False)
    except Exception:
        out_file = f"{out_base}.csv"
        result.to_csv(out_file, index=False)

    print(f"Generated {len(result)} debit orders for {year}-{month:02d}")
    print(f"Saved to {out_file}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic debit orders in monthly folders")
    parser.add_argument("--year", type=int, required=True, help="Year for generation")
    parser.add_argument("--month", type=int, help="Month 1-12. Omit to generate all months")
    args = parser.parse_args()

    if args.month is not None and (args.month < 1 or args.month > 12):
        raise ValueError("month must be in 1..12")

    if args.month is None:
        for m in range(1, 13):
            generate_for_month(args.year, m)
    else:
        generate_for_month(args.year, args.month)


if __name__ == "__main__":
    main()
