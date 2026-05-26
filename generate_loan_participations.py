from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"
KEYSTONE_BANK = "Keystone Retail Bank"
OUTGOING_PARTNER_BANKS = ["Absa", "Nedbank"]
INCOMING_ORIGINATORS = ["Standard Bank", "First National Bank", "Capitec"]


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def pick_candidate_loans(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df.copy()
    if "application_status" in candidates.columns:
        candidates = candidates[candidates["application_status"].astype(str).str.lower().eq("approved")]
    if "amount_granted" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["amount_granted"], errors="coerce").fillna(0) >= 100_000]
    if candidates.empty:
        return df.head(0)
    return candidates.sort_values(["loan_type", "loan_id"]).head(12)


def make_outgoing_rows(year: int, month: int, loans: pd.DataFrame, rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = pick_candidate_loans(loans)
    if candidates.empty:
        return rows

    picked = candidates.sample(n=min(2, len(candidates)), random_state=year * 100 + month)
    for idx, (_, loan) in enumerate(picked.iterrows(), start=1):
        partner_bank = OUTGOING_PARTNER_BANKS[(year + month + idx) % len(OUTGOING_PARTNER_BANKS)]
        pct = rng.choice([0.15, 0.20, 0.25, 0.30])
        amount_granted = float(clean_value(loan.get("amount_granted")) or 0.0)
        effective_date = clean_value(loan.get("booked_at")) or clean_value(loan.get("disbursed_at")) or clean_value(loan.get("decision_at"))
        if effective_date is None:
            effective_date = f"{year}-{month:02d}-15"

        rows.append(
            {
                "participation_id": f"LP-{year}{month:02d}-OUT-{idx:03d}",
                "loan_id": clean_value(loan.get("loan_id")),
                "external_loan_reference": None,
                "customer_id": clean_value(loan.get("customer_id")),
                "account_id": clean_value(loan.get("account_id")),
                "participation_direction": "outgoing_participation",
                "originating_bank": KEYSTONE_BANK,
                "servicing_bank": KEYSTONE_BANK,
                "participant_bank": partner_bank,
                "participant_role": "funding_participant",
                "loan_type": clean_value(loan.get("loan_type")),
                "effective_date": str(pd.to_datetime(effective_date).date()),
                "participation_pct": pct,
                "participation_amount": round(amount_granted * pct, 2),
                "retained_pct": round(1 - pct, 2),
                "risk_share_type": "pro_rata",
                "servicing_fee_bps": rng.choice([15, 20, 25, 30]),
                "status": "active",
                "notes": f"{partner_bank} participates in a small share of a Keystone-originated loan; Keystone remains servicer.",
            }
        )
    return rows


def make_incoming_rows(year: int, month: int, rng: random.Random) -> list[dict[str, Any]]:
    # Keystone occasionally takes a small participation in loans originated by other banks.
    if rng.random() > 0.65:
        return []

    count = 1 if rng.random() < 0.85 else 2
    rows: list[dict[str, Any]] = []
    for idx in range(1, count + 1):
        originator = rng.choice(INCOMING_ORIGINATORS)
        loan_type = rng.choice(["Business Loan", "Vehicle Loan", "Home Loan"])
        amount = rng.randint(350_000, 4_500_000)
        pct = rng.choice([0.10, 0.15, 0.20])
        rows.append(
            {
                "participation_id": f"LP-{year}{month:02d}-IN-{idx:03d}",
                "loan_id": None,
                "external_loan_reference": f"EXT-{originator[:3].upper().replace(' ', '')}-{year}{month:02d}-{idx:04d}",
                "customer_id": None,
                "account_id": None,
                "participation_direction": "incoming_participation",
                "originating_bank": originator,
                "servicing_bank": originator,
                "participant_bank": KEYSTONE_BANK,
                "participant_role": "funding_participant",
                "loan_type": loan_type,
                "effective_date": f"{year}-{month:02d}-{rng.randint(3, 24):02d}",
                "participation_pct": pct,
                "participation_amount": round(amount * pct, 2),
                "retained_pct": None,
                "risk_share_type": "pro_rata",
                "servicing_fee_bps": rng.choice([10, 15, 20]),
                "status": "active",
                "notes": f"Keystone participates in a small share of a loan originated and serviced by {originator}.",
            }
        )
    return rows


def generate_month(year: int, month: int) -> int:
    month_dir = BANKING_DIR / str(year) / f"{month:02d}"
    loans_path = month_dir / f"loans_{year}_{month:02d}.parquet"
    if not loans_path.exists():
        return 0

    rng = random.Random(20260526 + year * 100 + month)
    loans = pd.read_parquet(loans_path)
    rows = make_outgoing_rows(year, month, loans, rng)
    rows.extend(make_incoming_rows(year, month, rng))

    out_path = month_dir / f"loan_participations_{year}_{month:02d}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sparse bank-to-bank loan participation datasets.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    total = 0
    months = 0
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            count = generate_month(year, month)
            if count:
                months += 1
                total += count

    print(f"Generated {total} loan participation rows across {months} months.")


if __name__ == "__main__":
    main()
