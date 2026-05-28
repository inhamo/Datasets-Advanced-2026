"""Normalize company annual_income values in monthly customer parquet files.

Company annual_turnover can be large, but annual_income should not mirror
turnover. For company customers we model annual_income as the income of the
primary owner/director/authorised representative used in affordability-style
analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"


INCOME_BANDS = {
    "small": [
        (72_000, 180_000, 0.45),
        (180_000, 360_000, 0.42),
        (360_000, 650_000, 0.13),
    ],
    "medium": [
        (180_000, 420_000, 0.36),
        (420_000, 900_000, 0.49),
        (900_000, 1_400_000, 0.15),
    ],
    "large": [
        (360_000, 800_000, 0.28),
        (800_000, 1_500_000, 0.52),
        (1_500_000, 2_100_000, 0.20),
    ],
}


def stable_rng(customer_id: str, year: int, month: int) -> random.Random:
    seed_text = f"{customer_id}|{year}|{month}|company-income-v2"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def choose_band(size: str, rng: random.Random) -> tuple[int, int]:
    bands = INCOME_BANDS.get(size.lower(), INCOME_BANDS["small"])
    marker = rng.random()
    running = 0.0
    for low, high, weight in bands:
        running += weight
        if marker <= running:
            return low, high
    low, high, _ = bands[-1]
    return low, high


def representative_income(customer_id: str, company_size: str, year: int, month: int) -> int:
    rng = stable_rng(customer_id, year, month)
    low, high = choose_band(str(company_size or "small"), rng)
    mode = low + int((high - low) * rng.uniform(0.35, 0.65))
    income = int(round(rng.triangular(low, high, mode), -2))
    return max(0, income)


def normalize_file(path: Path) -> int:
    year = int(path.parts[-3])
    month = int(path.parts[-2])
    df = pd.read_parquet(path)
    required = {"customer_id", "customer_type", "company_size", "annual_income"}
    if df.empty or not required.issubset(df.columns):
        return 0

    mask = df["customer_type"].astype(str).str.lower().eq("company")
    if not mask.any():
        return 0

    df.loc[mask, "annual_income"] = [
        representative_income(row.customer_id, row.company_size, year, month)
        for row in df.loc[mask, ["customer_id", "company_size"]].itertuples(index=False)
    ]
    df.to_parquet(path, index=False)
    return int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize company annual_income in customer parquet files.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    total = 0
    for path in sorted(BANKING_DIR.glob("20*/??/customers_*.parquet")):
        year = int(path.parts[-3])
        if args.start_year <= year <= args.end_year:
            total += normalize_file(path)
    print(f"Normalized annual_income for {total:,} company customer rows.")


if __name__ == "__main__":
    main()
