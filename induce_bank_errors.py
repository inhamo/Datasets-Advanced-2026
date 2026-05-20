"""
Post-process banking transaction files to inject normal operational and data-quality errors.

Usage:
  py induce_bank_errors.py
  py induce_bank_errors.py --pattern loan_payment_transactions --operational-rate 0.015
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from commons.bank_errors import OPERATIONAL_FAILURE_REASONS, pick_failure_reason

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"

REASONS, REASON_WEIGHTS = zip(*OPERATIONAL_FAILURE_REASONS)


def _inject_data_errors_vectorized(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Return pipe-delimited error tags per row (empty string if none)."""
    n = len(df)
    tags: list[list[str]] = [[] for _ in range(n)]

    mask = rng.random(n) < 0.012
    if mask.any() and "description" in df.columns:
        desc = df.loc[mask, "description"].astype(str)
        df.loc[mask, "description"] = desc.str.slice(0, 25) + "..."
        for i in np.where(mask)[0]:
            tags[i].append("truncated_description")

    mask = rng.random(n) < 0.004
    if mask.any() and "amount" in df.columns:
        bump = rng.choice([-0.01, 0.01], size=int(mask.sum()))
        df.loc[mask, "amount"] = (df.loc[mask, "amount"].astype(float) + bump).round(2)
        for i in np.where(mask)[0]:
            tags[i].append("amount_rounding_error")

    mask = rng.random(n) < 0.006
    if mask.any() and "transaction_time" in df.columns:
        times = df.loc[mask, "transaction_time"].astype(str).str.split(":", expand=True)
        hours = (times[0].astype(int) + rng.choice([-3, -2, 2, 3, 5], size=int(mask.sum()))) % 24
        hour_str = hours.astype(int).astype(str).str.zfill(2)
        df.loc[mask, "transaction_time"] = hour_str + ":" + times[1] + ":" + times[2]
        for i in np.where(mask)[0]:
            tags[i].append("timestamp_shift")

    failed_mask = df["status"].eq("Failed") if "status" in df.columns else pd.Series(False, index=df.index)
    mask = failed_mask & (rng.random(n) < 0.003)
    if mask.any() and "failure_reason" in df.columns:
        df.loc[mask, "failure_reason"] = np.nan
        for i in np.where(mask)[0]:
            tags[i].append("missing_failure_reason")

    mask = rng.random(n) < 0.002
    if mask.any() and "status" in df.columns:
        flip_to_failed = mask & df["status"].eq("Completed")
        flip_to_ok = mask & df["status"].eq("Failed")
        if flip_to_failed.any():
            df.loc[flip_to_failed, "status"] = "Failed"
            df.loc[flip_to_failed, "failure_reason"] = rng.choice(
                REASONS, size=int(flip_to_failed.sum()), p=REASON_WEIGHTS
            )
        if flip_to_ok.any():
            df.loc[flip_to_ok, "status"] = "Completed"
            df.loc[flip_to_ok, "failure_reason"] = np.nan
        for i in np.where(mask)[0]:
            tags[i].append("status_inconsistency")

    mask = rng.random(n) < 0.002
    id_cols = [c for c in ("transaction_id", "account_id", "customer_id", "loan_id") if c in df.columns]
    if mask.any() and id_cols:
        col = rng.choice(id_cols)
        df.loc[mask, col] = " " + df.loc[mask, col].astype(str) + " "
        for i in np.where(mask)[0]:
            tags[i].append("whitespace_in_id")

    return pd.Series(["|".join(t) for t in tags], index=df.index)


def _process_file(
    path: Path,
    operational_rate: float,
    seed: int,
    dry_run: bool,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if df.empty:
        return {"rows": 0, "operational": 0, "data_errors": 0, "duplicates": 0}

    if "has_data_error" not in df.columns:
        df["has_data_error"] = False
    if "data_error_types" not in df.columns:
        df["data_error_types"] = ""

    operational_count = 0
    if "status" in df.columns:
        completed = df["status"].eq("Completed")
        flip = completed & (rng.random(len(df)) < operational_rate)
        operational_count = int(flip.sum())
        if operational_count:
            df.loc[flip, "status"] = "Failed"
            df.loc[flip, "failure_reason"] = rng.choice(
                REASONS, size=operational_count, p=REASON_WEIGHTS
            )

    new_tags = _inject_data_errors_vectorized(df, rng)
    merged = df["data_error_types"].fillna("").astype(str)
    merged = np.where(
        (merged == "") | merged.isna(),
        new_tags,
        np.where(new_tags == "", merged, merged + "|" + new_tags),
    )
    df["data_error_types"] = merged
    df["has_data_error"] = df["data_error_types"].astype(str).str.len() > 0
    data_error_count = int((new_tags != "").sum())

    duplicate_count = 0
    if len(df) > 1 and rng.random() < 0.35:
        dup_idx = int(rng.integers(0, len(df)))
        clone = df.iloc[[dup_idx]].copy()
        prev = str(clone.iloc[0].get("data_error_types") or "")
        tags = [t for t in prev.split("|") if t]
        tags.append("duplicate_transaction_id")
        clone["data_error_types"] = "|".join(sorted(set(tags)))
        clone["has_data_error"] = True
        df = pd.concat([df, clone], ignore_index=True)
        duplicate_count = 1

    if not dry_run:
        suffix = path.suffix.lower()
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=suffix,
            dir=path.parent,
            newline="",
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if suffix == ".parquet":
                df.to_parquet(tmp_path, index=False)
            else:
                df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    return {
        "rows": int(len(df)),
        "operational": operational_count,
        "data_errors": data_error_count,
        "duplicates": duplicate_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Induce normal banking errors in transaction files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pattern", default="loan_payment_transactions")
    parser.add_argument(
        "--operational-rate",
        type=float,
        default=0.015,
        help="Share of Completed rows flipped to Failed (default 1.5%%).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process files that already contain injected error columns.",
    )
    args = parser.parse_args()

    files = sorted(args.data_dir.rglob(f"{args.pattern}*.csv")) + sorted(
        args.data_dir.rglob(f"{args.pattern}*.parquet")
    )
    if not files:
        print(f"No files matching '{args.pattern}' under {args.data_dir}")
        return

    totals = {"files": 0, "rows": 0, "operational": 0, "data_errors": 0, "duplicates": 0}
    for i, path in enumerate(files):
        if not args.force and path.suffix.lower() == ".csv":
            try:
                header = pd.read_csv(path, nrows=0)
                if "has_data_error" in header.columns:
                    print(f"{path.name}: skipped (already processed)")
                    continue
            except OSError:
                pass
        stats = _process_file(path, args.operational_rate, args.seed + i, args.dry_run)
        totals["files"] += 1
        for k in ("rows", "operational", "data_errors", "duplicates"):
            totals[k] += stats[k]
        print(f"{path.name}: +{stats['operational']} failed, +{stats['data_errors']} data errors, rows={stats['rows']}")

    print(json.dumps(totals, indent=2))
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
