from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
BANKING = ROOT / "banking_data"
REPORT_DIR = ROOT / "migration_artifacts" / "global_customer_uniqueness"


def month_dirs() -> list[Path]:
    return sorted(
        (p for p in BANKING.glob("20*/[0-1][0-9]") if p.is_dir()),
        key=lambda p: (int(p.parent.name), int(p.name)),
    )


def schema_names(path: Path) -> list[str]:
    return pq.read_schema(path).names


def clean_id(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def repair_customers() -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    removed_rows = []
    for md in month_dirs():
        year, month = md.parent.name, md.name
        path = md / f"customers_{year}_{month}.parquet"
        if not path.exists() or "customer_id" not in schema_names(path):
            continue
        frame = pd.read_parquet(path)
        frame["customer_id"] = frame["customer_id"].map(clean_id)

        blank_mask = frame["customer_id"].eq("")
        duplicate_within_mask = frame.duplicated("customer_id", keep="first") & ~blank_mask
        duplicate_global_mask = frame["customer_id"].isin(seen) & ~blank_mask
        remove_mask = blank_mask | duplicate_within_mask | duplicate_global_mask

        if remove_mask.any():
            removed = frame.loc[remove_mask, ["customer_id"]].copy()
            removed["year"] = year
            removed["month"] = month
            removed["reason"] = "blank_customer_id"
            removed.loc[duplicate_within_mask.loc[remove_mask].values, "reason"] = "duplicate_within_file"
            removed.loc[duplicate_global_mask.loc[remove_mask].values, "reason"] = "duplicate_across_month_files"
            removed_rows.append(removed)
            frame = frame.loc[~remove_mask].copy()
            frame.to_parquet(path, index=False)

        seen.update(cid for cid in frame["customer_id"] if cid)

    report = pd.concat(removed_rows, ignore_index=True) if removed_rows else pd.DataFrame(columns=["customer_id", "year", "month", "reason"])
    report.to_csv(REPORT_DIR / "removed_duplicate_customer_rows.csv", index=False)
    return report


def verify() -> dict[str, object]:
    customer_seen: dict[str, str] = {}
    duplicate_customer_ids: dict[str, list[str]] = {}
    blank_customer_rows = 0
    account_blank_customer_rows = 0
    account_customer_ids: set[str] = set()
    all_customer_ids: set[str] = set()

    for md in month_dirs():
        year, month = md.parent.name, md.name
        cp = md / f"customers_{year}_{month}.parquet"
        ap = md / f"accounts_{year}_{month}.parquet"
        if cp.exists() and "customer_id" in schema_names(cp):
            customers = pd.read_parquet(cp, columns=["customer_id"])
            customers["customer_id"] = customers["customer_id"].map(clean_id)
            blank_customer_rows += int(customers["customer_id"].eq("").sum())
            for cid in customers["customer_id"]:
                if not cid:
                    continue
                all_customer_ids.add(cid)
                place = f"{year}/{month}"
                if cid in customer_seen:
                    duplicate_customer_ids.setdefault(cid, [customer_seen[cid]]).append(place)
                else:
                    customer_seen[cid] = place
        if ap.exists() and "customer_id" in schema_names(ap):
            accounts = pd.read_parquet(ap, columns=["account_id", "customer_id"])
            accounts["customer_id"] = accounts["customer_id"].map(clean_id)
            account_blank_customer_rows += int(accounts["customer_id"].eq("").sum())
            account_customer_ids.update(cid for cid in accounts["customer_id"] if cid)

    missing_account_customer_ids = sorted(account_customer_ids - all_customer_ids)
    result = {
        "unique_customer_ids": len(all_customer_ids),
        "duplicate_customer_ids": len(duplicate_customer_ids),
        "blank_customer_rows": blank_customer_rows,
        "account_blank_customer_rows": account_blank_customer_rows,
        "account_customer_ids_missing_from_global_customers": len(missing_account_customer_ids),
        "sample_duplicate_customer_ids": list(duplicate_customer_ids)[:20],
        "sample_missing_account_customer_ids": missing_account_customer_ids[:20],
    }
    pd.Series(result, dtype="object").to_json(REPORT_DIR / "global_customer_integrity_summary.json", indent=2)
    return result


def main() -> None:
    removed = repair_customers()
    result = verify()
    print(f"Removed duplicate/blank customer rows: {len(removed)}")
    print(result)


if __name__ == "__main__":
    main()
