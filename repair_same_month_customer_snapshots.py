from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
BANKING = ROOT / "banking_data"
REPORT_DIR = ROOT / "migration_artifacts" / "same_month_customer_snapshots"


def ym(month_dir: Path) -> int:
    return int(month_dir.parent.name) * 100 + int(month_dir.name)


def month_dirs() -> list[Path]:
    return sorted((p for p in BANKING.glob("20*/[0-1][0-9]") if p.is_dir()), key=ym)


def schema_names(path: Path) -> list[str]:
    return pq.read_schema(path).names


def clean_id(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def load_customer_history() -> pd.DataFrame:
    frames = []
    for md in month_dirs():
        year, month = md.parent.name, md.name
        path = md / f"customers_{year}_{month}.parquet"
        if not path.exists() or "customer_id" not in schema_names(path):
            continue
        frame = pd.read_parquet(path)
        frame["customer_id"] = frame["customer_id"].map(clean_id)
        frame["_source_year"] = int(year)
        frame["_source_month"] = int(month)
        frame["_source_ym"] = int(year) * 100 + int(month)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No customer files found.")
    return pd.concat(frames, ignore_index=True)


def repair() -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_customer_history()
    repairs = []

    for md in month_dirs():
        year, month = md.parent.name, md.name
        target_ym = int(year) * 100 + int(month)
        accounts_path = md / f"accounts_{year}_{month}.parquet"
        customers_path = md / f"customers_{year}_{month}.parquet"
        if not accounts_path.exists() or not customers_path.exists():
            continue
        if "customer_id" not in schema_names(accounts_path) or "customer_id" not in schema_names(customers_path):
            continue

        accounts = pd.read_parquet(accounts_path, columns=["account_id", "customer_id"])
        customers = pd.read_parquet(customers_path)
        accounts["customer_id"] = accounts["customer_id"].map(clean_id)
        accounts["account_id"] = accounts["account_id"].map(clean_id)
        customers["customer_id"] = customers["customer_id"].map(clean_id)

        missing = sorted(set(accounts["customer_id"]) - set(customers["customer_id"]))
        missing = [cid for cid in missing if cid]
        if not missing:
            continue

        available = history[history["_source_ym"] <= target_ym].sort_values("_source_ym")
        latest = available.drop_duplicates("customer_id", keep="last").set_index("customer_id")
        found = [cid for cid in missing if cid in latest.index]
        not_found = [cid for cid in missing if cid not in latest.index]
        if not_found:
            repairs.extend(
                {
                    "year": year,
                    "month": month,
                    "customer_id": cid,
                    "action": "unresolved_no_prior_customer_row",
                    "source_year": None,
                    "source_month": None,
                }
                for cid in not_found
            )

        if found:
            additions = latest.loc[found].reset_index()
            source_cols = ["_source_year", "_source_month", "_source_ym"]
            audit = additions[["customer_id", "_source_year", "_source_month"]].copy()
            for col in source_cols:
                if col in additions.columns:
                    additions = additions.drop(columns=col)
            for col in customers.columns:
                if col not in additions.columns:
                    additions[col] = None
            additions = additions[customers.columns]
            combined = pd.concat([customers, additions], ignore_index=True)
            combined = combined.drop_duplicates("customer_id", keep="last")
            combined.to_parquet(customers_path, index=False)

            for _, row in audit.iterrows():
                repairs.append(
                    {
                        "year": year,
                        "month": month,
                        "customer_id": row["customer_id"],
                        "action": "copied_prior_customer_row_into_month",
                        "source_year": int(row["_source_year"]),
                        "source_month": int(row["_source_month"]),
                    }
                )

            # Make newly added rows available to later months in this repair run.
            additions_for_history = additions.copy()
            additions_for_history["_source_year"] = int(year)
            additions_for_history["_source_month"] = int(month)
            additions_for_history["_source_ym"] = target_ym
            history = pd.concat([history, additions_for_history], ignore_index=True)

    report = pd.DataFrame(repairs)
    report.to_csv(REPORT_DIR / "same_month_customer_backfills.csv", index=False)
    return report


def verify() -> pd.DataFrame:
    rows = []
    for md in month_dirs():
        year, month = md.parent.name, md.name
        accounts_path = md / f"accounts_{year}_{month}.parquet"
        customers_path = md / f"customers_{year}_{month}.parquet"
        if not accounts_path.exists() or not customers_path.exists():
            continue
        if "customer_id" not in schema_names(accounts_path) or "customer_id" not in schema_names(customers_path):
            rows.append({"year": year, "month": month, "missing_customer_ids": "schema_issue", "sample": ""})
            continue
        accounts = pd.read_parquet(accounts_path, columns=["customer_id"])
        customers = pd.read_parquet(customers_path, columns=["customer_id"])
        accounts["customer_id"] = accounts["customer_id"].map(clean_id)
        customers["customer_id"] = customers["customer_id"].map(clean_id)
        missing = sorted(set(accounts["customer_id"]) - set(customers["customer_id"]))
        missing = [cid for cid in missing if cid]
        if missing:
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "missing_customer_ids": len(missing),
                    "sample": ";".join(missing[:10]),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "same_month_customer_verification.csv", index=False)
    return result


def main() -> None:
    report = repair()
    result = verify()
    print(f"Backfilled rows: {len(report[report['action'].eq('copied_prior_customer_row_into_month')]) if not report.empty else 0}")
    if result.empty:
        print("Same-month accounts -> customers integrity: PASS")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
