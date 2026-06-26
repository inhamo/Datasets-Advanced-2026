"""Repair joint-account links and early marketing outputs.

This script fixes two source-data issues:

1. ``accounts.linked_joint_accounts`` still contained pre-migration customer
   IDs, while ``signatories_json`` and ``account_signatories`` already had the
   repaired customer IDs.
2. Marketing campaign folders existed before the bank had six months of
   operating history. Campaign activity should start from July 2019 onward.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
BANKING = ROOT / "banking_data"
ARTIFACTS = ROOT / "migration_artifacts" / "joint_links_and_marketing_start"
FIRST_MARKETING_DATE = date(2019, 7, 1)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _linked_ids_from_signatories(raw: Any, primary_customer_id: str) -> str:
    text = _clean(raw)
    if not text:
        return ""
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(rows, list):
        return ""

    linked: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        customer_id = _clean(row.get("customer_id"))
        role = _clean(row.get("signatory_role")).lower()
        is_active = row.get("is_active", True)
        if not customer_id or customer_id == primary_customer_id:
            continue
        if role != "joint_holder":
            continue
        if isinstance(is_active, str) and is_active.lower() in {"false", "0", "no"}:
            continue
        if is_active is False:
            continue
        linked.append(customer_id)

    return ";".join(dict.fromkeys(linked))


def repair_account_links() -> dict[str, int]:
    files_scanned = 0
    files_changed = 0
    rows_changed = 0
    stale_ids_replaced = 0

    examples: list[dict[str, str]] = []

    for path in sorted(BANKING.glob("*/*/accounts_*.parquet")):
        files_scanned += 1
        frame = pd.read_parquet(path)
        if "linked_joint_accounts" not in frame.columns or "signatories_json" not in frame.columns:
            continue

        original = frame["linked_joint_accounts"].map(_clean)
        repaired = frame.apply(
            lambda row: _linked_ids_from_signatories(
                row.get("signatories_json"),
                _clean(row.get("customer_id")),
            ),
            axis=1,
        )

        mask = original != repaired
        if not mask.any():
            continue

        changed = int(mask.sum())
        rows_changed += changed
        stale_ids_replaced += sum(
            len([part for part in old.split(";") if part])
            for old in original[mask]
        )

        for _, row in frame.loc[mask].head(10).iterrows():
            examples.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "account_id": _clean(row.get("account_id")),
                    "before": _clean(row.get("linked_joint_accounts")),
                    "after": _linked_ids_from_signatories(
                        row.get("signatories_json"),
                        _clean(row.get("customer_id")),
                    ),
                }
            )

        frame.loc[mask, "linked_joint_accounts"] = repaired[mask].replace("", pd.NA)
        frame.to_parquet(path, index=False)
        files_changed += 1

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "joint_link_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(examples[:50], handle, indent=2)

    return {
        "account_files_scanned": files_scanned,
        "account_files_changed": files_changed,
        "account_rows_changed": rows_changed,
        "stale_linked_ids_replaced": stale_ids_replaced,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _date_from_row(row: dict[str, str], field: str) -> date | None:
    try:
        return date.fromisoformat(row.get(field, ""))
    except ValueError:
        return None


def repair_marketing_start() -> dict[str, int]:
    removed_dirs = 0
    adjusted_campaigns = 0
    removed_responses = 0
    files_rewritten = 0

    for month_dir in sorted(BANKING.glob("*/*/marketing_campaigns")):
        year = int(month_dir.parent.parent.name)
        month = int(month_dir.parent.name)
        if (year, month) < (FIRST_MARKETING_DATE.year, FIRST_MARKETING_DATE.month):
            shutil.rmtree(month_dir)
            removed_dirs += 1
            continue

        campaigns_path = month_dir / "campaigns.csv"
        responses_path = month_dir / "campaign_responses.csv"
        campaigns = _read_csv(campaigns_path)
        responses = _read_csv(responses_path)

        if campaigns:
            campaign_fields = list(campaigns[0].keys())
        else:
            campaign_fields = [
                "campaign_id",
                "campaign_name",
                "campaign_type",
                "target_segment",
                "channel",
                "product_focus",
                "offer_summary",
                "start_date",
                "end_date",
                "budget_zar",
                "target_customers_count",
                "region",
                "status",
                "success_metric",
            ]

        active_campaign_ids: set[str] = set()
        for row in campaigns:
            start = _date_from_row(row, "start_date")
            end = _date_from_row(row, "end_date")
            if start and start < FIRST_MARKETING_DATE:
                row["start_date"] = FIRST_MARKETING_DATE.isoformat()
                if not end or end < FIRST_MARKETING_DATE:
                    row["end_date"] = (FIRST_MARKETING_DATE + timedelta(days=21)).isoformat()
                adjusted_campaigns += 1
            active_campaign_ids.add(row.get("campaign_id", ""))

        if campaigns_path.exists():
            _write_csv(campaigns_path, campaigns, campaign_fields)
            files_rewritten += 1

        if responses_path.exists():
            response_fields = list(responses[0].keys()) if responses else [
                "response_id",
                "campaign_id",
                "customer_id",
                "account_id",
                "response_date",
                "response_type",
                "conversion_value_zar",
                "channel_used",
                "notes",
            ]
            kept: list[dict[str, str]] = []
            for row in responses:
                response_date = _date_from_row(row, "response_date")
                if response_date and response_date < FIRST_MARKETING_DATE:
                    removed_responses += 1
                    continue
                if active_campaign_ids and row.get("campaign_id", "") not in active_campaign_ids:
                    removed_responses += 1
                    continue
                kept.append(row)
            _write_csv(responses_path, kept, response_fields)
            files_rewritten += 1

    return {
        "pre_july_2019_marketing_dirs_removed": removed_dirs,
        "campaign_rows_adjusted_to_july_2019": adjusted_campaigns,
        "early_or_orphan_response_rows_removed": removed_responses,
        "marketing_csv_files_rewritten": files_rewritten,
    }


def verify() -> dict[str, int]:
    customer_ids: set[str] = set()
    for path in BANKING.glob("*/*/customers_*.parquet"):
        frame = pd.read_parquet(path, columns=["customer_id"])
        customer_ids.update(frame["customer_id"].dropna().astype(str))

    linked_ids = 0
    linked_missing_customers = 0
    linked_mismatch_with_signatories = 0
    for path in BANKING.glob("*/*/accounts_*.parquet"):
        frame = pd.read_parquet(
            path,
            columns=["customer_id", "linked_joint_accounts", "signatories_json"],
        )
        for _, row in frame.iterrows():
            expected = _linked_ids_from_signatories(
                row.get("signatories_json"),
                _clean(row.get("customer_id")),
            )
            actual = _clean(row.get("linked_joint_accounts"))
            if actual != expected:
                linked_mismatch_with_signatories += 1
            for customer_id in [part for part in actual.split(";") if part]:
                linked_ids += 1
                if customer_id not in customer_ids:
                    linked_missing_customers += 1

    early_marketing_dirs = sum(
        1
        for path in BANKING.glob("2019/0[1-6]/marketing_campaigns")
        if path.exists()
    )
    early_campaign_rows = 0
    early_response_rows = 0
    for path in BANKING.glob("*/*/marketing_campaigns/campaigns.csv"):
        for row in _read_csv(path):
            start = _date_from_row(row, "start_date")
            if start and start < FIRST_MARKETING_DATE:
                early_campaign_rows += 1
    for path in BANKING.glob("*/*/marketing_campaigns/campaign_responses.csv"):
        for row in _read_csv(path):
            response_date = _date_from_row(row, "response_date")
            if response_date and response_date < FIRST_MARKETING_DATE:
                early_response_rows += 1

    return {
        "linked_joint_customer_ids": linked_ids,
        "linked_joint_customer_ids_missing_from_customers": linked_missing_customers,
        "linked_joint_account_rows_mismatching_signatories": linked_mismatch_with_signatories,
        "early_marketing_dirs_before_2019_07": early_marketing_dirs,
        "campaign_rows_before_2019_07": early_campaign_rows,
        "response_rows_before_2019_07": early_response_rows,
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "joint_links": repair_account_links(),
        "marketing": repair_marketing_start(),
        "verification": verify(),
    }
    with (ARTIFACTS / "repair_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
