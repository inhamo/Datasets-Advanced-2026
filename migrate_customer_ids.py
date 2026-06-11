"""Repair reused customer IDs across monthly banking source datasets.

The generated source data reused customer IDs in different months for unrelated
people. This migration assigns every monthly customer a deterministic global ID
and updates direct references while preserving all account, transaction, loan,
application, and debit-order identifiers.

Run without --apply for a complete dry run. The apply mode writes the source
files in place and creates crosswalk artifacts under migration_artifacts/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "migration_artifacts" / "customer_id_migration"
PERIOD_PATH_RE = re.compile(r"banking_data[\\/](20\d{2})[\\/](0[1-9]|1[0-2])[\\/]")
CUSTOMER_FILE_RE = re.compile(r"^customers_\d{4}_\d{2}\.parquet$")

# These tables contain the owning account and should inherit its corrected owner.
ACCOUNT_OWNER_TABLES = {
    "accounts",
    "atm_logs",
    "debit_orders",
    "loans",
    "loan_participations",
    "collections_cases",
    "recovery_payments",
    "communications",
    "complaints",
    "suggestions",
    "campaign_responses",
    "correspondent_payment_alerts",
}

# A customer in these tables is not necessarily the account owner.
PERIOD_CUSTOMER_TABLES = {
    "customers",
    "account_signatories",
    "rejected_applications",
}

EMBEDDED_CUSTOMER_FIELDS = {"signatories_json"}
DIRECT_REFERENCE_COLUMNS = {"customer_id", "guardian_customer_id"}
SUPPORTED_SUFFIXES = {".parquet", ".csv", ".jsonl"}


@dataclass(frozen=True)
class Period:
    year: int
    month: int

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def yy(self) -> str:
        return f"{self.year % 100:02d}"

    @property
    def mm(self) -> str:
        return f"{self.month:02d}"


@dataclass
class MigrationStats:
    files_scanned: int = 0
    files_changed: int = 0
    rows_scanned: int = 0
    direct_ids_changed: int = 0
    guardian_ids_changed: int = 0
    embedded_ids_changed: int = 0
    account_anchored_rows: int = 0
    period_anchored_rows: int = 0
    account_customer_mismatches_corrected: int = 0


def period_from_path(path: Path) -> Period:
    match = PERIOD_PATH_RE.search(str(path.resolve()))
    if not match:
        raise ValueError(f"Cannot derive YYYY/MM from path: {path}")
    return Period(int(match.group(1)), int(match.group(2)))


def logical_table_name(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"_20\d{2}_\d{2}$", "", stem)
    return stem


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return next(csv.reader(handle), [])


def jsonl_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        line = handle.readline()
    return list(json.loads(line)) if line.strip() else []


def file_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return parquet_columns(path)
    if path.suffix.lower() == ".csv":
        return csv_columns(path)
    if path.suffix.lower() == ".jsonl":
        return jsonl_columns(path)
    return []


def valid_customer_file(path: Path) -> bool:
    return CUSTOMER_FILE_RE.match(path.name) is not None and "customer_id" in parquet_columns(path)


def split_customer_id(customer_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d{8})", customer_id.strip())
    if not match:
        raise ValueError(f"Unexpected customer ID format: {customer_id!r}")
    prefix, numeric = match.groups()
    return prefix.upper(), numeric[-6:]


def new_customer_id(original_customer_id: str, period: Period) -> str:
    prefix, serial = split_customer_id(original_customer_id)
    return f"{prefix}{period.yy}{period.mm}{serial}"


def identity_hash(row: pd.Series) -> str:
    fields = [
        row.get("customer_type"),
        row.get("full_name"),
        row.get("birth_date"),
        row.get("id_number"),
        row.get("email"),
        row.get("phone_number"),
    ]
    payload = "|".join("" if pd.isna(value) else str(value).strip().lower() for value in fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_customer_crosswalk() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = sorted(DATA_DIR.glob("20??/??/customers_*.parquet"))
    for path in tqdm(paths, desc="Building customer crosswalk", unit="file"):
        if not valid_customer_file(path):
            print(f"  Skipping placeholder customer file without customer_id: {path}")
            continue
        period = period_from_path(path)
        frame = pd.read_parquet(path)
        if frame["customer_id"].isna().any():
            raise ValueError(f"Null customer_id values found in {path}")
        if frame["customer_id"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate customer_id within one monthly file: {path}")
        for _, row in frame.iterrows():
            original = str(row["customer_id"])
            rows.append(
                {
                    "source_year": period.year,
                    "source_month": period.month,
                    "source_period": period.key,
                    "original_customer_id": original,
                    "new_customer_id": new_customer_id(original, period),
                    "customer_type": row.get("customer_type"),
                    "identity_hash": identity_hash(row),
                    "source_file": path.relative_to(BASE_DIR).as_posix(),
                }
            )

    crosswalk = pd.DataFrame(rows)
    if crosswalk.empty:
        raise RuntimeError("No customer rows were found.")
    source_key = ["source_year", "source_month", "original_customer_id"]
    if crosswalk.duplicated(source_key).any():
        duplicates = crosswalk[crosswalk.duplicated(source_key, keep=False)]
        raise ValueError(f"Crosswalk source keys are not unique:\n{duplicates.head(20)}")
    if crosswalk["new_customer_id"].duplicated().any():
        duplicates = crosswalk[crosswalk["new_customer_id"].duplicated(keep=False)]
        raise ValueError(f"Generated customer IDs are not unique:\n{duplicates.head(20)}")
    return crosswalk.sort_values(source_key).reset_index(drop=True)


def crosswalk_lookup(crosswalk: pd.DataFrame) -> dict[tuple[int, int, str], str]:
    return {
        (int(row.source_year), int(row.source_month), str(row.original_customer_id)): str(row.new_customer_id)
        for row in crosswalk.itertuples(index=False)
    }


def global_crosswalk_lookup(crosswalk: pd.DataFrame) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for original, rows in crosswalk.groupby("original_customer_id", sort=False):
        grouped[str(original)] = sorted(set(rows["new_customer_id"].astype(str)))
    return grouped


def map_period_customer_id(
    value: Any,
    period: Period,
    lookup: dict[tuple[int, int, str], str],
    global_lookup: dict[str, list[str]],
    *,
    allow_null: bool = True,
    allow_unresolved: bool = False,
    allow_external_reference: bool = False,
    unresolved: set[str] | None = None,
) -> Any:
    if pd.isna(value) or str(value).strip() in {"", "None", "nan", "<NA>"}:
        if allow_null:
            return value
        raise ValueError(f"Null customer ID for {period.key}")
    original = str(value).strip()
    if allow_external_reference and not re.fullmatch(r"(?:IND|COM)\d{8}", original):
        return original
    key = (period.year, period.month, original)
    if key in lookup:
        return lookup[key]
    candidates = global_lookup.get(original, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        if allow_unresolved:
            if unresolved is not None:
                unresolved.add(original)
            return original
        raise KeyError(
            f"Ambiguous customer reference for {period.key} / {original}; "
            f"it resolves to {len(candidates)} different people."
        )
    if allow_unresolved:
        if unresolved is not None:
            unresolved.add(original)
        return original
    raise KeyError(f"No customer crosswalk entry for {period.key} / {original}")


def build_account_bridge(
    lookup: dict[tuple[int, int, str], str],
    global_lookup: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unresolved: set[str] = set()
    paths = sorted(DATA_DIR.glob("20??/??/accounts_*.parquet"))
    for path in tqdm(paths, desc="Building account-owner bridge", unit="file"):
        columns = parquet_columns(path)
        if not {"account_id", "customer_id"}.issubset(columns):
            print(f"  Skipping placeholder account file: {path}")
            continue
        period = period_from_path(path)
        selected = ["account_id", "customer_id"]
        if "opening_date" in columns:
            selected.append("opening_date")
        frame = pd.read_parquet(path, columns=selected)
        for row in frame.to_dict("records"):
            original = str(row["customer_id"])
            corrected = map_period_customer_id(
                original,
                period,
                lookup,
                global_lookup,
                allow_null=False,
                allow_unresolved=True,
                unresolved=unresolved,
            )
            rows.append(
                {
                    "account_id": str(row["account_id"]),
                    "original_customer_id": original,
                    "new_customer_id": corrected,
                    "resolution_status": (
                        "unresolved_preexisting_orphan"
                        if corrected == original
                        else "resolved"
                    ),
                    "opening_date": row.get("opening_date"),
                    "source_year": period.year,
                    "source_month": period.month,
                    "source_period": period.key,
                    "source_file": path.relative_to(BASE_DIR).as_posix(),
                }
            )
    bridge = pd.DataFrame(rows)
    if bridge.empty:
        raise RuntimeError("No account ownership rows were found.")
    if bridge["account_id"].duplicated().any():
        duplicates = bridge[bridge["account_id"].duplicated(keep=False)]
        raise ValueError(f"Account IDs are not globally unique:\n{duplicates.head(20)}")
    if unresolved:
        print(
            f"  Preserving {len(unresolved):,} pre-existing orphan customer IDs "
            "referenced by accounts; they will be written to the exception report."
        )
    return bridge.sort_values("account_id").reset_index(drop=True)


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype_backend="numpy_nullable")
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file: {path}")


def atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        if path.suffix.lower() == ".parquet":
            frame.to_parquet(temp_path, index=False)
        elif path.suffix.lower() == ".csv":
            frame.to_csv(temp_path, index=False)
        elif path.suffix.lower() == ".jsonl":
            frame.to_json(temp_path, orient="records", lines=True, date_format="iso")
        else:
            raise ValueError(f"Unsupported file: {path}")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def replace_ids_in_json(
    value: Any,
    period: Period,
    lookup: dict[tuple[int, int, str], str],
    global_lookup: dict[str, list[str]],
    unresolved: set[str],
) -> tuple[Any, int]:
    if pd.isna(value) or value in ("", None):
        return value, 0
    parsed = json.loads(value) if isinstance(value, str) else value
    changes = 0

    def walk(node: Any) -> Any:
        nonlocal changes
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            updated = {}
            for key, item in node.items():
                if key.lower().endswith("customer_id") and item not in (None, ""):
                    mapped = map_period_customer_id(
                        item,
                        period,
                        lookup,
                        global_lookup,
                        allow_unresolved=True,
                        allow_external_reference=True,
                        unresolved=unresolved,
                    )
                    changes += int(str(mapped) != str(item))
                    updated[key] = mapped
                else:
                    updated[key] = walk(item)
            return updated
        return node

    updated = walk(parsed)
    if isinstance(value, str):
        return json.dumps(updated, ensure_ascii=True, separators=(",", ":")), changes
    return updated, changes


def candidate_files() -> list[Path]:
    paths: list[Path] = []
    for path in DATA_DIR.glob("20??/??/**/*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        columns = set(file_columns(path))
        if columns & DIRECT_REFERENCE_COLUMNS or columns & EMBEDDED_CUSTOMER_FIELDS:
            paths.append(path)
    return sorted(paths)


def account_owner_series(
    frame: pd.DataFrame,
    account_lookup: dict[str, str],
    path: Path,
) -> pd.Series:
    normalized_accounts = frame["account_id"].astype("string").str.strip()
    return normalized_accounts.map(account_lookup)


def migrate_frame(
    frame: pd.DataFrame,
    path: Path,
    period_lookup: dict[tuple[int, int, str], str],
    global_lookup: dict[str, list[str]],
    account_lookup: dict[str, str],
    account_original_lookup: dict[str, str],
    unresolved_references: set[str],
    stats: MigrationStats,
) -> tuple[pd.DataFrame, bool]:
    period = period_from_path(path)
    table = logical_table_name(path)
    changed = False
    stats.rows_scanned += len(frame)

    if "customer_id" in frame.columns:
        original = frame["customer_id"].copy()
        use_account = table in ACCOUNT_OWNER_TABLES and "account_id" in frame.columns
        if use_account:
            account_mapped = account_owner_series(frame, account_lookup, path)
            expected_original = (
                frame["account_id"].astype("string").str.strip().map(account_original_lookup)
            )
            mismatch = expected_original.notna() & original.notna() & (
                expected_original.astype(str).str.strip() != original.astype(str).str.strip()
            )
            stats.account_customer_mismatches_corrected += int(mismatch.sum())
            period_mapped = original.map(
                lambda value: map_period_customer_id(
                    value,
                    period,
                    period_lookup,
                    global_lookup,
                    allow_unresolved=True,
                    unresolved=unresolved_references,
                )
            )
            resolved = account_mapped.where(account_mapped.notna(), period_mapped)
            frame["customer_id"] = resolved
            stats.account_anchored_rows += int(account_mapped.notna().sum())
            stats.period_anchored_rows += int(account_mapped.isna().sum())
        else:
            permit_unresolved = table in {"account_signatories", "rejected_applications"}
            frame["customer_id"] = original.map(
                lambda value: map_period_customer_id(
                    value,
                    period,
                    period_lookup,
                    global_lookup,
                    allow_unresolved=permit_unresolved,
                    allow_external_reference=table == "account_signatories",
                    unresolved=unresolved_references,
                )
            )
            stats.period_anchored_rows += int(original.notna().sum())
        count = int(
            (
                original.fillna("<NULL>").astype(str)
                != frame["customer_id"].fillna("<NULL>").astype(str)
            ).sum()
        )
        stats.direct_ids_changed += count
        changed = changed or count > 0

    if "guardian_customer_id" in frame.columns:
        original = frame["guardian_customer_id"].copy()
        frame["guardian_customer_id"] = original.map(
            lambda value: map_period_customer_id(
                value,
                period,
                period_lookup,
                global_lookup,
                allow_unresolved=True,
                unresolved=unresolved_references,
            )
        )
        count = int(
            (
                original.fillna("<NULL>").astype(str)
                != frame["guardian_customer_id"].fillna("<NULL>").astype(str)
            ).sum()
        )
        stats.guardian_ids_changed += count
        changed = changed or count > 0

    for column in EMBEDDED_CUSTOMER_FIELDS & set(frame.columns):
        updated_values = []
        column_changes = 0
        for value in frame[column]:
            updated, count = replace_ids_in_json(
                value,
                period,
                period_lookup,
                global_lookup,
                unresolved_references,
            )
            updated_values.append(updated)
            column_changes += count
        frame[column] = updated_values
        stats.embedded_ids_changed += column_changes
        changed = changed or column_changes > 0

    return frame, changed


def validate_frame(
    frame: pd.DataFrame,
    path: Path,
    valid_customer_ids: set[str],
    account_lookup: dict[str, str],
    unresolved_references: set[str],
) -> None:
    table = logical_table_name(path)
    if "customer_id" in frame.columns:
        referenced = set(frame["customer_id"].dropna().astype(str))
        if table == "account_signatories":
            referenced = {
                value for value in referenced if re.fullmatch(r"(?:IND|COM)\d{10}", value)
            }
        invalid = referenced - valid_customer_ids - unresolved_references
        if invalid:
            raise ValueError(f"{path}: invalid corrected customer IDs: {sorted(invalid)[:10]}")
        if table in ACCOUNT_OWNER_TABLES and "account_id" in frame.columns:
            expected = frame["account_id"].astype("string").str.strip().map(account_lookup)
            comparable = expected.notna() & frame["customer_id"].notna()
            mismatch = comparable & (expected.astype(str) != frame["customer_id"].astype(str))
            if mismatch.any():
                raise ValueError(f"{path}: account-owner validation failed for {int(mismatch.sum())} rows.")

    if "guardian_customer_id" in frame.columns:
        guardians = set(frame["guardian_customer_id"].dropna().astype(str))
        invalid = guardians - valid_customer_ids - unresolved_references
        if invalid:
            raise ValueError(f"{path}: invalid guardian customer IDs: {sorted(invalid)[:10]}")


def write_artifacts(
    crosswalk: pd.DataFrame,
    bridge: pd.DataFrame,
    artifacts_dir: Path,
    stats: MigrationStats,
    unresolved_references: set[str],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(artifacts_dir / "customer_id_crosswalk.csv", index=False)
    bridge.to_csv(artifacts_dir / "account_customer_bridge.csv", index=False)
    summary = {
        "customer_crosswalk_rows": len(crosswalk),
        "unique_original_customer_ids": int(crosswalk["original_customer_id"].nunique()),
        "unique_new_customer_ids": int(crosswalk["new_customer_id"].nunique()),
        "account_bridge_rows": len(bridge),
        "unresolved_preexisting_orphan_accounts": int(
            (bridge["resolution_status"] == "unresolved_preexisting_orphan").sum()
        ),
        "unresolved_ambiguous_customer_references": len(unresolved_references),
        **stats.__dict__,
    }
    (artifacts_dir / "migration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    unresolved = bridge[bridge["resolution_status"] == "unresolved_preexisting_orphan"]
    unresolved.to_csv(artifacts_dir / "unresolved_preexisting_orphan_accounts.csv", index=False)
    pd.DataFrame(
        {"unresolved_original_customer_id": sorted(unresolved_references)}
    ).to_csv(artifacts_dir / "unresolved_ambiguous_customer_references.csv", index=False)


def run_migration(apply: bool, artifacts_dir: Path) -> None:
    mode = "APPLY" if apply else "DRY RUN"
    print(f"\nCustomer ID migration: {mode}")
    print(f"Data root: {DATA_DIR}")

    crosswalk = build_customer_crosswalk()
    period_lookup = crosswalk_lookup(crosswalk)
    global_lookup = global_crosswalk_lookup(crosswalk)
    bridge = build_account_bridge(period_lookup, global_lookup)
    account_lookup = dict(zip(bridge["account_id"].astype(str), bridge["new_customer_id"].astype(str)))
    account_original_lookup = dict(
        zip(bridge["account_id"].astype(str), bridge["original_customer_id"].astype(str))
    )
    unresolved_customer_ids = set(
        bridge.loc[
            bridge["resolution_status"] == "unresolved_preexisting_orphan",
            "new_customer_id",
        ].astype(str)
    )
    valid_customer_ids = set(crosswalk["new_customer_id"].astype(str)) | unresolved_customer_ids
    unresolved_references: set[str] = set()
    paths = candidate_files()
    stats = MigrationStats()

    print(f"\nCustomers to re-key: {len(crosswalk):,}")
    print(f"Original IDs reused across periods: {(crosswalk.groupby('original_customer_id').size() > 1).sum():,}")
    print(f"Accounts preserving their IDs: {len(bridge):,}")
    print(f"Files containing customer references: {len(paths):,}\n")

    for path in tqdm(paths, desc="Migrating customer references", unit="file"):
        stats.files_scanned += 1
        original_row_count: int
        frame = read_frame(path)
        original_row_count = len(frame)
        migrated, changed = migrate_frame(
            frame,
            path,
            period_lookup,
            global_lookup,
            account_lookup,
            account_original_lookup,
            unresolved_references,
            stats,
        )
        if len(migrated) != original_row_count:
            raise ValueError(f"Row count changed for {path}: {original_row_count} -> {len(migrated)}")
        validate_frame(
            migrated,
            path,
            valid_customer_ids,
            account_lookup,
            unresolved_references,
        )
        if changed:
            stats.files_changed += 1
            if apply:
                atomic_write_frame(migrated, path)

    print("\nValidation passed.")
    print(f"Files scanned: {stats.files_scanned:,}")
    print(f"Files requiring changes: {stats.files_changed:,}")
    print(f"Rows scanned: {stats.rows_scanned:,}")
    print(f"Direct customer IDs changed: {stats.direct_ids_changed:,}")
    print(f"Guardian IDs changed: {stats.guardian_ids_changed:,}")
    print(f"Embedded JSON IDs changed: {stats.embedded_ids_changed:,}")
    print(f"Account-anchored references: {stats.account_anchored_rows:,}")
    print(f"Period-anchored references: {stats.period_anchored_rows:,}")
    print(
        "Pre-existing account/customer mismatches corrected: "
        f"{stats.account_customer_mismatches_corrected:,}"
    )
    print(f"Ambiguous customer references preserved for review: {len(unresolved_references):,}")

    if apply:
        write_artifacts(crosswalk, bridge, artifacts_dir, stats, unresolved_references)
        print(f"Migration artifacts written to: {artifacts_dir}")
        print("Source data migration completed successfully.")
    else:
        print("\nNo files were changed. Run again with --apply after reviewing this output.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace reused monthly customer IDs and update linked source datasets."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrected IDs to source files. Without this flag, perform a dry run only.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for the crosswalk, account bridge, and migration summary.",
    )
    args = parser.parse_args()
    run_migration(apply=args.apply, artifacts_dir=args.artifacts_dir.resolve())


if __name__ == "__main__":
    main()
