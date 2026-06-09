"""Generate monthly ATM operational logs linked to accounts, cards and customers.

Outputs:
  banking_data/YYYY/MM/atm_logs_YYYY_MM.parquet

These are not a replacement for financial transactions. Financial ATM logs are
anchored to transactions.jsonl through linked_transaction_id. Non-financial ATM
events such as balance enquiries, failed PIN attempts, blocked/expired card
attempts, eWallet/cardless errors, mini statements and retained-card logs are
generated as operational-only events.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import random
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BANKING_DIR = BASE_DIR / "banking_data"

ATM_NETWORK = [
    ("ATM-GP-JHB-001", "Johannesburg CBD", "Gauteng", -26.2041, 28.0473),
    ("ATM-GP-SAN-002", "Sandton", "Gauteng", -26.1076, 28.0567),
    ("ATM-GP-PTA-003", "Pretoria Central", "Gauteng", -25.7479, 28.2293),
    ("ATM-WC-CPT-004", "Cape Town CBD", "Western Cape", -33.9249, 18.4241),
    ("ATM-WC-BEL-005", "Bellville", "Western Cape", -33.8943, 18.6294),
    ("ATM-KZN-DBN-006", "Durban CBD", "KwaZulu-Natal", -29.8587, 31.0218),
    ("ATM-KZN-PMB-007", "Pietermaritzburg", "KwaZulu-Natal", -29.6006, 30.3794),
    ("ATM-EC-ELS-008", "East London", "Eastern Cape", -33.0292, 27.8546),
    ("ATM-EC-PE-009", "Gqeberha", "Eastern Cape", -33.9608, 25.6022),
    ("ATM-FS-BLM-010", "Bloemfontein", "Free State", -29.0852, 26.1596),
    ("ATM-LP-PLK-011", "Polokwane", "Limpopo", -23.9045, 29.4689),
    ("ATM-MP-NEL-012", "Mbombela", "Mpumalanga", -25.4658, 30.9853),
    ("ATM-NW-RUS-013", "Rustenburg", "North West", -25.6544, 27.2559),
    ("ATM-NC-KIM-014", "Kimberley", "Northern Cape", -28.7282, 24.7499),
]

EVENT_WEIGHTS = [
    ("balance_enquiry", 0.38),
    ("mini_statement", 0.08),
    ("pin_change", 0.04),
    ("ewallet_cashout", 0.22),
    ("ewallet_send_voucher", 0.12),
    ("cardless_withdrawal", 0.10),
    ("card_retained", 0.02),
    ("card_status_check", 0.04),
]

CARD_LOOKUP: pd.DataFrame | None = None


def weighted_choice(items: list[tuple[str, float]], rng: random.Random) -> str:
    total = sum(w for _, w in items)
    marker = rng.random() * total
    running = 0.0
    for value, weight in items:
        running += weight
        if marker <= running:
            return value
    return items[-1][0]


def mask_card(card_number: Any) -> str | None:
    if pd.isna(card_number):
        return None
    value = str(card_number).split(".")[0]
    if len(value) < 10:
        return None
    return f"{value[:6]}******{value[-4:]}"


def hash_card(card_number: Any) -> str | None:
    if pd.isna(card_number):
        return None
    value = str(card_number).split(".")[0]
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def card_for_date(account: pd.Series, event_date: date) -> tuple[str | None, str | None, str | None]:
    def values(field: str) -> list[str]:
        value = account.get(field)
        if value is None or pd.isna(value):
            return []
        return [item.strip() for item in str(value).split(",") if item.strip()]

    numbers = values("card_number")
    types = values("card_type")
    issue_dates = [pd.to_datetime(value, errors="coerce") for value in values("card_issue_date")]
    expiry_dates = values("card_expiry_date")
    if not numbers:
        return None, None, None

    selected = 0
    for index, issued in enumerate(issue_dates):
        if not pd.isna(issued) and issued.date() <= event_date:
            selected = index
    selected = min(selected, len(numbers) - 1)
    card_type = types[min(selected, len(types) - 1)] if types else None
    expiry = expiry_dates[min(selected, len(expiry_dates) - 1)] if expiry_dates else None
    return numbers[selected], card_type, expiry


def rand_timestamp(year: int, month: int, rng: random.Random, opening_date: Any = None) -> datetime:
    last = monthrange(year, month)[1]
    start_day = 1
    if opening_date is not None and not pd.isna(opening_date):
        opened = pd.to_datetime(opening_date).date()
        if opened.year == year and opened.month == month:
            start_day = max(start_day, opened.day)
    day = rng.randint(start_day, last)
    hour = rng.choices(
        population=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        weights=[3, 5, 7, 8, 8, 7, 9, 8, 7, 8, 10, 12, 11, 8, 6, 4, 2],
        k=1,
    )[0]
    return datetime(year, month, day, hour, rng.randint(0, 59), rng.randint(0, 59))


def generate_result(event_type: str, account: pd.Series, ts: datetime, rng: random.Random) -> tuple[str, str | None]:
    status = str(account.get("account_status") or "").lower()
    _, _, expiry_raw = card_for_date(account, ts.date())
    expired = False
    if expiry_raw is not None and not pd.isna(expiry_raw):
        expired = pd.to_datetime(expiry_raw).date() < ts.date()

    if event_type == "card_retained":
        return "failed", rng.choice(["suspected_fraud", "card_reported_lost", "too_many_pin_attempts"])
    if status in {"restricted", "blocked", "suspended", "closed"}:
        return "failed", "blocked_card"
    if expired:
        return "failed", "expired_card"

    base_fail = {
        "balance_enquiry": 0.02,
        "mini_statement": 0.03,
        "cash_withdrawal": 0.10,
        "cash_deposit": 0.04,
        "pin_change": 0.07,
        "ewallet_cashout": 0.16,
        "ewallet_send_voucher": 0.13,
        "cardless_withdrawal": 0.14,
        "card_status_check": 0.03,
    }.get(event_type, 0.08)

    if rng.random() > base_fail:
        return "successful", None

    reasons = {
        "cash_withdrawal": ["incorrect_pin", "insufficient_funds", "daily_limit_exceeded", "atm_cash_unavailable"],
        "cash_deposit": ["cash_acceptor_jam", "deposit_bag_rejected", "note_validation_failed"],
        "balance_enquiry": ["incorrect_pin", "host_timeout"],
        "mini_statement": ["incorrect_pin", "printer_unavailable", "host_timeout"],
        "pin_change": ["incorrect_current_pin", "pin_policy_failed", "host_timeout"],
        "ewallet_cashout": ["wrong_voucher_pin", "voucher_expired", "recipient_msisdn_mismatch", "voucher_already_redeemed"],
        "ewallet_send_voucher": ["recipient_msisdn_invalid", "daily_limit_exceeded", "insufficient_funds"],
        "cardless_withdrawal": ["wrong_otp", "otp_expired", "recipient_msisdn_mismatch"],
        "card_status_check": ["host_timeout", "blocked_card"],
    }
    return "failed", rng.choice(reasons.get(event_type, ["host_timeout"]))


def event_amount(event_type: str, result: str, rng: random.Random) -> float | None:
    if event_type in {"balance_enquiry", "mini_statement", "pin_change", "card_retained", "card_status_check"}:
        return None
    if event_type in {"cash_withdrawal", "ewallet_cashout", "cardless_withdrawal"}:
        return float(rng.choice([50, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000]))
    if event_type == "ewallet_send_voucher":
        return float(rng.choice([50, 100, 150, 200, 250, 300, 500, 750, 1000]))
    if event_type == "cash_deposit":
        return float(rng.choice([100, 200, 300, 500, 1000, 1500, 2500, 5000]))
    return None


def ewallet_fields(event_type: str, failure_reason: str | None, rng: random.Random) -> dict[str, Any]:
    if event_type not in {"ewallet_cashout", "ewallet_send_voucher", "cardless_withdrawal"}:
        return {
            "ewallet_reference": None,
            "ewallet_recipient_msisdn_entered": None,
            "ewallet_error_type": None,
        }
    msisdn = f"+27{rng.randint(60, 84)}{rng.randint(1000000, 9999999)}"
    return {
        "ewallet_reference": f"EW{rng.randint(10_000_000, 99_999_999)}",
        "ewallet_recipient_msisdn_entered": msisdn,
        "ewallet_error_type": failure_reason if failure_reason in {
            "wrong_voucher_pin",
            "voucher_expired",
            "recipient_msisdn_mismatch",
            "voucher_already_redeemed",
            "recipient_msisdn_invalid",
            "wrong_otp",
            "otp_expired",
        } else None,
    }


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or pd.isna(value):
        return {}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def atm_from_metadata(metadata: dict[str, Any], rng: random.Random) -> tuple[str, str, str, float, float]:
    atm_id = metadata.get("atm_id")
    gps = metadata.get("gps_coordinates") or {}
    if atm_id:
        province = "Unknown"
        location = "ATM network terminal"
        lat = gps.get("latitude")
        lon = gps.get("longitude")
        return (
            str(atm_id),
            location,
            province,
            float(lat) if lat is not None else 0.0,
            float(lon) if lon is not None else 0.0,
        )
    return rng.choice(ATM_NETWORK)


def card_record_for_account(account_id: Any, lookup: pd.DataFrame) -> pd.Series | None:
    if lookup.empty or account_id is None or pd.isna(account_id):
        return None
    found = lookup.loc[lookup["account_id"].eq(account_id)]
    if found.empty:
        return None
    return found.iloc[0]


def failure_from_transaction(tx: pd.Series, rng: random.Random) -> str | None:
    status = str(tx.get("status") or "").lower()
    if status in {"completed", "successful", "success"}:
        return None
    return rng.choice(["insufficient_funds", "transaction_declined", "host_timeout", "daily_limit_exceeded"])


def event_type_from_transaction(tx: pd.Series) -> str:
    category = str(tx.get("category") or "").lower()
    debit_credit = str(tx.get("debit_credit") or "").lower()
    if category in {"airtime", "data", "prepaid"}:
        return "prepaid_purchase"
    if debit_credit == "credit":
        return "cash_deposit"
    return "cash_withdrawal"


def all_card_lookup() -> pd.DataFrame:
    global CARD_LOOKUP
    if CARD_LOOKUP is not None:
        return CARD_LOOKUP

    frames: list[pd.DataFrame] = []
    cols = [
        "account_id",
        "account_number",
        "customer_id",
        "account_status",
        "card_number",
        "card_type",
        "card_expiry_date",
        "currency",
        "opening_date",
    ]
    for path in sorted(BANKING_DIR.glob("20*/??/accounts_*.parquet")):
        df = pd.read_parquet(path)
        if df.empty or "card_number" not in df.columns:
            continue
        keep = [c for c in cols if c in df.columns]
        frames.append(df.loc[df["card_number"].notna(), keep].copy())

    if not frames:
        CARD_LOOKUP = pd.DataFrame(columns=cols)
        return CARD_LOOKUP

    lookup = pd.concat(frames, ignore_index=True)
    lookup = lookup.sort_values(["account_id", "opening_date"], na_position="last")
    lookup = lookup.drop_duplicates("account_id", keep="last")
    CARD_LOOKUP = lookup
    return CARD_LOOKUP


def load_month_accounts(year: int, month: int) -> pd.DataFrame:
    accounts_path = BANKING_DIR / str(year) / f"{month:02d}" / f"accounts_{year}_{month:02d}.parquet"
    if not accounts_path.exists():
        return pd.DataFrame()

    accounts = pd.read_parquet(accounts_path)
    if not accounts.empty and "card_number" in accounts.columns:
        return accounts[accounts["card_number"].notna()].copy()

    debit_orders_path = BANKING_DIR / str(year) / f"{month:02d}" / f"debit_orders_{year}_{month:02d}.parquet"
    if not debit_orders_path.exists():
        return pd.DataFrame()

    debit_orders = pd.read_parquet(debit_orders_path)
    if debit_orders.empty or "account_id" not in debit_orders.columns:
        return pd.DataFrame()

    active_pairs = debit_orders[["account_id", "customer_id"]].dropna().drop_duplicates()
    lookup = all_card_lookup()
    if lookup.empty:
        return pd.DataFrame()

    fallback = active_pairs.merge(lookup.drop(columns=["customer_id"], errors="ignore"), on="account_id", how="inner")
    if "customer_id_x" in fallback.columns:
        fallback = fallback.rename(columns={"customer_id_x": "customer_id"})
    return fallback[fallback["card_number"].notna()].copy()


def transaction_log_rows(year: int, month: int, lookup: pd.DataFrame, rng: random.Random) -> list[dict[str, Any]]:
    tx_path = BANKING_DIR / str(year) / f"{month:02d}" / "transactions.jsonl"
    if not tx_path.exists():
        return []

    tx = pd.read_json(tx_path, lines=True)
    if tx.empty or "channel" not in tx.columns:
        return []
    tx = tx[tx["channel"].astype(str).str.lower().eq("atm")].copy()
    if tx.empty:
        return []

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(tx.itertuples(index=False), start=1):
        row = pd.Series(item._asdict())
        account = card_record_for_account(row.get("account_id"), lookup)
        if account is None:
            continue
        ts = pd.to_datetime(row.get("transaction_timestamp")).to_pydatetime()
        metadata = parse_metadata(row.get("channel_metadata"))
        atm = atm_from_metadata(metadata, rng)
        failure_reason = failure_from_transaction(row, rng)
        result = "successful" if failure_reason is None else "failed"
        event_type = event_type_from_transaction(row)
        card_blocked = failure_reason == "blocked_card" or str(account.get("account_status")).lower() in {"restricted", "blocked", "suspended", "closed"}

        card_number, card_type, card_expiry = card_for_date(account, ts.date())
        rows.append(
            {
                "atm_log_id": f"ATMLOG-{year}{month:02d}-TX-{i:07d}",
                "event_timestamp": ts.isoformat(),
                "event_date": ts.date().isoformat(),
                "event_time": ts.time().isoformat(timespec="seconds"),
                "atm_id": atm[0],
                "terminal_id": metadata.get("terminal_id") or f"TERM-{str(atm[0]).replace('ATM-', '')}",
                "atm_location": atm[1],
                "atm_province": atm[2],
                "atm_latitude": atm[3],
                "atm_longitude": atm[4],
                "customer_id": row.get("customer_id"),
                "account_id": row.get("account_id"),
                "account_number": account.get("account_number"),
                "account_status": account.get("account_status"),
                "masked_card_number": mask_card(card_number),
                "card_number_hash": hash_card(card_number),
                "card_type": card_type,
                "card_expiry_date": card_expiry,
                "card_block_status": bool(card_blocked),
                "event_type": event_type,
                "attempt_result": result,
                "failure_reason": failure_reason,
                "amount": float(row.get("amount")) if not pd.isna(row.get("amount")) else None,
                "currency": account.get("currency") or "ZAR",
                "balance_enquiry_requested": False,
                "available_balance_returned": None,
                "pin_attempt_number": None,
                "cash_bin_status": rng.choice(["normal", "low_cash", "cash_out"]) if event_type == "cash_withdrawal" else None,
                "receipt_printed": bool(rng.random() < 0.74) if result == "successful" else False,
                "host_response_code": "00" if result == "successful" else rng.choice(["05", "51", "55", "57", "61", "68", "91"]),
                "network_latency_ms": row.get("network_latency_ms") if "network_latency_ms" in row else rng.randint(120, 1800),
                "linked_transaction_id": row.get("transaction_id"),
                "transaction_category": row.get("category"),
                "transaction_status": row.get("status"),
                "source_system": "atm_switch",
                "ewallet_reference": None,
                "ewallet_recipient_msisdn_entered": None,
                "ewallet_error_type": None,
            }
        )
    return rows


def generate_month(year: int, month: int) -> int:
    accounts = load_month_accounts(year, month)
    lookup = all_card_lookup()
    if accounts.empty and lookup.empty:
        return 0

    rng = random.Random(9100 + year * 100 + month)
    rows: list[dict[str, Any]] = transaction_log_rows(year, month, lookup, rng)

    if accounts.empty:
        accounts = lookup.sample(n=min(len(lookup), 100), random_state=9100 + year * 100 + month).copy()

    n = min(len(accounts), max(60, int(len(accounts) * rng.uniform(0.06, 0.10))))
    selected = accounts.sample(n=n, random_state=9100 + year * 100 + month, replace=False)

    sequence = 1
    for _, account in selected.iterrows():
        attempts = rng.choices([1, 2, 3, 4], weights=[72, 20, 6, 2], k=1)[0]
        if str(account.get("account_status")).lower() in {"restricted", "blocked", "suspended", "closed"}:
            attempts = max(attempts, rng.choice([1, 2]))

        for _ in range(attempts):
            event_type = weighted_choice(EVENT_WEIGHTS, rng)
            ts = rand_timestamp(year, month, rng, account.get("opening_date"))
            result, failure_reason = generate_result(event_type, account, ts, rng)
            atm = rng.choice(ATM_NETWORK)
            amount = event_amount(event_type, result, rng)
            ew = ewallet_fields(event_type, failure_reason, rng)
            card_blocked = failure_reason in {"blocked_card", "card_reported_lost", "suspected_fraud", "too_many_pin_attempts"} or str(account.get("account_status")).lower() in {"restricted", "blocked", "suspended", "closed"}

            card_number, card_type, card_expiry = card_for_date(account, ts.date())
            rows.append(
                {
                    "atm_log_id": f"ATMLOG-{year}{month:02d}-{sequence:07d}",
                    "event_timestamp": ts.isoformat(),
                    "event_date": ts.date().isoformat(),
                    "event_time": ts.time().isoformat(timespec="seconds"),
                    "atm_id": atm[0],
                    "terminal_id": f"TERM-{atm[0].replace('ATM-', '')}",
                    "atm_location": atm[1],
                    "atm_province": atm[2],
                    "atm_latitude": atm[3],
                    "atm_longitude": atm[4],
                    "customer_id": account.get("customer_id"),
                    "account_id": account.get("account_id"),
                    "account_number": account.get("account_number"),
                    "account_status": account.get("account_status"),
                    "masked_card_number": mask_card(card_number),
                    "card_number_hash": hash_card(card_number),
                    "card_type": card_type,
                    "card_expiry_date": card_expiry,
                    "card_block_status": bool(card_blocked),
                    "event_type": event_type,
                    "attempt_result": result,
                    "failure_reason": failure_reason,
                    "amount": amount,
                    "currency": account.get("currency") or "ZAR",
                    "balance_enquiry_requested": event_type == "balance_enquiry",
                    "available_balance_returned": round(rng.uniform(0, 35000), 2) if event_type == "balance_enquiry" and result == "successful" else None,
                    "pin_attempt_number": rng.randint(1, 3) if failure_reason in {"incorrect_pin", "incorrect_current_pin", "too_many_pin_attempts"} else None,
                    "cash_bin_status": rng.choice(["normal", "low_cash", "cash_out"]) if event_type == "cash_withdrawal" else None,
                    "receipt_printed": bool(rng.random() < 0.74) if result == "successful" else False,
                    "host_response_code": "00" if result == "successful" else rng.choice(["05", "51", "55", "57", "61", "68", "91"]),
                    "network_latency_ms": rng.randint(120, 1800) if failure_reason != "host_timeout" else rng.randint(5000, 15000),
                    "linked_transaction_id": None,
                    "transaction_category": None,
                    "transaction_status": None,
                    "source_system": "atm_switch",
                    **ew,
                }
            )
            sequence += 1

    out_path = BANKING_DIR / str(year) / f"{month:02d}" / f"atm_logs_{year}_{month:02d}.parquet"
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    pd.DataFrame(rows).sort_values(["event_timestamp", "atm_log_id"]).to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATM operational logs for monthly banking data.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    total = 0
    months = 0
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            count = generate_month(year, month)
            if count:
                total += count
                months += 1
    print(f"Generated {total:,} ATM log rows across {months} months.")


if __name__ == "__main__":
    main()
