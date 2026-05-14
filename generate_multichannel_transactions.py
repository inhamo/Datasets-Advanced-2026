from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "banking_data"
COMMONS_DIR = BASE_DIR / "commons"


# Simplified but production-usable schedule templates.
CUSTOMER_SCHEDULE_TEMPLATES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "Employed_FullTime": {
        "weekday": [
            {"window": "05:00-06:30", "activity": "wake_up", "channels": ["mobile_banking_app"], "location": "home"},
            {"window": "06:30-08:00", "activity": "commute_morning", "channels": ["pos", "mobile_banking_app", "contactless"], "location": "transit", "common_txns": ["fuel", "transport", "groceries", "airtime"]},
            {"window": "08:00-12:00", "activity": "work_morning", "channels": ["mobile_banking_app", "online_banking"], "location": "work", "common_txns": ["airtime", "utilities", "transfer"]},
            {"window": "12:00-13:30", "activity": "lunch", "channels": ["pos", "atm", "mobile_banking_app", "contactless", "ewallet"], "location": "work_area", "common_txns": ["restaurants", "retail", "groceries"]},
            {"window": "18:00-21:00", "activity": "leisure", "channels": ["pos", "mobile_banking_app", "online_banking", "ewallet", "contactless", "atm"], "location": "home_area", "common_txns": ["restaurants", "entertainment", "retail", "alcohol", "groceries"]},
            {"window": "21:00-23:00", "activity": "home_evening", "channels": ["mobile_banking_app", "online_banking", "ussd"], "location": "home", "common_txns": ["airtime", "utilities", "transfer"]},
            {"window": "23:00-05:00", "activity": "sleeping", "channels": [], "location": "home", "probability": 0.02},
        ],
        "weekend": [
            {"window": "10:00-13:00", "activity": "errands", "channels": ["pos", "atm", "mobile_banking_app", "contactless", "ewallet"], "location": "home_area", "common_txns": ["groceries", "retail", "fuel", "restaurants", "clothing"]},
            {"window": "15:00-19:00", "activity": "social", "channels": ["pos", "atm", "mobile_banking_app", "ewallet", "contactless"], "location": "social_area", "common_txns": ["alcohol", "restaurants", "retail", "entertainment"]},
            {"window": "23:00-07:00", "activity": "sleeping", "channels": [], "location": "home", "probability": 0.05},
        ],
    },
    "Self_Employed": {
        "weekday": [
            {"window": "06:00-09:00", "activity": "morning_flexible", "channels": ["mobile_banking_app", "online_banking", "pos", "atm"], "location": "variable", "common_txns": ["fuel", "groceries", "airtime"]},
            {"window": "09:00-16:00", "activity": "work_variable", "channels": ["mobile_banking_app", "online_banking", "pos", "ewallet", "atm"], "location": "variable", "common_txns": ["fuel", "groceries", "restaurants", "retail", "transfer"]},
            {"window": "16:00-20:00", "activity": "evening_flexible", "channels": ["pos", "mobile_banking_app", "online_banking", "ewallet", "atm", "contactless"], "location": "home_area", "common_txns": ["groceries", "retail", "restaurants", "alcohol"]},
            {"window": "20:00-06:00", "activity": "night", "channels": ["mobile_banking_app", "online_banking"], "location": "home", "probability": 0.15},
        ],
        "weekend": [],
    },
    "Unemployed": {
        "weekday": [
            {"window": "08:00-10:00", "activity": "morning", "channels": ["mobile_banking_app", "ussd"], "location": "home", "common_txns": ["airtime"]},
            {"window": "10:00-14:00", "activity": "day_activities", "channels": ["pos", "atm", "mobile_banking_app", "ussd"], "location": "home_area", "common_txns": ["groceries", "transport", "airtime", "utilities"]},
            {"window": "14:00-20:00", "activity": "afternoon", "channels": ["pos", "mobile_banking_app", "ussd", "atm"], "location": "home_area", "common_txns": ["groceries", "alcohol", "airtime", "entertainment"]},
            {"window": "20:00-08:00", "activity": "night", "channels": ["mobile_banking_app", "ussd"], "location": "home", "probability": 0.10},
        ],
        "weekend": [],
    },
    "Student": {
        "weekday": [
            {"window": "06:00-08:00", "activity": "morning_rush", "channels": ["mobile_banking_app", "pos", "contactless"], "location": "transit", "common_txns": ["transport", "groceries", "airtime"]},
            {"window": "08:00-15:00", "activity": "campus", "channels": ["mobile_banking_app", "pos", "ewallet"], "location": "campus", "common_txns": ["restaurants", "groceries", "airtime", "entertainment"]},
            {"window": "18:00-23:00", "activity": "evening", "channels": ["pos", "mobile_banking_app", "online_banking", "ewallet"], "location": "social_area", "common_txns": ["restaurants", "alcohol", "entertainment", "retail"]},
            {"window": "23:00-06:00", "activity": "night", "channels": ["mobile_banking_app"], "location": "home", "probability": 0.15},
        ],
        "weekend": [
            {"window": "14:00-23:30", "activity": "social_extended", "channels": ["pos", "mobile_banking_app", "ewallet", "atm", "contactless"], "location": "social_area", "common_txns": ["alcohol", "restaurants", "entertainment", "retail", "transport"]}
        ],
    },
}


CHANNEL_REGISTRY: dict[str, dict[str, Any]] = {
    "pos": {"category": "physical", "requires_presence": True, "min_interval_seconds": 120, "max_amount": 50000},
    "atm": {"category": "physical", "requires_presence": True, "min_interval_seconds": 60, "max_amount": 20000},
    "branch": {"category": "physical", "requires_presence": True, "min_interval_seconds": 300, "max_amount": 500000},
    "mobile_banking_app": {"category": "digital", "requires_presence": False, "min_interval_seconds": 10, "max_amount": 250000},
    "online_banking": {"category": "digital", "requires_presence": False, "min_interval_seconds": 15, "max_amount": 500000},
    "ewallet": {"category": "hybrid", "requires_presence": True, "min_interval_seconds": 30, "max_amount": 10000},
    "ussd": {"category": "digital", "requires_presence": False, "min_interval_seconds": 45, "max_amount": 5000},
    "contactless": {"category": "physical", "requires_presence": True, "min_interval_seconds": 5, "max_amount": 500},
    "scheduled_payment": {"category": "automated", "requires_presence": False, "min_interval_seconds": 0, "max_amount": 999999999},
}


CITY_CENTERS: dict[str, tuple[float, float]] = {
    "Johannesburg": (-26.2041, 28.0473),
    "Pretoria": (-25.7479, 28.2293),
    "Soweto": (-26.2485, 27.8540),
    "Sandton": (-26.1076, 28.0567),
    "Centurion": (-25.8640, 28.1881),
    "Midrand": (-25.9992, 28.1263),
    "Randburg": (-26.0936, 28.0066),
    "Cape Town": (-33.9249, 18.4241),
    "Stellenbosch": (-33.9321, 18.8602),
    "Bellville": (-33.8918, 18.6291),
    "Somerset West": (-34.0840, 18.8433),
    "Wynberg": (-34.0056, 18.4686),
    "Durban": (-29.8587, 31.0218),
    "Pietermaritzburg": (-29.6006, 30.3794),
    "Umhlanga": (-29.7272, 31.0850),
    "Richards Bay": (-28.7807, 32.0383),
    "Newcastle": (-27.7577, 29.9318),
    "Gqeberha": (-33.9608, 25.6022),
    "East London": (-33.0292, 27.8546),
    "Mthatha": (-31.5889, 28.7844),
    "Bhisho": (-32.8499, 27.4380),
    "Makhanda": (-33.3106, 26.5256),
    "Polokwane": (-23.9045, 29.4689),
    "Tzaneen": (-23.8332, 30.1635),
    "Mokopane": (-24.1944, 29.0097),
    "Thohoyandou": (-22.9456, 30.4849),
    "Lephalale": (-23.6664, 27.7448),
    "Mbombela": (-25.4753, 30.9694),
    "Emalahleni": (-25.8728, 29.2553),
    "Secunda": (-26.5160, 29.2020),
    "Middelburg": (-25.7751, 29.4648),
    "Bethal": (-26.4579, 29.4655),
    "Rustenburg": (-25.6544, 27.2559),
    "Mahikeng": (-25.8560, 25.6403),
    "Klerksdorp": (-26.8521, 26.6667),
    "Potchefstroom": (-26.7145, 27.0970),
    "Brits": (-25.6347, 27.7802),
    "Bloemfontein": (-29.0852, 26.1596),
    "Welkom": (-27.9777, 26.7351),
    "Kroonstad": (-27.6504, 27.2349),
    "Bethlehem": (-28.2308, 28.3071),
    "Sasolburg": (-26.8136, 27.8160),
    "Kimberley": (-28.7282, 24.7499),
    "Upington": (-28.4478, 21.2561),
    "Kuruman": (-27.4521, 23.4325),
    "Springbok": (-29.6643, 17.8865),
    "De Aar": (-30.6497, 24.0123),
}


CITY_FALLBACK_BY_PROVINCE: dict[str, tuple[float, float]] = {
    "Gauteng": (-26.2041, 28.0473),
    "Western Cape": (-33.9249, 18.4241),
    "KwaZulu-Natal": (-29.8587, 31.0218),
    "Eastern Cape": (-33.0292, 27.8546),
    "Limpopo": (-23.9045, 29.4689),
    "Mpumalanga": (-25.4753, 30.9694),
    "North West": (-26.8521, 26.6667),
    "Free State": (-29.0852, 26.1596),
    "Northern Cape": (-28.7282, 24.7499),
}


# Calibrated from recent fraud reports: financial-transaction suspected fraud is
# far below login/account-creation risk, while SA losses skew heavily digital/CNP.
TRANSACTION_FRAUD_BASE_RATE = 0.0035
TRAVEL_TIME_ANOMALY_RATE = 0.0008
TRAVEL_TIME_ANOMALY_COOLDOWN_DAYS = 60


RETAINED_OUTPUT_FILES = {"transactions.jsonl"}
GENERATED_SIDECAR_FILES = {
    "customer_sessions.jsonl",
    "fraud_cases.jsonl",
    "error_log.jsonl",
    "channel_analytics.json",
    "anomaly_flags.jsonl",
    "location_traces.jsonl",
    "behavioral_profiles.jsonl",
}


CATEGORY_BASE_AMOUNTS = {
    "groceries": 450,
    "fuel": 700,
    "transport": 120,
    "airtime": 80,
    "utilities": 900,
    "retail": 650,
    "restaurants": 350,
    "entertainment": 300,
    "alcohol": 250,
    "transfer": 1200,
    "clothing": 550,
}


FRAUD_PATTERN_WEIGHTS = {
    "smurfing": 0.15,
    "card_testing": 0.20,
    "account_takeover": 0.20,
    "money_mule": 0.20,
    "round_tripping": 0.15,
    "bust_out": 0.10,
}


ERROR_INJECTION = {
    "duplicate_transaction": 0.003,
    "reversed_transaction": 0.001,
    "delayed_settlement": 0.005,
    "missing_merchant_name": 0.010,
    "truncated_description": 0.020,
    "incorrect_timestamp": 0.005,
    "channel_misclassification": 0.003,
    "amount_corruption": 0.002,
    "status_inconsistency": 0.004,
    "encoding_errors": 0.001,
}


@dataclass
class LocationPoint:
    province: str
    city: str
    suburb: str
    area_type: str
    latitude: float
    longitude: float


@dataclass
class CustomerState:
    customer_id: str
    current_location: LocationPoint
    preferred_transport: str
    earliest_next_physical: datetime
    last_channel_time: dict[str, datetime] = field(default_factory=dict)
    last_physical_time: datetime | None = None
    last_physical_location: LocationPoint | None = None
    last_digital_time: datetime | None = None
    session_id: str | None = None
    session_start: datetime | None = None
    anchor_locations: dict[str, LocationPoint] = field(default_factory=dict)
    last_anomaly_time: dict[str, datetime] = field(default_factory=dict)
    monthly_anomaly_count: dict[str, int] = field(default_factory=dict)
    primary_device: str = ""
    secondary_device: str | None = None
    home_ip_prefix: str = ""
    mobile_ip_prefix: str = ""
    channel_affinity: dict[str, float] = field(default_factory=dict)
    category_affinity: dict[str, float] = field(default_factory=dict)
    spend_multiplier: float = 1.0


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_serializable(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, ensure_ascii=False, indent=2)


def cleanup_sidecar_outputs(out_dir: Path) -> list[str]:
    deleted: list[str] = []
    for filename in GENERATED_SIDECAR_FILES:
        if filename in RETAINED_OUTPUT_FILES:
            continue
        path = out_dir / filename
        if path.exists() and path.is_file():
            path.unlink()
            deleted.append(filename)
    return sorted(deleted)


def resolve_table_files(record_type: str, year: int, month: int | None) -> list[Path]:
    if month is not None:
        base = DATA_DIR / str(year) / f"{month:02d}" / f"{record_type}_{year}_{month:02d}"
        if base.with_suffix(".parquet").exists():
            return [base.with_suffix(".parquet")]
        if base.with_suffix(".csv").exists():
            return [base.with_suffix(".csv")]

    yearly_dir = DATA_DIR / str(year)
    monthly = sorted(yearly_dir.glob(f"*/{record_type}_{year}_*.parquet"))
    if monthly:
        return monthly
    monthly_csv = sorted(yearly_dir.glob(f"*/{record_type}_{year}_*.csv"))
    if monthly_csv:
        return monthly_csv

    flat = DATA_DIR / f"{record_type}_{year}.parquet"
    if flat.exists():
        return [flat]
    flat_csv = DATA_DIR / f"{record_type}_{year}.csv"
    if flat_csv.exists():
        return [flat_csv]
    return []


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _concat_files(files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in files:
        try:
            df = read_table(p)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _latest_non_empty(record_type: str, end_year: int) -> pd.DataFrame:
    for y in range(end_year, 2017, -1):
        files = resolve_table_files(record_type, y, None)
        df = _concat_files(files)
        if not df.empty:
            return df
    return pd.DataFrame()


def load_accounts_customers(year: int, month: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    account_files = resolve_table_files("accounts", year, month)
    customer_files = resolve_table_files("customers", year, month)

    accounts = _concat_files(account_files)
    customers = _concat_files(customer_files)

    # If month slice is sparse/empty, fallback to year-wide data.
    if accounts.empty:
        accounts = _concat_files(resolve_table_files("accounts", year, None))
    if customers.empty:
        customers = _concat_files(resolve_table_files("customers", year, None))

    # Last-resort fallback to latest prior non-empty data.
    if accounts.empty:
        accounts = _latest_non_empty("accounts", year)
    if customers.empty:
        customers = _latest_non_empty("customers", year)

    if accounts.empty or customers.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "customer_id" not in customers.columns and "CustomerID" in customers.columns:
        customers = customers.rename(columns={"CustomerID": "customer_id"})

    if "customer_id" not in customers.columns or "account_id" not in accounts.columns or "customer_id" not in accounts.columns:
        return pd.DataFrame(), pd.DataFrame()

    customers = customers.drop_duplicates(subset=["customer_id"]).copy()
    accounts = accounts.drop_duplicates(subset=["account_id"]).copy()
    return accounts, customers


def load_reference_locations() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with (COMMONS_DIR / "locations.json").open("r", encoding="utf-8") as f:
        locations = json.load(f)
    with (COMMONS_DIR / "branches.json").open("r", encoding="utf-8") as f:
        branches = json.load(f)
    return locations, branches


def random_location_for_province_city(province: str, city: str, radius_km: float = 8.0) -> LocationPoint:
    area_type = random.choice(["urban_dense", "suburban", "township", "rural"])
    suburb = f"{city} {random.choice(['Central', 'North', 'South', 'West', 'East'])}"
    center = CITY_CENTERS.get(city) or CITY_FALLBACK_BY_PROVINCE.get(province, (-26.2041, 28.0473))
    angle = random.uniform(0, math.tau)
    distance = radius_km * math.sqrt(random.random())
    lat_offset = (distance * math.cos(angle)) / 111.0
    lon_offset = (distance * math.sin(angle)) / max(1.0, 111.0 * math.cos(math.radians(center[0])))
    latitude = round(center[0] + lat_offset, 6)
    longitude = round(center[1] + lon_offset, 6)
    return LocationPoint(
        province=province,
        city=city,
        suburb=suburb,
        area_type=area_type,
        latitude=latitude,
        longitude=longitude,
    )


def nearby_location(origin: LocationPoint, radius_km: float, label: str | None = None) -> LocationPoint:
    angle = random.uniform(0, math.tau)
    distance = radius_km * math.sqrt(random.random())
    lat_offset = (distance * math.cos(angle)) / 111.0
    lon_offset = (distance * math.sin(angle)) / max(1.0, 111.0 * math.cos(math.radians(origin.latitude)))
    suffix = label or random.choice(["Central", "North", "South", "West", "East"])
    return LocationPoint(
        province=origin.province,
        city=origin.city,
        suburb=f"{origin.city} {suffix}",
        area_type=origin.area_type,
        latitude=round(origin.latitude + lat_offset, 6),
        longitude=round(origin.longitude + lon_offset, 6),
    )


def distant_location(provinces: list[str], cities_by_province: dict[str, list[str]], current: LocationPoint) -> LocationPoint:
    possible_provinces = [p for p in provinces if p != current.province] or provinces
    province = random.choice(possible_provinces)
    city = random.choice(cities_by_province.get(province, [current.city]))
    return random_location_for_province_city(province, city, radius_km=12.0)


def location_for_block(state: CustomerState, block_location: str) -> LocationPoint:
    anchors = state.anchor_locations
    if block_location == "home":
        return nearby_location(anchors["home"], 1.5, "Central")
    if block_location in ["home_area", "transit"]:
        return nearby_location(anchors["home"], 6.0)
    if block_location in ["work", "work_area", "campus"]:
        return nearby_location(anchors.get("work", anchors["home"]), 4.0)
    if block_location == "social_area":
        return nearby_location(anchors.get("social", anchors["home"]), 8.0)
    if block_location == "variable":
        anchor = random.choices(
            [anchors["home"], anchors.get("work", anchors["home"]), anchors.get("social", anchors["home"])],
            weights=[0.45, 0.35, 0.20],
            k=1,
        )[0]
        return nearby_location(anchor, 10.0)
    return nearby_location(state.current_location, 3.0)


def can_emit_anomaly(state: CustomerState, anomaly_type: str, tx_time: datetime, monthly_limit: int = 1) -> bool:
    if state.monthly_anomaly_count.get(anomaly_type, 0) >= monthly_limit:
        return False
    last_time = state.last_anomaly_time.get(anomaly_type)
    if last_time is not None and tx_time - last_time < timedelta(days=TRAVEL_TIME_ANOMALY_COOLDOWN_DAYS):
        return False
    return True


def build_channel_affinity(schedule_key: str, income: float) -> dict[str, float]:
    weights = {
        "pos": 1.0,
        "atm": 0.35,
        "mobile_banking_app": 1.15,
        "online_banking": 0.45,
        "ewallet": 0.35,
        "ussd": 0.20,
        "contactless": 0.75,
    }
    if schedule_key == "Student":
        weights.update({"contactless": 1.05, "mobile_banking_app": 1.35, "ussd": 0.12, "atm": 0.20})
    elif schedule_key == "Self_Employed":
        weights.update({"online_banking": 0.90, "ewallet": 0.65, "pos": 1.10})
    elif schedule_key == "Unemployed":
        weights.update({"ussd": 0.75, "atm": 0.55, "online_banking": 0.15, "contactless": 0.35})
    if income > 45000:
        weights["online_banking"] += 0.30
        weights["contactless"] += 0.25
        weights["ussd"] *= 0.4
    if income < 12000:
        weights["ussd"] += 0.35
        weights["atm"] += 0.20
        weights["online_banking"] *= 0.55
    return weights


def build_category_affinity(schedule_key: str) -> dict[str, float]:
    weights = {k: 1.0 for k in CATEGORY_BASE_AMOUNTS}
    if schedule_key == "Student":
        weights.update({"airtime": 1.4, "transport": 1.35, "restaurants": 1.2, "utilities": 0.45})
    elif schedule_key == "Self_Employed":
        weights.update({"fuel": 1.35, "transfer": 1.25, "restaurants": 1.15})
    elif schedule_key == "Unemployed":
        weights.update({"airtime": 1.5, "groceries": 1.35, "transport": 1.25, "retail": 0.55, "restaurants": 0.55})
    return weights


def weighted_choice(options: list[str], weights_by_key: dict[str, float]) -> str:
    weights = [max(0.01, weights_by_key.get(option, 1.0)) for option in options]
    return random.choices(options, weights=weights, k=1)[0]


def draw_category_for_customer(common: list[str] | None, state: CustomerState) -> str:
    options = common or list(CATEGORY_BASE_AMOUNTS.keys())
    return weighted_choice(options, state.category_affinity)


def amount_for_state(category: str, income: float, state: CustomerState) -> float:
    amount = amount_for_customer(category, income) * state.spend_multiplier
    if random.random() < 0.06:
        amount *= random.uniform(1.6, 2.8)
    return round(max(10.0, float(amount)), 2)


def network_type_for_channel(channel: str) -> str:
    if channel in ["pos", "contactless", "atm", "branch"]:
        return random.choices(["ethernet", "4g", "5g", "3g"], weights=[0.45, 0.32, 0.16, 0.07], k=1)[0]
    if channel == "ussd":
        return random.choices(["3g", "4g", "5g"], weights=[0.58, 0.35, 0.07], k=1)[0]
    return random.choices(["wifi", "4g", "5g", "3g"], weights=[0.46, 0.31, 0.18, 0.05], k=1)[0]


def ip_for_transaction(state: CustomerState, channel: str, network_type: str) -> str:
    if channel in ["pos", "contactless", "atm", "branch"]:
        prefix = f"102.{random.randint(1, 240)}.{random.randint(1, 240)}"
    elif network_type == "wifi":
        prefix = state.home_ip_prefix
    else:
        prefix = state.mobile_ip_prefix
    return f"{prefix}.{random.randint(2, 254)}"


def device_for_transaction(state: CustomerState, channel: str, is_anomaly: bool) -> str:
    if is_anomaly:
        return f"DEV-{state.customer_id[-6:]}-NEW-{random.randint(1000,9999)}"
    if channel in ["pos", "contactless", "atm"]:
        return state.secondary_device or state.primary_device
    return state.primary_device if random.random() < 0.96 or state.secondary_device is None else state.secondary_device


def behavioral_score_for_tx(state: CustomerState, tx_time: datetime, channel: str, category: str, amount: float, anomaly: bool) -> float:
    score = 94.0
    if channel not in state.channel_affinity:
        score -= 8.0
    elif state.channel_affinity.get(channel, 0) < 0.35:
        score -= 5.0
    if state.category_affinity.get(category, 1.0) < 0.65:
        score -= 4.0
    if amount > CATEGORY_BASE_AMOUNTS.get(category, 250) * state.spend_multiplier * 3.2:
        score -= 7.0
    if tx_time.hour <= 4 or tx_time.hour >= 23:
        score -= 6.0
    if anomaly:
        score -= 18.0
    return round(max(35.0, min(99.0, score + random.uniform(-4.0, 3.0))), 2)


def auth_events_for_session(session_start: datetime, channels: set[str], anomalies: list[str]) -> list[dict[str, Any]]:
    method = "device_binding" if "mobile_banking_app" in channels else "otp" if "online_banking" in channels or "ewallet" in channels else "pin"
    events: list[dict[str, Any]] = []
    if anomalies:
        events.append(
            {
                "time": session_start - timedelta(minutes=random.randint(1, 8)),
                "method": method,
                "success": False,
                "attempt_number": 1,
                "failure_reason": random.choice(["new_device_challenge", "otp_retry", "risk_step_up"]),
            }
        )
        attempt_number = 2
    else:
        attempt_number = 1
    events.append(
        {
            "time": session_start,
            "method": method,
            "success": True,
            "attempt_number": attempt_number,
        }
    )
    return events


def session_realism_score(tx_count: int, channels: set[str], anomalies: list[str], duration_minutes: float) -> float:
    score = 95.0
    if tx_count > 6:
        score -= (tx_count - 6) * 2.5
    if len(channels) > 4:
        score -= (len(channels) - 4) * 3.0
    if duration_minutes > 720:
        score -= 8.0
    if anomalies:
        score -= 16.0
    return round(max(45.0, min(99.0, score + random.uniform(-3.5, 2.5))), 2)


def customer_schedule_type(occupation: str) -> str:
    o = str(occupation).lower()
    if "student" in o:
        return "Student"
    if "self" in o:
        return "Self_Employed"
    if "unemploy" in o:
        return "Unemployed"
    return "Employed_FullTime"


def parse_window(day: datetime, window: str) -> tuple[datetime, datetime]:
    start, end = window.split("-")
    sh, sm = [int(x) for x in start.split(":")]
    eh, em = [int(x) for x in end.split(":")]
    start_dt = day.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = day.replace(hour=eh, minute=em, second=0, microsecond=0)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def pick_tx_time_in_window(day: datetime, window: str) -> datetime:
    start_dt, end_dt = parse_window(day, window)
    total_seconds = int((end_dt - start_dt).total_seconds())
    return start_dt + timedelta(seconds=random.randint(0, max(1, total_seconds - 1)))


def distance_km(a: LocationPoint, b: LocationPoint) -> float:
    # Approximation is enough for realism constraints.
    lat_factor = 111.0
    lon_factor = 91.0
    dlat = (a.latitude - b.latitude) * lat_factor
    dlon = (a.longitude - b.longitude) * lon_factor
    return math.sqrt(dlat * dlat + dlon * dlon)


def max_speed_kmh(transport: str) -> float:
    return {"walking": 5.0, "bus": 35.0, "taxi": 50.0, "car": 65.0}.get(transport, 40.0)


def can_place_transaction(state: CustomerState, tx_time: datetime, channel: str, location: LocationPoint) -> tuple[bool, str | None]:
    min_interval = CHANNEL_REGISTRY[channel]["min_interval_seconds"]
    prev_same = state.last_channel_time.get(channel)
    if prev_same is not None and (tx_time - prev_same).total_seconds() < min_interval:
        return False, "same_channel_rapid"

    if CHANNEL_REGISTRY[channel]["requires_presence"] and state.last_physical_time is not None and state.last_physical_location is not None:
        elapsed_hours = max(0.01, (tx_time - state.last_physical_time).total_seconds() / 3600.0)
        allowed_distance = max_speed_kmh(state.preferred_transport) * elapsed_hours
        need_distance = distance_km(state.last_physical_location, location)
        if need_distance > allowed_distance + 3.0:
            return False, "travel_time_violation"

    return True, None


def draw_category(common: list[str] | None) -> str:
    if common:
        return random.choice(common)
    return random.choice(list(CATEGORY_BASE_AMOUNTS.keys()))


def amount_for_customer(category: str, income: float) -> float:
    base = CATEGORY_BASE_AMOUNTS.get(category, 250)
    income_multiplier = 0.7 if income < 15000 else 1.0 if income < 40000 else 1.35 if income < 80000 else 1.75
    amount = np.random.normal(base * income_multiplier, base * 0.35)
    return round(max(10.0, float(amount)), 2)


def external_context(tx_time: datetime) -> dict[str, Any]:
    is_weekend = tx_time.weekday() >= 5
    is_payday_window = tx_time.day in [25, 26, 27, 28, 29, 30]
    month = tx_time.month

    weather = random.choices(["sunny", "cloudy", "rain", "storm"], weights=[0.45, 0.30, 0.20, 0.05], k=1)[0]
    if month in [10, 11, 12, 1, 2, 3] and random.random() < 0.35:
        weather = random.choice(["rain", "storm"])

    load_shedding_stage = random.choices([0, 1, 2, 3, 4, 5, 6], weights=[0.20, 0.13, 0.18, 0.17, 0.15, 0.11, 0.06], k=1)[0]

    return {
        "day_of_week": tx_time.strftime("%A"),
        "is_public_holiday": False,
        "is_school_holiday": tx_time.month in [6, 7, 12],
        "is_payday_window": is_payday_window,
        "weather_condition": weather,
        "load_shedding_stage": load_shedding_stage,
        "nearby_events": ["local_sports"] if random.random() < 0.03 else [],
        "is_weekend": is_weekend,
    }


def inject_fraud(tx: dict[str, Any], state: CustomerState) -> dict[str, Any] | None:
    # Transaction-level fraud should be sparse. Current public benchmarks put
    # financial-transaction suspected fraud below higher-risk login/account flows.
    base = TRANSACTION_FRAUD_BASE_RATE
    channel = str(tx.get("channel", ""))
    hour = pd.to_datetime(tx["transaction_timestamp"]).hour
    if channel in ["mobile_banking_app", "online_banking"]:
        base += 0.0018
    elif channel in ["pos", "contactless", "atm", "branch"]:
        base -= 0.0012
    elif channel in ["ewallet", "ussd"]:
        base += 0.0008
    if hour >= 23 or hour <= 4:
        base += 0.0015
    if tx["external_context"]["is_payday_window"]:
        base += 0.0010
    base = max(0.0005, min(base, 0.012))

    if random.random() >= base:
        tx["is_fraudulent"] = False
        tx["fraud_pattern"] = None
        tx["fraud_confidence"] = 0.0
        tx["fraud_metadata"] = {}
        return None

    pattern = random.choices(list(FRAUD_PATTERN_WEIGHTS.keys()), weights=list(FRAUD_PATTERN_WEIGHTS.values()), k=1)[0]
    tx["is_fraudulent"] = True
    tx["fraud_pattern"] = pattern
    tx["fraud_confidence"] = round(random.uniform(0.55, 0.99), 3)

    if pattern == "smurfing":
        tx["fraud_metadata"] = {
            "structuring_flag": True,
            "cumulative_daily_cash": round(float(tx["amount"]) * random.uniform(2.5, 8.0), 2),
            "branch_distance_km": round(random.uniform(5, 65), 2),
            "time_between_branches_minutes": random.randint(8, 60),
            "expected_travel_time_minutes": random.randint(25, 120),
            "anomaly_score": round(random.uniform(70, 98), 2),
        }
    elif pattern == "card_testing":
        tx["fraud_metadata"] = {
            "velocity_flag": True,
            "transactions_per_minute": round(random.uniform(5.5, 14.0), 2),
            "success_ratio": round(random.uniform(0.1, 0.45), 2),
            "amount_sequence_analysis": "incremental_small_values",
            "bin_matched_transactions": random.randint(3, 20),
        }
    elif pattern == "round_tripping":
        tx["fraud_metadata"] = {
            "circular_flow_detected": True,
            "hops_count": random.randint(2, 5),
            "amount_preserved_percentage": round(random.uniform(0.95, 0.985), 4),
            "interbank_transfer_fees_total": round(random.uniform(55, 850), 2),
        }
    elif pattern == "account_takeover":
        tx["fraud_metadata"] = {
            "takeover_risk_score": round(random.uniform(70, 99), 2),
            "device_fingerprint_changed": True,
            "location_anomaly_score": round(random.uniform(65, 98), 2),
            "behavioral_anomaly_score": round(random.uniform(65, 97), 2),
            "previous_failed_login_count": random.randint(3, 15),
            "beneficiary_trust_age_days": random.randint(0, 4),
        }
    elif pattern == "money_mule":
        tx["fraud_metadata"] = {
            "mule_account_flag": True,
            "funds_velocity_hours": round(random.uniform(0.5, 34), 2),
            "average_balance": round(random.uniform(20, 1400), 2),
            "incoming_outgoing_ratio": round(random.uniform(0.96, 1.02), 3),
            "beneficiary_count_30_days": random.randint(8, 40),
        }
    else:
        tx["fraud_metadata"] = {
            "bust_out_risk_score": round(random.uniform(55, 97), 2),
            "credit_building_duration_days": random.randint(80, 240),
            "days_since_last_payment": random.randint(35, 150),
            "credit_utilization_ratio": round(random.uniform(0.92, 1.0), 3),
        }

    return {
        "transaction_id": tx["transaction_id"],
        "customer_id": tx["customer_id"],
        "account_id": tx["account_id"],
        "fraud_pattern": tx["fraud_pattern"],
        "fraud_confidence": tx["fraud_confidence"],
        "fraud_metadata": tx["fraud_metadata"],
        "transaction_timestamp": tx["transaction_timestamp"],
    }


def inject_errors(tx: dict[str, Any], txs_out: list[dict[str, Any]]) -> dict[str, Any] | None:
    errors: list[str] = []
    meta: dict[str, Any] = {}

    if random.random() < ERROR_INJECTION["missing_merchant_name"]:
        tx["merchant_name"] = None
        errors.append("missing_merchant_name")

    if random.random() < ERROR_INJECTION["truncated_description"] and tx.get("description"):
        cut = random.choice([40, 50, 60])
        tx["description"] = str(tx["description"])[:cut]
        errors.append("truncated_description")
        meta["truncation_point"] = cut

    if random.random() < ERROR_INJECTION["incorrect_timestamp"]:
        t = pd.to_datetime(tx["transaction_timestamp"])
        mutation = random.choice(["midnight_default", "timezone_plus2", "future_plus1d"])
        if mutation == "midnight_default":
            t = t.replace(hour=0, minute=0, second=0)
        elif mutation == "timezone_plus2":
            t = t + timedelta(hours=2)
        else:
            t = t + timedelta(days=1)
        tx["transaction_timestamp"] = t
        errors.append("incorrect_timestamp")
        meta["timestamp_mutation"] = mutation

    if random.random() < ERROR_INJECTION["channel_misclassification"]:
        original = tx["channel"]
        mapping = {
            "pos": "online_banking",
            "mobile_banking_app": "online_banking",
            "contactless": "pos",
            "ewallet": "mobile_banking_app",
        }
        tx["channel"] = mapping.get(original, original)
        errors.append("channel_misclassification")
        meta["original_channel"] = original

    if random.random() < ERROR_INJECTION["amount_corruption"]:
        mutation = random.choice(["zero", "negative", "decimal_shift"])
        if mutation == "zero":
            tx["amount"] = 0.0
        elif mutation == "negative":
            tx["amount"] = -abs(float(tx["amount"]))
        else:
            tx["amount"] = round(float(tx["amount"]) * 100, 2)
        errors.append("amount_corruption")
        meta["amount_mutation"] = mutation

    if random.random() < ERROR_INJECTION["status_inconsistency"]:
        tx["status"] = random.choice(["pending", "failed"]) if tx["status"] == "completed" else "completed"
        errors.append("status_inconsistency")

    if random.random() < ERROR_INJECTION["duplicate_transaction"]:
        dup = tx.copy()
        dup["transaction_id"] = f"{tx['transaction_id']}-DUP"
        dt = pd.to_datetime(tx["transaction_timestamp"]) + timedelta(seconds=random.randint(1, 300))
        dup["transaction_timestamp"] = dt
        txs_out.append(dup)
        errors.append("duplicate_transaction")

    if not errors:
        return None

    return {
        "transaction_id": tx["transaction_id"],
        "customer_id": tx["customer_id"],
        "account_id": tx["account_id"],
        "error_types": errors,
        "error_metadata": meta,
        "transaction_timestamp": tx["transaction_timestamp"],
    }


def realism_scores(tx: dict[str, Any], violation_reason: str | None) -> tuple[float, float, float, float, float]:
    temporal = random.uniform(75, 99)
    spatial = random.uniform(70, 99) if violation_reason is None else random.uniform(20, 65)
    behavioral = random.uniform(72, 98)
    financial = random.uniform(70, 98)
    overall = (temporal + spatial + behavioral + financial) / 4.0
    return round(overall, 2), round(temporal, 2), round(spatial, 2), round(behavioral, 2), round(financial, 2)


def scheduled_transactions_for_month(year: int, month: int) -> pd.DataFrame:
    frames = []
    loan_files = resolve_table_files("loan_payment_transactions", year, month)
    debit_files = resolve_table_files("debit_order_transactions", year, month)

    for f in loan_files + debit_files:
        try:
            df = read_table(f)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    date_col = "transaction_date" if "transaction_date" in out.columns else None
    if date_col is None:
        return pd.DataFrame()

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[(out[date_col].dt.year == year) & (out[date_col].dt.month == month)].copy()
    return out


def select_customers(accounts: pd.DataFrame, customers: pd.DataFrame, target_customers: int | None) -> pd.DataFrame:
    merged = accounts.merge(customers, on="customer_id", how="inner")

    # If accounts and customers cannot be joined, fall back to synthetic account linkage.
    if merged.empty and not customers.empty:
        merged = customers.copy()
        merged["account_id"] = merged["customer_id"].astype(str).apply(lambda x: f"SYN-{x}")
        merged["account_status"] = "active"
        merged["status"] = "active"

    if merged.empty:
        return merged

    status_col = "account_status" if "account_status" in merged.columns else "status" if "status" in merged.columns else None
    if status_col is not None:
        filtered = merged[~merged[status_col].astype(str).str.lower().isin(["closed", "frozen"])]
        if not filtered.empty:
            merged = filtered

    merged = merged.drop_duplicates(subset=["customer_id"])
    if target_customers is not None and len(merged) > target_customers:
        merged = merged.sample(n=target_customers, random_state=year_seed())
    return merged.reset_index(drop=True)


def year_seed() -> int:
    return random.randint(1, 1_000_000)


def generate_month(year: int, month: int, target_customers: int | None = None) -> dict[str, Any]:
    random.seed(year * 100 + month)
    np.random.seed(year * 100 + month)

    accounts, customers = load_accounts_customers(year, month)
    if accounts.empty or customers.empty:
        print(f"Missing accounts/customers for {year}-{month:02d}")
        return {"generated": False, "reason": "missing_inputs"}

    locations_ref, branches_ref = load_reference_locations()
    province_rows = locations_ref.get("provinces", [])
    provinces = [p.get("name") for p in province_rows]
    cities_by_province = {p.get("name"): p.get("cities", []) for p in province_rows}

    population = select_customers(accounts, customers, target_customers)
    if population.empty:
        print(f"No eligible customers for {year}-{month:02d}")
        return {"generated": False, "reason": "no_population"}

    start = datetime(year, month, 1)
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).to_pydatetime().replace(hour=23, minute=59, second=59)

    tx_rows: list[dict[str, Any]] = []
    fraud_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    location_trace_rows: list[dict[str, Any]] = []

    # Add scheduled txns first to improve reconciliation with loan/debit pipelines.
    scheduled = scheduled_transactions_for_month(year, month)
    scheduled_daily_count: dict[tuple[str, str], int] = {}
    scheduled_daily_amt: dict[tuple[str, str], float] = {}

    tx_counter = 1
    if not scheduled.empty:
        for _, r in scheduled.iterrows():
            if pd.isna(r.get("account_id")):
                continue
            acc_id = str(r.get("account_id"))
            cust = population[population["account_id"].astype(str) == acc_id]
            if cust.empty:
                continue
            customer_id = str(cust.iloc[0]["customer_id"])

            tx_date = pd.to_datetime(r.get("transaction_date"), errors="coerce")
            if pd.isna(tx_date):
                continue
            tx_time = str(r.get("transaction_time", "06:00:00"))
            if len(tx_time.split(":")) == 2:
                tx_time = f"{tx_time}:00"
            tx_timestamp = pd.to_datetime(f"{tx_date.strftime('%Y-%m-%d')} {tx_time}", errors="coerce")
            if pd.isna(tx_timestamp):
                tx_timestamp = tx_date

            amount = float(pd.to_numeric(r.get("amount", 0), errors="coerce") or 0)
            dc = str(r.get("debit_credit", "Debit")).lower()
            amount_signed = -abs(amount) if dc == "debit" else abs(amount)

            key = (customer_id, tx_timestamp.strftime("%Y-%m-%d"))
            scheduled_daily_count[key] = scheduled_daily_count.get(key, 0) + 1
            scheduled_daily_amt[key] = scheduled_daily_amt.get(key, 0.0) + amount_signed

            tx_rows.append(
                {
                    "transaction_id": f"MTX{year}{month:02d}{tx_counter:09d}",
                    "batch_id": f"BATCH-{year}{month:02d}",
                    "generation_timestamp": pd.Timestamp.now(),
                    "transaction_timestamp": tx_timestamp,
                    "transaction_date": tx_timestamp.strftime("%Y-%m-%d"),
                    "transaction_time": tx_timestamp.strftime("%H:%M:%S"),
                    "customer_id": customer_id,
                    "account_id": acc_id,
                    "channel": "scheduled_payment",
                    "channel_metadata": {
                        "mandate_reference": r.get("debit_order_id") or r.get("loan_id"),
                        "frequency": r.get("frequency", "monthly"),
                        "retry_count_on_failure": 0,
                    },
                    "category": str(r.get("category", "scheduled_payment")).lower(),
                    "amount": round(amount, 2),
                    "debit_credit": "debit" if amount_signed < 0 else "credit",
                    "status": str(r.get("status", "completed")).lower(),
                    "description": str(r.get("description", "Scheduled payment")),
                    "merchant_name": r.get("merchant_name"),
                    "is_fraudulent": False,
                    "fraud_pattern": None,
                    "fraud_confidence": 0.0,
                    "fraud_metadata": {},
                    "has_error": False,
                    "error_types": [],
                    "error_metadata": {},
                    "network_latency_ms": random.randint(10, 200),
                    "authorization_time_ms": random.randint(50, 1200),
                    "third_party_timeout": random.random() < 0.01,
                    "stan": f"{random.randint(100000, 999999)}",
                    "rrn": f"{year}{month:02d}{random.randint(100000000, 999999999)}",
                    "source_table": "scheduled",
                }
            )
            tx_counter += 1

    states: dict[str, CustomerState] = {}
    behavioral_summary: dict[str, dict[str, Any]] = {}
    sessions: list[dict[str, Any]] = []

    for _, row in tqdm(population.iterrows(), total=len(population), desc=f"Multi-channel {year}-{month:02d}"):
        customer_id = str(row["customer_id"])
        account_id = str(row["account_id"])
        occupation = str(row.get("occupation", "Employed"))
        income = float(pd.to_numeric(row.get("annual_income", 300000), errors="coerce") or 300000) / 12.0

        province = str(row.get("branch_province", "") or row.get("province", ""))
        city = str(row.get("branch_city", "") or row.get("city", ""))
        if province not in provinces:
            province = random.choice(provinces)
        if city not in cities_by_province.get(province, []):
            city = random.choice(cities_by_province.get(province, ["Johannesburg"]))

        schedule_key = customer_schedule_type(occupation)
        home_location = random_location_for_province_city(province, city, radius_km=6.0)
        work_location = nearby_location(home_location, random.uniform(3.0, 18.0), "Work")
        social_location = nearby_location(home_location, random.uniform(2.0, 14.0), "Social")
        transport = random.choices(["walking", "bus", "taxi", "car"], weights=[0.08, 0.22, 0.45, 0.25], k=1)[0]
        primary_device = f"DEV-{customer_id[-6:]}-{random.randint(1000,9999)}"
        secondary_device = f"CARD-{customer_id[-6:]}-{random.randint(1000,9999)}"
        state = CustomerState(
            customer_id=customer_id,
            current_location=home_location,
            preferred_transport=transport,
            earliest_next_physical=start,
            anchor_locations={"home": home_location, "work": work_location, "social": social_location},
            primary_device=primary_device,
            secondary_device=secondary_device,
            home_ip_prefix=f"102.{random.randint(20, 90)}.{random.randint(1, 240)}",
            mobile_ip_prefix=f"102.{random.randint(100, 220)}.{random.randint(1, 240)}",
            channel_affinity=build_channel_affinity(schedule_key, income),
            category_affinity=build_category_affinity(schedule_key),
            spend_multiplier=random.uniform(0.82, 1.18),
        )
        states[customer_id] = state

        schedule = CUSTOMER_SCHEDULE_TEMPLATES[schedule_key]

        day = start
        customer_tx_count = 0
        customer_channel_counts: dict[str, int] = {}

        while day <= end:
            is_weekend = day.weekday() >= 5
            blocks = schedule["weekend"] if is_weekend and schedule["weekend"] else schedule["weekday"]

            state.session_id = f"SES-{customer_id}-{day.strftime('%Y%m%d')}"
            state.session_start = day.replace(hour=0, minute=0, second=0, microsecond=0)

            session_tx_ids: list[str] = []
            gps_traces: list[dict[str, Any]] = []
            session_transaction_events: list[dict[str, Any]] = []
            channels_used: set[str] = set()
            session_ip_addresses: set[str] = set()
            session_devices: set[str] = set()
            anomalies: list[str] = []

            for block in blocks:
                channels = block.get("channels", [])
                if not channels:
                    if random.random() >= float(block.get("probability", 0.0)):
                        continue
                    channels = ["mobile_banking_app"]

                tx_events = np.random.poisson(0.6 if is_weekend else 0.5)
                if tx_events == 0:
                    continue

                for _ in range(int(tx_events)):
                    tx_time = pick_tx_time_in_window(day, block["window"])
                    channel = weighted_choice(channels, state.channel_affinity)
                    if channel not in CHANNEL_REGISTRY:
                        continue

                    category = draw_category_for_customer(block.get("common_txns"), state)
                    amount = amount_for_state(category, income, state)
                    if amount > CHANNEL_REGISTRY[channel]["max_amount"]:
                        amount = round(CHANNEL_REGISTRY[channel]["max_amount"] * random.uniform(0.7, 0.98), 2)

                    tx_location = location_for_block(state, str(block.get("location", "home")))

                    is_injected_travel_anomaly = (
                        CHANNEL_REGISTRY[channel]["requires_presence"]
                        and state.last_physical_time is not None
                        and random.random() < TRAVEL_TIME_ANOMALY_RATE
                        and can_emit_anomaly(state, "travel_time_violation", tx_time)
                    )
                    if is_injected_travel_anomaly:
                        tx_location = distant_location(provinces, cities_by_province, state.last_physical_location or state.current_location)

                    ok, reason = can_place_transaction(state, tx_time, channel, tx_location)
                    if not ok:
                        if is_injected_travel_anomaly and reason == "travel_time_violation":
                            anomalies.append(reason)
                            anomaly_rows.append(
                                {
                                    "customer_id": customer_id,
                                    "account_id": account_id,
                                    "timestamp": tx_time,
                                    "anomaly_type": reason,
                                    "severity": "high",
                                }
                            )
                            state.last_anomaly_time[reason] = tx_time
                            state.monthly_anomaly_count[reason] = state.monthly_anomaly_count.get(reason, 0) + 1
                        else:
                            fallback_location = nearby_location(state.last_physical_location or state.current_location, 1.5)
                            ok, reason = can_place_transaction(state, tx_time, channel, fallback_location)
                            if not ok:
                                continue
                            tx_location = fallback_location

                    ext = external_context(tx_time)
                    network_type = network_type_for_channel(channel)
                    ip_address = ip_for_transaction(state, channel, network_type)
                    device_fingerprint = device_for_transaction(state, channel, is_injected_travel_anomaly)
                    status = "completed"
                    if ext["load_shedding_stage"] >= 4 and channel in ["pos", "atm"] and random.random() < 0.12:
                        status = "failed"

                    debit_credit = "debit"
                    if category in ["transfer"] and random.random() < 0.2:
                        debit_credit = "credit"

                    overall, temporal, spatial, behavioral, financial = realism_scores({}, reason)
                    behavioral = behavioral_score_for_tx(state, tx_time, channel, category, amount, is_injected_travel_anomaly)
                    overall = round((temporal + spatial + behavioral + financial) / 4.0, 2)
                    tx = {
                        "transaction_id": f"MTX{year}{month:02d}{tx_counter:09d}",
                        "batch_id": f"BATCH-{year}{month:02d}",
                        "generation_timestamp": pd.Timestamp.now(),
                        "transaction_timestamp": tx_time,
                        "transaction_date": tx_time.strftime("%Y-%m-%d"),
                        "transaction_time": tx_time.strftime("%H:%M:%S"),
                        "customer_id": customer_id,
                        "account_id": account_id,
                        "customer_session_id": state.session_id,
                        "customer_device_fingerprint": device_fingerprint,
                        "channel": channel,
                        "channel_metadata": {
                            "network_type": network_type,
                            "ip_address": ip_address,
                            "terminal_id": f"TERM-{random.randint(10000,99999)}" if channel in ["pos", "contactless"] else None,
                            "atm_id": f"ATM-{random.randint(1000,9999)}" if channel == "atm" else None,
                            "branch_code": random.choice(branches_ref).get("branch_id") if channel == "branch" else None,
                            "gps_coordinates": {"latitude": tx_location.latitude, "longitude": tx_location.longitude},
                            "session_duration_seconds": random.randint(8, 420),
                        },
                        "category": category,
                        "amount": amount,
                        "debit_credit": debit_credit,
                        "status": status,
                        "description": f"{category.title()} via {channel}",
                        "merchant_name": random.choice(["Shoprite", "Pick n Pay", "Spar", "Clicks", "Shell", "Checkers", None]),
                        "customer_location_state_before": {
                            "province": state.current_location.province,
                            "city": state.current_location.city,
                            "suburb": state.current_location.suburb,
                            "latitude": state.current_location.latitude,
                            "longitude": state.current_location.longitude,
                        },
                        "customer_location_state_after": {
                            "province": tx_location.province,
                            "city": tx_location.city,
                            "suburb": tx_location.suburb,
                            "latitude": tx_location.latitude,
                            "longitude": tx_location.longitude,
                        },
                        "customer_behavioral_score": behavioral,
                        "realism_score": overall,
                        "temporal_realism_score": temporal,
                        "spatial_realism_score": spatial,
                        "behavioral_realism_score": behavioral,
                        "financial_realism_score": financial,
                        "is_fraudulent": False,
                        "fraud_pattern": None,
                        "fraud_confidence": 0.0,
                        "fraud_metadata": {},
                        "has_error": False,
                        "error_types": [],
                        "error_metadata": {},
                        "account_balance_before": None,
                        "account_balance_after": None,
                        "daily_transaction_count_so_far": None,
                        "daily_total_amount_so_far": None,
                        "monthly_transaction_count_so_far": None,
                        "external_context": ext,
                        "network_latency_ms": random.randint(8, 500),
                        "authorization_time_ms": random.randint(45, 1800),
                        "third_party_timeout": random.random() < 0.01,
                        "stan": f"{random.randint(100000, 999999)}",
                        "rrn": f"{year}{month:02d}{random.randint(100000000, 999999999)}",
                        "source_table": "synthetic",
                    }

                    fraud_case = inject_fraud(tx, state)
                    if fraud_case is not None:
                        fraud_rows.append(fraud_case)
                    elif is_injected_travel_anomaly:
                        tx["is_fraudulent"] = True
                        tx["fraud_pattern"] = "account_takeover"
                        tx["fraud_confidence"] = round(random.uniform(0.72, 0.96), 3)
                        tx["fraud_metadata"] = {
                            "takeover_risk_score": round(random.uniform(78, 99), 2),
                            "device_fingerprint_changed": True,
                            "location_anomaly_score": round(random.uniform(85, 99), 2),
                            "behavioral_anomaly_score": round(random.uniform(70, 94), 2),
                            "previous_failed_login_count": random.randint(1, 5),
                            "beneficiary_trust_age_days": random.randint(0, 2),
                        }
                        fraud_rows.append(
                            {
                                "transaction_id": tx["transaction_id"],
                                "customer_id": tx["customer_id"],
                                "account_id": tx["account_id"],
                                "fraud_pattern": tx["fraud_pattern"],
                                "fraud_confidence": tx["fraud_confidence"],
                                "fraud_metadata": tx["fraud_metadata"],
                                "transaction_timestamp": tx["transaction_timestamp"],
                            }
                        )

                    error_case = inject_errors(tx, tx_rows)
                    if error_case is not None:
                        tx["has_error"] = True
                        tx["error_types"] = error_case["error_types"]
                        tx["error_metadata"] = error_case["error_metadata"]
                        error_rows.append(error_case)

                    tx_rows.append(tx)
                    tx_counter += 1
                    customer_tx_count += 1
                    customer_channel_counts[channel] = customer_channel_counts.get(channel, 0) + 1

                    # Update state.
                    state.last_channel_time[channel] = tx_time
                    if CHANNEL_REGISTRY[channel]["requires_presence"] and not is_injected_travel_anomaly:
                        state.current_location = tx_location
                        state.last_physical_time = tx_time
                        state.last_physical_location = tx_location
                        location_trace_rows.append(
                            {
                                "customer_id": customer_id,
                                "account_id": account_id,
                                "transaction_id": tx["transaction_id"],
                                "timestamp": tx_time,
                                "latitude": tx_location.latitude,
                                "longitude": tx_location.longitude,
                                "province": tx_location.province,
                                "city": tx_location.city,
                                "channel": channel,
                            }
                        )
                    else:
                        if not CHANNEL_REGISTRY[channel]["requires_presence"]:
                            state.last_digital_time = tx_time

                    session_tx_ids.append(tx["transaction_id"])
                    channels_used.add(channel)
                    session_ip_addresses.add(ip_address)
                    session_devices.add("card" if channel in ["pos", "contactless", "atm"] else "phone")
                    gps_traces.append(
                        {
                            "latitude": tx_location.latitude,
                            "longitude": tx_location.longitude,
                            "timestamp": tx_time,
                            "channel": channel,
                            "location_source": "terminal" if CHANNEL_REGISTRY[channel]["requires_presence"] else "device",
                        }
                    )
                    session_transaction_events.append(
                        {
                            "transaction_id": tx["transaction_id"],
                            "timestamp": tx_time,
                            "channel": channel,
                            "amount": amount,
                            "status": status,
                            "ip_address": ip_address,
                            "device_fingerprint": device_fingerprint,
                        }
                    )

            if session_tx_ids:
                session_transaction_events = sorted(session_transaction_events, key=lambda x: pd.to_datetime(x["timestamp"]))
                gps_traces = sorted(gps_traces, key=lambda x: pd.to_datetime(x["timestamp"]))
                session_start = pd.to_datetime(session_transaction_events[0]["timestamp"])
                session_end = pd.to_datetime(session_transaction_events[-1]["timestamp"])
                duration_minutes = max(1.0, (session_end - session_start).total_seconds() / 60.0)
                location_transitions = []
                for prev, curr in zip(gps_traces, gps_traces[1:]):
                    prev_loc = LocationPoint("", "", "", "", float(prev["latitude"]), float(prev["longitude"]))
                    curr_loc = LocationPoint("", "", "", "", float(curr["latitude"]), float(curr["longitude"]))
                    transition_minutes = max(1.0, (pd.to_datetime(curr["timestamp"]) - pd.to_datetime(prev["timestamp"])).total_seconds() / 60.0)
                    location_transitions.append(
                        {
                            "from_channel": prev["channel"],
                            "to_channel": curr["channel"],
                            "distance_km": round(distance_km(prev_loc, curr_loc), 2),
                            "elapsed_minutes": round(transition_minutes, 1),
                        }
                    )
                sessions.append(
                    {
                        "session_id": state.session_id,
                        "customer_id": customer_id,
                        "session_start": session_start,
                        "session_end": session_end,
                        "channels_used": sorted(list(channels_used)),
                        "devices_used": sorted(session_devices),
                        "ip_addresses": sorted(session_ip_addresses),
                        "gps_traces": gps_traces,
                        "transactions_in_session": [str(x["transaction_id"]) for x in session_transaction_events],
                        "transaction_events": session_transaction_events,
                        "session_realism_score": session_realism_score(len(session_tx_ids), channels_used, anomalies, duration_minutes),
                        "location_transitions": location_transitions,
                        "anomalies_detected": anomalies,
                        "authentication_events": auth_events_for_session(session_start, channels_used, anomalies),
                    }
                )

            day += timedelta(days=1)

        behavioral_summary[customer_id] = {
            "customer_id": customer_id,
            "account_id": account_id,
            "tx_count": customer_tx_count,
            "channel_preferences": customer_channel_counts,
            "time_pattern_adherence": round(random.uniform(0.70, 0.98), 3),
            "preferred_transport": transport,
            "home_province": province,
            "home_city": city,
        }

    if not tx_rows:
        print(f"No transactions generated for {year}-{month:02d}")
        return {"generated": False, "reason": "no_transactions"}

    tx_df = pd.DataFrame(tx_rows)
    tx_df["transaction_timestamp"] = pd.to_datetime(tx_df["transaction_timestamp"], errors="coerce")
    tx_df = tx_df.sort_values("transaction_timestamp").reset_index(drop=True)

    # Running counters and synthetic balances for metadata.
    tx_df["signed_amount"] = np.where(tx_df["debit_credit"].str.lower() == "debit", -tx_df["amount"].abs(), tx_df["amount"].abs())
    tx_df["date_key"] = tx_df["transaction_timestamp"].dt.strftime("%Y-%m-%d")

    tx_df["daily_transaction_count_so_far"] = tx_df.groupby(["customer_id", "date_key"]).cumcount() + 1
    tx_df["daily_total_amount_so_far"] = tx_df.groupby(["customer_id", "date_key"])["signed_amount"].cumsum().round(2)
    tx_df["monthly_transaction_count_so_far"] = tx_df.groupby(["customer_id"]).cumcount() + 1

    opening_balance = {str(cid): round(random.uniform(500, 15000), 2) for cid in tx_df["customer_id"].unique()}
    balances_before: list[float] = []
    balances_after: list[float] = []
    for _, row in tx_df.iterrows():
        cid = str(row["customer_id"])
        before = opening_balance[cid]
        after = before + float(row["signed_amount"])
        balances_before.append(round(before, 2))
        balances_after.append(round(after, 2))
        opening_balance[cid] = after

    tx_df["account_balance_before"] = balances_before
    tx_df["account_balance_after"] = balances_after

    enriched_behavioral_summary: list[dict[str, Any]] = []
    for cid, group in tx_df.groupby("customer_id"):
        base_profile = behavioral_summary.get(str(cid), {})
        synthetic_group = group[group["source_table"] == "synthetic"].copy()
        if synthetic_group.empty:
            synthetic_group = group.copy()
        channel_counts = synthetic_group["channel"].value_counts().to_dict()
        category_counts = synthetic_group["category"].value_counts().head(5).to_dict()
        active_days = int(synthetic_group["date_key"].nunique())
        tx_hours = synthetic_group["transaction_timestamp"].dt.hour
        preferred_hours = sorted([int(h) for h in tx_hours.mode().head(3).tolist()])
        digital_count = int(synthetic_group["channel"].isin(["mobile_banking_app", "online_banking", "ussd"]).sum())
        physical_count = int(synthetic_group["channel"].isin(["pos", "contactless", "atm", "branch"]).sum())
        weekend_count = int(synthetic_group["transaction_timestamp"].dt.weekday.ge(5).sum())
        failed_count = int((synthetic_group["status"] == "failed").sum())
        fraud_count = int(synthetic_group["is_fraudulent"].fillna(False).sum())
        enriched_behavioral_summary.append(
            {
                **base_profile,
                "tx_count": int(len(synthetic_group)),
                "active_days": active_days,
                "avg_transactions_per_active_day": round(float(len(synthetic_group) / max(1, active_days)), 2),
                "avg_transaction_amount": round(float(synthetic_group["amount"].mean()), 2),
                "median_transaction_amount": round(float(synthetic_group["amount"].median()), 2),
                "channel_preferences": channel_counts,
                "top_categories": category_counts,
                "preferred_transaction_hours": preferred_hours,
                "digital_ratio": round(digital_count / max(1, len(synthetic_group)), 3),
                "physical_ratio": round(physical_count / max(1, len(synthetic_group)), 3),
                "weekend_activity_ratio": round(weekend_count / max(1, len(synthetic_group)), 3),
                "failure_rate": round(failed_count / max(1, len(synthetic_group)), 4),
                "fraud_case_count": fraud_count,
                "avg_behavioral_score": round(float(synthetic_group["customer_behavioral_score"].mean()), 2),
                "time_pattern_adherence": round(float(synthetic_group["temporal_realism_score"].mean()) / 100.0, 3),
            }
        )

    channel_analytics = (
        tx_df.groupby("channel")
        .agg(
            tx_count=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            failed_count=("status", lambda s: int((s == "failed").sum())),
            fraud_count=("is_fraudulent", lambda s: int(pd.Series(s).fillna(False).sum())),
            unique_customers=("customer_id", "nunique"),
            avg_realism_score=("realism_score", "mean"),
            avg_authorization_time_ms=("authorization_time_ms", "mean"),
            timeout_count=("third_party_timeout", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
    )
    channel_analytics["failure_rate"] = (channel_analytics["failed_count"] / channel_analytics["tx_count"]).round(4)
    channel_analytics["fraud_rate"] = (channel_analytics["fraud_count"] / channel_analytics["tx_count"]).round(4)
    channel_analytics["timeout_rate"] = (channel_analytics["timeout_count"] / channel_analytics["tx_count"]).round(4)
    for col in ["total_amount", "avg_amount", "median_amount", "avg_realism_score", "avg_authorization_time_ms"]:
        channel_analytics[col] = channel_analytics[col].round(2)
    channel_analytics = channel_analytics.to_dict("records")

    out_dir = DATA_DIR / str(year) / f"{month:02d}"
    write_jsonl(out_dir / "transactions.jsonl", tx_df.drop(columns=["signed_amount", "date_key"]).to_dict("records"))
    write_jsonl(out_dir / "customer_sessions.jsonl", sessions)
    write_jsonl(out_dir / "fraud_cases.jsonl", fraud_rows)
    write_jsonl(out_dir / "error_log.jsonl", error_rows)
    write_json(out_dir / "channel_analytics.json", channel_analytics)
    write_jsonl(out_dir / "anomaly_flags.jsonl", anomaly_rows)
    write_jsonl(out_dir / "location_traces.jsonl", location_trace_rows)
    write_jsonl(out_dir / "behavioral_profiles.jsonl", enriched_behavioral_summary)
    deleted_sidecars = cleanup_sidecar_outputs(out_dir)

    print(f"Generated {len(tx_df)} transactions for {year}-{month:02d}")
    print(f"Outputs written to {out_dir}")
    if deleted_sidecars:
        print(f"Removed sidecar JSON outputs: {', '.join(deleted_sidecars)}")

    return {
        "generated": True,
        "transactions": int(len(tx_df)),
        "fraud_cases": int(len(fraud_rows)),
        "error_rows": int(len(error_rows)),
        "retained_outputs": sorted(RETAINED_OUTPUT_FILES),
        "deleted_sidecar_outputs": deleted_sidecars,
        "output_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultra-realistic multi-channel transaction generator (JSON/JSONL outputs)")
    parser.add_argument("--year", type=int, required=True, help="Target year")
    parser.add_argument("--month", type=int, help="Target month 1-12. Omit for full year")
    parser.add_argument("--target-customers", type=int, default=None, help="Optional cap on customer count for generation")
    args = parser.parse_args()

    if args.month is not None and (args.month < 1 or args.month > 12):
        raise ValueError("month must be in 1..12")

    if args.month is None:
        for m in range(1, 13):
            generate_month(args.year, m, args.target_customers)
    else:
        generate_month(args.year, args.month, args.target_customers)


if __name__ == "__main__":
    main()
