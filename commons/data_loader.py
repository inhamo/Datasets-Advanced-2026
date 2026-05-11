from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

COMMONS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_json(name: str) -> Any:
    path = COMMONS_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_occupations_data() -> dict[str, Any]:
    return load_json("occupations.json")


def get_locations_data() -> dict[str, Any]:
    return load_json("locations.json")


def get_location_profiles_data() -> dict[str, Any]:
    return load_json("location_profiles.json")


def get_branches_data() -> list[dict[str, Any]]:
    return load_json("branches.json")


def get_names_data() -> dict[str, Any]:
    return load_json("names_by_ethnicity.json")


def get_names_by_country_data() -> dict[str, Any]:
    return load_json("names_by_country.json")


def get_phone_rules_data() -> dict[str, Any]:
    return load_json("phone_rules.json")
