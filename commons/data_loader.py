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


def get_behavioral_profiles_data() -> dict[str, Any]:
    return load_json("behavioral_profiles.json")


def get_customer_scenarios_data() -> dict[str, Any]:
    return load_json("customer_scenarios.json")


def get_company_profiles_data() -> dict[str, Any]:
    return load_json("company_profiles.json")


def get_retail_bank_products_data() -> dict[str, Any]:
    return load_json("retail_bank_products_sa.json")


def get_branch_codes_by_city_data() -> dict[str, Any]:
    return load_json("branch_codes_by_city.json")


def get_sa_banking_realism_data() -> dict[str, Any]:
    return load_json("sa_banking_realism.json")


def get_realism_source_catalog_data() -> dict[str, Any]:
    return load_json("realism_source_catalog.json")
