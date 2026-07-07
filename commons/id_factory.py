from __future__ import annotations

import string


ALPHANUMERIC_VALUES = {char: str(index + 10) for index, char in enumerate(string.ascii_uppercase)}


def _mod97(value: str) -> int:
    converted = []
    for char in value.upper():
        if char.isdigit():
            converted.append(char)
        elif char.isalpha():
            converted.append(ALPHANUMERIC_VALUES[char])
    if not converted:
        return 0
    return int("".join(converted)) % 97


def check_digits(base: str) -> str:
    return f"{98 - _mod97(base + '00'):02d}"


def compact_id(*parts: object) -> str:
    return "".join(str(part).upper().replace("-", "").replace(" ", "") for part in parts if part is not None)


def make_customer_id(customer_kind: str, year: int, month: int | None, sequence: int) -> str:
    kind_code = {
        "individual": "P",
        "person": "P",
        "minor": "M",
        "company": "B",
        "business": "B",
        "organisation": "O",
        "organization": "O",
        "trust": "T",
        "stokvel": "S",
    }.get(customer_kind.lower(), "C")
    period = f"{year}{month or 0:02d}"
    base = compact_id("CIF", "ZA", kind_code, period, f"{sequence:07d}")
    return f"CIF-ZA-{kind_code}-{period}-{sequence:07d}-{check_digits(base)}"


def make_account_id(year: int, month: int | None, sequence: int) -> str:
    period = f"{year}{month or 0:02d}"
    base = compact_id("KSBACC", period, f"{sequence:08d}")
    return f"KSB-ACC-{period}-{sequence:08d}-{check_digits(base)}"


def make_application_id(prefix: str, year: int, month: int | None, sequence: int) -> str:
    period = f"{year}{month or 0:02d}"
    code = compact_id(prefix)[:6] or "APP"
    base = compact_id(code, period, f"{sequence:08d}")
    return f"{code}-{period}-{sequence:08d}-{check_digits(base)}"


def make_loan_id(year: int, month: int | None, sequence: int) -> str:
    period = f"{year}{month or 0:02d}"
    base = compact_id("KSBLOAN", period, f"{sequence:08d}")
    return f"KSB-LOAN-{period}-{sequence:08d}-{check_digits(base)}"
