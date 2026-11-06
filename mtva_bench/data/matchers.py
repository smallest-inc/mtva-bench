"""How a tool-call argument is compared with a scenario's matcher.

Strings are compared in normalised form, so formatting, case and the script of the digits
never decide a match.
"""

import contextlib
import re
import unicodedata
from datetime import date, datetime
from typing import Any

from mtva_bench.data.schema import Matcher, Scenario

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d.%m.%Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%A %B %d %Y",
    "%A %d %B %Y",
)
ORDINAL = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
PHONE_MIN_DIGITS = 7
PHONE_MAX_PREFIX = 3


def ascii_digits(text: str) -> str:
    return "".join(str(unicodedata.digit(char)) if char.isdigit() else char for char in text)


def normalize(value: Any) -> str:
    """NFKC, casefolded, any script's digits as ASCII, punctuation and symbols as spaces,
    whitespace collapsed."""
    text = ascii_digits(unicodedata.normalize("NFKC", str(value)).casefold())
    text = "".join(" " if unicodedata.category(char)[0] in "PSZ" else char for char in text)
    return " ".join(text.split())


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = ascii_digits(unicodedata.normalize("NFKC", str(value))).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def equal(expected: Any, actual: Any) -> bool:
    """Numbers compare numerically, an ISO date as a date in any common form, and anything
    with a digit in it, such as an ID or a phone number, ignoring its separators, since a
    caller reading out "MBR-0815533-01" says no hyphens. Anything else compares as normalised
    text."""
    expected_number, actual_number = number(expected), number(actual)
    if expected_number is not None and actual_number is not None:
        return expected_number == actual_number
    if isinstance(expected, str) and ISO_DATE.fullmatch(expected.strip()):
        return date.fromisoformat(expected.strip()) in dates(actual)
    left, right = normalize(expected), normalize(actual)
    if any(char.isdigit() for char in left) and any(char.isdigit() for char in right):
        return left.replace(" ", "") == right.replace(" ", "")
    return left == right


def dates(value: Any) -> set[date]:
    """The dates a value can be read as. An ISO date-time gives its date as written, in its
    own offset, so "2026-06-12T21:00:00+05:30" is June 12."""
    text = ascii_digits(unicodedata.normalize("NFKC", str(value))).replace(",", " ")
    text = " ".join(ORDINAL.sub("", text).split())
    found = set()
    with contextlib.suppress(ValueError):
        found.add(datetime.fromisoformat(text.removesuffix("Z")).date())
    for pattern in DATE_FORMATS:
        try:
            found.add(datetime.strptime(text, pattern).date())
        except ValueError:
            continue
    return found


def same_phone(expected: Any, actual: Any) -> bool:
    digits = [
        "".join(char for char in ascii_digits(str(value)) if char.isdigit())
        for value in (expected, actual)
    ]
    short, long = sorted(digits, key=len)
    return (
        len(short) >= PHONE_MIN_DIGITS
        and long.endswith(short)
        and len(long) - len(short) <= PHONE_MAX_PREFIX
    )


def matches(matcher: Matcher, actual: Any, scenario: Scenario) -> bool:
    if actual is None:
        return False
    if matcher.present:
        return normalize(actual) != ""
    if matcher.equals is not None:
        return equal(matcher.equals, actual)
    if matcher.ref is not None:
        return equal(scenario.resolve(matcher.ref), actual)
    if matcher.any_of is not None:
        return any(equal(option, actual) for option in matcher.any_of)
    if matcher.range is not None:
        value = number(actual)
        return value is not None and matcher.range[0] <= value <= matcher.range[1]
    if matcher.date is not None:
        return date.fromisoformat(matcher.date) in dates(actual)
    assert matcher.phone is not None
    return same_phone(matcher.phone, actual)
