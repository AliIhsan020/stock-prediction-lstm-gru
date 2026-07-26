from datetime import date

import pytest

from bist100_forecasting.instruments import (
    BIST100_CATALOG_EFFECTIVE_FROM,
    BIST100_CATALOG_EFFECTIVE_TO,
    BIST100_CODES,
    BIST100_INSTRUMENTS,
    get_bist100_instrument,
    normalize_bist_code,
    to_yahoo_symbol,
)


def test_catalog_contains_exactly_one_hundred_unique_instruments() -> None:
    assert len(BIST100_INSTRUMENTS) == 100
    assert len(BIST100_CODES) == 100
    assert len(set(BIST100_CODES)) == 100


def test_catalog_reflects_third_quarter_2026_review() -> None:
    assert {"ESEN", "IEYHO", "ODINE"} <= set(BIST100_CODES)
    assert {"AGHOL", "TABGD", "TUREX"}.isdisjoint(BIST100_CODES)
    assert BIST100_CATALOG_EFFECTIVE_FROM == date(2026, 7, 1)
    assert BIST100_CATALOG_EFFECTIVE_TO == date(2026, 9, 30)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("asels", "ASELS"),
        ("  thyao  ", "THYAO"),
        ("bimas.is", "BIMAS"),
        (" XU100.IS ", "XU100"),
    ],
)
def test_normalize_bist_code(value: str, expected: str) -> None:
    assert normalize_bist_code(value) == expected


@pytest.mark.parametrize("value", ["", ".IS", "AB", "THYAO.US", "THY AO"])
def test_normalize_bist_code_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid BIST code"):
        normalize_bist_code(value)


def test_normalize_bist_code_rejects_non_string_value() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        normalize_bist_code(123)  # type: ignore[arg-type]


def test_to_yahoo_symbol_is_idempotent_for_yahoo_suffix() -> None:
    assert to_yahoo_symbol("EREGL") == "EREGL.IS"
    assert to_yahoo_symbol("eregl.is") == "EREGL.IS"


def test_get_bist100_instrument_returns_display_metadata() -> None:
    instrument = get_bist100_instrument("sise.is")

    assert instrument.code == "SISE"
    assert instrument.name == "Şişecam"
    assert instrument.yahoo_symbol == "SISE.IS"
    assert instrument.label == "SISE — Şişecam"


def test_get_bist100_instrument_rejects_non_constituent() -> None:
    with pytest.raises(ValueError, match="not in the current BIST 100 catalogue"):
        get_bist100_instrument("XYZAB")
