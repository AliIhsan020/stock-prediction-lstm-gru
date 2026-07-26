"""Current BIST 100 instrument catalogue and ticker helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

BIST100_CATALOG_EFFECTIVE_FROM = date(2026, 7, 1)
BIST100_CATALOG_EFFECTIVE_TO = date(2026, 9, 30)
BIST100_REVIEW_URL = (
    "https://borsaistanbul.com/en/announcement/15483/bist-stock-indices-periodic-review"
)

_BIST_CODE_PATTERN = re.compile(r"[A-Z0-9]{3,6}")


def normalize_bist_code(value: str) -> str:
    """Return an uppercase BIST code without Yahoo Finance's ``.IS`` suffix."""
    if not isinstance(value, str):
        raise TypeError("BIST code must be a string.")

    code = value.strip().upper()
    if code.endswith(".IS"):
        code = code[:-3]
    if not _BIST_CODE_PATTERN.fullmatch(code):
        raise ValueError(f"Invalid BIST code: {value!r}.")
    return code


def to_yahoo_symbol(value: str) -> str:
    """Convert a BIST code to its Yahoo Finance ticker."""
    return f"{normalize_bist_code(value)}.IS"


@dataclass(frozen=True, slots=True)
class Instrument:
    """A selectable BIST 100 constituent."""

    code: str
    name: str

    @property
    def yahoo_symbol(self) -> str:
        """Return the instrument's Yahoo Finance ticker."""
        return to_yahoo_symbol(self.code)

    @property
    def label(self) -> str:
        """Return a compact user-interface label."""
        return f"{self.code} — {self.name}"


_CATALOG_ROWS = (
    ("ASELS", "Aselsan"),
    ("BIMAS", "BİM Birleşik Mağazalar"),
    ("TUPRS", "Tüpraş"),
    ("THYAO", "Türk Hava Yolları"),
    ("AKBNK", "Akbank"),
    ("DSTKF", "Destek Finans Faktoring"),
    ("EREGL", "Ereğli Demir Çelik"),
    ("KCHOL", "Koç Holding"),
    ("ASTOR", "Astor Enerji"),
    ("ODINE", "Odine Solutions"),
    ("YKBNK", "Yapı Kredi Bankası"),
    ("ISCTR", "İş Bankası (C)"),
    ("TCELL", "Turkcell"),
    ("SAHOL", "Sabancı Holding"),
    ("KTLEV", "Katılımevim"),
    ("CCOLA", "Coca-Cola İçecek"),
    ("GARAN", "Garanti Bankası"),
    ("SASA", "Sasa Polyester"),
    ("SISE", "Şişecam"),
    ("MGROS", "Migros"),
    ("FROTO", "Ford Otosan"),
    ("TRALT", "Türk Altın İşletmeleri"),
    ("TAVHL", "TAV Havalimanları"),
    ("ENKAI", "Enka İnşaat"),
    ("AEFES", "Anadolu Efes"),
    ("PGSUS", "Pegasus"),
    ("EKGYO", "Emlak Konut GYO"),
    ("TOASO", "Tofaş"),
    ("MPARK", "MLP Sağlık"),
    ("TURSG", "Türkiye Sigorta"),
    ("IEYHO", "Işıklar Enerji ve Yapı Holding"),
    ("KRDMD", "Kardemir (D)"),
    ("GUBRF", "Gübre Fabrikaları"),
    ("RALYH", "Ral Yatırım Holding"),
    ("KUYAS", "Kuyas Yatırım"),
    ("PETKM", "Petkim"),
    ("PASEU", "Pasifik Eurasia"),
    ("EUPWR", "Europower Enerji"),
    ("AKSEN", "Aksa Enerji"),
    ("ENJSA", "Enerjisa"),
    ("TTKOM", "Türk Telekom"),
    ("MAVI", "Mavi Giyim"),
    ("VAKBN", "Vakıfbank"),
    ("TRMET", "TR Anadolu Metal Madencilik"),
    ("SARKY", "Sarkuysan"),
    ("HALKB", "Halkbank"),
    ("CIMSA", "Çimsa"),
    ("OYAKC", "Oyak Çimento"),
    ("DOHOL", "Doğan Holding"),
    ("BTCIM", "Batıçim"),
    ("ANSGR", "Anadolu Sigorta"),
    ("BRSAN", "Borusan Mannesmann"),
    ("ENERY", "Enerya Enerji"),
    ("CVKMD", "CVK Maden"),
    ("DOAS", "Doğuş Otomotiv"),
    ("ISMEN", "İş Yatırım"),
    ("ALARK", "Alarko Holding"),
    ("ULKER", "Ülker Bisküvi"),
    ("AKSA", "Aksa Akrilik"),
    ("GESAN", "Girişim Elektrik"),
    ("BSOKE", "Batı Söke Çimento"),
    ("MAGEN", "Margün Enerji"),
    ("GLRMK", "Gülermak Ağır Sanayi"),
    ("SKBNK", "Şekerbank"),
    ("HEKTS", "Hektaş"),
    ("SOKM", "Şok Marketler"),
    ("TSKB", "TSKB"),
    ("KLRHO", "Kiler Holding"),
    ("EFOR", "Efor Yatırım"),
    ("TKFEN", "Tekfen Holding"),
    ("ARCLK", "Arçelik"),
    ("CWENE", "CW Enerji"),
    ("OTKAR", "Otokar"),
    ("ECILC", "Eczacıbaşı İlaç"),
    ("PSGYO", "Pasifik GYO"),
    ("BALSU", "Balsu Gıda"),
    ("GENIL", "Gen İlaç"),
    ("MIATK", "Mia Teknoloji"),
    ("CANTE", "Çan2 Termik"),
    ("ODAS", "Odaş Elektrik"),
    ("DAPGM", "DAP Gayrimenkul"),
    ("TRENJ", "TR Doğal Enerji Kaynakları"),
    ("GRSEL", "Gürsel Taşımacılık"),
    ("BRYAT", "Borusan Yatırım"),
    ("FENER", "Fenerbahçe Futbol"),
    ("IZENR", "İzdemir Enerji"),
    ("GRTHO", "Graintürk Holding"),
    ("ALTNY", "Altınay Savunma"),
    ("BERA", "Bera Holding"),
    ("PATEK", "Pasifik Teknoloji"),
    ("PAHOL", "Pasifik Holding"),
    ("GSRAY", "Galatasaray Sportif"),
    ("TUKAS", "Tukaş"),
    ("QUAGR", "QUA Granite"),
    ("ZOREN", "Zorlu Enerji"),
    ("EUREN", "Europen Endüstri"),
    ("OBAMS", "Oba Makarnacılık"),
    ("VESTL", "Vestel"),
    ("REEDR", "Reeder"),
    ("ESEN", "Esenboğa Elektrik"),
)

BIST100_INSTRUMENTS = tuple(
    Instrument(code=code, name=name) for code, name in _CATALOG_ROWS
)
BIST100_CODES = tuple(instrument.code for instrument in BIST100_INSTRUMENTS)
_BIST100_BY_CODE = {instrument.code: instrument for instrument in BIST100_INSTRUMENTS}


def get_bist100_instrument(value: str) -> Instrument:
    """Return a current constituent or reject codes outside the catalogue."""
    code = normalize_bist_code(value)
    try:
        return _BIST100_BY_CODE[code]
    except KeyError as error:
        raise ValueError(f"{code} is not in the current BIST 100 catalogue.") from error


def _validate_catalog() -> None:
    if len(BIST100_INSTRUMENTS) != 100:
        raise RuntimeError("BIST 100 catalogue must contain exactly 100 instruments.")
    if len(set(BIST100_CODES)) != len(BIST100_CODES):
        raise RuntimeError("BIST 100 catalogue contains duplicate instrument codes.")
    for instrument in BIST100_INSTRUMENTS:
        normalize_bist_code(instrument.code)


_validate_catalog()
