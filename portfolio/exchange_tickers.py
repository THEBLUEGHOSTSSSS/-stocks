from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from io import StringIO
from pathlib import Path

import requests

from config import EXCHANGE_TICKERS_FILE
from portfolio.custom_tickers import normalize_ticker_symbol, save_custom_ticker_metadata


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

OTHER_LISTED_EXCHANGE_LABELS = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "V": "IEX",
    "Z": "Cboe BZX",
    "M": "NYSE Texas",
}

_SKIPPED_SECURITY_NAME_TOKENS = (
    " warrant",
    " rights",
    " right",
    " units",
    " unit",
)


def _download_text(url: str, timeout: float) -> str:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    response.raise_for_status()
    if not response.encoding:
        response.encoding = "utf-8"
    return response.text


def _clean_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _should_skip_security(name: str) -> bool:
    lowered = f" {_clean_text(name).lower()} "
    return any(token in lowered for token in _SKIPPED_SECURITY_NAME_TOKENS)


def _build_category(exchange_label: str, is_etf: bool) -> str:
    base = f"交易所导入/{exchange_label}"
    return f"{base} ETF" if is_etf else base


def _parse_nasdaq_listed(text: str) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(StringIO(text), delimiter="|")
    for row in reader:
        raw_symbol = _clean_text(row.get("Symbol"))
        if not raw_symbol or raw_symbol.startswith("File Creation Time"):
            continue
        if str(row.get("Test Issue") or "").upper() == "Y":
            continue

        symbol = normalize_ticker_symbol(raw_symbol)
        name = _clean_text(row.get("Security Name")) or symbol
        if not symbol or _should_skip_security(name):
            continue

        is_etf = str(row.get("ETF") or "").upper() == "Y"
        metadata[symbol] = {
            "name": name,
            "category": _build_category("NASDAQ", is_etf),
        }
    return metadata


def _parse_other_listed(text: str) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(StringIO(text), delimiter="|")
    for row in reader:
        raw_symbol = _clean_text(row.get("NASDAQ Symbol") or row.get("ACT Symbol"))
        if not raw_symbol or raw_symbol.startswith("File Creation Time"):
            continue
        if str(row.get("Test Issue") or "").upper() == "Y":
            continue

        symbol = normalize_ticker_symbol(raw_symbol)
        name = _clean_text(row.get("Security Name")) or symbol
        if not symbol or _should_skip_security(name):
            continue

        exchange_code = _clean_text(row.get("Exchange")).upper()
        exchange_label = OTHER_LISTED_EXCHANGE_LABELS.get(exchange_code, exchange_code or "其他交易所")
        is_etf = str(row.get("ETF") or "").upper() == "Y"
        metadata[symbol] = {
            "name": name,
            "category": _build_category(exchange_label, is_etf),
        }
    return metadata


def import_public_exchange_tickers(path: Path = EXCHANGE_TICKERS_FILE, timeout: float = 20.0) -> dict[str, object]:
    nasdaq_metadata = _parse_nasdaq_listed(_download_text(NASDAQ_LISTED_URL, timeout=timeout))
    other_metadata = _parse_other_listed(_download_text(OTHER_LISTED_URL, timeout=timeout))
    merged_metadata = {**other_metadata, **nasdaq_metadata}
    saved_metadata = save_custom_ticker_metadata(path, merged_metadata)

    exchange_counter = Counter()
    etf_count = 0
    for entry in saved_metadata.values():
        category = str(entry.get("category") or "")
        if category.endswith(" ETF"):
            etf_count += 1
            exchange_label = category.removeprefix("交易所导入/").removesuffix(" ETF")
        else:
            exchange_label = category.removeprefix("交易所导入/")
        if exchange_label:
            exchange_counter[exchange_label] += 1

    return {
        "path": str(path),
        "saved_count": len(saved_metadata),
        "etf_count": etf_count,
        "source_counts": {
            "nasdaq_listed": len(nasdaq_metadata),
            "other_listed": len(other_metadata),
        },
        "exchange_counts": dict(sorted(exchange_counter.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="一次性导入公开交易所股票代码清单")
    parser.add_argument("--path", default=str(EXCHANGE_TICKERS_FILE), help="导入结果 JSON 文件路径")
    parser.add_argument("--timeout", type=float, default=20.0, help="下载超时时间，单位秒")
    args = parser.parse_args()

    summary = import_public_exchange_tickers(Path(args.path), timeout=args.timeout)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()