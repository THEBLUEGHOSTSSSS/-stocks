from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CUSTOM_TICKER_CATEGORY = "自定义代码"
_VALID_TICKER_RE = re.compile(r"^[A-Z0-9.\-\^]{1,15}$")


def normalize_ticker_symbol(value: Any) -> str:
    symbol = str(value or "").upper().strip()
    return symbol if _VALID_TICKER_RE.fullmatch(symbol) else ""


def _coerce_metadata_entry(symbol: str, raw_entry: Any) -> dict[str, str] | None:
    if not symbol:
        return None
    if not isinstance(raw_entry, dict):
        return {
            "name": symbol,
            "category": DEFAULT_CUSTOM_TICKER_CATEGORY,
        }
    name = str(raw_entry.get("name") or symbol).strip() or symbol
    category = str(raw_entry.get("category") or DEFAULT_CUSTOM_TICKER_CATEGORY).strip() or DEFAULT_CUSTOM_TICKER_CATEGORY
    return {
        "name": name,
        "category": category,
    }


def load_custom_ticker_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, list):
        iterator = ((item.get("ticker"), item) for item in raw if isinstance(item, dict))
    else:
        return {}

    metadata: dict[str, dict[str, str]] = {}
    for raw_symbol, raw_entry in iterator:
        symbol = normalize_ticker_symbol(raw_symbol)
        entry = _coerce_metadata_entry(symbol, raw_entry)
        if symbol and entry is not None:
            metadata[symbol] = entry
    return metadata


def save_custom_ticker_metadata(path: Path, metadata: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    sanitized: dict[str, dict[str, str]] = {}
    for raw_symbol, raw_entry in metadata.items():
        symbol = normalize_ticker_symbol(raw_symbol)
        entry = _coerce_metadata_entry(symbol, raw_entry)
        if symbol and entry is not None:
            sanitized[symbol] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sanitized


def register_custom_ticker(
    path: Path,
    ticker: Any,
    name: str | None = None,
    category: str | None = None,
) -> dict[str, dict[str, str]]:
    metadata = load_custom_ticker_metadata(path)
    symbol = normalize_ticker_symbol(ticker)
    if not symbol:
        return metadata

    existing = metadata.get(symbol, {})
    metadata[symbol] = {
        "name": str(name or existing.get("name") or symbol).strip() or symbol,
        "category": str(category or existing.get("category") or DEFAULT_CUSTOM_TICKER_CATEGORY).strip() or DEFAULT_CUSTOM_TICKER_CATEGORY,
    }
    return save_custom_ticker_metadata(path, metadata)