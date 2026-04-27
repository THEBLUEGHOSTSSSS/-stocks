from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import HOLDINGS_FILE


def _normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    ticker = str(row.get("ticker", "")).strip().upper()
    if not ticker:
        return None

    side = str(row.get("side", "LONG") or "LONG").strip().upper()
    if side not in {"LONG", "SHORT"}:
        side = "LONG"

    try:
        shares = abs(float(row.get("shares", 0.0) or 0.0))
    except (TypeError, ValueError):
        shares = 0.0

    try:
        cost_basis = float(row.get("cost_basis", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost_basis = 0.0

    return {
        "ticker": ticker,
        "side": side,
        "shares": shares,
        "cost_basis": cost_basis,
        "notes": str(row.get("notes", "") or ""),
    }


def normalize_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize_row(row)
        if item is not None:
            normalized.append(item)
    return normalized


def load_holdings(path: Path = HOLDINGS_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    return normalize_holdings(raw)


def save_holdings(rows: list[dict[str, Any]], path: Path = HOLDINGS_FILE) -> list[dict[str, Any]]:
    normalized = normalize_holdings(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def holdings_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ticker", "side", "shares", "cost_basis", "notes"])
    return pd.DataFrame(rows)


def frame_to_holdings(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame.fillna("").to_dict(orient="records")
    return normalize_holdings(rows)


def enrich_holdings_with_quotes(
    holdings: list[dict[str, Any]],
    quotes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for holding in holdings:
        quote = quotes.get(holding["ticker"], {})
        market_price = quote.get("close")
        shares = float(holding["shares"])
        cost_basis = float(holding["cost_basis"])
        side = str(holding.get("side") or "LONG").upper()
        direction = 1.0 if side == "LONG" else -1.0
        market_value = market_price * shares if market_price is not None else None
        cost_value = cost_basis * shares
        pnl = (market_price - cost_basis) * shares * direction if market_price is not None else None
        pnl_pct = (((market_price - cost_basis) / cost_basis) * 100.0 * direction) if market_price is not None and cost_basis else None
        net_exposure = market_value * direction if market_value is not None else None
        enriched.append(
            {
                **holding,
                "market_price": market_price,
                "market_value": market_value,
                "net_exposure": net_exposure,
                "cost_value": cost_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "market_session": quote.get("session_label"),
                "session_change_pct": quote.get("session_change_pct"),
                "quote_as_of": quote.get("as_of"),
            }
        )
    return enriched
