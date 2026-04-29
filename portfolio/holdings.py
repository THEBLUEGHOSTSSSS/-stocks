from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import HOLDINGS_FILE, HOLDINGS_HISTORY_FILE


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


def _holdings_signature(rows: list[dict[str, Any]]) -> str:
    normalized = normalize_holdings(rows)
    ordered = sorted(
        normalized,
        key=lambda item: (
            str(item.get("ticker") or ""),
            str(item.get("side") or ""),
            float(item.get("shares") or 0.0),
            float(item.get("cost_basis") or 0.0),
            str(item.get("notes") or ""),
        ),
    )
    return json.dumps(ordered, ensure_ascii=False, sort_keys=True)


def _normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    holdings = snapshot.get("holdings")
    if not isinstance(holdings, list):
        return None

    normalized_holdings = normalize_holdings(holdings)
    captured_at = str(snapshot.get("captured_at") or "").strip() or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source = str(snapshot.get("source") or "unknown").strip() or "unknown"
    signature = str(snapshot.get("signature") or _holdings_signature(normalized_holdings))
    return {
        "captured_at": captured_at,
        "source": source,
        "signature": signature,
        "holdings": normalized_holdings,
    }


def _aggregate_holdings(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in normalize_holdings(rows):
        ticker = str(row["ticker"])
        item = aggregated.setdefault(
            ticker,
            {
                "ticker": ticker,
                "shares": 0.0,
                "cost_value": 0.0,
                "side_weights": {},
                "notes": set(),
            },
        )
        shares = float(row.get("shares") or 0.0)
        side = str(row.get("side") or "LONG")
        item["shares"] += shares
        item["cost_value"] += shares * float(row.get("cost_basis") or 0.0)
        item["side_weights"][side] = float(item["side_weights"].get(side) or 0.0) + shares
        note = str(row.get("notes") or "").strip()
        if note:
            item["notes"].add(note)

    collapsed: dict[str, dict[str, Any]] = {}
    for ticker, item in aggregated.items():
        side_weights = item["side_weights"]
        side = "MIXED"
        if len(side_weights) == 1:
            side = next(iter(side_weights))
        elif side_weights:
            side = max(side_weights.items(), key=lambda pair: pair[1])[0]
        shares = float(item["shares"])
        cost_basis = float(item["cost_value"] / shares) if shares > 0 else 0.0
        collapsed[ticker] = {
            "ticker": ticker,
            "side": side,
            "shares": shares,
            "cost_basis": cost_basis,
            "notes": " / ".join(sorted(item["notes"])),
        }
    return collapsed


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


def load_holdings_history(path: Path = HOLDINGS_HISTORY_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    history: list[dict[str, Any]] = []
    for snapshot in raw:
        if not isinstance(snapshot, dict):
            continue
        normalized_snapshot = _normalize_snapshot(snapshot)
        if normalized_snapshot is not None:
            history.append(normalized_snapshot)
    return history


def save_holdings(rows: list[dict[str, Any]], path: Path = HOLDINGS_FILE) -> list[dict[str, Any]]:
    normalized = normalize_holdings(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def save_holdings_snapshot(
    rows: list[dict[str, Any]],
    path: Path = HOLDINGS_HISTORY_FILE,
    source: str = "manual_save",
    max_snapshots: int = 180,
) -> dict[str, Any]:
    normalized = normalize_holdings(rows)
    history = load_holdings_history(path)
    signature = _holdings_signature(normalized)
    latest = history[-1] if history else None
    if latest and str(latest.get("signature") or "") == signature:
        return latest

    snapshot = {
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source or "unknown"),
        "signature": signature,
        "holdings": normalized,
    }
    history.append(snapshot)
    if max_snapshots > 0:
        history = history[-max_snapshots:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def resolve_reference_snapshot(
    current_rows: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_signature = _holdings_signature(current_rows)
    for snapshot in reversed(history):
        if str(snapshot.get("signature") or "") != current_signature:
            return snapshot
    return None


def build_holdings_change_records(
    current_rows: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_map = _aggregate_holdings(current_rows)
    previous_map = _aggregate_holdings(previous_rows)
    records: list[dict[str, Any]] = []

    for ticker in sorted(set(current_map) | set(previous_map)):
        current = current_map.get(ticker)
        previous = previous_map.get(ticker)
        current_side = current.get("side") if current else None
        previous_side = previous.get("side") if previous else None
        current_shares = float((current or {}).get("shares") or 0.0)
        previous_shares = float((previous or {}).get("shares") or 0.0)
        share_delta = current_shares - previous_shares
        current_cost_basis = float((current or {}).get("cost_basis") or 0.0) if current else None
        previous_cost_basis = float((previous or {}).get("cost_basis") or 0.0) if previous else None
        current_notes = str((current or {}).get("notes") or "") if current else ""
        previous_notes = str((previous or {}).get("notes") or "") if previous else ""

        if previous is None:
            status = "新增"
        elif current is None:
            status = "已清仓"
        elif current_side != previous_side:
            status = "方向切换"
        elif share_delta > 1e-9:
            status = "加仓"
        elif share_delta < -1e-9:
            status = "减仓"
        elif abs(float(current_cost_basis or 0.0) - float(previous_cost_basis or 0.0)) > 1e-9 or current_notes != previous_notes:
            status = "信息更新"
        else:
            status = "未变化"

        records.append(
            {
                "ticker": ticker,
                "status": status,
                "previous_side": previous_side,
                "current_side": current_side,
                "previous_shares": previous_shares if previous is not None else None,
                "current_shares": current_shares if current is not None else None,
                "share_delta": share_delta,
                "previous_cost_basis": previous_cost_basis,
                "current_cost_basis": current_cost_basis,
                "previous_notes": previous_notes,
                "current_notes": current_notes,
            }
        )

    return sorted(records, key=lambda item: (item["status"] == "未变化", str(item["ticker"])))


def build_snapshot_order_action_records(
    orders: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_map = _aggregate_holdings(previous_rows)
    records: list[dict[str, Any]] = []

    for order in orders:
        ticker = str(order.get("Ticker") or "").strip().upper()
        if not ticker:
            continue
        previous = previous_map.get(ticker)
        records.append(
            {
                "ticker": ticker,
                "previous_side": (previous or {}).get("side"),
                "previous_shares": (previous or {}).get("shares"),
                "previous_cost_basis": (previous or {}).get("cost_basis"),
                "previous_notes": (previous or {}).get("notes"),
                "signal": order.get("Signal"),
                "position_side": order.get("Position_Side"),
                "trade_direction": order.get("Trade_Direction"),
                "suggested_shares": order.get("Suggested_Shares"),
                "reference_price": order.get("Reference_Price"),
                "gate_reason": order.get("Gate_Reason"),
                "signal_mode": order.get("Signal_Mode"),
            }
        )

    return records


def build_holdings_history_summary(history: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in reversed(history[-limit:]):
        holdings = snapshot.get("holdings") or []
        tickers = [str(item.get("ticker") or "") for item in holdings if item.get("ticker")]
        rows.append(
            {
                "captured_at": snapshot.get("captured_at"),
                "source": snapshot.get("source"),
                "position_count": int(len(holdings)),
                "tickers": "、".join(tickers) if tickers else "空仓",
            }
        )
    return rows


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
