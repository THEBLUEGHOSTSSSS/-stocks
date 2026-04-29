from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import (
    ACCOUNT_FILE,
    DEFAULT_LONG_TARGET_PCT,
    DEFAULT_SHORT_MARGIN_RATIO,
    DEFAULT_SHORT_TARGET_PCT,
    SHORT_MARGIN_CONFIG,
)
from data.fetcher import get_short_borrow_metrics


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_short_margin_profile(
    borrow_metrics: dict[str, Any] | None,
    base_margin_ratio: float = DEFAULT_SHORT_MARGIN_RATIO,
) -> dict[str, float]:
    metrics = borrow_metrics or {}
    borrow_fee_pct = max(_safe_float(metrics.get("latest_fee_pct"), 0.0), 0.0)
    initial_base = max(
        _safe_float(base_margin_ratio, DEFAULT_SHORT_MARGIN_RATIO),
        float(SHORT_MARGIN_CONFIG["reg_t_initial_pct"]),
    )
    initial_margin_pct = min(initial_base + borrow_fee_pct / 100.0, 1.0)
    maintenance_margin_pct = min(
        max(float(SHORT_MARGIN_CONFIG["maintenance_pct"]), initial_margin_pct - 0.2),
        initial_margin_pct,
    )
    return {
        "borrow_fee_pct": borrow_fee_pct,
        "initial_margin_pct": initial_margin_pct,
        "maintenance_margin_pct": maintenance_margin_pct,
    }


def normalize_account_state(raw: dict[str, Any] | None) -> dict[str, float]:
    state = raw or {}
    total_equity = max(_safe_float(state.get("total_equity"), 0.0), 0.0)
    short_margin_ratio = min(max(_safe_float(state.get("short_margin_ratio"), DEFAULT_SHORT_MARGIN_RATIO), 0.2), 1.0)
    max_long_position_pct = min(max(_safe_float(state.get("max_long_position_pct"), DEFAULT_LONG_TARGET_PCT), 0.01), 0.5)
    max_short_position_pct = min(max(_safe_float(state.get("max_short_position_pct"), DEFAULT_SHORT_TARGET_PCT), 0.01), 0.5)
    return {
        "total_equity": total_equity,
        "short_margin_ratio": short_margin_ratio,
        "max_long_position_pct": max_long_position_pct,
        "max_short_position_pct": max_short_position_pct,
    }


def load_account_state(path: Path = ACCOUNT_FILE) -> dict[str, float]:
    if not path.exists():
        return normalize_account_state(None)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return normalize_account_state(None)
    if not isinstance(raw, dict):
        return normalize_account_state(None)
    return normalize_account_state(raw)


def save_account_state(state: dict[str, Any], path: Path = ACCOUNT_FILE) -> dict[str, float]:
    normalized = normalize_account_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def compute_account_overview(
    holdings_enriched: list[dict[str, Any]],
    account_state: dict[str, Any],
    borrow_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_state = normalize_account_state(account_state)
    long_market_value = sum(
        float(item.get("market_value") or 0.0)
        for item in holdings_enriched
        if str(item.get("side") or "LONG").upper() == "LONG"
    )
    short_market_value = sum(
        float(item.get("market_value") or 0.0)
        for item in holdings_enriched
        if str(item.get("side") or "LONG").upper() == "SHORT"
    )
    total_pnl = sum(float(item.get("pnl") or 0.0) for item in holdings_enriched)
    net_exposure = sum(float(item.get("net_exposure") or 0.0) for item in holdings_enriched)
    base_margin_ratio = float(normalized_state["short_margin_ratio"])
    borrow_metrics_lookup = borrow_metrics or {}
    short_margin_used = 0.0
    short_maintenance_requirement = 0.0
    weighted_borrow_fee = 0.0
    ticker_margin_map: dict[str, float] = {}
    ticker_maintenance_margin_map: dict[str, float] = {}
    borrow_metrics_summary: dict[str, Any] = {
        "avg_borrow_fee_pct": 0.0,
        "htb_count": 0,
        "prohibit_count": 0,
        "tickers": {},
    }
    for item in holdings_enriched:
        if str(item.get("side") or "LONG").upper() != "SHORT":
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        market_value = float(item.get("market_value") or 0.0)
        metrics = borrow_metrics_lookup.get(ticker)
        if metrics is None and ticker:
            metrics = get_short_borrow_metrics(ticker)
        margin_profile = compute_short_margin_profile(metrics, base_margin_ratio=base_margin_ratio)
        initial_margin_pct = float(margin_profile["initial_margin_pct"])
        maintenance_margin_pct = float(margin_profile["maintenance_margin_pct"])
        borrow_fee_pct = float(margin_profile["borrow_fee_pct"])
        short_margin_used += market_value * initial_margin_pct
        short_maintenance_requirement += market_value * maintenance_margin_pct
        weighted_borrow_fee += market_value * borrow_fee_pct
        ticker_margin_map[ticker] = round(initial_margin_pct, 4)
        ticker_maintenance_margin_map[ticker] = round(maintenance_margin_pct, 4)
        is_hard_to_borrow = bool((metrics or {}).get("is_hard_to_borrow"))
        prohibit_short = bool((metrics or {}).get("prohibit_short"))
        if is_hard_to_borrow:
            borrow_metrics_summary["htb_count"] += 1
        if prohibit_short:
            borrow_metrics_summary["prohibit_count"] += 1
        borrow_metrics_summary["tickers"][ticker] = {
            "fee_pct": round(borrow_fee_pct, 2),
            "available_shares": int(_safe_float((metrics or {}).get("latest_available_shares"), 0.0)),
            "is_hard_to_borrow": is_hard_to_borrow,
            "prohibit_short": prohibit_short,
            "initial_margin_pct": round(initial_margin_pct * 100.0, 2),
            "maintenance_margin_pct": round(maintenance_margin_pct * 100.0, 2),
            "source": (metrics or {}).get("source"),
            "error": (metrics or {}).get("error"),
        }

    effective_margin_ratio = (short_margin_used / short_market_value) if short_market_value else base_margin_ratio
    maintenance_margin_ratio = (
        short_maintenance_requirement / short_market_value
        if short_market_value
        else float(SHORT_MARGIN_CONFIG["maintenance_pct"])
    )
    capital_in_use = long_market_value + short_margin_used
    total_equity = max(float(normalized_state["total_equity"]), capital_in_use)
    idle_cash = max(total_equity - capital_in_use, 0.0)
    short_buying_power = idle_cash / effective_margin_ratio if effective_margin_ratio else 0.0
    market_funds = long_market_value + short_market_value
    utilization = (capital_in_use / total_equity) if total_equity else 0.0
    if short_market_value:
        borrow_metrics_summary["avg_borrow_fee_pct"] = round(weighted_borrow_fee / short_market_value, 2)
    return {
        **normalized_state,
        "base_short_margin_ratio": base_margin_ratio,
        "short_margin_ratio": effective_margin_ratio,
        "short_maintenance_ratio": maintenance_margin_ratio,
        "market_funds": market_funds,
        "total_pnl": total_pnl,
        "long_market_value": long_market_value,
        "short_market_value": short_market_value,
        "short_margin_used": short_margin_used,
        "short_maintenance_requirement": short_maintenance_requirement,
        "capital_in_use": capital_in_use,
        "idle_cash": idle_cash,
        "short_buying_power": short_buying_power,
        "net_exposure": net_exposure,
        "utilization": utilization,
        "ticker_margin_map": ticker_margin_map,
        "ticker_maintenance_margin_map": ticker_maintenance_margin_map,
        "borrow_metrics_summary": borrow_metrics_summary,
    }