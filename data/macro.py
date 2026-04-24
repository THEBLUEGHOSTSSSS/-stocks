from __future__ import annotations

from typing import Any

from config import ETF_FLOW_PROXY_TICKERS, MACRO_TICKERS
from data.fetcher import batch_history, get_history


def _latest_change(history) -> tuple[float | None, float | None]:
    if history.empty:
        return None, None
    latest = float(history["close"].iloc[-1])
    baseline = float(history["close"].iloc[-2]) if len(history) > 1 else latest
    if baseline == 0:
        return latest, None
    return latest, ((latest / baseline) - 1.0) * 100.0


def get_macro_snapshot() -> dict[str, Any]:
    us10y_history = get_history(MACRO_TICKERS["us10y"], period="1mo", interval="1d")
    vix_history = get_history(MACRO_TICKERS["vix"], period="1mo", interval="1d")
    flow_histories = batch_history(ETF_FLOW_PROXY_TICKERS, period="2mo", interval="1d")

    us10y_value, us10y_change_pct = _latest_change(us10y_history)
    vix_value, vix_change_pct = _latest_change(vix_history)

    dollar_volume_ratios: list[float] = []
    for history in flow_histories.values():
        if history.empty or len(history) < 21:
            continue
        latest_dollar_volume = float(history["close"].iloc[-1] * history["volume"].iloc[-1])
        avg_dollar_volume = float((history["close"] * history["volume"]).iloc[-21:-1].mean())
        if avg_dollar_volume:
            dollar_volume_ratios.append(latest_dollar_volume / avg_dollar_volume)

    flow_proxy = sum(dollar_volume_ratios) / len(dollar_volume_ratios) if dollar_volume_ratios else None
    return {
        "us10y": {
            "value": us10y_value,
            "change_pct": us10y_change_pct,
        },
        "vix": {
            "value": vix_value,
            "change_pct": vix_change_pct,
        },
        "core_etf_flow_proxy": {
            "value": flow_proxy,
            "label": "ETF dollar-volume ratio vs 20d average",
        },
    }
