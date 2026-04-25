from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from data.fetcher import register_fetch_warning


def _top_strikes(frame: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    ranked = frame.sort_values("volume", ascending=False).head(top_n)
    return [
        {
            "strike": float(row["strike"]),
            "volume": float(row["volume"]),
            "open_interest": float(row.get("openInterest", 0.0)),
            "last_price": float(row.get("lastPrice", 0.0)),
        }
        for _, row in ranked.iterrows()
    ]


def get_near_term_options_summary(
    ticker: str,
    max_expiries: int = 2,
    top_n: int = 5,
) -> dict[str, Any]:
    ticker_obj = yf.Ticker(ticker)
    try:
        expiries = list(ticker_obj.options[:max_expiries])
    except Exception as exc:
        register_fetch_warning("期权链", ticker, exc)
        expiries = []
    if not expiries:
        return {
            "ticker": ticker,
            "expiries": [],
            "call_put_volume_ratio": None,
            "top_calls": [],
            "top_puts": [],
        }

    calls: list[pd.DataFrame] = []
    puts: list[pd.DataFrame] = []
    for expiry in expiries:
        try:
            chain = ticker_obj.option_chain(expiry)
        except Exception as exc:
            register_fetch_warning("期权链详情", f"{ticker} {expiry}", exc)
            continue
        if not chain.calls.empty:
            call_frame = chain.calls.copy()
            call_frame["expiry"] = expiry
            calls.append(call_frame)
        if not chain.puts.empty:
            put_frame = chain.puts.copy()
            put_frame["expiry"] = expiry
            puts.append(put_frame)

    all_calls = pd.concat(calls, ignore_index=True) if calls else pd.DataFrame()
    all_puts = pd.concat(puts, ignore_index=True) if puts else pd.DataFrame()

    total_call_volume = float(all_calls.get("volume", pd.Series(dtype=float)).fillna(0.0).sum())
    total_put_volume = float(all_puts.get("volume", pd.Series(dtype=float)).fillna(0.0).sum())
    volume_ratio = None
    if total_put_volume > 0.0:
        volume_ratio = total_call_volume / total_put_volume

    return {
        "ticker": ticker,
        "expiries": expiries,
        "call_volume_total": total_call_volume,
        "put_volume_total": total_put_volume,
        "call_put_volume_ratio": volume_ratio,
        "top_calls": _top_strikes(all_calls, top_n),
        "top_puts": _top_strikes(all_puts, top_n),
    }
