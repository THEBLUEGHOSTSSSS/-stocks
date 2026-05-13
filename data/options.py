from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from config import RISK_FREE_RATE
from data.fetcher import register_fetch_warning
from models.options_advisor import implied_volatility_from_price


IV_HISTORY_RETENTION_DAYS = 366
DEFAULT_ATM_STRIKES_PER_SIDE = 6
DEFAULT_MIN_DAYS_TO_EXPIRY = 7
DEFAULT_MAX_DAYS_TO_EXPIRY = 45
MIN_REASONABLE_IV = 0.03
MAX_REASONABLE_IV = 4.0


def _empty_summary(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "spot": None,
        "expiries": [],
        "near_term_expiry": None,
        "near_term_days_to_expiry": None,
        "call_volume_total": 0.0,
        "put_volume_total": 0.0,
        "call_put_volume_ratio": None,
        "top_calls": [],
        "top_puts": [],
        "near_term_atm_calls": [],
        "near_term_atm_puts": [],
        "near_term_atm_strikes": [],
        "atm_iv_call": None,
        "atm_iv_put": None,
        "atm_iv_current": None,
        "atm_iv_current_source": None,
        "atm_iv_history_52w": [],
        "atm_iv_history_points_52w": [],
        "atm_iv_cache_size": 0,
    }


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(resolved):
        return None
    return resolved


def _coerce_int(value: Any) -> int | None:
    resolved = _coerce_float(value)
    if resolved is None:
        return None
    return int(resolved)


def _clean_iv(value: Any) -> float | None:
    resolved = _coerce_float(value)
    if resolved is None:
        return None
    if resolved < MIN_REASONABLE_IV or resolved > MAX_REASONABLE_IV:
        return None
    return resolved


def _top_strikes(frame: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    ranked = frame.sort_values("volume", ascending=False).head(top_n)
    return [
        {
            "strike": float(row["strike"]),
            "volume": float(row.get("volume", 0.0) or 0.0),
            "open_interest": float(row.get("openInterest", 0.0) or 0.0),
            "last_price": float(row.get("lastPrice", 0.0) or 0.0),
        }
        for _, row in ranked.iterrows()
    ]


def _cache_file_path() -> Path:
    if sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base_dir = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base_dir / "us-stock-quant" / "atm_iv_history.json"


def _load_iv_history_cache() -> dict[str, list[dict[str, Any]]]:
    cache_path = _cache_file_path()
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): list(value) for key, value in payload.items() if isinstance(value, list)}


def _save_iv_history_cache(cache: dict[str, list[dict[str, Any]]]) -> None:
    cache_path = _cache_file_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def _normalize_iv_history_points(points: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=IV_HISTORY_RETENTION_DAYS)
    normalized: list[dict[str, Any]] = []
    for raw in list(points or []):
        date_str = str(raw.get("date") or "").strip()
        if not date_str:
            continue
        try:
            point_date = datetime.fromisoformat(date_str).date()
        except ValueError:
            continue
        if point_date < cutoff:
            continue
        atm_iv = _clean_iv(raw.get("atm_iv"))
        if atm_iv is None:
            continue
        normalized.append(
            {
                "date": point_date.isoformat(),
                "atm_iv": round(atm_iv, 6),
                "spot": round(_coerce_float(raw.get("spot")) or 0.0, 4),
                "expiry": str(raw.get("expiry") or ""),
                "days_to_expiry": _coerce_int(raw.get("days_to_expiry")) or 0,
            }
        )
    normalized.sort(key=lambda item: item["date"])
    deduped: dict[str, dict[str, Any]] = {}
    for point in normalized:
        deduped[point["date"]] = point
    return list(deduped.values())


def _load_ticker_iv_history(ticker: str) -> list[dict[str, Any]]:
    cache = _load_iv_history_cache()
    return _normalize_iv_history_points(cache.get(str(ticker).upper()))


def _update_ticker_iv_history(
    ticker: str,
    *,
    atm_iv: float | None,
    spot: float | None,
    expiry: str | None,
    days_to_expiry: int | None,
) -> list[dict[str, Any]]:
    normalized_ticker = str(ticker).upper()
    cache = _load_iv_history_cache()
    points = _normalize_iv_history_points(cache.get(normalized_ticker))
    if atm_iv is None:
        return points

    today = datetime.now(timezone.utc).date().isoformat()
    updated_point = {
        "date": today,
        "atm_iv": round(float(atm_iv), 6),
        "spot": round(float(spot or 0.0), 4),
        "expiry": str(expiry or ""),
        "days_to_expiry": int(days_to_expiry or 0),
    }
    points = [point for point in points if point["date"] != today]
    points.append(updated_point)
    points = _normalize_iv_history_points(points)
    cache[normalized_ticker] = points
    _save_iv_history_cache(cache)
    return points


def _resolve_spot_price(ticker_obj: yf.Ticker, ticker: str) -> float | None:
    try:
        history = ticker_obj.history(period="5d", interval="1d", auto_adjust=False)
    except Exception as exc:
        register_fetch_warning("期权标的现价", ticker, exc)
        history = pd.DataFrame()
    if not history.empty:
        for column in ("Close", "Adj Close"):
            if column in history.columns:
                close = pd.to_numeric(history[column], errors="coerce").dropna()
                if not close.empty:
                    return float(close.iloc[-1])
    try:
        fast_info = ticker_obj.fast_info
    except Exception:
        fast_info = None
    if fast_info:
        for key in ("lastPrice", "regularMarketPrice", "previousClose"):
            spot = _coerce_float(getattr(fast_info, key, None) if not isinstance(fast_info, dict) else fast_info.get(key))
            if spot is not None and spot > 0.0:
                return spot
    return None


def _select_near_term_expiry(
    expiries: list[str],
    *,
    min_days_to_expiry: int,
    max_days_to_expiry: int,
) -> tuple[str | None, int | None]:
    today = datetime.now(timezone.utc).date()
    normalized: list[tuple[str, int]] = []
    for expiry in expiries:
        try:
            expiry_date = datetime.fromisoformat(str(expiry)).date()
        except ValueError:
            continue
        days_to_expiry = (expiry_date - today).days
        if days_to_expiry < 0:
            continue
        normalized.append((str(expiry), days_to_expiry))

    if not normalized:
        return None, None

    preferred = [item for item in normalized if min_days_to_expiry <= item[1] <= max_days_to_expiry]
    if preferred:
        return preferred[0]

    positive = [item for item in normalized if item[1] > 0]
    if positive:
        return positive[0]
    return normalized[0]


def _select_atm_window_strikes(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    *,
    spot: float,
    strikes_per_side: int,
) -> list[float]:
    strike_series: list[pd.Series] = []
    for frame in (calls, puts):
        if frame.empty or "strike" not in frame.columns:
            continue
        numeric_strikes = pd.to_numeric(frame["strike"], errors="coerce").dropna()
        if not numeric_strikes.empty:
            strike_series.append(numeric_strikes)
    if not strike_series:
        return []

    combined = pd.concat(strike_series, ignore_index=True).drop_duplicates()
    ranked = pd.DataFrame({"strike": combined})
    ranked["distance"] = (ranked["strike"] - spot).abs()
    ranked = ranked.sort_values(["distance", "strike"], ascending=[True, True])
    selected = ranked.head(max(2 * strikes_per_side + 1, 5))["strike"].tolist()
    return [float(value) for value in sorted(selected)]


def _resolve_contract_premium(row: pd.Series) -> tuple[float | None, str | None, float | None]:
    bid = _coerce_float(row.get("bid"))
    ask = _coerce_float(row.get("ask"))
    last_price = _coerce_float(row.get("lastPrice"))
    if bid is not None and ask is not None and bid > 0.0 and ask > 0.0 and ask >= bid:
        return (bid + ask) / 2.0, "mid", (bid + ask) / 2.0
    if last_price is not None and last_price > 0.0:
        return last_price, "last", (bid + ask) / 2.0 if bid is not None and ask is not None and ask >= bid and ask > 0.0 else None
    if bid is not None and bid > 0.0:
        return bid, "bid", None
    if ask is not None and ask > 0.0:
        return ask, "ask", None
    return None, None, None


def _serialize_last_trade_date(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError):
        return None


def _build_contract_rows(
    frame: pd.DataFrame,
    *,
    option_type: str,
    expiry: str,
    days_to_expiry: int,
    spot: float,
    selected_strikes: list[float],
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    working = frame.copy()
    working["strike"] = pd.to_numeric(working.get("strike"), errors="coerce")
    working = working.dropna(subset=["strike"])
    if selected_strikes:
        working = working[working["strike"].isin(selected_strikes)]
    if working.empty:
        return []

    working = working.sort_values("strike", ascending=True)
    time_to_expiry = max(days_to_expiry / 365.0, 1.0 / 365.0)
    rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        strike = _coerce_float(row.get("strike"))
        if strike is None or strike <= 0.0:
            continue

        premium, premium_source, mid_price = _resolve_contract_premium(row)
        model_iv = None
        if premium is not None and premium > 0.0:
            model_iv = _clean_iv(
                implied_volatility_from_price(
                    premium,
                    strike,
                    spot,
                    RISK_FREE_RATE,
                    time_to_expiry,
                    option_type,
                )
            )
        raw_iv = _clean_iv(row.get("impliedVolatility"))
        effective_iv = model_iv if model_iv is not None else raw_iv
        iv_source = "premium_inversion" if model_iv is not None else ("raw_chain_iv" if raw_iv is not None else None)
        bid = _coerce_float(row.get("bid"))
        ask = _coerce_float(row.get("ask"))
        last_price = _coerce_float(row.get("lastPrice"))
        distance_dollar = abs(strike - spot)
        distance_pct = distance_dollar / max(spot, 1e-6)
        rows.append(
            {
                "contract_symbol": str(row.get("contractSymbol") or ""),
                "option_type": option_type,
                "expiry": expiry,
                "days_to_expiry": int(days_to_expiry),
                "strike": round(strike, 2),
                "spot": round(spot, 2),
                "last_trade_date": _serialize_last_trade_date(row.get("lastTradeDate")),
                "last_price": round(last_price, 4) if last_price is not None else None,
                "bid": round(bid, 4) if bid is not None else None,
                "ask": round(ask, 4) if ask is not None else None,
                "mid_price": round(mid_price, 4) if mid_price is not None else None,
                "premium": round(premium, 4) if premium is not None else None,
                "premium_source": premium_source,
                "volume": _coerce_int(row.get("volume")) or 0,
                "open_interest": _coerce_int(row.get("openInterest")) or 0,
                "in_the_money": bool(row.get("inTheMoney")),
                "distance_dollar": round(distance_dollar, 2),
                "distance_pct": round(distance_pct, 6),
                "raw_implied_volatility": round(raw_iv, 6) if raw_iv is not None else None,
                "implied_volatility": round(effective_iv, 6) if effective_iv is not None else None,
                "implied_volatility_source": iv_source,
            }
        )
    return rows


def _aggregate_atm_iv(rows: list[dict[str, Any]]) -> float | None:
    weighted_values: list[tuple[float, float]] = []
    for row in rows:
        implied_volatility = _clean_iv(row.get("implied_volatility"))
        if implied_volatility is None:
            continue
        distance_pct = _coerce_float(row.get("distance_pct")) or 0.0
        open_interest = _coerce_float(row.get("open_interest")) or 0.0
        volume = _coerce_float(row.get("volume")) or 0.0
        weight = 1.0 / max(distance_pct, 0.01)
        weight *= 1.0 + min((open_interest + volume) / 10000.0, 0.5)
        if row.get("premium_source") == "mid":
            weight *= 1.1
        weighted_values.append((implied_volatility, weight))

    if not weighted_values:
        return None
    values = np.asarray([value for value, _ in weighted_values], dtype=float)
    weights = np.asarray([weight for _, weight in weighted_values], dtype=float)
    return float(np.average(values, weights=weights))


@lru_cache(maxsize=256)
def get_near_term_options_summary(
    ticker: str,
    max_expiries: int = 2,
    top_n: int = 5,
    min_days_to_expiry: int = DEFAULT_MIN_DAYS_TO_EXPIRY,
    max_days_to_expiry: int = DEFAULT_MAX_DAYS_TO_EXPIRY,
    atm_strikes_per_side: int = DEFAULT_ATM_STRIKES_PER_SIDE,
) -> dict[str, Any]:
    normalized_ticker = str(ticker or "").strip().upper()
    summary = _empty_summary(normalized_ticker)
    ticker_obj = yf.Ticker(normalized_ticker)

    try:
        all_expiries = list(ticker_obj.options)
    except Exception as exc:
        register_fetch_warning("期权链", normalized_ticker, exc)
        return summary
    if not all_expiries:
        return summary

    summary["expiries"] = list(all_expiries[:max(max_expiries, 1)])
    near_term_expiry, near_term_days_to_expiry = _select_near_term_expiry(
        all_expiries,
        min_days_to_expiry=min_days_to_expiry,
        max_days_to_expiry=max_days_to_expiry,
    )
    summary["near_term_expiry"] = near_term_expiry
    summary["near_term_days_to_expiry"] = near_term_days_to_expiry
    summary["spot"] = _resolve_spot_price(ticker_obj, normalized_ticker)

    expiry_to_chain: dict[str, Any] = {}
    expiry_targets = list(dict.fromkeys([*summary["expiries"], *( [near_term_expiry] if near_term_expiry else [] )]))
    calls: list[pd.DataFrame] = []
    puts: list[pd.DataFrame] = []
    for expiry in expiry_targets:
        try:
            chain = ticker_obj.option_chain(expiry)
        except Exception as exc:
            register_fetch_warning("期权链详情", f"{normalized_ticker} {expiry}", exc)
            continue
        expiry_to_chain[expiry] = chain
        if expiry in summary["expiries"]:
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
    summary["call_volume_total"] = total_call_volume
    summary["put_volume_total"] = total_put_volume
    summary["call_put_volume_ratio"] = total_call_volume / total_put_volume if total_put_volume > 0.0 else None
    summary["top_calls"] = _top_strikes(all_calls, top_n)
    summary["top_puts"] = _top_strikes(all_puts, top_n)

    near_term_chain = expiry_to_chain.get(near_term_expiry or "")
    if near_term_chain is not None and summary["spot"] is not None and near_term_days_to_expiry is not None:
        selected_strikes = _select_atm_window_strikes(
            near_term_chain.calls,
            near_term_chain.puts,
            spot=float(summary["spot"]),
            strikes_per_side=atm_strikes_per_side,
        )
        summary["near_term_atm_strikes"] = [round(value, 2) for value in selected_strikes]
        summary["near_term_atm_calls"] = _build_contract_rows(
            near_term_chain.calls,
            option_type="call",
            expiry=str(near_term_expiry),
            days_to_expiry=int(near_term_days_to_expiry),
            spot=float(summary["spot"]),
            selected_strikes=selected_strikes,
        )
        summary["near_term_atm_puts"] = _build_contract_rows(
            near_term_chain.puts,
            option_type="put",
            expiry=str(near_term_expiry),
            days_to_expiry=int(near_term_days_to_expiry),
            spot=float(summary["spot"]),
            selected_strikes=selected_strikes,
        )

    summary["atm_iv_call"] = _aggregate_atm_iv(list(summary["near_term_atm_calls"]))
    summary["atm_iv_put"] = _aggregate_atm_iv(list(summary["near_term_atm_puts"]))
    summary["atm_iv_current"] = _aggregate_atm_iv(
        [*list(summary["near_term_atm_calls"]), *list(summary["near_term_atm_puts"])]
    )
    summary["atm_iv_current_source"] = "premium_inversion_atm_window" if summary["atm_iv_current"] is not None else None

    iv_history_points = _update_ticker_iv_history(
        normalized_ticker,
        atm_iv=summary["atm_iv_current"],
        spot=_coerce_float(summary["spot"]),
        expiry=near_term_expiry,
        days_to_expiry=near_term_days_to_expiry,
    )
    if not iv_history_points:
        iv_history_points = _load_ticker_iv_history(normalized_ticker)
    summary["atm_iv_history_points_52w"] = iv_history_points
    summary["atm_iv_history_52w"] = [float(point["atm_iv"]) for point in iv_history_points if _clean_iv(point.get("atm_iv")) is not None]
    summary["atm_iv_cache_size"] = len(summary["atm_iv_history_52w"])
    return summary