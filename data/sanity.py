from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class PriceSanityError(ValueError):
    """Raised when market data breaches hard sanity thresholds."""


@dataclass(frozen=True, slots=True)
class PriceSanityConfig:
    max_abs_daily_move: float = 0.50
    max_intraday_range: float = 0.60
    max_quote_gap: float = 0.50
    min_price: float = 0.01


@dataclass(frozen=True, slots=True)
class HoldingCostSanityConfig:
    max_cost_basis_gap: float = 0.80
    min_price: float = 0.01


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


def enforce_history_sanity(
    frame: pd.DataFrame,
    ticker: str,
    *,
    price_source: str = "",
    config: PriceSanityConfig | None = None,
) -> pd.DataFrame:
    cfg = config or PriceSanityConfig()
    if frame.empty:
        return frame

    clean = frame.copy()
    for column in ["open", "high", "low", "close"]:
        if column not in clean.columns:
            raise PriceSanityError(f"{ticker} 缺少 {column} 字段，无法做价格校验")
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=["open", "high", "low", "close"])
    if clean.empty:
        return clean

    if bool((clean[["open", "high", "low", "close"]] <= cfg.min_price).any().any()):
        raise PriceSanityError(f"{ticker} {price_source or 'history'} 出现非正常低价数据")

    previous_close = clean["close"].shift(1)
    daily_move = (clean["close"] / previous_close - 1.0).abs().replace([np.inf, -np.inf], np.nan)
    if bool((daily_move > cfg.max_abs_daily_move).fillna(False).any()):
        worst = float(daily_move.max(skipna=True) or 0.0)
        raise PriceSanityError(
            f"{ticker} {price_source or 'history'} 触发绝对极值熔断，单日偏移 {worst:.1%} 超过 {cfg.max_abs_daily_move:.0%}"
        )

    intraday_range = (clean["high"] / clean["low"] - 1.0).abs().replace([np.inf, -np.inf], np.nan)
    if bool((intraday_range > cfg.max_intraday_range).fillna(False).any()):
        worst = float(intraday_range.max(skipna=True) or 0.0)
        raise PriceSanityError(
            f"{ticker} {price_source or 'history'} 日内振幅 {worst:.1%} 超过 {cfg.max_intraday_range:.0%}"
        )

    return clean


def enforce_quote_sanity(
    quote: dict[str, Any],
    ticker: str,
    *,
    price_source: str = "",
    config: PriceSanityConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PriceSanityConfig()
    if not quote:
        return quote

    active_price = _coerce_float(
        quote.get("reference_price")
        or quote.get("close")
        or quote.get("regular_market_price")
        or quote.get("market_price")
    )
    previous_close = _coerce_float(quote.get("previous_close") or quote.get("regular_close"))

    if active_price is not None and active_price <= cfg.min_price:
        raise PriceSanityError(f"{ticker} {price_source or 'quote'} 出现非正常价格 {active_price}")

    if active_price is not None and previous_close is not None and previous_close > cfg.min_price:
        gap = abs(active_price / previous_close - 1.0)
        if gap > cfg.max_quote_gap:
            raise PriceSanityError(
                f"{ticker} {price_source or 'quote'} 触发绝对极值熔断，报价偏移 {gap:.1%} 超过 {cfg.max_quote_gap:.0%}"
            )

    return quote


def sanitize_cost_basis_against_market_price(
    cost_basis: Any,
    market_price: Any,
    ticker: str,
    *,
    config: HoldingCostSanityConfig | None = None,
) -> tuple[float | None, str | None]:
    cfg = config or HoldingCostSanityConfig()
    resolved_cost_basis = _coerce_float(cost_basis)
    resolved_market_price = _coerce_float(market_price)

    if resolved_cost_basis is None:
        return None, None
    if resolved_cost_basis <= cfg.min_price:
        return resolved_cost_basis, None
    if resolved_market_price is None or resolved_market_price <= cfg.min_price:
        return resolved_cost_basis, None

    gap = abs(resolved_cost_basis / resolved_market_price - 1.0)
    if gap <= cfg.max_cost_basis_gap:
        return resolved_cost_basis, None

    warning = (
        f"{ticker} 成本价 {resolved_cost_basis:.2f} 与当前市价 {resolved_market_price:.2f} "
        f"偏离 {gap:.1%}，超过 {cfg.max_cost_basis_gap:.0%}，已按现价熔断重置"
    )
    return resolved_market_price, warning