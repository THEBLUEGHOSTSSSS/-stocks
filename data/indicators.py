from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR


STRUCTURAL_MOMENTUM_LOOKBACK = 60
TREND_FILTER_LOOKBACK = 120


def compute_log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def compute_historical_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    log_returns = compute_log_return(close)
    return log_returns.rolling(window=window, min_periods=window).std() * math.sqrt(TRADING_DAYS_PER_YEAR)


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = volume.rolling(window=window, min_periods=window).mean()
    return volume / rolling_mean.replace(0.0, np.nan)


def compute_relative_strength(
    close: pd.Series,
    benchmark_close: pd.Series,
    window: int = 20,
) -> pd.Series:
    aligned = pd.concat(
        [close.rename("asset_close"), benchmark_close.rename("benchmark_close")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(index=close.index, dtype=float)

    asset_return = aligned["asset_close"].pct_change(window)
    benchmark_return = aligned["benchmark_close"].pct_change(window)
    relative_strength = asset_return - benchmark_return
    return relative_strength.reindex(close.index)


def build_factor_snapshot(
    history: pd.DataFrame,
    benchmark_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if history.empty:
        return {}

    close = history["close"]
    volume = history["volume"] if "volume" in history else pd.Series(index=history.index, dtype=float)
    rsi = compute_rsi(close)
    atr = compute_atr(history)
    volatility = compute_historical_volatility(close)
    # V2.1: 将核心趋势从 10 日对数收益切换到 60 日。
    # 数学上这是把原先高噪声的短周期漂移估计，替换为更慢、更稳定的结构性趋势项。
    log_return_60d = compute_log_return(close, periods=STRUCTURAL_MOMENTUM_LOOKBACK)
    volume_ratio = compute_volume_ratio(volume)
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    sma_60 = close.rolling(window=STRUCTURAL_MOMENTUM_LOOKBACK, min_periods=STRUCTURAL_MOMENTUM_LOOKBACK).mean()
    sma_120 = close.rolling(window=TREND_FILTER_LOOKBACK, min_periods=TREND_FILTER_LOOKBACK).mean()
    breakout_20d = close > close.shift(1).rolling(window=20, min_periods=20).max()
    breakout_60d = close > close.shift(1).rolling(
        window=STRUCTURAL_MOMENTUM_LOOKBACK,
        min_periods=STRUCTURAL_MOMENTUM_LOOKBACK,
    ).max()
    relative_strength = None
    if benchmark_history is not None and not benchmark_history.empty:
        relative_strength_series = compute_relative_strength(close, benchmark_history["close"])
        relative_strength = float(relative_strength_series.iloc[-1]) if not relative_strength_series.empty else None

    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else latest
    latest_close = float(latest.get("close", np.nan))
    latest_open = float(latest.get("open", np.nan))
    latest_high = float(latest.get("high", np.nan))
    latest_volatility = float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else None
    latest_log_return_60d = float(log_return_60d.iloc[-1]) if not pd.isna(log_return_60d.iloc[-1]) else None

    upper_shadow_pct = None
    if np.isfinite(latest_high) and np.isfinite(latest_open) and np.isfinite(latest_close) and latest_close > 0.0:
        upper_shadow_pct = float(max(latest_high - max(latest_open, latest_close), 0.0) / latest_close * 100.0)

    intraday_drawdown_pct = None
    if np.isfinite(latest_high) and np.isfinite(latest_close) and latest_high > 0.0:
        intraday_drawdown_pct = float(max(latest_high - latest_close, 0.0) / latest_high * 100.0)

    return {
        "close": latest_close,
        "daily_return_pct": float(((latest.get("close", np.nan) / previous.get("close", np.nan)) - 1.0) * 100.0),
        # 保留 legacy 字段是为了让旧版 Streamlit 表格和报告仍能正常消费，
        # 但其底层值已经切换为 60 日结构性动量，避免前端直接断裂。
        "log_return_10d": latest_log_return_60d,
        "log_return_60d": latest_log_return_60d,
        "hist_vol_20d": latest_volatility,
        "rsi_14": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        "atr_14": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
        "volume_ratio_20d": float(volume_ratio.iloc[-1]) if not pd.isna(volume_ratio.iloc[-1]) else None,
        "sma_20": float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
        "sma_60": float(sma_60.iloc[-1]) if not pd.isna(sma_60.iloc[-1]) else None,
        "sma_120": float(sma_120.iloc[-1]) if not pd.isna(sma_120.iloc[-1]) else None,
        "above_sma_120": bool(
            not pd.isna(sma_120.iloc[-1])
            and latest_close > float(sma_120.iloc[-1])
        ),
        "breakout_20d": bool(breakout_20d.fillna(False).iloc[-1]),
        "breakout_60d": bool(breakout_60d.fillna(False).iloc[-1]),
        "upper_shadow_pct": upper_shadow_pct,
        "intraday_drawdown_pct": intraday_drawdown_pct,
        "relative_strength_20d_vs_benchmark": relative_strength,
    }
