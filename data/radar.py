from __future__ import annotations

from typing import Any

import numpy as np

from data.fetcher import batch_history
from data.indicators import build_factor_snapshot


def _close_position_in_range(open_price: float, high: float, low: float, close: float) -> float:
    price_range = high - low
    if price_range <= 0.0:
        return 0.5
    return (close - low) / price_range


def _expected_return_model(snapshot: dict[str, Any], close_position: float) -> float:
    log_return_5d = float(snapshot.get("log_return_5d") or 0.0)
    hist_vol = float(snapshot.get("hist_vol_20d") or 0.0)
    volume_ratio = float(snapshot.get("volume_ratio_20d") or 1.0)
    breakout_bonus = 1.0 if snapshot.get("breakout_20d") else 0.0

    momentum_score = float(np.clip(log_return_5d / 0.08, -1.0, 1.0))
    volume_support = float(np.clip((volume_ratio - 1.0) / 2.0, 0.0, 1.0))
    volatility_penalty = float(np.clip(hist_vol / 0.8, 0.0, 1.0))
    close_strength = float(np.clip((close_position - 0.5) / 0.5, -1.0, 1.0))

    expected_return = (
        0.01
        + 0.04 * momentum_score
        + 0.025 * volume_support
        + 0.02 * breakout_bonus
        + 0.015 * close_strength
        - 0.03 * volatility_penalty
    )
    return float(np.clip(expected_return, -0.08, 0.12))


def scan_breakout_candidates(
    universe: list[str],
    benchmark: str = "QQQ",
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    tickers = list(dict.fromkeys(universe + [benchmark]))
    histories = batch_history(tickers, period="3mo", interval="1d")
    benchmark_history = histories.get(benchmark)
    if benchmark_history is None or benchmark_history.empty:
        benchmark_history = None

    candidates: list[dict[str, Any]] = []
    for ticker in universe:
        history = histories.get(ticker)
        if history is None or history.empty or len(history) < 25:
            continue

        snapshot = build_factor_snapshot(history, benchmark_history)
        latest = history.iloc[-1]
        previous_high_20d = float(history["high"].iloc[-21:-1].max())
        previous_low_20d = float(history["low"].iloc[-21:-1].min())
        close_position = _close_position_in_range(
            float(latest["open"]),
            float(latest["high"]),
            float(latest["low"]),
            float(latest["close"]),
        )
        price_change_pct = float(snapshot.get("daily_return_pct") or 0.0)
        volume_ratio = float(snapshot.get("volume_ratio_20d") or 0.0)
        valid_breakout = bool(
            float(latest["close"]) > previous_high_20d
            and volume_ratio >= 1.5
            and close_position >= 0.6
            and price_change_pct >= 3.0
        )
        fake_breakout = bool(
            float(latest["close"]) > previous_high_20d
            and (volume_ratio < 1.5 or close_position < 0.6)
        )
        breakdown = bool(float(latest["close"]) < previous_low_20d and volume_ratio >= 1.5)
        expected_return_5d = _expected_return_model(snapshot, close_position)
        signal_strength = expected_return_5d * max(volume_ratio, 0.1)

        candidates.append(
            {
                "ticker": ticker,
                "close": float(latest["close"]),
                "daily_return_pct": price_change_pct,
                "volume_ratio_20d": volume_ratio,
                "atr_14": snapshot.get("atr_14"),
                "hist_vol_20d": snapshot.get("hist_vol_20d"),
                "relative_strength": snapshot.get("relative_strength_20d_vs_benchmark"),
                "breakout_valid": valid_breakout,
                "fake_breakout": fake_breakout,
                "breakdown": breakdown,
                "expected_return_5d": expected_return_5d,
                "expected_return_10d": round(expected_return_5d * 1.6, 4),
                "signal_strength": signal_strength,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (item["breakout_valid"], item["signal_strength"]),
        reverse=True,
    )
    return ranked[:max_candidates]
