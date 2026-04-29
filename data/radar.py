from __future__ import annotations

from typing import Any

from data.fetcher import batch_history, get_news_items
from data.indicators import build_factor_snapshot
from data.macro import get_macro_snapshot
from data.options import get_near_term_options_summary
from data.sentiment import build_news_features
from models.regime import classify_market_regime, detect_breakdown_signal
from models.signals import build_candidate_trade_profile


def _close_position_in_range(open_price: float, high: float, low: float, close: float) -> float:
    price_range = high - low
    if price_range <= 0.0:
        return 0.5
    return (close - low) / price_range


def _usable_history(history: Any) -> bool:
    return history is not None and not history.empty


def _resolve_regime_context(
    histories: dict[str, Any],
    benchmark: str,
    regime_score: float | None,
    vix_value: float | None,
) -> tuple[float, float | None]:
    macro_snapshot = get_macro_snapshot()
    resolved_vix = vix_value
    if resolved_vix is None:
        resolved_vix = float((macro_snapshot.get("vix") or {}).get("value") or 0.0) or None

    if regime_score is not None:
        return float(regime_score), resolved_vix

    qqq_history = histories.get("QQQ")
    if not _usable_history(qqq_history):
        qqq_history = histories.get(benchmark)

    benchmark_history = histories.get(benchmark)
    if not _usable_history(benchmark_history):
        benchmark_history = qqq_history

    nvda_history = histories.get("NVDA")
    aaoi_history = histories.get("AAOI")

    qqq_factors = build_factor_snapshot(
        qqq_history,
        benchmark_history if _usable_history(benchmark_history) else None,
    ) if _usable_history(qqq_history) else {}
    nvda_factors = build_factor_snapshot(
        nvda_history,
        qqq_history if _usable_history(qqq_history) else None,
    ) if _usable_history(nvda_history) else {}
    aaoi_breakdown = detect_breakdown_signal(aaoi_history) if _usable_history(aaoi_history) else {"breakdown": False}
    nvda_news = build_news_features(get_news_items("NVDA", limit=5))

    regime = classify_market_regime(
        macro_snapshot,
        qqq_factors,
        nvda_factors,
        aaoi_breakdown,
        nvda_news,
        market_history=qqq_history if _usable_history(qqq_history) else None,
    )
    regime_score_value = float(regime.get("score") or 0.5)
    regime_vix_value = float(regime.get("vix_value") or 0.0) or None
    return regime_score_value, resolved_vix if resolved_vix is not None else regime_vix_value


def scan_breakout_candidates(
    universe: list[str],
    benchmark: str = "QQQ",
    max_candidates: int = 5,
    regime_score: float | None = None,
    vix_value: float | None = None,
) -> list[dict[str, Any]]:
    tickers = list(dict.fromkeys(universe + [benchmark, "QQQ", "NVDA", "AAOI"]))
    histories = batch_history(tickers, period="3mo", interval="1d")
    benchmark_history = histories.get(benchmark)
    if not _usable_history(benchmark_history):
        benchmark_history = None

    resolved_regime_score, resolved_vix_value = _resolve_regime_context(
        histories,
        benchmark,
        regime_score,
        vix_value,
    )

    candidates: list[dict[str, Any]] = []
    for ticker in universe:
        history = histories.get(ticker)
        if not _usable_history(history) or len(history) < 25:
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
        candidate_features = {
            "ticker": ticker,
            "close": float(latest["close"]),
            "daily_return_pct": price_change_pct,
            "log_return_10d": snapshot.get("log_return_10d"),
            "volume_ratio_20d": volume_ratio,
            "atr_14": snapshot.get("atr_14"),
            "hist_vol_20d": snapshot.get("hist_vol_20d"),
            "rsi_14": snapshot.get("rsi_14"),
            "sma_20": snapshot.get("sma_20"),
            "upper_shadow_pct": snapshot.get("upper_shadow_pct"),
            "intraday_drawdown_pct": snapshot.get("intraday_drawdown_pct"),
            "relative_strength": snapshot.get("relative_strength_20d_vs_benchmark"),
            "relative_strength_20d_vs_benchmark": snapshot.get("relative_strength_20d_vs_benchmark"),
            "breakout_20d": bool(snapshot.get("breakout_20d")),
            "breakout_valid": valid_breakout,
            "fake_breakout": fake_breakout,
            "breakdown": breakdown,
            "close_position_in_range": close_position,
            "trade_direction": "SHORT" if breakdown and not valid_breakout else ("LONG" if valid_breakout else "FLAT"),
        }
        candidate_news = build_news_features(get_news_items(ticker, limit=5))
        candidate_profile = build_candidate_trade_profile(
            candidate_features,
            resolved_regime_score,
            news_features=candidate_news,
            vix_value=resolved_vix_value,
            options_data=get_near_term_options_summary(ticker),
        )
        candidates.append(
            {
                **candidate_profile,
                "eligible_for_risk": bool(candidate_profile.get("eligible_for_risk")),
                "gate_reason": str(candidate_profile.get("gate_reason") or ""),
                "days_to_earnings": candidate_profile.get("days_to_earnings"),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item.get("breakout_valid") or item.get("breakdown")),
            abs(float(item.get("signal_strength") or 0.0)),
        ),
        reverse=True,
    )
    return ranked[:max_candidates]
