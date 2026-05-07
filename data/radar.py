from __future__ import annotations

from typing import Any

from data.fetcher import batch_history, get_news_items
from data.indicators import build_factor_snapshot
from data.macro import get_macro_snapshot
from data.options import get_near_term_options_summary
from data.sentiment import build_news_features
from models.regime import classify_market_regime, detect_breakdown_signal, detect_sigma_event
from models.signals import build_candidate_trade_profile


_EVENT_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "财报催化": ("earnings", "guidance", "revenue", "eps", "beats", "misses"),
    "并购合作": ("acquisition", "acquire", "merger", "partnership", "deal", "joint venture"),
    "AI/芯片": ("ai", "artificial intelligence", "chip", "gpu", "semiconductor", "data center"),
    "监管/诉讼": ("investigation", "fraud", "lawsuit", "sec", "antitrust", "approval"),
    "评级/指引": ("upgrade", "downgrade", "price target", "guidance cut", "guidance raise"),
    "新品/扩张": ("launch", "release", "product", "expansion", "factory", "capacity"),
}


def _close_position_in_range(open_price: float, high: float, low: float, close: float) -> float:
    price_range = high - low
    if price_range <= 0.0:
        return 0.5
    return (close - low) / price_range


def _usable_history(history: Any) -> bool:
    return history is not None and not history.empty


def _clip_unit(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _resolve_52w_high_context(history: Any) -> dict[str, Any]:
    if not _usable_history(history):
        return {
            "high_52w": None,
            "distance_to_52w_high_pct": None,
            "breakout_52w": False,
            "high_52w_strength": 0.0,
        }

    lookback = history.tail(min(len(history), 252))
    latest = lookback.iloc[-1]
    high_52w = float(lookback["high"].max())
    prior_high_52w = float(lookback["high"].iloc[:-1].max()) if len(lookback) > 1 else high_52w
    close = float(latest["close"])
    distance_to_52w_high_pct = ((close / high_52w) - 1.0) * 100.0 if high_52w else None
    breakout_52w = bool(len(lookback) > 1 and close > prior_high_52w)
    proximity_to_high = close / high_52w if high_52w else 0.0
    high_52w_strength = _clip_unit((proximity_to_high - 0.9) / 0.1)
    if breakout_52w:
        high_52w_strength = max(high_52w_strength, 0.95)
    return {
        "high_52w": high_52w,
        "distance_to_52w_high_pct": distance_to_52w_high_pct,
        "breakout_52w": breakout_52w,
        "high_52w_strength": high_52w_strength,
    }


def _infer_event_driver(
    news_features: dict[str, Any],
    price_change_pct: float,
    volume_ratio: float,
    sigma_event: dict[str, Any],
) -> dict[str, Any]:
    top_headlines = news_features.get("top_headlines") or []
    raw_tags: list[str] = []
    for headline in top_headlines[:3]:
        lowered = str((headline or {}).get("title") or "").lower()
        for tag, keywords in _EVENT_TAG_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                raw_tags.append(tag)

    if bool((sigma_event or {}).get("is_2sigma")) and abs(price_change_pct) >= 4.0 and volume_ratio >= 1.8:
        raw_tags.append("量价异动")

    event_tags = _dedupe_preserve_order(raw_tags)
    shock_component = _clip_unit(float(news_features.get("shock_score") or 0.0) / 5.0)
    sentiment_component = _clip_unit(abs(float(news_features.get("activated_sentiment") or 0.0)))
    headline_density = _clip_unit(float(news_features.get("headline_count") or 0.0) / 5.0)
    sigma_bonus = 0.16 if bool((sigma_event or {}).get("is_2sigma")) else 0.0
    price_volume_bonus = 0.12 if abs(price_change_pct) >= 4.0 and volume_ratio >= 1.8 else 0.0
    event_driven_score = float(
        min(
            1.0,
            0.32 * shock_component
            + 0.24 * sentiment_component
            + 0.18 * headline_density
            + sigma_bonus
            + price_volume_bonus
            + 0.08 * len(event_tags),
        )
    )
    event_driven = bool(event_tags) or event_driven_score >= 0.42
    event_driver = event_tags[0] if event_tags else ("事件催化" if event_driven else "")
    return {
        "event_driven": event_driven,
        "event_driver": event_driver,
        "event_tags": event_tags,
        "event_driven_score": event_driven_score,
    }


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


def _candidate_priority_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    breakout_bonus = 1.0 if bool(candidate.get("breakout_valid") or candidate.get("breakdown")) else 0.0
    expected_return = abs(float(candidate.get("expected_return_5d") or 0.0))
    target_ev = abs(float(candidate.get("target_ev") or 0.0))
    signal_strength = abs(float(candidate.get("signal_strength") or 0.0))
    predictive_confidence = float(candidate.get("predictive_confidence") or 0.0)
    volume_ratio = float(candidate.get("volume_ratio_20d") or 1.0)
    hist_vol = max(float(candidate.get("hist_vol_20d") or 0.35), 0.35)
    eligibility_bonus = 0.12 if bool(candidate.get("eligible_for_risk")) else 0.0
    normalized_volume = float(min(max(volume_ratio, 1.0), 2.5))
    high_52w_strength = float(candidate.get("high_52w_strength") or 0.0)
    event_driven_score = float(candidate.get("event_driven_score") or 0.0)
    trend_quality = expected_return * (0.7 + 0.3 * predictive_confidence) * normalized_volume / hist_vol
    return (
        breakout_bonus,
        trend_quality + target_ev * 4.0 + 0.08 * high_52w_strength + 0.06 * event_driven_score + eligibility_bonus,
        signal_strength,
        predictive_confidence,
    )


def scan_breakout_candidates(
    universe: list[str],
    benchmark: str = "QQQ",
    max_candidates: int = 5,
    regime_score: float | None = None,
    vix_value: float | None = None,
) -> list[dict[str, Any]]:
    tickers = list(dict.fromkeys(universe + [benchmark, "QQQ", "NVDA", "AAOI"]))
    histories = batch_history(tickers, period="1y", interval="1d")
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
        high_52w_context = _resolve_52w_high_context(history)
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
        sigma_event = detect_sigma_event(history)
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
            **high_52w_context,
            "sigma_return_pct": sigma_event.get("return_pct"),
            "sigma_z_score": sigma_event.get("z_score"),
            "sigma_2_event": bool(sigma_event.get("is_2sigma")),
        }
        candidate_news = build_news_features(get_news_items(ticker, limit=5))
        event_driver = _infer_event_driver(candidate_news, price_change_pct, volume_ratio, sigma_event)
        candidate_features.update(event_driver)
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
        key=_candidate_priority_key,
        reverse=True,
    )
    return ranked[:max_candidates]
