from __future__ import annotations

from typing import Any

from data.fetcher import batch_history, get_news_items
from data.indicators import build_factor_snapshot
from data.macro import get_macro_snapshot
from data.options import get_near_term_options_summary
from data.sentiment import build_news_features
from math_engine import build_market_neutral_math_snapshot
from models.regime import classify_market_regime, detect_breakdown_signal, detect_sigma_event
from models.signals import build_candidate_trade_profile
from portfolio.ticker_relationships import get_ticker_relationship_profile


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


def _trailing_return_pct(history: Any, lookback_days: int) -> float | None:
    if not _usable_history(history) or lookback_days <= 0:
        return None

    close_series = history["close"].dropna()
    if len(close_series) <= lookback_days:
        return None

    start_price = float(close_series.iloc[-(lookback_days + 1)])
    end_price = float(close_series.iloc[-1])
    if start_price <= 0.0:
        return None
    return (end_price / start_price - 1.0) * 100.0


def _resolve_relationship_spillover(
    ticker: str,
    relationship_profile: dict[str, Any],
    histories: dict[str, Any],
) -> dict[str, Any]:
    default = {
        "related_leader_ticker": "",
        "related_leader_return_5d_pct": None,
        "related_leader_return_20d_pct": None,
        "related_leader_return_60d_pct": None,
        "spillover_momentum_score": 0.0,
        "spillover_trigger": False,
        "spillover_reason": "",
    }

    related_tickers = list(relationship_profile.get("related_tickers") or [])
    relationship_strength = float(relationship_profile.get("relationship_strength") or 0.0)
    normalized_ticker = str(ticker or "").strip().upper()
    if not related_tickers or relationship_strength <= 0.0:
        return default

    best_leader: dict[str, Any] | None = None
    for related_ticker in related_tickers:
        normalized_related = str(related_ticker or "").strip().upper()
        if not normalized_related or normalized_related == normalized_ticker:
            continue

        related_history = histories.get(normalized_related)
        if not _usable_history(related_history):
            continue

        return_5d = _trailing_return_pct(related_history, 5)
        return_20d = _trailing_return_pct(related_history, 20)
        return_60d = _trailing_return_pct(related_history, 60)
        if return_5d is None and return_20d is None and return_60d is None:
            continue

        composite = (
            0.4 * _clip_unit(max(float(return_5d or 0.0), 0.0) / 10.0)
            + 0.4 * _clip_unit(max(float(return_20d or 0.0), 0.0) / 25.0)
            + 0.2 * _clip_unit(max(float(return_60d or 0.0), 0.0) / 45.0)
        )
        if best_leader is None or composite > float(best_leader.get("composite") or 0.0):
            best_leader = {
                "ticker": normalized_related,
                "return_5d": return_5d,
                "return_20d": return_20d,
                "return_60d": return_60d,
                "composite": composite,
            }

    if best_leader is None:
        return default

    spillover_momentum_score = float(min(1.0, relationship_strength * float(best_leader["composite"])))
    leader_return_5d = float(best_leader.get("return_5d") or 0.0)
    leader_return_20d = float(best_leader.get("return_20d") or 0.0)
    leader_return_60d = float(best_leader.get("return_60d") or 0.0)
    spillover_trigger = bool(
        spillover_momentum_score >= 0.38
        and ((leader_return_5d >= 4.0 and leader_return_20d >= 7.5) or leader_return_20d >= 15.0)
    )
    reason = (
        f"{best_leader['ticker']} 领涨 5日 {leader_return_5d:.1f}% / 20日 {leader_return_20d:.1f}%，"
        f"关联扩散强度 {spillover_momentum_score:.2f}"
    )
    return {
        "related_leader_ticker": str(best_leader["ticker"]),
        "related_leader_return_5d_pct": round(leader_return_5d, 2),
        "related_leader_return_20d_pct": round(leader_return_20d, 2),
        "related_leader_return_60d_pct": round(leader_return_60d, 2),
        "spillover_momentum_score": spillover_momentum_score,
        "spillover_trigger": spillover_trigger,
        "spillover_reason": reason,
    }


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
    relationship_strength = float(candidate.get("relationship_strength") or 0.0)
    spillover_momentum_score = float(candidate.get("spillover_momentum_score") or 0.0)
    orthogonal_alpha_score = abs(float(candidate.get("orthogonal_alpha_score") or 0.0))
    stat_arb_trigger = 0.12 if bool(candidate.get("stat_arb_trigger")) else 0.0
    trend_quality = expected_return * (0.7 + 0.3 * predictive_confidence) * normalized_volume / hist_vol
    return (
        breakout_bonus,
        trend_quality
        + target_ev * 4.0
        + 0.08 * high_52w_strength
        + 0.06 * event_driven_score
        + 0.05 * relationship_strength
        + 0.08 * spillover_momentum_score
        + 0.05 * orthogonal_alpha_score
        + stat_arb_trigger
        + eligibility_bonus,
        signal_strength,
        predictive_confidence,
    )


def _shortlist_priority_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        1.0 if bool(candidate.get("benchmark_above_sma_60")) else 0.0,
        max(float(candidate.get("idio_zscore") or 0.0), 0.0),
        1.0 if bool(float(candidate.get("close") or 0.0) >= float(candidate.get("sma_60") or candidate.get("close") or 0.0)) else 0.0,
        float(candidate.get("high_52w_strength") or 0.0),
        float(candidate.get("volume_ratio_20d") or 0.0),
        abs(float(candidate.get("daily_return_pct") or 0.0)),
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
    qqq_history = histories.get("QQQ")
    if not _usable_history(qqq_history):
        qqq_history = benchmark_history
    qqq_snapshot = build_factor_snapshot(qqq_history) if _usable_history(qqq_history) else {}
    qqq_close = qqq_snapshot.get("close")
    qqq_sma_60 = qqq_snapshot.get("sma_60")
    benchmark_above_sma_60 = bool(
        qqq_close is not None
        and qqq_sma_60 is not None
        and float(qqq_sma_60) > 0.0
        and float(qqq_close) >= float(qqq_sma_60)
    )

    resolved_regime_score, resolved_vix_value = _resolve_regime_context(
        histories,
        benchmark,
        regime_score,
        vix_value,
    )

    base_candidates: list[dict[str, Any]] = []
    candidate_news_map: dict[str, dict[str, Any]] = {}
    for ticker in universe:
        history = histories.get(ticker)
        if not _usable_history(history) or len(history) < 65:
            continue

        snapshot = build_factor_snapshot(history, benchmark_history)
        latest = history.iloc[-1]
        previous_high_60d = float(history["high"].iloc[-61:-1].max())
        previous_low_60d = float(history["low"].iloc[-61:-1].min())
        close_position = _close_position_in_range(
            float(latest["open"]),
            float(latest["high"]),
            float(latest["low"]),
            float(latest["close"]),
        )
        price_change_pct = float(snapshot.get("daily_return_pct") or 0.0)
        volume_ratio = float(snapshot.get("volume_ratio_20d") or 0.0)
        log_return_60d = float(snapshot.get("log_return_60d") or 0.0)
        sma_60 = float(snapshot.get("sma_60") or snapshot.get("sma_20") or latest["close"])
        high_52w_context = _resolve_52w_high_context(history)
        # V2.1: 候选池不再因为 20 日脉冲而误判突破，必须同时满足 60 日级别的新高与慢趋势同向。
        valid_breakout = bool(
            float(latest["close"]) > previous_high_60d
            and volume_ratio >= 1.5
            and close_position >= 0.6
            and price_change_pct >= 3.0
            and log_return_60d >= 0.08
            and float(latest["close"]) >= sma_60
        )
        fake_breakout = bool(
            float(latest["close"]) > previous_high_60d
            and (volume_ratio < 1.5 or close_position < 0.6 or log_return_60d < 0.08 or float(latest["close"]) < sma_60)
        )
        breakdown = bool(
            float(latest["close"]) < previous_low_60d
            and volume_ratio >= 1.5
            and log_return_60d <= -0.08
            and float(latest["close"]) <= sma_60
        )
        sigma_event = detect_sigma_event(history)
        relationship_profile = get_ticker_relationship_profile(ticker)
        spillover_context = _resolve_relationship_spillover(ticker, relationship_profile, histories)
        candidate_features = {
            "ticker": ticker,
            "close": float(latest["close"]),
            "daily_return_pct": price_change_pct,
            "log_return_10d": snapshot.get("log_return_10d"),
            "log_return_60d": snapshot.get("log_return_60d"),
            "volume_ratio_20d": volume_ratio,
            "atr_14": snapshot.get("atr_14"),
            "hist_vol_20d": snapshot.get("hist_vol_20d"),
            "rsi_14": snapshot.get("rsi_14"),
            "sma_20": snapshot.get("sma_20"),
            "sma_60": snapshot.get("sma_60"),
            "upper_shadow_pct": snapshot.get("upper_shadow_pct"),
            "intraday_drawdown_pct": snapshot.get("intraday_drawdown_pct"),
            "relative_strength": snapshot.get("relative_strength_20d_vs_benchmark"),
            "relative_strength_20d_vs_benchmark": snapshot.get("relative_strength_20d_vs_benchmark"),
            "breakout_20d": bool(snapshot.get("breakout_20d")),
            "breakout_60d": bool(snapshot.get("breakout_60d")),
            "breakout_valid": valid_breakout,
            "fake_breakout": fake_breakout,
            "breakdown": breakdown,
            "close_position_in_range": close_position,
            "trade_direction": "SHORT" if breakdown and not valid_breakout else ("LONG" if valid_breakout else "FLAT"),
            "benchmark_above_sma_60": benchmark_above_sma_60,
            "benchmark_close": qqq_close,
            "benchmark_sma_60": qqq_sma_60,
            **high_52w_context,
            **relationship_profile,
            **spillover_context,
            "sigma_return_pct": sigma_event.get("return_pct"),
            "sigma_z_score": sigma_event.get("z_score"),
            "sigma_2_event": bool(sigma_event.get("is_2sigma")),
        }
        event_driver = _infer_event_driver({}, price_change_pct, volume_ratio, sigma_event)
        candidate_features.update(event_driver)
        base_candidates.append(candidate_features)

    math_snapshot = build_market_neutral_math_snapshot(
        {str(candidate["ticker"]): candidate for candidate in base_candidates},
        histories,
        [str(candidate["ticker"]) for candidate in base_candidates],
    )
    for candidate in base_candidates:
        candidate.update(math_snapshot.signal_map.get(str(candidate["ticker"]), {}))

    shortlist_count = max(max_candidates * 3, 12)
    shortlisted = sorted(base_candidates, key=_shortlist_priority_key, reverse=True)[:shortlist_count]
    candidates: list[dict[str, Any]] = []
    for candidate in shortlisted:
        ticker = str(candidate["ticker"])
        candidate_news = build_news_features(get_news_items(ticker, limit=5))
        candidate_features = {
            **candidate,
        }
        candidate_features.update(
            _infer_event_driver(
                candidate_news,
                float(candidate.get("daily_return_pct") or 0.0),
                float(candidate.get("volume_ratio_20d") or 0.0),
                {
                    "is_2sigma": bool(candidate.get("sigma_2_event")),
                    "return_pct": candidate.get("sigma_return_pct"),
                    "z_score": candidate.get("sigma_z_score"),
                },
            )
        )
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
