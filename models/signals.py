from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.fetcher import get_upcoming_earnings
from models.regime import classify_factor_allocation_regime


GLOBAL_OBSERVE_REGIME_THRESHOLD = 0.34
STRATEGY_THRESHOLDS = {
    "momentum": {
        "min_regime_score": 0.48,
        "min_expected_return_5d": 0.02,
        "min_ev": 0.006,
        "min_win_rate": 0.57,
        "min_payoff_ratio": 1.05,
    },
    "short_breakdown": {
        "min_regime_score": 0.4,
        "min_expected_return_5d": 0.018,
        "min_ev": 0.005,
        "min_win_rate": 0.56,
        "min_payoff_ratio": 1.0,
    },
    "mean_reversion": {
        "min_regime_score": 0.42,
        "min_expected_return_5d": 0.015,
        "min_ev": 0.004,
        "min_win_rate": 0.58,
        "min_payoff_ratio": 0.85,
    },
}


def _clip_feature(value: float, scale: float, lower: float = -1.0, upper: float = 1.0) -> float:
    if scale == 0.0:
        return 0.0
    return float(np.clip(value / scale, lower, upper))


def _coerce_finite_float(value: Any, default: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(resolved):
        return default
    return resolved


def _resolve_momentum_core(market_features: dict[str, Any]) -> dict[str, float]:
    log_return_10d = _coerce_finite_float(market_features.get("log_return_10d"), 0.0)
    return {
        "log_return_10d": log_return_10d,
        "momentum_term": _clip_feature(log_return_10d, 0.08),
    }


def _assess_momentum_rsi_filter(
    market_features: dict[str, Any],
    breakout_valid: bool,
    relative_strength_term: float,
) -> dict[str, Any]:
    rsi_14 = _coerce_finite_float(market_features.get("rsi_14"), 50.0)
    momentum_term = max(float(_resolve_momentum_core(market_features)["momentum_term"]), 0.0)
    if not breakout_valid or momentum_term <= 0.0:
        return {
            "release_multiplier": 1.0,
            "reversal_penalty": 0.0,
            "blowoff_top_block": False,
            "state": "inactive",
        }

    quality_score = (
        0.55 * float(np.clip(momentum_term, 0.0, 1.0))
        + 0.25 * float(np.clip(relative_strength_term, 0.0, 1.0))
        + 0.20
    )
    medium_quality = quality_score >= 0.45
    high_quality = quality_score >= 0.72

    # Orthogonalize RSI away from trend alpha: RSI never adds positive EV.
    # It only gates whether already-valid intermediate momentum may pass through the model.
    if medium_quality and rsi_14 > 70.0:
        return {
            "release_multiplier": 0.0,
            "reversal_penalty": -(0.035 + 0.01 * float(np.clip((rsi_14 - 70.0) / 10.0, 0.0, 1.0))),
            "blowoff_top_block": True,
            "state": "blowoff_top_block",
        }
    if medium_quality and 40.0 <= rsi_14 <= 60.0:
        return {
            "release_multiplier": 1.0 if high_quality else 0.85,
            "reversal_penalty": 0.0,
            "blowoff_top_block": False,
            "state": "healthy_release",
        }
    if high_quality and 60.0 < rsi_14 <= 68.0:
        return {
            "release_multiplier": 0.55,
            "reversal_penalty": -0.006,
            "blowoff_top_block": False,
            "state": "stretched_but_allowed",
        }
    if medium_quality:
        return {
            "release_multiplier": 0.0,
            "reversal_penalty": -0.01,
            "blowoff_top_block": False,
            "state": "rsi_filter_block",
        }
    return {
        "release_multiplier": 0.25,
        "reversal_penalty": -0.004 if rsi_14 > 65.0 else 0.0,
        "blowoff_top_block": False,
        "state": "weak_momentum",
    }


def _assess_volume_divergence(
    market_features: dict[str, Any],
    breakout_valid: bool,
) -> dict[str, Any]:
    volume_ratio = _coerce_finite_float(market_features.get("volume_ratio_20d") or market_features.get("volume_ratio"), 1.0)
    upper_shadow_pct = _coerce_finite_float(market_features.get("upper_shadow_pct"), 0.0)
    intraday_drawdown_pct = _coerce_finite_float(market_features.get("intraday_drawdown_pct"), 0.0)

    default = {
        "distribution_risk": False,
        "win_rate_penalty": 0.0,
        "reversal_penalty": 0.0,
        "reason": "",
    }
    if not breakout_valid or volume_ratio <= 1.5:
        return default

    upper_shadow_risk = upper_shadow_pct >= 2.5
    intraday_drawdown_risk = intraday_drawdown_pct >= 3.0
    if not upper_shadow_risk and not intraday_drawdown_risk:
        return default

    reasons: list[str] = []
    if upper_shadow_risk:
        reasons.append(f"长上影 {upper_shadow_pct:.1f}%")
    if intraday_drawdown_risk:
        reasons.append(f"日内回撤 {intraday_drawdown_pct:.1f}%")
    win_rate_penalty = 0.16 + (0.05 if upper_shadow_risk else 0.0) + (0.05 if intraday_drawdown_risk else 0.0)
    reversal_penalty = -(0.012 + (0.006 if upper_shadow_risk and intraday_drawdown_risk else 0.0))
    return {
        "distribution_risk": True,
        "win_rate_penalty": float(min(win_rate_penalty, 0.26)),
        "reversal_penalty": reversal_penalty,
        "reason": "放量滞涨/派发：" + " / ".join(reasons),
    }


def detect_exhaustion_reversal(market_features: dict[str, Any]) -> dict[str, Any]:
    daily_return_pct = float(market_features.get("daily_return_pct") or 0.0)
    hist_vol_20d = float(market_features.get("hist_vol_20d") or 0.0)
    rsi_14 = float(market_features.get("rsi_14") or 50.0)
    volume_ratio = float(market_features.get("volume_ratio_20d") or market_features.get("volume_ratio") or 1.0)
    momentum_term = float(_resolve_momentum_core(market_features)["momentum_term"])

    # Downside capitulation keeps the original asymmetric logic: large down move plus prior overheat, then require
    # either elevated volatility or weak rebound volume to confirm panic-style exhaustion.
    downside_capitulation = bool(
        daily_return_pct <= -4.0
        and (rsi_14 >= 70.0 or momentum_term >= 0.85)
        and (hist_vol_20d >= 0.65 or volume_ratio < 1.0)
    )
    # Blow-off top uses a hard three-factor trigger; extreme volume is a weighted confirmation for distribution.
    upside_blowoff = bool(
        daily_return_pct >= 5.0
        and rsi_14 >= 75.0
        and momentum_term >= 0.95
    )

    if downside_capitulation:
        reasons: list[str] = []
        if daily_return_pct <= -4.0:
            reasons.append("单日急跌")
        if rsi_14 >= 70.0:
            reasons.append("RSI处于过热区")
        if momentum_term >= 0.85:
            reasons.append("10日动量过热")
        if hist_vol_20d >= 0.65:
            reasons.append("20日波动率偏高")
        if volume_ratio < 1.0:
            reasons.append("回撤日量能未放大")
        reason = " / ".join(reasons[:4])
        return {
            "exhaustion_reversal": True,
            "exhaustion_reason": f"向下恐慌反转：{reason}",
        }

    if upside_blowoff:
        reasons = ["单日逼空拉升", "RSI进入极热区", "10日动量过热"]
        if volume_ratio >= 2.5:
            reasons.append("天量疑似高位派发")
        reason = " / ".join(reasons[:4])
        return {
            "exhaustion_reversal": True,
            "exhaustion_reason": f"向上逼空过热：{reason}",
        }

    return {
        "exhaustion_reversal": False,
        "exhaustion_reason": "",
    }


def volatility_penalty_label(hist_vol_20d: float | None) -> str:
    if hist_vol_20d is None:
        return "Medium"
    if hist_vol_20d >= 0.65:
        return "High"
    if hist_vol_20d >= 0.35:
        return "Medium"
    return "Low"


def _fallback_alpha(sample: pd.DataFrame) -> float:
    if sample.empty:
        return 0.0
    excess_return = sample["asset"] - sample["benchmark"]
    if excess_return.empty:
        return 0.0
    return float(excess_return.mean())


def calculate_rolling_alpha(
    ticker_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> float:
    # OLS alpha uses the latest clean window; if observations are insufficient, fall back to mean excess return.
    aligned = pd.concat(
        [ticker_returns.rename("asset"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return 0.0

    sample = aligned.tail(window)
    if len(sample) < window:
        return _fallback_alpha(sample)

    benchmark = sample["benchmark"].to_numpy(dtype=float)
    asset = sample["asset"].to_numpy(dtype=float)
    if np.allclose(np.var(benchmark), 0.0):
        return _fallback_alpha(sample)

    design_matrix = np.column_stack([np.ones(len(sample)), benchmark])
    try:
        coefficients, *_ = np.linalg.lstsq(design_matrix, asset, rcond=None)
    except np.linalg.LinAlgError:
        return _fallback_alpha(sample)

    alpha = float(coefficients[0])
    if not np.isfinite(alpha):
        return _fallback_alpha(sample)
    return alpha


def _normalize_expert_weights(weights: dict[str, float]) -> dict[str, float]:
    sanitized = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = float(sum(sanitized.values()))
    if total <= 0.0:
        uniform = 1.0 / max(len(sanitized), 1)
        return {name: uniform for name in sanitized}
    return {name: value / total for name, value in sanitized.items()}


def _normalize_signed_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(abs(float(value)) for value in weights.values()))
    if total <= 0.0:
        fallback = {
            "momentum": 0.30,
            "relative_strength": 0.22,
            "breakout": 0.18,
            "rsi_alignment": 0.0,
            "sentiment_component": 0.12,
            "shock_component": 0.08,
            "headline_density": 0.04,
            "volatility_term": 0.0,
        }
        total = float(sum(abs(value) for value in fallback.values()))
        return {name: value / total for name, value in fallback.items()}
    return {name: float(value) / total for name, value in weights.items()}


def get_dynamic_weights(regime: str, hist_vol_20d: float) -> dict[str, float]:
    raw_weights_by_regime = {
        "Risk-On": {
            "momentum": 0.34,
            "relative_strength": 0.22,
            "breakout": 0.24,
            "rsi_alignment": 0.0,
            "sentiment_component": 0.09,
            "shock_component": 0.04,
            "headline_density": 0.02,
            "volatility_term": 0.0,
        },
        "Choppy": {
            "momentum": 0.12,
            "relative_strength": 0.28,
            "breakout": 0.08,
            "rsi_alignment": 0.0,
            "sentiment_component": 0.11,
            "shock_component": 0.06,
            "headline_density": 0.04,
            "volatility_term": 0.0,
        },
        "Risk-Off": {
            "momentum": 0.06,
            "relative_strength": 0.16,
            "breakout": 0.02,
            "rsi_alignment": 0.0,
            "sentiment_component": 0.22,
            "shock_component": 0.18,
            "headline_density": 0.08,
            "volatility_term": 0.0,
        },
    }
    del hist_vol_20d
    weights = dict(raw_weights_by_regime.get(str(regime or "Choppy"), raw_weights_by_regime["Choppy"]))
    return _normalize_signed_weights(weights)


def _assess_volatility_gate(
    hist_vol_20d: float,
    breakout_valid: bool,
    short_trend_confirmed: bool,
    regime_label: str,
    momentum_term: float,
) -> dict[str, Any]:
    if short_trend_confirmed:
        return {
            "release_multiplier": 1.0,
            "penalty": 0.0,
            "block": False,
            "state": "short_trend_allowed",
            "reason": "",
        }

    if hist_vol_20d >= 0.9 and breakout_valid:
        return {
            "release_multiplier": 0.0,
            "penalty": -0.012,
            "block": True,
            "state": "extreme_vol_block",
            "reason": "20日波动率极高，禁止追涨",
        }

    if hist_vol_20d >= 0.65:
        if breakout_valid and regime_label == "Risk-On" and momentum_term >= 0.55:
            return {
                "release_multiplier": 0.65,
                "penalty": -0.004,
                "block": False,
                "state": "high_vol_caution",
                "reason": "20日波动率偏高，仅释放部分动能",
            }
        return {
            "release_multiplier": 0.0,
            "penalty": -0.008,
            "block": True,
            "state": "high_vol_gate",
            "reason": "20日波动率偏高，暂不放行动能追涨",
        }

    return {
        "release_multiplier": 1.0,
        "penalty": 0.0,
        "block": False,
        "state": "allowed",
        "reason": "",
    }


def _extract_call_put_volume_ratio(options_data: dict[str, Any] | None) -> float:
    if not options_data:
        return 1.0
    raw_ratio = options_data.get("call_put_volume_ratio")
    if raw_ratio is None:
        return 1.0
    try:
        ratio = float(raw_ratio)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    return ratio


def _build_options_edge(
    options_data: dict[str, Any] | None,
    breakout_valid: bool,
    short_trend_confirmed: bool,
) -> dict[str, Any]:
    ratio = _extract_call_put_volume_ratio(options_data)
    default = {
        "call_put_volume_ratio": ratio,
        "options_flow_signal": "neutral",
        "options_edge": 0.0,
        "reversal_penalty_delta": 0.0,
        "confidence_delta": 0.0,
        "options_win_rate_multiplier": 1.0,
        "options_win_rate_delta": 0.0,
        "options_flow_reason": "期权量比中性",
    }

    if breakout_valid and ratio >= 3.5:
        intensity = float(np.clip((ratio - 3.5) / 4.0, 0.0, 1.0))
        return {
            "call_put_volume_ratio": ratio,
            "options_flow_signal": "gamma_squeeze",
            "options_edge": 0.04 + 0.02 * intensity,
            "reversal_penalty_delta": 0.0,
            "confidence_delta": 0.18 + 0.08 * intensity,
            "options_win_rate_multiplier": 1.18 + 0.08 * intensity,
            "options_win_rate_delta": 0.10 + 0.06 * intensity,
            "options_flow_reason": "Call/Put 量比极端偏多，机构扫 Call 触发 Gamma 挤压加成",
        }

    if breakout_valid and ratio <= 0.6:
        intensity = float(np.clip((0.6 - ratio) / 0.6, 0.0, 1.0))
        return {
            "call_put_volume_ratio": ratio,
            "options_flow_signal": "put_divergence",
            "options_edge": -(0.015 + 0.015 * intensity),
            "reversal_penalty_delta": -(0.05 + 0.015 * intensity),
            "confidence_delta": -(0.14 + 0.08 * intensity),
            "options_win_rate_multiplier": 0.72 - 0.12 * intensity,
            "options_win_rate_delta": -(0.12 + 0.08 * intensity),
            "options_flow_reason": "表面突破但 Put 防守异常放量，触发期权背离重罚",
        }

    if short_trend_confirmed and ratio <= 0.3:
        intensity = float(np.clip((0.3 - ratio) / 0.3, 0.0, 1.0))
        return {
            "call_put_volume_ratio": ratio,
            "options_flow_signal": "put_panic_short",
            "options_edge": -(0.025 + 0.02 * intensity),
            "reversal_penalty_delta": 0.0,
            "confidence_delta": 0.12 + 0.05 * intensity,
            "options_win_rate_multiplier": 1.22 + 0.10 * intensity,
            "options_win_rate_delta": 0.16 + 0.10 * intensity,
            "options_flow_reason": "极端 Put 恐慌叠加单边下跌，空头延续胜率显著抬升",
        }

    return default


def estimate_expert_context(
    market_features: dict[str, Any],
    news_features: dict[str, Any] | None,
    regime_score: float,
) -> dict[str, Any]:
    news = news_features or {}
    momentum_core = _resolve_momentum_core(market_features)
    hist_vol_20d = float(market_features.get("hist_vol_20d") or 0.0)
    relative_strength = float(market_features.get("relative_strength_20d_vs_benchmark") or market_features.get("relative_strength") or 0.0)
    rsi_14 = float(market_features.get("rsi_14") or 50.0)
    daily_return_pct = float(market_features.get("daily_return_pct") or 0.0)
    volume_ratio = float(market_features.get("volume_ratio_20d") or market_features.get("volume_ratio") or 1.0)
    breakout = 1.0 if (market_features.get("breakout_20d") or market_features.get("breakout_valid")) else 0.0

    momentum_term = float(momentum_core["momentum_term"])
    relative_strength_term = _clip_feature(relative_strength, 0.15)
    volatility_term = _clip_feature(hist_vol_20d, 0.8, lower=0.0, upper=1.0)
    oversold_reversion_gate = 1.0 if rsi_14 <= 35.0 else (0.35 if rsi_14 <= 42.0 else 0.0)
    tail_move_term = float(np.clip(abs(daily_return_pct) / 8.0, 0.0, 1.0))

    sentiment_component = float(news.get("activated_sentiment") or 0.0)
    shock_component = _clip_feature(float(news.get("shock_score") or 0.0), 4.0)
    headline_density = _clip_feature(float(news.get("headline_count") or 0.0), 8.0, lower=0.0, upper=1.0)

    expert_edges = {
        "trend": (
            0.58 * momentum_term
            + 0.27 * relative_strength_term
            + 0.15 * breakout
        ),
        "reversion": (
            oversold_reversion_gate
            * (
                0.55 * float(np.clip(-momentum_term, 0.0, 1.0))
                + 0.25 * _clip_feature(1.4 - volume_ratio, 1.4, lower=0.0, upper=1.0)
                + 0.2 * float(np.clip(-relative_strength_term, 0.0, 1.0))
            )
        ),
        "event": (
            0.55 * sentiment_component
            + 0.3 * shock_component
            + 0.15 * headline_density
        ),
    }
    expert_weights = _normalize_expert_weights(
        {
            "trend": (
                0.25
                + 0.42 * breakout
                + 0.2 * float(np.clip((regime_score - 0.45) / 0.25, 0.0, 1.0))
                + 0.18 * float(np.clip(relative_strength_term, 0.0, 1.0))
                + 0.12 * float(np.clip(momentum_term, 0.0, 1.0))
                - 0.2 * float(np.clip((volume_ratio - 2.5) / 2.0, 0.0, 1.0))
            ),
            "reversion": (
                0.12
                + 0.28 * float(np.clip(-momentum_term, 0.0, 1.0))
                + 0.42 * oversold_reversion_gate
                + 0.1 * (1.0 - breakout)
            ),
            "event": (
                0.15
                + 0.35 * abs(sentiment_component)
                + 0.25 * abs(shock_component)
                + 0.15 * headline_density
                + 0.1 * tail_move_term
            ),
        }
    )
    expert_consensus = float(sum(expert_weights[name] * expert_edges[name] for name in expert_edges))
    expert_disagreement = float(np.std(list(expert_edges.values())))
    dominant_expert = max(expert_weights, key=expert_weights.get)
    return {
        "expert_edges": expert_edges,
        "expert_weights": expert_weights,
        "expert_consensus": expert_consensus,
        "expert_disagreement": expert_disagreement,
        "dominant_expert": dominant_expert,
        "momentum_term": momentum_term,
        "relative_strength_term": relative_strength_term,
        "volatility_term": volatility_term,
        "sentiment_component": sentiment_component,
        "shock_component": shock_component,
        "headline_density": headline_density,
    }


def estimate_predictive_uncertainty(
    market_features: dict[str, Any],
    news_features: dict[str, Any] | None,
    regime_score: float,
    expert_context: dict[str, Any],
) -> dict[str, float]:
    news = news_features or {}
    rsi_14 = float(market_features.get("rsi_14") or 50.0)
    daily_return_pct = float(market_features.get("daily_return_pct") or 0.0)
    volume_ratio = float(market_features.get("volume_ratio_20d") or market_features.get("volume_ratio") or 1.0)
    breakout = bool(market_features.get("breakout_20d") or market_features.get("breakout_valid"))

    volatility_term = float(expert_context.get("volatility_term") or 0.0)
    shock_component = float(expert_context.get("shock_component") or _clip_feature(float(news.get("shock_score") or 0.0), 4.0))
    expert_disagreement_score = float(np.clip(float(expert_context.get("expert_disagreement") or 0.0) / 0.6, 0.0, 1.0))
    expert_weights = expert_context.get("expert_weights") or {"trend": 1.0}

    volume_anomaly = float(np.clip(abs(volume_ratio - 1.4) / 2.2, 0.0, 1.0))
    tail_move_term = float(np.clip(abs(daily_return_pct) / 7.0, 0.0, 1.0))
    news_uncertainty = float(np.clip(abs(shock_component) / 1.0, 0.0, 1.0))
    routing_uncertainty = float(np.clip(1.0 - max(float(value) for value in expert_weights.values()), 0.0, 1.0))
    weak_regime = float(np.clip((0.48 - regime_score) / 0.2, 0.0, 1.0))
    crowding_risk = float(
        np.clip(
            max(rsi_14 - 68.0, 0.0) / 14.0
            + max(volume_ratio - 2.5, 0.0) / 2.5
            + (0.3 if breakout and regime_score < 0.5 else 0.0),
            0.0,
            1.0,
        )
    )

    uncertainty_score = (
        0.22 * volatility_term
        + 0.16 * volume_anomaly
        + 0.14 * tail_move_term
        + 0.14 * news_uncertainty
        + 0.14 * expert_disagreement_score
        + 0.1 * routing_uncertainty
        + 0.1 * weak_regime
        + 0.1 * crowding_risk
    )
    uncertainty_score = float(np.clip(uncertainty_score, 0.0, 1.0))
    predictive_confidence = float(np.clip(1.0 - uncertainty_score, 0.05, 0.95))
    prediction_interval_5d = float(np.clip(0.012 + 0.04 * uncertainty_score + 0.025 * volatility_term, 0.012, 0.1))
    return {
        "uncertainty_score": uncertainty_score,
        "predictive_confidence": predictive_confidence,
        "prediction_interval_5d": prediction_interval_5d,
        "expert_disagreement_score": expert_disagreement_score,
        "routing_uncertainty": routing_uncertainty,
        "crowding_risk": crowding_risk,
    }


def estimate_expected_return(
    ticker: str,
    market_features: dict[str, Any],
    news_features: dict[str, Any],
    regime_score: float,
    rolling_alpha: float | None = None,
    vix_value: float | None = None,
    options_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    momentum_core = _resolve_momentum_core(market_features)
    hist_vol_20d = float(market_features.get("hist_vol_20d") or 0.0)
    relative_strength = float(market_features.get("relative_strength_20d_vs_benchmark") or market_features.get("relative_strength") or 0.0)
    daily_return_pct = float(market_features.get("daily_return_pct") or 0.0)
    breakout = 1.0 if (market_features.get("breakout_20d") or market_features.get("breakout_valid")) else 0.0
    exhaustion_signal = detect_exhaustion_reversal(market_features)
    expert_context = estimate_expert_context(market_features, news_features, regime_score)
    uncertainty = estimate_predictive_uncertainty(market_features, news_features, regime_score, expert_context)
    allocation_regime = classify_factor_allocation_regime(regime_score, vix_value)
    dynamic_weights = get_dynamic_weights(str(allocation_regime["label"]), hist_vol_20d)
    breakout_valid = bool(market_features.get("breakout_valid"))
    short_trend_confirmed = bool(
        market_features.get("short_trend_confirmed")
        or market_features.get("breakdown")
        or str(market_features.get("trade_direction") or "").upper() == "SHORT"
    )
    options_edge = _build_options_edge(options_data, breakout_valid, short_trend_confirmed)
    relative_strength_term = _clip_feature(relative_strength, 0.15)
    momentum_filter = _assess_momentum_rsi_filter(market_features, breakout_valid, relative_strength_term)
    volume_divergence = _assess_volume_divergence(market_features, breakout_valid)
    volatility_gate = _assess_volatility_gate(
        hist_vol_20d,
        breakout_valid,
        short_trend_confirmed,
        str(allocation_regime["label"]),
        float(momentum_core["momentum_term"]),
    )

    momentum_release_multiplier = float(momentum_filter["release_multiplier"]) * float(volatility_gate["release_multiplier"])
    momentum_term = float(momentum_core["momentum_term"]) * momentum_release_multiplier
    breakout_term = breakout * momentum_release_multiplier
    sentiment_component = float(news_features.get("activated_sentiment") or 0.0)
    shock_component = _clip_feature(float(news_features.get("shock_score") or 0.0), 4.0)
    headline_density = _clip_feature(float(news_features.get("headline_count") or 0.0), 8.0, lower=0.0, upper=1.0)

    # RSI is deliberately excluded from the additive score stack. It only gates momentum release.
    price_factor_component = (
        dynamic_weights["momentum"] * momentum_term
        + dynamic_weights["relative_strength"] * relative_strength_term
        + dynamic_weights["breakout"] * breakout_term
    )
    news_component = (
        dynamic_weights["sentiment_component"] * sentiment_component
        + dynamic_weights["shock_component"] * shock_component
        + dynamic_weights["headline_density"] * headline_density
    )
    base_market_component = price_factor_component + news_component
    expert_blend = {
        "Risk-On": 0.26,
        "Choppy": 0.22,
        "Risk-Off": 0.36,
    }.get(str(allocation_regime["label"]), 0.28)
    market_component = (1.0 - expert_blend) * base_market_component + expert_blend * float(expert_context["expert_consensus"])

    # The regression alpha is estimated on daily returns, so rescale it to the 5-day forecast horizon.
    resolved_alpha = float(rolling_alpha) if rolling_alpha is not None and np.isfinite(rolling_alpha) else 0.0
    alpha_component_5d = resolved_alpha * 5.0
    if allocation_regime["label"] == "Risk-On":
        regime_bias = 0.006 + (regime_score - 0.5) * 0.025
        synergy = 0.012 if breakout_term and sentiment_component > 0 else 0.0
    elif allocation_regime["label"] == "Risk-Off":
        regime_bias = -0.008 + (regime_score - 0.5) * 0.015
        synergy = 0.004 if sentiment_component > 0.2 and shock_component >= 0.0 else 0.0
    else:
        regime_bias = (regime_score - 0.5) * 0.008
        synergy = 0.004 if breakout_term and relative_strength_term > -0.2 else 0.0
    reversal_penalty = 0.0
    if exhaustion_signal["exhaustion_reversal"]:
        reversal_penalty -= min(abs(daily_return_pct) / 100.0 * 0.4, 0.045)
        if hist_vol_20d >= 0.65:
            reversal_penalty -= 0.008
        if float(market_features.get("volume_ratio_20d") or market_features.get("volume_ratio") or 1.0) < 1.0:
            reversal_penalty -= 0.005
        if float(market_features.get("volume_ratio_20d") or market_features.get("volume_ratio") or 1.0) >= 2.5:
            reversal_penalty -= 0.01
    reversal_penalty += float(momentum_filter["reversal_penalty"])
    reversal_penalty += float(volume_divergence["reversal_penalty"])
    reversal_penalty += float(options_edge["reversal_penalty_delta"])
    reversal_penalty += float(volatility_gate["penalty"])

    effective_uncertainty_score = float(np.clip(float(uncertainty["uncertainty_score"]) - float(options_edge["confidence_delta"]), 0.0, 1.0))
    effective_predictive_confidence = float(np.clip(1.0 - effective_uncertainty_score, 0.05, 0.99))
    prediction_interval_multiplier = float(np.clip(1.0 - 0.5 * float(options_edge["confidence_delta"]), 0.7, 1.35))
    effective_prediction_interval_5d = float(
        np.clip(float(uncertainty["prediction_interval_5d"]) * prediction_interval_multiplier, 0.01, 0.12)
    )

    uncertainty_penalty = float(
        effective_prediction_interval_5d
        * (0.08 + 0.18 * uncertainty["expert_disagreement_score"])
    )

    raw_expected_return_5d = (
        alpha_component_5d
        + 0.055 * market_component
        + regime_bias
        + synergy
        + float(options_edge["options_edge"])
    )
    confidence_shrink = 0.7 + 0.3 * effective_predictive_confidence
    expected_return_5d = raw_expected_return_5d * confidence_shrink + reversal_penalty - uncertainty_penalty
    if momentum_filter["blowoff_top_block"]:
        expected_return_5d = min(expected_return_5d, -0.008)
    expected_return_5d = float(np.clip(expected_return_5d, -0.12, 0.15))
    dominant_dynamic_factor = max(dynamic_weights, key=lambda factor_name: abs(dynamic_weights[factor_name]))

    if momentum_filter["blowoff_top_block"]:
        driver = "RSI 过热阻断主导"
    elif volatility_gate["block"]:
        driver = "高波动门控主导"
    elif volume_divergence["distribution_risk"]:
        driver = "放量派发抑制主导"
    elif exhaustion_signal["exhaustion_reversal"]:
        driver = "过热反转风控主导"
    elif uncertainty["predictive_confidence"] < 0.45:
        driver = "不确定性抑制主导"
    elif abs(price_factor_component) >= abs(news_component):
        driver = "量价因子主导"
    else:
        driver = "消息情绪主导"
    if options_edge["options_flow_signal"] == "gamma_squeeze":
        driver = "期权逼空主导"
    elif options_edge["options_flow_signal"] == "put_divergence":
        driver = "期权背离抑制主导"
    elif options_edge["options_flow_signal"] == "put_panic_short":
        driver = "Put 恐慌空头延续主导"
    inference_log = (
        f"{driver}，配置状态 {allocation_regime['label']}，5日EV {expected_return_5d:.2%}，"
        f"置信度 {effective_predictive_confidence:.0%}，"
        f"主导专家 {expert_context['dominant_expert']}，"
        f"主导权重 {dominant_dynamic_factor}"
    )

    return {
        "expected_return_5d": expected_return_5d,
        "expected_return_5d_raw": raw_expected_return_5d,
        "expected_return_daily": expected_return_5d / 5.0,
        "rolling_alpha_daily": resolved_alpha,
        "momentum_core_10d": float(momentum_core["log_return_10d"]),
        "alpha_component_5d": alpha_component_5d,
        "market_component": market_component,
        "base_market_component": base_market_component,
        "price_factor_component": price_factor_component,
        "news_component": news_component,
        "allocation_regime": allocation_regime["label"],
        "allocation_reason": allocation_regime["reason"],
        "vix_value": allocation_regime["vix_value"],
        "expert_consensus_blend": expert_blend,
        "reversal_penalty": reversal_penalty,
        "uncertainty_penalty": uncertainty_penalty,
        "volatility_penalty": volatility_penalty_label(hist_vol_20d),
        "expert_consensus": float(expert_context["expert_consensus"]),
        "expert_disagreement": float(expert_context["expert_disagreement"]),
        "dominant_expert": str(expert_context["dominant_expert"]),
        "expert_weight_trend": float(expert_context["expert_weights"]["trend"]),
        "expert_weight_reversion": float(expert_context["expert_weights"]["reversion"]),
        "expert_weight_event": float(expert_context["expert_weights"]["event"]),
        "dynamic_weight_momentum": float(dynamic_weights["momentum"]),
        "dynamic_weight_relative_strength": float(dynamic_weights["relative_strength"]),
        "dynamic_weight_breakout": float(dynamic_weights["breakout"]),
        "dynamic_weight_rsi_alignment": float(dynamic_weights["rsi_alignment"]),
        "dynamic_weight_sentiment": float(dynamic_weights["sentiment_component"]),
        "dynamic_weight_shock": float(dynamic_weights["shock_component"]),
        "dynamic_weight_headline": float(dynamic_weights["headline_density"]),
        "dynamic_weight_volatility": float(dynamic_weights["volatility_term"]),
        "volatility_gate_state": str(volatility_gate["state"]),
        "volatility_gate_reason": str(volatility_gate["reason"]),
        "volatility_gate_block": bool(volatility_gate["block"]),
        "volatility_gate_release_multiplier": float(volatility_gate["release_multiplier"]),
        "call_put_volume_ratio": options_edge["call_put_volume_ratio"],
        "options_flow_signal": options_edge["options_flow_signal"],
        "options_flow_reason": options_edge["options_flow_reason"],
        "options_edge": float(options_edge["options_edge"]),
        "options_confidence_delta": float(options_edge["confidence_delta"]),
        "options_reversal_penalty_delta": float(options_edge["reversal_penalty_delta"]),
        "options_win_rate_multiplier": float(options_edge["options_win_rate_multiplier"]),
        "options_win_rate_delta": float(options_edge["options_win_rate_delta"]),
        **{
            **uncertainty,
            "uncertainty_score": effective_uncertainty_score,
            "predictive_confidence": effective_predictive_confidence,
            "prediction_interval_5d": effective_prediction_interval_5d,
        },
        **exhaustion_signal,
        "inference_log": inference_log,
    }


def classify_signal_mode(market_features: dict[str, Any]) -> dict[str, str]:
    rsi_14 = float(market_features.get("rsi_14") or 50.0)
    momentum_term = float(_resolve_momentum_core(market_features)["momentum_term"])
    volume_ratio = float(market_features.get("volume_ratio_20d") or 1.0)
    relative_strength = float(market_features.get("relative_strength_20d_vs_benchmark") or 0.0)
    breakout = bool(market_features.get("breakout_20d"))
    daily_return_pct = float(market_features.get("daily_return_pct") or 0.0)
    close = float(market_features.get("close") or 0.0)
    sma_20 = float(market_features.get("sma_20") or close or 0.0)
    strong_intermediate_trend = momentum_term >= 0.9
    medium_intermediate_trend = momentum_term >= 0.45
    healthy_rsi = 40.0 <= rsi_14 <= 60.0
    stretched_rsi = 60.0 < rsi_14 <= 68.0

    # Medium-quality momentum only gets released in a healthy RSI corridor.
    # Stronger intermediate momentum may survive a mildly stretched RSI, but never a blow-off RSI.
    if breakout and volume_ratio >= 1.2 and relative_strength >= 0.0 and rsi_14 <= 70.0:
        if medium_intermediate_trend and healthy_rsi:
            return {
                "signal_mode": "momentum",
                "signal_label": "动能延续",
                "signal_reason": "10日动量在健康 RSI 区间释放",
            }
        if strong_intermediate_trend and (healthy_rsi or stretched_rsi):
            return {
                "signal_mode": "momentum",
                "signal_label": "动能延续",
                "signal_reason": "10日动量强、相对强度同步确认",
            }

    if (
        not breakout
        and daily_return_pct <= -2.0
        and momentum_term <= -0.55
        and volume_ratio >= 1.2
        and relative_strength <= 0.0
        and close <= sma_20
        and rsi_14 <= 48.0
    ):
        return {
            "signal_mode": "short_breakdown",
            "signal_label": "破位沽空",
            "signal_reason": "10日动量转弱、放量、弱于基准",
        }

    if (
        not breakout
        and rsi_14 <= 35.0
        and momentum_term <= -0.4
        and daily_return_pct <= -1.5
        and close <= sma_20
        and volume_ratio <= 1.8
    ):
        return {
            "signal_mode": "mean_reversion",
            "signal_label": "均值回归",
            "signal_reason": "10日动量过弱，进入超跌反抽观察区",
        }

    return {
        "signal_mode": "neutral",
        "signal_label": "观望",
        "signal_reason": "未触发动量或均值回归高胜率模板",
    }


def _earnings_event_reason(days_to_earnings: int | None) -> str:
    if days_to_earnings is None:
        return ""
    if days_to_earnings <= 3:
        return "财报高危期，强制空仓"
    if days_to_earnings <= 5:
        return "财报临近，降低信号置信度"
    return ""


def evaluate_trade_gate(
    signal_mode: str,
    expected_return_5d: float,
    trade_stats: dict[str, Any],
    regime_score: float,
) -> dict[str, Any]:
    if trade_stats.get("earnings_event_block"):
        return {
            "eligible": False,
            "reason": str(trade_stats.get("earnings_gate_reason") or "财报高危期，强制空仓"),
        }

    if regime_score <= GLOBAL_OBSERVE_REGIME_THRESHOLD:
        return {
            "eligible": False,
            "reason": "市场环境过弱，触发空仓/观望约束",
        }

    if signal_mode == "neutral":
        return {
            "eligible": False,
            "reason": "未触发历史高胜率模板",
        }

    thresholds = STRATEGY_THRESHOLDS[signal_mode]
    if regime_score < thresholds["min_regime_score"]:
        return {
            "eligible": False,
            "reason": "市场状态不支持新增风险暴露",
        }
    edge_return_5d = abs(expected_return_5d) if signal_mode == "short_breakdown" else expected_return_5d
    if edge_return_5d < thresholds["min_expected_return_5d"]:
        return {
            "eligible": False,
            "reason": "5日EV未达到入场阈值",
        }
    if trade_stats["target_ev"] < thresholds["min_ev"]:
        return {
            "eligible": False,
            "reason": "目标EV低于强制阈值",
        }
    if trade_stats["win_rate"] < thresholds["min_win_rate"]:
        return {
            "eligible": False,
            "reason": "统计胜率不足",
        }
    if trade_stats["payoff_ratio"] < thresholds["min_payoff_ratio"]:
        return {
            "eligible": False,
            "reason": "盈亏比不足",
        }
    return {
        "eligible": True,
        "reason": "通过高胜率阈值",
    }


def build_trade_profile(
    ticker: str,
    market_features: dict[str, Any],
    news_features: dict[str, Any],
    regime_score: float,
    rolling_alpha: float | None = None,
    vix_value: float | None = None,
    options_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    earnings_event = get_upcoming_earnings(ticker)
    signal_mode = classify_signal_mode(market_features)
    resolved_alpha = rolling_alpha
    if resolved_alpha is None:
        cached_alpha = market_features.get("rolling_alpha")
        if isinstance(cached_alpha, (int, float)) and np.isfinite(cached_alpha):
            resolved_alpha = float(cached_alpha)

    volume_ratio = float(market_features.get("volume_ratio_20d") or 1.0)
    relative_strength = float(market_features.get("relative_strength_20d_vs_benchmark") or 0.0)
    breakout = bool(market_features.get("breakout_20d"))
    breakdown_valid = bool(signal_mode["signal_mode"] == "short_breakdown")
    breakout_valid = bool(signal_mode["signal_mode"] in {"momentum", "short_breakdown"} and ((breakout and volume_ratio >= 1.2 and relative_strength >= 0.0) or breakdown_valid))
    fake_breakout = bool(breakout and not breakout_valid)

    expected = estimate_expected_return(
        ticker,
        {
            **market_features,
            "breakout_valid": breakout_valid,
            "fake_breakout": fake_breakout,
            "short_trend_confirmed": breakdown_valid,
        },
        news_features,
        regime_score,
        rolling_alpha=resolved_alpha,
        vix_value=vix_value,
        options_data=options_data,
    )
    trade_stats = estimate_trade_stats(
        expected["expected_return_5d"],
        volume_ratio,
        breakout_valid,
        fake_breakout,
        float(expected.get("uncertainty_score") or 0.0),
        days_to_earnings=earnings_event.get("days_to_earnings"),
        options_data=options_data,
        short_trend_confirmed=breakdown_valid,
        long_breakout_confirmed=bool(signal_mode["signal_mode"] == "momentum" and breakout_valid),
        market_features=market_features,
    )
    if expected.get("volatility_gate_block"):
        gate = {
            "eligible": False,
            "reason": str(expected.get("volatility_gate_reason") or "20日波动率偏高，暂不放行动能追涨"),
        }
    else:
        gate = evaluate_trade_gate(
            signal_mode["signal_mode"],
            expected["expected_return_5d"],
            trade_stats,
            regime_score,
        )
    if expected.get("exhaustion_reversal"):
        gate = {
            "eligible": False,
            "reason": str(expected.get("exhaustion_reason") or "过热反转风控触发"),
        }

    inference_log = (
        f"{signal_mode['signal_label']} / {expected['inference_log']} / "
        f"{gate['reason']}"
    )
    return {
        **expected,
        **signal_mode,
        **trade_stats,
        **earnings_event,
        "trade_direction": "SHORT" if breakdown_valid else ("LONG" if signal_mode["signal_mode"] == "momentum" else "FLAT"),
        "breakdown": breakdown_valid,
        "breakout_valid": breakout_valid,
        "fake_breakout": fake_breakout,
        "eligible_for_risk": gate["eligible"],
        "gate_reason": gate["reason"],
        "inference_log": inference_log,
    }


def build_candidate_trade_profile(
    candidate: dict[str, Any],
    regime_score: float,
    news_features: dict[str, Any] | None = None,
    vix_value: float | None = None,
    options_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exhaustion_signal = detect_exhaustion_reversal(candidate)
    earnings_event = get_upcoming_earnings(str(candidate.get("ticker") or ""))
    candidate_news = news_features or {}
    candidate_expected = estimate_expected_return(
        str(candidate.get("ticker") or ""),
        candidate,
        candidate_news,
        regime_score,
        vix_value=vix_value,
        options_data=options_data,
    )
    if candidate.get("breakdown"):
        signal_mode = {
            "signal_mode": "short_breakdown",
            "signal_label": "破位沽空",
            "signal_reason": "放量跌破模板",
        }
    elif candidate.get("breakout_valid"):
        signal_mode = {
            "signal_mode": "momentum",
            "signal_label": "动能延续",
            "signal_reason": "放量突破模板",
        }
    else:
        signal_mode = {
            "signal_mode": "neutral",
            "signal_label": "观望",
            "signal_reason": "未触发突破阈值",
        }

    confirmed_setup = bool(candidate.get("breakout_valid") or candidate.get("breakdown"))
    trade_stats = estimate_trade_stats(
        float(candidate_expected.get("expected_return_5d") or 0.0),
        float(candidate.get("volume_ratio_20d") or 0.0),
        confirmed_setup,
        bool(candidate.get("fake_breakout")),
        float(candidate_expected.get("uncertainty_score") or 0.0),
        days_to_earnings=earnings_event.get("days_to_earnings"),
        options_data=options_data,
        short_trend_confirmed=bool(candidate.get("breakdown") or str(candidate.get("trade_direction") or "").upper() == "SHORT"),
        long_breakout_confirmed=bool(candidate.get("breakout_valid") and not candidate.get("breakdown")),
        market_features=candidate,
    )
    if candidate_expected.get("volatility_gate_block"):
        gate = {
            "eligible": False,
            "reason": str(candidate_expected.get("volatility_gate_reason") or "20日波动率偏高，暂不放行动能追涨"),
        }
    else:
        gate = evaluate_trade_gate(
            signal_mode["signal_mode"],
            float(candidate_expected.get("expected_return_5d") or 0.0),
            trade_stats,
            regime_score,
        )
    if exhaustion_signal["exhaustion_reversal"] and str(candidate.get("trade_direction") or "LONG").upper() == "LONG":
        gate = {
            "eligible": False,
            "reason": str(exhaustion_signal.get("exhaustion_reason") or "过热反转风控触发"),
        }
    return {
        **candidate,
        **candidate_expected,
        **signal_mode,
        **trade_stats,
        **earnings_event,
        **exhaustion_signal,
        "expected_return_10d": round(float(candidate_expected.get("expected_return_5d") or 0.0) * 1.6, 4),
        "signal_strength": abs(float(candidate_expected.get("expected_return_5d") or 0.0)) * max(float(candidate.get("volume_ratio_20d") or 0.1), 0.1),
        "trade_direction": "SHORT" if candidate.get("breakdown") else ("LONG" if candidate.get("breakout_valid") else "FLAT"),
        "eligible_for_risk": gate["eligible"],
        "gate_reason": gate["reason"],
    }


def should_force_observe(
    core_profiles: dict[str, dict[str, Any]],
    candidate_profiles: list[dict[str, Any]],
    regime_score: float,
) -> tuple[bool, str]:
    if regime_score <= GLOBAL_OBSERVE_REGIME_THRESHOLD:
        return True, "市场风险偏好过低"

    core_eligible = any(profile.get("eligible_for_risk") for profile in core_profiles.values())
    candidate_eligible = any(profile.get("eligible_for_risk") for profile in candidate_profiles)
    if not core_eligible and not candidate_eligible:
        return True, "未触发任何高胜率阈值"

    return False, ""


def estimate_trade_stats(
    expected_return_5d: float,
    volume_ratio: float,
    breakout_valid: bool,
    fake_breakout: bool,
    uncertainty_score: float = 0.0,
    days_to_earnings: int | None = None,
    options_data: dict[str, Any] | None = None,
    short_trend_confirmed: bool = False,
    long_breakout_confirmed: bool = False,
    market_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edge_return_5d = abs(expected_return_5d)
    win_rate = 0.42
    volume_divergence = _assess_volume_divergence(market_features or {}, long_breakout_confirmed or breakout_valid)
    if breakout_valid:
        win_rate += 0.12
    if fake_breakout:
        win_rate -= 0.15
    # Volume enters with a piecewise score: 1.2-2.5x confirms participation, while extreme volume above 3.0x
    # is treated as a discontinuity/distribution risk and subtracts confidence instead of linearly adding to it.
    if breakout_valid and 1.2 < volume_ratio < 2.5 and not volume_divergence["distribution_risk"]:
        win_rate += 0.08 * float(np.clip((volume_ratio - 1.2) / 1.3, 0.0, 1.0))
    elif volume_ratio > 3.0:
        win_rate -= 0.1 * float(np.clip((volume_ratio - 3.0) / 1.5, 0.0, 1.0))
    if volume_divergence["distribution_risk"]:
        win_rate -= float(volume_divergence["win_rate_penalty"])
    win_rate -= 0.14 * float(np.clip(uncertainty_score, 0.0, 1.0))
    options_edge = _build_options_edge(options_data, long_breakout_confirmed, short_trend_confirmed)
    win_rate = win_rate * float(options_edge["options_win_rate_multiplier"]) + float(options_edge["options_win_rate_delta"])
    earnings_event_block = bool(days_to_earnings is not None and days_to_earnings <= 3)
    earnings_gate_reason = _earnings_event_reason(days_to_earnings)
    if earnings_event_block:
        win_rate = min(win_rate, 0.2)
    elif days_to_earnings is not None and days_to_earnings <= 5:
        win_rate -= 0.08
    win_rate += 0.12 * float(np.clip(edge_return_5d / 0.08, 0.0, 1.0))
    win_rate = float(np.clip(win_rate, 0.2, 0.8))

    earnings_stop_buffer = 0.03 if earnings_event_block else (0.01 if days_to_earnings is not None and days_to_earnings <= 5 else 0.0)
    stop_loss_pct = float(np.clip(0.03 + edge_return_5d * 0.6 + 0.02 * float(np.clip(uncertainty_score, 0.0, 1.0)) + earnings_stop_buffer, 0.03, 0.1))
    reward_pct = float(max(edge_return_5d * (1.0 - 0.25 * float(np.clip(uncertainty_score, 0.0, 1.0))), 0.01))
    payoff_ratio = reward_pct / stop_loss_pct if stop_loss_pct else 0.0
    ev = win_rate * reward_pct - (1.0 - win_rate) * stop_loss_pct
    return {
        "win_rate": win_rate,
        "stop_loss_pct": stop_loss_pct,
        "payoff_ratio": payoff_ratio,
        "target_ev": ev,
        "days_to_earnings": days_to_earnings,
        "earnings_event_block": earnings_event_block,
        "earnings_gate_reason": earnings_gate_reason,
        "call_put_volume_ratio": options_edge["call_put_volume_ratio"],
        "options_flow_signal": options_edge["options_flow_signal"],
        "options_flow_reason": options_edge["options_flow_reason"],
    }
