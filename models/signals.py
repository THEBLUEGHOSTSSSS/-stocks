from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.fetcher import get_upcoming_earnings
from models.regime import classify_factor_allocation_regime


GLOBAL_OBSERVE_REGIME_THRESHOLD = 0.34
RSI_REVERSAL_BUY_THRESHOLD = 35.0
RSI_REVERSAL_SELL_THRESHOLD = 70.0
RSI_TREND_FILTER_LOOKBACK = 120
RSI_REVERSAL_BASE_EV = 0.02
RSI_REVERSAL_EV_MULTIPLIER = 0.002
RSI_NEUTRAL_BLOCK_EV = -0.01
RSI_LIQUIDATE_EV = -0.05
STRATEGY_THRESHOLDS = {
    "rsi_reversion_long": {
        "min_regime_score": 0.0,
        "min_expected_return_5d": 0.02,
        "min_ev": 0.0,
        "min_win_rate": 0.45,
        "min_payoff_ratio": 0.70,
    },
}


def _is_rsi_reversion_signal_mode(signal_mode: str) -> bool:
    return signal_mode == "rsi_reversion_long"


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
    del market_features
    return {
        "log_return_60d": 0.0,
        "momentum_term": 0.0,
    }


def _resolve_breakout_flag(market_features: dict[str, Any]) -> bool:
    del market_features
    return False


def _resolve_rsi_trend_filter(market_features: dict[str, Any]) -> dict[str, Any]:
    close = _coerce_finite_float(market_features.get("close"), np.nan)
    sma_120 = _coerce_finite_float(market_features.get("sma_120"), np.nan)
    if not np.isfinite(close) or not np.isfinite(sma_120) or sma_120 <= 0.0:
        return {
            "trend_filter_pass": False,
            "knife_catch_block": True,
            "trend_filter_reason": f"{RSI_TREND_FILTER_LOOKBACK} 日均线数据不足，禁止接飞刀式抄底",
        }
    if close > sma_120:
        return {
            "trend_filter_pass": True,
            "knife_catch_block": False,
            "trend_filter_reason": f"close={close:.2f} > sma_120={sma_120:.2f}，通过 {RSI_TREND_FILTER_LOOKBACK} 日趋势过滤",
        }
    return {
        "trend_filter_pass": False,
        "knife_catch_block": True,
        "trend_filter_reason": f"close={close:.2f} <= sma_120={sma_120:.2f}，判定为接飞刀风险，禁止低吸",
    }


def _resolve_rsi_reversion_signal_mode(market_features: dict[str, Any]) -> dict[str, Any]:
    rsi_14 = _coerce_finite_float(market_features.get("rsi_14"), 50.0)
    trend_state = _resolve_rsi_trend_filter(market_features)
    if rsi_14 < RSI_REVERSAL_BUY_THRESHOLD:
        if not trend_state["trend_filter_pass"]:
            return {
                "signal_mode": "neutral",
                "signal_label": "观望",
                "signal_reason": str(
                    market_features.get("signal_reason")
                    or (
                        f"rsi_14={rsi_14:.1f} < {RSI_REVERSAL_BUY_THRESHOLD:.0f}，但 {trend_state['trend_filter_reason']}，"
                        "V2.5 趋势过滤 RSI 低吸判定为接飞刀，强制观望"
                    )
                ),
                **trend_state,
            }
        return {
            "signal_mode": "rsi_reversion_long",
            "signal_label": "RSI 趋势过滤低吸",
            "signal_reason": str(
                market_features.get("signal_reason")
                or (
                    f"rsi_14={rsi_14:.1f} < {RSI_REVERSAL_BUY_THRESHOLD:.0f}，且 {trend_state['trend_filter_reason']}，"
                    "执行 V2.5 趋势过滤 RSI 低吸"
                )
            ),
            **trend_state,
        }
    if rsi_14 >= RSI_REVERSAL_SELL_THRESHOLD:
        return {
            "signal_mode": "liquidate",
            "signal_label": "RSI 趋势过滤止盈",
            "signal_reason": str(
                market_features.get("signal_reason")
                or f"rsi_14={rsi_14:.1f} >= {RSI_REVERSAL_SELL_THRESHOLD:.0f}，进入贪婪止盈区，强制收缩仓位"
            ),
            **trend_state,
        }
    return {
        "signal_mode": "neutral",
        "signal_label": "观望",
        "signal_reason": (
            f"rsi_14={rsi_14:.1f} 位于 {RSI_REVERSAL_BUY_THRESHOLD:.0f}-{RSI_REVERSAL_SELL_THRESHOLD:.0f} 之间，"
            "V2.5 趋势过滤 RSI 低吸策略强制管住手"
        ),
        **trend_state,
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
        if high_quality and rsi_14 <= 76.0:
            return {
                "release_multiplier": 0.38,
                "reversal_penalty": -(0.012 + 0.008 * float(np.clip((rsi_14 - 70.0) / 6.0, 0.0, 1.0))),
                "blowoff_top_block": False,
                "state": "high_quality_overheat_release",
            }
        if rsi_14 <= 74.0:
            return {
                "release_multiplier": 0.18,
                "reversal_penalty": -(0.016 + 0.008 * float(np.clip((rsi_14 - 70.0) / 4.0, 0.0, 1.0))),
                "blowoff_top_block": False,
                "state": "cautious_overheat_release",
            }
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
        daily_return_pct >= 7.0
        and rsi_14 >= 78.0
        and momentum_term >= 0.98
        and (volume_ratio >= 2.5 or hist_vol_20d >= 0.75)
    )

    if downside_capitulation:
        reasons: list[str] = []
        if daily_return_pct <= -4.0:
            reasons.append("单日急跌")
        if rsi_14 >= 70.0:
            reasons.append("RSI处于过热区")
        if momentum_term >= 0.85:
            reasons.append("60日动量过热")
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
        reasons = ["单日逼空拉升", "RSI进入极热区", "60日动量过热"]
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
        if breakout_valid and momentum_term >= 0.85:
            return {
                "release_multiplier": 0.55 if regime_label == "Risk-On" else 0.42,
                "penalty": -0.0035 if regime_label == "Risk-On" else -0.005,
                "block": False,
                "state": "high_vol_trend_release",
                "reason": "高波动但趋势强，缩量放行部分动能",
            }
        if breakout_valid and regime_label == "Risk-On" and momentum_term >= 0.55:
            return {
                "release_multiplier": 0.72,
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
    del ticker, news_features, options_data
    hist_vol_20d = float(market_features.get("hist_vol_20d") or 0.0)
    orthogonal_alpha_score = float(market_features.get("orthogonal_alpha_score") or 0.0)
    idio_zscore = float(market_features.get("idio_zscore") or 0.0)
    rsi_14 = _coerce_finite_float(market_features.get("rsi_14"), 50.0)
    signal_mode = _resolve_rsi_reversion_signal_mode(market_features)
    rsi_reversion_active = _is_rsi_reversion_signal_mode(signal_mode["signal_mode"])
    knife_catch_block = bool(signal_mode.get("knife_catch_block"))
    allocation_regime = classify_factor_allocation_regime(regime_score, vix_value)
    resolved_alpha = float(rolling_alpha) if rolling_alpha is not None and np.isfinite(rolling_alpha) else 0.0
    adjusted_ev = RSI_NEUTRAL_BLOCK_EV
    effective_uncertainty_score = 0.95
    effective_predictive_confidence = 0.05
    effective_prediction_interval_5d = 0.12
    rsi_reversion_alpha_bonus_5d = 0.0
    if signal_mode["signal_mode"] == "liquidate":
        adjusted_ev = RSI_LIQUIDATE_EV
        effective_uncertainty_score = 0.18
        effective_predictive_confidence = 0.82
        effective_prediction_interval_5d = 0.025
    elif rsi_reversion_active:
        oversold_gap = float(max(RSI_REVERSAL_BUY_THRESHOLD - rsi_14, 0.0))
        # V2.5 趋势过滤 RSI 低吸：只有超卖且站上 120 日均线时才给正 EV。
        adjusted_ev = float(np.clip(RSI_REVERSAL_BASE_EV + oversold_gap * RSI_REVERSAL_EV_MULTIPLIER, 0.02, 0.08))
        rsi_reversion_alpha_bonus_5d = adjusted_ev
        effective_uncertainty_score = float(np.clip(0.34 - 0.012 * oversold_gap, 0.08, 0.34))
        effective_predictive_confidence = float(np.clip(1.0 - effective_uncertainty_score, 0.66, 0.92))
        effective_prediction_interval_5d = float(np.clip(0.036 - 0.0015 * oversold_gap, 0.012, 0.036))
    elif knife_catch_block:
        adjusted_ev = min(RSI_NEUTRAL_BLOCK_EV, -0.02)
        effective_uncertainty_score = 0.88
        effective_predictive_confidence = 0.12
        effective_prediction_interval_5d = 0.08

    inference_log = (
        f"V2.5 趋势过滤 RSI 低吸 / 配置状态 {allocation_regime['label']} / "
        f"rsi_14 {rsi_14:.1f} / idio_zscore {idio_zscore:.2f} / adjusted_ev {adjusted_ev:.2%} / "
        f"{signal_mode['signal_reason']}"
    )

    return {
        "expected_return_5d": adjusted_ev,
        "expected_return_5d_raw": adjusted_ev,
        "adjusted_ev": adjusted_ev,
        "expected_return_daily": adjusted_ev / 5.0,
        "rolling_alpha_daily": resolved_alpha,
        "momentum_core_10d": 0.0,
        "momentum_core_60d": 0.0,
        "alpha_component_5d": rsi_reversion_alpha_bonus_5d,
        "stat_arb_alpha_bonus_5d": 0.0,
        "idio_momentum_alpha_bonus_5d": 0.0,
        "rsi_reversion_alpha_bonus_5d": rsi_reversion_alpha_bonus_5d,
        "market_component": 0.0,
        "base_market_component": 0.0,
        "price_factor_component": 0.0,
        "orthogonal_alpha_score": orthogonal_alpha_score,
        "orthogonal_alpha_component": 0.0,
        "idio_zscore": idio_zscore,
        "rsi_14": rsi_14,
        "idiosyncratic_component": adjusted_ev,
        "stat_arb_priority": False,
        "spillover_component": 0.0,
        "news_component": 0.0,
        "allocation_regime": allocation_regime["label"],
        "allocation_reason": allocation_regime["reason"],
        "vix_value": allocation_regime["vix_value"],
        "trend_filter_pass": bool(signal_mode.get("trend_filter_pass")),
        "trend_filter_reason": str(signal_mode.get("trend_filter_reason") or ""),
        "knife_catch_block": knife_catch_block,
        "expert_consensus_blend": 0.0,
        "reversal_penalty": 0.0,
        "uncertainty_penalty": 0.0,
        "volatility_penalty": volatility_penalty_label(hist_vol_20d),
        "expert_consensus": 1.0 if rsi_reversion_active else 0.0,
        "expert_disagreement": 0.0,
        "dominant_expert": "rsi_reversion" if rsi_reversion_active else "risk_control",
        "expert_weight_trend": 0.0,
        "expert_weight_reversion": 1.0 if rsi_reversion_active else 0.0,
        "expert_weight_event": 0.0,
        "dynamic_weight_momentum": 0.0,
        "dynamic_weight_relative_strength": 0.0,
        "dynamic_weight_breakout": 0.0,
        "dynamic_weight_rsi_alignment": 0.0,
        "dynamic_weight_sentiment": 0.0,
        "dynamic_weight_shock": 0.0,
        "dynamic_weight_headline": 0.0,
        "dynamic_weight_volatility": 0.0,
        "volatility_gate_state": "rsi_extreme_only_mode",
        "volatility_gate_reason": "V2.5 趋势过滤 RSI 低吸：只有 RSI 进入 35 以下且价格站上 SMA120 才允许新增风险暴露",
        "volatility_gate_block": bool(not rsi_reversion_active),
        "volatility_gate_release_multiplier": 1.0 if rsi_reversion_active else 0.0,
        "call_put_volume_ratio": 1.0,
        "options_flow_signal": "neutral",
        "options_flow_reason": "V2.5 趋势过滤 RSI 低吸：期权流不再主导买卖方向",
        "options_edge": 0.0,
        "options_confidence_delta": 0.0,
        "options_reversal_penalty_delta": 0.0,
        "options_win_rate_multiplier": 1.0,
        "options_win_rate_delta": 0.0,
        **{
            "uncertainty_score": effective_uncertainty_score,
            "predictive_confidence": effective_predictive_confidence,
            "prediction_interval_5d": effective_prediction_interval_5d,
            "expert_disagreement_score": 0.0,
            "routing_uncertainty": 0.0,
            "crowding_risk": 0.0,
        },
        **{
            "exhaustion_reversal": False,
            "exhaustion_reason": "",
        },
        "inference_log": inference_log,
    }


def classify_signal_mode(market_features: dict[str, Any]) -> dict[str, str]:
    return _resolve_rsi_reversion_signal_mode(market_features)


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
    signal_reason: str | None = None,
) -> dict[str, Any]:
    if trade_stats.get("earnings_event_block"):
        return {
            "eligible": False,
            "reason": str(trade_stats.get("earnings_gate_reason") or "财报高危期，强制空仓"),
        }

    if signal_mode == "liquidate":
        return {
            "eligible": False,
            "reason": "rsi_14 已进入 70 以上贪婪区，执行止盈/清仓",
        }

    if signal_mode == "neutral":
        return {
            "eligible": False,
            "reason": str(signal_reason or "rsi_14 未进入 35 以下极端低吸区，强制观望"),
        }

    if not _is_rsi_reversion_signal_mode(signal_mode):
        return {
            "eligible": False,
            "reason": "仅允许 RSI 极端超卖低吸进入风险预算",
        }

    thresholds = STRATEGY_THRESHOLDS[signal_mode]
    if regime_score < thresholds["min_regime_score"]:
        return {
            "eligible": False,
            "reason": "市场状态不支持新增风险暴露",
        }
    edge_return_5d = abs(expected_return_5d)
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
        "reason": "通过 V2.5 趋势过滤 RSI 低吸阈值",
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
    rsi_reversion_confirmed = _is_rsi_reversion_signal_mode(signal_mode["signal_mode"])
    resolved_alpha = rolling_alpha
    if resolved_alpha is None:
        cached_alpha = market_features.get("rolling_alpha")
        if isinstance(cached_alpha, (int, float)) and np.isfinite(cached_alpha):
            resolved_alpha = float(cached_alpha)

    volume_ratio = float(market_features.get("volume_ratio_20d") or 1.0)

    expected = estimate_expected_return(
        ticker,
        market_features,
        news_features,
        regime_score,
        rolling_alpha=resolved_alpha,
        vix_value=vix_value,
        options_data=options_data,
    )
    trade_stats = estimate_trade_stats(
        expected["expected_return_5d"],
        volume_ratio,
        False,
        False,
        float(expected.get("uncertainty_score") or 0.0),
        days_to_earnings=earnings_event.get("days_to_earnings"),
        options_data=options_data,
        short_trend_confirmed=False,
        long_breakout_confirmed=False,
        stat_arb_confirmed=False,
        rsi_reversion_confirmed=rsi_reversion_confirmed,
        market_features=market_features,
    )
    gate = evaluate_trade_gate(
        signal_mode["signal_mode"],
        expected["expected_return_5d"],
        trade_stats,
        regime_score,
        signal_reason=str(signal_mode.get("signal_reason") or ""),
    )

    trade_direction = "FLAT"
    if signal_mode["signal_mode"] == "rsi_reversion_long":
        trade_direction = "LONG"

    inference_log = (
        f"{signal_mode['signal_label']} / {expected['inference_log']} / "
        f"{gate['reason']}"
    )
    return {
        **expected,
        **signal_mode,
        **trade_stats,
        **earnings_event,
        "trade_direction": trade_direction,
        "breakdown": False,
        "breakout_valid": False,
        "fake_breakout": False,
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
    exhaustion_signal = {
        "exhaustion_reversal": False,
        "exhaustion_reason": "",
    }
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
    signal_mode = classify_signal_mode(candidate)
    rsi_reversion_confirmed = _is_rsi_reversion_signal_mode(signal_mode["signal_mode"])

    trade_stats = estimate_trade_stats(
        float(candidate_expected.get("expected_return_5d") or 0.0),
        float(candidate.get("volume_ratio_20d") or 0.0),
        False,
        False,
        float(candidate_expected.get("uncertainty_score") or 0.0),
        days_to_earnings=earnings_event.get("days_to_earnings"),
        options_data=options_data,
        short_trend_confirmed=False,
        long_breakout_confirmed=False,
        stat_arb_confirmed=False,
        rsi_reversion_confirmed=rsi_reversion_confirmed,
        market_features=candidate,
    )
    gate = evaluate_trade_gate(
        signal_mode["signal_mode"],
        float(candidate_expected.get("expected_return_5d") or 0.0),
        trade_stats,
        regime_score,
        signal_reason=str(signal_mode.get("signal_reason") or ""),
    )

    trade_direction = "FLAT"
    if signal_mode["signal_mode"] == "rsi_reversion_long":
        trade_direction = "LONG"
    return {
        **candidate,
        **candidate_expected,
        **signal_mode,
        **trade_stats,
        **earnings_event,
        **exhaustion_signal,
        "expected_return_10d": round(float(candidate_expected.get("expected_return_5d") or 0.0) * 1.6, 4),
        "signal_strength": (
            max(RSI_REVERSAL_BUY_THRESHOLD - float(candidate.get("rsi_14") or 50.0), 0.0)
            * abs(float(candidate_expected.get("expected_return_5d") or 0.0))
            if signal_mode["signal_mode"] == "rsi_reversion_long"
            else 0.0
        ),
        "trade_direction": trade_direction,
        "eligible_for_risk": gate["eligible"],
        "gate_reason": gate["reason"],
    }


def should_force_observe(
    core_profiles: dict[str, dict[str, Any]],
    candidate_profiles: list[dict[str, Any]],
    regime_score: float,
) -> tuple[bool, str]:
    del regime_score
    core_eligible = any(profile.get("eligible_for_risk") for profile in core_profiles.values())
    candidate_eligible = any(profile.get("eligible_for_risk") for profile in candidate_profiles)
    if core_eligible or candidate_eligible:
        return False, ""
    return True, "未触发任何趋势过滤 RSI 低吸阈值"


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
    stat_arb_confirmed: bool = False,
    rsi_reversion_confirmed: bool = False,
    market_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edge_return_5d = abs(expected_return_5d)
    win_rate = 0.42
    volume_divergence = _assess_volume_divergence(market_features or {}, long_breakout_confirmed or breakout_valid)
    if breakout_valid:
        win_rate += 0.12
    if rsi_reversion_confirmed:
        rsi_14 = _coerce_finite_float((market_features or {}).get("rsi_14"), 50.0)
        oversold_gap = max(RSI_REVERSAL_BUY_THRESHOLD - rsi_14, 0.0)
        win_rate += 0.10 + 0.08 * float(np.clip(oversold_gap / 15.0, 0.0, 1.0))
    if stat_arb_confirmed:
        idio_zscore = abs(float((market_features or {}).get("idio_zscore") or 0.0))
        win_rate += 0.10 * float(np.clip(idio_zscore / 3.0, 0.0, 1.0))
        if idio_zscore >= 1.0:
            win_rate += 0.06 * float(np.clip((idio_zscore - 1.0) / 1.5, 0.0, 1.0))
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
    uncertainty_term = float(np.clip(uncertainty_score, 0.0, 1.0))
    if rsi_reversion_confirmed:
        stop_loss_pct = float(np.clip(0.022 + edge_return_5d * 0.30 + 0.014 * uncertainty_term + earnings_stop_buffer, 0.022, 0.07))
        reward_pct = float(max(edge_return_5d * (1.40 - 0.10 * uncertainty_term), 0.012))
    elif stat_arb_confirmed:
        stop_loss_pct = float(np.clip(0.022 + edge_return_5d * 0.28 + 0.016 * uncertainty_term + earnings_stop_buffer, 0.022, 0.075))
        reward_pct = float(max(edge_return_5d * (1.35 - 0.10 * uncertainty_term), 0.012))
    elif long_breakout_confirmed:
        stop_loss_pct = float(np.clip(0.024 + edge_return_5d * 0.35 + 0.015 * uncertainty_term + earnings_stop_buffer, 0.025, 0.08))
        reward_pct = float(max(edge_return_5d * (1.55 - 0.15 * uncertainty_term), 0.012))
    elif breakout_valid or short_trend_confirmed:
        stop_loss_pct = float(np.clip(0.028 + edge_return_5d * 0.45 + 0.018 * uncertainty_term + earnings_stop_buffer, 0.028, 0.09))
        reward_pct = float(max(edge_return_5d * (1.30 - 0.20 * uncertainty_term), 0.011))
    else:
        stop_loss_pct = float(np.clip(0.03 + edge_return_5d * 0.6 + 0.02 * uncertainty_term + earnings_stop_buffer, 0.03, 0.1))
        reward_pct = float(max(edge_return_5d * (1.0 - 0.25 * uncertainty_term), 0.01))
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
