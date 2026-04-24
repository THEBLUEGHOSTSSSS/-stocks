from __future__ import annotations

from typing import Any

import numpy as np


GLOBAL_OBSERVE_REGIME_THRESHOLD = 0.34
STRATEGY_THRESHOLDS = {
    "momentum": {
        "min_regime_score": 0.48,
        "min_expected_return_5d": 0.02,
        "min_ev": 0.006,
        "min_win_rate": 0.57,
        "min_payoff_ratio": 1.05,
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


def volatility_penalty_label(hist_vol_20d: float | None) -> str:
    if hist_vol_20d is None:
        return "Medium"
    if hist_vol_20d >= 0.65:
        return "High"
    if hist_vol_20d >= 0.35:
        return "Medium"
    return "Low"


def estimate_expected_return(
    ticker: str,
    market_features: dict[str, Any],
    news_features: dict[str, Any],
    regime_score: float,
) -> dict[str, Any]:
    log_return_5d = float(market_features.get("log_return_5d") or 0.0)
    hist_vol_20d = float(market_features.get("hist_vol_20d") or 0.0)
    relative_strength = float(market_features.get("relative_strength_20d_vs_benchmark") or 0.0)
    rsi_14 = float(market_features.get("rsi_14") or 50.0)
    breakout = 1.0 if market_features.get("breakout_20d") else 0.0

    momentum_term = _clip_feature(log_return_5d, 0.08)
    relative_strength_term = _clip_feature(relative_strength, 0.15)
    volatility_term = _clip_feature(hist_vol_20d, 0.8, lower=0.0, upper=1.0)

    if rsi_14 > 75.0:
        rsi_alignment = -0.45
    elif rsi_14 < 35.0:
        rsi_alignment = 0.2
    elif 55.0 <= rsi_14 <= 68.0:
        rsi_alignment = 0.35
    else:
        rsi_alignment = 0.0

    market_component = (
        0.45 * momentum_term
        + 0.2 * relative_strength_term
        + 0.15 * breakout
        + 0.2 * rsi_alignment
        - 0.35 * volatility_term
    )

    sentiment_component = float(news_features.get("activated_sentiment") or 0.0)
    shock_component = _clip_feature(float(news_features.get("shock_score") or 0.0), 4.0)
    headline_density = _clip_feature(float(news_features.get("headline_count") or 0.0), 8.0, lower=0.0, upper=1.0)
    news_component = 0.6 * sentiment_component + 0.25 * shock_component + 0.15 * headline_density

    intercept = {
        "QQQ": 0.01,
        "NVDA": 0.014,
        "AAOI": 0.012,
    }.get(ticker, 0.01)
    regime_bias = (regime_score - 0.5) * 0.03
    synergy = 0.01 if breakout and sentiment_component > 0 else 0.0
    expected_return_5d = intercept + 0.04 * market_component + 0.025 * news_component + regime_bias + synergy
    expected_return_5d = float(np.clip(expected_return_5d, -0.12, 0.15))

    if market_component >= news_component:
        driver = "量价因子主导"
    else:
        driver = "消息情绪主导"
    inference_log = f"{driver}，5日EV {expected_return_5d:.2%}"

    return {
        "expected_return_5d": expected_return_5d,
        "expected_return_daily": expected_return_5d / 5.0,
        "market_component": market_component,
        "news_component": news_component,
        "volatility_penalty": volatility_penalty_label(hist_vol_20d),
        "inference_log": inference_log,
    }


def classify_signal_mode(market_features: dict[str, Any]) -> dict[str, str]:
    rsi_14 = float(market_features.get("rsi_14") or 50.0)
    log_return_5d = float(market_features.get("log_return_5d") or 0.0)
    volume_ratio = float(market_features.get("volume_ratio_20d") or 1.0)
    relative_strength = float(market_features.get("relative_strength_20d_vs_benchmark") or 0.0)
    breakout = bool(market_features.get("breakout_20d"))
    daily_return_pct = float(market_features.get("daily_return_pct") or 0.0)
    close = float(market_features.get("close") or 0.0)
    sma_20 = float(market_features.get("sma_20") or close or 0.0)

    if breakout and log_return_5d >= 0.015 and volume_ratio >= 1.2 and relative_strength >= 0.0 and 52.0 <= rsi_14 <= 78.0:
        return {
            "signal_mode": "momentum",
            "signal_label": "动能延续",
            "signal_reason": "突破、放量、相对强度同步确认",
        }

    if (
        not breakout
        and rsi_14 <= 35.0
        and log_return_5d <= -0.015
        and daily_return_pct <= -1.5
        and close <= sma_20
        and volume_ratio <= 1.8
    ):
        return {
            "signal_mode": "mean_reversion",
            "signal_label": "均值回归",
            "signal_reason": "短线超跌，等待反抽确认",
        }

    return {
        "signal_mode": "neutral",
        "signal_label": "观望",
        "signal_reason": "未触发动量或均值回归高胜率模板",
    }


def evaluate_trade_gate(
    signal_mode: str,
    expected_return_5d: float,
    trade_stats: dict[str, float],
    regime_score: float,
) -> dict[str, Any]:
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
    if expected_return_5d < thresholds["min_expected_return_5d"]:
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
) -> dict[str, Any]:
    expected = estimate_expected_return(ticker, market_features, news_features, regime_score)
    signal_mode = classify_signal_mode(market_features)

    volume_ratio = float(market_features.get("volume_ratio_20d") or 1.0)
    relative_strength = float(market_features.get("relative_strength_20d_vs_benchmark") or 0.0)
    breakout = bool(market_features.get("breakout_20d"))
    breakout_valid = bool(signal_mode["signal_mode"] == "momentum" and breakout and volume_ratio >= 1.2 and relative_strength >= 0.0)
    fake_breakout = bool(breakout and not breakout_valid)
    trade_stats = estimate_trade_stats(
        expected["expected_return_5d"],
        volume_ratio,
        breakout_valid,
        fake_breakout,
    )
    gate = evaluate_trade_gate(
        signal_mode["signal_mode"],
        expected["expected_return_5d"],
        trade_stats,
        regime_score,
    )

    inference_log = (
        f"{signal_mode['signal_label']} / {expected['inference_log']} / "
        f"{gate['reason']}"
    )
    return {
        **expected,
        **signal_mode,
        **trade_stats,
        "breakout_valid": breakout_valid,
        "fake_breakout": fake_breakout,
        "eligible_for_risk": gate["eligible"],
        "gate_reason": gate["reason"],
        "inference_log": inference_log,
    }


def build_candidate_trade_profile(candidate: dict[str, Any], regime_score: float) -> dict[str, Any]:
    signal_mode = {
        "signal_mode": "momentum" if candidate.get("breakout_valid") else "neutral",
        "signal_label": "动能延续" if candidate.get("breakout_valid") else "观望",
        "signal_reason": "放量突破模板" if candidate.get("breakout_valid") else "未触发突破阈值",
    }
    trade_stats = estimate_trade_stats(
        float(candidate.get("expected_return_5d") or 0.0),
        float(candidate.get("volume_ratio_20d") or 0.0),
        bool(candidate.get("breakout_valid")),
        bool(candidate.get("fake_breakout")),
    )
    gate = evaluate_trade_gate(
        signal_mode["signal_mode"],
        float(candidate.get("expected_return_5d") or 0.0),
        trade_stats,
        regime_score,
    )
    return {
        **candidate,
        **signal_mode,
        **trade_stats,
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
) -> dict[str, float]:
    win_rate = 0.42
    if breakout_valid:
        win_rate += 0.12
    if fake_breakout:
        win_rate -= 0.15
    win_rate += 0.08 * float(np.clip((volume_ratio - 1.0) / 2.0, 0.0, 1.0))
    win_rate += 0.12 * float(np.clip(expected_return_5d / 0.08, -1.0, 1.0))
    win_rate = float(np.clip(win_rate, 0.2, 0.8))

    stop_loss_pct = float(np.clip(0.03 + abs(expected_return_5d) * 0.6, 0.03, 0.08))
    reward_pct = float(max(expected_return_5d, 0.01))
    payoff_ratio = reward_pct / stop_loss_pct if stop_loss_pct else 0.0
    ev = win_rate * reward_pct - (1.0 - win_rate) * stop_loss_pct
    return {
        "win_rate": win_rate,
        "stop_loss_pct": stop_loss_pct,
        "payoff_ratio": payoff_ratio,
        "target_ev": ev,
    }
