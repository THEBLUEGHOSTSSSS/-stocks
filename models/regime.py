from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_sigma_event(history: pd.DataFrame, window: int = 60) -> dict[str, Any]:
    if history is None or history.empty or len(history) < 6:
        return {
            "return_pct": None,
            "z_score": None,
            "is_2sigma": False,
        }

    returns = history["close"].pct_change().dropna()
    if len(returns) < 2:
        return {
            "return_pct": None,
            "z_score": None,
            "is_2sigma": False,
        }

    sample = returns.iloc[-(window + 1):-1] if len(returns) > window else returns.iloc[:-1]
    if sample.empty:
        sample = returns.iloc[:-1]
    latest_return = float(returns.iloc[-1])
    sigma = float(sample.std(ddof=0)) if not sample.empty else 0.0
    mean = float(sample.mean()) if not sample.empty else 0.0
    z_score = 0.0 if sigma == 0.0 else (latest_return - mean) / sigma
    return {
        "return_pct": latest_return * 100.0,
        "z_score": z_score,
        "is_2sigma": abs(z_score) >= 2.0,
    }


def detect_breakdown_signal(
    history: pd.DataFrame,
    lookback: int = 20,
    volume_ratio_threshold: float = 1.3,
) -> dict[str, Any]:
    if history is None or history.empty or len(history) < lookback + 1:
        return {
            "breakdown": False,
            "close": None,
            "support": None,
            "volume_ratio": None,
        }

    latest = history.iloc[-1]
    rolling_support = float(history["low"].iloc[-(lookback + 1):-1].min())
    average_volume = float(history["volume"].iloc[-(lookback + 1):-1].mean())
    volume_ratio = float(latest["volume"] / average_volume) if average_volume else None
    close = float(latest["close"])
    breakdown = bool(close < rolling_support and (volume_ratio or 0.0) >= volume_ratio_threshold)
    return {
        "breakdown": breakdown,
        "close": close,
        "support": rolling_support,
        "volume_ratio": volume_ratio,
    }


def classify_market_regime(
    macro_snapshot: dict[str, Any],
    qqq_factors: dict[str, Any],
    nvda_factors: dict[str, Any],
    aaoi_breakdown: dict[str, Any],
    nvda_news: dict[str, Any],
) -> dict[str, Any]:
    score = 0.5
    reasons: list[str] = []

    vix_value = float((macro_snapshot.get("vix") or {}).get("value") or 0.0)
    if vix_value:
        if vix_value < 16.0:
            score += 0.18
            reasons.append("低VIX")
        elif vix_value < 22.0:
            score += 0.06
            reasons.append("VIX温和")
        elif vix_value > 28.0:
            score -= 0.22
            reasons.append("高VIX")
        else:
            score -= 0.08
            reasons.append("VIX偏高")

    us10y_change = float((macro_snapshot.get("us10y") or {}).get("change_pct") or 0.0)
    if us10y_change > 1.2:
        score -= 0.15
        reasons.append("利率冲击")
    elif us10y_change < -1.0:
        score += 0.08
        reasons.append("利率回落")

    flow_proxy = (macro_snapshot.get("core_etf_flow_proxy") or {}).get("value")
    if flow_proxy is not None:
        flow_proxy_value = float(flow_proxy)
        if flow_proxy_value > 1.12:
            score += 0.08
            reasons.append("ETF活跃")
        elif flow_proxy_value < 0.9:
            score -= 0.08
            reasons.append("ETF转弱")

    if qqq_factors.get("breakout_20d"):
        score += 0.1
        reasons.append("QQQ突破")
    if qqq_factors.get("close") and qqq_factors.get("sma_20"):
        if float(qqq_factors["close"]) > float(qqq_factors["sma_20"]):
            score += 0.06
            reasons.append("QQQ站上20日线")

    qqq_rsi = float(qqq_factors.get("rsi_14") or 50.0)
    if qqq_rsi > 75.0:
        score -= 0.05
        reasons.append("QQQ超买")
    elif 55.0 <= qqq_rsi <= 68.0:
        score += 0.05
        reasons.append("QQQ趋势稳")
    elif qqq_rsi < 40.0:
        score -= 0.05
        reasons.append("QQQ偏弱")

    if nvda_factors.get("breakout_20d"):
        score += 0.06
        reasons.append("NVDA突破")

    if aaoi_breakdown.get("breakdown"):
        score -= 0.12
        reasons.append("光模块破位")

    activated_sentiment = float(nvda_news.get("activated_sentiment") or 0.0)
    shock_score = float(nvda_news.get("shock_score") or 0.0)
    if activated_sentiment > 0.3:
        score += 0.04
        reasons.append("NVDA情绪正")
    elif activated_sentiment < -0.3:
        score -= 0.06
        reasons.append("NVDA情绪负")
    if shock_score > 2.5 and activated_sentiment < 0.0:
        score -= 0.05
        reasons.append("负面事件放大")

    clipped_score = float(np.clip(score, 0.0, 1.0))
    if clipped_score >= 0.67:
        label = "单边上涨"
    elif clipped_score <= 0.33:
        label = "系统性风险"
    else:
        label = "宽幅震荡"

    return {
        "score": clipped_score,
        "label": label,
        "reasons": reasons,
    }
