from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_KELLY_FRACTION, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def single_asset_kelly(probability: float, payoff_ratio: float) -> float:
    if payoff_ratio <= 0.0:
        return 0.0
    loss_probability = 1.0 - probability
    return max((probability * payoff_ratio - loss_probability) / payoff_ratio, 0.0)


def continuous_kelly(
    expected_returns: dict[str, float],
    returns_frame: pd.DataFrame,
    risk_free_rate: float | None = None,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    asset_volatility: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not expected_returns or returns_frame.empty:
        return {
            "weights": {},
            "total_exposure": 0.0,
            "covariance": pd.DataFrame(),
            "volatility_scalars": {},
            "edge_relief_factors": {},
            "asset_volatility": {},
        }

    columns = [column for column in returns_frame.columns if column in expected_returns]
    if not columns:
        return {
            "weights": {},
            "total_exposure": 0.0,
            "covariance": pd.DataFrame(),
            "volatility_scalars": {},
            "edge_relief_factors": {},
            "asset_volatility": {},
        }

    aligned_returns = returns_frame[columns].dropna(how="any")
    if aligned_returns.empty:
        return {
            "weights": {},
            "total_exposure": 0.0,
            "covariance": pd.DataFrame(),
            "volatility_scalars": {},
            "edge_relief_factors": {},
            "asset_volatility": {},
        }

    covariance = aligned_returns.cov()
    ridge = np.eye(len(columns)) * 1e-6
    inverse_covariance = np.linalg.pinv(covariance.to_numpy() + ridge)
    rf = (risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE) / TRADING_DAYS_PER_YEAR
    mu = np.array([expected_returns[column] for column in columns]) - rf
    raw_weights = inverse_covariance @ mu
    raw_weights = np.maximum(raw_weights, 0.0)
    scaled_weights = raw_weights * max(kelly_fraction, 0.0)

    fallback_volatility = aligned_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    resolved_asset_volatility: dict[str, float] = {}
    volatility_scalars: dict[str, float] = {}
    edge_relief_factors: dict[str, float] = {}
    for column in columns:
        raw_volatility = (asset_volatility or {}).get(column)
        try:
            resolved_volatility = float(raw_volatility)
        except (TypeError, ValueError):
            resolved_volatility = float(fallback_volatility.get(column, np.nan))
        if not np.isfinite(resolved_volatility) or resolved_volatility <= 0.0:
            resolved_volatility = float(fallback_volatility.get(column, np.nan))
        if not np.isfinite(resolved_volatility) or resolved_volatility <= 0.0:
            volatility_scalars[column] = 1.0
            edge_relief_factors[column] = 0.0
            continue
        resolved_asset_volatility[column] = resolved_volatility
        # V2.1: 风险收紧。旧公式是 clip(0.35 / sigma, 0.35, 1.0)，
        # 会给高波动资产保留过高仓位。新公式改成 clip(0.20 / sigma, 0.10, 0.80)，
        # 把 20 日波动率直接映射为更陡峭的惩罚曲线，让高 sigma 个股被机械压权。
        base_scalar = float(np.clip(0.20 / resolved_volatility, 0.10, 0.80))
        expected_edge = max(float(expected_returns.get(column) or 0.0), 0.0)
        edge_relief = float(np.clip((expected_edge - 0.003) / 0.009, 0.0, 1.0))
        high_volatility_intensity = float(np.clip((resolved_volatility - 0.20) / 0.45, 0.0, 1.0))
        relief_multiplier = 1.0 + 0.65 * edge_relief * high_volatility_intensity
        volatility_scalars[column] = float(np.clip(base_scalar * relief_multiplier, base_scalar, 0.80))
        edge_relief_factors[column] = edge_relief

    scaled_weights = scaled_weights * np.array([volatility_scalars.get(column, 1.0) for column in columns], dtype=float)

    total_exposure = float(scaled_weights.sum())
    if total_exposure > 1.0:
        scaled_weights = scaled_weights / total_exposure
        total_exposure = 1.0

    weights = {column: float(weight) for column, weight in zip(columns, scaled_weights, strict=True)}
    return {
        "weights": weights,
        "total_exposure": total_exposure,
        "covariance": covariance,
        "volatility_scalars": volatility_scalars,
        "edge_relief_factors": edge_relief_factors,
        "asset_volatility": resolved_asset_volatility,
    }
