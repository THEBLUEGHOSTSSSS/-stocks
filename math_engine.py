from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA


FloatArray = NDArray[np.float64]

DEFAULT_FACTOR_COLUMNS: tuple[str, ...] = (
    "daily_return_pct",
    # V2.1: 横截面正交化不再喂入 10 日脉冲项，改为 60 日慢趋势因子，
    # 让 PCA/SVD 更聚焦结构性漂移而不是短线噪声。
    "log_return_60d",
    "hist_vol_20d",
    "rsi_14",
    "atr_14",
    "volume_ratio_20d",
    "sma_20",
    "upper_shadow_pct",
    "intraday_drawdown_pct",
    "relative_strength_20d_vs_benchmark",
)


@dataclass(frozen=True, slots=True)
class SVDPCAConfig:
    orth_components: int = 6
    pca_components: int = 5
    ridge_lambda: float = 1e-4
    residual_z_window: int = 60
    residual_z_threshold: float = 2.0
    eps: float = 1e-12


@dataclass(slots=True)
class OrthogonalAlphaResult:
    orthogonal_factor_frame: pd.DataFrame
    original_feature_weights: pd.Series
    singular_values: pd.Series
    explained_variance_ratio: pd.Series
    factor_rotation: pd.DataFrame
    orthogonal_alpha_scores: pd.Series


@dataclass(slots=True)
class ResidualRiskResult:
    residual_returns: pd.DataFrame
    residual_zscores: pd.DataFrame
    systematic_returns: pd.DataFrame
    component_loadings: pd.DataFrame
    explained_variance_ratio: pd.Series
    latest_signals: pd.DataFrame


@dataclass(slots=True)
class MarketNeutralMathSnapshot:
    signal_map: dict[str, dict[str, Any]]
    orthogonal_weights: pd.Series
    orthogonal_scores: pd.Series
    residual_signals: pd.DataFrame
    residual_explained_variance_ratio: pd.Series


class SVDPCAAlphaEngine:
    """SVD 正交化 + PCA 系统性风险剥离引擎。"""

    def __init__(self, config: SVDPCAConfig | None = None) -> None:
        self.config = config or SVDPCAConfig()

    def _standardize_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        clean = frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = clean.mean(axis=0)
        std = clean.std(axis=0, ddof=0).replace(0.0, 1.0)
        standardized = (clean - mean) / std
        return standardized, mean, std

    def orthogonalize_factors(
        self,
        factor_frame: pd.DataFrame,
        target_returns: pd.Series,
        n_components: int | None = None,
    ) -> OrthogonalAlphaResult:
        aligned_factors, aligned_target = factor_frame.align(target_returns.rename("target"), join="inner", axis=0)
        if aligned_factors.empty or len(aligned_factors.columns) == 0:
            raise ValueError("因子矩阵为空，无法做 SVD 正交化")

        x_std, _, _ = self._standardize_frame(aligned_factors)
        x: FloatArray = np.ascontiguousarray(x_std.to_numpy(dtype=np.float64))
        y: FloatArray = np.ascontiguousarray(aligned_target.to_numpy(dtype=np.float64).reshape(-1))

        u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
        max_rank = min(x.shape[0], x.shape[1])
        rank = min(n_components or self.config.orth_components, max_rank)
        if rank <= 0:
            raise ValueError("有效秩不足，无法做 SVD 正交化")

        orthogonal_factors: FloatArray = x @ vt[:rank].T
        gram: FloatArray = (orthogonal_factors.T @ orthogonal_factors) / max(len(orthogonal_factors), 1)
        rhs: FloatArray = (orthogonal_factors.T @ y) / max(len(orthogonal_factors), 1)
        beta_orth = np.linalg.solve(gram + self.config.ridge_lambda * np.eye(rank, dtype=np.float64), rhs)

        raw_weights: FloatArray = vt[:rank].T @ beta_orth
        weight_norm = float(np.sum(np.abs(raw_weights)))
        if weight_norm > self.config.eps:
            raw_weights = raw_weights / weight_norm

        raw_scores: FloatArray = x @ raw_weights
        centered_scores = raw_scores - float(raw_scores.mean())
        score_std = float(centered_scores.std(ddof=0))
        z_scores = centered_scores / (score_std if score_std > self.config.eps else 1.0)

        explained = (singular_values[:rank] ** 2) / max(float(np.sum(singular_values**2)), self.config.eps)
        orth_columns = [f"orth_factor_{idx + 1}" for idx in range(rank)]

        return OrthogonalAlphaResult(
            orthogonal_factor_frame=pd.DataFrame(orthogonal_factors, index=x_std.index, columns=orth_columns),
            original_feature_weights=pd.Series(raw_weights, index=x_std.columns, name="alpha_weight"),
            singular_values=pd.Series(singular_values[:rank], index=orth_columns, name="singular_value"),
            explained_variance_ratio=pd.Series(explained, index=orth_columns, name="explained_variance_ratio"),
            factor_rotation=pd.DataFrame(vt[:rank].T, index=x_std.columns, columns=orth_columns),
            orthogonal_alpha_scores=pd.Series(z_scores, index=x_std.index, name="orthogonal_alpha_score"),
        )

    def decompose_systematic_risk(
        self,
        returns_frame: pd.DataFrame,
        n_components: int | None = None,
    ) -> ResidualRiskResult:
        clean = returns_frame.astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if clean.empty or clean.shape[1] < 2:
            raise ValueError("收益矩阵不足，无法做 PCA 残差分解")

        cross_section_mean = clean.mean(axis=0)
        x: FloatArray = np.ascontiguousarray((clean - cross_section_mean).to_numpy(dtype=np.float64))
        rank = min(n_components or self.config.pca_components, x.shape[0], x.shape[1])
        if rank <= 0:
            raise ValueError("有效主成分数量不足")

        pca = PCA(n_components=rank, svd_solver="randomized", whiten=False, random_state=42)
        scores: FloatArray = pca.fit_transform(x)
        common: FloatArray = pca.inverse_transform(scores)
        residual: FloatArray = x - common

        systematic_returns = pd.DataFrame(common + cross_section_mean.to_numpy(dtype=np.float64), index=clean.index, columns=clean.columns)
        residual_returns = pd.DataFrame(residual, index=clean.index, columns=clean.columns)

        rolling_mean = residual_returns.rolling(self.config.residual_z_window).mean()
        rolling_std = residual_returns.rolling(self.config.residual_z_window).std(ddof=0).replace(0.0, np.nan)
        residual_z = (residual_returns - rolling_mean) / rolling_std
        latest_z = residual_z.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        latest_residual = residual_returns.iloc[-1].fillna(0.0)

        signal_side = np.where(
            latest_z >= self.config.residual_z_threshold,
            "SHORT",
            np.where(latest_z <= -self.config.residual_z_threshold, "LONG", "FLAT"),
        )
        latest_signals = pd.DataFrame(
            {
                "ticker": latest_z.index,
                "idiosyncratic_return": latest_residual.values,
                "idio_zscore": latest_z.values,
                "stat_arb_signal": signal_side,
                "stat_arb_trigger": np.abs(latest_z.values) >= self.config.residual_z_threshold,
                "signal_strength": np.abs(latest_z.values),
            }
        ).sort_values(["stat_arb_trigger", "signal_strength"], ascending=[False, False])

        component_names = [f"pc_{idx + 1}" for idx in range(rank)]
        return ResidualRiskResult(
            residual_returns=residual_returns,
            residual_zscores=residual_z.replace([np.inf, -np.inf], np.nan),
            systematic_returns=systematic_returns,
            component_loadings=pd.DataFrame(pca.components_.T, index=clean.columns, columns=component_names),
            explained_variance_ratio=pd.Series(pca.explained_variance_ratio_, index=component_names, name="explained_variance_ratio"),
            latest_signals=latest_signals.reset_index(drop=True),
        )


def build_cross_sectional_factor_frame(
    feature_map: dict[str, dict[str, Any]],
    tickers: list[str],
    feature_columns: tuple[str, ...] = DEFAULT_FACTOR_COLUMNS,
) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        features = feature_map.get(ticker) or {}
        rows[ticker] = {
            column: float(features.get(column) or 0.0)
            for column in feature_columns
        }
    if not rows:
        return pd.DataFrame(columns=feature_columns, dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index").replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_realized_return_series(
    histories: dict[str, pd.DataFrame],
    tickers: list[str],
    lookback_days: int = 5,
) -> pd.Series:
    realized: dict[str, float] = {}
    for ticker in tickers:
        history = histories.get(ticker, pd.DataFrame())
        if history.empty or "close" not in history.columns or len(history) <= lookback_days:
            continue
        close = history["close"].astype(float).dropna()
        if len(close) <= lookback_days:
            continue
        start_price = float(close.iloc[-(lookback_days + 1)])
        end_price = float(close.iloc[-1])
        if start_price <= 0.0:
            continue
        realized[ticker] = end_price / start_price - 1.0
    return pd.Series(realized, dtype=float, name="realized_return")


def build_returns_frame_from_histories(
    histories: dict[str, pd.DataFrame],
    tickers: list[str],
    lookback_days: int = 180,
) -> pd.DataFrame:
    returns_map: dict[str, pd.Series] = {}
    min_observations = max(40, lookback_days // 2)
    for ticker in tickers:
        history = histories.get(ticker, pd.DataFrame())
        if history.empty or "close" not in history.columns:
            continue
        close = history["close"].astype(float).dropna()
        if len(close) < min_observations:
            continue
        returns_map[ticker] = close.pct_change().dropna().tail(lookback_days)

    if not returns_map:
        return pd.DataFrame()
    frame = pd.DataFrame(returns_map).dropna(axis=1, thresh=min_observations).dropna(axis=0, how="any")
    return frame


def build_market_neutral_math_snapshot(
    feature_map: dict[str, dict[str, Any]],
    histories: dict[str, pd.DataFrame],
    tickers: list[str],
    config: SVDPCAConfig | None = None,
) -> MarketNeutralMathSnapshot:
    engine = SVDPCAAlphaEngine(config)
    factor_frame = build_cross_sectional_factor_frame(feature_map, tickers)
    realized_returns = build_realized_return_series(histories, tickers)
    returns_frame = build_returns_frame_from_histories(histories, tickers)

    orth_weights = pd.Series(dtype=float, name="alpha_weight")
    orth_scores = pd.Series(dtype=float, name="orthogonal_alpha_score")
    residual_signals = pd.DataFrame(columns=["ticker", "idiosyncratic_return", "idio_zscore", "stat_arb_signal", "stat_arb_trigger", "signal_strength"])
    residual_explained = pd.Series(dtype=float, name="explained_variance_ratio")

    if len(factor_frame) >= 3 and len(realized_returns) >= 3:
        try:
            orth_result = engine.orthogonalize_factors(factor_frame, realized_returns)
            orth_weights = orth_result.original_feature_weights
            orth_scores = orth_result.orthogonal_alpha_scores
        except Exception:
            orth_weights = pd.Series(dtype=float, name="alpha_weight")
            orth_scores = pd.Series(dtype=float, name="orthogonal_alpha_score")

    if returns_frame.shape[1] >= 2 and len(returns_frame) >= max(40, engine.config.residual_z_window // 2):
        try:
            residual_result = engine.decompose_systematic_risk(returns_frame)
            residual_signals = residual_result.latest_signals
            residual_explained = residual_result.explained_variance_ratio
        except Exception:
            residual_signals = pd.DataFrame(columns=["ticker", "idiosyncratic_return", "idio_zscore", "stat_arb_signal", "stat_arb_trigger", "signal_strength"])
            residual_explained = pd.Series(dtype=float, name="explained_variance_ratio")

    orth_rank = {}
    if not orth_scores.empty:
        orth_rank = {
            str(ticker): int(rank)
            for rank, ticker in enumerate(orth_scores.abs().sort_values(ascending=False).index, start=1)
        }

    residual_lookup = residual_signals.set_index("ticker").to_dict(orient="index") if not residual_signals.empty else {}
    signal_map: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        residual_row = residual_lookup.get(ticker, {})
        idio_zscore = float(residual_row.get("idio_zscore") or 0.0)
        stat_arb_signal = str(residual_row.get("stat_arb_signal") or "FLAT").upper()
        stat_arb_trigger = bool(residual_row.get("stat_arb_trigger"))
        signal_map[ticker] = {
            "orthogonal_alpha_score": float(orth_scores.get(ticker, 0.0) or 0.0),
            "orthogonal_alpha_rank": orth_rank.get(ticker),
            "idiosyncratic_return": float(residual_row.get("idiosyncratic_return") or 0.0),
            "idio_zscore": idio_zscore,
            "stat_arb_signal": stat_arb_signal,
            "stat_arb_trigger": stat_arb_trigger,
            "stat_arb_reason": f"特质收益 Z 分数 {idio_zscore:.2f} 超过阈值，触发均值回归 {stat_arb_signal}" if stat_arb_trigger and stat_arb_signal in {"LONG", "SHORT"} else "",
        }

    return MarketNeutralMathSnapshot(
        signal_map=signal_map,
        orthogonal_weights=orth_weights,
        orthogonal_scores=orth_scores,
        residual_signals=residual_signals,
        residual_explained_variance_ratio=residual_explained,
    )