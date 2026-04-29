from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR: Final[int] = 252
REGIME_LABEL_SCORES: Final[dict[str, float]] = {
    "Risk-On": 0.82,
    "Range": 0.5,
    "Risk-Off": 0.18,
}


def classify_factor_allocation_regime(regime_score: float, vix_value: float | None) -> dict[str, Any]:
    resolved_score = float(np.clip(regime_score, 0.0, 1.0))
    resolved_vix = None
    if vix_value is not None and np.isfinite(vix_value):
        resolved_vix = float(vix_value)

    if (resolved_vix is not None and resolved_vix >= 26.0) or resolved_score <= 0.38:
        return {
            "label": "Risk-Off",
            "reason": "VIX高位或风险偏好显著走弱",
            "score": resolved_score,
            "vix_value": resolved_vix,
        }
    if (resolved_vix is None or resolved_vix <= 18.0) and resolved_score >= 0.62:
        return {
            "label": "Risk-On",
            "reason": "风险偏好抬升且VIX处于低位/常态",
            "score": resolved_score,
            "vix_value": resolved_vix,
        }
    return {
        "label": "Choppy",
        "reason": "风险偏好与波动率信号分化，进入震荡配置模式",
        "score": resolved_score,
        "vix_value": resolved_vix,
    }


@dataclass(frozen=True)
class RegimeStateStats:
    state_id: int
    mean_return: float
    return_variance: float
    avg_volatility: float
    avg_volume_change: float
    sample_size: int


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


def prepare_regime_features(
    history: pd.DataFrame,
    volatility_window: int = 20,
) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["log_return", "historical_volatility", "volume_change_rate"])

    frame = history.copy()
    frame = frame.sort_index()
    frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")
    frame["close"] = frame["close"].replace([np.inf, -np.inf], np.nan)
    frame["volume"] = frame["volume"].replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    frame = frame.dropna(subset=["close", "volume"])
    if frame.empty:
        return pd.DataFrame(columns=["log_return", "historical_volatility", "volume_change_rate"])

    min_periods = max(5, volatility_window // 3)
    frame["log_return"] = np.log(frame["close"]).diff()
    frame["historical_volatility"] = (
        frame["log_return"].rolling(volatility_window, min_periods=min_periods).std(ddof=0)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    frame["volume_change_rate"] = frame["volume"].pct_change()

    features = frame[["log_return", "historical_volatility", "volume_change_rate"]].replace([np.inf, -np.inf], np.nan)
    for column in features.columns:
        series = features[column]
        if series.dropna().empty:
            continue
        median = float(series.median())
        mad = float((series - median).abs().median())
        if np.isfinite(mad) and mad > 0.0:
            robust_sigma = 1.4826 * mad
            series = series.clip(median - 6.0 * robust_sigma, median + 6.0 * robust_sigma)
        lower = float(series.quantile(0.01))
        upper = float(series.quantile(0.99))
        features[column] = series.clip(lower, upper)

    return features.dropna().astype(float)


class GaussianHMMRegimeModel:
    def __init__(
        self,
        n_states: int = 3,
        volatility_window: int = 20,
        min_history: int = 80,
        n_iter: int = 300,
        restarts: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.volatility_window = volatility_window
        self.min_history = max(min_history, volatility_window * 2)
        self.n_iter = n_iter
        self.restarts = max(restarts, 1)
        self.random_state = random_state
        self.feature_columns = ["log_return", "historical_volatility", "volume_change_rate"]
        self.model_: GaussianHMM | None = None
        self.feature_means_: pd.Series | None = None
        self.feature_stds_: pd.Series | None = None
        self.state_label_map_: dict[int, str] = {}
        self.state_stats_: dict[int, RegimeStateStats] = {}

    def fit(self, history: pd.DataFrame) -> GaussianHMMRegimeModel:
        features = prepare_regime_features(history, volatility_window=self.volatility_window)
        if len(features) < self.min_history:
            raise ValueError("insufficient_history_for_hmm")

        transformed = self._fit_transform(features)
        self.model_ = self._fit_best_model(transformed)
        hidden_states = self.model_.predict(transformed)
        self.state_stats_ = self._summarize_states(features, hidden_states)
        self.state_label_map_ = self._align_state_labels(self.state_stats_)
        return self

    def predict_current_regime(self, latest_data: pd.DataFrame) -> dict[str, Any]:
        if self.model_ is None or self.feature_means_ is None or self.feature_stds_ is None:
            raise RuntimeError("hmm_model_not_fitted")

        features = prepare_regime_features(latest_data, volatility_window=self.volatility_window)
        if features.empty:
            raise ValueError("empty_feature_frame")

        transformed = self._transform(features)
        posterior = self.model_.predict_proba(transformed)
        latest_posterior = posterior[-1]
        current_state = int(np.argmax(latest_posterior))
        label_probabilities = {
            self.state_label_map_.get(state_id, f"State-{state_id}"): float(latest_posterior[state_id])
            for state_id in range(self.n_states)
        }
        current_label = self.state_label_map_.get(current_state, f"State-{current_state}")
        score = float(
            sum(
                latest_posterior[state_id] * REGIME_LABEL_SCORES.get(self.state_label_map_.get(state_id, "Range"), 0.5)
                for state_id in range(self.n_states)
            )
        )
        state_stats = self.state_stats_.get(current_state)
        reasons = [
            f"HMM主状态 {current_label}",
            f"后验概率 {label_probabilities.get(current_label, 0.0):.0%}",
        ]
        if state_stats is not None:
            reasons.append(f"状态均值收益 {state_stats.mean_return:.2%}")
            reasons.append(f"状态波动 {state_stats.avg_volatility:.2%}")

        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "label": current_label,
            "reasons": reasons,
            "posterior_probabilities": label_probabilities,
            "state_mapping": {state_id: label for state_id, label in self.state_label_map_.items()},
            "state_statistics": {state_id: asdict(stats) for state_id, stats in self.state_stats_.items()},
        }

    def _fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        self.feature_means_ = features.mean()
        self.feature_stds_ = features.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        return self._transform(features)

    def _transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_means_ is None or self.feature_stds_ is None:
            raise RuntimeError("scaler_not_fitted")
        aligned = features.reindex(columns=self.feature_columns)
        standardized = (aligned - self.feature_means_) / self.feature_stds_
        return standardized.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float)

    def _fit_best_model(self, transformed: np.ndarray) -> GaussianHMM:
        best_model: GaussianHMM | None = None
        best_score = -np.inf
        for trial in range(self.restarts):
            seed = self.random_state + trial
            candidate = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.n_iter,
                min_covar=1e-4,
                random_state=seed,
            )
            try:
                candidate.fit(transformed)
                candidate_score = float(candidate.score(transformed))
            except ValueError:
                continue
            if np.isfinite(candidate_score) and candidate_score > best_score:
                best_score = candidate_score
                best_model = candidate
        if best_model is None:
            raise RuntimeError("hmm_fit_failed")
        return best_model

    def _summarize_states(
        self,
        features: pd.DataFrame,
        hidden_states: np.ndarray,
    ) -> dict[int, RegimeStateStats]:
        summaries: dict[int, RegimeStateStats] = {}
        default_mean_return = float(features["log_return"].mean())
        default_variance = float(features["log_return"].var(ddof=0))
        default_volatility = float(features["historical_volatility"].mean())
        default_volume_change = float(features["volume_change_rate"].mean())

        for state_id in range(self.n_states):
            mask = hidden_states == state_id
            state_frame = features.loc[mask]
            if state_frame.empty:
                summaries[state_id] = RegimeStateStats(
                    state_id=state_id,
                    mean_return=default_mean_return,
                    return_variance=default_variance,
                    avg_volatility=default_volatility,
                    avg_volume_change=default_volume_change,
                    sample_size=0,
                )
                continue
            summaries[state_id] = RegimeStateStats(
                state_id=state_id,
                mean_return=float(state_frame["log_return"].mean()),
                return_variance=float(state_frame["log_return"].var(ddof=0)),
                avg_volatility=float(state_frame["historical_volatility"].mean()),
                avg_volume_change=float(state_frame["volume_change_rate"].mean()),
                sample_size=int(len(state_frame)),
            )
        return summaries

    def _align_state_labels(
        self,
        state_stats: dict[int, RegimeStateStats],
    ) -> dict[int, str]:
        states = list(state_stats)
        risk_off_state = min(
            states,
            key=lambda state_id: (
                state_stats[state_id].mean_return,
                -state_stats[state_id].return_variance,
            ),
        )
        remaining = [state_id for state_id in states if state_id != risk_off_state]
        risk_on_state = max(
            remaining,
            key=lambda state_id: (
                state_stats[state_id].mean_return - 0.35 * state_stats[state_id].return_variance,
                -state_stats[state_id].avg_volatility,
            ),
        )
        range_state = next(state_id for state_id in states if state_id not in {risk_on_state, risk_off_state})
        return {
            risk_on_state: "Risk-On",
            range_state: "Range",
            risk_off_state: "Risk-Off",
        }


def predict_current_regime(
    latest_data: pd.DataFrame,
    model: GaussianHMMRegimeModel,
) -> dict[str, Any]:
    return model.predict_current_regime(latest_data)


def classify_market_regime(
    macro_snapshot: dict[str, Any],
    qqq_factors: dict[str, Any],
    nvda_factors: dict[str, Any],
    aaoi_breakdown: dict[str, Any],
    nvda_news: dict[str, Any],
    market_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    detector = GaussianHMMRegimeModel()
    if market_history is None:
        allocation_regime = classify_factor_allocation_regime(0.5, float((macro_snapshot.get("vix") or {}).get("value") or 0.0) or None)
        return {
            "score": 0.5,
            "label": "Range",
            "reasons": ["缺少市场历史数据，HMM未启用"],
            "posterior_probabilities": {"Risk-On": 0.0, "Range": 1.0, "Risk-Off": 0.0},
            "state_mapping": {},
            "state_statistics": {},
            "allocation_regime": allocation_regime["label"],
            "allocation_reason": allocation_regime["reason"],
            "vix_value": allocation_regime["vix_value"],
        }

    try:
        detector.fit(market_history)
        regime = detector.predict_current_regime(market_history)
    except (RuntimeError, ValueError):
        allocation_regime = classify_factor_allocation_regime(0.5, float((macro_snapshot.get("vix") or {}).get("value") or 0.0) or None)
        return {
            "score": 0.5,
            "label": "Range",
            "reasons": ["HMM训练失败，回落到中性状态"],
            "posterior_probabilities": {"Risk-On": 0.0, "Range": 1.0, "Risk-Off": 0.0},
            "state_mapping": {},
            "state_statistics": {},
            "allocation_regime": allocation_regime["label"],
            "allocation_reason": allocation_regime["reason"],
            "vix_value": allocation_regime["vix_value"],
        }

    if qqq_factors.get("breakout_20d"):
        regime["reasons"].append("QQQ突破")
    if nvda_factors.get("breakout_20d"):
        regime["reasons"].append("NVDA突破")
    if aaoi_breakdown.get("breakdown"):
        regime["reasons"].append("AAOI破位")

    vix_value = float((macro_snapshot.get("vix") or {}).get("value") or 0.0)
    if vix_value >= 28.0:
        regime["reasons"].append("VIX高位")
    elif 0.0 < vix_value <= 16.0:
        regime["reasons"].append("VIX低位")

    activated_sentiment = float(nvda_news.get("activated_sentiment") or 0.0)
    if activated_sentiment >= 0.3:
        regime["reasons"].append("NVDA情绪偏正")
    elif activated_sentiment <= -0.3:
        regime["reasons"].append("NVDA情绪偏负")

    allocation_regime = classify_factor_allocation_regime(float(regime.get("score") or 0.5), vix_value if vix_value > 0.0 else None)
    regime["allocation_regime"] = allocation_regime["label"]
    regime["allocation_reason"] = allocation_regime["reason"]
    regime["vix_value"] = allocation_regime["vix_value"]

    return regime
