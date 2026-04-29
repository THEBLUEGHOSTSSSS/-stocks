from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import pandas as pd

from data.fetcher import batch_history
from data.indicators import (
    compute_historical_volatility,
    compute_log_return,
    compute_relative_strength,
    compute_rsi,
    compute_volume_ratio,
)


SearchMethod = Literal["grid", "random"]


@dataclass(frozen=True)
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class PreprocessingStats:
    medians: pd.Series
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    means: pd.Series
    stds: pd.Series


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_weights: dict[str, float]
    train_sharpe: float
    train_max_drawdown: float
    test_sharpe: float
    test_max_drawdown: float
    test_calmar: float
    train_observations: int
    test_observations: int


@dataclass(frozen=True)
class WalkForwardResult:
    equity_curve: pd.DataFrame
    fold_summary: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, float]


@dataclass(frozen=True)
class RollForwardPathSpec:
    path_id: str
    random_state: int
    fold_offset: int
    fold_step: int | None = None


class WalkForwardValidator:
    def __init__(
        self,
        panel_data: pd.DataFrame,
        factor_cols: Sequence[str],
        target_col: str,
        train_window: int,
        test_window: int,
        date_col: str = "date",
        asset_col: str = "asset",
        search_method: SearchMethod = "grid",
        param_grid: dict[str, Sequence[float]] | None = None,
        n_iter: int = 100,
        annualization: int = 252,
        transaction_cost_bps: float = 0.0,
        winsor_quantiles: tuple[float, float] = (0.01, 0.99),
        position_cap: float = 1.0,
        random_state: int = 42,
        fold_offset: int = 0,
        fold_step: int | None = None,
    ) -> None:
        if train_window <= 1:
            raise ValueError("train_window_must_be_greater_than_one")
        if test_window <= 0:
            raise ValueError("test_window_must_be_positive")
        if not factor_cols:
            raise ValueError("factor_cols_must_not_be_empty")
        if search_method not in {"grid", "random"}:
            raise ValueError("unsupported_search_method")

        lower_q, upper_q = winsor_quantiles
        if not 0.0 <= lower_q < upper_q <= 1.0:
            raise ValueError("winsor_quantiles_must_be_between_zero_and_one")

        self.date_col = date_col
        self.asset_col = asset_col
        self.factor_cols = tuple(factor_cols)
        self.target_col = target_col
        self.train_window = int(train_window)
        self.test_window = int(test_window)
        self.search_method = search_method
        self.param_grid = self._resolve_param_grid(param_grid)
        self.n_iter = int(max(n_iter, 1))
        self.annualization = int(max(annualization, 1))
        self.transaction_cost_bps = float(max(transaction_cost_bps, 0.0))
        self.winsor_quantiles = winsor_quantiles
        self.position_cap = float(max(position_cap, 1e-6))
        self.random_state = int(random_state)
        self.fold_offset = int(max(fold_offset, 0))
        self.fold_step = int(max(fold_step if fold_step is not None else self.test_window, 1))
        self.panel_data = self._prepare_panel(panel_data)

    def run(self) -> WalkForwardResult:
        folds = list(self.iter_folds())
        if not folds:
            raise ValueError("insufficient_dates_for_walk_forward")

        prediction_frames: list[pd.DataFrame] = []
        fold_results: list[FoldResult] = []
        oos_returns: list[pd.Series] = []

        for fold in folds:
            train_frame = self.panel_data[self.panel_data[self.date_col].isin(fold.train_dates)].copy()
            test_frame = self.panel_data[self.panel_data[self.date_col].isin(fold.test_dates)].copy()
            if train_frame.empty or test_frame.empty:
                continue

            stats = self._fit_preprocessor(train_frame)
            transformed_train = self._transform(train_frame, stats)
            transformed_test = self._transform(test_frame, stats)

            best_weights, train_metrics = self._optimize_weights(transformed_train, fold.fold_id)
            train_output = self._simulate_portfolio(transformed_train, best_weights, fold.fold_id, "train")
            test_output = self._simulate_portfolio(transformed_test, best_weights, fold.fold_id, "test")

            prediction_frames.append(test_output["predictions"])
            oos_returns.append(test_output["daily_returns"])
            fold_results.append(
                FoldResult(
                    fold_id=fold.fold_id,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    best_weights=best_weights,
                    train_sharpe=train_metrics["sharpe"],
                    train_max_drawdown=train_metrics["max_drawdown"],
                    test_sharpe=test_output["metrics"]["sharpe"],
                    test_max_drawdown=test_output["metrics"]["max_drawdown"],
                    test_calmar=test_output["metrics"]["calmar"],
                    train_observations=int(len(transformed_train)),
                    test_observations=int(len(transformed_test)),
                )
            )

        if not oos_returns:
            raise RuntimeError("walk_forward_produced_no_test_results")

        oos_return_series = pd.concat(oos_returns).sort_index()
        oos_return_series = oos_return_series[~oos_return_series.index.duplicated(keep="first")]
        equity_curve = self._build_equity_curve(oos_return_series)
        metrics = self._compute_metrics(oos_return_series)
        prediction_frame = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        fold_summary = pd.DataFrame([self._fold_result_to_row(result) for result in fold_results])

        return WalkForwardResult(
            equity_curve=equity_curve,
            fold_summary=fold_summary,
            predictions=prediction_frame,
            metrics=metrics,
        )

    def iter_folds(self) -> Iterator[FoldWindow]:
        unique_dates = tuple(pd.Index(self.panel_data[self.date_col].drop_duplicates()).sort_values())
        if len(unique_dates) < self.train_window + self.test_window:
            return

        fold_id = 0
        start_idx = self.train_window + self.fold_offset
        if start_idx > len(unique_dates) - self.test_window:
            return

        for train_end_idx in range(start_idx, len(unique_dates) - self.test_window + 1, self.fold_step):
            train_dates = unique_dates[train_end_idx - self.train_window:train_end_idx]
            test_dates = unique_dates[train_end_idx:train_end_idx + self.test_window]
            yield FoldWindow(
                fold_id=fold_id,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_dates=tuple(train_dates),
                test_dates=tuple(test_dates),
            )
            fold_id += 1

    def _prepare_panel(self, panel_data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {self.date_col, self.asset_col, self.target_col, *self.factor_cols}
        missing_cols = sorted(required_cols.difference(panel_data.columns))
        if missing_cols:
            raise KeyError(f"missing_required_columns: {missing_cols}")

        frame = panel_data.loc[:, list(required_cols)].copy()
        frame[self.date_col] = pd.to_datetime(frame[self.date_col], errors="coerce")
        frame = frame.dropna(subset=[self.date_col, self.asset_col, self.target_col])
        if frame.empty:
            raise ValueError("panel_data_is_empty_after_basic_cleaning")

        for column in (*self.factor_cols, self.target_col):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=[self.target_col])
        frame[self.asset_col] = frame[self.asset_col].astype(str)
        frame = frame.sort_values([self.date_col, self.asset_col]).reset_index(drop=True)

        duplicated = frame.duplicated(subset=[self.date_col, self.asset_col], keep=False)
        if bool(duplicated.any()):
            raise ValueError("duplicate_date_asset_rows_detected")

        return frame

    def _resolve_param_grid(
        self,
        param_grid: dict[str, Sequence[float]] | None,
    ) -> dict[str, tuple[float, ...]]:
        default_grid = (-1.0, -0.5, 0.0, 0.5, 1.0)
        resolved: dict[str, tuple[float, ...]] = {}
        for factor in self.factor_cols:
            values = param_grid.get(factor) if param_grid is not None else None
            if not values:
                values = default_grid
            resolved[factor] = tuple(float(value) for value in values)
        return resolved

    def _fit_preprocessor(self, train_frame: pd.DataFrame) -> PreprocessingStats:
        features = train_frame.loc[:, self.factor_cols].copy()
        medians = features.median(numeric_only=True)
        filled = features.fillna(medians)
        lower_q, upper_q = self.winsor_quantiles
        lower_bounds = filled.quantile(lower_q)
        upper_bounds = filled.quantile(upper_q)
        clipped = filled.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
        means = clipped.mean()
        stds = clipped.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        return PreprocessingStats(
            medians=medians,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            means=means,
            stds=stds,
        )

    def _transform(self, frame: pd.DataFrame, stats: PreprocessingStats) -> pd.DataFrame:
        transformed = frame.copy()
        features = transformed.loc[:, self.factor_cols].copy()
        features = features.fillna(stats.medians)
        features = features.clip(lower=stats.lower_bounds, upper=stats.upper_bounds, axis=1)
        standardized = (features - stats.means) / stats.stds
        standardized = standardized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        transformed.loc[:, self.factor_cols] = standardized
        return transformed

    def _optimize_weights(
        self,
        train_frame: pd.DataFrame,
        fold_id: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        best_weights: dict[str, float] | None = None
        best_metrics: dict[str, float] | None = None
        best_objective = -np.inf

        for weights in self._iter_weight_candidates(fold_id):
            simulation = self._simulate_portfolio(train_frame, weights, fold_id, "train")
            metrics = simulation["metrics"]
            objective = float(metrics["sharpe"] - 0.25 * abs(metrics["max_drawdown"]))
            if np.isfinite(objective) and objective > best_objective:
                best_objective = objective
                best_weights = weights
                best_metrics = metrics

        if best_weights is None or best_metrics is None:
            raise RuntimeError("parameter_search_failed")

        return best_weights, best_metrics

    def _iter_weight_candidates(self, fold_id: int) -> Iterator[dict[str, float]]:
        if self.search_method == "grid":
            seen: set[tuple[float, ...]] = set()
            values = [self.param_grid[factor] for factor in self.factor_cols]
            for combo in product(*values):
                normalized = self._normalize_weight_tuple(combo)
                if normalized is None or normalized in seen:
                    continue
                seen.add(normalized)
                yield dict(zip(self.factor_cols, normalized, strict=True))
            return

        rng = np.random.default_rng(self.random_state + fold_id)
        seen_random: set[tuple[float, ...]] = set()
        max_unique = int(np.prod([len(self.param_grid[factor]) for factor in self.factor_cols], dtype=np.int64))
        target_iterations = min(self.n_iter, max_unique)
        attempts = 0
        max_attempts = max(target_iterations * 20, 100)

        while len(seen_random) < target_iterations and attempts < max_attempts:
            attempts += 1
            combo = tuple(
                self.param_grid[factor][int(rng.integers(0, len(self.param_grid[factor])))]
                for factor in self.factor_cols
            )
            normalized = self._normalize_weight_tuple(combo)
            if normalized is None or normalized in seen_random:
                continue
            seen_random.add(normalized)
            yield dict(zip(self.factor_cols, normalized, strict=True))

    def _normalize_weight_tuple(self, combo: Sequence[float]) -> tuple[float, ...] | None:
        weights = np.asarray(combo, dtype=float)
        gross = float(np.abs(weights).sum())
        if gross <= 0.0:
            return None
        normalized = tuple(np.round(weights / gross, 8))
        return normalized

    def _simulate_portfolio(
        self,
        frame: pd.DataFrame,
        weights: dict[str, float],
        fold_id: int,
        split_label: str,
    ) -> dict[str, Any]:
        scores = self._compute_scores(frame, weights)
        predictions = frame.loc[:, [self.date_col, self.asset_col, self.target_col]].copy()
        predictions["fold_id"] = fold_id
        predictions["split"] = split_label
        predictions["signal_score"] = scores
        predictions["position"] = predictions.groupby(self.date_col, group_keys=False)["signal_score"].apply(self._score_to_position)

        position_matrix = predictions.pivot(index=self.date_col, columns=self.asset_col, values="position").fillna(0.0)
        return_matrix = predictions.pivot(index=self.date_col, columns=self.asset_col, values=self.target_col).fillna(0.0)
        gross_return = (position_matrix * return_matrix).sum(axis=1)
        turnover = position_matrix.diff().abs().sum(axis=1)
        if not turnover.empty:
            turnover.iloc[0] = position_matrix.iloc[0].abs().sum()
        cost = turnover * (self.transaction_cost_bps / 10000.0)
        net_return = (gross_return - cost).rename("portfolio_return")

        return {
            "predictions": predictions,
            "daily_returns": net_return,
            "metrics": self._compute_metrics(net_return),
        }

    def _compute_scores(self, frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
        weight_vector = np.asarray([weights[factor] for factor in self.factor_cols], dtype=float)
        feature_matrix = frame.loc[:, self.factor_cols].to_numpy(dtype=float)
        scores = feature_matrix @ weight_vector
        return pd.Series(scores, index=frame.index, dtype=float)

    def _score_to_position(self, signal: pd.Series) -> pd.Series:
        cleaned = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if cleaned.empty:
            return cleaned
        if len(cleaned) == 1:
            value = float(np.tanh(cleaned.iloc[0]))
            return pd.Series([value], index=cleaned.index, dtype=float)

        centered = cleaned - float(cleaned.mean())
        dispersion = float(centered.std(ddof=0))
        if dispersion <= 1e-12:
            scaled = np.sign(centered)
        else:
            scaled = centered / dispersion
        clipped = scaled.clip(-self.position_cap, self.position_cap)
        gross = float(clipped.abs().sum())
        if gross <= 1e-12:
            return pd.Series(0.0, index=cleaned.index, dtype=float)
        return (clipped / gross).astype(float)

    def _compute_metrics(self, daily_returns: pd.Series) -> dict[str, float]:
        cleaned = daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if cleaned.empty:
            return {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "calmar": 0.0,
                "annualized_return": 0.0,
            }

        volatility = float(cleaned.std(ddof=0))
        sharpe = 0.0 if volatility <= 1e-12 else float(np.sqrt(self.annualization) * cleaned.mean() / volatility)
        equity_curve = self._build_equity_curve(cleaned)
        max_drawdown = float(equity_curve["drawdown"].min())
        total_return = float(equity_curve["equity_curve"].iloc[-1])
        periods = len(cleaned)
        annualized_return = 0.0
        if periods > 0 and total_return > 0.0:
            annualized_return = float(total_return ** (self.annualization / periods) - 1.0)
        calmar = 0.0 if abs(max_drawdown) <= 1e-12 else float(annualized_return / abs(max_drawdown))
        return {
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "annualized_return": annualized_return,
        }

    def _build_equity_curve(self, daily_returns: pd.Series) -> pd.DataFrame:
        cleaned = daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        equity_curve = (1.0 + cleaned).cumprod()
        running_peak = equity_curve.cummax()
        drawdown = equity_curve / running_peak - 1.0
        return pd.DataFrame(
            {
                "portfolio_return": cleaned,
                "equity_curve": equity_curve,
                "drawdown": drawdown,
            }
        )

    def _fold_result_to_row(self, fold_result: FoldResult) -> dict[str, Any]:
        row = asdict(fold_result)
        for factor, weight in fold_result.best_weights.items():
            row[f"weight_{factor}"] = weight
        return row


def build_project_panel_data(
    histories: dict[str, pd.DataFrame],
    benchmark_ticker: str = "QQQ",
    target_horizon: int = 1,
    min_rows_per_asset: int = 80,
) -> pd.DataFrame:
    benchmark_history = histories.get(benchmark_ticker, pd.DataFrame())
    benchmark_close = benchmark_history.get("close") if not benchmark_history.empty else None
    panel_frames: list[pd.DataFrame] = []

    for ticker, history in histories.items():
        if history is None or history.empty:
            continue

        frame = history.copy().sort_index()
        if "close" not in frame or "volume" not in frame:
            continue

        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        hist_vol_20d = compute_historical_volatility(close, window=20)
        hist_vol_60d = compute_historical_volatility(close, window=60)
        log_return_10d = compute_log_return(close, periods=10)
        log_return_60d = compute_log_return(close, periods=60)
        factor_frame = pd.DataFrame(index=frame.index)
        factor_frame["asset"] = ticker
        factor_frame["log_return_10d"] = log_return_10d
        factor_frame["log_return_60d"] = log_return_60d
        factor_frame["hist_vol_20d"] = hist_vol_20d
        factor_frame["hist_vol_60d"] = hist_vol_60d
        factor_frame["volume_ratio_20d"] = compute_volume_ratio(volume, window=20)
        factor_frame["rsi_14"] = compute_rsi(close, window=14)
        factor_frame["forward_return_1d"] = close.pct_change(target_horizon).shift(-target_horizon)
        factor_frame["daily_return_pct"] = close.pct_change() * 100.0
        factor_frame["breakout_20d"] = (
            close > close.shift(1).rolling(window=20, min_periods=20).max()
        ).astype(float)

        if benchmark_close is not None and ticker != benchmark_ticker:
            factor_frame["relative_strength_20d_vs_benchmark"] = compute_relative_strength(
                close,
                benchmark_close,
                window=20,
            )
        else:
            factor_frame["relative_strength_20d_vs_benchmark"] = 0.0

        factor_frame = factor_frame.reset_index(names="date")
        factor_frame = factor_frame.replace([np.inf, -np.inf], np.nan)
        factor_frame = factor_frame.dropna(subset=["forward_return_1d"])
        if len(factor_frame) < min_rows_per_asset:
            continue
        panel_frames.append(factor_frame)

    if not panel_frames:
        return pd.DataFrame(
            columns=[
                "date",
                "asset",
                "log_return_10d",
                "log_return_60d",
                "hist_vol_20d",
                "hist_vol_60d",
                "volume_ratio_20d",
                "rsi_14",
                "relative_strength_20d_vs_benchmark",
                "daily_return_pct",
                "breakout_20d",
                "forward_return_1d",
            ]
        )

    panel = pd.concat(panel_frames, ignore_index=True)
    return panel.sort_values(["date", "asset"]).reset_index(drop=True)


def load_project_panel_data(
    tickers: Sequence[str],
    benchmark_ticker: str = "QQQ",
    period: str = "18mo",
    interval: str = "1d",
    target_horizon: int = 1,
    auto_adjust: bool = True,
    min_rows_per_asset: int = 80,
) -> pd.DataFrame:
    requested = list(dict.fromkeys([benchmark_ticker, *tickers]))
    histories = batch_history(
        requested,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    return build_project_panel_data(
        histories=histories,
        benchmark_ticker=benchmark_ticker,
        target_horizon=target_horizon,
        min_rows_per_asset=min_rows_per_asset,
    )


def build_roll_forward_path_specs(
    random_states: Sequence[int] | None = None,
    fold_offsets: Sequence[int] | None = None,
    fold_step: int | None = None,
) -> list[RollForwardPathSpec]:
    resolved_random_states = tuple(int(value) for value in (random_states or (11, 23, 37, 41, 53, 67, 79, 97)))
    resolved_fold_offsets = tuple(int(value) for value in (fold_offsets or tuple(index * 3 for index in range(len(resolved_random_states)))))
    if len(resolved_random_states) != len(resolved_fold_offsets):
        raise ValueError("random_states_and_fold_offsets_must_match")

    return [
        RollForwardPathSpec(
            path_id=f"path_{index:02d}",
            random_state=random_state,
            fold_offset=max(fold_offset, 0),
            fold_step=fold_step,
        )
        for index, (random_state, fold_offset) in enumerate(zip(resolved_random_states, resolved_fold_offsets, strict=True))
    ]


def _summarize_factor_path_distribution(path_frame: pd.DataFrame) -> dict[str, Any]:
    if path_frame.empty:
        return {}
    return {
        "factor_set": str(path_frame["factor_set"].iloc[0]),
        "path_count": int(len(path_frame)),
        "mean_sharpe": float(path_frame["sharpe"].mean()),
        "median_sharpe": float(path_frame["sharpe"].median()),
        "sharpe_std": float(path_frame["sharpe"].std(ddof=0)),
        "min_sharpe": float(path_frame["sharpe"].min()),
        "max_sharpe": float(path_frame["sharpe"].max()),
        "positive_sharpe_ratio": float((path_frame["sharpe"] > 0.0).mean()),
        "mean_calmar": float(path_frame["calmar"].mean()),
        "mean_max_drawdown": float(path_frame["max_drawdown"].mean()),
        "mean_annualized_return": float(path_frame["annualized_return"].mean()),
    }


def _compare_factor_set_paths(
    path_metrics: pd.DataFrame,
    candidate_label: str,
    reference_label: str,
) -> dict[str, Any]:
    candidate_frame = path_metrics[path_metrics["factor_set"] == candidate_label].copy()
    reference_frame = path_metrics[path_metrics["factor_set"] == reference_label].copy()
    if candidate_frame.empty or reference_frame.empty:
        return {}

    merged = candidate_frame.merge(
        reference_frame,
        on="path_id",
        suffixes=("_candidate", "_reference"),
    )
    if merged.empty:
        return {}

    sharpe_diff = merged["sharpe_candidate"] - merged["sharpe_reference"]
    calmar_diff = merged["calmar_candidate"] - merged["calmar_reference"]
    annualized_return_diff = merged["annualized_return_candidate"] - merged["annualized_return_reference"]
    max_drawdown_diff = merged["max_drawdown_candidate"] - merged["max_drawdown_reference"]
    improvement_ratio = float((sharpe_diff > 0.0).mean())
    return {
        "candidate_factor_set": candidate_label,
        "reference_factor_set": reference_label,
        "path_count": int(len(merged)),
        "improvement_ratio": improvement_ratio,
        "mean_sharpe_diff": float(sharpe_diff.mean()),
        "median_sharpe_diff": float(sharpe_diff.median()),
        "mean_calmar_diff": float(calmar_diff.mean()),
        "mean_annualized_return_diff": float(annualized_return_diff.mean()),
        "mean_max_drawdown_diff": float(max_drawdown_diff.mean()),
        "statistically_significant_improvement": bool(improvement_ratio >= 0.8),
    }


def run_roll_forward_factor_study(
    panel_data: pd.DataFrame,
    factor_sets: dict[str, Sequence[str]],
    target_col: str,
    train_window: int,
    test_window: int,
    path_specs: Sequence[RollForwardPathSpec] | None = None,
    search_method: SearchMethod = "random",
    param_grid: dict[str, Sequence[float]] | None = None,
    n_iter: int = 24,
    annualization: int = 252,
    transaction_cost_bps: float = 0.0,
    winsor_quantiles: tuple[float, float] = (0.01, 0.99),
    position_cap: float = 1.0,
    comparison_pairs: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    resolved_paths = list(path_specs or build_roll_forward_path_specs())
    if not resolved_paths:
        raise ValueError("roll_forward_paths_must_not_be_empty")

    path_metric_rows: list[dict[str, Any]] = []
    for factor_set_name, factor_cols in factor_sets.items():
        for path_spec in resolved_paths:
            validator = WalkForwardValidator(
                panel_data=panel_data,
                factor_cols=factor_cols,
                target_col=target_col,
                train_window=train_window,
                test_window=test_window,
                search_method=search_method,
                param_grid=param_grid,
                n_iter=n_iter,
                annualization=annualization,
                transaction_cost_bps=transaction_cost_bps,
                winsor_quantiles=winsor_quantiles,
                position_cap=position_cap,
                random_state=path_spec.random_state,
                fold_offset=path_spec.fold_offset,
                fold_step=path_spec.fold_step,
            )
            result = validator.run()
            path_metric_rows.append(
                {
                    "factor_set": factor_set_name,
                    "path_id": path_spec.path_id,
                    "random_state": path_spec.random_state,
                    "fold_offset": path_spec.fold_offset,
                    "fold_step": path_spec.fold_step if path_spec.fold_step is not None else test_window,
                    "sharpe": float(result.metrics.get("sharpe") or 0.0),
                    "max_drawdown": float(result.metrics.get("max_drawdown") or 0.0),
                    "calmar": float(result.metrics.get("calmar") or 0.0),
                    "annualized_return": float(result.metrics.get("annualized_return") or 0.0),
                    "fold_count": int(len(result.fold_summary)),
                }
            )

    path_metrics = pd.DataFrame(path_metric_rows)
    factor_summary = pd.DataFrame(
        [
            _summarize_factor_path_distribution(group.copy())
            for _, group in path_metrics.groupby("factor_set", sort=False)
        ]
    )
    if not factor_summary.empty:
        factor_summary = factor_summary.sort_values(["mean_sharpe", "mean_calmar"], ascending=False).reset_index(drop=True)

    resolved_pairs = list(comparison_pairs or [])
    pairwise_rows = [
        _compare_factor_set_paths(path_metrics, candidate_label, reference_label)
        for candidate_label, reference_label in resolved_pairs
    ]
    pairwise_rows = [row for row in pairwise_rows if row]
    pairwise_summary = pd.DataFrame(pairwise_rows)
    if not pairwise_summary.empty:
        pairwise_summary = pairwise_summary.sort_values(["statistically_significant_improvement", "improvement_ratio", "mean_sharpe_diff"], ascending=[False, False, False]).reset_index(drop=True)

    best_factor_set = str(factor_summary.iloc[0]["factor_set"]) if not factor_summary.empty else ""
    return {
        "path_specs": pd.DataFrame([asdict(path) for path in resolved_paths]),
        "path_metrics": path_metrics,
        "factor_summary": factor_summary,
        "pairwise_summary": pairwise_summary,
        "best_factor_set": best_factor_set,
    }