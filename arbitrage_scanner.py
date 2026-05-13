from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint

from config import EXCHANGE_TICKERS_FILE, TICKER_METADATA
from portfolio.custom_tickers import load_custom_ticker_metadata
from portfolio.exchange_tickers import import_public_exchange_tickers
from portfolio.ticker_relationships import get_ticker_relationship_profile


@dataclass(frozen=True, slots=True)
class EdgeSeed:
    leader: str
    follower: str
    relation_type: str
    weight: float
    direction: int = 1
    leader_fetch_symbol: str | None = None
    leader_display: str | None = None


@dataclass(frozen=True, slots=True)
class ArbitrageScannerConfig:
    daily_corr_floor: float = 0.72
    coint_pvalue_max: float = 0.05
    max_lag_bars: int = 3
    shadow_gap_trigger: float = 0.015
    score_floor: float = 0.55


EDGE_SEEDS: tuple[EdgeSeed, ...] = (
    EdgeSeed("SOXL", "NVDA", "leveraged_etf", 0.98, 1),
    EdgeSeed("SOXL", "AMD", "leveraged_etf", 0.92, 1),
    EdgeSeed("SOXL", "AVGO", "leveraged_etf", 0.90, 1),
    EdgeSeed("TQQQ", "QQQ", "leveraged_index", 0.96, 1),
    EdgeSeed("NVDL", "NVDA", "leveraged_single_name", 0.99, 1),
    EdgeSeed("SIVE.ST", "SIVEF", "foreign_primary_link", 0.90, 1, leader_fetch_symbol="SIVE.ST", leader_display="STO:SIVE"),
)


class AsyncYahooAdapter:
    async def batch_history(self, tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
        unique = list(dict.fromkeys(str(ticker or "").strip().upper() if ":" not in str(ticker or "") else str(ticker or "").strip() for ticker in tickers if str(ticker or "").strip()))
        if not unique:
            return {}

        try:
            raw = await asyncio.to_thread(
                yf.download,
                tickers=unique,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            return {}

        history_map: dict[str, pd.DataFrame] = {}
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in unique:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    frame = raw[ticker].copy()
                    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
                    frame = frame.rename(columns={"adj_close": "close", "adj close": "close"}).dropna(subset=["close"])
                    if not frame.empty:
                        history_map[ticker] = frame
            else:
                frame = raw.copy()
                frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
                frame = frame.rename(columns={"adj_close": "close", "adj close": "close"}).dropna(subset=["close"])
                if unique and not frame.empty:
                    history_map[unique[0]] = frame
        return history_map


def load_ticker_master(path: Path = EXCHANGE_TICKERS_FILE) -> pd.DataFrame:
    if not path.exists():
        import_public_exchange_tickers(path)

    metadata = load_custom_ticker_metadata(path)
    rows = [
        {
            "ticker": ticker,
            "name": entry.get("name") or ticker,
            "category": entry.get("category") or "",
        }
        for ticker, entry in metadata.items()
    ]
    return pd.DataFrame(rows)


class KnowledgeGraphEngine:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def build(self, target_symbols: list[str], master: pd.DataFrame) -> nx.MultiDiGraph:
        for row in master.itertuples(index=False):
            ticker = str(getattr(row, "ticker", "") or "").upper()
            if not ticker:
                continue
            self.graph.add_node(
                ticker,
                ticker=ticker,
                name=getattr(row, "name", ticker),
                category=getattr(row, "category", ""),
            )

        for seed in EDGE_SEEDS:
            self.graph.add_edge(
                seed.leader,
                seed.follower,
                relation_type=seed.relation_type,
                weight=seed.weight,
                direction=seed.direction,
                leader_fetch_symbol=seed.leader_fetch_symbol or seed.leader,
                leader_display=seed.leader_display or seed.leader,
            )

        normalized_targets = sorted({str(symbol or "").strip().upper() for symbol in target_symbols if str(symbol or "").strip()})
        for follower in normalized_targets:
            profile = get_ticker_relationship_profile(follower, metadata=TICKER_METADATA)
            relation_type = str(profile.get("relation_type") or "theme_peer")
            weight = float(profile.get("relationship_strength") or 0.0)
            for leader in list(profile.get("related_tickers") or [])[:10]:
                if not leader or leader == follower:
                    continue
                self.graph.add_edge(
                    leader,
                    follower,
                    relation_type=relation_type or "theme_peer",
                    weight=max(weight, 0.55),
                    direction=1,
                    leader_fetch_symbol=leader,
                    leader_display=leader,
                )
        return self.graph


class CorrelationGraphScanner:
    def __init__(self, graph: nx.MultiDiGraph, market: AsyncYahooAdapter, config: ArbitrageScannerConfig | None = None) -> None:
        self.graph = graph
        self.market = market
        self.config = config or ArbitrageScannerConfig()

    @staticmethod
    def _log_returns(history: pd.DataFrame) -> pd.Series:
        close = history["close"].astype(float)
        return np.log(close).diff().dropna()

    @staticmethod
    def _pearson(a: pd.Series, b: pd.Series, tail: int) -> float:
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna().tail(tail)
        if len(joined) < max(20, tail // 2):
            return 0.0
        return float(joined["a"].corr(joined["b"]))

    def _lead_lag_corr(self, leader_ret: pd.Series, follower_ret: pd.Series) -> tuple[float, int]:
        best_corr = -1.0
        best_lag = 0
        for lag in range(1, self.config.max_lag_bars + 1):
            joined = pd.concat(
                [leader_ret.rename("leader"), follower_ret.shift(-lag).rename("follower_future")],
                axis=1,
            ).dropna()
            if len(joined) < 30:
                continue
            corr = float(joined["leader"].corr(joined["follower_future"]))
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag
        return max(best_corr, 0.0), best_lag

    @staticmethod
    def _cointegration_stats(leader_close: pd.Series, follower_close: pd.Series) -> tuple[float, float, float]:
        joined = pd.concat([leader_close.rename("leader"), follower_close.rename("follower")], axis=1).dropna()
        if len(joined) < 80:
            return 1.0, 0.0, 0.0
        hedge_beta = float(np.polyfit(joined["leader"], joined["follower"], 1)[0])
        spread = joined["follower"] - hedge_beta * joined["leader"]
        spread_std = float(spread.std(ddof=0))
        spread_zscore = 0.0 if spread_std == 0.0 else float((spread.iloc[-1] - spread.mean()) / spread_std)
        pvalue = float(coint(joined["follower"], joined["leader"])[1])
        return pvalue, hedge_beta, spread_zscore

    @staticmethod
    def _latest_return(history: pd.DataFrame) -> float:
        close = history["close"].astype(float)
        if len(close) < 2:
            return 0.0
        return float(close.iloc[-1] / close.iloc[-2] - 1.0)

    def _score(self, leader_shock: float, lead_lag_corr: float, coint_pvalue: float, graph_weight: float, spread_zscore: float) -> float:
        shock_term = float(np.clip(abs(leader_shock) / 0.08, 0.0, 1.0))
        coint_term = 1.0 - float(np.clip(coint_pvalue, 0.0, 1.0))
        z_term = float(np.clip(abs(spread_zscore) / 3.0, 0.0, 1.0))
        return 0.30 * shock_term + 0.25 * lead_lag_corr + 0.20 * coint_term + 0.15 * float(np.clip(graph_weight, 0.0, 1.0)) + 0.10 * z_term

    async def scan(self, target_symbols: list[str]) -> pd.DataFrame:
        tradable = {str(symbol or "").strip().upper() for symbol in target_symbols if str(symbol or "").strip()}
        candidate_edges = [(leader, follower, edge) for leader, follower, edge in self.graph.edges(data=True) if follower in tradable]
        if not candidate_edges:
            return pd.DataFrame()

        fetch_symbols = sorted(
            {str(edge.get("leader_fetch_symbol") or leader) for leader, _, edge in candidate_edges}
            | {follower for _, follower, _ in candidate_edges}
        )
        daily_map, intraday_map = await asyncio.gather(
            self.market.batch_history(fetch_symbols, period="9mo", interval="1d"),
            self.market.batch_history(fetch_symbols, period="5d", interval="1m"),
        )

        rows: list[dict[str, Any]] = []
        for leader, follower, edge in candidate_edges:
            leader_fetch_symbol = str(edge.get("leader_fetch_symbol") or leader)
            leader_display = str(edge.get("leader_display") or leader)
            leader_daily = daily_map.get(leader_fetch_symbol)
            follower_daily = daily_map.get(follower)
            if leader_daily is None or follower_daily is None or leader_daily.empty or follower_daily.empty:
                continue

            leader_daily_ret = self._log_returns(leader_daily)
            follower_daily_ret = self._log_returns(follower_daily)
            pearson_20d = self._pearson(leader_daily_ret, follower_daily_ret, tail=20)
            pearson_60d = self._pearson(leader_daily_ret, follower_daily_ret, tail=60)
            if max(abs(pearson_20d), abs(pearson_60d)) < self.config.daily_corr_floor:
                continue

            leader_intraday = intraday_map.get(leader_fetch_symbol, leader_daily)
            follower_intraday = intraday_map.get(follower, follower_daily)
            lead_lag_corr, lead_lag_bars = self._lead_lag_corr(self._log_returns(leader_intraday), self._log_returns(follower_intraday))
            coint_pvalue, hedge_beta, spread_zscore = self._cointegration_stats(leader_daily["close"].astype(float), follower_daily["close"].astype(float))
            if coint_pvalue > self.config.coint_pvalue_max and lead_lag_corr < 0.30:
                continue

            direction = int(edge.get("direction") or 1)
            leader_shock = self._latest_return(leader_intraday)
            follower_shock = self._latest_return(follower_intraday)
            shock_spread = direction * leader_shock - follower_shock
            if shock_spread <= self.config.shadow_gap_trigger:
                continue

            signal_score = self._score(shock_spread, lead_lag_corr, coint_pvalue, float(edge.get("weight") or 0.5), spread_zscore)
            if signal_score < self.config.score_floor:
                continue

            execution_direction = "BUY" if direction > 0 else "SELL_SHORT"
            rows.append(
                {
                    "leader": leader_display,
                    "leader_fetch_symbol": leader_fetch_symbol,
                    "follower": follower,
                    "relation_type": str(edge.get("relation_type") or ""),
                    "pearson_20d": pearson_20d,
                    "pearson_60d": pearson_60d,
                    "lead_lag_corr": lead_lag_corr,
                    "lead_lag_bars": lead_lag_bars,
                    "coint_pvalue": coint_pvalue,
                    "hedge_beta": hedge_beta,
                    "spread_zscore": spread_zscore,
                    "leader_shock": leader_shock,
                    "follower_shock": follower_shock,
                    "shock_spread": shock_spread,
                    "graph_weight": float(edge.get("weight") or 0.5),
                    "signal_score": signal_score,
                    "Execution_Direction": execution_direction,
                    "execution_direction": execution_direction,
                    "thesis": f"{leader_display} 先行异动，{follower} 滞后；corr20={pearson_20d:.2f}，corr60={pearson_60d:.2f}，lag={lead_lag_bars}，coint_p={coint_pvalue:.3f}，spread_z={spread_zscore:.2f}",
                    "follower_close": float(follower_daily["close"].iloc[-1]),
                    "leader_close": float(leader_daily["close"].iloc[-1]),
                }
            )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["signal_score", "lead_lag_corr", "pearson_20d"], ascending=False).reset_index(drop=True)


def arbitrage_opportunities_to_candidates(opportunities: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(opportunities, pd.DataFrame) or opportunities.empty:
        return []

    candidates: list[dict[str, Any]] = []
    for row in opportunities.to_dict(orient="records"):
        execution_direction = str(row.get("Execution_Direction") or "BUY").upper()
        trade_direction = "SHORT" if execution_direction == "SELL_SHORT" else "LONG"
        shock_spread = float(row.get("shock_spread") or 0.0)
        signal_score = float(row.get("signal_score") or 0.0)
        lead_lag_corr = float(row.get("lead_lag_corr") or 0.0)
        spread_zscore = float(row.get("spread_zscore") or 0.0)
        raw_edge = min(max(shock_spread * (0.9 + 0.35 * lead_lag_corr), 0.012), 0.12)
        expected_return_5d = raw_edge if trade_direction == "LONG" else -raw_edge
        stop_loss_pct = float(np.clip(0.02 + abs(shock_spread) * 0.55, 0.03, 0.08))
        win_rate = float(np.clip(0.50 + 0.18 * signal_score + 0.08 * max(lead_lag_corr - 0.30, 0.0), 0.45, 0.78))
        payoff_ratio = float(np.clip(0.92 + abs(spread_zscore) * 0.12 + lead_lag_corr * 0.35, 0.92, 1.85))
        target_ev = win_rate * abs(expected_return_5d) - (1.0 - win_rate) * stop_loss_pct
        eligible_for_risk = bool(signal_score >= 0.60 and target_ev >= 0.0045 and (float(row.get("coint_pvalue") or 1.0) <= 0.10 or lead_lag_corr >= 0.40))
        gate_reason = "通过图谱套利阈值" if eligible_for_risk else "图谱套利 EV 或协整/滞后相关不足"
        signal_label = "跨国影子套利" if str(row.get("relation_type") or "") == "foreign_primary_link" else "图谱联动套利"
        candidates.append(
            {
                "ticker": str(row.get("follower") or "").upper(),
                "close": float(row.get("follower_close") or 0.0),
                "reference_price": float(row.get("follower_close") or 0.0),
                "regular_close": float(row.get("follower_close") or 0.0),
                "trade_direction": trade_direction,
                "Signal_Mode": "graph_arbitrage",
                "signal_mode": "graph_arbitrage",
                "signal_label": signal_label,
                "signal_reason": str(row.get("thesis") or ""),
                "expected_return_5d": expected_return_5d,
                "expected_return_10d": expected_return_5d * 1.6,
                "stop_loss_pct": stop_loss_pct,
                "payoff_ratio": payoff_ratio,
                "win_rate": win_rate,
                "target_ev": target_ev,
                "eligible_for_risk": eligible_for_risk,
                "gate_reason": gate_reason,
                "signal_strength": signal_score,
                "graph_signal_score": signal_score,
                "relation_type": row.get("relation_type"),
                "relation_group": "图谱套利链",
                "related_leader_ticker": row.get("leader"),
                "lead_lag_corr": row.get("lead_lag_corr"),
                "lead_lag_bars": row.get("lead_lag_bars"),
                "pearson_20d": row.get("pearson_20d"),
                "pearson_60d": row.get("pearson_60d"),
                "coint_pvalue": row.get("coint_pvalue"),
                "hedge_beta": row.get("hedge_beta"),
                "spread_zscore": row.get("spread_zscore"),
                "graph_weight": row.get("graph_weight"),
                "relationship_strength": row.get("graph_weight"),
                "relationship_logic": row.get("thesis"),
                "Execution_Direction": execution_direction,
                "execution_direction": execution_direction,
            }
        )
    return candidates


async def _scan_graph_arbitrage_async(target_symbols: list[str], refresh_master: bool = False) -> pd.DataFrame:
    if refresh_master or not EXCHANGE_TICKERS_FILE.exists():
        import_public_exchange_tickers(EXCHANGE_TICKERS_FILE)
    master = load_ticker_master(EXCHANGE_TICKERS_FILE)
    graph = KnowledgeGraphEngine().build(target_symbols, master)
    scanner = CorrelationGraphScanner(graph=graph, market=AsyncYahooAdapter())
    return await scanner.scan(target_symbols)


def scan_graph_arbitrage_candidates(target_symbols: list[str], refresh_master: bool = False) -> pd.DataFrame:
    try:
        return asyncio.run(_scan_graph_arbitrage_async(target_symbols, refresh_master=refresh_master))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_scan_graph_arbitrage_async(target_symbols, refresh_master=refresh_master))
        finally:
            loop.close()