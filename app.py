from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from config import (
    AAOX_UNDERLYING,
    CACHE_TTL_SECONDS,
    CORE_SIGNAL_TICKERS,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_HOLDING_TICKER,
    HOLDINGS_FILE,
    OPTIONS_TICKER,
    RADAR_UNIVERSE,
    REGIME_SUPPORT_TICKERS,
    RISK_FREE_RATE,
    TICKER_METADATA,
    TICKER_SUGGESTION_UNIVERSE,
)
from data.fetcher import batch_history, get_latest_quote, get_market_session_quote, get_news_items
from data.indicators import build_factor_snapshot
from data.macro import get_macro_snapshot
from data.options import get_near_term_options_summary
from data.radar import scan_breakout_candidates
from data.sentiment import build_news_features
from models.kelly import continuous_kelly
from models.regime import classify_market_regime, detect_breakdown_signal, detect_sigma_event
from models.signals import build_candidate_trade_profile, build_trade_profile, should_force_observe
from portfolio.holdings import enrich_holdings_with_quotes, frame_to_holdings, load_holdings, save_holdings
from reports.generator import build_execution_payload, build_markdown_report, save_report_bundle


st.set_page_config(page_title="美股量化半自动助手", layout="wide")


def _ticker_meta(ticker: str) -> dict[str, str]:
    return TICKER_METADATA.get(ticker.upper().strip(), {})


def _format_ticker_option(ticker: str) -> str:
    meta = _ticker_meta(ticker)
    name = meta.get("name", "")
    category = meta.get("category", "")
    extras = " | ".join(part for part in [name, category] if part)
    return f"{ticker} - {extras}" if extras else ticker


def _new_holding_row(ticker: str = "") -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "ticker": ticker,
        "shares": 0.0,
        "cost_basis": 0.0,
        "notes": "",
    }


def _editor_rows_from_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if rows:
        return [
            {
                "id": str(uuid4()),
                "ticker": row.get("ticker", ""),
                "shares": float(row.get("shares", 0.0) or 0.0),
                "cost_basis": float(row.get("cost_basis", 0.0) or 0.0),
                "notes": row.get("notes", ""),
            }
            for row in rows
        ]
    return [_new_holding_row(DEFAULT_HOLDING_TICKER)]


def _ensure_holding_editor_rows(saved_rows: list[dict[str, Any]]) -> None:
    if "holding_editor_rows" not in st.session_state:
        st.session_state["holding_editor_rows"] = _editor_rows_from_holdings(saved_rows)


def _collect_sidebar_holdings() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in st.session_state.get("holding_editor_rows", []):
        row_id = row["id"]
        rows.append(
            {
                "ticker": st.session_state.get(f"holding_ticker_{row_id}", row.get("ticker", "")),
                "shares": st.session_state.get(f"holding_shares_{row_id}", float(row.get("shares", 0.0) or 0.0)),
                "cost_basis": st.session_state.get(f"holding_cost_{row_id}", float(row.get("cost_basis", 0.0) or 0.0)),
                "notes": st.session_state.get(f"holding_notes_{row_id}", row.get("notes", "")),
            }
        )
    return frame_to_holdings(pd.DataFrame(rows)) if rows else []


def _reset_holding_editor(rows: list[dict[str, Any]]) -> None:
    st.session_state["holding_editor_rows"] = _editor_rows_from_holdings(rows)


def _render_sidebar_holding_editor(saved_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _ensure_holding_editor_rows(saved_rows)

    toolbar_left, toolbar_right = st.columns(2)
    with toolbar_left:
        if st.button("新增一行", use_container_width=True):
            st.session_state["holding_editor_rows"].append(_new_holding_row())
            st.rerun()
    with toolbar_right:
        if st.button("重置为已保存", use_container_width=True):
            _reset_holding_editor(saved_rows)
            st.rerun()

    st.caption("代码框支持直接输入字母搜索候选代码，也支持手动输入未收录的新代码。")

    for index, row in enumerate(st.session_state.get("holding_editor_rows", []), start=1):
        row_id = row["id"]
        ticker_key = f"holding_ticker_{row_id}"
        shares_key = f"holding_shares_{row_id}"
        cost_key = f"holding_cost_{row_id}"
        note_key = f"holding_notes_{row_id}"
        auto_note_key = f"holding_auto_note_{row_id}"
        suggestion_key = f"holding_suggestion_{row_id}"

        st.session_state.setdefault(ticker_key, row.get("ticker", ""))
        st.session_state.setdefault(shares_key, float(row.get("shares", 0.0) or 0.0))
        st.session_state.setdefault(cost_key, float(row.get("cost_basis", 0.0) or 0.0))
        st.session_state.setdefault(note_key, row.get("notes", ""))

        with st.container(border=True):
            st.markdown(f"**持仓 {index}**")
            top_left, top_right = st.columns(2)
            with top_left:
                current_ticker = str(st.session_state.get(ticker_key, row.get("ticker", "")) or "").upper().strip()
                current_index = TICKER_SUGGESTION_UNIVERSE.index(current_ticker) if current_ticker in TICKER_SUGGESTION_UNIVERSE else None
                ticker_value = st.selectbox(
                    "代码",
                    options=TICKER_SUGGESTION_UNIVERSE,
                    index=current_index,
                    format_func=_format_ticker_option,
                    placeholder="输入代码，例如 NVDA",
                    key=ticker_key,
                    accept_new_options=True,
                    filter_mode="fuzzy",
                )
                normalized_ticker = str(ticker_value or "").upper().strip()
                meta = _ticker_meta(normalized_ticker)
                previous_auto_note = st.session_state.get(auto_note_key, "")
                current_note = st.session_state.get(note_key, "")
                auto_note = meta.get("category", "")
                if auto_note and (not current_note or current_note == previous_auto_note):
                    st.session_state[note_key] = auto_note
                if auto_note:
                    st.session_state[auto_note_key] = auto_note

                if meta:
                    st.caption(f"自动识别：{meta.get('name', normalized_ticker)} | {meta.get('category', '未分类')}")
                else:
                    st.caption("自动识别：未收录，仍可保存并参与行情分析")
            with top_right:
                st.number_input(
                    "股数",
                    min_value=0.0,
                    value=float(row.get("shares", 0.0) or 0.0),
                    step=1.0,
                    key=shares_key,
                )

            bottom_left, bottom_right = st.columns(2)
            with bottom_left:
                st.number_input(
                    "成本价",
                    min_value=0.0,
                    value=float(row.get("cost_basis", 0.0) or 0.0),
                    step=0.01,
                    format="%.2f",
                    key=cost_key,
                )
            with bottom_right:
                st.text_input(
                    "备注",
                    value=row.get("notes", ""),
                    placeholder="例如 核心仓 / 波段仓",
                    key=note_key,
                )

            if st.button("删除这一行", key=f"delete_holding_{row_id}", use_container_width=True):
                st.session_state["holding_editor_rows"] = [
                    item for item in st.session_state["holding_editor_rows"] if item["id"] != row_id
                ]
                if not st.session_state["holding_editor_rows"]:
                    st.session_state["holding_editor_rows"] = [_new_holding_row()]
                st.rerun()

    return _collect_sidebar_holdings()


def _display_holdings_frame(holdings_enriched: list[dict[str, Any]]) -> pd.DataFrame:
    if not holdings_enriched:
        return pd.DataFrame()
    frame = pd.DataFrame(holdings_enriched)
    return frame.rename(
        columns={
            "ticker": "代码",
            "shares": "股数",
            "cost_basis": "成本价",
            "notes": "备注",
            "market_price": "最新价",
            "market_value": "市值",
            "cost_value": "成本市值",
            "pnl": "浮盈亏",
            "pnl_pct": "浮盈亏%",
            "market_session": "会话阶段",
            "session_change_pct": "扩展时段涨跌幅%",
            "quote_as_of": "价格抓取时间",
        }
    )


def _display_orders_frame(orders: list[dict[str, Any]]) -> pd.DataFrame:
    if not orders:
        return pd.DataFrame()
    frame = pd.DataFrame(orders).copy()

    signal_value_map = {
        "BUY": "买入",
        "IGNORE": "忽略",
        "ADD": "加仓",
        "HOLD": "持有/观望",
        "REDUCE": "减仓",
        "LIQUIDATE": "清仓",
    }
    mode_value_map = {
        "momentum": "动能延续",
        "neutral": "观望",
        "mean_reversion": "均值回归",
    }
    channel_value_map = {
        "Google News RSS": "Google 新闻 RSS",
        "Yahoo Finance": "Yahoo Finance 聚合",
        "yfinance.info": "yfinance 实时快照",
    }

    if "Signal" in frame.columns:
        frame["Signal"] = frame["Signal"].map(lambda value: signal_value_map.get(value, value))

    for column in ["Signal_Mode", "signal_mode"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: mode_value_map.get(value, value))

    for column in ["Price_Source", "price_source", "Latest_News_Channel", "latest_news_channel"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: channel_value_map.get(value, value))

    for column in ["News_Channels", "news_channels"]:
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: " + ".join(channel_value_map.get(item, item) for item in value)
                if isinstance(value, list)
                else value
            )

    for column in ["breakout_valid", "fake_breakout", "breakdown", "eligible_for_risk"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: "是" if bool(value) else "否")

    return frame.rename(
        columns={
            "Ticker": "代码",
            "ticker": "代码",
            "Signal": "信号",
            "Action_Price": "动作价格",
            "Trailing_Stop_Update": "移动止损",
            "Hard_Stop_Loss": "硬止损",
            "Signal_Mode": "信号模式",
            "signal_mode": "信号模式",
            "signal_label": "信号标签",
            "signal_reason": "触发原因",
            "Gate_Reason": "门控原因",
            "gate_reason": "门控原因",
            "Reference_Price": "参考现价",
            "reference_price": "参考现价",
            "close": "收盘价",
            "Market_Session": "会话阶段",
            "session_label": "会话阶段",
            "Regular_Close": "昨收",
            "regular_close": "昨收",
            "Session_Change_Pct": "扩展时段涨跌幅%",
            "session_change_pct": "扩展时段涨跌幅%",
            "daily_return_pct": "当日涨跌幅%",
            "volume_ratio_20d": "20日量比",
            "atr_14": "ATR14",
            "hist_vol_20d": "20日历史波动率",
            "relative_strength": "相对强弱",
            "breakout_valid": "有效突破",
            "fake_breakout": "假突破嫌疑",
            "breakdown": "跌破支撑",
            "expected_return_5d": "5日预期收益",
            "expected_return_10d": "10日预期收益",
            "signal_strength": "信号强度",
            "Entry_Limit_Price": "买入价",
            "Initial_Stop_Loss": "止损价",
            "Take_Profit_Price": "止盈价",
            "stop_loss_pct": "止损幅度",
            "payoff_ratio": "盈亏比",
            "Session_Impact": "会话影响",
            "Price_As_Of": "价格抓取时间",
            "price_as_of": "价格抓取时间",
            "Price_Source": "价格来源",
            "price_source": "价格来源",
            "Latest_News_Publisher": "最新消息来源",
            "latest_news_publisher": "最新消息来源",
            "Latest_News_Channel": "新闻通道",
            "latest_news_channel": "新闻通道",
            "News_Channels": "消息通道汇总",
            "news_channels": "消息通道汇总",
            "Latest_News_Time": "最新消息时间",
            "latest_news_time": "最新消息时间",
            "Latest_News_Age_Hours": "消息距今小时",
            "latest_news_age_hours": "消息距今小时",
            "Latest_Headline": "最新消息标题",
            "latest_headline": "最新消息标题",
            "Target_EV": "目标EV",
            "target_ev": "目标EV",
            "Win_Rate": "胜率",
            "win_rate": "胜率",
            "eligible_for_risk": "允许承担风险",
        }
    )


def _history_to_quote(history: pd.DataFrame, ticker: str) -> dict[str, Any]:
    if history.empty:
        return get_latest_quote(ticker)

    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else latest
    previous_close = float(previous.get("close", latest.get("close", 0.0)))
    close = float(latest.get("close", previous_close))
    change_pct = ((close / previous_close) - 1.0) * 100.0 if previous_close else None
    return {
        "ticker": ticker,
        "close": close,
        "open": float(latest.get("open", close)),
        "high": float(latest.get("high", close)),
        "low": float(latest.get("low", close)),
        "volume": float(latest.get("volume", 0.0)),
        "previous_close": previous_close,
        "change_pct": change_pct,
        "as_of": str(history.index[-1].date()) if isinstance(history.index, pd.DatetimeIndex) else None,
    }


def _build_core_math_inference(
    factor_map: dict[str, dict[str, Any]],
    expected_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    inference: dict[str, dict[str, Any]] = {}
    for ticker, expected in expected_map.items():
        inference[ticker] = {
            "Expected_Return_Log": expected["inference_log"],
            "Signal_Mode": expected.get("signal_label", "观望"),
            "Trade_Gate": "允许" if expected.get("eligible_for_risk") else "观望",
            "Gate_Reason": expected.get("gate_reason", ""),
            "Target_EV": round(float(expected.get("target_ev") or 0.0), 4),
            "Win_Rate": round(float(expected.get("win_rate") or 0.0), 4),
            "Volatility_Penalty": expected["volatility_penalty"],
            "Target_Weight": f"{expected.get('target_weight', 0.0) * 100:.2f}%",
            "RSI_14": round(float(factor_map[ticker].get("rsi_14") or 0.0), 2),
            "ATR_14": round(float(factor_map[ticker].get("atr_14") or 0.0), 2),
        }
    return inference


def _legacy_order_for_ticker(
    ticker: str,
    quote: dict[str, Any],
    expected: dict[str, Any] | None,
    factors: dict[str, Any] | None,
    regime: dict[str, Any],
    observe_mode: bool = False,
    observe_reason: str = "",
    sigma_event: dict[str, Any] | None = None,
    breakdown_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    close = float(quote.get("close") or 0.0)
    session_label = str(quote.get("session_label") or "休市")
    session_change_pct = quote.get("session_change_pct")
    regular_close = quote.get("regular_close") or quote.get("previous_close")
    atr = float((factors or {}).get("atr_14") or 0.0)
    trailing_multiplier = 2.6 if sigma_event and sigma_event.get("is_2sigma") else 2.0
    trailing_stop = close - atr * trailing_multiplier if atr else close * 0.94

    if session_label == "盘前" and session_change_pct is not None and abs(float(session_change_pct)) >= 2.0:
        session_impact = "盘前价格偏离昨收较大，执行价与止损已按盘前价重算。"
    elif session_label == "盘后" and session_change_pct is not None and abs(float(session_change_pct)) >= 2.0:
        session_impact = "盘后波动较大，次日开盘前建议复核。"
    elif session_label == "盘中":
        session_impact = "盘中价格实时变化，执行价为当前会话参考价。"
    else:
        session_impact = "当前主要按常规盘结构参考。"

    if ticker == "AAOX":
        should_liquidate = bool((breakdown_signal or {}).get("breakdown")) or regime["score"] < 0.4
        hard_stop_loss = close * 0.93 if close else None
        return {
            "Ticker": ticker,
            "Signal": "LIQUIDATE" if should_liquidate else "HOLD",
            "Reference_Price": round(close, 2) if close else None,
            "Market_Session": session_label,
            "Regular_Close": round(float(regular_close), 2) if regular_close else None,
            "Session_Change_Pct": round(float(session_change_pct), 2) if session_change_pct is not None else None,
            "Hard_Stop_Loss": round(hard_stop_loss, 2) if hard_stop_loss else None,
            "Session_Impact": session_impact,
            "Price_As_Of": quote.get("as_of"),
            "Price_Source": quote.get("price_source"),
            "Signal_Mode": "风控清仓" if should_liquidate else "观望持有",
            "Gate_Reason": "底层板块破位或市场过弱" if should_liquidate else (observe_reason if observe_mode else "底层未破位"),
        }

    expected_profile = expected or {}
    expected_return_5d = float(expected_profile.get("expected_return_5d") or 0.0)
    signal_mode = expected_profile.get("signal_mode", "neutral")
    if expected_return_5d < -0.015 or regime["score"] < 0.35:
        signal = "REDUCE"
    elif observe_mode or not expected_profile.get("eligible_for_risk"):
        signal = "HOLD"
    elif signal_mode == "momentum" and expected_return_5d > 0.02:
        signal = "ADD"
    else:
        signal = "HOLD"

    return {
        "Ticker": ticker,
        "Signal": signal,
        "Reference_Price": round(close, 2) if close else None,
        "Market_Session": session_label,
        "Regular_Close": round(float(regular_close), 2) if regular_close else None,
        "Session_Change_Pct": round(float(session_change_pct), 2) if session_change_pct is not None else None,
        "Action_Price": round(close, 2) if close else None,
        "Trailing_Stop_Update": round(trailing_stop, 2) if trailing_stop else None,
        "Session_Impact": session_impact,
        "Price_As_Of": quote.get("as_of"),
        "Price_Source": quote.get("price_source"),
        "Signal_Mode": expected_profile.get("signal_label", "观望"),
        "Gate_Reason": observe_reason if observe_mode and signal != "REDUCE" else expected_profile.get("gate_reason", ""),
    }


def _build_new_alpha_targets(
    candidates: list[dict[str, Any]],
    observe_mode: bool,
    observe_reason: str,
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for candidate in candidates[:3]:
        close = float(candidate.get("close") or 0.0)
        reference_price = float(candidate.get("reference_price") or close or 0.0)
        stop_pct = float(candidate.get("stop_loss_pct") or 0.03)
        expected_return_5d = float(candidate.get("expected_return_5d") or 0.0)
        payoff_ratio = float(candidate.get("payoff_ratio") or 0.0)
        entry_buffer = 0.005 if candidate.get("breakout_valid") else 0.01
        entry_limit_price = round(reference_price * (1.0 - entry_buffer), 2) if reference_price else None
        initial_stop_loss = round(entry_limit_price * (1.0 - stop_pct), 2) if entry_limit_price else None
        take_profit_pct = max(expected_return_5d, stop_pct * max(payoff_ratio, 1.0), 0.01)
        take_profit_price = round(entry_limit_price * (1.0 + take_profit_pct), 2) if entry_limit_price else None
        session_label = str(candidate.get("session_label") or "休市")
        session_change_pct = candidate.get("session_change_pct")
        if session_label == "盘前" and session_change_pct is not None and float(session_change_pct) >= 2.5:
            session_impact = "盘前跳空较大，目标价已按盘前价重算，避免直接追高。"
        elif session_label == "盘后" and session_change_pct is not None and abs(float(session_change_pct)) >= 2.0:
            session_impact = "盘后波动明显，次日开盘前需再次确认承接。"
        elif session_label == "盘中":
            session_impact = "盘中价格实时变化，挂单价需预留滑点。"
        else:
            session_impact = "当前主要按常规盘结构参考。"
        signal = "BUY" if not observe_mode and candidate.get("eligible_for_risk") else "IGNORE"
        orders.append(
            {
                "Ticker": candidate["ticker"],
                "Signal": signal,
                "Reference_Price": round(reference_price, 2) if reference_price else None,
                "Market_Session": session_label,
                "Regular_Close": round(float(candidate.get("regular_close") or close or 0.0), 2) if (candidate.get("regular_close") or close) else None,
                "Session_Change_Pct": round(float(session_change_pct), 2) if session_change_pct is not None else None,
                "Entry_Limit_Price": entry_limit_price,
                "Initial_Stop_Loss": initial_stop_loss,
                "Take_Profit_Price": take_profit_price,
                "Session_Impact": session_impact,
                "Price_As_Of": candidate.get("price_as_of"),
                "Price_Source": candidate.get("price_source"),
                "Latest_News_Publisher": candidate.get("latest_news_publisher"),
                "Latest_News_Channel": candidate.get("latest_news_channel"),
                "News_Channels": " + ".join(candidate.get("news_channels") or []),
                "Latest_News_Time": candidate.get("latest_news_time"),
                "Latest_News_Age_Hours": candidate.get("latest_news_age_hours"),
                "Latest_Headline": candidate.get("latest_headline"),
                "Target_EV": round(float(candidate.get("target_ev") or 0.0), 4),
                "Win_Rate": round(float(candidate.get("win_rate") or 0.0), 4),
                "Signal_Mode": candidate.get("signal_label", "观望"),
                "Gate_Reason": observe_reason if observe_mode else candidate.get("gate_reason", ""),
            }
        )
    return orders


def _build_quant_logic_log(
    regime: dict[str, Any],
    qqq_sigma: dict[str, Any],
    nvda_sigma: dict[str, Any],
    aaoi_breakdown: dict[str, Any],
    observe_mode: bool,
    observe_reason: str,
) -> str:
    parts: list[str] = []
    if observe_mode:
        parts.append("观望")
        parts.append(observe_reason)
    parts.extend(regime.get("reasons", [])[:3])
    if qqq_sigma.get("is_2sigma"):
        parts.append("QQQ 2σ")
    if nvda_sigma.get("is_2sigma"):
        parts.append("NVDA 2σ")
    if aaoi_breakdown.get("breakdown"):
        parts.append("AAOI破位")
    return " / ".join(parts) or "量价与波动未触发极端阈值"


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def run_analysis_pipeline(holding_tickers: tuple[str, ...], kelly_fraction: float) -> dict[str, Any]:
    tracked_tickers = sorted(set(CORE_SIGNAL_TICKERS + REGIME_SUPPORT_TICKERS + list(holding_tickers) + [AAOX_UNDERLYING]))
    histories = batch_history(tracked_tickers, period="6mo", interval="1d")
    qqq_history = histories.get("QQQ", pd.DataFrame())

    factor_map: dict[str, dict[str, Any]] = {}
    news_map: dict[str, dict[str, Any]] = {}
    for ticker in tracked_tickers:
        history = histories.get(ticker, pd.DataFrame())
        benchmark_history = None if ticker == "QQQ" else qqq_history
        factor_map[ticker] = build_factor_snapshot(history, benchmark_history)
        news_map[ticker] = build_news_features(get_news_items(ticker, limit=8))

    macro_snapshot = get_macro_snapshot()
    qqq_sigma = detect_sigma_event(histories.get("QQQ", pd.DataFrame()))
    nvda_sigma = detect_sigma_event(histories.get("NVDA", pd.DataFrame()))
    aaoi_breakdown = detect_breakdown_signal(histories.get(AAOX_UNDERLYING, pd.DataFrame()))
    regime = classify_market_regime(
        macro_snapshot,
        factor_map.get("QQQ", {}),
        factor_map.get("NVDA", {}),
        aaoi_breakdown,
        news_map.get("NVDA", {}),
    )

    core_signal_profiles: dict[str, dict[str, Any]] = {}
    for ticker in CORE_SIGNAL_TICKERS:
        core_signal_profiles[ticker] = build_trade_profile(
            ticker,
            factor_map.get(ticker, {}),
            news_map.get(ticker, {}),
            regime["score"],
        )

    decision_tickers = sorted({ticker for ticker in holding_tickers if ticker and ticker != "AAOX"})
    if "AAOX" in holding_tickers:
        decision_tickers.append("AAOI")
    holding_profiles: dict[str, dict[str, Any]] = {}
    for ticker in sorted(set(decision_tickers)):
        holding_profiles[ticker] = build_trade_profile(
            ticker,
            factor_map.get(ticker, {}),
            news_map.get(ticker, {}),
            regime["score"],
        )

    options_summary = get_near_term_options_summary(OPTIONS_TICKER)
    radar_candidates = [
        build_candidate_trade_profile(candidate, regime["score"])
        for candidate in scan_breakout_candidates(RADAR_UNIVERSE)
    ]
    for candidate in radar_candidates:
        ticker = candidate["ticker"]
        session_quote = get_market_session_quote(ticker)
        candidate_news = build_news_features(get_news_items(ticker, limit=5))
        candidate.update(
            {
                "reference_price": session_quote.get("reference_price") or candidate.get("close"),
                "regular_close": session_quote.get("regular_close") or candidate.get("close"),
                "session_label": session_quote.get("session_label"),
                "session_change_pct": session_quote.get("session_change_pct"),
                "price_as_of": session_quote.get("as_of"),
                "price_source": session_quote.get("price_source"),
                "latest_news_publisher": candidate_news.get("latest_publisher"),
                "latest_news_channel": candidate_news.get("latest_source_channel"),
                "news_channels": candidate_news.get("news_channels"),
                "latest_news_time": candidate_news.get("latest_published_at"),
                "latest_news_age_hours": candidate_news.get("latest_age_hours"),
                "latest_headline": candidate_news.get("latest_headline"),
            }
        )
    trade_reference_profiles = {**core_signal_profiles, **holding_profiles}
    observe_mode, observe_reason = should_force_observe(trade_reference_profiles, radar_candidates, regime["score"])

    daily_returns = {}
    for ticker in sorted(set(CORE_SIGNAL_TICKERS + list(holding_profiles.keys()))):
        history = histories.get(ticker, pd.DataFrame())
        if not history.empty:
            daily_returns[ticker] = history["close"].pct_change()
    returns_frame = pd.DataFrame(daily_returns).dropna(how="any")
    kelly_inputs = {}
    if not observe_mode:
        kelly_inputs = {
            ticker: trade_reference_profiles[ticker]["expected_return_daily"]
            for ticker in trade_reference_profiles
            if trade_reference_profiles[ticker].get("eligible_for_risk")
        }
    kelly_result = continuous_kelly(
        kelly_inputs,
        returns_frame,
        risk_free_rate=RISK_FREE_RATE,
        kelly_fraction=kelly_fraction,
    )
    for ticker in trade_reference_profiles:
        trade_reference_profiles[ticker]["target_weight"] = 0.0
    for ticker, weight in kelly_result["weights"].items():
        trade_reference_profiles[ticker]["target_weight"] = weight

    quote_map: dict[str, dict[str, Any]] = {}
    for ticker in tracked_tickers:
        history = histories.get(ticker, pd.DataFrame())
        quote_map[ticker] = {
            **_history_to_quote(history, ticker),
            **get_market_session_quote(ticker),
        }

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "macro_snapshot": macro_snapshot,
        "factor_map": factor_map,
        "news_map": news_map,
        "core_signal_profiles": core_signal_profiles,
        "holding_profiles": holding_profiles,
        "regime": regime,
        "sigma_events": {
            "QQQ": qqq_sigma,
            "NVDA": nvda_sigma,
        },
        "aaoi_breakdown": aaoi_breakdown,
        "kelly": kelly_result,
        "options_summary": options_summary,
        "radar_candidates": radar_candidates,
        "observe_mode": observe_mode,
        "observe_reason": observe_reason,
        "quote_map": quote_map,
    }


st.title("美股量化半自动助手")
st.caption("本地采集公开数据，生成信号与报告。所有下单由你自行决定。")

saved_holdings = load_holdings(HOLDINGS_FILE)
with st.sidebar:
    st.subheader("持仓录入")
    current_editor_holdings = _render_sidebar_holding_editor(saved_holdings)
    if st.button("保存持仓", use_container_width=True):
        normalized = save_holdings(current_editor_holdings, HOLDINGS_FILE)
        st.session_state["saved_holdings"] = normalized
        st.success("持仓已保存")

    st.caption("持仓会保存到 portfolio/holdings.json")

    kelly_fraction = st.slider(
        "凯利折扣",
        min_value=0.1,
        max_value=1.0,
        value=float(DEFAULT_KELLY_FRACTION),
        step=0.05,
        help="建议先用半凯利，避免高波动环境过度暴露。",
    )

run_clicked = st.button("运行分析", type="primary", use_container_width=True)

holdings = current_editor_holdings
holding_tickers = tuple(sorted({item["ticker"] for item in holdings if item.get("ticker")}))

if run_clicked:
    with st.spinner("正在抓取行情、新闻、期权和雷达数据..."):
        analysis = run_analysis_pipeline(holding_tickers, kelly_fraction)

    quote_map = analysis["quote_map"]
    holdings_enriched = enrich_holdings_with_quotes(holdings, quote_map)
    core_signal_profiles = analysis["core_signal_profiles"]
    holding_profiles = analysis["holding_profiles"]
    factor_map = analysis["factor_map"]
    sigma_events = analysis["sigma_events"]
    aaoi_breakdown = analysis["aaoi_breakdown"]
    regime = analysis["regime"]
    observe_mode = bool(analysis.get("observe_mode"))
    observe_reason = str(analysis.get("observe_reason") or "")

    legacy_orders: list[dict[str, Any]] = []
    for holding in holdings_enriched:
        ticker = holding["ticker"]
        if ticker == "AAOX":
            legacy_orders.append(
                _legacy_order_for_ticker(
                    ticker,
                    quote_map.get(ticker, get_latest_quote(ticker)),
                    holding_profiles.get("AAOI"),
                    factor_map.get("AAOI"),
                    regime,
                    observe_mode,
                    observe_reason,
                    breakdown_signal=aaoi_breakdown,
                )
            )
        else:
            legacy_orders.append(
                _legacy_order_for_ticker(
                    ticker,
                    quote_map.get(ticker, {}),
                    holding_profiles.get(ticker),
                    factor_map.get(ticker),
                    regime,
                    observe_mode,
                    observe_reason,
                    sigma_events.get(ticker),
                )
            )

    new_alpha_targets = _build_new_alpha_targets(analysis["radar_candidates"], observe_mode, observe_reason)
    mathematical_inference = _build_core_math_inference(factor_map, core_signal_profiles)
    quant_logic_log = _build_quant_logic_log(
        regime,
        sigma_events.get("QQQ", {}),
        sigma_events.get("NVDA", {}),
        aaoi_breakdown,
        observe_mode,
        observe_reason,
    )

    payload = build_execution_payload(
        analysis_date=datetime.now().strftime("%Y-%m-%d"),
        regime=regime,
        macro_snapshot=analysis["macro_snapshot"],
        mathematical_inference=mathematical_inference,
        legacy_orders=legacy_orders,
        new_alpha_targets=new_alpha_targets,
        kelly_weights=analysis["kelly"]["weights"],
        quant_logic_log=quant_logic_log,
    )
    markdown = build_markdown_report(payload)

    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("市场状态", regime["label"])
    top_mid.metric("风险偏好分数", f"{regime['score']:.2f}")
    top_right.metric("总建议风险敞口", f"{analysis['kelly']['total_exposure'] * 100:.1f}%")

    if observe_mode:
        st.warning(f"当前触发空仓/观望强约束：{observe_reason}")

    overview_tab, holdings_tab, radar_tab, report_tab = st.tabs(["核心信号", "持仓盈亏", "异动雷达", "报告与JSON"])

    with overview_tab:
        st.subheader("核心指数信号")
        st.caption("这里默认用 SPY、QQQ、DIA、IWM 判断大盘与风格状态；个股信号请看持仓页。")
        st.dataframe(pd.DataFrame.from_dict(mathematical_inference, orient="index"), use_container_width=True)
        st.subheader("期权流")
        st.json(analysis["options_summary"])

    with holdings_tab:
        st.subheader("持仓市值")
        if holdings_enriched:
            holdings_frame = _display_holdings_frame(holdings_enriched)
            st.dataframe(holdings_frame, use_container_width=True)
        else:
            st.info("先在左侧录入持仓。")
        if legacy_orders:
            st.subheader("旧仓位执行单")
            st.dataframe(_display_orders_frame(legacy_orders), use_container_width=True)

    with radar_tab:
        st.subheader("全市场异动雷达")
        st.dataframe(_display_orders_frame(analysis["radar_candidates"]), use_container_width=True)
        st.subheader("新 Alpha 目标")
        st.dataframe(_display_orders_frame(new_alpha_targets), use_container_width=True)

    with report_tab:
        st.subheader("Markdown 报告预览")
        st.code(markdown, language="markdown")
        st.subheader("JSON 执行单")
        st.json(payload)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("保存 Markdown + JSON 到 snapshots"):
                bundle = save_report_bundle(payload, markdown)
                st.success(f"已保存: {bundle['markdown_path']} 和 {bundle['json_path']}")
        with col2:
            st.download_button(
                "下载 JSON",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
else:
    st.info("左侧录入持仓后，点击“运行分析”开始抓取数据。")