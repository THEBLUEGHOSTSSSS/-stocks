from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    AAOX_UNDERLYING,
    ACCOUNT_FILE,
    CACHE_TTL_SECONDS,
    DEFAULT_LONG_TARGET_PCT,
    CORE_SIGNAL_TICKERS,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_HOLDING_TICKER,
    DEFAULT_SHORT_MARGIN_RATIO,
    DEFAULT_SHORT_TARGET_PCT,
    HOLDINGS_FILE,
    HOLDINGS_HISTORY_FILE,
    OPTIONS_TICKER,
    RADAR_UNIVERSE,
    REGIME_SUPPORT_TICKERS,
    RISK_FREE_RATE,
    SHORT_MARGIN_CONFIG,
    TICKER_METADATA,
    TICKER_SUGGESTION_UNIVERSE,
)
from data.fetcher import (
    batch_history,
    collect_fetch_notices,
    collect_fetch_warnings,
    get_short_borrow_metrics,
    get_latest_quote,
    get_market_session_quote,
    get_news_items,
    reset_fetch_warnings,
)
from data.indicators import build_factor_snapshot
from data.macro import get_macro_snapshot
from data.options import get_near_term_options_summary
from data.radar import scan_breakout_candidates
from data.sentiment import build_news_features
from models.kelly import continuous_kelly
from models.regime import classify_market_regime, detect_breakdown_signal, detect_sigma_event
from models.signals import calculate_rolling_alpha, build_candidate_trade_profile, build_trade_profile, should_force_observe
from portfolio.account import compute_account_overview, compute_short_margin_profile, load_account_state, save_account_state
from portfolio.holdings import (
    build_holdings_change_records,
    build_holdings_history_summary,
    build_snapshot_order_action_records,
    enrich_holdings_with_quotes,
    frame_to_holdings,
    load_holdings,
    load_holdings_history,
    resolve_reference_snapshot,
    save_holdings,
    save_holdings_snapshot,
)
from reports.generator import build_execution_payload, build_markdown_report, save_report_bundle
from walk_forward_validation import (
    WalkForwardValidator,
    build_roll_forward_path_specs,
    load_project_panel_data,
    run_roll_forward_factor_study,
)


st.set_page_config(page_title="美股量化半自动助手", layout="wide")


SIDE_LABELS = {
    "LONG": "多头",
    "SHORT": "空头",
    "FLAT": "空仓",
}

REGIME_BILINGUAL_LABELS = {
    "Risk-On": "风险偏好 / Risk-On",
    "Risk-Off": "风险规避 / Risk-Off",
    "Choppy": "震荡市 / Choppy",
    "Range": "区间整理 / Range",
}

SIGNAL_LABELS = {
    "ADD": "加仓",
    "ADD_SHORT": "加空",
    "HOLD": "持有",
    "HOLD_SHORT": "持有空单",
    "REDUCE": "减仓",
    "LIQUIDATE": "清仓",
    "COVER": "回补",
}

HOLDINGS_SNAPSHOT_SOURCE_LABELS = {
    "manual_save": "手动保存",
    "analysis_run": "运行分析",
    "unknown": "未知来源",
}

FACTOR_SET_DISPLAY_LABELS = {
    "baseline_log_return_5d": "旧基线 / 5日动量",
    "pruned_no_short_momentum": "旧裁剪版 / 无短周期动量",
    "momentum_10d": "10日动量",
    "momentum_20d": "20日动量",
    "momentum_60d": "60日动量",
    "volatility_adjusted_momentum_20d": "旧波动率调整 / 20日动量",
    "momentum_20d_plus_vol_adjusted": "旧复合 / 20日动量+波动率调整",
    "momentum_60d_plus_vol_adjusted": "旧复合 / 60日动量+波动率调整",
    "core_without_momentum": "无动量核心",
    "momentum_10d_core": "10日动量核心",
    "momentum_60d_shadow": "60日动量对照",
}


def _bool_display(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return "是" if bool(value) else "否"


def _format_regime_label(label: Any) -> str:
    normalized = str(label or "").strip()
    if not normalized:
        return ""
    return REGIME_BILINGUAL_LABELS.get(normalized, normalized)


def _format_factor_set_label(label: Any) -> str:
    normalized = str(label or "").strip()
    if not normalized:
        return ""
    return FACTOR_SET_DISPLAY_LABELS.get(normalized, normalized)


def _format_factor_set_frame(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in ("factor_set", "candidate_factor_set", "reference_factor_set", "best_factor_set"):
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_factor_set_label)
    return formatted


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
        "side": "LONG",
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
                "side": str(row.get("side", "LONG") or "LONG"),
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
                "side": st.session_state.get(f"holding_side_{row_id}", str(row.get("side", "LONG") or "LONG")),
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
        side_key = f"holding_side_{row_id}"
        shares_key = f"holding_shares_{row_id}"
        cost_key = f"holding_cost_{row_id}"
        note_key = f"holding_notes_{row_id}"
        auto_note_key = f"holding_auto_note_{row_id}"
        suggestion_key = f"holding_suggestion_{row_id}"

        st.session_state.setdefault(ticker_key, row.get("ticker", ""))
        st.session_state.setdefault(side_key, str(row.get("side", "LONG") or "LONG"))
        st.session_state.setdefault(shares_key, float(row.get("shares", 0.0) or 0.0))
        st.session_state.setdefault(cost_key, float(row.get("cost_basis", 0.0) or 0.0))
        st.session_state.setdefault(note_key, row.get("notes", ""))

        with st.container(border=True):
            st.markdown(f"**持仓 {index}**")
            top_left, top_mid, top_right = st.columns([1.5, 1.0, 1.0])
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
            with top_mid:
                st.selectbox(
                    "方向",
                    options=["LONG", "SHORT"],
                    format_func=lambda value: SIDE_LABELS.get(value, value),
                    key=side_key,
                )
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


def _render_sidebar_account_editor(saved_account: dict[str, Any]) -> dict[str, float]:
    st.subheader("账户资金")
    total_equity = st.number_input(
        "账户总权益",
        min_value=0.0,
        value=float(saved_account.get("total_equity", 0.0) or 0.0),
        step=1000.0,
        format="%.2f",
        help="如果不填，系统会按当前持仓占用资金估算总权益，闲余资金默认为 0。",
    )
    short_margin_ratio = st.slider(
        "空头基准保证金折算",
        min_value=0.2,
        max_value=1.0,
        value=float(saved_account.get("short_margin_ratio", DEFAULT_SHORT_MARGIN_RATIO) or DEFAULT_SHORT_MARGIN_RATIO),
        step=0.05,
        help="Reg T 初始保证金至少按 50% 起算；实际空头占用会在此基础上叠加借券费率与 HTB 约束。",
    )
    max_long_position_pct = st.slider(
        "单笔多头目标仓位上限",
        min_value=0.02,
        max_value=0.3,
        value=float(saved_account.get("max_long_position_pct", DEFAULT_LONG_TARGET_PCT) or DEFAULT_LONG_TARGET_PCT),
        step=0.01,
        help="建议挂单不会超过总权益的这一比例。",
    )
    max_short_position_pct = st.slider(
        "单笔空头目标仓位上限",
        min_value=0.02,
        max_value=0.2,
        value=float(saved_account.get("max_short_position_pct", DEFAULT_SHORT_TARGET_PCT) or DEFAULT_SHORT_TARGET_PCT),
        step=0.01,
        help="空头默认比多头更保守。",
    )
    return {
        "total_equity": total_equity,
        "short_margin_ratio": short_margin_ratio,
        "max_long_position_pct": max_long_position_pct,
        "max_short_position_pct": max_short_position_pct,
    }


def _display_holdings_frame(
    holdings_enriched: list[dict[str, Any]],
    holding_profiles: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if not holdings_enriched:
        return pd.DataFrame()
    frame = pd.DataFrame(holdings_enriched).copy()
    profile_map = holding_profiles or {}
    if profile_map:
        frame["earnings_date"] = frame["ticker"].map(
            lambda ticker: (profile_map.get(AAOX_UNDERLYING if str(ticker or "").upper() == "AAOX" else str(ticker or "")) or {}).get("earnings_date")
        )
        frame["days_to_earnings"] = frame["ticker"].map(
            lambda ticker: (profile_map.get(AAOX_UNDERLYING if str(ticker or "").upper() == "AAOX" else str(ticker or "")) or {}).get("days_to_earnings")
        )
        frame["earnings_gate_reason"] = frame["ticker"].map(
            lambda ticker: (profile_map.get(AAOX_UNDERLYING if str(ticker or "").upper() == "AAOX" else str(ticker or "")) or {}).get("earnings_gate_reason")
        )
    if "side" in frame.columns:
        frame["side"] = frame["side"].map(lambda value: SIDE_LABELS.get(str(value).upper(), value))
    return frame.rename(
        columns={
            "ticker": "代码",
            "side": "方向",
            "shares": "股数",
            "cost_basis": "成本价",
            "notes": "备注",
            "market_price": "最新价",
            "market_value": "持仓名义金额",
            "net_exposure": "净敞口",
            "cost_value": "成本市值",
            "pnl": "浮盈亏",
            "pnl_pct": "浮盈亏%",
            "market_session": "会话阶段",
            "session_change_pct": "扩展时段涨跌幅%",
            "quote_as_of": "价格抓取时间",
            "earnings_date": "下次财报日",
            "days_to_earnings": "距财报交易日",
            "earnings_gate_reason": "财报风控",
        }
    )


def _snapshot_source_label(source: Any) -> str:
    return HOLDINGS_SNAPSHOT_SOURCE_LABELS.get(str(source or "unknown"), str(source or "未知来源"))


def _display_holdings_change_frame(change_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not change_records:
        return pd.DataFrame()
    frame = pd.DataFrame(change_records).copy()
    for column in ["previous_side", "current_side"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: SIDE_LABELS.get(str(value).upper(), value) if value else value)
    return frame.rename(
        columns={
            "ticker": "代码",
            "status": "变更类型",
            "previous_side": "上次方向",
            "current_side": "当前方向",
            "previous_shares": "上次股数",
            "current_shares": "当前股数",
            "share_delta": "股数变化",
            "previous_cost_basis": "上次成本价",
            "current_cost_basis": "当前成本价",
            "previous_notes": "上次备注",
            "current_notes": "当前备注",
        }
    )


def _display_holdings_history_frame(history_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not history_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(history_rows).copy()
    if "source" in frame.columns:
        frame["source"] = frame["source"].map(_snapshot_source_label)
    return frame.rename(
        columns={
            "captured_at": "快照时间",
            "source": "记录来源",
            "position_count": "持仓数",
            "tickers": "持仓代码",
        }
    )


def _display_snapshot_order_action_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).copy()
    for column in ["previous_side", "position_side", "trade_direction"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: SIDE_LABELS.get(str(value).upper(), value) if value else value)
    if "signal" in frame.columns:
        frame["signal"] = frame["signal"].map(lambda value: SIGNAL_LABELS.get(str(value).upper(), value) if value else value)
    return frame.rename(
        columns={
            "ticker": "代码",
            "previous_side": "上次方向",
            "previous_shares": "上次股数",
            "previous_cost_basis": "上次成本价",
            "previous_notes": "上次备注",
            "signal": "今日挂单动作",
            "position_side": "当前持仓方向",
            "trade_direction": "挂单方向",
            "suggested_shares": "建议股数",
            "reference_price": "参考价",
            "gate_reason": "门控原因",
            "signal_mode": "信号模式",
        }
    )


def _display_account_overview_frame(account_overview: dict[str, Any]) -> pd.DataFrame:
    money_fields = {
        "total_equity",
        "market_funds",
        "total_pnl",
        "long_market_value",
        "short_market_value",
        "short_margin_used",
        "short_maintenance_requirement",
        "capital_in_use",
        "idle_cash",
        "idle_cash_after_legacy_orders",
        "short_buying_power",
        "net_exposure",
    }
    pct_fields = {
        "max_long_position_pct",
        "max_short_position_pct",
        "base_short_margin_ratio",
        "short_margin_ratio",
        "short_maintenance_ratio",
        "utilization",
    }
    field_labels = [
        ("total_equity", "账户总权益"),
        ("market_funds", "市场资金"),
        ("total_pnl", "总盈利(浮动)"),
        ("long_market_value", "多头敞口"),
        ("short_market_value", "空头敞口"),
        ("short_margin_used", "空头初始保证金占用"),
        ("short_maintenance_requirement", "空头维持担保占用"),
        ("capital_in_use", "总资金占用"),
        ("idle_cash", "当前闲余资金"),
        ("idle_cash_after_legacy_orders", "旧仓执行后闲余资金"),
        ("short_buying_power", "空头剩余可开仓名义金额"),
        ("net_exposure", "净敞口"),
        ("max_long_position_pct", "单笔多头上限"),
        ("max_short_position_pct", "单笔空头上限"),
        ("base_short_margin_ratio", "空头基准保证金折算"),
        ("short_margin_ratio", "空头实际初始保证金折算"),
        ("short_maintenance_ratio", "空头实际维持担保折算"),
        ("utilization", "资金使用率"),
    ]

    rows: list[dict[str, Any]] = []
    for key, label in field_labels:
        if key not in account_overview:
            continue
        raw_value = account_overview.get(key)
        if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
            display_value = None
        elif key in money_fields:
            display_value = f"${float(raw_value):,.2f}"
        elif key in pct_fields:
            display_value = f"{float(raw_value) * 100.0:.2f}%"
        else:
            display_value = raw_value
        rows.append({"项目": label, "数值": display_value})
    return pd.DataFrame(rows)


def _display_borrow_summary_frame(account_overview: dict[str, Any]) -> pd.DataFrame:
    borrow_summary = (account_overview.get("borrow_metrics_summary") or {}).get("tickers") or {}
    if not borrow_summary:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for ticker, summary in borrow_summary.items():
        rows.append(
            {
                "代码": ticker,
                "借券费率%": f"{float(summary.get('fee_pct') or 0.0):.2f}%",
                "可借股数": int(summary.get("available_shares") or 0),
                "难借状态": _bool_display(summary.get("is_hard_to_borrow")),
                "禁空": _bool_display(summary.get("prohibit_short")),
                "初始保证金%": f"{float(summary.get('initial_margin_pct') or 0.0):.2f}%",
                "维持担保%": f"{float(summary.get('maintenance_margin_pct') or 0.0):.2f}%",
                "数据源": summary.get("source"),
                "抓取状态": summary.get("error") or "正常",
            }
        )
    return pd.DataFrame(rows)


def _display_orders_frame(orders: list[dict[str, Any]]) -> pd.DataFrame:
    if not orders:
        return pd.DataFrame()
    frame = pd.DataFrame(orders).copy()

    signal_value_map = {
        "BUY": "买入",
        "SELL_SHORT": "沽空开仓",
        "IGNORE": "忽略",
        "ADD": "加仓",
        "ADD_SHORT": "加空",
        "HOLD": "持有/观望",
        "HOLD_SHORT": "持有空单",
        "REDUCE": "减仓",
        "COVER": "回补",
        "LIQUIDATE": "清仓",
    }
    mode_value_map = {
        "momentum": "动能延续",
        "neutral": "观望",
        "mean_reversion": "均值回归",
        "short_breakdown": "破位沽空",
    }
    allocation_regime_map = {
        "Risk-On": "风险偏好 / Risk-On",
        "Risk-Off": "风险规避 / Risk-Off",
        "Choppy": "震荡市 / Choppy",
        "Range": "区间整理 / Range",
    }
    options_flow_signal_map = {
        "neutral": "中性",
        "gamma_squeeze": "Gamma 挤压",
        "put_divergence": "Put 背离",
        "put_panic_short": "Put 恐慌空头延续",
    }
    expert_value_map = {
        "trend": "趋势专家",
        "reversion": "回归专家",
        "event": "事件专家",
    }
    volatility_penalty_map = {
        "High": "高",
        "Medium": "中",
        "Low": "低",
    }
    channel_value_map = {
        "Google News RSS": "Google 新闻 RSS",
        "Yahoo Finance": "Yahoo Finance 聚合",
        "yfinance.info": "yfinance 实时快照",
        "yfinance.history": "yfinance 历史行情",
        "eastmoney.realtime": "东方财富实时行情",
        "eastmoney.history": "东方财富历史行情",
        "unavailable": "数据不可用",
    }

    if "Signal" in frame.columns:
        frame["Signal"] = frame["Signal"].map(lambda value: signal_value_map.get(value, value))

    for column in ["Signal_Mode", "signal_mode"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: mode_value_map.get(value, value))

    for column in ["allocation_regime"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: allocation_regime_map.get(value, value))

    for column in ["options_flow_signal"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: options_flow_signal_map.get(value, value))

    for column in ["dominant_expert"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: expert_value_map.get(value, value))

    for column in ["volatility_penalty"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: volatility_penalty_map.get(value, value))

    for column in ["Trade_Direction", "trade_direction", "Position_Side", "position_side"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: SIDE_LABELS.get(str(value).upper(), value))

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

    for column in [
        "breakout_valid",
        "fake_breakout",
        "breakdown",
        "eligible_for_risk",
        "earnings_event_block",
        "Earnings_Event_Block",
        "exhaustion_reversal",
        "Is_Hard_To_Borrow",
        "is_hard_to_borrow",
        "Prohibit_Short",
        "prohibit_short",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].map(_bool_display)

    return frame.rename(
        columns={
            "Ticker": "代码",
            "ticker": "代码",
            "Signal": "信号",
            "Trade_Direction": "交易方向",
            "trade_direction": "交易方向",
            "Position_Side": "持仓方向",
            "position_side": "持仓方向",
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
            "Suggested_Shares": "建议股数",
            "Suggested_Notional": "建议名义金额",
            "Capital_Usage": "预计资金占用",
            "Budget_Pct": "建议仓位%",
            "Sizing_Note": "资金约束说明",
            "Borrow_Fee_Pct": "借券费率%",
            "borrow_fee_pct": "借券费率%",
            "Available_Shares": "可借股数",
            "available_shares": "可借股数",
            "Is_Hard_To_Borrow": "难借状态",
            "is_hard_to_borrow": "难借状态",
            "Prohibit_Short": "禁空",
            "prohibit_short": "禁空",
            "Short_Initial_Margin_Pct": "初始保证金%",
            "short_initial_margin_pct": "初始保证金%",
            "Short_Maintenance_Margin_Pct": "维持担保%",
            "short_maintenance_margin_pct": "维持担保%",
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
            "Earnings_Date": "下次财报日",
            "earnings_date": "下次财报日",
            "Days_To_Earnings": "距财报交易日",
            "days_to_earnings": "距财报交易日",
            "Earnings_Risk": "财报风控",
            "earnings_gate_reason": "财报风控",
            "allocation_regime": "配置状态",
            "allocation_reason": "配置说明",
            "call_put_volume_ratio": "Call/Put量比",
            "options_flow_signal": "期权流信号",
            "options_flow_reason": "期权流说明",
            "options_edge": "期权边际贡献",
            "options_confidence_delta": "期权置信度调整",
            "options_reversal_penalty_delta": "期权反转惩罚调整",
            "options_win_rate_multiplier": "期权胜率乘数",
            "options_win_rate_delta": "期权胜率调整",
            "expected_return_5d_raw": "5日原始预期收益",
            "expected_return_daily": "日度预期收益",
            "rolling_alpha_daily": "滚动Alpha(日)",
            "alpha_component_5d": "5日Alpha贡献",
            "market_component": "市场综合因子",
            "base_market_component": "基础市场因子",
            "price_factor_component": "量价因子贡献",
            "news_component": "新闻因子贡献",
            "expert_consensus_blend": "专家融合权重",
            "reversal_penalty": "反转惩罚",
            "uncertainty_penalty": "不确定性惩罚",
            "volatility_penalty": "波动率标签",
            "expert_consensus": "专家共识",
            "expert_disagreement": "专家分歧",
            "dominant_expert": "主导专家",
            "expert_weight_trend": "趋势专家权重",
            "expert_weight_reversion": "回归专家权重",
            "expert_weight_event": "事件专家权重",
            "dynamic_weight_momentum": "动态权重-动能",
            "dynamic_weight_relative_strength": "动态权重-相对强弱",
            "dynamic_weight_breakout": "动态权重-突破",
            "dynamic_weight_rsi_alignment": "动态权重-RSI对齐",
            "dynamic_weight_sentiment": "动态权重-情绪",
            "dynamic_weight_shock": "动态权重-冲击",
            "dynamic_weight_headline": "动态权重-新闻密度",
            "dynamic_weight_volatility": "动态权重-波动率",
            "uncertainty_score": "不确定性分数",
            "predictive_confidence": "预测置信度",
            "prediction_interval_5d": "5日预测区间",
            "expert_disagreement_score": "专家分歧分数",
            "routing_uncertainty": "路由不确定性",
            "crowding_risk": "拥挤风险",
            "vix_value": "VIX",
            "exhaustion_reversal": "过热反转",
            "exhaustion_reason": "过热反转说明",
            "inference_log": "推理日志",
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


def _rolling_alpha_from_history(history: pd.DataFrame, benchmark_history: pd.DataFrame) -> float:
    if history.empty or benchmark_history.empty:
        return 0.0
    if "close" not in history or "close" not in benchmark_history:
        return 0.0
    return calculate_rolling_alpha(history["close"].pct_change(), benchmark_history["close"].pct_change())


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


def _allocation_strength(expected_return_5d: float, win_rate: float) -> float:
    edge_strength = float(np.clip(abs(expected_return_5d) / 0.08, 0.35, 1.0))
    quality_strength = float(np.clip(win_rate / 0.65, 0.45, 1.0))
    return (edge_strength + quality_strength) / 2.0


def _short_borrow_block_message(borrow_metrics: dict[str, Any] | None) -> str:
    metrics = borrow_metrics or {}
    fee_pct = float(metrics.get("latest_fee_pct") or 0.0)
    available_shares = int(float(metrics.get("latest_available_shares") or 0.0))
    reasons: list[str] = []
    if available_shares <= 0:
        reasons.append("当前无可借股数")
    if fee_pct >= float(SHORT_MARGIN_CONFIG["max_annual_borrow_fee_pct"]):
        reasons.append(
            f"借券费率 {fee_pct:.2f}%/年 超过 {float(SHORT_MARGIN_CONFIG['max_annual_borrow_fee_pct']):.0f}%/年 上限"
        )
    if not reasons:
        reasons.append("借券条件不满足")
    return "；".join(reasons) + "，禁止新增空单。"


def _short_borrow_order_fields(
    borrow_metrics: dict[str, Any] | None,
    account_overview: dict[str, Any],
) -> dict[str, Any]:
    if not borrow_metrics:
        return {
            "Borrow_Fee_Pct": None,
            "Available_Shares": None,
            "Is_Hard_To_Borrow": None,
            "Prohibit_Short": None,
            "Short_Initial_Margin_Pct": None,
            "Short_Maintenance_Margin_Pct": None,
        }

    margin_profile = compute_short_margin_profile(
        borrow_metrics,
        base_margin_ratio=float(
            account_overview.get("base_short_margin_ratio")
            or account_overview.get("short_margin_ratio")
            or DEFAULT_SHORT_MARGIN_RATIO
        ),
    )
    available_shares = int(float(borrow_metrics.get("latest_available_shares") or 0.0))
    return {
        "Borrow_Fee_Pct": round(float(margin_profile["borrow_fee_pct"]), 2),
        "Available_Shares": available_shares,
        "Is_Hard_To_Borrow": bool(borrow_metrics.get("is_hard_to_borrow")),
        "Prohibit_Short": bool(borrow_metrics.get("prohibit_short")),
        "Short_Initial_Margin_Pct": round(float(margin_profile["initial_margin_pct"]) * 100.0, 2),
        "Short_Maintenance_Margin_Pct": round(float(margin_profile["maintenance_margin_pct"]) * 100.0, 2),
    }


def _suggest_order_budget(
    reference_price: float,
    trade_direction: str,
    expected_return_5d: float,
    win_rate: float,
    account_overview: dict[str, Any],
    available_cash: float,
    current_position_notional: float = 0.0,
    borrow_metrics: dict[str, Any] | None = None,
) -> tuple[int, float, float, float, str]:
    if reference_price <= 0.0 or available_cash <= 0.0:
        return 0, 0.0, 0.0, available_cash, "闲余资金不足，暂不建议新增挂单。"

    strength = _allocation_strength(expected_return_5d, win_rate)
    if trade_direction == "SHORT":
        if (borrow_metrics or {}).get("prohibit_short"):
            return 0, 0.0, 0.0, available_cash, _short_borrow_block_message(borrow_metrics)

        base_pct = float(account_overview.get("max_short_position_pct") or DEFAULT_SHORT_TARGET_PCT)
        margin_profile = compute_short_margin_profile(
            borrow_metrics,
            base_margin_ratio=float(
                account_overview.get("base_short_margin_ratio")
                or account_overview.get("short_margin_ratio")
                or DEFAULT_SHORT_MARGIN_RATIO
            ),
        )
        margin_ratio = float(margin_profile["initial_margin_pct"])
        maintenance_ratio = float(margin_profile["maintenance_margin_pct"])
        borrow_fee_pct = float(margin_profile["borrow_fee_pct"])
        target_notional = float(account_overview.get("total_equity") or 0.0) * base_pct * strength
        sizing_parts = [
            f"按空头仓位上限与动态保证金约束计算，初始保证金 {margin_ratio * 100.0:.1f}% ，维持担保 {maintenance_ratio * 100.0:.1f}%。"
        ]
        if borrow_fee_pct > 0.0:
            sizing_parts.append(f"借券费率 {borrow_fee_pct:.2f}%/年。")
        if (borrow_metrics or {}).get("is_hard_to_borrow"):
            target_notional *= 1.0 - float(SHORT_MARGIN_CONFIG["htb_size_haircut_pct"])
            sizing_parts.append(
                f"HTB 标的，目标仓位额外收缩 {float(SHORT_MARGIN_CONFIG['htb_size_haircut_pct']) * 100.0:.0f}%。"
            )
        incremental_notional = max(target_notional - current_position_notional, 0.0)
        max_notional_from_cash = available_cash / margin_ratio if margin_ratio else 0.0
        suggested_notional = min(incremental_notional, max_notional_from_cash)
        capital_usage = suggested_notional * margin_ratio
        sizing_note = " ".join(sizing_parts)
    else:
        base_pct = float(account_overview.get("max_long_position_pct") or DEFAULT_LONG_TARGET_PCT)
        target_notional = float(account_overview.get("total_equity") or 0.0) * base_pct * strength
        incremental_notional = max(target_notional - current_position_notional, 0.0)
        suggested_notional = min(incremental_notional, available_cash)
        capital_usage = suggested_notional
        sizing_note = "按多头仓位上限与闲余资金约束计算。"

    suggested_shares = int(suggested_notional // reference_price)
    if trade_direction == "SHORT":
        available_shares = int(float((borrow_metrics or {}).get("latest_available_shares") or 0.0))
        if available_shares > 0:
            suggested_shares = min(suggested_shares, available_shares)
    if suggested_shares <= 0:
        return 0, 0.0, 0.0, available_cash, "预算不足以形成最小 1 股挂单。"

    suggested_notional = suggested_shares * reference_price
    if trade_direction == "SHORT":
        capital_usage = suggested_notional * margin_ratio
    else:
        capital_usage = suggested_notional
    remaining_cash = max(available_cash - capital_usage, 0.0)
    return suggested_shares, suggested_notional, capital_usage, remaining_cash, sizing_note


def _legacy_order_for_ticker(
    holding: dict[str, Any],
    ticker: str,
    quote: dict[str, Any],
    expected: dict[str, Any] | None,
    factors: dict[str, Any] | None,
    account_overview: dict[str, Any],
    available_cash: float,
    regime: dict[str, Any],
    observe_mode: bool = False,
    observe_reason: str = "",
    sigma_event: dict[str, Any] | None = None,
    breakdown_signal: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    expected_profile = expected or {}
    close = float(quote.get("close") or 0.0)
    session_label = str(quote.get("session_label") or "休市")
    session_change_pct = quote.get("session_change_pct")
    regular_close = quote.get("regular_close") or quote.get("previous_close")
    atr = float((factors or {}).get("atr_14") or 0.0)
    trailing_multiplier = 2.6 if sigma_event and sigma_event.get("is_2sigma") else 2.0
    trailing_stop = close - atr * trailing_multiplier if atr else close * 0.94
    position_side = str(holding.get("side") or "LONG").upper()
    position_notional = float(holding.get("market_value") or 0.0)
    shares = float(holding.get("shares") or 0.0)

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
            "Position_Side": position_side,
            "Trade_Direction": position_side,
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
            "Earnings_Date": expected_profile.get("earnings_date"),
            "Days_To_Earnings": expected_profile.get("days_to_earnings"),
            "Earnings_Risk": expected_profile.get("earnings_gate_reason"),
        }, available_cash

    expected_return_5d = float(expected_profile.get("expected_return_5d") or 0.0)
    signal_mode = expected_profile.get("signal_mode", "neutral")
    earnings_event_block = bool(expected_profile.get("earnings_event_block"))
    borrow_metrics = get_short_borrow_metrics(ticker) if position_side == "SHORT" else None

    if position_side == "SHORT":
        if earnings_event_block:
            signal = "COVER"
        elif signal_mode == "short_breakdown" and not observe_mode and expected_profile.get("eligible_for_risk"):
            signal = "ADD_SHORT"
        elif expected_return_5d > 0.015 or regime["score"] > 0.55 or signal_mode in {"momentum", "mean_reversion"}:
            signal = "COVER"
        else:
            signal = "HOLD_SHORT"
    else:
        if earnings_event_block:
            signal = "LIQUIDATE"
        elif expected_return_5d < -0.015 or regime["score"] < 0.35:
            signal = "REDUCE"
        elif observe_mode or not expected_profile.get("eligible_for_risk"):
            signal = "HOLD"
        elif signal_mode == "momentum" and expected_return_5d > 0.02:
            signal = "ADD"
        else:
            signal = "HOLD"

    suggested_shares = 0
    suggested_notional = 0.0
    capital_usage = 0.0
    sizing_note = "当前以风控和跟踪为主，不建议新增资金。"
    remaining_cash = available_cash
    if signal in {"ADD", "ADD_SHORT"} and close > 0.0:
        trade_direction = "SHORT" if signal == "ADD_SHORT" else "LONG"
        if trade_direction == "SHORT" and (borrow_metrics or {}).get("prohibit_short"):
            signal = "HOLD_SHORT"
            sizing_note = _short_borrow_block_message(borrow_metrics)
        else:
            suggested_shares, suggested_notional, capital_usage, remaining_cash, sizing_note = _suggest_order_budget(
                close,
                trade_direction,
                expected_return_5d,
                float(expected_profile.get("win_rate") or 0.0),
                account_overview,
                available_cash,
                current_position_notional=position_notional,
                borrow_metrics=borrow_metrics if trade_direction == "SHORT" else None,
            )
            if suggested_shares <= 0:
                signal = "HOLD_SHORT" if signal == "ADD_SHORT" else "HOLD"
    elif signal == "REDUCE":
        suggested_shares = max(int(np.ceil(shares * 0.5)), 1) if shares > 0 else 0
        sizing_note = "默认建议先减半，保留观察仓。"
    elif signal in {"COVER", "LIQUIDATE"}:
        suggested_shares = int(np.ceil(shares)) if shares > 0 else 0
        sizing_note = "默认建议回补/平掉当前全部仓位。"

    gate_reason = observe_reason if observe_mode and signal not in {"REDUCE", "LIQUIDATE", "COVER"} else expected_profile.get("gate_reason", "")
    if earnings_event_block:
        gate_reason = str(expected_profile.get("earnings_gate_reason") or expected_profile.get("gate_reason") or "财报高危期，强制空仓")
    if position_side == "SHORT" and borrow_metrics:
        if borrow_metrics.get("prohibit_short") and signal != "COVER":
            gate_reason = _short_borrow_block_message(borrow_metrics)
        elif borrow_metrics.get("is_hard_to_borrow"):
            htb_reason = f"HTB：借券费率 {float(borrow_metrics.get('latest_fee_pct') or 0.0):.2f}%/年"
            gate_reason = f"{gate_reason} / {htb_reason}" if gate_reason else htb_reason

    return {
        "Ticker": ticker,
        "Signal": signal,
        "Position_Side": position_side,
        "Trade_Direction": "SHORT" if signal in {"ADD_SHORT", "COVER", "HOLD_SHORT"} else "LONG",
        "Reference_Price": round(close, 2) if close else None,
        "Market_Session": session_label,
        "Regular_Close": round(float(regular_close), 2) if regular_close else None,
        "Session_Change_Pct": round(float(session_change_pct), 2) if session_change_pct is not None else None,
        "Action_Price": round(close, 2) if close else None,
        "Trailing_Stop_Update": round(trailing_stop, 2) if trailing_stop else None,
        "Suggested_Shares": suggested_shares,
        "Suggested_Notional": round(suggested_notional, 2) if suggested_notional else 0.0,
        "Capital_Usage": round(capital_usage, 2) if capital_usage else 0.0,
        "Budget_Pct": round((suggested_notional / float(account_overview.get("total_equity") or 1.0)) * 100.0, 2) if suggested_notional and account_overview.get("total_equity") else 0.0,
        "Sizing_Note": sizing_note,
        "Session_Impact": session_impact,
        "Price_As_Of": quote.get("as_of"),
        "Price_Source": quote.get("price_source"),
        "Signal_Mode": expected_profile.get("signal_label", "观望"),
        "Gate_Reason": gate_reason,
        "Earnings_Date": expected_profile.get("earnings_date"),
        "Days_To_Earnings": expected_profile.get("days_to_earnings"),
        "Earnings_Risk": expected_profile.get("earnings_gate_reason"),
        "allocation_regime": expected_profile.get("allocation_regime"),
        **_short_borrow_order_fields(borrow_metrics, account_overview),
    }, remaining_cash


def _build_new_alpha_targets(
    candidates: list[dict[str, Any]],
    account_overview: dict[str, Any],
    observe_mode: bool,
    observe_reason: str,
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    remaining_cash = float(account_overview.get("idle_cash") or 0.0)
    for candidate in candidates[:5]:
        close = float(candidate.get("close") or 0.0)
        reference_price = float(candidate.get("reference_price") or close or 0.0)
        stop_pct = float(candidate.get("stop_loss_pct") or 0.03)
        expected_return_5d = float(candidate.get("expected_return_5d") or 0.0)
        payoff_ratio = float(candidate.get("payoff_ratio") or 0.0)
        trade_direction = str(candidate.get("trade_direction") or ("SHORT" if candidate.get("breakdown") else "LONG")).upper()
        borrow_metrics = get_short_borrow_metrics(candidate["ticker"]) if trade_direction == "SHORT" else None
        entry_buffer = 0.005 if candidate.get("breakout_valid") or candidate.get("breakdown") else 0.01
        if trade_direction == "SHORT":
            entry_limit_price = round(reference_price * (1.0 + entry_buffer), 2) if reference_price else None
            initial_stop_loss = round(entry_limit_price * (1.0 + stop_pct), 2) if entry_limit_price else None
        else:
            entry_limit_price = round(reference_price * (1.0 - entry_buffer), 2) if reference_price else None
            initial_stop_loss = round(entry_limit_price * (1.0 - stop_pct), 2) if entry_limit_price else None
        take_profit_pct = max(expected_return_5d, stop_pct * max(payoff_ratio, 1.0), 0.01)
        if trade_direction == "SHORT":
            take_profit_price = round(entry_limit_price * (1.0 - max(abs(take_profit_pct), 0.01)), 2) if entry_limit_price else None
        else:
            take_profit_price = round(entry_limit_price * (1.0 + take_profit_pct), 2) if entry_limit_price else None
        session_label = str(candidate.get("session_label") or "休市")
        session_change_pct = candidate.get("session_change_pct")
        if session_label == "盘前" and session_change_pct is not None and float(session_change_pct) >= 2.5:
            session_impact = "盘前跳空较大，目标价已按盘前价重算，避免直接追高。"
        elif session_label == "盘前" and session_change_pct is not None and float(session_change_pct) <= -2.5:
            session_impact = "盘前向下跳空较大，空头挂单价已按盘前价重算，避免追空过度。"
        elif session_label == "盘后" and session_change_pct is not None and abs(float(session_change_pct)) >= 2.0:
            session_impact = "盘后波动明显，次日开盘前需再次确认承接。"
        elif session_label == "盘中":
            session_impact = "盘中价格实时变化，挂单价需预留滑点。"
        else:
            session_impact = "当前主要按常规盘结构参考。"

        signal = "IGNORE"
        suggested_shares = 0
        suggested_notional = 0.0
        capital_usage = 0.0
        sizing_note = "当前不建议新开仓。"
        gate_reason = observe_reason if observe_mode else str(candidate.get("gate_reason") or "")
        if borrow_metrics and borrow_metrics.get("prohibit_short"):
            block_reason = _short_borrow_block_message(borrow_metrics)
            gate_reason = f"{gate_reason} / {block_reason}" if gate_reason and not observe_mode else block_reason if not observe_mode else f"{gate_reason} / {block_reason}"
        elif borrow_metrics and borrow_metrics.get("is_hard_to_borrow"):
            htb_reason = f"HTB：借券费率 {float(borrow_metrics.get('latest_fee_pct') or 0.0):.2f}%/年"
            gate_reason = f"{gate_reason} / {htb_reason}" if gate_reason else htb_reason
        if not observe_mode and candidate.get("eligible_for_risk") and entry_limit_price:
            if trade_direction == "SHORT" and (borrow_metrics or {}).get("prohibit_short"):
                signal = "IGNORE"
                sizing_note = _short_borrow_block_message(borrow_metrics)
            else:
                signal = "SELL_SHORT" if trade_direction == "SHORT" else "BUY"
                suggested_shares, suggested_notional, capital_usage, remaining_cash, sizing_note = _suggest_order_budget(
                    float(entry_limit_price),
                    trade_direction,
                    expected_return_5d,
                    float(candidate.get("win_rate") or 0.0),
                    account_overview,
                    remaining_cash,
                    borrow_metrics=borrow_metrics if trade_direction == "SHORT" else None,
                )
                if suggested_shares <= 0:
                    signal = "IGNORE"
        orders.append(
            {
                "Ticker": candidate["ticker"],
                "Signal": signal,
                "Trade_Direction": trade_direction,
                "Reference_Price": round(reference_price, 2) if reference_price else None,
                "Market_Session": session_label,
                "Regular_Close": round(float(candidate.get("regular_close") or close or 0.0), 2) if (candidate.get("regular_close") or close) else None,
                "Session_Change_Pct": round(float(session_change_pct), 2) if session_change_pct is not None else None,
                "Entry_Limit_Price": entry_limit_price,
                "Initial_Stop_Loss": initial_stop_loss,
                "Take_Profit_Price": take_profit_price,
                "Suggested_Shares": suggested_shares,
                "Suggested_Notional": round(suggested_notional, 2) if suggested_notional else 0.0,
                "Capital_Usage": round(capital_usage, 2) if capital_usage else 0.0,
                "Budget_Pct": round((suggested_notional / float(account_overview.get("total_equity") or 1.0)) * 100.0, 2) if suggested_notional and account_overview.get("total_equity") else 0.0,
                "Sizing_Note": sizing_note,
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
                "Gate_Reason": gate_reason,
                "Earnings_Date": candidate.get("earnings_date"),
                "Days_To_Earnings": candidate.get("days_to_earnings"),
                "Earnings_Risk": candidate.get("earnings_gate_reason"),
                "allocation_regime": candidate.get("allocation_regime"),
                **_short_borrow_order_fields(borrow_metrics, account_overview),
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


def _run_walk_forward_backtest(holding_tickers: tuple[str, ...]) -> dict[str, Any]:
    requested_tickers = sorted(
        {
            "QQQ",
            *CORE_SIGNAL_TICKERS,
            *[
                AAOX_UNDERLYING if ticker == "AAOX" else ticker
                for ticker in holding_tickers
                if ticker
            ],
        }
    )
    pruned_factors = [
        "log_return_5d",
        "log_return_20d",
        "volatility_adjusted_momentum_20d",
        "volatility_adjusted_momentum_60d",
    ]
    factor_cols = [
        "hist_vol_20d",
        "volume_ratio_20d",
        "rsi_14",
        "relative_strength_20d_vs_benchmark",
    ]
    factor_study_sets = {
        "core_without_momentum": factor_cols,
        "momentum_10d_core": [
            *factor_cols,
            "log_return_10d",
        ],
        "momentum_60d_shadow": [
            *factor_cols,
            "log_return_60d",
        ],
    }
    robustness_random_states = [11, 23, 37, 41, 53, 67, 79, 97]
    robustness_fold_offsets = [0, 3, 6, 9, 12, 15, 18, 21]
    config = {
        "benchmark_ticker": "QQQ",
        "train_window": 126,
        "test_window": 21,
        "history_period": "18mo",
        "target_horizon": 1,
        "search_method": "random",
        "n_iter": 32,
        "transaction_cost_bps": 2.0,
        "pruned_factors": pruned_factors,
        "factor_cols": factor_cols,
        "robustness_random_states": robustness_random_states,
        "robustness_fold_offsets": robustness_fold_offsets,
        "robustness_n_iter": 24,
        "factor_study_sets": {name: list(values) for name, values in factor_study_sets.items()},
    }

    try:
        panel = load_project_panel_data(
            tickers=requested_tickers,
            benchmark_ticker=str(config["benchmark_ticker"]),
            period=str(config["history_period"]),
            interval="1d",
            target_horizon=int(config["target_horizon"]),
            min_rows_per_asset=int(config["train_window"] + config["test_window"]),
        )
        if panel.empty or panel["date"].nunique() < int(config["train_window"] + config["test_window"]):
            return {
                "available": False,
                "error": "历史样本不足，无法运行滚动前向验证",
                "config": config,
                "requested_assets": requested_tickers,
                "asset_universe": [],
                "panel_rows": int(len(panel)),
                "date_count": int(panel["date"].nunique()) if not panel.empty else 0,
                "equity_curve": pd.DataFrame(),
                "fold_summary": pd.DataFrame(),
                "metrics": {},
            }

        validator = WalkForwardValidator(
            panel_data=panel,
            factor_cols=factor_cols,
            target_col="forward_return_1d",
            train_window=int(config["train_window"]),
            test_window=int(config["test_window"]),
            search_method=str(config["search_method"]),
            n_iter=int(config["n_iter"]),
            transaction_cost_bps=float(config["transaction_cost_bps"]),
        )
        result = validator.run()
        equity_curve = result.equity_curve.reset_index(names="date")
        fold_summary = result.fold_summary.copy()
        return {
            "available": True,
            "error": "",
            "config": config,
            "requested_assets": requested_tickers,
            "asset_universe": sorted(panel["asset"].dropna().astype(str).unique().tolist()),
            "panel_rows": int(len(panel)),
            "date_count": int(panel["date"].nunique()),
            "equity_curve": equity_curve,
            "fold_summary": fold_summary,
            "metrics": result.metrics,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "config": config,
            "requested_assets": requested_tickers,
            "asset_universe": [],
            "panel_rows": 0,
            "date_count": 0,
            "equity_curve": pd.DataFrame(),
            "fold_summary": pd.DataFrame(),
            "metrics": {},
        }


@st.cache_data(ttl=7200, show_spinner="运行多路径稳健性研究（约3-5分钟）…")
def _run_robustness_study(holding_tickers: tuple[str, ...]) -> dict[str, Any]:
    """独立稳健性研究，与主分析流程解耦，仅在用户主动触发时运行。"""
    requested_tickers = sorted(
        {
            "QQQ",
            *CORE_SIGNAL_TICKERS,
            *[
                AAOX_UNDERLYING if ticker == "AAOX" else ticker
                for ticker in holding_tickers
                if ticker
            ],
        }
    )
    factor_cols = [
        "hist_vol_20d",
        "volume_ratio_20d",
        "rsi_14",
        "relative_strength_20d_vs_benchmark",
    ]
    factor_study_sets = {
        "core_without_momentum": factor_cols,
        "momentum_10d_core": [*factor_cols, "log_return_10d"],
        "momentum_60d_shadow": [*factor_cols, "log_return_60d"],
    }
    random_states = [11, 23, 37, 41, 53, 67, 79, 97]
    fold_offsets = [0, 3, 6, 9, 12, 15, 18, 21]
    panel = load_project_panel_data(
        tickers=requested_tickers,
        benchmark_ticker="QQQ",
        period="18mo",
        interval="1d",
        target_horizon=1,
        min_rows_per_asset=147,
    )
    if panel.empty:
        return {"error": "数据不足，无法运行稳健性研究"}
    path_specs = build_roll_forward_path_specs(
        random_states=random_states,
        fold_offsets=fold_offsets,
    )
    study = run_roll_forward_factor_study(
        panel_data=panel,
        factor_sets=factor_study_sets,
        target_col="forward_return_1d",
        train_window=126,
        test_window=21,
        path_specs=path_specs,
        search_method="random",
        n_iter=24,
        transaction_cost_bps=2.0,
        comparison_pairs=[
            ("momentum_10d_core", "core_without_momentum"),
            ("momentum_10d_core", "momentum_60d_shadow"),
        ],
    )
    return study


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def run_analysis_pipeline(holding_tickers: tuple[str, ...], kelly_fraction: float) -> dict[str, Any]:
    reset_fetch_warnings()
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
    vix_value = float((macro_snapshot.get("vix") or {}).get("value") or 0.0) or None
    qqq_sigma = detect_sigma_event(histories.get("QQQ", pd.DataFrame()))
    nvda_sigma = detect_sigma_event(histories.get("NVDA", pd.DataFrame()))
    aaoi_breakdown = detect_breakdown_signal(histories.get(AAOX_UNDERLYING, pd.DataFrame()))
    try:
        regime = classify_market_regime(
            macro_snapshot,
            factor_map.get("QQQ", {}),
            factor_map.get("NVDA", {}),
            aaoi_breakdown,
            news_map.get("NVDA", {}),
            market_history=qqq_history,
        )
    except TypeError as exc:
        if "market_history" not in str(exc):
            raise
        regime = classify_market_regime(
            macro_snapshot,
            factor_map.get("QQQ", {}),
            factor_map.get("NVDA", {}),
            aaoi_breakdown,
            news_map.get("NVDA", {}),
        )
        regime.setdefault("reasons", []).append("市场状态接口回退到旧版签名")

    decision_tickers = sorted({ticker for ticker in holding_tickers if ticker and ticker != "AAOX"})
    if "AAOX" in holding_tickers:
        decision_tickers.append("AAOI")
    signal_options_data = {
        ticker: get_near_term_options_summary(ticker)
        for ticker in sorted(set(CORE_SIGNAL_TICKERS + list(decision_tickers)))
    }

    core_signal_profiles: dict[str, dict[str, Any]] = {}
    for ticker in CORE_SIGNAL_TICKERS:
        history = histories.get(ticker, pd.DataFrame())
        core_signal_profiles[ticker] = build_trade_profile(
            ticker,
            factor_map.get(ticker, {}),
            news_map.get(ticker, {}),
            regime["score"],
            rolling_alpha=_rolling_alpha_from_history(history, qqq_history),
            vix_value=vix_value,
            options_data=signal_options_data.get(ticker),
        )

    holding_profiles: dict[str, dict[str, Any]] = {}
    for ticker in sorted(set(decision_tickers)):
        history = histories.get(ticker, pd.DataFrame())
        holding_profiles[ticker] = build_trade_profile(
            ticker,
            factor_map.get(ticker, {}),
            news_map.get(ticker, {}),
            regime["score"],
            rolling_alpha=_rolling_alpha_from_history(history, qqq_history),
            vix_value=vix_value,
            options_data=signal_options_data.get(ticker),
        )

    options_summary = get_near_term_options_summary(OPTIONS_TICKER)
    raw_radar_candidates = scan_breakout_candidates(
        RADAR_UNIVERSE,
        max_candidates=len(RADAR_UNIVERSE),
        regime_score=regime["score"],
        vix_value=vix_value,
    )
    radar_candidates: list[dict[str, Any]] = []
    for candidate in raw_radar_candidates:
        ticker = candidate["ticker"]
        session_quote = get_market_session_quote(ticker)
        candidate_news = build_news_features(get_news_items(ticker, limit=5))
        candidate_profile = candidate.copy()
        candidate_profile.update(
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
        radar_candidates.append(candidate_profile)
    radar_candidates = sorted(
        radar_candidates,
        key=lambda item: (
            bool(item.get("eligible_for_risk")),
            bool(item.get("breakout_valid") or item.get("breakdown")),
            abs(float(item.get("target_ev") or 0.0)),
            abs(float(item.get("expected_return_5d") or 0.0)),
        ),
        reverse=True,
    )[:5]
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
    kelly_asset_volatility = {
        ticker: float(factor_map.get(ticker, {}).get("hist_vol_20d") or 0.0)
        for ticker in kelly_inputs
    }
    kelly_result = continuous_kelly(
        kelly_inputs,
        returns_frame,
        risk_free_rate=RISK_FREE_RATE,
        kelly_fraction=kelly_fraction,
        asset_volatility=kelly_asset_volatility,
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
        "walk_forward_backtest": None,
        "data_notices": collect_fetch_notices(),
        "quote_map": quote_map,
        "data_warnings": collect_fetch_warnings(),
    }


st.title("美股量化半自动助手")
st.caption("本地采集公开数据，生成信号与报告。所有下单由你自行决定。")

saved_holdings = load_holdings(HOLDINGS_FILE)
saved_holdings_history = load_holdings_history(HOLDINGS_HISTORY_FILE)
saved_account_state = load_account_state(ACCOUNT_FILE)
with st.sidebar:
    st.subheader("持仓录入")
    current_editor_holdings = _render_sidebar_holding_editor(saved_holdings)
    if st.button("保存持仓", use_container_width=True):
        normalized = save_holdings(current_editor_holdings, HOLDINGS_FILE)
        save_holdings_snapshot(normalized, HOLDINGS_HISTORY_FILE, source="manual_save")
        saved_holdings_history = load_holdings_history(HOLDINGS_HISTORY_FILE)
        st.session_state["saved_holdings"] = normalized
        st.success("持仓已保存，并记录历史快照")

    st.caption("持仓会保存到 portfolio/holdings.json；每次保存或运行分析都会自动记录历史快照。")

    current_account_state = _render_sidebar_account_editor(saved_account_state)
    if st.button("保存资金参数", use_container_width=True):
        normalized_account = save_account_state(current_account_state, ACCOUNT_FILE)
        st.session_state["saved_account_state"] = normalized_account
        st.success("资金参数已保存")

    st.caption("资金参数会保存到 portfolio/account.json")

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
analysis_signature = (holding_tickers, round(float(kelly_fraction), 4))

if run_clicked:
    save_holdings_snapshot(holdings, HOLDINGS_HISTORY_FILE, source="analysis_run")
    saved_holdings_history = load_holdings_history(HOLDINGS_HISTORY_FILE)
    with st.spinner("正在抓取行情、新闻、期权和雷达数据..."):
        analysis = run_analysis_pipeline(holding_tickers, kelly_fraction)
    st.session_state["analysis_result"] = analysis
    st.session_state["analysis_signature"] = analysis_signature

analysis = None
if st.session_state.get("analysis_signature") == analysis_signature:
    cached_analysis = st.session_state.get("analysis_result")
    if isinstance(cached_analysis, dict):
        analysis = cached_analysis

if analysis is not None:
    holdings_reference_snapshot = resolve_reference_snapshot(holdings, saved_holdings_history)
    holdings_change_records = build_holdings_change_records(
        holdings,
        list((holdings_reference_snapshot or {}).get("holdings") or []),
    )
    snapshot_order_action_records: list[dict[str, Any]] = []
    holdings_history_rows = build_holdings_history_summary(saved_holdings_history, limit=12)
    quote_map = analysis["quote_map"]
    holdings_enriched = enrich_holdings_with_quotes(holdings, quote_map)
    short_borrow_metrics = {
        item["ticker"]: get_short_borrow_metrics(item["ticker"])
        for item in holdings_enriched
        if str(item.get("side") or "LONG").upper() == "SHORT" and item.get("ticker")
    }
    account_overview = compute_account_overview(holdings_enriched, current_account_state, borrow_metrics=short_borrow_metrics)
    core_signal_profiles = analysis["core_signal_profiles"]
    holding_profiles = analysis["holding_profiles"]
    factor_map = analysis["factor_map"]
    sigma_events = analysis["sigma_events"]
    aaoi_breakdown = analysis["aaoi_breakdown"]
    regime = analysis["regime"]
    observe_mode = bool(analysis.get("observe_mode"))
    observe_reason = str(analysis.get("observe_reason") or "")

    legacy_orders: list[dict[str, Any]] = []
    remaining_cash = float(account_overview.get("idle_cash") or 0.0)
    for holding in holdings_enriched:
        ticker = holding["ticker"]
        if ticker == "AAOX":
            order, remaining_cash = _legacy_order_for_ticker(
                holding,
                    ticker,
                    quote_map.get(ticker, get_latest_quote(ticker)),
                    holding_profiles.get("AAOI"),
                    factor_map.get("AAOI"),
                    account_overview,
                    remaining_cash,
                    regime,
                    observe_mode,
                    observe_reason,
                    breakdown_signal=aaoi_breakdown,
                )
            legacy_orders.append(order)
        else:
            order, remaining_cash = _legacy_order_for_ticker(
                holding,
                    ticker,
                    quote_map.get(ticker, {}),
                    holding_profiles.get(ticker),
                    factor_map.get(ticker),
                    account_overview,
                    remaining_cash,
                    regime,
                    observe_mode,
                    observe_reason,
                    sigma_events.get(ticker),
                )
            legacy_orders.append(order)

    account_overview["idle_cash_after_legacy_orders"] = remaining_cash
    snapshot_order_action_records = build_snapshot_order_action_records(
        legacy_orders,
        list((holdings_reference_snapshot or {}).get("holdings") or []),
    )
    new_alpha_targets = _build_new_alpha_targets(analysis["radar_candidates"], account_overview, observe_mode, observe_reason)
    mathematical_inference = _build_core_math_inference(factor_map, core_signal_profiles)
    quant_logic_log = _build_quant_logic_log(
        regime,
        sigma_events.get("QQQ", {}),
        sigma_events.get("NVDA", {}),
        aaoi_breakdown,
        observe_mode,
        observe_reason,
    )
    data_notices = analysis.get("data_notices", [])
    data_warnings = analysis.get("data_warnings", [])

    if data_notices:
        st.info("部分海外数据源不可达时，系统已自动切换到中国大陆/香港可达的数据源。海外源恢复后仍会优先使用海外源。")
        with st.expander("查看自动切源记录", expanded=False):
            for notice in data_notices:
                st.write(f"- {notice}")

    if data_warnings:
        st.warning("仍有部分数据在海外源和大陆/香港回退源上都未成功取回，当前已按空结果降级显示。")
        with st.expander("查看抓取告警", expanded=False):
            for warning in data_warnings:
                st.write(f"- {warning}")

    display_regime = {
        **regime,
        "label": _format_regime_label(regime.get("label")),
    }

    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("市场状态", display_regime["label"])
    top_mid.metric("风险偏好分数", f"{regime['score']:.2f}")
    top_right.metric("总建议风险敞口", f"{analysis['kelly']['total_exposure'] * 100:.1f}%")

    budget_col1, budget_col2, budget_col3, budget_col4, budget_col5, budget_col6 = st.columns(6)
    budget_col1.metric("账户总权益", f"${account_overview['total_equity']:,.0f}")
    budget_col2.metric("市场资金", f"${account_overview['market_funds']:,.0f}")
    budget_col3.metric("总盈利(浮动)", f"${account_overview['total_pnl']:,.0f}")
    budget_col4.metric("闲余资金", f"${account_overview['idle_cash']:,.0f}")
    budget_col5.metric("多头敞口", f"${account_overview['long_market_value']:,.0f}")
    budget_col6.metric("空头敞口", f"${account_overview['short_market_value']:,.0f}")

    if observe_mode:
        st.warning(f"当前触发空仓/观望强约束：{observe_reason}")

    backtest_signature = tuple(holding_tickers)
    if st.session_state.get("walk_forward_backtest_tickers") != backtest_signature:
        st.session_state.pop("walk_forward_backtest_result", None)
        st.session_state["walk_forward_backtest_tickers"] = backtest_signature

    backtest = st.session_state.get("walk_forward_backtest_result")
    if not isinstance(backtest, dict):
        backtest = analysis.get("walk_forward_backtest") or {}

    overview_tab, holdings_tab, radar_tab, backtest_tab, report_tab = st.tabs(["核心信号", "持仓盈亏", "异动雷达", "样本外回测", "报告与JSON"])

    with overview_tab:
        st.subheader("核心指数信号")
        st.caption("这里默认用 SPY、QQQ、DIA、IWM 判断大盘与风格状态；个股信号请看持仓页。")
        st.dataframe(pd.DataFrame.from_dict(mathematical_inference, orient="index"), use_container_width=True)
        st.subheader("期权流")
        st.json(analysis["options_summary"])

    with holdings_tab:
        st.subheader("账户资金概览")
        st.dataframe(_display_account_overview_frame(account_overview), use_container_width=True)
        borrow_summary_frame = _display_borrow_summary_frame(account_overview)
        if not borrow_summary_frame.empty:
            st.caption("空头借券与保证金明细")
            st.dataframe(borrow_summary_frame, use_container_width=True)
        st.subheader("持仓市值")
        if holdings_enriched:
            holdings_frame = _display_holdings_frame(holdings_enriched, holding_profiles)
            st.dataframe(holdings_frame, use_container_width=True)
        else:
            st.info("先在左侧录入持仓。")
        st.subheader("持仓历史对比")
        if holdings_reference_snapshot:
            st.caption(
                f"当前持仓对比基准：{holdings_reference_snapshot.get('captured_at')} · {_snapshot_source_label(holdings_reference_snapshot.get('source'))}"
            )
            change_frame = _display_holdings_change_frame(holdings_change_records)
            changed_only_frame = change_frame[change_frame["变更类型"] != "未变化"] if not change_frame.empty and "变更类型" in change_frame.columns else pd.DataFrame()
            if not changed_only_frame.empty:
                st.dataframe(changed_only_frame, use_container_width=True)
            elif not change_frame.empty:
                st.info("当前持仓与上次历史快照一致，没有发现新增、减仓或清仓变化。")
            else:
                st.info("当前持仓与历史快照之间暂时没有可展示的差异。")
        else:
            st.info("还没有更早的持仓快照。先保存一次持仓或运行一次分析，后续就能看到历史对比。")
        history_frame = _display_holdings_history_frame(holdings_history_rows)
        if not history_frame.empty:
            with st.expander("查看近几次持仓快照", expanded=False):
                st.dataframe(history_frame, use_container_width=True)
        if holdings_reference_snapshot and snapshot_order_action_records:
            st.subheader("上次持仓 vs 今日挂单动作")
            st.caption("这里能直接看到昨天还持有的仓位，今天系统给出的动作是否已经变成减仓、清仓或回补。")
            st.dataframe(_display_snapshot_order_action_frame(snapshot_order_action_records), use_container_width=True)
        if legacy_orders:
            st.subheader("旧仓位执行单")
            st.dataframe(_display_orders_frame(legacy_orders), use_container_width=True)

    with radar_tab:
        st.subheader("全市场异动雷达")
        st.dataframe(_display_orders_frame(analysis["radar_candidates"]), use_container_width=True)
        st.subheader("新 Alpha 目标")
        st.dataframe(_display_orders_frame(new_alpha_targets), use_container_width=True)

    with backtest_tab:
        st.subheader("滚动前向验证")
        st.caption("主分析不再默认阻塞式跑回测；这里按需触发并缓存当前持仓结果。")
        if st.button("▶ 运行样本外回测", key="run_walk_forward_backtest_btn"):
            st.session_state["walk_forward_backtest_result"] = _run_walk_forward_backtest(holding_tickers)
            backtest = st.session_state["walk_forward_backtest_result"]

        if not backtest:
            st.info("点击“运行样本外回测”后查看结果。")
        elif not backtest.get("available"):
            error_message = str(backtest.get("error") or "回测不可用")
            st.info(error_message)
            if backtest.get("config"):
                with st.expander("查看回测配置", expanded=False):
                    st.json(backtest)
        else:
            metrics = backtest.get("metrics") or {}
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Sharpe", f"{float(metrics.get('sharpe') or 0.0):.2f}")
            metric_col2.metric("Max Drawdown", f"{float(metrics.get('max_drawdown') or 0.0):.2%}")
            metric_col3.metric("Calmar", f"{float(metrics.get('calmar') or 0.0):.2f}")
            metric_col4.metric("Annualized Return", f"{float(metrics.get('annualized_return') or 0.0):.2%}")

            equity_curve = backtest.get("equity_curve")
            if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
                equity_chart = equity_curve.copy()
                equity_chart["date"] = pd.to_datetime(equity_chart["date"], errors="coerce")
                equity_chart = equity_chart.dropna(subset=["date"]).set_index("date")
                st.caption("样本外净值曲线")
                st.line_chart(equity_chart[["equity_curve"]], use_container_width=True)
                st.caption("样本外回撤曲线")
                st.line_chart(equity_chart[["drawdown"]], use_container_width=True)

            fold_summary = backtest.get("fold_summary")
            if isinstance(fold_summary, pd.DataFrame) and not fold_summary.empty:
                st.subheader("Fold 汇总")
                st.dataframe(fold_summary, use_container_width=True)

        st.divider()
        st.subheader("多路径因子稳健性研究")
        st.caption("运行 8 随机种子 × 8 路径偏移，对 3 组新因子口径做多路径滚动验证。结果会缓存 2 小时。")
        robustness_signature = tuple(holding_tickers)
        if st.session_state.get("robustness_study_tickers") != robustness_signature:
            st.session_state.pop("robustness_study_result", None)
            st.session_state["robustness_study_tickers"] = robustness_signature

        if st.button("▶ 运行稳健性研究", key="run_robustness_study_btn"):
            st.session_state["robustness_study_result"] = _run_robustness_study(holding_tickers)

        robustness_study = st.session_state.get("robustness_study_result")
        if isinstance(robustness_study, dict):
            robustness_error = str(robustness_study.get("error") or "")
            factor_summary = robustness_study.get("factor_summary")
            pairwise_summary = robustness_study.get("pairwise_summary")
            path_metrics = robustness_study.get("path_metrics")
            path_specs = robustness_study.get("path_specs")
            best_factor_set = str(robustness_study.get("best_factor_set") or "")

            if robustness_error:
                st.warning(f"稳健性研究执行失败：{robustness_error}")
            elif robustness_study:
                study_col1, study_col2, study_col3 = st.columns(3)
                study_col1.metric("最佳因子组", _format_factor_set_label(best_factor_set) or "无")
                study_col2.metric(
                    "路径数量",
                    str(len(path_specs)) if isinstance(path_specs, pd.DataFrame) and not path_specs.empty else "0",
                )
                significant_count = 0
                if isinstance(pairwise_summary, pd.DataFrame) and not pairwise_summary.empty and "statistically_significant_improvement" in pairwise_summary.columns:
                    significant_count = int(pairwise_summary["statistically_significant_improvement"].fillna(False).sum())
                study_col3.metric("显著改善对比数", str(significant_count))

                if isinstance(factor_summary, pd.DataFrame) and not factor_summary.empty:
                    factor_summary_display = _format_factor_set_frame(factor_summary)
                    st.caption("因子组路径分布摘要")
                    st.dataframe(
                        factor_summary_display.rename(
                            columns={
                                "factor_set": "因子组",
                                "path_count": "路径数",
                                "mean_sharpe": "平均夏普",
                                "median_sharpe": "夏普中位数",
                                "sharpe_std": "夏普波动",
                                "min_sharpe": "最差夏普",
                                "max_sharpe": "最佳夏普",
                                "positive_sharpe_ratio": "正夏普占比",
                                "mean_calmar": "平均Calmar",
                                "mean_max_drawdown": "平均最大回撤",
                                "mean_annualized_return": "平均年化收益",
                            }
                        ),
                        use_container_width=True,
                    )

                if isinstance(pairwise_summary, pd.DataFrame) and not pairwise_summary.empty:
                    pairwise_summary_display = _format_factor_set_frame(pairwise_summary)
                    st.caption("统计显著性对比")
                    st.dataframe(
                        pairwise_summary_display.rename(
                            columns={
                                "candidate_factor_set": "候选因子组",
                                "reference_factor_set": "基准因子组",
                                "path_count": "路径数",
                                "improvement_ratio": "改善路径占比",
                                "mean_sharpe_diff": "平均夏普差",
                                "median_sharpe_diff": "夏普中位差",
                                "mean_calmar_diff": "平均Calmar差",
                                "mean_annualized_return_diff": "平均年化收益差",
                                "mean_max_drawdown_diff": "平均最大回撤差",
                                "statistically_significant_improvement": "达到80%显著改善",
                            }
                        ),
                        use_container_width=True,
                    )

                if isinstance(path_metrics, pd.DataFrame) and not path_metrics.empty:
                    path_metrics_display = _format_factor_set_frame(path_metrics)
                    with st.expander("查看全部路径结果", expanded=False):
                        st.dataframe(
                            path_metrics_display.rename(columns={"factor_set": "因子组"}),
                            use_container_width=True,
                        )

                if isinstance(path_specs, pd.DataFrame) and not path_specs.empty:
                    with st.expander("查看路径配置", expanded=False):
                        st.dataframe(path_specs, use_container_width=True)

        if backtest:
            with st.expander("查看回测配置", expanded=False):
                st.json(
                    {
                        "config": backtest.get("config"),
                        "requested_assets": backtest.get("requested_assets"),
                        "asset_universe": backtest.get("asset_universe"),
                        "panel_rows": backtest.get("panel_rows"),
                        "date_count": backtest.get("date_count"),
                    }
                )

    payload_backtest = backtest.copy() if isinstance(backtest, dict) and backtest else None
    robustness_payload = st.session_state.get("robustness_study_result")
    if payload_backtest is not None and isinstance(robustness_payload, dict):
        payload_backtest = {
            **payload_backtest,
            "robustness_study": robustness_payload,
            "robustness_error": str(robustness_payload.get("error") or ""),
        }

    payload = build_execution_payload(
        analysis_date=datetime.now().strftime("%Y-%m-%d"),
        regime=display_regime,
        macro_snapshot=analysis["macro_snapshot"],
        account_overview=account_overview,
        mathematical_inference=mathematical_inference,
        legacy_orders=legacy_orders,
        new_alpha_targets=new_alpha_targets,
        kelly_weights=analysis["kelly"]["weights"],
        quant_logic_log=quant_logic_log,
        walk_forward_validation=payload_backtest,
    )
    markdown = build_markdown_report(payload)

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