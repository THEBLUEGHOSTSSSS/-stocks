from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import SNAPSHOTS_DIR


RESEARCH_THEME_LABELS = {
    "momentum": "动量",
    "mean_reversion": "均值回归",
    "volatility": "波动率",
    "options_flow": "期权流",
    "liquidity": "流动性",
    "macro": "宏观",
    "cross_asset": "跨资产联动",
    "earnings": "财报漂移",
}

RESEARCH_STATUS_LABELS = {
    "ok": "已完成",
    "error": "抓取失败",
}


def _format_research_theme(theme: Any) -> str:
    normalized = str(theme or "").strip()
    if not normalized:
        return ""
    return RESEARCH_THEME_LABELS.get(normalized, normalized)


def _format_research_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if not normalized:
        return "未知"
    return RESEARCH_STATUS_LABELS.get(normalized, str(status))


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def _serialize_frame(frame: Any) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    serializable = frame.copy()
    for column in serializable.columns:
        serializable[column] = serializable[column].map(_json_safe_value)
    return serializable.to_dict(orient="records")


def _serialize_walk_forward_validation(walk_forward_validation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not walk_forward_validation:
        return None
    metrics = {
        key: round(float(value), 6)
        for key, value in (walk_forward_validation.get("metrics") or {}).items()
        if value is not None
    }
    robustness_study = walk_forward_validation.get("robustness_study") or {}
    return {
        "Available": bool(walk_forward_validation.get("available")),
        "Error": walk_forward_validation.get("error"),
        "Config": walk_forward_validation.get("config") or {},
        "Requested_Assets": walk_forward_validation.get("requested_assets") or [],
        "Asset_Universe": walk_forward_validation.get("asset_universe") or [],
        "Panel_Rows": int(walk_forward_validation.get("panel_rows") or 0),
        "Date_Count": int(walk_forward_validation.get("date_count") or 0),
        "Metrics": metrics,
        "Fold_Summary": _serialize_frame(walk_forward_validation.get("fold_summary")),
        "Equity_Curve": _serialize_frame(walk_forward_validation.get("equity_curve")),
        "Robustness_Error": walk_forward_validation.get("robustness_error"),
        "Robustness_Study": {
            "Best_Factor_Set": robustness_study.get("best_factor_set"),
            "Path_Specs": _serialize_frame(robustness_study.get("path_specs")),
            "Path_Metrics": _serialize_frame(robustness_study.get("path_metrics")),
            "Factor_Summary": _serialize_frame(robustness_study.get("factor_summary")),
            "Pairwise_Summary": _serialize_frame(robustness_study.get("pairwise_summary")),
        }
        if robustness_study or walk_forward_validation.get("robustness_error")
        else {},
    }


def _order_earnings_summary(order: dict[str, Any]) -> str:
    earnings_date = order.get("Earnings_Date") or order.get("earnings_date")
    days_to_earnings = order.get("Days_To_Earnings") or order.get("days_to_earnings")
    earnings_risk = order.get("Earnings_Risk") or order.get("earnings_gate_reason")
    parts: list[str] = []
    if earnings_date:
        parts.append(f"财报日 {earnings_date}")
    if days_to_earnings is not None:
        parts.append(f"距财报 {int(days_to_earnings)} 个交易日")
    if earnings_risk:
        parts.append(str(earnings_risk))
    return " / ".join(parts) if parts else "无近期财报风险标记"


def _resolve_execution_direction_value(order: dict[str, Any]) -> Any:
    return order.get("Execution_Direction") or order.get("execution_direction") or order.get("Trade_Direction") or order.get("trade_direction")


def _normalize_execution_order(order: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(order)
    execution_direction = _resolve_execution_direction_value(normalized)
    if execution_direction is not None:
        normalized["Execution_Direction"] = execution_direction
        normalized["execution_direction"] = execution_direction
    return normalized


def _serialize_alpha_research(alpha_research: dict[str, Any] | None) -> dict[str, Any]:
    if not alpha_research:
        return {}
    insights = []
    for insight in list(alpha_research.get("actionable_insights") or [])[:8]:
        insights.append(
            {
                "paper_id": insight.get("paper_id"),
                "source": insight.get("source"),
                "title": insight.get("title"),
                "published_at": insight.get("published_at"),
                "url": insight.get("url"),
                "themes": list(insight.get("themes") or []),
                "signal_family": insight.get("signal_family"),
                "factor_formula": insight.get("factor_formula"),
                "entry_rule": insight.get("entry_rule"),
                "exit_rule": insight.get("exit_rule"),
                "risk_filters": list(insight.get("risk_filters") or []),
                "summary": insight.get("summary"),
            }
        )
    return {
        "Status": alpha_research.get("status"),
        "Error": alpha_research.get("error"),
        "Fetched_At": alpha_research.get("fetched_at"),
        "Paper_Count": int(alpha_research.get("paper_count") or 0),
        "Theme_Counts": alpha_research.get("theme_counts") or {},
        "Summary": alpha_research.get("summary") or "",
        "Actionable_Insights": insights,
    }


def build_execution_payload(
    analysis_date: str,
    regime: dict[str, Any],
    macro_snapshot: dict[str, Any],
    account_overview: dict[str, Any],
    mathematical_inference: dict[str, Any],
    legacy_orders: list[dict[str, Any]],
    new_alpha_targets: list[dict[str, Any]],
    kelly_weights: dict[str, float],
    quant_logic_log: str,
    walk_forward_validation: dict[str, Any] | None = None,
    alpha_research: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_legacy_orders = [_normalize_execution_order(order) for order in legacy_orders]
    normalized_new_alpha_targets = [_normalize_execution_order(order) for order in new_alpha_targets]
    payload = {
        "Date": analysis_date,
        "Market_Regime_Score": round(float(regime.get("score", 0.0)), 4),
        "Market_Regime_Label": regime.get("label"),
        "Macro_Snapshot": macro_snapshot,
        "Portfolio_Budget": account_overview,
        "Execution_Orders": {
            "Legacy_Positions": normalized_legacy_orders,
            "New_Alpha_Targets": normalized_new_alpha_targets,
        },
        "Mathematical_Inference": mathematical_inference,
        "Academic_Alpha_Research": _serialize_alpha_research(alpha_research),
        "Kelly_Weights": {ticker: round(weight, 4) for ticker, weight in kelly_weights.items()},
        "Quant_Logic_Log": quant_logic_log[:50],
    }
    serialized_backtest = _serialize_walk_forward_validation(walk_forward_validation)
    if serialized_backtest is not None:
        payload["Walk_Forward_Validation"] = serialized_backtest
    return payload


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    borrow_summary = (payload.get("Portfolio_Budget") or {}).get("borrow_metrics_summary") or {}
    lines.append(f"# 美股量化执行报告 {payload['Date']}")
    lines.append("")
    lines.append("## 市场环境")
    lines.append(f"- 市场状态: {payload.get('Market_Regime_Label')}")
    lines.append(f"- 风险偏好分数: {payload.get('Market_Regime_Score')}")
    lines.append(f"- 核心逻辑: {payload.get('Quant_Logic_Log')}")
    lines.append("")
    lines.append("## 宏观快照")
    for key, value in payload.get("Macro_Snapshot", {}).items():
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("")
    alpha_research = payload.get("Academic_Alpha_Research") or {}
    if alpha_research:
        lines.append("## 学术 Alpha 追踪")
        lines.append(f"- 状态: {_format_research_status(alpha_research.get('Status'))}")
        if alpha_research.get("Error"):
            lines.append(f"- 错误: {alpha_research.get('Error')}")
        lines.append(f"- 抓取时间: {alpha_research.get('Fetched_At')}")
        lines.append(f"- 论文数量: {alpha_research.get('Paper_Count')}")
        lines.append(f"- 摘要: {alpha_research.get('Summary')}")
        for theme, count in (alpha_research.get("Theme_Counts") or {}).items():
            lines.append(f"- 主题 {_format_research_theme(theme)}: {count}")
        insights = alpha_research.get("Actionable_Insights") or []
        for insight in insights[:5]:
            lines.append(
                "- "
                f"{insight.get('title')} / {insight.get('signal_family')} / 公式: {insight.get('factor_formula')} / "
                f"入场: {insight.get('entry_rule')} / 出场: {insight.get('exit_rule')}"
            )
        lines.append("")
    walk_forward_validation = payload.get("Walk_Forward_Validation") or {}
    if walk_forward_validation:
        lines.append("## 样本外回测")
        if not walk_forward_validation.get("Available"):
            lines.append(f"- 状态: 不可用 ({walk_forward_validation.get('Error')})")
        else:
            metrics = walk_forward_validation.get("Metrics") or {}
            lines.append(f"- Sharpe: {float(metrics.get('sharpe') or 0.0):.4f}")
            lines.append(f"- Max Drawdown: {float(metrics.get('max_drawdown') or 0.0):.2%}")
            lines.append(f"- Calmar: {float(metrics.get('calmar') or 0.0):.4f}")
            lines.append(f"- Annualized Return: {float(metrics.get('annualized_return') or 0.0):.2%}")
            lines.append(f"- 回测资产: {', '.join(walk_forward_validation.get('Asset_Universe') or [])}")
            lines.append(f"- Fold 数: {len(walk_forward_validation.get('Fold_Summary') or [])}")
            fold_summary = walk_forward_validation.get("Fold_Summary") or []
            for fold in fold_summary:
                lines.append(
                    "- "
                    f"Fold {int(fold.get('fold_id') or 0)}: "
                    f"测试区间 {fold.get('test_start')} -> {fold.get('test_end')}, "
                    f"Sharpe {float(fold.get('test_sharpe') or 0.0):.2f}, "
                    f"MaxDD {float(fold.get('test_max_drawdown') or 0.0):.2%}, "
                    f"Calmar {float(fold.get('test_calmar') or 0.0):.2f}"
                )
            equity_curve = walk_forward_validation.get("Equity_Curve") or []
            if equity_curve:
                last_point = equity_curve[-1]
                lines.append(f"- 期末净值: {float(last_point.get('equity_curve') or 0.0):.4f}")
                preview_points = equity_curve[-5:] if len(equity_curve) > 5 else equity_curve
                for point in preview_points:
                    lines.append(
                        "- "
                        f"净值快照 {point.get('date')}: "
                        f"equity {float(point.get('equity_curve') or 0.0):.4f}, "
                        f"drawdown {float(point.get('drawdown') or 0.0):.2%}"
                    )
            robustness_error = walk_forward_validation.get("Robustness_Error")
            robustness_study = walk_forward_validation.get("Robustness_Study") or {}
            if robustness_error:
                lines.append(f"- 稳健性研究: 不可用 ({robustness_error})")
            elif robustness_study:
                best_factor_set = robustness_study.get("Best_Factor_Set")
                if best_factor_set:
                    lines.append(f"- 最优因子组: {best_factor_set}")
                factor_summary = robustness_study.get("Factor_Summary") or []
                if factor_summary:
                    lines.append("- 因子组分布摘要:")
                    for row in factor_summary[:5]:
                        lines.append(
                            "- "
                            f"{row.get('factor_set')}: 平均夏普 {float(row.get('mean_sharpe') or 0.0):.3f}, "
                            f"夏普中位数 {float(row.get('median_sharpe') or 0.0):.3f}, "
                            f"正夏普占比 {float(row.get('positive_sharpe_ratio') or 0.0):.0%}"
                        )
                pairwise_summary = robustness_study.get("Pairwise_Summary") or []
                if pairwise_summary:
                    lines.append("- 统计显著性对比:")
                    for row in pairwise_summary:
                        significance_label = "是" if bool(row.get("statistically_significant_improvement")) else "否"
                        lines.append(
                            "- "
                            f"{row.get('candidate_factor_set')} vs {row.get('reference_factor_set')}: "
                            f"改善路径占比 {float(row.get('improvement_ratio') or 0.0):.0%}, "
                            f"平均夏普差 {float(row.get('mean_sharpe_diff') or 0.0):.3f}, "
                            f"达到80%显著改善 {significance_label}"
                        )
        lines.append("")
    lines.append("## 账户资金")
    for key, value in payload.get("Portfolio_Budget", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    if borrow_summary:
        lines.append("## 借券情况")
        lines.append(f"- 平均借券费率: {float(borrow_summary.get('avg_borrow_fee_pct') or 0.0):.2f}%/年")
        lines.append(f"- 难借个股数: {int(borrow_summary.get('htb_count') or 0)}")
        lines.append(f"- 禁空个股数: {int(borrow_summary.get('prohibit_count') or 0)}")
        for ticker, summary in (borrow_summary.get("tickers") or {}).items():
            lines.append(
                "- "
                f"{ticker}: 借券费率 {float(summary.get('fee_pct') or 0.0):.2f}%/年, "
                f"可借股数 {int(summary.get('available_shares') or 0)}, "
                f"HTB={bool(summary.get('is_hard_to_borrow'))}, "
                f"禁空={bool(summary.get('prohibit_short'))}, "
                f"初始保证金 {float(summary.get('initial_margin_pct') or 0.0):.2f}%, "
                f"维持担保 {float(summary.get('maintenance_margin_pct') or 0.0):.2f}%"
            )
        lines.append("")
    lines.append("## 核心推演")
    for ticker, inference in payload.get("Mathematical_Inference", {}).items():
        lines.append(f"### {ticker}")
        for field, value in inference.items():
            lines.append(f"- {field}: {value}")
        lines.append("")
    lines.append("## 执行单")
    lines.append("### 旧仓位")
    for order in payload.get("Execution_Orders", {}).get("Legacy_Positions", []):
        lines.append(
            "- "
            f"{order.get('Ticker')} / {order.get('Signal')} / {order.get('Position_Side') or _resolve_execution_direction_value(order)} / "
            f"财报信息: {_order_earnings_summary(order)} / 门控: {order.get('Gate_Reason')}"
        )
    lines.append("")
    lines.append("### 新目标")
    for order in payload.get("Execution_Orders", {}).get("New_Alpha_Targets", []):
        lines.append(
            "- "
            f"{order.get('Ticker')} / {order.get('Signal')} / {_resolve_execution_direction_value(order)} / "
            f"财报信息: {_order_earnings_summary(order)} / 门控: {order.get('Gate_Reason')}"
        )
    lines.append("")
    lines.append("## 原始 JSON")
    lines.append("```json")
    lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def save_report_bundle(
    payload: dict[str, Any],
    markdown: str,
    snapshot_dir: Path = SNAPSHOTS_DIR,
) -> dict[str, str]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_name = f"quant_report_{timestamp}"
    json_path = snapshot_dir / f"{base_name}.json"
    md_path = snapshot_dir / f"{base_name}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }
