from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SNAPSHOTS_DIR


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
) -> dict[str, Any]:
    return {
        "Date": analysis_date,
        "Market_Regime_Score": round(float(regime.get("score", 0.0)), 4),
        "Market_Regime_Label": regime.get("label"),
        "Macro_Snapshot": macro_snapshot,
        "Portfolio_Budget": account_overview,
        "Execution_Orders": {
            "Legacy_Positions": legacy_orders,
            "New_Alpha_Targets": new_alpha_targets,
        },
        "Mathematical_Inference": mathematical_inference,
        "Kelly_Weights": {ticker: round(weight, 4) for ticker, weight in kelly_weights.items()},
        "Quant_Logic_Log": quant_logic_log[:50],
    }


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
        lines.append(f"- {json.dumps(order, ensure_ascii=False)}")
    lines.append("")
    lines.append("### 新目标")
    for order in payload.get("Execution_Orders", {}).get("New_Alpha_Targets", []):
        lines.append(f"- {json.dumps(order, ensure_ascii=False)}")
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
