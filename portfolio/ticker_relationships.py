from __future__ import annotations

from typing import Any

from config import TICKER_METADATA
from portfolio.custom_tickers import normalize_ticker_symbol


STATIC_TICKER_RELATIONSHIPS: dict[str, dict[str, Any]] = {
    "AAOX": {
        "relation_type": "leveraged_single_name",
        "relation_group": "光模块杠杆链",
        "related_tickers": ["AAOI", "CIEN", "GLW", "ANET"],
        "related_external_assets": [],
        "relationship_logic": "2倍做多光模块/AAOI，常跟随底层与互联链事件放大波动。",
        "relationship_strength": 0.98,
    },
    "SOXL": {
        "relation_type": "leveraged_theme",
        "relation_group": "半导体杠杆链",
        "related_tickers": ["SOXX", "SMH", "NVDA", "AMD", "AVGO", "MU", "ASML", "LRCX", "KLAC", "MRVL"],
        "related_external_assets": [],
        "relationship_logic": "3倍做多半导体主题，对板块龙头和设备链共振最敏感。",
        "relationship_strength": 0.96,
    },
    "SOXX": {
        "relation_type": "theme_index",
        "relation_group": "半导体产业链",
        "related_tickers": ["SMH", "SOXL", "NVDA", "AMD", "AVGO", "MU", "ASML", "LRCX", "KLAC", "MRVL"],
        "related_external_assets": [],
        "relationship_logic": "半导体 ETF，常作为半导体产业链的板块锚。",
        "relationship_strength": 0.84,
    },
    "SMH": {
        "relation_type": "theme_index",
        "relation_group": "半导体产业链",
        "related_tickers": ["SOXX", "SOXL", "NVDA", "AMD", "AVGO", "MU", "ASML", "LRCX", "KLAC", "MRVL"],
        "related_external_assets": [],
        "relationship_logic": "VanEck 半导体 ETF，适合做半导体情绪与权重龙头映射。",
        "relationship_strength": 0.82,
    },
    "NVDL": {
        "relation_type": "leveraged_single_name",
        "relation_group": "NVDA杠杆链",
        "related_tickers": ["NVDA", "SOXL", "SOXX", "SMH"],
        "related_external_assets": [],
        "relationship_logic": "2倍做多 NVDA，常放大单一龙头事件与板块 beta。",
        "relationship_strength": 0.97,
    },
    "TQQQ": {
        "relation_type": "leveraged_index",
        "relation_group": "纳指杠杆链",
        "related_tickers": ["QQQ", "NVDA", "MSFT", "AAPL", "AMZN", "META"],
        "related_external_assets": [],
        "relationship_logic": "3倍做多纳指100，对大盘科技和高beta风格最敏感。",
        "relationship_strength": 0.93,
    },
    "SQQQ": {
        "relation_type": "inverse_index",
        "relation_group": "纳指反向链",
        "related_tickers": ["QQQ", "TQQQ", "NVDA", "MSFT", "AAPL"],
        "related_external_assets": [],
        "relationship_logic": "3倍做空纳指100，通常用于风险偏好急剧收缩阶段。",
        "relationship_strength": 0.9,
    },
    "FNGU": {
        "relation_type": "leveraged_theme",
        "relation_group": "高beta科技链",
        "related_tickers": ["META", "AMZN", "NFLX", "TSLA", "MSFT", "NVDA"],
        "related_external_assets": [],
        "relationship_logic": "3倍做多高beta科技篮子，适合捕捉事件驱动后的风格扩散。",
        "relationship_strength": 0.88,
    },
    "SIVEF": {
        "relation_type": "foreign_primary_link",
        "relation_group": "海外主上市映射",
        "related_tickers": [],
        "related_external_assets": ["STO:SIVE"],
        "relationship_logic": "OTC F-share 与瑞典主上市 SIVE 存在逻辑挂钩，主板价格与流动性变化可能向美股场外报价传导。",
        "relationship_strength": 0.9,
    },
}

CATEGORY_RELATION_RULES: tuple[dict[str, Any], ...] = (
    {
        "contains": ("半导体", "芯片", "CPU GPU", "存储", "网络芯片", "半导体设备"),
        "relation_group": "半导体产业链",
        "relation_type": "theme_peer",
        "anchor_tickers": ["SOXL", "SOXX", "SMH", "NVDA", "AMD", "AVGO", "MU", "ASML", "LRCX", "KLAC", "MRVL"],
        "logic": "半导体主题常在龙头、设备与杠杆ETF之间扩散。",
        "strength": 0.72,
    },
    {
        "contains": ("光模块", "CPO", "光通信", "AI互联"),
        "relation_group": "光模块/CPO链",
        "relation_type": "theme_peer",
        "anchor_tickers": ["AAOI", "AAOX", "CIEN", "GLW", "ANET"],
        "logic": "光模块与AI互联链存在订单、资本开支与主题传导。",
        "strength": 0.68,
    },
    {
        "contains": ("AI", "算力", "服务器", "AI基础设施"),
        "relation_group": "AI算力链",
        "relation_type": "theme_peer",
        "anchor_tickers": ["NVDA", "SMCI", "ANET", "NBIS", "AAOI", "SOXL"],
        "logic": "AI算力链会在芯片、服务器、互联和杠杆ETF之间传导。",
        "strength": 0.66,
    },
    {
        "contains": ("核能", "SMR"),
        "relation_group": "先进核能链",
        "relation_type": "theme_peer",
        "anchor_tickers": ["OKLO", "SMR", "NNE"],
        "logic": "先进核能小盘链条常在情绪驱动下联动扩散。",
        "strength": 0.62,
    },
    {
        "contains": ("比特币", "加密", "交易所/金融科技"),
        "relation_group": "加密β链",
        "relation_type": "theme_peer",
        "anchor_tickers": ["MSTR", "COIN"],
        "logic": "加密 beta 通常围绕交易所和资产负债表敞口标的放大。",
        "strength": 0.58,
    },
)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _normalize_ticker_list(values: list[Any], exclude: str = "") -> list[str]:
    normalized_values: list[str] = []
    excluded = normalize_ticker_symbol(exclude)
    for value in values:
        symbol = normalize_ticker_symbol(value)
        if symbol and symbol != excluded:
            normalized_values.append(symbol)
    return _dedupe_preserve_order(normalized_values)


def _normalize_relationship_seed(ticker: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": normalize_ticker_symbol(ticker),
        "relation_type": str(raw.get("relation_type") or "").strip(),
        "relation_group": str(raw.get("relation_group") or "").strip(),
        "related_tickers": _normalize_ticker_list(list(raw.get("related_tickers") or []), exclude=ticker),
        "related_external_assets": _dedupe_preserve_order([str(item or "").strip() for item in list(raw.get("related_external_assets") or [])]),
        "relationship_logic": str(raw.get("relationship_logic") or "").strip(),
        "relationship_strength": float(raw.get("relationship_strength") or 0.0),
    }


NORMALIZED_STATIC_RELATIONSHIPS = {
    ticker: _normalize_relationship_seed(ticker, raw)
    for ticker, raw in STATIC_TICKER_RELATIONSHIPS.items()
    if normalize_ticker_symbol(ticker)
}


def _infer_category_relationships(ticker: str, metadata: dict[str, dict[str, str]]) -> dict[str, Any]:
    category = str((metadata.get(ticker) or {}).get("category") or "").strip()
    if not category:
        return {
            "relation_type": "",
            "relation_group": "",
            "related_tickers": [],
            "relationship_logic": "",
            "relationship_strength": 0.0,
        }

    related_tickers: list[str] = []
    logic_parts: list[str] = []
    relation_groups: list[str] = []
    relation_type = ""
    relationship_strength = 0.0
    for rule in CATEGORY_RELATION_RULES:
        if any(token in category for token in rule["contains"]):
            related_tickers.extend(rule["anchor_tickers"])
            logic_parts.append(str(rule["logic"]))
            relation_groups.append(str(rule["relation_group"]))
            relation_type = relation_type or str(rule["relation_type"])
            relationship_strength = max(relationship_strength, float(rule["strength"]))

    return {
        "relation_type": relation_type,
        "relation_group": relation_groups[0] if relation_groups else "",
        "related_tickers": _normalize_ticker_list(related_tickers, exclude=ticker),
        "relationship_logic": " / ".join(_dedupe_preserve_order(logic_parts)),
        "relationship_strength": relationship_strength,
    }


def get_ticker_relationship_profile(
    ticker: Any,
    metadata: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    normalized_ticker = normalize_ticker_symbol(ticker)
    if not normalized_ticker:
        return {
            "relation_type": "",
            "relation_group": "",
            "related_tickers": [],
            "related_external_assets": [],
            "relationship_logic": "",
            "relationship_strength": 0.0,
        }

    metadata_lookup = metadata or TICKER_METADATA
    seeded = NORMALIZED_STATIC_RELATIONSHIPS.get(normalized_ticker, {})
    inferred = _infer_category_relationships(normalized_ticker, metadata_lookup)
    related_tickers = _dedupe_preserve_order(
        _normalize_ticker_list(list(seeded.get("related_tickers") or []), exclude=normalized_ticker)
        + _normalize_ticker_list(list(inferred.get("related_tickers") or []), exclude=normalized_ticker)
    )
    related_external_assets = _dedupe_preserve_order(list(seeded.get("related_external_assets") or []))
    relationship_logic = " / ".join(
        _dedupe_preserve_order(
            [
                str(seeded.get("relationship_logic") or "").strip(),
                str(inferred.get("relationship_logic") or "").strip(),
            ]
        )
    )
    return {
        "relation_type": str(seeded.get("relation_type") or inferred.get("relation_type") or "").strip(),
        "relation_group": str(seeded.get("relation_group") or inferred.get("relation_group") or "").strip(),
        "related_tickers": related_tickers,
        "related_external_assets": related_external_assets,
        "relationship_logic": relationship_logic,
        "relationship_strength": max(
            float(seeded.get("relationship_strength") or 0.0),
            float(inferred.get("relationship_strength") or 0.0),
        ),
    }


def expand_universe_with_related_tickers(
    base_universe: list[Any],
    metadata: dict[str, dict[str, str]] | None = None,
    include_seed_tickers: bool = True,
    max_related_per_ticker: int = 8,
) -> list[str]:
    metadata_lookup = metadata or TICKER_METADATA
    expanded = _normalize_ticker_list(list(base_universe))
    if include_seed_tickers:
        expanded = _dedupe_preserve_order(expanded + list(NORMALIZED_STATIC_RELATIONSHIPS))

    for ticker in list(expanded):
        profile = get_ticker_relationship_profile(ticker, metadata_lookup)
        expanded = _dedupe_preserve_order(expanded + list(profile.get("related_tickers") or [])[:max_related_per_ticker])
    return expanded