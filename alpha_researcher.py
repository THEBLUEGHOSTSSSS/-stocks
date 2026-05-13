from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from xml.etree import ElementTree

import aiohttp


ARXIV_QFIN_FEED = "https://export.arxiv.org/api/query?search_query=cat:q-fin*&sortBy=submittedDate&sortOrder=descending&max_results=50"
NBER_WORKING_PAPERS = "https://www.nber.org/papers?page=1&perPage=25"

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "momentum": ("momentum", "trend", "continuation", "relative strength"),
    "mean_reversion": ("mean reversion", "reversal", "overreaction", "mispricing"),
    "volatility": ("volatility", "variance", "vix", "tail risk"),
    "options_flow": ("option", "gamma", "implied volatility", "skew"),
    "liquidity": ("liquidity", "order flow", "microstructure", "market making"),
    "macro": ("macro", "inflation", "rates", "monetary", "treasury"),
    "cross_asset": ("cross-section", "cross asset", "spillover", "network", "cointegration", "etf"),
    "earnings": ("earnings", "announcement", "guidance", "post-earnings"),
}

THEME_LABELS: dict[str, str] = {
    "momentum": "动量",
    "mean_reversion": "均值回归",
    "volatility": "波动率",
    "options_flow": "期权流",
    "liquidity": "流动性",
    "macro": "宏观",
    "cross_asset": "跨资产联动",
    "earnings": "财报漂移",
}

THEME_TEMPLATES: dict[str, dict[str, Any]] = {
    "momentum": {
        "signal_family": "动量延续",
        "factor_formula": "20日相对强弱 + 放量确认 + 52周新高强度",
        "entry_rule": "相对强弱为正且量比 > 1.2 时顺势跟进",
        "exit_rule": "5日动量转负或跌破 ATR 止损时退出",
        "risk_filters": ["避开财报前3个交易日", "VIX 急升时降权"],
    },
    "mean_reversion": {
        "signal_family": "残差均值回归",
        "factor_formula": "特质收益 Z 分数 + PCA 残差偏离 + 协整价差",
        "entry_rule": "|idio_zscore| > 2 且协整/残差触发时逆向建仓",
        "exit_rule": "价差回归到 0.5σ 内或止损触发时退出",
        "risk_filters": ["避免基本面断层日", "禁用无流动性小票"],
    },
    "volatility": {
        "signal_family": "波动率交易",
        "factor_formula": "隐含/实现波动率差 + 波动率斜率 + 事件冲击",
        "entry_rule": "波动率风险溢价显著偏离历史分位时介入",
        "exit_rule": "波动率回归中枢或事件落地后退出",
        "risk_filters": ["避免极端流动性抽离时段", "限制高杠杆暴露"],
    },
    "options_flow": {
        "signal_family": "期权流偏离",
        "factor_formula": "Call/Put 量比 + Gamma 挤压迹象 + IV 斜率",
        "entry_rule": "期权流和现货方向共振时顺势介入",
        "exit_rule": "期权流背离消失或现货放量反转时退出",
        "risk_filters": ["避免过宽价差期权链", "盘后不追单"],
    },
    "liquidity": {
        "signal_family": "流动性错配",
        "factor_formula": "成交量冲击 + 盘口失衡 + 振幅扩散",
        "entry_rule": "流动性收缩导致价格偏离时做回归/跟随",
        "exit_rule": "流动性恢复常态后退出",
        "risk_filters": ["低成交额标的限仓", "避开异常停牌与复牌日"],
    },
    "macro": {
        "signal_family": "宏观轮动",
        "factor_formula": "利率变化 + 通胀预期 + 风格相对强弱",
        "entry_rule": "宏观变量变点出现后切换成长/价值/防御暴露",
        "exit_rule": "宏观阈值回落或风格强弱反转后退出",
        "risk_filters": ["事件窗口减仓", "跨资产相关性飙升时降杠杆"],
    },
    "cross_asset": {
        "signal_family": "联动套利",
        "factor_formula": "相关图谱 + 先行滞后相关 + 协整价差",
        "entry_rule": "龙头先动、跟随标的滞后且相关/协整达标时切入",
        "exit_rule": "滞后差收敛或联动失效时退出",
        "risk_filters": ["仅保留相关性稳定链路", "异常公告/汇率冲击时禁用"],
    },
    "earnings": {
        "signal_family": "财报漂移",
        "factor_formula": "财报超预期幅度 + 指引修正 + 财报后漂移强度",
        "entry_rule": "财报超预期且后续扩散确认时跟进",
        "exit_rule": "漂移衰减或情绪见顶时退出",
        "risk_filters": ["避免财报前裸露持仓", "盘后跳空过大不追价"],
    },
}


@dataclass(frozen=True, slots=True)
class PaperRecord:
    source: str
    paper_id: str
    title: str
    abstract: str
    published_at: str
    url: str


class LLMClientProtocol(Protocol):
    async def generate_json(self, prompt: dict[str, Any]) -> dict[str, Any]:
        ...


class AcademicAlphaResearcher:
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def _fetch_text(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
            response.raise_for_status()
            return await response.text()

    async def fetch_arxiv_qfin(self) -> list[PaperRecord]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = await self._fetch_text(session, ARXIV_QFIN_FEED)

        root = ElementTree.fromstring(payload)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        records: list[PaperRecord] = []
        for entry in root.findall("atom:entry", namespace):
            paper_id = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()
            title = " ".join((entry.findtext("atom:title", default="", namespaces=namespace) or "").split())
            abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=namespace) or "").split())
            published_at = (entry.findtext("atom:published", default="", namespaces=namespace) or "").strip()
            if not paper_id or not title:
                continue
            records.append(
                PaperRecord(
                    source="arXiv",
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    published_at=published_at,
                    url=paper_id,
                )
            )
        return records

    async def fetch_nber(self) -> list[PaperRecord]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            html = await self._fetch_text(session, NBER_WORKING_PAPERS)

        records: list[PaperRecord] = []
        for chunk in html.split('<article'):
            if '/papers/' not in chunk:
                continue
            href_marker = 'href="'
            href_start = chunk.find(href_marker)
            if href_start < 0:
                continue
            href_start += len(href_marker)
            href_end = chunk.find('"', href_start)
            url = chunk[href_start:href_end]
            if not url.startswith('/papers/'):
                continue
            title_start = chunk.find('title="')
            if title_start >= 0:
                title_start += len('title="')
                title_end = chunk.find('"', title_start)
                title = chunk[title_start:title_end].strip()
            else:
                title = ""
            paper_id = url.rsplit('/', 1)[-1]
            if paper_id and title:
                records.append(
                    PaperRecord(
                        source="NBER",
                        paper_id=paper_id,
                        title=title,
                        abstract="",
                        published_at="",
                        url=f"https://www.nber.org{url}",
                    )
                )
        return records

    async def fetch_latest_papers(self) -> list[PaperRecord]:
        arxiv_records, nber_records = await asyncio.gather(self.fetch_arxiv_qfin(), self.fetch_nber(), return_exceptions=False)
        return arxiv_records + nber_records

    async def compile_factor_draft(self, llm_client: LLMClientProtocol, paper: PaperRecord) -> dict[str, Any]:
        prompt = {
            "role": "quant_research_compiler",
            "title": paper.title,
            "abstract": paper.abstract,
            "source": paper.source,
            "required_output": {
                "anomaly_name": "str",
                "feature_formula": "str",
                "entry_rule": "str",
                "exit_rule": "str",
                "risk_filters": ["str"],
                "python_factor_draft": "str",
            },
        }
        response = await llm_client.generate_json(prompt)
        return {
            "paper_id": paper.paper_id,
            "source": paper.source,
            "title": paper.title,
            "llm_factor_draft": response,
        }

    @staticmethod
    def _infer_themes(paper: PaperRecord) -> list[str]:
        haystack = f"{paper.title} {paper.abstract}".lower()
        themes = [
            theme
            for theme, keywords in THEME_KEYWORDS.items()
            if any(keyword in haystack for keyword in keywords)
        ]
        return themes or ["cross_asset"]

    @staticmethod
    def _paper_summary(paper: PaperRecord) -> str:
        text = (paper.abstract or paper.title or "").strip()
        if len(text) <= 220:
            return text
        return text[:217].rstrip() + "..."

    @staticmethod
    def _theme_label(theme: str) -> str:
        return THEME_LABELS.get(theme, theme)

    def _paper_to_insight(self, paper: PaperRecord) -> dict[str, Any]:
        themes = self._infer_themes(paper)
        primary_theme = themes[0]
        template = THEME_TEMPLATES.get(primary_theme, THEME_TEMPLATES["cross_asset"])
        return {
            "paper_id": paper.paper_id,
            "source": paper.source,
            "title": paper.title,
            "published_at": paper.published_at,
            "url": paper.url,
            "themes": themes,
            "signal_family": template["signal_family"],
            "factor_formula": template["factor_formula"],
            "entry_rule": template["entry_rule"],
            "exit_rule": template["exit_rule"],
            "risk_filters": template["risk_filters"],
            "summary": self._paper_summary(paper),
        }

    async def build_research_digest(self, max_papers: int = 8) -> dict[str, Any]:
        try:
            papers = await self.fetch_latest_papers()
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "paper_count": 0,
                "theme_counts": {},
                "summary": f"学术 Alpha 抓取失败，已降级为空结果: {exc}",
                "actionable_insights": [],
            }

        insights = [self._paper_to_insight(paper) for paper in papers[:max_papers]]
        theme_counts: dict[str, int] = {}
        for insight in insights:
            for theme in insight.get("themes", []):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

        dominant_themes = sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        summary = " / ".join(f"{self._theme_label(theme)}:{count}" for theme, count in dominant_themes) or "暂无明显论文主题聚集"
        return {
            "status": "ok",
            "error": "",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "paper_count": len(papers),
            "theme_counts": theme_counts,
            "summary": f"近期论文主题聚焦 {summary}",
            "actionable_insights": insights,
        }


def get_academic_alpha_digest(max_papers: int = 8) -> dict[str, Any]:
    researcher = AcademicAlphaResearcher()
    try:
        return asyncio.run(researcher.build_research_digest(max_papers=max_papers))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(researcher.build_research_digest(max_papers=max_papers))
        finally:
            loop.close()