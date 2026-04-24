from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from config import NEGATIVE_KEYWORDS, POSITIVE_KEYWORDS


def nonlinear_activation(value: float) -> float:
    return math.tanh(1.5 * value)


def _headline_weight(title: str, keyword_map: dict[str, float]) -> float:
    lowered = title.lower()
    return sum(weight for keyword, weight in keyword_map.items() if keyword in lowered)


def score_headline(title: str) -> float:
    return _headline_weight(title, POSITIVE_KEYWORDS) + _headline_weight(title, NEGATIVE_KEYWORDS)


def event_importance(title: str) -> int:
    lowered = title.lower()
    if any(keyword in lowered for keyword in ["earnings", "guidance", "acquisition", "fraud", "investigation"]):
        return 5
    if any(keyword in lowered for keyword in ["beats", "misses", "approval", "lawsuit", "downgrade"]):
        return 4
    if any(keyword in lowered for keyword in ["partnership", "launch", "upgrade", "expansion"]):
        return 3
    if any(keyword in lowered for keyword in ["ai", "chip", "cloud", "tariff"]):
        return 2
    return 1


def _recency_weight(published_at: Any, half_life_hours: float = 24.0) -> float:
    if published_at is None:
        return 1.0

    published = _parse_published_at(published_at)
    if published is None:
        return 1.0

    now = datetime.now(tz=timezone.utc)
    hours = max((now - published).total_seconds() / 3600.0, 0.0)
    return 0.5 ** (hours / half_life_hours)


def _parse_published_at(published_at: Any) -> datetime | None:
    if published_at is None:
        return None
    if isinstance(published_at, (int, float)):
        return datetime.fromtimestamp(published_at, tz=timezone.utc)
    if isinstance(published_at, str):
        try:
            return datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def build_news_features(news_items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    weighted_scores: list[float] = []
    weighted_importance: list[float] = []
    headlines: list[dict[str, Any]] = []
    latest_published_at: datetime | None = None
    latest_headline_title = ""
    latest_publisher = ""
    latest_source_channel = ""
    publishers: set[str] = set()
    source_channels: set[str] = set()

    for item in news_items:
        title = item.get("title")
        if not title:
            continue
        published_at = item.get("published_at")
        published_dt = _parse_published_at(published_at)
        base_score = score_headline(title)
        importance = event_importance(title)
        recency = _recency_weight(published_at)
        weighted = base_score * importance * recency
        weighted_scores.append(weighted)
        weighted_importance.append(importance * recency)
        publisher = str(item.get("publisher") or "Unknown")
        source_channel = str(item.get("source_channel") or "unknown")
        publishers.add(publisher)
        source_channels.add(source_channel)
        if published_dt and (latest_published_at is None or published_dt > latest_published_at):
            latest_published_at = published_dt
            latest_headline_title = title
            latest_publisher = publisher
            latest_source_channel = source_channel
        headlines.append(
            {
                "title": title,
                "publisher": publisher,
                "source_channel": source_channel,
                "score": round(base_score, 4),
                "importance": importance,
                "recency_weight": round(recency, 4),
                "published_at": published_at,
                "link": item.get("link"),
            }
        )

    if not weighted_scores:
        return {
            "headline_count": 0,
            "average_sentiment": 0.0,
            "activated_sentiment": 0.0,
            "shock_score": 0.0,
            "half_life_hours": 24.0,
            "news_source_count": 0,
            "news_channels": [],
            "latest_published_at": None,
            "latest_age_hours": None,
            "latest_headline": "",
            "latest_publisher": "",
            "latest_source_channel": "",
            "top_headlines": [],
        }

    total_importance = sum(weighted_importance) or 1.0
    average_sentiment = sum(weighted_scores) / total_importance
    shock_score = max(abs(score) for score in weighted_scores)
    top_headlines = sorted(headlines, key=lambda item: abs(item["score"]) * item["importance"], reverse=True)[:5]
    latest_age_hours = None
    if latest_published_at is not None:
        latest_age_hours = max((datetime.now(tz=timezone.utc) - latest_published_at).total_seconds() / 3600.0, 0.0)
    return {
        "headline_count": len(headlines),
        "average_sentiment": average_sentiment,
        "activated_sentiment": nonlinear_activation(average_sentiment),
        "shock_score": shock_score,
        "half_life_hours": 24.0,
        "news_source_count": len(publishers),
        "news_channels": sorted(source_channels),
        "latest_published_at": latest_published_at.isoformat() if latest_published_at is not None else None,
        "latest_age_hours": round(latest_age_hours, 2) if latest_age_hours is not None else None,
        "latest_headline": latest_headline_title,
        "latest_publisher": latest_publisher,
        "latest_source_channel": latest_source_channel,
        "top_headlines": top_headlines,
    }
