from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import pandas as pd
import requests
import yfinance as yf

from config import TICKER_METADATA


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    normalized = frame.copy()
    normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
    if isinstance(normalized.index, pd.DatetimeIndex):
        try:
            normalized.index = normalized.index.tz_localize(None)
        except TypeError:
            pass
    return normalized


def get_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    history = yf.Ticker(ticker).history(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    return _normalize_history_frame(history)


def batch_history(
    tickers: Iterable[str],
    period: str = "3mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        return {}

    raw = yf.download(
        tickers=unique_tickers,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames: dict[str, pd.DataFrame] = {}
    if not isinstance(raw.columns, pd.MultiIndex):
        frames[unique_tickers[0]] = _normalize_history_frame(raw)
        return frames

    for ticker in unique_tickers:
        if ticker not in raw.columns.get_level_values(0):
            frames[ticker] = pd.DataFrame()
            continue
        frame = raw[ticker].copy()
        frames[ticker] = _normalize_history_frame(frame)
    return frames


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_session_label(market_state: str | None) -> str:
    normalized = str(market_state or "").upper()
    if normalized.startswith("PRE"):
        return "盘前"
    if normalized == "REGULAR":
        return "盘中"
    if normalized.startswith("POST"):
        return "盘后"
    if normalized in {"CLOSED", "CLOSE"}:
        return "休市"
    return "未知"


def get_market_session_quote(ticker: str) -> dict[str, Any]:
    ticker_obj = yf.Ticker(ticker)
    info: dict[str, Any] = {}
    try:
        info = ticker_obj.get_info() or {}
    except Exception:
        info = {}

    fast_info = dict(getattr(ticker_obj, "fast_info", {}) or {})
    regular_previous_close = _safe_float(
        info.get("regularMarketPreviousClose")
        or fast_info.get("regularMarketPreviousClose")
        or fast_info.get("previousClose")
    )
    regular_market_price = _safe_float(
        info.get("regularMarketPrice")
        or info.get("currentPrice")
        or fast_info.get("lastPrice")
    )
    pre_market_price = _safe_float(info.get("preMarketPrice"))
    post_market_price = _safe_float(info.get("postMarketPrice"))
    market_state = str(info.get("marketState") or "").upper()
    session_label = _market_session_label(market_state)

    if session_label == "盘前" and pre_market_price is not None:
        active_price = pre_market_price
    elif session_label == "盘后" and post_market_price is not None:
        active_price = post_market_price
    else:
        active_price = regular_market_price

    change_pct = None
    if active_price is not None and regular_previous_close:
        change_pct = ((active_price / regular_previous_close) - 1.0) * 100.0

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": ticker,
        "close": active_price,
        "reference_price": active_price,
        "regular_market_price": regular_market_price,
        "previous_close": regular_previous_close,
        "regular_close": regular_previous_close,
        "pre_market_price": pre_market_price,
        "post_market_price": post_market_price,
        "market_state": market_state or None,
        "session_label": session_label,
        "session_change_pct": change_pct,
        "change_pct": change_pct,
        "as_of": fetched_at,
        "price_source": "yfinance.info",
    }


def get_latest_quote(ticker: str) -> dict[str, Any]:
    history = get_history(ticker, period="10d", interval="1d")
    if history.empty:
        session_quote = get_market_session_quote(ticker)
        return {
            "ticker": ticker,
            "close": session_quote.get("close"),
            "previous_close": session_quote.get("previous_close"),
            "change_pct": session_quote.get("change_pct"),
            "session_label": session_quote.get("session_label"),
            "session_change_pct": session_quote.get("session_change_pct"),
            "price_source": session_quote.get("price_source"),
            "as_of": session_quote.get("as_of"),
        }

    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else latest
    previous_close = float(previous.get("close", latest.get("close", 0.0)))
    close = float(latest.get("close", previous_close))
    change_pct = ((close / previous_close) - 1.0) * 100.0 if previous_close else None

    as_of_value = history.index[-1]
    as_of = as_of_value.strftime("%Y-%m-%d") if isinstance(as_of_value, datetime) else str(as_of_value)
    return {
        "ticker": ticker,
        "close": close,
        "open": float(latest.get("open", close)),
        "high": float(latest.get("high", close)),
        "low": float(latest.get("low", close)),
        "volume": float(latest.get("volume", 0.0)),
        "previous_close": previous_close,
        "change_pct": change_pct,
        "as_of": as_of,
    }


def _extract_news_item(raw_item: dict[str, Any]) -> dict[str, Any] | None:
    content = raw_item.get("content", {}) if isinstance(raw_item, dict) else {}
    if not isinstance(content, dict):
        content = {}

    click_through = content.get("clickThroughUrl")
    if not isinstance(click_through, dict):
        click_through = {}

    canonical = content.get("canonicalUrl")
    if not isinstance(canonical, dict):
        canonical = {}

    provider = content.get("provider")
    if not isinstance(provider, dict):
        provider = {}

    title = raw_item.get("title") or content.get("title")
    if not title:
        return None

    publisher = raw_item.get("publisher")
    if not publisher:
        publisher = provider.get("displayName")

    link = raw_item.get("link")
    if not link:
        link = click_through.get("url") or canonical.get("url")

    published = raw_item.get("providerPublishTime") or content.get("pubDate")
    return {
        "title": title,
        "publisher": publisher or "Unknown",
        "link": link,
        "published_at": published,
        "source_channel": "Yahoo Finance",
    }


def _parse_google_news_pubdate(pubdate: str | None) -> str | None:
    if not pubdate:
        return None
    try:
        return parsedate_to_datetime(pubdate).isoformat()
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


@lru_cache(maxsize=256)
def _ticker_news_profile(ticker: str) -> dict[str, str]:
    cleaned = ticker.replace("^", "").strip().upper()
    profile = {
        "ticker": cleaned,
        "name": str((TICKER_METADATA.get(cleaned) or {}).get("name") or "").strip(),
        "exchange_label": "",
    }

    try:
        info = yf.Ticker(cleaned).get_info() or {}
    except Exception:
        info = {}

    if not profile["name"]:
        profile["name"] = str(info.get("longName") or info.get("shortName") or "").strip()

    exchange = str(info.get("exchange") or "").upper()
    if exchange in {"NMS", "NAS", "NASDAQ"}:
        profile["exchange_label"] = "NASDAQ"
    elif exchange in {"NYQ", "NYE", "NYSE"}:
        profile["exchange_label"] = "NYSE"
    return profile


def _google_news_query(ticker: str) -> str:
    profile = _ticker_news_profile(ticker)
    query_parts = [f'"{profile["ticker"]} stock"', f'"{profile["ticker"]} shares"']
    if profile["exchange_label"]:
        query_parts.append(f'"{profile["exchange_label"]}:{profile["ticker"]}"')
    if profile["name"]:
        query_parts.append(f'"{profile["name"]}"')
    return f"({' OR '.join(query_parts)}) when:2d"


def _looks_like_finance_news(title: str) -> bool:
    lowered = title.lower()
    finance_keywords = [
        "stock",
        "shares",
        "earnings",
        "guidance",
        "analyst",
        "price target",
        "after-hours",
        "premarket",
        "pre-market",
        "market",
        "nasdaq",
        "nyse",
        "chip",
        "chips",
        "semiconductor",
        "ai",
        "revenue",
        "profit",
        "etf",
        "futures",
        "upgrade",
        "downgrade",
        "rally",
        "selloff",
    ]
    return any(keyword in lowered for keyword in finance_keywords)


def _fetch_google_news_items(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    query = quote_plus(_google_news_query(ticker))
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ElementTree.fromstring(response.content)
    except ElementTree.ParseError:
        return []

    items: list[dict[str, Any]] = []
    for raw_item in root.findall("./channel/item"):
        title = (raw_item.findtext("title") or "").strip()
        link = (raw_item.findtext("link") or "").strip()
        publisher = (raw_item.findtext("source") or "Google News").strip()
        published_at = _parse_google_news_pubdate(raw_item.findtext("pubDate"))
        if not title or not _looks_like_finance_news(title):
            continue
        items.append(
            {
                "title": title,
                "publisher": publisher,
                "link": link or None,
                "published_at": published_at,
                "source_channel": "Google News RSS",
            }
        )
        if len(items) >= limit:
            break
    return items


def _published_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    published_at = item.get("published_at")
    if isinstance(published_at, str) and published_at:
        return (1, published_at)
    return (0, "")


def get_news_items(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    ticker_obj = yf.Ticker(ticker)
    raw_news: list[dict[str, Any]] = []
    try:
        raw_news = list(getattr(ticker_obj, "news", []) or [])
    except Exception:
        raw_news = []

    items: list[dict[str, Any]] = []
    for raw_item in raw_news:
        item = _extract_news_item(raw_item)
        if item is not None:
            items.append(item)

    items.extend(_fetch_google_news_items(ticker, limit=limit))

    deduped: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for item in sorted(items, key=_published_sort_key, reverse=True):
        normalized_title = str(item.get("title") or "").strip().casefold()
        if not normalized_title or normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped
