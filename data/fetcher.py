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


_FETCH_WARNINGS: dict[str, str] = {}
_FETCH_NOTICES: dict[str, str] = {}
_DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}
_EASTMONEY_SEARCH_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"
_EASTMONEY_SEARCH_URL = "https://searchapi.eastmoney.com/api/suggest/get"
_EASTMONEY_HISTORY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_EASTMONEY_ALIAS_MAP = {
    "^TNX": "US10Y",
}
_EASTMONEY_STATIC_QUOTE_IDS = {
    "^TNX": "171.US10Y",
    "US10Y": "171.US10Y",
}


def _summarize_exception(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    if len(message) > 180:
        message = f"{message[:177]}..."
    return f"{type(exc).__name__}: {message}"


def register_fetch_warning(
    scope: str,
    ticker: str | None,
    exc: Exception,
    provider: str = "Yahoo Finance",
) -> None:
    symbol = (ticker or "全局").strip() or "全局"
    summary = _summarize_exception(exc)
    key = f"{provider}:{scope}:{symbol}:{summary}"
    _FETCH_WARNINGS.setdefault(key, f"{provider} {scope}失败（{symbol}）：{summary}")


def register_fetch_notice(scope: str, message: str) -> None:
    key = f"{scope}:{message}"
    _FETCH_NOTICES.setdefault(key, message)


def reset_fetch_warnings() -> None:
    _FETCH_WARNINGS.clear()
    _FETCH_NOTICES.clear()


def collect_fetch_warnings() -> list[str]:
    return list(_FETCH_WARNINGS.values())


def collect_fetch_notices() -> list[str]:
    return list(_FETCH_NOTICES.values())


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


def _history_limit_from_period(period: str) -> int:
    normalized = str(period or "6mo").strip().lower()
    if normalized.endswith("d"):
        return max(int(normalized[:-1] or 10) + 2, 10)
    if normalized.endswith("mo"):
        return max(int(normalized[:-2] or 1) * 22 + 5, 22)
    if normalized.endswith("y"):
        return max(int(normalized[:-1] or 1) * 252 + 5, 252)
    return 132


def _eastmoney_price(value: Any) -> float | None:
    scaled = _safe_float(value)
    if scaled is None:
        return None
    return scaled / 1000.0


def _eastmoney_pct(value: Any) -> float | None:
    scaled = _safe_float(value)
    if scaled is None:
        return None
    return scaled / 100.0


@lru_cache(maxsize=512)
def _resolve_eastmoney_quote_id(ticker: str) -> str | None:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        return None
    if normalized in _EASTMONEY_STATIC_QUOTE_IDS:
        return _EASTMONEY_STATIC_QUOTE_IDS[normalized]

    query = _EASTMONEY_ALIAS_MAP.get(normalized, normalized.replace("^", ""))
    response = requests.get(
        _EASTMONEY_SEARCH_URL,
        params={"input": query, "type": "14", "token": _EASTMONEY_SEARCH_TOKEN},
        timeout=10,
        headers=_DEFAULT_HEADERS,
    )
    response.raise_for_status()
    payload = response.json()
    items = (((payload or {}).get("QuotationCodeTable") or {}).get("Data")) or []
    target_code = query.upper()

    for item in items:
        code = str(item.get("Code") or "").upper()
        quote_id = str(item.get("QuoteID") or "").strip()
        if code == target_code and quote_id:
            return quote_id

    for item in items:
        quote_id = str(item.get("QuoteID") or "").strip()
        classify = str(item.get("Classify") or "")
        if quote_id and classify in {"UsStock", "UB"}:
            return quote_id
    return None


def _fetch_eastmoney_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    if interval != "1d":
        return pd.DataFrame()

    quote_id = _resolve_eastmoney_quote_id(ticker)
    if not quote_id:
        return pd.DataFrame()

    response = requests.get(
        _EASTMONEY_HISTORY_URL,
        params={
            "secid": quote_id,
            "klt": "101",
            "fqt": "1" if auto_adjust else "0",
            "lmt": str(_history_limit_from_period(period)),
            "end": "20500101",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        },
        timeout=15,
        headers=_DEFAULT_HEADERS,
    )
    response.raise_for_status()
    payload = response.json()
    raw_klines = ((payload or {}).get("data") or {}).get("klines") or []
    if not raw_klines:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for raw_line in raw_klines:
        parts = str(raw_line).split(",")
        if len(parts) < 7:
            continue
        records.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6],
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    for column in ["open", "close", "high", "low", "volume", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    normalized = _normalize_history_frame(frame)
    normalized.attrs["price_source"] = "eastmoney.history"
    return normalized


def _fetch_eastmoney_realtime_quote(ticker: str) -> dict[str, Any]:
    quote_id = _resolve_eastmoney_quote_id(ticker)
    if not quote_id:
        return {}

    response = requests.get(
        _EASTMONEY_QUOTE_URL,
        params={
            "secid": quote_id,
            "fields": "f43,f44,f45,f46,f47,f57,f58,f60,f169,f170,f171",
        },
        timeout=10,
        headers=_DEFAULT_HEADERS,
    )
    response.raise_for_status()
    payload = response.json()
    data = (payload or {}).get("data") or {}
    if not data:
        return {}

    current_price = _eastmoney_price(data.get("f43"))
    previous_close = _eastmoney_price(data.get("f60"))
    change_pct = _eastmoney_pct(data.get("f170"))
    if change_pct is None and current_price is not None and previous_close:
        change_pct = ((current_price / previous_close) - 1.0) * 100.0

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "ticker": str(data.get("f57") or ticker).upper(),
        "close": current_price,
        "reference_price": current_price,
        "regular_market_price": current_price,
        "previous_close": previous_close,
        "regular_close": previous_close,
        "pre_market_price": None,
        "post_market_price": None,
        "market_state": None,
        "session_label": "未知",
        "session_change_pct": change_pct,
        "change_pct": change_pct,
        "as_of": fetched_at,
        "price_source": "eastmoney.realtime",
    }


def get_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    primary_error: Exception | None = None
    try:
        history = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
        )
        normalized = _normalize_history_frame(history)
        if not normalized.empty:
            normalized.attrs["price_source"] = "yfinance.history"
            return normalized
        primary_error = RuntimeError("海外源返回空数据")
    except Exception as exc:
        primary_error = exc

    try:
        fallback = _fetch_eastmoney_history(ticker, period=period, interval=interval, auto_adjust=auto_adjust)
        if not fallback.empty:
            register_fetch_notice("历史行情", "部分海外历史行情不可达，已自动切换到东方财富数据源。")
            return fallback
        fallback_error: Exception | None = RuntimeError("东方财富未返回历史行情")
    except Exception as exc:
        fallback_error = exc

    if primary_error is not None:
        register_fetch_warning("历史行情", ticker, primary_error)
    if fallback_error is not None:
        register_fetch_warning("历史行情回退", ticker, fallback_error, provider="东方财富")
    return pd.DataFrame()


def batch_history(
    tickers: Iterable[str],
    period: str = "3mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        return {}

    raw: pd.DataFrame | None = None
    try:
        raw = yf.download(
            tickers=unique_tickers,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        raw = None

    frames: dict[str, pd.DataFrame] = {}
    if raw is not None and not raw.empty:
        if not isinstance(raw.columns, pd.MultiIndex):
            if len(unique_tickers) == 1:
                normalized = _normalize_history_frame(raw)
                normalized.attrs["price_source"] = "yfinance.history"
                frames[unique_tickers[0]] = normalized
            else:
                for ticker in unique_tickers:
                    frames[ticker] = pd.DataFrame()
        else:
            for ticker in unique_tickers:
                if ticker not in raw.columns.get_level_values(0):
                    frames[ticker] = pd.DataFrame()
                    continue
                frame = raw[ticker].copy()
                normalized = _normalize_history_frame(frame)
                if not normalized.empty:
                    normalized.attrs["price_source"] = "yfinance.history"
                frames[ticker] = normalized

    for ticker in unique_tickers:
        if ticker in frames and not frames[ticker].empty:
            continue
        frames[ticker] = get_history(ticker, period=period, interval=interval, auto_adjust=auto_adjust)
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
    primary_errors: list[Exception] = []
    try:
        info = ticker_obj.get_info() or {}
    except Exception as exc:
        primary_errors.append(exc)
        info = {}

    try:
        fast_info = dict(getattr(ticker_obj, "fast_info", {}) or {})
    except Exception as exc:
        primary_errors.append(exc)
        fast_info = {}
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

    price_source = "yfinance.info"
    if all(
        value is None
        for value in [
            active_price,
            regular_market_price,
            regular_previous_close,
            pre_market_price,
            post_market_price,
        ]
    ):
        try:
            fallback_quote = _fetch_eastmoney_realtime_quote(ticker)
        except Exception as exc:
            fallback_quote = {}
            register_fetch_warning("实时快照回退", ticker, exc, provider="东方财富")
        if fallback_quote:
            register_fetch_notice("实时快照", "部分海外实时行情不可达，已自动切换到东方财富数据源。")
            return fallback_quote
        price_source = "unavailable"
        if primary_errors:
            register_fetch_warning("实时快照", ticker, primary_errors[0])
        else:
            register_fetch_warning("实时快照", ticker, RuntimeError("海外源与东方财富均未返回可用快照"))

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
        "price_source": price_source,
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
        "price_source": str(history.attrs.get("price_source") or "yfinance.history"),
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
            headers=_DEFAULT_HEADERS,
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
