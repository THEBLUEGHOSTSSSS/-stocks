from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq, newton
from scipy.stats import norm


InstrumentType = Literal["stock", "call", "put"]
PositionSide = Literal["long", "short"]

DEFAULT_RISK_FREE_RATE = 0.045
DEFAULT_DAYS_TO_EXPIRY = 30.0
MIN_TIME_TO_EXPIRY = 1.0 / 365.0
MIN_VOLATILITY = 1e-4
MAX_VOLATILITY = 5.0
HIGH_IVR_THRESHOLD = 80.0
LOW_IVR_THRESHOLD = 20.0
IVR_PROXY_LOW_MULTIPLIER = 0.65
IVR_PROXY_HIGH_MULTIPLIER = 1.75


@dataclass(frozen=True, slots=True)
class StrategyLeg:
    instrument: InstrumentType
    direction: PositionSide
    quantity: int = 1
    strike: float | None = None
    premium: float = 0.0
    entry_price: float | None = None
    label: str = ""


@dataclass(frozen=True, slots=True)
class OptionStrategy:
    code: str
    name: str
    ticker: str
    summary: str
    scenario: str
    rationale: str
    spot: float
    support: float
    resistance: float
    max_profit: str
    max_loss: str
    breakeven_points: tuple[float, ...]
    legs: tuple[StrategyLeg, ...]


@dataclass(frozen=True, slots=True)
class OptionMarketSnapshot:
    option_type: Literal["call", "put"]
    strike: float
    premium: float
    time_to_expiry: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    label: str = ""


@dataclass(frozen=True, slots=True)
class VolatilityContext:
    current_iv: float
    current_iv_source: str
    iv_rank: float
    iv_rank_method: str
    historical_iv_low: float
    historical_iv_high: float
    risk_free_rate: float
    dividend_yield: float
    time_to_expiry: float
    days_to_expiry: float
    expected_move_pct: float
    expected_move_lower: float
    expected_move_upper: float
    volatility_regime: str


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(resolved):
        return None
    return resolved


def _normalize_option_type(value: Any) -> Literal["call", "put"] | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"call", "c"}:
        return "call"
    if normalized in {"put", "p"}:
        return "put"
    return None


def _strike_step(spot: float) -> float:
    return 1.0 if spot >= 50.0 else 0.5


def _round_strike(value: float, step: float, mode: Literal["up", "down", "nearest"] = "nearest") -> float:
    scaled = value / step
    if mode == "up":
        return round(np.ceil(scaled) * step, 2)
    if mode == "down":
        return round(np.floor(scaled) * step, 2)
    return round(np.round(scaled) * step, 2)


def _normalize_days_to_expiry(days_to_expiry: float | None) -> tuple[float, float]:
    resolved = _coerce_float(days_to_expiry) or DEFAULT_DAYS_TO_EXPIRY
    if resolved <= 1.0:
        time_to_expiry = max(resolved, MIN_TIME_TO_EXPIRY)
        return time_to_expiry, time_to_expiry * 365.0
    time_to_expiry = max(resolved / 365.0, MIN_TIME_TO_EXPIRY)
    return time_to_expiry, max(resolved, 1.0)


def _annualize_implied_move(
    implied_move_pct: float | None,
    time_to_expiry: float,
    historical_volatility: float | None,
) -> float:
    candidates: list[float] = []
    resolved_move = _coerce_float(implied_move_pct)
    if resolved_move is not None and resolved_move > 0.0 and time_to_expiry > 0.0:
        candidates.append(resolved_move / math.sqrt(time_to_expiry))
    resolved_hist_vol = _coerce_float(historical_volatility)
    if resolved_hist_vol is not None and resolved_hist_vol > 0.0:
        candidates.append(resolved_hist_vol)
    if not candidates:
        return 0.25
    return float(np.clip(np.median(candidates), 0.05, 1.5))


def _normalize_market_option_snapshots(
    market_option_snapshots: list[dict[str, Any]] | None,
    *,
    default_time_to_expiry: float,
    default_risk_free_rate: float,
    default_dividend_yield: float,
) -> list[OptionMarketSnapshot]:
    normalized: list[OptionMarketSnapshot] = []
    for raw in list(market_option_snapshots or []):
        option_type = _normalize_option_type(raw.get("option_type") or raw.get("instrument") or raw.get("right"))
        strike = _coerce_float(raw.get("strike"))
        premium = _coerce_float(raw.get("premium") or raw.get("last_price") or raw.get("mark") or raw.get("mid"))
        if option_type is None or strike is None or strike <= 0.0 or premium is None or premium <= 0.0:
            continue

        snapshot_time = _coerce_float(raw.get("time_to_expiry"))
        snapshot_days = _coerce_float(raw.get("days_to_expiry"))
        if snapshot_time is None and snapshot_days is not None:
            snapshot_time = max(snapshot_days / 365.0, MIN_TIME_TO_EXPIRY)
        elif snapshot_time is not None and snapshot_time > 1.0:
            snapshot_time = max(snapshot_time / 365.0, MIN_TIME_TO_EXPIRY)
        else:
            snapshot_time = max(snapshot_time or default_time_to_expiry, MIN_TIME_TO_EXPIRY)

        snapshot_rate = _coerce_float(raw.get("risk_free_rate")) or default_risk_free_rate
        snapshot_dividend = _coerce_float(raw.get("dividend_yield")) or default_dividend_yield
        normalized.append(
            OptionMarketSnapshot(
                option_type=option_type,
                strike=float(strike),
                premium=float(premium),
                time_to_expiry=float(snapshot_time),
                risk_free_rate=float(snapshot_rate),
                dividend_yield=float(snapshot_dividend),
                label=str(raw.get("label") or ""),
            )
        )
    return normalized


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: Literal["call", "put"],
    *,
    dividend_yield: float = 0.0,
) -> float:
    resolved_spot = max(float(spot), 1e-8)
    resolved_strike = max(float(strike), 1e-8)
    resolved_time = max(float(time_to_expiry), 0.0)
    resolved_vol = max(float(volatility), MIN_VOLATILITY)
    resolved_rate = float(risk_free_rate)
    resolved_dividend = float(dividend_yield)

    if resolved_time <= 0.0:
        intrinsic = max(resolved_spot - resolved_strike, 0.0) if option_type == "call" else max(resolved_strike - resolved_spot, 0.0)
        return float(intrinsic)

    sqrt_time = math.sqrt(resolved_time)
    d1 = (
        math.log(resolved_spot / resolved_strike)
        + (resolved_rate - resolved_dividend + 0.5 * resolved_vol * resolved_vol) * resolved_time
    ) / (resolved_vol * sqrt_time)
    d2 = d1 - resolved_vol * sqrt_time
    discounted_spot = resolved_spot * math.exp(-resolved_dividend * resolved_time)
    discounted_strike = resolved_strike * math.exp(-resolved_rate * resolved_time)

    if option_type == "call":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def black_scholes_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    dividend_yield: float = 0.0,
) -> float:
    resolved_spot = max(float(spot), 1e-8)
    resolved_strike = max(float(strike), 1e-8)
    resolved_time = max(float(time_to_expiry), MIN_TIME_TO_EXPIRY)
    resolved_vol = max(float(volatility), MIN_VOLATILITY)
    resolved_rate = float(risk_free_rate)
    resolved_dividend = float(dividend_yield)

    sqrt_time = math.sqrt(resolved_time)
    d1 = (
        math.log(resolved_spot / resolved_strike)
        + (resolved_rate - resolved_dividend + 0.5 * resolved_vol * resolved_vol) * resolved_time
    ) / (resolved_vol * sqrt_time)
    return float(resolved_spot * math.exp(-resolved_dividend * resolved_time) * norm.pdf(d1) * sqrt_time)


def implied_volatility_from_price(
    premium: float,
    strike: float,
    spot: float,
    risk_free_rate: float,
    time_to_expiry: float,
    option_type: Literal["call", "put"],
    *,
    dividend_yield: float = 0.0,
    initial_guess: float | None = None,
) -> float | None:
    resolved_premium = _coerce_float(premium)
    resolved_strike = _coerce_float(strike)
    resolved_spot = _coerce_float(spot)
    resolved_rate = _coerce_float(risk_free_rate)
    resolved_time = _coerce_float(time_to_expiry)
    if (
        resolved_premium is None
        or resolved_strike is None
        or resolved_spot is None
        or resolved_rate is None
        or resolved_time is None
        or resolved_premium <= 0.0
        or resolved_strike <= 0.0
        or resolved_spot <= 0.0
    ):
        return None

    resolved_time = max(resolved_time, MIN_TIME_TO_EXPIRY)
    discounted_spot = resolved_spot * math.exp(-float(dividend_yield) * resolved_time)
    discounted_strike = resolved_strike * math.exp(-resolved_rate * resolved_time)
    lower_bound = max(discounted_spot - discounted_strike, 0.0) if option_type == "call" else max(discounted_strike - discounted_spot, 0.0)
    if resolved_premium < lower_bound - 1e-8:
        return None

    def objective(volatility: float) -> float:
        return black_scholes_price(
            resolved_spot,
            resolved_strike,
            resolved_time,
            resolved_rate,
            volatility,
            option_type,
            dividend_yield=dividend_yield,
        ) - resolved_premium

    def derivative(volatility: float) -> float:
        return black_scholes_vega(
            resolved_spot,
            resolved_strike,
            resolved_time,
            resolved_rate,
            volatility,
            dividend_yield=dividend_yield,
        )

    seed = float(np.clip(initial_guess if initial_guess is not None else 0.3, 0.05, 1.5))
    try:
        solved = float(newton(objective, x0=seed, fprime=derivative, tol=1e-8, maxiter=50))
        if np.isfinite(solved) and MIN_VOLATILITY <= solved <= MAX_VOLATILITY:
            return solved
    except (RuntimeError, OverflowError, ZeroDivisionError):
        pass

    try:
        lower = MIN_VOLATILITY
        upper = MAX_VOLATILITY
        if objective(lower) * objective(upper) <= 0.0:
            solved = float(brentq(objective, lower, upper, maxiter=100, xtol=1e-8))
            if np.isfinite(solved) and MIN_VOLATILITY <= solved <= MAX_VOLATILITY:
                return solved
    except (RuntimeError, OverflowError, ValueError):
        pass
    return None


def _estimate_current_iv(
    spot: float,
    snapshots: list[OptionMarketSnapshot],
    *,
    default_time_to_expiry: float,
    default_risk_free_rate: float,
    default_dividend_yield: float,
    base_iv_guess: float,
) -> tuple[float, str]:
    weighted_ivs: list[tuple[float, float]] = []
    for snapshot in snapshots:
        implied_vol = implied_volatility_from_price(
            snapshot.premium,
            snapshot.strike,
            spot,
            snapshot.risk_free_rate,
            snapshot.time_to_expiry,
            snapshot.option_type,
            dividend_yield=snapshot.dividend_yield,
            initial_guess=base_iv_guess,
        )
        if implied_vol is None:
            continue
        moneyness_gap = abs(snapshot.strike - spot) / max(spot, 1e-6)
        weight = 1.0 / max(moneyness_gap, 0.015)
        weighted_ivs.append((implied_vol, weight))

    if weighted_ivs:
        values = np.asarray([value for value, _ in weighted_ivs], dtype=float)
        weights = np.asarray([weight for _, weight in weighted_ivs], dtype=float)
        return float(np.clip(np.average(values, weights=weights), 0.05, 2.5)), "market_premium_inversion"

    synthetic_atm_premium = black_scholes_price(
        spot,
        spot,
        default_time_to_expiry,
        default_risk_free_rate,
        base_iv_guess,
        "call",
        dividend_yield=default_dividend_yield,
    )
    synthetic_iv = implied_volatility_from_price(
        synthetic_atm_premium,
        spot,
        spot,
        default_risk_free_rate,
        default_time_to_expiry,
        "call",
        dividend_yield=default_dividend_yield,
        initial_guess=base_iv_guess,
    )
    if synthetic_iv is not None:
        return float(np.clip(synthetic_iv, 0.05, 2.5)), "implied_move_proxy"
    return float(np.clip(base_iv_guess, 0.05, 2.5)), "implied_move_proxy"


def _estimate_iv_rank(
    current_iv: float,
    *,
    iv_history: list[float] | np.ndarray | None,
    historical_volatility: float | None,
) -> tuple[float, str, float, float]:
    raw_history = iv_history.tolist() if isinstance(iv_history, np.ndarray) else list(iv_history or [])
    normalized_history = np.asarray(
        [value for value in (_coerce_float(item) for item in raw_history) if value is not None and value > 0.0],
        dtype=float,
    )
    if normalized_history.size >= 2:
        iv_low = float(np.min(normalized_history))
        iv_high = float(np.max(normalized_history))
        method = "52w_iv_history"
    else:
        resolved_hist_vol = _coerce_float(historical_volatility)
        if resolved_hist_vol is not None and resolved_hist_vol > 0.0:
            iv_low = max(0.05, resolved_hist_vol * IVR_PROXY_LOW_MULTIPLIER)
            iv_high = max(iv_low + 0.05, resolved_hist_vol * IVR_PROXY_HIGH_MULTIPLIER)
            method = "proxy_vs_realized_vol_band"
        else:
            iv_low = max(0.05, current_iv * 0.75)
            iv_high = max(iv_low + 0.05, current_iv * 1.25)
            method = "fallback_static_band"

    if abs(iv_high - iv_low) <= 1e-8:
        return 50.0, method, iv_low, iv_high
    iv_rank = float(np.clip(100.0 * (current_iv - iv_low) / (iv_high - iv_low), 0.0, 100.0))
    return iv_rank, method, iv_low, iv_high


def _build_volatility_context(
    spot: float,
    *,
    implied_move_pct: float | None,
    historical_volatility: float | None,
    iv_history: list[float] | np.ndarray | None,
    market_option_snapshots: list[dict[str, Any]] | None,
    risk_free_rate: float,
    dividend_yield: float,
    days_to_expiry: float,
) -> VolatilityContext:
    time_to_expiry, resolved_days_to_expiry = _normalize_days_to_expiry(days_to_expiry)
    base_iv_guess = _annualize_implied_move(implied_move_pct, time_to_expiry, historical_volatility)
    snapshots = _normalize_market_option_snapshots(
        market_option_snapshots,
        default_time_to_expiry=time_to_expiry,
        default_risk_free_rate=risk_free_rate,
        default_dividend_yield=dividend_yield,
    )
    current_iv, current_iv_source = _estimate_current_iv(
        spot,
        snapshots,
        default_time_to_expiry=time_to_expiry,
        default_risk_free_rate=risk_free_rate,
        default_dividend_yield=dividend_yield,
        base_iv_guess=base_iv_guess,
    )
    iv_rank, iv_rank_method, historical_iv_low, historical_iv_high = _estimate_iv_rank(
        current_iv,
        iv_history=iv_history,
        historical_volatility=historical_volatility,
    )
    expected_move_pct = max(current_iv * math.sqrt(time_to_expiry), 0.02)
    expected_move_lower = max(spot * math.exp(-expected_move_pct), 0.01)
    expected_move_upper = max(spot * math.exp(expected_move_pct), expected_move_lower * 1.01)

    if iv_rank >= HIGH_IVR_THRESHOLD:
        volatility_regime = "高 IVR / 卖波动"
    elif iv_rank <= LOW_IVR_THRESHOLD:
        volatility_regime = "低 IVR / 买波动"
    else:
        volatility_regime = "中性 IVR / 场景驱动"

    return VolatilityContext(
        current_iv=current_iv,
        current_iv_source=current_iv_source,
        iv_rank=iv_rank,
        iv_rank_method=iv_rank_method,
        historical_iv_low=historical_iv_low,
        historical_iv_high=historical_iv_high,
        risk_free_rate=float(risk_free_rate),
        dividend_yield=float(dividend_yield),
        time_to_expiry=time_to_expiry,
        days_to_expiry=resolved_days_to_expiry,
        expected_move_pct=expected_move_pct,
        expected_move_lower=expected_move_lower,
        expected_move_upper=expected_move_upper,
        volatility_regime=volatility_regime,
    )


def _model_option_premium(
    spot: float,
    strike: float,
    option_type: Literal["call", "put"],
    context: VolatilityContext,
) -> float:
    premium = black_scholes_price(
        spot,
        strike,
        context.time_to_expiry,
        context.risk_free_rate,
        context.current_iv,
        option_type,
        dividend_yield=context.dividend_yield,
    )
    return round(max(premium, 0.01), 2)


def _resolve_sigma_strike(
    spot: float,
    option_type: Literal["call", "put"],
    context: VolatilityContext,
    *,
    scale: float,
) -> float:
    step = _strike_step(spot)
    if option_type == "call":
        return _round_strike(spot * math.exp(context.expected_move_pct * scale), step, mode="up")
    return _round_strike(max(spot * math.exp(-context.expected_move_pct * scale), step), step, mode="down")


def _serialize_strategy(strategy: OptionStrategy, primary_code: str, context: VolatilityContext) -> dict[str, Any]:
    return {
        "code": strategy.code,
        "name": strategy.name,
        "ticker": strategy.ticker,
        "summary": strategy.summary,
        "scenario": strategy.scenario,
        "rationale": strategy.rationale,
        "spot": strategy.spot,
        "support": strategy.support,
        "resistance": strategy.resistance,
        "max_profit": strategy.max_profit,
        "max_loss": strategy.max_loss,
        "breakeven_points": list(strategy.breakeven_points),
        "is_primary": strategy.code == primary_code,
        "legs": [asdict(leg) for leg in strategy.legs],
        "current_iv": round(context.current_iv, 6),
        "current_iv_pct": round(context.current_iv * 100.0, 2),
        "current_iv_source": context.current_iv_source,
        "iv_rank": round(context.iv_rank, 2),
        "iv_rank_method": context.iv_rank_method,
        "historical_iv_low": round(context.historical_iv_low, 6),
        "historical_iv_high": round(context.historical_iv_high, 6),
        "risk_free_rate": round(context.risk_free_rate, 6),
        "dividend_yield": round(context.dividend_yield, 6),
        "time_to_expiry_years": round(context.time_to_expiry, 6),
        "days_to_expiry": round(context.days_to_expiry, 2),
        "expected_move_pct": round(context.expected_move_pct, 6),
        "expected_move_lower": round(context.expected_move_lower, 2),
        "expected_move_upper": round(context.expected_move_upper, 2),
        "volatility_regime": context.volatility_regime,
        "pricing_model": "black_scholes_merton",
    }


def _build_covered_call(
    ticker: str,
    spot: float,
    support: float,
    resistance: float,
    context: VolatilityContext,
) -> OptionStrategy:
    step = _strike_step(spot)
    call_strike = _round_strike(max(resistance, _resolve_sigma_strike(spot, "call", context, scale=0.65)), step, mode="up")
    call_premium = _model_option_premium(spot, call_strike, "call", context)
    max_profit_value = max(call_strike - spot + call_premium, 0.0)
    max_loss_value = max(spot - call_premium, 0.0)
    breakeven = max(spot - call_premium, 0.0)
    return OptionStrategy(
        code="covered_call",
        name="IV 驱动备兑开仓",
        ticker=ticker,
        summary=f"当前 IV {context.current_iv * 100.0:.1f}% ，若已有正股，可在 {call_strike:.2f} 附近卖出 Call，用高估权利金降本。",
        scenario="已有正股，波动率不低，且不预期短期暴力突破上沿。",
        rationale="备兑 Call 仍属于卖波动结构，适合中高 IV 环境下把时间价值和波动溢价收回来。",
        spot=spot,
        support=support,
        resistance=resistance,
        max_profit=f"{max_profit_value:.2f} / 股",
        max_loss=f"{max_loss_value:.2f} / 股",
        breakeven_points=(round(breakeven, 2),),
        legs=(
            StrategyLeg(instrument="stock", direction="long", quantity=1, entry_price=spot, label="持有正股"),
            StrategyLeg(instrument="call", direction="short", quantity=1, strike=call_strike, premium=call_premium, label="卖出 Call"),
        ),
    )


def _build_bear_put_spread(
    ticker: str,
    spot: float,
    support: float,
    resistance: float,
    context: VolatilityContext,
) -> OptionStrategy:
    step = _strike_step(spot)
    long_put_strike = _round_strike(min(spot, max(support, spot * math.exp(-context.expected_move_pct * 0.45))), step, mode="nearest")
    short_put_strike = _resolve_sigma_strike(spot, "put", context, scale=0.95)
    if short_put_strike >= long_put_strike:
        short_put_strike = max(_round_strike(long_put_strike - max(step, 1.0), step, mode="down"), step)

    long_put_premium = _model_option_premium(spot, long_put_strike, "put", context)
    short_put_premium = _model_option_premium(spot, short_put_strike, "put", context)
    net_debit = max(long_put_premium - short_put_premium, 0.01)
    spread_width = max(long_put_strike - short_put_strike, 0.0)
    max_profit_value = max(spread_width - net_debit, 0.0)
    breakeven = max(long_put_strike - net_debit, 0.0)
    return OptionStrategy(
        code="bear_put_spread",
        name="IV 校准看跌价差",
        ticker=ticker,
        summary=f"若担心跌破 {support:.2f} 支撑，可买入 {long_put_strike:.2f} Put 并卖出 {short_put_strike:.2f} Put 做防守。",
        scenario="趋势破坏或跌破长期均线，需要用有限风险结构做下行保护。",
        rationale="熊市 Put 价差仍适合方向性防守，但净借记和盈亏平衡点改由 Black-Scholes 校准后的权利金决定。",
        spot=spot,
        support=support,
        resistance=resistance,
        max_profit=f"{max_profit_value:.2f} / 股",
        max_loss=f"{net_debit:.2f} / 股",
        breakeven_points=(round(breakeven, 2),),
        legs=(
            StrategyLeg(instrument="put", direction="long", quantity=1, strike=long_put_strike, premium=long_put_premium, label="买入 Put"),
            StrategyLeg(instrument="put", direction="short", quantity=1, strike=short_put_strike, premium=short_put_premium, label="卖出 Put"),
        ),
    )


def _build_long_straddle(
    ticker: str,
    spot: float,
    support: float,
    resistance: float,
    context: VolatilityContext,
) -> OptionStrategy:
    step = _strike_step(spot)
    atm_strike = _round_strike(spot, step, mode="nearest")
    call_premium = _model_option_premium(spot, atm_strike, "call", context)
    put_premium = _model_option_premium(spot, atm_strike, "put", context)
    total_debit = round(call_premium + put_premium, 2)
    lower_breakeven = max(atm_strike - total_debit, 0.0)
    upper_breakeven = atm_strike + total_debit
    downside_profit_cap = max(atm_strike - total_debit, 0.0)
    return OptionStrategy(
        code="long_straddle",
        name="低 IVR 埋伏长跨式",
        ticker=ticker,
        summary=f"IVR {context.iv_rank:.1f}% 处于极低区间，{atm_strike:.2f} 的 ATM 长跨式适合埋伏波动扩张。",
        scenario="波动率被压到低位，市场死水一潭，但你预期后面会出现大幅单边行情或波动扩张。",
        rationale="当 IVR < 20% 时，买波动比卖波动更合理，Long Straddle 可以同时押注方向突破和 IV 抬升。",
        spot=spot,
        support=support,
        resistance=resistance,
        max_profit=f"上行理论无限 / 下行上限 {downside_profit_cap:.2f} / 股",
        max_loss=f"{total_debit:.2f} / 股",
        breakeven_points=(round(lower_breakeven, 2), round(upper_breakeven, 2)),
        legs=(
            StrategyLeg(instrument="call", direction="long", quantity=1, strike=atm_strike, premium=call_premium, label="买入 ATM Call"),
            StrategyLeg(instrument="put", direction="long", quantity=1, strike=atm_strike, premium=put_premium, label="买入 ATM Put"),
        ),
    )


def _build_short_strangle(
    ticker: str,
    spot: float,
    support: float,
    resistance: float,
    context: VolatilityContext,
) -> OptionStrategy:
    short_put_strike = _resolve_sigma_strike(spot, "put", context, scale=1.0)
    short_call_strike = _resolve_sigma_strike(spot, "call", context, scale=1.0)
    put_premium = _model_option_premium(spot, short_put_strike, "put", context)
    call_premium = _model_option_premium(spot, short_call_strike, "call", context)
    total_credit = round(put_premium + call_premium, 2)
    lower_breakeven = max(short_put_strike - total_credit, 0.0)
    upper_breakeven = short_call_strike + total_credit
    downside_floor = max(lower_breakeven, 0.0)
    return OptionStrategy(
        code="short_strangle",
        name="高 IVR 卖出宽跨式",
        ticker=ticker,
        summary=f"IVR {context.iv_rank:.1f}% 已极高，优先卖出 {short_put_strike:.2f} Put + {short_call_strike:.2f} Call 收取高权利金。",
        scenario="期权极贵、市场情绪极端，预期未来是波动回落而不是继续失控扩张。",
        rationale="当 IVR > 80% 时，净买入期权通常性价比很差，卖出宽跨式更直接地收取高估时间价值和波动溢价。",
        spot=spot,
        support=support,
        resistance=resistance,
        max_profit=f"{total_credit:.2f} / 股",
        max_loss=f"上行理论无限 / 下行至 0 仍有 {downside_floor:.2f} / 股风险暴露",
        breakeven_points=(round(lower_breakeven, 2), round(upper_breakeven, 2)),
        legs=(
            StrategyLeg(instrument="put", direction="short", quantity=1, strike=short_put_strike, premium=put_premium, label="卖出 OTM Put"),
            StrategyLeg(instrument="call", direction="short", quantity=1, strike=short_call_strike, premium=call_premium, label="卖出 OTM Call"),
        ),
    )


def _build_iron_condor(
    ticker: str,
    spot: float,
    support: float,
    resistance: float,
    context: VolatilityContext,
) -> OptionStrategy:
    step = _strike_step(spot)
    short_put_strike = _resolve_sigma_strike(spot, "put", context, scale=0.85)
    short_call_strike = _resolve_sigma_strike(spot, "call", context, scale=0.85)
    wing_width = max(step * 3.0, _round_strike(spot * context.expected_move_pct * 0.35, step, mode="nearest"))
    long_put_strike = max(_round_strike(short_put_strike - wing_width, step, mode="down"), step)
    long_call_strike = _round_strike(short_call_strike + wing_width, step, mode="up")
    if long_put_strike >= short_put_strike:
        long_put_strike = max(_round_strike(short_put_strike - step, step, mode="down"), step)
    if long_call_strike <= short_call_strike:
        long_call_strike = _round_strike(short_call_strike + step, step, mode="up")

    short_put_premium = _model_option_premium(spot, short_put_strike, "put", context)
    long_put_premium = _model_option_premium(spot, long_put_strike, "put", context)
    short_call_premium = _model_option_premium(spot, short_call_strike, "call", context)
    long_call_premium = _model_option_premium(spot, long_call_strike, "call", context)

    total_credit = round(short_put_premium + short_call_premium - long_put_premium - long_call_premium, 2)
    widest_spread = max(short_put_strike - long_put_strike, long_call_strike - short_call_strike, 0.0)
    max_loss_value = max(widest_spread - total_credit, 0.0)
    lower_breakeven = max(short_put_strike - total_credit, 0.0)
    upper_breakeven = short_call_strike + total_credit
    return OptionStrategy(
        code="iron_condor",
        name="高 IVR 铁鹰组合",
        ticker=ticker,
        summary=f"IVR {context.iv_rank:.1f}% 偏极端，若想把裸卖风险封顶，可改用铁鹰在 {short_put_strike:.2f} / {short_call_strike:.2f} 两侧收信用金。",
        scenario="高 IV 环境下希望卖波动，但又不愿承担裸卖宽跨式的无限风险。",
        rationale="铁鹰本质是定义风险版 Short Strangle，同样受益于时间流逝和 IV 回落，更适合公开展示和实盘风控。",
        spot=spot,
        support=support,
        resistance=resistance,
        max_profit=f"{total_credit:.2f} / 股",
        max_loss=f"{max_loss_value:.2f} / 股",
        breakeven_points=(round(lower_breakeven, 2), round(upper_breakeven, 2)),
        legs=(
            StrategyLeg(instrument="put", direction="long", quantity=1, strike=long_put_strike, premium=long_put_premium, label="买入保护 Put"),
            StrategyLeg(instrument="put", direction="short", quantity=1, strike=short_put_strike, premium=short_put_premium, label="卖出 Put"),
            StrategyLeg(instrument="call", direction="short", quantity=1, strike=short_call_strike, premium=short_call_premium, label="卖出 Call"),
            StrategyLeg(instrument="call", direction="long", quantity=1, strike=long_call_strike, premium=long_call_premium, label="买入保护 Call"),
        ),
    )


def _dedupe_strategies(strategies: list[OptionStrategy]) -> list[OptionStrategy]:
    deduped: list[OptionStrategy] = []
    seen: set[str] = set()
    for strategy in strategies:
        if strategy.code in seen:
            continue
        seen.add(strategy.code)
        deduped.append(strategy)
    return deduped


def suggest_option_strategies(
    ticker: str,
    spot: float,
    support: float,
    resistance: float,
    *,
    has_shares: bool = False,
    breakdown_risk: bool = False,
    event_risk: bool = False,
    implied_move_pct: float | None = None,
    historical_volatility: float | None = None,
    iv_history: list[float] | np.ndarray | None = None,
    market_option_snapshots: list[dict[str, Any]] | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    dividend_yield: float = 0.0,
    days_to_expiry: float = DEFAULT_DAYS_TO_EXPIRY,
) -> dict[str, Any]:
    resolved_spot = float(max(spot, 0.01))
    resolved_support = float(min(support, resolved_spot * 0.995)) if support >= resolved_spot else float(max(support, 0.01))
    resolved_resistance = float(max(resistance, resolved_spot * 1.01))
    resolved_implied_move_pct = float(max(implied_move_pct or abs(resolved_resistance - resolved_support) / max(resolved_spot, 1e-6) / 2.0, 0.04))

    context = _build_volatility_context(
        resolved_spot,
        implied_move_pct=resolved_implied_move_pct,
        historical_volatility=historical_volatility,
        iv_history=iv_history,
        market_option_snapshots=market_option_snapshots,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        days_to_expiry=days_to_expiry,
    )

    if context.iv_rank >= HIGH_IVR_THRESHOLD:
        strategies = _dedupe_strategies(
            [
                _build_short_strangle(ticker, resolved_spot, resolved_support, resolved_resistance, context),
                _build_iron_condor(ticker, resolved_spot, resolved_support, resolved_resistance, context),
            ]
        )
        primary_code = "iron_condor" if event_risk or breakdown_risk else "short_strangle"
        regime_reason = "IVR > 80%，期权权利金处于极贵区间，禁止净买入期权，优先建议卖波动结构。"
    elif context.iv_rank <= LOW_IVR_THRESHOLD:
        strategies = [_build_long_straddle(ticker, resolved_spot, resolved_support, resolved_resistance, context)]
        primary_code = "long_straddle"
        regime_reason = "IVR < 20%，期权权利金处于极便宜区间，优先建议买入长跨式埋伏波动扩张。"
    else:
        strategies: list[OptionStrategy] = []
        if has_shares and resolved_resistance > resolved_spot:
            strategies.append(_build_covered_call(ticker, resolved_spot, resolved_support, resolved_resistance, context))
        if breakdown_risk:
            strategies.append(_build_bear_put_spread(ticker, resolved_spot, resolved_support, resolved_resistance, context))
        if event_risk and context.current_iv <= 0.25:
            strategies.append(_build_long_straddle(ticker, resolved_spot, resolved_support, resolved_resistance, context))
        if context.iv_rank >= 60.0 or not strategies:
            strategies.append(_build_iron_condor(ticker, resolved_spot, resolved_support, resolved_resistance, context))
        strategies = _dedupe_strategies(strategies)
        primary_code = strategies[0].code
        regime_reason = "IVR 位于中间区间，按持仓方向、趋势破坏和事件风险选择有限风险或中性卖波动结构。"

    return {
        "ticker": ticker,
        "spot": round(resolved_spot, 2),
        "support": round(resolved_support, 2),
        "resistance": round(resolved_resistance, 2),
        "implied_move_pct": round(resolved_implied_move_pct, 6),
        "current_iv": round(context.current_iv, 6),
        "current_iv_pct": round(context.current_iv * 100.0, 2),
        "current_iv_source": context.current_iv_source,
        "iv_rank": round(context.iv_rank, 2),
        "iv_rank_method": context.iv_rank_method,
        "historical_iv_low": round(context.historical_iv_low, 6),
        "historical_iv_high": round(context.historical_iv_high, 6),
        "risk_free_rate": round(context.risk_free_rate, 6),
        "dividend_yield": round(context.dividend_yield, 6),
        "time_to_expiry_years": round(context.time_to_expiry, 6),
        "days_to_expiry": round(context.days_to_expiry, 2),
        "expected_move_pct": round(context.expected_move_pct, 6),
        "expected_move_lower": round(context.expected_move_lower, 2),
        "expected_move_upper": round(context.expected_move_upper, 2),
        "volatility_regime": context.volatility_regime,
        "iv_regime_reason": regime_reason,
        "options_purchase_blocked": bool(context.iv_rank >= HIGH_IVR_THRESHOLD),
        "has_shares": bool(has_shares),
        "breakdown_risk": bool(breakdown_risk),
        "event_risk": bool(event_risk),
        "primary_strategy": primary_code,
        "strategies": [_serialize_strategy(strategy, primary_code, context) for strategy in strategies],
    }


def _leg_payoff(leg: StrategyLeg, prices: np.ndarray, reference_spot: float) -> np.ndarray:
    direction = 1.0 if leg.direction == "long" else -1.0
    quantity = float(max(leg.quantity, 1))
    if leg.instrument == "stock":
        entry_price = float(leg.entry_price if leg.entry_price is not None else reference_spot)
        return direction * quantity * (prices - entry_price)

    strike = float(leg.strike or 0.0)
    premium = float(leg.premium or 0.0)
    if leg.instrument == "call":
        intrinsic = np.maximum(prices - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - prices, 0.0)
    return direction * quantity * (intrinsic - premium)


def _find_breakevens(prices: np.ndarray, payoff: np.ndarray) -> list[float]:
    breakevens: list[float] = []
    close_to_zero = np.where(np.isclose(payoff, 0.0, atol=1e-6))[0]
    for index in close_to_zero.tolist():
        breakevens.append(float(prices[index]))

    sign_change = np.where(np.signbit(payoff[:-1]) != np.signbit(payoff[1:]))[0]
    for index in sign_change.tolist():
        left_x = float(prices[index])
        right_x = float(prices[index + 1])
        left_y = float(payoff[index])
        right_y = float(payoff[index + 1])
        if abs(right_y - left_y) <= 1e-12:
            breakevens.append(left_x)
            continue
        crossing = left_x - left_y * (right_x - left_x) / (right_y - left_y)
        breakevens.append(float(crossing))

    return sorted({round(value, 2) for value in breakevens if value >= 0.0})


def _resolve_payoff_range(structure: dict[str, Any], reference_spot: float) -> tuple[float, float]:
    current_iv = _coerce_float(structure.get("current_iv"))
    time_to_expiry = _coerce_float(structure.get("time_to_expiry_years"))

    if current_iv is not None and current_iv > 0.0 and time_to_expiry is not None and time_to_expiry > 0.0:
        sigma_move = current_iv * math.sqrt(time_to_expiry)
        lower_bound = max(reference_spot * math.exp(-sigma_move), 0.01)
        upper_bound = max(reference_spot * math.exp(sigma_move), lower_bound * 1.01)
    else:
        support = float(structure.get("support") or 0.0)
        resistance = float(structure.get("resistance") or reference_spot)
        lower_bound = max(min(support, reference_spot * 0.7), 0.01)
        upper_bound = max(reference_spot * 1.3, resistance * 1.15, lower_bound * 1.05)

    strikes = [float(leg.get("strike")) for leg in list(structure.get("legs") or []) if _coerce_float(leg.get("strike")) is not None]
    breakevens = [float(value) for value in list(structure.get("breakeven_points") or []) if _coerce_float(value) is not None]
    display_points = [value for value in [*strikes, *breakevens, lower_bound, upper_bound] if value > 0.0]
    lower_bound = max(min(lower_bound, min(display_points) * 0.98), 0.01)
    upper_bound = max(upper_bound, max(display_points) * 1.02)
    if upper_bound <= lower_bound:
        upper_bound = lower_bound * 1.1
    return lower_bound, upper_bound


def plot_option_payoff(structure: dict[str, Any], spot: float | None = None, title: str | None = None) -> go.Figure:
    legs = tuple(StrategyLeg(**leg) for leg in structure.get("legs") or [])
    reference_spot = float(spot or structure.get("spot") or 0.0)
    lower_bound, upper_bound = _resolve_payoff_range(structure, reference_spot)
    prices = np.linspace(lower_bound, upper_bound, 320)
    payoff = np.zeros_like(prices)
    for leg in legs:
        payoff += _leg_payoff(leg, prices, reference_spot)

    breakevens = structure.get("breakeven_points") or _find_breakevens(prices, payoff)
    max_profit_index = int(np.argmax(payoff))
    max_loss_index = int(np.argmin(payoff))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prices,
            y=payoff,
            mode="lines",
            name="净盈亏",
            line={"color": "#0f766e", "width": 3},
            hovertemplate="到期价格: %{x:.2f}<br>净盈亏: %{y:.2f}<extra></extra>",
        )
    )
    figure.add_hline(y=0.0, line_dash="dot", line_color="#334155")
    figure.add_vline(x=reference_spot, line_dash="dash", line_color="#64748b", annotation_text="现价", annotation_position="top")
    expected_move_lower = _coerce_float(structure.get("expected_move_lower"))
    expected_move_upper = _coerce_float(structure.get("expected_move_upper"))
    if expected_move_lower is not None:
        figure.add_vline(x=expected_move_lower, line_dash="dot", line_color="#ef4444", annotation_text="-1σ", annotation_position="bottom")
    if expected_move_upper is not None:
        figure.add_vline(x=expected_move_upper, line_dash="dot", line_color="#2563eb", annotation_text="+1σ", annotation_position="bottom")
    for breakeven in breakevens:
        figure.add_vline(x=float(breakeven), line_dash="dash", line_color="#16a34a", annotation_text="盈亏平衡点", annotation_position="bottom")

    figure.add_annotation(
        x=float(prices[max_profit_index]),
        y=float(payoff[max_profit_index]),
        text=f"最大盈利：{structure.get('max_profit')}",
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(255,255,255,0.92)",
    )
    figure.add_annotation(
        x=float(prices[max_loss_index]),
        y=float(payoff[max_loss_index]),
        text=f"最大亏损：{structure.get('max_loss')}",
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(255,255,255,0.92)",
    )
    figure.update_layout(
        title=title or str(structure.get("name") or "期权到期盈亏图"),
        template="plotly_white",
        xaxis_title="到期价格",
        yaxis_title="净盈亏",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return figure


def build_hk_yingli_execution_guide(structure: dict[str, Any]) -> dict[str, list[str] | str]:
    strategy_name = str(structure.get("name") or "期权策略")
    legs = list(structure.get("legs") or [])
    put_legs = [leg for leg in legs if str(leg.get("instrument") or "") == "put"]
    call_legs = [leg for leg in legs if str(leg.get("instrument") or "") == "call"]
    short_call = next((leg for leg in call_legs if str(leg.get("direction") or "") == "short"), None)
    long_call = next((leg for leg in call_legs if str(leg.get("direction") or "") == "long"), None)
    short_put = next((leg for leg in put_legs if str(leg.get("direction") or "") == "short"), None)
    long_put = next((leg for leg in put_legs if str(leg.get("direction") or "") == "long"), None)

    preflight = [
        "确认香港盈立证券账户已开通美股期权权限，并且账户有足够美元购买力或保证金额度。",
        "确认标的是美股期权，不要误下港股期权；美股标准合约默认 1 张 = 100 股。",
        "优先使用限价单，不直接追市价；先核对到期日、执行价、方向和张数，再提交。",
    ]
    execution_steps: list[str]
    position_management: list[str]

    code = str(structure.get("code") or "")
    if code == "covered_call" and short_call is not None:
        strike = float(short_call.get("strike") or 0.0)
        premium = float(short_call.get("premium") or 0.0)
        execution_steps = [
            "在香港盈立证券 App 或桌面端搜索标的代码，进入美股期权链，切到与你计划一致的到期日。",
            f"确认账户里对应正股至少有 100 股后，选择 {strike:.2f} 的 Call，方向选 卖出开仓 / Sell to Open。",
            f"委托方式选限价，首挂可参考买卖盘中间价附近，权利金参考 {premium:.2f} 美元/股。",
            "若券商显示该单会被识别为 Covered Call，再提交；若没有被识别为备兑，先取消，避免裸卖 Call。",
        ]
        position_management = [
            "若股价快速逼近执行价，优先考虑买回该 Call 并滚动到更远到期日或更高执行价。",
            "若你愿意按执行价卖出正股，可以持有到期；若不愿被行权，就不要拖到临近到期才处理。",
        ]
    elif code == "bear_put_spread" and long_put is not None and short_put is not None:
        long_strike = float(long_put.get("strike") or 0.0)
        short_strike = float(short_put.get("strike") or 0.0)
        execution_steps = [
            "在香港盈立证券期权链里先找到同一到期日的两条 Put 腿，确认高执行价买入、低执行价卖出。",
            f"若界面支持组合单，直接选择 Bear Put Spread：买入 {long_strike:.2f} Put，同时卖出 {short_strike:.2f} Put。",
            f"若界面不支持组合单，先下 买入开仓 / Buy to Open {long_strike:.2f} Put，再下 卖出开仓 / Sell to Open {short_strike:.2f} Put。",
            f"整组委托尽量按净借记控制，最大亏损就是净权利金支出，当前预言机给出的上限是 {structure.get('max_loss') or '见图表'}。",
        ]
        position_management = [
            "若组合浮盈已接近最大盈利的 50% 到 70%，可以优先止盈，不必死拿到期。",
            "若标的重新站回关键阻力、下跌逻辑失效，优先整组平仓，不要只留一条腿裸奔。",
        ]
    elif code == "long_straddle" and long_put is not None and long_call is not None:
        put_strike = float(long_put.get("strike") or 0.0)
        call_strike = float(long_call.get("strike") or 0.0)
        execution_steps = [
            "在香港盈立证券期权链里选择同一到期日、同一执行价附近的 ATM Put 和 ATM Call。",
            f"若支持组合单，直接下 Long Straddle；若不支持，则分别用 买入开仓 / Buy to Open 买入 {put_strike:.2f} Put 和 {call_strike:.2f} Call。",
            f"两腿都用限价单，总权利金支出最好贴近预言机的最大亏损 {structure.get('max_loss') or '见图表'}。",
            "该结构本质是买波动，若事件后价格不动且 IV 回落，需要快速止损，不要把时间价值耗尽。",
        ]
        position_management = [
            "若任一方向出现快速单边走势，可优先止盈盈利腿，并评估是否保留另一腿继续博弈。",
            "若波动扩张兑现但方向尚未完全走完，可考虑部分减仓，而不是机械等到到期。",
        ]
    elif code == "short_strangle" and short_put is not None and short_call is not None:
        put_strike = float(short_put.get("strike") or 0.0)
        call_strike = float(short_call.get("strike") or 0.0)
        execution_steps = [
            "在香港盈立证券期权链中选择同一到期日的 OTM Put 和 OTM Call，先确认账户保证金足够承受裸卖风险。",
            f"若界面支持策略单，直接下 Short Strangle；若不支持，则分别用 卖出开仓 / Sell to Open 卖出 {put_strike:.2f} Put 和 {call_strike:.2f} Call。",
            "所有委托优先用限价单，成交前务必再次核对净信用金、保证金占用和最坏情形。",
            "这是高 IVR 卖波动结构，若券商风控提示保证金不足或风险过高，应优先改用铁鹰，而不是硬做裸卖。",
        ]
        position_management = [
            "若标的逼近任一短腿执行价，优先处理受测一侧，可选择回补、滚动或改造成定义风险结构。",
            "若信用金利润达到 40% 到 60%，通常应主动止盈，不要贪到最后一分时间价值。",
        ]
    elif code == "iron_condor" and long_put is not None and short_put is not None and short_call is not None and long_call is not None:
        long_put_strike = float(long_put.get("strike") or 0.0)
        short_put_strike = float(short_put.get("strike") or 0.0)
        short_call_strike = float(short_call.get("strike") or 0.0)
        long_call_strike = float(long_call.get("strike") or 0.0)
        execution_steps = [
            "在香港盈立证券期权链中选择同一到期日的四条腿，构造成一个 Put Credit Spread 和一个 Call Credit Spread。",
            f"若支持组合单，直接下 Iron Condor：买入 {long_put_strike:.2f} Put，卖出 {short_put_strike:.2f} Put；卖出 {short_call_strike:.2f} Call，买入 {long_call_strike:.2f} Call。",
            "若界面不支持四腿组合单，先下认购价差，再下认沽价差，保证四条腿最终都在同一到期日。",
            f"铁鹰是定义风险版卖波动，最大亏损已封顶为 {structure.get('max_loss') or '见图表'}，但仍要检查净信用金和保证金冻结。",
        ]
        position_management = [
            "若价格持续停留在两条短腿之间，可在利润达到 40% 到 60% 时主动止盈，减少尾部跳空风险。",
            "若单侧短腿被快速测试，不要死扛，优先平掉受测侧或整体减仓。",
        ]
    else:
        execution_steps = [
            "在香港盈立证券里搜索标的并进入美股期权链，确认到期日、执行价、方向和张数后，用限价单提交。",
            "若该策略包含多条腿，优先用组合单；若券商前端不支持，再按每条腿的方向分别下单。",
        ]
        position_management = [
            "成交后继续盯隐含波动率、时间价值和标的趋势，不要只看方向是否判断正确。",
        ]

    return {
        "title": f"{strategy_name} · 香港盈立证券具体下单步骤",
        "preflight": preflight,
        "execution_steps": execution_steps,
        "position_management": position_management,
    }