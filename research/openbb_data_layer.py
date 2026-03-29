from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class DataLayerConfig:
    primary_data_source: str = "openbb"
    fallback_data_sources: tuple[str, ...] = ("yfinance",)
    openbb_provider: Optional[str] = None
    download_period: str = "10y"
    start_date: Optional[str] = None
    min_history_days: int = 180
    max_missing_ratio: float = 0.20
    max_staleness_days: int = 14


def normalize_weights(w: pd.Series, index: pd.Index) -> pd.Series:
    w = pd.Series(w).reindex(index).fillna(0.0).astype(float)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return pd.Series(1 / len(index), index=index)
    return w / s


def coerce_openbb_output_to_series(output, symbol: str) -> Optional[pd.Series]:
    if output is None:
        return None

    df = None
    if isinstance(output, pd.Series):
        s = output.copy()
        s.name = symbol
        return s
    if isinstance(output, pd.DataFrame):
        df = output.copy()
    elif hasattr(output, "to_dataframe"):
        df = output.to_dataframe()
    elif hasattr(output, "to_df"):
        df = output.to_df()
    elif hasattr(output, "results"):
        try:
            df = pd.DataFrame(output.results)
        except Exception:
            df = None

    if df is None or len(df) == 0:
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        date_candidates = [c for c in df.columns if str(c).lower() in ["date", "datetime", "timestamp"]]
        if date_candidates:
            df[date_candidates[0]] = pd.to_datetime(df[date_candidates[0]])
            df = df.set_index(date_candidates[0])

    close_candidates = [c for c in df.columns if str(c).lower() in ["close", "adj_close", "adjusted_close"]]
    if len(close_candidates) == 0:
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) == 0:
            return None
        close_col = numeric_cols[0]
    else:
        close_col = close_candidates[0]

    s = pd.Series(df[close_col]).dropna()
    s.name = symbol
    return s if len(s) else None


def validate_series(s: Optional[pd.Series], symbol: str, config: DataLayerConfig) -> Tuple[Optional[pd.Series], List[str]]:
    issues: List[str] = []
    if s is None or len(s) == 0:
        return None, ["empty"]

    s = pd.Series(s).copy()
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    s = s.replace([np.inf, -np.inf], np.nan)

    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            issues.append("bad_index")

    s = s.dropna()

    if len(s) < config.min_history_days:
        issues.append("short_history")

    if not s.index.is_monotonic_increasing:
        issues.append("non_monotonic_index")

    if len(s) > 1:
        full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
        missing_ratio = 1 - (len(s.index.unique()) / max(len(full_range), 1))
        if missing_ratio > config.max_missing_ratio:
            issues.append(f"missing_ratio>{config.max_missing_ratio:.0%}")

    if len(s) and (pd.Timestamp.today().normalize() - s.index.max().normalize()).days > config.max_staleness_days:
        issues.append("stale")

    return s, issues


def fetch_series_from_yfinance(symbol: str, config: DataLayerConfig) -> Optional[pd.Series]:
    df = yf.download(symbol, period=config.download_period, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        close = df.xs("Close", axis=1, level=0) if "Close" in level0 else df.iloc[:, :1]
    else:
        close = df[["Close"]] if "Close" in df.columns else df.iloc[:, :1]

    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            return None
        close = close.iloc[:, 0]

    s = pd.Series(close).dropna()
    s.name = symbol
    return s if len(s) else None


def fetch_series_from_openbb(symbol: str, config: DataLayerConfig) -> Optional[pd.Series]:
    try:
        from openbb import obb
    except Exception:
        return None

    attempts = []
    if config.start_date is not None and config.openbb_provider:
        attempts.append(lambda: obb.equity.price.historical(symbol, start_date=config.start_date, provider=config.openbb_provider))
    if config.start_date is not None:
        attempts.append(lambda: obb.equity.price.historical(symbol, start_date=config.start_date))
    if config.openbb_provider:
        attempts.append(lambda: obb.equity.price.historical(symbol, provider=config.openbb_provider))
    attempts.append(lambda: obb.equity.price.historical(symbol))

    for fn in attempts:
        try:
            output = fn()
            s = coerce_openbb_output_to_series(output, symbol)
            if s is not None and len(s):
                return s
        except Exception:
            continue

    return None


def fetch_series(symbol: str, config: DataLayerConfig) -> Tuple[Optional[pd.Series], Optional[str], List[str]]:
    provider_order = [config.primary_data_source] + [p for p in config.fallback_data_sources if p != config.primary_data_source]
    errors: List[str] = []

    for provider in provider_order:
        s = None
        if provider == "openbb":
            s = fetch_series_from_openbb(symbol, config)
        elif provider == "yfinance":
            s = fetch_series_from_yfinance(symbol, config)
        else:
            errors.append(f"unknown_provider:{provider}")
            continue

        s, issues = validate_series(s, symbol, config)
        if s is not None and len(s):
            return s, provider, issues
        errors.extend(issues)

    return None, None, list(dict.fromkeys(errors))


def download_market_series(
    stock_tickers: List[str],
    fx_ticker: str,
    gold_usd_ticker: str,
    asset_labels: Dict[str, str],
    config: DataLayerConfig,
) -> Tuple[Dict[str, Optional[pd.Series]], pd.DataFrame]:
    rows = []
    downloaded: Dict[str, Optional[pd.Series]] = {}

    symbols = stock_tickers + [fx_ticker, gold_usd_ticker]
    for symbol in symbols:
        s, provider_used, issues = fetch_series(symbol, config)
        downloaded[symbol] = s
        rows.append({
            "source": symbol,
            "asset": asset_labels.get(symbol, symbol),
            "provider_used": provider_used,
            "n": 0 if s is None else len(s),
            "start": None if s is None else str(s.index.min().date()),
            "end": None if s is None else str(s.index.max().date()),
            "issues": ", ".join(issues) if issues else "",
            "status": "ok" if s is not None and len(s) else "empty",
        })

    return downloaded, pd.DataFrame(rows)


def build_gold_try(downloaded: Dict[str, Optional[pd.Series]], fx_ticker: str, gold_usd_ticker: str, asset_labels: Dict[str, str], config: DataLayerConfig) -> Optional[pd.Series]:
    gold_usd = downloaded.get(gold_usd_ticker)
    usdtry = downloaded.get(fx_ticker)
    if gold_usd is None or usdtry is None:
        return None
    gold_try = (gold_usd * usdtry).dropna()
    gold_try.name = asset_labels["GOLD_TRY"]
    gold_try, _issues = validate_series(gold_try, "GOLD_TRY", config)
    return gold_try if gold_try is not None and len(gold_try) else None


def prepare_price_matrix(
    downloaded: Dict[str, Optional[pd.Series]],
    gold_try: Optional[pd.Series],
    stock_tickers: List[str],
    fx_ticker: str,
    asset_labels: Dict[str, str],
    min_history_days: int,
    target_test_months: int,
    target_lookback_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    series = []
    rows = []

    for symbol in stock_tickers + [fx_ticker]:
        s = downloaded.get(symbol)
        eligible = s is not None and len(s) >= min_history_days
        if eligible:
            s = s.copy()
            s.name = asset_labels[symbol]
            series.append(s)
        rows.append({"asset": asset_labels.get(symbol, symbol), "n": 0 if s is None else len(s), "eligible": bool(eligible)})

    gold_eligible = gold_try is not None and len(gold_try) >= min_history_days
    if gold_eligible:
        series.append(gold_try)
    rows.append({"asset": asset_labels["GOLD_TRY"], "n": 0 if gold_try is None else len(gold_try), "eligible": bool(gold_eligible)})

    eligible_assets = pd.DataFrame(rows)

    if len(series) < 3:
        raise ValueError("Yeterli tarihçesi olan en az 3 varlık yok.")

    prices = pd.concat(series, axis=1).sort_index().ffill()
    common_start = prices.apply(lambda s: s.first_valid_index()).max()
    prices = prices.loc[common_start:].dropna()

    monthly_points = prices.resample("MS").first().dropna()
    if len(monthly_points) < target_test_months + 1:
        raise ValueError(f"Yeterli ortak tarihçe yok. Aylık nokta sayısı: {len(monthly_points)}")

    lookback = min(target_lookback_days, max(60, len(prices) - 40))
    months = target_test_months
    return prices, eligible_assets, lookback, months
