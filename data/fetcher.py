"""
Data Fetcher - Downloads OHLCV data from Binance (no API key needed for public data).
Uses CCXT library. Falls back to synthetic data for offline testing.
"""
import ccxt
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1h",
                days: int = 365, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance.
    Returns DataFrame with columns: open, high, low, close, volume
    """
    cache_file = DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_{days}d.parquet"

    if use_cache and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 3600:  # Cache for 1 hour
            print(f"[Cache] Loading {symbol} {timeframe} from cache...")
            return pd.read_parquet(cache_file)

    print(f"[Fetch] Downloading {symbol} {timeframe} ({days} days) from Binance...")

    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_candles = []

        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < 1000:
                break
            time.sleep(0.1)

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        df = df[~df.index.duplicated(keep='last')].sort_index()

        df.to_parquet(cache_file)
        print(f"[Fetch] Downloaded {len(df):,} candles. Saved to cache.")
        return df

    except Exception as e:
        print(f"[Fetch] Error fetching live data: {e}")
        print("[Fetch] Generating synthetic BTC/USDT data for offline testing...")
        return _generate_synthetic_data(days=days, timeframe=timeframe)


def _generate_synthetic_data(days: int = 365, timeframe: str = "1h") -> pd.DataFrame:
    """Generate realistic synthetic BTC/USDT data for testing."""
    tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    mins = tf_minutes.get(timeframe, 60)
    n = (days * 24 * 60) // mins

    np.random.seed(42)
    dt = pd.date_range(end=datetime.utcnow(), periods=n, freq=f'{mins}min')

    # Realistic GBM with regime changes
    returns = np.zeros(n)
    price = 30000.0
    prices = [price]

    for i in range(1, n):
        # Regime: trending (60% of time) vs ranging (40%)
        regime = np.sin(i / (n / 10)) * 0.5
        mu = 0.0003 * (1 + regime)
        sigma = 0.015 + 0.005 * abs(regime)

        ret = np.random.normal(mu, sigma)
        ret = np.clip(ret, -0.08, 0.08)
        price *= (1 + ret)
        price = max(price, 1000)
        prices.append(price)

    prices = np.array(prices)

    # Build OHLCV
    close = prices
    volatility = np.abs(np.random.normal(0, 0.008, n))
    high = close * (1 + volatility)
    low = close * (1 - volatility)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.lognormal(10, 1, n) * (1 + volatility * 5)

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume
    }, index=dt)

    return df.astype(float)
