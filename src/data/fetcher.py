"""
Data Fetcher - Download OHLCV data from Binance via CCXT
Stores 1-minute data as parquet for future flexibility.
"""
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    RAW_DATA_DIR, SYMBOL, TIMEFRAME_RAW, LOOKBACK_DAYS,
    BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET,
    RAW_OUTLIER_Z_THRESHOLD, WICK_RATIO_THRESHOLD
)


def get_exchange(use_testnet: bool = False):
    """Initialize Binance exchange connection."""
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY if use_testnet else '',
        'secret': BINANCE_SECRET_KEY if use_testnet else '',
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Only use testnet for trading, not for fetching historical data
    if use_testnet:
        exchange.set_sandbox_mode(True)
    
    return exchange


def _filter_raw_noise(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Remove obvious raw-market spikes and malformed candles."""
    df = df.copy().sort_index()

    # Remove candles with excessive wicks relative to candle body.
    body = (df['close'] - df['open']).abs()
    candle_range = (df['high'] - df['low']).abs()
    wick_ratio = candle_range / (body + 1e-8)
    df = df[(wick_ratio <= (1.0 / max(WICK_RATIO_THRESHOLD, 1e-6))) | body.isna()]

    # Use returns-based robust filtering to catch single-candle spikes.
    returns = df['close'].pct_change()
    rolling_median = returns.rolling(window=window, min_periods=max(10, window // 4)).median()
    rolling_mad = (returns - rolling_median).abs().rolling(window=window, min_periods=max(10, window // 4)).median()
    robust_z = (returns - rolling_median).abs() / (1.4826 * rolling_mad + 1e-8)
    df = df[(robust_z.isna()) | (robust_z <= RAW_OUTLIER_Z_THRESHOLD)]

    # Keep volumes sane.
    df['volume'] = df['volume'].clip(lower=0)
    return df


def fetch_ohlcv(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME_RAW,
    days: int = LOOKBACK_DAYS,
    save: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1m', '15m', '1h')
        days: Number of days of historical data
        save: Whether to save to parquet
        
    Returns:
        DataFrame with OHLCV data
    """
    # Use public API (no testnet) for historical data
    exchange = get_exchange(use_testnet=False)
    
    # Calculate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    since = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    # Binance limits: 1000 candles per request
    all_candles = []
    current_since = since
    
    # Calculate expected number of candles
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }
    minutes = timeframe_minutes.get(timeframe, 1)
    total_candles = (days * 24 * 60) // minutes
    
    print(f"Fetching {symbol} {timeframe} data for {days} days (~{total_candles:,} candles)")
    
    with tqdm(total=total_candles, desc="Downloading") as pbar:
        while current_since < end_ms:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                pbar.update(len(candles))
                
                # Move to next batch
                current_since = candles[-1][0] + 1
                
                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)
                
            except ccxt.NetworkError as e:
                print(f"Network error: {e}, retrying...")
                time.sleep(5)
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {e}")
                break
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    if df.empty:
        raise ValueError("No candles returned from exchange")

    # Basic data sanity: coerce numeric fields and remove malformed rows
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.dropna(how='any')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)

    # Remove impossible candles and cap negative volume
    df = df[(df['high'] >= df[['open', 'close', 'low']].max(axis=1)) &
            (df['low'] <= df[['open', 'close', 'high']].min(axis=1))]
    df['volume'] = df['volume'].clip(lower=0)
    df = df.dropna(how='any')

    # Filter out obvious raw-market noise spikes
    df = _filter_raw_noise(df)

    if df.empty:
        raise ValueError("All fetched candles were removed by sanity checks")
    
    print(f"Downloaded {len(df):,} candles from {df.index[0]} to {df.index[-1]}")
    
    # Save to parquet
    if save:
        filename = f"{symbol.replace('/', '_')}_{timeframe}_{days}d.parquet"
        filepath = RAW_DATA_DIR / filename
        df.to_parquet(filepath)
        print(f"Saved to {filepath}")
    
    return df


def load_raw_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME_RAW) -> pd.DataFrame:
    """Load raw data from parquet file."""
    pattern = f"{symbol.replace('/', '_')}_{timeframe}_*.parquet"
    files = list(RAW_DATA_DIR.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No data files found matching {pattern}")
    
    # Load most recent file
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"Loading {latest_file}")
    
    return pd.read_parquet(latest_file)


if __name__ == "__main__":
    # Test fetch
    df = fetch_ohlcv(days=7)  # Start with 7 days for testing
    print(df.head())
    print(df.tail())
