"""
Data Resampler - Convert 1-minute candles to 15-minute candles.
"""
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SYMBOL, 
    TIMEFRAME_RAW, TIMEFRAME_TRADE,
    RESAMPLE_SMOOTHING_SPAN, WICK_RATIO_THRESHOLD
)


def _filter_resampled_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Remove low-timeframe noise that survives resampling."""
    df = df.copy().sort_index()

    # Reject candles whose wick structure is suspiciously large.
    body = (df['close'] - df['open']).abs()
    candle_range = (df['high'] - df['low']).abs()
    wick_ratio = candle_range / (body + 1e-8)
    df = df[(wick_ratio <= (1.0 / max(WICK_RATIO_THRESHOLD, 1e-6))) | body.isna()]

    # Suppress tiny one-bar spikes after aggregation.
    df['close'] = df['close'].ewm(span=RESAMPLE_SMOOTHING_SPAN, adjust=False).mean()
    return df


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str = TIMEFRAME_TRADE
) -> pd.DataFrame:
    """
    Resample OHLCV data to a larger timeframe.
    
    Args:
        df: DataFrame with OHLCV data (datetime index)
        target_timeframe: Target timeframe ('15m', '1h', etc.)
        
    Returns:
        Resampled DataFrame
    """
    # Map timeframe string to pandas resample rule
    timeframe_map = {
        '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': '1H', '4h': '4H', '1d': '1D'
    }
    
    rule = timeframe_map.get(target_timeframe, '15T')

    # Clean input first to reduce noisy or malformed candles before aggregation
    df = df.copy().sort_index()
    df = df[~df.index.duplicated(keep='first')]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.dropna(how='any')
    
    # Resample using OHLCV aggregation rules
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Drop candles with invalid structure after aggregation
    resampled = resampled[(resampled['high'] >= resampled[['open', 'close', 'low']].max(axis=1)) &
                          (resampled['low'] <= resampled[['open', 'close', 'high']].min(axis=1))]

    # Extra suppression for low-timeframe noise
    resampled = _filter_resampled_noise(resampled)
    resampled = resampled.dropna(how='any')
    
    print(f"Resampled from {len(df):,} to {len(resampled):,} candles ({target_timeframe})")
    
    return resampled


def resample_and_save(
    symbol: str = SYMBOL,
    source_timeframe: str = TIMEFRAME_RAW,
    target_timeframe: str = TIMEFRAME_TRADE
) -> pd.DataFrame:
    """
    Load raw data, resample, and save to processed folder.
    
    Args:
        symbol: Trading pair
        source_timeframe: Source timeframe
        target_timeframe: Target timeframe
        
    Returns:
        Resampled DataFrame
    """
    # Find source file
    pattern = f"{symbol.replace('/', '_')}_{source_timeframe}_*.parquet"
    files = list(RAW_DATA_DIR.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No source files found: {pattern}")
    
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    print(f"Loading {latest_file}")
    
    # Load and resample
    df = pd.read_parquet(latest_file)
    resampled = resample_ohlcv(df, target_timeframe)
    
    # Save
    filename = f"{symbol.replace('/', '_')}_{target_timeframe}.parquet"
    filepath = PROCESSED_DATA_DIR / filename
    resampled.to_parquet(filepath)
    print(f"Saved to {filepath}")
    
    return resampled


def load_processed_data(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME_TRADE
) -> pd.DataFrame:
    """Load processed (resampled) data."""
    filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
    filepath = PROCESSED_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")
    
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    # Test resample
    df = resample_and_save()
    print(df.head())
    print(df.tail())
