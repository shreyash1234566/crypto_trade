"""
Feature Engineering - Calculate technical indicators for trading.
"""
import pandas as pd
import numpy as np
import ta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import PROCESSED_DATA_DIR, SYMBOL, TIMEFRAME_TRADE


def fractional_diff_ffd(series: pd.Series, d: float = 0.4, thres: float = 1e-5) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differencing (FFD).
    
    Args:
        series: Input time series (e.g., close prices)
        d: Differencing order (0 < d < 1)
        thres: Threshold for weight cutoff
        
    Returns:
        Fractionally differenced series
    """
    # 1. Compute weights
    w = [1.]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        
    w = np.array(w[::-1]).reshape(-1, 1)
    
    # 2. Apply weights using rolling window
    # Fill NaNs to avoid issues
    series_filled = series.ffill().bfill()
    
    # Apply rolling dot product
    result = series_filled.rolling(window=len(w)).apply(
        lambda x: np.dot(x, w)[0], 
        raw=True
    )
    
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and derived features to OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()
    
    # ==========================================================================
    # PRICE FEATURES
    # ==========================================================================
    
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility (20-period rolling std of returns)
    df['volatility'] = df['log_return'].rolling(window=20).std()
    
    # ==========================================================================
    # RSI (Relative Strength Index)
    # ==========================================================================
    df['rsi'] = ta.momentum.RSIIndicator(
        close=df['close'],
        window=14
    ).rsi()
    
    # ==========================================================================
    # MACD (Moving Average Convergence Divergence)
    # ==========================================================================
    macd = ta.trend.MACD(
        close=df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # ==========================================================================
    # BOLLINGER BANDS
    # ==========================================================================
    bb = ta.volatility.BollingerBands(
        close=df['close'],
        window=20,
        window_dev=2
    )
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_pct'] = bb.bollinger_pband()  # %B indicator
    
    # ==========================================================================
    # ADDITIONAL FEATURES
    # ==========================================================================
    
    # Price position relative to range
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # Volume change
    df['volume_change'] = df['volume'].pct_change()
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['price_sma_ratio'] = df['close'] / df['sma_20']
    
    # ==========================================================================
    # FRACTIONAL DIFFERENCING
    # ==========================================================================
    try:
        df['frac_diff'] = fractional_diff_ffd(df['close'], d=0.4)
    except Exception as e:
        print(f"Error calculating fractional diff: {e}")
        df['frac_diff'] = df['log_return'] # Fallback

    # ==========================================================================
    # TARGET (for Bi-LSTM pre-training)
    # ==========================================================================
    
    # Next candle direction (1 = up, 0 = down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaN rows from rolling calculations
    df.dropna(inplace=True)
    
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Normalize features using rolling z-score (prevents look-ahead bias).
    
    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize
        
    Returns:
        DataFrame with normalized features
    """
    df = df.copy()
    window = 60  # Rolling window for normalization
    
    for col in feature_cols:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    return df


def prepare_features(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME_TRADE,
    save: bool = True
) -> pd.DataFrame:
    """
    Load processed data, add features, and save.
    
    Args:
        symbol: Trading pair
        timeframe: Timeframe
        save: Whether to save to file
        
    Returns:
        DataFrame with features
    """
    # Load processed OHLCV data
    filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
    filepath = PROCESSED_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Run resampler first: {filepath}")
    
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} candles")
    
    # Add features
    df = add_features(df)
    print(f"After features: {len(df):,} candles, {len(df.columns)} columns")
    
    # Normalize selected features
    normalize_cols = [
        'log_return', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_pct', 'price_range', 'volume_change', 'price_sma_ratio', 'frac_diff'
    ]
    df = normalize_features(df, normalize_cols)
    
    # Save
    if save:
        output_file = PROCESSED_DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_features.parquet"
        df.to_parquet(output_file)
        print(f"Saved to {output_file}")
    
    return df


def load_featured_data(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME_TRADE
) -> pd.DataFrame:
    """Load data with features."""
    filename = f"{symbol.replace('/', '_')}_{timeframe}_features.parquet"
    filepath = PROCESSED_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Features not found: {filepath}")
    
    return pd.read_parquet(filepath)


def get_feature_columns() -> list:
    """Get list of feature columns for model input."""
    return [
        'open', 'high', 'low', 'close', 'volume',
        'log_return_norm', 'volatility_norm',
        'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'bb_pct_norm', 'price_range_norm', 'volume_change_norm', 'price_sma_ratio_norm',
        'frac_diff_norm'
    ]


if __name__ == "__main__":
    df = prepare_features()
    print("\nFeature columns:")
    print(df.columns.tolist())
    print("\nSample data:")
    print(df.head())
