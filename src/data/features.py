"""
Feature Engineering - Calculate technical indicators for trading.
"""
import pandas as pd
import numpy as np
import json
import ta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import PROCESSED_DATA_DIR, SYMBOL, TIMEFRAME_TRADE
from config.settings import (
    PRICE_SMOOTHING_SPAN,
    OUTLIER_Z_THRESHOLD,
    NORMALIZATION_CLIP,
    FEATURE_CORR_THRESHOLD,
)


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


def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, and coerce OHLCV data before feature engineering."""
    df = df.copy().sort_index()
    df = df[~df.index.duplicated(keep='first')]

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns])
    return df


def _ema_smooth(series: pd.Series, span: int = PRICE_SMOOTHING_SPAN) -> pd.Series:
    """Apply EMA smoothing to reduce short-term market noise."""
    return series.ewm(span=span, adjust=False).mean()


def _clip_rolling_zscore(series: pd.Series, window: int = 60, z_threshold: float = OUTLIER_Z_THRESHOLD) -> pd.Series:
    """Clip outliers using a rolling z-score style band."""
    rolling_mean = series.rolling(window=window, min_periods=max(10, window // 4)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(10, window // 4)).std()
    upper = rolling_mean + z_threshold * rolling_std
    lower = rolling_mean - z_threshold * rolling_std
    return series.clip(lower=lower, upper=upper)


def _save_feature_columns(symbol: str, timeframe: str, columns: list[str]) -> None:
    """Persist selected feature columns alongside processed data."""
    meta_path = PROCESSED_DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_feature_columns.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(columns, f, indent=2)


def _load_saved_feature_columns(symbol: str = SYMBOL, timeframe: str = TIMEFRAME_TRADE) -> list[str] | None:
    meta_path = PROCESSED_DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_feature_columns.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return list(data)
    except Exception:
        return None


def select_feature_columns(df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    """Select a compact, low-redundancy feature set using correlation pruning."""
    available = [c for c in candidate_cols if c in df.columns]
    if not available:
        return []

    target_corrs = {}
    for col in available:
        if 'target' in df.columns and df[col].std(skipna=True) not in (0, None):
            target_corrs[col] = abs(df[col].corr(df['target']))
        else:
            target_corrs[col] = 0.0

    ranked = sorted(available, key=lambda c: (target_corrs.get(c, 0.0), c), reverse=True)
    selected: list[str] = []

    for col in ranked:
        if col == 'fear_greed_norm':
            selected.append(col)
            continue

        keep = True
        for kept in selected:
            if kept == 'fear_greed_norm':
                continue
            corr = df[col].corr(df[kept])
            if pd.notna(corr) and abs(corr) >= FEATURE_CORR_THRESHOLD:
                keep = False
                break

        if keep:
            selected.append(col)

    return selected


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and derived features to OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with additional feature columns
    """
    df = _sanitize_ohlcv(df)
    df = df.dropna(how='any')

    # ======================================================================
    # PRICE SMOOTHING
    # ======================================================================
    df['close_ema'] = _ema_smooth(df['close'], span=PRICE_SMOOTHING_SPAN)
    df['volume_ema'] = _ema_smooth(df['volume'], span=PRICE_SMOOTHING_SPAN)
    
    # ==========================================================================
    # PRICE FEATURES
    # ==========================================================================
    
    # Log returns on smoothed close to reduce 1-bar noise
    df['log_return'] = np.log(df['close_ema'] / df['close_ema'].shift(1))
    df['log_return'] = _clip_rolling_zscore(df['log_return'])
    
    # Volatility (20-period rolling std of returns)
    df['volatility'] = df['log_return'].rolling(window=20).std()
    
    # ==========================================================================
    # RSI (Relative Strength Index)
    # ==========================================================================
    df['rsi'] = ta.momentum.RSIIndicator(
        close=df['close_ema'],
        window=14
    ).rsi()
    
    # ==========================================================================
    # MACD (Moving Average Convergence Divergence)
    # ==========================================================================
    macd = ta.trend.MACD(
        close=df['close_ema'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['macd'] = macd.macd()
    df['macd_hist'] = macd.macd_diff()
    
    # ==========================================================================
    # BOLLINGER BANDS
    # ==========================================================================
    bb = ta.volatility.BollingerBands(
        close=df['close_ema'],
        window=20,
        window_dev=2
    )
    df['bb_pct'] = bb.bollinger_pband()  # %B indicator
    
    # ==========================================================================
    # ADDITIONAL FEATURES
    # ==========================================================================
    
    # Price position relative to range
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['price_range'] = _clip_rolling_zscore(df['price_range'])
    
    # Volume change
    df['volume_change'] = df['volume_ema'].pct_change()
    df['volume_change'] = _clip_rolling_zscore(df['volume_change'])
    
    # Smoothed trend feature instead of a stack of correlated moving averages
    df['price_ema_ratio'] = df['close'] / df['close_ema']
    
    # ==========================================================================
    # FRACTIONAL DIFFERENCING
    # ==========================================================================
    try:
        df['frac_diff'] = fractional_diff_ffd(df['close_ema'], d=0.4)
    except Exception as e:
        print(f"Error calculating fractional diff: {e}")
        df['frac_diff'] = df['log_return'] # Fallback

    df['frac_diff'] = _clip_rolling_zscore(df['frac_diff'])

    # ==========================================================================
    # TARGET (for Bi-LSTM pre-training)
    # ==========================================================================
    
    # Next candle direction (1 = up, 0 = down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Keep a stable trend proxy for downstream models
    df['price_ema_ratio'] = _clip_rolling_zscore(df['price_ema_ratio'])

    # Drop intermediate helper columns to keep the final dataset compact
    df.drop(columns=['close_ema', 'volume_ema', 'macd'], inplace=True, errors='ignore')
    
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
            rolling_mean = df[col].rolling(window=window, min_periods=max(10, window // 4)).mean()
            rolling_std = df[col].rolling(window=window, min_periods=max(10, window // 4)).std()
            df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            df[f'{col}_norm'] = df[f'{col}_norm'].clip(-NORMALIZATION_CLIP, NORMALIZATION_CLIP)
    
    return df


def prepare_features(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME_TRADE,
    save: bool = True,
    df: pd.DataFrame | None = None
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
    # Load processed OHLCV data if not provided directly
    if df is None:
        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        filepath = PROCESSED_DATA_DIR / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Run resampler first: {filepath}")
        
        df = pd.read_parquet(filepath)
        print(f"Loaded {len(df):,} candles")
    else:
        print(f"Using in-memory dataframe with {len(df):,} candles")
    
    # Add features
    df = add_features(df)
    print(f"After features: {len(df):,} candles, {len(df.columns)} columns")
    
    # Select a compact, low-redundancy feature set before normalization
    candidate_raw_cols = [
        'log_return', 'volatility', 'rsi', 'macd_hist', 'bb_pct',
        'price_range', 'volume_change', 'price_ema_ratio', 'frac_diff',
        'fear_greed_norm'
    ]
    selected_raw_cols = select_feature_columns(df, candidate_raw_cols)

    # Normalize selected features
    normalize_cols = [col for col in selected_raw_cols if col != 'fear_greed_norm']
    df = normalize_features(df, normalize_cols)

    # Persist selected features so training / evaluation use the same denoised set
    feature_columns = ['close', 'volume'] + [
        'fear_greed_norm' if col == 'fear_greed_norm' else f'{col}_norm'
        for col in selected_raw_cols
    ]
    _save_feature_columns(symbol, timeframe, feature_columns)
    print(f"Selected features: {feature_columns}")
    
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
    saved = _load_saved_feature_columns()
    if saved:
        return saved

    return [
        'close', 'volume',
        'log_return_norm', 'volatility_norm', 'rsi_norm',
        'macd_hist_norm', 'bb_pct_norm', 'price_range_norm',
        'volume_change_norm', 'price_ema_ratio_norm', 'frac_diff_norm',
        'fear_greed_norm'
    ]


if __name__ == "__main__":
    df = prepare_features()
    print("\nFeature columns:")
    print(df.columns.tolist())
    print("\nSample data:")
    print(df.head())
