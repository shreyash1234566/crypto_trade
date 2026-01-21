"""
Fear & Greed Index - Optional sentiment data from alternative.me API.
"""
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import PROCESSED_DATA_DIR


def fetch_fear_greed(days: int = 180) -> pd.DataFrame | None:
    """
    Fetch Fear & Greed Index history from alternative.me API.
    
    Args:
        days: Number of days of history
        
    Returns:
        DataFrame with fear_greed column, or None if API fails
    """
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data:
            print("Fear & Greed API: No data in response")
            return None
        
        # Parse response
        records = []
        for item in data['data']:
            records.append({
                'timestamp': pd.to_datetime(int(item['timestamp']), unit='s'),
                'fear_greed': int(item['value']),
                'classification': item['value_classification']
            })
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"Fear & Greed: Fetched {len(df)} days of data")
        return df
        
    except requests.RequestException as e:
        print(f"Fear & Greed API error: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Fear & Greed parsing error: {e}")
        return None


def merge_fear_greed(
    df: pd.DataFrame,
    fear_greed_df: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Merge Fear & Greed index into OHLCV data.
    
    The index is daily, so we forward-fill to match the trading timeframe.
    
    Args:
        df: OHLCV DataFrame with datetime index
        fear_greed_df: Fear & Greed DataFrame, or None
        
    Returns:
        DataFrame with fear_greed column added
    """
    df = df.copy()
    
    if fear_greed_df is None or fear_greed_df.empty:
        # Add NaN column if API failed
        df['fear_greed'] = 50  # Neutral default
        df['fear_greed_norm'] = 0.0
        print("Fear & Greed: Using neutral default (50)")
        return df
    
    # Resample fear_greed to match df index (forward fill daily value)
    fear_greed_df = fear_greed_df[['fear_greed']].copy()
    
    # Create date column for merging
    df['date'] = df.index.date
    fear_greed_df['date'] = fear_greed_df.index.date
    
    # Merge on date
    merged = df.merge(
        fear_greed_df[['date', 'fear_greed']].drop_duplicates('date'),
        on='date',
        how='left',
        suffixes=('', '_fg')
    )
    
    # Forward fill missing values
    merged['fear_greed'] = merged['fear_greed'].ffill().bfill()
    
    # Normalize (0-100 -> -1 to 1)
    merged['fear_greed_norm'] = (merged['fear_greed'] - 50) / 50
    
    # Restore index
    merged.set_index(df.index, inplace=True)
    merged.drop(columns=['date'], inplace=True)
    
    print(f"Fear & Greed: Merged {merged['fear_greed'].notna().sum()} values")
    
    return merged


def save_fear_greed(days: int = 365):
    """Fetch and save Fear & Greed data."""
    df = fetch_fear_greed(days)
    if df is not None:
        filepath = PROCESSED_DATA_DIR / "fear_greed.parquet"
        df.to_parquet(filepath)
        print(f"Saved to {filepath}")
    return df


def load_fear_greed() -> pd.DataFrame | None:
    """Load saved Fear & Greed data."""
    filepath = PROCESSED_DATA_DIR / "fear_greed.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None


if __name__ == "__main__":
    # Test fetch
    df = fetch_fear_greed(30)
    if df is not None:
        print(df.head(10))
        print(f"\nLatest: {df.iloc[-1]['fear_greed']} ({df.iloc[-1]['classification']})")
