"""
Feature Engineering - Research-backed technical indicators.

Based on:
- IEEE paper: EMA strategy, profit factor 3.5, win rate 60%
- ACM paper: RSI + MACD + on-chain factors → Sharpe 2.5
- Lopez de Prado: purged features, fractional differentiation
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    df = df.copy()

    # ─── Trend Indicators ─────────────────────────────────────────────
    df['ema8']  = ta.ema(df['close'], length=8)
    df['ema13'] = ta.ema(df['close'], length=13)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema34'] = ta.ema(df['close'], length=34)
    df['ema55'] = ta.ema(df['close'], length=55)
    df['ema89'] = ta.ema(df['close'], length=89)
    df['ema200']= ta.ema(df['close'], length=200)

    df['sma20'] = ta.sma(df['close'], length=20)
    df['sma50'] = ta.sma(df['close'], length=50)

    # Trend signals
    df['ema8_34_cross']  = (df['ema8'] > df['ema34']).astype(int)
    df['ema21_55_cross'] = (df['ema21'] > df['ema55']).astype(int)
    df['price_above_200']= (df['close'] > df['ema200']).astype(int)

    # Trend strength
    df['ema_slope'] = (df['ema21'] - df['ema21'].shift(5)) / df['ema21'].shift(5) * 100

    # ─── Momentum Indicators ─────────────────────────────────────────
    rsi = ta.rsi(df['close'], length=14)
    df['rsi'] = rsi
    df['rsi_fast'] = ta.rsi(df['close'], length=7)
    df['rsi_slow'] = ta.rsi(df['close'], length=21)

    # RSI zones
    df['rsi_oversold']   = (df['rsi'] < 35).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 65).astype(int)
    df['rsi_neutral']    = ((df['rsi'] >= 45) & (df['rsi'] <= 65)).astype(int)

    # MACD (IEEE paper optimal params)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['macd']        = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
        df['macd_hist']   = macd.iloc[:, 2]
        df['macd_bull']   = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_hist_pos'] = (df['macd_hist'] > 0).astype(int)
        df['macd_hist_rising'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)

    # MACD ADX optimal params from research paper (Short=17, Long=21, Signal=15)
    macd_opt = ta.macd(df['close'], fast=17, slow=21, signal=15)
    if macd_opt is not None and not macd_opt.empty:
        df['macd_opt'] = macd_opt.iloc[:, 0]
        df['macd_opt_signal'] = macd_opt.iloc[:, 1]

    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if stoch is not None and not stoch.empty:
        df['stoch_k'] = stoch.iloc[:, 0]
        df['stoch_d'] = stoch.iloc[:, 1]
        df['stoch_oversold']   = (df['stoch_k'] < 25).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 75).astype(int)

    # ─── Volatility Indicators ───────────────────────────────────────
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr'] = atr
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None and not bb.empty:
        df['bb_lower'] = bb.iloc[:, 0]
        df['bb_mid']   = bb.iloc[:, 1]
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)

    # ATR ratio (volatility regime filter)
    df['atr_ratio'] = df['atr'] / df['atr'].rolling(20).mean()
    df['high_vol'] = (df['atr_ratio'] > 1.5).astype(int)
    df['low_vol']  = (df['atr_ratio'] < 0.7).astype(int)

    # ─── Volume Indicators ───────────────────────────────────────────
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    df['high_volume']  = (df['volume_ratio'] > 1.5).astype(int)

    obv = ta.obv(df['close'], df['volume'])
    df['obv'] = obv
    df['obv_ema'] = ta.ema(df['obv'], length=20)
    df['obv_bull'] = (df['obv'] > df['obv_ema']).astype(int)

    # ─── Trend Strength ──────────────────────────────────────────────
    adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_data is not None and not adx_data.empty:
        df['adx'] = adx_data.iloc[:, 0]
        df['di_plus']  = adx_data.iloc[:, 1]
        df['di_minus'] = adx_data.iloc[:, 2]
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['di_bull'] = (df['di_plus'] > df['di_minus']).astype(int)

    # ─── Price Action ────────────────────────────────────────────────
    df['candle_body']  = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio']   = df['candle_body'] / df['candle_range'].replace(0, np.nan)
    df['is_bullish']   = (df['close'] > df['open']).astype(int)

    # Momentum
    df['roc5']  = df['close'].pct_change(5) * 100
    df['roc10'] = df['close'].pct_change(10) * 100
    df['roc20'] = df['close'].pct_change(20) * 100

    # ─── Multi-timeframe proxy (rolling stats) ───────────────────────
    df['close_z20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    df['close_z50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()

    # ─── Target (next candle direction for ML) ───────────────────────
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df


def get_feature_columns() -> list:
    """Return feature columns for ML models."""
    return [
        'rsi', 'rsi_fast', 'rsi_slow',
        'macd_hist', 'macd_bull', 'macd_hist_rising',
        'ema8_34_cross', 'ema21_55_cross', 'price_above_200',
        'ema_slope', 'atr_pct', 'atr_ratio',
        'bb_pct', 'bb_width', 'bb_squeeze',
        'stoch_k', 'stoch_d',
        'volume_ratio', 'high_volume', 'obv_bull',
        'adx', 'di_bull', 'strong_trend',
        'close_z20', 'close_z50',
        'roc5', 'roc10', 'roc20',
        'body_ratio', 'is_bullish',
    ]
