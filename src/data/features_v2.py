"""
features_v2.py -- All indicators in pure pandas/numpy. No pandas_ta needed.

Key findings from report_OptimizedTrend_BTC_USDT.txt (+7.96% ROI):
  Win rate:       42.2%  <- LOW win rate is FINE
  Win/Loss ratio: 2.527  <- wins are 2.5x bigger than losses
  Profit Factor:  1.842  <- for every $1 lost, makes $1.84
  Max Drawdown:   5.96%  <- very controlled

  The 3 rules that make this work:
  1. price > EMA200  -> master regime filter (no bear market trades)
  2. EMA8 crosses above EMA21 -> fresh momentum (cross, not continuation)
  3. Exit when price < EMA21 -> trailing stop cuts losses fast

Place at: src/data/features_v2.py
"""
import numpy as np
import pandas as pd


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, 1e-8))

def _macd(s, fast, slow, sig):
    m = _ema(s, fast) - _ema(s, slow)
    sl = _ema(m, sig)
    return m, sl, m - sl

def _atr(h, l, c, p=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, min_periods=p).mean()

def _bb(s, p=20, std=2.0):
    mid = s.rolling(p).mean()
    sig = s.rolling(p).std()
    return mid - std * sig, mid, mid + std * sig

def _stoch(h, l, c, k=14, d=3):
    lo = l.rolling(k).min()
    hi = h.rolling(k).max()
    K = 100 * (c - lo) / (hi - lo + 1e-8)
    return K, K.rolling(d).mean()

def _adx(h, l, c, p=14):
    ph, pl = h.shift(1), l.shift(1)
    up, down = h - ph, pl - l
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(h, l, c, p)
    pdi = 100 * pdm.ewm(com=p-1, min_periods=p).mean() / atr.replace(0, 1e-8)
    mdi = 100 * mdm.ewm(com=p-1, min_periods=p).mean() / atr.replace(0, 1e-8)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-8)
    return dx.ewm(com=p-1, min_periods=p).mean(), pdi, mdi

def _obv(c, v):
    return (np.sign(c.diff().fillna(0)) * v).cumsum()

def _norm(s, w=252, clip=5.0):
    mu = s.rolling(w, min_periods=20).mean()
    sig = s.rolling(w, min_periods=20).std().replace(0, 1e-8)
    return ((s - mu) / sig).clip(-clip, clip)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, lo, v = df['close'], df['high'], df['low'], df['volume']

    # -- EMAs ------------------------------------------------------------------
    for span in [8, 13, 21, 34, 55, 89, 200]:
        df[f'ema{span}'] = _ema(c, span)

    # -- THE 3 RULES from optimized_trend.py -----------------------------------
    # Rule 1: Regime filter
    df['price_above_200']   = (c > df['ema200']).astype(int)
    # Rule 2: Fresh momentum cross
    df['ema8_above_21']     = (df['ema8'] > df['ema21']).astype(int)
    df['ema8_21_cross']     = (
        (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
    ).astype(int)
    # Rule 3: Trailing stop condition
    df['price_above_ema21'] = (c > df['ema21']).astype(int)

    # Other trend
    df['ema8_34_cross']  = (df['ema8'] > df['ema34']).astype(int)
    df['ema21_55_cross'] = (df['ema21'] > df['ema55']).astype(int)
    df['ema_slope']      = (df['ema21'] - df['ema21'].shift(5)) / df['ema21'].shift(5) * 100

    # -- RSI -------------------------------------------------------------------
    df['rsi']      = _rsi(c, 14)
    df['rsi_fast'] = _rsi(c, 7)
    df['rsi_slow'] = _rsi(c, 21)
    df['rsi_oversold']   = (df['rsi'] < 35).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 65).astype(int)
    df['rsi_neutral']    = ((df['rsi'] >= 45) & (df['rsi'] <= 65)).astype(int)

    # -- MACD (12,26,9) --------------------------------------------------------
    m, ms, mh = _macd(c, 12, 26, 9)
    df['macd']             = m
    df['macd_signal']      = ms
    df['macd_hist']        = mh
    df['macd_bull']        = (m > ms).astype(int)
    df['macd_hist_rising'] = (mh > mh.shift(1)).astype(int)

    # -- MACD research-optimal (17,21,15) --------------------------------------
    mo, mos, _ = _macd(c, 17, 21, 15)
    df['macd_opt']        = mo
    df['macd_opt_signal'] = mos

    # -- ATR -------------------------------------------------------------------
    df['atr']       = _atr(h, lo, c, 14)
    df['atr_pct']   = df['atr'] / c * 100
    df['atr_ratio'] = df['atr'] / df['atr'].rolling(20).mean()

    # -- Bollinger Bands -------------------------------------------------------
    bbl, bbm, bbu  = _bb(c, 20, 2.0)
    df['bb_lower'] = bbl
    df['bb_mid']   = bbm
    df['bb_upper'] = bbu
    df['bb_pct']   = (c - bbl) / (bbu - bbl + 1e-8)
    df['bb_width'] = (bbu - bbl) / bbm * 100
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)

    # -- Stochastic ------------------------------------------------------------
    df['stoch_k'], df['stoch_d'] = _stoch(h, lo, c)
    df['stoch_oversold'] = (df['stoch_k'] < 25).astype(int)

    # -- ADX -------------------------------------------------------------------
    df['adx'], df['di_plus'], df['di_minus'] = _adx(h, lo, c, 14)
    df['strong_trend'] = (df['adx'] > 25).astype(int)
    df['di_bull']      = (df['di_plus'] > df['di_minus']).astype(int)

    # -- Volume ----------------------------------------------------------------
    df['volume_sma20'] = v.rolling(20).mean()
    df['volume_ratio'] = v / df['volume_sma20']
    df['high_volume']  = (df['volume_ratio'] > 1.5).astype(int)
    df['obv']          = _obv(c, v)
    df['obv_ema']      = _ema(df['obv'], 20)
    df['obv_bull']     = (df['obv'] > df['obv_ema']).astype(int)

    # -- Price action ----------------------------------------------------------
    df['body_ratio'] = (c - df['open']).abs() / (h - lo).replace(0, np.nan)
    df['is_bullish'] = (c > df['open']).astype(int)
    df['roc5']       = c.pct_change(5) * 100
    df['roc20']      = c.pct_change(20) * 100
    df['close_z20']  = (c - c.rolling(20).mean()) / c.rolling(20).std()
    df['close_z50']  = (c - c.rolling(50).mean()) / c.rolling(50).std()

    # -- Normalised versions for BiLSTM input ----------------------------------
    w = 252
    df['rsi_norm']          = _norm(df['rsi'], w)
    df['macd_hist_norm']    = _norm(df['macd_hist'], w)
    df['bb_pct_norm']       = _norm(df['bb_pct'], w)
    df['adx_norm']          = _norm(df['adx'], w)
    df['volume_ratio_norm'] = _norm(df['volume_ratio'], w)
    df['atr_ratio_norm']    = _norm(df['atr_ratio'], w)
    df['stoch_k_norm']      = _norm(df['stoch_k'], w)
    df['roc5_norm']         = _norm(df['roc5'], w)
    df['roc20_norm']        = _norm(df['roc20'], w)
    df['ema_slope_norm']    = _norm(df['ema_slope'], w)
    df['macd_opt_norm']     = _norm(df['macd_opt'], w)
    df['log_return']        = np.log(c / c.shift(1))
    df['log_return_norm']   = _norm(df['log_return'], w)
    df['volatility_norm']   = _norm(df['log_return'].rolling(20).std(), w)

    # -- Target for BiLSTM -----------------------------------------------------
    # Target: meaningful price move (1.5% in 12 candles) in bull regime
    # Old target (0.3% in 3 candles) was noise -- BTC moves that much randomly
    forward_return = df['close'].shift(-12) / df['close'] - 1
    df['target'] = (
        (forward_return > 0.015) &               # 1.5% upside (was 0.3% -- too noisy)
        (df['price_above_200'] == 1) &            # bull regime
        (df['rsi'] < 70)                          # not already overbought
    ).astype(float)

    df = df.dropna().reset_index(drop=True)
    print(f"Features built: {df.shape[0]:,} rows x {df.shape[1]} cols  |  "
          f"Target positives: {df['target'].mean():.2%}")
    return df


# Features fed into BiLSTM (30 rich features)
BILSTM_FEATURES = [
    'log_return_norm', 'volatility_norm',
    'rsi_norm', 'macd_hist_norm', 'bb_pct_norm',
    'adx_norm', 'volume_ratio_norm', 'atr_ratio_norm',
    'stoch_k_norm', 'roc5_norm', 'roc20_norm',
    'ema_slope_norm', 'macd_opt_norm',
    'close_z20', 'close_z50',
    # Binary signals (0/1) -- directly learnable patterns
    'price_above_200',    # <- master regime filter
    'ema8_21_cross',      # <- the fresh momentum signal
    'price_above_ema21',  # <- trailing stop condition
    'ema8_34_cross', 'ema21_55_cross',
    'macd_bull', 'macd_hist_rising',
    'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
    'strong_trend', 'di_bull',
    'stoch_oversold', 'high_volume', 'obv_bull',
]

# Strategy features added directly to PPO observation (gives PPO regime awareness)
PPO_STRATEGY_FEATURES = [
    'price_above_200',    # bear/bull regime -- most important
    'ema8_21_cross',      # fresh entry signal
    'price_above_ema21',  # trailing stop -- exit signal
    'adx_norm',           # trend strength
    'rsi_norm',           # momentum state
    'bb_pct_norm',        # mean reversion position
]
