"""
ML Signal Enhancer
==================
Based on research paper LogisticRegressionStrategy:
- Dynamically re-trains using rolling windows of 300 candles
- Incorporates RSI, ADX, fast/slow EMAs, MACD
- Entry requires predicted price increase + RSI < 60 + bullish EMA crossover
- Exit: negative prediction + RSI > 65 or bearish EMA crossover

This wraps any base strategy and uses ML to filter signals (reduce false positives).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class MLSignalEnhancer:
    """
    Wraps a base strategy and filters/confirms signals using a rolling
    logistic regression model. Re-trains every N candles.
    """

    def __init__(self, base_strategy, config: dict = None):
        self.base_strategy = base_strategy
        self.config = config or {}
        self.train_window = self.config.get('train_window', 300)
        self.retrain_freq = self.config.get('retrain_freq', 50)
        self.min_confidence = self.config.get('min_confidence', 0.55)
        self.name = f"ML_{base_strategy.name}"
        self.description = f"ML-enhanced {base_strategy.description}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate base signals then filter with rolling ML model.
        """
        base_signals = self.base_strategy.generate_signals(df)
        df_enhanced = base_signals.copy()

        feature_cols = [
            'rsi', 'rsi_fast', 'macd_hist', 'macd_bull',
            'ema8_34_cross', 'price_above_200', 'adx', 'di_bull',
            'atr_ratio', 'bb_pct', 'volume_ratio', 'close_z20',
            'roc5', 'roc10', 'stoch_k'
        ]
        # Only use available features
        feature_cols = [c for c in feature_cols if c in df.columns]

        if len(feature_cols) < 5:
            return base_signals  # Not enough features for ML

        ml_confidence = np.full(len(df), 0.5)

        # Rolling training
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=200, C=1.0, class_weight='balanced'))
        ])

        for i in range(self.train_window, len(df), self.retrain_freq):
            train_start = max(0, i - self.train_window)
            train_df = df.iloc[train_start:i]

            X_train = train_df[feature_cols].values
            y_train = train_df['target'].values

            if len(np.unique(y_train)) < 2:
                continue  # Need both classes

            try:
                pipeline.fit(X_train, y_train)
                # Apply to next retrain_freq candles
                predict_end = min(i + self.retrain_freq, len(df))
                X_pred = df.iloc[i:predict_end][feature_cols].values
                probs = pipeline.predict_proba(X_pred)[:, 1]
                ml_confidence[i:predict_end] = probs
            except Exception:
                pass

        df_enhanced['ml_confidence'] = ml_confidence
        df_enhanced['ml_buy'] = (ml_confidence >= self.min_confidence).astype(int)

        # Filter: only keep buy signals where ML agrees
        original_buys = df_enhanced['signal'] == 1
        df_enhanced.loc[original_buys & (df_enhanced['ml_buy'] == 0), 'signal'] = 0

        return df_enhanced
