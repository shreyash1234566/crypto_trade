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
    logistic regression model. Supports:
    * Precision‑floor threshold tuning (target minimum precision)
    * Scalable positive‑class weight (w_pos) to reduce bias toward positives
    * Optional calibration step using a validation slice to pick the best
      decision threshold achieving the precision floor.
    """

    def __init__(self, base_strategy, config: dict = None):
        self.base_strategy = base_strategy
        self.config = config or {}
        self.train_window = self.config.get('train_window', 300)
        self.retrain_freq = self.config.get('retrain_freq', 50)
        self.min_confidence = self.config.get('min_confidence', 0.55)
        # New parameters for precision/weight tuning
        self.precision_floor = self.config.get('precision_floor', 0.80)   # target minimum precision
        self.weight_scale = self.config.get('weight_scale', 1.0)        # scale for positive class weight
        self.calibrate = self.config.get('calibrate', True)            # enable calibration step
        self.threshold = None                                         # will hold calibrated threshold
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
        pipeline = None

        for i in range(self.train_window, len(df), self.retrain_freq):
            train_start = max(0, i - self.train_window)
            train_df = df.iloc[train_start:i]

            X_train = train_df[feature_cols].values
            y_train = train_df['target'].values

            if len(np.unique(y_train)) < 2:
                continue  # Need both classes

            try:
                # Compute class weight with scaling
                pos_rate = y_train.mean()
                neg_rate = 1 - pos_rate
                w_pos = neg_rate / (2 * pos_rate + 1e-8) * self.weight_scale
                w_neg = pos_rate / (2 * neg_rate + 1e-8)
                class_weight = {0: w_neg, 1: w_pos}

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(max_iter=200, C=1.0, class_weight=class_weight))
                ])
                pipeline.fit(X_train, y_train)

                # Apply to next retrain_freq candles
                predict_end = min(i + self.retrain_freq, len(df))
                X_pred = df.iloc[i:predict_end][feature_cols].values
                probs = pipeline.predict_proba(X_pred)[:, 1]
                ml_confidence[i:predict_end] = probs
            except Exception:
                pass

        df_enhanced['ml_confidence'] = ml_confidence

        # Calibration: find threshold achieving precision floor if calibrate is True
        if self.calibrate and self.threshold is None:
            # Use a validation slice (e.g., last retrain_freq candles) to calibrate
            val_start = max(0, len(df) - self.retrain_freq)
            val_end = len(df)
            if val_start < val_end:
                X_val = df.iloc[val_start:val_end][feature_cols].values
                y_val = df.iloc[val_start:val_end]['target'].values
                if len(X_val) > 0 and len(y_val) > 0:
                    try:
                        val_probs = pipeline.predict_proba(X_val)[:, 1] if pipeline else np.full(len(X_val), 0.5)
                        # Compute precision-recall curve
                        from sklearn.metrics import precision_recall_curve
                        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
                        # precision array is len(thresholds)+1; trim to match
                        prec_at_thr = precision[:-1]
                        # Find thresholds where precision >= floor
                        mask = prec_at_thr >= self.precision_floor
                        if mask.any():
                            # Pick the *lowest* threshold that still meets the floor
                            # (maximises recall while keeping precision above target)
                            self.threshold = float(thresholds[mask][0])
                        else:
                            self.threshold = self.min_confidence
                    except Exception:
                        self.threshold = self.min_confidence
                else:
                    self.threshold = self.min_confidence
            else:
                self.threshold = self.min_confidence

        # Apply threshold (if calibrated) or use min_confidence
        threshold = self.threshold if self.threshold is not None else self.min_confidence
        df_enhanced['ml_buy'] = (ml_confidence >= threshold).astype(int)

        # Filter: only keep buy signals where ML agrees
        original_buys = df_enhanced['signal'] == 1
        df_enhanced.loc[original_buys & (df_enhanced['ml_buy'] == 0), 'signal'] = 0

        return df_enhanced
