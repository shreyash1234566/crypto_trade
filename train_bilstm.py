"""
train_bilstm.py — Train the PyTorch BiLSTM with strategy-aware target.

What makes this different from train_bilstm_refined.py:
  1. PyTorch (not TensorFlow) — produces bilstm_best.pt for Phase 4
  2. 30 rich features (not 5 raw OHLCV) — BiLSTM sees RSI, MACD, EMA crosses
  3. Strategy-aware target: predict WHEN the +7.96% entry setup fires
     target = 1 when (EMA8 crosses EMA21) AND (price > EMA200)
     This teaches the BiLSTM to recognize high-quality entry conditions
     Not lookahead bias — uses only current candle data

Run: python train_bilstm.py
Output: models_saved/bilstm_best.pt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import SEQUENCE_LENGTH, MODELS_DIR
from src.models.bilstm import (
    BiLSTMFeatureExtractor, SequenceDataset, train_bilstm, save_model
)
from src.data.features_v2 import build_features, BILSTM_FEATURES

import torch
from torch.utils.data import DataLoader

DATA_PATH  = ROOT / 'btc_usdt_1h.csv'
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 1e-4
PATIENCE   = 15
TRAIN_FRAC = 0.80


print("\n" + "="*55)
print("  STEP 1 — Load data")
print("="*55)
assert DATA_PATH.exists(), f"ERROR: {DATA_PATH} not found"
raw = pd.read_csv(DATA_PATH)
print(f"  {len(raw):,} rows  (~{len(raw)//24} days)")


print("\n" + "="*55)
print("  STEP 2 — Build 30 strategy features (no pandas_ta)")
print("="*55)
df = build_features(raw)
feat_cols = [c for c in BILSTM_FEATURES if c in df.columns]
print(f"  Using {len(feat_cols)} features")
print(f"  Target: strategy entry signal")
print(f"  Positive rate: {df['target'].mean():.2%}  (entries are rare — correct)")


print("\n" + "="*55)
print("  STEP 3 — Train/val split")
print("="*55)
n      = len(df)
split  = int(n * TRAIN_FRAC)
df_tr  = df.iloc[:split].reset_index(drop=True)
df_val = df.iloc[split:].reset_index(drop=True)
print(f"  Train: {len(df_tr):,}   Val: {len(df_val):,}")

X_tr  = df_tr[feat_cols].values.astype(np.float32)
y_tr  = df_tr['target'].values.astype(np.float32)
X_val = df_val[feat_cols].values.astype(np.float32)
y_val = df_val['target'].values.astype(np.float32)
X_tr  = np.nan_to_num(X_tr,  nan=0.0, posinf=1.0, neginf=-1.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)

train_ds = SequenceDataset(X_tr,  y_tr,  SEQUENCE_LENGTH)
val_ds   = SequenceDataset(X_val, y_val, SEQUENCE_LENGTH)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
print(f"  Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")


print("\n" + "="*55)
print("  STEP 4 — Build BiLSTM model")
print("="*55)
model    = BiLSTMFeatureExtractor(input_size=len(feat_cols), use_attention=True)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Input size:  {len(feat_cols)} features")
print(f"  Output size: 64-dim state vector")
print(f"  Parameters:  {n_params:,}")

# Verify shapes
dummy = torch.randn(4, SEQUENCE_LENGTH, len(feat_cols))
with torch.no_grad():
    s = model(dummy, return_state=True)
    p = model(dummy, return_state=False)
print(f"  State shape: {s.shape}  Pred shape: {p.shape}  ✓")


print("\n" + "="*55)
print("  STEP 5 — Class weights (rare entry = imbalanced classes)")
print("="*55)
# Strategy entries are rare (~2-5% of candles)
# Weight the positive class heavily so BiLSTM learns to find entries
pos_rate = y_tr.mean()
neg_rate = 1 - pos_rate
w_pos = neg_rate / (2 * pos_rate + 1e-8)    # oversample minority
w_neg = pos_rate / (2 * neg_rate + 1e-8)
class_weights = torch.FloatTensor([w_neg, w_pos])
print(f"  Entry rate: {pos_rate:.2%}  Weight pos: {w_pos:.2f}  Weight neg: {w_neg:.2f}")


print("\n" + "="*55)
print(f"  STEP 6 — Train (max {EPOCHS} epochs, patience={PATIENCE})")
print("="*55)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

history = train_bilstm(
    model          = model,
    train_loader   = train_loader,
    val_loader     = val_loader,
    epochs         = EPOCHS,
    learning_rate  = LR,
    patience       = PATIENCE,
    class_weights  = class_weights,
    use_focal_loss = True,     # Focal loss handles the class imbalance better
    label_smoothing= 0.05,
    use_mixup      = True,
    mixup_alpha    = 0.2,
)

print("\n" + "="*55)
print("  STEP 7 — Verify saved file")
print("="*55)
best_pt = MODELS_DIR / 'bilstm_best.pt'
assert best_pt.exists(), f"ERROR: {best_pt} not created"
import os
print(f"  ✓ models_saved/bilstm_best.pt  ({os.path.getsize(best_pt)//1024} KB)")
print(f"  Best val accuracy: {max(history['val_acc']):.4f}")
print(f"\n  Done. Now run: python train_v2.py")
