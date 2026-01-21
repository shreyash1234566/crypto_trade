"""
Configuration settings for the trading bot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_saved"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# BINANCE API
# =============================================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# =============================================================================
# TRADING PARAMETERS
# =============================================================================
SYMBOL = "BTC/USDT"
TIMEFRAME_RAW = "1m"       # Raw data timeframe
TIMEFRAME_TRADE = "15m"    # Trading timeframe
LOOKBACK_DAYS = 180        # 6 months of historical data

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
# Bi-LSTM
SEQUENCE_LENGTH = 60       # 60 candles lookback (15 hours for 15-min)
LSTM_HIDDEN_SIZE = 128     # Hidden units per LSTM layer
LSTM_NUM_LAYERS = 2        # Number of Bi-LSTM layers
STATE_DIM = 64             # Output state vector dimension

# =============================================================================
# PPO HYPERPARAMETERS (Optimized by Optuna)
# =============================================================================
PPO_LEARNING_RATE = 0.000044
PPO_N_STEPS = 4096
PPO_BATCH_SIZE = 128
PPO_N_EPOCHS = 3
PPO_GAMMA = 0.979575
PPO_GAE_LAMBDA = 0.9452
PPO_CLIP_RANGE = 0.1154
PPO_ENT_COEF = 0.027865
PPO_VF_COEF = 0.6633
PPO_MAX_GRAD_NORM = 0.4677
PPO_LSTM_HIDDEN_SIZE = 256
PPO_NET_ARCH = "large"
PPO_ACTIVATION_FN = "relu"
PPO_ORTHO_INIT = False

# LSTM Policy specific
PPO_N_LSTM_LAYERS = 1      # Single layer LSTM in policy

# Learning rate schedule: linear decay from initial to 0
# This helps early exploration while stabilizing later
PPO_LR_SCHEDULE = "linear"  # Options: "constant", "linear"

# Entropy schedule: start high, decay to encourage exploration early
# Your logs show entropy collapsed too fast -> use higher initial + slower decay
PPO_ENT_COEF_INITIAL = 0.02  # Start with more exploration
PPO_ENT_COEF_FINAL = 0.001   # End with some exploration
PPO_USE_ENT_SCHEDULE = True  # Enable entropy schedule

# =============================================================================
# VALIDATION PARAMETERS (Purged Walk-Forward)
# =============================================================================
N_SPLITS = 5               # Number of CV folds
EMBARGO_PCT = 0.01         # 1% purge (~1 hour for 15-min data)
TRAIN_DAYS = 30            # Days per training window
TEST_DAYS = 7              # Days per test window

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
MAX_RISK_PER_TRADE = 0.02  # 2% max risk per trade
MAX_POSITIONS = 1          # Max concurrent positions
VOLATILITY_THRESHOLD = 2.0 # Reject if vol > 2x average
STOP_LOSS_PCT = 0.03       # 3% stop loss

# =============================================================================
# REWARD FUNCTION (Dense Shaping for Better Learning)
# =============================================================================
REWARD_PROFIT = 1.0          # Reward for profitable trade (realized)
REWARD_LOSS = -1.2           # Penalty for losing trade (slight asymmetry)
REWARD_HOLD_WINNING = 0.001  # Small reward for holding winners
REWARD_HOLD_LOSING = -0.005  # Penalty per step holding losing position
REWARD_UNREALIZED_SCALE = 0.1  # Scale for step-by-step unrealized PnL reward
REWARD_FLAT_PENALTY = -1e-4   # Penalty when staying flat (encourages engagement)

# =============================================================================
# FEATURES
# =============================================================================
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'log_return', 'volatility',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_pct',
    'fear_greed'  # Optional, will be NaN if unavailable
]

# =============================================================================
# DEVICE (GPU/CPU)
# =============================================================================
import torch
# Bi-LSTM uses GPU for efficient sequence processing
DEVICE_LSTM = "cuda" if torch.cuda.is_available() else "cpu"
# PPO now prefers GPU when available for faster policy updates
DEVICE_PPO = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device - LSTM: {DEVICE_LSTM}, PPO: {DEVICE_PPO}")
