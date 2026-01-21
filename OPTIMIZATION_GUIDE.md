# Model Optimization Implementation - Quick Start

## What Was Changed

### ✅ Phase 1 Critical Fixes Implemented

#### 1. Device Configuration (LSTM=GPU, PPO=CPU)
- **Changed**: Separated device config in `settings.py`
  - `DEVICE_LSTM = "cuda"` for Bi-LSTM feature extraction
  - `DEVICE_PPO = "cpu"` for PPO training
- **Why**: Eliminates GPU↔CPU transfer overhead, better utilizes RTX 3050
- **Files Modified**: `settings.py`, `bilstm.py`, `ppo_agent.py`, `trading_env.py`

#### 2. Reward Function Redesign (Dense Shaping)
- **Changed**: `trading_env.py` reward calculation
  - ❌ Removed: `× 10` scaling on PnL rewards
  - ✅ Added: Step-by-step unrealized PnL change reward (`× 0.1`)
  - ✅ Added: Small incentive for holding winners (`+0.001/step`)
  - ✅ Changed: Holding loser penalty (`-0.01` → `-0.005/step`)
  - ✅ Changed: Stop-loss penalty (reduced from `2×` to `1×`)
- **Why**: Dense rewards provide continuous learning signal, not just sparse on close
- **Files Modified**: `trading_env.py`, `settings.py`

#### 3. Training Configuration Optimization
- **Changed**: PPO hyperparameters in `settings.py`
  - `learning_rate`: `3e-4` → `1e-4` (more conservative)
  - `n_steps`: `2048` → `4096` (longer rollouts)
  - `batch_size`: `64` → `256` (more stable updates)
  - `ent_coef`: `0.01` → `0.05` (more exploration)
- **Why**: Better sample efficiency and exploration for sparse trading signals
- **Files Modified**: `settings.py`, `ppo_agent.py`

#### 4. Evaluation Tools
- **Created**: `scripts/detailed_eval.py`
  - Evaluates all trained models
  - Compares to buy & hold baseline
  - Calculates Sharpe ratio, win rate, action distribution
  - Generates comparison plots and CSV reports
- **Why**: Visibility into what the agent actually learned

---

## How to Use

### Step 1: Test the Current Models (Optional)
Before retraining, see how old models perform with new evaluation:

```powershell
python scripts\detailed_eval.py
```

This will show you the baseline performance to compare against.

---

### Step 2: Retrain with Optimized Configuration

#### Option A: Quick Test (1 Fold, 200k Steps)
Test if new reward function is working:

```powershell
python main.py train --timesteps 200000 --use-lstm --train-days 30 --test-days 7
```

**Expected**: 
- Non-zero rewards during evaluation callbacks
- Action distribution: 20-40% buy/sell (not 100% hold)
- Training should complete in ~1-2 hours

#### Option B: Full Training (Multiple Folds, 500k Steps)
If Step 1 looks good, train full walk-forward validation:

```powershell
python main.py train --timesteps 500000 --use-lstm --train-days 30 --test-days 7
```

**Expected**: 
- Will run ~20 folds automatically
- Each fold takes ~3-4 hours
- Total time: ~60-80 hours for all folds

**Tip**: Run overnight or use screen/tmux if on Linux

---

### Step 3: Evaluate Results

After training completes:

```powershell
python scripts\detailed_eval.py
```

This will:
- Compare all fold models + best_model
- Calculate aggregate metrics
- Generate `models_saved/evaluation_results.csv`
- Create plots: `models_saved/model_comparison.png`

---

### Step 4: Interpret Results

#### ✅ Good Signs (Model is Learning)
- **Mean return > 0%** (profitable)
- **Win rate > 35-40%** (better than random)
- **Alpha > 0** (beats buy & hold)
- **Action distribution**: 20-40% buy, 40-60% hold, 20-40% sell
- **Sharpe ratio > 0.3** (risk-adjusted positive)

#### ⚠️ Warning Signs (Needs More Work)
- **Mean return < 0%** → Review reward function
- **Win rate < 30%** → Increase exploration (higher `ent_coef`)
- **All actions = hold** → Reward not incentivizing trades
- **Sharpe < 0** → High variance, unstable learning

#### ❌ Bad Signs (Major Issues)
- **Zero rewards throughout training** → Check environment logic
- **NaN losses** → Reduce learning rate further
- **Crashes/errors** → Debug environment/model integration

---

## Monitoring Training Progress

### During Training
Watch the console output for:
- **Evaluation callbacks** (every ~50k steps): Check mean_reward
- **Entropy** (should stay >0.5 early on): Exploration happening
- **Value loss** (should decrease): Model learning value estimates

### Tensorboard (Real-Time)
```powershell
tensorboard --logdir models_saved\tensorboard
```

Open: `http://localhost:6006`

Watch for:
- `train/entropy_loss`: Should start high, slowly decrease
- `train/explained_variance`: Should increase toward 1.0
- `eval/mean_reward`: Should trend upward over time

---

## Troubleshooting

### Issue: Training Too Slow
- **Reduce timesteps**: Use `--timesteps 100000` for faster iteration
- **Reduce folds**: Manually set fewer test periods in purged CV

### Issue: Still Getting 0 Rewards
1. **Check data**: Run `python evaluate_model.py` to see current performance
2. **Simplify environment**: 
   - Set `transaction_fee=0` in `trading_env.py` temporarily
   - Disable stop-loss temporarily
   - Test with simpler reward: `reward = current_balance - previous_balance`

### Issue: High Loss, NaN, or Crash
- **Lower learning rate**: Change to `5e-5` in `settings.py`
- **Check for NaN in data**: Run `df.isna().sum()` on loaded features
- **Increase batch size**: Try `512` if memory allows

---

## Next Steps After Phase 1

### If Results Are Good (>0% return, >35% win rate)
1. **Phase 3**: Hyperparameter tuning
   - Test different `gamma`, `gae_lambda`, `clip_range`
   - Use single fold for faster iteration
2. **Phase 5**: Out-of-sample testing
   - Reserve final 10% of data for unseen test
   - Only proceed to paper trading if OOS > 2% return

### If Results Are Marginal (0-2% return)
1. **Reward ablation**: Test sparse vs dense vs hybrid
2. **Check Bi-LSTM**: Train without `--use-lstm` flag, compare
3. **More training**: Increase to `--timesteps 1000000`

### If Results Are Still Negative
1. **Simplify**:
   - Remove shorts (only long positions)
   - Remove transaction fees temporarily
   - Use direct portfolio value reward
2. **Debug**:
   - Manually simulate 100 steps, print (state, action, reward, next_state)
   - Check for data leakage in features
   - Validate Bi-LSTM pre-training quality

---

## Configuration Reference

### Current Optimized Settings

#### Device
- `DEVICE_LSTM = "cuda"` (RTX 3050)
- `DEVICE_PPO = "cpu"` (8 cores)

#### Reward Function
- `REWARD_PROFIT = 1.0`
- `REWARD_LOSS = -1.2`
- `REWARD_HOLD_WINNING = 0.001`
- `REWARD_HOLD_LOSING = -0.005`
- `REWARD_UNREALIZED_SCALE = 0.1`

#### PPO Hyperparameters
- `PPO_LEARNING_RATE = 1e-4`
- `PPO_N_STEPS = 4096`
- `PPO_BATCH_SIZE = 256`
- `PPO_N_EPOCHS = 10`
- `PPO_GAMMA = 0.99`
- `PPO_GAE_LAMBDA = 0.95`
- `PPO_ENT_COEF = 0.05`

#### Risk Management
- `STOP_LOSS_PCT = 0.03` (3%)
- `MAX_RISK_PER_TRADE = 0.02` (2%)
- `transaction_fee = 0.001` (0.1%)

---

## Quick Commands Cheat Sheet

```powershell
# Evaluate existing models
python scripts\detailed_eval.py

# Quick training test (1 fold, 200k steps)
python main.py train --timesteps 200000 --use-lstm

# Full training (all folds, 500k steps)
python main.py train --timesteps 500000 --use-lstm

# Monitor with tensorboard
tensorboard --logdir models_saved\tensorboard

# Paper trade with best model
python main.py paper --use-lstm --testnet --balance 10000

# Check current data
python -c "from src.data.features import load_featured_data; df = load_featured_data(); print(f'Samples: {len(df)}, Features: {len(df.columns)}'); print(df.tail())"
```

---

## Success Metrics Checklist

### Training (Must See)
- [ ] Mean episode reward > 0 within first 100k steps
- [ ] Action distribution: not 100% hold
- [ ] Value loss decreasing
- [ ] Entropy > 0.5 early on

### Validation (Target)
- [ ] Mean return > 5% across folds
- [ ] Win rate > 40%
- [ ] Sharpe ratio > 0.5
- [ ] Max drawdown < 20%

### Out-of-Sample (Deploy Threshold)
- [ ] OOS return > 2%
- [ ] OOS Sharpe > 0.3
- [ ] Beats buy & hold

---

## Contact/Issues

If you encounter issues not covered here:
1. Check logs in `logs/` directory
2. Review tensorboard graphs for anomalies
3. Run `python evaluate_model.py` for detailed debugging
4. Check the full plan document for deeper analysis

**Remember**: RL training is noisy. Run multiple seeds and look at average performance, not single runs.
