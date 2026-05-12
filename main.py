"""
Main Orchestrator - CLI for running the trading bot.

Commands:
- fetch: Download historical data
- process: Resample and add features
- pretrain: Pre-train Bi-LSTM on direction prediction
- train: Train PPO agent with purged validation
- validate: Run purged walk-forward validation
- paper: Run paper trading
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    SYMBOL, TIMEFRAME_RAW, TIMEFRAME_TRADE, LOOKBACK_DAYS,
    SEQUENCE_LENGTH, DEVICE_LSTM, MODELS_DIR,
    REWARD_FLAT_PENALTY
)


def cmd_fetch(args):
    """Fetch historical data from Binance."""
    from src.data.fetcher import fetch_ohlcv
    
    print(f"Fetching {args.symbol} {args.timeframe} data for {args.days} days...")
    df = fetch_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        save=True
    )
    print(f"Fetched {len(df):,} candles")


def cmd_process(args):
    """Process data: resample and add features."""
    from src.data.resampler import resample_and_save
    from src.data.features import prepare_features
    from src.data.fear_greed import fetch_fear_greed, merge_fear_greed, save_fear_greed
    
    print(f"Resampling to {args.timeframe}...")
    df = resample_and_save(
        symbol=args.symbol,
        source_timeframe=TIMEFRAME_RAW,
        target_timeframe=args.timeframe
    )

    if args.fear_greed:
        print("Fetching Fear & Greed Index...")
        fg_df = fetch_fear_greed(days=365)
        if fg_df is not None:
            save_fear_greed(days=365)
            df = merge_fear_greed(df, fg_df)
    
    print("Adding features...")
    df = prepare_features(symbol=args.symbol, timeframe=args.timeframe, df=df)
    
    print(f"Processed {len(df):,} candles with {len(df.columns)} features")


def cmd_pretrain(args):
    """Pre-train Bi-LSTM on direction prediction with improved diagnostics and fixes."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from src.data.features import load_featured_data, get_feature_columns
    from src.models.bilstm import (
        BiLSTMFeatureExtractor, SequenceDataset, 
        create_sequences, train_bilstm, save_model
    )
    import numpy as np
    
    print("Loading featured data...")
    df = load_featured_data(symbol=args.symbol, timeframe=args.timeframe)
    
    # Get feature columns (use normalized versions)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # ============================================================
    # FIX #1: Ensure target column exists and is properly created
    # ============================================================
    if 'target' not in df.columns:
        print("\n⚠️  Creating target column (price direction)...")
        # Create target: 1 if next candle closes higher, 0 otherwise
        df['future_return'] = df['close'].pct_change().shift(-1)
        df['target'] = (df['future_return'] > 0).astype(int)
        df = df.dropna()
        print(f"Target column created. Dataset size: {len(df)}")
    
    # ============================================================
    # FIX #2: Check and report class distribution
    # ============================================================
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    target_counts = df['target'].value_counts()
    target_pcts = df['target'].value_counts(normalize=True)
    
    print(f"Class 0 (Down): {target_counts.get(0, 0):,} samples ({target_pcts.get(0, 0):.2%})")
    print(f"Class 1 (Up):   {target_counts.get(1, 0):,} samples ({target_pcts.get(1, 0):.2%})")
    
    imbalance_ratio = target_pcts.max() / target_pcts.min()
    print(f"Imbalance ratio: {imbalance_ratio:.3f}x")
    
    if imbalance_ratio > 1.15:
        print("⚠️  IMBALANCED CLASSES DETECTED")
        print("   Applying class weighting to loss function...")
        use_class_weights = True
        # Calculate weights: inverse of class frequency
        class_weights = torch.FloatTensor([
            len(df) / (2 * target_counts.get(0, 1)),
            len(df) / (2 * target_counts.get(1, 1))
        ]).to(DEVICE_LSTM)
    else:
        print("✅ Classes are balanced")
        use_class_weights = False
        class_weights = None
    
    # ============================================================
    # FIX #3: Check feature quality
    # ============================================================
    print("\n" + "="*60)
    print("FEATURE QUALITY CHECK")
    print("="*60)
    
    correlations = {}
    for col in feature_cols:
        corr = df[col].corr(df['target'])
        correlations[col] = abs(corr)
    
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 most predictive features:")
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat}: {corr:.4f}")
    
    max_corr = sorted_corrs[0][1]
    if max_corr < 0.03:
        print("\n⚠️  WARNING: Very weak feature-target correlation!")
        print("   Max correlation:", max_corr)
        print("   The model may struggle to learn. Consider:")
        print("   - Adding more informative features")
        print("   - Using longer sequence lengths")
        print("   - Increasing training data")
    else:
        print(f"\n✅ Detectable signal (max correlation: {max_corr:.4f})")
    
    # ============================================================
    # Create sequences
    # ============================================================
    print("\n" + "="*60)
    print("CREATING SEQUENCES")
    print("="*60)
    
    features, targets = create_sequences(df, feature_cols, 'target', SEQUENCE_LENGTH)
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Total sequences: {len(features)}")
    
    # Create dataset
    dataset = SequenceDataset(features, targets, SEQUENCE_LENGTH)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train sequences: {train_size:,}")
    print(f"Val sequences:   {val_size:,}")
    
    # ============================================================
    # FIX #4: Reduce learning rate for better convergence
    # ============================================================
    if args.lr > 0.0003:
        print(f"\n⚠️  Learning rate {args.lr} may be too high for LSTM")
        print("   Consider using --lr 0.0001 or --lr 0.00005")
    
    # ============================================================
    # Create model
    # ============================================================
    print("\n" + "="*60)
    print("MODEL INITIALIZATION")
    print("="*60)
    
    model = BiLSTMFeatureExtractor(input_size=len(feature_cols))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Device: {DEVICE_LSTM}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    
    # ============================================================
    # FIX #5: Pass class weights to training function
    # ============================================================
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    if use_class_weights:
        print("Using weighted loss function")
    
    # You'll need to modify train_bilstm to accept class_weights
    # For now, we'll call it with the existing signature
    try:
        history = train_bilstm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            class_weights=class_weights  # Add this parameter
        )
    except TypeError:
        # If train_bilstm doesn't accept class_weights yet
        print("⚠️  Class weights not supported by train_bilstm yet")
        print("   Training without class weights...")
        history = train_bilstm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience
        )
    
    # ============================================================
    # Results
    # ============================================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    best_val_acc = max(history['val_acc'])
    best_val_loss = min(history['val_loss'])
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Baseline comparison
    baseline_acc = target_pcts.max()
    print(f"Baseline (always predict majority): {baseline_acc:.4f}")
    
    improvement = best_val_acc - baseline_acc
    if improvement > 0.01:
        print(f"✅ Model beats baseline by {improvement:.4f}")
    else:
        print(f"⚠️  Model is NOT better than baseline (diff: {improvement:.4f})")
        print("\nRECOMMENDATIONS:")
        print("1. Try lower learning rate: --lr 0.00005")
        print("2. Increase sequence length in config/settings.py")
        print("3. Add dropout to model (0.2-0.3)")
        print("4. Engineer better features")
        print("\nFor now, proceed with --no-lstm for RL training")


def cmd_train(args):
    """Train PPO agent with purged validation."""
    import numpy as np
    from src.data.features import load_featured_data, get_feature_columns
    from src.environment.trading_env import CryptoTradingEnv
    from src.validation.purged_cv import PurgedWalkForward
    from src.models.bilstm import load_model as load_bilstm, freeze_model
    from src.models.ppo_agent import create_ppo_agent, train_ppo, save_ppo_agent, evaluate_agent
    
    print("Loading data...")
    df = load_featured_data(symbol=args.symbol, timeframe=args.timeframe)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Load pre-trained Bi-LSTM if available
    bilstm_model = None
    if args.use_lstm:
        try:
            bilstm_model = load_bilstm('bilstm_best.pt')
            freeze_model(bilstm_model)
            print("Loaded pre-trained Bi-LSTM")
        except FileNotFoundError:
            print("No pre-trained Bi-LSTM found, using raw features")
    
    # Set up purged walk-forward validation
    cv = PurgedWalkForward(
        train_period=args.train_days,
        test_period=args.test_days,
        embargo_period=4  # 1 hour for 15-min data
    )
    
    # Get splits
    X = df[feature_cols].values
    n_splits = cv.get_n_splits(X)
    print(f"Running {n_splits} walk-forward folds")
    
    all_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        # Create environments
        train_env = CryptoTradingEnv(
            train_df, feature_cols,
            use_lstm_features=args.use_lstm,
            lstm_model=bilstm_model,
            flat_penalty=REWARD_FLAT_PENALTY
        )
        test_env = CryptoTradingEnv(
            test_df, feature_cols,
            use_lstm_features=args.use_lstm,
            lstm_model=bilstm_model,
            flat_penalty=REWARD_FLAT_PENALTY
        )
        
        # Wrap test_env for RecurrentPPO evaluation
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        test_env = DummyVecEnv([lambda: test_env])
        # We need to normalize it using the training stats
        # But we don't have easy access to the training env's stats here unless we extract them from the model
        # create_ppo_agent wraps train_env in VecNormalize.
        
        # Create and train PPO agent
        model = create_ppo_agent(
            train_env,
            bilstm_model=bilstm_model,
            tensorboard_log=str(MODELS_DIR / 'tensorboard' / f'fold_{fold+1}')
        )
        
        # Get the VecNormalize wrapper from the model
        train_vec_env = model.get_env()
        # Apply same normalization to test_env
        if isinstance(train_vec_env, VecNormalize):
             # We can't easily share the stats object, but we can save/load or just use a new one if we don't care about exact match (but we do!)
             # Actually, train_ppo handles eval_env normalization internally if passed.
             pass

        model = train_ppo(
            model,
            total_timesteps=args.timesteps,
            eval_env=test_env, # train_ppo will wrap this if needed, but we already wrapped it in DummyVecEnv
            eval_freq=args.timesteps // 10
        )
        
        # For final evaluation, we need to ensure test_env is normalized with training stats
        # train_ppo doesn't return the wrapped eval_env.
        # We should manually wrap test_env with VecNormalize using stats from model.get_env()
        
        if isinstance(train_vec_env, VecNormalize):
            test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, training=False)
            test_env.obs_rms = train_vec_env.obs_rms
            test_env.ret_rms = train_vec_env.ret_rms
        
        # Evaluate
        metrics = evaluate_agent(model, test_env, n_episodes=5)
        metrics['fold'] = fold + 1
        all_metrics.append(metrics)
        
        # Save fold model
        save_ppo_agent(model, f'ppo_fold_{fold+1}')
    
    # Print summary
    print(f"\n{'='*50}")
    print("WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string())
    
    print(f"\nMean Return: {metrics_df['mean_return'].mean()*100:.2f}%")
    print(f"Mean Win Rate: {metrics_df['win_rate'].mean()*100:.1f}%")


def cmd_paper(args):
    """Run paper trading."""
    import time
    import numpy as np
    from src.data.features import load_featured_data, get_feature_columns
    from src.environment.trading_env import CryptoTradingEnv
    from src.models.ppo_agent import load_ppo_agent
    from src.models.bilstm import load_model as load_bilstm, freeze_model
    from src.execution.paper_trader import PaperTrader
    
    print("Starting paper trading...")
    
    # Load data
    df = load_featured_data(symbol=args.symbol, timeframe=args.timeframe)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Load models
    bilstm_model = None
    if args.use_lstm:
        try:
            bilstm_model = load_bilstm('bilstm_best.pt')
            freeze_model(bilstm_model)
        except FileNotFoundError:
            print("No Bi-LSTM found")
    
    # Create environment
    env = CryptoTradingEnv(
        df, feature_cols,
        use_lstm_features=args.use_lstm,
        lstm_model=bilstm_model,
        flat_penalty=REWARD_FLAT_PENALTY
    )
    
    # Load PPO agent with VecNormalize stats
    try:
        model, vec_env = load_ppo_agent(env, args.model_name, bilstm_model)
    except FileNotFoundError:
        print(f"Model {args.model_name} not found. Run training first.")
        return
    base_env = vec_env.envs[0]
    
    # Create paper trader
    trader = PaperTrader(
        initial_balance=args.balance,
        use_testnet=args.testnet,
        symbol=args.symbol
    )
    
    # Run paper trading
    obs = vec_env.reset()
    info = base_env._get_info()
    done = False
    step = 0
    
    # Initialize LSTM states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    print(f"\nRunning paper trading on {len(df)} candles...")
    
    while not done:
        # Get action from model
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts,
            deterministic=True
        )
        episode_starts = np.zeros((1,), dtype=bool)
        
        action_scalar = int(action[0] if hasattr(action, '__len__') else action)
        
        # Get current state
        current_price = info['current_price']
        current_vol = df.iloc[info['step']].get('volatility', 0.02)
        
        # Execute through paper trader
        trader.execute_signal(action_scalar, current_price, current_vol)
        trader.check_stop_loss(current_price)
        
        # Step environment
        obs, rewards_vec, dones, infos = vec_env.step(action)
        reward = float(rewards_vec[0])
        info = infos[0]
        done = bool(dones[0])
        
        step += 1
        if step % 100 == 0:
            portfolio_value = trader.get_portfolio_value(current_price)
            print(f"Step {step}: Portfolio ${portfolio_value:,.2f}")
    
    # Print summary
    trader.print_summary()
    trader.save_trades()
    trader.save_metrics()


def main():
    parser = argparse.ArgumentParser(
        description="Bi-LSTM + PPO Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch historical data')
    fetch_parser.add_argument('--symbol', default=SYMBOL, help='Trading pair')
    fetch_parser.add_argument('--timeframe', default=TIMEFRAME_RAW, help='Timeframe')
    fetch_parser.add_argument('--days', type=int, default=LOOKBACK_DAYS, help='Days of history')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process data')
    process_parser.add_argument('--symbol', default=SYMBOL, help='Trading pair')
    process_parser.add_argument('--timeframe', default=TIMEFRAME_TRADE, help='Target timeframe')
    process_parser.add_argument('--fear-greed', action='store_true', help='Include Fear & Greed')
    
    # Pretrain command
    pretrain_parser = subparsers.add_parser('pretrain', help='Pre-train Bi-LSTM')
    pretrain_parser.add_argument('--symbol', default=SYMBOL, help='Trading pair')
    pretrain_parser.add_argument('--timeframe', default=TIMEFRAME_TRADE, help='Timeframe')
    pretrain_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    pretrain_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    pretrain_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    pretrain_parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train PPO agent')
    train_parser.add_argument('--symbol', default=SYMBOL, help='Trading pair')
    train_parser.add_argument('--timeframe', default=TIMEFRAME_TRADE, help='Timeframe')
    # Use a higher default to avoid training starvation with PPO_N_STEPS=4096
    train_parser.add_argument('--timesteps', type=int, default=500000,
                              help='Total timesteps per fold (default: 500000)')
    train_parser.add_argument('--train-days', type=int, default=30, help='Training period (days)')
    train_parser.add_argument('--test-days', type=int, default=7, help='Test period (days)')
    train_parser.add_argument('--use-lstm', action='store_true', help='Use pre-trained Bi-LSTM')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--symbol', default=SYMBOL, help='Trading pair')
    paper_parser.add_argument('--timeframe', default=TIMEFRAME_TRADE, help='Timeframe')
    paper_parser.add_argument('--model-name', default='ppo_fold_1', help='Model to use')
    paper_parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    paper_parser.add_argument('--testnet', action='store_true', help='Use Binance Testnet')
    paper_parser.add_argument('--use-lstm', action='store_true', help='Use Bi-LSTM features')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    commands = {
        'fetch': cmd_fetch,
        'process': cmd_process,
        'pretrain': cmd_pretrain,
        'train': cmd_train,
        'paper': cmd_paper
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
