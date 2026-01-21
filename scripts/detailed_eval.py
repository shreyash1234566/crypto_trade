"""
Detailed Model Evaluation Script

Evaluates all trained PPO models against:
- Buy & hold baseline
- Multiple metrics (return, Sharpe, win rate, max drawdown)
- Trade analysis and visualization
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.features import load_featured_data, get_feature_columns
from src.environment.trading_env import CryptoTradingEnv
from src.models.bilstm import load_model as load_bilstm, freeze_model
from src.models.ppo_agent import load_ppo_agent
from config.settings import MODELS_DIR, REWARD_FLAT_PENALTY


def evaluate_model(model, vec_env, n_episodes=10):
    """Evaluate model with detailed metrics using VecNormalize env."""
    all_returns = []
    all_trades = []
    all_actions = []
    base_env = vec_env.envs[0]
    
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        episode_actions = []
        info = base_env._get_info()
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards_vec, dones, infos = vec_env.step(action)
            episode_actions.append(int(action[0] if hasattr(action, '__len__') else action))
            done = bool(dones[0])
            info = infos[0]
        
        all_returns.append(info.get('total_return', 0))
        all_trades.append(info.get('n_trades', 0))
        all_actions.extend(episode_actions)
    
    # Calculate metrics
    returns = np.array(all_returns)
    actions = np.array(all_actions)
    
    metrics = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'sharpe': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        'win_rate': sum(1 for r in returns if r > 0) / len(returns),
        'mean_trades': np.mean(all_trades),
        'action_buy_pct': 100 * np.sum(actions == 0) / len(actions),
        'action_hold_pct': 100 * np.sum(actions == 1) / len(actions),
        'action_sell_pct': 100 * np.sum(actions == 2) / len(actions),
    }
    
    return metrics


def calculate_buy_hold(df):
    """Calculate buy & hold return."""
    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    return (end_price - start_price) / start_price


def print_comparison_table(results_df):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    numeric_cols = ['mean_return', 'sharpe', 'win_rate', 'mean_trades']
    for col in numeric_cols:
        if col in results_df.columns:
            print(f"{col:20s}: mean={results_df[col].mean():.4f}, std={results_df[col].std():.4f}")


def plot_model_comparison(results_df, output_path='models_saved/model_comparison.png'):
    """Plot model performance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Return comparison
    axes[0, 0].bar(results_df['model'], results_df['mean_return'])
    axes[0, 0].set_title('Mean Return by Model')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Sharpe ratio
    axes[0, 1].bar(results_df['model'], results_df['sharpe'])
    axes[0, 1].set_title('Sharpe Ratio by Model')
    axes[0, 1].set_ylabel('Sharpe')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Win rate
    axes[1, 0].bar(results_df['model'], results_df['win_rate'])
    axes[1, 0].set_title('Win Rate by Model')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='50%')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()
    
    # Number of trades
    axes[1, 1].bar(results_df['model'], results_df['mean_trades'])
    axes[1, 1].set_title('Mean Trades per Episode')
    axes[1, 1].set_ylabel('Trades')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def main():
    print("="*80)
    print("DETAILED MODEL EVALUATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_featured_data()
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Use last 20% for test
    test_start = int(len(df) * 0.8)
    test_df = df.iloc[test_start:].reset_index(drop=True)
    print(f"Test data: {len(test_df)} candles ({len(test_df)/96:.1f} days)")
    
    # Calculate buy & hold baseline
    buy_hold_return = calculate_buy_hold(test_df)
    print(f"Buy & Hold return: {buy_hold_return*100:.2f}%")
    
    # Load Bi-LSTM
    print("\nLoading Bi-LSTM...")
    try:
        bilstm = load_bilstm('bilstm_best.pt')
        freeze_model(bilstm)
        use_lstm = True
    except FileNotFoundError:
        print("No Bi-LSTM found, using raw features")
        bilstm = None
        use_lstm = False
    
    # Find all PPO models
    model_files = sorted(MODELS_DIR.glob('ppo_fold_*.zip'))
    if (MODELS_DIR / 'best_model.zip').exists():
        model_files = [MODELS_DIR / 'best_model.zip'] + model_files
    
    print(f"\nFound {len(model_files)} models to evaluate")
    
    # Evaluate each model
    results = []
    for model_path in model_files:
        model_name = model_path.stem
        print(f"\nEvaluating {model_name}...")
        
        # Create environment
        env = CryptoTradingEnv(
            test_df, feature_cols,
            use_lstm_features=use_lstm,
            lstm_model=bilstm,
            flat_penalty=REWARD_FLAT_PENALTY
        )
        
        # Load model
        try:
            model, vec_env = load_ppo_agent(env, model_name, bilstm)
            metrics = evaluate_model(model, vec_env, n_episodes=10)
            
            # Add model name and alpha
            metrics['model'] = model_name
            metrics['alpha'] = metrics['mean_return'] - buy_hold_return
            
            results.append(metrics)
            
            print(f"  Return: {metrics['mean_return']*100:.2f}% (alpha: {metrics['alpha']*100:+.2f}%)")
            print(f"  Sharpe: {metrics['sharpe']:.3f}, Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  Actions: Buy {metrics['action_buy_pct']:.1f}%, Hold {metrics['action_hold_pct']:.1f}%, Sell {metrics['action_sell_pct']:.1f}%")
            
        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
    
    if not results:
        print("\nNo models successfully evaluated!")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols_order = ['model', 'mean_return', 'alpha', 'sharpe', 'win_rate', 'mean_trades', 
                  'action_buy_pct', 'action_hold_pct', 'action_sell_pct']
    results_df = results_df[[c for c in cols_order if c in results_df.columns]]
    
    # Print comparison table
    print_comparison_table(results_df)
    
    # Save results
    output_csv = MODELS_DIR / 'evaluation_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Plot comparison
    if len(results_df) > 1:
        plot_model_comparison(results_df)
    
    # Identify best model
    best_idx = results_df['sharpe'].idxmax()
    best_model = results_df.loc[best_idx]
    
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    print(f"Model: {best_model['model']}")
    print(f"Return: {best_model['mean_return']*100:.2f}%")
    print(f"Alpha vs Buy & Hold: {best_model['alpha']*100:+.2f}%")
    print(f"Sharpe: {best_model['sharpe']:.3f}")
    print(f"Win Rate: {best_model['win_rate']*100:.1f}%")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if best_model['mean_return'] > 0.02 and best_model['win_rate'] > 0.35:
        print("✓ Model shows promise! Consider paper trading.")
    elif best_model['mean_return'] > 0:
        print("○ Model is profitable but marginal. More training recommended.")
    else:
        print("✗ Model underperforming. Review reward function and training process.")


if __name__ == "__main__":
    main()
