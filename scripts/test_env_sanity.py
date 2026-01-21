"""
Quick diagnostic to verify CryptoTradingEnv produces reasonable returns.
Tests random agent over 10 episodes to establish baseline expectations.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.trading_env import CryptoTradingEnv
from src.data.features import load_featured_data, get_feature_columns

def main():
    print("=" * 60)
    print("ENVIRONMENT SANITY CHECK - RANDOM AGENT BASELINE")
    print("=" * 60)
    
    # Load data
    df = load_featured_data("BTC/USDT", "15m")
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"Using last 500 candles from {len(df)} total")
    test_df = df.iloc[-500:].reset_index(drop=True)
    
    # Strong default params
    reward_params = {
        "reward_profit": 1.7,
        "reward_loss": -1.0,
        "reward_hold_winning": 0.0005,
        "reward_hold_losing": -0.003,
        "reward_unrealized_scale": 0.005,
        "flat_penalty": -0.00005
    }
    
    print(f"Reward params: {reward_params}")
    print("\nRunning 10 random episodes...\n")
    
    # Create environment
    env = CryptoTradingEnv(
        test_df,
        feature_cols,
        use_lstm_features=False,
        lstm_model=None,
        transaction_fee=0.001,  # Revert to realistic fee for RL
        **reward_params
    )
    
    returns = []
    equities = []
    trade_counts = []

    for ep in range(10):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, info = reset_out
        else:
            obs = reset_out

        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            total_reward += reward
            step_count += 1

            if done:
                total_return = info.get('total_return')
                final_equity = info.get('final_equity')
                initial_equity = getattr(env, 'initial_equity', 1000.0)
                n_trades = info.get('n_trades', 0)
                trade_counts.append(n_trades)

                if total_return is None and final_equity is not None:
                    total_return = (final_equity / initial_equity) - 1.0

                if total_return is not None:
                    returns.append(total_return)
                    equities.append(final_equity if final_equity else initial_equity)
                    print(f"Episode {ep+1:2d}: ROI={total_return:+7.2%}, "
                          f"Final Equity=${final_equity if final_equity is not None else 'N/A'}, "
                          f"Steps={step_count}, "
                          f"Total Reward={total_reward:+.2f}, "
                          f"Trades={n_trades}")
                else:
                    print(f"Episode {ep+1:2d}: WARNING - no total_return in info. Keys: {list(info.keys())}")
    
    print("\n" + "=" * 60)
    if returns:
        mean_roi = np.mean(returns)
        std_roi = np.std(returns)
        min_roi = np.min(returns)
        max_roi = np.max(returns)
        
        print(f"Random Agent Statistics ({len(returns)} episodes):")
        print(f"  Mean ROI:   {mean_roi:+7.2%}")
        print(f"  Std ROI:    {std_roi:7.2%}")
        print(f"  Min ROI:    {min_roi:+7.2%}")
        print(f"  Max ROI:    {max_roi:+7.2%}")
        print(f"  Mean Equity: ${np.mean(equities):.2f}")
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS:")
        if abs(mean_roi) > 0.20:
            print("❌ CRITICAL: Random agent has extreme bias (>±20%)")
            print("   → Check transaction costs, data leakage, or reward calculation")
        elif abs(mean_roi) > 0.10:
            print("⚠️  WARNING: Random agent shows significant bias (>±10%)")
            print("   → Environment may have structural issues")
        else:
            print("✅ PASS: Random agent ROI is reasonable (within ±10%)")
            print("   → Environment appears healthy for RL training")
    else:
        print("❌ CRITICAL: No valid returns collected!")
        print("   → Environment info dict missing 'total_return' or 'final_equity'")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
