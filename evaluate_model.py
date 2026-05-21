"""
Evaluate trained model performance with directional accuracy metrics.

Metrics reported:
- Directional accuracy (next-candle up/down prediction accuracy)
- Action distribution (buy/hold/sell breakdown)
- Reward analysis
- Trade results (win rate, PnL)
- Market context (buy & hold comparison, alpha)
"""
import pandas as pd
import numpy as np
from config.settings import REWARD_FLAT_PENALTY
from src.data.features import load_featured_data, get_feature_columns
from src.environment.trading_env import CryptoTradingEnv
from src.models.bilstm import load_model as load_bilstm, freeze_model
from src.models.ppo_agent import load_ppo_agent

print('Loading data and models...')
df = load_featured_data()
feature_cols = get_feature_columns()
feature_cols = [c for c in feature_cols if c in df.columns]

bilstm = load_bilstm('bilstm_best.pt')
freeze_model(bilstm)

# Use last 20% for test
test_start = int(len(df) * 0.8)
test_df = df.iloc[test_start:].reset_index(drop=True)
print(f'Test data: {len(test_df)} candles ({len(test_df)/96:.1f} days)')

env = CryptoTradingEnv(
    test_df,
    feature_cols,
    use_lstm_features=True,
    lstm_model=bilstm,
    flat_penalty=REWARD_FLAT_PENALTY
)
model, vec_env = load_ppo_agent(env, 'best_model', bilstm)
base_env = vec_env.envs[0]

# Run one episode manually to see trades
obs = vec_env.reset()
info = base_env._get_info()
done = False
actions_taken = []
rewards = []
step_info = []
step_info.append(info.copy())

print('\nRunning episode...')
while not done:
    action, _ = model.predict(obs, deterministic=True)
    action_scalar = int(action[0] if hasattr(action, '__len__') else action)
    obs, rewards_vec, dones, infos = vec_env.step(action)
    reward = float(rewards_vec[0])
    info = infos[0]
    done = bool(dones[0])
    actions_taken.append(action_scalar)
    rewards.append(reward)
    step_info.append(info.copy())

# Analyze actions
actions_arr = np.array(actions_taken)
print(f'\n{"="*60}')
print('ACTION DISTRIBUTION')
print(f'{"="*60}')
print(f'Total steps: {len(actions_arr)}')
print(f'Buy (0):  {np.sum(actions_arr == 0):5d} ({100*np.sum(actions_arr == 0)/len(actions_arr):5.1f}%)')
print(f'Hold (1): {np.sum(actions_arr == 1):5d} ({100*np.sum(actions_arr == 1)/len(actions_arr):5.1f}%)')
print(f'Sell (2): {np.sum(actions_arr == 2):5d} ({100*np.sum(actions_arr == 2)/len(actions_arr):5.1f}%)')

# ================================================================
# DIRECTIONAL ACCURACY -- The core metric for 70-75% target
# ================================================================
print(f'\n{"="*60}')
print('DIRECTIONAL ACCURACY (Next-Candle Prediction)')
print(f'{"="*60}')

# Map actions to predicted direction: Buy=Up(1), Sell=Down(0), Hold=Skip
correct_dir = 0
total_dir = 0
correct_by_class = {'up': 0, 'down': 0}
total_by_class = {'up': 0, 'down': 0}

for i in range(len(actions_arr)):
    action = actions_arr[i]
    
    # Skip Hold actions -- they don't make a directional prediction
    if action == 1:
        continue
    
    # Predicted direction: Buy(0)=Up, Sell(2)=Down
    predicted_up = (action == 0)
    
    # Actual direction: next candle close vs current close
    current_step = step_info[i]['step']
    if current_step + 1 >= len(test_df):
        continue
        
    actual_up = test_df.iloc[current_step + 1]['close'] > test_df.iloc[current_step]['close']
    
    total_dir += 1
    if actual_up:
        total_by_class['up'] += 1
    else:
        total_by_class['down'] += 1
    
    if predicted_up == actual_up:
        correct_dir += 1
        if actual_up:
            correct_by_class['up'] += 1
        else:
            correct_by_class['down'] += 1

if total_dir > 0:
    dir_accuracy = correct_dir / total_dir
    print(f'Directional predictions made: {total_dir}')
    print(f'Correct predictions:          {correct_dir}')
    print(f'Directional Accuracy:         {dir_accuracy:.2%}')
    
    # Per-class accuracy
    if total_by_class['up'] > 0:
        up_acc = correct_by_class['up'] / total_by_class['up']
        print(f'  Up prediction accuracy:     {up_acc:.2%} ({correct_by_class["up"]}/{total_by_class["up"]})')
    if total_by_class['down'] > 0:
        down_acc = correct_by_class['down'] / total_by_class['down']
        print(f'  Down prediction accuracy:   {down_acc:.2%} ({correct_by_class["down"]}/{total_by_class["down"]})')
    
    # Target assessment
    if dir_accuracy >= 0.75:
        print(f'\n🏆 EXCEEDS TARGET: {dir_accuracy:.1%} > 75%')
    elif dir_accuracy >= 0.70:
        print(f'\n🎯 TARGET ACHIEVED: {dir_accuracy:.1%} ∈ [70%, 75%]')
    elif dir_accuracy >= 0.60:
        print(f'\n📈 PROGRESS: {dir_accuracy:.1%} -- approaching 70% target')
    else:
        print(f'\n[WARN]  BELOW TARGET: {dir_accuracy:.1%} -- needs improvement')
else:
    print('No directional predictions made (all Hold actions)')

# Analyze rewards
print(f'\n{"="*60}')
print('REWARD ANALYSIS')
print(f'{"="*60}')
total_reward = sum(rewards)
pos_rewards = sum(r for r in rewards if r > 0)
neg_rewards = sum(r for r in rewards if r < 0)
print(f'Total reward:     {total_reward:8.2f}')
print(f'Positive rewards: {pos_rewards:8.2f}')
print(f'Negative rewards: {neg_rewards:8.2f}')

# Analyze trades
print(f'\n{"="*60}')
print('TRADE RESULTS')
print(f'{"="*60}')
final_info = step_info[-1]
print(f'Number of trades: {final_info.get("n_trades", 0)}')
print(f'Initial balance:  $10,000.00')
print(f'Final balance:    ${final_info.get("balance", 0):,.2f}')
print(f'Total return:     {final_info.get("total_return", 0)*100:.2f}%')

# Get trade history from environment
if hasattr(base_env, 'trades') and base_env.trades:
    print(f'\n{"="*60}')
    print('INDIVIDUAL TRADES')
    print(f'{"="*60}')
    wins = 0
    losses = 0
    total_pnl = 0
    for i, trade in enumerate(base_env.trades):
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_pct', 0) * 100
        direction = 'LONG' if trade.get('direction', 1) == 1 else 'SHORT'
        status = '✓' if pnl > 0 else '✗'
        print(f'Trade {i+1:2d}: {direction:5s} | PnL: ${pnl:8.2f} ({pnl_pct:+6.2f}%) {status}')
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    
    print(f'\n{"="*60}')
    print('TRADE SUMMARY')
    print(f'{"="*60}')
    print(f'Wins:       {wins}')
    print(f'Losses:     {losses}')
    print(f'Win Rate:   {100*wins/(wins+losses) if (wins+losses) > 0 else 0:.1f}%')
    print(f'Total PnL:  ${total_pnl:,.2f}')
    
    if wins + losses > 0:
        avg_win = sum(t['pnl'] for t in base_env.trades if t['pnl'] > 0) / max(wins, 1)
        avg_loss = sum(t['pnl'] for t in base_env.trades if t['pnl'] <= 0) / max(losses, 1)
        print(f'Avg Win:    ${avg_win:,.2f}')
        print(f'Avg Loss:   ${avg_loss:,.2f}')
        if abs(avg_loss) > 0:
            print(f'Risk/Reward:{abs(avg_win/avg_loss):.2f}x')

# Price movement during test period
print(f'\n{"="*60}')
print('MARKET CONTEXT')
print(f'{"="*60}')
start_price = test_df.iloc[0]['close']
end_price = test_df.iloc[-1]['close']
buy_hold = (end_price - start_price) / start_price * 100
print(f'BTC Start Price: ${start_price:,.2f}')
print(f'BTC End Price:   ${end_price:,.2f}')
print(f'Buy & Hold:      {buy_hold:+.2f}%')
print(f'Model Return:    {final_info.get("total_return", 0)*100:+.2f}%')
print(f'Alpha:           {final_info.get("total_return", 0)*100 - buy_hold:+.2f}%')
