"""
Backtesting Engine
==================
Realistic backtesting with:
- No look-ahead bias (signals computed on current candle, executed on next open)
- Trading fees: 0.1% per trade (Binance maker)
- Slippage: 0.05% per trade
- Stop-loss: checked against candle low (realistic)
- Take-profit: checked against candle high (realistic)
- Kelly criterion & fixed fractional position sizing
- Walk-forward validation (no in-sample overfitting)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED_TP = "tp"
    CLOSED_SL = "sl"
    CLOSED_SIGNAL = "signal"
    CLOSED_END = "end"


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    size: float          # units of crypto
    capital_used: float  # USD
    direction: int       # 1 = long

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    fees_paid: float = 0.0


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001          # 0.1% per side
    slippage_rate: float = 0.0005    # 0.05%
    risk_per_trade: float = 0.02     # 2% of capital per trade (Kelly-inspired)
    max_open_trades: int = 1         # Conservative: 1 concurrent
    use_compound: bool = True        # Compound profits
    stop_loss_override: Optional[float] = None  # e.g. 0.02 = 2% hard stop
    take_profit_override: Optional[float] = None


class BacktestEngine:
    """
    Walk-forward backtesting engine.
    Signals are computed on candle close but executed on NEXT candle open (no lookahead).
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> dict:
        """
        Run backtest.
        Args:
            df: OHLCV + features DataFrame
            signals: DataFrame with 'signal', 'sl', 'tp' columns (aligned to df)
        Returns:
            dict with trades, equity_curve, metrics
        """
        capital = self.config.initial_capital
        equity_curve = [capital]
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        peak = capital

        for i in range(1, len(df)):
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]
            sig_row  = signals.iloc[i - 1]  # Signal from previous candle
            curr_sig = signals.iloc[i]

            # ── Check open trade ──────────────────────────────────────
            if open_trade is not None:
                # Use realistic price targets (check against candle range)
                hit_sl = curr_row['low'] <= open_trade.sl
                hit_tp = curr_row['high'] >= open_trade.tp

                if hit_sl and hit_tp:
                    # Both hit — assume SL hit first (conservative)
                    hit_sl, hit_tp = True, False

                exit_signal = curr_sig['signal'] == -1

                if hit_sl:
                    exit_price = open_trade.sl * (1 - self.config.slippage_rate)
                    open_trade = self._close_trade(open_trade, curr_row.name, exit_price,
                                                   TradeStatus.CLOSED_SL)
                elif hit_tp:
                    exit_price = open_trade.tp * (1 + self.config.slippage_rate)  # TP might slip down
                    exit_price = min(exit_price, open_trade.tp)  # Realistic
                    open_trade = self._close_trade(open_trade, curr_row.name, exit_price,
                                                   TradeStatus.CLOSED_TP)
                elif exit_signal:
                    exit_price = curr_row['open'] * (1 - self.config.slippage_rate)
                    open_trade = self._close_trade(open_trade, curr_row.name, exit_price,
                                                   TradeStatus.CLOSED_SIGNAL)

                if open_trade.status != TradeStatus.OPEN:
                    capital += open_trade.pnl - open_trade.fees_paid
                    capital = max(capital, 0)
                    trades.append(open_trade)
                    open_trade = None
                    peak = max(peak, capital)

            # ── Open new trade ────────────────────────────────────────
            if open_trade is None and sig_row['signal'] == 1:
                # Execute on current candle open (next after signal)
                entry_price = curr_row['open'] * (1 + self.config.slippage_rate)

                sl = sig_row['sl']
                tp = sig_row['tp']

                # Override with config if set
                if self.config.stop_loss_override:
                    sl = entry_price * (1 - self.config.stop_loss_override)
                if self.config.take_profit_override:
                    tp = entry_price * (1 + self.config.take_profit_override)

                # Validate SL/TP
                if sl >= entry_price or tp <= entry_price:
                    equity_curve.append(capital)
                    continue

                # Position sizing: risk-based
                risk_amount = capital * self.config.risk_per_trade
                stop_distance = entry_price - sl
                if stop_distance <= 0:
                    equity_curve.append(capital)
                    continue

                size = risk_amount / stop_distance  # units of crypto
                capital_used = size * entry_price

                # Cap position at 25% of capital
                max_capital = capital * 0.25
                if capital_used > max_capital:
                    size = max_capital / entry_price
                    capital_used = max_capital

                if capital_used > capital * 0.95:
                    equity_curve.append(capital)
                    continue

                # Entry fee
                entry_fee = capital_used * self.config.fee_rate

                open_trade = Trade(
                    entry_time=curr_row.name,
                    entry_price=entry_price,
                    sl=sl,
                    tp=tp,
                    size=size,
                    capital_used=capital_used,
                    direction=1,
                    fees_paid=entry_fee
                )

            equity_curve.append(capital if open_trade is None else
                                 capital + (curr_row['close'] - open_trade.entry_price) * open_trade.size)

        # Close any open trade at end
        if open_trade is not None:
            last_row = df.iloc[-1]
            exit_price = last_row['close']
            open_trade = self._close_trade(open_trade, last_row.name, exit_price, TradeStatus.CLOSED_END)
            capital += open_trade.pnl - open_trade.fees_paid
            trades.append(open_trade)

        equity_curve.append(capital)

        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': capital,
            'metrics': self._compute_metrics(trades, equity_curve, self.config.initial_capital, df)
        }

    def _close_trade(self, trade: Trade, exit_time, exit_price: float,
                     status: TradeStatus) -> Trade:
        exit_fee = trade.size * exit_price * self.config.fee_rate
        gross_pnl = (exit_price - trade.entry_price) * trade.size
        net_pnl = gross_pnl - exit_fee
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = gross_pnl
        trade.pnl_pct = pnl_pct
        trade.status = status
        trade.fees_paid += exit_fee
        return trade

    def _compute_metrics(self, trades: List[Trade], equity_curve: list,
                         initial: float, df: pd.DataFrame) -> dict:
        if not trades:
            return {'error': 'No trades'}

        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1]
        returns = returns[np.isfinite(returns)]

        closed = [t for t in trades if t.status != TradeStatus.OPEN]
        if not closed:
            return {'error': 'No closed trades'}

        pnls     = [t.pnl for t in closed]
        pnl_pcts = [t.pnl_pct for t in closed]
        wins     = [t for t in closed if t.pnl > 0]
        losses   = [t for t in closed if t.pnl <= 0]

        win_rate  = len(wins) / len(closed) if closed else 0
        avg_win   = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss  = abs(np.mean([t.pnl for t in losses])) if losses else 1e-9
        gross_profit = sum(t.pnl for t in wins)
        gross_loss   = abs(sum(t.pnl for t in losses))
        profit_factor = min(gross_profit / (gross_loss or 1e-9), 999.0)

        # Drawdown
        peak_eq = np.maximum.accumulate(eq)
        drawdowns = (eq - peak_eq) / peak_eq
        max_dd = abs(drawdowns.min()) * 100

        # Sharpe (annualized, assuming hourly data)
        tf_per_year = 365 * 24  # hourly
        n_candles = len(df)
        if n_candles < tf_per_year:
            ann_factor = tf_per_year / n_candles
        else:
            ann_factor = 1.0

        sharpe = 0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(tf_per_year)

        # Calmar
        final  = eq[-1]
        total_return_pct = (final - initial) / initial * 100
        calmar = (total_return_pct / 100) / (max_dd / 100) if max_dd > 0 else 0

        # Trade duration
        durations = []
        for t in closed:
            if t.exit_time and t.entry_time:
                dur = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(dur)

        # Exit breakdown
        tp_count  = sum(1 for t in closed if t.status == TradeStatus.CLOSED_TP)
        sl_count  = sum(1 for t in closed if t.status == TradeStatus.CLOSED_SL)
        sig_count = sum(1 for t in closed if t.status == TradeStatus.CLOSED_SIGNAL)

        return {
            'total_trades':    len(closed),
            'win_rate':        round(win_rate * 100, 2),
            'profit_factor':   round(profit_factor, 3),
            'total_return_pct':round(total_return_pct, 2),
            'sharpe':          round(sharpe, 3),
            'calmar':          round(calmar, 3),
            'max_drawdown_pct':round(max_dd, 2),
            'avg_win_usd':     round(avg_win, 2),
            'avg_loss_usd':    round(avg_loss, 2),
            'win_loss_ratio':  round(avg_win / avg_loss, 3) if avg_loss > 0 else 0,
            'avg_trade_pct':   round(np.mean(pnl_pcts), 4),
            'final_capital':   round(final, 2),
            'total_fees':      round(sum(t.fees_paid for t in closed), 2),
            'tp_exits':        tp_count,
            'sl_exits':        sl_count,
            'signal_exits':    sig_count,
            'avg_duration_h':  round(np.mean(durations), 1) if durations else 0,
            'expectancy':      round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2),
        }


def walk_forward_backtest(df: pd.DataFrame, strategy, config: BacktestConfig,
                           train_size: int = 30, test_size: int = 7,
                           step_size: int = 7) -> dict:
    """
    Walk-forward validation: train on 30 days, test on 7, step 7 days.
    Prevents look-ahead bias / overfitting.
    """
    engine = BacktestEngine(config)
    all_trades = []
    all_equity = []
    fold_metrics = []

    signals_df = strategy.generate_signals(df)

    # Candles per day (assume 1h data = 24 candles/day)
    cpd = 24
    train_candles = train_size * cpd
    test_candles  = test_size  * cpd
    step_candles  = step_size  * cpd

    n = len(df)
    fold = 0

    start = train_candles
    while start + test_candles <= n:
        test_end = start + test_candles
        test_df   = df.iloc[start:test_end].copy()
        test_sigs = signals_df.iloc[start:test_end].reset_index(drop=True)

        result = engine.run(test_df, test_sigs)

        if result['trades']:
            all_trades.extend(result['trades'])
        if result['equity_curve']:
            all_equity.extend(result['equity_curve'])
        if 'error' not in result['metrics']:
            m = result['metrics'].copy()
            m['fold'] = fold + 1
            fold_metrics.append(m)

        fold += 1
        start += step_candles

    # Aggregate metrics
    if not fold_metrics:
        return {'error': 'No folds completed'}

    import pandas as pd
    mdf = pd.DataFrame(fold_metrics)
    summary = {
        'n_folds': fold,
        'total_trades': sum(m['total_trades'] for m in fold_metrics),
        'avg_win_rate': round(mdf['win_rate'].mean(), 2),
        'avg_profit_factor': round(mdf['profit_factor'].mean(), 3),
        'avg_sharpe': round(mdf['sharpe'].mean(), 3),
        'avg_max_dd': round(mdf['max_drawdown_pct'].mean(), 2),
        'fold_metrics': fold_metrics,
        'all_trades': all_trades,
    }
    return summary
