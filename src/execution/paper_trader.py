"""
Paper Trader - Simulated trading on Binance Vision API.

Executes trades through risk manager and logs all activity.
Uses Binance Vision API for unrestricted public market data (no credentials required).
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import json
import time
import sys

# Note: Adjusting sys.path if necessary, but for this snippet we'll keep the logic self-contained
# sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET,
    SYMBOL, PROJECT_ROOT
)
from src.execution.risk_manager import (
    RiskManager, TradeRequest, TradeResponse, TradeDecision
)


@dataclass
class Trade:
    """Trade record."""
    timestamp: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    value: float
    fee: float
    stop_loss: float
    take_profit: float
    status: str  # 'open', 'closed', 'stopped'
    pnl: float = 0.0
    close_price: float = 0.0
    close_time: str = ""


class PaperTrader:
    """
    Paper trading system for testing strategies.
    
    Can operate in two modes:
    1. Vision API mode: Connects to Binance Vision for real prices (No credentials)
    2. Simulation mode: Pure simulation without exchange connection
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        use_testnet: bool = BINANCE_TESTNET,
        symbol: str = SYMBOL,
        log_dir: Path = None
    ):
        """
        Initialize paper trader.
        
        Args:
            initial_balance: Starting balance in USDT
            use_testnet: Whether to connect to Binance (Vision API)
            symbol: Trading pair
            log_dir: Directory for trade logs
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.symbol = symbol
        self.use_testnet = use_testnet
        self.log_dir = log_dir or PROJECT_ROOT / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Risk manager
        self.risk_manager = RiskManager()
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        
        # Exchange connection (optional)
        self.exchange = None
        if use_testnet:
            self._connect_exchange()
        
        # Metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': initial_balance
        }
    
    def _connect_exchange(self):
        """Connect to Binance for public market data (no credentials needed)."""
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            # Override the public API URL to use the Vision endpoint
            # (geo-unrestricted for public market data, no API key required).
            # CCXT 4.x expects the full base path including /v3.
            self.exchange.urls['api']['public'] = 'https://data-api.binance.vision/api/v3'

            # Test connection by fetching a ticker
            ticker = self.exchange.fetch_ticker(self.symbol)
            print(f"Connected to Binance Vision API. {self.symbol}: ${ticker['last']:,.2f}")

        except Exception as e:
            print(f"Vision API failed ({e}), trying default api.binance.com...")
            try:
                # Fallback: standard Binance public API (works in most regions)
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                ticker = self.exchange.fetch_ticker(self.symbol)
                print(f"Connected to Binance API. {self.symbol}: ${ticker['last']:,.2f}")
            except Exception as e2:
                print(f"Binance API also failed: {e2}")
                print("Running in pure simulation mode.")
                self.exchange = None

    
    def get_current_price(self) -> float:
        """Get current market price."""
        if self.exchange:
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                return ticker['last']
            except Exception as e:
                print(f"Error fetching price: {e}")
        
        # Fallback: return 0 (caller should handle)
        return 0.0
    
    def execute_signal(
        self,
        action: int,
        current_price: float,
        current_volatility: float
    ) -> Optional[Trade]:
        """
        Execute a trading signal.
        
        Args:
            action: 0=Buy, 1=Hold, 2=Sell
            current_price: Current market price
            current_volatility: Current volatility reading
            
        Returns:
            Trade object if executed, None otherwise
        """
        # Check if we need to close existing position first
        if action == 2 and self.symbol in self.open_positions:
            return self._close_position(current_price, "signal")
        
        if action == 0 and self.symbol in self.open_positions:
            # Already have a position, skip
            return None
        
        # Create trade request
        request = TradeRequest(
            action=action,
            symbol=self.symbol,
            current_price=current_price,
            current_volatility=current_volatility,
            balance=self.balance,
            current_positions=len(self.open_positions)
        )
        
        # Evaluate through risk manager
        response = self.risk_manager.evaluate_trade(request)
        
        if response.decision != TradeDecision.APPROVED:
            print(f"Trade rejected: {response.reason}")
            return None
        
        if response.position_size == 0:
            return None
        
        # Execute trade
        trade = self._execute_trade(action, response, current_price)
        
        return trade
    
    def _execute_trade(
        self,
        action: int,
        response: TradeResponse,
        current_price: float
    ) -> Trade:
        """Execute the actual trade."""
        side = 'buy' if action == 0 else 'sell'
        fee = response.position_size * 0.001  # 0.1% fee
        
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            symbol=self.symbol,
            side=side,
            quantity=response.quantity,
            price=current_price,
            value=response.position_size,
            fee=fee,
            stop_loss=response.stop_loss_price,
            take_profit=response.take_profit_price,
            status='open'
        )
        
        # Update balance
        self.balance -= response.position_size + fee
        
        # Track position
        self.open_positions[self.symbol] = trade
        self.trades.append(trade)
        self.metrics['total_trades'] += 1
        
        print(f"Executed {side.upper()}: {response.quantity:.6f} {self.symbol} @ ${current_price:.2f}")
        print(f"  Stop Loss: ${response.stop_loss_price:.2f}, Take Profit: ${response.take_profit_price:.2f}")
        
        return trade
    
    def _close_position(self, current_price: float, reason: str) -> Optional[Trade]:
        """Close an open position."""
        if self.symbol not in self.open_positions:
            return None
        
        trade = self.open_positions[self.symbol]
        
        # Calculate PnL
        if trade.side == 'buy':
            pnl = (current_price - trade.price) / trade.price * trade.value
        else:
            pnl = (trade.price - current_price) / trade.price * trade.value
        
        # Subtract closing fee
        close_fee = trade.value * 0.001
        pnl -= close_fee
        
        # Update trade record
        trade.pnl = pnl
        trade.close_price = current_price
        trade.close_time = datetime.now().isoformat()
        trade.status = 'closed' if reason == 'signal' else 'stopped'
        
        # Update balance
        self.balance += trade.value + pnl
        
        # Update metrics
        self.metrics['total_pnl'] += pnl
        if pnl > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        
        # Track drawdown
        if self.balance > self.metrics['peak_balance']:
            self.metrics['peak_balance'] = self.balance
        drawdown = (self.metrics['peak_balance'] - self.balance) / self.metrics['peak_balance']
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
        
        # Remove from open positions
        del self.open_positions[self.symbol]
        
        print(f"Closed position: PnL ${pnl:.2f} ({pnl/trade.value*100:.2f}%)")
        
        return trade
    
    def check_stop_loss(self, current_price: float) -> Optional[Trade]:
        """Check and execute stop-loss if triggered."""
        if self.symbol not in self.open_positions:
            return None
        
        trade = self.open_positions[self.symbol]
        is_long = trade.side == 'buy'
        
        should_close, pnl_pct = self.risk_manager.check_stop_loss(
            trade.price, current_price, is_long
        )
        
        if should_close:
            print(f"Stop-loss triggered at {pnl_pct*100:.2f}%")
            return self._close_position(current_price, "stop_loss")
        
        return None
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Get total portfolio value including open positions."""
        value = self.balance
        
        for symbol, trade in self.open_positions.items():
            if trade.side == 'buy':
                unrealized = (current_price - trade.price) / trade.price * trade.value
            else:
                unrealized = (trade.price - current_price) / trade.price * trade.value
            value += trade.value + unrealized
        
        return value
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        total_trades = self.metrics['total_trades']
        
        metrics = {
            **self.metrics,
            'current_balance': self.balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance,
            'win_rate': self.metrics['winning_trades'] / max(1, total_trades),
            'open_positions': len(self.open_positions)
        }
        
        # Calculate Sharpe ratio (simplified)
        if self.trades:
            pnls = [t.pnl for t in self.trades if t.status != 'open']
            if pnls:
                metrics['sharpe_ratio'] = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
        
        return metrics
    
    def save_trades(self, filename: str = None):
        """Save trade history to CSV."""
        if not self.trades:
            print("No trades to save")
            return
        
        filename = filename or f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.log_dir / filename
        
        df = pd.DataFrame([asdict(t) for t in self.trades])
        df.to_csv(filepath, index=False)
        print(f"Trades saved to {filepath}")
    
    def save_metrics(self, filename: str = None):
        """Save metrics to JSON."""
        filename = filename or f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        metrics = self.get_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {filepath}")
    
    def print_summary(self):
        """Print trading summary."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 50)
        print("PAPER TRADING SUMMARY")
        print("=" * 50)
        print(f"Initial Balance:  ${self.initial_balance:,.2f}")
        print(f"Current Balance:  ${metrics['current_balance']:,.2f}")
        print(f"Total Return:     {metrics['total_return']*100:.2f}%")
        print(f"Total PnL:        ${metrics['total_pnl']:,.2f}")
        print("-" * 50)
        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"Winning Trades:   {metrics['winning_trades']}")
        print(f"Losing Trades:    {metrics['losing_trades']}")
        print(f"Win Rate:         {metrics['win_rate']*100:.1f}%")
        print(f"Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
        if 'sharpe_ratio' in metrics:
            print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    # Test paper trader
    # By setting use_testnet=True, it will now connect to the Vision API
    trader = PaperTrader(initial_balance=10000, use_testnet=True)
    
    # Simulate some trades
    prices = [50000, 50500, 51000, 50800, 50200, 49500, 50000]
    volatilities = [0.02, 0.025, 0.022, 0.03, 0.028, 0.035, 0.025]
    actions = [0, 1, 1, 1, 2, 0, 2]  # Buy, Hold, Hold, Hold, Sell, Buy, Sell
    
    for price, vol, action in zip(prices, volatilities, actions):
        trader.execute_signal(action, price, vol)
        trader.check_stop_loss(price)
        time.sleep(0.1)
    
    trader.print_summary()
    trader.save_trades()
    trader.save_metrics()