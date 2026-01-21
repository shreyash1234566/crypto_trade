"""
Risk Manager - Position sizing and trade filtering.

Implements:
- Fixed fractional position sizing (2% risk per trade)
- Volatility filter (reject high volatility trades)
- Maximum position limits
- Stop-loss enforcement
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    MAX_RISK_PER_TRADE, MAX_POSITIONS, 
    VOLATILITY_THRESHOLD, STOP_LOSS_PCT
)


class TradeDecision(Enum):
    """Trade decision outcomes."""
    APPROVED = "approved"
    REJECTED_VOLATILITY = "rejected_high_volatility"
    REJECTED_MAX_POSITIONS = "rejected_max_positions"
    REJECTED_INSUFFICIENT_BALANCE = "rejected_insufficient_balance"
    REJECTED_RISK_LIMIT = "rejected_risk_limit"


@dataclass
class TradeRequest:
    """Trade request data."""
    action: int  # 0=Buy, 1=Hold, 2=Sell
    symbol: str
    current_price: float
    current_volatility: float
    balance: float
    current_positions: int


@dataclass
class TradeResponse:
    """Trade response with position sizing."""
    decision: TradeDecision
    position_size: float  # In quote currency (USDT)
    quantity: float  # In base currency (BTC)
    stop_loss_price: float
    take_profit_price: float
    reason: str


class RiskManager:
    """
    Risk manager for controlling trade execution.
    
    Responsibilities:
    - Filter trades based on volatility
    - Calculate position size based on risk
    - Enforce position limits
    - Set stop-loss levels
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = MAX_RISK_PER_TRADE,
        max_positions: int = MAX_POSITIONS,
        volatility_threshold: float = VOLATILITY_THRESHOLD,
        stop_loss_pct: float = STOP_LOSS_PCT
    ):
        """
        Initialize risk manager.
        
        Args:
            max_risk_per_trade: Maximum risk per trade (e.g., 0.02 = 2%)
            max_positions: Maximum concurrent positions
            volatility_threshold: Reject if volatility > threshold * avg
            stop_loss_pct: Stop-loss percentage (e.g., 0.03 = 3%)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.volatility_threshold = volatility_threshold
        self.stop_loss_pct = stop_loss_pct
        
        # Track volatility history for comparison
        self.volatility_history = []
        self.avg_volatility = 0.0
        
    def update_volatility(self, volatility: float):
        """Update volatility history."""
        self.volatility_history.append(volatility)
        # Keep last 100 readings
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)
        self.avg_volatility = np.mean(self.volatility_history)
    
    def evaluate_trade(self, request: TradeRequest) -> TradeResponse:
        """
        Evaluate a trade request and return decision with sizing.
        
        Args:
            request: Trade request details
            
        Returns:
            Trade response with decision and sizing
        """
        # Update volatility tracking
        self.update_volatility(request.current_volatility)
        
        # Check if action is hold (no trade needed)
        if request.action == 1:
            return TradeResponse(
                decision=TradeDecision.APPROVED,
                position_size=0,
                quantity=0,
                stop_loss_price=0,
                take_profit_price=0,
                reason="Hold action - no trade"
            )
        
        # Check volatility
        if self.avg_volatility > 0:
            vol_ratio = request.current_volatility / self.avg_volatility
            if vol_ratio > self.volatility_threshold:
                return TradeResponse(
                    decision=TradeDecision.REJECTED_VOLATILITY,
                    position_size=0,
                    quantity=0,
                    stop_loss_price=0,
                    take_profit_price=0,
                    reason=f"Volatility too high: {vol_ratio:.2f}x average"
                )
        
        # Check position limits
        if request.current_positions >= self.max_positions:
            return TradeResponse(
                decision=TradeDecision.REJECTED_MAX_POSITIONS,
                position_size=0,
                quantity=0,
                stop_loss_price=0,
                take_profit_price=0,
                reason=f"Max positions reached: {request.current_positions}/{self.max_positions}"
            )
        
        # Calculate position size based on risk
        risk_amount = request.balance * self.max_risk_per_trade
        position_size = risk_amount / self.stop_loss_pct
        
        # Don't exceed available balance
        position_size = min(position_size, request.balance * 0.95)  # Keep 5% buffer
        
        if position_size < 10:  # Minimum trade size
            return TradeResponse(
                decision=TradeDecision.REJECTED_INSUFFICIENT_BALANCE,
                position_size=0,
                quantity=0,
                stop_loss_price=0,
                take_profit_price=0,
                reason=f"Position size too small: ${position_size:.2f}"
            )
        
        # Calculate quantity
        quantity = position_size / request.current_price
        
        # Calculate stop-loss and take-profit
        if request.action == 0:  # Buy (Long)
            stop_loss_price = request.current_price * (1 - self.stop_loss_pct)
            take_profit_price = request.current_price * (1 + self.stop_loss_pct * 2)  # 2:1 R:R
        else:  # Sell (Short)
            stop_loss_price = request.current_price * (1 + self.stop_loss_pct)
            take_profit_price = request.current_price * (1 - self.stop_loss_pct * 2)
        
        return TradeResponse(
            decision=TradeDecision.APPROVED,
            position_size=position_size,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            reason="Trade approved"
        )
    
    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        is_long: bool
    ) -> Tuple[bool, float]:
        """
        Check if stop-loss should be triggered.
        
        Args:
            entry_price: Entry price of position
            current_price: Current market price
            is_long: True if long position
            
        Returns:
            Tuple of (should_close, pnl_percent)
        """
        if is_long:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        should_close = pnl_pct <= -self.stop_loss_pct
        
        return should_close, pnl_pct
    
    def get_status(self) -> Dict:
        """Get current risk manager status."""
        return {
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_positions': self.max_positions,
            'volatility_threshold': self.volatility_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'avg_volatility': self.avg_volatility,
            'volatility_readings': len(self.volatility_history)
        }


if __name__ == "__main__":
    # Test risk manager
    rm = RiskManager()
    
    # Simulate some volatility history
    for v in np.random.uniform(0.01, 0.03, 50):
        rm.update_volatility(v)
    
    print(f"Average volatility: {rm.avg_volatility:.4f}")
    
    # Test trade request
    request = TradeRequest(
        action=0,  # Buy
        symbol="BTC/USDT",
        current_price=50000,
        current_volatility=0.025,
        balance=10000,
        current_positions=0
    )
    
    response = rm.evaluate_trade(request)
    print(f"\nTrade Decision: {response.decision.value}")
    print(f"Position Size: ${response.position_size:.2f}")
    print(f"Quantity: {response.quantity:.6f} BTC")
    print(f"Stop Loss: ${response.stop_loss_price:.2f}")
    print(f"Take Profit: ${response.take_profit_price:.2f}")
    print(f"Reason: {response.reason}")
    
    # Test high volatility rejection
    request_high_vol = TradeRequest(
        action=0,
        symbol="BTC/USDT",
        current_price=50000,
        current_volatility=0.08,  # High volatility
        balance=10000,
        current_positions=0
    )
    
    response_high_vol = rm.evaluate_trade(request_high_vol)
    print(f"\nHigh Vol Decision: {response_high_vol.decision.value}")
    print(f"Reason: {response_high_vol.reason}")
