"""
MT5 Signal Executor - Trading Signal Execution Engine
Sends SMC trading signals to MetaTrader 5 for order execution
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order type classifications"""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"

class OrderResult(Enum):
    """Order execution result"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"

@dataclass
class TradingSignal:
    """Trading signal structure for MT5 execution"""
    symbol: str
    signal_type: OrderType
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    comment: str = ""
    magic_number: int = 12345
    confidence: float = 0.0
    quality_grade: str = ""
    confluence_factors: List[str] = None
    risk_reward_ratio: float = 0.0
    
class MT5SignalExecutor:
    """
    Professional MT5 Signal Execution Engine
    
    Provides:
    - Market and pending order execution
    - Position management and monitoring
    - Risk validation before order placement
    - Order modification and closure
    - Real-time position tracking
    - Paper trading simulation mode
    """
    
    def __init__(self,
                 paper_trading: bool = True,
                 magic_number: int = 12345,
                 max_spread_pips: float = 3.0,
                 max_slippage_points: int = 10,
                 enable_risk_checks: bool = True):
        """
        Initialize MT5 Signal Executor
        
        Args:
            paper_trading: Enable paper trading mode (no real orders)
            magic_number: Magic number for order identification
            max_spread_pips: Maximum spread allowed for execution
            max_slippage_points: Maximum slippage allowed
            enable_risk_checks: Enable pre-execution risk validation
        """
        self.paper_trading = paper_trading
        self.magic_number = magic_number
        self.max_spread_pips = max_spread_pips
        self.max_slippage_points = max_slippage_points
        self.enable_risk_checks = enable_risk_checks
        
        # Paper trading storage
        self.paper_positions = {}
        self.paper_orders = {}
        self.paper_balance = 10000.0  # Starting paper balance
        self.paper_equity = 10000.0
        self.order_counter = 1
        
        # Performance tracking
        self.executed_signals = []
        self.position_history = []
    
    def validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Validate trading signal before execution
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic validation
            if signal.volume <= 0:
                return False, "Invalid volume: must be positive"
            
            if signal.entry_price <= 0:
                return False, "Invalid entry price: must be positive"
            
            if signal.stop_loss <= 0:
                return False, "Invalid stop loss: must be positive"
            
            if signal.take_profit <= 0:
                return False, "Invalid take profit: must be positive"
            
            # Risk/reward validation
            if signal.signal_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP]:
                # Buy signal validation
                if signal.stop_loss >= signal.entry_price:
                    return False, "Stop loss must be below entry price for buy signals"
                if signal.take_profit <= signal.entry_price:
                    return False, "Take profit must be above entry price for buy signals"
            else:
                # Sell signal validation  
                if signal.stop_loss <= signal.entry_price:
                    return False, "Stop loss must be above entry price for sell signals"
                if signal.take_profit >= signal.entry_price:
                    return False, "Take profit must be below entry price for sell signals"
            
            # Spread validation (if not paper trading)
            if not self.paper_trading and self.enable_risk_checks:
                current_price = self._get_current_price(signal.symbol)
                if current_price:
                    spread = current_price['ask'] - current_price['bid']
                    symbol_info = self._get_symbol_info(signal.symbol)
                    if symbol_info:
                        spread_pips = spread / symbol_info['point']
                        if spread_pips > self.max_spread_pips:
                            return False, f"Spread too wide: {spread_pips:.1f} pips > {self.max_spread_pips} pips"
            
            return True, "Signal validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def execute_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Execute trading signal
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Execution result dictionary
        """
        # Validate signal first
        is_valid, validation_message = self.validate_signal(signal)
        if not is_valid:
            logger.error(f"Signal validation failed: {validation_message}")
            return {
                'result': OrderResult.REJECTED,
                'message': validation_message,
                'signal': signal,
                'order_id': None
            }
        
        if self.paper_trading:
            return self._execute_paper_signal(signal)
        else:
            return self._execute_live_signal(signal)
    
    def _execute_paper_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute signal in paper trading mode"""
        try:
            order_id = f"PAPER_{self.order_counter}"
            self.order_counter += 1
            
            # Simulate order execution
            execution_price = signal.entry_price
            
            # For market orders, simulate current market price
            if signal.signal_type in [OrderType.BUY, OrderType.SELL]:
                current_price = self._get_current_price(signal.symbol)
                if current_price:
                    if signal.signal_type == OrderType.BUY:
                        execution_price = current_price['ask']
                    else:
                        execution_price = current_price['bid']
            
            # Create paper position
            position = {
                'order_id': order_id,
                'signal': signal,
                'execution_price': execution_price,
                'execution_time': datetime.now(),
                'volume': signal.volume,
                'unrealized_pnl': 0.0,
                'status': 'OPEN'
            }
            
            self.paper_positions[order_id] = position
            self.executed_signals.append(signal)
            
            logger.info(f"Paper trade executed: {signal.signal_type.value} {signal.volume} {signal.symbol} at {execution_price}")
            
            return {
                'result': OrderResult.SUCCESS,
                'message': f"Paper trade executed successfully",
                'signal': signal,
                'order_id': order_id,
                'execution_price': execution_price,
                'execution_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Paper trade execution failed: {str(e)}")
            return {
                'result': OrderResult.FAILED,
                'message': f"Paper trade execution error: {str(e)}",
                'signal': signal,
                'order_id': None
            }
    
    def _execute_live_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute signal in live trading mode"""
        try:
            # Check MT5 connection
            if not self._ensure_mt5_connection():
                return {
                    'result': OrderResult.FAILED,
                    'message': "MT5 connection failed",
                    'signal': signal,
                    'order_id': None
                }
            
            # Prepare order request
            request = self._prepare_order_request(signal)
            if request is None:
                return {
                    'result': OrderResult.FAILED,
                    'message': "Failed to prepare order request",
                    'signal': signal,
                    'order_id': None
                }
            
            # Send order to MT5
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed: {error}")
                return {
                    'result': OrderResult.FAILED,
                    'message': f"MT5 order send failed: {error}",
                    'signal': signal,
                    'order_id': None
                }
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order execution failed: {result.retcode} - {result.comment}")
                return {
                    'result': OrderResult.FAILED,
                    'message': f"Order failed: {result.comment}",
                    'signal': signal,
                    'order_id': None
                }
            
            # Order executed successfully
            self.executed_signals.append(signal)
            
            logger.info(f"Live order executed: {signal.signal_type.value} {signal.volume} {signal.symbol}")
            logger.info(f"Order ID: {result.order}, Price: {result.price}")
            
            return {
                'result': OrderResult.SUCCESS,
                'message': "Live order executed successfully",
                'signal': signal,
                'order_id': result.order,
                'execution_price': result.price,
                'execution_time': datetime.now(),
                'mt5_result': result
            }
            
        except Exception as e:
            logger.error(f"Live order execution failed: {str(e)}")
            return {
                'result': OrderResult.FAILED,
                'message': f"Live order execution error: {str(e)}",
                'signal': signal,
                'order_id': None
            }
    
    def _prepare_order_request(self, signal: TradingSignal) -> Optional[Dict]:
        """Prepare MT5 order request"""
        try:
            # Get symbol info for proper volume and price formatting
            symbol_info = self._get_symbol_info(signal.symbol)
            if symbol_info is None:
                logger.error(f"Symbol info not available for {signal.symbol}")
                return None
            
            # Determine order type
            if signal.signal_type == OrderType.BUY:
                order_type = mt5.ORDER_TYPE_BUY
                price = 0.0  # Market price
            elif signal.signal_type == OrderType.SELL:
                order_type = mt5.ORDER_TYPE_SELL
                price = 0.0  # Market price
            elif signal.signal_type == OrderType.BUY_LIMIT:
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
                price = signal.entry_price
            elif signal.signal_type == OrderType.SELL_LIMIT:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT
                price = signal.entry_price
            elif signal.signal_type == OrderType.BUY_STOP:
                order_type = mt5.ORDER_TYPE_BUY_STOP
                price = signal.entry_price
            elif signal.signal_type == OrderType.SELL_STOP:
                order_type = mt5.ORDER_TYPE_SELL_STOP
                price = signal.entry_price
            else:
                logger.error(f"Unsupported order type: {signal.signal_type}")
                return None
            
            # Normalize volume
            volume = max(signal.volume, symbol_info['volume_min'])
            volume = min(volume, symbol_info['volume_max'])
            
            # Round volume to step size
            volume_steps = volume / symbol_info['volume_step']
            volume = round(volume_steps) * symbol_info['volume_step']
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL if signal.signal_type in [OrderType.BUY, OrderType.SELL] else mt5.TRADE_ACTION_PENDING,
                "symbol": signal.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": self.max_slippage_points,
                "magic": self.magic_number,
                "comment": signal.comment or f"SMC_{signal.quality_grade}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            return request
            
        except Exception as e:
            logger.error(f"Error preparing order request: {str(e)}")
            return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if self.paper_trading:
            return self._get_paper_positions()
        else:
            return self._get_live_positions()
    
    def _get_paper_positions(self) -> List[Dict]:
        """Get paper trading positions"""
        positions = []
        
        for order_id, position in self.paper_positions.items():
            if position['status'] == 'OPEN':
                # Update unrealized PnL
                current_price = self._get_current_price(position['signal'].symbol)
                if current_price:
                    self._update_paper_position_pnl(position, current_price)
                
                positions.append({
                    'order_id': order_id,
                    'symbol': position['signal'].symbol,
                    'type': position['signal'].signal_type.value,
                    'volume': position['volume'],
                    'entry_price': position['execution_price'],
                    'current_pnl': position['unrealized_pnl'],
                    'stop_loss': position['signal'].stop_loss,
                    'take_profit': position['signal'].take_profit,
                    'open_time': position['execution_time']
                })
        
        return positions
    
    def _get_live_positions(self) -> List[Dict]:
        """Get live MT5 positions"""
        if not self._ensure_mt5_connection():
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                if pos.magic == self.magic_number:  # Only our positions
                    position_list.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'volume': pos.volume,
                        'entry_price': pos.price_open,
                        'current_price': pos.price_current,
                        'current_pnl': pos.profit,
                        'stop_loss': pos.sl,
                        'take_profit': pos.tp,
                        'open_time': datetime.fromtimestamp(pos.time),
                        'swap': pos.swap,
                        'commission': pos.commission
                    })
            
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting live positions: {str(e)}")
            return []
    
    def close_position(self, order_id: str, reason: str = "Manual close") -> Dict[str, Any]:
        """Close a position"""
        if self.paper_trading:
            return self._close_paper_position(order_id, reason)
        else:
            return self._close_live_position(order_id, reason)
    
    def _close_paper_position(self, order_id: str, reason: str) -> Dict[str, Any]:
        """Close paper trading position"""
        if order_id not in self.paper_positions:
            return {
                'result': False,
                'message': f"Position {order_id} not found"
            }
        
        position = self.paper_positions[order_id]
        if position['status'] != 'OPEN':
            return {
                'result': False,
                'message': f"Position {order_id} is not open"
            }
        
        # Update final PnL
        current_price = self._get_current_price(position['signal'].symbol)
        if current_price:
            self._update_paper_position_pnl(position, current_price)
        
        # Close position
        position['status'] = 'CLOSED'
        position['close_time'] = datetime.now()
        position['close_reason'] = reason
        
        # Update paper balance
        self.paper_balance += position['unrealized_pnl']
        self.paper_equity = self.paper_balance
        
        # Add to history
        self.position_history.append(position.copy())
        
        logger.info(f"Paper position closed: {order_id}, PnL: ${position['unrealized_pnl']:.2f}")
        
        return {
            'result': True,
            'message': f"Paper position closed successfully",
            'pnl': position['unrealized_pnl'],
            'close_time': position['close_time']
        }
    
    def _close_live_position(self, ticket: int, reason: str) -> Dict[str, Any]:
        """Close live MT5 position"""
        # Implementation for live position closure
        # This would use mt5.order_send with TRADE_ACTION_DEAL to close
        pass
    
    def _update_paper_position_pnl(self, position: Dict, current_price: Dict):
        """Update paper position PnL"""
        try:
            signal = position['signal']
            entry_price = position['execution_price']
            volume = position['volume']
            
            # Get symbol info for proper PnL calculation
            symbol_info = self._get_symbol_info(signal.symbol)
            if symbol_info is None:
                return
            
            # Determine current price based on position type
            if signal.signal_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP]:
                current = current_price['bid']  # Exit at bid for buy positions
                price_diff = current - entry_price
            else:
                current = current_price['ask']  # Exit at ask for sell positions  
                price_diff = entry_price - current
            
            # Calculate PnL in account currency (simplified)
            pip_value = symbol_info['contract_size'] * symbol_info['point']
            pnl = price_diff * volume * pip_value
            
            position['unrealized_pnl'] = pnl
            
        except Exception as e:
            logger.error(f"Error updating paper position PnL: {str(e)}")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        if self.paper_trading:
            return self._get_paper_summary()
        else:
            return self._get_live_summary()
    
    def _get_paper_summary(self) -> Dict[str, Any]:
        """Get paper trading summary"""
        open_positions = self._get_paper_positions()
        closed_positions = [p for p in self.position_history if p['status'] == 'CLOSED']
        
        total_trades = len(closed_positions)
        winning_trades = len([p for p in closed_positions if p['unrealized_pnl'] > 0])
        losing_trades = len([p for p in closed_positions if p['unrealized_pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(p['unrealized_pnl'] for p in closed_positions)
        
        return {
            'mode': 'paper_trading',
            'balance': self.paper_balance,
            'equity': self.paper_equity,
            'open_positions': len(open_positions),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'executed_signals': len(self.executed_signals)
        }
    
    def _get_live_summary(self) -> Dict[str, Any]:
        """Get live trading summary"""
        # Implementation for live trading summary
        return {'mode': 'live_trading', 'message': 'Live summary not implemented yet'}
    
    def _get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for symbol"""
        try:
            if not self._ensure_mt5_connection():
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid
            }
        except Exception:
            return None
    
    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        try:
            if not self._ensure_mt5_connection():
                return None
            
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            
            return {
                'point': info.point,
                'digits': info.digits,
                'volume_min': info.volume_min,
                'volume_max': info.volume_max,
                'volume_step': info.volume_step,
                'contract_size': info.contract_size
            }
        except Exception:
            return None
    
    def _ensure_mt5_connection(self) -> bool:
        """Ensure MT5 connection for live trading"""
        if self.paper_trading:
            return True
        
        try:
            account_info = mt5.account_info()
            return account_info is not None
        except Exception:
            return False

# Factory function for signal executor
def create_signal_executor(paper_trading: bool = True,
                          magic_number: int = 12345) -> MT5SignalExecutor:
    """
    Factory function to create signal executor
    
    Args:
        paper_trading: Enable paper trading mode
        magic_number: Magic number for order identification
        
    Returns:
        Configured MT5SignalExecutor instance
    """
    return MT5SignalExecutor(
        paper_trading=paper_trading,
        magic_number=magic_number
    )