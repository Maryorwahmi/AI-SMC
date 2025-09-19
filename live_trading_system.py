"""
Live SMC Trading System - Complete Integration
Combines SMC analysis with MT5 live data and signal execution
"""

import time
import logging
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from threading import Thread, Event
import signal
import sys

# SMC Engine imports
from analyzer import SMCAnalyzer
from config.settings import create_settings, Settings

# MT5 Integration imports
from mt5_integration.mt5_data_provider import MT5DataProvider, MT5Connection
from mt5_integration.mt5_signal_executor import (
    MT5SignalExecutor, TradingSignal, OrderType, create_signal_executor
)

logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """
    Complete Live SMC Trading System
    
    Orchestrates:
    - Live MT5 data fetching
    - Real-time SMC analysis
    - Signal generation and execution
    - Position monitoring and management
    - Performance tracking
    - Risk management
    """
    
    def __init__(self, 
                 settings: Optional[Settings] = None,
                 paper_trading: bool = True,
                 analysis_interval_minutes: int = 5,  # Reduced from 15 to 5 for more frequent analysis
                 symbols: Optional[List[str]] = None):
        """
        Initialize Live Trading System
        
        Args:
            settings: SMC analysis settings
            paper_trading: Enable paper trading mode
            analysis_interval_minutes: Analysis frequency in minutes
            symbols: List of symbols to trade (None for default)
        """
        self.settings = settings or create_settings("conservative", "signals_only")
        self.paper_trading = paper_trading
        self.analysis_interval = analysis_interval_minutes
        
        # Trading symbols - Updated based on Exness broker availability
        self.symbols = symbols or [
            'EURUSDm', 'AUDUSDm', 'EURAUDm', 'EURCADm', 'EURCHFm', 
            'EURJPYm', 'AUDJPYm', 'AUDCADm', 'CADJPYm', 'CHFJPYm'
        ]
        
        # Initialize components
        self.smc_analyzer = SMCAnalyzer(self.settings)
        self.data_provider = MT5DataProvider()
        self.signal_executor = create_signal_executor(
            paper_trading=paper_trading,
            magic_number=12345
        )
        
        # System state
        self.running = False
        self.shutdown_event = Event()
        self.last_analysis_time = {}
        self.active_signals = {}
        self.performance_stats = {
            'total_analyses': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'execution_errors': 0,
            'start_time': None
        }
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def start(self) -> bool:
        """Start the live trading system"""
        try:
            logger.info("[START] Starting Live SMC Trading System...")
            
            # Connect to MT5
            if not self.data_provider.connect():
                logger.error("[ERROR] Failed to connect to MT5")
                return False
            
            # Validate symbols
            valid_symbols = self._validate_symbols()
            if not valid_symbols:
                logger.error("[ERROR] No valid symbols found")
                return False
            
            self.symbols = valid_symbols
            logger.info(f"[SUCCESS] Validated symbols: {', '.join(self.symbols)}")
            
            # Initialize performance tracking
            self.performance_stats['start_time'] = datetime.now()
            
            # Schedule analysis
            self._schedule_analysis()
            
            # Start main loop
            self.running = True
            logger.info(f"[SUCCESS] Live trading system started in {'PAPER' if self.paper_trading else 'LIVE'} mode")
            logger.info(f"[CONFIG] Analysis interval: {self.analysis_interval} minutes")
            logger.info(f"[CONFIG] Trading symbols: {len(self.symbols)}")
            
            self._main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start trading system: {str(e)}")
            return False
    
    def stop(self):
        """Stop the trading system gracefully"""
        logger.info("[STOP] Stopping Live Trading System...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Close all open positions if paper trading
        if self.paper_trading:
            self._close_all_paper_positions("System shutdown")
        
        # Disconnect from MT5
        self.data_provider.disconnect()
        
        # Print final statistics
        self._print_final_stats()
        
        logger.info("[SUCCESS] Live Trading System stopped")
    
    def _schedule_analysis(self):
        """Schedule regular analysis"""
        # Schedule analysis every X minutes
        schedule.every(self.analysis_interval).minutes.do(self._run_analysis_cycle)
        
        # Schedule daily summary at market close
        schedule.every().day.at("17:00").do(self._daily_summary)  # 17:00 UTC (London close)
        
        # Schedule position monitoring every 5 minutes
        schedule.every(5).minutes.do(self._monitor_positions)
    
    def _main_loop(self):
        """Main system loop"""
        try:
            # Run initial analysis
            self._run_analysis_cycle()
            
            while self.running and not self.shutdown_event.is_set():
                # Run scheduled tasks
                schedule.run_pending()
                
                # Check for immediate market conditions
                self._check_market_conditions()
                
                # Sleep for 1 minute
                if self.shutdown_event.wait(60):  # 60 seconds timeout
                    break
                    
        except KeyboardInterrupt:
            logger.info("[PAUSE] Keyboard interrupt received")
        except Exception as e:
            logger.error(f"[ERROR] Main loop error: {str(e)}")
        finally:
            self.stop()
    
    def _run_analysis_cycle(self):
        """Run complete analysis cycle for all symbols"""
        try:
            logger.info("[ANALYSIS] Starting analysis cycle...")
            analysis_start = datetime.now()
            
            # Check market hours
            market_info = self.data_provider.get_market_hours()
            if not market_info.get('is_major_session', False):
                logger.info("[TIME] Outside major trading sessions, skipping analysis")
                return
            
            symbols_analyzed = 0
            signals_generated = 0
            
            for symbol in self.symbols:
                try:
                    logger.info(f"[SYMBOL] Analyzing {symbol}...")
                    
                    # Get multi-timeframe data
                    symbol_data = self.data_provider.get_multi_timeframe_data(
                        symbol, 
                        timeframes=['H4', 'H1', 'M15'],
                        count=200
                    )
                    
                    if len(symbol_data) < 3:
                        logger.warning(f"[WARNING] Insufficient data for {symbol}")
                        continue
                    
                    logger.info(f"[DATA] {symbol} - Retrieved {len(symbol_data)} timeframes")
                    for tf, data in symbol_data.items():
                        logger.info(f"  └─ {tf}: {len(data)} bars, Latest: {data.index[-1] if len(data) > 0 else 'No data'}")
                    
                    # Get current price
                    current_price_info = self.data_provider.get_current_price(symbol)
                    if not current_price_info:
                        logger.warning(f"[WARNING] No current price for {symbol}")
                        continue
                    
                    current_price = (current_price_info['bid'] + current_price_info['ask']) / 2
                    logger.info(f"[PRICE] {symbol} - Bid: {current_price_info['bid']}, Ask: {current_price_info['ask']}, Mid: {current_price:.5f}")
                    
                    # Perform SMC analysis
                    logger.info(f"[SMC] Running SMC analysis for {symbol}...")
                    analysis = self.smc_analyzer.analyze_symbol(
                        symbol, symbol_data, current_price
                    )
                    
                    if 'error' in analysis:
                        logger.error(f"[ERROR] Analysis error for {symbol}: {analysis['error']}")
                        continue
                    
                    symbols_analyzed += 1
                    
                    # Log analysis results
                    logger.info(f"[RESULTS] {symbol} Analysis Results:")
                    logger.info(f"  ├─ Market Bias: {analysis.get('market_bias', 'Unknown')}")
                    logger.info(f"  ├─ Strength: {analysis.get('strength', 'Unknown')}")
                    logger.info(f"  ├─ Confluence Score: {analysis.get('confluence_score', 0):.2f}")
                    logger.info(f"  ├─ Active Factors: {len(analysis.get('confluence_factors', []))}")
                    
                    # Log confluence factors
                    factors = analysis.get('confluence_factors', [])
                    if factors:
                        logger.info(f"  ├─ Confluence Factors:")
                        for factor in factors[:5]:  # Show first 5
                            logger.info(f"  │  └─ {factor}")
                    
                    # Process signals
                    signals = analysis.get('signals', [])
                    logger.info(f"  └─ Signals Generated: {len(signals)}")
                    
                    if signals:
                        for i, signal_data in enumerate(signals, 1):
                            logger.info(f"    └─ Signal {i}: {signal_data.get('type', 'Unknown')} at {signal_data.get('entry_price', 'N/A')}")
                            logger.info(f"       ├─ Quality: {signal_data.get('quality_rating', 'Unknown')}")
                            logger.info(f"       ├─ Confidence: {signal_data.get('confidence_percentage', 0):.1f}%")
                            logger.info(f"       └─ R:R Ratio: {signal_data.get('rr_ratio', 0):.1f}")
                    
                    for signal_data in signals:
                        signal = self._convert_to_trading_signal(symbol, signal_data, current_price_info)
                        if signal:
                            execution_result = self.signal_executor.execute_signal(signal)
                            if execution_result['result'].value == 'SUCCESS':
                                signals_generated += 1
                                self.active_signals[execution_result['order_id']] = {
                                    'signal': signal,
                                    'execution_result': execution_result,
                                    'analysis': analysis
                                }
                                
                                logger.info(f"[SUCCESS] Signal executed: {signal.signal_type.value} {signal.symbol}")
                            else:
                                logger.warning(f"[WARNING] Signal execution failed: {execution_result['message']}")
                    
                    # Update last analysis time
                    self.last_analysis_time[symbol] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"[ERROR] Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Update statistics
            self.performance_stats['total_analyses'] += symbols_analyzed
            self.performance_stats['signals_generated'] += signals_generated
            
            analysis_duration = (datetime.now() - analysis_start).total_seconds()
            logger.info(f"[ANALYSIS] Analysis cycle completed in {analysis_duration:.1f}s")
            logger.info(f"[SUMMARY] Cycle Summary:")
            logger.info(f"  ├─ Symbols processed: {symbols_analyzed}")
            logger.info(f"  ├─ Signals generated: {signals_generated}")
            logger.info(f"  ├─ Processing time: {analysis_duration:.1f}s")
            logger.info(f"  └─ Average time per symbol: {analysis_duration/max(symbols_analyzed,1):.1f}s")
            
        except Exception as e:
            logger.error(f"[ERROR] Analysis cycle error: {str(e)}")
    
    def _convert_to_trading_signal(self, symbol: str, signal_data: Dict, price_info: Dict) -> Optional[TradingSignal]:
        """Convert SMC signal to trading signal"""
        try:
            # Determine order type based on signal
            signal_type_str = signal_data.get('type', '').upper()
            current_price = (price_info['bid'] + price_info['ask']) / 2
            entry_price = signal_data.get('entry_price', current_price)
            
            # Map signal types
            if 'BUY' in signal_type_str:
                if entry_price <= current_price * 1.001:  # Within 0.1% of current price
                    order_type = OrderType.BUY  # Market buy
                else:
                    order_type = OrderType.BUY_LIMIT  # Buy limit
            elif 'SELL' in signal_type_str:
                if entry_price >= current_price * 0.999:  # Within 0.1% of current price
                    order_type = OrderType.SELL  # Market sell
                else:
                    order_type = OrderType.SELL_LIMIT  # Sell limit
            else:
                logger.warning(f"[WARNING] Unknown signal type: {signal_type_str}")
                return None
            
            # Calculate position size using risk management
            stop_loss = signal_data.get('stop_loss')
            if not stop_loss:
                logger.warning(f"[WARNING] No stop loss for {symbol} signal")
                return None
            
            # Use conservative position sizing
            base_volume = 0.1  # Default volume
            risk_percentage = self.settings.trading.risk_per_trade
            
            # Calculate volume based on risk
            if order_type in [OrderType.BUY, OrderType.BUY_LIMIT]:
                risk_distance = abs(entry_price - stop_loss)
            else:
                risk_distance = abs(stop_loss - entry_price)
            
            if risk_distance > 0:
                # Calculate position size (simplified)
                account_balance = 10000.0  # Default for paper trading
                risk_amount = account_balance * risk_percentage
                
                # Get symbol info for proper calculation
                symbol_info = self.data_provider.get_symbol_info(symbol)
                if symbol_info:
                    pip_value = symbol_info['contract_size'] * symbol_info['point']
                    risk_distance_pips = risk_distance / symbol_info['point']
                    
                    if risk_distance_pips > 0:
                        volume = risk_amount / (risk_distance_pips * pip_value)
                        volume = max(volume, symbol_info['volume_min'])
                        volume = min(volume, symbol_info['volume_max'])
                        
                        # Round to step size
                        volume_steps = volume / symbol_info['volume_step']
                        volume = round(volume_steps) * symbol_info['volume_step']
                        base_volume = volume
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=order_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=signal_data.get('take_profit', entry_price * 1.02 if order_type in [OrderType.BUY, OrderType.BUY_LIMIT] else entry_price * 0.98),
                volume=base_volume,
                comment=f"SMC_{signal_data.get('quality_grade', 'GOOD')}",
                confidence=signal_data.get('confidence', 0.7),
                quality_grade=signal_data.get('quality_grade', 'GOOD'),
                confluence_factors=signal_data.get('confluence_factors', []),
                risk_reward_ratio=signal_data.get('rr_ratio', 2.0)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"[ERROR] Error converting signal for {symbol}: {str(e)}")
            return None
    
    def _monitor_positions(self):
        """Monitor open positions and manage them"""
        try:
            open_positions = self.signal_executor.get_open_positions()
            
            if not open_positions:
                return
            
            logger.debug(f"[MONITOR] Monitoring {len(open_positions)} open positions")
            
            for position in open_positions:
                # Check for position management rules
                self._manage_position(position)
                
        except Exception as e:
            logger.error(f"[ERROR] Position monitoring error: {str(e)}")
    
    def _manage_position(self, position: Dict):
        """Manage individual position"""
        try:
            symbol = position['symbol']
            order_id = position.get('order_id', position.get('ticket'))
            
            # Get current price
            current_price_info = self.data_provider.get_current_price(symbol)
            if not current_price_info:
                return
            
            # Check for trailing stop or other management rules
            # This can be expanded with more sophisticated position management
            
            # For now, just log position status
            pnl = position.get('current_pnl', 0)
            logger.debug(f"[POSITION] Position {order_id} {symbol}: PnL ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"[ERROR] Position management error: {str(e)}")
    
    def _check_market_conditions(self):
        """Check for immediate market conditions requiring action"""
        try:
            # Check if we're in a major session
            market_info = self.data_provider.get_market_hours()
            
            # Log session changes
            current_sessions = set(market_info.get('active_sessions', []))
            if hasattr(self, '_last_sessions') and self._last_sessions != current_sessions:
                logger.info(f"[SESSION] Session change: {list(current_sessions)}")
            
            self._last_sessions = current_sessions
            
        except Exception as e:
            logger.error(f"[ERROR] Market condition check error: {str(e)}")
    
    def _close_all_paper_positions(self, reason: str):
        """Close all paper trading positions"""
        if not self.paper_trading:
            return
        
        try:
            open_positions = self.signal_executor.get_open_positions()
            for position in open_positions:
                order_id = position.get('order_id')
                if order_id:
                    self.signal_executor.close_position(order_id, reason)
                    
        except Exception as e:
            logger.error(f"[ERROR] Error closing paper positions: {str(e)}")
    
    def _daily_summary(self):
        """Generate daily trading summary"""
        try:
            logger.info("[SUMMARY] Daily Trading Summary")
            logger.info("=" * 50)
            
            # Get trading summary
            summary = self.signal_executor.get_trading_summary()
            
            logger.info(f"Trading Mode: {summary.get('mode', 'unknown').upper()}")
            logger.info(f"Total Trades: {summary.get('total_trades', 0)}")
            logger.info(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
            logger.info(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
            
            if self.paper_trading:
                logger.info(f"Paper Balance: ${summary.get('balance', 0):.2f}")
                logger.info(f"Paper Equity: ${summary.get('equity', 0):.2f}")
            
            logger.info(f"Signals Generated Today: {self.performance_stats['signals_generated']}")
            logger.info(f"Analyses Completed: {self.performance_stats['total_analyses']}")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"[ERROR] Daily summary error: {str(e)}")
    
    def _print_final_stats(self):
        """Print final statistics"""
        try:
            runtime = datetime.now() - self.performance_stats['start_time']
            
            logger.info("[SUMMARY] Final Trading Statistics")
            logger.info("=" * 50)
            logger.info(f"Runtime: {runtime}")
            logger.info(f"Total Analyses: {self.performance_stats['total_analyses']}")
            logger.info(f"Signals Generated: {self.performance_stats['signals_generated']}")
            logger.info(f"Signals Executed: {self.performance_stats['signals_executed']}")
            
            summary = self.signal_executor.get_trading_summary()
            logger.info(f"Final Balance: ${summary.get('balance', 0):.2f}")
            logger.info(f"Total PnL: ${summary.get('total_pnl', 0):.2f}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"[ERROR] Final stats error: {str(e)}")
    
    def _validate_symbols(self) -> List[str]:
        """Validate and filter available symbols"""
        try:
            available_symbols = self.data_provider.get_available_symbols()
            valid_symbols = []
            
            for symbol in self.symbols:
                if symbol in available_symbols:
                    if self.data_provider.validate_symbol(symbol):
                        valid_symbols.append(symbol)
                    else:
                        logger.warning(f"[WARNING] Symbol {symbol} not tradeable")
                else:
                    # Try variations
                    for suffix in ['m', '', '.raw']:
                        test_symbol = symbol + suffix
                        if test_symbol in available_symbols and self.data_provider.validate_symbol(test_symbol):
                            valid_symbols.append(test_symbol)
                            logger.info(f"[SUCCESS] Found {symbol} as {test_symbol}")
                            break
                    else:
                        logger.warning(f"[WARNING] Symbol {symbol} not found")
            
            return valid_symbols
            
        except Exception as e:
            logger.error(f"[ERROR] Symbol validation error: {str(e)}")
            return []
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"[STOP] Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        sys.exit(0)

# Factory function for easy system creation
def create_live_trading_system(risk_level: str = "conservative",
                              paper_trading: bool = True,
                              symbols: Optional[List[str]] = None) -> LiveTradingSystem:
    """
    Factory function to create live trading system
    
    Args:
        risk_level: "conservative", "moderate", or "aggressive"
        paper_trading: Enable paper trading mode
        symbols: List of symbols to trade
        
    Returns:
        Configured LiveTradingSystem instance
    """
    settings = create_settings(
        risk_level=risk_level,
        trading_mode="paper_trading" if paper_trading else "live_trading"
    )
    
    return LiveTradingSystem(
        settings=settings,
        paper_trading=paper_trading,
        symbols=symbols
    )

if __name__ == "__main__":
    # Configure logging with UTF-8 encoding
    import sys
    
    # Set console encoding to UTF-8 for Windows
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # Configure logging handlers with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler('live_trading.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )
    
    print("[START] AI-SMC Live Trading System")
    print("=" * 50)
    print()
    
    # Get user preferences
    try:
        risk_level = input("Risk level (conservative/moderate/aggressive) [conservative]: ").strip().lower()
        if risk_level not in ['conservative', 'moderate', 'aggressive']:
            risk_level = 'conservative'
        
        paper_mode = input("Paper trading mode? (y/N) [Y]: ").strip().lower()
        paper_trading = paper_mode in ['y', 'yes', ''] or paper_mode == ''
        
        # Create and start system
        system = create_live_trading_system(
            risk_level=risk_level,
            paper_trading=paper_trading
        )
        
        print(f"\n[CONFIG] Starting system with {risk_level} risk level")
        print(f"[MODE] Mode: {'Paper Trading' if paper_trading else 'Live Trading'}")
        print("\n[INFO] Press Ctrl+C to stop gracefully")
        print("=" * 50)
        print()
        
        # Start the system
        if system.start():
            print("[SUCCESS] System started successfully")
        else:
            print("[ERROR] Failed to start system")
            
    except KeyboardInterrupt:
        print("\n[EXIT] Goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        logging.error(f"Main execution error: {str(e)}")