"""
MT5 Data Provider - Live Market Data Integration
Fetches real-time OHLC data from MetaTrader 5 for SMC analysis
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
import logging
import time

logger = logging.getLogger(__name__)

class MT5DataProvider:
    """
    Professional MT5 Data Provider for SMC Analysis
    
    Provides:
    - Multi-timeframe data fetching (H4, H1, M15)
    - Real-time price updates
    - Symbol information and specifications
    - Connection management with auto-reconnect
    - Data validation and cleaning
    """
    
    def __init__(self, 
                 login: Optional[int] = None,
                 password: Optional[str] = None,
                 server: Optional[str] = None,
                 timeout: int = 60,
                 max_retries: int = 3):
        """
        Initialize MT5 Data Provider
        
        Args:
            login: MT5 account login (None for auto-detect)
            password: MT5 account password (None for auto-detect)
            server: MT5 server (None for auto-detect)
            timeout: Connection timeout in seconds
            max_retries: Maximum connection retry attempts
        """
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.max_retries = max_retries
        self.connected = False
        self.account_info = None
        
        # Timeframe mappings
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        # Default symbols for forex
        self.default_symbols = [
            'EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'USDCADm',
            'USDCHFm', 'NZDUSDm', 'EURGBPm', 'EURJPYm', 'GBPJPYm'
        ]
    
    def connect(self) -> bool:
        """
        Connect to MetaTrader 5
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False
            
            # If credentials provided, login with them
            if self.login and self.password and self.server:
                if not mt5.login(self.login, password=self.password, server=self.server):
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    mt5.shutdown()
                    return False
            
            # Get account information
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account information")
                mt5.shutdown()
                return False
            
            self.connected = True
            logger.info(f"MT5 connected successfully")
            logger.info(f"Account: {self.account_info.login}, Server: {self.account_info.server}")
            logger.info(f"Balance: {self.account_info.balance}, Equity: {self.account_info.equity}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
    
    def ensure_connection(self) -> bool:
        """Ensure MT5 connection is active"""
        if not self.connected:
            return self.connect()
        
        # Test connection with a simple call
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("MT5 connection lost, attempting to reconnect...")
                self.connected = False
                return self.connect()
            return True
        except Exception:
            logger.warning("MT5 connection test failed, attempting to reconnect...")
            self.connected = False
            return self.connect()
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol information and specifications
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol information or None
        """
        if not self.ensure_connection():
            return None
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Symbol {symbol} not found")
                return None
            
            return {
                'symbol': symbol_info.name,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': symbol_info.spread,
                'trade_mode': symbol_info.trade_mode,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'contract_size': getattr(symbol_info, 'trade_contract_size', 100000.0)  # Default to 100k for forex
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current bid/ask prices for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with current prices or None
        """
        if not self.ensure_connection():
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}")
                return None
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time),
                'volume': tick.volume
            }
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, 
                           symbol: str, 
                           timeframe: str = 'H1', 
                           count: int = 200,
                           start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Get historical OHLC data for symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            count: Number of bars to fetch
            start_date: Start date (uses count from current time if None)
            
        Returns:
            DataFrame with OHLC data or None
        """
        if not self.ensure_connection():
            return None
        
        if timeframe not in self.timeframes:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        try:
            mt5_timeframe = self.timeframes[timeframe]
            
            if start_date:
                # Get data from specific date
                rates = mt5.copy_rates_from(symbol, mt5_timeframe, start_date, count)
            else:
                # Get most recent data
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to standard format
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # Remove any duplicate timestamps
            df = df[~df.index.duplicated(keep='last')]
            
            # Sort by time
            df.sort_index(inplace=True)
            
            logger.debug(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol} {timeframe}: {str(e)}")
            return None
    
    def get_multi_timeframe_data(self, 
                                symbol: str, 
                                timeframes: List[str] = ['H4', 'H1', 'M15'],
                                count: int = 200) -> Dict[str, pd.DataFrame]:
        """
        Get multi-timeframe data for SMC analysis
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to fetch
            count: Number of bars per timeframe
            
        Returns:
            Dictionary with timeframe data
        """
        data = {}
        
        for tf in timeframes:
            df = self.get_historical_data(symbol, tf, count)
            if df is not None:
                data[tf] = df
            else:
                logger.warning(f"Failed to get {tf} data for {symbol}")
        
        return data
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols
        
        Returns:
            List of available symbols
        """
        if not self.ensure_connection():
            return []
        
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            
            # Filter for forex symbols (customize as needed)
            forex_symbols = []
            for symbol in symbols:
                symbol_name = symbol.name
                # Common forex symbol patterns
                if (any(pair in symbol_name.upper() for pair in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']) and
                    len(symbol_name) >= 6 and 
                    symbol.trade_mode != 0):  # Tradeable symbols only
                    forex_symbols.append(symbol_name)
            
            return forex_symbols[:50]  # Return first 50 forex symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available for trading
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if symbol is valid and tradeable
        """
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return False
        
        return symbol_info['trade_mode'] != 0  # 0 = disabled
    
    def get_market_hours(self) -> Dict:
        """
        Get current market session information
        
        Returns:
            Dictionary with market session info
        """
        try:
            now = datetime.now(pytz.UTC)
            
            # Define major session times (UTC)
            sessions = {
                'asian': {'start': 23, 'end': 8},     # 23:00-08:00 UTC
                'london': {'start': 8, 'end': 17},    # 08:00-17:00 UTC  
                'new_york': {'start': 13, 'end': 22}, # 13:00-22:00 UTC
                'overlap': {'start': 13, 'end': 17}   # 13:00-17:00 UTC (London/NY)
            }
            
            current_hour = now.hour
            active_sessions = []
            
            for session, times in sessions.items():
                start = times['start']
                end = times['end']
                
                if start < end:  # Same day session
                    if start <= current_hour < end:
                        active_sessions.append(session)
                else:  # Crosses midnight
                    if current_hour >= start or current_hour < end:
                        active_sessions.append(session)
            
            return {
                'current_time_utc': now,
                'current_hour_utc': current_hour,
                'active_sessions': active_sessions,
                'is_major_session': len(active_sessions) > 0,
                'is_overlap': 'overlap' in active_sessions
            }
            
        except Exception as e:
            logger.error(f"Error getting market hours: {str(e)}")
            return {'active_sessions': [], 'is_major_session': False}
    
    def get_account_summary(self) -> Optional[Dict]:
        """
        Get account summary information
        
        Returns:
            Dictionary with account information
        """
        if not self.ensure_connection():
            return None
        
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            return {
                'login': account.login,
                'server': account.server,
                'name': account.name,
                'company': account.company,
                'currency': account.currency,
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'profit': account.profit,
                'credit': account.credit
            }
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            return None

# Context manager for automatic connection handling
class MT5Connection:
    """Context manager for MT5 connections"""
    
    def __init__(self, provider: MT5DataProvider):
        self.provider = provider
    
    def __enter__(self):
        if not self.provider.connect():
            raise ConnectionError("Failed to connect to MT5")
        return self.provider
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.provider.disconnect()

# Factory function for easy provider creation
def create_mt5_provider(login: Optional[int] = None,
                       password: Optional[str] = None,
                       server: Optional[str] = None) -> MT5DataProvider:
    """
    Factory function to create MT5 data provider
    
    Args:
        login: MT5 account login (None for auto-detect)
        password: MT5 account password (None for auto-detect)  
        server: MT5 server (None for auto-detect)
        
    Returns:
        Configured MT5DataProvider instance
    """
    return MT5DataProvider(login=login, password=password, server=server)