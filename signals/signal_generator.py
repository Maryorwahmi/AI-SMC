"""
Signal Generator
Generates trading signals from SMC analysis
"""

from typing import List, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Trading Signal Generator
    Generates trading signals based on SMC confluence analysis
    """
    
    def __init__(self, min_confluence_factors: float = 2.0, 
                 min_rr_ratio: float = 2.0, 
                 enhanced_mode: bool = True,
                 require_perfect_structure: bool = True):
        self.min_confluence_score = min_confluence_factors
        self.min_active_factors = 2
        self.min_rr_ratio = min_rr_ratio
        self.enhanced_mode = enhanced_mode
        self.require_perfect_structure = require_perfect_structure
        
        logger.info(f"[INIT] Signal Generator initialized with:")
        logger.info(f"  ├─ Min confluence score: {self.min_confluence_score}")
        logger.info(f"  ├─ Min active factors: {self.min_active_factors}")
        logger.info(f"  ├─ Min R:R ratio: {min_rr_ratio}")
        logger.info(f"  ├─ Enhanced mode: {enhanced_mode}")
        logger.info(f"  └─ Require perfect structure: {self.require_perfect_structure}")
    
    def generate_signals(self, symbol: str, timeframe_analysis: Dict,
                        confluence: Dict, current_price: Optional[float]) -> List[Dict]:
        """Generate trading signals from SMC analysis"""
        logger.info(f"[SIGNAL] Generating signals for {symbol}")
        logger.info(f"[SIGNAL] Input confluence: score={confluence.get('total_score', 0)}, "
                   f"bias={confluence.get('bias', 'none')}, "
                   f"active_factors={len(confluence.get('active_factors', []))}")
        
        signals = []
        
        # Check minimum requirements
        score = confluence.get('total_score', 0)
        active_factors = confluence.get('active_factors', [])
        active_factor_count = len(active_factors) if isinstance(active_factors, list) else 0
        bias = confluence.get('bias', 'none')
        
        # Check for perfect structure if required
        if self.require_perfect_structure:
            has_choch = any(tf_analysis.get('choch_signals') for tf_analysis in timeframe_analysis.values())
            has_mss = any(tf_analysis.get('mss_signals') for tf_analysis in timeframe_analysis.values())
            
            if not (has_choch or has_mss):
                logger.warning(f"[SIGNAL] Rejected: Perfect structure (CHoCH/MSS) required but not found")
                return []
        
        logger.info(f"[SIGNAL] Checking requirements:")
        logger.info(f"  ├─ Score {score} >= {self.min_confluence_score}: {score >= self.min_confluence_score}")
        logger.info(f"  ├─ Active factors {active_factor_count} >= {self.min_active_factors}: {active_factor_count >= self.min_active_factors}")
        logger.info(f"  └─ Has bias '{bias}': {bias in ['bullish', 'bearish']}")
        
        if (score >= self.min_confluence_score and 
            active_factor_count >= self.min_active_factors and 
            bias in ['bullish', 'bearish'] and 
            current_price is not None):
            
            logger.info(f"[SIGNAL] Requirements met - generating {bias} signal")
            
            # Calculate entry, stop loss, and take profit levels
            signal = self._create_signal(symbol, bias, current_price, timeframe_analysis, confluence)
            if signal:
                signals.append(signal)
                logger.info(f"[SIGNAL] Created signal: {signal['type']} at {signal['entry_price']}")
            else:
                logger.warning(f"[SIGNAL] Failed to create signal for {symbol}")
        else:
            logger.info(f"[SIGNAL] Requirements not met - no signal generated")
        
        logger.info(f"[SIGNAL] Generated {len(signals)} signals for {symbol}")
        return signals
    
    def _create_signal(self, symbol: str, bias: str, current_price: float, 
                      timeframe_analysis: Dict, confluence: Dict) -> Optional[Dict]:
        """Create a trading signal with entry, SL, and TP levels"""
        try:
            # Calculate ATR-based levels (simplified approach)
            # In a real implementation, this would use actual SMC levels
            atr_estimate = current_price * 0.01  # 1% of price as ATR estimate
            
            if bias == 'bullish':
                entry_price = current_price
                stop_loss = current_price - (atr_estimate * 1.5)
                take_profit = current_price + (atr_estimate * 3.0)
                signal_type = 'BUY'
            else:  # bearish
                entry_price = current_price
                stop_loss = current_price + (atr_estimate * 1.5)
                take_profit = current_price - (atr_estimate * 3.0)
                signal_type = 'SELL'
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            logger.info(f"[SIGNAL] Calculated levels:")
            logger.info(f"  ├─ Entry: {entry_price:.5f}")
            logger.info(f"  ├─ Stop Loss: {stop_loss:.5f}")
            logger.info(f"  ├─ Take Profit: {take_profit:.5f}")
            logger.info(f"  └─ R:R Ratio: {rr_ratio:.2f}")
            
            if rr_ratio >= self.min_rr_ratio - 0.01:  # Allow small floating point variance
                return {
                    'symbol': symbol,
                    'type': signal_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'rr_ratio': rr_ratio,
                    'confluence_score': confluence.get('total_score', 0),
                    'active_factors': len(confluence.get('active_factors', [])),
                    'bias': bias,
                    'strength': confluence.get('strength', 'unknown'),
                    'timestamp': pd.Timestamp.now()
                }
            else:
                logger.warning(f"[SIGNAL] R:R ratio {rr_ratio:.2f} below minimum {self.min_rr_ratio}")
                return None
                
        except Exception as e:
            logger.error(f"[SIGNAL] Error creating signal for {symbol}: {e}")
            return None