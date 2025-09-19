"""
AI-SMC Main Analyzer - Institutional-Grade SMC Engine Coordinator
Orchestrates all SMC components for comprehensive market analysis
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from config.settings import Settings
from config.constants import (
    QualityGrade, QUALITY_THRESHOLDS, TIMEFRAME_WEIGHTS, 
    QUALITY_FILTER_WEIGHTS, SESSION_CONFIG
)

# SMC Components
from smc_components.order_blocks import OrderBlockDetector
from smc_components.fair_value_gaps import FairValueGapAnalyzer
from smc_components.liquidity_zones import LiquidityZoneMapper
from smc_components.choch_detector import CHoCHDetector
from smc_components.mss_detector import MSSDetector

# Analysis modules (to be implemented)
from market_structure.structure_analyzer import MarketStructureAnalyzer
from session_analysis.session_manager import SessionManager
from signals.signal_generator import SignalGenerator
from quality.quality_filter import QualityFilter
from risk_management.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class SMCAnalyzer:
    """
    Main AI-SMC Analyzer Engine
    
    Coordinates all SMC components to provide institutional-grade analysis:
    - Multi-timeframe SMC analysis (H4/H1/M15)
    - 12-point quality filtering system
    - Session-based optimization
    - Advanced risk management
    - Real-time signal generation
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the SMC Analyzer
        
        Args:
            settings: Configuration settings (uses defaults if None)
        """
        self.settings = settings or Settings()
        
        # Initialize core SMC components
        self._initialize_smc_components()
        
        # Initialize analysis modules
        self._initialize_analysis_modules()
        
        # Initialize support systems
        self._initialize_support_systems()
        
        logger.info("AI-SMC Analyzer initialized successfully")
    
    def _initialize_smc_components(self):
        """Initialize all SMC detection components"""
        self.order_block_detector = OrderBlockDetector(
            min_displacement_pips=20.0,
            lookback_period=50,
            min_body_ratio=0.6
        )
        
        self.fvg_analyzer = FairValueGapAnalyzer(
            min_gap_size_pips=self.settings.analysis.fvg_min_size,
            enable_volume_analysis=True,
            require_momentum_confirmation=True
        )
        
        self.liquidity_mapper = LiquidityZoneMapper(
            equal_level_tolerance_pips=3.0,
            min_touch_count=2,
            sweep_threshold_pips=2.0
        )
        
        self.choch_detector = CHoCHDetector(
            swing_length=self.settings.analysis.swing_length,
            min_displacement_pips=15.0,
            confirmation_bars=3
        )
        
        self.mss_detector = MSSDetector(
            swing_length=15,
            min_displacement_pips=25.0,
            trend_validation_swings=3
        )
    
    def _initialize_analysis_modules(self):
        """Initialize analysis coordination modules"""
        self.structure_analyzer = MarketStructureAnalyzer(
            swing_length=self.settings.analysis.swing_length
        )
        
        self.session_manager = SessionManager(
            enable_session_analysis=self.settings.session.enable_session_analysis
        )
        
        self.signal_generator = SignalGenerator(
            min_confluence_factors=self.settings.analysis.min_confluence_score,
            min_rr_ratio=self.settings.analysis.min_rr_ratio,
            enhanced_mode=self.settings.quality.enable_quality_analysis,
            require_perfect_structure=self.settings.analysis.require_perfect_structure
        )
        
        self.quality_filter = QualityFilter(
            enable_12_point_filter=self.settings.quality.enable_12_point_filter,
            min_quality_score=self.settings.quality.min_quality_score
        )
    
    def _initialize_support_systems(self):
        """Initialize support systems"""
        self.risk_manager = RiskManager(
            initial_balance=self.settings.backtest.initial_balance,
            risk_level=self.settings.trading.risk_level,
            max_portfolio_risk=self.settings.trading.max_portfolio_risk
        )
    
    def analyze_symbol(self, symbol: str, data: Dict[str, pd.DataFrame],
                      current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform comprehensive SMC analysis for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            data: Dictionary with timeframe data {'H4': df, 'H1': df, 'M15': df}
            current_price: Current market price
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting comprehensive SMC analysis for {symbol}")
        
        try:
            analysis_results = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'timeframe_analysis': {},
                'smc_confluence': {},
                'signals': [],
                'quality_scores': {},
                'session_analysis': {},
                'risk_assessment': {}
            }
            
            # Log input data
            logger.info(f"[DEBUG] Input data for {symbol}:")
            for tf, df in data.items():
                if df is not None and len(df) > 0:
                    logger.info(f"  └─ {tf}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
                else:
                    logger.warning(f"  └─ {tf}: No data or empty DataFrame")
            
            # 1. Multi-timeframe SMC analysis
            logger.info(f"[DEBUG] Step 1: Multi-timeframe SMC analysis for {symbol}")
            analysis_results['timeframe_analysis'] = self._analyze_all_timeframes(
                symbol, data
            )
            
            # Log timeframe analysis results
            logger.info(f"[DEBUG] Timeframe analysis results:")
            for tf, tf_results in analysis_results['timeframe_analysis'].items():
                logger.info(f"  └─ {tf}:")
                logger.info(f"     ├─ Order Blocks: {len(tf_results.get('order_blocks', []))}")
                logger.info(f"     ├─ Fair Value Gaps: {len(tf_results.get('fair_value_gaps', []))}")
                logger.info(f"     ├─ Liquidity Zones: {len(tf_results.get('liquidity_zones', []))}")
                logger.info(f"     ├─ CHoCH Signals: {len(tf_results.get('choch_signals', []))}")
                logger.info(f"     └─ MSS Signals: {len(tf_results.get('mss_signals', []))}")
            
            # 2. Calculate SMC confluence
            logger.info(f"[DEBUG] Step 2: Calculate SMC confluence for {symbol}")
            analysis_results['smc_confluence'] = self._calculate_smc_confluence(
                analysis_results['timeframe_analysis'], current_price
            )
            
            # Log confluence results
            confluence = analysis_results['smc_confluence']
            logger.info(f"[DEBUG] SMC Confluence results:")
            logger.info(f"  ├─ Total Score: {confluence.get('total_score', 0):.2f}")
            logger.info(f"  ├─ Bias: {confluence.get('bias', 'unknown')}")
            logger.info(f"  ├─ Strength: {confluence.get('strength', 'unknown')}")
            logger.info(f"  └─ Active Factors: {len(confluence.get('active_factors', []))}")
            
            # 3. Session analysis
            logger.info(f"[DEBUG] Step 3: Session analysis for {symbol}")
            analysis_results['session_analysis'] = self.session_manager.analyze_current_session(
                current_price, analysis_results['smc_confluence']
            )
            
            # 4. Generate signals
            logger.info(f"[DEBUG] Step 4: Generate signals for {symbol}")
            signals = self.signal_generator.generate_signals(
                symbol, analysis_results['timeframe_analysis'],
                analysis_results['smc_confluence'], current_price
            )
            logger.info(f"[DEBUG] Generated {len(signals)} raw signals")
            
            # 5. Apply quality filtering
            logger.info(f"[DEBUG] Step 5: Apply quality filtering for {symbol}")
            filtered_signals = self.quality_filter.filter_signals(
                signals, analysis_results
            )
            logger.info(f"[DEBUG] Filtered to {len(filtered_signals)} quality signals")
            
            analysis_results['signals'] = filtered_signals
            
            # 6. Risk assessment
            if filtered_signals:
                logger.info(f"[DEBUG] Step 6: Risk assessment for {symbol}")
                analysis_results['risk_assessment'] = self._assess_risk(
                    symbol, filtered_signals, current_price
                )
            
            # Add summary data for external logging
            analysis_results['market_bias'] = confluence.get('bias', 'Unknown')
            analysis_results['strength'] = confluence.get('strength', 'Unknown')
            analysis_results['confluence_score'] = confluence.get('total_score', 0)
            analysis_results['confluence_factors'] = confluence.get('active_factors', [])
            
            logger.info(f"Analysis completed for {symbol}. Generated {len(filtered_signals)} quality signals")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _analyze_all_timeframes(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze all timeframes for comprehensive SMC view"""
        timeframe_results = {}
        
        for timeframe, df in data.items():
            if df is None or len(df) < 50:
                continue
                
            logger.debug(f"Analyzing {timeframe} timeframe for {symbol}")
            
            timeframe_results[timeframe] = {
                'order_blocks': self.order_block_detector.detect_order_blocks(df, symbol),
                'fair_value_gaps': self.fvg_analyzer.detect_fair_value_gaps(df, symbol),
                'liquidity_zones': self.liquidity_mapper.detect_liquidity_zones(df, symbol),
                'choch_signals': self.choch_detector.detect_choch_signals(df, symbol),
                'mss_signals': self.mss_detector.detect_mss_signals(df, symbol),
                'market_structure': self.structure_analyzer.analyze_structure(df),
                'timeframe_weight': TIMEFRAME_WEIGHTS.get(timeframe, 0.1)
            }
        
        return timeframe_results
    
    def _calculate_smc_confluence(self, timeframe_analysis: Dict, current_price: Optional[float]) -> Dict:
        """Calculate SMC confluence across timeframes"""
        confluence_data = {
            'total_score': 0.0,
            'factor_scores': {},
            'bias': 'neutral',
            'strength': 'weak',
            'active_factors': []
        }
        
        try:
            logger.info(f"[DEBUG] Starting confluence calculation...")
            factor_scores = {}
            
            # Analyze each SMC factor across timeframes
            for factor in ['order_blocks', 'fair_value_gaps', 'liquidity_zones', 'choch_signals', 'mss_signals']:
                logger.info(f"[DEBUG] Analyzing factor: {factor}")
                factor_score = 0.0
                bullish_weight = 0.0
                bearish_weight = 0.0
                
                for timeframe, analysis in timeframe_analysis.items():
                    if factor not in analysis:
                        logger.warning(f"[DEBUG] Factor {factor} not found in {timeframe} analysis")
                        continue
                    
                    tf_weight = analysis['timeframe_weight']
                    factor_data = analysis[factor]
                    
                    logger.info(f"[DEBUG] {timeframe} {factor}: {len(factor_data) if factor_data else 0} items, weight: {tf_weight}")
                    
                    if factor == 'order_blocks':
                        # Analyze order block bias
                        active_obs = [ob for ob in factor_data if ob.valid] if factor_data else []
                        logger.info(f"[DEBUG] {timeframe} active order blocks: {len(active_obs)}")
                        for ob in active_obs:
                            weight = tf_weight * (ob.displacement_pips / 50.0)
                            if ob.type.value == 'BULLISH':
                                bullish_weight += weight
                                logger.info(f"[DEBUG] Bullish OB: +{weight:.2f}")
                            else:
                                bearish_weight += weight
                                logger.info(f"[DEBUG] Bearish OB: +{weight:.2f}")
                    
                    elif factor == 'fair_value_gaps':
                        # Analyze FVG bias
                        active_fvgs = [fvg for fvg in factor_data if not fvg.filled] if factor_data else []
                        logger.info(f"[DEBUG] {timeframe} active FVGs: {len(active_fvgs)}")
                        for fvg in active_fvgs:
                            weight = tf_weight * (fvg.size_pips / 20.0)
                            if fvg.type.value == 'BULLISH':
                                bullish_weight += weight
                                logger.info(f"[DEBUG] Bullish FVG: +{weight:.2f}")
                            else:
                                bearish_weight += weight
                                logger.info(f"[DEBUG] Bearish FVG: +{weight:.2f}")

                    elif factor == 'liquidity_zones':
                        # Analyze liquidity zone bias
                        active_zones = [zone for zone in factor_data if zone.get('valid', True)] if factor_data else []
                        logger.info(f"[DEBUG] {timeframe} active liquidity zones: {len(active_zones)}")
                        for zone in active_zones:
                            if current_price is None: continue
                            # Weight based on proximity to current price
                            distance = abs(zone['price'] - current_price)
                            proximity_weight = max(0, 1 - (distance / (current_price * 0.01))) # Stronger if within 1%
                            weight = tf_weight * proximity_weight
                            
                            if zone['type'] == 'high': # Price is likely to go up to sweep liquidity
                                bearish_weight += weight
                                logger.info(f"[DEBUG] Bearish Liquidity High: +{weight:.2f}")
                            else: # Price is likely to go down to sweep liquidity
                                bullish_weight += weight
                                logger.info(f"[DEBUG] Bullish Liquidity Low: +{weight:.2f}")
                    
                    elif factor == 'choch_signals':
                        # Analyze recent CHoCH signals
                        recent_choch = self.choch_detector.get_recent_choch_signals(factor_data) if factor_data else []
                        logger.info(f"[DEBUG] {timeframe} recent CHoCH: {len(recent_choch)}")
                        for choch in recent_choch:
                            weight = tf_weight * 2.0  # CHoCH has high weight
                            if choch.type.value == 'BULLISH_CHOCH':
                                bullish_weight += weight
                                logger.info(f"[DEBUG] Bullish CHoCH: +{weight:.2f}")
                            else:
                                bearish_weight += weight
                                logger.info(f"[DEBUG] Bearish CHoCH: +{weight:.2f}")
                    
                    elif factor == 'mss_signals':
                        # Analyze recent MSS signals
                        recent_mss = self.mss_detector.get_recent_mss_signals(factor_data) if factor_data else []
                        logger.info(f"[DEBUG] {timeframe} recent MSS: {len(recent_mss)}")
                        for mss in recent_mss:
                            weight = tf_weight * 3.0  # MSS has highest weight
                            if mss.type.value == 'BULLISH_MSS':
                                bullish_weight += weight
                                logger.info(f"[DEBUG] Bullish MSS: +{weight:.2f}")
                            else:
                                bearish_weight += weight
                                logger.info(f"[DEBUG] Bearish MSS: +{weight:.2f}")
                
                # Calculate factor score and bias
                factor_score = max(bullish_weight, bearish_weight)
                factor_bias = 'bullish' if bullish_weight > bearish_weight else 'bearish'
                
                factor_scores[factor] = {
                    'score': factor_score,
                    'bias': factor_bias,
                    'bullish_weight': bullish_weight,
                    'bearish_weight': bearish_weight
                }
                
                logger.info(f"[DEBUG] Factor {factor} final: score={factor_score:.2f}, bias={factor_bias}, bull={bullish_weight:.2f}, bear={bearish_weight:.2f}")
                
                if factor_score > 0.5:  # Significant factor
                    confluence_data['active_factors'].append(factor)
                    logger.info(f"[DEBUG] Factor {factor} added to active factors (score > 0.5)")
            
            confluence_data['factor_scores'] = factor_scores
            
            # Calculate overall confluence
            total_bullish = sum(f['bullish_weight'] for f in factor_scores.values())
            total_bearish = sum(f['bearish_weight'] for f in factor_scores.values())
            
            confluence_data['total_score'] = max(total_bullish, total_bearish)
            confluence_data['bias'] = 'bullish' if total_bullish > total_bearish else 'bearish'
            
            # Classify strength
            if confluence_data['total_score'] >= 5.0:
                confluence_data['strength'] = 'very_strong'
            elif confluence_data['total_score'] >= 3.0:
                confluence_data['strength'] = 'strong'
            elif confluence_data['total_score'] >= 1.5:
                confluence_data['strength'] = 'moderate'
            else:
                confluence_data['strength'] = 'weak'
                
        except Exception as e:
            logger.error(f"Error calculating SMC confluence: {str(e)}")
        
        return confluence_data
    
    def _assess_risk(self, symbol: str, signals: List[Dict], current_price: float) -> Dict:
        """Assess risk for potential trades"""
        risk_assessment = {
            'position_sizing': {},
            'portfolio_impact': {},
            'recommendations': []
        }
        
        try:
            for signal in signals:
                entry_price = signal.get('entry_price', current_price)
                stop_loss = signal.get('stop_loss')
                
                if stop_loss:
                    position_size, position_risk = self.risk_manager.calculate_position_size(
                        symbol, entry_price, stop_loss, signal.get('confidence', 0.7)
                    )
                    
                    risk_assessment['position_sizing'][f"signal_{signal.get('id', 'unknown')}"] = {
                        'position_size': position_size,
                        'risk_amount': position_risk.risk_amount if position_risk else 0,
                        'risk_percentage': position_risk.risk_percentage if position_risk else 0
                    }
            
            # Portfolio-level assessment
            portfolio_summary = self.risk_manager.get_risk_summary()
            risk_assessment['portfolio_impact'] = portfolio_summary
            
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
        
        return risk_assessment
    
    def get_market_bias(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict:
        """Get overall market bias for symbol"""
        try:
            # This is a simplified call, so we might not have current_price
            # A full analysis would require it.
            analysis = self.analyze_symbol(symbol, data, data['M15']['close'].iloc[-1] if 'M15' in data and not data['M15'].empty else None)
            confluence = analysis.get('smc_confluence', {})
            
            return {
                'symbol': symbol,
                'bias': confluence.get('bias', 'neutral'),
                'strength': confluence.get('strength', 'weak'),
                'confidence': min(100, confluence.get('total_score', 0) * 20),
                'active_factors': confluence.get('active_factors', []),
                'timeframe_agreement': self._calculate_timeframe_agreement(analysis)
            }
            
        except Exception as e:
            logger.error(f"Error getting market bias for {symbol}: {str(e)}")
            return {'symbol': symbol, 'bias': 'neutral', 'error': str(e)}

    
    def _calculate_timeframe_agreement(self, analysis: Dict) -> Dict:
        """Calculate agreement between timeframes"""
        timeframe_analysis = analysis.get('timeframe_analysis', {})
        
        if len(timeframe_analysis) < 2:
            return {'agreement_percentage': 0, 'agreeing_timeframes': []}
        
        # Get bias from each timeframe
        timeframe_biases = {}
        for tf, tf_analysis in timeframe_analysis.items():
            # Simple bias calculation based on recent signals
            choch_signals = tf_analysis.get('choch_signals', [])
            mss_signals = tf_analysis.get('mss_signals', [])
            
            recent_choch = self.choch_detector.get_recent_choch_signals(choch_signals)
            recent_mss = self.mss_detector.get_recent_mss_signals(mss_signals)
            
            bullish_score = 0
            bearish_score = 0
            
            for signal in recent_choch + recent_mss:
                if 'BULLISH' in signal.type.value:
                    bullish_score += 1
                else:
                    bearish_score += 1
            
            if bullish_score > bearish_score:
                timeframe_biases[tf] = 'bullish'
            elif bearish_score > bullish_score:
                timeframe_biases[tf] = 'bearish'
            else:
                timeframe_biases[tf] = 'neutral'
        
        # Calculate agreement
        bias_counts = {}
        for bias in timeframe_biases.values():
            bias_counts[bias] = bias_counts.get(bias, 0) + 1
        
        if bias_counts:
            max_agreement = max(bias_counts.values())
            total_timeframes = len(timeframe_biases)
            agreement_percentage = (max_agreement / total_timeframes) * 100
            
            dominant_bias = max(bias_counts.items(), key=lambda x: x[1])[0]
            agreeing_timeframes = [tf for tf, bias in timeframe_biases.items() if bias == dominant_bias]
        else:
            agreement_percentage = 0
            agreeing_timeframes = []
        
        return {
            'agreement_percentage': agreement_percentage,
            'agreeing_timeframes': agreeing_timeframes,
            'timeframe_biases': timeframe_biases
        }
    
    def get_trading_opportunities(self, symbols: List[str], 
                                 data_provider_func) -> List[Dict]:
        """
        Get trading opportunities for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            data_provider_func: Function that returns timeframe data for a symbol
            
        Returns:
            List of trading opportunities sorted by quality
        """
        opportunities = []
        
        for symbol in symbols:
            try:
                logger.info(f"Analyzing trading opportunities for {symbol}")
                
                # Get data for all timeframes
                symbol_data = data_provider_func(symbol)
                if not symbol_data:
                    continue
                
                # Perform analysis
                analysis = self.analyze_symbol(symbol, symbol_data)
                
                # Extract signals
                signals = analysis.get('signals', [])
                if signals:
                    for signal in signals:
                        opportunity = {
                            'symbol': symbol,
                            'signal': signal,
                            'analysis': analysis,
                            'quality_grade': self._classify_signal_quality(signal),
                            'timestamp': datetime.now()
                        }
                        opportunities.append(opportunity)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by quality and strength
        opportunities.sort(key=lambda x: (
            x['quality_grade'].value,
            x['signal'].get('quality_score', 0),
            x['signal'].get('confluence_score', 0)
        ), reverse=True)
        
        return opportunities[:10]  # Return top 10 opportunities
    
    def _classify_signal_quality(self, signal: Dict) -> QualityGrade:
        """Classify signal quality based on total score"""
        total_score = signal.get('quality_score', 0) * 12  # Convert to 12-point scale
        
        for grade, threshold in QUALITY_THRESHOLDS.items():
            if total_score >= threshold:
                return grade
        
        return QualityGrade.POOR