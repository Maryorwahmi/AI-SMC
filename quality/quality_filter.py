"""
Quality Filter - Institutional-Grade Signal Assessment
Implements comprehensive 12-point quality filtering system for SMC analysis
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

class QualityRating(Enum):
    """Signal quality classification"""
    EXCELLENT = "EXCELLENT"     # 90-100% confidence
    HIGH = "HIGH"              # 75-89% confidence  
    GOOD = "GOOD"              # 60-74% confidence
    MODERATE = "MODERATE"      # 45-59% confidence
    LOW = "LOW"                # 30-44% confidence
    POOR = "POOR"              # Below 30% confidence

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    confluence_score: float = 0.0
    active_factors: int = 0
    timeframe_alignment: float = 0.0
    structure_strength: float = 0.0
    volume_confirmation: float = 0.0
    session_timing: float = 0.0
    risk_reward_ratio: float = 0.0
    market_structure_bias: float = 0.0
    liquidity_zones: float = 0.0
    order_block_quality: float = 0.0
    fvg_quality: float = 0.0
    trend_alignment: float = 0.0
    overall_score: float = 0.0
    confidence_percentage: float = 0.0
    quality_rating: QualityRating = QualityRating.POOR

class QualityFilter:
    """
    Professional 12-Point Quality Assessment System
    
    Evaluates signals across 12 institutional criteria:
    1. Confluence Score (Multi-factor confirmation)
    2. Active SMC Factors (Order blocks, FVGs, etc.)
    3. Multi-timeframe Alignment 
    4. Market Structure Strength
    5. Volume Confirmation
    6. Session Timing (London/NY overlap)
    7. Risk-Reward Ratio
    8. Market Structure Bias
    9. Liquidity Zone Proximity
    10. Order Block Quality
    11. Fair Value Gap Quality
    12. Trend Alignment
    """
    
    def __init__(self, 
                 enable_12_point_filter: bool = True, 
                 min_quality_score: float = 0.45,  # Lowered from 0.6 to 0.45
                 min_confidence: float = 45.0):    # Lowered from 50.0 to 45.0
        """
        Initialize Quality Filter
        
        Args:
            enable_12_point_filter: Enable comprehensive quality assessment
            min_quality_score: Minimum quality score (0.0-1.0)
            min_confidence: Minimum confidence percentage
        """
        self.enable_12_point_filter = enable_12_point_filter
        self.min_quality_score = min_quality_score
        self.min_confidence = min_confidence
        
        # Quality scoring weights (must sum to 1.0)
        self.weights = {
            'confluence': 0.20,      # 20% - Core SMC confluence
            'active_factors': 0.15,  # 15% - Number of confirming factors
            'timeframe_alignment': 0.12,  # 12% - Multi-TF confirmation
            'structure_strength': 0.10,   # 10% - Market structure quality
            'volume_confirmation': 0.08,  # 8% - Volume analysis
            'session_timing': 0.08,      # 8% - Trading session
            'risk_reward': 0.07,         # 7% - R:R ratio
            'market_bias': 0.06,         # 6% - Bias strength
            'liquidity_zones': 0.05,     # 5% - Liquidity proximity
            'order_blocks': 0.04,        # 4% - OB quality
            'fvg_quality': 0.03,         # 3% - FVG quality
            'trend_alignment': 0.02      # 2% - Trend confirmation
        }
    
    def filter_signals(self, signals: List[Dict], analysis: Dict) -> List[Dict]:
        """
        Filter signals using comprehensive 12-point quality system
        
        Args:
            signals: List of raw trading signals
            analysis: Complete SMC analysis data
            
        Returns:
            List of quality-filtered signals with metrics
        """
        if not self.enable_12_point_filter:
            return signals
        
        filtered_signals = []
        
        for signal in signals:
            # Assess signal quality
            quality_metrics = self._assess_signal_quality(signal, analysis)
            
            # Add quality metrics to signal
            signal.update({
                'quality_metrics': quality_metrics,
                'quality_rating': quality_metrics.quality_rating.value,
                'confidence_percentage': quality_metrics.confidence_percentage,
                'overall_quality_score': quality_metrics.overall_score
            })
            
            # Filter based on quality requirements
            if (quality_metrics.overall_score >= self.min_quality_score and 
                quality_metrics.confidence_percentage >= self.min_confidence):
                filtered_signals.append(signal)
                
                # Debug logging for passed signals
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[QUALITY] ✅ Signal PASSED quality filter:")
                logger.info(f"  ├─ Symbol: {signal.get('symbol', 'Unknown')}")
                logger.info(f"  ├─ Overall Score: {quality_metrics.overall_score:.3f} >= {self.min_quality_score}")
                logger.info(f"  ├─ Confidence: {quality_metrics.confidence_percentage:.1f}% >= {self.min_confidence}%")
                logger.info(f"  └─ Quality Rating: {quality_metrics.quality_rating.value}")
            else:
                # Debug logging for rejected signals
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[QUALITY] ❌ Signal REJECTED by quality filter:")
                logger.info(f"  ├─ Symbol: {signal.get('symbol', 'Unknown')}")
                logger.info(f"  ├─ Overall Score: {quality_metrics.overall_score:.3f} vs Required: {self.min_quality_score}")
                logger.info(f"  ├─ Confidence: {quality_metrics.confidence_percentage:.1f}% vs Required: {self.min_confidence}%")
                logger.info(f"  ├─ Confluence Score: {quality_metrics.confluence_score:.3f}")
                logger.info(f"  ├─ Active Factors: {quality_metrics.active_factors:.3f}")
                logger.info(f"  └─ Quality Rating: {quality_metrics.quality_rating.value}")
        
        return filtered_signals
    
    def _assess_signal_quality(self, signal: Dict, analysis: Dict) -> QualityMetrics:
        """Comprehensive quality assessment using 12-point system"""
        
        metrics = QualityMetrics()
        
        # 1. Confluence Score Assessment (20%)
        confluence_data = analysis.get('smc_confluence', {})
        confluence_score = confluence_data.get('total_score', 0)
        metrics.confluence_score = min(confluence_score / 10.0, 1.0)  # Normalize to 0-1
        
        # 2. Active SMC Factors (15%)
        active_factors = confluence_data.get('active_factors', [])
        factor_count = len(active_factors) if isinstance(active_factors, list) else active_factors
        metrics.active_factors = min(factor_count / 5.0, 1.0)  # Max 5 factors
        
        # 3. Multi-timeframe Alignment (12%)
        timeframe_data = analysis.get('timeframe_analysis', {})
        metrics.timeframe_alignment = self._assess_timeframe_alignment(timeframe_data, signal)
        
        # 4. Market Structure Strength (10%)
        structure_strength = confluence_data.get('strength', 'weak')
        strength_map = {
            'very_strong': 1.0,
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.3,
            'very_weak': 0.1
        }
        metrics.structure_strength = strength_map.get(structure_strength, 0.3)
        
        # 5. Volume Confirmation (8%)
        metrics.volume_confirmation = self._assess_volume_confirmation(analysis)
        
        # 6. Session Timing (8%)
        session_data = analysis.get('session_analysis', {})
        metrics.session_timing = self._assess_session_timing(session_data, signal)
        
        # 7. Risk-Reward Ratio (7%)
        rr_ratio = signal.get('risk_reward_ratio', 0)
        metrics.risk_reward_ratio = min(rr_ratio / 3.0, 1.0)  # Normalize to max 3:1
        
        # 8. Market Structure Bias (6%)
        bias = confluence_data.get('bias', 'neutral')
        signal_direction = signal.get('direction', 'unknown')
        bias_alignment = self._assess_bias_alignment(bias, signal_direction)
        metrics.market_structure_bias = bias_alignment
        
        # 9. Liquidity Zone Proximity (5%)
        liquidity_data = timeframe_data.get('liquidity_zones', [])
        metrics.liquidity_zones = self._assess_liquidity_proximity(liquidity_data, signal)
        
        # 10. Order Block Quality (4%)
        ob_data = timeframe_data.get('order_blocks', [])
        metrics.order_block_quality = self._assess_order_block_quality(ob_data, signal)
        
        # 11. Fair Value Gap Quality (3%)
        fvg_data = timeframe_data.get('fair_value_gaps', [])
        metrics.fvg_quality = self._assess_fvg_quality(fvg_data, signal)
        
        # 12. Trend Alignment (2%)
        metrics.trend_alignment = self._assess_trend_alignment(analysis, signal)
        
        # Calculate overall score using weighted average
        metrics.overall_score = (
            metrics.confluence_score * self.weights['confluence'] +
            metrics.active_factors * self.weights['active_factors'] +
            metrics.timeframe_alignment * self.weights['timeframe_alignment'] +
            metrics.structure_strength * self.weights['structure_strength'] +
            metrics.volume_confirmation * self.weights['volume_confirmation'] +
            metrics.session_timing * self.weights['session_timing'] +
            metrics.risk_reward_ratio * self.weights['risk_reward'] +
            metrics.market_structure_bias * self.weights['market_bias'] +
            metrics.liquidity_zones * self.weights['liquidity_zones'] +
            metrics.order_block_quality * self.weights['order_blocks'] +
            metrics.fvg_quality * self.weights['fvg_quality'] +
            metrics.trend_alignment * self.weights['trend_alignment']
        )
        
        # Convert to confidence percentage
        metrics.confidence_percentage = metrics.overall_score * 100
        
        # Determine quality rating
        metrics.quality_rating = self._determine_quality_rating(metrics.confidence_percentage)
        
        return metrics
    
    def _assess_timeframe_alignment(self, timeframe_data: Dict, signal: Dict) -> float:
        """Assess multi-timeframe alignment"""
        timeframes = ['H4', 'H1', 'M15']
        alignment_score = 0
        valid_timeframes = 0
        
        signal_direction = signal.get('direction', '').upper()
        
        for tf in timeframes:
            tf_data = timeframe_data.get(tf, {})
            if not tf_data:
                continue
                
            valid_timeframes += 1
            
            # Check order block alignment
            ob_count = len(tf_data.get('order_blocks', []))
            if ob_count > 0:
                alignment_score += 0.3
            
            # Check FVG alignment  
            fvg_count = len(tf_data.get('fair_value_gaps', []))
            if fvg_count > 0:
                alignment_score += 0.2
        
        return alignment_score / max(valid_timeframes, 1) if valid_timeframes > 0 else 0
    
    def _assess_volume_confirmation(self, analysis: Dict) -> float:
        """Assess volume confirmation"""
        # Placeholder - would analyze volume patterns
        return 0.7  # Default moderate volume confirmation
    
    def _assess_session_timing(self, session_data: Dict, signal: Dict) -> float:
        """Assess trading session timing quality"""
        # Check if signal occurs during high-activity sessions
        session_score = session_data.get('activity_level', 0.5)
        overlap_bonus = session_data.get('overlap_bonus', 0)
        
        return min(session_score + overlap_bonus, 1.0)
    
    def _assess_bias_alignment(self, market_bias: str, signal_direction: str) -> float:
        """Assess alignment between market bias and signal direction"""
        if market_bias == 'neutral' or signal_direction == 'unknown':
            return 0.5
        
        # Check if signal aligns with market bias
        bullish_alignment = (market_bias == 'bullish' and signal_direction in ['BUY', 'LONG'])
        bearish_alignment = (market_bias == 'bearish' and signal_direction in ['SELL', 'SHORT'])
        
        return 1.0 if (bullish_alignment or bearish_alignment) else 0.2
    
    def _assess_liquidity_proximity(self, liquidity_data: List, signal: Dict) -> float:
        """Assess proximity to liquidity zones"""
        if not liquidity_data:
            return 0.3
        
        entry_price = signal.get('entry_price', 0)
        if entry_price == 0:
            return 0.3
        
        # Calculate distance to nearest liquidity zone
        min_distance = float('inf')
        for zone in liquidity_data:
            zone_price = zone.get('price', 0)
            if zone_price > 0:
                distance = abs(entry_price - zone_price) / entry_price
                min_distance = min(min_distance, distance)
        
        # Closer to liquidity = higher score
        if min_distance == float('inf'):
            return 0.3
        
        return max(0, 1.0 - min_distance * 100)  # Penalize if >1% away
    
    def _assess_order_block_quality(self, ob_data: List, signal: Dict) -> float:
        """Assess order block quality"""
        if not ob_data:
            return 0.2
        
        # Count relevant order blocks
        relevant_obs = len([ob for ob in ob_data if ob.get('active', False)])
        
        return min(relevant_obs / 3.0, 1.0)  # Normalize to max 3 OBs
    
    def _assess_fvg_quality(self, fvg_data: List, signal: Dict) -> float:
        """Assess Fair Value Gap quality"""
        if not fvg_data:
            return 0.2
        
        # Count active FVGs
        active_fvgs = len([fvg for fvg in fvg_data if fvg.get('active', False)])
        
        return min(active_fvgs / 2.0, 1.0)  # Normalize to max 2 FVGs
    
    def _assess_trend_alignment(self, analysis: Dict, signal: Dict) -> float:
        """Assess trend alignment"""
        # Check if signal aligns with higher timeframe trend
        confluence = analysis.get('confluence', {})
        bias = confluence.get('bias', 'neutral')
        strength = confluence.get('strength', 'weak')
        
        if bias == 'neutral':
            return 0.5
        
        signal_direction = signal.get('direction', '').upper()
        
        # Assess alignment
        trend_aligned = (
            (bias == 'bullish' and signal_direction in ['BUY', 'LONG']) or
            (bias == 'bearish' and signal_direction in ['SELL', 'SHORT'])
        )
        
        base_score = 1.0 if trend_aligned else 0.2
        
        # Bonus for strong trends
        strength_bonus = {
            'very_strong': 0.2,
            'strong': 0.1,
            'moderate': 0.05,
            'weak': 0,
            'very_weak': -0.1
        }.get(strength, 0)
        
        return min(base_score + strength_bonus, 1.0)
    
    def _determine_quality_rating(self, confidence: float) -> QualityRating:
        """Determine quality rating from confidence percentage"""
        if confidence >= 90:
            return QualityRating.EXCELLENT
        elif confidence >= 75:
            return QualityRating.HIGH
        elif confidence >= 60:
            return QualityRating.GOOD
        elif confidence >= 45:
            return QualityRating.MODERATE
        elif confidence >= 30:
            return QualityRating.LOW
        else:
            return QualityRating.POOR
    
    def get_quality_summary(self, metrics: QualityMetrics) -> Dict:
        """Get detailed quality assessment summary"""
        return {
            'overall_score': round(metrics.overall_score, 3),
            'confidence_percentage': round(metrics.confidence_percentage, 1),
            'quality_rating': metrics.quality_rating.value,
            'component_scores': {
                'confluence': round(metrics.confluence_score, 2),
                'active_factors': round(metrics.active_factors, 2),
                'timeframe_alignment': round(metrics.timeframe_alignment, 2),
                'structure_strength': round(metrics.structure_strength, 2),
                'volume_confirmation': round(metrics.volume_confirmation, 2),
                'session_timing': round(metrics.session_timing, 2),
                'risk_reward': round(metrics.risk_reward_ratio, 2),
                'market_bias': round(metrics.market_structure_bias, 2),
                'liquidity_zones': round(metrics.liquidity_zones, 2),
                'order_blocks': round(metrics.order_block_quality, 2),
                'fvg_quality': round(metrics.fvg_quality, 2),
                'trend_alignment': round(metrics.trend_alignment, 2)
            },
            'strengths': self._identify_strengths(metrics),
            'weaknesses': self._identify_weaknesses(metrics)
        }
    
    def _identify_strengths(self, metrics: QualityMetrics) -> List[str]:
        """Identify quality strengths"""
        strengths = []
        
        if metrics.confluence_score > 0.8:
            strengths.append("Strong SMC confluence")
        if metrics.active_factors > 0.8:
            strengths.append("Multiple confirming factors")
        if metrics.structure_strength > 0.8:
            strengths.append("Strong market structure")
        if metrics.risk_reward_ratio > 0.8:
            strengths.append("Excellent risk-reward ratio")
        if metrics.market_structure_bias > 0.8:
            strengths.append("Perfect bias alignment")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: QualityMetrics) -> List[str]:
        """Identify quality weaknesses"""
        weaknesses = []
        
        if metrics.confluence_score < 0.4:
            weaknesses.append("Low confluence score")
        if metrics.active_factors < 0.4:
            weaknesses.append("Few confirming factors")
        if metrics.volume_confirmation < 0.4:
            weaknesses.append("Weak volume confirmation")
        if metrics.session_timing < 0.4:
            weaknesses.append("Poor session timing")
        if metrics.timeframe_alignment < 0.4:
            weaknesses.append("Weak multi-timeframe alignment")
        
        return weaknesses