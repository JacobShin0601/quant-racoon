#!/usr/bin/env python3
"""
RLMF (Reinforcement Learning from Market Feedback) 기반 동적 적응 시스템

Reference: arXiv:2406.15508 - "What Teaches Robots to Walk, Teaches Them to Trade too"
- Market feedback을 활용한 실시간 학습 시스템
- Statistical Arbitrage 요소 통합 (Keybot the Quant 방식)
- 동적 가중치 조정 및 적응 메커니즘
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging


class RLMFRegimeAdaptation:
    """
    RLMF (Reinforcement Learning from Market Feedback) 기반 
    동적 regime 적응 시스템
    
    Reference: "What Teaches Robots to Walk, Teaches Them to Trade too"
    - arXiv:2406.15508 논문의 방법론 적용
    """
    
    def __init__(self, learning_rate: float = 0.01, decay_factor: float = 0.95, 
                 feedback_window: int = 20, min_confidence_threshold: float = 0.3):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.feedback_window = feedback_window
        self.min_confidence_threshold = min_confidence_threshold
        
        # Market feedback 저장소
        self.feedback_history = []
        self.regime_performance = {}
        self.adaptation_weights = {
            'trend_strength': 1.0,
            'volatility_regime': 1.0, 
            'momentum_persistence': 1.0,
            'macro_alignment': 1.0,
            'correlation_shift': 1.0
        }
        
        # Statistical Arbitrage Key Metrics (Keybot the Quant 방식)
        self.key_metrics = {
            'XRT': {'weight': 0.32, 'threshold': 0.032},  # Retail stocks
            'XLF': {'weight': 0.27, 'threshold': 0.027},  # Financial sector
            'GTX': {'weight': 0.25, 'threshold': 0.025},  # Commodities
            'VIX': {'weight': 0.16, 'threshold': 0.015}   # Volatility
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_market_feedback(self, prediction: str, actual_returns: pd.Series, 
                                spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Market feedback 계산 - 예측 정확도와 수익성을 종합 평가
        """
        try:
            feedback = {
                'prediction_accuracy': 0.0,
                'return_alignment': 0.0,
                'volatility_prediction': 0.0,
                'regime_persistence': 0.0,
                'macro_consistency': 0.0
            }
            
            if len(actual_returns) < 5:
                return feedback
            
            # 1. 예측 정확도 (Direction Accuracy)
            predicted_direction = 1 if prediction in ['TRENDING_UP', 'BULLISH'] else -1
            if prediction in ['SIDEWAYS', 'UNCERTAIN']:
                predicted_direction = 0
            
            actual_direction_series = np.sign(actual_returns)
            direction_accuracy = np.mean(actual_direction_series == predicted_direction)
            feedback['prediction_accuracy'] = max(0.0, direction_accuracy)
            
            # 2. 수익률 정렬도 (Return Alignment)
            if prediction == 'TRENDING_UP':
                expected_return = 0.05  # 5% 상승 기대
            elif prediction == 'TRENDING_DOWN': 
                expected_return = -0.03  # 3% 하락 기대
            else:
                expected_return = 0.0  # 중립
            
            actual_return = actual_returns.mean()
            return_alignment = 1.0 - abs(expected_return - actual_return) / 0.1
            feedback['return_alignment'] = max(0.0, min(1.0, return_alignment))
            
            # 3. 변동성 예측 정확도
            predicted_volatility = 0.15 if prediction == 'VOLATILE' else 0.08
            actual_volatility = actual_returns.std()
            vol_accuracy = 1.0 - abs(predicted_volatility - actual_volatility) / 0.1
            feedback['volatility_prediction'] = max(0.0, min(1.0, vol_accuracy))
            
            # 4. Regime 지속성 (Persistence)
            regime_changes = self._count_regime_changes(actual_returns)
            if prediction in ['TRENDING_UP', 'TRENDING_DOWN']:
                expected_changes = 2  # 트렌드는 변화가 적어야 함
            else:
                expected_changes = 5  # 횡보/변동성은 변화가 많을 수 있음
            
            persistence_score = 1.0 - abs(regime_changes - expected_changes) / max(regime_changes, expected_changes)
            feedback['regime_persistence'] = max(0.0, persistence_score)
            
            # 5. 매크로 일관성 (Macro Consistency)
            macro_score = self._calculate_macro_consistency(prediction, macro_data)
            feedback['macro_consistency'] = macro_score
            
            return feedback
            
        except Exception as e:
            self.logger.warning(f"Market feedback 계산 중 오류: {e}")
            return {'prediction_accuracy': 0.5, 'return_alignment': 0.5, 
                    'volatility_prediction': 0.5, 'regime_persistence': 0.5,
                    'macro_consistency': 0.5}
    
    def _count_regime_changes(self, returns: pd.Series) -> int:
        """수익률 시리즈에서 regime 변화 횟수 계산"""
        if len(returns) < 3:
            return 0
        
        # 3일 이동평균으로 regime 변화 감지
        smoothed_returns = returns.rolling(3).mean()
        regime_indicators = np.sign(smoothed_returns)
        changes = np.sum(np.diff(regime_indicators) != 0)
        return int(changes)
    
    def _calculate_macro_consistency(self, prediction: str, macro_data: Dict[str, pd.DataFrame]) -> float:
        """매크로 데이터와 예측의 일관성 계산"""
        try:
            consistency_score = 0.5  # 기본값
            
            # VIX 일관성 확인
            if '^VIX' in macro_data and not macro_data['^VIX'].empty:
                vix_data = macro_data['^VIX']
                close_col = 'close' if 'close' in vix_data.columns else 'Close'
                current_vix = vix_data[close_col].iloc[-1]
                
                if prediction == 'VOLATILE' and current_vix > 25:
                    consistency_score += 0.2
                elif prediction in ['TRENDING_UP', 'TRENDING_DOWN'] and current_vix < 20:
                    consistency_score += 0.1
            
            # 금리 환경 일관성 (10년 국채)
            if '^TNX' in macro_data and not macro_data['^TNX'].empty:
                tnx_data = macro_data['^TNX']
                close_col = 'close' if 'close' in tnx_data.columns else 'Close'
                if len(tnx_data) > 5:
                    rate_trend = tnx_data[close_col].pct_change(5).iloc[-1]
                    
                    if prediction == 'TRENDING_DOWN' and rate_trend > 0.05:  # 금리 상승 시 주식 하락
                        consistency_score += 0.15
                    elif prediction == 'TRENDING_UP' and rate_trend < -0.02:  # 금리 하락 시 주식 상승
                        consistency_score += 0.15
            
            return min(1.0, consistency_score)
            
        except Exception as e:
            self.logger.warning(f"매크로 일관성 계산 중 오류: {e}")
            return 0.5
    
    def update_adaptation_weights(self, feedback: Dict[str, float]):
        """Market feedback을 기반으로 적응 가중치 업데이트"""
        try:
            # Feedback history에 추가
            self.feedback_history.append({
                'timestamp': datetime.now(),
                'feedback': feedback.copy()
            })
            
            # 최근 feedback_window 개만 유지
            if len(self.feedback_history) > self.feedback_window:
                self.feedback_history = self.feedback_history[-self.feedback_window:]
            
            # 가중치 업데이트 (Gradient-based adaptation)
            if len(self.feedback_history) >= 3:
                recent_performance = np.mean([f['feedback']['prediction_accuracy'] 
                                            for f in self.feedback_history[-3:]])
                
                # 성과에 따른 가중치 조정
                if recent_performance > 0.6:  # 좋은 성과
                    adjustment_factor = 1.0 + self.learning_rate
                else:  # 부족한 성과
                    adjustment_factor = 1.0 - self.learning_rate
                
                # 각 요소별 가중치 조정
                for key in self.adaptation_weights:
                    if key in feedback:
                        current_weight = self.adaptation_weights[key]
                        feedback_score = feedback.get(key, 0.5)
                        
                        # 성과가 좋은 요소는 가중치 증가, 나쁜 요소는 감소
                        if feedback_score > 0.6:
                            new_weight = current_weight * adjustment_factor
                        else:
                            new_weight = current_weight * (2.0 - adjustment_factor)
                        
                        self.adaptation_weights[key] = max(0.1, min(2.0, new_weight))
                
                # Decay factor 적용 (과적합 방지)
                for key in self.adaptation_weights:
                    self.adaptation_weights[key] *= self.decay_factor
                    self.adaptation_weights[key] = max(0.1, self.adaptation_weights[key])
            
        except Exception as e:
            self.logger.warning(f"적응 가중치 업데이트 중 오류: {e}")
    
    def calculate_statistical_arbitrage_signal(self, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Statistical Arbitrage 신호 계산 (Keybot the Quant 방식)
        
        Key Metrics: XRT, XLF, GTX, VIX
        """
        try:
            signals = {}
            overall_signal = 0.0
            confidence = 0.0
            
            for metric, config in self.key_metrics.items():
                weight = config['weight']
                threshold = config['threshold']
                
                symbol_map = {
                    'XRT': 'XRT',    # SPDR S&P Retail ETF
                    'XLF': 'XLF',    # Financial Select Sector SPDR
                    'GTX': 'DJP',    # iPath Bloomberg Commodity Index (GTX 대체)
                    'VIX': '^VIX'    # VIX Volatility Index
                }
                
                symbol = symbol_map.get(metric, metric)
                
                if symbol in macro_data and not macro_data[symbol].empty:
                    data = macro_data[symbol]
                    close_col = 'close' if 'close' in data.columns else 'Close'
                    
                    if len(data) >= 5:
                        # 5일 수익률 계산
                        returns = data[close_col].pct_change(5).iloc[-1]
                        
                        # 신호 계산 (Keybot 방식: threshold 기반)
                        if returns > threshold:
                            signal = 1.0  # 강세
                        elif returns < -threshold:
                            signal = -1.0  # 약세
                        else:
                            signal = 0.0  # 중립
                        
                        signals[metric] = {
                            'signal': signal,
                            'return': returns,
                            'weight': weight
                        }
                        
                        # 전체 신호에 가중치 적용
                        overall_signal += signal * weight
                        confidence += weight if abs(signal) > 0.5 else 0
            
            # 신호 정규화
            total_weight = sum(config['weight'] for config in self.key_metrics.values())
            if total_weight > 0:
                overall_signal /= total_weight
                confidence /= total_weight
            
            return {
                'overall_signal': overall_signal,
                'confidence': confidence,
                'individual_signals': signals,
                'signal_strength': abs(overall_signal),
                'direction': 'BULLISH' if overall_signal > 0.1 else 'BEARISH' if overall_signal < -0.1 else 'NEUTRAL'
            }
            
        except Exception as e:
            self.logger.warning(f"Statistical arbitrage 신호 계산 중 오류: {e}")
            return {
                'overall_signal': 0.0,
                'confidence': 0.0,
                'individual_signals': {},
                'signal_strength': 0.0,
                'direction': 'NEUTRAL'
            }
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """현재 적응 상태 반환"""
        if not self.feedback_history:
            return {
                'status': 'initializing',
                'weights': self.adaptation_weights.copy(),
                'performance': 0.5,
                'feedback_count': 0
            }
        
        recent_performance = np.mean([f['feedback']['prediction_accuracy'] 
                                    for f in self.feedback_history[-5:]])
        
        return {
            'status': 'adapting',
            'weights': self.adaptation_weights.copy(),
            'performance': recent_performance,
            'feedback_count': len(self.feedback_history),
            'last_update': self.feedback_history[-1]['timestamp'].isoformat()
        } 