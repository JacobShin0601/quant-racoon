#!/usr/bin/env python3
"""
동적 Regime Switching 감지 시스템

상관관계 변화와 구조적 변화를 실시간으로 감지하여
시장 regime의 전환점을 포착하는 시스템
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import logging


class DynamicRegimeSwitchingDetector:
    """
    동적 regime switching 감지 시스템
    상관관계 변화와 구조적 변화를 실시간으로 감지
    """
    
    def __init__(self, window_size: int = 60, shift_threshold: float = 0.3):
        self.window_size = window_size
        self.shift_threshold = shift_threshold
        self.correlation_history = []
        self.regime_change_points = []
        
        self.logger = logging.getLogger(__name__)
    
    def detect_regime_shifts(self, spy_data: pd.DataFrame, 
                           macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Regime shift 감지"""
        try:
            if len(spy_data) < self.window_size * 2:
                return {'regime_shift_detected': False, 'confidence': 0.0}
            
            # SPY 수익률 계산
            spy_returns = spy_data['close'].pct_change().dropna()
            
            # 현재 window와 이전 window의 상관관계 매트릭스 계산
            current_window = spy_returns.tail(self.window_size)
            previous_window = spy_returns.tail(self.window_size * 2).head(self.window_size)
            
            # VIX와의 상관관계 변화 확인
            correlation_changes = {}
            
            if '^VIX' in macro_data and not macro_data['^VIX'].empty:
                vix_data = macro_data['^VIX']
                close_col = 'close' if 'close' in vix_data.columns else 'Close'
                vix_returns = vix_data[close_col].pct_change().dropna()
                
                # 대응되는 기간의 VIX 데이터 추출
                vix_current = vix_returns.tail(self.window_size)
                vix_previous = vix_returns.tail(self.window_size * 2).head(self.window_size)
                
                if len(vix_current) >= 30 and len(vix_previous) >= 30:
                    # 상관관계 계산
                    corr_current = spy_returns.tail(len(vix_current)).corr(vix_current)
                    corr_previous = spy_returns.tail(self.window_size * 2).head(len(vix_previous)).corr(vix_previous)
                    
                    correlation_changes['VIX'] = abs(corr_current - corr_previous)
            
            # 변동성 regime 변화 감지
            volatility_current = current_window.std()
            volatility_previous = previous_window.std()
            volatility_change = abs(volatility_current - volatility_previous) / volatility_previous
            
            # Trend strength 변화 감지
            trend_current = np.mean(current_window > 0)
            trend_previous = np.mean(previous_window > 0)
            trend_change = abs(trend_current - trend_previous)
            
            # 종합 regime shift 점수 계산
            shift_score = 0.0
            
            # 상관관계 변화 점수
            if correlation_changes:
                correlation_score = np.mean(list(correlation_changes.values()))
                shift_score += correlation_score * 0.4
            
            # 변동성 변화 점수
            volatility_score = min(volatility_change, 1.0)
            shift_score += volatility_score * 0.3
            
            # 트렌드 변화 점수
            trend_score = trend_change
            shift_score += trend_score * 0.3
            
            # Regime shift 감지
            regime_shift_detected = shift_score > self.shift_threshold
            
            if regime_shift_detected:
                self.regime_change_points.append({
                    'timestamp': datetime.now(),
                    'shift_score': shift_score,
                    'correlation_changes': correlation_changes,
                    'volatility_change': volatility_change,
                    'trend_change': trend_change
                })
            
            return {
                'regime_shift_detected': regime_shift_detected,
                'shift_score': shift_score,
                'confidence': min(shift_score / self.shift_threshold, 1.0),
                'components': {
                    'correlation_changes': correlation_changes,
                    'volatility_change': volatility_change,
                    'trend_change': trend_change
                },
                'change_points_count': len(self.regime_change_points)
            }
            
        except Exception as e:
            self.logger.warning(f"Regime shift 감지 중 오류: {e}")
            return {
                'regime_shift_detected': False,
                'shift_score': 0.0,
                'confidence': 0.0,
                'components': {},
                'change_points_count': 0
            }
    
    def analyze_regime_stability(self, spy_data: pd.DataFrame, lookback_periods: int = 5) -> Dict[str, Any]:
        """최근 regime 안정성 분석 (고도화)"""
        try:
            if len(spy_data) < self.window_size * lookback_periods:
                return {'stability_score': 0.5, 'trend_consistency': 0.5}
            
            spy_returns = spy_data['close'].pct_change().dropna()
            
            # 여러 기간에 걸친 안정성 평가
            stability_scores = []
            trend_scores = []
            volatility_scores = []
            momentum_scores = []
            correlation_scores = []
            
            for i in range(lookback_periods):
                start_idx = -(self.window_size * (i + 1))
                end_idx = -(self.window_size * i) if i > 0 else None
                
                period_returns = spy_returns.iloc[start_idx:end_idx]
                
                # 1. 변동성 안정성
                vol_stability = 1.0 - (period_returns.std() / period_returns.mean()) if period_returns.mean() != 0 else 0.5
                stability_scores.append(max(0.0, min(1.0, vol_stability)))
                
                # 2. 트렌드 일관성
                positive_ratio = np.mean(period_returns > 0)
                trend_consistency = abs(positive_ratio - 0.5) * 2  # 0.5에서 멀수록 일관된 트렌드
                trend_scores.append(trend_consistency)
                
                # 3. 변동성 일관성 (새로 추가)
                vol_consistency = 1.0 - period_returns.rolling(5).std().std() / period_returns.std() if period_returns.std() > 0 else 0.5
                volatility_scores.append(max(0.0, min(1.0, vol_consistency)))
                
                # 4. 모멘텀 안정성 (새로 추가)
                momentum = period_returns.rolling(5).mean()
                momentum_stability = 1.0 - momentum.std() / abs(momentum.mean()) if abs(momentum.mean()) > 0 else 0.5
                momentum_scores.append(max(0.0, min(1.0, momentum_stability)))
                
                # 5. 상관관계 안정성 (새로 추가)
                if len(period_returns) > 10:
                    # 5일 이동상관관계의 변동성
                    rolling_corr = period_returns.rolling(5).corr(period_returns.shift(1))
                    corr_stability = 1.0 - rolling_corr.std() if not rolling_corr.isna().all() else 0.5
                    correlation_scores.append(max(0.0, min(1.0, corr_stability)))
                else:
                    correlation_scores.append(0.5)
            
            # 종합 안정성 점수 계산 (가중 평균)
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # 변동성 > 트렌드 > 변동성일관성 > 모멘텀 > 상관관계
            stability_score = np.average([
                np.mean(stability_scores),
                np.mean(trend_scores),
                np.mean(volatility_scores),
                np.mean(momentum_scores),
                np.mean(correlation_scores)
            ], weights=weights)
            
            return {
                'stability_score': stability_score,
                'trend_consistency': np.mean(trend_scores),
                'volatility_consistency': np.mean(volatility_scores),
                'momentum_stability': np.mean(momentum_scores),
                'correlation_stability': np.mean(correlation_scores),
                'volatility_trend': 'increasing' if stability_scores[-1] < stability_scores[0] else 'decreasing',
                'overall_trend': 'improving' if stability_score > 0.6 else 'deteriorating' if stability_score < 0.4 else 'stable',
                'period_analysis': {
                    'stability_scores': stability_scores,
                    'trend_scores': trend_scores,
                    'volatility_scores': volatility_scores,
                    'momentum_scores': momentum_scores,
                    'correlation_scores': correlation_scores
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Regime 안정성 분석 중 오류: {e}")
            return {'stability_score': 0.5, 'trend_consistency': 0.5}
    
    def get_regime_change_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 regime 변화 이력 반환"""
        try:
            # 최근 변화점들을 시간순으로 정렬
            sorted_changes = sorted(self.regime_change_points, 
                                  key=lambda x: x['timestamp'], 
                                  reverse=True)
            
            return sorted_changes[:limit]
            
        except Exception as e:
            self.logger.warning(f"Regime 변화 이력 조회 중 오류: {e}")
            return []
    
    def calculate_regime_persistence(self, spy_data: pd.DataFrame) -> Dict[str, Any]:
        """현재 regime의 지속성 분석"""
        try:
            if len(spy_data) < self.window_size:
                return {'persistence_score': 0.5, 'expected_duration': 'unknown'}
            
            spy_returns = spy_data['close'].pct_change().dropna()
            recent_returns = spy_returns.tail(self.window_size)
            
            # 현재 regime 특성 분석
            current_trend = np.mean(recent_returns)
            current_volatility = recent_returns.std()
            
            # 유사한 과거 regime들과 비교
            historical_durations = []
            
            # 단순화된 지속성 계산 (실제로는 더 복잡한 알고리즘 사용)
            trend_strength = abs(current_trend) / current_volatility if current_volatility > 0 else 0
            
            if trend_strength > 0.5:
                persistence_score = min(0.9, 0.5 + trend_strength * 0.4)
                expected_duration = 'long'
            elif trend_strength > 0.2:
                persistence_score = 0.5 + trend_strength * 0.3
                expected_duration = 'medium'
            else:
                persistence_score = max(0.1, 0.5 - (0.2 - trend_strength) * 2)
                expected_duration = 'short'
            
            return {
                'persistence_score': persistence_score,
                'expected_duration': expected_duration,
                'trend_strength': trend_strength,
                'volatility_level': 'high' if current_volatility > recent_returns.std() * 1.5 else 'normal'
            }
            
        except Exception as e:
            self.logger.warning(f"Regime 지속성 분석 중 오류: {e}")
            return {'persistence_score': 0.5, 'expected_duration': 'unknown'} 