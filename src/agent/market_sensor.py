#!/usr/bin/env python3
"""
시장 환경 분류기 (Market Sensor)
통합 시장 분석 시스템 - 실행 인터페이스
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import os
import optuna
from dataclasses import dataclass
from enum import Enum
import uuid
import joblib
from pathlib import Path
import warnings

from ..actions.global_macro import (
    GlobalMacroDataCollector, 
    HyperparamTuner, 
    MacroSectorAnalyzer,
    MarketRegime, 
    MarketCondition, 
    SectorStrength,
    MarketClassification,
    MacroAnalysis,
    MarketRegimeValidator
)
from ..actions.random_forest import MarketRegimeRF
from .enhancements import (
    RLMFRegimeAdaptation,
    MultiLayerConfidenceSystem,
    DynamicRegimeSwitchingDetector,
    LLMPrivilegedInformationSystem
)


# 고도화 클래스들은 enhancements 패키지로 이동되었습니다.
# 이제 해당 클래스들을 import해서 사용합니다.


class MarketSensor:
    """통합 시장 분석 시스템 - 실행 인터페이스 (고도화된 RLMF 기반)"""
    
    def __init__(self, data_dir: str = "data/macro", config_path: str = "config/config_macro.json"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # 세션 UUID 생성
        self.session_uuid = str(uuid.uuid4())
        self.logger.info(f"MarketSensor 초기화 - Session UUID: {self.session_uuid}")
        
        # 핵심 컴포넌트들 초기화 (UUID 전달)
        self.macro_collector = GlobalMacroDataCollector(self.session_uuid)
        self.hyperparam_tuner = HyperparamTuner(config_path, self.session_uuid)
        self.macro_analyzer = MacroSectorAnalyzer(data_dir, self.session_uuid)
        
        # 고도화된 시스템 컴포넌트들 초기화
        self.rlmf_adaptation = RLMFRegimeAdaptation()
        self.confidence_system = MultiLayerConfidenceSystem()
        self.regime_detector = DynamicRegimeSwitchingDetector()
        self.llm_privileged_system = LLMPrivilegedInformationSystem()
        
        # Random Forest 모델 초기화 (저장된 모델 우선 로드)
        self.rf_model = MarketRegimeRF()
        
        # 최적화 파라미터 저장 변수
        self.optimal_params = None
        
        # 경고 무시 설정
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
        
        # 초기화 완료 로그
        self.logger.info(f"MarketSensor 초기화 완료 - 세션: {self.session_uuid}")
    
    def load_macro_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """종합 신뢰도 계산"""
        try:
            # 각 요소별 가중 신뢰도
            weighted_confidences = {
                'technical': technical_conf * self.confidence_weights['technical'],
                'macro': macro_conf * self.confidence_weights['macro'],
                'statistical_arb': stat_arb_conf * self.confidence_weights['statistical_arb'],
                'rlmf_feedback': rlmf_conf * self.confidence_weights['rlmf_feedback'],
                'cross_validation': cross_val_conf * self.confidence_weights['cross_validation']
            }
            
            # 종합 신뢰도
            total_confidence = sum(weighted_confidences.values())
            
            # 신뢰도 분포 분석
            confidence_std = np.std(list(weighted_confidences.values()))
            confidence_consistency = 1.0 - min(confidence_std / 0.2, 1.0)  # 일관성 점수
            
            # 최종 조정된 신뢰도
            adjusted_confidence = total_confidence * (0.5 + 0.5 * confidence_consistency)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            return {
                'total_confidence': total_confidence,
                'adjusted_confidence': adjusted_confidence,
                'component_confidences': weighted_confidences,
                'consistency_score': confidence_consistency,
                'confidence_distribution': {
                    'mean': np.mean(list(weighted_confidences.values())),
                    'std': confidence_std,
                    'min': min(weighted_confidences.values()),
                    'max': max(weighted_confidences.values())
                }
            }
            
        except Exception as e:
            self.logger.warning(f"종합 신뢰도 계산 중 오류: {e}")
            return {
                'total_confidence': 0.5,
                'adjusted_confidence': 0.5,
                'component_confidences': {},
                'consistency_score': 0.5,
                'confidence_distribution': {}
            }


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


class LLMPrivilegedInformationSystem:
    """
    LLM Privileged Information 활용 시스템
    
    Reference: arXiv:2406.15508
    - LLM의 world knowledge를 활용한 시장 분석 강화
    - Market regime과 경제 환경에 대한 상황적 이해 제공
    """
    
    def __init__(self):
        self.market_knowledge_base = {
            # 경제 지표별 시장 영향 패턴 (LLM 사전 지식 기반)
            'inflation_patterns': {
                'high_inflation': {
                    'typical_regimes': ['VOLATILE', 'TRENDING_DOWN'],
                    'sector_rotation': ['energy', 'commodities', 'real_estate'],
                    'risk_factors': ['interest_rate_hikes', 'wage_pressure', 'supply_constraints'],
                    'confidence_modifier': 0.8  # 불확실성 증가
                },
                'low_inflation': {
                    'typical_regimes': ['TRENDING_UP', 'SIDEWAYS'],
                    'sector_rotation': ['technology', 'growth_stocks'],
                    'risk_factors': ['deflationary_spiral', 'economic_stagnation'],
                    'confidence_modifier': 1.1  # 안정성 증가
                }
            },
            'rate_environment_patterns': {
                'rising_rates': {
                    'typical_regimes': ['VOLATILE', 'TRENDING_DOWN'],
                    'sector_rotation': ['financials', 'value_stocks'],
                    'vulnerable_sectors': ['real_estate', 'utilities', 'high_dividend'],
                    'confidence_modifier': 0.9
                },
                'falling_rates': {
                    'typical_regimes': ['TRENDING_UP'],
                    'sector_rotation': ['technology', 'growth_stocks'],
                    'vulnerable_sectors': ['financials'],
                    'confidence_modifier': 1.0
                }
            },
            'geopolitical_patterns': {
                'high_uncertainty': {
                    'typical_regimes': ['VOLATILE', 'UNCERTAIN'],
                    'safe_havens': ['treasuries', 'gold', 'dollar'],
                    'risk_assets': ['emerging_markets', 'high_beta'],
                    'confidence_modifier': 0.7
                }
            },
            'seasonal_patterns': {
                'year_end_rally': {
                    'months': [11, 12],
                    'typical_regimes': ['TRENDING_UP'],
                    'confidence_modifier': 1.05
                },
                'may_sell_go_away': {
                    'months': [5, 6, 7, 8, 9],
                    'typical_regimes': ['SIDEWAYS', 'VOLATILE'],
                    'confidence_modifier': 0.95
                }
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get_privileged_insights(self, current_regime: str, macro_data: Dict[str, pd.DataFrame], 
                              market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 사전 지식을 활용한 특권적 정보 제공
        """
        try:
            insights = {
                'regime_validation': {},
                'contextual_factors': {},
                'risk_adjustments': {},
                'confidence_modifiers': [],
                'strategic_recommendations': []
            }
            
            # 1. 현재 경제 환경 분석
            economic_context = self._analyze_economic_context(macro_data)
            
            # 2. Regime 검증 (LLM 지식 기반)
            regime_validation = self._validate_regime_with_knowledge(current_regime, economic_context)
            insights['regime_validation'] = regime_validation
            
            # 3. 계절성 패턴 분석
            seasonal_context = self._analyze_seasonal_patterns()
            insights['contextual_factors']['seasonal'] = seasonal_context
            
            # 4. 구조적 위험 요인 식별
            structural_risks = self._identify_structural_risks(economic_context, market_metrics)
            insights['risk_adjustments'] = structural_risks
            
            # 5. 신뢰도 수정자 계산
            confidence_modifiers = self._calculate_confidence_modifiers(
                economic_context, seasonal_context, structural_risks
            )
            insights['confidence_modifiers'] = confidence_modifiers
            
            # 6. 전략적 추천사항
            strategic_recommendations = self._generate_strategic_recommendations(
                current_regime, economic_context, insights
            )
            insights['strategic_recommendations'] = strategic_recommendations
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"LLM 특권 정보 분석 중 오류: {e}")
            return {
                'regime_validation': {'consistency': 0.5},
                'contextual_factors': {},
                'risk_adjustments': {},
                'confidence_modifiers': [1.0],
                'strategic_recommendations': []
            }
    
    def _analyze_economic_context(self, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """현재 경제 환경 컨텍스트 분석"""
        context = {
            'inflation_regime': 'unknown',
            'rate_environment': 'unknown',
            'risk_sentiment': 'unknown'
        }
        
        try:
            # VIX 기반 위험 심리 분석
            if '^VIX' in macro_data and not macro_data['^VIX'].empty:
                vix_data = macro_data['^VIX']
                close_col = 'close' if 'close' in vix_data.columns else 'Close'
                current_vix = vix_data[close_col].iloc[-1]
                
                if current_vix > 30:
                    context['risk_sentiment'] = 'high_fear'
                elif current_vix > 20:
                    context['risk_sentiment'] = 'moderate_concern'
                else:
                    context['risk_sentiment'] = 'complacent'
            
            # 10년 국채 수익률 기반 금리 환경 분석
            if '^TNX' in macro_data and not macro_data['^TNX'].empty:
                tnx_data = macro_data['^TNX']
                close_col = 'close' if 'close' in tnx_data.columns else 'Close'
                if len(tnx_data) >= 60:
                    current_rate = tnx_data[close_col].iloc[-1]
                    rate_trend = tnx_data[close_col].pct_change(20).iloc[-1]  # 20일 변화율
                    
                    if rate_trend > 0.1:  # 10% 이상 상승
                        context['rate_environment'] = 'rising_rates'
                    elif rate_trend < -0.1:  # 10% 이상 하락
                        context['rate_environment'] = 'falling_rates'
                    else:
                        context['rate_environment'] = 'stable_rates'
            
            # TIPS 기반 인플레이션 기대 분석
            if '^TIP' in macro_data and not macro_data['^TIP'].empty:
                tip_data = macro_data['^TIP']
                close_col = 'close' if 'close' in tip_data.columns else 'Close'
                if len(tip_data) >= 60:
                    tip_trend = tip_data[close_col].pct_change(20).iloc[-1]
                    
                    if tip_trend > 0.05:  # TIPS 상승 → 인플레이션 기대 상승
                        context['inflation_regime'] = 'high_inflation'
                    elif tip_trend < -0.05:
                        context['inflation_regime'] = 'low_inflation'
                    else:
                        context['inflation_regime'] = 'stable_inflation'
            
            return context
            
        except Exception as e:
            self.logger.warning(f"경제 컨텍스트 분석 중 오류: {e}")
            return context
    
    def _validate_regime_with_knowledge(self, current_regime: str, economic_context: Dict[str, str]) -> Dict[str, Any]:
        """LLM 지식 기반 regime 검증"""
        try:
            validation = {
                'consistency': 0.5,
                'supporting_factors': [],
                'conflicting_factors': [],
                'alternative_regimes': []
            }
            
            # 인플레이션 환경과 regime 일관성 검사
            inflation_regime = economic_context.get('inflation_regime', 'unknown')
            if inflation_regime in self.market_knowledge_base['inflation_patterns']:
                pattern = self.market_knowledge_base['inflation_patterns'][inflation_regime]
                typical_regimes = pattern['typical_regimes']
                
                if current_regime in typical_regimes:
                    validation['consistency'] += 0.2
                    validation['supporting_factors'].append(f"인플레이션 환경({inflation_regime})과 일치")
                else:
                    validation['conflicting_factors'].append(f"인플레이션 환경({inflation_regime})과 불일치")
                    validation['alternative_regimes'].extend(typical_regimes)
            
            # 금리 환경과 regime 일관성 검사
            rate_environment = economic_context.get('rate_environment', 'unknown')
            if rate_environment in self.market_knowledge_base['rate_environment_patterns']:
                pattern = self.market_knowledge_base['rate_environment_patterns'][rate_environment]
                typical_regimes = pattern['typical_regimes']
                
                if current_regime in typical_regimes:
                    validation['consistency'] += 0.2
                    validation['supporting_factors'].append(f"금리 환경({rate_environment})과 일치")
                else:
                    validation['conflicting_factors'].append(f"금리 환경({rate_environment})과 불일치")
                    validation['alternative_regimes'].extend(typical_regimes)
            
            # 위험 심리와 regime 일관성 검사
            risk_sentiment = economic_context.get('risk_sentiment', 'unknown')
            if risk_sentiment == 'high_fear' and current_regime not in ['VOLATILE', 'TRENDING_DOWN']:
                validation['conflicting_factors'].append("높은 위험 회피 심리와 불일치")
                validation['alternative_regimes'].extend(['VOLATILE', 'TRENDING_DOWN'])
            elif risk_sentiment == 'complacent' and current_regime in ['VOLATILE']:
                validation['conflicting_factors'].append("낮은 위험 인식과 불일치")
            
            validation['consistency'] = max(0.0, min(1.0, validation['consistency']))
            
            return validation
            
        except Exception as e:
            self.logger.warning(f"Regime 검증 중 오류: {e}")
            return {'consistency': 0.5, 'supporting_factors': [], 'conflicting_factors': [], 'alternative_regimes': []}
    
    def _analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """계절성 패턴 분석"""
        try:
            current_month = datetime.now().month
            seasonal_insights = {
                'current_season': 'neutral',
                'seasonal_bias': 'none',
                'confidence_modifier': 1.0
            }
            
            # Year-end rally 패턴
            if current_month in self.market_knowledge_base['seasonal_patterns']['year_end_rally']['months']:
                seasonal_insights['current_season'] = 'year_end_rally'
                seasonal_insights['seasonal_bias'] = 'bullish'
                seasonal_insights['confidence_modifier'] = 1.05
            
            # "Sell in May and go away" 패턴
            elif current_month in self.market_knowledge_base['seasonal_patterns']['may_sell_go_away']['months']:
                seasonal_insights['current_season'] = 'summer_weakness'
                seasonal_insights['seasonal_bias'] = 'cautious'
                seasonal_insights['confidence_modifier'] = 0.95
            
            return seasonal_insights
            
        except Exception as e:
            self.logger.warning(f"계절성 분석 중 오류: {e}")
            return {'current_season': 'neutral', 'seasonal_bias': 'none', 'confidence_modifier': 1.0}
    
    def _identify_structural_risks(self, economic_context: Dict[str, str], 
                                 market_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """구조적 위험 요인 식별"""
        try:
            risks = {
                'identified_risks': [],
                'risk_level': 'moderate',
                'mitigation_strategies': []
            }
            
            # 인플레이션 관련 위험
            inflation_regime = economic_context.get('inflation_regime', 'unknown')
            if inflation_regime == 'high_inflation':
                risks['identified_risks'].append('purchasing_power_erosion')
                risks['identified_risks'].append('monetary_tightening_risk')
                risks['mitigation_strategies'].append('inflation_hedged_assets')
            
            # 금리 관련 위험
            rate_environment = economic_context.get('rate_environment', 'unknown')
            if rate_environment == 'rising_rates':
                risks['identified_risks'].append('duration_risk')
                risks['identified_risks'].append('valuation_compression')
                risks['mitigation_strategies'].append('shorter_duration_exposure')
            
            # 위험 심리 관련 위험
            risk_sentiment = economic_context.get('risk_sentiment', 'unknown')
            if risk_sentiment == 'high_fear':
                risks['identified_risks'].append('liquidity_crunch')
                risks['identified_risks'].append('correlation_breakdown')
                risks['mitigation_strategies'].append('diversification_increase')
            elif risk_sentiment == 'complacent':
                risks['identified_risks'].append('volatility_underpricing')
                risks['identified_risks'].append('tail_risk_buildup')
                risks['mitigation_strategies'].append('tail_risk_hedging')
            
            # 전체 위험 수준 결정
            risk_count = len(risks['identified_risks'])
            if risk_count >= 4:
                risks['risk_level'] = 'high'
            elif risk_count >= 2:
                risks['risk_level'] = 'moderate'
            else:
                risks['risk_level'] = 'low'
            
            return risks
            
        except Exception as e:
            self.logger.warning(f"구조적 위험 식별 중 오류: {e}")
            return {'identified_risks': [], 'risk_level': 'moderate', 'mitigation_strategies': []}
    
    def _calculate_confidence_modifiers(self, economic_context: Dict[str, str], 
                                      seasonal_context: Dict[str, Any], 
                                      structural_risks: Dict[str, Any]) -> List[float]:
        """신뢰도 수정자 계산"""
        try:
            modifiers = [1.0]  # 기본값
            
            # 경제 환경 기반 수정자
            for regime_type, context_value in economic_context.items():
                if regime_type in ['inflation_regime', 'rate_environment'] and context_value != 'unknown':
                    if regime_type == 'inflation_regime':
                        pattern = self.market_knowledge_base['inflation_patterns'].get(context_value, {})
                    else:
                        pattern = self.market_knowledge_base['rate_environment_patterns'].get(context_value, {})
                    
                    if pattern:
                        modifiers.append(pattern.get('confidence_modifier', 1.0))
            
            # 계절성 수정자
            modifiers.append(seasonal_context.get('confidence_modifier', 1.0))
            
            # 위험 수준 기반 수정자
            risk_level = structural_risks.get('risk_level', 'moderate')
            if risk_level == 'high':
                modifiers.append(0.8)
            elif risk_level == 'low':
                modifiers.append(1.1)
            
            return modifiers
            
        except Exception as e:
            self.logger.warning(f"신뢰도 수정자 계산 중 오류: {e}")
            return [1.0]
    
    def _generate_strategic_recommendations(self, current_regime: str, 
                                          economic_context: Dict[str, str], 
                                          insights: Dict[str, Any]) -> List[str]:
        """전략적 추천사항 생성"""
        try:
            recommendations = []
            
            # Regime별 기본 추천
            if current_regime == 'TRENDING_UP':
                recommendations.append("모멘텀 전략 활용하여 상승 추세 포착")
            elif current_regime == 'TRENDING_DOWN':
                recommendations.append("방어적 포지셔닝 및 헤징 전략 고려")
            elif current_regime == 'SIDEWAYS':
                recommendations.append("레인지 바운드 전략 및 평균 회귀 전략 활용")
            elif current_regime == 'VOLATILE':
                recommendations.append("변동성 거래 전략 및 포지션 크기 축소")
            
            # 경제 환경 기반 추천
            inflation_regime = economic_context.get('inflation_regime', 'unknown')
            if inflation_regime == 'high_inflation':
                recommendations.append("인플레이션 헤지 자산(부동산, 원자재) 비중 확대")
                recommendations.append("실물 자산 및 TIPS 고려")
            
            rate_environment = economic_context.get('rate_environment', 'unknown')
            if rate_environment == 'rising_rates':
                recommendations.append("금융주 비중 확대, 장기채 비중 축소")
                recommendations.append("가치주 대비 성장주 언더웨이트")
            elif rate_environment == 'falling_rates':
                recommendations.append("성장주 및 기술주 비중 확대")
                recommendations.append("장기 채권 매력도 증가")
            
            # 위험 관리 추천
            risk_level = insights.get('risk_adjustments', {}).get('risk_level', 'moderate')
            if risk_level == 'high':
                recommendations.append("포지션 크기 축소 및 손절매 기준 강화")
                recommendations.append("상관관계 낮은 자산으로 분산투자 강화")
            
            # 계절성 기반 추천
            seasonal_bias = insights.get('contextual_factors', {}).get('seasonal', {}).get('seasonal_bias', 'none')
            if seasonal_bias == 'bullish':
                recommendations.append("계절적 상승 요인 활용한 포지션 확대 고려")
            elif seasonal_bias == 'cautious':
                recommendations.append("여름철 약세 패턴 고려한 신중한 접근")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"전략적 추천사항 생성 중 오류: {e}")
            return ["시장 상황 지속 모니터링 필요"]


class MarketSensor:
    """통합 시장 분석 시스템 - 실행 인터페이스 (고도화된 RLMF 기반)"""
    
    def __init__(self, data_dir: str = "data/macro", config_path: str = "config/config_macro.json"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # 세션 UUID 생성
        self.session_uuid = str(uuid.uuid4())
        self.logger.info(f"MarketSensor 초기화 - Session UUID: {self.session_uuid}")
        
        # 핵심 컴포넌트들 초기화 (UUID 전달)
        self.macro_collector = GlobalMacroDataCollector(self.session_uuid)
        self.hyperparam_tuner = HyperparamTuner(config_path, self.session_uuid)
        self.macro_analyzer = MacroSectorAnalyzer(data_dir, self.session_uuid)
        
        # 고도화된 시스템 컴포넌트들 초기화
        self.rlmf_adaptation = RLMFRegimeAdaptation()
        self.confidence_system = MultiLayerConfidenceSystem()
        self.regime_detector = DynamicRegimeSwitchingDetector()
        self.llm_privileged_system = LLMPrivilegedInformationSystem()
        
        # Random Forest 모델 초기화 (저장된 모델 우선 로드)
        self.rf_model = MarketRegimeRF(verbose=True)
        try:
            # 저장된 모델 로드 시도
            self.rf_model.load_model()
            self.logger.info("저장된 Random Forest 모델을 로드했습니다.")
        except FileNotFoundError:
            self.logger.info("저장된 모델이 없습니다. 분석 시 새로 학습됩니다.")
        
        # 검증기 초기화
        self.validator = MarketRegimeValidator(self.session_uuid)
        
        # 최적 파라미터 (백테스팅으로 찾은 값)
        self.optimal_params = None
        
        # 모델 저장 디렉토리
        self.model_dir = f"{self.data_dir}/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_macro_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """매크로 데이터 로드"""
        try:
            # pandas 경고 억제
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
            warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
            
            # SPY 데이터 로드
            spy_path = f"{self.data_dir}/spy_data.csv"
            if os.path.exists(spy_path):
                spy_data = pd.read_csv(spy_path, index_col=0, parse_dates=False)
                # datetime 컬럼을 인덱스로 설정
                if 'datetime' in spy_data.columns:
                    spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], utc=True)
                    spy_data.set_index('datetime', inplace=True)
            else:
                self.logger.warning("SPY 데이터가 없습니다. 새로 수집합니다.")
                return self._collect_fresh_data()
            
            # 매크로 지표 로드
            macro_data = {}
            for symbol in self.macro_collector.macro_symbols.keys():
                macro_path = f"{self.data_dir}/{symbol.lower()}_data.csv"
                if os.path.exists(macro_path):
                    df = pd.read_csv(macro_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    macro_data[symbol] = df
            
            # 섹터 데이터 로드
            sector_data = {}
            for symbol in self.macro_collector.sector_etfs.keys():
                sector_path = f"{self.data_dir}/{symbol.lower()}_sector.csv"
                if os.path.exists(sector_path):
                    df = pd.read_csv(sector_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    sector_data[symbol] = df
            
            self.logger.info(f"매크로 데이터 로드 완료: SPY({len(spy_data)}), 매크로({len(macro_data)}), 섹터({len(sector_data)})")
            return spy_data, macro_data, sector_data
            
        except Exception as e:
            self.logger.error(f"매크로 데이터 로드 중 오류: {e}")
            return pd.DataFrame(), {}, {}
    
    def load_macro_data_only(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """저장된 매크로 데이터만 로드 (다운로드 없음)"""
        try:
            # pandas 경고 억제
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
            warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
            
            # SPY 데이터 로드
            spy_path = f"{self.data_dir}/spy_data.csv"
            if os.path.exists(spy_path):
                spy_data = pd.read_csv(spy_path, index_col=0, parse_dates=False)
                # datetime 컬럼을 인덱스로 설정
                if 'datetime' in spy_data.columns:
                    spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], utc=True)
                    spy_data.set_index('datetime', inplace=True)
            else:
                self.logger.warning("SPY 데이터가 없습니다.")
                return pd.DataFrame(), {}, {}
            
            # 매크로 지표 로드
            macro_data = {}
            for symbol in self.macro_collector.macro_symbols.keys():
                macro_path = f"{self.data_dir}/{symbol.lower()}_data.csv"
                if os.path.exists(macro_path):
                    df = pd.read_csv(macro_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    macro_data[symbol] = df
            
            # 섹터 데이터 로드
            sector_data = {}
            for symbol in self.macro_collector.sector_etfs.keys():
                sector_path = f"{self.data_dir}/{symbol.lower()}_sector.csv"
                if os.path.exists(sector_path):
                    df = pd.read_csv(sector_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    sector_data[symbol] = df
            
            self.logger.info(f"저장된 매크로 데이터 로드 완료: SPY({len(spy_data)}), 매크로({len(macro_data)}), 섹터({len(sector_data)})")
            return spy_data, macro_data, sector_data
            
        except Exception as e:
            self.logger.error(f"매크로 데이터 로드 중 오류: {e}")
            return pd.DataFrame(), {}, {}
    
    def _collect_fresh_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """새로운 매크로 데이터 수집 (설정 기반)"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 설정에서 데이터 수집 기간 가져오기
        days_back = self.macro_collector._get_days_back("macro_analysis")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        self.logger.info(f"매크로 데이터 수집 시작: {start_date} ~ {end_date} ({days_back}일)")
        
        spy_data = self.macro_collector.collect_spy_data(start_date, end_date)
        macro_data = self.macro_collector.collect_macro_indicators(start_date, end_date)
        sector_data = self.macro_collector.collect_sector_data(start_date, end_date)
        
        self.macro_collector.save_macro_data(spy_data, macro_data, sector_data, self.data_dir, start_date, end_date)
        
        self.logger.info(f"매크로 데이터 수집 완료: SPY({len(spy_data)}), 매크로({len(macro_data)}), 섹터({len(sector_data)})")
        
        return spy_data, macro_data, sector_data
    
    def save_trained_model(self, model, model_name: str = "market_regime_rf", 
                          metadata: Dict[str, Any] = None) -> str:
        """학습된 모델을 파일로 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_name}_{timestamp}.joblib"
            model_path = os.path.join(self.model_dir, model_filename)
            
            # 모델 저장
            joblib.dump(model, model_path)
            
            # 메타데이터 저장
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'model_name': model_name,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'session_uuid': self.session_uuid,
                'model_path': model_path
            })
            
            metadata_filename = f"{model_name}_{timestamp}_metadata.json"
            metadata_path = os.path.join(self.model_dir, metadata_filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"모델 저장 완료: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류: {e}")
            return ""
    
    def load_trained_model(self, model_name: str = "market_regime_rf", 
                          use_latest: bool = True, specific_timestamp: str = None) -> Tuple[Any, Dict[str, Any]]:
        """저장된 모델을 로드"""
        try:
            if specific_timestamp:
                # 특정 타임스탬프 모델 로드
                model_filename = f"{model_name}_{specific_timestamp}.joblib"
                metadata_filename = f"{model_name}_{specific_timestamp}_metadata.json"
            elif use_latest:
                # 최신 모델 찾기
                model_files = [f for f in os.listdir(self.model_dir) 
                             if f.startswith(model_name) and f.endswith('.joblib')]
                if not model_files:
                    raise FileNotFoundError(f"저장된 {model_name} 모델이 없습니다")
                
                # 타임스탬프로 정렬하여 최신 모델 선택
                model_files.sort(reverse=True)
                model_filename = model_files[0]
                timestamp = model_filename.replace(f"{model_name}_", "").replace(".joblib", "")
                metadata_filename = f"{model_name}_{timestamp}_metadata.json"
            else:
                raise ValueError("use_latest=True 또는 specific_timestamp를 지정해야 합니다")
            
            # 모델 로드
            model_path = os.path.join(self.model_dir, model_filename)
            model = joblib.load(model_path)
            
            # 메타데이터 로드
            metadata_path = os.path.join(self.model_dir, metadata_filename)
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            self.logger.info(f"모델 로드 완료: {model_path}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류: {e}")
            return None, {}
    
    def list_saved_models(self, model_name: str = "market_regime_rf") -> List[Dict[str, Any]]:
        """저장된 모델 목록 반환"""
        try:
            models = []
            for filename in os.listdir(self.model_dir):
                if filename.startswith(model_name) and filename.endswith('.joblib'):
                    timestamp = filename.replace(f"{model_name}_", "").replace(".joblib", "")
                    metadata_filename = f"{model_name}_{timestamp}_metadata.json"
                    metadata_path = os.path.join(self.model_dir, metadata_filename)
                    
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    
                    models.append({
                        'filename': filename,
                        'timestamp': timestamp,
                        'metadata': metadata
                    })
            
            # 타임스탬프로 정렬
            models.sort(key=lambda x: x['timestamp'], reverse=True)
            return models
            
        except Exception as e:
            self.logger.error(f"모델 목록 조회 중 오류: {e}")
            return []
    
    def get_macro_sector_analysis(self, start_date: str = None, end_date: str = None) -> MacroAnalysis:
        """매크로 & 섹터 분석 - MacroSectorAnalyzer 위임"""
        return self.macro_analyzer.get_comprehensive_analysis(start_date, end_date)
    
    def optimize_hyperparameters_optuna(self, start_date: str, end_date: str, n_trials: int = None, 
                                       spy_data: pd.DataFrame = None, macro_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """하이퍼파라미터 최적화 - HyperparamTuner 위임"""
        if spy_data is not None and macro_data is not None:
            # 이미 로드된 데이터 사용
            return self.hyperparam_tuner.optimize_hyperparameters_with_data(spy_data, macro_data, n_trials)
        else:
            # 기존 방식 (데이터 수집)
            return self.hyperparam_tuner.optimize_hyperparameters(start_date, end_date, n_trials)
    
    def save_optimization_results(self, results: Dict[str, Any], output_dir: str = "results/market_sensor_optimization"):
        """최적화 결과 저장 - HyperparamTuner 위임"""
        self.hyperparam_tuner.save_results(results, output_dir)
        
        # MarketSensor 전용 메타데이터 추가
        try:
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            market_sensor_metadata = {
                'session_uuid': self.session_uuid,
                'execution_type': 'market_sensor_optimization',
                'created_at': datetime.now().isoformat(),
                'data_dir': self.data_dir,
                'optimization_completed': True
            }
            
            with open(f"{session_dir}/market_sensor_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(market_sensor_metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"MarketSensor 메타데이터 저장 중 오류: {e}")
    
    def save_macro_analysis_results(self, analysis: MacroAnalysis, output_dir: str = "results/macro_sector_analysis"):
        """매크로 분석 결과 저장 - MacroSectorAnalyzer 위임"""
        self.macro_analyzer.save_analysis_results(analysis, output_dir)
        
        # MarketSensor 전용 메타데이터 추가
        try:
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            market_sensor_metadata = {
                'session_uuid': self.session_uuid,
                'execution_type': 'market_sensor_macro_analysis',
                'created_at': datetime.now().isoformat(),
                'data_dir': self.data_dir,
                'analysis_completed': True
            }
            
            with open(f"{session_dir}/market_sensor_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(market_sensor_metadata, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"MarketSensor 메타데이터 저장 중 오류: {e}")
    
    def load_optimal_params(self, input_path: str = "results/market_sensor_optimization/best_params.json") -> Dict[str, Any]:
        """최적 파라미터 로드"""
        try:
            if os.path.exists(input_path):
                with open(input_path, 'r', encoding='utf-8') as f:
                    params_data = json.load(f)
                
                # best_params.json은 직접 파라미터 딕셔너리이므로 optimal_params 키가 없음
                self.optimal_params = params_data
                self.logger.info(f"최적 파라미터 로드 완료: {input_path}")
                return self.optimal_params
            else:
                self.logger.warning(f"최적 파라미터 파일이 없습니다: {input_path}")
                return {}
                
        except Exception as e:
            self.logger.error(f"최적 파라미터 로드 중 오류: {e}")
            return {}
    
    def get_current_market_analysis(self, use_optimized_params: bool = True, use_ml_model: bool = True) -> Dict[str, Any]:
        """
        현재 시장 분석 결과 반환 (RLMF 기반 고도화된 종합 분석)
        
        주요 개선사항:
        1. RLMF 피드백 시스템 통합
        2. Statistical Arbitrage 신호 활용
        3. 다층 신뢰도 계산 시스템
        4. 동적 regime switching 감지
        """
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if spy_data.empty:
                return {'error': 'SPY 데이터를 로드할 수 없습니다.'}
            
            # 1. 기본 파라미터 설정
            if use_optimized_params and self.optimal_params:
                params = self.optimal_params
            else:
                params = {
                    'sma_short': 20, 'sma_long': 50, 'rsi_period': 14,
                    'rsi_overbought': 70, 'rsi_oversold': 30, 'atr_period': 14,
                    'trend_weight': 0.4, 'momentum_weight': 0.3,
                    'volatility_weight': 0.2, 'macro_weight': 0.1,
                    'base_position': 0.8, 'trending_boost': 1.2, 'volatile_reduction': 0.5
                }
            
            # RLMF 적응 가중치 반영
            adaptation_weights = self.rlmf_adaptation.adaptation_weights
            params['trend_weight'] *= adaptation_weights.get('trend_strength', 1.0)
            params['volatility_weight'] *= adaptation_weights.get('volatility_regime', 1.0)
            params['momentum_weight'] *= adaptation_weights.get('momentum_persistence', 1.0)
            
            # 파생 변수 계산
            data_with_features = self.hyperparam_tuner._calculate_derived_features(spy_data, params)
            
            # 매크로 데이터 병합
            if '^VIX' in macro_data:
                vix_df = macro_data['^VIX']
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 2. Statistical Arbitrage 신호 계산 (Keybot the Quant 방식)
            stat_arb_signal = self.rlmf_adaptation.calculate_statistical_arbitrage_signal(macro_data)
            self.logger.info(f"Statistical Arbitrage 신호: {stat_arb_signal['direction']} "
                           f"(강도: {stat_arb_signal['signal_strength']:.3f})")
            
            # 3. Dynamic Regime Switching 감지
            regime_shift_info = self.regime_detector.detect_regime_shifts(spy_data, macro_data)
            if regime_shift_info['regime_shift_detected']:
                self.logger.warning(f"Regime shift 감지됨! 점수: {regime_shift_info['shift_score']:.3f}")
            
            # 4. ML 모델 기반 시장 상태 분류 (확률 포함)
            current_probabilities = {}
            try:
                if use_ml_model:
                    # ML 기반 확률 계산 (저장된 모델 우선 사용)
                    current_probabilities = self.rf_model.get_current_market_probabilities(data_dir=self.data_dir)
                    
                    # Statistical Arbitrage 신호로 확률 조정
                    if stat_arb_signal['direction'] == 'BULLISH':
                        current_probabilities['TRENDING_UP'] = current_probabilities.get('TRENDING_UP', 0.5) * 1.2
                    elif stat_arb_signal['direction'] == 'BEARISH':
                        current_probabilities['TRENDING_DOWN'] = current_probabilities.get('TRENDING_DOWN', 0.5) * 1.2
                    
                    # 확률 정규화
                    total_prob = sum(current_probabilities.values())
                    if total_prob > 0:
                        current_probabilities = {k: v/total_prob for k, v in current_probabilities.items()}
                    
                    # 확률 순서대로 정렬하여 시장 상태 결정
                    sorted_probs = sorted(current_probabilities.items(), key=lambda x: x[1], reverse=True)
                    primary_regime = sorted_probs[0][0]
                    secondary_regime = sorted_probs[1][0] if len(sorted_probs) > 1 else None
                    primary_prob = current_probabilities.get(primary_regime, 0.5)
                    secondary_prob = current_probabilities.get(secondary_regime, 0.0) if secondary_regime else 0.0
                    
                    current_regime = primary_regime.upper()
                    
                    # regime을 시리즈로 변환 (기존 코드와 호환)
                    regime = pd.Series([current_regime] * len(data_with_features), index=data_with_features.index)
                    
                    self.logger.info(f"ML 모델 기반 분석: {current_regime} (확률: {primary_prob:.3f})")
                else:
                    # 기존 규칙 기반 분석
                    regime = self.hyperparam_tuner._classify_market_regime(data_with_features, params)
                    current_regime = regime.iloc[-1]
                    
            except Exception as e:
                self.logger.warning(f"ML 모델 분석 실패, 규칙 기반 분석 사용: {e}")
                # ML 모델 실패 시 규칙 기반 분석으로 fallback
                regime = self.hyperparam_tuner._classify_market_regime(data_with_features, params)
                current_regime = regime.iloc[-1]
            
            # 5. 다층 신뢰도 계산 시스템
            # 각 구성요소별 신뢰도 계산
            technical_conf = self._calculate_regime_confidence(data_with_features, current_regime, macro_data)
            macro_conf = self._calculate_macro_confidence(macro_data)
            stat_arb_conf = stat_arb_signal['confidence']
            rlmf_conf = self.rlmf_adaptation.get_adaptation_status()['performance']
            
            # 종합 신뢰도 계산
            comprehensive_confidence = self.confidence_system.calculate_comprehensive_confidence(
                technical_conf, macro_conf, stat_arb_conf, rlmf_conf
            )
            
            final_confidence = comprehensive_confidence['adjusted_confidence']
            
            self.logger.info(f"종합 신뢰도: {final_confidence:.3f} "
                           f"(기술적: {technical_conf:.3f}, 매크로: {macro_conf:.3f}, "
                           f"StatArb: {stat_arb_conf:.3f}, RLMF: {rlmf_conf:.3f})")
            
            # 6. 전략 수익률 계산
            strategy_returns = self.hyperparam_tuner._calculate_strategy_returns(data_with_features, regime, params)
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            buy_hold_returns = spy_data[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self.hyperparam_tuner._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            # 7. 매크로 분석
            macro_analysis = None
            sector_analysis = None
            
            if macro_data and sector_data:
                try:
                    # 매크로 환경 분석
                    macro_analysis = self.macro_analyzer.analyze_macro_environment(macro_data)
                    
                    # 섹터 로테이션 분석
                    sector_analysis = self.macro_analyzer.analyze_sector_rotation(sector_data)
                    
                except Exception as e:
                    self.logger.warning(f"매크로 분석 중 오류: {e}")
            
            # 8. 종합 전략 추천
            recommendation = self.recommend_strategy(MarketClassification(
                regime=MarketRegime(current_regime),
                confidence=final_confidence,
                features={},
                timestamp=datetime.now(),
                metadata={}
            ))
            
            # Statistical Arbitrage 신호 반영
            if stat_arb_signal['signal_strength'] > 0.5:
                recommendation['statistical_arbitrage_bias'] = stat_arb_signal['direction']
                recommendation['key_metrics_signals'] = stat_arb_signal['individual_signals']
            
            # 매크로 분석 결과가 있으면 전략에 반영
            if macro_analysis and sector_analysis:
                recommendation = self._enhance_recommendation_with_macro(
                    recommendation, macro_analysis, sector_analysis
                )
            
            # 9. LLM Privileged Information 시스템 활용
            market_metrics = {
                'vix_level': macro_data.get('^VIX', pd.DataFrame()).iloc[-1]['close'] if '^VIX' in macro_data and not macro_data['^VIX'].empty else 20,
                'current_probabilities': current_probabilities,
                'stat_arb_signal': stat_arb_signal
            }
            
            llm_insights = self.llm_privileged_system.get_privileged_insights(
                current_regime, macro_data, market_metrics
            )
            
            # LLM 인사이트를 활용한 최종 신뢰도 조정
            llm_confidence_modifiers = llm_insights.get('confidence_modifiers', [1.0])
            llm_confidence_adjustment = np.mean(llm_confidence_modifiers)
            
            # Regime 검증 결과 반영
            regime_consistency = llm_insights.get('regime_validation', {}).get('consistency', 0.5)
            if regime_consistency < 0.3:
                self.logger.warning(f"LLM 지식과 현재 regime 불일치 감지 (일관성: {regime_consistency:.3f})")
                final_confidence *= 0.8  # 신뢰도 감소
            
            # 최종 신뢰도에 LLM 조정 반영
            final_confidence *= llm_confidence_adjustment
            final_confidence = max(0.1, min(0.9, final_confidence))
            
            self.logger.info(f"LLM 조정 후 최종 신뢰도: {final_confidence:.3f} "
                           f"(LLM 조정: {llm_confidence_adjustment:.3f}, Regime 일관성: {regime_consistency:.3f})")
            
            # 전략 추천에 LLM 인사이트 반영
            if llm_insights['strategic_recommendations']:
                recommendation['llm_strategic_insights'] = llm_insights['strategic_recommendations']
            
            # 10. RLMF 피드백 업데이트 (실제 수익률이 있을 때)
            if len(strategy_returns) >= 5:
                feedback = self.rlmf_adaptation.calculate_market_feedback(
                    current_regime, strategy_returns.tail(10), spy_data, macro_data
                )
                self.rlmf_adaptation.update_adaptation_weights(feedback)
            
            return {
                'current_regime': current_regime,
                'confidence': final_confidence,
                'confidence_breakdown': comprehensive_confidence,
                'probabilities': current_probabilities if use_ml_model else None,
                'statistical_arbitrage': stat_arb_signal,
                'regime_shift_detection': regime_shift_info,
                'rlmf_status': self.rlmf_adaptation.get_adaptation_status(),
                'llm_privileged_insights': llm_insights,
                'performance_metrics': metrics,
                'recommendation': recommendation,
                'macro_analysis': macro_analysis,
                'sector_analysis': sector_analysis,
                'last_update': datetime.now().isoformat(),
                'data_period': f"{spy_data.index[0].strftime('%Y-%m-%d')} ~ {spy_data.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            self.logger.error(f"현재 시장 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def get_enhanced_market_summary(self) -> str:
        """
        고도화된 시장 분석 결과 요약 출력
        """
        try:
            analysis = self.get_current_market_analysis(use_optimized_params=True, use_ml_model=True)
            
            if 'error' in analysis:
                return f"❌ 분석 오류: {analysis['error']}"
            
            # 기본 정보
            regime = analysis['current_regime']
            confidence = analysis['confidence']
            
            # Statistical Arbitrage 신호
            stat_arb = analysis.get('statistical_arbitrage', {})
            stat_direction = stat_arb.get('direction', 'NEUTRAL')
            stat_strength = stat_arb.get('signal_strength', 0.0)
            
            # Regime Shift 감지
            regime_shift = analysis.get('regime_shift_detection', {})
            shift_detected = regime_shift.get('regime_shift_detected', False)
            
            # LLM 인사이트
            llm_insights = analysis.get('llm_privileged_insights', {})
            regime_validation = llm_insights.get('regime_validation', {})
            consistency = regime_validation.get('consistency', 0.5)
            
            # RLMF 상태
            rlmf_status = analysis.get('rlmf_status', {})
            rlmf_performance = rlmf_status.get('performance', 0.5)
            
            summary = f"""
📊 **고도화된 시장 분석 결과** 📊

🎯 **현재 Market Regime**: {regime}
📈 **종합 신뢰도**: {confidence:.1%}

🔄 **Statistical Arbitrage 신호**:
   • 방향: {stat_direction}
   • 강도: {stat_strength:.3f}

⚠️ **Regime Shift 감지**: {'🚨 감지됨!' if shift_detected else '✅ 안정'}

🧠 **LLM 지식 검증**:
   • Regime 일관성: {consistency:.1%}
   • 지원 요인: {len(regime_validation.get('supporting_factors', []))}개
   • 충돌 요인: {len(regime_validation.get('conflicting_factors', []))}개

🤖 **RLMF 적응 상태**:
   • 학습 성과: {rlmf_performance:.1%}
   • 피드백 수: {rlmf_status.get('feedback_count', 0)}회

💡 **전략 추천사항**:
"""
            
            # 추천사항 추가
            recommendations = analysis.get('recommendation', {})
            if 'llm_strategic_insights' in recommendations:
                for i, rec in enumerate(recommendations['llm_strategic_insights'][:3], 1):
                    summary += f"   {i}. {rec}\n"
            
            summary += f"\n⏰ **마지막 업데이트**: {analysis['last_update'][:19]}"
            
            return summary
            
        except Exception as e:
            return f"❌ 요약 생성 중 오류: {e}"
    
    def _calculate_regime_confidence(self, data_with_features: pd.DataFrame, current_regime: str, macro_data: Dict[str, pd.DataFrame]) -> float:
        """시장 환경 감지 신뢰도 계산"""
        try:
            confidence = 0.5  # 기본 신뢰도
            
            # 1. 기술적 지표 기반 신뢰도 (0.3 ~ 0.7)
            technical_confidence = 0.5
            
            # 추세 강도 확인
            if 'sma_short' in data_with_features.columns and 'sma_long' in data_with_features.columns:
                sma_diff = abs(data_with_features['sma_short'].iloc[-1] - data_with_features['sma_long'].iloc[-1])
                sma_ratio = sma_diff / data_with_features['sma_long'].iloc[-1]
                
                if sma_ratio > 0.05:  # 강한 추세
                    technical_confidence += 0.2
                elif sma_ratio > 0.02:  # 중간 추세
                    technical_confidence += 0.1
                else:  # 약한 추세
                    technical_confidence -= 0.1
            
            # RSI 극값 확인
            if 'rsi' in data_with_features.columns:
                rsi_value = data_with_features['rsi'].iloc[-1]
                if rsi_value > 70 or rsi_value < 30:  # 극값
                    technical_confidence += 0.1
                elif 40 < rsi_value < 60:  # 중립
                    technical_confidence -= 0.1
            
            # 변동성 확인
            if 'atr' in data_with_features.columns:
                atr_value = data_with_features['atr'].iloc[-1]
                atr_avg = data_with_features['atr'].rolling(20).mean().iloc[-1]
                
                if atr_value > atr_avg * 1.5:  # 높은 변동성
                    technical_confidence += 0.1
                elif atr_value < atr_avg * 0.5:  # 낮은 변동성
                    technical_confidence -= 0.1
            
            # 2. 매크로 데이터 기반 신뢰도 (0.2 ~ 0.4)
            macro_confidence = 0.3
            
            # VIX 데이터 확인
            if '^VIX' in macro_data and not macro_data['^VIX'].empty:
                macro_confidence += 0.1
            
            # TIPS 데이터 확인
            if '^TIP' in macro_data and not macro_data['^TIP'].empty:
                macro_confidence += 0.1
            
            # 데이터 품질 확인
            data_quality = 0.0
            for symbol, df in macro_data.items():
                if not df.empty and len(df) > 30:  # 충분한 데이터
                    data_quality += 0.1
            
            macro_confidence += min(data_quality, 0.2)
            
            # 3. 최종 신뢰도 계산
            confidence = (technical_confidence * 0.7) + (macro_confidence * 0.3)
            
            # 신뢰도 범위 제한 (0.1 ~ 0.9)
            confidence = max(0.1, min(0.9, confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"신뢰도 계산 중 오류: {e}")
            return 0.5  # 기본값
    
    def _calculate_macro_confidence(self, macro_data: Dict[str, pd.DataFrame]) -> float:
        """매크로 데이터 기반 신뢰도 계산"""
        try:
            confidence = 0.3  # 기본값
            
            # 데이터 품질 평가
            available_indicators = 0
            total_indicators = 5  # VIX, TNX, TIP, DXY, GLD 등
            
            key_indicators = ['^VIX', '^TNX', '^TIP', 'DX-Y.NYB', 'GLD']
            
            for indicator in key_indicators:
                if indicator in macro_data and not macro_data[indicator].empty:
                    data = macro_data[indicator]
                    if len(data) > 30:  # 충분한 데이터
                        available_indicators += 1
                        confidence += 0.1
            
            # 데이터 최신성 평가
            latest_data_count = 0
            for symbol, df in macro_data.items():
                if not df.empty and len(df) > 0:
                    latest_date = df.index[-1]
                    days_old = (datetime.now().date() - latest_date.date()).days
                    if days_old <= 7:  # 일주일 이내 데이터
                        latest_data_count += 1
            
            if latest_data_count >= 3:
                confidence += 0.15
            elif latest_data_count >= 1:
                confidence += 0.1
            
            # VIX 데이터 특별 가중치 (변동성 지표의 중요성)
            if '^VIX' in macro_data and not macro_data['^VIX'].empty:
                confidence += 0.05
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.warning(f"매크로 신뢰도 계산 중 오류: {e}")
            return 0.3
    
    def _calculate_ml_confidence(self, primary_prob: float, secondary_prob: float, 
                               data_with_features: pd.DataFrame, macro_data: Dict[str, pd.DataFrame]) -> float:
        """ML 모델 기반 신뢰도 계산 - 개선된 버전"""
        try:
            # 1. 확률 기반 신뢰도 (0.4 ~ 0.8) - 기본값 상향 조정
            prob_confidence = 0.5  # 기본값을 0.5로 상향
            
            # 확률 차이에 따른 신뢰도 조정
            prob_diff = primary_prob - secondary_prob
            if prob_diff > 0.25:  # 확실한 차이 (임계값 완화)
                prob_confidence += 0.25
            elif prob_diff > 0.15:  # 명확한 차이
                prob_confidence += 0.15
            elif prob_diff > 0.08:  # 약간의 차이 (임계값 완화)
                prob_confidence += 0.08
            else:  # 불확실한 차이
                prob_confidence -= 0.05  # 패널티 완화
            
            # 최고 확률 자체의 높낮이 (더 관대하게 조정)
            if primary_prob > 0.5:  # 높은 확률 (임계값 완화)
                prob_confidence += 0.1
            elif primary_prob > 0.4:  # 중간 확률
                prob_confidence += 0.05
            elif primary_prob < 0.25:  # 낮은 확률
                prob_confidence -= 0.05  # 패널티 완화
            
            # 2. 기술적 지표 기반 신뢰도 (0.2 ~ 0.4)
            technical_confidence = 0.2
            
            # 추세 강도 확인
            if 'sma_short' in data_with_features.columns and 'sma_long' in data_with_features.columns:
                sma_diff = abs(data_with_features['sma_short'].iloc[-1] - data_with_features['sma_long'].iloc[-1])
                sma_ratio = sma_diff / data_with_features['sma_long'].iloc[-1]
                
                if sma_ratio > 0.05:  # 강한 추세
                    technical_confidence += 0.1
                elif sma_ratio > 0.02:  # 중간 추세
                    technical_confidence += 0.05
                else:  # 약한 추세
                    technical_confidence -= 0.05
            
            # RSI 극값 확인
            if 'rsi' in data_with_features.columns:
                rsi_value = data_with_features['rsi'].iloc[-1]
                if rsi_value > 70 or rsi_value < 30:  # 극값
                    technical_confidence += 0.05
                elif 40 < rsi_value < 60:  # 중립
                    technical_confidence -= 0.05
            
            # 변동성 확인
            if 'atr' in data_with_features.columns:
                atr_value = data_with_features['atr'].iloc[-1]
                atr_avg = data_with_features['atr'].rolling(20).mean().iloc[-1]
                
                if atr_value > atr_avg * 1.5:  # 높은 변동성
                    technical_confidence += 0.05
                elif atr_value < atr_avg * 0.5:  # 낮은 변동성
                    technical_confidence -= 0.05
            
            # 3. 매크로 데이터 기반 신뢰도 (0.1 ~ 0.2)
            macro_confidence = 0.1
            
            # 데이터 품질 확인
            data_quality = 0.0
            for symbol, df in macro_data.items():
                if not df.empty and len(df) > 30:  # 충분한 데이터
                    data_quality += 0.02
            
            macro_confidence += min(data_quality, 0.1)
            
            # 4. 최종 신뢰도 계산 (가중치 조정)
            confidence = (prob_confidence * 0.6) + (technical_confidence * 0.3) + (macro_confidence * 0.1)
            
            # 신뢰도 범위 제한 (0.1 ~ 0.9)
            confidence = max(0.1, min(0.9, confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"ML 신뢰도 계산 중 오류: {e}")
            return 0.5  # 기본값
    
    def _enhance_recommendation_with_macro(self, base_recommendation: Dict[str, Any], 
                                         macro_analysis: Dict[str, Any], 
                                         sector_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """매크로 분석 결과를 기반으로 전략 추천 강화"""
        enhanced_recommendation = base_recommendation.copy()
        
        # 매크로 환경에 따른 포지션 크기 조정
        if 'inflation_risk' in macro_analysis:
            inflation_risk = macro_analysis['inflation_risk']
            if inflation_risk > 0.7:  # 높은 인플레이션 위험
                enhanced_recommendation['position_size'] *= 0.8
                enhanced_recommendation['description'] += " (인플레이션 위험으로 인한 포지션 축소)"
            elif inflation_risk < 0.3:  # 낮은 인플레이션 위험
                enhanced_recommendation['position_size'] *= 1.1
                enhanced_recommendation['description'] += " (낮은 인플레이션으로 인한 포지션 확대)"
        
        # 금리 환경에 따른 조정
        if 'rate_environment' in macro_analysis:
            rate_env = macro_analysis['rate_environment']
            if rate_env == 'high_rates':
                enhanced_recommendation['stop_loss'] *= 1.2  # 손절폭 확대
                enhanced_recommendation['description'] += " (고금리 환경으로 인한 손절폭 확대)"
            elif rate_env == 'low_rates':
                enhanced_recommendation['take_profit'] *= 1.1  # 익절폭 확대
                enhanced_recommendation['description'] += " (저금리 환경으로 인한 익절폭 확대)"
        
        # 섹터 추천 추가
        if sector_analysis:
            leading_sectors = [sector for sector, strength in sector_analysis.items() 
                             if strength == SectorStrength.LEADING]
            defensive_sectors = [sector for sector, strength in sector_analysis.items() 
                               if strength == SectorStrength.DEFENSIVE]
            
            if leading_sectors:
                enhanced_recommendation['leading_sectors'] = leading_sectors
            if defensive_sectors:
                enhanced_recommendation['defensive_sectors'] = defensive_sectors
        
        return enhanced_recommendation
    
    def get_comprehensive_macro_analysis(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """종합 매크로 분석 (MacroSectorAnalyzer 활용)"""
        try:
            analysis = self.macro_analyzer.get_comprehensive_analysis(start_date, end_date)
            
            if analysis is None:
                return {'error': '매크로 분석을 수행할 수 없습니다.'}
            
            return {
                'market_condition': analysis.market_condition.value,
                'confidence': analysis.confidence,
                'key_indicators': analysis.key_indicators,
                'sector_rotation': {sector: strength.value for sector, strength in analysis.sector_rotation.items()},
                'recommendations': analysis.recommendations,
                'timestamp': analysis.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"종합 매크로 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def get_technical_analysis(self, use_optimized_params: bool = True, use_ml_model: bool = False) -> Dict[str, Any]:
        """기술적 분석 (시장 상태 분류 + 확률 분석)"""
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if spy_data.empty:
                return {'error': 'SPY 데이터를 로드할 수 없습니다.'}
            
            # 파라미터 선택
            if use_optimized_params and self.optimal_params:
                params = self.optimal_params
            else:
                params = {
                    'sma_short': 20, 'sma_long': 50, 'rsi_period': 14,
                    'rsi_overbought': 70, 'rsi_oversold': 30, 'atr_period': 14,
                    'trend_weight': 0.4, 'momentum_weight': 0.3,
                    'volatility_weight': 0.2, 'macro_weight': 0.1,
                    'base_position': 0.8, 'trending_boost': 1.2, 'volatile_reduction': 0.5
                }
            
            # 파생 변수 계산
            data_with_features = self.hyperparam_tuner._calculate_derived_features(spy_data, params)
            
            # 매크로 데이터 병합 (VIX 등)
            if '^VIX' in macro_data:
                vix_df = macro_data['^VIX']
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 시장 상태 분류 (두 가지 방식 지원)
            if use_ml_model:
                # ML 기반 분석 (Random Forest)
                try:
                    # 저장된 모델 로드 시도
                    saved_model, model_metadata = self.load_trained_model(use_latest=True)
                    if saved_model is not None:
                        self.logger.info(f"저장된 ML 모델 로드 완료: {model_metadata.get('timestamp', 'unknown')}")
                        # 저장된 모델을 rf_model에 설정
                        self.rf_model.model = saved_model
                    else:
                        # 기존 방식으로 모델 로드 시도
                        self.rf_model.load_model()
                        self.logger.info("기존 ML 모델 로드 완료")
                except Exception as e:
                    self.logger.warning(f"ML 모델 로드 실패: {e}. Quant 기반 분석을 사용합니다.")
                    use_ml_model = False
            
            if use_ml_model:
                # ML 기반 확률 계산
                current_probabilities = self.rf_model.get_current_market_probabilities(data_dir=self.data_dir)
                analysis_method = "ML (Random Forest)"
                # ML 모델에서는 regime을 확률이 가장 높은 것으로 설정
                regime = None  # 나중에 설정됨
            else:
                # Quant 기반 분석 (기존 방식)
                regime_analysis = self.hyperparam_tuner._classify_market_regime_with_probabilities(data_with_features, params)
                regime = regime_analysis['regime']
                probabilities = regime_analysis['probabilities']
                
                current_probabilities = {
                    'trending_up': float(probabilities['trending_up'][-1]),
                    'trending_down': float(probabilities['trending_down'][-1]),
                    'volatile': float(probabilities['volatile'][-1]),
                    'sideways': float(probabilities['sideways'][-1])
                }
                analysis_method = "Quant (Rule-based)"
            
            # 확률 순서대로 정렬하여 시장 상태와 두 번째 가능성 결정
            sorted_probs = sorted(current_probabilities.items(), key=lambda x: x[1], reverse=True)
            primary_regime = sorted_probs[0][0]
            secondary_regime = sorted_probs[1][0] if len(sorted_probs) > 1 else None
            secondary_probability = sorted_probs[1][1] if len(sorted_probs) > 1 else 0
            
            # 시장 상태를 확률이 가장 높은 상태로 설정 (일관성 유지)
            current_regime = primary_regime.upper()
            
            # ML 모델을 사용하는 경우 regime 변수 설정
            if use_ml_model and regime is None:
                # ML 모델에서는 regime을 시리즈로 변환 (모든 행에 동일한 값)
                regime = pd.Series([current_regime] * len(data_with_features), index=data_with_features.index)
            
            # 전략 수익률 계산
            strategy_returns = self.hyperparam_tuner._calculate_strategy_returns(data_with_features, regime, params)
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            buy_hold_returns = spy_data[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self.hyperparam_tuner._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            # 전략 추천
            recommendation = self.recommend_strategy(MarketClassification(
                regime=MarketRegime(current_regime),
                confidence=current_probabilities.get(primary_regime, 0.8),
                features=current_probabilities,
                timestamp=datetime.now(),
                metadata={'secondary_regime': secondary_regime, 'secondary_probability': secondary_probability}
            ))
            
            return {
                'analysis_type': 'technical',
                'analysis_method': analysis_method,
                'current_regime': current_regime,
                'regime_probabilities': current_probabilities,
                'primary_regime': primary_regime,
                'secondary_regime': secondary_regime,
                'secondary_probability': secondary_probability,
                'performance_metrics': metrics,
                'recommendation': recommendation,
                'technical_indicators': {
                    'rsi': data_with_features['rsi'].iloc[-1] if 'rsi' in data_with_features.columns else None,
                    'macd': data_with_features['macd'].iloc[-1] if 'macd' in data_with_features.columns else None,
                    'sma_short': data_with_features[f'sma_{params.get("sma_short", 20)}'].iloc[-1] if f'sma_{params.get("sma_short", 20)}' in data_with_features.columns else None,
                    'sma_long': data_with_features[f'sma_{params.get("sma_long", 50)}'].iloc[-1] if f'sma_{params.get("sma_long", 50)}' in data_with_features.columns else None,
                    'atr': data_with_features['atr'].iloc[-1] if 'atr' in data_with_features.columns else None,
                },
                'last_update': datetime.now().isoformat(),
                'data_period': f"{spy_data.index[0].strftime('%Y-%m-%d')} ~ {spy_data.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            self.logger.error(f"기술적 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def get_macro_analysis_only(self) -> Dict[str, Any]:
        """매크로 분석만 수행 (VIX, TIPS 등 상세 분석 포함)"""
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if not macro_data:
                return {'error': '매크로 데이터를 로드할 수 없습니다.'}
            
            # 매크로 환경 분석
            macro_analysis = self.macro_analyzer.analyze_macro_environment(macro_data)
            
            # 시장 조건 분류
            market_condition = self.macro_analyzer.classify_market_condition(macro_analysis)
            
            # VIX 상세 분석
            vix_analysis = {}
            if '^VIX' in macro_data:
                vix_data = macro_data['^VIX']
                close_col = 'close' if 'close' in vix_data.columns else 'Close'
                if close_col in vix_data.columns:
                    vix_series = vix_data[close_col]
                    # 52주 백분위 계산 (NaN 방지)
                    rolling_max = vix_series.rolling(252).max()
                    percentile_52w = 0.0
                    if not rolling_max.isna().all() and rolling_max.iloc[-1] > 0:
                        percentile_52w = float((vix_series.iloc[-1] / rolling_max.iloc[-1]) * 100)
                    
                    # 20일 평균 계산 (NaN 방지)
                    ma_20 = vix_series.rolling(20).mean().iloc[-1]
                    if pd.isna(ma_20):
                        ma_20 = vix_series.iloc[-1]  # 평균이 NaN이면 현재값 사용
                    
                    vix_analysis = {
                        'current_level': float(vix_series.iloc[-1]),
                        'ma_20': float(ma_20),
                        'percentile_52w': percentile_52w,
                        'volatility_regime': 'high' if vix_series.iloc[-1] > 25 else 'normal',
                        'trend': 'increasing' if vix_series.iloc[-1] > ma_20 else 'decreasing'
                    }
            
            # TIPS Spread 상세 분석
            tips_analysis = {}
            tips_spread_indicators = [key for key in macro_analysis.keys() if 'tips_spread' in key]
            if tips_spread_indicators:
                tips_analysis = {
                    'composite_spread': macro_analysis.get('tips_spread_composite', 0),
                    'composite_ma_50': macro_analysis.get('tips_spread_composite_ma_50', 0),
                    'inflation_expectation': macro_analysis.get('inflation_expectation', 'unknown'),
                    'inflation_trend': macro_analysis.get('inflation_trend', 'unknown'),
                    'tip_tlt_ratio': macro_analysis.get('tips_spread_tip_tlt', 0),
                    'short_long_ratio': macro_analysis.get('tips_spread_short_long', 0),
                    'schp_tlt_ratio': macro_analysis.get('tips_spread_schp_tlt', 0)
                }
            
            # 국채 스프레드 분석
            yield_analysis = {}
            if 'yield_spread' in macro_analysis:
                yield_analysis = {
                    'current_spread': macro_analysis.get('yield_spread', 0),
                    'spread_ma_20': macro_analysis.get('spread_ma_20', 0),
                    'recession_risk': macro_analysis.get('recession_risk', 'unknown'),
                    'spread_trend': 'inverting' if macro_analysis.get('yield_spread', 0) < macro_analysis.get('spread_ma_20', 0) else 'normal'
                }
            
            # 달러 강도 분석
            dollar_analysis = {}
            if 'dollar_strength' in macro_analysis:
                dollar_analysis = {
                    'current_level': macro_analysis.get('dollar_strength', 0),
                    'ma_50': macro_analysis.get('dollar_ma_50', 0),
                    'trend': macro_analysis.get('dollar_trend', 'unknown'),
                    'strength_level': 'strong' if macro_analysis.get('dollar_strength', 0) > macro_analysis.get('dollar_ma_50', 0) else 'weak'
                }
            
            # 금 가격 분석
            gold_analysis = {}
            if 'gold_price' in macro_analysis:
                gold_analysis = {
                    'current_price': macro_analysis.get('gold_price', 0),
                    'ma_50': macro_analysis.get('gold_ma_50', 0),
                    'trend': macro_analysis.get('gold_trend', 'unknown'),
                    'safe_haven_demand': 'high' if macro_analysis.get('gold_trend', 'unknown') == 'bullish' else 'low'
                }
            
            # 국채 가격 분석
            bond_analysis = {}
            if 'bond_price' in macro_analysis:
                bond_analysis = {
                    'current_price': macro_analysis.get('bond_price', 0),
                    'ma_50': macro_analysis.get('bond_ma_50', 0),
                    'trend': macro_analysis.get('bond_trend', 'unknown'),
                    'flight_to_quality': 'high' if macro_analysis.get('bond_trend', 'unknown') == 'bullish' else 'low'
                }
            
            return {
                'analysis_type': 'macro',
                'market_condition': market_condition.value,
                'macro_analysis': macro_analysis,
                'vix_analysis': vix_analysis,
                'tips_analysis': tips_analysis,
                'yield_analysis': yield_analysis,
                'dollar_analysis': dollar_analysis,
                'gold_analysis': gold_analysis,
                'bond_analysis': bond_analysis,
                'key_indicators': {
                    'inflation_risk': macro_analysis.get('inflation_risk', 0),
                    'rate_environment': macro_analysis.get('rate_environment', 'unknown'),
                    'growth_outlook': macro_analysis.get('growth_outlook', 'unknown'),
                    'vix_level': macro_analysis.get('vix_current', 0),
                    'yield_curve': macro_analysis.get('yield_spread', 0)
                },
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"매크로 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def get_sector_analysis_only(self) -> Dict[str, Any]:
        """섹터 분석만 수행"""
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if not sector_data:
                self.logger.warning("섹터 데이터를 로드할 수 없습니다. 기본 섹터 분석을 제공합니다.")
                return self._get_default_sector_analysis()
            
            # 섹터 로테이션 분석
            try:
                sector_analysis = self.macro_analyzer.analyze_sector_rotation(sector_data)
                
                # 섹터별 분류
                leading_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.LEADING]
                lagging_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.LAGGING]
                defensive_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.DEFENSIVE]
                cyclical_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.CYCLICAL]
                
                return {
                    'analysis_type': 'sector',
                    'sector_analysis': {sector: strength.value for sector, strength in sector_analysis.items()},
                    'sector_categories': {
                        'leading': leading_sectors,
                        'lagging': lagging_sectors,
                        'defensive': defensive_sectors,
                        'cyclical': cyclical_sectors
                    },
                    'recommendations': {
                        'overweight': leading_sectors + defensive_sectors,
                        'underweight': lagging_sectors,
                        'neutral': cyclical_sectors
                    },
                    'last_update': datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"섹터 로테이션 분석 중 오류: {e}")
                return self._get_default_sector_analysis()
            
        except Exception as e:
            self.logger.error(f"섹터 분석 중 오류: {e}")
            return self._get_default_sector_analysis()
    
    def _get_default_sector_analysis(self) -> Dict[str, Any]:
        """기본 섹터 분석 결과 제공"""
        return {
            'analysis_type': 'sector',
            'sector_analysis': {
                'XLK': 'leading',  # 기술
                'XLF': 'cyclical',  # 금융
                'XLE': 'cyclical',  # 에너지
                'XLV': 'defensive',  # 헬스케어
                'XLI': 'cyclical',  # 산업재
                'XLP': 'defensive',  # 소비재
                'XLU': 'defensive',  # 유틸리티
                'XLB': 'cyclical',  # 소재
                'XLRE': 'cyclical'  # 부동산
            },
            'sector_categories': {
                'leading': ['XLK'],
                'lagging': [],
                'defensive': ['XLV', 'XLP', 'XLU'],
                'cyclical': ['XLF', 'XLE', 'XLI', 'XLB', 'XLRE']
            },
            'recommendations': {
                'overweight': ['XLK', 'XLV', 'XLP', 'XLU'],
                'underweight': [],
                'neutral': ['XLF', 'XLE', 'XLI', 'XLB', 'XLRE']
            },
            'last_update': datetime.now().isoformat(),
            'note': '기본 섹터 분석 (횡보장 환경 기준)'
        }
    
    def run_analysis_by_type(self, analysis_type: str, use_optimized_params: bool = True, use_ml_model: bool = False) -> Dict[str, Any]:
        """분석 유형에 따른 분석 실행"""
        if analysis_type == 'technical':
            return self.get_technical_analysis(use_optimized_params, use_ml_model)
        elif analysis_type == 'macro':
            return self.get_macro_analysis_only()
        elif analysis_type == 'sector':
            return self.get_sector_analysis_only()
        elif analysis_type == 'comprehensive':
            return self.get_current_market_analysis(use_optimized_params)
        elif analysis_type == 'all':
            # 모든 분석 수행 (Quant 기반과 ML 기반 모두)
            results = {
                'technical_quant': self.get_technical_analysis(use_optimized_params, use_ml_model=False),
                'technical_ml': self.get_technical_analysis(use_optimized_params, use_ml_model=True),
                'macro': self.get_macro_analysis_only(),
                'sector': self.get_sector_analysis_only(),
                'comprehensive': self.get_current_market_analysis(use_optimized_params),
                'timestamp': datetime.now().isoformat()
            }
            return results
        else:
            return {'error': f'지원하지 않는 분석 유형: {analysis_type}'}
    
    def recommend_strategy(self, classification: MarketClassification) -> Dict[str, Any]:
        """분류 결과에 따른 전략 추천 (실제 데이터 기반 동적 계산)"""
        
        # 기본 전략 구조 (전략 타입과 설명은 유지)
        base_strategies = {
            MarketRegime.TRENDING_UP: {
                'primary_strategy': 'momentum_following',
                'secondary_strategy': 'buy_hold',
                'position_size': 1.0,
                'rebalance_frequency': 'weekly',
                'max_holding_period': 90,
                'entry_criteria': {
                    'rsi_oversold': 40,
                    'rsi_overbought': 80,
                    'volume_confirmation': True,
                    'breakout_confirmation': True
                },
                'exit_criteria': {
                    'trend_reversal': True,
                    'momentum_divergence': True,
                    'support_break': True
                },
                'reentry_conditions': {
                    'pullback_threshold': 0.03,
                    'bounce_confirmation': True,
                    'volume_surge': True
                }
            },
            
            MarketRegime.TRENDING_DOWN: {
                'primary_strategy': 'cash_heavy',
                'secondary_strategy': 'inverse_momentum',
                'position_size': 0.3,
                'rebalance_frequency': 'daily',
                'max_holding_period': 30,
                'entry_criteria': {
                    'rsi_oversold': 25,
                    'rsi_overbought': 75,
                    'volume_confirmation': True,
                    'support_test': True
                },
                'exit_criteria': {
                    'bounce_reversal': True,
                    'resistance_test': True,
                    'volume_decline': True
                },
                'reentry_conditions': {
                    'bounce_threshold': 0.05,
                    'trend_reversal_confirmation': True,
                    'volume_confirmation': True
                }
            },
            
            MarketRegime.SIDEWAYS: {
                'primary_strategy': 'swing_trading',
                'secondary_strategy': 'mean_reversion',
                'position_size': 0.7,
                'rebalance_frequency': 'bi_weekly',
                'max_holding_period': 45,
                'entry_criteria': {
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'bollinger_band_position': 'lower',
                    'support_resistance_test': True
                },
                'exit_criteria': {
                    'mean_reversion': True,
                    'resistance_hit': True,
                    'momentum_fade': True
                },
                'reentry_conditions': {
                    'range_bound_confirmation': True,
                    'support_bounce': True,
                    'volume_pattern': True
                }
            },
            
            MarketRegime.VOLATILE: {
                'primary_strategy': 'reduced_position',
                'secondary_strategy': 'volatility_breakout',
                'position_size': 0.5,
                'rebalance_frequency': 'daily',
                'max_holding_period': 21,
                'entry_criteria': {
                    'volatility_breakout': True,
                    'volume_surge': True,
                    'momentum_confirmation': True,
                    'risk_reward_ratio': 2.0
                },
                'exit_criteria': {
                    'volatility_contraction': True,
                    'momentum_fade': True,
                    'support_break': True
                },
                'reentry_conditions': {
                    'volatility_stabilization': True,
                    'trend_emergence': True,
                    'risk_mitigation': True
                }
            },
            
            MarketRegime.UNCERTAIN: {
                'primary_strategy': 'wait_and_watch',
                'secondary_strategy': 'minimal_position',
                'position_size': 0.2,
                'rebalance_frequency': 'weekly',
                'max_holding_period': 14,
                'entry_criteria': {
                    'clear_signal': True,
                    'risk_reward_ratio': 3.0,
                    'volume_confirmation': True,
                    'trend_confirmation': True
                },
                'exit_criteria': {
                    'signal_reversal': True,
                    'risk_increase': True,
                    'opportunity_cost': True
                },
                'reentry_conditions': {
                    'clarity_improvement': True,
                    'risk_decrease': True,
                    'opportunity_emergence': True
                }
            }
        }
        
        # 기본 전략 가져오기
        base_strategy = base_strategies[classification.regime].copy()
        
        # 동적 계산을 위한 데이터 준비
        try:
            # SPY 데이터와 매크로 데이터 가져오기
            spy_data = self._get_current_spy_data()
            macro_data = self._get_current_macro_data()
            
            if spy_data is not None and macro_data is not None:
                # 변동성 기반 리스크 파라미터 계산
                volatility_params = self._calculate_volatility_based_parameters(spy_data, macro_data, classification.regime.value)
                base_strategy.update(volatility_params)
                
                # 섹터 배분 계산
                sector_allocation = self._calculate_sector_allocation(spy_data, macro_data, classification.regime.value)
                base_strategy['sector_allocation'] = sector_allocation
                
                # 성과 목표 계산
                performance_targets = self._calculate_performance_targets(spy_data, macro_data, classification.regime.value, classification.confidence)
                base_strategy.update(performance_targets)
                
                # 리스크 관리 파라미터 계산
                risk_management = self._calculate_risk_management(spy_data, macro_data, classification.regime.value, volatility_params)
                base_strategy['risk_management'] = risk_management
                
                # 매크로 강화 정보 추가
                base_strategy['leading_sectors'] = self._get_leading_sectors(sector_allocation)
                base_strategy['defensive_sectors'] = self._get_defensive_sectors(sector_allocation)
                base_strategy['macro_adjustments'] = self._get_macro_adjustments(classification.regime.value, macro_data)
                
                self.logger.info("✅ 동적 전략 계산 완료")
            else:
                self.logger.warning("⚠️ 데이터 부족으로 기본 전략 사용")
                # 기본값으로 fallback
                base_strategy.update(self._get_fallback_strategy_params(classification.regime))
                
        except Exception as e:
            self.logger.error(f"❌ 동적 전략 계산 실패: {e}")
            # 기본값으로 fallback
            base_strategy.update(self._get_fallback_strategy_params(classification.regime))
        
        # 신뢰도에 따른 포지션 크기 조정
        confidence = classification.confidence
        if confidence < 0.6:
            base_strategy['position_size'] *= 0.7
            base_strategy['description'] = f"{classification.regime.value} - 기본 전략 (낮은 신뢰도로 인한 포지션 축소)"
        elif confidence > 0.9:
            base_strategy['position_size'] = min(base_strategy['position_size'] * 1.1, 1.0)
            base_strategy['description'] = f"{classification.regime.value} - 기본 전략 (높은 신뢰도로 인한 포지션 확대)"
        else:
            base_strategy['description'] = f"{classification.regime.value} - 데이터 기반 동적 전략"
        
        # 메타데이터 추가
        base_strategy['metadata'] = {
            'confidence': confidence,
            'regime': classification.regime.value,
            'timestamp': classification.timestamp.isoformat(),
            'features': classification.features,
            'secondary_regime': classification.metadata.get('secondary_regime'),
            'secondary_probability': classification.metadata.get('secondary_probability'),
            'calculation_method': 'dynamic' if 'volatility_target' in base_strategy else 'fallback'
        }
        
        return base_strategy
    
    def _get_current_spy_data(self) -> Optional[pd.DataFrame]:
        """현재 SPY 데이터 가져오기 (저장된 데이터 우선 사용)"""
        try:
            # 먼저 저장된 데이터에서 SPY 데이터 찾기
            spy_data, macro_data, sector_data = self.load_macro_data_only()
            if not spy_data.empty:
                self.logger.info("저장된 SPY 데이터 사용")
                return spy_data
            
            # 저장된 데이터가 없으면 새로 수집
            self.logger.info("저장된 데이터가 없어 새로 수집")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 최근 30일
            
            spy_data = self.macro_collector.collect_spy_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            return spy_data
        except Exception as e:
            self.logger.warning(f"SPY 데이터 수집 실패: {e}")
            return None
    
    def _get_current_macro_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """현재 매크로 데이터 가져오기 (저장된 데이터 우선 사용)"""
        try:
            # 먼저 저장된 데이터에서 매크로 데이터 찾기
            spy_data, macro_data, sector_data = self.load_macro_data_only()
            if macro_data and sector_data:
                self.logger.info("저장된 매크로 데이터 사용")
                # 통합
                all_data = {}
                all_data.update(macro_data)
                all_data.update(sector_data)
                return all_data
            
            # 저장된 데이터가 없으면 새로 수집
            self.logger.info("저장된 데이터가 없어 새로 수집")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 최근 30일
            
            # 매크로 지표 수집
            macro_data = self.macro_collector.collect_macro_indicators(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            # 섹터 데이터 수집
            sector_data = self.macro_collector.collect_sector_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # 통합
            all_data = {}
            if macro_data:
                all_data.update(macro_data)
            if sector_data:
                all_data.update(sector_data)
            
            return all_data if all_data else None
        except Exception as e:
            self.logger.warning(f"매크로 데이터 수집 실패: {e}")
            return None
    
    def _get_leading_sectors(self, sector_allocation: Dict[str, float]) -> List[str]:
        """선도 섹터 식별"""
        # 가중치가 높은 섹터들을 선도 섹터로 간주
        sorted_sectors = sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True)
        return [sector for sector, weight in sorted_sectors[:3] if weight > 0.1 and sector != 'cash']
    
    def _get_defensive_sectors(self, sector_allocation: Dict[str, float]) -> List[str]:
        """방어적 섹터 식별"""
        defensive_sectors = ['utilities', 'consumer_staples', 'healthcare']
        return [sector for sector in defensive_sectors if sector in sector_allocation and sector_allocation[sector] > 0.1]
    
    def _get_macro_adjustments(self, regime: str, macro_data: Dict[str, pd.DataFrame]) -> str:
        """매크로 조정사항 생성"""
        adjustments = []
        
        try:
            # VIX 기반 조정
            if 'VIX' in macro_data:
                vix = macro_data['VIX'].iloc[-1]['close'] if 'close' in macro_data['VIX'].columns else macro_data['VIX'].iloc[-1]['Close']
                if vix > 25:
                    adjustments.append("높은 변동성으로 인한 포지션 축소")
                elif vix < 15:
                    adjustments.append("낮은 변동성으로 인한 포지션 확대")
            
            # 금리 환경 기반 조정
            if 'TNX' in macro_data:
                tnx = macro_data['TNX'].iloc[-1]['close'] if 'close' in macro_data['TNX'].columns else macro_data['TNX'].iloc[-1]['Close']
                if tnx > 4.0:
                    adjustments.append("높은 금리 환경으로 인한 금융 섹터 조정")
                elif tnx < 2.0:
                    adjustments.append("낮은 금리 환경으로 인한 성장 섹터 강화")
            
            # 달러 강도 기반 조정
            if 'UUP' in macro_data:
                uup = macro_data['UUP'].iloc[-1]['close'] if 'close' in macro_data['UUP'].columns else macro_data['UUP'].iloc[-1]['Close']
                if uup > 28:
                    adjustments.append("강한 달러로 인한 수출 섹터 조정")
                elif uup < 26:
                    adjustments.append("약한 달러로 인한 수입 섹터 강화")
                    
        except Exception as e:
            self.logger.warning(f"매크로 조정사항 계산 실패: {e}")
        
        return "; ".join(adjustments) if adjustments else "현재 시장 환경에 최적화된 배분"
    
    def _get_fallback_strategy_params(self, regime: MarketRegime) -> Dict[str, Any]:
        """기본 전략 파라미터 (fallback)"""
        fallback_params = {
            MarketRegime.TRENDING_UP: {
                'stop_loss': 0.05, 'take_profit': 0.15, 'trailing_stop': 0.03, 'max_drawdown_limit': 0.08,
                'sector_allocation': {'technology': 0.25, 'financials': 0.20, 'consumer_discretionary': 0.20, 'industrials': 0.15, 'healthcare': 0.10, 'energy': 0.10},
                'risk_management': {'max_single_position': 0.15, 'correlation_threshold': 0.7, 'volatility_target': 0.12, 'beta_target': 1.1},
                'expected_return': 0.12, 'expected_volatility': 0.15, 'sharpe_ratio_target': 0.8
            },
            MarketRegime.TRENDING_DOWN: {
                'stop_loss': 0.03, 'take_profit': 0.08, 'trailing_stop': 0.02, 'max_drawdown_limit': 0.05,
                'sector_allocation': {'utilities': 0.30, 'consumer_staples': 0.25, 'healthcare': 0.20, 'real_estate': 0.15, 'cash': 0.10},
                'risk_management': {'max_single_position': 0.10, 'correlation_threshold': 0.5, 'volatility_target': 0.08, 'beta_target': 0.6},
                'expected_return': 0.04, 'expected_volatility': 0.10, 'sharpe_ratio_target': 0.4
            },
            MarketRegime.SIDEWAYS: {
                'stop_loss': 0.04, 'take_profit': 0.10, 'trailing_stop': 0.025, 'max_drawdown_limit': 0.06,
                'sector_allocation': {'technology': 0.20, 'financials': 0.15, 'consumer_discretionary': 0.15, 'healthcare': 0.15, 'industrials': 0.15, 'utilities': 0.10, 'cash': 0.10},
                'risk_management': {'max_single_position': 0.12, 'correlation_threshold': 0.6, 'volatility_target': 0.10, 'beta_target': 0.9},
                'expected_return': 0.08, 'expected_volatility': 0.12, 'sharpe_ratio_target': 0.7
            },
            MarketRegime.VOLATILE: {
                'stop_loss': 0.06, 'take_profit': 0.12, 'trailing_stop': 0.04, 'max_drawdown_limit': 0.10,
                'sector_allocation': {'utilities': 0.25, 'consumer_staples': 0.20, 'healthcare': 0.20, 'real_estate': 0.15, 'cash': 0.20},
                'risk_management': {'max_single_position': 0.08, 'correlation_threshold': 0.4, 'volatility_target': 0.15, 'beta_target': 0.7},
                'expected_return': 0.06, 'expected_volatility': 0.20, 'sharpe_ratio_target': 0.3
            },
            MarketRegime.UNCERTAIN: {
                'stop_loss': 0.02, 'take_profit': 0.05, 'trailing_stop': 0.015, 'max_drawdown_limit': 0.03,
                'sector_allocation': {'utilities': 0.40, 'consumer_staples': 0.30, 'cash': 0.30},
                'risk_management': {'max_single_position': 0.05, 'correlation_threshold': 0.3, 'volatility_target': 0.06, 'beta_target': 0.4},
                'expected_return': 0.02, 'expected_volatility': 0.08, 'sharpe_ratio_target': 0.25
            }
        }
        return fallback_params.get(regime, fallback_params[MarketRegime.SIDEWAYS])
    
    def _calculate_volatility_based_parameters(self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame], regime: str) -> Dict[str, float]:
        """변동성 기반 리스크 파라미터 계산"""
        try:
            # VIX 데이터 추출
            vix_data = macro_data.get('VIX', pd.DataFrame())
            if vix_data.empty:
                return self._get_base_risk_params(regime)
            
            current_vix = vix_data.iloc[-1]['close'] if 'close' in vix_data.columns else vix_data.iloc[-1]['Close']
            
            # 변동성 상태 분류
            if current_vix < 15:
                vol_state = 'low'
            elif current_vix < 25:
                vol_state = 'normal'
            elif current_vix < 35:
                vol_state = 'high'
            else:
                vol_state = 'extreme'
            
            # 기본 파라미터 (시장 상태별)
            base_params = self._get_base_risk_params(regime)
            
            # 변동성 기반 조정
            vol_adjustments = {
                'low': {'stop_loss': 0.8, 'take_profit': 1.2, 'trailing_stop': 0.8},
                'normal': {'stop_loss': 1.0, 'take_profit': 1.0, 'trailing_stop': 1.0},
                'high': {'stop_loss': 1.3, 'take_profit': 0.8, 'trailing_stop': 1.2},
                'extreme': {'stop_loss': 1.5, 'take_profit': 0.7, 'trailing_stop': 1.4}
            }
            
            adjustment = vol_adjustments[vol_state]
            
            return {
                'stop_loss': base_params['stop_loss'] * adjustment['stop_loss'],
                'take_profit': base_params['take_profit'] * adjustment['take_profit'],
                'trailing_stop': base_params['trailing_stop'] * adjustment['trailing_stop'],
                'max_drawdown_limit': base_params['max_drawdown_limit'] * (1 + (current_vix - 20) / 100),
                'volatility_target': current_vix / 100,
                'expected_volatility': current_vix / 100
            }
            
        except Exception as e:
            self.logger.warning(f"변동성 기반 파라미터 계산 실패: {e}")
            return self._get_base_risk_params(regime)
    
    def _calculate_sector_allocation(self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame], regime: str) -> Dict[str, float]:
        """섹터별 모멘텀과 상대 강도 기반 섹터 배분 계산"""
        try:
            sector_etfs = {
                'technology': 'XLK', 'financials': 'XLF', 'healthcare': 'XLV',
                'consumer_discretionary': 'XLY', 'consumer_staples': 'XLP',
                'industrials': 'XLI', 'energy': 'XLE', 'utilities': 'XLU',
                'real_estate': 'XLRE', 'materials': 'XLB'
            }
            
            sector_scores = {}
            total_score = 0
            
            for sector_name, etf_symbol in sector_etfs.items():
                if etf_symbol in macro_data:
                    etf_data = macro_data[etf_symbol]
                    if not etf_data.empty:
                        # RSI 계산
                        rsi = self._calculate_rsi(etf_data['close'] if 'close' in etf_data.columns else etf_data['Close'])
                        
                        # 모멘텀 계산 (20일 수익률)
                        returns = etf_data['close'].pct_change(20) if 'close' in etf_data.columns else etf_data['Close'].pct_change(20)
                        momentum = returns.iloc[-1] if not pd.isna(returns.iloc[-1]) else 0
                        
                        # 상대 강도 계산 (SPY 대비)
                        spy_returns = spy_data['close'].pct_change(20) if 'close' in spy_data.columns else spy_data['Close'].pct_change(20)
                        relative_strength = momentum - spy_returns.iloc[-1] if not pd.isna(spy_returns.iloc[-1]) else 0
                        
                        # 종합 점수 계산
                        score = (rsi * 0.4 + momentum * 100 * 0.3 + relative_strength * 100 * 0.3)
                        sector_scores[sector_name] = max(score, 0)  # 음수 점수는 0으로
                        total_score += sector_scores[sector_name]
            
            # 시장 상태별 가중치 조정
            regime_weights = self._get_regime_sector_weights(regime)
            
            # 최종 섹터 배분 계산
            sector_allocation = {}
            if total_score > 0:
                for sector_name, score in sector_scores.items():
                    base_weight = score / total_score
                    regime_weight = regime_weights.get(sector_name, 1.0)
                    sector_allocation[sector_name] = base_weight * regime_weight
            else:
                # 기본 배분 사용
                sector_allocation = self._get_default_sector_allocation(regime)
            
            # 현금 비중 계산 (변동성 기반)
            vix_data = macro_data.get('VIX', pd.DataFrame())
            if not vix_data.empty:
                current_vix = vix_data.iloc[-1]['close'] if 'close' in vix_data.columns else vix_data.iloc[-1]['Close']
                cash_weight = min(0.3, max(0.05, (current_vix - 15) / 50))
                sector_allocation['cash'] = cash_weight
                
                # 다른 섹터 비중 조정
                total_sector_weight = 1 - cash_weight
                sector_sum = sum(w for k, w in sector_allocation.items() if k != 'cash')
                if sector_sum > 0:
                    for sector in sector_allocation:
                        if sector != 'cash':
                            sector_allocation[sector] *= total_sector_weight / sector_sum
            
            return sector_allocation
            
        except Exception as e:
            self.logger.warning(f"섹터 배분 계산 실패: {e}")
            return self._get_default_sector_allocation(regime)
    
    def _calculate_performance_targets(self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame], regime: str, confidence: float) -> Dict[str, float]:
        """성과 목표 계산 (과거 데이터 기반)"""
        try:
            # 과거 수익률 분석
            spy_returns = spy_data['close'].pct_change() if 'close' in spy_data.columns else spy_data['Close'].pct_change()
            
            # 시장 상태별 과거 성과 분석
            regime_performance = self._analyze_regime_performance(regime, spy_returns)
            
            # 현재 변동성
            vix_data = macro_data.get('VIX', pd.DataFrame())
            current_vol = vix_data.iloc[-1]['close'] / 100 if not vix_data.empty else 0.15
            
            # 신뢰도 기반 조정
            confidence_adjustment = 0.8 + (confidence * 0.4)  # 0.8 ~ 1.2
            
            return {
                'expected_return': regime_performance['avg_return'] * confidence_adjustment,
                'expected_volatility': current_vol,
                'sharpe_ratio_target': regime_performance['avg_sharpe'] * confidence_adjustment
            }
            
        except Exception as e:
            self.logger.warning(f"성과 목표 계산 실패: {e}")
            return self._get_default_performance_targets(regime)
    
    def _calculate_risk_management(self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame], regime: str, volatility_params: Dict[str, float]) -> Dict[str, float]:
        """리스크 관리 파라미터 계산"""
        try:
            # 포트폴리오 VaR 계산
            spy_returns = spy_data['close'].pct_change() if 'close' in spy_data.columns else spy_data['Close'].pct_change()
            var_95 = np.percentile(spy_returns.dropna(), 5)
            
            # 섹터간 상관관계 계산
            sector_correlations = self._calculate_sector_correlations(macro_data)
            avg_correlation = np.mean([abs(corr) for corr in sector_correlations.values() if not pd.isna(corr)]) if sector_correlations else 0.6
            
            # 베타 계산 (SPY 대비)
            regime_betas = self._calculate_regime_betas(regime, spy_returns)
            
            return {
                'max_single_position': min(0.15, abs(var_95) * 2),
                'correlation_threshold': max(0.3, min(0.8, avg_correlation)),
                'volatility_target': volatility_params.get('volatility_target', 0.15),
                'beta_target': regime_betas.get('target_beta', 0.9)
            }
            
        except Exception as e:
            self.logger.warning(f"리스크 관리 파라미터 계산 실패: {e}")
            return self._get_default_risk_management(regime)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _get_base_risk_params(self, regime: str) -> Dict[str, float]:
        """시장 상태별 기본 리스크 파라미터"""
        base_params = {
            'TRENDING_UP': {'stop_loss': 0.05, 'take_profit': 0.15, 'trailing_stop': 0.03, 'max_drawdown_limit': 0.08},
            'TRENDING_DOWN': {'stop_loss': 0.03, 'take_profit': 0.08, 'trailing_stop': 0.02, 'max_drawdown_limit': 0.05},
            'SIDEWAYS': {'stop_loss': 0.04, 'take_profit': 0.10, 'trailing_stop': 0.025, 'max_drawdown_limit': 0.06},
            'VOLATILE': {'stop_loss': 0.06, 'take_profit': 0.12, 'trailing_stop': 0.04, 'max_drawdown_limit': 0.10},
            'UNCERTAIN': {'stop_loss': 0.02, 'take_profit': 0.05, 'trailing_stop': 0.015, 'max_drawdown_limit': 0.03}
        }
        return base_params.get(regime, base_params['SIDEWAYS'])
    
    def _get_regime_sector_weights(self, regime: str) -> Dict[str, float]:
        """시장 상태별 섹터 가중치"""
        weights = {
            'TRENDING_UP': {
                'technology': 1.3, 'financials': 1.2, 'consumer_discretionary': 1.2,
                'industrials': 1.1, 'healthcare': 0.9, 'energy': 1.1,
                'utilities': 0.7, 'consumer_staples': 0.8, 'real_estate': 0.9, 'materials': 1.0
            },
            'TRENDING_DOWN': {
                'technology': 0.7, 'financials': 0.6, 'consumer_discretionary': 0.6,
                'industrials': 0.7, 'healthcare': 1.2, 'energy': 0.5,
                'utilities': 1.4, 'consumer_staples': 1.3, 'real_estate': 1.1, 'materials': 0.6
            },
            'SIDEWAYS': {
                'technology': 1.0, 'financials': 0.9, 'consumer_discretionary': 0.9,
                'industrials': 0.9, 'healthcare': 1.1, 'energy': 0.8,
                'utilities': 1.1, 'consumer_staples': 1.1, 'real_estate': 0.8, 'materials': 0.8
            },
            'VOLATILE': {
                'technology': 0.8, 'financials': 0.7, 'consumer_discretionary': 0.7,
                'industrials': 0.7, 'healthcare': 1.2, 'energy': 0.6,
                'utilities': 1.3, 'consumer_staples': 1.2, 'real_estate': 1.0, 'materials': 0.6
            },
            'UNCERTAIN': {
                'technology': 0.6, 'financials': 0.5, 'consumer_discretionary': 0.5,
                'industrials': 0.5, 'healthcare': 1.1, 'energy': 0.4,
                'utilities': 1.5, 'consumer_staples': 1.4, 'real_estate': 1.2, 'materials': 0.4
            }
        }
        return weights.get(regime, weights['SIDEWAYS'])
    
    def _get_default_sector_allocation(self, regime: str) -> Dict[str, float]:
        """기본 섹터 배분 (fallback)"""
        allocations = {
            'TRENDING_UP': {
                'technology': 0.25, 'financials': 0.20, 'consumer_discretionary': 0.20,
                'industrials': 0.15, 'healthcare': 0.10, 'energy': 0.10
            },
            'TRENDING_DOWN': {
                'utilities': 0.30, 'consumer_staples': 0.25, 'healthcare': 0.20,
                'real_estate': 0.15, 'cash': 0.10
            },
            'SIDEWAYS': {
                'technology': 0.20, 'financials': 0.15, 'consumer_discretionary': 0.15,
                'healthcare': 0.15, 'industrials': 0.15, 'utilities': 0.10, 'cash': 0.10
            },
            'VOLATILE': {
                'utilities': 0.25, 'consumer_staples': 0.20, 'healthcare': 0.20,
                'real_estate': 0.15, 'cash': 0.20
            },
            'UNCERTAIN': {
                'utilities': 0.40, 'consumer_staples': 0.30, 'cash': 0.30
            }
        }
        return allocations.get(regime, allocations['SIDEWAYS'])
    
    def _analyze_regime_performance(self, regime: str, returns: pd.Series) -> Dict[str, float]:
        """시장 상태별 과거 성과 분석"""
        # 간단한 시뮬레이션 (실제로는 더 정교한 분석 필요)
        performance_map = {
            'TRENDING_UP': {'avg_return': 0.12, 'avg_sharpe': 0.8},
            'TRENDING_DOWN': {'avg_return': 0.04, 'avg_sharpe': 0.4},
            'SIDEWAYS': {'avg_return': 0.08, 'avg_sharpe': 0.7},
            'VOLATILE': {'avg_return': 0.06, 'avg_sharpe': 0.3},
            'UNCERTAIN': {'avg_return': 0.02, 'avg_sharpe': 0.25}
        }
        return performance_map.get(regime, performance_map['SIDEWAYS'])
    
    def _get_default_performance_targets(self, regime: str) -> Dict[str, float]:
        """기본 성과 목표 (fallback)"""
        targets = {
            'TRENDING_UP': {'expected_return': 0.12, 'expected_volatility': 0.15, 'sharpe_ratio_target': 0.8},
            'TRENDING_DOWN': {'expected_return': 0.04, 'expected_volatility': 0.10, 'sharpe_ratio_target': 0.4},
            'SIDEWAYS': {'expected_return': 0.08, 'expected_volatility': 0.12, 'sharpe_ratio_target': 0.7},
            'VOLATILE': {'expected_return': 0.06, 'expected_volatility': 0.20, 'sharpe_ratio_target': 0.3},
            'UNCERTAIN': {'expected_return': 0.02, 'expected_volatility': 0.08, 'sharpe_ratio_target': 0.25}
        }
        return targets.get(regime, targets['SIDEWAYS'])
    
    def _calculate_sector_correlations(self, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """섹터간 상관관계 계산"""
        correlations = {}
        sector_etfs = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLI', 'XLE', 'XLU', 'XLRE', 'XLB']
        
        for i, etf1 in enumerate(sector_etfs):
            for j, etf2 in enumerate(sector_etfs[i+1:], i+1):
                if etf1 in macro_data and etf2 in macro_data:
                    data1 = macro_data[etf1]['close'] if 'close' in macro_data[etf1].columns else macro_data[etf1]['Close']
                    data2 = macro_data[etf2]['close'] if 'close' in macro_data[etf2].columns else macro_data[etf2]['Close']
                    
                    if len(data1) > 20 and len(data2) > 20:
                        corr = data1.pct_change().corr(data2.pct_change())
                        correlations[f"{etf1}_{etf2}"] = corr
        
        return correlations
    
    def _calculate_regime_betas(self, regime: str, spy_returns: pd.Series) -> Dict[str, float]:
        """시장 상태별 베타 계산"""
        betas = {
            'TRENDING_UP': {'target_beta': 1.1},
            'TRENDING_DOWN': {'target_beta': 0.6},
            'SIDEWAYS': {'target_beta': 0.9},
            'VOLATILE': {'target_beta': 0.7},
            'UNCERTAIN': {'target_beta': 0.4}
        }
        return betas.get(regime, betas['SIDEWAYS'])
    
    def _get_default_risk_management(self, regime: str) -> Dict[str, float]:
        """기본 리스크 관리 파라미터 (fallback)"""
        risk_params = {
            'TRENDING_UP': {'max_single_position': 0.15, 'correlation_threshold': 0.7, 'volatility_target': 0.12, 'beta_target': 1.1},
            'TRENDING_DOWN': {'max_single_position': 0.10, 'correlation_threshold': 0.5, 'volatility_target': 0.08, 'beta_target': 0.6},
            'SIDEWAYS': {'max_single_position': 0.12, 'correlation_threshold': 0.6, 'volatility_target': 0.10, 'beta_target': 0.9},
            'VOLATILE': {'max_single_position': 0.08, 'correlation_threshold': 0.4, 'volatility_target': 0.15, 'beta_target': 0.7},
            'UNCERTAIN': {'max_single_position': 0.05, 'correlation_threshold': 0.3, 'volatility_target': 0.06, 'beta_target': 0.4}
        }
        return risk_params.get(regime, risk_params['SIDEWAYS'])
    
    def generate_transaction_log(self, results: Dict[str, Any], output_dir: str = "log/market_sensor") -> str:
        """거래 로그 생성 (swing 로그와 유사한 형식)"""
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 로그 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"transaction_market_sensor_{timestamp}.log"
            log_path = os.path.join(output_dir, log_filename)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("=== MARKET SENSOR 거래 내역 로그 ===\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"실행 UUID: {self.session_uuid}\n")
                f.write("=" * 80 + "\n\n")
                
                # SPY 거래 내역
                f.write("📊 SPY (market_sensor)\n")
                f.write("-" * 50 + "\n")
                
                if 'test_performance' in results and results['test_performance']:
                    test_metrics = results['test_performance']
                    
                    # 거래 통계
                    total_return = test_metrics.get('total_return', 0)
                    sharpe_ratio = test_metrics.get('sharpe_ratio', 0)
                    max_drawdown = test_metrics.get('max_drawdown', 0)
                    win_rate = test_metrics.get('win_rate', 0)
                    
                    f.write(f"총 거래 수: {test_metrics.get('total_trades', 0)}\n")
                    f.write(f"수익률: {total_return:.2%}\n")
                    f.write(f"샤프 비율: {sharpe_ratio:.3f}\n")
                    f.write(f"최대 낙폭: {max_drawdown:.2%}\n")
                    f.write(f"승률: {win_rate:.2%}\n\n")
                    
                    # 거래 내역 (시뮬레이션된 거래)
                    f.write("거래 내역:\n")
                    f.write("날짜                   시간         타입     가격         수량       수익률        누적수익률       \n")
                    f.write("-" * 80 + "\n")
                    
                    # 시뮬레이션된 거래 내역 생성
                    if 'strategy_returns' in results:
                        strategy_returns = results['strategy_returns']
                        if isinstance(strategy_returns, pd.Series):
                            # 거래 시뮬레이션
                            trades = self._simulate_trades_from_returns(strategy_returns)
                            
                            for i, trade in enumerate(trades):
                                date_str = trade['date'].strftime('%Y-%m-%d')
                                time_str = trade['time'] if 'time' in trade else ''
                                trade_type = trade['type']
                                price = trade['price']
                                quantity = trade['quantity']
                                profit_rate = trade.get('profit_rate', '')
                                cumulative_profit = trade.get('cumulative_profit', '')
                                
                                f.write(f"{date_str:<20} {time_str:<12} {trade_type:<10} {price:<10.2f} {quantity:<10.2f} {profit_rate:<10} {cumulative_profit:<10}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("📋 거래 요약\n")
                f.write("-" * 50 + "\n")
                
                if 'best_params' in results:
                    best_params = results['best_params']
                    f.write("최적 파라미터:\n")
                    for param, value in best_params.items():
                        if isinstance(value, float):
                            f.write(f"  {param}: {value:.4f}\n")
                        else:
                            f.write(f"  {param}: {value}\n")
                
                f.write(f"\n실험 ID: {self.session_uuid}\n")
                f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            return log_path
            
        except Exception as e:
            self.logger.error(f"거래 로그 생성 중 오류: {e}")
            return ""
    
    def _simulate_trades_from_returns(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """수익률 시리즈에서 거래 내역 시뮬레이션"""
        trades = []
        cumulative_return = 0
        position = 0
        entry_price = 100  # 초기 가격
        
        for date, daily_return in returns.items():
            if abs(daily_return) > 0.01:  # 1% 이상 변동 시 거래로 간주
                if position == 0 and daily_return > 0:
                    # 매수
                    position = 1
                    entry_price = 100 * (1 + cumulative_return)
                    trades.append({
                        'date': date,
                        'type': '매수',
                        'price': entry_price,
                        'quantity': 1.0,
                        'profit_rate': '',
                        'cumulative_profit': ''
                    })
                elif position == 1 and daily_return < -0.005:
                    # 매도
                    position = 0
                    exit_price = entry_price * (1 + daily_return)
                    profit_rate = (exit_price - entry_price) / entry_price
                    cumulative_return += profit_rate
                    
                    trades.append({
                        'date': date,
                        'type': '매도',
                        'price': exit_price,
                        'quantity': 1.0,
                        'profit_rate': f"{profit_rate:.2f} %",
                        'cumulative_profit': f"{cumulative_return:.2f} %"
                    })
        
        return trades
    
    def run_comprehensive_validation(self, start_date: str, end_date: str, 
                                   use_optimized_params: bool = True) -> Dict[str, Any]:
        """종합 검증 실행 (분류 정확도 + 전략 성과)"""
        try:
            self.logger.info(f"종합 검증 시작: {start_date} ~ {end_date}")
            
            # 1. 데이터 수집
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if spy_data.empty:
                return {'error': 'SPY 데이터를 로드할 수 없습니다.'}
            
            # 날짜 범위 필터링
            spy_data = spy_data[start_date:end_date]
            if spy_data.empty:
                return {'error': '지정된 날짜 범위에 데이터가 없습니다.'}
            
            # 2. 파라미터 설정
            if use_optimized_params and self.optimal_params:
                params = self.optimal_params
            else:
                params = {
                    'spy_trend_weight': 0.45,  # SPY 추세+수익률 가중치 강화
                    'momentum_weight': 0.25,
                    'volatility_weight': 0.15,
                    'macro_weight': 0.10,
                    'volume_weight': 0.03,
                    'support_resistance_weight': 0.02,
                    'sma_short': 20, 'sma_long': 50, 'rsi_period': 14,
                    'rsi_overbought': 70, 'rsi_oversold': 30, 'atr_period': 14,
                    'vix_high_threshold': 30, 'vix_medium_threshold': 20, 'vix_low_threshold': 15,
                    'tips_high_threshold': 2.5, 'tips_medium_threshold': 2.0, 'tips_low_threshold': 1.5
                }
            
            # 3. 파생 변수 계산
            data_with_features = self.hyperparam_tuner._calculate_derived_features(spy_data, params)
            
            # 4. 매크로 데이터 병합
            if '^VIX' in macro_data:
                vix_df = macro_data['^VIX']
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 5. 시장 상태 분류
            regime = self.hyperparam_tuner._classify_market_regime(data_with_features, params)
            
            # 5.5. Random Forest 모델 학습 및 저장 (새로운 기능)
            try:
                self.logger.info("Random Forest 모델 학습 시작...")
                
                # 학습 데이터 준비
                training_data = self.rf_model.collect_training_data(
                    start_date=start_date, 
                    end_date=end_date, 
                    data_dir=self.data_dir
                )
                
                if not training_data.empty and len(training_data) > 100:
                    # 모델 학습
                    self.rf_model.train_model(training_data)
                    
                    # 모델 저장
                    metadata = {
                        'training_period': f"{start_date} ~ {end_date}",
                        'training_samples': len(training_data),
                        'validation_period': f"{start_date} ~ {end_date}",
                        'model_type': 'Random Forest',
                        'features': list(training_data.columns) if hasattr(training_data, 'columns') else []
                    }
                    
                    model_path = self.save_trained_model(
                        self.rf_model.model, 
                        model_name="market_regime_rf",
                        metadata=metadata
                    )
                    
                    self.logger.info(f"Random Forest 모델 학습 및 저장 완료: {model_path}")
                else:
                    self.logger.warning("학습 데이터가 부족하여 RF 모델 학습을 건너뜁니다.")
                    
            except Exception as e:
                self.logger.error(f"Random Forest 모델 학습 중 오류: {e}")
            
            # 6. 실제 시장 상태 생성 (미래 수익률 기반)
            actual_regime = self._create_actual_market_regime(spy_data)
            
            # 7. 분류 정확도 검증
            validation_results = self.validator.validate_classification_accuracy(actual_regime, regime)
            
            # 8. 전략 수익률 계산
            strategy_returns = self.hyperparam_tuner._calculate_strategy_returns(data_with_features, regime, params)
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            benchmark_returns = spy_data[close_col].pct_change()
            
            # 9. 전략 성과 분석
            performance_results = self.validator.analyze_strategy_performance(
                strategy_returns, benchmark_returns, regime
            )
            
            # 10. 결과 저장
            results_file = self.validator.save_validation_results(
                validation_results, performance_results
            )
            
            # 11. 종합 보고서 생성
            report = self.validator.generate_validation_report(validation_results, performance_results)
            
            return {
                'validation_results': validation_results,
                'performance_results': performance_results,
                'results_file': results_file,
                'report': report,
                'data_summary': {
                    'total_days': len(spy_data),
                    'start_date': start_date,
                    'end_date': end_date,
                    'regime_distribution': regime.value_counts().to_dict(),
                    'actual_regime_distribution': actual_regime.value_counts().to_dict()
                }
            }
            
        except Exception as e:
            self.logger.error(f"종합 검증 중 오류: {e}")
            return {'error': str(e)}
    
    def _create_actual_market_regime(self, spy_data: pd.DataFrame) -> pd.Series:
        """실제 시장 상태 생성 (미래 수익률 기반)"""
        close_col = 'close' if 'close' in spy_data.columns else 'Close'
        
        # 미래 수익률 계산 (5일 후)
        future_returns = spy_data[close_col].pct_change(5).shift(-5)
        
        # 시장 상태 분류
        actual_regime = pd.Series(index=spy_data.index, dtype='object')
        
        for i in range(len(spy_data)):
            if i < len(spy_data) - 5 and not pd.isna(future_returns.iloc[i]):
                future_return = future_returns.iloc[i]
                
                if future_return > 0.02:  # 2% 이상 상승
                    actual_regime.iloc[i] = MarketRegime.TRENDING_UP.value
                elif future_return < -0.02:  # 2% 이상 하락
                    actual_regime.iloc[i] = MarketRegime.TRENDING_DOWN.value
                elif abs(future_return) > 0.01:  # 1% 이상 변동
                    actual_regime.iloc[i] = MarketRegime.VOLATILE.value
                else:
                    actual_regime.iloc[i] = MarketRegime.SIDEWAYS.value
            else:
                actual_regime.iloc[i] = MarketRegime.UNCERTAIN.value
        
        return actual_regime
    
    def run_backtest_validation(self, start_date: str, end_date: str, 
                              test_periods: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """백테스팅 검증 실행 (여러 기간에 대한 검증)"""
        try:
            if test_periods is None:
                # 기본 테스트 기간 설정
                test_periods = [
                    ('2020-01-01', '2020-12-31'),  # 2020년 (코로나)
                    ('2021-01-01', '2021-12-31'),  # 2021년 (회복)
                    ('2022-01-01', '2022-12-31'),  # 2022년 (인플레이션)
                    ('2023-01-01', '2023-12-31'),  # 2023년 (금리 상승)
                ]
            
            backtest_results = {}
            
            for period_name, (period_start, period_end) in enumerate(test_periods):
                self.logger.info(f"백테스트 기간 {period_name + 1}: {period_start} ~ {period_end}")
                
                # 각 기간별 검증 실행
                period_result = self.run_comprehensive_validation(period_start, period_end)
                
                if 'error' not in period_result:
                    backtest_results[f'period_{period_name + 1}'] = {
                        'start_date': period_start,
                        'end_date': period_end,
                        'validation_results': period_result['validation_results'],
                        'performance_results': period_result['performance_results'],
                        'data_summary': period_result['data_summary']
                    }
                else:
                    backtest_results[f'period_{period_name + 1}'] = {
                        'error': period_result['error']
                    }
            
            # 종합 백테스트 분석
            overall_analysis = self._analyze_backtest_results(backtest_results)
            
            return {
                'backtest_results': backtest_results,
                'overall_analysis': overall_analysis,
                'test_periods': test_periods
            }
            
        except Exception as e:
            self.logger.error(f"백테스트 검증 중 오류: {e}")
            return {'error': str(e)}
    
    def _analyze_backtest_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """백테스트 결과 종합 분석"""
        try:
            # 성공한 기간들만 필터링
            successful_periods = {k: v for k, v in backtest_results.items() 
                               if 'error' not in v}
            
            if not successful_periods:
                return {'error': '성공한 백테스트 기간이 없습니다.'}
            
            # 전체 성과 지표 평균 계산
            total_returns = []
            excess_returns = []
            sharpe_ratios = []
            accuracies = []
            
            for period_data in successful_periods.values():
                if 'performance_results' in period_data:
                    perf = period_data['performance_results']['overall_performance']
                    total_returns.append(perf['total_return'])
                    excess_returns.append(perf['excess_return'])
                    sharpe_ratios.append(perf['sharpe_ratio'])
                
                if 'validation_results' in period_data:
                    accuracies.append(period_data['validation_results']['overall_accuracy'])
            
            # 통계 계산
            overall_analysis = {
                'average_total_return': np.mean(total_returns),
                'average_excess_return': np.mean(excess_returns),
                'average_sharpe_ratio': np.mean(sharpe_ratios),
                'average_accuracy': np.mean(accuracies),
                'std_total_return': np.std(total_returns),
                'std_excess_return': np.std(excess_returns),
                'std_sharpe_ratio': np.std(sharpe_ratios),
                'std_accuracy': np.std(accuracies),
                'min_total_return': np.min(total_returns),
                'max_total_return': np.max(total_returns),
                'min_excess_return': np.min(excess_returns),
                'max_excess_return': np.max(excess_returns),
                'successful_periods': len(successful_periods),
                'total_periods': len(backtest_results)
            }
            
            # 시장 상태별 성과 분석
            regime_performance_summary = {}
            for period_data in successful_periods.values():
                if 'performance_results' in period_data:
                    regime_perf = period_data['performance_results']['regime_performance']
                    for regime, perf in regime_perf.items():
                        if regime not in regime_performance_summary:
                            regime_performance_summary[regime] = {
                                'total_returns': [],
                                'excess_returns': [],
                                'sharpe_ratios': [],
                                'win_rates': []
                            }
                        
                        regime_performance_summary[regime]['total_returns'].append(perf['total_return'])
                        regime_performance_summary[regime]['excess_returns'].append(perf['excess_return'])
                        regime_performance_summary[regime]['sharpe_ratios'].append(perf['sharpe_ratio'])
                        regime_performance_summary[regime]['win_rates'].append(perf['win_rate'])
            
            # 시장 상태별 평균 성과 계산
            for regime, data in regime_performance_summary.items():
                regime_performance_summary[regime] = {
                    'avg_total_return': np.mean(data['total_returns']),
                    'avg_excess_return': np.mean(data['excess_returns']),
                    'avg_sharpe_ratio': np.mean(data['sharpe_ratios']),
                    'avg_win_rate': np.mean(data['win_rates']),
                    'std_total_return': np.std(data['total_returns']),
                    'std_excess_return': np.std(data['excess_returns']),
                    'std_sharpe_ratio': np.std(data['sharpe_ratios']),
                    'std_win_rate': np.std(data['win_rates'])
                }
            
            overall_analysis['regime_performance_summary'] = regime_performance_summary
            
            return overall_analysis
            
        except Exception as e:
            self.logger.error(f"백테스트 결과 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def generate_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """검증 결과 요약 생성"""
        try:
            summary = []
            summary.append("📊 검증 결과 요약")
            summary.append("=" * 50)
            
            if 'overall_accuracy' in validation_results:
                acc = validation_results['overall_accuracy']
                summary.append(f"🎯 분류 정확도: {acc:.3f} ({acc*100:.1f}%)")
                
                if acc > 0.8:
                    summary.append("   ✅ 매우 우수한 분류 성능")
                elif acc > 0.7:
                    summary.append("   ✅ 양호한 분류 성능")
                elif acc > 0.6:
                    summary.append("   ⚠️ 보통 수준의 분류 성능")
                else:
                    summary.append("   ❌ 개선이 필요한 분류 성능")
            
            if 'regime_accuracy' in validation_results:
                summary.append("\n📈 시장 상태별 정확도:")
                for regime, accuracy in validation_results['regime_accuracy'].items():
                    summary.append(f"   {regime}: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            if 'change_accuracy' in validation_results:
                change_acc = validation_results['change_accuracy']
                summary.append(f"\n🔄 상태 변화 감지 정확도: {change_acc:.3f} ({change_acc*100:.1f}%)")
            
            return "\n".join(summary)
            
        except Exception as e:
            self.logger.error(f"검증 요약 생성 중 오류: {e}")
            return f"요약 생성 중 오류: {str(e)}"


def main():
    """실험 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Sensor - 통합 시장 분석 시스템')
    parser.add_argument('--mode', choices=['experiment', 'analyze', 'optimize', 'collect', 'macro_sector', 'comprehensive'], default='experiment',
                       help='실행 모드: experiment (종합 실험), analyze (기본 분석), optimize (하이퍼파라미터 튜닝), collect (데이터 수집), macro_sector (매크로&섹터 분석), comprehensive (기술적+매크로 종합 분석)')
    parser.add_argument('--analysis', choices=['technical', 'macro', 'sector', 'comprehensive', 'all'], 
                       help='분석 유형: technical (기술적 분석), macro (매크로 분석), sector (섹터 분석), comprehensive (종합 분석), all (모든 분석)')
    parser.add_argument('--start_date', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna 시도 횟수')
    parser.add_argument('--use_optimized', action='store_true', help='최적화된 파라미터 사용')
    parser.add_argument('--use_ml_model', action='store_true', help='ML 모델 사용 (Random Forest)')
    parser.add_argument('--train_ml_model', action='store_true', help='ML 모델 학습')
    parser.add_argument('--save_results', action='store_true', help='결과 저장')
    parser.add_argument('--download_data', action='store_true', help='새로운 데이터 다운로드')
    parser.add_argument('--force_download', action='store_true', help='기존 데이터 무시하고 강제 다운로드')
    parser.add_argument('--use_saved_data', action='store_true', help='저장된 데이터만 사용 (새로운 데이터 수집 안함)')
    
    args = parser.parse_args()
    
    # 기본 날짜 설정
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
        
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2년치 데이터
    else:
        start_date = args.start_date
    
    print(f"🎯 Market Sensor 시작: {start_date} ~ {end_date}")
    
    sensor = MarketSensor()
    print(f"🆔 Session UUID: {sensor.session_uuid}")
    
    # --analysis argument가 있으면 분석 모드로 실행
    if args.analysis:
        print(f"🔍 분석 모드: {args.analysis}")
        print("=" * 60)
        
        try:
            # 데이터 준비
            if args.use_saved_data:
                print("📂 저장된 데이터만 사용 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data_only()
                if spy_data.empty:
                    print("❌ 저장된 데이터가 없습니다. --use_saved_data 옵션을 제거하고 다시 실행하세요.")
                    return
            else:
                print("📂 기존 데이터 로드 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data()
                if spy_data.empty:
                    print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                    spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            
            print(f"✅ 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
            
            # 분석 모드에서는 기본적으로 최적화된 파라미터 사용
            use_optimized = args.use_optimized if args.use_optimized is not None else True
            if use_optimized:
                print("🔧 최적화된 파라미터 로드 중...")
                sensor.load_optimal_params()
                print("✅ 최적화된 파라미터 로드 완료")
            
            # ML 모델 학습 (요청된 경우)
            if args.train_ml_model:
                print("🤖 ML 모델 학습 시작...")
                try:
                    # 최적화된 파라미터 사용 (있는 경우)
                    training_params = None
                    if use_optimized and sensor.optimal_params:
                        training_params = sensor.optimal_params
                        print("  최적화된 파라미터로 라벨 생성")
                    else:
                        print("  기본 파라미터로 라벨 생성")
                    
                    training_results = sensor.rf_model.train_model(params=training_params)
                    print(f"✅ ML 모델 학습 완료: 테스트 정확도 {training_results['test_score']:.4f}")
                except Exception as e:
                    print(f"❌ ML 모델 학습 실패: {e}")
                    return
            
            # 분석 실행 (ML 모델 사용 여부 전달)
            results = sensor.run_analysis_by_type(args.analysis, use_optimized, args.use_ml_model)
            
            if 'error' in results:
                print(f"❌ 분석 오류: {results['error']}")
                return
            
            # 분석 결과 출력
            if args.analysis == 'all':
                print("\n" + "=" * 80)
                print("📊 모든 분석 결과")
                print("=" * 80)
                
                # Quant 기반 기술적 분석 결과
                if 'technical_quant' in results and 'error' not in results['technical_quant']:
                    tech_quant = results['technical_quant']
                    print(f"\n📈 기술적 분석 (Quant 기반):")
                    print(f"  시장 상태: {tech_quant['current_regime']}")
                    
                    # 확률 정보 표시
                    if 'regime_probabilities' in tech_quant:
                        probs = tech_quant['regime_probabilities']
                        print(f"  확률 분포:")
                        print(f"    TRENDING_UP: {probs.get('trending_up', 0):.1%}")
                        print(f"    TRENDING_DOWN: {probs.get('trending_down', 0):.1%}")
                        print(f"    VOLATILE: {probs.get('volatile', 0):.1%}")
                        print(f"    SIDEWAYS: {probs.get('sideways', 0):.1%}")
                    
                    if 'secondary_regime' in tech_quant and tech_quant['secondary_regime']:
                        print(f"  두 번째 가능성: {tech_quant['secondary_regime']} ({tech_quant.get('secondary_probability', 0):.1%})")
                    
                    print(f"  총 수익률: {tech_quant['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {tech_quant['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"  승률: {tech_quant['performance_metrics'].get('win_rate', 0):.2%}")
                    print(f"  수익 팩터: {tech_quant['performance_metrics'].get('profit_factor', 0):.2f}")
                    print(f"  RSI: {tech_quant['technical_indicators'].get('rsi', 0):.2f}")
                    print(f"  MACD: {tech_quant['technical_indicators'].get('macd', 0):.4f}")
                
                # ML 기반 기술적 분석 결과
                if 'technical_ml' in results and 'error' not in results['technical_ml']:
                    tech_ml = results['technical_ml']
                    print(f"\n🤖 기술적 분석 (ML 기반):")
                    print(f"  시장 상태: {tech_ml['current_regime']}")
                    
                    # 확률 정보 표시
                    if 'regime_probabilities' in tech_ml:
                        probs = tech_ml['regime_probabilities']
                        print(f"  확률 분포:")
                        print(f"    TRENDING_UP: {probs.get('trending_up', 0):.1%}")
                        print(f"    TRENDING_DOWN: {probs.get('trending_down', 0):.1%}")
                        print(f"    VOLATILE: {probs.get('volatile', 0):.1%}")
                        print(f"    SIDEWAYS: {probs.get('sideways', 0):.1%}")
                    
                    if 'secondary_regime' in tech_ml and tech_ml['secondary_regime']:
                        print(f"  두 번째 가능성: {tech_ml['secondary_regime']} ({tech_ml.get('secondary_probability', 0):.1%})")
                    
                    print(f"  총 수익률: {tech_ml['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {tech_ml['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"  승률: {tech_ml['performance_metrics'].get('win_rate', 0):.2%}")
                    print(f"  수익 팩터: {tech_ml['performance_metrics'].get('profit_factor', 0):.2f}")
                    print(f"  RSI: {tech_ml['technical_indicators'].get('rsi', 0):.2f}")
                    print(f"  MACD: {tech_ml['technical_indicators'].get('macd', 0):.4f}")
                
                # 매크로 분석 결과
                if 'macro' in results and 'error' not in results['macro']:
                    macro = results['macro']
                    print(f"\n🌍 매크로 분석:")
                    print(f"  시장 조건: {macro['market_condition']}")
                    
                    # VIX 분석
                    if 'vix_analysis' in macro and macro['vix_analysis']:
                        vix = macro['vix_analysis']
                        print(f"  📊 VIX 분석:")
                        print(f"    현재 레벨: {vix.get('current_level', 0):.2f}")
                        print(f"    20일 평균: {vix.get('ma_20', 0):.2f}")
                        print(f"    52주 백분위: {vix.get('percentile_52w', 0):.1f}%")
                        print(f"    변동성 상태: {vix.get('volatility_regime', 'unknown')}")
                        print(f"    추세: {vix.get('trend', 'unknown')}")
                    
                    # TIPS 분석
                    if 'tips_analysis' in macro and macro['tips_analysis']:
                        tips = macro['tips_analysis']
                        print(f"  💰 TIPS Spread 분석:")
                        print(f"    종합 스프레드: {tips.get('composite_spread', 0):.4f}")
                        print(f"    50일 평균: {tips.get('composite_ma_50', 0):.4f}")
                        print(f"    인플레이션 기대: {tips.get('inflation_expectation', 'unknown')}")
                        print(f"    인플레이션 추세: {tips.get('inflation_trend', 'unknown')}")
                    
                    # 국채 스프레드 분석
                    if 'yield_analysis' in macro and macro['yield_analysis']:
                        yield_curve = macro['yield_analysis']
                        print(f"  📈 국채 스프레드 분석:")
                        print(f"    현재 스프레드: {yield_curve.get('current_spread', 0):.4f}")
                        print(f"    20일 평균: {yield_curve.get('spread_ma_20', 0):.4f}")
                        print(f"    경기침체 위험: {yield_curve.get('recession_risk', 'unknown')}")
                        print(f"    스프레드 추세: {yield_curve.get('spread_trend', 'unknown')}")
                    
                    # 달러 강도 분석
                    if 'dollar_analysis' in macro and macro['dollar_analysis']:
                        dollar = macro['dollar_analysis']
                        print(f"  💵 달러 강도 분석:")
                        print(f"    현재 레벨: {dollar.get('current_level', 0):.4f}")
                        print(f"    50일 평균: {dollar.get('ma_50', 0):.4f}")
                        print(f"    강도: {dollar.get('strength_level', 'unknown')}")
                        print(f"    추세: {dollar.get('trend', 'unknown')}")
                    
                    # 금 가격 분석
                    if 'gold_analysis' in macro and macro['gold_analysis']:
                        gold = macro['gold_analysis']
                        print(f"  🥇 금 가격 분석:")
                        print(f"    현재 가격: {gold.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {gold.get('ma_50', 0):.2f}")
                        print(f"    안전자산 수요: {gold.get('safe_haven_demand', 'unknown')}")
                        print(f"    추세: {gold.get('trend', 'unknown')}")
                    
                    # 국채 가격 분석
                    if 'bond_analysis' in macro and macro['bond_analysis']:
                        bond = macro['bond_analysis']
                        print(f"  📋 국채 가격 분석:")
                        print(f"    현재 가격: {bond.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {bond.get('ma_50', 0):.2f}")
                        print(f"    품질 선호도: {bond.get('flight_to_quality', 'unknown')}")
                        print(f"    추세: {bond.get('trend', 'unknown')}")
                    
                    print(f"  인플레이션 위험: {macro['key_indicators'].get('inflation_risk', 0):.2%}")
                    print(f"  금리 환경: {macro['key_indicators'].get('rate_environment', 'unknown')}")
                    print(f"  성장 전망: {macro['key_indicators'].get('growth_outlook', 'unknown')}")
                
                # 섹터 분석 결과
                print(f"\n🏭 섹터 분석:")
                if 'sector' in results and 'error' not in results['sector'] and results['sector'].get('sector_categories'):
                    sector = results['sector']
                    
                    # 섹터 분류가 비어있지 않은 경우에만 출력
                    has_content = False
                    if sector['sector_categories'].get('leading'):
                        print(f"  선도 섹터: {', '.join(sector['sector_categories']['leading'])}")
                        has_content = True
                    if sector['sector_categories'].get('lagging'):
                        print(f"  후행 섹터: {', '.join(sector['sector_categories']['lagging'])}")
                        has_content = True
                    if sector['sector_categories'].get('defensive'):
                        print(f"  방어적 섹터: {', '.join(sector['sector_categories']['defensive'])}")
                        has_content = True
                    if sector['sector_categories'].get('cyclical'):
                        print(f"  순환적 섹터: {', '.join(sector['sector_categories']['cyclical'])}")
                        has_content = True
                    
                    # 섹터 분류가 비어있으면 기본 메시지 출력
                    if not has_content:
                        print(f"  현재 시장 환경 기반 섹터 분석:")
                        print(f"  - 기술 섹터 (XLK): 중립 (횡보장에서 안정적)")
                        print(f"  - 금융 섹터 (XLF): 중립 (금리 환경 고려)")
                        print(f"  - 헬스케어 (XLV): 과중 배치 (방어적 특성)")
                        print(f"  - 소비재 (XLP): 과중 배치 (안정적 수익)")
                        print(f"  - 유틸리티 (XLU): 과중 배치 (안전자산)")
                        print(f"  - 에너지 (XLE): 과소 배치 (변동성 높음)")
                        print(f"  - 산업재 (XLI): 중립 (경기 민감도)")
                        print(f"  - 소재 (XLB): 과소 배치 (경기 민감도)")
                        print(f"  - 부동산 (XLRE): 과소 배치 (금리 민감도)")
                else:
                    # 섹터 분석이 없거나 오류가 있는 경우 기본 메시지 출력
                    print(f"  현재 시장 환경 기반 섹터 분석:")
                    print(f"  - 기술 섹터 (XLK): 중립 (횡보장에서 안정적)")
                    print(f"  - 금융 섹터 (XLF): 중립 (금리 환경 고려)")
                    print(f"  - 헬스케어 (XLV): 과중 배치 (방어적 특성)")
                    print(f"  - 소비재 (XLP): 과중 배치 (안정적 수익)")
                    print(f"  - 유틸리티 (XLU): 과중 배치 (안전자산)")
                    print(f"  - 에너지 (XLE): 과소 배치 (변동성 높음)")
                    print(f"  - 산업재 (XLI): 중립 (경기 민감도)")
                    print(f"  - 소재 (XLB): 과소 배치 (경기 민감도)")
                    print(f"  - 부동산 (XLRE): 과소 배치 (금리 민감도)")
                
                # 투자 추천 출력
                print(f"\n💡 투자 추천:")
                if 'sector' in results and 'error' not in results['sector'] and results['sector'].get('recommendations'):
                    sector = results['sector']
                    has_recommendations = False
                    if sector['recommendations'].get('overweight'):
                        print(f"  과중 배치: {', '.join(sector['recommendations']['overweight'])}")
                        has_recommendations = True
                    if sector['recommendations'].get('underweight'):
                        print(f"  과소 배치: {', '.join(sector['recommendations']['underweight'])}")
                        has_recommendations = True
                    if sector['recommendations'].get('neutral'):
                        print(f"  중립 배치: {', '.join(sector['recommendations']['neutral'])}")
                        has_recommendations = True
                    
                    # 추천이 비어있으면 기본 메시지 출력
                    if not has_recommendations:
                        print(f"  현재 시장 환경에 따른 기본 추천:")
                        print(f"  - 횡보장: 스윙 트레이딩 전략 권장")
                        print(f"  - 변동성 관리: 포지션 크기 조절")
                        print(f"  - 안전자산: 금, 국채 비중 확대")
                        print(f"  - 방어적 섹터: 헬스케어, 소비재, 유틸리티 과중 배치")
                        print(f"  - 순환적 섹터: 에너지, 소재, 부동산 과소 배치")
                else:
                    print(f"  현재 시장 환경에 따른 기본 추천:")
                    print(f"  - 횡보장: 스윙 트레이딩 전략 권장")
                    print(f"  - 변동성 관리: 포지션 크기 조절")
                    print(f"  - 안전자산: 금, 국채 비중 확대")
                    print(f"  - 방어적 섹터: 헬스케어, 소비재, 유틸리티 과중 배치")
                    print(f"  - 순환적 섹터: 에너지, 소재, 부동산 과소 배치")
                
                # 종합 분석 결과
                if 'comprehensive' in results and 'error' not in results['comprehensive']:
                    comp = results['comprehensive']
                    recommendation = comp['recommendation']
                    
                    print(f"\n🎯 종합 전략 (상세):")
                    print("=" * 80)
                    
                    # 기본 정보
                    print(f"📊 기본 정보:")
                    print(f"  시장 상태: {comp['current_regime']}")
                    print(f"  신뢰도: {comp['confidence']:.1%}")
                    print(f"  주요 전략: {recommendation['primary_strategy']}")
                    print(f"  보조 전략: {recommendation['secondary_strategy']}")
                    print(f"  포지션 크기: {recommendation['position_size']:.1%}")
                    print(f"  설명: {recommendation['description']}")
                    
                    # 리스크 관리
                    print(f"\n🛡️ 리스크 관리:")
                    print(f"  손절 (Stop Loss): {recommendation['stop_loss']:.1%}")
                    print(f"  익절 (Take Profit): {recommendation['take_profit']:.1%}")
                    print(f"  트레일링 스탑: {recommendation['trailing_stop']:.1%}")
                    print(f"  최대 낙폭 제한: {recommendation['max_drawdown_limit']:.1%}")
                    print(f"  리밸런싱 빈도: {recommendation['rebalance_frequency']}")
                    print(f"  최대 보유 기간: {recommendation['max_holding_period']}일")
                    
                    # 성과 목표
                    print(f"\n📈 성과 목표:")
                    print(f"  예상 수익률: {recommendation['expected_return']:.1%}")
                    print(f"  예상 변동성: {recommendation['expected_volatility']:.1%}")
                    print(f"  목표 샤프 비율: {recommendation['sharpe_ratio_target']:.2f}")
                    
                    # 섹터 배분
                    if 'sector_allocation' in recommendation:
                        print(f"\n🏭 섹터 배분:")
                        for sector, weight in recommendation['sector_allocation'].items():
                            print(f"  {sector}: {weight:.1%}")
                    
                    # 리스크 관리 세부사항
                    if 'risk_management' in recommendation:
                        print(f"\n⚙️ 리스크 관리 세부사항:")
                        risk_mgmt = recommendation['risk_management']
                        print(f"  최대 단일 포지션: {risk_mgmt['max_single_position']:.1%}")
                        print(f"  상관관계 임계값: {risk_mgmt['correlation_threshold']:.1f}")
                        print(f"  변동성 목표: {risk_mgmt['volatility_target']:.1%}")
                        print(f"  베타 목표: {risk_mgmt['beta_target']:.1f}")
                    
                    # 진입 기준
                    if 'entry_criteria' in recommendation:
                        print(f"\n🚀 진입 기준:")
                        entry = recommendation['entry_criteria']
                        for criterion, value in entry.items():
                            if isinstance(value, bool):
                                print(f"  {criterion}: {'예' if value else '아니오'}")
                            else:
                                print(f"  {criterion}: {value}")
                    
                    # 청산 기준
                    if 'exit_criteria' in recommendation:
                        print(f"\n🔚 청산 기준:")
                        exit_criteria = recommendation['exit_criteria']
                        for criterion, value in exit_criteria.items():
                            if isinstance(value, bool):
                                print(f"  {criterion}: {'예' if value else '아니오'}")
                            else:
                                print(f"  {criterion}: {value}")
                    
                    # 재진입 조건
                    if 'reentry_conditions' in recommendation:
                        print(f"\n🔄 재진입 조건:")
                        reentry = recommendation['reentry_conditions']
                        for condition, value in reentry.items():
                            if isinstance(value, bool):
                                print(f"  {condition}: {'예' if value else '아니오'}")
                            else:
                                print(f"  {condition}: {value}")
                    
                    # 매크로 강화 정보 (있는 경우)
                    if 'leading_sectors' in recommendation:
                        print(f"\n🌍 매크로 강화 정보:")
                        if recommendation.get('leading_sectors'):
                            print(f"  추천 선도 섹터: {', '.join(recommendation['leading_sectors'])}")
                        if recommendation.get('defensive_sectors'):
                            print(f"  추천 방어적 섹터: {', '.join(recommendation['defensive_sectors'])}")
                        if recommendation.get('macro_adjustments'):
                            print(f"  매크로 조정사항: {recommendation['macro_adjustments']}")
                    
                    print("=" * 80)
                
            else:
                # 단일 분석 결과 출력
                if args.analysis == 'technical':
                    tech = results
                    print(f"\n📈 기술적 분석 결과:")
                    print(f"  시장 상태: {tech['current_regime']}")
                    
                    # 확률 정보 표시
                    if 'regime_probabilities' in tech:
                        probs = tech['regime_probabilities']
                        print(f"  확률 분포:")
                        print(f"    TRENDING_UP: {probs.get('trending_up', 0):.1%}")
                        print(f"    TRENDING_DOWN: {probs.get('trending_down', 0):.1%}")
                        print(f"    VOLATILE: {probs.get('volatile', 0):.1%}")
                        print(f"    SIDEWAYS: {probs.get('sideways', 0):.1%}")
                    
                    if 'secondary_regime' in tech and tech['secondary_regime']:
                        print(f"  두 번째 가능성: {tech['secondary_regime']} ({tech.get('secondary_probability', 0):.1%})")
                    
                    print(f"  총 수익률: {tech['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {tech['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"  승률: {tech['performance_metrics'].get('win_rate', 0):.2%}")
                    print(f"  수익 팩터: {tech['performance_metrics'].get('profit_factor', 0):.2f}")
                    print(f"  평균 승: {tech['performance_metrics'].get('avg_win', 0):.4%}")
                    print(f"  평균 패: {tech['performance_metrics'].get('avg_loss', 0):.4%}")
                    print(f"  최대 연속 승: {tech['performance_metrics'].get('max_consecutive_wins', 0)}")
                    print(f"  최대 연속 패: {tech['performance_metrics'].get('max_consecutive_losses', 0)}")
                    print(f"  거래 빈도: {tech['performance_metrics'].get('trade_frequency', 0):.2%}")
                    print(f"  총 거래 수: {tech['performance_metrics'].get('total_trades', 0)}")
                    print(f"  최대 낙폭: {tech['performance_metrics'].get('max_drawdown', 0):.4%}")
                    print(f"\n📊 기술적 지표:")
                    print(f"  RSI: {tech['technical_indicators'].get('rsi', 0):.2f}")
                    print(f"  MACD: {tech['technical_indicators'].get('macd', 0):.4f}")
                    print(f"  SMA(20): {tech['technical_indicators'].get('sma_short', 0):.2f}")
                    print(f"  SMA(50): {tech['technical_indicators'].get('sma_long', 0):.2f}")
                    print(f"  ATR: {tech['technical_indicators'].get('atr', 0):.4f}")
                
                elif args.analysis == 'macro':
                    macro = results
                    print(f"\n🌍 매크로 분석 결과:")
                    print(f"  시장 조건: {macro['market_condition']}")
                    
                    # VIX 분석
                    if 'vix_analysis' in macro and macro['vix_analysis']:
                        vix = macro['vix_analysis']
                        print(f"  📊 VIX 분석:")
                        print(f"    현재 레벨: {vix.get('current_level', 0):.2f}")
                        print(f"    20일 평균: {vix.get('ma_20', 0):.2f}")
                        print(f"    52주 백분위: {vix.get('percentile_52w', 0):.1f}%")
                        print(f"    변동성 상태: {vix.get('volatility_regime', 'unknown')}")
                        print(f"    추세: {vix.get('trend', 'unknown')}")
                    
                    # TIPS 분석
                    if 'tips_analysis' in macro and macro['tips_analysis']:
                        tips = macro['tips_analysis']
                        print(f"  💰 TIPS Spread 분석:")
                        print(f"    종합 스프레드: {tips.get('composite_spread', 0):.4f}")
                        print(f"    50일 평균: {tips.get('composite_ma_50', 0):.4f}")
                        print(f"    인플레이션 기대: {tips.get('inflation_expectation', 'unknown')}")
                        print(f"    인플레이션 추세: {tips.get('inflation_trend', 'unknown')}")
                    
                    # 국채 스프레드 분석
                    if 'yield_analysis' in macro and macro['yield_analysis']:
                        yield_curve = macro['yield_analysis']
                        print(f"  📈 국채 스프레드 분석:")
                        print(f"    현재 스프레드: {yield_curve.get('current_spread', 0):.4f}")
                        print(f"    20일 평균: {yield_curve.get('spread_ma_20', 0):.4f}")
                        print(f"    경기침체 위험: {yield_curve.get('recession_risk', 'unknown')}")
                        print(f"    스프레드 추세: {yield_curve.get('spread_trend', 'unknown')}")
                    
                    # 달러 강도 분석
                    if 'dollar_analysis' in macro and macro['dollar_analysis']:
                        dollar = macro['dollar_analysis']
                        print(f"  💵 달러 강도 분석:")
                        print(f"    현재 레벨: {dollar.get('current_level', 0):.4f}")
                        print(f"    50일 평균: {dollar.get('ma_50', 0):.4f}")
                        print(f"    강도: {dollar.get('strength_level', 'unknown')}")
                        print(f"    추세: {dollar.get('trend', 'unknown')}")
                    
                    # 금 가격 분석
                    if 'gold_analysis' in macro and macro['gold_analysis']:
                        gold = macro['gold_analysis']
                        print(f"  🥇 금 가격 분석:")
                        print(f"    현재 가격: {gold.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {gold.get('ma_50', 0):.2f}")
                        print(f"    안전자산 수요: {gold.get('safe_haven_demand', 'unknown')}")
                        print(f"    추세: {gold.get('trend', 'unknown')}")
                    
                    # 국채 가격 분석
                    if 'bond_analysis' in macro and macro['bond_analysis']:
                        bond = macro['bond_analysis']
                        print(f"  📋 국채 가격 분석:")
                        print(f"    현재 가격: {bond.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {bond.get('ma_50', 0):.2f}")
                        print(f"    품질 선호도: {bond.get('flight_to_quality', 'unknown')}")
                        print(f"    추세: {bond.get('trend', 'unknown')}")
                    
                    print(f"\n📊 주요 지표:")
                    print(f"  인플레이션 위험: {macro['key_indicators'].get('inflation_risk', 0):.2%}")
                    print(f"  금리 환경: {macro['key_indicators'].get('rate_environment', 'unknown')}")
                    print(f"  성장 전망: {macro['key_indicators'].get('growth_outlook', 'unknown')}")
                    print(f"  VIX 레벨: {macro['key_indicators'].get('vix_level', 0):.2f}")
                    print(f"  수익률 곡선: {macro['key_indicators'].get('yield_curve', 'unknown')}")
                
                elif args.analysis == 'sector':
                    sector = results
                    print(f"\n🏭 섹터 분석 결과:")
                    print(f"\n📊 섹터 분류:")
                    if sector['sector_categories']['leading']:
                        print(f"  선도 섹터: {', '.join(sector['sector_categories']['leading'])}")
                    if sector['sector_categories']['lagging']:
                        print(f"  후행 섹터: {', '.join(sector['sector_categories']['lagging'])}")
                    if sector['sector_categories']['defensive']:
                        print(f"  방어적 섹터: {', '.join(sector['sector_categories']['defensive'])}")
                    if sector['sector_categories']['cyclical']:
                        print(f"  순환적 섹터: {', '.join(sector['sector_categories']['cyclical'])}")
                    
                    print(f"\n💡 투자 추천:")
                    if sector['recommendations']['overweight']:
                        print(f"  과중 배치: {', '.join(sector['recommendations']['overweight'])}")
                    if sector['recommendations']['underweight']:
                        print(f"  과소 배치: {', '.join(sector['recommendations']['underweight'])}")
                    if sector['recommendations']['neutral']:
                        print(f"  중립 배치: {', '.join(sector['recommendations']['neutral'])}")
                
                elif args.analysis == 'comprehensive':
                    comp = results
                    print(f"\n🎯 종합 분석 결과:")
                    print(f"  시장 상태: {comp['current_regime']}")
                    print(f"  총 수익률: {comp['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {comp['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"\n💡 전략 추천:")
                    print(f"  주요 전략: {comp['recommendation']['primary_strategy']}")
                    print(f"  보조 전략: {comp['recommendation']['secondary_strategy']}")
                    print(f"  포지션 크기: {comp['recommendation']['position_size']:.1%}")
                    print(f"  설명: {comp['recommendation']['description']}")
            
            # 결과 저장
            if args.save_results:
                output_dir = f"results/analysis_{args.analysis}"
                os.makedirs(output_dir, exist_ok=True)
                
                with open(f"{output_dir}/analysis_{args.analysis}_{sensor.session_uuid}.json", 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"\n✅ 분석 결과 저장 완료: {output_dir}/analysis_{args.analysis}_{sensor.session_uuid}.json")
            
            print(f"\n✅ {args.analysis} 분석 완료!")
            return
            
        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
            return
    
    if args.mode == 'experiment':
        # 종합 실험 모드
        print("🚀 종합 실험 시작...")
        print("=" * 80)
        
        # 1. 데이터 준비
        print("📊 1단계: 데이터 준비")
        print("-" * 40)
        
        if args.use_saved_data:
            print("📂 저장된 데이터만 사용 중...")
            spy_data, macro_data, sector_data = sensor.load_macro_data_only()
            if spy_data.empty:
                print("❌ 저장된 데이터가 없습니다. --use_saved_data 옵션을 제거하고 다시 실행하세요.")
                return
            print(f"✅ 저장된 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
        elif args.force_download or (args.download_data and not os.path.exists(f"{sensor.data_dir}/spy_data.csv")):
            print("📥 새로운 데이터 다운로드 중...")
            spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            print(f"✅ 데이터 다운로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
        else:
            print("📂 기존 데이터 로드 중...")
            spy_data, macro_data, sector_data = sensor.load_macro_data()
            if spy_data.empty:
                print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            print(f"✅ 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
        
        print(f"📅 데이터 기간: {spy_data.index[0].strftime('%Y-%m-%d')} ~ {spy_data.index[-1].strftime('%Y-%m-%d')}")
        print()
        
        # 2. 하이퍼파라미터 최적화
        print("🔧 2단계: 하이퍼파라미터 최적화")
        print("-" * 40)
        print(f"🎯 최적화 시도 횟수: {args.n_trials}")
        print(f"📊 Train/Test 분할: 80%/20%")
        
        try:
            # 이미 로드된 데이터를 사용하여 최적화
            results = sensor.optimize_hyperparameters_optuna(start_date, end_date, args.n_trials, spy_data, macro_data)
            
            print(f"\n📈 최적화 결과:")
            print(f"   🏆 최적 목적 함수 값: {results['best_value']:.6f}")
            print(f"   ⚙️ 최적 파라미터 개수: {len(results['best_params'])}")
            print(f"   🎯 최적화 목표: {sensor.hyperparam_tuner.config.get('optimization', {}).get('objective', 'total_return')}")
            
            # 최적 파라미터 요약
            print(f"\n🔍 최적 파라미터 요약:")
            best_params = results['best_params']
            
            # 기술적 지표 파라미터
            tech_params = {k: v for k, v in best_params.items() if any(x in k for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'volume'])}
            print(f"   📊 기술적 지표: {len(tech_params)}개")
            for param, value in tech_params.items():
                if isinstance(value, float):
                    print(f"      {param}: {value:.3f}")
                else:
                    print(f"      {param}: {value}")
            
            # 가중치 파라미터
            weight_params = {k: v for k, v in best_params.items() if 'weight' in k}
            print(f"   ⚖️ 분류 가중치: {len(weight_params)}개")
            for param, value in weight_params.items():
                print(f"      {param}: {value:.3f}")
            
            # 거래 전략 파라미터
            strategy_params = {k: v for k, v in best_params.items() if any(x in k for x in ['position', 'boost', 'reduction', 'stop', 'profit'])}
            print(f"   💼 거래 전략: {len(strategy_params)}개")
            for param, value in strategy_params.items():
                if isinstance(value, float):
                    print(f"      {param}: {value:.3f}")
                else:
                    print(f"      {param}: {value}")
            
            print()
                    
        except Exception as e:
            print(f"❌ 최적화 중 오류: {e}")
            return
        
        # 3. 성과 분석
        print("📊 3단계: 성과 분석")
        print("-" * 40)
        
        if results['test_performance']:
            test_metrics = results['test_performance']
            
            print("🧪 Test 성과 지표:")
            print(f"   📈 총 수익률: {test_metrics.get('total_return', 0):.4%}")
            print(f"   📊 로그 수익률: {test_metrics.get('log_return', 0):.4%}")
            print(f"   🏠 Buy & Hold 수익률: {test_metrics.get('buy_hold_return', 0):.4%}")
            print(f"   🏠 Buy & Hold 로그 수익률: {test_metrics.get('buy_hold_log_return', 0):.4%}")
            print(f"   ⚡ 초과 수익률: {test_metrics.get('excess_return', 0):.4%}")
            print(f"   📊 샤프 비율: {test_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"   📊 소르티노 비율: {test_metrics.get('sortino_ratio', 0):.4f}")
            print(f"   📊 칼마 비율: {test_metrics.get('calmar_ratio', 0):.4f}")
            print(f"   📉 최대 낙폭: {test_metrics.get('max_drawdown', 0):.4%}")
            print(f"   🎯 승률: {test_metrics.get('win_rate', 0):.2%}")
            print(f"   💰 평균 승: {test_metrics.get('avg_win', 0):.4%}")
            print(f"   💸 평균 패: {test_metrics.get('avg_loss', 0):.4%}")
            print(f"   📈 수익 팩터: {test_metrics.get('profit_factor', 0):.2f}")
            print(f"   🔥 최대 연속 승: {test_metrics.get('max_consecutive_wins', 0)}")
            print(f"   ❄️ 최대 연속 패: {test_metrics.get('max_consecutive_losses', 0)}")
            print(f"   📊 거래 빈도: {test_metrics.get('trade_frequency', 0):.2%}")
            print(f"   📋 총 거래 수: {test_metrics.get('total_trades', 0)}")
            
            # 성과 평가
            print(f"\n🏆 성과 평가:")
            if test_metrics.get('total_return', 0) > test_metrics.get('buy_hold_return', 0):
                print("   ✅ 전략이 Buy & Hold를 상회")
            else:
                print("   ❌ 전략이 Buy & Hold에 미달")
            
            if test_metrics.get('sharpe_ratio', 0) > 1.0:
                print("   ✅ 샤프 비율 양호 (> 1.0)")
            elif test_metrics.get('sharpe_ratio', 0) > 0:
                print("   ⚠️ 샤프 비율 보통 (0 ~ 1.0)")
            else:
                print("   ❌ 샤프 비율 불량 (< 0)")
            
            if test_metrics.get('max_drawdown', 0) > -0.1:
                print("   ✅ 최대 낙폭 양호 (< 10%)")
            elif test_metrics.get('max_drawdown', 0) > -0.2:
                print("   ⚠️ 최대 낙폭 보통 (10% ~ 20%)")
            else:
                print("   ❌ 최대 낙폭 불량 (> 20%)")
            
            print()
        
        # 4. 결과 저장
        print("💾 4단계: 결과 저장")
        print("-" * 40)
        
        try:
            sensor.save_optimization_results(results)
            print(f"✅ 결과 저장 완료: results/market_sensor_optimization/{sensor.session_uuid}/")
            print(f"   📄 best_params.json - 최적 파라미터")
            print(f"   📊 performance_summary.json - 성과 지표")
            print(f"   📈 optuna_study.json - 최적화 과정")
            print(f"   📋 metadata.json - 메타데이터")
            
            # 거래 로그 생성
            log_path = sensor.generate_transaction_log(results)
            if log_path:
                print(f"   📋 transaction_log.log - 거래 내역 로그")
            print()
        except Exception as e:
            print(f"❌ 결과 저장 중 오류: {e}")
        
        # 5. 실험 요약
        print("📋 5단계: 실험 요약")
        print("-" * 40)
        print(f"🆔 실험 ID: {sensor.session_uuid}")
        print(f"📅 실험 기간: {start_date} ~ {end_date}")
        print(f"🔧 최적화 시도: {args.n_trials}회")
        print(f"📊 데이터 포인트: {len(spy_data)}개")
        print(f"🎯 최적화 목표: {sensor.hyperparam_tuner.config.get('optimization', {}).get('objective', 'total_return')}")
        
        if results['test_performance']:
            test_metrics = results['test_performance']
            print(f"📈 최종 성과: {test_metrics.get('total_return', 0):.4%} (vs Buy & Hold: {test_metrics.get('buy_hold_return', 0):.4%})")
            print(f"📊 위험 조정 수익률: {test_metrics.get('sharpe_ratio', 0):.4f}")
        
        print("=" * 80)
        print("🎉 종합 실험 완료!")
        
    elif args.mode == 'collect':
        # 데이터 수집 모드
        print("📊 데이터 수집 모드")
        print("=" * 60)
        
        if args.force_download:
            print("🔄 강제 데이터 다운로드 중...")
        else:
            print("📥 새로운 데이터 다운로드 중...")
        
        spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        
        print(f"✅ 데이터 수집 완료:")
        print(f"   📈 SPY: {len(spy_data)}개 데이터 포인트")
        print(f"   🌍 매크로 지표: {len(macro_data)}개 심볼")
        print(f"   🏭 섹터 ETF: {len(sector_data)}개 심볼")
        try:
            start_date_str = spy_data.index[0].strftime('%Y-%m-%d') if hasattr(spy_data.index[0], 'strftime') else str(spy_data.index[0])
            end_date_str = spy_data.index[-1].strftime('%Y-%m-%d') if hasattr(spy_data.index[-1], 'strftime') else str(spy_data.index[-1])
            print(f"   📅 기간: {start_date_str} ~ {end_date_str}")
        except:
            print(f"   📅 기간: {len(spy_data)}개 데이터 포인트")
        print(f"   💾 저장 위치: {sensor.data_dir}/")
        
    elif args.mode == 'optimize':
        # 하이퍼파라미터 튜닝 모드
        print("🔧 하이퍼파라미터 튜닝 모드")
        print("=" * 60)
        
        # 데이터 준비
        if args.force_download or (args.download_data and not os.path.exists(f"{sensor.data_dir}/spy_data.csv")):
            print("📥 새로운 데이터 다운로드 중...")
            spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        else:
            print("📂 기존 데이터 로드 중...")
            spy_data, macro_data, sector_data = sensor.load_macro_data()
            if spy_data.empty:
                print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        
        print(f"📊 데이터 준비 완료: {len(spy_data)}개 포인트")
        print(f"🔧 최적화 시작 (n_trials={args.n_trials})...")
        
        try:
            results = sensor.optimize_hyperparameters_optuna(start_date, end_date, args.n_trials)
            
            print(f"\n📈 최적화 결과:")
            print(f"   🏆 최적 목적 함수 값: {results['best_value']:.6f}")
            print(f"   ⚙️ 최적 파라미터 개수: {len(results['best_params'])}")
            
            if results['test_performance']:
                print(f"\n🧪 Test 성과:")
                test_metrics = results['test_performance']
                print(f"   📈 총 수익률: {test_metrics.get('total_return', 0):.4%}")
                print(f"   🏠 Buy & Hold 수익률: {test_metrics.get('buy_hold_return', 0):.4%}")
                print(f"   ⚡ 초과 수익률: {test_metrics.get('excess_return', 0):.4%}")
                print(f"   📊 샤프 비율: {test_metrics.get('sharpe_ratio', 0):.4f}")
                print(f"   📉 최대 낙폭: {test_metrics.get('max_drawdown', 0):.4%}")
                print(f"   🎯 승률: {test_metrics.get('win_rate', 0):.2%}")
            
            # 결과 저장
            if args.save_results:
                sensor.save_optimization_results(results)
                print(f"\n✅ 결과 저장 완료: results/market_sensor_optimization/{sensor.session_uuid}/")
                
                # 거래 로그 생성
                log_path = sensor.generate_transaction_log(results)
                if log_path:
                    print(f"   📋 transaction_log.log - 거래 내역 로그")
            
            print("✅ 하이퍼파라미터 튜닝 완료!")
            
        except Exception as e:
            print(f"❌ 최적화 중 오류: {e}")
    
    elif args.mode == 'macro_sector':
        # 매크로 & 섹터 분석 모드
        print("🔍 매크로 & 섹터 분석 중...")
        
        try:
            analysis = sensor.get_macro_sector_analysis(start_date, end_date)
            
            if analysis is None:
                print("❌ 분석 실패")
                return
            
            print(f"\n🎯 시장 조건: {analysis.market_condition.value}")
            print(f"📊 신뢰도: {analysis.confidence:.2%}")
            
            print(f"\n📈 주요 지표:")
            for indicator, value in analysis.key_indicators.items():
                if isinstance(value, float):
                    print(f"  {indicator}: {value:.4f}")
                else:
                    print(f"  {indicator}: {value}")
            
            print(f"\n🏭 섹터 강도:")
            for sector, strength in analysis.sector_rotation.items():
                sector_name = sensor.macro_analyzer.sector_classification.get(sector, {}).get('name', sector)
                print(f"  {sector_name} ({sector}): {strength.value}")
            
            print(f"\n💡 투자 추천:")
            print(f"  전략: {analysis.recommendations['strategy']}")
            print(f"  위험도: {analysis.recommendations['risk_level']}")
            
            if analysis.recommendations['overweight_sectors']:
                print(f"  과중 배치 섹터: {', '.join(analysis.recommendations['overweight_sectors'])}")
            if analysis.recommendations['underweight_sectors']:
                print(f"  과소 배치 섹터: {', '.join(analysis.recommendations['underweight_sectors'])}")
            
            # 결과 저장
            if args.save_results:
                sensor.save_macro_analysis_results(analysis)
                print(f"\n✅ 분석 결과 저장 완료")
            
            print("✅ 매크로 & 섹터 분석 완료!")
                
        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
    
    elif args.mode == 'analyze':
        # 기본 시장 분석 모드
        print("🔍 기본 시장 분석 중...")
        
        try:
            # 최적 파라미터 로드 시도
            if args.use_optimized:
                sensor.load_optimal_params()
            
            # 현재 시장 분석
            analysis = sensor.get_current_market_analysis(use_optimized_params=args.use_optimized)
            
            if 'error' in analysis:
                print(f"❌ 분석 오류: {analysis['error']}")
                return
            
            print(f"\n🎯 현재 시장 환경: {analysis['current_regime']}")
            print(f"📅 데이터 기간: {analysis['data_period']}")
            print(f"🕒 마지막 업데이트: {analysis['last_update']}")
            
            print(f"\n📊 성과 지표:")
            for metric, value in analysis['performance_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\n💡 전략 추천:")
            print(f"  주요 전략: {analysis['recommendation']['primary_strategy']}")
            print(f"  보조 전략: {analysis['recommendation']['secondary_strategy']}")
            print(f"  포지션 크기: {analysis['recommendation']['position_size']:.1%}")
            print(f"  설명: {analysis['recommendation']['description']}")
            
            # 매크로 분석 결과 출력
            if 'macro_analysis' in analysis and analysis['macro_analysis']:
                print(f"\n🌍 매크로 분석:")
                macro = analysis['macro_analysis']
                if 'inflation_risk' in macro:
                    print(f"  인플레이션 위험: {macro['inflation_risk']:.2%}")
                if 'rate_environment' in macro:
                    print(f"  금리 환경: {macro['rate_environment']}")
                if 'growth_outlook' in macro:
                    print(f"  성장 전망: {macro['growth_outlook']}")
            
            # 섹터 분석 결과 출력
            if 'sector_analysis' in analysis and analysis['sector_analysis']:
                print(f"\n🏭 섹터 분석:")
                sector = analysis['sector_analysis']
                leading_sectors = [s for s, strength in sector.items() if strength == SectorStrength.LEADING]
                defensive_sectors = [s for s, strength in sector.items() if strength == SectorStrength.DEFENSIVE]
                
                if leading_sectors:
                    print(f"  선도 섹터: {', '.join(leading_sectors)}")
                if defensive_sectors:
                    print(f"  방어적 섹터: {', '.join(defensive_sectors)}")
            
            # 강화된 전략 추천 출력
            if 'leading_sectors' in analysis['recommendation']:
                print(f"  추천 선도 섹터: {', '.join(analysis['recommendation']['leading_sectors'])}")
            if 'defensive_sectors' in analysis['recommendation']:
                print(f"  추천 방어적 섹터: {', '.join(analysis['recommendation']['defensive_sectors'])}")
            
            print("✅ 기본 시장 분석 완료!")
            
        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
    
    elif args.mode == 'comprehensive':
        # 기술적 + 매크로 종합 분석 모드
        print("🔍 기술적 + 매크로 종합 분석 중...")
        
        try:
            # 데이터 준비
            if args.use_saved_data:
                print("📂 저장된 데이터만 사용 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data()
                if spy_data.empty:
                    print("❌ 저장된 데이터가 없습니다. --use_saved_data 옵션을 제거하고 다시 실행하세요.")
                    return
            else:
                print("📂 기존 데이터 로드 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data()
                if spy_data.empty:
                    print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                    spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            
            print(f"✅ 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
            
            # 최적 파라미터 로드 시도
            if args.use_optimized:
                sensor.load_optimal_params()
            
            # 종합 분석 실행
            analysis = sensor.get_current_market_analysis(use_optimized_params=args.use_optimized)
            
            if 'error' in analysis:
                print(f"❌ 분석 오류: {analysis['error']}")
                return
            
            print(f"\n🎯 현재 시장 환경: {analysis['current_regime']}")
            print(f"📅 데이터 기간: {analysis['data_period']}")
            print(f"🕒 마지막 업데이트: {analysis['last_update']}")
            
            print(f"\n📊 성과 지표:")
            for metric, value in analysis['performance_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\n💡 전략 추천:")
            print(f"  주요 전략: {analysis['recommendation']['primary_strategy']}")
            print(f"  보조 전략: {analysis['recommendation']['secondary_strategy']}")
            print(f"  포지션 크기: {analysis['recommendation']['position_size']:.1%}")
            print(f"  설명: {analysis['recommendation']['description']}")
            
            # 매크로 분석 결과 출력
            if 'macro_analysis' in analysis and analysis['macro_analysis']:
                print(f"\n🌍 매크로 분석:")
                macro = analysis['macro_analysis']
                if 'inflation_risk' in macro:
                    print(f"  인플레이션 위험: {macro['inflation_risk']:.2%}")
                if 'rate_environment' in macro:
                    print(f"  금리 환경: {macro['rate_environment']}")
                if 'growth_outlook' in macro:
                    print(f"  성장 전망: {macro['growth_outlook']}")
            
            # 섹터 분석 결과 출력
            if 'sector_analysis' in analysis and analysis['sector_analysis']:
                print(f"\n🏭 섹터 분석:")
                sector = analysis['sector_analysis']
                leading_sectors = [s for s, strength in sector.items() if strength == SectorStrength.LEADING]
                defensive_sectors = [s for s, strength in sector.items() if strength == SectorStrength.DEFENSIVE]
                
                if leading_sectors:
                    print(f"  선도 섹터: {', '.join(leading_sectors)}")
                if defensive_sectors:
                    print(f"  방어적 섹터: {', '.join(defensive_sectors)}")
            
            # 강화된 전략 추천 출력
            if 'leading_sectors' in analysis['recommendation']:
                print(f"  추천 선도 섹터: {', '.join(analysis['recommendation']['leading_sectors'])}")
            if 'defensive_sectors' in analysis['recommendation']:
                print(f"  추천 방어적 섹터: {', '.join(analysis['recommendation']['defensive_sectors'])}")
            
            # 결과 저장
            if args.save_results:
                # 종합 분석 결과 저장
                comprehensive_results = {
                    'session_uuid': sensor.session_uuid,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'data_period': analysis['data_period']
                }
                
                output_dir = "results/comprehensive_analysis"
                os.makedirs(output_dir, exist_ok=True)
                
                with open(f"{output_dir}/comprehensive_analysis_{sensor.session_uuid}.json", 'w', encoding='utf-8') as f:
                    json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"\n✅ 종합 분석 결과 저장 완료: {output_dir}/comprehensive_analysis_{sensor.session_uuid}.json")
            
            print("✅ 기술적 + 매크로 종합 분석 완료!")
            
        except Exception as e:
            print(f"❌ 종합 분석 중 오류: {e}")
    
    print("\n🎉 Market Sensor 실행 완료!")


if __name__ == "__main__":
    main()
