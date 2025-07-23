#!/usr/bin/env python3
"""
LLM Privileged Information 활용 시스템

Reference: arXiv:2406.15508
- LLM의 world knowledge를 활용한 시장 분석 강화
- Market regime과 경제 환경에 대한 상황적 이해 제공
- 계절성, 경제 사이클, 지정학적 요인 등 종합 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import logging


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
    
    def analyze_macro_regime_correlation(self, current_regime: str, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """매크로 환경과 regime 상관관계 심화 분석"""
        try:
            correlation_analysis = {
                'economic_consistency': 0.5,
                'macro_regime_alignment': 'neutral',
                'supporting_indicators': [],
                'conflicting_indicators': []
            }
            
            economic_context = self._analyze_economic_context(macro_data)
            
            # 각 매크로 지표별 regime 지지/반대 점수 계산
            support_score = 0
            total_indicators = 0
            
            for indicator, regime_value in economic_context.items():
                if regime_value != 'unknown':
                    total_indicators += 1
                    
                    if indicator == 'inflation_regime':
                        pattern = self.market_knowledge_base['inflation_patterns'].get(regime_value, {})
                        if current_regime in pattern.get('typical_regimes', []):
                            support_score += 1
                            correlation_analysis['supporting_indicators'].append(f"인플레이션({regime_value})")
                        else:
                            correlation_analysis['conflicting_indicators'].append(f"인플레이션({regime_value})")
                    
                    elif indicator == 'rate_environment':
                        pattern = self.market_knowledge_base['rate_environment_patterns'].get(regime_value, {})
                        if current_regime in pattern.get('typical_regimes', []):
                            support_score += 1
                            correlation_analysis['supporting_indicators'].append(f"금리환경({regime_value})")
                        else:
                            correlation_analysis['conflicting_indicators'].append(f"금리환경({regime_value})")
            
            # 경제적 일관성 점수 계산
            if total_indicators > 0:
                correlation_analysis['economic_consistency'] = support_score / total_indicators
            
            # 매크로-regime 정렬도 결정
            if correlation_analysis['economic_consistency'] > 0.7:
                correlation_analysis['macro_regime_alignment'] = 'strong_support'
            elif correlation_analysis['economic_consistency'] > 0.5:
                correlation_analysis['macro_regime_alignment'] = 'moderate_support'
            elif correlation_analysis['economic_consistency'] < 0.3:
                correlation_analysis['macro_regime_alignment'] = 'conflicting'
            else:
                correlation_analysis['macro_regime_alignment'] = 'neutral'
            
            return correlation_analysis
            
        except Exception as e:
            self.logger.warning(f"매크로-regime 상관관계 분석 중 오류: {e}")
            return {'economic_consistency': 0.5, 'macro_regime_alignment': 'neutral'} 