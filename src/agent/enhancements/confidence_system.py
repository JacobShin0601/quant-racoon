#!/usr/bin/env python3
"""
다층 신뢰도 계산 시스템 (Multi-Layer Confidence System)

기술적 분석, 매크로 환경, 통계적 차익거래, RLMF 피드백을 종합하여
정교한 신뢰도 계산을 수행하는 시스템
"""

import numpy as np
from typing import Dict, Any
import logging


class MultiLayerConfidenceSystem:
    """
    다층 신뢰도 계산 시스템
    기술적, 매크로, 통계적 차익거래, RLMF 피드백을 종합
    """
    
    def __init__(self):
        self.confidence_weights = {
            'technical': 0.25,      # 기술적 분석
            'macro': 0.20,         # 매크로 환경
            'statistical_arb': 0.25, # 통계적 차익거래
            'rlmf_feedback': 0.20,   # RLMF 피드백
            'cross_validation': 0.10  # 교차 검증
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_confidence(self, 
                                         technical_conf: float,
                                         macro_conf: float, 
                                         stat_arb_conf: float,
                                         rlmf_conf: float,
                                         cross_val_conf: float = 0.5) -> Dict[str, Any]:
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
    
    def update_confidence_weights(self, performance_feedback: Dict[str, float]):
        """성과 피드백을 기반으로 신뢰도 가중치 동적 조정"""
        try:
            # 성과가 좋은 구성요소의 가중치를 증가
            total_performance = sum(performance_feedback.values())
            
            if total_performance > 0:
                for component, performance in performance_feedback.items():
                    if component in self.confidence_weights:
                        # 성과 비율에 따른 가중치 조정 (최대 ±20%)
                        adjustment = (performance / total_performance - 1/len(performance_feedback)) * 0.2
                        new_weight = self.confidence_weights[component] * (1 + adjustment)
                        self.confidence_weights[component] = max(0.05, min(0.5, new_weight))
                
                # 가중치 정규화
                total_weight = sum(self.confidence_weights.values())
                if total_weight > 0:
                    for component in self.confidence_weights:
                        self.confidence_weights[component] /= total_weight
                        
        except Exception as e:
            self.logger.warning(f"신뢰도 가중치 업데이트 중 오류: {e}")
    
    def get_confidence_explanation(self, confidence_result: Dict[str, Any]) -> str:
        """신뢰도 계산 결과에 대한 설명 생성"""
        try:
            total_conf = confidence_result.get('total_confidence', 0.5)
            adjusted_conf = confidence_result.get('adjusted_confidence', 0.5)
            consistency = confidence_result.get('consistency_score', 0.5)
            components = confidence_result.get('component_confidences', {})
            
            explanation = f"종합 신뢰도: {adjusted_conf:.1%}\n"
            explanation += f"일관성 점수: {consistency:.1%}\n\n"
            explanation += "구성요소별 기여도:\n"
            
            for component, value in components.items():
                percentage = (value / total_conf) * 100 if total_conf > 0 else 0
                explanation += f"  • {component}: {value:.3f} ({percentage:.1f}%)\n"
            
            if consistency < 0.5:
                explanation += "\n⚠️ 구성요소 간 일관성이 낮음 - 신중한 판단 필요"
            elif consistency > 0.8:
                explanation += "\n✅ 구성요소 간 높은 일관성 - 신뢰할 만한 신호"
                
            return explanation
            
        except Exception as e:
            return f"설명 생성 중 오류: {e}" 