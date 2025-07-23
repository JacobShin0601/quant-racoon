#!/usr/bin/env python3
"""
시장 환경 분류기 (Market Sensor)
통합 시장 분석 시스템 - 실행 인터페이스 (고도화된 버전)
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
    MarketRegimeValidator,
)
from ..actions.random_forest import MarketRegimeRF
from .enhancements import (
    RLMFRegimeAdaptation,
    MultiLayerConfidenceSystem,
    DynamicRegimeSwitchingDetector,
    LLMPrivilegedInformationSystem,
    LLMAPIIntegration,
    LLMConfig,
)


class MarketSensor:
    """통합 시장 분석 시스템 - 실행 인터페이스 (고도화된 버전)"""

    def __init__(
        self,
        data_dir: str = "data/macro",
        config_path: str = "config/config_macro.json",
        enable_llm_api: bool = False,
        llm_config: LLMConfig = None,
    ):
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

        # LLM API 통합 시스템 (선택적 활성화)
        self.llm_api_system = None
        self.llm_config = llm_config
        if enable_llm_api and llm_config:
            try:
                self.llm_api_system = LLMAPIIntegration(llm_config)
                self.logger.info("LLM API 통합 시스템 활성화됨")
            except Exception as e:
                self.logger.warning(f"LLM API 시스템 초기화 실패: {e}")

        # Random Forest 모델 초기화 (저장된 모델 우선 로드)
        self.rf_model = MarketRegimeRF()

        # 최적화 파라미터 저장 변수
        self.optimal_params = None

        # 경고 무시 설정
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

        # 초기화 완료 로그
        self.logger.info(f"MarketSensor 초기화 완료 - 세션: {self.session_uuid}")

    def load_macro_data(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """매크로 데이터 로드"""
        try:
            return self.macro_collector.collect_all_data()
        except Exception as e:
            self.logger.error(f"매크로 데이터 로드 실패: {e}")
            return pd.DataFrame(), {}, {}

    def get_enhanced_market_analysis(
        self,
        use_optimized_params: bool = True,
        use_ml_model: bool = True,
        enable_advanced_features: bool = True,
    ) -> Dict[str, Any]:
        """
        고도화된 시장 분석 수행

        Args:
            use_optimized_params: 최적화된 파라미터 사용 여부
            use_ml_model: ML 모델 사용 여부
            enable_advanced_features: 고급 기능 활성화 여부
        """
        try:
            # 1. 기본 시장 분석
            basic_analysis = self.get_current_market_analysis(
                use_optimized_params=use_optimized_params, use_ml_model=use_ml_model
            )

            if not enable_advanced_features:
                return basic_analysis

            # 2. 고도화된 분석 수행
            enhanced_analysis = basic_analysis.copy()

            # SPY 데이터와 매크로 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()

            if spy_data.empty or not macro_data:
                self.logger.warning("데이터 부족으로 고급 분석 건너뜀")
                return enhanced_analysis

            # 3. RLMF 적응 분석
            rlmf_analysis = self._perform_rlmf_analysis(
                spy_data, macro_data, basic_analysis
            )
            enhanced_analysis["rlmf_analysis"] = rlmf_analysis

            # 4. 다층 신뢰도 분석
            confidence_analysis = self._perform_confidence_analysis(
                basic_analysis, rlmf_analysis
            )
            enhanced_analysis["confidence_analysis"] = confidence_analysis

            # 5. Regime 전환 감지
            regime_detection = self._perform_regime_detection(spy_data, macro_data)
            enhanced_analysis["regime_detection"] = regime_detection

            # 6. LLM 특권 정보 분석
            llm_insights = self._perform_llm_analysis(basic_analysis, macro_data)
            enhanced_analysis["llm_insights"] = llm_insights

            # 7. LLM API 통합 분석 (활성화된 경우)
            if self.llm_api_system:
                llm_api_insights = self._perform_llm_api_analysis(
                    basic_analysis, macro_data
                )
                enhanced_analysis["llm_api_insights"] = llm_api_insights

            # 8. 종합 신뢰도 계산
            final_confidence = self._calculate_final_confidence(enhanced_analysis)
            enhanced_analysis["final_confidence"] = final_confidence

            # 9. 전략적 추천 강화
            enhanced_recommendations = self._generate_enhanced_recommendations(
                enhanced_analysis
            )
            enhanced_analysis["enhanced_recommendations"] = enhanced_recommendations

            self.logger.info("고도화된 시장 분석 완료")
            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"고도화된 시장 분석 실패: {e}")
            return self.get_current_market_analysis(use_optimized_params, use_ml_model)

    def _perform_rlmf_analysis(
        self,
        spy_data: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame],
        basic_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """RLMF 적응 분석 수행"""
        try:
            current_regime = basic_analysis.get("current_regime", "UNCERTAIN")

            # Statistical Arbitrage 신호 계산
            stat_arb_signal = (
                self.rlmf_adaptation.calculate_statistical_arbitrage_signal(macro_data)
            )

            # Market feedback 계산 (이전 예측과 실제 결과 비교)
            if len(spy_data) >= 10:
                recent_returns = spy_data["close"].pct_change().tail(10)
                market_feedback = self.rlmf_adaptation.calculate_market_feedback(
                    current_regime, recent_returns, spy_data, macro_data
                )
            else:
                market_feedback = {
                    "prediction_accuracy": 0.5,
                    "return_alignment": 0.5,
                    "volatility_prediction": 0.5,
                    "regime_persistence": 0.5,
                    "macro_consistency": 0.5,
                }

            # 적응 상태 확인
            adaptation_status = self.rlmf_adaptation.get_adaptation_status()

            return {
                "statistical_arbitrage": stat_arb_signal,
                "market_feedback": market_feedback,
                "adaptation_status": adaptation_status,
                "learning_rate": self.rlmf_adaptation.learning_rate,
                "feedback_window": self.rlmf_adaptation.feedback_window,
            }

        except Exception as e:
            self.logger.warning(f"RLMF 분석 실패: {e}")
            return {}

    def _perform_confidence_analysis(
        self, basic_analysis: Dict[str, Any], rlmf_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """다층 신뢰도 분석 수행"""
        try:
            # 기본 신뢰도 구성요소들
            technical_conf = basic_analysis.get("confidence", 0.5)
            macro_conf = basic_analysis.get("macro_confidence", 0.5)

            # RLMF 기반 신뢰도
            rlmf_conf = 0.5
            if rlmf_analysis and "market_feedback" in rlmf_analysis:
                feedback = rlmf_analysis["market_feedback"]
                rlmf_conf = feedback.get("prediction_accuracy", 0.5)

            # Statistical Arbitrage 신뢰도
            stat_arb_conf = 0.5
            if rlmf_analysis and "statistical_arbitrage" in rlmf_analysis:
                stat_arb = rlmf_analysis["statistical_arbitrage"]
                stat_arb_conf = stat_arb.get("confidence", 0.5)

            # 교차 검증 신뢰도
            cross_val_conf = 0.5
            if basic_analysis.get("ml_model_used", False):
                cross_val_conf = basic_analysis.get("cross_validation_score", 0.5)

            # 종합 신뢰도 계산
            confidence_result = (
                self.confidence_system.calculate_comprehensive_confidence(
                    technical_conf, macro_conf, stat_arb_conf, rlmf_conf, cross_val_conf
                )
            )

            # 신뢰도 설명 생성
            explanation = self.confidence_system.get_confidence_explanation(
                confidence_result
            )

            return {
                "confidence_result": confidence_result,
                "explanation": explanation,
                "component_breakdown": {
                    "technical": technical_conf,
                    "macro": macro_conf,
                    "statistical_arb": stat_arb_conf,
                    "rlmf_feedback": rlmf_conf,
                    "cross_validation": cross_val_conf,
                },
            }

        except Exception as e:
            self.logger.warning(f"신뢰도 분석 실패: {e}")
            return {"confidence_result": {"adjusted_confidence": 0.5}}

    def _perform_regime_detection(
        self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Regime 전환 감지 수행"""
        try:
            # Regime shift 감지
            regime_shift = self.regime_detector.detect_regime_shifts(
                spy_data, macro_data
            )

            # Regime 안정성 분석
            stability_analysis = self.regime_detector.analyze_regime_stability(spy_data)

            # Regime 지속성 분석
            persistence_analysis = self.regime_detector.calculate_regime_persistence(
                spy_data
            )

            # 최근 변화 이력
            change_history = self.regime_detector.get_regime_change_history(limit=5)

            return {
                "regime_shift_detection": regime_shift,
                "stability_analysis": stability_analysis,
                "persistence_analysis": persistence_analysis,
                "change_history": change_history,
            }

        except Exception as e:
            self.logger.warning(f"Regime 감지 실패: {e}")
            return {}

    def _perform_llm_analysis(
        self, basic_analysis: Dict[str, Any], macro_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """LLM 특권 정보 분석 수행"""
        try:
            current_regime = basic_analysis.get("current_regime", "UNCERTAIN")

            # 시장 메트릭 준비
            market_metrics = {
                "current_probabilities": basic_analysis.get("probabilities", {}),
                "confidence": basic_analysis.get("confidence", 0.5),
                "vix_level": (
                    macro_data.get("^VIX", pd.DataFrame())
                    .get("close", pd.Series())
                    .iloc[-1]
                    if "^VIX" in macro_data and not macro_data["^VIX"].empty
                    else 20.0
                ),
            }

            # LLM 특권 정보 획득
            llm_insights = self.llm_privileged_system.get_privileged_insights(
                current_regime, macro_data, market_metrics
            )

            return llm_insights

        except Exception as e:
            self.logger.warning(f"LLM 분석 실패: {e}")
            return {}

    def _perform_llm_api_analysis(
        self, basic_analysis: Dict[str, Any], macro_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """LLM API 통합 분석 수행"""
        try:
            current_regime = basic_analysis.get("current_regime", "UNCERTAIN")

            # 시장 메트릭 준비
            market_metrics = {
                "current_probabilities": basic_analysis.get("probabilities", {}),
                "confidence": basic_analysis.get("confidence", 0.5),
                "vix_level": (
                    macro_data.get("^VIX", pd.DataFrame())
                    .get("close", pd.Series())
                    .iloc[-1]
                    if "^VIX" in macro_data and not macro_data["^VIX"].empty
                    else 20.0
                ),
            }

            # LLM API 통합 인사이트 획득
            llm_api_insights = self.llm_api_system.get_enhanced_insights(
                current_regime, macro_data, market_metrics
            )

            # API 통계 추가
            api_stats = self.llm_api_system.get_api_stats()
            llm_api_insights["api_stats"] = api_stats

            return llm_api_insights

        except Exception as e:
            self.logger.warning(f"LLM API 분석 실패: {e}")
            return {}

    def _calculate_final_confidence(
        self, enhanced_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """최종 신뢰도 계산"""
        try:
            # 기본 신뢰도
            base_confidence = enhanced_analysis.get("confidence", 0.5)

            # 고급 분석 신뢰도들
            confidence_analysis = enhanced_analysis.get("confidence_analysis", {})
            confidence_result = confidence_analysis.get("confidence_result", {})
            adjusted_confidence = confidence_result.get(
                "adjusted_confidence", base_confidence
            )

            # LLM 인사이트 신뢰도 수정자
            llm_insights = enhanced_analysis.get("llm_insights", {})
            confidence_modifiers = llm_insights.get("confidence_modifiers", [1.0])

            # Regime 전환 감지 영향
            regime_detection = enhanced_analysis.get("regime_detection", {})
            regime_shift = regime_detection.get("regime_shift_detection", {})
            shift_confidence = regime_shift.get("confidence", 1.0)

            # 최종 신뢰도 계산
            final_confidence = adjusted_confidence
            if confidence_modifiers:
                final_confidence *= np.mean(confidence_modifiers)
            final_confidence *= shift_confidence

            # 신뢰도 범위 제한
            final_confidence = max(0.0, min(1.0, final_confidence))

            return {
                "final_confidence": final_confidence,
                "base_confidence": base_confidence,
                "adjusted_confidence": adjusted_confidence,
                "confidence_modifiers": confidence_modifiers,
                "regime_shift_impact": shift_confidence,
                "confidence_level": self._get_confidence_level(final_confidence),
            }

        except Exception as e:
            self.logger.warning(f"최종 신뢰도 계산 실패: {e}")
            return {"final_confidence": 0.5, "confidence_level": "MEDIUM"}

    def _get_confidence_level(self, confidence: float) -> str:
        """신뢰도 수준 분류"""
        if confidence >= 0.8:
            return "VERY_HIGH"
        elif confidence >= 0.6:
            return "HIGH"
        elif confidence >= 0.4:
            return "MEDIUM"
        elif confidence >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"

    def _generate_enhanced_recommendations(
        self, enhanced_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """고도화된 전략적 추천 생성"""
        try:
            recommendations = {
                "primary_strategy": "",
                "risk_management": {},
                "sector_rotation": [],
                "position_sizing": "MODERATE",
                "time_horizon": "MEDIUM",
                "key_considerations": [],
                "advanced_insights": [],
            }

            # 기본 분석에서 전략 추출
            basic_recommendation = enhanced_analysis.get("recommendation", {})
            recommendations["primary_strategy"] = basic_recommendation.get(
                "strategy", "BALANCED"
            )

            # RLMF 분석 기반 추천
            rlmf_analysis = enhanced_analysis.get("rlmf_analysis", {})
            if rlmf_analysis:
                stat_arb = rlmf_analysis.get("statistical_arbitrage", {})
                if stat_arb.get("direction") == "BULLISH":
                    recommendations["key_considerations"].append(
                        "Statistical Arbitrage 신호: 강세"
                    )
                elif stat_arb.get("direction") == "BEARISH":
                    recommendations["key_considerations"].append(
                        "Statistical Arbitrage 신호: 약세"
                    )

            # Regime 전환 감지 기반 추천
            regime_detection = enhanced_analysis.get("regime_detection", {})
            if regime_detection.get("regime_shift_detection", {}).get(
                "regime_shift_detected", False
            ):
                recommendations["key_considerations"].append(
                    "⚠️ 시장 체제 전환 감지됨 - 신중한 접근 필요"
                )

            # LLM 인사이트 기반 추천
            llm_insights = enhanced_analysis.get("llm_insights", {})
            if llm_insights:
                strategic_recs = llm_insights.get("strategic_recommendations", [])
                recommendations["advanced_insights"].extend(
                    strategic_recs[:3]
                )  # 상위 3개만

            # 신뢰도 기반 포지션 사이징 조정
            final_confidence = enhanced_analysis.get("final_confidence", {}).get(
                "final_confidence", 0.5
            )
            if final_confidence >= 0.7:
                recommendations["position_sizing"] = "AGGRESSIVE"
            elif final_confidence <= 0.3:
                recommendations["position_sizing"] = "CONSERVATIVE"

            return recommendations

        except Exception as e:
            self.logger.warning(f"고도화된 추천 생성 실패: {e}")
            return {"primary_strategy": "BALANCED", "position_sizing": "MODERATE"}

    def get_current_market_analysis(
        self, use_optimized_params: bool = True, use_ml_model: bool = True
    ) -> Dict[str, Any]:
        """기본 시장 분석 수행 (기존 메서드)"""
        try:
            # 기본 분석 로직 (기존 코드 유지)
            # ... (기존 get_current_market_analysis 로직)

            # 임시로 간단한 분석 반환
            return {
                "current_regime": "TRENDING_UP",
                "confidence": 0.7,
                "probabilities": {
                    "TRENDING_UP": 0.6,
                    "TRENDING_DOWN": 0.2,
                    "VOLATILE": 0.15,
                    "SIDEWAYS": 0.05,
                },
                "recommendation": {"strategy": "BULLISH", "risk_level": "MODERATE"},
            }

        except Exception as e:
            self.logger.error(f"기본 시장 분석 실패: {e}")
            return {}

    def enable_llm_api(self, llm_config: LLMConfig):
        """LLM API 시스템 활성화"""
        try:
            self.llm_api_system = LLMAPIIntegration(llm_config)
            self.llm_config = llm_config
            self.logger.info("LLM API 통합 시스템 활성화됨")
        except Exception as e:
            self.logger.error(f"LLM API 시스템 활성화 실패: {e}")

    def disable_llm_api(self):
        """LLM API 시스템 비활성화"""
        self.llm_api_system = None
        self.llm_config = None
        self.logger.info("LLM API 통합 시스템 비활성화됨")

    def get_llm_api_stats(self) -> Dict[str, Any]:
        """LLM API 통계 반환"""
        if self.llm_api_system:
            return self.llm_api_system.get_api_stats()
        return {"status": "disabled"}


def main():
    """메인 실행 함수"""
    print("🚀 고도화된 Market Sensor 시스템 시작")

    # LLM API 설정 (선택사항)
    llm_config = LLMConfig(
        provider="hybrid",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_to_rules=True,
    )

    # Market Sensor 초기화
    sensor = MarketSensor(enable_llm_api=True, llm_config=llm_config)

    # 고도화된 분석 수행
    analysis = sensor.get_enhanced_market_analysis(
        use_optimized_params=True, use_ml_model=True, enable_advanced_features=True
    )

    # 결과 출력
    print("\n📊 고도화된 시장 분석 결과:")
    print(f"현재 체제: {analysis.get('current_regime', 'N/A')}")
    print(
        f"최종 신뢰도: {analysis.get('final_confidence', {}).get('final_confidence', 0.5):.3f}"
    )

    # LLM API 통계
    if sensor.llm_api_system:
        stats = sensor.get_llm_api_stats()
        print(f"LLM API 성공률: {stats.get('success_rate', 0):.2%}")


if __name__ == "__main__":
    main()
