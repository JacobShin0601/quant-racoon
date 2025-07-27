#!/usr/bin/env python3
"""
LLM API 통합 시스템 (LangChain 기반)

LangChain을 활용한 안정적이고 확장 가능한 시장 분석 강화 시스템
기존 규칙 기반 시스템과 하이브리드로 동작하여 안정성과 성능을 모두 확보
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
import json
import time
import hashlib
from dataclasses import dataclass
import warnings

# LangChain 관련 import
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    from langchain_aws import ChatBedrock
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_community.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    from langchain.schema import HumanMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    warnings.warn(f"LangChain not available: {e}. Using rule-based system only.")
    # LangChain이 없을 때를 위한 대체 클래스들
    BaseModel = object
    Field = lambda **kwargs: lambda x: x

from .llm_insights import LLMPrivilegedInformationSystem


@dataclass
class LLMConfig:
    """LLM 설정 클래스"""

    provider: str = "bedrock"  # "bedrock", "openai", "anthropic", "hybrid"
    model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    api_key: Optional[str] = None
    region: str = "us-east-1"
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 30
    retry_attempts: int = 3
    fallback_to_rules: bool = True


class MarketAnalysisOutput(BaseModel):
    """시장 분석 결과를 위한 Pydantic 모델"""

    def __init__(self, **kwargs):
        if LANGCHAIN_AVAILABLE:
            super().__init__(**kwargs)
        else:
            # LangChain이 없을 때는 단순한 객체로 동작
            self.comprehensive_analysis = kwargs.get("comprehensive_analysis", {})
            self.risk_assessment = kwargs.get("risk_assessment", {})
            self.strategic_recommendations = kwargs.get("strategic_recommendations", {})
            self.scenario_analysis = kwargs.get("scenario_analysis", {})
            self.confidence_modifier = kwargs.get("confidence_modifier", 1.0)
            self.key_insights = kwargs.get("key_insights", [])

    if LANGCHAIN_AVAILABLE:
        comprehensive_analysis: Dict[str, Any] = Field(
            description="종합적인 시장 분석 결과"
        )
        risk_assessment: Dict[str, Any] = Field(description="리스크 평가 결과")
        strategic_recommendations: Dict[str, Any] = Field(description="전략적 제언")
        scenario_analysis: Dict[str, Any] = Field(description="시나리오 분석")
        confidence_modifier: float = Field(
            description="신뢰도 수정자 (0.5-1.5)", ge=0.5, le=1.5
        )
        key_insights: List[str] = Field(description="핵심 인사이트 목록")


class LLMAPIIntegration:
    """
    LLM API 통합 시스템 (LangChain 기반)

    LangChain을 활용하여 안정적이고 확장 가능한 시장 분석 강화 시스템
    기존 규칙 기반 시스템과 하이브리드로 동작
    """

    def __init__(self, config: LLMConfig = None):
        # 딕셔너리로 전달된 설정을 LLMConfig 객체로 변환
        if isinstance(config, dict):
            self.config = LLMConfig(**config)
        else:
            self.config = config or LLMConfig()

        self.logger = logging.getLogger(__name__)

        # 기존 규칙 기반 시스템 (fallback용)
        self.rule_based_system = LLMPrivilegedInformationSystem()

        # LangChain 캐시 설정
        if LANGCHAIN_AVAILABLE:
            set_llm_cache(InMemoryCache())

        # LLM 모델 초기화
        self.llm_model = self._initialize_llm_model()

        # 출력 파서 초기화
        if LANGCHAIN_AVAILABLE:
            self.output_parser = JsonOutputParser(pydantic_object=MarketAnalysisOutput)
        else:
            self.output_parser = None

        # 성능 모니터링
        self.api_call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_response_time": 0.0,
        }

    def _initialize_llm_model(self) -> Optional[Any]:
        """LangChain LLM 모델 초기화"""
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning(
                "LangChain not available. Using rule-based system only."
            )
            return None

        try:
            self.logger.info(
                f"Initializing LangChain LLM with provider: {self.config.provider}"
            )
            self.logger.info(f"Model name: {self.config.model_name}")

            if self.config.provider == "bedrock":
                self.logger.info(
                    f"Creating Bedrock model for region: {self.config.region}"
                )
                model = ChatBedrock(
                    model_id=self.config.model_name,
                    region_name=self.config.region,
                    model_kwargs={
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    },
                )
                self.logger.info("Bedrock model created successfully")
                return model

            elif self.config.provider == "openai":
                if not self.config.api_key:
                    self.logger.warning(
                        "OpenAI API key not provided. Using rule-based system only."
                    )
                    return None

                self.logger.info("Creating OpenAI model")
                model = ChatOpenAI(
                    model=self.config.model_name,
                    openai_api_key=self.config.api_key,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout,
                )
                self.logger.info("OpenAI model created successfully")
                return model

            elif self.config.provider == "anthropic":
                if not self.config.api_key:
                    self.logger.warning(
                        "Anthropic API key not provided. Using rule-based system only."
                    )
                    return None

                self.logger.info("Creating Anthropic model")
                model = ChatAnthropic(
                    model=self.config.model_name,
                    anthropic_api_key=self.config.api_key,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                self.logger.info("Anthropic model created successfully")
                return model

            else:
                self.logger.warning(f"Unknown LLM provider: {self.config.provider}")
                return None

        except Exception as e:
            self.logger.error(f"LangChain LLM initialization failed: {e}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def get_enhanced_insights(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
        analysis_results: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        향상된 인사이트 획득 (LangChain LLM + 규칙 기반 하이브리드)

        Args:
            current_regime: 현재 시장 체제
            macro_data: 매크로 데이터
            market_metrics: 시장 메트릭
            analysis_results: 기존 분석 결과
        """
        try:
            # 1. 규칙 기반 분석 (기본)
            rule_based_insights = self.rule_based_system.get_privileged_insights(
                current_regime, macro_data, market_metrics
            )

            # 2. LangChain LLM 호출 (향상된 분석)
            if self.llm_model and self.config.provider != "rule_only":
                try:
                    llm_insights = self._call_langchain_llm(
                        current_regime, macro_data, market_metrics, analysis_results
                    )

                    # 3. 두 결과 융합
                    enhanced_insights = self._combine_insights(
                        rule_based_insights, llm_insights
                    )

                    self.logger.info("LangChain LLM 통합 분석 완료")
                    return enhanced_insights

                except Exception as e:
                    self.logger.warning(
                        f"LangChain LLM 호출 실패, 규칙 기반 분석 사용: {e}"
                    )
                    self.api_call_stats["failed_calls"] += 1

            # LLM API 실패 시 규칙 기반만 사용
            return rule_based_insights

        except Exception as e:
            self.logger.error(f"Enhanced insights generation failed: {e}")
            return self._get_fallback_insights(
                current_regime, macro_data, market_metrics
            )

    def _call_langchain_llm(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
        analysis_results: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """LangChain LLM 호출"""
        # LangChain이 없으면 규칙 기반 분석만 반환
        if not LANGCHAIN_AVAILABLE or not self.llm_model:
            self.logger.warning(
                "LangChain not available, using rule-based analysis only"
            )
            return {}

        start_time = time.time()

        try:
            # 프롬프트 생성
            prompt = self._create_langchain_prompt(
                current_regime, macro_data, market_metrics, analysis_results
            )

            self.logger.info(
                f"Calling LangChain LLM with prompt length: {len(prompt)} characters"
            )

            # LangChain 체인 실행
            chain = prompt | self.llm_model | self.output_parser

            # API 호출 (재시도 로직 포함)
            response = None
            for attempt in range(self.config.retry_attempts):
                try:
                    response = chain.invoke({})
                    if response:
                        break
                except Exception as e:
                    self.logger.warning(
                        f"LangChain LLM call attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(2**attempt)  # 지수 백오프

            # 응답 처리
            if response:
                # 통계 업데이트
                response_time = time.time() - start_time
                self.api_call_stats["successful_calls"] += 1
                self.api_call_stats["total_calls"] += 1
                self._update_avg_response_time(response_time)

                self.logger.info("LangChain LLM response received successfully")
                return response

            # API 호출 실패
            self.api_call_stats["failed_calls"] += 1
            self.api_call_stats["total_calls"] += 1
            raise Exception("LangChain LLM call failed")

        except Exception as e:
            self.logger.error(f"LangChain LLM call failed: {e}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _create_langchain_prompt(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
        analysis_results: Dict[str, Any] = None,
    ) -> ChatPromptTemplate:
        """LangChain 프롬프트 템플릿 생성"""
        # LangChain이 없으면 더미 객체 반환
        if not LANGCHAIN_AVAILABLE:
            return None

        # 시장 요약 생성
        market_summary = self._create_market_summary(macro_data)
        metrics_summary = self._create_metrics_summary(market_metrics)
        analysis_summary = (
            self._create_analysis_summary(analysis_results)
            if analysis_results
            else "기존 분석 결과 없음"
        )

        # LangChain 프롬프트 템플릿
        template = f"""
당신은 금융 시장 분석 전문가입니다. 제공된 종합적인 시장 분석 결과를 바탕으로 다면적이고 심층적인 해석을 제공해주세요.

## 현재 시장 상황
- 감지된 시장 체제: {current_regime}
- VIX 수준: {market_summary.get('vix_level', 'N/A')}
- 10년 국채 수익률: {market_summary.get('tnx_level', 'N/A')}
- TIPS 수준: {market_summary.get('tips_level', 'N/A')}
- 달러 인덱스: {market_summary.get('dxy_level', 'N/A')}

## 시장 메트릭
{metrics_summary}

## 기존 분석 결과
{analysis_summary}

## 종합 분석 요청사항
다음 관점에서 심층적인 분석을 제공해주세요:

1. **지표 해석**: 각 기술적/매크로 지표의 의미와 현재 시장에서의 중요성
2. **다면적 분석**: 기술적, 매크로, 섹터 분석 결과의 일관성과 상충점
3. **시장 역학**: 현재 시장의 주요 동인과 변동성 원인
4. **리스크 평가**: 단기/중기/장기 관점에서의 위험 요인
5. **전략적 제언**: 현재 상황에 최적화된 포트폴리오 구성 방안
6. **시나리오 분석**: 다양한 시장 시나리오별 대응 방안

다음 JSON 형태로 답변해주세요:

{{
    "comprehensive_analysis": {{
        "market_dynamics": {{
            "primary_drivers": ["주요 동인1", "주요 동인2"],
            "volatility_factors": ["변동성 요인1", "변동성 요인2"],
            "trend_strength": "strong/moderate/weak",
            "momentum_quality": "high/medium/low"
        }},
        "indicator_interpretation": {{
            "technical_indicators": {{
                "rsi_interpretation": "RSI 해석",
                "macd_interpretation": "MACD 해석",
                "volume_analysis": "거래량 분석"
            }},
            "macro_indicators": {{
                "yield_curve_analysis": "수익률 곡선 분석",
                "inflation_outlook": "인플레이션 전망",
                "growth_prospects": "성장 전망"
            }}
        }},
        "consistency_analysis": {{
            "technical_macro_alignment": 0.0-1.0,
            "sector_macro_alignment": 0.0-1.0,
            "conflicting_signals": ["상충 신호1", "상충 신호2"],
            "supporting_signals": ["지지 신호1", "지지 신호2"]
        }}
    }},
    "risk_assessment": {{
        "short_term_risks": ["단기 위험1", "단기 위험2"],
        "medium_term_risks": ["중기 위험1", "중기 위험2"],
        "long_term_risks": ["장기 위험1", "장기 위험2"],
        "risk_mitigation": {{
            "portfolio_hedging": ["헤징 전략1", "헤징 전략2"],
            "position_sizing": "conservative/moderate/aggressive",
            "stop_loss_levels": "적정 손절 수준"
        }}
    }},
    "strategic_recommendations": {{
        "portfolio_allocation": {{
            "equity_allocation": "0-100%",
            "bond_allocation": "0-100%",
            "cash_allocation": "0-100%",
            "alternative_allocation": "0-100%"
        }},
        "sector_focus": {{
            "overweight_sectors": ["과중 배치 섹터1", "과중 배치 섹터2"],
            "underweight_sectors": ["과소 배치 섹터1", "과소 배치 섹터2"],
            "avoid_sectors": ["회피 섹터1", "회피 섹터2"]
        }},
        "trading_strategy": {{
            "entry_timing": "immediate/gradual/wait",
            "holding_period": "short/medium/long",
            "exit_strategy": "exit 전략"
        }}
    }},
    "scenario_analysis": {{
        "bull_scenario": {{
            "probability": 0.0-1.0,
            "triggers": ["상승 트리거1", "상승 트리거2"],
            "actions": ["상승 시 행동1", "상승 시 행동2"]
        }},
        "bear_scenario": {{
            "probability": 0.0-1.0,
            "triggers": ["하락 트리거1", "하락 트리거2"],
            "actions": ["하락 시 행동1", "하락 시 행동2"]
        }},
        "sideways_scenario": {{
            "probability": 0.0-1.0,
            "triggers": ["횡보 트리거1", "횡보 트리거2"],
            "actions": ["횡보 시 행동1", "횡보 시 행동2"]
        }}
    }},
    "confidence_modifier": 0.5-1.5,
    "key_insights": ["핵심 인사이트1", "핵심 인사이트2", "핵심 인사이트3"]
}}
"""

        return ChatPromptTemplate.from_template(template)

    def _create_market_summary(
        self, macro_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """시장 요약 생성"""
        summary = {}

        try:
            # VIX 분석
            if "VIX" in macro_data:
                vix_data = macro_data["VIX"]
                if not vix_data.empty:
                    latest_vix = vix_data.iloc[-1]["Close"]
                    summary["vix_level"] = latest_vix

                    if latest_vix < 15:
                        summary["vix_level"] = "낮음 (안정)"
                    elif latest_vix < 25:
                        summary["vix_level"] = "보통"
                    else:
                        summary["vix_level"] = "높음 (변동성)"

            # TNX (10년 국채 수익률) 분석
            if "TNX" in macro_data:
                tnx_data = macro_data["TNX"]
                if not tnx_data.empty:
                    latest_tnx = tnx_data.iloc[-1]["Close"]
                    summary["tnx_level"] = latest_tnx

                    if latest_tnx < 2.0:
                        summary["tnx_level"] = "낮음 (성장 우려)"
                    elif latest_tnx < 4.0:
                        summary["tnx_level"] = "보통"
                    else:
                        summary["tnx_level"] = "높음 (인플레이션 우려)"

            # TIPS 분석
            if "TIPS" in macro_data:
                tips_data = macro_data["TIPS"]
                if not tips_data.empty:
                    latest_tips = tips_data.iloc[-1]["Close"]
                    summary["tips_level"] = latest_tips

                    if latest_tips < 0:
                        summary["tips_level"] = "음수 (디플레이션 우려)"
                    else:
                        summary["tips_level"] = "양수 (인플레이션 우려)"

            # DXY (달러 인덱스) 분석
            if "DXY" in macro_data:
                dxy_data = macro_data["DXY"]
                if not dxy_data.empty:
                    latest_dxy = dxy_data.iloc[-1]["Close"]
                    summary["dxy_level"] = latest_dxy

                    if latest_dxy < 95:
                        summary["dxy_level"] = "약세"
                    elif latest_dxy < 105:
                        summary["dxy_level"] = "보통"
                    else:
                        summary["dxy_level"] = "강세"

        except Exception as e:
            self.logger.warning(f"Market summary creation failed: {e}")

        return summary

    def _create_metrics_summary(self, market_metrics: Dict[str, Any]) -> str:
        """메트릭 요약 생성"""
        try:
            summary_parts = []

            # 기본 메트릭
            if "probabilities" in market_metrics:
                probs = market_metrics["probabilities"]
                summary_parts.append(
                    f"시장 체제 확률: 상승 {probs.get('UP', 0):.1%}, 하락 {probs.get('DOWN', 0):.1%}, 횡보 {probs.get('SIDEWAYS', 0):.1%}"
                )

            # 통계적 차익거래 신호
            if "stat_arb_signals" in market_metrics:
                signals = market_metrics["stat_arb_signals"]
                summary_parts.append(f"통계적 차익거래 신호: {signals}")

            # 기타 메트릭들
            for key, value in market_metrics.items():
                if key not in ["probabilities", "stat_arb_signals"]:
                    summary_parts.append(f"{key}: {value}")

            return "\n".join(summary_parts) if summary_parts else "메트릭 데이터 없음"

        except Exception as e:
            self.logger.warning(f"Metrics summary creation failed: {e}")
            return "메트릭 요약 생성 실패"

    def _create_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """기존 분석 결과 요약 생성"""
        try:
            if not analysis_results:
                return "기존 분석 결과 없음"

            summary_parts = []

            # 현재 체제
            if "current_regime" in analysis_results:
                summary_parts.append(f"현재 체제: {analysis_results['current_regime']}")

            # 신뢰도
            if "confidence" in analysis_results:
                summary_parts.append(f"신뢰도: {analysis_results['confidence']:.2f}")

            # 확률
            if "probabilities" in analysis_results:
                probs = analysis_results["probabilities"]
                summary_parts.append(f"체제 확률: {probs}")

            # 최적화 성능
            if "optimization_performance" in analysis_results:
                perf = analysis_results["optimization_performance"]
                summary_parts.append(f"최적화 성능: {perf}")

            # 검증 결과
            if "validation_results" in analysis_results:
                validation = analysis_results["validation_results"]
                summary_parts.append(f"검증 결과: {validation}")

            return "\n".join(summary_parts) if summary_parts else "분석 결과 요약 없음"

        except Exception as e:
            self.logger.warning(f"Analysis summary creation failed: {e}")
            return "분석 결과 요약 생성 실패"

    def _combine_insights(
        self, rule_insights: Dict[str, Any], llm_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """규칙 기반과 LLM 인사이트 융합"""
        try:
            combined = {
                "analysis_timestamp": datetime.now().isoformat(),
                "data_source": "hybrid_llm_rules",
                "rule_based_insights": rule_insights,
                "llm_enhanced_insights": llm_insights,
            }

            # LLM 인사이트에서 주요 섹션들 추출
            if "comprehensive_analysis" in llm_insights:
                combined["market_dynamics"] = llm_insights[
                    "comprehensive_analysis"
                ].get("market_dynamics", {})
                combined["indicator_interpretation"] = llm_insights[
                    "comprehensive_analysis"
                ].get("indicator_interpretation", {})
                combined["consistency_analysis"] = llm_insights[
                    "comprehensive_analysis"
                ].get("consistency_analysis", {})

            if "risk_assessment" in llm_insights:
                combined["risk_assessment"] = llm_insights["risk_assessment"]

            if "strategic_recommendations" in llm_insights:
                combined["strategic_recommendations"] = llm_insights[
                    "strategic_recommendations"
                ]

            if "scenario_analysis" in llm_insights:
                combined["scenario_analysis"] = llm_insights["scenario_analysis"]

            if "key_insights" in llm_insights:
                combined["key_insights"] = llm_insights["key_insights"]

            # 신뢰도 수정자 적용
            confidence_modifier = llm_insights.get("confidence_modifier", 1.0)
            if "confidence" in rule_insights:
                original_confidence = rule_insights["confidence"]
                combined["adjusted_confidence"] = min(
                    1.0, original_confidence * confidence_modifier
                )

            # API 통계 추가
            combined["api_stats"] = self.get_api_stats()

            return combined

        except Exception as e:
            self.logger.error(f"Insights combination failed: {e}")
            return rule_insights

    def _update_avg_response_time(self, new_response_time: float):
        """평균 응답 시간 업데이트"""
        total_calls = self.api_call_stats["successful_calls"]
        current_avg = self.api_call_stats["avg_response_time"]

        if total_calls == 1:
            self.api_call_stats["avg_response_time"] = new_response_time
        else:
            self.api_call_stats["avg_response_time"] = (
                current_avg * (total_calls - 1) + new_response_time
            ) / total_calls

    def _get_fallback_insights(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback 인사이트 생성"""
        return self.rule_based_system.get_privileged_insights(
            current_regime, macro_data, market_metrics
        )

    def get_api_stats(self) -> Dict[str, Any]:
        """API 통계 반환"""
        total_calls = self.api_call_stats["total_calls"]
        success_rate = (
            self.api_call_stats["successful_calls"] / total_calls * 100
            if total_calls > 0
            else 0
        )

        return {
            **self.api_call_stats,
            "success_rate": success_rate,
            "provider": self.config.provider,
            "model": self.config.model_name,
        }

    def clear_cache(self):
        """캐시 클리어"""
        if LANGCHAIN_AVAILABLE:
            # LangChain 캐시는 자동으로 관리됨
            pass

    def update_config(self, new_config: LLMConfig):
        """설정 업데이트"""
        self.config = new_config
        self.llm_model = self._initialize_llm_model()


def test_langchain_llm_integration():
    """LangChain LLM 통합 테스트"""
    print("🧪 LangChain LLM 통합 테스트")
    print("=" * 50)

    # 테스트 설정
    config = LLMConfig(
        provider="bedrock",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens=2000,
        temperature=0.1,
    )

    try:
        # LLM API 통합 시스템 초기화
        llm_system = LLMAPIIntegration(config)

        print(f"✅ LangChain LLM 시스템 초기화 성공")
        print(f"🤖 Provider: {config.provider}")
        print(f"📊 Model: {config.model_name}")

        # 테스트 데이터
        test_macro_data = {
            "VIX": pd.DataFrame({"Close": [20.5]}, index=[pd.Timestamp.now()]),
            "TNX": pd.DataFrame({"Close": [3.2]}, index=[pd.Timestamp.now()]),
        }

        test_market_metrics = {
            "probabilities": {"UP": 0.6, "DOWN": 0.2, "SIDEWAYS": 0.2},
            "stat_arb_signals": "neutral",
        }

        test_analysis_results = {
            "current_regime": "TRENDING_UP",
            "confidence": 0.75,
            "probabilities": {"UP": 0.6, "DOWN": 0.2, "SIDEWAYS": 0.2},
        }

        # 향상된 인사이트 획득 테스트
        print("\n🚀 향상된 인사이트 획득 테스트...")
        insights = llm_system.get_enhanced_insights(
            "TRENDING_UP", test_macro_data, test_market_metrics, test_analysis_results
        )

        print(f"✅ 인사이트 획득 성공")
        print(f"📊 API 통계: {llm_system.get_api_stats()}")

        # 결과 출력
        if "llm_enhanced_insights" in insights:
            llm_result = insights["llm_enhanced_insights"]
            print(f"\n🤖 LLM 종합 분석 결과:")
            print(
                f"   - 시장 역학: {llm_result.get('comprehensive_analysis', {}).get('market_dynamics', {})}"
            )
            print(f"   - 핵심 인사이트: {llm_result.get('key_insights', [])}")
            print(f"   - 신뢰도 수정자: {llm_result.get('confidence_modifier', 1.0)}")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_langchain_llm_integration()
