#!/usr/bin/env python3
"""
LLM API 통합 시스템

실제 LLM API (Bedrock, OpenAI 등)를 활용한 시장 분석 강화 시스템
기존 규칙 기반 시스템과 하이브리드로 동작하여 안정성과 성능을 모두 확보
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
import json
import time
import asyncio
from dataclasses import dataclass
import warnings

# LLM API 관련 import
try:
    import boto3

    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    warnings.warn("boto3 not available. Bedrock API will not be available.")

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("openai not available. OpenAI API will not be available.")

from .llm_insights import LLMPrivilegedInformationSystem


@dataclass
class LLMConfig:
    """LLM 설정 클래스"""

    provider: str = "bedrock"  # "bedrock", "openai", "hybrid"
    model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    api_key: Optional[str] = None
    region: str = "us-east-1"
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout: int = 30
    retry_attempts: int = 3
    fallback_to_rules: bool = True


class LLMAPIIntegration:
    """
    LLM API 통합 시스템

    실제 LLM API를 활용하여 시장 분석을 강화하는 시스템
    기존 규칙 기반 시스템과 하이브리드로 동작
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.logger = logging.getLogger(__name__)

        # 기존 규칙 기반 시스템 (fallback용)
        self.rule_based_system = LLMPrivilegedInformationSystem()

        # LLM API 클라이언트 초기화
        self.llm_client = self._initialize_llm_client()

        # 캐시 시스템 (API 호출 최적화)
        self.response_cache = {}
        self.cache_ttl = 300  # 5분 캐시

        # 성능 모니터링
        self.api_call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_response_time": 0.0,
        }

    def _initialize_llm_client(self) -> Optional[Any]:
        """LLM API 클라이언트 초기화"""
        try:
            if self.config.provider == "bedrock":
                if not BEDROCK_AVAILABLE:
                    self.logger.warning(
                        "Bedrock not available. Using rule-based system only."
                    )
                    return None

                return boto3.client("bedrock-runtime", region_name=self.config.region)

            elif self.config.provider == "openai":
                if not OPENAI_AVAILABLE:
                    self.logger.warning(
                        "OpenAI not available. Using rule-based system only."
                    )
                    return None

                if not self.config.api_key:
                    self.logger.warning(
                        "OpenAI API key not provided. Using rule-based system only."
                    )
                    return None

                return openai.OpenAI(api_key=self.config.api_key)

            else:
                self.logger.warning(f"Unknown LLM provider: {self.config.provider}")
                return None

        except Exception as e:
            self.logger.error(f"LLM client initialization failed: {e}")
            return None

    def get_enhanced_insights(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        향상된 인사이트 획득 (LLM API + 규칙 기반 하이브리드)
        """
        try:
            # 1. 규칙 기반 분석 (기본)
            rule_based_insights = self.rule_based_system.get_privileged_insights(
                current_regime, macro_data, market_metrics
            )

            # 2. LLM API 호출 (향상된 분석)
            if self.llm_client and self.config.provider != "rule_only":
                try:
                    llm_insights = self._call_llm_api(
                        current_regime, macro_data, market_metrics
                    )

                    # 3. 두 결과 융합
                    enhanced_insights = self._combine_insights(
                        rule_based_insights, llm_insights
                    )

                    self.logger.info("LLM API 통합 분석 완료")
                    return enhanced_insights

                except Exception as e:
                    self.logger.warning(f"LLM API 호출 실패, 규칙 기반 분석 사용: {e}")
                    self.api_call_stats["failed_calls"] += 1

            # LLM API 실패 시 규칙 기반만 사용
            return rule_based_insights

        except Exception as e:
            self.logger.error(f"Enhanced insights generation failed: {e}")
            return self._get_fallback_insights(
                current_regime, macro_data, market_metrics
            )

    def _call_llm_api(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM API 호출"""
        start_time = time.time()

        # 캐시 키 생성
        cache_key = self._generate_cache_key(current_regime, macro_data, market_metrics)

        # 캐시 확인
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            if time.time() - cached_response["timestamp"] < self.cache_ttl:
                self.logger.info("Using cached LLM response")
                return cached_response["data"]

        # 프롬프트 생성
        prompt = self._create_llm_prompt(current_regime, macro_data, market_metrics)

        # API 호출
        response = None
        for attempt in range(self.config.retry_attempts):
            try:
                if self.config.provider == "bedrock":
                    response = self._call_bedrock_api(prompt)
                elif self.config.provider == "openai":
                    response = self._call_openai_api(prompt)

                if response:
                    break

            except Exception as e:
                self.logger.warning(f"LLM API call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(2**attempt)  # 지수 백오프

        # 응답 처리
        if response:
            try:
                parsed_response = self._parse_llm_response(response)

                # 캐시 저장
                self.response_cache[cache_key] = {
                    "data": parsed_response,
                    "timestamp": time.time(),
                }

                # 통계 업데이트
                response_time = time.time() - start_time
                self.api_call_stats["successful_calls"] += 1
                self.api_call_stats["total_calls"] += 1
                self._update_avg_response_time(response_time)

                return parsed_response

            except Exception as e:
                self.logger.error(f"LLM response parsing failed: {e}")

        # API 호출 실패
        self.api_call_stats["failed_calls"] += 1
        self.api_call_stats["total_calls"] += 1
        raise Exception("LLM API call failed")

    def _call_bedrock_api(self, prompt: str) -> Optional[str]:
        """Bedrock API 호출"""
        try:
            response = self.llm_client.invoke_model(
                modelId=self.config.model_name,
                body=json.dumps(
                    {
                        "prompt": prompt,
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    }
                ),
            )

            result = json.loads(response["body"].read())
            return result.get("completion", "")

        except Exception as e:
            self.logger.error(f"Bedrock API call failed: {e}")
            return None

    def _call_openai_api(self, prompt: str) -> Optional[str]:
        """OpenAI API 호출"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return None

    def _create_llm_prompt(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
    ) -> str:
        """LLM 프롬프트 생성"""

        # 매크로 데이터 요약
        market_summary = self._create_market_summary(macro_data)

        # 시장 메트릭 요약
        metrics_summary = self._create_metrics_summary(market_metrics)

        prompt = f"""
당신은 금융 시장 분석 전문가입니다. 현재 시장 상황을 분석하고 투자 전략을 제시해주세요.

## 현재 시장 상황
- 감지된 시장 체제: {current_regime}
- VIX 수준: {market_summary.get('vix_level', 'N/A')}
- 10년 국채 수익률: {market_summary.get('tnx_level', 'N/A')}
- TIPS 수준: {market_summary.get('tips_level', 'N/A')}
- 달러 인덱스: {market_summary.get('dxy_level', 'N/A')}

## 시장 메트릭
{metrics_summary}

## 분석 요청사항
다음 질문들에 답변해주세요:

1. **Regime 일관성**: 현재 감지된 시장 체제가 경제 환경과 일치합니까?
2. **위험 요인**: 현재 시장에서 주목해야 할 위험 요인들은 무엇입니까?
3. **투자 전략**: 현재 상황에서 어떤 투자 전략을 추천하시겠습니까?
4. **섹터 로테이션**: 어떤 섹터에 집중해야 할까요?
5. **리스크 관리**: 어떤 리스크 관리 전략을 사용해야 할까요?

## 답변 형식
다음 JSON 형태로 답변해주세요:

{{
    "regime_validation": {{
        "consistency": 0.0-1.0,
        "supporting_factors": ["요인1", "요인2"],
        "conflicting_factors": ["요인1", "요인2"],
        "alternative_regimes": ["대안1", "대안2"]
    }},
    "risk_analysis": {{
        "identified_risks": ["위험1", "위험2"],
        "risk_level": "low/moderate/high",
        "mitigation_strategies": ["전략1", "전략2"]
    }},
    "investment_strategy": {{
        "primary_strategy": "전략명",
        "sector_rotation": ["섹터1", "섹터2"],
        "position_sizing": "conservative/moderate/aggressive",
        "time_horizon": "short/medium/long"
    }},
    "confidence_modifier": 0.5-1.5,
    "strategic_recommendations": ["추천1", "추천2", "추천3"]
}}

분석은 객관적이고 실용적이어야 하며, 구체적인 수치와 근거를 포함해주세요.
"""

        return prompt

    def _create_market_summary(
        self, macro_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """매크로 데이터 요약"""
        summary = {}

        try:
            if "^VIX" in macro_data and not macro_data["^VIX"].empty:
                close_col = (
                    "close" if "close" in macro_data["^VIX"].columns else "Close"
                )
                summary["vix_level"] = round(macro_data["^VIX"][close_col].iloc[-1], 2)

            if "^TNX" in macro_data and not macro_data["^TNX"].empty:
                close_col = (
                    "close" if "close" in macro_data["^TNX"].columns else "Close"
                )
                summary["tnx_level"] = round(macro_data["^TNX"][close_col].iloc[-1], 2)

            if "^TIP" in macro_data and not macro_data["^TIP"].empty:
                close_col = (
                    "close" if "close" in macro_data["^TIP"].columns else "Close"
                )
                summary["tips_level"] = round(macro_data["^TIP"][close_col].iloc[-1], 2)

            if "DX-Y.NYB" in macro_data and not macro_data["DX-Y.NYB"].empty:
                close_col = (
                    "close" if "close" in macro_data["DX-Y.NYB"].columns else "Close"
                )
                summary["dxy_level"] = round(
                    macro_data["DX-Y.NYB"][close_col].iloc[-1], 2
                )

        except Exception as e:
            self.logger.warning(f"Market summary creation failed: {e}")

        return summary

    def _create_metrics_summary(self, market_metrics: Dict[str, Any]) -> str:
        """시장 메트릭 요약"""
        summary_parts = []

        try:
            if "current_probabilities" in market_metrics:
                probs = market_metrics["current_probabilities"]
                summary_parts.append(f"- 시장 체제 확률: {dict(probs)}")

            if "stat_arb_signal" in market_metrics:
                signal = market_metrics["stat_arb_signal"]
                summary_parts.append(
                    f"- 통계적 차익거래 신호: {signal.get('direction', 'N/A')} (강도: {signal.get('signal_strength', 0):.3f})"
                )

            if "vix_level" in market_metrics:
                summary_parts.append(f"- VIX 수준: {market_metrics['vix_level']:.2f}")

        except Exception as e:
            self.logger.warning(f"Metrics summary creation failed: {e}")

        return "\n".join(summary_parts) if summary_parts else "메트릭 데이터 없음"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            # JSON 추출 시도
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)

                # 필수 필드 검증 및 기본값 설정
                validated_response = {
                    "regime_validation": parsed.get(
                        "regime_validation",
                        {
                            "consistency": 0.5,
                            "supporting_factors": [],
                            "conflicting_factors": [],
                            "alternative_regimes": [],
                        },
                    ),
                    "risk_analysis": parsed.get(
                        "risk_analysis",
                        {
                            "identified_risks": [],
                            "risk_level": "moderate",
                            "mitigation_strategies": [],
                        },
                    ),
                    "investment_strategy": parsed.get(
                        "investment_strategy",
                        {
                            "primary_strategy": "balanced",
                            "sector_rotation": [],
                            "position_sizing": "moderate",
                            "time_horizon": "medium",
                        },
                    ),
                    "confidence_modifier": parsed.get("confidence_modifier", 1.0),
                    "strategic_recommendations": parsed.get(
                        "strategic_recommendations", []
                    ),
                }

                return validated_response
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            self.logger.error(f"LLM response parsing failed: {e}")
            self.logger.debug(f"Raw response: {response}")

            # 파싱 실패 시 기본 응답 반환
            return self._get_default_llm_response()

    def _get_default_llm_response(self) -> Dict[str, Any]:
        """기본 LLM 응답"""
        return {
            "regime_validation": {
                "consistency": 0.5,
                "supporting_factors": [],
                "conflicting_factors": [],
                "alternative_regimes": [],
            },
            "risk_analysis": {
                "identified_risks": [],
                "risk_level": "moderate",
                "mitigation_strategies": [],
            },
            "investment_strategy": {
                "primary_strategy": "balanced",
                "sector_rotation": [],
                "position_sizing": "moderate",
                "time_horizon": "medium",
            },
            "confidence_modifier": 1.0,
            "strategic_recommendations": [],
        }

    def _combine_insights(
        self, rule_insights: Dict[str, Any], llm_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """규칙 기반과 LLM 인사이트 융합"""
        combined = rule_insights.copy()

        # LLM의 regime 검증 결과 반영
        if "regime_validation" in llm_insights:
            llm_validation = llm_insights["regime_validation"]
            rule_validation = combined.get("regime_validation", {})

            # 일관성 점수 융합 (가중 평균)
            rule_consistency = rule_validation.get("consistency", 0.5)
            llm_consistency = llm_validation.get("consistency", 0.5)
            combined_consistency = (rule_consistency * 0.6) + (llm_consistency * 0.4)

            combined["regime_validation"] = {
                "consistency": combined_consistency,
                "supporting_factors": rule_validation.get("supporting_factors", [])
                + llm_validation.get("supporting_factors", []),
                "conflicting_factors": rule_validation.get("conflicting_factors", [])
                + llm_validation.get("conflicting_factors", []),
                "alternative_regimes": list(
                    set(
                        rule_validation.get("alternative_regimes", [])
                        + llm_validation.get("alternative_regimes", [])
                    )
                ),
            }

        # 위험 분석 융합
        if "risk_analysis" in llm_insights:
            llm_risks = llm_insights["risk_analysis"]
            rule_risks = combined.get("risk_adjustments", {})

            combined["risk_adjustments"] = {
                "identified_risks": list(
                    set(
                        rule_risks.get("identified_risks", [])
                        + llm_risks.get("identified_risks", [])
                    )
                ),
                "risk_level": self._determine_risk_level(
                    rule_risks.get("risk_level", "moderate"),
                    llm_risks.get("risk_level", "moderate"),
                ),
                "mitigation_strategies": list(
                    set(
                        rule_risks.get("mitigation_strategies", [])
                        + llm_risks.get("mitigation_strategies", [])
                    )
                ),
            }

        # 전략적 추천사항 융합
        if "strategic_recommendations" in llm_insights:
            rule_recommendations = combined.get("strategic_recommendations", [])
            llm_recommendations = llm_insights["strategic_recommendations"]

            # 중복 제거하면서 융합
            all_recommendations = rule_recommendations + llm_recommendations
            unique_recommendations = list(
                dict.fromkeys(all_recommendations)
            )  # 순서 유지하면서 중복 제거

            combined["strategic_recommendations"] = unique_recommendations

        # 신뢰도 수정자 융합
        if "confidence_modifier" in llm_insights:
            rule_modifiers = combined.get("confidence_modifiers", [1.0])
            llm_modifier = llm_insights["confidence_modifier"]

            # LLM 수정자 추가
            rule_modifiers.append(llm_modifier)
            combined["confidence_modifiers"] = rule_modifiers

        # 투자 전략 정보 추가
        if "investment_strategy" in llm_insights:
            combined["investment_strategy"] = llm_insights["investment_strategy"]

        return combined

    def _determine_risk_level(self, rule_level: str, llm_level: str) -> str:
        """위험 수준 결정 (두 수준 중 더 높은 것 선택)"""
        risk_hierarchy = {"low": 1, "moderate": 2, "high": 3}

        rule_score = risk_hierarchy.get(rule_level, 2)
        llm_score = risk_hierarchy.get(llm_level, 2)

        max_score = max(rule_score, llm_score)

        for level, score in risk_hierarchy.items():
            if score == max_score:
                return level

        return "moderate"

    def _generate_cache_key(
        self,
        current_regime: str,
        macro_data: Dict[str, pd.DataFrame],
        market_metrics: Dict[str, Any],
    ) -> str:
        """캐시 키 생성"""
        # 간단한 해시 기반 캐시 키
        key_parts = [
            current_regime,
            str(hash(str(macro_data.keys()))),
            str(hash(str(market_metrics.keys()))),
        ]
        return "_".join(key_parts)

    def _update_avg_response_time(self, new_response_time: float):
        """평균 응답 시간 업데이트"""
        current_avg = self.api_call_stats["avg_response_time"]
        total_calls = self.api_call_stats["successful_calls"]

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
        """fallback 인사이트"""
        return self.rule_based_system.get_privileged_insights(
            current_regime, macro_data, market_metrics
        )

    def get_api_stats(self) -> Dict[str, Any]:
        """API 통계 반환"""
        stats = self.api_call_stats.copy()

        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def clear_cache(self):
        """캐시 클리어"""
        self.response_cache.clear()
        self.logger.info("LLM response cache cleared")

    def update_config(self, new_config: LLMConfig):
        """설정 업데이트"""
        self.config = new_config
        self.llm_client = self._initialize_llm_client()
        self.logger.info("LLM configuration updated")


# 사용 예시 및 테스트 함수
def test_llm_api_integration():
    """LLM API 통합 테스트"""

    # 설정
    config = LLMConfig(
        provider="hybrid",  # "bedrock", "openai", "hybrid", "rule_only"
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        api_key="your_api_key_here",  # OpenAI 사용 시
        region="us-east-1",
    )

    # 시스템 초기화
    llm_system = LLMAPIIntegration(config)

    # 테스트 데이터
    macro_data = {
        "^VIX": pd.DataFrame({"close": [25.5]}),
        "^TNX": pd.DataFrame({"close": [4.2]}),
        "^TIP": pd.DataFrame({"close": [105.3]}),
    }

    market_metrics = {
        "current_probabilities": {
            "TRENDING_UP": 0.6,
            "TRENDING_DOWN": 0.2,
            "VOLATILE": 0.15,
            "SIDEWAYS": 0.05,
        },
        "vix_level": 25.5,
    }

    # 향상된 인사이트 획득
    insights = llm_system.get_enhanced_insights(
        current_regime="TRENDING_UP",
        macro_data=macro_data,
        market_metrics=market_metrics,
    )

    print("향상된 인사이트 결과:")
    print(json.dumps(insights, indent=2, ensure_ascii=False))

    # API 통계 확인
    stats = llm_system.get_api_stats()
    print(f"\nAPI 통계: {stats}")


if __name__ == "__main__":
    test_llm_api_integration()
