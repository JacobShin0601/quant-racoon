#!/usr/bin/env python3
"""
LLM API 통합 시스템 사용 예시

실제 LLM API (Bedrock, OpenAI)를 활용한 시장 분석 강화 시스템 테스트
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.enhancements import LLMAPIIntegration, LLMConfig


def create_sample_macro_data():
    """샘플 매크로 데이터 생성"""
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

    # VIX 데이터 (변동성 지수)
    vix_data = pd.DataFrame(
        {
            "close": np.random.normal(20, 5, len(dates)),
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    # 10년 국채 수익률
    tnx_data = pd.DataFrame(
        {
            "close": np.random.normal(4.0, 0.5, len(dates)),
            "volume": np.random.randint(500, 5000, len(dates)),
        },
        index=dates,
    )

    # TIPS (인플레이션 보호 국채)
    tips_data = pd.DataFrame(
        {
            "close": np.random.normal(105, 2, len(dates)),
            "volume": np.random.randint(200, 2000, len(dates)),
        },
        index=dates,
    )

    # 달러 인덱스
    dxy_data = pd.DataFrame(
        {
            "close": np.random.normal(100, 3, len(dates)),
            "volume": np.random.randint(100, 1000, len(dates)),
        },
        index=dates,
    )

    return {"^VIX": vix_data, "^TNX": tnx_data, "^TIP": tips_data, "DX-Y.NYB": dxy_data}


def create_sample_market_metrics():
    """샘플 시장 메트릭 생성"""
    return {
        "current_probabilities": {
            "TRENDING_UP": 0.65,
            "TRENDING_DOWN": 0.15,
            "VOLATILE": 0.12,
            "SIDEWAYS": 0.08,
        },
        "stat_arb_signal": {
            "direction": "BULLISH",
            "signal_strength": 0.72,
            "confidence": 0.85,
            "individual_signals": {"XRT": 0.8, "XLF": 0.7, "GTX": 0.6, "VIX": 0.5},
        },
        "vix_level": 22.5,
        "regime_shift_detected": False,
        "confidence": 0.78,
    }


def test_rule_only_mode():
    """규칙 기반 모드 테스트"""
    print("🔧 규칙 기반 모드 테스트")
    print("=" * 50)

    # 설정
    config = LLMConfig(provider="rule_only", fallback_to_rules=True)

    # 시스템 초기화
    llm_system = LLMAPIIntegration(config)

    # 샘플 데이터
    macro_data = create_sample_macro_data()
    market_metrics = create_sample_market_metrics()

    # 인사이트 획득
    insights = llm_system.get_enhanced_insights(
        current_regime="TRENDING_UP",
        macro_data=macro_data,
        market_metrics=market_metrics,
    )

    print("📊 규칙 기반 분석 결과:")
    print(f"Regime 일관성: {insights['regime_validation']['consistency']:.3f}")
    print(f"지지 요인: {insights['regime_validation']['supporting_factors']}")
    print(f"충돌 요인: {insights['regime_validation']['conflicting_factors']}")
    print(f"전략적 추천: {insights['strategic_recommendations'][:3]}")  # 처음 3개만
    print()


def test_hybrid_mode():
    """하이브리드 모드 테스트 (LLM API + 규칙 기반)"""
    print("🤖 하이브리드 모드 테스트")
    print("=" * 50)

    # 설정 (실제 API 키가 없으면 규칙 기반으로 fallback)
    config = LLMConfig(
        provider="hybrid",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        api_key=None,  # 실제 API 키 필요
        region="us-east-1",
        fallback_to_rules=True,
    )

    # 시스템 초기화
    llm_system = LLMAPIIntegration(config)

    # 샘플 데이터
    macro_data = create_sample_macro_data()
    market_metrics = create_sample_market_metrics()

    # 인사이트 획득
    insights = llm_system.get_enhanced_insights(
        current_regime="TRENDING_UP",
        macro_data=macro_data,
        market_metrics=market_metrics,
    )

    print("📊 하이브리드 분석 결과:")
    print(f"Regime 일관성: {insights['regime_validation']['consistency']:.3f}")
    print(f"지지 요인: {insights['regime_validation']['supporting_factors']}")
    print(f"충돌 요인: {insights['regime_validation']['conflicting_factors']}")
    print(f"전략적 추천: {insights['strategic_recommendations'][:3]}")

    # API 통계 확인
    stats = llm_system.get_api_stats()
    print(f"\n📈 API 통계: {stats}")
    print()


def test_different_regimes():
    """다양한 시장 체제 테스트"""
    print("🎯 다양한 시장 체제 테스트")
    print("=" * 50)

    config = LLMConfig(provider="rule_only")
    llm_system = LLMAPIIntegration(config)

    macro_data = create_sample_macro_data()
    market_metrics = create_sample_market_metrics()

    regimes = ["TRENDING_UP", "TRENDING_DOWN", "VOLATILE", "SIDEWAYS"]

    for regime in regimes:
        print(f"\n📊 {regime} 체제 분석:")

        # 메트릭 조정
        adjusted_metrics = market_metrics.copy()
        if regime == "TRENDING_UP":
            adjusted_metrics["current_probabilities"] = {
                "TRENDING_UP": 0.75,
                "TRENDING_DOWN": 0.10,
                "VOLATILE": 0.10,
                "SIDEWAYS": 0.05,
            }
        elif regime == "TRENDING_DOWN":
            adjusted_metrics["current_probabilities"] = {
                "TRENDING_UP": 0.10,
                "TRENDING_DOWN": 0.75,
                "VOLATILE": 0.10,
                "SIDEWAYS": 0.05,
            }
        elif regime == "VOLATILE":
            adjusted_metrics["current_probabilities"] = {
                "TRENDING_UP": 0.20,
                "TRENDING_DOWN": 0.20,
                "VOLATILE": 0.50,
                "SIDEWAYS": 0.10,
            }
            adjusted_metrics["vix_level"] = 35.0
        else:  # SIDEWAYS
            adjusted_metrics["current_probabilities"] = {
                "TRENDING_UP": 0.25,
                "TRENDING_DOWN": 0.25,
                "VOLATILE": 0.20,
                "SIDEWAYS": 0.30,
            }
            adjusted_metrics["vix_level"] = 15.0

        insights = llm_system.get_enhanced_insights(
            current_regime=regime,
            macro_data=macro_data,
            market_metrics=adjusted_metrics,
        )

        print(f"  일관성: {insights['regime_validation']['consistency']:.3f}")
        print(f"  위험 수준: {insights['risk_adjustments']['risk_level']}")
        print(
            f"  추천 전략: {insights['strategic_recommendations'][0] if insights['strategic_recommendations'] else 'N/A'}"
        )


def test_api_configuration():
    """API 설정 테스트"""
    print("⚙️ API 설정 테스트")
    print("=" * 50)

    # Bedrock 설정
    bedrock_config = LLMConfig(
        provider="bedrock",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        region="us-east-1",
    )

    # OpenAI 설정
    openai_config = LLMConfig(
        provider="openai", model_name="gpt-4", api_key="your_openai_api_key_here"
    )

    # 하이브리드 설정
    hybrid_config = LLMConfig(
        provider="hybrid",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_to_rules=True,
    )

    configs = [
        ("Bedrock", bedrock_config),
        ("OpenAI", openai_config),
        ("Hybrid", hybrid_config),
    ]

    for name, config in configs:
        print(f"\n🔧 {name} 설정:")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model_name}")
        print(f"  Region: {config.region}")
        print(f"  Max Tokens: {config.max_tokens}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Fallback to Rules: {config.fallback_to_rules}")

        # 시스템 초기화 테스트
        try:
            llm_system = LLMAPIIntegration(config)
            print(f"  ✅ 초기화 성공")
        except Exception as e:
            print(f"  ❌ 초기화 실패: {e}")


def test_performance_monitoring():
    """성능 모니터링 테스트"""
    print("📊 성능 모니터링 테스트")
    print("=" * 50)

    config = LLMConfig(provider="rule_only")
    llm_system = LLMAPIIntegration(config)

    macro_data = create_sample_macro_data()
    market_metrics = create_sample_market_metrics()

    # 여러 번 호출하여 통계 생성
    for i in range(5):
        insights = llm_system.get_enhanced_insights(
            current_regime="TRENDING_UP",
            macro_data=macro_data,
            market_metrics=market_metrics,
        )

        stats = llm_system.get_api_stats()
        print(
            f"호출 {i+1}: 성공률 {stats['success_rate']:.2f}, 평균 응답시간 {stats['avg_response_time']:.3f}s"
        )

    # 최종 통계
    final_stats = llm_system.get_api_stats()
    print(f"\n📈 최종 통계:")
    print(f"  총 호출: {final_stats['total_calls']}")
    print(f"  성공: {final_stats['successful_calls']}")
    print(f"  실패: {final_stats['failed_calls']}")
    print(f"  성공률: {final_stats['success_rate']:.2%}")
    print(f"  평균 응답시간: {final_stats['avg_response_time']:.3f}s")


def main():
    """메인 테스트 함수"""
    print("🚀 LLM API 통합 시스템 테스트 시작")
    print("=" * 60)

    try:
        # 1. 규칙 기반 모드 테스트
        test_rule_only_mode()

        # 2. 하이브리드 모드 테스트
        test_hybrid_mode()

        # 3. 다양한 시장 체제 테스트
        test_different_regimes()

        # 4. API 설정 테스트
        test_api_configuration()

        # 5. 성능 모니터링 테스트
        test_performance_monitoring()

        print("\n✅ 모든 테스트 완료!")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
