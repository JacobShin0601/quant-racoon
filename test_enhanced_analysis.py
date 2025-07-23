#!/usr/bin/env python3
"""
고도화된 시장 분석 시스템 테스트 스크립트

다양한 분석 유형과 LLM API 통합을 테스트합니다.
"""

import sys
import os
import json
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.market_sensor import MarketSensor
from src.agent.enhancements import LLMConfig


def test_basic_analysis():
    """기본 분석 테스트"""
    print("🔧 기본 분석 테스트")
    print("=" * 50)

    sensor = MarketSensor()
    analysis = sensor.get_current_market_analysis(
        use_optimized_params=True, use_ml_model=True
    )

    print(f"현재 체제: {analysis.get('current_regime', 'N/A')}")
    print(f"신뢰도: {analysis.get('confidence', 0.5):.3f}")
    print(f"추천 전략: {analysis.get('recommendation', {}).get('strategy', 'N/A')}")
    print()


def test_enhanced_analysis():
    """고도화된 분석 테스트 (LLM API 없이)"""
    print("🚀 고도화된 분석 테스트")
    print("=" * 50)

    sensor = MarketSensor()
    analysis = sensor.get_enhanced_market_analysis(
        use_optimized_params=True, use_ml_model=True, enable_advanced_features=True
    )

    print(f"현재 체제: {analysis.get('current_regime', 'N/A')}")

    # RLMF 분석 결과
    if "rlmf_analysis" in analysis:
        rlmf = analysis["rlmf_analysis"]
        print(f"RLMF 분석: {len(rlmf)} 개 항목")

        if "statistical_arbitrage" in rlmf:
            sa = rlmf["statistical_arbitrage"]
            print(
                f"  Statistical Arbitrage: {sa.get('direction', 'N/A')} (신뢰도: {sa.get('confidence', 0):.3f})"
            )

    # 신뢰도 분석 결과
    if "confidence_analysis" in analysis:
        conf = analysis["confidence_analysis"]
        final_conf = conf.get("confidence_result", {}).get("adjusted_confidence", 0.5)
        print(f"조정된 신뢰도: {final_conf:.3f}")

    # Regime 감지 결과
    if "regime_detection" in analysis:
        regime = analysis["regime_detection"]
        shift_det = regime.get("regime_shift_detection", {})
        if shift_det.get("regime_shift_detected", False):
            print("⚠️ 시장 체제 전환 감지됨!")

    print()


def test_llm_api_analysis():
    """LLM API 통합 분석 테스트"""
    print("🤖 LLM API 통합 분석 테스트")
    print("=" * 50)

    # LLM 설정 (규칙 기반 fallback)
    llm_config = LLMConfig(
        provider="hybrid",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_to_rules=True,
    )

    sensor = MarketSensor(enable_llm_api=True, llm_config=llm_config)

    analysis = sensor.get_enhanced_market_analysis(
        use_optimized_params=True, use_ml_model=True, enable_advanced_features=True
    )

    print(f"현재 체제: {analysis.get('current_regime', 'N/A')}")

    # LLM API 인사이트
    if "llm_api_insights" in analysis:
        llm_insights = analysis["llm_api_insights"]
        print(f"LLM API 인사이트: {len(llm_insights)} 개 항목")

        if "regime_validation" in llm_insights:
            validation = llm_insights["regime_validation"]
            print(f"  Regime 일관성: {validation.get('consistency', 0.5):.3f}")

        if "strategic_recommendations" in llm_insights:
            recs = llm_insights["strategic_recommendations"]
            print(f"  전략적 추천: {len(recs)} 개")
            for i, rec in enumerate(recs[:3], 1):
                print(f"    {i}. {rec}")

    # API 통계
    if sensor.llm_api_system:
        stats = sensor.get_llm_api_stats()
        print(f"API 통계: 성공률 {stats.get('success_rate', 0):.2%}")

    print()


def test_full_analysis():
    """전체 기능 통합 분석 테스트"""
    print("🎯 전체 기능 통합 분석 테스트")
    print("=" * 50)

    # LLM 설정
    llm_config = LLMConfig(
        provider="hybrid",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        fallback_to_rules=True,
    )

    sensor = MarketSensor(enable_llm_api=True, llm_config=llm_config)

    analysis = sensor.get_enhanced_market_analysis(
        use_optimized_params=True, use_ml_model=True, enable_advanced_features=True
    )

    # 결과 요약
    print("📊 전체 분석 결과 요약:")
    print(f"현재 체제: {analysis.get('current_regime', 'N/A')}")

    # 최종 신뢰도
    if "final_confidence" in analysis:
        final_conf = analysis["final_confidence"]
        print(f"최종 신뢰도: {final_conf.get('final_confidence', 0.5):.3f}")
        print(f"신뢰도 수준: {final_conf.get('confidence_level', 'MEDIUM')}")

    # 고도화된 추천
    if "enhanced_recommendations" in analysis:
        recs = analysis["enhanced_recommendations"]
        print(f"주요 전략: {recs.get('primary_strategy', 'N/A')}")
        print(f"포지션 사이징: {recs.get('position_sizing', 'N/A')}")

        considerations = recs.get("key_considerations", [])
        if considerations:
            print("주요 고려사항:")
            for i, consideration in enumerate(considerations, 1):
                print(f"  {i}. {consideration}")

    # 성능 통계
    print("\n📈 성능 통계:")
    if sensor.llm_api_system:
        stats = sensor.get_llm_api_stats()
        print(f"LLM API 호출: {stats.get('total_calls', 0)}")
        print(f"성공률: {stats.get('success_rate', 0):.2%}")
        print(f"평균 응답시간: {stats.get('avg_response_time', 0):.3f}초")

    print()


def test_llm_api_configurations():
    """다양한 LLM API 설정 테스트"""
    print("⚙️ LLM API 설정 테스트")
    print("=" * 50)

    configs = [
        (
            "Bedrock",
            LLMConfig(
                provider="bedrock", model_name="anthropic.claude-3-sonnet-20240229-v1:0"
            ),
        ),
        ("OpenAI", LLMConfig(provider="openai", model_name="gpt-4")),
        (
            "Hybrid",
            LLMConfig(
                provider="hybrid", model_name="anthropic.claude-3-sonnet-20240229-v1:0"
            ),
        ),
        ("Rule-only", LLMConfig(provider="rule_only")),
    ]

    for name, config in configs:
        print(f"\n🔧 {name} 설정 테스트:")
        try:
            sensor = MarketSensor(enable_llm_api=True, llm_config=config)
            print(f"  ✅ 초기화 성공")

            # 간단한 분석 테스트
            analysis = sensor.get_enhanced_market_analysis(
                enable_advanced_features=True
            )
            print(f"  ✅ 분석 완료")

        except Exception as e:
            print(f"  ❌ 실패: {e}")

    print()


def save_test_results():
    """테스트 결과 저장"""
    print("💾 테스트 결과 저장")
    print("=" * 50)

    # 결과 디렉토리 생성
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 기본 분석 결과
    sensor = MarketSensor()
    basic_analysis = sensor.get_current_market_analysis()

    with open(
        f"{results_dir}/basic_analysis_{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(basic_analysis, f, indent=2, ensure_ascii=False, default=str)

    # 고도화된 분석 결과
    enhanced_analysis = sensor.get_enhanced_market_analysis(
        enable_advanced_features=True
    )

    with open(
        f"{results_dir}/enhanced_analysis_{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False, default=str)

    print(f"✅ 결과 저장 완료: {results_dir}/")
    print()


def main():
    """메인 테스트 함수"""
    print("🚀 고도화된 시장 분석 시스템 테스트 시작")
    print("=" * 60)

    try:
        # 1. 기본 분석 테스트
        test_basic_analysis()

        # 2. 고도화된 분석 테스트
        test_enhanced_analysis()

        # 3. LLM API 통합 분석 테스트
        test_llm_api_analysis()

        # 4. 전체 기능 통합 분석 테스트
        test_full_analysis()

        # 5. LLM API 설정 테스트
        test_llm_api_configurations()

        # 6. 결과 저장
        save_test_results()

        print("✅ 모든 테스트 완료!")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
