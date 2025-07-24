#!/usr/bin/env python3
"""
LangChain 기반 LLM API 통합 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.enhancements.llm_api_integration import LLMAPIIntegration, LLMConfig


def test_langchain_llm_integration():
    """LangChain LLM 통합 테스트"""
    print("🧪 LangChain LLM 통합 테스트")
    print("=" * 60)

    # 테스트 설정
    config = LLMConfig(
        provider="bedrock",
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens=2000,
        temperature=0.1
    )

    try:
        # LLM API 통합 시스템 초기화
        print("🔧 LLM API 통합 시스템 초기화 중...")
        llm_system = LLMAPIIntegration(config)
        
        print(f"✅ LangChain LLM 시스템 초기화 성공")
        print(f"🤖 Provider: {config.provider}")
        print(f"📊 Model: {config.model_name}")
        
        # 테스트 데이터 생성
        print("\n📊 테스트 데이터 생성 중...")
        test_macro_data = {
            "VIX": pd.DataFrame({
                "Close": [20.5, 21.2, 19.8, 22.1, 20.9],
                "Volume": [1000000, 1200000, 900000, 1100000, 1050000]
            }, index=pd.date_range('2024-01-01', periods=5, freq='D')),
            "TNX": pd.DataFrame({
                "Close": [3.2, 3.4, 3.1, 3.5, 3.3],
                "Volume": [500000, 600000, 450000, 550000, 520000]
            }, index=pd.date_range('2024-01-01', periods=5, freq='D')),
            "TIPS": pd.DataFrame({
                "Close": [105.3, 105.8, 104.9, 106.2, 105.5],
                "Volume": [300000, 350000, 280000, 320000, 310000]
            }, index=pd.date_range('2024-01-01', periods=5, freq='D')),
            "DXY": pd.DataFrame({
                "Close": [102.5, 102.8, 102.1, 103.2, 102.9],
                "Volume": [800000, 850000, 780000, 900000, 870000]
            }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        }
        
        test_market_metrics = {
            "probabilities": {"UP": 0.6, "DOWN": 0.2, "SIDEWAYS": 0.2},
            "stat_arb_signals": "neutral",
            "rsi": 65.5,
            "macd": 0.25,
            "volume_ratio": 1.2
        }
        
        test_analysis_results = {
            "current_regime": "TRENDING_UP",
            "confidence": 0.75,
            "probabilities": {"UP": 0.6, "DOWN": 0.2, "SIDEWAYS": 0.2},
            "optimization_performance": {
                "sharpe_ratio": 1.85,
                "total_return": 0.12,
                "max_drawdown": -0.08
            },
            "validation_results": {
                "backtest_accuracy": 0.72,
                "out_of_sample_performance": 0.68
            }
        }
        
        # 향상된 인사이트 획득 테스트
        print("\n🚀 향상된 인사이트 획득 테스트...")
        start_time = datetime.now()
        
        insights = llm_system.get_enhanced_insights(
            "TRENDING_UP",
            test_macro_data,
            test_market_metrics,
            test_analysis_results
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 인사이트 획득 완료 (소요시간: {duration:.2f}초)")
        
        # API 통계 출력
        api_stats = llm_system.get_api_stats()
        print(f"\n📊 API 통계:")
        print(f"   - 총 호출 수: {api_stats['total_calls']}")
        print(f"   - 성공 호출 수: {api_stats['successful_calls']}")
        print(f"   - 실패 호출 수: {api_stats['failed_calls']}")
        print(f"   - 성공률: {api_stats['success_rate']:.1f}%")
        print(f"   - 평균 응답시간: {api_stats['avg_response_time']:.2f}초")
        
        # 결과 분석
        print(f"\n🔍 결과 분석:")
        
        if "llm_enhanced_insights" in insights:
            llm_result = insights["llm_enhanced_insights"]
            print(f"✅ LLM 향상된 인사이트 획득 성공")
            
            # 시장 역학 분석
            if "comprehensive_analysis" in llm_result:
                comp_analysis = llm_result["comprehensive_analysis"]
                market_dynamics = comp_analysis.get("market_dynamics", {})
                print(f"\n📈 시장 역학:")
                print(f"   - 주요 동인: {market_dynamics.get('primary_drivers', [])}")
                print(f"   - 변동성 요인: {market_dynamics.get('volatility_factors', [])}")
                print(f"   - 트렌드 강도: {market_dynamics.get('trend_strength', 'N/A')}")
                print(f"   - 모멘텀 품질: {market_dynamics.get('momentum_quality', 'N/A')}")
            
            # 리스크 평가
            if "risk_assessment" in llm_result:
                risk_assessment = llm_result["risk_assessment"]
                print(f"\n⚠️ 리스크 평가:")
                print(f"   - 단기 위험: {risk_assessment.get('short_term_risks', [])}")
                print(f"   - 중기 위험: {risk_assessment.get('medium_term_risks', [])}")
                print(f"   - 장기 위험: {risk_assessment.get('long_term_risks', [])}")
            
            # 전략적 제언
            if "strategic_recommendations" in llm_result:
                strategic_rec = llm_result["strategic_recommendations"]
                portfolio_alloc = strategic_rec.get("portfolio_allocation", {})
                print(f"\n💼 포트폴리오 배분:")
                print(f"   - 주식: {portfolio_alloc.get('equity_allocation', 'N/A')}")
                print(f"   - 채권: {portfolio_alloc.get('bond_allocation', 'N/A')}")
                print(f"   - 현금: {portfolio_alloc.get('cash_allocation', 'N/A')}")
                print(f"   - 대체자산: {portfolio_alloc.get('alternative_allocation', 'N/A')}")
            
            # 시나리오 분석
            if "scenario_analysis" in llm_result:
                scenario_analysis = llm_result["scenario_analysis"]
                print(f"\n🎯 시나리오 분석:")
                for scenario, details in scenario_analysis.items():
                    if isinstance(details, dict):
                        prob = details.get('probability', 0)
                        print(f"   - {scenario}: {prob:.1%}")
            
            # 핵심 인사이트
            if "key_insights" in llm_result:
                key_insights = llm_result["key_insights"]
                print(f"\n💡 핵심 인사이트:")
                for i, insight in enumerate(key_insights, 1):
                    print(f"   {i}. {insight}")
            
            # 신뢰도 수정자
            confidence_modifier = llm_result.get("confidence_modifier", 1.0)
            print(f"\n🎚️ 신뢰도 수정자: {confidence_modifier:.2f}")
            
        else:
            print("❌ LLM 향상된 인사이트 없음 - 규칙 기반 분석만 사용됨")
        
        # 규칙 기반 인사이트 확인
        if "rule_based_insights" in insights:
            rule_insights = insights["rule_based_insights"]
            print(f"\n📋 규칙 기반 인사이트:")
            print(f"   - 신뢰도: {rule_insights.get('confidence', 'N/A')}")
            print(f"   - 전략적 제언: {rule_insights.get('strategic_recommendations', [])}")
        
        # 조정된 신뢰도
        if "adjusted_confidence" in insights:
            print(f"\n🎚️ 조정된 신뢰도: {insights['adjusted_confidence']:.3f}")
        
        print(f"\n🎉 LangChain LLM 통합 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_config_parsing():
    """LLM 설정 파싱 테스트"""
    print("\n🔧 LLM 설정 파싱 테스트")
    print("=" * 40)
    
    try:
        # 딕셔너리로 설정 전달
        config_dict = {
            "provider": "bedrock",
            "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        llm_system = LLMAPIIntegration(config_dict)
        print("✅ 딕셔너리 설정 파싱 성공")
        
        # LLMConfig 객체로 설정 전달
        config_obj = LLMConfig(
            provider="bedrock",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=2000,
            temperature=0.1
        )
        
        llm_system2 = LLMAPIIntegration(config_obj)
        print("✅ LLMConfig 객체 설정 파싱 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 설정 파싱 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    print("🚀 LangChain 기반 LLM API 통합 테스트 시작")
    print("=" * 80)
    
    # 설정 파싱 테스트
    config_test = test_llm_config_parsing()
    
    # 메인 통합 테스트
    main_test = test_langchain_llm_integration()
    
    # 최종 결과
    print("\n" + "=" * 80)
    if config_test and main_test:
        print("🎉 모든 테스트 통과!")
    else:
        print("❌ 일부 테스트 실패")
    
    print("=" * 80) 