#!/usr/bin/env python3
"""
Enhanced LLM 분석 기능 테스트 스크립트

새로 업데이트된 --enhanced 옵션의 LLM 종합 분석 기능을 테스트합니다.
"""

import sys
import os
import json
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.market_sensor import MarketSensor
from src.agent.enhancements import LLMConfig


def test_enhanced_llm_analysis():
    """Enhanced LLM 분석 테스트"""
    print("🚀 Enhanced LLM 분석 테스트")
    print("=" * 60)

    # LLM 설정 (종합 분석 모드)
    llm_config = {
        'provider': 'hybrid',
        'model_name': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'fallback_to_rules': True,
        'max_tokens': 2000,
        'temperature': 0.1
    }

    try:
        # Market Sensor 초기화 (LLM API 활성화)
        sensor = MarketSensor(
            enable_llm_api=True,
            llm_config=llm_config,
            use_cached_data=True,  # 캐시된 데이터 사용
            use_random_forest=True
        )
        
        print("✅ Market Sensor 초기화 성공")
        print(f"🤖 LLM API 설정: {llm_config['provider']} - {llm_config['model_name']}")
        
        # Enhanced 분석 수행
        print("\n🚀 Enhanced 분석 수행 중...")
        result = sensor.run_enhanced_analysis(
            output_dir="results/macro/enhanced_test",
            verbose=True
        )
        
        if result:
            print("\n✅ Enhanced 분석 완료!")
            print(f"세션 UUID: {result.session_uuid}")
            
            # LLM API 분석 결과 확인
            if hasattr(result, 'llm_api_insights') and result.llm_api_insights:
                llm_insights = result.llm_api_insights
                print("\n🤖 LLM 종합 분석 결과:")
                
                # 종합 분석 결과
                if 'comprehensive_analysis' in llm_insights:
                    comp = llm_insights['comprehensive_analysis']
                    print(f"  📊 시장 역학:")
                    if 'market_dynamics' in comp:
                        dynamics = comp['market_dynamics']
                        print(f"    - 주요 동인: {dynamics.get('primary_drivers', [])}")
                        print(f"    - 추세 강도: {dynamics.get('trend_strength', 'N/A')}")
                        print(f"    - 모멘텀 품질: {dynamics.get('momentum_quality', 'N/A')}")
                
                # 위험 평가
                if 'risk_assessment' in llm_insights:
                    risk = llm_insights['risk_assessment']
                    print(f"  ⚠️ 위험 평가:")
                    print(f"    - 단기 위험: {risk.get('short_term_risks', [])}")
                    print(f"    - 중기 위험: {risk.get('medium_term_risks', [])}")
                    print(f"    - 장기 위험: {risk.get('long_term_risks', [])}")
                
                # 전략적 추천
                if 'strategic_recommendations' in llm_insights:
                    strategy = llm_insights['strategic_recommendations']
                    print(f"  📈 전략적 추천:")
                    if 'portfolio_allocation' in strategy:
                        alloc = strategy['portfolio_allocation']
                        print(f"    - 포트폴리오 배분:")
                        print(f"      * 주식: {alloc.get('equity_allocation', 'N/A')}")
                        print(f"      * 채권: {alloc.get('bond_allocation', 'N/A')}")
                        print(f"      * 현금: {alloc.get('cash_allocation', 'N/A')}")
                    
                    if 'sector_focus' in strategy:
                        sector = strategy['sector_focus']
                        print(f"    - 섹터 포커스:")
                        print(f"      * 과중 배치: {sector.get('overweight_sectors', [])}")
                        print(f"      * 과소 배치: {sector.get('underweight_sectors', [])}")
                
                # 시나리오 분석
                if 'scenario_analysis' in llm_insights:
                    scenario = llm_insights['scenario_analysis']
                    print(f"  🔮 시나리오 분석:")
                    for scenario_type, details in scenario.items():
                        if isinstance(details, dict):
                            prob = details.get('probability', 0)
                            triggers = details.get('triggers', [])
                            print(f"    - {scenario_type}: {prob:.1%} 확률")
                            if triggers:
                                print(f"      트리거: {triggers[:2]}")  # 처음 2개만
                
                # 핵심 인사이트
                if 'key_insights' in llm_insights:
                    insights = llm_insights['key_insights']
                    print(f"  💡 핵심 인사이트:")
                    for i, insight in enumerate(insights[:5], 1):  # 처음 5개만
                        print(f"    {i}. {insight}")
                
                # API 통계
                if 'api_stats' in llm_insights:
                    stats = llm_insights['api_stats']
                    print(f"\n📊 LLM API 통계:")
                    print(f"  - 총 호출: {stats.get('total_calls', 0)}")
                    print(f"  - 성공률: {stats.get('success_rate', 0):.2%}")
                    print(f"  - 평균 응답시간: {stats.get('avg_response_time', 0):.3f}초")
            else:
                print("⚠️ LLM API 분석 결과가 없습니다.")
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/macro/enhanced_test/enhanced_llm_test_{timestamp}.json"
            
            # 결과를 JSON으로 변환 가능한 형태로 준비
            result_dict = {
                "session_uuid": result.session_uuid,
                "current_regime": result.current_regime.value if hasattr(result.current_regime, 'value') else str(result.current_regime),
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "llm_api_insights": result.llm_api_insights,
                "timestamp": result.timestamp.isoformat() if result.timestamp else datetime.now().isoformat(),
                "analysis_type": "enhanced_llm_test"
            }
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 결과 저장: {output_file}")
            
        else:
            print("❌ Enhanced 분석 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 Enhanced LLM 분석 테스트 완료!")
    return True


def test_llm_config_parsing():
    """LLM 설정 파싱 테스트"""
    print("\n🔧 LLM 설정 파싱 테스트")
    print("=" * 40)
    
    from src.agent.enhancements.llm_api_integration import LLMAPIIntegration
    
    # 딕셔너리 설정 테스트
    config_dict = {
        'provider': 'hybrid',
        'model_name': 'test-model',
        'max_tokens': 2000,
        'temperature': 0.1
    }
    
    try:
        llm_system = LLMAPIIntegration(config_dict)
        print(f"✅ 딕셔너리 설정 파싱 성공")
        print(f"  - Provider: {llm_system.config.provider}")
        print(f"  - Model: {llm_system.config.model_name}")
        print(f"  - Max Tokens: {llm_system.config.max_tokens}")
        print(f"  - Temperature: {llm_system.config.temperature}")
        
    except Exception as e:
        print(f"❌ 딕셔너리 설정 파싱 실패: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 Enhanced LLM 분석 시스템 테스트 시작")
    print("=" * 60)
    
    # LLM 설정 파싱 테스트
    if not test_llm_config_parsing():
        print("❌ LLM 설정 파싱 테스트 실패")
        sys.exit(1)
    
    # Enhanced LLM 분석 테스트
    if test_enhanced_llm_analysis():
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n❌ 일부 테스트 실패")
        sys.exit(1) 