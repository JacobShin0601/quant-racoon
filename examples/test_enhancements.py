#!/usr/bin/env python3
"""
Enhancements 패키지 사용 예시 스크립트

터미널에서 실행: python examples/test_enhancements.py
"""

import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_rlmf_system():
    """RLMF 시스템 테스트"""
    print("🤖 RLMF (Reinforcement Learning from Market Feedback) 시스템 테스트")
    print("=" * 70)
    
    try:
        from src.agent.enhancements.rlmf_adaptation import RLMFRegimeAdaptation
        
        # RLMF 인스턴스 생성
        rlmf = RLMFRegimeAdaptation(learning_rate=0.02, feedback_window=15)
        
        print("✅ RLMF 시스템 초기화 완료!")
        print(f"📊 학습률: {rlmf.learning_rate}")
        print(f"📈 피드백 윈도우: {rlmf.feedback_window}")
        print(f"🔑 Key Metrics: {list(rlmf.key_metrics.keys())}")
        
        # 적응 상태 확인
        status = rlmf.get_adaptation_status()
        print(f"📌 현재 상태: {status['status']}")
        print(f"📊 성과: {status['performance']:.1%}")
        
        # 가상 매크로 데이터로 Statistical Arbitrage 신호 테스트
        print("\n🔄 Statistical Arbitrage 신호 계산 테스트:")
        mock_macro_data = {
            '^VIX': pd.DataFrame({
                'close': [20, 22, 21, 19, 18]
            }),
            'XRT': pd.DataFrame({
                'close': [50, 51, 52, 51.5, 52.2]
            })
        }
        
        signal = rlmf.calculate_statistical_arbitrage_signal(mock_macro_data)
        print(f"   📈 신호 방향: {signal['direction']}")
        print(f"   💪 신호 강도: {signal['signal_strength']:.3f}")
        print(f"   🎯 신뢰도: {signal['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ RLMF 테스트 오류: {e}")
        return False

def test_confidence_system():
    """다층 신뢰도 시스템 테스트"""
    print("\n📊 Multi-Layer Confidence System 테스트")
    print("=" * 70)
    
    try:
        from src.agent.enhancements.confidence_system import MultiLayerConfidenceSystem
        
        # 신뢰도 시스템 생성
        confidence_sys = MultiLayerConfidenceSystem()
        
        print("✅ 다층 신뢰도 시스템 초기화 완료!")
        print(f"⚖️ 가중치: {confidence_sys.confidence_weights}")
        
        # 가상 신뢰도 데이터로 테스트
        result = confidence_sys.calculate_comprehensive_confidence(
            technical_conf=0.75,
            macro_conf=0.65,
            stat_arb_conf=0.80,
            rlmf_conf=0.70
        )
        
        print(f"\n📈 종합 신뢰도: {result['adjusted_confidence']:.1%}")
        print(f"🔗 일관성 점수: {result['consistency_score']:.1%}")
        print("🔍 구성요소별 기여도:")
        for component, value in result['component_confidences'].items():
            print(f"   • {component}: {value:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 신뢰도 시스템 테스트 오류: {e}")
        return False

def test_regime_detection():
    """동적 Regime Switching 감지 시스템 테스트"""
    print("\n⚡ Dynamic Regime Switching Detector 테스트")
    print("=" * 70)
    
    try:
        from src.agent.enhancements.regime_detection import DynamicRegimeSwitchingDetector
        
        # Regime 감지기 생성
        detector = DynamicRegimeSwitchingDetector(window_size=30, shift_threshold=0.3)
        
        print("✅ Regime Switching 감지기 초기화 완료!")
        print(f"🪟 윈도우 크기: {detector.window_size}")
        print(f"📏 변화 임계값: {detector.shift_threshold}")
        
        # 가상 SPY 데이터 생성
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        spy_data = pd.DataFrame({
            'close': np.random.normal(100, 2, 100).cumsum() + 4000
        }, index=dates)
        
        # 가상 매크로 데이터
        mock_macro_data = {
            '^VIX': pd.DataFrame({
                'close': np.random.normal(20, 5, 100)
            }, index=dates)
        }
        
        # Regime shift 감지 테스트
        result = detector.detect_regime_shifts(spy_data, mock_macro_data)
        
        print(f"\n🔍 Regime Shift 감지됨: {'🚨 YES' if result['regime_shift_detected'] else '✅ NO'}")
        print(f"📊 변화 점수: {result['shift_score']:.3f}")
        print(f"🎯 감지 신뢰도: {result['confidence']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Regime 감지 테스트 오류: {e}")
        return False

def test_llm_insights():
    """LLM Privileged Information 시스템 테스트"""
    print("\n🧠 LLM Privileged Information System 테스트")
    print("=" * 70)
    
    try:
        from src.agent.enhancements.llm_insights import LLMPrivilegedInformationSystem
        
        # LLM 인사이트 시스템 생성
        llm_system = LLMPrivilegedInformationSystem()
        
        print("✅ LLM 인사이트 시스템 초기화 완료!")
        print(f"📚 지식 베이스 카테고리: {list(llm_system.market_knowledge_base.keys())}")
        
        # 가상 매크로 데이터로 테스트
        mock_macro_data = {
            '^VIX': pd.DataFrame({'close': [25]}),  # 높은 VIX
            '^TNX': pd.DataFrame({'close': [4.5]}), # 금리
        }
        
        market_metrics = {'vix_level': 25, 'current_probabilities': {}}
        
        insights = llm_system.get_privileged_insights('VOLATILE', mock_macro_data, market_metrics)
        
        print(f"\n🔍 Regime 검증 일관성: {insights['regime_validation']['consistency']:.1%}")
        print(f"🎯 신뢰도 수정자: {len(insights['confidence_modifiers'])}개")
        print(f"💡 전략 추천: {len(insights['strategic_recommendations'])}개")
        
        if insights['strategic_recommendations']:
            print("📋 주요 추천사항:")
            for i, rec in enumerate(insights['strategic_recommendations'][:3], 1):
                print(f"   {i}. {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM 인사이트 테스트 오류: {e}")
        return False

def test_integrated_system():
    """통합 시스템 테스트"""
    print("\n🎯 통합 MarketSensor 시스템 테스트")
    print("=" * 70)
    
    try:
        from src.agent.market_sensor import MarketSensor
        
        # MarketSensor 초기화
        print("🚀 MarketSensor 초기화 중...")
        sensor = MarketSensor()
        
        print("✅ MarketSensor 초기화 완료!")
        print(f"🆔 세션 UUID: {sensor.session_uuid}")
        print(f"🤖 RLMF 상태: {sensor.rlmf_adaptation.get_adaptation_status()['status']}")
        
        # 고도화된 요약 정보 테스트 (실제 데이터 없이는 오류 발생 가능)
        print("\n📊 고도화된 분석 시스템 구성 요소:")
        print(f"   • RLMF 적응 시스템: ✅")
        print(f"   • 다층 신뢰도 시스템: ✅") 
        print(f"   • Regime 감지 시스템: ✅")
        print(f"   • LLM 인사이트 시스템: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ 통합 시스템 테스트 오류: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🎉 Enhancements 패키지 종합 테스트")
    print("=" * 80)
    
    test_results = []
    
    # 각 시스템 개별 테스트
    test_results.append(("RLMF System", test_rlmf_system()))
    test_results.append(("Confidence System", test_confidence_system()))
    test_results.append(("Regime Detection", test_regime_detection()))
    test_results.append(("LLM Insights", test_llm_insights()))
    test_results.append(("Integrated System", test_integrated_system()))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)
    
    passed = 0
    for name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 전체 결과: {passed}/{len(test_results)} 테스트 통과")
    
    if passed == len(test_results):
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("⚠️  일부 테스트에서 오류가 발생했습니다. 설정을 확인해주세요.")

if __name__ == "__main__":
    main() 