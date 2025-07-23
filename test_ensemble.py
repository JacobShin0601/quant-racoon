#!/usr/bin/env python3
"""
앙상블 전략 테스트 스크립트
"""

import sys
import os
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.actions.ensemble import EnsembleStrategy


def test_ensemble_initialization():
    """앙상블 전략 초기화 테스트"""
    print("🧪 앙상블 전략 초기화 테스트")
    
    try:
        ensemble = EnsembleStrategy(
            config_path="config/config_ensemble.json",
            market_sensor_config="config/config_macro.json"
        )
        
        print("✅ 앙상블 전략 초기화 성공")
        print(f"📊 로드된 시장 환경 설정: {len(ensemble.regime_configs)}개")
        
        for regime in ensemble.regime_configs.keys():
            print(f"  - {regime}")
        
        return ensemble
        
    except Exception as e:
        print(f"❌ 앙상블 전략 초기화 실패: {e}")
        return None


def test_market_regime_detection(ensemble):
    """시장 환경 감지 테스트"""
    print("\n🧪 시장 환경 감지 테스트")
    
    try:
        regime_detection = ensemble.detect_market_regime()
        
        print("✅ 시장 환경 감지 성공")
        print(f"📊 감지된 환경: {regime_detection['regime']}")
        print(f"🎯 신뢰도: {regime_detection['confidence']:.3f}")
        print(f"📅 감지 날짜: {regime_detection['detection_date']}")
        
        return regime_detection
        
    except Exception as e:
        print(f"❌ 시장 환경 감지 실패: {e}")
        return None


def test_regime_config_loading(ensemble):
    """시장 환경별 설정 로딩 테스트"""
    print("\n🧪 시장 환경별 설정 로딩 테스트")
    
    test_regimes = ["TRENDING_UP", "TRENDING_DOWN", "VOLATILE", "SIDEWAYS"]
    
    for regime in test_regimes:
        try:
            config = ensemble.get_regime_config(regime)
            print(f"✅ {regime} 설정 로드 성공")
            print(f"  - 전략 수: {len(config.get('strategies', []))}")
            print(f"  - 시간대: {config.get('time_horizon', 'N/A')}")
            
        except Exception as e:
            print(f"❌ {regime} 설정 로드 실패: {e}")


def test_ensemble_pipeline(ensemble):
    """앙상블 파이프라인 테스트 (간단 버전)"""
    print("\n🧪 앙상블 파이프라인 테스트")
    
    try:
        # 시장 환경 감지만 테스트 (실제 파이프라인 실행은 시간이 오래 걸림)
        regime_detection = ensemble.detect_market_regime()
        detected_regime = regime_detection["regime"]
        
        print(f"✅ 시장 환경 감지: {detected_regime}")
        
        # 해당 환경의 설정 확인
        regime_config = ensemble.get_regime_config(detected_regime)
        print(f"✅ {detected_regime} 환경 설정 확인")
        print(f"  - 전략 목록: {regime_config.get('strategies', [])[:3]}...")  # 처음 3개만 표시
        
        print("⚠️ 실제 파이프라인 실행은 시간이 오래 걸려 생략했습니다.")
        print("   전체 실행을 원하시면 run_ensemble.sh를 사용하세요.")
        
    except Exception as e:
        print(f"❌ 앙상블 파이프라인 테스트 실패: {e}")


def test_backtest_ensemble(ensemble):
    """백테스팅 앙상블 테스트 (짧은 기간)"""
    print("\n🧪 백테스팅 앙상블 테스트 (짧은 기간)")
    
    try:
        # 최근 7일간 백테스팅
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        print(f"📅 백테스팅 기간: {start_date} ~ {end_date}")
        
        backtest_results = ensemble.run_backtest_ensemble(start_date, end_date)
        
        print("✅ 백테스팅 완료")
        print(f"📊 시장 환경별 감지 횟수:")
        
        for regime, data in backtest_results.get("performance_by_regime", {}).items():
            print(f"  - {regime}: {data.get('detection_count', 0)}회")
        
    except Exception as e:
        print(f"❌ 백테스팅 테스트 실패: {e}")


def main():
    """메인 테스트 함수"""
    print("🎯 앙상블 전략 테스트 시작")
    print("=" * 50)
    
    # 1. 초기화 테스트
    ensemble = test_ensemble_initialization()
    if not ensemble:
        print("❌ 초기화 실패로 인해 테스트를 중단합니다.")
        return
    
    # 2. 시장 환경 감지 테스트
    regime_detection = test_market_regime_detection(ensemble)
    
    # 3. 설정 로딩 테스트
    test_regime_config_loading(ensemble)
    
    # 4. 파이프라인 테스트 (간단 버전)
    test_ensemble_pipeline(ensemble)
    
    # 5. 백테스팅 테스트 (짧은 기간)
    test_backtest_ensemble(ensemble)
    
    print("\n" + "=" * 50)
    print("🎉 앙상블 전략 테스트 완료!")
    print("\n📋 다음 단계:")
    print("  1. 전체 실행: ./run_ensemble.sh")
    print("  2. 백테스팅: python -m src.actions.ensemble --mode backtest --start-date 2023-01-01 --end-date 2024-12-31")
    print("  3. 설정 확인: config/config_ensemble.json")


if __name__ == "__main__":
    main() 