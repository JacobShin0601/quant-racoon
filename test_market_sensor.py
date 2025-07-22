#!/usr/bin/env python3
"""
통합 Market Sensor 테스트 스크립트
시장 환경 분류, 하이퍼파라미터 튜닝, 전략 추천 기능을 종합적으로 테스트
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).resolve().parent))

from src.agent.market_sensor import MarketSensor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_data_collection():
    """데이터 수집 테스트"""
    print("🔍 데이터 수집 테스트 시작...")
    
    sensor = MarketSensor()
    
    # 최근 6개월 데이터 수집
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    print(f"📊 데이터 수집 기간: {start_date} ~ {end_date}")
    
    try:
        spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        print(f"SPY 데이터: {len(spy_data)}개 행, {len(spy_data.columns)}개 컬럼")
        print(f"매크로 지표: {len(macro_data)}개 수집됨")
        print(f"섹터 데이터: {len(sector_data)}개 수집됨")
        
        print("✅ 데이터 수집 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        return False

def test_market_analysis():
    """시장 분석 테스트"""
    print("\n🔍 시장 분석 테스트 시작...")
    
    sensor = MarketSensor()
    
    try:
        # 현재 시장 분석 (기본 파라미터)
        print("📊 기본 파라미터로 시장 분석...")
        analysis_basic = sensor.get_current_market_analysis(use_optimized_params=False)
        
        if 'error' in analysis_basic:
            print(f"❌ 기본 분석 실패: {analysis_basic['error']}")
            return False
        
        print(f"현재 시장 환경: {analysis_basic['current_regime']}")
        print(f"데이터 기간: {analysis_basic['data_period']}")
        
        # 최적화된 파라미터로 분석 (있는 경우)
        print("\n📊 최적화된 파라미터로 시장 분석...")
        analysis_optimized = sensor.get_current_market_analysis(use_optimized_params=True)
        
        if 'error' not in analysis_optimized:
            print(f"최적화된 분석 - 현재 시장 환경: {analysis_optimized['current_regime']}")
            
            print(f"\n📈 성과 비교:")
            print("기본 파라미터:")
            for metric, value in analysis_basic['performance_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print("\n최적화된 파라미터:")
            for metric, value in analysis_optimized['performance_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        else:
            print("최적화된 파라미터가 없습니다.")
        
        print("✅ 시장 분석 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 시장 분석 실패: {e}")
        return False

def test_hyperparameter_optimization():
    """하이퍼파라미터 최적화 테스트"""
    print("\n🔧 하이퍼파라미터 최적화 테스트 시작...")
    
    sensor = MarketSensor()
    
    # 최근 1년 데이터로 테스트
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"📊 튜닝 기간: {start_date} ~ {end_date}")
    
    try:
        # 작은 시도 횟수로 빠른 테스트
        results = sensor.optimize_hyperparameters_optuna(start_date, end_date, n_trials=10)
        
        print(f"\n📈 튜닝 결과:")
        print(f"최적 샤프 비율: {results['best_value']:.4f}")
        print(f"최적 파라미터 개수: {len(results['best_params'])}")
        
        if results['test_performance']:
            print(f"\n🧪 Test 성과:")
            for metric, value in results['test_performance'].items():
                print(f"  {metric}: {value:.4f}")
        
        # 결과 저장
        sensor.save_optimization_results(results, "test_results/market_sensor_optimization")
        
        print("✅ 하이퍼파라미터 최적화 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 하이퍼파라미터 최적화 실패: {e}")
        return False

def test_market_regime_classification():
    """시장 상태 분류 테스트"""
    print("\n🎯 시장 상태 분류 테스트 시작...")
    
    sensor = MarketSensor()
    
    # 최근 3개월 데이터로 테스트
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    try:
        # 데이터 수집
        spy_data = sensor.macro_collector.collect_spy_data(start_date, end_date)
        macro_data = sensor.macro_collector.collect_macro_indicators(start_date, end_date)
        
        if spy_data.empty:
            print("❌ SPY 데이터 수집 실패")
            return False
        
        # 샘플 파라미터로 테스트
        sample_params = {
            'sma_short': 20,
            'sma_long': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'vix_threshold': 25,
            'trend_weight': 0.4,
            'momentum_weight': 0.3,
            'volatility_weight': 0.2,
            'macro_weight': 0.1,
            'base_position': 0.8,
            'trending_boost': 1.2,
            'volatile_reduction': 0.5
        }
        
        # 파생 변수 계산
        data_with_features = sensor._calculate_derived_features(spy_data, sample_params)
        
        # 매크로 데이터 병합
        if '^VIX' in macro_data:
            vix_data = macro_data['^VIX'][['Close']].rename(columns={'Close': '^VIX'})
            data_with_features = data_with_features.join(vix_data, how='left')
        
        # 시장 상태 분류
        regime = sensor._classify_market_regime_optimized(data_with_features, sample_params)
        
        # 분류 결과 분석
        regime_counts = regime.value_counts()
        print(f"\n📊 시장 상태 분류 결과:")
        for regime_type, count in regime_counts.items():
            percentage = (count / len(regime)) * 100
            print(f"  {regime_type}: {count}일 ({percentage:.1f}%)")
        
        # 전략 수익률 계산
        strategy_returns = sensor._calculate_strategy_returns(data_with_features, regime, sample_params)
        close_col = 'close' if 'close' in spy_data.columns else 'Close'
        buy_hold_returns = spy_data[close_col].pct_change()
        
        # 성과 지표 계산
        metrics = sensor._calculate_performance_metrics(strategy_returns, buy_hold_returns)
        
        print(f"\n📈 성과 지표:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("✅ 시장 상태 분류 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 시장 상태 분류 실패: {e}")
        return False

def test_integrated_workflow():
    """통합 워크플로우 테스트"""
    print("\n🔄 통합 워크플로우 테스트 시작...")
    
    sensor = MarketSensor()
    
    try:
        # 1. 데이터 수집
        print("1️⃣ 데이터 수집...")
        spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        print(f"   ✅ SPY 데이터: {len(spy_data)}개")
        
        # 2. 빠른 하이퍼파라미터 최적화
        print("2️⃣ 하이퍼파라미터 최적화...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        results = sensor.optimize_hyperparameters_optuna(start_date, end_date, n_trials=5)
        print(f"   ✅ 최적 샤프 비율: {results['best_value']:.4f}")
        
        # 3. 최적화된 파라미터로 현재 시장 분석
        print("3️⃣ 현재 시장 분석...")
        analysis = sensor.get_current_market_analysis(use_optimized_params=True)
        
        if 'error' not in analysis:
            print(f"   ✅ 현재 시장 환경: {analysis['current_regime']}")
            print(f"   📊 샤프 비율: {analysis['performance_metrics']['sharpe_ratio']:.4f}")
            print(f"   💡 추천 전략: {analysis['recommendation']['primary_strategy']}")
        
        # 4. 결과 저장
        print("4️⃣ 결과 저장...")
        sensor.save_optimization_results(results)
        print("   ✅ 결과 저장 완료")
        
        print("✅ 통합 워크플로우 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 통합 워크플로우 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 통합 Market Sensor 테스트 시작")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # 1. 데이터 수집 테스트
        test_results['data_collection'] = test_data_collection()
        
        # 2. 시장 분석 테스트
        test_results['market_analysis'] = test_market_analysis()
        
        # 3. 시장 상태 분류 테스트
        test_results['regime_classification'] = test_market_regime_classification()
        
        # 4. 하이퍼파라미터 최적화 테스트 (시간이 오래 걸릴 수 있음)
        print("\n⚠️  하이퍼파라미터 최적화는 시간이 오래 걸릴 수 있습니다.")
        response = input("하이퍼파라미터 최적화 테스트를 실행하시겠습니까? (y/n): ")
        
        if response.lower() == 'y':
            test_results['hyperparameter_optimization'] = test_hyperparameter_optimization()
        else:
            print("하이퍼파라미터 최적화 테스트를 건너뜁니다.")
            test_results['hyperparameter_optimization'] = False
        
        # 5. 통합 워크플로우 테스트
        test_results['integrated_workflow'] = test_integrated_workflow()
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        for test_name, result in test_results.items():
            status = "✅ 통과" if result else "❌ 실패"
            print(f"{test_name}: {status}")
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\n전체 결과: {passed_tests}/{total_tests} 테스트 통과")
        
        if passed_tests == total_tests:
            print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        else:
            print("⚠️  일부 테스트가 실패했습니다. 로그를 확인해주세요.")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 