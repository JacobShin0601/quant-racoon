#!/usr/bin/env python3
"""
매크로 하이퍼파라미터 튜너 테스트 스크립트
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).resolve().parent))

from src.actions.global_macro import HyperparamTuner, GlobalMacroDataCollector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_data_collection():
    """데이터 수집 테스트"""
    print("🔍 데이터 수집 테스트 시작...")
    
    collector = GlobalMacroDataCollector()
    
    # 최근 6개월 데이터 수집
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    print(f"📊 데이터 수집 기간: {start_date} ~ {end_date}")
    
    # SPY 데이터 수집
    spy_data = collector.collect_spy_data(start_date, end_date)
    print(f"SPY 데이터: {len(spy_data)}개 행, {len(spy_data.columns)}개 컬럼")
    
    # 매크로 지표 수집
    macro_data = collector.collect_macro_indicators(start_date, end_date)
    print(f"매크로 지표: {len(macro_data)}개 수집됨")
    
    # 데이터 저장
    collector.save_macro_data(spy_data, macro_data, {})
    
    print("✅ 데이터 수집 테스트 완료!")
    return spy_data, macro_data

def test_hyperparam_tuner():
    """하이퍼파라미터 튜너 테스트"""
    print("\n🔧 하이퍼파라미터 튜너 테스트 시작...")
    
    tuner = HyperparamTuner()
    
    # 최근 1년 데이터로 테스트
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"📊 튜닝 기간: {start_date} ~ {end_date}")
    
    # 작은 시도 횟수로 빠른 테스트
    results = tuner.optimize_hyperparameters(start_date, end_date, n_trials=10)
    
    print("\n📈 튜닝 결과:")
    print(f"최적 샤프 비율: {results['best_value']:.4f}")
    print(f"최적 파라미터 개수: {len(results['best_params'])}")
    
    if results['test_performance']:
        print(f"\n🧪 Test 성과:")
        for metric, value in results['test_performance'].items():
            print(f"  {metric}: {value:.4f}")
    
    # 결과 저장
    tuner.save_results(results, "test_results/macro_optimization")
    
    print("✅ 하이퍼파라미터 튜너 테스트 완료!")
    return results

def test_market_regime_classification():
    """시장 상태 분류 테스트"""
    print("\n🎯 시장 상태 분류 테스트 시작...")
    
    tuner = HyperparamTuner()
    
    # 최근 3개월 데이터로 테스트
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # 데이터 수집
    spy_data = tuner.collector.collect_spy_data(start_date, end_date)
    macro_data = tuner.collector.collect_macro_indicators(start_date, end_date)
    
    if spy_data.empty:
        print("❌ SPY 데이터 수집 실패")
        return
    
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
    data_with_features = tuner._calculate_derived_features(spy_data, sample_params)
    
    # 매크로 데이터 병합
    if '^VIX' in macro_data:
        vix_data = macro_data['^VIX'][['Close']].rename(columns={'Close': '^VIX'})
        data_with_features = data_with_features.join(vix_data, how='left')
    
    # 시장 상태 분류
    regime = tuner._classify_market_regime(data_with_features, sample_params)
    
    # 분류 결과 분석
    regime_counts = regime.value_counts()
    print(f"\n📊 시장 상태 분류 결과:")
    for regime_type, count in regime_counts.items():
        percentage = (count / len(regime)) * 100
        print(f"  {regime_type}: {count}일 ({percentage:.1f}%)")
    
    # 전략 수익률 계산
    strategy_returns = tuner._calculate_strategy_returns(data_with_features, regime, sample_params)
    buy_hold_returns = spy_data['Close'].pct_change()
    
    # 성과 지표 계산
    metrics = tuner._calculate_performance_metrics(strategy_returns, buy_hold_returns)
    
    print(f"\n📈 성과 지표:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("✅ 시장 상태 분류 테스트 완료!")

def main():
    """메인 테스트 함수"""
    print("🚀 매크로 하이퍼파라미터 튜너 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 데이터 수집 테스트
        test_data_collection()
        
        # 2. 시장 상태 분류 테스트
        test_market_regime_classification()
        
        # 3. 하이퍼파라미터 튜너 테스트 (시간이 오래 걸릴 수 있음)
        print("\n⚠️  하이퍼파라미터 튜닝은 시간이 오래 걸릴 수 있습니다.")
        response = input("하이퍼파라미터 튜닝을 실행하시겠습니까? (y/n): ")
        
        if response.lower() == 'y':
            test_hyperparam_tuner()
        else:
            print("하이퍼파라미터 튜닝을 건너뜁니다.")
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 