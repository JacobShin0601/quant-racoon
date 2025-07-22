#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝 수정사항 테스트 스크립트
컬럼명 문제와 중복 파라미터 문제 해결 확인
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.market_sensor import MarketSensor
from src.actions.global_macro import HyperparamTuner

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_hyperparam_tuning():
    """하이퍼파라미터 튜닝 테스트"""
    print("\n🔧 하이퍼파라미터 튜닝 수정사항 테스트")
    print("=" * 60)
    
    # 날짜 설정 (짧은 기간으로 테스트)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"📅 테스트 기간: {start_date} ~ {end_date}")
    print(f"🎯 테스트 목표: 컬럼명 문제와 중복 파라미터 문제 해결 확인")
    
    try:
        # HyperparamTuner 초기화
        print("\n1️⃣ HyperparamTuner 초기화...")
        tuner = HyperparamTuner("config/config_macro.json")
        print("   ✅ 초기화 완료")
        
        # 설정 파일 확인
        print("\n2️⃣ 설정 파일 확인...")
        config = tuner.config
        print(f"   📊 최적화 시도 횟수: {config.get('optimization', {}).get('n_trials', 'N/A')}")
        print(f"   🎯 목적 함수: {config.get('optimization', {}).get('objective', 'N/A')}")
        print(f"   📈 사용 가능한 지표: {config.get('optimization', {}).get('metrics', [])}")
        
        # 중복 파라미터 확인
        strategy_config = config.get('trading_strategy', {})
        stop_loss_params = strategy_config.get('stop_loss', {}).keys()
        take_profit_params = strategy_config.get('take_profit', {}).keys()
        
        print(f"   🛑 Stop Loss 파라미터: {list(stop_loss_params)}")
        print(f"   📈 Take Profit 파라미터: {list(take_profit_params)}")
        
        # 중복 확인
        duplicates = set(stop_loss_params) & set(take_profit_params)
        if duplicates:
            print(f"   ⚠️ 중복 파라미터 발견: {duplicates}")
        else:
            print("   ✅ 중복 파라미터 없음")
        
        # 데이터 수집 테스트
        print("\n3️⃣ 데이터 수집 테스트...")
        spy_data = tuner.collector.collect_spy_data(start_date, end_date)
        macro_data = tuner.collector.collect_macro_indicators(start_date, end_date)
        
        if not spy_data.empty:
            print(f"   ✅ SPY 데이터 수집 완료: {len(spy_data)}개 행")
            print(f"   📊 SPY 컬럼: {list(spy_data.columns)}")
            
            # 컬럼명 확인
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            print(f"   🎯 사용할 Close 컬럼: {close_col}")
        else:
            print("   ❌ SPY 데이터 수집 실패")
            return
        
        if macro_data:
            print(f"   ✅ 매크로 데이터 수집 완료: {len(macro_data)}개 심볼")
            for symbol, data in macro_data.items():
                if not data.empty:
                    print(f"      {symbol}: {len(data)}개 행, 컬럼: {list(data.columns)}")
        else:
            print("   ❌ 매크로 데이터 수집 실패")
            return
        
        # 파생 변수 계산 테스트
        print("\n4️⃣ 파생 변수 계산 테스트...")
        test_params = {
            'sma_short': 20,
            'sma_medium': 50,
            'sma_long': 100,
            'ema_short': 12,
            'ema_long': 26,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0,
            'atr_period': 14,
            'volume_ma_period': 20,
            'trend_weight': 0.4,
            'momentum_weight': 0.3,
            'volatility_weight': 0.2,
            'macro_weight': 0.1,
            'base_position': 0.8,
            'volatile_reduction': 0.5,
            'trending_boost': 1.2,
            'stop_atr_multiplier': 2.0,
            'stop_fixed_percentage': 0.05,
            'profit_atr_multiplier': 3.0,
            'profit_fixed_percentage': 0.1
        }
        
        features_data = tuner._calculate_derived_features(spy_data, test_params)
        print(f"   ✅ 파생 변수 계산 완료: {len(features_data)}개 행")
        print(f"   📊 추가된 컬럼: {[col for col in features_data.columns if col not in spy_data.columns]}")
        
        # 시장 분류 테스트
        print("\n5️⃣ 시장 분류 테스트...")
        regime = tuner._classify_market_regime(features_data, test_params)
        print(f"   ✅ 시장 분류 완료: {len(regime)}개 분류")
        print(f"   📊 분류 결과 샘플: {regime.value_counts().to_dict()}")
        
        # 전략 수익률 계산 테스트
        print("\n6️⃣ 전략 수익률 계산 테스트...")
        strategy_returns = tuner._calculate_strategy_returns(features_data, regime, test_params)
        print(f"   ✅ 전략 수익률 계산 완료: {len(strategy_returns)}개 수익률")
        print(f"   📊 수익률 통계: 평균={strategy_returns.mean():.4f}, 표준편차={strategy_returns.std():.4f}")
        
        # 성과 지표 계산 테스트
        print("\n7️⃣ 성과 지표 계산 테스트...")
        close_col = 'close' if 'close' in spy_data.columns else 'Close'
        buy_hold_returns = spy_data[close_col].pct_change()
        metrics = tuner._calculate_performance_metrics(strategy_returns, buy_hold_returns)
        print(f"   ✅ 성과 지표 계산 완료")
        for metric, value in metrics.items():
            print(f"      {metric}: {value:.4f}")
        
        # 간단한 최적화 테스트 (5회만)
        print("\n8️⃣ 간단한 최적화 테스트 (5회)...")
        try:
            results = tuner.optimize_hyperparameters(start_date, end_date, n_trials=5)
            print(f"   ✅ 최적화 완료!")
            print(f"   🎯 최적 샤프 비율: {results['best_value']:.4f}")
            print(f"   📊 최적 파라미터 개수: {len(results['best_params'])}")
            
            # 중복 파라미터 확인
            param_names = list(results['best_params'].keys())
            duplicates = [name for name in param_names if param_names.count(name) > 1]
            if duplicates:
                print(f"   ⚠️ 최적화 결과에서 중복 파라미터 발견: {duplicates}")
            else:
                print("   ✅ 최적화 결과에서 중복 파라미터 없음")
                
        except Exception as e:
            print(f"   ❌ 최적화 중 오류: {e}")
            logger.error(f"최적화 오류: {e}", exc_info=True)
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        logger.error(f"테스트 오류: {e}", exc_info=True)


def main():
    """메인 함수"""
    print("🚀 하이퍼파라미터 튜닝 수정사항 테스트 시작")
    print("=" * 80)
    
    test_hyperparam_tuning()


if __name__ == "__main__":
    main() 