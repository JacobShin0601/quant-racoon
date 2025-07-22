#!/usr/bin/env python3
"""
전략 수익률 계산 디버깅 스크립트
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.actions.global_macro import HyperparamTuner, MarketRegime

def debug_strategy_returns():
    """전략 수익률 계산 디버깅"""
    print("🔍 전략 수익률 계산 디버깅 시작")
    print("=" * 60)
    
    # HyperparamTuner 초기화
    tuner = HyperparamTuner("config/config_macro.json")
    
    # 테스트 데이터 생성 (간단한 상승 추세)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    prices = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(50)]  # 상승 추세 + 노이즈
    
    test_data = pd.DataFrame({
        'datetime': dates,
        'close': prices,
        'open': [p - np.random.uniform(-1, 1) for p in prices],
        'high': [p + np.random.uniform(0, 2) for p in prices],
        'low': [p - np.random.uniform(0, 2) for p in prices],
        'volume': [np.random.randint(1000000, 5000000) for _ in range(50)]
    })
    test_data.set_index('datetime', inplace=True)
    
    # 기술적 지표 추가
    test_data['sma_20'] = test_data['close'].rolling(20).mean()
    test_data['sma_50'] = test_data['close'].rolling(50).mean()
    test_data['rsi'] = 50 + np.random.normal(0, 10, 50)  # RSI 50 근처
    test_data['atr'] = test_data['close'] * 0.02  # ATR 2%
    test_data['^VIX'] = 20 + np.random.normal(0, 5, 50)  # VIX 20 근처
    
    print(f"📊 테스트 데이터 생성: {len(test_data)}개 포인트")
    print(f"   시작가: {test_data['close'].iloc[0]:.2f}")
    print(f"   종가: {test_data['close'].iloc[-1]:.2f}")
    print(f"   총 수익률: {(test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100:.2f}%")
    
    # 테스트 파라미터
    test_params = {
        'sma_short': 20,
        'sma_long': 50,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'trend_weight': 0.4,
        'momentum_weight': 0.3,
        'volatility_weight': 0.2,
        'macro_weight': 0.1,
        'base_position': 0.8,
        'trending_boost': 1.2,
        'volatile_reduction': 0.5,
        'vix_threshold': 25
    }
    
    print(f"\n🔧 테스트 파라미터:")
    for key, value in test_params.items():
        print(f"   {key}: {value}")
    
    # 파생 변수 계산
    print(f"\n📈 파생 변수 계산...")
    data_with_features = tuner._calculate_derived_features(test_data, test_params)
    print(f"   추가된 컬럼: {[col for col in data_with_features.columns if col not in test_data.columns]}")
    
    # 시장 분류
    print(f"\n🎯 시장 분류...")
    regime = tuner._classify_market_regime(data_with_features, test_params)
    regime_counts = regime.value_counts()
    print(f"   분류 결과:")
    for regime_type, count in regime_counts.items():
        print(f"     {regime_type}: {count}개 ({count/len(regime)*100:.1f}%)")
    
    # 전략 수익률 계산
    print(f"\n💰 전략 수익률 계산...")
    strategy_returns = tuner._calculate_strategy_returns(data_with_features, regime, test_params)
    
    print(f"   전략 수익률 통계:")
    print(f"     평균: {strategy_returns.mean():.6f}")
    print(f"     표준편차: {strategy_returns.std():.6f}")
    print(f"     최소값: {strategy_returns.min():.6f}")
    print(f"     최대값: {strategy_returns.max():.6f}")
    print(f"     非0 개수: {(strategy_returns != 0).sum()}")
    
    # Buy & Hold 수익률
    buy_hold_returns = test_data['close'].pct_change()
    
    # 성과 지표 계산
    print(f"\n📊 성과 지표 계산...")
    metrics = tuner._calculate_performance_metrics(strategy_returns, buy_hold_returns)
    
    print(f"   성과 지표:")
    for metric, value in metrics.items():
        if 'return' in metric or 'drawdown' in metric:
            print(f"     {metric}: {value:.4%}")
        else:
            print(f"     {metric}: {value:.4f}")
    
    # 포지션 분석
    print(f"\n📋 포지션 분석...")
    position = pd.Series(index=data_with_features.index, dtype=float)
    
    for i, current_regime in enumerate(regime):
        if current_regime == MarketRegime.TRENDING_UP.value:
            position.iloc[i] = test_params.get('trending_boost', 1.0)
        elif current_regime == MarketRegime.TRENDING_DOWN.value:
            position.iloc[i] = -test_params.get('base_position', 0.5)
        elif current_regime == MarketRegime.SIDEWAYS.value:
            position.iloc[i] = 0
        elif current_regime == MarketRegime.VOLATILE.value:
            position.iloc[i] = test_params.get('volatile_reduction', 0.5) * test_params.get('base_position', 0.8)
        else:
            position.iloc[i] = 0
    
    print(f"   포지션 통계:")
    print(f"     평균: {position.mean():.4f}")
    print(f"     표준편차: {position.std():.4f}")
    print(f"     최소값: {position.min():.4f}")
    print(f"     최대값: {position.max():.4f}")
    print(f"     양수 개수: {(position > 0).sum()}")
    print(f"     음수 개수: {(position < 0).sum()}")
    print(f"     제로 개수: {(position == 0).sum()}")
    
    print("\n✅ 디버깅 완료!")

if __name__ == "__main__":
    debug_strategy_returns() 