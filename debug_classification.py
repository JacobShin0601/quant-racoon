#!/usr/bin/env python3
"""
시장 분류 로직 디버깅 스크립트
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.actions.global_macro import HyperparamTuner, MarketRegime

def debug_classification():
    """시장 분류 로직 디버깅"""
    print("🔍 시장 분류 로직 디버깅 시작")
    print("=" * 60)
    
    # HyperparamTuner 초기화
    tuner = HyperparamTuner("config/config_macro.json")
    
    # 테스트 데이터 생성 (명확한 상승 추세)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    prices = [100 + i * 1.0 for i in range(50)]  # 선형 상승 추세
    
    test_data = pd.DataFrame({
        'datetime': dates,
        'close': prices,
        'open': [p - 0.5 for p in prices],
        'high': [p + 1.0 for p in prices],
        'low': [p - 1.0 for p in prices],
        'volume': [1000000 for _ in range(50)]
    })
    test_data.set_index('datetime', inplace=True)
    
    # 기술적 지표 추가
    test_data['sma_20'] = test_data['close'].rolling(20).mean()
    test_data['sma_50'] = test_data['close'].rolling(50).mean()
    test_data['rsi'] = [60] * 50  # 중립적 RSI (모든 행에 60)
    test_data['atr'] = test_data['close'] * 0.01  # 낮은 ATR
    test_data['^VIX'] = [15] * 50  # 낮은 VIX (모든 행에 15)
    
    print(f"📊 테스트 데이터 생성: {len(test_data)}개 포인트")
    print(f"   시작가: {test_data['close'].iloc[0]:.2f}")
    print(f"   종가: {test_data['close'].iloc[-1]:.2f}")
    print(f"   총 수익률: {(test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100:.2f}%")
    print(f"   SMA20: {test_data['sma_20'].iloc[-1]:.2f}")
    print(f"   SMA50: {test_data['sma_50'].iloc[-1]:.2f}")
    print(f"   RSI: {test_data['rsi'].iloc[-1]:.2f}")
    print(f"   VIX: {test_data['^VIX'].iloc[-1]:.2f}")
    
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
    
    # 분류 과정 디버깅
    print(f"\n🎯 분류 과정 디버깅...")
    
    # 1. 트렌드 점수 계산
    sma_short = data_with_features[f'sma_{test_params.get("sma_short", 20)}']
    sma_long = data_with_features[f'sma_{test_params.get("sma_long", 50)}']
    trend_score = np.where(sma_short > sma_long, 1, -1)
    trend_weighted = trend_score * test_params['trend_weight']
    
    print(f"   트렌드 점수:")
    print(f"     SMA20 > SMA50: {(sma_short > sma_long).sum()}개")
    print(f"     SMA20 < SMA50: {(sma_short < sma_long).sum()}개")
    print(f"     평균 트렌드 점수: {trend_score.mean():.3f}")
    print(f"     평균 가중 트렌드 점수: {trend_weighted.mean():.3f}")
    
    # 2. 모멘텀 점수 계산
    rsi = data_with_features['rsi']
    momentum_score = np.where(
        (rsi > test_params.get('rsi_oversold', 30)) & (rsi < test_params.get('rsi_overbought', 70)),
        0, np.where(rsi > test_params.get('rsi_overbought', 70), -1, 1)
    )
    momentum_weighted = momentum_score * test_params['momentum_weight']
    
    print(f"   모멘텀 점수:")
    print(f"     RSI 중간: {(momentum_score == 0).sum()}개")
    print(f"     RSI 과매수: {(momentum_score == -1).sum()}개")
    print(f"     RSI 과매도: {(momentum_score == 1).sum()}개")
    print(f"     평균 모멘텀 점수: {momentum_score.mean():.3f}")
    print(f"     평균 가중 모멘텀 점수: {momentum_weighted.mean():.3f}")
    
    # 3. 변동성 점수 계산
    atr_ratio = data_with_features['atr'] / data_with_features['close']
    volatility_score = np.where(atr_ratio > 0.02, 1, 0)
    volatility_weighted = volatility_score * test_params['volatility_weight']
    
    print(f"   변동성 점수:")
    print(f"     높은 변동성: {(volatility_score == 1).sum()}개")
    print(f"     낮은 변동성: {(volatility_score == 0).sum()}개")
    print(f"     평균 변동성 점수: {volatility_score.mean():.3f}")
    print(f"     평균 가중 변동성 점수: {volatility_weighted.mean():.3f}")
    
    # 4. 매크로 점수 계산
    vix = data_with_features['^VIX']
    macro_score = np.where(vix > test_params.get('vix_threshold', 25), 1, 0)
    macro_weighted = macro_score * test_params['macro_weight']
    
    print(f"   매크로 점수:")
    print(f"     높은 VIX: {(macro_score == 1).sum()}개")
    print(f"     낮은 VIX: {(macro_score == 0).sum()}개")
    print(f"     평균 매크로 점수: {macro_score.mean():.3f}")
    print(f"     평균 가중 매크로 점수: {macro_weighted.mean():.3f}")
    
    # 5. 총점 계산
    total_score = trend_weighted + momentum_weighted + volatility_weighted + macro_weighted
    
    print(f"   총점:")
    print(f"     평균 총점: {total_score.mean():.3f}")
    print(f"     최소 총점: {total_score.min():.3f}")
    print(f"     최대 총점: {total_score.max():.3f}")
    
    # 6. 분류 결과
    regime = tuner._classify_market_regime(data_with_features, test_params)
    regime_counts = regime.value_counts()
    
    print(f"\n🎯 최종 분류 결과:")
    for regime_type, count in regime_counts.items():
        print(f"   {regime_type}: {count}개 ({count/len(regime)*100:.1f}%)")
    
    # 7. 직접 계산으로 검증
    print(f"\n🔍 직접 계산 검증:")
    print(f"   RSI 값들: {rsi.head(10).tolist()}")
    print(f"   RSI > 70: {(rsi > 70).sum()}개")
    print(f"   RSI < 30: {(rsi < 30).sum()}개")
    print(f"   RSI 30-70: {((rsi >= 30) & (rsi <= 70)).sum()}개")
    
    # 직접 모멘텀 점수 계산
    direct_momentum = np.where(
        (rsi >= 30) & (rsi <= 70),
        0, np.where(rsi > 70, -1, 1)
    )
    print(f"   직접 모멘텀 점수: {direct_momentum[:10]}")
    print(f"   직접 모멘텀 평균: {direct_momentum.mean():.3f}")
    
    # SMA 디버깅
    print(f"\n📊 SMA 디버깅:")
    sma_20 = data_with_features['sma_20']
    sma_50 = data_with_features['sma_50']
    print(f"   SMA20 NaN 개수: {sma_20.isna().sum()}")
    print(f"   SMA50 NaN 개수: {sma_50.isna().sum()}")
    print(f"   SMA20 값들: {sma_20.head(10).tolist()}")
    print(f"   SMA50 값들: {sma_50.head(10).tolist()}")
    print(f"   SMA20 > SMA50 (NaN 제외): {(sma_20 > sma_50).sum()}개")
    print(f"   SMA20 < SMA50 (NaN 제외): {(sma_20 < sma_50).sum()}개")
    
    # 직접 트렌드 점수 계산
    valid_mask = ~(sma_20.isna() | sma_50.isna())
    direct_trend = np.where(valid_mask, np.where(sma_20 > sma_50, 1, -1), 0)
    print(f"   직접 트렌드 점수: {direct_trend[:10]}")
    print(f"   직접 트렌드 평균: {direct_trend.mean():.3f}")
    
    # 8. 분류 임계값 확인
    print(f"\n📊 분류 임계값:")
    print(f"   TRENDING_UP 임계값: > 0.2")
    print(f"   TRENDING_DOWN 임계값: < -0.2")
    print(f"   VOLATILE 임계값: volatility_score > 0.1")
    print(f"   SIDEWAYS: 기타")
    
    print("\n✅ 분류 디버깅 완료!")

if __name__ == "__main__":
    debug_classification() 