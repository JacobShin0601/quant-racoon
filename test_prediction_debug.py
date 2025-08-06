#!/usr/bin/env python3
"""
신경망 예측 디버깅 테스트
"""
import sys
import pandas as pd
import numpy as np
from src.actions.neural_stock_predictor import NeuralStockPredictor

def test_prediction():
    """예측 테스트"""
    # 모델 초기화
    predictor = NeuralStockPredictor()
    
    # 모델 로드
    model_path = "models/trader/neural_predictor"
    if not predictor.load_model(model_path):
        print("❌ 모델 로드 실패")
        return
    
    print("✅ 모델 로드 성공")
    
    # 테스트 데이터 준비 (SPY)
    test_data = pd.read_csv("data/trader/SPY_data.csv")
    test_data['datetime'] = pd.to_datetime(test_data['datetime'])
    test_data.set_index('datetime', inplace=True)
    
    # 최근 30일 데이터만 사용
    test_features = test_data.tail(30)
    
    print(f"\n📊 테스트 데이터: {len(test_features)}일")
    print(f"날짜 범위: {test_features.index[0]} ~ {test_features.index[-1]}")
    
    # 예측 실행
    print("\n🔮 SPY 예측 실행...")
    prediction = predictor.predict(test_features, "SPY")
    
    print("\n📈 예측 결과:")
    print(f"타입: {type(prediction)}")
    
    if isinstance(prediction, dict):
        for key, value in prediction.items():
            print(f"  {key}: {value}")
            if 'target_22d' in key and isinstance(value, (int, float)):
                print(f"    → 22일 예상 수익률: {value*100:.2f}%")
    else:
        print(f"예측값: {prediction}")
    
    # 추가 디버깅을 위한 개별 모델 확인
    if hasattr(predictor, 'individual_target_stats') and 'SPY' in predictor.individual_target_stats:
        stats = predictor.individual_target_stats['SPY']
        print(f"\n📊 SPY 정규화 통계:")
        print(f"  mean: {stats['mean']}")
        print(f"  std: {stats['std']}")
        print(f"  mean type: {type(stats['mean'])}")
        if hasattr(stats['mean'], 'shape'):
            print(f"  mean shape: {stats['mean'].shape}")

if __name__ == "__main__":
    test_prediction()