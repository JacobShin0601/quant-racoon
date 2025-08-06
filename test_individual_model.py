#!/usr/bin/env python3
"""
개별 모델 예측 문제 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.actions.neural_stock_predictor import StockPredictionNetwork
from src.utils.centralized_logger import get_logger

# 로거 설정
logger = get_logger("test_individual_model", log_level="DEBUG")

def test_individual_model_prediction():
    """개별 모델 예측 테스트"""
    try:
        # 설정 로드
        config = {
            "neural_network": {
                "train_ratio": 0.7,
                "ensemble": {
                    "universal_weight": 0.7,
                    "individual_weight": 0.3,
                    "enable_individual_models": True,
                    "enable_weight_learning": True
                }
            }
        }
        
        # 신경망 예측기 초기화
        predictor = StockPredictionNetwork(config)
        
        # 모델 로드
        model_path = "models/trader/neural_predictor"
        if not os.path.exists(f"{model_path}_meta.pkl"):
            logger.error(f"모델 파일이 없습니다: {model_path}")
            return
            
        success = predictor.load_model(model_path)
        if not success:
            logger.error("모델 로드 실패")
            return
            
        logger.info("✅ 모델 로드 성공")
        
        # 테스트 데이터 로드 (SPY)
        data_path = "data/trader/SPY.csv"
        if not os.path.exists(data_path):
            logger.error(f"데이터 파일이 없습니다: {data_path}")
            return
            
        test_data = pd.read_csv(data_path)
        logger.info(f"📊 테스트 데이터 로드: {test_data.shape}")
        logger.info(f"   컬럼: {list(test_data.columns)}")
        
        # 최근 100개 데이터로 예측
        recent_data = test_data.tail(100)
        
        # 예측 수행
        logger.info("\n🔮 예측 시작...")
        prediction = predictor.predict(recent_data, "SPY")
        
        if prediction is not None:
            logger.info(f"✅ 예측 성공!")
            logger.info(f"   예측 결과: {prediction}")
        else:
            logger.error("❌ 예측 실패")
            
    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)

if __name__ == "__main__":
    print("="*60)
    print("개별 모델 예측 문제 테스트")
    print("="*60)
    test_individual_model_prediction()