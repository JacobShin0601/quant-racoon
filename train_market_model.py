#!/usr/bin/env python3
"""
시장 환경 분류 모델 학습 스크립트
Random Forest 모델을 학습하고 저장합니다.
"""

import sys
import os
import logging
from datetime import datetime
import argparse

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.actions.random_forest import MarketRegimeRF
from src.agent.market_sensor import MarketSensor

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('log/market_model_training.log'),
            logging.StreamHandler()
        ]
    )

def train_market_model(data_dir: str = "data/macro", force_retrain: bool = False):
    """
    시장 환경 분류 모델 학습
    
    Args:
        data_dir: 데이터 디렉토리 경로
        force_retrain: 강제 재학습 여부
    """
    logger = logging.getLogger(__name__)
    
    print("🎯 시장 환경 분류 모델 학습 시작...")
    print(f"📁 데이터 디렉토리: {data_dir}")
    print(f"🔄 강제 재학습: {force_retrain}")
    
    try:
        # Random Forest 모델 초기화
        rf_model = MarketRegimeRF(verbose=True)
        
        # 기존 모델 확인
        if not force_retrain:
            try:
                rf_model.load_model()
                logger.info("기존 모델을 로드했습니다.")
                print("✅ 기존 모델이 존재합니다. 강제 재학습을 원하면 --force-retrain 옵션을 사용하세요.")
                return
            except FileNotFoundError:
                logger.info("기존 모델이 없습니다. 새로 학습을 시작합니다.")
        
        # 모델 학습
        print("🔄 모델 학습 중...")
        results = rf_model.train_model(save_model=True)
        
        print("✅ 모델 학습 완료!")
        print(f"📊 학습 결과:")
        print(f"  - 훈련 정확도: {results['train_score']:.4f}")
        print(f"  - 테스트 정확도: {results['test_score']:.4f}")
        print(f"  - 교차 검증: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        print(f"  - 샘플 수: {results['n_samples']}")
        print(f"  - 특성 수: {results['n_features']}")
        
        # 현재 시장 상태 예측 테스트
        print("\n🔍 현재 시장 상태 예측 테스트...")
        probabilities = rf_model.get_current_market_probabilities(data_dir)
        
        print("📊 현재 시장 상태 확률:")
        for regime, prob in probabilities.items():
            print(f"  - {regime.upper()}: {prob:.1%}")
        
        # Market Sensor 테스트
        print("\n🔍 Market Sensor 테스트...")
        market_sensor = MarketSensor(data_dir=data_dir)
        analysis = market_sensor.get_current_market_analysis(use_ml_model=True)
        
        if 'error' not in analysis:
            print(f"✅ Market Sensor 분석 성공:")
            print(f"  - 현재 환경: {analysis['current_regime']}")
            print(f"  - 신뢰도: {analysis['confidence']:.3f}")
        else:
            print(f"❌ Market Sensor 분석 실패: {analysis['error']}")
        
        print("\n🎉 모델 학습 및 테스트 완료!")
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {e}")
        print(f"❌ 모델 학습 실패: {e}")
        raise

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='시장 환경 분류 모델 학습')
    parser.add_argument('--data-dir', default='data/macro', help='데이터 디렉토리 경로')
    parser.add_argument('--force-retrain', action='store_true', help='강제 재학습')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    # 로그 디렉토리 생성
    os.makedirs('log', exist_ok=True)
    
    # 모델 학습
    train_market_model(data_dir=args.data_dir, force_retrain=args.force_retrain)

if __name__ == "__main__":
    main() 