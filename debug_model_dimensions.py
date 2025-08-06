#!/usr/bin/env python3
"""
모델 차원 확인 스크립트
"""

import sys
import os
import pickle
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def check_model_dimensions():
    """저장된 모델의 차원 정보 확인"""
    
    # 메타 데이터 로드
    meta_path = "models/trader/neural_predictor_meta.pkl"
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)
            
        print("=" * 80)
        print("📊 모델 메타 데이터")
        print("=" * 80)
        
        # 피처 정보 확인
        if "feature_info" in meta_data:
            feature_info = meta_data["feature_info"]
            print("\n✅ 피처 정보:")
            
            # 개별 모델 피처
            if "individual_features" in feature_info:
                print("\n📌 개별 모델 피처:")
                for symbol, info in feature_info["individual_features"].items():
                    feature_names = info.get("feature_names", [])
                    print(f"   {symbol}: {len(feature_names)}개 피처")
                    if len(feature_names) > 0:
                        print(f"      예시: {feature_names[:5]}")
                        
            # 통합 모델 피처
            if "universal_features" in feature_info:
                universal_info = feature_info["universal_features"]
                feature_names = universal_info.get("feature_names", [])
                print(f"\n📌 통합 모델 피처: {len(feature_names)}개")
                if len(feature_names) > 0:
                    print(f"   예시: {feature_names[:5]}")
                    
        # 모델 차원 확인
        if "model_config" in meta_data:
            model_config = meta_data["model_config"]
            print(f"\n✅ 모델 설정:")
            print(f"   입력 차원: {model_config.get('input_dim', 'N/A')}")
            print(f"   출력 차원: {model_config.get('output_size', 'N/A')}")
            
        # 학습시 사용된 피처 이름 확인
        if "feature_names" in meta_data:
            feature_names = meta_data["feature_names"]
            print(f"\n✅ 통합 모델 학습시 사용된 피처: {len(feature_names)}개")
            print(f"   예시: {feature_names[:10]}")
            
        # 개별 모델 차원 확인
        print("\n✅ 개별 모델 파일 확인:")
        model_dir = Path("models/trader/")
        individual_models = list(model_dir.glob("neural_predictor_pytorch_individual_*.pth"))
        for model_path in individual_models:
            symbol = model_path.stem.split("_")[-1]
            print(f"   {symbol}: {model_path.name}")
            
    else:
        print(f"❌ 메타 데이터 파일이 없습니다: {meta_path}")
        
    # 피처 정보 JSON 확인
    feature_info_path = "models/trader/neural_predictor_feature_info.json"
    if os.path.exists(feature_info_path):
        print("\n" + "=" * 80)
        print("📊 피처 정보 JSON")
        print("=" * 80)
        
        with open(feature_info_path, "r") as f:
            feature_info = json.load(f)
            
        if "individual_features" in feature_info:
            print("\n✅ 개별 모델 피처 (JSON):")
            for symbol, info in feature_info["individual_features"].items():
                feature_names = info.get("feature_names", [])
                print(f"   {symbol}: {len(feature_names)}개 피처")
                print(f"      차원: {info.get('input_dim', 'N/A')}")
                print(f"      Lookback: {info.get('lookback_days', 'N/A')}일")
                print(f"      총 입력 크기: {len(feature_names)} x {info.get('lookback_days', 'N/A')} = {len(feature_names) * info.get('lookback_days', 0)}")

if __name__ == "__main__":
    check_model_dimensions()