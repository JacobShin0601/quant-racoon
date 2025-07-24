#!/usr/bin/env python3
"""
고도화된 시장 분석 실행 스크립트
생성 시간: 2025. 07. 24. (목) 10:33:49 KST
분석 유형: enhanced
LLM 제공자: bedrock
모델: claude-opus-4
"""

import sys
import os
import json
from datetime import datetime

# 프로젝트 루트 경로 추가 (수정된 경로 계산)
current_dir = os.path.dirname(os.path.abspath(__file__))
# results/macro/basic -> results/macro -> results -> quant-racoon (프로젝트 루트)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)
print(f"현재 디렉토리: {current_dir}")
print(f"프로젝트 루트: {project_root}")

from src.agent.market_sensor import MarketSensor
from src.agent.enhancements import LLMConfig

def main():
    print("🚀 시장 분석 시작")
    print(f"분석 유형: enhanced")
    
    # LLM 설정
    llm_config = None
    enable_llm_api = False
    
    # enhanced, llm-api, full 옵션에서 LLM API 활성화
    if "enhanced" in ["enhanced", "llm-api", "full"]:
        llm_config = LLMConfig(
            provider="bedrock",
            model_name="us.anthropic.claude-opus-4-20250514-v1:0",
            api_key="" if "" else None,
            region="us-east-1",
            fallback_to_rules=True
        )
        enable_llm_api = True
        print(f"🤖 LLM API 설정: {llm_config.provider} - {llm_config.model_name}")
    
    # Market Sensor 초기화
    sensor = MarketSensor(
        enable_llm_api=enable_llm_api,
        llm_config=llm_config
    )
    
    # 분석 수행 (분석 유형에 따라 올바른 메서드 호출)
    if "enhanced" == "basic":
        print("📊 기본 분석 수행 중...")
        analysis = sensor.run_basic_analysis(
            output_dir="results/macro/enhanced",
            verbose=True
        )
    elif "enhanced" == "enhanced":
        print("🚀 고도화된 분석 수행 중...")
        analysis = sensor.run_enhanced_analysis(
            output_dir="results/macro/enhanced",
            verbose=True
        )
    elif "enhanced" == "llm-api":
        print("🤖 LLM API 분석 수행 중...")
        analysis = sensor.run_enhanced_analysis(
            output_dir="results/macro/enhanced",
            verbose=True
        )
    elif "enhanced" == "full":
        print("🎯 전체 통합 분석 수행 중...")
        analysis = sensor.run_enhanced_analysis(
            output_dir="results/macro/enhanced",
            verbose=True
        )
    else:
        print("❌ 알 수 없는 분석 유형: enhanced")
        return
    
    print("✅ 분석 완료!")
    print(f"결과 저장 위치: results/macro/enhanced")

if __name__ == "__main__":
    main()
