#!/usr/bin/env python3
"""
기본 시장 분석 실행 스크립트
src/actions 폴더에서 실행
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.agent.market_sensor import MarketSensor
from src.agent.enhancements import LLMConfig


def run_basic_analysis(
    output_dir: str = "results/macro/basic",
    enable_llm_api: bool = False,
    llm_config: LLMConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    기본 시장 분석 실행
    
    Args:
        output_dir: 결과 저장 디렉토리
        enable_llm_api: LLM API 활성화 여부
        llm_config: LLM 설정
        verbose: 상세 로그 출력 여부
    
    Returns:
        분석 결과 딕셔너리
    """
    if verbose:
        print("🚀 기본 시장 분석 시작")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Market Sensor 초기화
        sensor = MarketSensor(
            enable_llm_api=enable_llm_api,
            llm_config=llm_config
        )
        
        if verbose:
            print("✅ Market Sensor 초기화 성공")
        
        # 기본 분석 수행
        if verbose:
            print("📊 기본 분석 수행 중...")
        
        analysis = sensor.get_current_market_analysis(
            use_optimized_params=True,
            use_ml_model=True
        )
        
        if verbose:
            print("✅ 기본 분석 완료")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/analysis_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            print(f"✅ 결과 저장: {output_file}")
        
        # 요약 출력
        if verbose:
            print("\n📊 분석 결과 요약:")
            print(f"현재 체제: {analysis.get('current_regime', 'N/A')}")
            
            if 'confidence' in analysis:
                print(f"신뢰도: {analysis.get('confidence', 0.5):.3f}")
            
            if 'probabilities' in analysis:
                probs = analysis['probabilities']
                print("체제 확률:")
                for regime, prob in probs.items():
                    print(f"  {regime}: {prob:.3f}")
            
            print("🎉 기본 분석 완료!")
        
        return analysis
        
    except Exception as e:
        if verbose:
            print(f"❌ 기본 분석 실패: {e}")
        return {}


def main():
    """메인 실행 함수"""
    # 명령행 인수 파싱
    import argparse
    
    parser = argparse.ArgumentParser(description="기본 시장 분석")
    parser.add_argument("--output", type=str, default="results/macro/basic", 
                       help="결과 출력 디렉토리")
    parser.add_argument("--enable-llm", action="store_true", 
                       help="LLM API 활성화")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # LLM 설정 (활성화된 경우)
    llm_config = None
    if args.enable_llm:
        llm_config = LLMConfig(
            provider="hybrid",
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            fallback_to_rules=True
        )
    
    # 분석 실행
    result = run_basic_analysis(
        output_dir=args.output,
        enable_llm_api=args.enable_llm,
        llm_config=llm_config,
        verbose=args.verbose
    )
    
    return result


if __name__ == "__main__":
    main() 