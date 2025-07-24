#!/usr/bin/env python3
"""
고도화된 시장 분석 실행 스크립트
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


def run_enhanced_analysis(
    output_dir: str = "results/macro/enhanced",
    enable_llm_api: bool = False,
    llm_config: LLMConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    고도화된 시장 분석 실행
    
    Args:
        output_dir: 결과 저장 디렉토리
        enable_llm_api: LLM API 활성화 여부
        llm_config: LLM 설정
        verbose: 상세 로그 출력 여부
    
    Returns:
        분석 결과 딕셔너리
    """
    if verbose:
        print("🚀 고도화된 시장 분석 시작")
    
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
        
        # 고도화된 분석 수행
        if verbose:
            print("🚀 고도화된 분석 수행 중...")
        
        analysis = sensor.get_enhanced_market_analysis(
            use_optimized_params=True,
            use_ml_model=True,
            enable_advanced_features=True
        )
        
        if verbose:
            print("✅ 고도화된 분석 완료")
        
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
            
            if 'final_confidence' in analysis:
                final_conf = analysis['final_confidence'].get('final_confidence', 0.5)
                print(f"최종 신뢰도: {final_conf:.3f}")
            
            if 'rlmf_analysis' in analysis:
                rlmf = analysis['rlmf_analysis']
                if 'statistical_arbitrage' in rlmf:
                    stat_arb = rlmf['statistical_arbitrage']
                    print(f"Statistical Arbitrage: {stat_arb.get('direction', 'N/A')} (신뢰도: {stat_arb.get('confidence', 0):.3f})")
            
            if 'regime_detection' in analysis:
                regime_det = analysis['regime_detection']
                if 'regime_shift_detection' in regime_det:
                    shift_det = regime_det['regime_shift_detection']
                    if shift_det.get('regime_shift_detected', False):
                        print("⚠️ 시장 체제 전환 감지됨!")
            
            if 'llm_api_insights' in analysis:
                print("🤖 LLM API 분석 완료")
                api_stats = analysis['llm_api_insights'].get('api_stats', {})
                if api_stats:
                    print(f"API 성공률: {api_stats.get('success_rate', 0):.2%}")
            
            # LLM API 통계 (활성화된 경우)
            if sensor.llm_api_system:
                stats = sensor.get_llm_api_stats()
                print(f"\n📈 LLM API 통계:")
                print(f"총 호출: {stats.get('total_calls', 0)}")
                print(f"성공률: {stats.get('success_rate', 0):.2%}")
                print(f"평균 응답시간: {stats.get('avg_response_time', 0):.3f}초")
            
            print("🎉 고도화된 분석 완료!")
        
        return analysis
        
    except Exception as e:
        if verbose:
            print(f"❌ 고도화된 분석 실패: {e}")
        return {}


def main():
    """메인 실행 함수"""
    # 명령행 인수 파싱
    import argparse
    
    parser = argparse.ArgumentParser(description="고도화된 시장 분석")
    parser.add_argument("--output", type=str, default="results/macro/enhanced", 
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
    result = run_enhanced_analysis(
        output_dir=args.output,
        enable_llm_api=args.enable_llm,
        llm_config=llm_config,
        verbose=args.verbose
    )
    
    return result


if __name__ == "__main__":
    main() 