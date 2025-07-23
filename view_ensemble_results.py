#!/usr/bin/env python3
"""
앙상블 결과 조회 스크립트
Evaluator와 유사한 기능을 제공하여 앙상블 전략 결과를 쉽게 확인할 수 있습니다.
"""

import sys
import os
import argparse

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.actions.ensemble import EnsembleStrategy


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="앙상블 전략 결과 조회 도구",
        epilog="""
사용 예시:
  # 최신 결과 조회
  python view_ensemble_results.py
  
  # 특정 UUID 결과 조회
  python view_ensemble_results.py --uuid 20250123_154722_17b6186e
  
  # 상세 정보 포함 조회
  python view_ensemble_results.py --uuid 20250123_154722_17b6186e --detailed
  
  # 모든 결과 목록 조회
  python view_ensemble_results.py --list
  
  # 두 결과 비교
  python view_ensemble_results.py --compare --uuid 20250123_154722_17b6186e --uuid2 20250124_101245_28c7297f
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--uuid", help="조회할 UUID (지정하지 않으면 최신 결과)")
    parser.add_argument("--uuid2", help="비교할 두 번째 UUID (--compare와 함께 사용)")
    parser.add_argument("--detailed", action="store_true", help="상세 정보 출력")
    parser.add_argument("--list", action="store_true", help="모든 결과 목록 조회")
    parser.add_argument("--compare", action="store_true", help="두 결과 비교")
    parser.add_argument("--config", default="config/config_ensemble.json", help="앙상블 설정 파일")
    
    args = parser.parse_args()
    
    try:
        # 앙상블 전략 초기화
        ensemble = EnsembleStrategy(config_path=args.config)
        
        if args.list:
            # 모든 결과 목록 조회
            ensemble.list_all_results()
            
        elif args.compare:
            # 두 결과 비교
            if not args.uuid or not args.uuid2:
                print("❌ 비교 모드에서는 --uuid와 --uuid2가 모두 필요합니다.")
                print("예시: python view_ensemble_results.py --compare --uuid UUID1 --uuid2 UUID2")
                return
            
            ensemble.compare_results(args.uuid, args.uuid2)
            
        else:
            # 개별 결과 조회
            ensemble.view_results(uuid=args.uuid, detailed=args.detailed)
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 