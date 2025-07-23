#!/usr/bin/env python3
"""
Swing 전략 결과 조회 스크립트
config_swing.json 기반 evaluator 결과를 쉽게 확인할 수 있습니다.
"""

import sys
import os
import argparse
from pathlib import Path
import json
import glob
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.agent.evaluator import TrainTestEvaluator


def list_available_results(results_dir: str = "results/swing"):
    """사용 가능한 결과 파일들 나열"""
    print("📂 사용 가능한 Swing 전략 결과 파일들:")
    print("=" * 60)
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ 결과 디렉토리가 존재하지 않습니다: {results_dir}")
        return
    
    # 최적화 결과 파일들
    optimization_files = list(results_path.glob("hyperparam_optimization_*.json"))
    portfolio_files = list(results_path.glob("portfolio_optimization_*.json"))
    performance_files = list(results_path.glob("performance_evaluation_*.txt"))
    
    print(f"📊 하이퍼파라미터 최적화 결과: {len(optimization_files)}개")
    for i, file in enumerate(sorted(optimization_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5], 1):
        mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. {file.name} ({mod_time})")
    
    print(f"\n💼 포트폴리오 최적화 결과: {len(portfolio_files)}개")
    for i, file in enumerate(sorted(portfolio_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5], 1):
        mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. {file.name} ({mod_time})")
    
    print(f"\n📈 성과 평가 결과: {len(performance_files)}개")
    for i, file in enumerate(sorted(performance_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5], 1):
        mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. {file.name} ({mod_time})")
    
    print("=" * 60)


def run_evaluator(config_path: str = "config/config_swing.json", 
                 data_dir: str = "data/swing",
                 symbols: list = None,
                 optimization_results: str = None,
                 portfolio_results: str = None):
    """Evaluator 실행"""
    print("🚀 Train/Test 평가 시스템 실행")
    print("=" * 60)
    
    try:
        # Evaluator 초기화
        evaluator = TrainTestEvaluator(
            data_dir=data_dir,
            log_mode="summary",
            config_path=config_path,
            optimization_results_path=optimization_results,
            portfolio_results_path=portfolio_results,
        )
        
        # 평가 실행
        results = evaluator.run_train_test_evaluation(
            symbols=symbols,
            save_results=True,
        )
        
        if results:
            print("\n✅ 평가 완료!")
            if results.get("table_path"):
                print(f"📄 성과 테이블: {results['table_path']}")
        else:
            print("❌ 평가 실패")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def view_latest_performance(results_dir: str = "results/swing"):
    """최신 성과 평가 결과 파일 내용 출력"""
    results_path = Path(results_dir)
    
    # 최신 성과 평가 파일 찾기
    performance_files = list(results_path.glob("performance_evaluation_*.txt"))
    if not performance_files:
        print(f"❌ 성과 평가 결과 파일을 찾을 수 없습니다: {results_dir}")
        return
    
    latest_file = max(performance_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📄 최신 성과 평가 결과: {latest_file.name}")
    print("=" * 80)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")


def analyze_optimization_results(results_dir: str = "results/swing"):
    """최적화 결과 분석"""
    results_path = Path(results_dir)
    
    # 최신 최적화 결과 파일 찾기
    optimization_files = list(results_path.glob("hyperparam_optimization_*.json"))
    if not optimization_files:
        print(f"❌ 최적화 결과 파일을 찾을 수 없습니다: {results_dir}")
        return
    
    latest_file = max(optimization_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📊 최적화 결과 분석: {latest_file.name}")
    print("=" * 80)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 결과 분석
        strategy_scores = {}
        symbol_scores = {}
        
        for key, result in results.items():
            strategy = result.get('strategy_name', 'UNKNOWN')
            symbol = result.get('symbol', 'UNKNOWN')
            score = result.get('best_score', 0)
            
            if strategy not in strategy_scores:
                strategy_scores[strategy] = []
            strategy_scores[strategy].append(score)
            
            if symbol not in symbol_scores:
                symbol_scores[symbol] = []
            symbol_scores[symbol].append(score)
        
        # 전략별 평균 점수
        print("📈 전략별 평균 점수:")
        print("-" * 40)
        strategy_avg = {s: sum(scores)/len(scores) for s, scores in strategy_scores.items()}
        for strategy, avg_score in sorted(strategy_avg.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy:<25}: {avg_score:>8.2f}")
        
        # 종목별 평균 점수
        print(f"\n💼 종목별 평균 점수:")
        print("-" * 40)
        symbol_avg = {s: sum(scores)/len(scores) for s, scores in symbol_scores.items()}
        for symbol, avg_score in sorted(symbol_avg.items(), key=lambda x: x[1], reverse=True):
            print(f"  {symbol:<8}: {avg_score:>8.2f}")
        
        # 전체 통계
        all_scores = [result.get('best_score', 0) for result in results.values()]
        valid_scores = [s for s in all_scores if s > -999999]
        
        print(f"\n📊 전체 통계:")
        print("-" * 40)
        print(f"  총 조합 수: {len(results)}")
        print(f"  성공 조합: {len(valid_scores)}")
        print(f"  성공률: {len(valid_scores)/len(results)*100:.1f}%")
        if valid_scores:
            print(f"  평균 점수: {sum(valid_scores)/len(valid_scores):.2f}")
            print(f"  최고 점수: {max(valid_scores):.2f}")
            print(f"  최저 점수: {min(valid_scores):.2f}")
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Swing 전략 결과 조회 및 평가 도구",
        epilog="""
사용 예시:
  # 사용 가능한 결과 파일 목록
  python view_swing_results.py --list
  
  # Evaluator 실행 (기본 설정)
  python view_swing_results.py --run
  
  # 특정 종목만 평가
  python view_swing_results.py --run --symbols AAPL META NFLX
  
  # 최신 성과 결과 보기
  python view_swing_results.py --view
  
  # 최적화 결과 분석
  python view_swing_results.py --analyze
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--list", action="store_true", help="사용 가능한 결과 파일 목록")
    parser.add_argument("--run", action="store_true", help="Evaluator 실행")
    parser.add_argument("--view", action="store_true", help="최신 성과 결과 보기")
    parser.add_argument("--analyze", action="store_true", help="최적화 결과 분석")
    
    parser.add_argument("--config", default="config/config_swing.json", help="설정 파일")
    parser.add_argument("--data-dir", default="data/swing", help="데이터 디렉토리")
    parser.add_argument("--results-dir", default="results/swing", help="결과 디렉토리")
    parser.add_argument("--symbols", nargs="+", help="평가할 종목 목록")
    parser.add_argument("--optimization-results", help="최적화 결과 파일 경로")
    parser.add_argument("--portfolio-results", help="포트폴리오 결과 파일 경로")
    
    args = parser.parse_args()
    
    # 옵션이 없으면 기본적으로 list 실행
    if not any([args.list, args.run, args.view, args.analyze]):
        args.list = True
    
    try:
        if args.list:
            list_available_results(args.results_dir)
        
        if args.run:
            run_evaluator(
                config_path=args.config,
                data_dir=args.data_dir,
                symbols=args.symbols,
                optimization_results=args.optimization_results,
                portfolio_results=args.portfolio_results
            )
        
        if args.view:
            view_latest_performance(args.results_dir)
        
        if args.analyze:
            analyze_optimization_results(args.results_dir)
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 