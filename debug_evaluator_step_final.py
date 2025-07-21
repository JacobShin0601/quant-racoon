#!/usr/bin/env python3

import traceback
from src.agent.evaluator import TrainTestEvaluator

try:
    print("=== Evaluator 단계별 디버깅 시작 ===")
    
    evaluator = TrainTestEvaluator(
        data_dir='data/swing',
        config_path='config/config_swing.json',
        optimization_results_path='results/hyperparam_optimization_20250721_20250721_090734.json',
        portfolio_results_path='results/portfolio_optimization_20250721_094703.json'
    )
    
    print("✅ Evaluator 초기화 성공")
    
    # 1단계: 데이터 로드 및 분할
    print("\n=== 1단계: 데이터 로드 및 분할 ===")
    train_data_dict, test_data_dict = evaluator.load_data_and_split()
    print(f"Train 데이터: {len(train_data_dict)}개 종목")
    print(f"Test 데이터: {len(test_data_dict)}개 종목")
    
    # 2단계: 최적화 결과 로드
    print("\n=== 2단계: 최적화 결과 로드 ===")
    optimization_results = evaluator.load_optimization_results()
    print(f"최적화 결과: {len(optimization_results)}개 조합")
    
    # 3단계: 포트폴리오 결과 로드
    print("\n=== 3단계: 포트폴리오 결과 로드 ===")
    portfolio_results = evaluator.load_portfolio_results()
    print(f"포트폴리오 결과 키: {list(portfolio_results.keys())}")
    
    # 4단계: 전략별 평가
    print("\n=== 4단계: 전략별 평가 ===")
    individual_results = evaluator.evaluate_all_strategies(
        train_data_dict, test_data_dict, optimization_results, portfolio_results
    )
    print(f"개별 결과 키: {list(individual_results.keys())}")
    
    # 5단계: 포트폴리오 성과 계산
    print("\n=== 5단계: 포트폴리오 성과 계산 ===")
    portfolio_weights = portfolio_results.get("portfolio_weights", {})
    portfolio_performance = evaluator.calculate_portfolio_performance(
        individual_results, portfolio_results
    )
    print(f"포트폴리오 성과 키: {list(portfolio_performance.keys())}")
    
    # 6단계: 거래 내역 로그 저장
    print("\n=== 6단계: 거래 내역 로그 저장 ===")
    evaluator.save_transaction_logs(
        individual_results, train_data_dict, test_data_dict
    )
    print("✅ 거래 내역 로그 저장 완료")
    
    # 7단계: 성과 요약 출력
    print("\n=== 7단계: 성과 요약 출력 ===")
    evaluator._print_performance_summary(
        individual_results, portfolio_performance, portfolio_weights,
        train_data_dict, test_data_dict
    )
    print("✅ 성과 요약 출력 완료")
    
    print("\n=== 모든 단계 완료 ===")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    traceback.print_exc() 