#!/usr/bin/env python3

import traceback
from src.agent.evaluator import TrainTestEvaluator

try:
    print("=== Evaluator 최종 디버깅 시작 ===")
    
    evaluator = TrainTestEvaluator(
        config_path='config/config_swing.json',
        optimization_results_path='results/hyperparam_optimization_20250721_20250721_090734.json',
        portfolio_results_path='results/portfolio_optimization_20250721_094703.json'
    )
    
    print("✅ Evaluator 초기화 성공")
    
    result = evaluator.run_train_test_evaluation()
    print(f"평가 완료: {'성공' if result else '실패'}")
    
    if result:
        print(f"결과 키: {list(result.keys())}")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    traceback.print_exc() 