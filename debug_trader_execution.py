#!/usr/bin/env python3
"""
실제 트레이더 실행 시 Threshold 사용 디버깅 스크립트
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

from src.agent.trader import HybridTrader


def debug_trader_execution():
    """실제 트레이더 실행 시 threshold 사용 현황 디버깅"""

    print("🔍 실제 트레이더 실행 시 Threshold 사용 현황 디버깅")
    print("=" * 70)

    # 1. HybridTrader 초기화 (실제 실행과 동일한 방식)
    print("1️⃣ HybridTrader 초기화")
    trader = HybridTrader("config/config_trader.json", analysis_mode=False)

    # 2. 초기 threshold 확인
    print("\n2️⃣ 초기 threshold 확인")
    if hasattr(trader, "signal_generator") and trader.signal_generator:
        initial_thresholds = trader.signal_generator.signal_thresholds
        print(f"   - 초기 threshold: {initial_thresholds}")
    else:
        print(f"   ❌ 신호 생성기를 찾을 수 없음")
        return

    # 3. 최적화된 threshold 로드
    print("\n3️⃣ 최적화된 threshold 로드")
    load_success = trader.load_optimized_thresholds()
    print(f"   - 로드 성공: {load_success}")

    # 4. 로드 후 threshold 확인
    print("\n4️⃣ 로드 후 threshold 확인")
    if hasattr(trader, "signal_generator") and trader.signal_generator:
        updated_thresholds = trader.signal_generator.signal_thresholds
        print(f"   - 업데이트된 threshold: {updated_thresholds}")

        # 변경 여부 확인
        if initial_thresholds != updated_thresholds:
            print(f"   ✅ Threshold가 업데이트됨!")
        else:
            print(f"   ⚠️ Threshold가 변경되지 않음")
    else:
        print(f"   ❌ 신호 생성기를 찾을 수 없음")

    # 5. 실제 신호 생성 테스트
    print("\n5️⃣ 실제 신호 생성 테스트")
    try:
        # 테스트용 투자 점수
        test_investment_score = {
            "symbol": "AAPL",
            "final_score": 0.8,
            "confidence": 0.7,
            "components": {
                "neural_score": 0.8,
                "technical_score": 0.7,
                "fundamental_score": 0.6,
            },
        }

        # 신호 생성
        signal = trader.signal_generator.generate_signal(test_investment_score)
        action = signal.get("action", "UNKNOWN")

        print(f"   - 테스트 점수: {test_investment_score['final_score']}")
        print(f"   - 생성된 신호: {action}")
        print(f"   - 사용된 threshold: {trader.signal_generator.signal_thresholds}")

    except Exception as e:
        print(f"   ❌ 신호 생성 테스트 실패: {e}")

    # 6. run_analysis() 실행 시 threshold 확인
    print("\n6️⃣ run_analysis() 실행 시 threshold 확인")
    try:
        # 분석 실행
        analysis_results = trader.run_analysis()

        # 개별 종목 결과에서 신호 확인
        individual_results = analysis_results.get("individual_results", [])
        if individual_results:
            print(f"   - 분석된 종목 수: {len(individual_results)}")

            # 첫 번째 종목의 신호 확인
            first_result = individual_results[0]
            trading_signal = first_result.get("trading_signal", {})
            symbol = trading_signal.get("symbol", "UNKNOWN")
            action = trading_signal.get("action", "UNKNOWN")
            score = first_result.get("investment_score", {}).get("final_score", 0.0)

            print(f"   - 첫 번째 종목 ({symbol}):")
            print(f"     - 점수: {score:.4f}")
            print(f"     - 신호: {action}")
            print(
                f"     - 사용된 threshold: {trader.signal_generator.signal_thresholds}"
            )
        else:
            print(f"   ❌ 개별 종목 결과가 없음")

    except Exception as e:
        print(f"   ❌ run_analysis() 실행 실패: {e}")

    # 7. 포트폴리오 관리자 threshold 확인
    print("\n7️⃣ 포트폴리오 관리자 threshold 확인")
    if hasattr(trader, "portfolio_manager") and trader.portfolio_manager:
        portfolio_thresholds = trader.portfolio_manager.get_signal_thresholds()
        print(f"   - 포트폴리오 관리자 threshold: {portfolio_thresholds}")
    else:
        print(f"   ❌ 포트폴리오 관리자가 초기화되지 않음")


if __name__ == "__main__":
    debug_trader_execution()
