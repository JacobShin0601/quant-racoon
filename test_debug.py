#!/usr/bin/env python3
"""
시그널 생성과 시뮬레이션 디버그 테스트
"""

import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.actions.strategies import *
from src.actions.log_pl import TradingSimulator
from src.actions.calculate_index import StrategyParams
from src.agent.helper import load_and_preprocess_data


def test_dual_momentum_signals():
    """dual_momentum 전략의 시그널 생성 테스트"""
    print("=== DualMomentum 전략 시그널 생성 테스트 ===")

    # 데이터 로드
    data_dict = load_and_preprocess_data("data/swing", ["AAPL"])
    if not data_dict or "AAPL" not in data_dict:
        print("❌ AAPL 데이터를 로드할 수 없습니다")
        return

    df = data_dict["AAPL"]
    print(f"📊 데이터 로드 완료: {len(df)} 행")
    print(f"📅 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 전략 파라미터 설정
    params = StrategyParams()
    params.donchian_period = 20
    params.rsi_period = 14
    params.rsi_oversold = 30
    params.rsi_overbought = 70
    params.momentum_period = 10
    params.momentum_threshold = 0.02

    # 전략 인스턴스 생성
    strategy = DualMomentumStrategy(params)

    # 시그널 생성
    signals = strategy.generate_signals(df)

    if signals is None or signals.empty:
        print("❌ 시그널 생성 실패")
        return

    # 시그널 통계 확인
    signal_counts = signals["signal"].value_counts()
    print(f"📈 시그널 통계: {signal_counts.to_dict()}")

    # 시그널이 모두 0인지 확인
    if len(signal_counts) == 1 and 0 in signal_counts:
        print("⚠️ 모든 시그널이 0입니다!")
        return

    # 시그널 샘플 확인
    non_zero_signals = signals[signals["signal"] != 0]
    if not non_zero_signals.empty:
        print(f"✅ 0이 아닌 시그널 {len(non_zero_signals)}개 발견")
        print("📋 첫 5개 시그널:")
        print(non_zero_signals[["datetime", "close", "signal"]].head())
    else:
        print("⚠️ 0이 아닌 시그널이 없습니다!")
        return

    # 거래 시뮬레이션 테스트
    print("\n=== 거래 시뮬레이션 테스트 ===")
    simulator = TradingSimulator("config/config_swing.json")
    simulation_result = simulator.simulate_trading(df, signals, "dual_momentum_test")

    if not simulation_result:
        print("❌ 시뮬레이션 실패")
        return

    trades = simulation_result.get("trades", [])
    results = simulation_result.get("results", {})

    print(f"💰 거래 수: {len(trades)}")
    print(f"📊 총 수익률: {results.get('total_return', 0)*100:.2f}%")
    print(f"📈 승률: {results.get('win_rate', 0)*100:.1f}%")
    print(f"📉 최대 낙폭: {results.get('max_drawdown', 0)*100:.2f}%")

    if trades:
        print("📋 첫 번째 거래:")
        print(trades[0])
    else:
        print("⚠️ 거래가 없습니다!")


if __name__ == "__main__":
    test_dual_momentum_signals()
