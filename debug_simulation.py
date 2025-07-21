#!/usr/bin/env python3
"""
거래 시뮬레이션 디버깅 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.portfolio_manager import AdvancedPortfolioManager
from src.actions.log_pl import TradingSimulator
from src.strategies.strategy_manager import StrategyManager

def debug_simulation():
    """거래 시뮬레이션 디버깅"""
    print("🔍 거래 시뮬레이션 디버깅 시작")
    
    # 설정 파일 경로
    config_path = "config/config_swing.json"
    
    # 포트폴리오 매니저 초기화
    portfolio_manager = AdvancedPortfolioManager(config_path)
    
    # 데이터 로드
    print("📊 데이터 로딩 중...")
    data_dict = portfolio_manager.load_data()
    
    if not data_dict:
        print("❌ 데이터 로딩 실패")
        return
    
    print(f"✅ 데이터 로딩 완료: {len(data_dict)}개 종목")
    
    # 첫 번째 종목으로 테스트
    symbol = list(data_dict.keys())[0]
    data = data_dict[symbol]
    print(f"🔍 테스트 종목: {symbol}")
    print(f"📊 데이터 shape: {data.shape}")
    print(f"📊 데이터 컬럼: {list(data.columns)}")
    print(f"📊 데이터 샘플:")
    print(data.head())
    
    # 전략 매니저 초기화
    strategy_manager = StrategyManager()
    
    # dual_momentum 전략으로 테스트
    strategy_name = "dual_momentum"
    strategy = strategy_manager.strategies.get(strategy_name)
    
    if not strategy:
        print(f"❌ 전략을 찾을 수 없습니다: {strategy_name}")
        return
    
    print(f"✅ 전략 로드: {strategy_name}")
    
    # 신호 생성
    print("📈 신호 생성 중...")
    signals = strategy.generate_signals(data)
    
    if signals is None or signals.empty:
        print("❌ 신호 생성 실패")
        return
    
    print(f"✅ 신호 생성 완료: {signals.shape}")
    print(f"📊 신호 컬럼: {list(signals.columns)}")
    print(f"📊 신호 샘플:")
    print(signals.head(10))
    
    # 신호 분포 확인
    signal_counts = signals["signal"].value_counts()
    print(f"📊 신호 분포:")
    print(signal_counts)
    
    # 0이 아닌 신호가 있는지 확인
    non_zero_signals = signals[signals["signal"] != 0]
    print(f"📊 0이 아닌 신호 개수: {len(non_zero_signals)}")
    
    if len(non_zero_signals) > 0:
        print(f"📊 0이 아닌 신호 샘플:")
        print(non_zero_signals.head())
    
    # 거래 시뮬레이션
    print("🔄 거래 시뮬레이션 시작...")
    simulator = TradingSimulator(config_path)
    simulation_result = simulator.simulate_trading(data, signals, strategy_name)
    
    if not simulation_result:
        print("❌ 시뮬레이션 실패")
        return
    
    print("✅ 시뮬레이션 완료")
    
    # 결과 분석
    results = simulation_result.get("results", {})
    trades = simulation_result.get("trades", [])
    returns = simulation_result.get("returns", [])
    
    print(f"📊 시뮬레이션 결과:")
    print(f"  - 총 수익률: {results.get('total_return', 0)*100:.2f}%")
    print(f"  - 거래 횟수: {len(trades)}")
    print(f"  - 승률: {results.get('win_rate', 0)*100:.1f}%")
    print(f"  - 샤프 비율: {results.get('sharpe_ratio', 0):.3f}")
    print(f"  - 최대 낙폭: {results.get('max_drawdown', 0)*100:.2f}%")
    
    if trades:
        print(f"📊 거래 기록 (처음 5개):")
        for i, trade in enumerate(trades[:5]):
            print(f"  거래 {i+1}: {trade}")
    
    if returns:
        print(f"📊 수익률 데이터:")
        print(f"  - 수익률 개수: {len(returns)}")
        print(f"  - 평균 수익률: {np.mean(returns)*100:.4f}%")
        print(f"  - 수익률 표준편차: {np.std(returns)*100:.4f}%")
        print(f"  - 최대 수익률: {np.max(returns)*100:.4f}%")
        print(f"  - 최소 수익률: {np.min(returns)*100:.4f}%")

if __name__ == "__main__":
    debug_simulation() 