#!/usr/bin/env python3
"""
수익률 계산 디버깅 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.portfolio_manager import AdvancedPortfolioManager

def debug_returns():
    """수익률 계산 디버깅"""
    print("🔍 수익률 계산 디버깅 시작")
    
    # 설정 파일 경로
    config_path = "config/config_swing.json"
    
    # 포트폴리오 매니저 초기화
    portfolio_manager = AdvancedPortfolioManager(config_path)
    
    # 데이터 로드
    print("📊 데이터 로딩 중...")
    data_dir = "data/swing"  # 직접 data/swing 경로 사용
    data_dict = portfolio_manager.load_portfolio_data(data_dir)
    
    if not data_dict:
        print("❌ 데이터 로딩 실패")
        return
    
    print(f"✅ 데이터 로딩 완료: {len(data_dict)}개 종목")
    
    # 첫 번째 종목으로 상세 분석
    symbol = list(data_dict.keys())[0]
    data = data_dict[symbol]
    print(f"🔍 분석 종목: {symbol}")
    
    # 데이터 기간 확인
    print(f"📊 데이터 기간:")
    print(f"  - 시작: {data.index.min()}")
    print(f"  - 종료: {data.index.max()}")
    print(f"  - 총 기간: {(data.index.max() - data.index.min()).days}일")
    
    # 훈련/테스트 분할
    train_ratio = 0.6
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"📊 데이터 분할:")
    print(f"  - 훈련 기간: {train_data.index.min()} ~ {train_data.index.max()}")
    print(f"  - 테스트 기간: {test_data.index.min()} ~ {test_data.index.max()}")
    print(f"  - 훈련 데이터: {len(train_data)}개 행")
    print(f"  - 테스트 데이터: {len(test_data)}개 행")
    
    # 가격 변화 확인
    train_start_price = train_data['close'].iloc[0]
    train_end_price = train_data['close'].iloc[-1]
    train_return = (train_end_price - train_start_price) / train_start_price
    
    test_start_price = test_data['close'].iloc[0]
    test_end_price = test_data['close'].iloc[-1]
    test_return = (test_end_price - test_start_price) / test_start_price
    
    print(f"📊 Buy & Hold 수익률:")
    print(f"  - 훈련 기간: {train_start_price:.2f} → {train_end_price:.2f} = {train_return*100:.2f}%")
    print(f"  - 테스트 기간: {test_start_price:.2f} → {test_end_price:.2f} = {test_return*100:.2f}%")
    
    # 전략별 수익률 계산
    from src.actions.strategies import StrategyManager
    strategy_manager = StrategyManager()
    
    # dual_momentum 전략으로 테스트
    strategy_name = "dual_momentum"
    strategy = strategy_manager.strategies.get(strategy_name)
    
    if strategy:
        print(f"📊 {strategy_name} 전략 분석:")
        
        # 훈련 데이터로 신호 생성
        train_signals = strategy.generate_signals(train_data)
        if train_signals is not None and not train_signals.empty:
            signal_counts = train_signals['signal'].value_counts()
            print(f"  - 훈련 신호 분포: {dict(signal_counts)}")
            
            # 거래 시뮬레이션
            from src.actions.log_pl import TradingSimulator
            simulator = TradingSimulator(config_path)
            train_result = simulator.simulate_trading(train_data, train_signals, strategy_name)
            
            if train_result:
                results = train_result.get('results', {})
                trades = train_result.get('trades', [])
                print(f"  - 훈련 거래 횟수: {len(trades)}")
                print(f"  - 훈련 총 수익률: {results.get('total_return', 0)*100:.4f}%")
                
                if trades:
                    print(f"  - 훈련 거래 상세:")
                    for i, trade in enumerate(trades[:3]):  # 처음 3개만
                        entry_price = trade.get('entry_price', 0)
                        exit_price = trade.get('exit_price', 0)
                        pnl = trade.get('pnl', 0)
                        print(f"    거래 {i+1}: {entry_price:.2f} → {exit_price:.2f} = {pnl*100:.2f}%")
        
        # 테스트 데이터로 신호 생성
        test_signals = strategy.generate_signals(test_data)
        if test_signals is not None and not test_signals.empty:
            signal_counts = test_signals['signal'].value_counts()
            print(f"  - 테스트 신호 분포: {dict(signal_counts)}")
            
            # 거래 시뮬레이션
            test_result = simulator.simulate_trading(test_data, test_signals, strategy_name)
            
            if test_result:
                results = test_result.get('results', {})
                trades = test_result.get('trades', [])
                print(f"  - 테스트 거래 횟수: {len(trades)}")
                print(f"  - 테스트 총 수익률: {results.get('total_return', 0)*100:.4f}%")
                
                if trades:
                    print(f"  - 테스트 거래 상세:")
                    for i, trade in enumerate(trades[:3]):  # 처음 3개만
                        entry_price = trade.get('entry_price', 0)
                        exit_price = trade.get('exit_price', 0)
                        pnl = trade.get('pnl', 0)
                        print(f"    거래 {i+1}: {entry_price:.2f} → {exit_price:.2f} = {pnl*100:.2f}%")

if __name__ == "__main__":
    debug_returns() 