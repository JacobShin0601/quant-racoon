#!/usr/bin/env python3
"""
최적화 프로세스 디버깅 스크립트
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

from src.actions.optimize_threshold import ThresholdOptimizer


def debug_optimization():
    """최적화 프로세스 디버깅"""

    print("🔍 최적화 프로세스 디버깅 시작")
    print("=" * 60)

    # 1. 설정 로드
    print("1️⃣ 설정 로드")
    with open("config/config_trader.json", "r") as f:
        config = json.load(f)
    print(f"   - 최적화 방법: {config['optimization']['method']}")
    print(f"   - 목표 지표: {config['optimization']['objective_metric']}")
    print(f"   - Threshold 범위: {config['optimization']['threshold_ranges']}")

    # 2. ThresholdOptimizer 초기화
    print("\n2️⃣ ThresholdOptimizer 초기화")
    optimizer = ThresholdOptimizer(config)
    print(f"   - Train 비율: {optimizer.train_ratio}")
    print(f"   - 최소 데이터 포인트: {optimizer.min_data_points}")

    # 3. 데이터 로드 및 분할
    print("\n3️⃣ 데이터 로드 및 분할")
    symbols = ["AAPL", "META", "QQQ", "SPY"]
    train_data, test_data = optimizer.load_and_split_data(symbols)

    print(f"   - Train 데이터 종목: {list(train_data.keys())}")
    print(f"   - Test 데이터 종목: {list(test_data.keys())}")

    for symbol in symbols:
        if symbol in train_data:
            print(f"   - {symbol} Train: {len(train_data[symbol])}일")
        if symbol in test_data:
            print(f"   - {symbol} Test: {len(test_data[symbol])}일")

    # 4. 테스트 threshold로 백테스팅
    print("\n4️⃣ 테스트 threshold로 백테스팅")
    test_thresholds = {
        "strong_buy": 0.5,
        "buy": 0.3,
        "hold_upper": 0.1,
        "hold_lower": -0.1,
        "sell": -0.3,
        "strong_sell": -0.5,
    }

    print(f"   - 테스트 threshold: {test_thresholds}")

    # Train 데이터로 백테스팅
    print("\n   📊 Train 데이터 백테스팅:")
    train_results = optimizer.backtest_with_thresholds(train_data, test_thresholds)
    print(f"   - 총 거래 수: {train_results['total_trades']}")
    print(f"   - 포트폴리오 성과: {train_results['portfolio_performance']}")

    # Test 데이터로 백테스팅
    print("\n   📊 Test 데이터 백테스팅:")
    test_results = optimizer.backtest_with_thresholds(test_data, test_thresholds)
    print(f"   - 총 거래 수: {test_results['total_trades']}")
    print(f"   - 포트폴리오 성과: {test_results['portfolio_performance']}")

    # 5. 개별 종목 상세 분석
    print("\n5️⃣ 개별 종목 상세 분석")
    for symbol in symbols:
        if symbol in train_data:
            print(f"\n   🔍 {symbol} Train 데이터:")
            symbol_train_results = train_results["symbol_results"].get(symbol, {})
            if symbol_train_results:
                trades = symbol_train_results.get("trades", [])
                performance = symbol_train_results.get("performance", {})
                print(f"   - 거래 수: {len(trades)}")
                print(f"   - 수익률: {performance.get('total_return', 0):.4f}")
                print(f"   - 샤프 비율: {performance.get('sharpe_ratio', 0):.4f}")

                # 처음 3개 거래 상세
                if trades:
                    print(f"   - 처음 3개 거래:")
                    for i, trade in enumerate(trades[:3]):
                        print(
                            f"     {i+1}. {trade.get('entry_date')} ~ {trade.get('exit_date')}: {trade.get('pnl', 0):.4f}"
                        )
                else:
                    print(f"   - 거래 없음")

    # 6. 예측값 범위 확인
    print("\n6️⃣ 예측값 범위 확인")
    for symbol in symbols:
        if symbol in train_data:
            data = train_data[symbol]
            close = data["close"]

            # RSI 계산
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # 이동평균 계산
            ma_20 = close.rolling(20).mean()
            ma_50 = close.rolling(50).mean()

            # 예측값 계산
            predictions = []
            for i in range(len(data)):
                if i < 50:
                    predictions.append(0.0)
                    continue

                rsi_current = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
                rsi_prediction = (rsi_current - 50) / 50

                if not pd.isna(ma_20.iloc[i]) and not pd.isna(ma_50.iloc[i]):
                    ma_signal = 1 if ma_20.iloc[i] > ma_50.iloc[i] else -1
                    combined_prediction = (rsi_prediction + ma_signal * 0.3) / 2
                else:
                    combined_prediction = rsi_prediction

                final_prediction = float(np.clip(combined_prediction, -1, 1))
                predictions.append(final_prediction)

            predictions_array = np.array(predictions)
            print(
                f"   - {symbol}: {predictions_array.min():.4f} ~ {predictions_array.max():.4f} (평균: {predictions_array.mean():.4f})"
            )

            # threshold별 신호 수
            strong_buy_count = sum(
                1 for p in predictions if p >= test_thresholds["strong_buy"]
            )
            buy_count = sum(
                1
                for p in predictions
                if test_thresholds["buy"] <= p < test_thresholds["strong_buy"]
            )
            sell_count = sum(
                1
                for p in predictions
                if test_thresholds["sell"] < p <= test_thresholds["hold_lower"]
            )
            strong_sell_count = sum(
                1 for p in predictions if p <= test_thresholds["strong_sell"]
            )

            print(
                f"     신호: STRONG_BUY={strong_buy_count}, BUY={buy_count}, SELL={sell_count}, STRONG_SELL={strong_sell_count}"
            )


if __name__ == "__main__":
    debug_optimization()
