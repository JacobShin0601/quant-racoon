#!/usr/bin/env python3
"""
Threshold 최적화 테스트 스크립트
간단한 예제로 threshold 최적화 시스템을 테스트합니다.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleThresholdOptimizer:
    """간단한 Threshold 최적화 시스템 (테스트용)"""

    def __init__(self):
        self.results_dir = "results/trader"
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_mock_data(
        self, symbols: List[str], days: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """모의 주가 데이터 생성"""
        logger.info(f"모의 데이터 생성: {symbols}")

        data = {}
        start_date = datetime.now() - timedelta(days=days)

        for symbol in symbols:
            # 랜덤 주가 데이터 생성
            np.random.seed(hash(symbol) % 1000)  # 심볼별 고정 시드

            # 기본 가격 (100-500 범위)
            base_price = 100 + hash(symbol) % 400

            # 일별 수익률 (정규분포)
            daily_returns = np.random.normal(
                0.0005, 0.02, days
            )  # 평균 0.05%, 표준편차 2%

            # 가격 시계열 생성
            prices = [base_price]
            for ret in daily_returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # OHLCV 데이터 생성
            dates = pd.date_range(start=start_date, periods=days, freq="D")

            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, days),
                },
                index=dates,
            )

            # 고가/저가 조정
            df["high"] = df[["open", "close"]].max(axis=1) * (
                1 + abs(np.random.normal(0, 0.005))
            )
            df["low"] = df[["open", "close"]].min(axis=1) * (
                1 - abs(np.random.normal(0, 0.005))
            )

            data[symbol] = df
            logger.info(f"  {symbol}: {len(df)}일 데이터 생성")

        return data

    def simulate_trading(
        self, data: pd.DataFrame, thresholds: Dict[str, float]
    ) -> Dict[str, float]:
        """간단한 거래 시뮬레이션"""
        try:
            # 간단한 기술적 지표 계산
            data = data.copy()
            data["returns"] = data["close"].pct_change()
            data["ma_20"] = data["close"].rolling(20).mean()
            data["rsi"] = self.calculate_rsi(data["close"])

            # 매매 신호 생성
            signals = []
            position = 0
            entry_price = 0
            trades = []

            for i in range(20, len(data)):
                current_price = data["close"].iloc[i]
                current_ma = data["ma_20"].iloc[i]
                current_rsi = data["rsi"].iloc[i]

                # 간단한 매매 로직
                score = 0.0

                # 이동평균 기반 점수
                if current_price > current_ma:
                    score += 0.3
                else:
                    score -= 0.3

                # RSI 기반 점수
                if current_rsi < 30:
                    score += 0.4
                elif current_rsi > 70:
                    score -= 0.4

                # 매수/매도 신호 결정
                action = "HOLD"
                if score > thresholds["buy"]:
                    action = "BUY"
                elif score < thresholds["sell"]:
                    action = "SELL"

                # 거래 실행
                if action == "BUY" and position <= 0:
                    if position == -1:  # 매도 포지션 청산
                        exit_price = current_price
                        pnl = (entry_price - exit_price) / entry_price
                        trades.append(pnl)

                    position = 1
                    entry_price = current_price

                elif action == "SELL" and position >= 0:
                    if position == 1:  # 매수 포지션 청산
                        exit_price = current_price
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append(pnl)

                    position = -1
                    entry_price = current_price

            # 마지막 포지션 청산
            if position != 0:
                last_price = data["close"].iloc[-1]
                if position == 1:
                    pnl = (last_price - entry_price) / entry_price
                    trades.append(pnl)
                else:
                    pnl = (entry_price - last_price) / entry_price
                    trades.append(pnl)

            # 성과 계산
            if not trades:
                return {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                }

            total_return = sum(trades)
            returns = np.array(trades)

            # 샤프 비율
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            )

            # 소르티노 비율
            negative_returns = returns[returns < 0]
            sortino_ratio = (
                np.mean(returns) / np.std(negative_returns)
                if len(negative_returns) > 0 and np.std(negative_returns) > 0
                else 0.0
            )

            # 최대 낙폭
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

            # 승률
            win_rate = np.sum(returns > 0) / len(returns)

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": len(trades),
            }

        except Exception as e:
            logger.error(f"거래 시뮬레이션 실패: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def optimize_thresholds(
        self, symbols: List[str], n_trials: int = 10
    ) -> Dict[str, Any]:
        """Threshold 최적화"""
        logger.info(f"Threshold 최적화 시작: {symbols}")

        # 모의 데이터 생성
        data_dict = self.generate_mock_data(symbols)

        # Train/Test 분할
        train_data = {}
        test_data = {}

        for symbol, data in data_dict.items():
            split_idx = int(len(data) * 0.7)
            train_data[symbol] = data.iloc[:split_idx]
            test_data[symbol] = data.iloc[split_idx:]

        # Threshold 조합 생성
        threshold_combinations = []

        # 간단한 그리드 서치
        buy_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        sell_thresholds = [-0.5, -0.4, -0.3, -0.2, -0.1]

        for buy_thresh in buy_thresholds:
            for sell_thresh in sell_thresholds:
                if buy_thresh > sell_thresh:
                    thresholds = {
                        "strong_buy": buy_thresh + 0.2,
                        "buy": buy_thresh,
                        "hold_upper": buy_thresh - 0.1,
                        "hold_lower": sell_thresh + 0.1,
                        "sell": sell_thresh,
                        "strong_sell": sell_thresh - 0.2,
                    }
                    threshold_combinations.append(thresholds)

        # 최적화 실행
        best_score = float("-inf")
        best_thresholds = None
        all_results = []

        for i, thresholds in enumerate(threshold_combinations[:n_trials]):
            logger.info(
                f"조합 {i+1}/{min(n_trials, len(threshold_combinations))} 테스트: {thresholds}"
            )

            # Train 데이터로 백테스팅
            train_scores = []
            for symbol, data in train_data.items():
                performance = self.simulate_trading(data, thresholds)
                train_scores.append(performance["sharpe_ratio"])

            # Test 데이터로 백테스팅
            test_scores = []
            for symbol, data in test_data.items():
                performance = self.simulate_trading(data, thresholds)
                test_scores.append(performance["sharpe_ratio"])

            # 평균 점수 계산
            avg_train_score = np.mean(train_scores) if train_scores else 0.0
            avg_test_score = np.mean(test_scores) if test_scores else 0.0
            avg_score = (avg_train_score + avg_test_score) / 2

            result = {
                "thresholds": thresholds,
                "train_score": avg_train_score,
                "test_score": avg_test_score,
                "avg_score": avg_score,
                "train_scores": train_scores,
                "test_scores": test_scores,
            }

            all_results.append(result)

            # 최고 점수 업데이트
            if avg_score > best_score:
                best_score = avg_score
                best_thresholds = thresholds.copy()
                logger.info(f"  새로운 최고 점수: {best_score:.4f}")

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            "timestamp": timestamp,
            "symbols": symbols,
            "n_trials": n_trials,
            "best_thresholds": best_thresholds,
            "best_score": best_score,
            "all_results": all_results,
        }

        result_file = os.path.join(
            self.results_dir, f"simple_threshold_optimization_{timestamp}.json"
        )
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"결과 저장: {result_file}")

        return summary


def main():
    """메인 함수"""
    # 테스트 실행
    optimizer = SimpleThresholdOptimizer()

    # 간단한 테스트
    symbols = ["AAPL", "META"]
    results = optimizer.optimize_thresholds(symbols, n_trials=5)

    print("\n" + "=" * 60)
    print("🎯 Threshold 최적화 결과")
    print("=" * 60)
    print(f"테스트 종목: {results['symbols']}")
    print(f"시도 횟수: {results['n_trials']}")
    print(f"최고 점수: {results['best_score']:.4f}")
    print(f"최적 threshold: {results['best_thresholds']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
