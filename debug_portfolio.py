#!/usr/bin/env python3
"""
포트폴리오 계산 디버깅 스크립트
"""

import json
import pandas as pd
from pathlib import Path
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.evaluator import TrainTestEvaluator


def load_latest_results():
    """최신 결과 파일 로드"""
    results_dir = Path("results")

    # 최신 individual_evaluation 파일 찾기
    individual_files = list(results_dir.glob("individual_evaluation_*.json"))
    if not individual_files:
        print("❌ individual_evaluation 파일을 찾을 수 없습니다.")
        return None, None

    latest_individual = max(individual_files, key=lambda x: x.stat().st_mtime)

    # 최신 portfolio_weights 파일 찾기
    weights_files = list(results_dir.glob("portfolio_weights_*.json"))
    if not weights_files:
        print("❌ portfolio_weights 파일을 찾을 수 없습니다.")
        return None, None

    latest_weights = max(weights_files, key=lambda x: x.stat().st_mtime)

    print(f"📁 개별 평가 파일: {latest_individual.name}")
    print(f"📁 포트폴리오 비중 파일: {latest_weights.name}")

    # 파일 로드
    with open(latest_individual, "r") as f:
        individual_results = json.load(f)

    with open(latest_weights, "r") as f:
        portfolio_weights = json.load(f)

    return individual_results, portfolio_weights


def test_evaluator_portfolio_calculation(individual_results, portfolio_weights):
    """evaluator의 포트폴리오 계산 함수 테스트"""
    print("\n" + "=" * 60)
    print("🔍 Evaluator 포트폴리오 계산 테스트")
    print("=" * 60)

    # Evaluator 인스턴스 생성
    evaluator = TrainTestEvaluator()

    # Train 데이터 확인
    train_data = individual_results.get("train", {})
    print(f"📊 Train 데이터 종목 수: {len(train_data)}")

    # Evaluator의 포트폴리오 계산 함수 테스트
    try:
        portfolio_metrics = evaluator._calculate_real_portfolio_metrics(
            train_data, portfolio_weights
        )

        print(
            f"📈 포트폴리오 누적 수익률: {portfolio_metrics.get('cumulative_return', 0):.6f} ({portfolio_metrics.get('cumulative_return', 0)*100:.2f}%)"
        )
        print(f"📊 샤프 비율: {portfolio_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"📊 소르티노 비율: {portfolio_metrics.get('sortino_ratio', 0):.4f}")
        print(
            f"📊 최대 낙폭: {portfolio_metrics.get('max_drawdown', 0):.4f} ({portfolio_metrics.get('max_drawdown', 0)*100:.2f}%)"
        )
        print(
            f"📊 변동성: {portfolio_metrics.get('volatility', 0):.4f} ({portfolio_metrics.get('volatility', 0)*100:.2f}%)"
        )
        print(f"📊 총 거래 수: {portfolio_metrics.get('total_trades', 0)}")
        print(f"📊 일별 수익률 개수: {len(portfolio_metrics.get('returns', []))}")

        # 일별 수익률 확인
        returns = portfolio_metrics.get("returns", [])
        if returns:
            returns_series = pd.Series(returns)
            print(f"📈 평균 일별 수익률: {returns_series.mean():.6f}")
            print(f"📊 일별 수익률 표준편차: {returns_series.std():.6f}")

    except Exception as e:
        print(f"❌ Evaluator 포트폴리오 계산 오류: {e}")
        import traceback

        print(f"상세 오류: {traceback.format_exc()}")


def debug_portfolio_calculation(individual_results, portfolio_weights):
    """포트폴리오 계산 디버깅"""
    print("\n" + "=" * 60)
    print("🔍 수동 포트폴리오 계산 디버깅")
    print("=" * 60)

    # Train 데이터 확인
    train_data = individual_results.get("train", {})
    print(f"📊 Train 데이터 종목 수: {len(train_data)}")

    total_trades = 0
    for symbol, data in train_data.items():
        trades = data.get("trades", [])
        total_trades += len(trades)
        weight = portfolio_weights.get(symbol, 0.0)
        print(f"  {symbol}: {len(trades)}개 거래, 비중: {weight:.4f}")

    print(f"📈 총 거래 수: {total_trades}")

    # 포트폴리오 누적 수익률 계산 (가중 평균 방식)
    print("\n📊 가중 평균 방식 포트폴리오 수익률:")
    weighted_return = 0.0
    for symbol, data in train_data.items():
        cumulative_return = data.get("cumulative_return", 0.0)
        weight = portfolio_weights.get(symbol, 0.0)
        weighted_return += cumulative_return * weight
        print(
            f"  {symbol}: {cumulative_return:.4f} × {weight:.4f} = {cumulative_return * weight:.6f}"
        )

    print(
        f"🎯 가중 평균 누적 수익률: {weighted_return:.6f} ({weighted_return*100:.2f}%)"
    )

    # 거래별 포트폴리오 수익률 계산
    print("\n📊 거래별 포트폴리오 수익률:")
    all_trades = []
    for symbol, data in train_data.items():
        weight = portfolio_weights.get(symbol, 0.0)
        trades = data.get("trades", [])

        for trade in trades:
            pnl = trade.get("pnl", 0.0)  # 이미 수익률로 저장됨
            weighted_pnl = pnl * weight
            trade_info = {
                "symbol": symbol,
                "weight": weight,
                "pnl": pnl,
                "weighted_pnl": weighted_pnl,
                "entry_time": trade.get("entry_time", "N/A"),
                "exit_time": trade.get("exit_time", "N/A"),
            }
            all_trades.append(trade_info)

    # 거래를 시간순으로 정렬
    all_trades.sort(key=lambda x: str(x["entry_time"]))

    print(f"📈 총 가중 거래 수: {len(all_trades)}")

    # 포트폴리오 누적 수익률 계산 (복리 효과)
    portfolio_value = 1.0
    daily_returns = []

    for i, trade in enumerate(all_trades):
        weighted_pnl = trade["weighted_pnl"]
        if weighted_pnl != 0:
            daily_return = weighted_pnl / portfolio_value
            daily_returns.append(daily_return)
            portfolio_value *= 1 + daily_return

            if i < 10:  # 처음 10개 거래만 출력
                print(
                    f"  거래 {i+1}: {trade['symbol']} PnL={trade['pnl']:.4f}, 가중PnL={weighted_pnl:.6f}, 일별수익률={daily_return:.6f}, 포트폴리오값={portfolio_value:.6f}"
                )

    final_return = portfolio_value - 1.0
    print(f"\n🎯 복리 효과 누적 수익률: {final_return:.6f} ({final_return*100:.2f}%)")
    print(f"📊 일별 수익률 개수: {len(daily_returns)}")

    if daily_returns:
        returns_series = pd.Series(daily_returns)
        mean_return = returns_series.mean()
        std_return = returns_series.std()
        sharpe_ratio = (
            (mean_return * 252) / (std_return * (252**0.5)) if std_return > 0 else 0
        )

        print(f"📈 평균 일별 수익률: {mean_return:.6f}")
        print(f"📊 일별 수익률 표준편차: {std_return:.6f}")
        print(f"📈 샤프 비율: {sharpe_ratio:.4f}")


def debug_daily_returns_comparison(individual_results, portfolio_weights):
    """일별 수익률 계산 방식 비교"""
    print("\n" + "=" * 60)
    print("🔍 일별 수익률 계산 방식 비교")
    print("=" * 60)

    # 현재 방식 (거래별 수익률)
    print("📊 현재 방식 (거래별 수익률):")
    all_trades = []
    for symbol, data in individual_results.get("train", {}).items():
        weight = portfolio_weights.get(symbol, 0.0)
        trades = data.get("trades", [])

        for trade in trades:
            pnl = trade.get("pnl", 0.0)
            weighted_pnl = pnl * weight
            all_trades.append(
                {
                    "symbol": symbol,
                    "weighted_pnl": weighted_pnl,
                    "exit_time": trade.get("exit_time", 0),
                }
            )

    # exit_time이 None인 경우 처리
    all_trades.sort(key=lambda x: x["exit_time"] if x["exit_time"] is not None else 0)

    current_portfolio_value = 1.0
    trade_returns = []

    for i, trade in enumerate(all_trades):
        weighted_pnl = trade["weighted_pnl"]
        if weighted_pnl != 0:
            daily_return = weighted_pnl / current_portfolio_value
            trade_returns.append(daily_return)
            current_portfolio_value *= 1 + daily_return

            if i < 5:
                print(
                    f"  거래 {i+1}: {trade['symbol']} 수익률={daily_return:.6f}, 포트폴리오값={current_portfolio_value:.6f}"
                )

    if trade_returns:
        returns_series = pd.Series(trade_returns)
        mean_return = returns_series.mean()
        std_return = returns_series.std()
        sharpe_ratio = (
            (mean_return * 252) / (std_return * (252**0.5)) if std_return > 0 else 0
        )

        print(f"📈 거래별 수익률 - 평균: {mean_return:.6f}, 표준편차: {std_return:.6f}")
        print(f"📊 샤프 비율: {sharpe_ratio:.4f}")
        print(f"📊 거래 수: {len(trade_returns)}")

    # 실제 일별 수익률 계산 (간단한 시뮬레이션)
    print("\n📊 실제 일별 수익률 (시뮬레이션):")

    # 365일 동안의 일별 수익률 시뮬레이션
    daily_returns_sim = []
    portfolio_value = 1.0

    # 거래가 발생한 날에만 수익률 적용, 나머지는 0
    trade_days = set()
    for trade in all_trades:
        if (
            isinstance(trade["exit_time"], (int, float))
            and trade["exit_time"] is not None
        ):
            trade_days.add(int(trade["exit_time"]))

    for day in range(365):
        if day in trade_days:
            # 해당 날짜의 거래 찾기
            day_trades = [
                t
                for t in all_trades
                if isinstance(t["exit_time"], (int, float))
                and t["exit_time"] is not None
                and int(t["exit_time"]) == day
            ]
            daily_return = sum(t["weighted_pnl"] for t in day_trades) / portfolio_value
        else:
            daily_return = 0.0

        daily_returns_sim.append(daily_return)
        portfolio_value *= 1 + daily_return

    if daily_returns_sim:
        returns_series_sim = pd.Series(daily_returns_sim)
        mean_return_sim = returns_series_sim.mean()
        std_return_sim = returns_series_sim.std()
        sharpe_ratio_sim = (
            (mean_return_sim * 252) / (std_return_sim * (252**0.5))
            if std_return_sim > 0
            else 0
        )

        print(
            f"📈 일별 수익률 - 평균: {mean_return_sim:.6f}, 표준편차: {std_return_sim:.6f}"
        )
        print(f"📊 샤프 비율: {sharpe_ratio_sim:.4f}")
        print(f"📊 거래일 수: {len([r for r in daily_returns_sim if r != 0])}")
        print(f"📊 총 일수: {len(daily_returns_sim)}")

    print(f"\n🎯 최종 포트폴리오 값: {portfolio_value:.6f} ({portfolio_value-1:.6f})")


def main():
    """메인 함수"""
    print("🚀 포트폴리오 계산 디버깅 시작")

    # 결과 파일 로드
    individual_results, portfolio_weights = load_latest_results()
    if individual_results is None or portfolio_weights is None:
        return

    # 수동 포트폴리오 계산 디버깅
    debug_portfolio_calculation(individual_results, portfolio_weights)

    # 일별 수익률 계산 방식 비교
    debug_daily_returns_comparison(individual_results, portfolio_weights)

    # Evaluator 포트폴리오 계산 테스트
    test_evaluator_portfolio_calculation(individual_results, portfolio_weights)


if __name__ == "__main__":
    main()
