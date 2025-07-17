#!/usr/bin/env python3
"""
실제 데이터로 퀀트 전략 백테스팅
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# 프로젝트 루트를 Python 경로에 추가
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

from .strategies import (
    StrategyManager,
    DualMomentumStrategy,
    VolatilityAdjustedBreakoutStrategy,
    RiskParityLeverageStrategy,
    SwingEMACrossoverStrategy,
    SwingRSIReversalStrategy,
    DonchianSwingBreakoutStrategy,
    BaseStrategy,
)
from .calculate_index import StrategyParams, TechnicalIndicators
from .log_pl import TradingSimulator


class BacktestEngine:
    """백테스팅 엔진"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}

    def run_backtest(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        params: StrategyParams,
    ) -> Dict[str, Any]:
        """전략 백테스팅 실행"""

        # 기술적 지표 계산
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, params)

        # 신호 생성
        signals = strategy.generate_signals(df_with_indicators)

        # 백테스팅 실행
        portfolio_values = []
        positions = []
        trades = []

        current_position = 0
        current_capital = self.initial_capital
        entry_price = 0
        entry_time = None

        for i, row in signals.iterrows():
            if i == 0:  # 첫 번째 행은 건너뛰기
                portfolio_values.append(current_capital)
                positions.append(current_position)
                continue

            current_price = row["close"]
            signal = row["signal"]

            # 포지션 진입
            if current_position == 0 and signal != 0:
                current_position = signal
                entry_price = current_price
                entry_time = row["datetime"]

                # 포지션 사이즈 계산
                position_size = strategy.calculate_position_size(
                    df_with_indicators.loc[:i], signal
                )

                # 손절/익절 계산
                stop_loss = strategy.calculate_stop_loss(
                    df_with_indicators.loc[:i], entry_price, signal
                )
                take_profit = strategy.calculate_take_profit(
                    df_with_indicators.loc[:i], entry_price, signal
                )

                trades.append(
                    {
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "position": signal,
                        "position_size": position_size,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }
                )

            # 포지션 청산 조건 확인
            elif current_position != 0:
                should_exit = False
                exit_reason = ""

                # 손절/익절 확인
                if current_position > 0:  # 롱 포지션
                    if stop_loss and current_price <= stop_loss:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif take_profit and current_price >= take_profit:
                        should_exit = True
                        exit_reason = "take_profit"
                else:  # 숏 포지션
                    if stop_loss and current_price >= stop_loss:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif take_profit and current_price <= take_profit:
                        should_exit = True
                        exit_reason = "take_profit"

                # 신호 반전
                if signal != 0 and signal != current_position:
                    should_exit = True
                    exit_reason = "signal_reversal"

                # 포지션 청산
                if should_exit:
                    # 수익률 계산
                    if current_position > 0:
                        pnl = (current_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - current_price) / entry_price

                    # 자본 업데이트
                    current_capital *= 1 + pnl

                    # 거래 기록 업데이트
                    trades[-1].update(
                        {
                            "exit_time": row["datetime"],
                            "exit_price": current_price,
                            "pnl": pnl,
                            "exit_reason": exit_reason,
                        }
                    )

                    current_position = 0
                    entry_price = 0
                    entry_time = None

            # 포트폴리오 가치 계산
            if current_position != 0:
                # 현재 포지션의 미실현 손익
                if current_position > 0:
                    unrealized_pnl = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price

                portfolio_value = current_capital * (1 + unrealized_pnl)
            else:
                portfolio_value = current_capital

            portfolio_values.append(portfolio_value)
            positions.append(current_position)

        # 성과 지표 계산
        portfolio_series = pd.Series(portfolio_values, index=signals["datetime"])
        returns = portfolio_series.pct_change().dropna()

        # 기본 성과 지표
        total_return = (
            portfolio_series.iloc[-1] - self.initial_capital
        ) / self.initial_capital
        annual_return = total_return * (252 / len(returns))
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(portfolio_series)

        # 거래 통계
        if trades:
            winning_trades = [t for t in trades if "pnl" in t and t["pnl"] > 0]
            losing_trades = [t for t in trades if "pnl" in t and t["pnl"] <= 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = (
                np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
            )
            avg_loss = (
                np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
            )
            profit_factor = (
                abs(
                    sum([t["pnl"] for t in winning_trades])
                    / sum([t["pnl"] for t in losing_trades])
                )
                if losing_trades and sum([t["pnl"] for t in losing_trades]) != 0
                else float("inf")
            )
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        return {
            "strategy_name": strategy_name,
            "portfolio_values": portfolio_series,
            "positions": positions,
            "trades": trades,
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "num_trades": len(trades),
            "params": params,
        }

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """최대 낙폭 계산"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """백테스팅 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{results["strategy_name"]} 백테스팅 결과', fontsize=16)

        # 1. 포트폴리오 가치 변화
        axes[0, 0].plot(
            results["portfolio_values"].index, results["portfolio_values"].values
        )
        axes[0, 0].set_title("포트폴리오 가치 변화")
        axes[0, 0].set_ylabel("포트폴리오 가치")
        axes[0, 0].grid(True)

        # 2. 수익률 분포
        returns = results["portfolio_values"].pct_change().dropna()
        axes[0, 1].hist(returns, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("수익률 분포")
        axes[0, 1].set_xlabel("수익률")
        axes[0, 1].set_ylabel("빈도")
        axes[0, 1].grid(True)

        # 3. 포지션 변화
        positions = pd.Series(
            results["positions"], index=results["portfolio_values"].index
        )
        axes[1, 0].plot(positions.index, positions.values, alpha=0.7)
        axes[1, 0].set_title("포지션 변화")
        axes[1, 0].set_ylabel("포지션")
        axes[1, 0].grid(True)

        # 4. 성과 지표 요약
        metrics = [
            f'총 수익률: {results["total_return"]:.2%}',
            f'연간 수익률: {results["annual_return"]:.2%}',
            f'변동성: {results["volatility"]:.2%}',
            f'샤프 비율: {results["sharpe_ratio"]:.2f}',
            f'최대 낙폭: {results["max_drawdown"]:.2%}',
            f'승률: {results["win_rate"]:.2%}',
            f'거래 횟수: {results["num_trades"]}',
        ]

        axes[1, 1].text(
            0.1,
            0.9,
            "\n".join(metrics),
            transform=axes[1, 1].transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 1].set_title("성과 지표")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"차트가 {save_path}에 저장되었습니다.")

        plt.show()


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """CSV 파일에서 데이터 로드"""
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def main():
    """메인 백테스팅 함수"""
    print("퀀트 전략 백테스팅 시작\n")

    # input_for_backtesting.json 읽기
    input_path = os.path.join(
        os.path.dirname(__file__), "../../input_for_backtesting.json"
    )
    with open(input_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    symbols = config.get("symbols", [])
    strategies_to_run = config.get(
        "strategies",
        [
            "dual_momentum",
            "volatility_breakout",
            "swing_ema",
            "swing_rsi",
            "swing_donchian",
        ],
    )
    param_ranges = config.get("param_ranges", {})
    initial_capital = config.get("initial_capital", 100000)

    # 백테스팅 엔진 초기화
    backtest_engine = BacktestEngine(initial_capital=initial_capital)

    # 데이터 디렉토리 확인
    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    if not os.path.exists(data_dir):
        print(f"데이터 디렉토리 {data_dir}가 존재하지 않습니다.")
        return

    # CSV 파일들 찾기
    csv_files = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and any(f.startswith(sym) for sym in symbols)
    ]
    if not csv_files:
        print("CSV 파일을 찾을 수 없습니다.")
        return

    print(f"발견된 데이터 파일들: {csv_files}")

    # 전략 매니저 초기화
    manager = StrategyManager()

    # 기본 파라미터
    params = StrategyParams()

    # 전략들 등록
    manager.add_strategy("dual_momentum", DualMomentumStrategy(params))
    manager.add_strategy(
        "volatility_breakout", VolatilityAdjustedBreakoutStrategy(params)
    )
    manager.add_strategy("swing_ema", SwingEMACrossoverStrategy(params))
    manager.add_strategy("swing_rsi", SwingRSIReversalStrategy(params))
    manager.add_strategy("swing_donchian", DonchianSwingBreakoutStrategy(params))

    # 각 파일에 대해 백테스팅 실행
    all_results = {}

    for csv_file in csv_files:
        symbol = csv_file.split("_")[0]  # 파일명에서 심볼 추출
        file_path = os.path.join(data_dir, csv_file)

        print(f"\n=== {symbol} 백테스팅 ===")

        try:
            # 데이터 로드
            df = load_data_from_csv(file_path)
            print(f"데이터 포인트 수: {len(df)}")
            print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

            # 각 전략에 대해 백테스팅 실행
            for strategy_name in strategies_to_run:
                print(f"\n{strategy_name} 전략 실행 중...")

                strategy = manager.strategies[strategy_name]

                # 파라미터 최적화가 필요한 경우
                if param_ranges:
                    best_params = manager.optimize_parameters(
                        strategy_name, df, param_ranges
                    )
                    print(f"최적 파라미터: {best_params}")
                    result = backtest_engine.run_backtest(
                        strategy_name, df, strategy, best_params
                    )
                else:
                    result = backtest_engine.run_backtest(
                        strategy_name, df, strategy, params
                    )

                # 결과 출력
                print(f"총 수익률: {result['total_return']:.2%}")
                print(f"샤프 비율: {result['sharpe_ratio']:.2f}")
                print(f"최대 낙폭: {result['max_drawdown']:.2%}")
                print(f"거래 횟수: {result['num_trades']}")
                print(f"승률: {result['win_rate']:.2%}")

                # 실제 매매 시뮬레이션 실행
                print(f"\n=== {symbol} {strategy_name} 실제 매매 시뮬레이션 ===")
                simulator = TradingSimulator()  # 새로운 설정 파일 사용
                simulation_result = simulator.simulate_trading(
                    df, result["signals"], f"{symbol}_{strategy_name}"
                )
                simulator.print_logs(simulation_result["log_lines"])

                # 결과 저장
                key = f"{symbol}_{strategy_name}"
                all_results[key] = result
                all_results[f"{key}_simulation"] = simulation_result

                # 차트 저장
                chart_path = os.path.join(
                    os.path.dirname(__file__), f"../../log/backtest_results_{key}.png"
                )
                backtest_engine.plot_results(result, chart_path)

        except Exception as e:
            print(f"{symbol} 백테스팅 중 오류: {e}")
            continue

    # 전체 결과 요약
    print("\n=== 전체 백테스팅 결과 요약 ===")
    summary_data = []

    for key, result in all_results.items():
        summary_data.append(
            {
                "Symbol_Strategy": key,
                "Total_Return": result["total_return"],
                "Sharpe_Ratio": result["sharpe_ratio"],
                "Max_Drawdown": result["max_drawdown"],
                "Win_Rate": result["win_rate"],
                "Num_Trades": result["num_trades"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # 결과를 JSON으로 저장
    results_summary = {}
    for key, result in all_results.items():
        results_summary[key] = {
            "total_return": result["total_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "win_rate": result["win_rate"],
            "num_trades": result["num_trades"],
            "annual_return": result["annual_return"],
            "volatility": result["volatility"],
        }

    with open(
        os.path.join(os.path.dirname(__file__), "../../log/backtest_results.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n백테스팅 결과가 log/backtest_results.json에 저장되었습니다.")


if __name__ == "__main__":
    main()
