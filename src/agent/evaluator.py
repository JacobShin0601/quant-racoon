#!/usr/bin/env python3
"""
전략 평가 및 비교 분석 시스템
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavior.strategies import (
    StrategyManager,
    DualMomentumStrategy,
    VolatilityAdjustedBreakoutStrategy,
    RiskParityLeverageStrategy,
    SwingEMACrossoverStrategy,
    SwingRSIReversalStrategy,
    DonchianSwingBreakoutStrategy,
    StochasticStrategy,
    WilliamsRStrategy,
    CCIStrategy,
    # 휩쏘 방지 전략들 추가
    WhipsawPreventionStrategy,
    DonchianRSIWhipsawStrategy,
    VolatilityFilteredBreakoutStrategy,
    MultiTimeframeWhipsawStrategy,
    AdaptiveWhipsawStrategy,
    # 결합 전략들 추가
    CCIBollingerStrategy,
    StochDonchianStrategy,
    # 스켈핑 전략들 추가
    VWAPMACDScalpingStrategy,
    KeltnerRSIScalpingStrategy,
    AbsorptionScalpingStrategy,
    RSIBollingerScalpingStrategy,
    MeanReversionStrategy,
)
from behavior.calculate_index import StrategyParams
from behavior.log_pl import TradingSimulator
from behavior.portfolio_weight import PortfolioWeightCalculator
from .portfolio_manager import AdvancedPortfolioManager
from .helper import (
    StrategyResult,
    PortfolioWeights,
    Logger,
    load_config,
    load_and_preprocess_data,
    print_section_header,
    print_subsection_header,
    format_percentage,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
)


class StrategyEvaluator:
    """전략 평가 및 비교 분석 클래스"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        log_mode: str = "summary",
        portfolio_mode: bool = False,
        config_path: str = DEFAULT_CONFIG_PATH,
        portfolio_weights: PortfolioWeights = None,
        portfolio_method: str = "fixed",  # "fixed", "strategy_weights", "signal_combined"
        analysis_results_path: str = None,  # 정량 분석 결과 파일 경로
    ):
        self.data_dir = data_dir
        self.log_mode = log_mode
        self.portfolio_mode = portfolio_mode
        self.config = load_config(config_path)
        self.strategy_manager = StrategyManager()
        self.params = StrategyParams()
        self.simulator = TradingSimulator(config_path)
        self.weight_calculator = PortfolioWeightCalculator(config_path)
        self.portfolio_manager = AdvancedPortfolioManager(config_path)
        self.portfolio_weights = portfolio_weights
        self.portfolio_method = portfolio_method
        self.analysis_results_path = analysis_results_path
        self.results = {}
        self.logger = Logger()
        self.evaluation_start_time = datetime.now()

        # 전략 등록
        self._register_strategies()

    def _calculate_strategy_based_weights(
        self, strategy_name: str, data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """전략별 포트폴리오 비중 계산 (옵션 A)"""
        self.logger.log_info(f"📋 {strategy_name} 전략 기반 포트폴리오 비중 계산")

        # 전략별 최적화 방법 매핑
        strategy_optimization_map = {
            "dual_momentum": "sharpe_maximization",
            "volatility_breakout": "minimum_variance",
            "swing_ema": "risk_parity",
            "swing_rsi": "sortino_maximization",
            "swing_donchian": "maximum_diversification",
            # 데이터 사이언스 전략들
            "trend_following_ds": "maximum_diversification",  # 추세 추종은 분산 극대화
            "predictive_ds": "sharpe_maximization",  # 예측 기반은 샤프 비율 최대화
            "bayesian": "risk_parity",  # 베이지안은 리스크 패리티
            "ensemble_ds": "sortino_maximization",  # 앙상블은 소르티노 비율 최대화
        }

        # 전략에 맞는 최적화 방법 선택
        optimization_method = strategy_optimization_map.get(
            strategy_name, "sharpe_maximization"
        )

        # PortfolioWeightCalculator의 method를 임시로 변경
        original_method = self.weight_calculator.method
        self.weight_calculator.method = optimization_method

        try:
            weights_df = self.weight_calculator.calculate_optimal_weights(data_dict)
        finally:
            # 원래 method로 복원
            self.weight_calculator.method = original_method

        return weights_df

    def _combine_signals_with_weights(
        self,
        strategy_name: str,
        data_dict: Dict[str, pd.DataFrame],
        base_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """전략 신호와 포트폴리오 비중 결합 (옵션 B)"""
        self.logger.log_info(f"📋 {strategy_name} 신호와 포트폴리오 비중 결합")

        # 기본 비중 복사
        combined_weights = base_weights.copy()

        # 기본 비중 정보 로깅
        avg_base_weights = base_weights.mean()
        self.logger.log_info(f"📊 기본 포트폴리오 비중:")
        for symbol, weight in avg_base_weights.items():
            if symbol != "cash" and weight > 0.01:
                self.logger.log_info(f"  {symbol}: {weight*100:.1f}%")

        # 각 종목별로 전략 신호 생성 및 비중 조정
        signal_adjustments = {}
        for symbol in data_dict.keys():
            if symbol in data_dict:
                data = data_dict[symbol]
                strategy = self.strategy_manager.strategies[strategy_name]
                signals = strategy.generate_signals(data)

                # 신호 기반 비중 조정
                adjustment_factor = self._calculate_signal_adjustment(signals, symbol)
                signal_adjustments[symbol] = adjustment_factor

                self.logger.log_info(
                    f"📈 {symbol} 신호 조정 팩터: {adjustment_factor:.3f}"
                )

        # 조정된 비중 계산
        for symbol in combined_weights.columns:
            if symbol != "cash" and symbol in signal_adjustments:
                adjustment = signal_adjustments[symbol]
                combined_weights[symbol] = combined_weights[symbol] * adjustment

        # 비중 정규화 (합계가 1이 되도록)
        combined_weights = self._normalize_weights(combined_weights)

        # 조정 후 비중 정보 로깅
        avg_combined_weights = combined_weights.mean()
        self.logger.log_info(f"📊 신호 조정 후 포트폴리오 비중:")
        for symbol, weight in avg_combined_weights.items():
            if symbol != "cash" and weight > 0.01:
                self.logger.log_info(f"  {symbol}: {weight*100:.1f}%")

        return combined_weights

    def _calculate_signal_adjustment(self, signals: pd.DataFrame, symbol: str) -> float:
        """신호에 따른 비중 조정 팩터 계산"""
        try:
            # 신호 컬럼 확인
            if "signal" not in signals.columns:
                self.logger.log_warning(
                    f"⚠️ {symbol}: 'signal' 컬럼이 없습니다. 기본값 1.0 사용"
                )
                return 1.0

            # 최근 10개 신호의 평균 계산
            recent_signals = signals["signal"].tail(10)
            if len(recent_signals) == 0:
                self.logger.log_warning(
                    f"⚠️ {symbol}: 신호 데이터가 없습니다. 기본값 1.0 사용"
                )
                return 1.0

            avg_signal = recent_signals.mean()

            # 신호 강도에 따른 조정 팩터 계산
            # 신호 범위: -1 (강한 매도) ~ 1 (강한 매수)
            if avg_signal > 0.3:  # 매수 신호
                adjustment = 1.0 + (avg_signal - 0.3) * 0.5  # 최대 1.35배
            elif avg_signal < -0.3:  # 매도 신호
                adjustment = 1.0 + (avg_signal + 0.3) * 0.5  # 최소 0.65배
            else:  # 중립 신호
                adjustment = 1.0

            # 조정 팩터 범위 제한 (0.5 ~ 1.5)
            adjustment = max(0.5, min(1.5, adjustment))

            self.logger.log_info(f"  📊 {symbol} 신호 분석:")
            self.logger.log_info(f"    평균 신호: {avg_signal:.3f}")
            self.logger.log_info(f"    조정 팩터: {adjustment:.3f}")

            return adjustment

        except Exception as e:
            self.logger.log_error(f"❌ {symbol} 신호 조정 계산 중 오류: {str(e)}")
            return 1.0

    def _normalize_weights(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """비중 정규화 (합계가 1이 되도록)"""
        # 현금을 제외한 비중 합계
        non_cash_weights = weights_df.drop(columns=["cash"], errors="ignore")
        total_weight = non_cash_weights.sum(axis=1)

        # 정규화
        for col in non_cash_weights.columns:
            weights_df[col] = weights_df[col] / total_weight

        # 현금 비중 조정
        if "cash" in weights_df.columns:
            weights_df["cash"] = 1 - non_cash_weights.sum(axis=1)

        return weights_df

    def _register_strategies(self):
        """모든 전략을 매니저에 등록"""
        strategies = {
            "dual_momentum": DualMomentumStrategy(self.params),
            "volatility_breakout": VolatilityAdjustedBreakoutStrategy(self.params),
            "swing_ema": SwingEMACrossoverStrategy(self.params),
            "swing_rsi": SwingRSIReversalStrategy(self.params),
            "swing_donchian": DonchianSwingBreakoutStrategy(self.params),
            # 신규 전략 등록
            "stochastic": StochasticStrategy(self.params),
            "williams_r": WilliamsRStrategy(self.params),
            "cci": CCIStrategy(self.params, threshold=80),
            # 휩쏘 방지 전략들 등록
            "whipsaw_prevention": WhipsawPreventionStrategy(self.params),
            "donchian_rsi_whipsaw": DonchianRSIWhipsawStrategy(self.params),
            "volatility_filtered_breakout": VolatilityFilteredBreakoutStrategy(
                self.params
            ),
            "multi_timeframe_whipsaw": MultiTimeframeWhipsawStrategy(self.params),
            "adaptive_whipsaw": AdaptiveWhipsawStrategy(self.params),
            # 새로운 결합 전략들 등록
            "cci_bollinger": CCIBollingerStrategy(self.params),
            "stoch_donchian": StochDonchianStrategy(self.params),
            # 스켈핑 전략들 등록
            "vwap_macd_scalping": VWAPMACDScalpingStrategy(self.params),
            "keltner_rsi_scalping": KeltnerRSIScalpingStrategy(self.params),
            "absorption_scalping": AbsorptionScalpingStrategy(self.params),
            "rsi_bollinger_scalping": RSIBollingerScalpingStrategy(self.params),
            # 평균회귀 전략 등록
            "mean_reversion": MeanReversionStrategy(self.params),
        }
        for name, strategy in strategies.items():
            self.strategy_manager.add_strategy(name, strategy)

    def load_data(self, symbol: str = None) -> Dict[str, pd.DataFrame]:
        """데이터 로드"""
        # config에서 심볼 목록 가져오기
        config_symbols = self.config.get("data", {}).get("symbols", [])

        return load_and_preprocess_data(self.data_dir, config_symbols, symbol)

    def evaluate_strategy(
        self, strategy_name: str, data_dict: Dict[str, pd.DataFrame]
    ) -> StrategyResult:
        """단일 전략 평가"""
        # 로거 설정 (간소화된 로그 파일명)
        symbols = list(data_dict.keys())
        self.logger.setup_logger(strategy=strategy_name, symbols=symbols, mode="eval")

        self.logger.log_section(f"🔍 {strategy_name} 전략 평가 중...")

        # 데이터 분석 정보 로깅
        first_symbol = list(data_dict.keys())[0]
        data = data_dict[first_symbol]
        start_date = data["datetime"].min()
        end_date = data["datetime"].max()
        total_days = (end_date - start_date).days
        total_points = len(data)

        self.logger.log_info(
            f"📅 분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({total_days}일)"
        )
        self.logger.log_info(f"📊 분석 종목: {', '.join(symbols)} ({len(symbols)}개)")
        self.logger.log_info(f"📈 데이터 포인트: {total_points:,}개")
        self.logger.log_info(
            f"💰 초기 자본: ${self.config.get('trading', {}).get('initial_capital', 100000):,}"
        )

        try:
            if self.portfolio_mode:
                # 포트폴리오 모드 - 멀티-에셋 비중 기반 시뮬레이션
                self.logger.log_info(
                    f"📊 {strategy_name} 포트폴리오 비중 기반 시뮬레이션 ({self.portfolio_method})"
                )

                # 포트폴리오 비중 계산 방법에 따른 분기
                if self.portfolio_method == "fixed":
                    # 기존 방식: 고정된 비중 사용
                    if self.portfolio_weights is not None:
                        self.logger.log_info(f"📋 미리 계산된 포트폴리오 비중 사용")
                        weights_df = self.portfolio_weights.weights
                    else:
                        self.logger.log_info(f"📋 기본 포트폴리오 비중 계산")
                        weights_df = self.weight_calculator.calculate_optimal_weights(
                            data_dict
                        )
                        # 비중 요약 출력
                        self.weight_calculator.print_weight_summary(weights_df)

                elif self.portfolio_method == "strategy_weights":
                    # 옵션 A: 전략별 다른 최적화 방법으로 비중 계산
                    weights_df = self._calculate_strategy_based_weights(
                        strategy_name, data_dict
                    )

                elif self.portfolio_method == "signal_combined":
                    # 옵션 B: 기본 비중 + 전략 신호 결합
                    if self.portfolio_weights is not None:
                        base_weights = self.portfolio_weights.weights
                    else:
                        base_weights = self.weight_calculator.calculate_optimal_weights(
                            data_dict
                        )

                    weights_df = self._combine_signals_with_weights(
                        strategy_name, data_dict, base_weights
                    )

                else:
                    # 기본값: 고정된 비중 사용
                    self.logger.log_warning(
                        f"알 수 없는 포트폴리오 방법: {self.portfolio_method}, 기본 방식 사용"
                    )
                    weights_df = self.weight_calculator.calculate_optimal_weights(
                        data_dict
                    )

                # 포트폴리오 비중 정보 로깅
                avg_weights = weights_df.mean()
                self.logger.log_info(f"📊 평균 포트폴리오 비중:")
                for symbol, weight in avg_weights.items():
                    if symbol != "cash" and weight > 0.01:  # 1% 이상인 종목만
                        self.logger.log_info(f"  {symbol}: {weight*100:.1f}%")

                # 전략별 신호 생성 (포트폴리오 모드에서도 전략별 차이를 위해)
                strategy_signals = {}
                for symbol, data in data_dict.items():
                    strategy = self.strategy_manager.strategies[strategy_name]
                    signals = strategy.generate_signals(data)
                    strategy_signals[symbol] = signals

                # 실시간 로그 모드인 경우
                if self.log_mode == "real_time":
                    print(f"\n📊 {strategy_name} 실시간 포트폴리오 시뮬레이션")
                    print("-" * 50)

                    # 포트폴리오 시뮬레이션 실행 (전략별 신호 포함)
                    simulation_result = self.simulator.simulate_portfolio_trading(
                        data_dict, weights_df, strategy_name, strategy_signals
                    )

                    # 실시간 로그 출력
                    self.simulator.print_logs(simulation_result["log_lines"])

                    results = simulation_result["results"]
                    trades = simulation_result["trades"]
                    portfolio_values = simulation_result["portfolio_values"]

                else:
                    # 요약 모드 - 포트폴리오 성과 지표 계산 (전략별 신호 포함)
                    simulation_result = self.simulator.simulate_portfolio_trading(
                        data_dict, weights_df, strategy_name, strategy_signals
                    )
                    results = simulation_result["results"]
                    trades = simulation_result["trades"]
                    portfolio_values = simulation_result["portfolio_values"]

            else:
                # 단일 종목 모드 (기존 방식)
                # 첫 번째 종목 사용
                first_symbol = list(data_dict.keys())[0]
                data = data_dict[first_symbol]

                # 전략 실행
                strategy = self.strategy_manager.strategies[strategy_name]
                signals = strategy.generate_signals(data)

                # 실시간 로그 모드인 경우
                if self.log_mode == "real_time":
                    print(f"\n📊 {strategy_name} 실시간 매매 시뮬레이션")
                    print("-" * 50)

                    # 시뮬레이션 실행
                    simulation_result = self.simulator.simulate_trading(
                        data, signals, strategy_name
                    )

                    # 실시간 로그 출력
                    self.simulator.print_logs(simulation_result["log_lines"])

                    results = simulation_result["results"]
                    trades = simulation_result["trades"]
                    portfolio_values = simulation_result["portfolio_values"]

                    # total_trades 키 추가
                    results["total_trades"] = len(trades)

                else:
                    # 요약 모드 - 실제 거래 시뮬레이션 실행하여 정확한 승률 계산
                    simulation_result = self.simulator.simulate_trading(
                        data, signals, strategy_name
                    )
                    results = simulation_result["results"]
                    trades = simulation_result["trades"]
                    portfolio_values = simulation_result["portfolio_values"]

                    # total_trades 키 추가
                    results["total_trades"] = len(trades)

            # 성과 지표 로깅 (간소화)
            self.logger.log_success(f"✅ {strategy_name} 전략 평가 완료")
            self.logger.log_info(
                f"📈 총 수익률: {results['total_return']*100:.2f}% | "
                f"📊 샤프 비율: {results['sharpe_ratio']:.2f} | "
                f"📉 최대 낙폭: {results['max_drawdown']*100:.2f}% | "
                f"🔄 거래 횟수: {results['total_trades']}회"
            )

            # 종합 요약용 결과 저장
            self.logger.add_evaluation_result(strategy_name, results)

            # 거래 통계 로깅
            if trades:
                profitable_trades = [t for t in trades if t["pnl"] > 0]
                losing_trades = [t for t in trades if t["pnl"] < 0]
                avg_profit = (
                    np.mean([t["pnl"] for t in profitable_trades])
                    if profitable_trades
                    else 0
                )
                avg_loss = (
                    np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
                )

                self.logger.log_info(f"📊 거래 통계:")
                self.logger.log_info(
                    f"  수익 거래: {len(profitable_trades)}회 (평균 ${avg_profit:.2f})"
                )
                self.logger.log_info(
                    f"  손실 거래: {len(losing_trades)}회 (평균 ${avg_loss:.2f})"
                )
                self.logger.log_info(
                    f"  최대 수익: ${max([t['pnl'] for t in trades]):.2f}"
                )
                self.logger.log_info(
                    f"  최대 손실: ${min([t['pnl'] for t in trades]):.2f}"
                )

            # StrategyResult 객체 생성
            if self.portfolio_mode:
                # 포트폴리오 모드에서는 전략별 신호와 weights_df 모두 저장
                strategy_result = StrategyResult(
                    name=strategy_name,
                    total_return=results["total_return"],
                    sharpe_ratio=results["sharpe_ratio"],
                    max_drawdown=results["max_drawdown"],
                    win_rate=results["win_rate"],
                    profit_factor=results["profit_factor"],
                    sqn=results["sqn"],
                    total_trades=results["total_trades"],
                    avg_hold_duration=results["avg_hold_duration"],
                    trades=trades,
                    portfolio_values=portfolio_values,
                    signals=strategy_signals,  # 전략별 신호 저장
                )
            else:
                strategy_result = StrategyResult(
                    name=strategy_name,
                    total_return=results["total_return"],
                    sharpe_ratio=results["sharpe_ratio"],
                    max_drawdown=results["max_drawdown"],
                    win_rate=results["win_rate"],
                    profit_factor=results["profit_factor"],
                    sqn=results["sqn"],
                    total_trades=results["total_trades"],
                    avg_hold_duration=results["avg_hold_duration"],
                    trades=trades,
                    portfolio_values=portfolio_values,
                    signals=signals,
                )

            return strategy_result

        except Exception as e:
            print(f"❌ {strategy_name} 전략 평가 중 오류: {e}")
            # 기본 결과 객체 반환 (예외 발생 시)
            default_results = {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sqn": 0.0,
                "total_trades": 0,
                "avg_hold_duration": 0.0,
            }

            strategy_result = StrategyResult(
                name=strategy_name,
                total_return=default_results["total_return"],
                sharpe_ratio=default_results["sharpe_ratio"],
                max_drawdown=default_results["max_drawdown"],
                win_rate=default_results["win_rate"],
                profit_factor=default_results["profit_factor"],
                sqn=default_results["sqn"],
                total_trades=default_results["total_trades"],
                avg_hold_duration=default_results["avg_hold_duration"],
                trades=[],
                portfolio_values=pd.DataFrame(),
                signals=pd.DataFrame(),
            )

            return strategy_result

    def evaluate_strategy_with_params(
        self,
        strategy_name: str,
        data_dict: Dict[str, pd.DataFrame],
        optimized_params: Dict[str, Any],
    ) -> "StrategyResult":
        """최적화된 파라미터로 전략 평가"""
        self.logger.log_section(f"🔍 {strategy_name} 최적화된 파라미터로 평가")
        self.logger.log_info(f"최적화된 파라미터: {optimized_params}")

        try:
            # StrategyParams 객체 생성 (최적화된 파라미터로)
            strategy_params = StrategyParams(**optimized_params)

            # 전략 인스턴스 생성
            strategy_class = self.strategy_manager.strategies[strategy_name].__class__
            strategy = strategy_class(strategy_params)

            # 기존 전략 임시 저장
            original_strategy = self.strategy_manager.strategies[strategy_name]

            # 새로운 전략으로 교체
            self.strategy_manager.strategies[strategy_name] = strategy

            try:
                # 전략 평가 실행
                result = self.evaluate_strategy(strategy_name, data_dict)
                return result
            finally:
                # 원래 전략으로 복원
                self.strategy_manager.strategies[strategy_name] = original_strategy

        except Exception as e:
            self.logger.log_error(f"최적화된 파라미터로 평가 중 오류: {str(e)}")
            return None

    def _calculate_basic_metrics(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> Dict[str, float]:
        """기본 성과 지표 계산"""
        returns = data["close"].pct_change()
        strategy_returns = signals["signal"].shift(1) * returns

        total_return = strategy_returns.sum()
        sharpe_ratio = (
            (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
            if strategy_returns.std() > 0
            else 0
        )

        # 최대 낙폭 계산
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 실제 거래 기반 승률 계산
        win_rate = self._calculate_actual_win_rate(data, signals)

        # 수익 팩터 계산
        profit_factor = self._calculate_profit_factor(strategy_returns)

        # SQN (System Quality Number) 계산
        sqn = self._calculate_sqn(strategy_returns)

        # 거래 횟수 계산
        signal_changes = signals["signal"].diff()
        buy_signals = len(signals[signal_changes == 1])
        sell_signals = len(signals[signal_changes == -1])
        total_trades = min(buy_signals, sell_signals)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sqn": sqn,
            "total_trades": total_trades,
            "avg_hold_duration": 0.0,  # 기본값
        }

    def _calculate_actual_win_rate(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> float:
        """실제 거래 결과 기반 승률 계산"""
        try:
            # 신호 변화점 찾기
            signal_changes = signals["signal"].diff()
            entry_points = signals[signal_changes != 0].index

            if len(entry_points) < 2:
                return 0.0

            wins = 0
            total_trades = 0

            for i in range(len(entry_points) - 1):
                entry_idx = entry_points[i]
                exit_idx = entry_points[i + 1]

                if entry_idx >= len(data) or exit_idx >= len(data):
                    continue

                entry_price = data.loc[entry_idx, "close"]
                exit_price = data.loc[exit_idx, "close"]
                position = signals.loc[entry_idx, "signal"]

                # 수익/손실 계산
                if position == 1:  # 롱 포지션
                    pnl = (exit_price - entry_price) / entry_price
                elif position == -1:  # 숏 포지션
                    pnl = (entry_price - exit_price) / entry_price
                else:
                    continue

                if pnl > 0:
                    wins += 1
                total_trades += 1

            return wins / total_trades if total_trades > 0 else 0.0

        except Exception as e:
            self.logger.log_warning(f"승률 계산 중 오류: {e}")
            return 0.0

    def _calculate_profit_factor(self, strategy_returns: pd.Series) -> float:
        """수익 팩터 계산"""
        try:
            positive_returns = strategy_returns[strategy_returns > 0]
            negative_returns = strategy_returns[strategy_returns < 0]

            gross_profit = positive_returns.sum() if len(positive_returns) > 0 else 0
            gross_loss = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0

            return gross_profit / gross_loss if gross_loss > 0 else 1.0

        except Exception as e:
            self.logger.log_warning(f"수익 팩터 계산 중 오류: {e}")
            return 1.0

    def _calculate_sqn(self, strategy_returns: pd.Series) -> float:
        """SQN (System Quality Number) 계산"""
        try:
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return 0.0

            # 연간화된 수익률과 표준편차
            annual_return = strategy_returns.mean() * 252
            annual_std = strategy_returns.std() * np.sqrt(252)

            return annual_return / annual_std if annual_std > 0 else 0.0

        except Exception as e:
            self.logger.log_warning(f"SQN 계산 중 오류: {e}")
            return 0.0

    def compare_strategies(
        self, data_dict: Dict[str, pd.DataFrame], strategies: List[str] = None
    ) -> Dict[str, StrategyResult]:
        """여러 전략 비교 분석"""
        if strategies is None:
            strategies = list(self.strategy_manager.strategies.keys())

        # 로거 설정 (간소화)
        symbols = list(data_dict.keys())
        self.logger.setup_logger(strategy="comparison", symbols=symbols, mode="comp")

        self.logger.log_section("🚀 전략 비교 분석 시작")

        first_symbol = list(data_dict.keys())[0]
        data = data_dict[first_symbol]
        start_date = data["datetime"].min()
        end_date = data["datetime"].max()

        self.logger.log_info(
            f"📅 분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        )
        self.logger.log_info(f"📊 분석 종목: {', '.join(symbols)} ({len(symbols)}개)")
        self.logger.log_info(
            f"🎯 분석 전략: {', '.join(strategies)} ({len(strategies)}개)"
        )
        self.logger.log_info(
            f"📈 분석 모드: {'포트폴리오' if self.portfolio_mode else '단일 종목'}"
        )

        results = {}
        completed_count = 0

        for strategy_name in strategies:
            self.logger.log_info(
                f"🔄 {strategy_name} 전략 평가 중... ({completed_count + 1}/{len(strategies)})"
            )
            result = self.evaluate_strategy(strategy_name, data_dict)
            if result:
                results[strategy_name] = result
                completed_count += 1
                self.logger.log_success(f"✅ {strategy_name} 전략 평가 완료")
            else:
                self.logger.log_error(f"❌ {strategy_name} 전략 평가 실패")

        self.logger.log_success(
            f"🎉 전략 비교 분석 완료! ({completed_count}/{len(strategies)} 전략 성공)"
        )

        return results

    def generate_comparison_report(self, results: Dict[str, StrategyResult]) -> str:
        """전략 비교 리포트 생성"""
        if not results:
            return "평가 결과가 없습니다."

        report_lines = []
        report_lines.append("\n" + "=" * 80)
        report_lines.append("📈 전략 비교 분석 리포트")
        report_lines.append("=" * 80)

        # 성과 지표 테이블
        report_lines.append("\n📊 성과 지표 비교")
        report_lines.append("-" * 100)
        report_lines.append(
            f"{'전략명':<20} {'수익률':<10} {'샤프비율':<10} {'최대낙폭':<10} {'승률':<8} {'거래횟수':<8} {'매매의견':<10}"
        )
        report_lines.append("-" * 100)

        for name, result in results.items():
            # 실제 승률 계산 (거래 데이터 기반)
            actual_win_rate = self._calculate_actual_win_rate_from_trades(result.trades)

            # 마지막 신호 확인 (매매의견)
            current_signal = self._get_current_signal(result.signals)

            report_lines.append(
                f"{name:<20} {result.total_return*100:>8.2f}% {result.sharpe_ratio:>8.2f} "
                f"{result.max_drawdown*100:>8.2f}% {actual_win_rate*100:>6.1f}% {result.total_trades:>6d} {current_signal:<10}"
            )

        # 최고 성과 전략 찾기 (실제 승률 기준)
        best_return = max(results.values(), key=lambda x: x.total_return)
        best_sharpe = max(results.values(), key=lambda x: x.sharpe_ratio)
        best_winrate = max(
            results.values(),
            key=lambda x: self._calculate_actual_win_rate_from_trades(x.trades),
        )

        report_lines.append("\n🏆 최고 성과 전략")
        report_lines.append("-" * 40)
        report_lines.append(
            f"최고 수익률: {best_return.name} ({best_return.total_return*100:.2f}%)"
        )
        report_lines.append(
            f"최고 샤프비율: {best_sharpe.name} ({best_sharpe.sharpe_ratio:.2f})"
        )
        report_lines.append(
            f"최고 승률: {best_winrate.name} ({self._calculate_actual_win_rate_from_trades(best_winrate.trades)*100:.1f}%)"
        )

        # 상세 분석
        if self.log_mode == "real_time":
            report_lines.append("\n📋 상세 거래 분석")
            report_lines.append("-" * 40)

            for name, result in results.items():
                if result.trades:
                    actual_win_rate = self._calculate_actual_win_rate_from_trades(
                        result.trades
                    )
                    report_lines.append(f"\n{name} 전략:")
                    report_lines.append(f"  총 거래: {len(result.trades)}회")
                    report_lines.append(f"  실제 승률: {actual_win_rate*100:.1f}%")
                    report_lines.append(
                        f"  평균 보유기간: {result.avg_hold_duration:.1f}시간"
                    )
                    report_lines.append(f"  수익 팩터: {result.profit_factor:.2f}")
                    report_lines.append(f"  SQN: {result.sqn:.2f}")

        return "\n".join(report_lines)

    def _calculate_actual_win_rate_from_trades(self, trades: List[Dict]) -> float:
        """거래 데이터에서 실제 승률 계산"""
        if not trades:
            return 0.0

        winning_trades = [trade for trade in trades if trade.get("pnl", 0) > 0]
        return len(winning_trades) / len(trades)

    def _get_current_signal(self, signals) -> str:
        """마지막 신호를 기반으로 현재 매매의견 반환"""
        if signals is None:
            return "중립"

        # 포트폴리오 모드: signals가 dict인 경우
        if isinstance(signals, dict):
            # 모든 종목의 신호를 종합
            total_signal = 0
            signal_count = 0
            for symbol, signal_df in signals.items():
                if (
                    isinstance(signal_df, pd.DataFrame)
                    and "signal" in signal_df.columns
                ):
                    last_signal = signal_df["signal"].iloc[-1]
                    total_signal += last_signal
                    signal_count += 1

            if signal_count > 0:
                avg_signal = total_signal / signal_count
                # 더 세밀한 신호 분석
                if avg_signal > 0.05:  # 임계값 낮춤: 0.1 → 0.05
                    return "매수"
                elif avg_signal < -0.05:  # 임계값 낮춤: -0.1 → -0.05
                    return "매도"
                elif avg_signal > 0.01:  # 약한 매수 신호
                    return "약매수"
                elif avg_signal < -0.01:  # 약한 매도 신호
                    return "약매도"
                else:
                    return "중립"
            else:
                return "중립"

        # 단일 종목 모드: signals가 DataFrame인 경우
        elif isinstance(signals, pd.DataFrame):
            if "signal" in signals.columns:
                last_signal = signals["signal"].iloc[-1]
                # 더 세밀한 신호 분석
                if last_signal > 0.5:  # 강한 매수 신호
                    return "매수"
                elif last_signal < -0.5:  # 강한 매도 신호
                    return "매도"
                elif last_signal > 0.1:  # 약한 매수 신호
                    return "약매수"
                elif last_signal < -0.1:  # 약한 매도 신호
                    return "약매도"
                else:
                    return "중립"
            else:
                return "중립"
        else:
            return "중립"

    def plot_comparison(
        self, results: Dict[str, StrategyResult], save_path: str = None
    ):
        """전략 비교 차트 생성"""
        if not results:
            print("차트를 생성할 데이터가 없습니다.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("전략 비교 분석", fontsize=16, fontweight="bold")

        # 1. 수익률 비교
        names = list(results.keys())
        returns = [r.total_return * 100 for r in results.values()]

        bars1 = axes[0, 0].bar(
            names,
            returns,
            color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8B5A3C"],
        )
        axes[0, 0].set_title("총 수익률 비교 (%)")
        axes[0, 0].set_ylabel("수익률 (%)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 수치 표시
        for bar, return_val in zip(bars1, returns):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{return_val:.1f}%",
                ha="center",
                va="bottom",
            )

        # 2. 샤프 비율 비교
        sharpes = [r.sharpe_ratio for r in results.values()]
        bars2 = axes[0, 1].bar(
            names,
            sharpes,
            color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8B5A3C"],
        )
        axes[0, 1].set_title("샤프 비율 비교")
        axes[0, 1].set_ylabel("샤프 비율")
        axes[0, 1].tick_params(axis="x", rotation=45)

        for bar, sharpe in zip(bars2, sharpes):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{sharpe:.2f}",
                ha="center",
                va="bottom",
            )

        # 3. 최대 낙폭 비교
        drawdowns = [abs(r.max_drawdown) * 100 for r in results.values()]
        bars3 = axes[1, 0].bar(
            names,
            drawdowns,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        )
        axes[1, 0].set_title("최대 낙폭 비교 (%)")
        axes[1, 0].set_ylabel("낙폭 (%)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        for bar, dd in zip(bars3, drawdowns):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{dd:.1f}%",
                ha="center",
                va="bottom",
            )

        # 4. 승률 비교
        winrates = [r.win_rate * 100 for r in results.values()]
        bars4 = axes[1, 1].bar(
            names,
            winrates,
            color=["#2ECC71", "#3498DB", "#9B59B6", "#E67E22", "#E74C3C"],
        )
        axes[1, 1].set_title("승률 비교 (%)")
        axes[1, 1].set_ylabel("승률 (%)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        for bar, wr in zip(bars4, winrates):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{wr:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"차트가 {save_path}에 저장되었습니다.")

        plt.show()

    def load_and_evaluate_optimized_strategy(
        self, strategy_name: str, symbol: str = None, results_dir: str = "results"
    ) -> StrategyResult:
        """최적화된 파라미터를 로드하여 전략 평가"""
        import glob
        import json

        self.logger.log_section(f"🔍 {strategy_name} 최적화된 파라미터 로드 및 평가")

        # 최적화 결과 파일 찾기
        pattern = f"{results_dir}/evaluation_{strategy_name}_{symbol}_*.json"
        if symbol is None:
            pattern = f"{results_dir}/evaluation_{strategy_name}_*.json"

        files = glob.glob(pattern)

        if not files:
            self.logger.log_error(
                f"❌ {strategy_name}의 최적화 결과 파일을 찾을 수 없습니다."
            )
            self.logger.log_info(f"검색 패턴: {pattern}")
            return None

        # 가장 최근 파일 선택
        latest_file = max(files, key=os.path.getctime)
        self.logger.log_info(f"📁 최적화 결과 파일: {latest_file}")

        try:
            # 최적화 결과 로드
            with open(latest_file, "r", encoding="utf-8") as f:
                optimization_result = json.load(f)

            # 최적 파라미터 추출
            best_params = optimization_result.get("best_params", {})
            optimization_score = optimization_result.get("optimization_score", 0)

            self.logger.log_info(f"📊 최적화 점수: {optimization_score:.4f}")
            self.logger.log_info(f"🔧 최적 파라미터: {best_params}")

            # 데이터 로드
            data_dict = self.load_data(symbol)
            if not data_dict:
                self.logger.log_error("데이터를 로드할 수 없습니다.")
                return None

            # 최적화된 파라미터로 전략 평가
            result = self.evaluate_strategy_with_params(
                strategy_name, data_dict, best_params
            )

            if result:
                self.logger.log_success(f"✅ 최적화된 {strategy_name} 전략 평가 완료")
                self.logger.log_info(
                    f"📈 최적화된 수익률: {result.total_return*100:.2f}%"
                )
                self.logger.log_info(f"📊 최적화된 샤프비율: {result.sharpe_ratio:.4f}")

                # 최적화 전후 비교
                self._compare_optimization_results(
                    strategy_name, data_dict, best_params, optimization_result
                )

            return result

        except Exception as e:
            self.logger.log_error(f"❌ 최적화된 전략 평가 중 오류: {str(e)}")
            return None

    def _compare_optimization_results(
        self,
        strategy_name: str,
        data_dict: Dict[str, pd.DataFrame],
        best_params: Dict[str, Any],
        optimization_result: Dict[str, Any],
    ):
        """최적화 전후 성과 비교"""
        self.logger.log_section(f"📊 {strategy_name} 최적화 전후 비교")

        # 기본 파라미터로 평가
        default_result = self.evaluate_strategy(strategy_name, data_dict)

        # 최적화된 파라미터로 평가
        optimized_result = self.evaluate_strategy_with_params(
            strategy_name, data_dict, best_params
        )

        if default_result and optimized_result:
            print_subsection_header("최적화 전후 성과 비교")
            print(f"{'지표':<15} {'최적화 전':<12} {'최적화 후':<12} {'개선도':<10}")
            print("-" * 55)

            # 수익률 비교
            return_improvement = (
                (optimized_result.total_return - default_result.total_return)
                / abs(default_result.total_return)
                * 100
                if default_result.total_return != 0
                else 0
            )
            print(
                f"{'수익률':<15} {default_result.total_return*100:>10.2f}% {optimized_result.total_return*100:>10.2f}% {return_improvement:>8.1f}%"
            )

            # 샤프비율 비교
            sharpe_improvement = (
                (optimized_result.sharpe_ratio - default_result.sharpe_ratio)
                / abs(default_result.sharpe_ratio)
                * 100
                if default_result.sharpe_ratio != 0
                else 0
            )
            print(
                f"{'샤프비율':<15} {default_result.sharpe_ratio:>10.2f} {optimized_result.sharpe_ratio:>10.2f} {sharpe_improvement:>8.1f}%"
            )

            # 승률 비교
            win_rate_improvement = (
                (optimized_result.win_rate - default_result.win_rate)
                / abs(default_result.win_rate)
                * 100
                if default_result.win_rate != 0
                else 0
            )
            print(
                f"{'승률':<15} {default_result.win_rate*100:>10.1f}% {optimized_result.win_rate*100:>10.1f}% {win_rate_improvement:>8.1f}%"
            )

            # 거래횟수 비교
            trades_improvement = (
                (optimized_result.total_trades - default_result.total_trades)
                / max(default_result.total_trades, 1)
                * 100
            )
            print(
                f"{'거래횟수':<15} {default_result.total_trades:>10d} {optimized_result.total_trades:>10d} {trades_improvement:>8.1f}%"
            )

            # 최대낙폭 비교
            dd_improvement = (
                (abs(optimized_result.max_drawdown) - abs(default_result.max_drawdown))
                / abs(default_result.max_drawdown)
                * 100
                if default_result.max_drawdown != 0
                else 0
            )
            print(
                f"{'최대낙폭':<15} {default_result.max_drawdown*100:>10.2f}% {optimized_result.max_drawdown*100:>10.2f}% {dd_improvement:>8.1f}%"
            )

    def evaluate_all_optimized_strategies(
        self,
        strategies: List[str] = None,
        symbols: List[str] = None,
        results_dir: str = "results",
    ) -> Dict[str, StrategyResult]:
        """모든 최적화된 전략들을 평가"""
        self.logger.log_section("🚀 모든 최적화된 전략 평가")

        if strategies is None:
            strategies = list(self.strategy_manager.strategies.keys())

        if symbols is None:
            symbols = self.config.get("data", {}).get("symbols", [])

        results = {}

        for strategy_name in strategies:
            for symbol in symbols:
                key = f"{strategy_name}_{symbol}"
                self.logger.log_info(f"🔄 {key} 최적화된 전략 평가 중...")

                result = self.load_and_evaluate_optimized_strategy(
                    strategy_name, symbol, results_dir
                )

                if result:
                    results[key] = result
                    self.logger.log_success(f"✅ {key} 평가 완료")
                else:
                    self.logger.log_warning(f"⚠️ {key} 평가 실패")

        # 종합 리포트 생성
        if results:
            self._generate_optimization_summary_report(results)

        return results

    def _generate_optimization_summary_report(self, results: Dict[str, StrategyResult]):
        """최적화된 전략들의 종합 리포트 생성"""
        self.logger.log_section("📊 최적화된 전략 종합 리포트")

        # 성과별 정렬
        sorted_results = sorted(
            results.items(), key=lambda x: x[1].total_return, reverse=True
        )

        print_subsection_header("최적화된 전략 성과 순위")
        print(
            f"{'순위':<4} {'전략-심볼':<25} {'수익률':<10} {'샤프비율':<10} {'승률':<8} {'거래횟수':<8} {'매매의견':<12}"
        )
        print("-" * 85)

        for i, (key, result) in enumerate(sorted_results, 1):
            # 현재 매매의견 계산
            current_signal = self._get_current_signal(result.signals)
            print(
                f"{i:<4} {key:<25} {result.total_return*100:>8.2f}% {result.sharpe_ratio:>8.2f} {result.win_rate*100:>6.1f}% {result.total_trades:>6d} {current_signal:<12}"
            )

        # 최고 성과 전략
        best_strategy = sorted_results[0]
        best_signal = self._get_current_signal(best_strategy[1].signals)
        print(
            f"\n🏆 최고 성과: {best_strategy[0]} ({best_strategy[1].total_return*100:.2f}%) - 현재 의견: {best_signal}"
        )

        # 평균 성과
        avg_return = np.mean([r.total_return for r in results.values()])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results.values()])
        print(f"📊 평균 수익률: {avg_return*100:.2f}%")
        print(f"📊 평균 샤프비율: {avg_sharpe:.2f}")

        # 매매의견 분포
        signal_counts = {}
        for result in results.values():
            signal = self._get_current_signal(result.signals)
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        print(f"\n📊 현재 매매의견 분포:")
        for signal, count in signal_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {signal}: {count}개 ({percentage:.1f}%)")

    def run_evaluation(
        self, symbol: str = None, strategies: List[str] = None, save_chart: bool = False
    ) -> Dict[str, StrategyResult]:
        """전체 평가 프로세스 실행"""
        # 메인 로거 설정 (간소화)
        symbols = [symbol] if symbol else self.config.get("data", {}).get("symbols", [])
        self.logger.setup_logger(strategy="main", symbols=symbols, mode="main")

        # 종합 요약 로거 설정
        self.logger.setup_summary_logger(
            symbols=symbols, timestamp=self.evaluation_start_time
        )

        self.logger.log_section("🎯 전략 평가 시스템 시작")
        self.logger.log_info(f"📁 데이터 디렉토리: {self.data_dir}")
        self.logger.log_info(f"📝 로그 모드: {self.log_mode}")
        self.logger.log_info(
            f"📈 분석 모드: {'포트폴리오' if self.portfolio_mode else '단일 종목'}"
        )

        # 데이터 로드
        self.logger.log_info("📂 데이터 로딩 중...")
        data_dict = self.load_data(symbol)
        self.logger.log_success(f"✅ 데이터 로딩 완료 ({len(data_dict)}개 종목)")

        # 전략 비교 분석
        results = self.compare_strategies(data_dict, strategies)

        # 최종 결과 요약
        if results:
            best_strategy = max(results.values(), key=lambda x: x.total_return)
            worst_strategy = min(results.values(), key=lambda x: x.total_return)

            self.logger.log_section("🏆 최종 평가 결과")
            self.logger.log_success(
                f"🥇 최고 수익률: {best_strategy.name} ({best_strategy.total_return*100:.2f}%)"
            )
            self.logger.log_warning(
                f"🥉 최저 수익률: {worst_strategy.name} ({worst_strategy.total_return*100:.2f}%)"
            )

            avg_return = np.mean([r.total_return for r in results.values()])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results.values()])
            self.logger.log_info(f"📊 평균 수익률: {avg_return*100:.2f}%")
            self.logger.log_info(f"📊 평균 샤프 비율: {avg_sharpe:.2f}")

        # 종합 요약 로그 생성
        self.logger.generate_final_summary(
            portfolio_mode=self.portfolio_mode, portfolio_method=self.portfolio_method
        )

        # 리포트 생성 및 출력
        report = self.generate_comparison_report(results)
        print(report)

        # 차트 생성
        if save_chart:
            chart_path = f"log/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.logger.log_info(f"📊 차트 생성 중: {chart_path}")
            self.plot_comparison(results, chart_path)
            self.logger.log_success(f"✅ 차트 저장 완료: {chart_path}")

        return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="전략 평가 및 비교 분석")
    parser.add_argument("--data_dir", default="data", help="데이터 디렉토리 경로")
    parser.add_argument("--symbol", help="분석할 특정 심볼 (예: CONL)")
    parser.add_argument("--strategies", nargs="+", help="분석할 전략 목록")
    parser.add_argument(
        "--log",
        choices=["summary", "real_time"],
        default="summary",
        help="로그 출력 모드",
    )
    parser.add_argument(
        "--portfolio", action="store_true", help="포트폴리오 모드 활성화"
    )
    parser.add_argument(
        "--portfolio_method",
        choices=["fixed", "strategy_weights", "signal_combined"],
        default="fixed",
        help="포트폴리오 비중 계산 방법 (fixed: 고정비중, strategy_weights: 전략별비중, signal_combined: 신호결합)",
    )
    parser.add_argument("--analysis_results", help="정량 분석 결과 JSON 파일 경로")
    parser.add_argument("--save_chart", action="store_true", help="차트 저장 여부")
    parser.add_argument(
        "--optimized", action="store_true", help="최적화된 파라미터로 평가"
    )
    parser.add_argument("--results_dir", default="results", help="최적화 결과 디렉토리")

    args = parser.parse_args()

    # 평가기 초기화
    evaluator = StrategyEvaluator(
        data_dir=args.data_dir,
        log_mode=args.log,
        portfolio_mode=args.portfolio,
        portfolio_method=args.portfolio_method,
        analysis_results_path=args.analysis_results,
    )

    # 평가 실행
    if args.optimized:
        # 최적화된 파라미터로 평가
        results = evaluator.evaluate_all_optimized_strategies(
            strategies=args.strategies,
            symbols=[args.symbol] if args.symbol else None,
            results_dir=args.results_dir,
        )
    else:
        # 기본 파라미터로 평가
        results = evaluator.run_evaluation(
            symbol=args.symbol, strategies=args.strategies, save_chart=args.save_chart
        )


if __name__ == "__main__":
    main()
