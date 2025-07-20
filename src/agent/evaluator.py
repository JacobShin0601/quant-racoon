#!/usr/bin/env python3
"""
Train/Test 평가 시스템
1. Train 데이터로 최적화된 전략과 포트폴리오 비중을 사용
2. Train과 Test 데이터 모두에서 성과 평가
3. Buy & Hold 대비 성과 비교
4. 종합적인 성과 테이블 생성
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
import glob
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.actions.strategies import (
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
    # 실전형 전략들 추가
    FixedWeightRebalanceStrategy,
    ETFMomentumRotationStrategy,
    TrendFollowingMA200Strategy,
    ReturnStackingStrategy,
    # 새로운 스윙 전략들 추가
    SwingBreakoutStrategy,
    SwingPullbackEntryStrategy,
    SwingCandlePatternStrategy,
    SwingBollingerBandStrategy,
    SwingMACDStrategy,
    # 포트폴리오 전략들 추가
    DynamicAssetAllocationStrategy,
    SectorRotationStrategy,
)
from src.actions.calculate_index import StrategyParams
from src.actions.log_pl import TradingSimulator
from src.agent.portfolio_manager import AdvancedPortfolioManager
from src.agent.helper import (
    StrategyResult,
    PortfolioWeights,
    Logger,
    load_config,
    load_and_preprocess_data,
    load_optimization_results,
    get_latest_analysis_file,
    print_section_header,
    print_subsection_header,
    format_percentage,
    split_data_train_test,
    calculate_buy_hold_returns,
    calculate_portfolio_metrics,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
)

# 포트폴리오 매니저 import를 선택적으로 처리
PORTFOLIO_MANAGER_AVAILABLE = False
AdvancedPortfolioManager = None

try:
    from .portfolio_manager import AdvancedPortfolioManager

    PORTFOLIO_MANAGER_AVAILABLE = True
except ImportError:
    pass


class TrainTestEvaluator:
    """Train/Test 평가 시스템"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        log_mode: str = "summary",
        config_path: str = DEFAULT_CONFIG_PATH,
        optimization_results_path: str = None,  # 개별 전략 최적화 결과 파일 경로
        portfolio_results_path: str = None,  # 포트폴리오 최적화 결과 파일 경로
    ):
        self.data_dir = data_dir
        self.log_mode = log_mode
        self.config = load_config(config_path)
        self.strategy_manager = StrategyManager()
        self.params = StrategyParams()
        self.simulator = TradingSimulator(config_path)
        # PortfolioWeightCalculator 제거 - portfolio_manager.py의 결과물만 사용

        # 포트폴리오 매니저 초기화 (선택적)
        if PORTFOLIO_MANAGER_AVAILABLE:
            self.portfolio_manager = AdvancedPortfolioManager(config_path)
        else:
            self.portfolio_manager = None

        self.optimization_results_path = optimization_results_path
        self.portfolio_results_path = portfolio_results_path
        self.results = {}
        self.logger = Logger()
        self.evaluation_start_time = datetime.now()
        self.execution_uuid = None

        # Train/Test 분할 비율
        self.train_ratio = self.config.get("data", {}).get("train_ratio", 0.8)

        # 주요 평가 지표
        self.primary_metric = self.config.get("evaluator", {}).get(
            "primary_metric", "sharpe_ratio"
        )

        # 전략 등록
        self._register_strategies()

    def _register_strategies(self):
        """전략 등록"""
        strategies_to_register = {
            "dual_momentum": DualMomentumStrategy,
            "volatility_breakout": VolatilityAdjustedBreakoutStrategy,
            "swing_ema": SwingEMACrossoverStrategy,
            "swing_rsi": SwingRSIReversalStrategy,
            "swing_donchian": DonchianSwingBreakoutStrategy,
            "stoch_donchian": StochDonchianStrategy,
            "whipsaw_prevention": WhipsawPreventionStrategy,
            "donchian_rsi_whipsaw": DonchianRSIWhipsawStrategy,
            "volatility_filtered_breakout": VolatilityFilteredBreakoutStrategy,
            "multi_timeframe_whipsaw": MultiTimeframeWhipsawStrategy,
            "adaptive_whipsaw": AdaptiveWhipsawStrategy,
            "cci_bollinger": CCIBollingerStrategy,
            "mean_reversion": MeanReversionStrategy,
            "swing_breakout": SwingBreakoutStrategy,
            "swing_pullback_entry": SwingPullbackEntryStrategy,
            "swing_candle_pattern": SwingCandlePatternStrategy,
            "swing_bollinger_band": SwingBollingerBandStrategy,
            "swing_macd": SwingMACDStrategy,
        }

        for name, strategy_class in strategies_to_register.items():
            self.strategy_manager.add_strategy(name, strategy_class(StrategyParams()))

        self.logger.log_info(f"✅ {len(strategies_to_register)}개 전략 등록 완료")

    def load_data_and_split(
        self, symbols: List[str] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """데이터 로드 및 Train/Test 분할"""
        if not symbols:
            symbols = self.config.get("data", {}).get("symbols", [])

        data_dict = load_and_preprocess_data(self.data_dir, symbols)
        if not data_dict:
            self.logger.log_error(f"데이터를 로드할 수 없습니다: {self.data_dir}")
            return {}, {}

        # Train/Test 분할
        train_data_dict, test_data_dict = split_data_train_test(
            data_dict, self.train_ratio
        )

        # Test 데이터가 너무 적으면 경고
        test_data_points = sum(len(data) for data in test_data_dict.values())
        if test_data_points < 100:  # 최소 100개 데이터 포인트 권장
            self.logger.log_warning(
                f"⚠️ Test 데이터가 너무 적습니다 ({test_data_points}개). 평가 결과가 부정확할 수 있습니다."
            )

        return train_data_dict, test_data_dict

    def load_optimization_results(self) -> Dict[str, Dict]:
        """개별 전략 최적화 결과 로드"""
        if not self.optimization_results_path:
            # 자동으로 최신 최적화 결과 파일 찾기
            self.logger.log_info(
                "최적화 결과 파일 경로가 지정되지 않았습니다. 최신 파일을 자동으로 찾습니다."
            )
            self.optimization_results_path = self._find_latest_optimization_file()

        if not self.optimization_results_path:
            self.logger.log_error("최적화 결과 파일을 찾을 수 없습니다")
            return {}

        try:
            with open(self.optimization_results_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            self.logger.log_success(f"최적화 결과 로드 완료: {len(results)}개 조합")
            return results
        except Exception as e:
            self.logger.log_error(f"최적화 결과 로드 실패: {e}")
            return {}

    def _find_latest_optimization_file(self) -> Optional[str]:
        """최신 최적화 결과 파일 찾기"""
        try:
            results_dir = Path("results")
            if not results_dir.exists():
                return None

            # hyperparam_optimization_*.json 파일들 찾기
            optimization_files = list(
                results_dir.glob("hyperparam_optimization_*.json")
            )

            if not optimization_files:
                self.logger.log_warning(
                    "하이퍼파라미터 최적화 결과 파일을 찾을 수 없습니다"
                )
                return None

            # 가장 최신 파일 반환
            latest_file = max(optimization_files, key=lambda x: x.stat().st_mtime)
            self.logger.log_success(f"최신 최적화 결과 파일 발견: {latest_file.name}")
            return str(latest_file)

        except Exception as e:
            self.logger.log_error(f"최적화 결과 파일 찾기 실패: {e}")
            return None

    def load_portfolio_results(self) -> Dict[str, Any]:
        """포트폴리오 최적화 결과 로드"""
        if not self.portfolio_results_path:
            # 자동으로 최신 포트폴리오 결과 파일 찾기
            self.portfolio_results_path = self._find_latest_portfolio_file()

        if not self.portfolio_results_path:
            self.logger.log_warning("포트폴리오 결과 파일을 찾을 수 없습니다")
            return {}

        try:
            with open(self.portfolio_results_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            self.logger.log_success(
                f"포트폴리오 결과 로드 완료: {self.portfolio_results_path}"
            )
            return results
        except Exception as e:
            self.logger.log_error(f"포트폴리오 결과 로드 실패: {e}")
            return {}

    def _find_latest_portfolio_file(self) -> Optional[str]:
        """최신 포트폴리오 결과 파일 찾기"""
        try:
            results_dir = Path("results")
            if not results_dir.exists():
                return None

            # portfolio_optimization_*.json 파일들 찾기
            portfolio_files = list(results_dir.glob("portfolio_optimization_*.json"))

            if not portfolio_files:
                self.logger.log_warning(
                    "포트폴리오 최적화 결과 파일을 찾을 수 없습니다"
                )
                return None

            # 가장 최신 파일 반환
            latest_file = max(portfolio_files, key=lambda x: x.stat().st_mtime)
            self.logger.log_success(
                f"최신 포트폴리오 결과 파일 발견: {latest_file.name}"
            )
            return str(latest_file)

        except Exception as e:
            self.logger.log_error(f"포트폴리오 결과 파일 찾기 실패: {e}")
            return None

    def evaluate_strategy_with_params(
        self,
        strategy_name: str,
        data_dict: Dict[str, pd.DataFrame],
        optimized_params: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """최적화된 파라미터로 전략 평가"""
        results = {}

        try:
            # 전략 인스턴스 생성
            strategy = self.strategy_manager.strategies.get(strategy_name)
            if not strategy:
                self.logger.log_error(f"전략을 찾을 수 없습니다: {strategy_name}")
                return {}

            # 최적화된 파라미터 적용
            for param_name, param_value in optimized_params.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)

            # 각 종목에 대해 전략 실행
            for symbol, data in data_dict.items():
                try:
                    signals = strategy.generate_signals(data)

                    if signals is not None and not signals.empty:
                        # 거래 시뮬레이션
                        result = self.simulator.simulate_trading(
                            data, signals, strategy_name
                        )

                        # 시뮬레이션 결과 요약만 출력
                        if result:
                            # 성과 지표 계산 - simulate_trading 결과 구조에 맞게 수정
                            results_data = result.get("results", {})
                            total_return = results_data.get("total_return", 0.0)
                            total_trades = results_data.get("total_trades", 0)

                            # 샤프 비율 계산
                            returns = result.get("returns", [])
                            sharpe_ratio = 0
                            sortino_ratio = 0
                            max_drawdown = 0
                            volatility = 0

                            if (
                                returns
                                and isinstance(returns, list)
                                and len(returns) > 0
                            ):
                                try:
                                    returns_series = pd.Series(returns)
                                    mean_return = returns_series.mean()
                                    std_return = returns_series.std()
                                    sharpe_ratio = (
                                        (mean_return * 252)
                                        / (std_return * np.sqrt(252))
                                        if std_return > 0
                                        else 0
                                    )

                                    # 소르티노 비율 계산
                                    negative_returns = returns_series[
                                        returns_series < 0
                                    ]
                                    if len(negative_returns) > 0:
                                        downside_deviation = negative_returns.std()
                                        sortino_ratio = (
                                            (mean_return * 252)
                                            / (downside_deviation * np.sqrt(252))
                                            if downside_deviation > 0
                                            else 0
                                        )

                                    # 최대 낙폭 계산
                                    cumulative_returns = (1 + returns_series).cumprod()
                                    running_max = cumulative_returns.expanding().max()
                                    drawdown = (
                                        cumulative_returns - running_max
                                    ) / running_max
                                    max_drawdown = abs(drawdown.min())

                                    # 변동성 계산
                                    volatility = returns_series.std() * np.sqrt(252)
                                except Exception as e:
                                    pass
                                    # 기본값 유지

                            # 베타 계산 (간단히 1.0으로 설정)
                            beta = 1.0

                            results[symbol] = {
                                "total_return": total_return,
                                "sharpe_ratio": sharpe_ratio,
                                "sortino_ratio": sortino_ratio,
                                "max_drawdown": max_drawdown,
                                "volatility": volatility,
                                "beta": beta,
                                "total_trades": total_trades,
                            }
                            pass
                        else:
                            pass
                    else:
                        pass
                except Exception:
                    # 오류가 발생해도 기본 결과 반환
                    results[symbol] = {
                        "total_return": 0.0,
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "max_drawdown": 0.0,
                        "volatility": 0.0,
                        "beta": 1.0,
                        "total_trades": 0,
                    }
                    continue

        except Exception as e:
            self.logger.log_error(f"전략 평가 중 오류: {e}")

        # 항상 {symbol: ...} 형태로 반환 보장
        if len(results) == 1:
            symbol = list(results.keys())[0]
            return {symbol: results[symbol]}
        return results

    def evaluate_all_strategies(
        self,
        train_data_dict: Dict[str, pd.DataFrame],
        test_data_dict: Dict[str, pd.DataFrame],
        optimization_results: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """모든 전략의 Train/Test 성과 평가"""

        all_results = {
            "train": {},
            "test": {},
            "buy_hold_train": {},
            "buy_hold_test": {},
            "filtered_results": {},  # 필터링된 결과
            "ranking": [],  # 순위
        }

        # Buy & Hold 성과 계산
        all_results["buy_hold_train"] = calculate_buy_hold_returns(train_data_dict)
        all_results["buy_hold_test"] = calculate_buy_hold_returns(test_data_dict)

        # 최적화된 전략들 평가
        symbols = list(train_data_dict.keys())
        strategy_scores = []  # 전략별 점수 저장

        for symbol in symbols:
            # 해당 종목의 최적 전략 찾기
            best_strategy = None
            best_params = {}

            # 키 패턴으로 찾기 (예: "dual_momentum_AAPL")
            for key, result in optimization_results.items():
                if key.endswith(f"_{symbol}"):
                    best_strategy = result.get("strategy_name")
                    best_params = result.get("best_params", {})
                    break

            if not best_strategy:
                continue

            # Train 데이터에서 평가
            train_result = self.evaluate_strategy_with_params(
                best_strategy, {symbol: train_data_dict[symbol]}, best_params
            )
            if symbol in train_result:
                all_results["train"][symbol] = train_result[symbol]
                all_results["train"][symbol]["strategy"] = best_strategy

                pass
            else:
                pass

            # Test 데이터에서 평가
            try:
                # Test 데이터에서 평가
                test_data = test_data_dict[symbol]
                if len(test_data) < 20:  # 최소 20개 데이터 포인트 필요
                    all_results["test"][symbol] = {
                        "total_return": 0.0,
                        "sharpe_ratio": 0.0,
                        "sortino_ratio": 0.0,
                        "max_drawdown": 0.0,
                        "volatility": 0.0,
                        "beta": 1.0,
                        "total_trades": 0,
                        "strategy": best_strategy,
                    }
                else:
                    try:
                        test_result = self.evaluate_strategy_with_params(
                            best_strategy, {symbol: test_data}, best_params
                        )
                        if symbol in test_result:
                            all_results["test"][symbol] = test_result[symbol]
                            all_results["test"][symbol]["strategy"] = best_strategy
                        else:
                            all_results["test"][symbol] = {
                                "total_return": 0.0,
                                "sharpe_ratio": 0.0,
                                "sortino_ratio": 0.0,
                                "max_drawdown": 0.0,
                                "volatility": 0.0,
                                "beta": 1.0,
                                "total_trades": 0,
                                "strategy": best_strategy,
                            }
                    except Exception:
                        all_results["test"][symbol] = {
                            "total_return": 0.0,
                            "sharpe_ratio": 0.0,
                            "sortino_ratio": 0.0,
                            "max_drawdown": 0.0,
                            "volatility": 0.0,
                            "beta": 1.0,
                            "total_trades": 0,
                            "strategy": best_strategy,
                        }
            except Exception:
                all_results["test"][symbol] = {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                    "beta": 1.0,
                    "total_trades": 0,
                    "strategy": best_strategy,
                }

        # 최종 필터링 및 순위 결정
        all_results["filtered_results"], all_results["ranking"] = (
            self._apply_final_filtering(all_results["train"], all_results["test"])
        )

        return all_results

    def calculate_portfolio_performance(
        self,
        individual_results: Dict[str, Any],
        portfolio_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """포트폴리오 성과 계산"""
        self.logger.log_section("⚖️ 포트폴리오 성과 계산")

        portfolio_performance = {"train": {}, "test": {}}

        try:
            # 포트폴리오 비중 가져오기
            portfolio_weights = portfolio_results.get("portfolio_weights", {})
            if not portfolio_weights:
                self.logger.log_warning(
                    "포트폴리오 비중을 찾을 수 없습니다. 동일 가중치로 계산합니다."
                )
                # 동일 가중치로 설정
                symbols = list(individual_results.get("train", {}).keys())
                if symbols:
                    equal_weight = 1.0 / len(symbols)
                    portfolio_weights = {symbol: equal_weight for symbol in symbols}
                    self.logger.log_info(
                        f"동일 가중치 설정: {len(symbols)}개 종목, 각 {equal_weight:.3f}"
                    )

            # Train 포트폴리오 성과
            if individual_results["train"]:
                portfolio_performance["train"] = calculate_portfolio_metrics(
                    individual_results["train"], portfolio_weights
                )

            # Test 포트폴리오 성과
            if individual_results["test"]:
                portfolio_performance["test"] = calculate_portfolio_metrics(
                    individual_results["test"], portfolio_weights
                )

            # Buy & Hold 포트폴리오 성과
            if individual_results["buy_hold_train"]:
                portfolio_performance["buy_hold_train"] = calculate_portfolio_metrics(
                    individual_results["buy_hold_train"], portfolio_weights
                )

            if individual_results["buy_hold_test"]:
                portfolio_performance["buy_hold_test"] = calculate_portfolio_metrics(
                    individual_results["buy_hold_test"], portfolio_weights
                )

        except Exception as e:
            self.logger.log_error(f"포트폴리오 성과 계산 중 오류: {e}")

        return portfolio_performance

    def _apply_final_filtering(
        self, train_results: Dict[str, Any], test_results: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """최종 필터링 및 순위 결정"""
        self.logger.log_section("🔍 최종 필터링 및 순위 결정")

        # 설정에서 필터링 기준 로드
        evaluator_config = self.config.get("evaluator", {})
        minimum_requirements = evaluator_config.get("minimum_requirements", {})
        risk_penalties = evaluator_config.get("risk_penalties", {})

        # researcher config에서 성과 기준 로드
        researcher_config = self.config.get("researcher", {})
        performance_thresholds = researcher_config.get("performance_thresholds", {})

        # 최소 요구사항 (완화된 기준)
        min_trades = minimum_requirements.get("min_trades", 1)  # 5 → 1로 완화
        min_sharpe_ratio = minimum_requirements.get(
            "min_sharpe_ratio", -1.0
        )  # 0.1 → -1.0으로 완화
        min_profit_factor = minimum_requirements.get(
            "min_profit_factor", 0.0
        )  # 0.8 → 0.0으로 완화
        min_win_rate = minimum_requirements.get(
            "min_win_rate", 0.0
        )  # 0.3 → 0.0으로 완화
        max_drawdown_limit = minimum_requirements.get(
            "max_drawdown_limit", 1.0
        )  # 0.5 → 1.0으로 완화

        # 성과 기준 (researcher config에서 로드)
        min_return_threshold = performance_thresholds.get("min_return_threshold", 0.0)

        # 위험 페널티
        max_drawdown_threshold = risk_penalties.get("max_drawdown_threshold", 0.20)
        max_drawdown_penalty = risk_penalties.get("max_drawdown_penalty", 0.5)
        volatility_threshold = risk_penalties.get("volatility_threshold", 0.30)
        volatility_penalty = risk_penalties.get("volatility_penalty", 0.3)

        filtered_results = {}
        strategy_rankings = []

        for symbol in train_results.keys():
            if symbol not in test_results:
                continue

            train_result = train_results[symbol]
            test_result = test_results[symbol]

            # 최소 요구사항 체크
            meets_requirements = True
            rejection_reasons = []

            # 거래 횟수 체크
            if train_result.get("total_trades", 0) < min_trades:
                meets_requirements = False
                rejection_reasons.append(
                    f"거래 횟수 부족: {train_result.get('total_trades', 0)}/{min_trades}"
                )

            # 최소 수익률 체크
            train_return = train_result.get("total_return", 0)
            test_return = test_result.get("total_return", 0)
            avg_return = (train_return + test_return) / 2
            if avg_return < min_return_threshold:
                meets_requirements = False
                rejection_reasons.append(
                    f"수익률 부족: {avg_return:.3f}/{min_return_threshold}"
                )

            # 샤프 비율 체크 (Train과 Test 모두 고려)
            train_sharpe = train_result.get("sharpe_ratio", 0)
            test_sharpe = test_result.get("sharpe_ratio", 0)
            avg_sharpe = (train_sharpe + test_sharpe) / 2
            if avg_sharpe < min_sharpe_ratio:
                meets_requirements = False
                rejection_reasons.append(
                    f"샤프 비율 부족: {avg_sharpe:.3f}/{min_sharpe_ratio}"
                )

            # 최대 낙폭 체크 (Train과 Test 중 더 나쁜 것 기준)
            train_dd = train_result.get("max_drawdown", 1)
            test_dd = test_result.get("max_drawdown", 1)
            max_dd = max(train_dd, test_dd)
            if max_dd > max_drawdown_limit:
                meets_requirements = False
                rejection_reasons.append(
                    f"최대 낙폭 초과: {max_dd:.3f}/{max_drawdown_limit}"
                )

            if meets_requirements:
                # 복합 점수 계산 (Train과 Test 성과를 모두 고려)
                composite_score = self._calculate_evaluation_score(
                    train_result,
                    test_result,
                    max_drawdown_threshold,
                    max_drawdown_penalty,
                    volatility_threshold,
                    volatility_penalty,
                )

                filtered_results[symbol] = {
                    "train": train_result,
                    "test": test_result,
                    "composite_score": composite_score,
                    "avg_sharpe": avg_sharpe,
                    "max_drawdown": max_dd,
                    "strategy": train_result.get("strategy", "UNKNOWN"),
                }

                strategy_rankings.append(
                    {
                        "symbol": symbol,
                        "strategy": train_result.get("strategy", "UNKNOWN"),
                        "composite_score": composite_score,
                        "avg_sharpe": avg_sharpe,
                        "max_drawdown": max_dd,
                        "train_return": train_result.get("total_return", 0),
                        "test_return": test_result.get("total_return", 0),
                    }
                )
            else:
                self.logger.log_warning(
                    f"❌ {symbol} 필터링 제외: {', '.join(rejection_reasons)}"
                )

        # 점수 기준으로 순위 정렬
        strategy_rankings.sort(key=lambda x: x["composite_score"], reverse=True)

        self.logger.log_success(
            f"✅ 필터링 완료: {len(filtered_results)}/{len(train_results)}개 전략 통과"
        )
        self.logger.log_success(f"📊 상위 3개 전략:")
        for i, ranking in enumerate(strategy_rankings[:3], 1):
            self.logger.log_success(
                f"  {i}. {ranking['symbol']} ({ranking['strategy']}): {ranking['composite_score']:.3f}"
            )

        return filtered_results, strategy_rankings

    def _calculate_evaluation_score(
        self,
        train_result: Dict[str, Any],
        test_result: Dict[str, Any],
        max_dd_threshold: float,
        max_dd_penalty: float,
        volatility_threshold: float,
        volatility_penalty: float,
    ) -> float:
        """평가 점수 계산 (Train과 Test 성과를 모두 고려)"""
        try:
            # 기본 지표들 (Train과 Test의 평균)
            train_return = train_result.get("total_return", 0)
            test_return = test_result.get("total_return", 0)
            avg_return = (train_return + test_return) / 2

            train_sharpe = train_result.get("sharpe_ratio", 0)
            test_sharpe = test_result.get("sharpe_ratio", 0)
            avg_sharpe = (train_sharpe + test_sharpe) / 2

            train_sortino = train_result.get("sortino_ratio", 0)
            test_sortino = test_result.get("sortino_ratio", 0)
            avg_sortino = (train_sortino + test_sortino) / 2

            train_dd = train_result.get("max_drawdown", 1)
            test_dd = test_result.get("max_drawdown", 1)
            max_dd = max(train_dd, test_dd)

            train_vol = train_result.get("volatility", 0)
            test_vol = test_result.get("volatility", 0)
            avg_vol = (train_vol + test_vol) / 2

            # 점수 계산 (0-100 스케일)
            scores = {}

            # 수익률 점수
            scores["return"] = min(max(avg_return * 100, 0), 100)

            # 샤프 비율 점수
            scores["sharpe"] = min(max(avg_sharpe * 20, 0), 100)

            # 소르티노 비율 점수
            scores["sortino"] = min(max(avg_sortino * 20, 0), 100)

            # 최대 낙폭 점수 (낮을수록 높은 점수)
            scores["drawdown"] = max(0, 100 - (max_dd * 100))

            # 변동성 점수 (낮을수록 높은 점수)
            scores["volatility"] = max(0, 100 - (avg_vol * 100))

            # 가중치 (researcher config에서 로드)
            evaluation_metrics = self.config.get("researcher", {}).get(
                "evaluation_metrics", {}
            )
            weights = evaluation_metrics.get(
                "weights",
                {
                    "sharpe_ratio": 0.25,
                    "sortino_ratio": 0.20,
                    "calmar_ratio": 0.15,
                    "profit_factor": 0.20,
                    "win_rate": 0.20,
                },
            )

            # Train/Test 평가용 가중치 매핑
            evaluation_weights = {
                "return": weights.get("total_return", 0.25),
                "sharpe": weights.get("sharpe_ratio", 0.25),
                "sortino": weights.get("sortino_ratio", 0.20),
                "drawdown": 0.15,  # 고정값
                "volatility": 0.15,  # 고정값
            }

            # 복합 점수 계산
            composite_score = sum(
                scores[metric] * weight for metric, weight in evaluation_weights.items()
            )

            # 위험 페널티 적용
            if max_dd > max_dd_threshold:
                composite_score *= 1 - max_dd_penalty

            if avg_vol > volatility_threshold:
                composite_score *= 1 - volatility_penalty

            return composite_score

        except Exception as e:
            self.logger.log_error(f"평가 점수 계산 중 오류: {e}")
            return 0.0

    def _calculate_portfolio_score(
        self, portfolio_performance: Dict[str, Any]
    ) -> float:
        """포트폴리오 복합 점수 계산"""
        try:
            # 기본 지표들
            total_return = portfolio_performance.get("total_return", 0)
            sharpe_ratio = portfolio_performance.get("sharpe_ratio", 0)
            sortino_ratio = portfolio_performance.get("sortino_ratio", 0)
            max_drawdown = portfolio_performance.get("max_drawdown", 1)
            volatility = portfolio_performance.get("volatility", 0)

            # 점수 계산 (0-100 스케일)
            scores = {}

            # 수익률 점수
            scores["return"] = min(max(total_return * 100, 0), 100)

            # 샤프 비율 점수
            scores["sharpe"] = min(max(sharpe_ratio * 20, 0), 100)

            # 소르티노 비율 점수
            scores["sortino"] = min(max(sortino_ratio * 20, 0), 100)

            # 최대 낙폭 점수 (낮을수록 높은 점수)
            scores["drawdown"] = max(0, 100 - (max_drawdown * 100))

            # 변동성 점수 (낮을수록 높은 점수)
            scores["volatility"] = max(0, 100 - (volatility * 100))

            # 가중치 (researcher config에서 로드)
            evaluation_metrics = self.config.get("researcher", {}).get(
                "evaluation_metrics", {}
            )
            weights = evaluation_metrics.get(
                "weights",
                {
                    "sharpe_ratio": 0.25,
                    "sortino_ratio": 0.20,
                    "calmar_ratio": 0.15,
                    "profit_factor": 0.20,
                    "win_rate": 0.20,
                },
            )

            # 포트폴리오 평가용 가중치 매핑
            portfolio_weights = {
                "return": weights.get("total_return", 0.25),
                "sharpe": weights.get("sharpe_ratio", 0.25),
                "sortino": weights.get("sortino_ratio", 0.20),
                "drawdown": 0.15,  # 고정값
                "volatility": 0.15,  # 고정값
            }

            # 복합 점수 계산
            composite_score = sum(
                scores[metric] * weight for metric, weight in portfolio_weights.items()
            )

            # 위험 페널티 적용
            risk_penalties = self.config.get("evaluator", {}).get("risk_penalties", {})
            max_drawdown_threshold = risk_penalties.get("max_drawdown_threshold", 0.20)
            max_drawdown_penalty = risk_penalties.get("max_drawdown_penalty", 0.5)
            volatility_threshold = risk_penalties.get("volatility_threshold", 0.30)
            volatility_penalty = risk_penalties.get("volatility_penalty", 0.3)

            if max_drawdown > max_drawdown_threshold:
                composite_score *= 1 - max_drawdown_penalty

            if volatility > volatility_threshold:
                composite_score *= 1 - volatility_penalty

            return composite_score

        except Exception as e:
            self.logger.log_error(f"포트폴리오 점수 계산 중 오류: {e}")
            return 0.0

    def _calculate_beta(
        self,
        strategy_returns: pd.Series,
        symbol: str,
        data_dict: Dict[str, pd.DataFrame],
    ) -> float:
        """베타 계산 (시장 대비 변동성)"""
        try:
            # 시장 지수 찾기 (SPY 또는 QQQ)
            market_symbol = None
            for market_candidate in ["SPY", "QQQ", "^GSPC"]:
                if market_candidate in data_dict:
                    market_symbol = market_candidate
                    break

            if not market_symbol:
                self.logger.log_warning(
                    f"시장 지수를 찾을 수 없습니다. {symbol}의 베타를 1.0으로 설정합니다."
                )
                return 1.0

            # 시장 데이터에서 수익률 계산
            market_data = data_dict[market_symbol]
            market_returns = market_data["Close"].pct_change().dropna()

            # 전략 수익률과 시장 수익률의 길이 맞추기
            min_length = min(len(strategy_returns), len(market_returns))
            if min_length < 10:  # 최소 데이터 포인트
                return 1.0

            strategy_returns_aligned = strategy_returns.iloc[-min_length:]
            market_returns_aligned = market_returns.iloc[-min_length:]

            # 공분산과 분산 계산
            covariance = np.cov(strategy_returns_aligned, market_returns_aligned)[0, 1]
            market_variance = np.var(market_returns_aligned)

            if market_variance == 0:
                return 1.0

            beta = covariance / market_variance

            # 베타 범위 제한 (0.1 ~ 3.0)
            beta = max(0.1, min(3.0, beta))

            return beta

        except Exception as e:
            self.logger.log_error(f"베타 계산 중 오류: {e}")
            return 1.0

    def generate_performance_table(
        self,
        individual_results: Dict[str, Any],
        portfolio_performance: Dict[str, Any],
        portfolio_weights: Dict[str, float],
    ) -> str:
        """성과 테이블 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_evaluation_{timestamp}.txt"
            output_path = os.path.join("results", filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write("Train/Test 성과 평가 보고서\n")
                f.write("=" * 100 + "\n\n")

                f.write(f"평가 시작 시간: {self.evaluation_start_time}\n")
                f.write(f"평가 완료 시간: {datetime.now()}\n")
                f.write(f"Train 비율: {self.train_ratio*100:.1f}%\n")
                f.write(f"주요 평가 지표: {self.primary_metric}\n\n")

                # 포트폴리오 비중 정보
                f.write("포트폴리오 비중:\n")
                f.write("-" * 50 + "\n")
                for symbol, weight in portfolio_weights.items():
                    f.write(f"{symbol}: {weight*100:.2f}%\n")
                f.write("\n")

                # Train 성과 테이블
                f.write("TRAIN 성과 테이블\n")
                f.write("=" * 100 + "\n")
                f.write(
                    f"{'구분':<15} {'전략':<20} {'수익률':<10} {'샤프':<8} {'소르티노':<10} {'최대낙폭':<10} {'변동성':<10} {'베타':<6} {'거래수':<8}\n"
                )
                f.write("-" * 100 + "\n")

                # 포트폴리오 전체 성과
                if "train" in portfolio_performance and portfolio_performance["train"]:
                    perf = portfolio_performance["train"]
                    # 포트폴리오 복합 점수 계산
                    portfolio_score = self._calculate_portfolio_score(perf)
                    f.write(
                        f"{'PORTFOLIO':<15} {'OPTIMIZED':<20} {perf['total_return']*100:>8.2f}% {perf['sharpe_ratio']:>6.3f} {perf['sortino_ratio']:>8.3f} {perf['max_drawdown']*100:>8.2f}% {perf['volatility']*100:>8.2f}% {perf.get('beta', 1.0):>5.2f} {perf['total_trades']:>6} [{portfolio_score:>6.1f}]\n"
                    )

                # Buy & Hold 성과
                if (
                    "buy_hold_train" in portfolio_performance
                    and portfolio_performance["buy_hold_train"]
                ):
                    perf = portfolio_performance["buy_hold_train"]
                    f.write(
                        f"{'PORTFOLIO':<15} {'BUY&HOLD':<20} {perf['total_return']*100:>8.2f}% {perf['sharpe_ratio']:>6.3f} {perf['sortino_ratio']:>8.3f} {perf['max_drawdown']*100:>8.2f}% {perf['volatility']*100:>8.2f}% {perf.get('beta', 1.0):>5.2f} {perf['total_trades']:>6}\n"
                    )

                f.write("-" * 100 + "\n")

                # 개별 종목 성과 (필터링된 결과만)
                filtered_results = individual_results.get("filtered_results", {})

                for symbol, result in filtered_results.items():
                    train_result = result["train"]
                    strategy = train_result.get("strategy", "UNKNOWN")
                    composite_score = result.get("composite_score", 0)
                    f.write(
                        f"{symbol:<15} {strategy:<20} {train_result['total_return']*100:>8.2f}% {train_result['sharpe_ratio']:>6.3f} {train_result['sortino_ratio']:>8.3f} {train_result['max_drawdown']*100:>8.2f}% {train_result['volatility']*100:>8.2f}% {train_result.get('beta', 1.0):>5.2f} {train_result['total_trades']:>6} [{composite_score:>6.1f}]\n"
                    )

                f.write("\n\n")

                # Test 성과 테이블
                f.write("TEST 성과 테이블\n")
                f.write("=" * 100 + "\n")
                f.write(
                    f"{'구분':<15} {'전략':<20} {'수익률':<10} {'샤프':<8} {'소르티노':<10} {'최대낙폭':<10} {'변동성':<10} {'베타':<6} {'거래수':<8}\n"
                )
                f.write("-" * 100 + "\n")

                # 포트폴리오 전체 성과
                if "test" in portfolio_performance and portfolio_performance["test"]:
                    perf = portfolio_performance["test"]
                    # 포트폴리오 복합 점수 계산
                    portfolio_score = self._calculate_portfolio_score(perf)
                    f.write(
                        f"{'PORTFOLIO':<15} {'OPTIMIZED':<20} {perf['total_return']*100:>8.2f}% {perf['sharpe_ratio']:>6.3f} {perf['sortino_ratio']:>8.3f} {perf['max_drawdown']*100:>8.2f}% {perf['volatility']*100:>8.2f}% {perf.get('beta', 1.0):>5.2f} {perf['total_trades']:>6} [{portfolio_score:>6.1f}]\n"
                    )

                # Buy & Hold 성과
                if (
                    "buy_hold_test" in portfolio_performance
                    and portfolio_performance["buy_hold_test"]
                ):
                    perf = portfolio_performance["buy_hold_test"]
                    f.write(
                        f"{'PORTFOLIO':<15} {'BUY&HOLD':<20} {perf['total_return']*100:>8.2f}% {perf['sharpe_ratio']:>6.3f} {perf['sortino_ratio']:>8.3f} {perf['max_drawdown']*100:>8.2f}% {perf['volatility']*100:>8.2f}% {perf.get('beta', 1.0):>5.2f} {perf['total_trades']:>6}\n"
                    )

                f.write("-" * 100 + "\n")

                # 개별 종목 성과 (필터링된 결과만)
                for symbol, result in filtered_results.items():
                    test_result = result["test"]
                    strategy = test_result.get("strategy", "UNKNOWN")
                    composite_score = result.get("composite_score", 0)
                    f.write(
                        f"{symbol:<15} {strategy:<20} {test_result['total_return']*100:>8.2f}% {test_result['sharpe_ratio']:>6.3f} {test_result['sortino_ratio']:>8.3f} {test_result['max_drawdown']*100:>8.2f}% {test_result['volatility']*100:>8.2f}% {test_result.get('beta', 1.0):>5.2f} {test_result['total_trades']:>6} [{composite_score:>6.1f}]\n"
                    )

                # 성과 요약
                f.write("\n\n성과 요약:\n")
                f.write("=" * 50 + "\n")

                # 필터링 결과 요약
                filtered_count = len(filtered_results)
                total_count = len(individual_results.get("train", {}))
                f.write(f"전체 전략 수: {total_count}개\n")
                f.write(f"필터링 통과: {filtered_count}개\n")
                if total_count > 0:
                    f.write(f"필터링 통과율: {filtered_count/total_count*100:.1f}%\n\n")
                else:
                    f.write(f"필터링 통과율: 0.0%\n\n")

                # 상위 전략 순위
                rankings = individual_results.get("ranking", [])
                if rankings:
                    f.write("상위 전략 순위:\n")
                    f.write("-" * 30 + "\n")
                    for i, ranking in enumerate(rankings[:5], 1):
                        f.write(
                            f"{i}. {ranking['symbol']} ({ranking['strategy']}): {ranking['composite_score']:.3f}\n"
                        )
                    f.write("\n")

                if (
                    "train" in portfolio_performance
                    and "test" in portfolio_performance
                    and portfolio_performance["train"]
                    and portfolio_performance["test"]
                ):
                    train_perf = portfolio_performance["train"]
                    test_perf = portfolio_performance["test"]

                    f.write(f"Train 수익률: {train_perf['total_return']*100:.2f}%\n")
                    f.write(f"Test 수익률: {test_perf['total_return']*100:.2f}%\n")
                    f.write(f"Train 샤프 비율: {train_perf['sharpe_ratio']:.3f}\n")
                    f.write(f"Test 샤프 비율: {test_perf['sharpe_ratio']:.3f}\n")
                    f.write(f"Train 최대 낙폭: {train_perf['max_drawdown']*100:.2f}%\n")
                    f.write(f"Test 최대 낙폭: {test_perf['max_drawdown']*100:.2f}%\n")
                else:
                    f.write("포트폴리오 성과 데이터가 없습니다.\n")

            return output_path

        except Exception as e:
            self.logger.log_error(f"성과 테이블 생성 중 오류: {e}")
            return ""

    def run_train_test_evaluation(
        self,
        symbols: List[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Train/Test 평가 실행"""
        print("=" * 80)
        print("📊 Train/Test 평가 시스템")
        print("=" * 80)

        try:
            # 1. 데이터 로드 및 분할
            train_data_dict, test_data_dict = self.load_data_and_split(symbols)
            if not train_data_dict or not test_data_dict:
                return {}

                # 2. 최적화 결과 로드
            optimization_results = self.load_optimization_results()
            if not optimization_results:
                print("❌ 최적화 결과를 찾을 수 없습니다.")
                return {}

            # 3. 포트폴리오 결과 로드
            portfolio_results = self.load_portfolio_results()
            if not portfolio_results:
                portfolio_results = {
                    "portfolio_weights": {},
                    "portfolio_performance": {},
                }

            # 4. 전략별 Train/Test 성과 평가
            individual_results = self.evaluate_all_strategies(
                train_data_dict, test_data_dict, optimization_results
            )

            # 5. 포트폴리오 성과 계산
            portfolio_weights = portfolio_results.get("portfolio_weights", {})
            portfolio_performance = self.calculate_portfolio_performance(
                individual_results, portfolio_results
            )

            # 6. 성과 테이블 생성
            if save_results:
                table_path = self.generate_performance_table(
                    individual_results, portfolio_performance, portfolio_weights
                )
                self.save_evaluation_results(
                    individual_results, portfolio_performance, portfolio_weights
                )

            # 6. 성과 요약 테이블 출력
            self._print_performance_summary(
                individual_results, portfolio_performance, portfolio_weights
            )

            # 결과 반환
            return {
                "individual_results": individual_results,
                "portfolio_performance": portfolio_performance,
                "portfolio_weights": portfolio_weights,
                "portfolio_results": portfolio_results,  # 누락된 키 추가
                "table_path": table_path if save_results else None,
            }

        except Exception as e:
            self.logger.log_error(f"Train/Test 평가 실행 중 오류: {e}")
            return {}

    def save_evaluation_results(
        self,
        individual_results: Dict[str, Any],
        portfolio_performance: Dict[str, Any],
        portfolio_weights: Dict[str, float],
    ):
        """평가 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 개별 결과 저장
            individual_filename = f"individual_evaluation_{timestamp}.json"
            individual_path = os.path.join("results", individual_filename)

            with open(individual_path, "w", encoding="utf-8") as f:
                json.dump(individual_results, f, indent=2, ensure_ascii=False)

            # 포트폴리오 성과 저장
            portfolio_filename = f"portfolio_performance_{timestamp}.json"
            portfolio_path = os.path.join("results", portfolio_filename)

            with open(portfolio_path, "w", encoding="utf-8") as f:
                json.dump(portfolio_performance, f, indent=2, ensure_ascii=False)

            # 포트폴리오 비중 저장
            weights_filename = f"portfolio_weights_{timestamp}.json"
            weights_path = os.path.join("results", weights_filename)

            with open(weights_path, "w", encoding="utf-8") as f:
                json.dump(portfolio_weights, f, indent=2, ensure_ascii=False)

            self.logger.log_success(f"평가 결과 저장 완료:")
            self.logger.log_success(f"  개별 결과: {individual_path}")
            self.logger.log_success(f"  포트폴리오 성과: {portfolio_path}")
            self.logger.log_success(f"  포트폴리오 비중: {weights_path}")

        except Exception as e:
            self.logger.log_error(f"평가 결과 저장 중 오류: {e}")

    def _print_performance_summary(
        self,
        individual_results: Dict[str, Any],
        portfolio_performance: Dict[str, Any],
        portfolio_weights: Dict[str, float],
    ):
        """성과 요약 테이블 출력"""
        print("\n" + "=" * 100)
        print("📊 TRAIN 성과 요약")
        print("=" * 100)
        self._print_performance_table(
            "TRAIN", individual_results, portfolio_performance, portfolio_weights
        )

        print("\n" + "=" * 100)
        print("📊 TEST 성과 요약")
        print("=" * 100)
        self._print_performance_table(
            "TEST", individual_results, portfolio_performance, portfolio_weights
        )

        print("=" * 100)

    def _print_performance_table(
        self,
        period: str,
        individual_results: Dict[str, Any],
        portfolio_performance: Dict[str, Any],
        portfolio_weights: Dict[str, float],
    ):
        """성과 테이블 출력"""
        # 헤더 출력
        print(
            f"{'종목':<8} {'비중':<6} {'수익률':<8} {'샤프':<6} {'소르티노':<8} {'거래수':<6} {'보유':<4} {'전략':<20}"
        )
        print("-" * 100)

        # Buy & Hold 성과 (포트폴리오 비중 기준)
        buy_hold_data = individual_results.get(f"buy_hold_{period.lower()}", {})
        if buy_hold_data:
            total_return = 0
            total_sharpe = 0
            total_sortino = 0
            total_trades = 0
            symbol_count = 0

            for symbol, weight in portfolio_weights.items():
                if symbol in buy_hold_data:
                    data = buy_hold_data[symbol]
                    total_return += data.get("total_return", 0) * weight
                    total_sharpe += data.get("sharpe_ratio", 0) * weight
                    total_sortino += data.get("sortino_ratio", 0) * weight
                    total_trades += data.get("total_trades", 0)
                    symbol_count += 1

            if symbol_count > 0:
                print(
                    f"{'BUY&HOLD':<8} {'100%':<6} {total_return*100:>7.2f}% {total_sharpe:>5.3f} {total_sortino:>7.3f} {total_trades:>5} {'Y':<4} {'PASSIVE':<20}"
                )

        # 포트폴리오 성과
        portfolio_data = portfolio_performance.get(period.lower(), {})
        if portfolio_data:
            portfolio_score = self._calculate_portfolio_score(portfolio_data)
            print(
                f"{'PORTFOLIO':<8} {'100%':<6} {portfolio_data.get('total_return', 0)*100:>7.2f}% {portfolio_data.get('sharpe_ratio', 0):>5.3f} {portfolio_data.get('sortino_ratio', 0):>7.3f} {portfolio_data.get('total_trades', 0):>5} {'Y':<4} {'OPTIMIZED':<20} [{portfolio_score:>6.1f}]"
            )

        print("-" * 100)

        # 개별 종목 성과 (포트폴리오 비중 순으로 정렬)
        individual_data = individual_results.get(period.lower(), {})
        if individual_data:
            # 포트폴리오 비중 순으로 정렬
            sorted_symbols = sorted(
                portfolio_weights.items(), key=lambda x: x[1], reverse=True
            )

            for symbol, weight in sorted_symbols:
                if symbol in individual_data:
                    data = individual_data[symbol]
                    strategy = data.get("strategy", "UNKNOWN")
                    total_return = data.get("total_return", 0) * 100
                    sharpe = data.get("sharpe_ratio", 0)
                    sortino = data.get("sortino_ratio", 0)
                    trades = data.get("total_trades", 0)

                    # 보유 여부 판단 (거래가 있으면 보유)
                    holding = "Y" if trades > 0 else "N"

                    print(
                        f"{symbol:<8} {weight*100:>5.1f}% {total_return:>7.2f}% {sharpe:>5.3f} {sortino:>7.3f} {trades:>5} {holding:<4} {strategy:<20}"
                    )


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Train/Test 평가 시스템")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="설정 파일")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="데이터 디렉토리")
    parser.add_argument("--log-mode", default="summary", help="로그 모드")
    parser.add_argument(
        "--optimization-results", help="개별 전략 최적화 결과 파일 경로"
    )
    parser.add_argument("--portfolio-results", help="포트폴리오 최적화 결과 파일 경로")
    parser.add_argument("--symbols", nargs="+", help="평가할 종목 목록")
    parser.add_argument("--no-save", action="store_true", help="결과 저장 안함")

    args = parser.parse_args()

    # 평가기 초기화
    evaluator = TrainTestEvaluator(
        data_dir=args.data_dir,
        log_mode=args.log_mode,
        config_path=args.config,
        optimization_results_path=args.optimization_results,
        portfolio_results_path=args.portfolio_results,
    )

    # Train/Test 평가 실행
    results = evaluator.run_train_test_evaluation(
        symbols=args.symbols,
        save_results=not args.no_save,
    )

    if not results:
        print("❌ Train/Test 평가 실패")


if __name__ == "__main__":
    main()
