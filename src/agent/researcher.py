#!/usr/bin/env python3
"""
개별 전략 연구 및 최적화 시스템
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.agent.evaluator import TrainTestEvaluator
from src.actions.log_pl import TradingSimulator
from src.actions.strategies import *
from src.agent.helper import (
    load_and_preprocess_data,
    split_data_train_test,
    load_config,
    Logger,
)

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OptimizationResult:
    """최적화 결과 클래스"""

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        best_params: Dict[str, Any],
        best_score: float,
        optimization_method: str,
        execution_time: float,
        n_combinations_tested: int,
        all_results: List[Dict[str, Any]],
    ):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.best_params = best_params
        self.best_score = best_score
        self.optimization_method = optimization_method
        self.execution_time = execution_time
        self.n_combinations_tested = n_combinations_tested
        self.all_results = all_results


class IndividualStrategyResearcher:
    """개별 종목별 전략 최적화 연구자"""

    def __init__(
        self,
        research_config_path: str = "config/config_research.json",
        source_config_path: str = "config/config_swing.json",  # swing config를 기본값으로 설정
        data_dir: str = "data",
        results_dir: str = "results",
        log_dir: str = "log",
        analysis_dir: Optional[str] = None,
        auto_detect_source_config: bool = False,  # 자동 감지 비활성화
        uuid: Optional[str] = None,
    ):
        self.research_config_path = research_config_path
        self.source_config_path = source_config_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.log_dir = log_dir
        self.analysis_dir = analysis_dir
        self.uuid = uuid

        # 설정 로드
        self.research_config = self._load_research_config(research_config_path)
        self.source_config = load_config(source_config_path)

        # 자동 감지 및 설정 (orchestrator에서 호출할 때는 비활성화)
        if auto_detect_source_config:
            self._auto_detect_and_set_source_config()

        # 로거 설정
        self.logger = Logger()
        if log_dir:
            self.logger.set_log_dir(log_dir)

        # 평가기 초기화 (단일 종목 모드)
        self.evaluator = TrainTestEvaluator(
            data_dir=self.data_dir,
            log_mode="summary",
            config_path=self.source_config_path,
        )

        # 로거 설정
        if log_dir:
            self.evaluator.logger.set_log_dir(log_dir)

        self.strategy_manager = StrategyManager()

        # 전략 등록
        self._register_strategies()

        # 연구 결과 저장
        self.research_results = {}
        self.start_time = datetime.now()

    def _load_research_config(self, config_path: str) -> Dict[str, Any]:
        """연구 설정 파일 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"연구 설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"연구 설정 파일 형식이 잘못되었습니다: {config_path}")
            return {}

    def _auto_detect_and_set_source_config(self):
        """자동으로 최적의 source config 감지 및 설정"""
        logger.info("🔍 자동 source config 감지 중...")

        # 사용 가능한 config 파일들 찾기
        config_dir = Path("config")
        available_configs = []

        for config_file in config_dir.glob("config_*.json"):
            if config_file.name != "config_research.json":
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config_data = json.load(f)

                    # 기본 정보 추출
                    config_info = {
                        "name": config_file.name,
                        "path": str(config_file),
                        "time_horizon": config_data.get("time_horizon", "unknown"),
                        "symbol_count": len(
                            config_data.get("data", {}).get("symbols", [])
                        ),
                        "strategy_count": len(config_data.get("strategies", [])),
                        "portfolio_mode": False,  # 단일 종목 모드로 변경
                    }
                    available_configs.append(config_info)

                except Exception as e:
                    logger.warning(f"Config 파일 로드 실패: {config_file} - {e}")

        if not available_configs:
            logger.warning("사용 가능한 config 파일이 없습니다. 기본 설정 사용")
            return

        # 최적의 config 선택
        best_config = self._select_best_source_config(available_configs)
        if best_config:
            self.source_config_path = best_config["path"]
            self.source_config = load_config(self.source_config_path)
            logger.info(f"✅ 선택된 source config: {best_config['name']}")
        else:
            logger.warning("적절한 config를 찾을 수 없습니다. 기본 설정 사용")

    def _select_best_source_config(
        self, available_configs: List[Dict]
    ) -> Optional[Dict]:
        """최적의 source config 선택"""
        if not available_configs:
            return None

        def config_score(config):
            score = 0

            # 심볼 수 점수 (최대 50점)
            symbol_score = min(config["symbol_count"] * 10, 50)
            score += symbol_score

            # time_horizon 점수 (최대 30점) - swing을 우선으로 설정
            horizon = config["time_horizon"].lower()
            if "swing" in horizon:
                score += 30  # swing을 최고 점수로 설정
            elif "long" in horizon:
                score += 20
            elif "scalping" in horizon:
                score += 15
            else:
                score += 10

            # 전략 수 점수 (최대 20점)
            strategy_score = min(config["strategy_count"] * 2, 20)
            score += strategy_score

            return score

        # 점수로 정렬
        sorted_configs = sorted(available_configs, key=config_score, reverse=True)

        logger.info("📊 Config 파일 우선순위:")
        for i, config in enumerate(sorted_configs[:3], 1):
            logger.info(
                f"  {i}. {config['name']} (심볼: {config['symbol_count']}, "
                f"전략: {config['strategy_count']}, 시간대: {config['time_horizon']})"
            )

        return sorted_configs[0] if sorted_configs else None

    def _load_source_config_symbols(self) -> List[str]:
        """source config에서 심볼 목록 로드"""
        symbols = self.source_config.get("data", {}).get("symbols", [])
        logger.info(f"🔍 로드된 심볼들: {symbols}")
        logger.info(f"🔍 source_config_path: {self.source_config_path}")
        return symbols

    def _load_source_config_settings(self) -> Dict[str, Any]:
        """source config에서 설정 로드"""
        return self.source_config.get("data", {})

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
            "trend_following_ma200": TrendFollowingMA200Strategy,
            "swing_breakout": SwingBreakoutStrategy,
            "swing_pullback_entry": SwingPullbackEntryStrategy,
            "swing_candle_pattern": SwingCandlePatternStrategy,
            "swing_bollinger_band": SwingBollingerBandStrategy,
            "swing_macd": SwingMACDStrategy,
        }

        for name, strategy_class in strategies_to_register.items():
            self.strategy_manager.add_strategy(name, strategy_class(StrategyParams()))

        logger.info(f"✅ {len(strategies_to_register)}개 전략 등록 완료")

    def create_evaluation_function(
        self,
        strategy_name: str,
        data_dict: Dict[str, pd.DataFrame],
        symbol: str,
    ):
        """평가 함수 생성 (단일 종목용)"""

        def evaluation_function(params: Dict[str, Any]) -> float:
            try:
                # 전략 인스턴스 생성
                strategy = self.strategy_manager.strategies.get(strategy_name)
                if not strategy:
                    logger.error(f"전략을 찾을 수 없습니다: {strategy_name}")
                    return -999999.0

                # 파라미터 설정
                for param_name, param_value in params.items():
                    if hasattr(strategy, param_name):
                        setattr(strategy, param_name, param_value)

                # 단일 종목 데이터로 전략 실행
                symbol_data = data_dict[symbol]
                signals = strategy.generate_signals(symbol_data)

                if signals is None or signals.empty:
                    logger.error(f"시그널 생성 실패: {strategy_name} - {symbol}")
                    return -999999.0

                # 디버그: 시그널 통계 확인 (간단하게)
                signal_counts = signals["signal"].value_counts()
                if len(signal_counts) == 1 and 0 in signal_counts:
                    logger.info(f"모든 시그널이 0입니다: {strategy_name} - {symbol}")
                    return -999999.0

                # 거래 시뮬레이션
                simulator = TradingSimulator(self.source_config_path)
                simulation_result = simulator.simulate_trading(
                    symbol_data, signals, strategy_name
                )

                if not simulation_result:
                    logger.error(f"거래 시뮬레이션 실패: {strategy_name} - {symbol}")
                    return -999999.0

                # TradingSimulator 결과에서 performance metrics 추출
                strategy_result = simulation_result.get("results", {})
                trades = simulation_result.get("trades", [])

                # 거래가 없는 경우 체크
                if not trades:
                    logger.info(f"거래가 없음: {strategy_name} - {symbol}")
                    return -999999.0

                # 추가 정보 추가
                strategy_result["trades"] = trades

                # 복합 점수 계산
                composite_score = self._calculate_composite_score(strategy_result)

                # 디버깅: 점수가 -999999인 경우만 로그 출력
                if composite_score == -999999.0:
                    logger.debug(f"복합 점수가 -999999: {strategy_name} - {symbol}")

                return composite_score

            except Exception as e:
                logger.error(f"평가 함수 실행 중 오류: {e}")
                return -999999.0

        return evaluation_function

    def _calculate_composite_score(self, strategy_result) -> float:
        """복합 점수 계산 (단일 종목용)"""
        try:
            # 기본 지표들
            total_return = strategy_result.get("total_return", 0)
            sharpe_ratio = strategy_result.get("sharpe_ratio", 0)
            max_drawdown = strategy_result.get("max_drawdown", 1)
            win_rate = strategy_result.get("win_rate", 0)
            profit_factor = strategy_result.get("profit_factor", 0)
            total_trades = strategy_result.get("total_trades", 0)
            trades = strategy_result.get("trades", [])

            # 거래가 없는 경우 체크
            if not trades or total_trades == 0:
                return -999999.0

            # 추가 지표들
            sortino_ratio = self._calculate_sortino_ratio(strategy_result)
            calmar_ratio = self._calculate_calmar_ratio(strategy_result)

            # 성과 기준 체크 (설정 파일에서 로드)
            performance_thresholds = self.source_config.get("researcher", {}).get(
                "performance_thresholds", {}
            )
            min_return_threshold = performance_thresholds.get(
                "min_return_threshold", 0.0
            )
            min_sharpe_ratio = performance_thresholds.get("min_sharpe_ratio", -1.0)
            min_profit_factor = performance_thresholds.get("min_profit_factor", 0.0)
            min_win_rate = performance_thresholds.get("min_win_rate", 0.0)
            min_trades = performance_thresholds.get("min_trades", 1)

            # 거래가 없는 경우 체크 (설정 파일에서 로드)
            if not trades or total_trades < min_trades:
                return -999999.0

            # 최소 기준 체크
            if total_return < min_return_threshold:
                return -999999.0

            if sharpe_ratio < min_sharpe_ratio:
                return -999999.0

            if profit_factor < min_profit_factor:
                return -999999.0

            if win_rate < min_win_rate:
                return -999999.0

            # config에서 가중치 설정 로드
            evaluation_metrics = self.source_config.get("researcher", {}).get(
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

            # total_return과 max_drawdown은 별도 처리
            weights["total_return"] = 0.30  # 수익률 가중치 증가
            weights["max_drawdown"] = 0.10  # 낙폭 가중치 증가

            # 점수 계산
            scores = {}

            # 수익률 점수 (음수 수익률에 페널티 적용)
            if total_return >= 0:
                # 양수 수익률: 0-100점
                scores["total_return"] = min(total_return * 100, 100)
            else:
                # 음수 수익률: 페널티 적용 (최대 -50점)
                penalty_score = max(total_return * 100, -50)
                scores["total_return"] = penalty_score

            # 샤프 비율 점수 (0-100)
            scores["sharpe_ratio"] = min(max(sharpe_ratio * 20, 0), 100)

            # 소르티노 비율 점수 (0-100)
            scores["sortino_ratio"] = min(max(sortino_ratio * 20, 0), 100)

            # 칼마 비율 점수 (0-100)
            scores["calmar_ratio"] = min(max(calmar_ratio * 10, 0), 100)

            # 수익 팩터 점수 (0-100)
            scores["profit_factor"] = min(max(profit_factor * 20, 0), 100)

            # 승률 점수 (0-100)
            scores["win_rate"] = min(max(win_rate * 100, 0), 100)

            # 최대 낙폭 점수 (낮을수록 높은 점수)
            scores["max_drawdown"] = max(0, 100 - (max_drawdown * 100))

            # 복합 점수 계산
            composite_score = sum(
                scores[metric] * weight for metric, weight in weights.items()
            )

            # 위험 페널티 적용 (researcher config에서 로드)
            risk_penalties = self.source_config.get("researcher", {}).get(
                "risk_penalties", {}
            )
            max_drawdown_threshold = risk_penalties.get("max_drawdown_threshold", 0.20)
            max_drawdown_penalty = risk_penalties.get("max_drawdown_penalty", 0.5)
            volatility_threshold = risk_penalties.get("volatility_threshold", 0.30)
            volatility_penalty = risk_penalties.get("volatility_penalty", 0.3)

            # 최대 낙폭 페널티 (더 엄격하게)
            if max_drawdown > max_drawdown_threshold:
                composite_score *= 1 - max_drawdown_penalty

            # 변동성 페널티
            volatility = strategy_result.get("volatility", 0)
            if volatility > volatility_threshold:
                composite_score *= 1 - volatility_penalty

            # 수익률 페널티 (음수 수익률에 추가 페널티)
            if total_return < 0:
                return_penalty = abs(total_return) * 0.5  # 수익률 절댓값의 50% 페널티
                composite_score *= 1 - return_penalty

            return composite_score

        except Exception as e:
            logger.error(f"복합 점수 계산 중 오류: {e}")
            return -999999.0

    def _calculate_sortino_ratio(self, strategy_result) -> float:
        """소르티노 비율 계산"""
        try:
            trades = strategy_result.get("trades", [])
            if not trades:
                return 0

            # 거래별 수익률 계산
            returns = []
            for trade in trades:
                pnl = trade.get("pnl", 0)
                returns.append(pnl)

            if not returns:
                return 0

            returns_array = np.array(returns)
            negative_returns = returns_array[returns_array < 0]

            if len(negative_returns) == 0:
                return 0

            downside_deviation = np.std(negative_returns)
            if downside_deviation == 0:
                return 0

            mean_return = np.mean(returns_array)
            risk_free_rate = 0.02 / 252  # 일간 무위험 수익률

            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
            return sortino_ratio

        except Exception as e:
            logger.error(f"소르티노 비율 계산 중 오류: {e}")
            return 0

    def _calculate_calmar_ratio(self, strategy_result) -> float:
        """칼마 비율 계산"""
        try:
            total_return = strategy_result.get("total_return", 0)
            max_drawdown = strategy_result.get("max_drawdown", 1)

            if max_drawdown == 0:
                return 0

            calmar_ratio = total_return / max_drawdown
            return calmar_ratio

        except Exception as e:
            logger.error(f"칼마 비율 계산 중 오류: {e}")
            return 0

    def optimize_single_strategy_for_symbol(
        self,
        strategy_name: str,
        symbol: str,
        optimization_method: str = None,  # None이면 config에서 로드
    ) -> Optional[OptimizationResult]:
        """단일 전략을 단일 종목에 대해 최적화"""
        logger.info(f"🔬 {symbol} - {strategy_name} 최적화 시작")

        start_time = datetime.now()

        try:
            # 최적화 방법 설정 (config에서 로드)
            if optimization_method is None:
                optimization_method = self.source_config.get("researcher", {}).get(
                    "optimization_method", "bayesian_optimization"
                )
            logger.info(f"🔧 최적화 방법: {optimization_method}")

            # 데이터 로드
            data_dict = load_and_preprocess_data(self.data_dir, [symbol])
            if not data_dict or symbol not in data_dict:
                logger.error(f"{symbol} 데이터를 로드할 수 없습니다")
                return None

            # 평가 함수 생성
            evaluation_function = self.create_evaluation_function(
                strategy_name, data_dict, symbol
            )

            # 파라미터 범위 설정 (research_config에서 로드)
            param_ranges = (
                self.research_config.get("strategies", {})
                .get(strategy_name, {})
                .get("param_ranges", {})
            )

            logger.info(f"🔍 {strategy_name} 파라미터 범위 로드: {param_ranges}")
            logger.info(f"🔍 파라미터 개수: {len(param_ranges)}")

            # 최적화 설정 (source_config에서 로드)
            settings = self.source_config.get("researcher", {}).get(
                "optimization_settings", {}
            )

            # 최적화 실행
            if optimization_method == "grid_search":
                best_result = self._grid_search_optimization(
                    evaluation_function, param_ranges, settings
                )
            elif optimization_method == "bayesian_optimization":
                best_result = self._bayesian_optimization(
                    evaluation_function, param_ranges, settings
                )
            elif optimization_method == "genetic_algorithm":
                best_result = self._genetic_algorithm_optimization(
                    evaluation_function, param_ranges, settings
                )
            else:
                logger.error(f"지원하지 않는 최적화 방법: {optimization_method}")
                return None

            if not best_result:
                logger.warning(f"{symbol} - {strategy_name} 최적화 실패")
                return None

            execution_time = (datetime.now() - start_time).total_seconds()

            # 결과 생성
            result = OptimizationResult(
                strategy_name=strategy_name,
                symbol=symbol,
                best_params=best_result["params"],
                best_score=best_result["score"],
                optimization_method=optimization_method,
                execution_time=execution_time,
                n_combinations_tested=best_result.get("n_combinations", 0),
                all_results=best_result.get("all_results", []),
            )

            logger.info(
                f"✅ {symbol} - {strategy_name} 최적화 완료 "
                f"(점수: {best_result['score']:.2f}, 시간: {execution_time:.1f}초)"
            )

            return result

        except Exception as e:
            logger.error(f"{symbol} - {strategy_name} 최적화 중 오류: {e}")
            return None

    def _grid_search_optimization(
        self, evaluation_function, param_ranges: Dict, settings: Dict
    ) -> Optional[Dict]:
        """그리드 서치 최적화"""
        try:
            from itertools import product

            # 파라미터 조합 생성
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())

            all_combinations = list(product(*param_values))
            max_combinations = settings.get("max_combinations", 1000)

            if len(all_combinations) > max_combinations:
                logger.warning(
                    f"조합 수가 너무 많습니다 ({len(all_combinations)}). "
                    f"처음 {max_combinations}개만 테스트합니다."
                )
                all_combinations = all_combinations[:max_combinations]

            best_score = -999999.0
            best_params = {}
            # all_results는 상위 10개만 저장 (메모리 절약)
            top_results = []

            logger.info(f"🚀 그리드 서치 최적화 시작: {len(all_combinations)}개 조합")

            for i, combination in enumerate(all_combinations):
                params = dict(zip(param_names, combination))
                score = evaluation_function(params)

                # 상위 10개 결과만 저장
                if len(top_results) < 10:
                    top_results.append({"params": params, "score": score})
                    top_results.sort(key=lambda x: x["score"], reverse=True)
                elif score > top_results[-1]["score"]:
                    top_results.append({"params": params, "score": score})
                    top_results.sort(key=lambda x: x["score"], reverse=True)
                    top_results = top_results[:10]  # 상위 10개만 유지

                # 새로운 최고 점수 발견 시 로그 출력
                if score > best_score and score > -999999.0:
                    best_score = score
                    best_params = params
                    progress = (i + 1) / len(all_combinations) * 100
                    logger.info(
                        f"🎯 조합 {i+1}/{len(all_combinations)}: 새로운 최고 점수 {score:.2f} (진행률: {progress:.1f}%)"
                    )
                    logger.info(f"   최적 파라미터: {best_params}")

                # 진행률 로그를 10% 단위로만 출력
                if (i + 1) % max(1, len(all_combinations) // 10) == 0:
                    progress = (i + 1) / len(all_combinations) * 100
                    logger.info(
                        f"📊 진행률: {progress:.0f}% ({i+1}/{len(all_combinations)})"
                    )

            # 최적화 결과 요약
            logger.info(f"✅ 그리드 서치 최적화 완료: {len(all_combinations)} 조합")
            logger.info(f"🏆 최종 최고 점수: {best_score:.2f}")
            if best_score > -999999.0:
                logger.info(f"🎯 최적 파라미터: {best_params}")
            else:
                logger.warning("⚠️ 유효한 최적화 결과를 찾지 못했습니다")

            return {
                "params": best_params,
                "score": best_score,
                "n_combinations": len(all_combinations),
                "all_results": top_results,  # 상위 10개만 반환
            }

        except Exception as e:
            logger.error(f"그리드 서치 최적화 중 오류: {e}")
            return None

    def _bayesian_optimization(
        self, evaluation_function, param_ranges: Dict, settings: Dict
    ) -> Optional[Dict]:
        """Optuna를 사용한 베이지안 최적화"""
        try:
            import optuna

            # 파라미터 범위는 한 번만 출력 (디버그용)
            if len(param_ranges) <= 5:  # 파라미터가 적을 때만 출력
                logger.info(f"🔍 베이지안 최적화 파라미터 개수: {len(param_ranges)}")

            best_score_so_far = -999999.0
            best_params_so_far = {}

            def objective(trial):
                nonlocal best_score_so_far, best_params_so_far

                # 파라미터 샘플링
                params = {}
                for param_name, param_range in param_ranges.items():
                    if isinstance(param_range[0], bool):
                        # Boolean 파라미터 처리
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_range
                        )
                    elif isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, param_range[0], param_range[-1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_range[0], param_range[-1]
                        )

                # 평가 함수 실행
                score = evaluation_function(params)

                # 최고 점수 업데이트 및 로그 출력
                if score > best_score_so_far and score > -999999.0:
                    best_score_so_far = score
                    best_params_so_far = params.copy()

                    # 현재 trial 번호와 진행률 계산
                    current_trial = trial.number + 1
                    total_trials = (
                        trial.study.n_trials
                        if hasattr(trial.study, "n_trials")
                        else "unknown"
                    )
                    progress = (
                        (current_trial / trial.study.n_trials * 100)
                        if hasattr(trial.study, "n_trials")
                        else 0
                    )

                    logger.info(
                        f"🎯 Trial {current_trial}: 새로운 최고 점수 {score:.2f} (진행률: {progress:.1f}%)"
                    )
                    logger.info(f"   최적 파라미터: {best_params_so_far}")

                return score

            # 최적화 실행
            bayesian_settings = settings.get("bayesian_optimization", {})
            n_trials = bayesian_settings.get("n_trials", 50)
            n_startup_trials = bayesian_settings.get("n_startup_trials", 5)
            early_stopping_patience = bayesian_settings.get(
                "early_stopping_patience", 10
            )

            logger.info(f"🚀 베이지안 최적화 시작: {n_trials} trials")

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_score = study.best_value

            # 최적화 결과 요약
            logger.info(f"✅ 베이지안 최적화 완료: {n_trials} trials")
            logger.info(f"🏆 최종 최고 점수: {best_score:.2f}")
            if best_score > -999999.0:
                logger.info(f"🎯 최적 파라미터: {best_params}")
            else:
                logger.warning("⚠️ 유효한 최적화 결과를 찾지 못했습니다")

            return {
                "params": best_params,
                "score": best_score,
                "n_combinations": n_trials,
                "all_results": [],  # 베이지안 최적화는 all_results 없음
            }

        except ImportError:
            logger.warning("optuna가 설치되지 않았습니다. 그리드 서치를 사용합니다.")
            return self._grid_search_optimization(
                evaluation_function, param_ranges, settings
            )
        except Exception as e:
            logger.error(f"베이지안 최적화 중 오류: {e}")
            return None

    def _genetic_algorithm_optimization(
        self, evaluation_function, param_ranges: Dict, settings: Dict
    ) -> Optional[Dict]:
        """유전 알고리즘 최적화"""
        try:
            import random

            # 유전 알고리즘 설정
            population_size = settings.get("population_size", 50)
            generations = settings.get("generations", 30)
            mutation_rate = settings.get("mutation_rate", 0.1)
            crossover_rate = settings.get("crossover_rate", 0.8)

            logger.info(
                f"🚀 유전 알고리즘 최적화 시작: {generations}세대, {population_size}개체"
            )

            def create_individual():
                """개체 생성"""
                individual = {}
                for param_name, param_range in param_ranges.items():
                    if isinstance(param_range[0], int):
                        individual[param_name] = random.randint(
                            param_range[0], param_range[-1]
                        )
                    else:
                        individual[param_name] = random.uniform(
                            param_range[0], param_range[-1]
                        )
                return individual

            def evaluate(individual):
                """개체 평가"""
                return evaluation_function(individual)

            def crossover(parent1, parent2):
                """교차"""
                if random.random() > crossover_rate:
                    return parent1, parent2

                child1, child2 = {}, {}
                for param_name in param_ranges.keys():
                    if random.random() < 0.5:
                        child1[param_name] = parent1[param_name]
                        child2[param_name] = parent2[param_name]
                    else:
                        child1[param_name] = parent2[param_name]
                        child2[param_name] = parent1[param_name]

                return child1, child2

            def mutate(individual):
                """돌연변이"""
                for param_name, param_range in param_ranges.items():
                    if random.random() < mutation_rate:
                        if isinstance(param_range[0], int):
                            individual[param_name] = random.randint(
                                param_range[0], param_range[-1]
                            )
                        else:
                            individual[param_name] = random.uniform(
                                param_range[0], param_range[-1]
                            )
                return individual

            # 초기 개체군 생성
            population = [create_individual() for _ in range(population_size)]

            best_individual = None
            best_score = -999999.0

            # 진화
            for generation in range(generations):
                # 평가
                fitness_scores = [
                    (evaluate(individual), individual) for individual in population
                ]
                fitness_scores.sort(reverse=True)

                # 최고 개체 업데이트 및 로그 출력
                current_best_score = fitness_scores[0][0]
                if current_best_score > best_score and current_best_score > -999999.0:
                    best_score = current_best_score
                    best_individual = fitness_scores[0][1]

                    progress = (generation + 1) / generations * 100
                    logger.info(
                        f"🎯 세대 {generation+1}/{generations}: 새로운 최고 점수 {best_score:.2f} (진행률: {progress:.1f}%)"
                    )
                    logger.info(f"   최적 파라미터: {best_individual}")

                # 새로운 개체군 생성
                new_population = fitness_scores[: population_size // 2]  # 상위 50% 유지

                # 나머지는 교차와 돌연변이로 생성
                while len(new_population) < population_size:
                    parent1 = random.choice(fitness_scores[: population_size // 2])[1]
                    parent2 = random.choice(fitness_scores[: population_size // 2])[1]

                    child1, child2 = crossover(parent1, parent2)
                    child1 = mutate(child1)
                    child2 = mutate(child2)

                    new_population.extend([child1, child2])

                population = [
                    individual for _, individual in new_population[:population_size]
                ]

                # 5세대마다 진행상황 출력
                if generation % 5 == 0:
                    progress = (generation + 1) / generations * 100
                    logger.info(
                        f"📊 세대 {generation+1}/{generations}: 현재 최고 점수 = {best_score:.2f} (진행률: {progress:.1f}%)"
                    )

            # 최적화 결과 요약
            logger.info(f"✅ 유전 알고리즘 최적화 완료: {generations}세대")
            logger.info(f"🏆 최종 최고 점수: {best_score:.2f}")
            if best_score > -999999.0:
                logger.info(f"🎯 최적 파라미터: {best_individual}")
            else:
                logger.warning("⚠️ 유효한 최적화 결과를 찾지 못했습니다")

            return {
                "params": best_individual,
                "score": best_score,
                "n_combinations": population_size * generations,
                "all_results": [],
            }

        except Exception as e:
            logger.error(f"유전 알고리즘 최적화 중 오류: {e}")
            return None

    def run_comprehensive_research(
        self,
        strategies: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        optimization_method: str = None,  # config에서 로드
        use_train_data_only: bool = True,
    ) -> Dict[str, OptimizationResult]:
        """종합 연구 실행"""
        logger.info("🚀 종합 연구 시작")

        # 전략과 심볼 설정
        if not strategies:
            strategies = list(self.research_config.get("strategies", {}).keys())
        if not symbols:
            symbols = self._load_source_config_symbols()

        logger.info(f"📊 대상 전략: {len(strategies)}개")
        logger.info(f"📈 대상 심볼: {len(symbols)}개")

        # 최적화 방법 설정 (config에서 로드)
        if optimization_method is None:
            optimization_method = self.source_config.get("researcher", {}).get(
                "optimization_method", "bayesian_optimization"
            )
        logger.info(f"🔧 최적화 방법: {optimization_method}")

        # 데이터 로드 및 Train/Test 분할
        data_dict = load_and_preprocess_data(self.data_dir, symbols)
        if not data_dict:
            logger.error(f"데이터를 로드할 수 없습니다: {self.data_dir}")
            return {}

        # Train/Test 분할 (train 데이터만 사용)
        if use_train_data_only:
            train_ratio = self.source_config.get("data", {}).get("train_ratio", 0.8)
            train_data_dict, test_data_dict = split_data_train_test(
                data_dict, train_ratio
            )
            data_dict = train_data_dict  # train 데이터만 사용
            logger.info(
                f"Train/Test 분할 완료: Train {len(train_data_dict)}개 종목, Test {len(test_data_dict)}개 종목"
            )

        logger.info(f"데이터 로드 완료: {list(data_dict.keys())}")

        results = {}
        total_combinations = len(strategies) * len(symbols)
        current_combination = 0

        for strategy_name in strategies:
            for symbol in symbols:
                current_combination += 1
                logger.info(
                    f"🔬 진행률: {current_combination}/{total_combinations} "
                    f"({strategy_name} - {symbol})"
                )

                result = self.optimize_single_strategy_for_symbol(
                    strategy_name, symbol, optimization_method
                )

                if result:
                    key = f"{strategy_name}_{symbol}"
                    results[key] = result
                    logger.info(f"✅ {key}: 점수 {result.best_score:.2f}")
                else:
                    logger.warning(f"❌ {strategy_name} - {symbol}: 최적화 실패")

        logger.info(f"✅ 종합 연구 완료: {len(results)}개 조합 최적화됨")
        return results

    def save_research_results(self, results: Dict[str, OptimizationResult]):
        """연구 결과 저장"""
        try:
            # 날짜와 UUID로 파일명 생성
            current_date = datetime.now().strftime("%Y%m%d")
            if hasattr(self, "uuid") and self.uuid:
                filename = f"hyperparam_optimization_{current_date}_{self.uuid}.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hyperparam_optimization_{timestamp}.json"

            # 결과를 JSON 직렬화 가능한 형태로 변환
            serializable_results = {}
            for key, result in results.items():
                serializable_results[key] = {
                    "strategy_name": result.strategy_name,
                    "symbol": result.symbol,
                    "best_params": result.best_params,
                    "best_score": result.best_score,
                    "optimization_method": result.optimization_method,
                    "execution_time": result.execution_time,
                    "n_combinations_tested": result.n_combinations_tested,
                    "all_results": result.all_results,
                }

            # 파일 저장
            output_path = os.path.join(self.results_dir, filename)
            os.makedirs(self.results_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 연구 결과 저장 완료: {output_path}")

            # 최신 파일 경로 반환
            return output_path

        except Exception as e:
            logger.error(f"연구 결과 저장 중 오류: {e}")
            return None

    def generate_research_report(self, results: Dict[str, OptimizationResult]):
        """연구 보고서 생성"""
        try:
            # UUID가 있으면 사용, 없으면 현재 시간 사용
            if hasattr(self, "uuid") and self.uuid:
                timestamp = self.uuid
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.txt"
            output_path = os.path.join(self.results_dir, filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("개별 종목별 전략 최적화 연구 보고서\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"연구 시작 시간: {self.start_time}\n")
                f.write(f"연구 완료 시간: {datetime.now()}\n")
                f.write(f"총 조합 수: {len(results)}\n\n")

                # 전략별 요약
                strategy_summary = {}
                for key, result in results.items():
                    strategy_name = result.strategy_name
                    if strategy_name not in strategy_summary:
                        strategy_summary[strategy_name] = []
                    strategy_summary[strategy_name].append(result)

                f.write("전략별 최적화 결과 요약:\n")
                f.write("-" * 50 + "\n")
                for strategy_name, strategy_results in strategy_summary.items():
                    avg_score = np.mean([r.best_score for r in strategy_results])
                    best_score = max([r.best_score for r in strategy_results])
                    f.write(f"{strategy_name}:\n")
                    f.write(f"  평균 점수: {avg_score:.2f}\n")
                    f.write(f"  최고 점수: {best_score:.2f}\n")
                    f.write(f"  최적화된 종목 수: {len(strategy_results)}\n\n")

                # 상위 결과들
                f.write("상위 10개 최적화 결과:\n")
                f.write("-" * 50 + "\n")
                sorted_results = sorted(
                    results.items(), key=lambda x: x[1].best_score, reverse=True
                )
                for i, (key, result) in enumerate(sorted_results[:10], 1):
                    f.write(
                        f"{i}. {result.strategy_name} - {result.symbol}: "
                        f"점수 {result.best_score:.2f}\n"
                    )

            logger.info(f"📄 연구 보고서 생성 완료: {output_path}")

        except Exception as e:
            logger.error(f"연구 보고서 생성 중 오류: {e}")

    def run_quick_test(
        self, strategy_name: str = "dual_momentum", symbol: str = "AAPL"
    ):
        """빠른 테스트 실행"""
        logger.info(f"🧪 빠른 테스트: {strategy_name} - {symbol}")

        # config에서 최적화 방법 로드
        optimization_method = self.source_config.get("researcher", {}).get(
            "optimization_method", "bayesian_optimization"
        )
        logger.info(f"🔧 사용할 최적화 방법: {optimization_method}")

        result = self.optimize_single_strategy_for_symbol(
            strategy_name, symbol, optimization_method
        )

        if result:
            logger.info(f"✅ 테스트 성공: 점수 {result.best_score:.2f}")
        else:
            logger.error("❌ 테스트 실패")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="개별 종목별 전략 최적화 연구자")
    parser.add_argument(
        "--config", default="config/config_research.json", help="연구 설정 파일"
    )
    parser.add_argument(
        "--source-config", default="config/config_swing.json", help="소스 설정 파일"
    )
    parser.add_argument("--data-dir", default="data", help="데이터 디렉토리")
    parser.add_argument("--results-dir", default="results", help="결과 디렉토리")
    parser.add_argument("--log-dir", default="log", help="로그 디렉토리")
    parser.add_argument(
        "--optimization-method",
        choices=["grid_search", "bayesian_optimization", "genetic_algorithm"],
        help="최적화 방법 (기본값: config에서 로드)",
    )
    parser.add_argument("--quick-test", action="store_true", help="빠른 테스트 실행")

    args = parser.parse_args()

    # 연구자 초기화
    researcher = IndividualStrategyResearcher(
        research_config_path=args.config,
        source_config_path=args.source_config,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
    )

    if args.quick_test:
        researcher.run_quick_test()
    else:
        # 종합 연구 실행
        results = researcher.run_comprehensive_research(
            optimization_method=(
                args.optimization_method if args.optimization_method else None
            )
        )

        if results:
            # 결과 저장
            output_file = researcher.save_research_results(results)
            if output_file:
                logger.info(f"💾 결과 저장됨: {output_file}")

            # 보고서 생성
            researcher.generate_research_report(results)


if __name__ == "__main__":
    main()
