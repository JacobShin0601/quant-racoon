#!/usr/bin/env python3
"""
하이퍼파라미터 연구 및 최적화 시스템
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
import logging
import warnings
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavior.strategies import (
    StrategyManager,
    DualMomentumStrategy,
    VolatilityAdjustedBreakoutStrategy,
    SwingEMACrossoverStrategy,
    SwingRSIReversalStrategy,
    DonchianSwingBreakoutStrategy,
    StochasticStrategy,
    WilliamsRStrategy,
    CCIStrategy,
    WhipsawPreventionStrategy,
    DonchianRSIWhipsawStrategy,
    VolatilityFilteredBreakoutStrategy,
    MultiTimeframeWhipsawStrategy,
    AdaptiveWhipsawStrategy,
    CCIBollingerStrategy,
    StochDonchianStrategy,
    VWAPMACDScalpingStrategy,
    KeltnerRSIScalpingStrategy,
    AbsorptionScalpingStrategy,
    RSIBollingerScalpingStrategy,
)
from behavior.calculate_index import StrategyParams
from behavior.grid_search import HyperparameterOptimizer, OptimizationResult
from agent.evaluator import StrategyEvaluator
from agent.helper import (
    load_config,
    load_and_preprocess_data,
    print_section_header,
    print_subsection_header,
    format_percentage,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
)

warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterResearcher:
    """하이퍼파라미터 연구 및 최적화 클래스"""

    def __init__(
        self,
        research_config_path: str = "config_research.json",
        trading_config_path: str = DEFAULT_CONFIG_PATH,
        data_dir: str = DEFAULT_DATA_DIR,
        results_dir: str = "results",
        log_dir: str = "log",
    ):
        self.research_config = self._load_research_config(research_config_path)
        self.trading_config = load_config(trading_config_path)
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.log_dir = log_dir

        # 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 컴포넌트 초기화
        self.optimizer = HyperparameterOptimizer(research_config_path)
        self.evaluator = StrategyEvaluator(
            data_dir=data_dir,
            log_mode="summary",
            portfolio_mode=self.research_config.get("research_config", {}).get(
                "portfolio_mode", False
            ),
        )
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

    def _register_strategies(self):
        """모든 전략을 매니저에 등록"""
        params = StrategyParams()

        strategies = {
            "dual_momentum": DualMomentumStrategy(params),
            "volatility_breakout": VolatilityAdjustedBreakoutStrategy(params),
            "swing_ema": SwingEMACrossoverStrategy(params),
            "swing_rsi": SwingRSIReversalStrategy(params),
            "swing_donchian": DonchianSwingBreakoutStrategy(params),
            "stochastic": StochasticStrategy(params),
            "williams_r": WilliamsRStrategy(params),
            "cci": CCIStrategy(params),
            "whipsaw_prevention": WhipsawPreventionStrategy(params),
            "donchian_rsi_whipsaw": DonchianRSIWhipsawStrategy(params),
            "volatility_filtered_breakout": VolatilityFilteredBreakoutStrategy(params),
            "multi_timeframe_whipsaw": MultiTimeframeWhipsawStrategy(params),
            "adaptive_whipsaw": AdaptiveWhipsawStrategy(params),
            "cci_bollinger": CCIBollingerStrategy(params),
            "stoch_donchian": StochDonchianStrategy(params),
            "vwap_macd_scalping": VWAPMACDScalpingStrategy(params),
            "keltner_rsi_scalping": KeltnerRSIScalpingStrategy(params),
            "absorption_scalping": AbsorptionScalpingStrategy(params),
            "rsi_bollinger_scalping": RSIBollingerScalpingStrategy(params),
        }

        for name, strategy in strategies.items():
            self.strategy_manager.add_strategy(name, strategy)

    def create_evaluation_function(
        self, strategy_name: str, data_dict: Dict[str, pd.DataFrame], symbol: str = None
    ):
        """전략 평가 함수 생성"""

        def evaluation_function(params: Dict[str, Any]) -> float:
            """하이퍼파라미터 조합 평가 함수"""
            try:
                # StrategyParams 객체 생성
                strategy_params = StrategyParams(**params)

                # 전략 인스턴스 생성 (새로운 파라미터로)
                strategy_class = self.strategy_manager.strategies[
                    strategy_name
                ].__class__
                strategy = strategy_class(strategy_params)

                # 전략 평가
                if self.research_config.get("research_config", {}).get(
                    "portfolio_mode", False
                ):
                    # 포트폴리오 모드
                    strategy_result = self.evaluator.evaluate_strategy(
                        strategy_name, data_dict
                    )
                    if strategy_result is None:
                        # 기본 결과 반환
                        result = {
                            "total_return": 0.0,
                            "sharpe_ratio": 0.0,
                            "max_drawdown": 0.0,
                            "win_rate": 0.0,
                            "profit_factor": 0.0,
                            "sqn": 0.0,
                            "total_trades": 0,
                            "avg_hold_duration": 0.0,
                        }
                    else:
                        result = {
                            "total_return": strategy_result.total_return,
                            "sharpe_ratio": strategy_result.sharpe_ratio,
                            "max_drawdown": strategy_result.max_drawdown,
                            "win_rate": strategy_result.win_rate,
                            "profit_factor": strategy_result.profit_factor,
                            "sqn": strategy_result.sqn,
                            "total_trades": strategy_result.total_trades,
                            "avg_hold_duration": strategy_result.avg_hold_duration,
                        }
                else:
                    # 단일 종목 모드
                    if symbol:
                        symbol_data = data_dict[symbol]
                    else:
                        symbol_data = list(data_dict.values())[0]

                    # 신호 생성
                    signals = strategy.generate_signals(symbol_data)

                    # 시뮬레이션 실행
                    simulation_result = self.evaluator.simulator.simulate_trading(
                        symbol_data, signals, strategy_name
                    )
                    result = simulation_result["results"]

                # 평가 지표 추출
                primary_metric = self.research_config.get("research_config", {}).get(
                    "optimization_metric", "sharpe_ratio"
                )

                # 기본 지표들 추출
                sharpe = result.get("sharpe_ratio", 0)
                total_return = result.get("total_return", 0)
                win_rate = result.get("win_rate", 0)
                profit_factor = result.get("profit_factor", 0)
                sqn = result.get("sqn", 0)
                max_dd = abs(result.get("max_drawdown", 0))
                total_trades = result.get("total_trades", 0)

                if primary_metric == "sharpe_ratio":
                    score = sharpe
                elif primary_metric == "total_return":
                    score = total_return
                elif primary_metric == "win_rate":
                    score = win_rate
                elif primary_metric == "profit_factor":
                    score = profit_factor
                elif primary_metric == "sqn":
                    score = sqn
                else:
                    # 복합 점수 (여러 지표 조합)
                    score = (
                        sharpe * 0.4
                        + total_return * 0.3
                        + win_rate * 0.2
                        - max_dd * 0.1
                    )

                # 최소 거래 수 필터
                min_trades = self.research_config.get("evaluation_settings", {}).get(
                    "min_trades", 10
                )

                if total_trades < min_trades:
                    score *= 0.5  # 페널티 적용

                # 최소 수익률 필터
                min_return = self.research_config.get("evaluation_settings", {}).get(
                    "min_return", -0.5
                )
                if total_return < min_return:
                    score *= 0.3  # 강한 페널티

                return score

            except Exception as e:
                logger.warning(f"평가 함수 실행 중 오류: {str(e)}")
                return float("-inf")

        return evaluation_function

    def optimize_single_strategy(
        self,
        strategy_name: str,
        symbol: str = None,
        optimization_method: str = "grid_search",
    ) -> OptimizationResult:
        """단일 전략 최적화"""

        logger.info(f"🔬 {strategy_name} 전략 최적화 시작")

        # 데이터 로드
        data_dict = load_and_preprocess_data(
            self.data_dir, [symbol] if symbol else None
        )
        if not data_dict:
            logger.error(f"데이터를 로드할 수 없습니다: {self.data_dir}")
            return None

        # 전략 설정 가져오기
        strategy_config = self.research_config.get("strategies", {}).get(
            strategy_name, {}
        )
        if not strategy_config:
            logger.error(f"전략 설정을 찾을 수 없습니다: {strategy_name}")
            return None

        param_ranges = strategy_config.get("param_ranges", {})
        if not param_ranges:
            logger.error(f"파라미터 범위를 찾을 수 없습니다: {strategy_name}")
            return None

        # 평가 함수 생성
        evaluation_function = self.create_evaluation_function(
            strategy_name, data_dict, symbol
        )

        # 최적화 설정
        optimization_settings = self.research_config.get("optimization_settings", {})

        # 최적화 실행
        if optimization_method == "grid_search":
            grid_settings = optimization_settings.get("grid_search", {})
            result = self.optimizer.grid_search(
                strategy_name=strategy_name,
                param_ranges=param_ranges,
                evaluation_function=evaluation_function,
                max_combinations=grid_settings.get("max_combinations", 50),
                random_sampling=grid_settings.get("random_sampling", True),
                sampling_ratio=grid_settings.get("sampling_ratio", 0.3),
            )
        elif optimization_method == "bayesian_optimization":
            bayesian_settings = optimization_settings.get("bayesian_optimization", {})
            result = self.optimizer.bayesian_optimization(
                strategy_name=strategy_name,
                param_ranges=param_ranges,
                evaluation_function=evaluation_function,
                n_trials=bayesian_settings.get("n_trials", 100),
                n_startup_trials=bayesian_settings.get("n_startup_trials", 10),
            )
        elif optimization_method == "genetic_algorithm":
            ga_settings = optimization_settings.get("genetic_algorithm", {})
            result = self.optimizer.genetic_algorithm(
                strategy_name=strategy_name,
                param_ranges=param_ranges,
                evaluation_function=evaluation_function,
                population_size=ga_settings.get("population_size", 50),
                generations=ga_settings.get("generations", 20),
                mutation_rate=ga_settings.get("mutation_rate", 0.1),
                crossover_rate=ga_settings.get("crossover_rate", 0.8),
            )
        else:
            logger.error(f"지원하지 않는 최적화 방법: {optimization_method}")
            return None

        # 심볼 정보 추가
        if result:
            result.symbol = symbol

        logger.info(f"✅ {strategy_name} 전략 최적화 완료")
        return result

    def run_comprehensive_research(
        self,
        strategies: List[str] = None,
        symbols: List[str] = None,
        optimization_method: str = "grid_search",
    ) -> Dict[str, OptimizationResult]:
        """종합 연구 실행"""

        print_section_header("🔬 하이퍼파라미터 종합 연구 시작")

        # 설정에서 전략과 심볼 가져오기
        if strategies is None:
            strategies = list(self.research_config.get("strategies", {}).keys())

        if symbols is None:
            symbols = self.research_config.get("research_config", {}).get("symbols", [])

        logger.info(f"📊 연구 대상 전략: {len(strategies)}개")
        logger.info(f"📈 연구 대상 심볼: {len(symbols)}개")
        logger.info(f"🔧 최적화 방법: {optimization_method}")

        all_results = {}
        total_strategies = len(strategies) * len(symbols)
        completed = 0

        for strategy_name in strategies:
            for symbol in symbols:
                try:
                    logger.info(f"🔄 진행률: {completed + 1}/{total_strategies}")
                    logger.info(f"  전략: {strategy_name}, 심볼: {symbol}")

                    # 최적화 실행
                    result = self.optimize_single_strategy(
                        strategy_name, symbol, optimization_method
                    )

                    if result:
                        key = f"{strategy_name}_{symbol}"
                        all_results[key] = result
                        logger.info(f"  ✅ 완료 - 최고 점수: {result.best_score:.4f}")
                    else:
                        logger.warning(f"  ⚠️ 실패")

                    completed += 1

                except Exception as e:
                    logger.error(f"  ❌ 오류: {str(e)}")
                    completed += 1
                    continue

        # 결과 저장
        self.save_research_results(all_results)

        # 리포트 생성
        self.generate_research_report(all_results)

        # 최적화된 전략들을 evaluator로 평가
        self.run_comprehensive_evaluation(all_results)

        print_section_header("🎉 하이퍼파라미터 종합 연구 완료")

        return all_results

    def save_research_results(self, results: Dict[str, OptimizationResult]):
        """연구 결과 저장"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 결과를 리스트로 변환
        results_list = list(results.values())

        # 최적화기로 결과 저장
        json_path, csv_path = self.optimizer.save_results(
            results_list, output_dir=self.results_dir
        )

        # 추가 분석 결과 저장
        analysis_results = self._analyze_results(results)
        analysis_path = os.path.join(
            self.results_dir, f"analysis_results_{timestamp}.json"
        )

        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        logger.info(f"📁 연구 결과 저장 완료:")
        logger.info(f"  최적화 결과: {json_path}")
        logger.info(f"  요약 결과: {csv_path}")
        logger.info(f"  분석 결과: {analysis_path}")

    def _analyze_results(
        self, results: Dict[str, OptimizationResult]
    ) -> Dict[str, Any]:
        """결과 분석"""

        analysis = {
            "summary": {
                "total_strategies": len(results),
                "total_execution_time": sum(r.execution_time for r in results.values()),
                "total_combinations_tested": sum(
                    r.n_combinations_tested for r in results.values()
                ),
                "average_score": np.mean([r.best_score for r in results.values()]),
                "best_score": max([r.best_score for r in results.values()]),
                "worst_score": min([r.best_score for r in results.values()]),
            },
            "strategy_performance": {},
            "symbol_performance": {},
            "optimization_method_performance": {},
            "top_performers": [],
        }

        # 전략별 성과
        strategy_scores = {}
        symbol_scores = {}
        method_scores = {}

        for key, result in results.items():
            strategy_name = result.strategy_name
            symbol = result.symbol

            # 전략별 성과
            if strategy_name not in strategy_scores:
                strategy_scores[strategy_name] = []
            strategy_scores[strategy_name].append(result.best_score)

            # 심볼별 성과
            if symbol not in symbol_scores:
                symbol_scores[symbol] = []
            symbol_scores[symbol].append(result.best_score)

            # 최적화 방법별 성과
            method = result.optimization_method
            if method not in method_scores:
                method_scores[method] = []
            method_scores[method].append(result.best_score)

        # 평균 계산
        for strategy_name, scores in strategy_scores.items():
            analysis["strategy_performance"][strategy_name] = {
                "average_score": np.mean(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "std_score": np.std(scores),
                "count": len(scores),
            }

        for symbol, scores in symbol_scores.items():
            analysis["symbol_performance"][symbol] = {
                "average_score": np.mean(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "std_score": np.std(scores),
                "count": len(scores),
            }

        for method, scores in method_scores.items():
            analysis["optimization_method_performance"][method] = {
                "average_score": np.mean(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "std_score": np.std(scores),
                "count": len(scores),
            }

        # 상위 성과자
        sorted_results = sorted(
            results.items(), key=lambda x: x[1].best_score, reverse=True
        )
        analysis["top_performers"] = [
            {
                "key": key,
                "strategy_name": result.strategy_name,
                "symbol": result.symbol,
                "score": result.best_score,
                "params": result.best_params,
                "method": result.optimization_method,
            }
            for key, result in sorted_results[:10]  # 상위 10개
        ]

        return analysis

    def generate_research_report(self, results: Dict[str, OptimizationResult]):
        """연구 리포트 생성"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"research_report_{timestamp}.txt")

        # 최적화기 리포트 생성
        results_list = list(results.values())
        report_content = self.optimizer.generate_optimization_report(results_list)

        # 추가 분석 정보
        analysis = self._analyze_results(results)

        report_lines = [report_content]
        report_lines.append("\n" + "=" * 80)
        report_lines.append("📊 추가 분석 결과")
        report_lines.append("=" * 80)

        # 전략별 성과
        report_lines.append("\n🏆 전략별 평균 성과 (내림차순):")
        strategy_performance = analysis["strategy_performance"]
        sorted_strategies = sorted(
            strategy_performance.items(),
            key=lambda x: x[1]["average_score"],
            reverse=True,
        )

        for strategy_name, perf in sorted_strategies:
            report_lines.append(
                f"  {strategy_name:<25} 평균: {perf['average_score']:<8.4f} "
                f"최고: {perf['max_score']:<8.4f} 최저: {perf['min_score']:<8.4f} "
                f"표준편차: {perf['std_score']:<8.4f}"
            )

        # 심볼별 성과
        report_lines.append("\n📈 심볼별 평균 성과 (내림차순):")
        symbol_performance = analysis["symbol_performance"]
        sorted_symbols = sorted(
            symbol_performance.items(),
            key=lambda x: x[1]["average_score"],
            reverse=True,
        )

        for symbol, perf in sorted_symbols:
            report_lines.append(
                f"  {symbol:<10} 평균: {perf['average_score']:<8.4f} "
                f"최고: {perf['max_score']:<8.4f} 최저: {perf['min_score']:<8.4f} "
                f"표준편차: {perf['std_score']:<8.4f}"
            )

        # 상위 성과자 상세
        report_lines.append("\n🥇 상위 10개 성과자:")
        for i, performer in enumerate(analysis["top_performers"][:10], 1):
            report_lines.append(
                f"  {i:2d}. {performer['strategy_name']:<20} {performer['symbol']:<8} "
                f"점수: {performer['score']:<8.4f} 방법: {performer['method']}"
            )
            report_lines.append(f"      파라미터: {performer['params']}")

        # 파일 저장
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"📄 연구 리포트 생성 완료: {report_path}")

        # 콘솔 출력
        print("\n".join(report_lines))

    def run_quick_test(
        self, strategy_name: str = "dual_momentum", symbol: str = "TSLL"
    ):
        """빠른 테스트 실행"""

        print_section_header("🧪 빠른 테스트 실행")

        logger.info(f"테스트 전략: {strategy_name}")
        logger.info(f"테스트 심볼: {symbol}")

        result = self.optimize_single_strategy(strategy_name, symbol, "grid_search")

        if result:
            print_subsection_header("테스트 결과")
            print(f"전략: {result.strategy_name}")
            print(f"심볼: {result.symbol}")
            print(f"최고 점수: {result.best_score:.4f}")
            print(f"최적 파라미터: {result.best_params}")
            print(f"실행 시간: {result.execution_time:.2f}초")
            print(f"테스트 조합 수: {result.n_combinations_tested}")

            # 최적 파라미터로 evaluator 실행
            self.evaluate_optimized_strategy(result)
        else:
            print("❌ 테스트 실패")

    def evaluate_optimized_strategy(self, optimization_result: OptimizationResult):
        """최적화된 전략을 evaluator로 평가"""
        print_subsection_header("🔍 최적화된 전략 평가")

        strategy_name = optimization_result.strategy_name
        symbol = optimization_result.symbol
        best_params = optimization_result.best_params

        logger.info(f"최적화된 파라미터로 {strategy_name} 전략 평가 시작")
        logger.info(f"심볼: {symbol}")
        logger.info(f"파라미터: {best_params}")

        try:
            # 데이터 로드
            data_dict = load_and_preprocess_data(
                self.data_dir, [symbol] if symbol else None
            )
            if not data_dict:
                logger.error(f"데이터를 로드할 수 없습니다: {self.data_dir}")
                return

            # StrategyParams 객체 생성 (최적화된 파라미터로)
            strategy_params = StrategyParams(**best_params)

            # 전략 인스턴스 생성
            strategy_class = self.strategy_manager.strategies[strategy_name].__class__
            strategy = strategy_class(strategy_params)

            # evaluator에 전략 등록 (임시로)
            original_strategy = self.evaluator.strategy_manager.strategies[
                strategy_name
            ]
            self.evaluator.strategy_manager.strategies[strategy_name] = strategy

            try:
                # 전략 평가 실행
                if self.research_config.get("research_config", {}).get(
                    "portfolio_mode", False
                ):
                    # 포트폴리오 모드
                    result = self.evaluator.evaluate_strategy(strategy_name, data_dict)
                else:
                    # 단일 종목 모드
                    symbol_data = (
                        data_dict[symbol]
                        if symbol in data_dict
                        else list(data_dict.values())[0]
                    )
                    result = self.evaluator.evaluate_strategy(
                        strategy_name, {symbol: symbol_data}
                    )

                if result:
                    print_subsection_header("📊 최적화된 전략 평가 결과")
                    print(f"전략: {result.name}")
                    print(f"총 수익률: {result.total_return*100:.2f}%")
                    print(f"샤프 비율: {result.sharpe_ratio:.4f}")
                    print(f"최대 낙폭: {result.max_drawdown*100:.2f}%")
                    print(f"승률: {result.win_rate*100:.1f}%")
                    print(f"수익 팩터: {result.profit_factor:.2f}")
                    print(f"SQN: {result.sqn:.2f}")
                    print(f"총 거래 수: {result.total_trades}")
                    print(f"평균 보유 기간: {result.avg_hold_duration:.1f}시간")

                    # 거래 상세 정보
                    if result.trades:
                        profitable_trades = [t for t in result.trades if t["pnl"] > 0]
                        losing_trades = [t for t in result.trades if t["pnl"] < 0]

                        print(f"\n📈 거래 상세:")
                        print(f"  수익 거래: {len(profitable_trades)}회")
                        print(f"  손실 거래: {len(losing_trades)}회")

                        if profitable_trades:
                            avg_profit = np.mean([t["pnl"] for t in profitable_trades])
                            print(f"  평균 수익: ${avg_profit:.2f}")

                        if losing_trades:
                            avg_loss = np.mean([t["pnl"] for t in losing_trades])
                            print(f"  평균 손실: ${avg_loss:.2f}")

                    # 결과 저장
                    self.save_evaluation_result(
                        result, best_params, optimization_result
                    )

                else:
                    logger.error("전략 평가 결과가 없습니다.")

            finally:
                # 원래 전략으로 복원
                self.evaluator.strategy_manager.strategies[strategy_name] = (
                    original_strategy
                )

        except Exception as e:
            logger.error(f"최적화된 전략 평가 중 오류: {str(e)}")

    def save_evaluation_result(self, result, best_params, optimization_result):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        evaluation_result = {
            "timestamp": timestamp,
            "strategy_name": result.name,
            "symbol": optimization_result.symbol,
            "optimization_method": optimization_result.optimization_method,
            "best_params": best_params,
            "optimization_score": optimization_result.best_score,
            "evaluation_results": {
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "sqn": result.sqn,
                "total_trades": result.total_trades,
                "avg_hold_duration": result.avg_hold_duration,
            },
            "trades_summary": {
                "total_trades": len(result.trades),
                "profitable_trades": len([t for t in result.trades if t["pnl"] > 0]),
                "losing_trades": len([t for t in result.trades if t["pnl"] < 0]),
                "max_profit": (
                    max([t["pnl"] for t in result.trades]) if result.trades else 0
                ),
                "max_loss": (
                    min([t["pnl"] for t in result.trades]) if result.trades else 0
                ),
            },
        }

        # JSON 파일로 저장
        filename = (
            f"evaluation_{result.name}_{optimization_result.symbol}_{timestamp}.json"
        )
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)

        logger.info(f"📁 평가 결과 저장: {filepath}")

    def run_comprehensive_evaluation(self, results: Dict[str, OptimizationResult]):
        """종합 연구 결과를 evaluator로 평가"""
        print_section_header("🔍 종합 최적화 결과 평가")

        evaluation_results = {}

        for key, optimization_result in results.items():
            logger.info(f"평가 중: {key}")
            try:
                self.evaluate_optimized_strategy(optimization_result)
                evaluation_results[key] = optimization_result
            except Exception as e:
                logger.error(f"{key} 평가 중 오류: {str(e)}")
                continue

        # 종합 평가 리포트 생성
        self.generate_comprehensive_evaluation_report(evaluation_results)

        return evaluation_results

    def generate_comprehensive_evaluation_report(
        self, evaluation_results: Dict[str, OptimizationResult]
    ):
        """종합 평가 리포트 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.results_dir, f"comprehensive_evaluation_{timestamp}.txt"
        )

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🔍 종합 최적화 결과 평가 리포트")
        report_lines.append("=" * 80)
        report_lines.append(
            f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"평가된 전략-심볼 조합: {len(evaluation_results)}개")
        report_lines.append("")

        # 상위 성과자 정렬
        sorted_results = sorted(
            evaluation_results.items(), key=lambda x: x[1].best_score, reverse=True
        )

        report_lines.append("🏆 상위 성과자 (최적화 점수 기준):")
        report_lines.append("-" * 80)

        for i, (key, result) in enumerate(sorted_results[:10], 1):
            report_lines.append(f"{i:2d}. {key}")
            report_lines.append(f"    최적화 점수: {result.best_score:.4f}")
            report_lines.append(f"    최적화 방법: {result.optimization_method}")
            report_lines.append(f"    최적 파라미터: {result.best_params}")
            report_lines.append(f"    실행 시간: {result.execution_time:.2f}초")
            report_lines.append("")

        # 파일 저장
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"📄 종합 평가 리포트 생성: {report_path}")

        # 콘솔 출력
        print("\n".join(report_lines))


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="하이퍼파라미터 연구 및 최적화")
    parser.add_argument(
        "--config", default="config_research.json", help="연구 설정 파일 경로"
    )
    parser.add_argument(
        "--trading_config", default="config.json", help="거래 설정 파일 경로"
    )
    parser.add_argument("--data_dir", default="data", help="데이터 디렉토리 경로")
    parser.add_argument("--results_dir", default="results", help="결과 저장 디렉토리")
    parser.add_argument("--log_dir", default="log", help="로그 디렉토리")
    parser.add_argument("--strategies", nargs="+", help="연구할 전략 목록")
    parser.add_argument("--symbols", nargs="+", help="연구할 심볼 목록")
    parser.add_argument(
        "--method",
        choices=["grid_search", "bayesian_optimization", "genetic_algorithm"],
        default="grid_search",
        help="최적화 방법",
    )
    parser.add_argument("--quick_test", action="store_true", help="빠른 테스트 실행")
    parser.add_argument(
        "--test_strategy", default="dual_momentum", help="테스트할 전략"
    )
    parser.add_argument("--test_symbol", default="TSLL", help="테스트할 심볼")

    args = parser.parse_args()

    # 연구자 초기화
    researcher = HyperparameterResearcher(
        research_config_path=args.config,
        trading_config_path=args.trading_config,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
    )

    if args.quick_test:
        # 빠른 테스트
        researcher.run_quick_test(args.test_strategy, args.test_symbol)
    else:
        # 종합 연구
        results = researcher.run_comprehensive_research(
            strategies=args.strategies,
            symbols=args.symbols,
            optimization_method=args.method,
        )

        print(f"\n🎉 연구 완료! 총 {len(results)}개 전략-심볼 조합 최적화 완료")


if __name__ == "__main__":
    main()
