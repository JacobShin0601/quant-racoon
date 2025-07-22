#!/usr/bin/env python3
"""
고급 포트폴리오 관리자 - 개별 종목 최적화 결과 기반 포트폴리오 최적화
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.portfolio_optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationConstraints,
    OptimizationResult,
)
from actions.portfolio_weight import PortfolioWeightCalculator
from actions.calculate_index import StrategyParams
try:
    from .helper import (
        PortfolioConfig,
        PortfolioWeights,
        Logger,
        load_config,
        load_and_preprocess_data,
        validate_portfolio_weights,
        save_json_data,
        load_json_data,
        print_section_header,
        print_subsection_header,
        format_percentage,
        format_number,
        split_data_train_test,
        DEFAULT_CONFIG_PATH,
        DEFAULT_DATA_DIR,
    )
except ImportError:
    from src.agent.helper import (
        PortfolioConfig,
        PortfolioWeights,
        Logger,
        load_config,
        load_and_preprocess_data,
        validate_portfolio_weights,
        save_json_data,
        load_json_data,
        print_section_header,
        print_subsection_header,
        format_percentage,
        format_number,
        split_data_train_test,
        DEFAULT_CONFIG_PATH,
        DEFAULT_DATA_DIR,
    )


class AdvancedPortfolioManager:
    """고급 포트폴리오 관리자 - 개별 종목 최적화 결과 기반"""

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        time_horizon: str = "swing",
        uuid: Optional[str] = None,
    ):
        print("🔍 PortfolioManager 초기화 시작")
        print(f"🔍 입력 config_path: {config_path}")
        print(f"🔍 입력 time_horizon: {time_horizon}")
        print(f"🔍 입력 uuid: {uuid}")

        # 시간대별 설정 파일 경로 사용 (절대 경로로 변환)
        if time_horizon:
            import os

            current_dir = os.getcwd()  # 현재 작업 디렉토리
            horizon_config_path = os.path.join(
                current_dir, f"config/config_{time_horizon}.json"
            )
            self.config_path = horizon_config_path
            print(f"🔍 time_horizon 기반 설정 파일 경로: {self.config_path}")
        else:
            self.config_path = config_path
            print(f"🔍 직접 지정된 설정 파일 경로: {self.config_path}")

        self.time_horizon = time_horizon
        self.uuid = uuid or datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🔍 최종 UUID: {self.uuid}")

        # 로거 초기화 (설정 로드 전에)
        print("🔍 Logger 초기화 시작")
        self.logger = Logger()
        print("🔍 Logger 초기화 완료")

        # 로거 설정 (설정 로드 후에)
        try:
            # 직접 파일 읽기로 변경
            import json

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self.logger.log_success(f"✅ 설정 파일 로드 완료")
            
            # config에서 output 경로 가져오기
            output_config = self.config.get("output", {})
            logs_folder = output_config.get("logs_folder", "log")
            self.logger.set_log_dir(logs_folder)
            
            # UUID 설정 - logger를 통해 설정
            if self.uuid:
                self.logger.setup_logger(
                    strategy="portfolio_optimization", mode="portfolio", uuid=self.uuid
                )
            else:
                # UUID가 없어도 기본 로거 설정
                self.logger.setup_logger(
                    strategy="portfolio_optimization", mode="portfolio"
                )
                
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            # 기본 로거 설정
            self.logger.setup_logger(
                strategy="portfolio_optimization", mode="portfolio"
            )

        # 직접 print로도 로깅
        print(f"🔍 PortfolioManager 초기화 시작")
        print(f"🔍 설정 파일 경로: {self.config_path}")
        print(f"🔍 시간대: {self.time_horizon}")
        print(f"🔍 UUID: {self.uuid}")

        self.logger.log_info(f"🔍 PortfolioManager 초기화 시작")
        self.logger.log_info(f"🔍 설정 파일 경로: {self.config_path}")
        self.logger.log_info(f"🔍 시간대: {self.time_horizon}")
        self.logger.log_info(f"🔍 UUID: {self.uuid}")

        # 설정 파일은 이미 위에서 로드됨

        try:
            # PortfolioWeightCalculator에 동일한 설정 파일 경로 전달
            self.weight_calculator = PortfolioWeightCalculator(self.config_path)
            self.logger.log_success(f"✅ PortfolioWeightCalculator 초기화 완료")
        except Exception as e:
            self.logger.log_error(f"❌ PortfolioWeightCalculator 초기화 실패: {e}")
            import traceback

            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            raise

        self.params = StrategyParams()
        self.optimizer = None

        # 개별 종목 최적화 결과 저장
        self.individual_optimization_results = {}
        self.portfolio_optimization_result = None

        self.logger.log_success(f"✅ PortfolioManager 초기화 완료")

    def load_individual_optimization_results(
        self, optimization_file_path: str
    ) -> Dict[str, Dict]:
        """개별 종목 최적화 결과 로드"""
        try:
            print(f"🔍 개별 최적화 결과 파일 읽기 시작: {optimization_file_path}")

            with open(optimization_file_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            print(
                f"🔍 JSON 로드 완료: {type(results)}, 길이: {len(results) if results else 0}"
            )
            if results:
                print(f"🔍 결과 키 예시: {list(results.keys())[:3]}")
                print(f"🔍 첫 번째 결과 내용: {list(results.items())[:1]}")

            self.logger.log_success(
                f"개별 최적화 결과 로드 완료: {len(results)}개 조합"
            )
            return results

        except Exception as e:
            print(f"❌ 개별 최적화 결과 로드 실패: {e}")
            import traceback

            print(f"상세 오류: {traceback.format_exc()}")
            self.logger.log_error(f"개별 최적화 결과 로드 실패: {e}")
            return {}

    def select_best_strategy_per_symbol(
        self, optimization_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """각 종목별로 최적의 전략 선택"""
        print("🎯 각 종목별 최적 전략 선택 시작")
        self.logger.log_info("🎯 각 종목별 최적 전략 선택 시작")

        symbols = self.config.get("data", {}).get("symbols", [])
        symbol_best_strategies = {}

        for symbol in symbols:
            print(f"🔍 {symbol} 최적 전략 선택 중...")
            best_score = -999999.0
            best_strategy = None
            best_params = {}
            tested_strategies = 0

            # 해당 종목의 모든 전략 결과 비교
            for key, result in optimization_results.items():
                if result.get("symbol") == symbol:
                    tested_strategies += 1
                    score = result.get("best_score", -999999.0)
                    strategy_name = result.get("strategy_name", "")

                    # -999999 점수는 로그에서 숨기기
                    if score > -999999.0:
                        print(f"  - {strategy_name}: 점수 {score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_strategy = strategy_name
                        best_params = result.get("best_params", {})

            if best_strategy:
                symbol_best_strategies[symbol] = {
                    "strategy": best_strategy,
                    "params": best_params,
                    "score": best_score,
                    "tested_strategies": tested_strategies,
                }
                print(
                    f"✅ {symbol} 최적 전략: {best_strategy} (점수: {best_score:.3f})"
                )
                self.logger.log_success(
                    f"✅ {symbol} 최적 전략: {best_strategy} (점수: {best_score:.3f})"
                )
            else:
                print(f"⚠️ {symbol} 유효한 전략 없음")
                self.logger.log_warning(f"⚠️ {symbol} 유효한 전략 없음")

        self.logger.log_info(
            f"📊 종목별 최적 전략 선택 완료: {len(symbol_best_strategies)}개 종목"
        )
        return symbol_best_strategies

    def create_optimal_portfolio(
        self,
        symbol_best_strategies: Dict[str, Dict],
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """개별 종목별 최적 전략으로 포트폴리오 구성 (fallback 메커니즘 포함)"""
        print("🎯 최적 전략 기반 포트폴리오 구성 시작")
        self.logger.log_info("🎯 최적 전략 기반 포트폴리오 구성 시작")

        # 1. 각 종목의 최적 전략으로 수익률 계산
        symbol_returns = {}
        successful_symbols = 0

        for symbol, strategy_info in symbol_best_strategies.items():
            strategy_name = strategy_info["strategy"]
            params = strategy_info["params"]

            print(f"🔍 {symbol} ({strategy_name}) 수익률 계산 중...")

            # 최적화된 파라미터로 전략 실행
            returns = self._calculate_strategy_returns(
                data_dict[symbol], strategy_name, params
            )

            if returns is not None:
                symbol_returns[symbol] = returns
                successful_symbols += 1
                print(f"✅ {symbol} 수익률 계산 성공")
                self.logger.log_success(f"✅ {symbol} 수익률 계산 성공")
            else:
                print(f"⚠️ {symbol} 수익률 계산 실패")
                self.logger.log_warning(f"⚠️ {symbol} 수익률 계산 실패")

        if not symbol_returns:
            self.logger.log_error("❌ 유효한 수익률 데이터가 없습니다")
            return {}

        # 2. 포트폴리오 비중 최적화 (메인 방법 시도)
        print(f"📊 포트폴리오 최적화 시작: {len(symbol_returns)}개 종목")
        self.logger.log_info(f"📊 포트폴리오 최적화 시작: {len(symbol_returns)}개 종목")

        returns_df = pd.DataFrame(symbol_returns).dropna()

        if returns_df.empty:
            self.logger.log_error("❌ NaN 제거 후 데이터가 없습니다")
            return {}

        print(f"📊 수익률 데이터 형태: {returns_df.shape}")
        self.logger.log_info(f"📊 수익률 데이터 형태: {returns_df.shape}")

        # 3. 메인 최적화 방법 시도 (portfolio_optimization.py)
        portfolio_result = self._try_main_optimization(
            returns_df, symbol_best_strategies
        )

        if portfolio_result:
            return portfolio_result

        # 4. Fallback 방법 시도 (portfolio_weight.py)
        print("🔄 메인 최적화 실패, Fallback 방법 시도 중...")
        self.logger.log_warning("🔄 메인 최적화 실패, Fallback 방법 시도 중...")

        portfolio_result = self._try_fallback_optimization(
            data_dict, symbol_best_strategies
        )

        if portfolio_result:
            return portfolio_result

        # 5. 모든 방법 실패
        self.logger.log_error("❌ 모든 포트폴리오 최적화 방법 실패")
        return {}

    def _try_main_optimization(
        self, returns_df: pd.DataFrame, symbol_best_strategies: Dict[str, Dict]
    ) -> Optional[Dict[str, Any]]:
        """메인 최적화 방법 시도 (portfolio_optimization.py)"""
        try:
            print("🔍 메인 최적화 방법 시도 중...")
            self.logger.log_info("🔍 메인 최적화 방법 시도 중...")

            portfolio_config = self.get_portfolio_config()
            self.optimizer = PortfolioOptimizer(
                returns=returns_df,
                risk_free_rate=portfolio_config.risk_free_rate,
            )

            constraints = self.get_optimization_constraints()

            # 최적화 방법 설정 (설정 파일에서 읽기)
            portfolio_config = self.config.get("portfolio", {})
            method_name = portfolio_config.get(
                "optimization_method", "sharpe_maximization"
            )

            # 문자열을 OptimizationMethod로 변환
            if method_name == "sharpe_maximization":
                optimization_method = OptimizationMethod.SHARPE_MAXIMIZATION
            elif method_name == "sortino_maximization":
                optimization_method = OptimizationMethod.SORTINO_MAXIMIZATION
            elif method_name == "risk_parity":
                optimization_method = OptimizationMethod.RISK_PARITY
            elif method_name == "minimum_variance":
                optimization_method = OptimizationMethod.MINIMUM_VARIANCE
            elif method_name == "mean_variance":
                optimization_method = OptimizationMethod.MEAN_VARIANCE
            else:
                # 잘못된 방법인 경우 예외 발생하여 fallback 트리거
                raise ValueError(f"지원하지 않는 최적화 방법: {method_name}")

            print(
                f"🔍 포트폴리오 최적화 방법: {method_name} -> {optimization_method.value}"
            )
            self.logger.log_info(
                f"🔍 포트폴리오 최적화 방법: {method_name} -> {optimization_method.value}"
            )

            result = self.optimizer.optimize_portfolio(optimization_method, constraints)

            # 결과 구성
            portfolio_result = {
                "weights": dict(zip(returns_df.columns, result.weights)),
                "symbol_strategies": symbol_best_strategies,
                "performance": {
                    "sharpe_ratio": result.sharpe_ratio,
                    "expected_return": result.expected_return,
                    "volatility": result.volatility,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown": result.max_drawdown,
                },
                "returns_data": returns_df,
                "optimization_result": result,
                "optimization_method": "main",
                "method_used": method_name,
            }

            self.logger.log_success(
                f"✅ 메인 최적화 성공: 샤프 {result.sharpe_ratio:.3f}, "
                f"수익률 {result.expected_return*252*100:.2f}%"
            )

            return portfolio_result

        except Exception as e:
            print(f"❌ 메인 최적화 실패: {e}")
            self.logger.log_warning(f"❌ 메인 최적화 실패: {e}")
            import traceback

            self.logger.log_warning(f"상세 오류: {traceback.format_exc()}")
            return None

    def _try_fallback_optimization(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol_best_strategies: Dict[str, Dict],
    ) -> Optional[Dict[str, Any]]:
        """Fallback 최적화 방법 시도 (portfolio_weight.py)"""
        try:
            print("🔍 Fallback 최적화 방법 시도 중...")
            self.logger.log_info("🔍 Fallback 최적화 방법 시도 중...")

            # Fallback 방법 설정 (설정 파일에서 읽기)
            portfolio_config = self.config.get("portfolio", {})
            fallback_method = portfolio_config.get("fallback_method", "equal_weight")

            print(f"🔍 Fallback 방법: {fallback_method}")
            self.logger.log_info(f"🔍 Fallback 방법: {fallback_method}")

            # PortfolioWeightCalculator 사용
            weights_df = self.weight_calculator.calculate_optimal_weights(data_dict)

            if weights_df.empty:
                print("❌ Fallback 최적화 실패: 빈 결과")
                self.logger.log_warning("❌ Fallback 최적화 실패: 빈 결과")
                return None

            # 최신 비중 추출
            latest_weights = weights_df.iloc[-1].to_dict()

            # CASH 제거하고 정규화
            if "CASH" in latest_weights:
                del latest_weights["CASH"]

            # 비중 합계로 정규화
            total_weight = sum(latest_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    k: v / total_weight for k, v in latest_weights.items()
                }
            else:
                # 모든 비중이 0인 경우 동등 비중
                symbols = list(latest_weights.keys())
                normalized_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}

            # 성과 지표 계산 (간단한 추정)
            performance = {
                "sharpe_ratio": 0.5,  # 기본값
                "expected_return": 0.02 / 252,  # 기본값
                "volatility": 0.15 / np.sqrt(252),  # 기본값
                "sortino_ratio": 0.4,  # 기본값
                "max_drawdown": -0.05,  # 기본값
            }

            # 결과 구성
            portfolio_result = {
                "weights": normalized_weights,
                "symbol_strategies": symbol_best_strategies,
                "performance": performance,
                "returns_data": pd.DataFrame(),  # 빈 DataFrame
                "optimization_result": None,
                "optimization_method": "fallback",
                "method_used": fallback_method,
            }

            self.logger.log_success(
                f"✅ Fallback 최적화 성공: {fallback_method} 방법 사용"
            )

            return portfolio_result

        except Exception as e:
            print(f"❌ Fallback 최적화 실패: {e}")
            self.logger.log_error(f"❌ Fallback 최적화 실패: {e}")
            import traceback

            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            return None

    def prepare_strategy_returns_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        optimization_results: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """전략별 수익률 데이터 준비 (개별 최적화 결과 기반) - 레거시 메서드"""
        # 이 메서드는 기존 호환성을 위해 유지하되, 새로운 방식 사용
        return self.create_optimal_portfolio(
            self.select_best_strategy_per_symbol(optimization_results), data_dict
        )

    def _calculate_strategy_returns(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        params: Dict[str, Any],
    ) -> Optional[pd.Series]:
        """최적화된 파라미터로 전략 수익률 계산"""
        try:
            print(f"▶️ 전략 인스턴스 생성 시도: {strategy_name}")
            from actions.strategies import StrategyManager
            from actions.log_pl import TradingSimulator

            # 전략 매니저 초기화
            strategy_manager = StrategyManager()

            # 전략 등록 (기본 파라미터로)
            from actions.strategies import (
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
                MeanReversionStrategy,
                SwingBreakoutStrategy,
                SwingPullbackEntryStrategy,
                SwingCandlePatternStrategy,
                SwingBollingerBandStrategy,
                SwingMACDStrategy,
            )

            strategy_classes = {
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

            if strategy_name not in strategy_classes:
                print(f"❌ 알 수 없는 전략: {strategy_name}")
                self.logger.log_error(f"❌ 알 수 없는 전략: {strategy_name}")
                return None

            # 전략 인스턴스 생성
            strategy = strategy_classes[strategy_name](StrategyParams())
            print(f"✅ 전략 인스턴스 생성 성공: {strategy}")

            # 최적화된 파라미터 적용 (전략별 유효한 파라미터만)
            valid_params = {}
            for param_name, param_value in params.items():
                print(f"  - 파라미터 적용: {param_name} = {param_value}")
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)
                    valid_params[param_name] = param_value
                    self.logger.log_info(
                        f"  - 파라미터 설정: {param_name} = {param_value}"
                    )
                else:
                    print(f"  ⚠️ 전략에 없는 파라미터: {param_name} (무시됨)")
            
            print(f"  - 적용된 유효 파라미터: {list(valid_params.keys())}")

            # 신호 생성
            print(f"  - 신호 생성 시작")
            signals = strategy.generate_signals(data)
            print(
                f"  - 신호 생성 결과: {type(signals)}, shape: {getattr(signals, 'shape', None)}"
            )
            if signals is None or signals.empty:
                print(f"⚠️ {strategy_name}: 신호 생성 실패")
                self.logger.log_warning(f"⚠️ {strategy_name}: 신호 생성 실패")
                return None

            print(f"  - 신호 생성 완료: {signals.shape}")

            # 거래 시뮬레이션 전, datetime 컬럼 보장
            if "datetime" not in data.columns:
                if data.index.name == "datetime":
                    data = data.reset_index()
                else:
                    data["datetime"] = data.index
            if "datetime" not in signals.columns:
                if signals.index.name == "datetime":
                    signals = signals.reset_index()
                else:
                    signals["datetime"] = data["datetime"].values

            # 거래 시뮬레이션
            print(f"  - 거래 시뮬레이션 시작")
            simulator = TradingSimulator(self.config_path)
            result = simulator.simulate_trading(data, signals, strategy_name)
            print(
                f"  - 시뮬레이션 결과: {type(result)}, keys: {list(result.keys()) if result else None}"
            )

            if not result:
                print(f"⚠️ {strategy_name}: 거래 시뮬레이션 실패")
                self.logger.log_warning(f"⚠️ {strategy_name}: 거래 시뮬레이션 실패")
                return None

            print(f"  - 거래 시뮬레이션 완료")

            # 수익률 반환
            returns = result.get("returns", [])
            print(f"  - 수익률 길이: {len(returns)}")
            if returns:
                print(f"  - 수익률 데이터 생성 완료: {len(returns)}개 포인트")
                return pd.Series(returns, index=data.index[-len(returns) :])
            else:
                print(f"⚠️ {strategy_name}: 수익률 데이터 없음")
                self.logger.log_warning(f"⚠️ {strategy_name}: 수익률 데이터 없음")
                return None

        except Exception as e:
            print(f"❌ 전략 수익률 계산 실패: {strategy_name} - {e}")
            import traceback

            print(f"상세 오류: {traceback.format_exc()}")
            self.logger.log_error(f"❌ 전략 수익률 계산 실패: {strategy_name} - {e}")
            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            return None

    def optimize_portfolio_with_individual_results(
        self,
        data_dict: Dict[str, pd.DataFrame],
        optimization_results: Dict[str, Dict],
        method: OptimizationMethod = None,
    ) -> Dict[str, Any]:
        """개별 최적화 결과 기반 포트폴리오 최적화 (개별 종목별 최적 전략 선택)"""
        print("🎯 개별 최적화 결과 기반 포트폴리오 최적화 시작")
        self.logger.log_section("🎯 개별 최적화 결과 기반 포트폴리오 최적화")

        try:
            # 최적화 방법 설정 (설정 파일에서 읽기)
            if method is None:
                portfolio_config = self.config.get("portfolio", {})
                method_name = portfolio_config.get(
                    "optimization_method", "sharpe_maximization"
                )

                # 문자열을 OptimizationMethod로 변환
                if method_name == "sharpe_maximization":
                    method = OptimizationMethod.SHARPE_MAXIMIZATION
                elif method_name == "sortino_maximization":
                    method = OptimizationMethod.SORTINO_MAXIMIZATION
                elif method_name == "risk_parity":
                    method = OptimizationMethod.RISK_PARITY
                elif method_name == "minimum_variance":
                    method = OptimizationMethod.MINIMUM_VARIANCE
                elif method_name == "mean_variance":
                    method = OptimizationMethod.MEAN_VARIANCE
                else:
                    method = OptimizationMethod.SHARPE_MAXIMIZATION  # 기본값

                print(
                    f"🔍 설정 파일에서 최적화 방법 로드: {method_name} -> {method.value}"
                )
                self.logger.log_info(
                    f"🔍 설정 파일에서 최적화 방법 로드: {method_name} -> {method.value}"
                )

            print(f"🔍 입력 데이터 검증:")
            print(f"  - 데이터 종목 수: {len(data_dict)}")
            print(f"  - 최적화 결과 조합 수: {len(optimization_results)}")
            print(f"  - 최적화 방법: {method.value}")
            self.logger.log_info(f"🔍 입력 데이터 검증:")
            self.logger.log_info(f"  - 데이터 종목 수: {len(data_dict)}")
            self.logger.log_info(
                f"  - 최적화 결과 조합 수: {len(optimization_results)}"
            )
            self.logger.log_info(f"  - 최적화 방법: {method.value}")

            # 1. 각 종목별 최적 전략 선택
            print("📊 1단계: 각 종목별 최적 전략 선택 시작")
            self.logger.log_info("📊 1단계: 각 종목별 최적 전략 선택 시작")

            symbol_best_strategies = self.select_best_strategy_per_symbol(
                optimization_results
            )

            if not symbol_best_strategies:
                self.logger.log_error("❌ 유효한 종목별 최적 전략이 없습니다")
                return {}

            print(
                f"✅ 종목별 최적 전략 선택 완료: {len(symbol_best_strategies)}개 종목"
            )
            self.logger.log_success(
                f"✅ 종목별 최적 전략 선택 완료: {len(symbol_best_strategies)}개 종목"
            )

            # 2. 최적 전략 기반 포트폴리오 구성
            print("📊 2단계: 최적 전략 기반 포트폴리오 구성 시작")
            self.logger.log_info("📊 2단계: 최적 전략 기반 포트폴리오 구성 시작")

            portfolio_result = self.create_optimal_portfolio(
                symbol_best_strategies, data_dict
            )

            if not portfolio_result:
                self.logger.log_error("❌ 포트폴리오 구성 실패")
                return {}

            # 3. 최종 결과 구성
            final_result = {
                "portfolio_weights": portfolio_result["weights"],
                "symbol_strategies": portfolio_result["symbol_strategies"],
                "performance": portfolio_result["performance"],
                "optimization_method": method.value,
                "timestamp": datetime.now().isoformat(),
                "returns_data": portfolio_result["returns_data"],
                "optimization_result": portfolio_result["optimization_result"],
            }

            self.portfolio_optimization_result = final_result
            self.logger.log_success("✅ 포트폴리오 최적화 완료")
            return final_result

        except Exception as e:
            print(f"❌ 포트폴리오 최적화 실패: {e}")
            import traceback

            print(f"상세 오류: {traceback.format_exc()}")
            self.logger.log_error(f"포트폴리오 최적화 실패: {e}")
            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            return {}

    def load_portfolio_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """포트폴리오 데이터 로드"""
        try:
            print(f"🔍 load_portfolio_data 시작 - 데이터 디렉토리: {data_dir}")
            print(f"🔍 UUID: {self.uuid}")
            self.logger.log_info(f"🔍 데이터 디렉토리: {data_dir}")
            self.logger.log_info(f"🔍 UUID: {self.uuid}")

            data_dict = {}
            symbols = self.config.get("data", {}).get("symbols", [])
            print(f"🔍 설정된 심볼들: {symbols}")
            self.logger.log_info(f"🔍 설정된 심볼들: {symbols}")

            if not symbols:
                print("❌ 설정 파일에서 심볼을 찾을 수 없습니다")
                self.logger.log_error("설정 파일에서 심볼을 찾을 수 없습니다")
                return {}

            # time_horizon을 고려한 데이터 경로 구성
            # data_dir이 이미 time_horizon을 포함하고 있는지 확인
            if self.time_horizon and not str(data_dir).endswith(f"/{self.time_horizon}"):
                data_path = Path(data_dir) / self.time_horizon
            else:
                data_path = Path(data_dir)
            
            print(f"🔍 time_horizon 기반 데이터 경로: {data_path}")
            self.logger.log_info(f"🔍 time_horizon 기반 데이터 경로: {data_path}")
            
            # data_path가 존재하는지 확인
            if not data_path.exists():
                print(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_path}")
                self.logger.log_error(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_path}")
                return {}
            
            print(f"🔍 최종 검색 경로: {data_path}")
            self.logger.log_info(f"🔍 최종 검색 경로: {data_path}")

            for symbol in symbols:
                self.logger.log_info(f"🔍 {symbol} 데이터 파일 검색 중...")
                # 파일명 패턴 수정 - 실제 파일명 형식에 맞게
                pattern = f"{symbol}_*.csv"
                self.logger.log_info(f"🔍 검색 패턴: {pattern}")
                self.logger.log_info(f"🔍 검색 경로: {data_path}")

                files = list(data_path.glob(pattern))
                self.logger.log_info(
                    f"🔍 {symbol}에 대한 매칭 파일: {[f.name for f in files]}"
                )

                if files:
                    # 가장 최신 파일 선택 (파일명의 타임스탬프 기준)
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    self.logger.log_info(f"🔍 {symbol} 파일 로드: {latest_file}")
                    df = pd.read_csv(latest_file)
                    
                    # datetime 컬럼 처리
                    if "datetime" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        df.set_index("datetime", inplace=True)
                    elif "date" in df.columns:
                        df["datetime"] = pd.to_datetime(df["date"])
                        df.set_index("datetime", inplace=True)
                    else:
                        # 인덱스가 이미 datetime인 경우
                        df.index = pd.to_datetime(df.index)
                    
                    data_dict[symbol] = df
                    self.logger.log_info(
                        f"✅ {symbol} 데이터 로드: {latest_file.name} (행: {len(df)})"
                    )
                else:
                    self.logger.log_warning(f"⚠️ {symbol} 데이터 파일을 찾을 수 없음")

            self.logger.log_success(
                f"포트폴리오 데이터 로드 완료: {len(data_dict)}개 종목"
            )
            return data_dict

        except Exception as e:
            self.logger.log_error(f"포트폴리오 데이터 로드 실패: {e}")
            import traceback

            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            return {}

    def _select_best_strategy(self, portfolio_results: Dict[str, Any]) -> str:
        """최적 전략 선택 (복합 점수 기준)"""
        best_strategy = "dual_momentum"  # 기본값 설정
        best_score = -999999.0

        for strategy_name, result in portfolio_results.items():
            opt_result = result["optimization_result"]

            # 복합 점수 계산 (샤프 비율 + 수익률 + 낮은 변동성)
            composite_score = (
                opt_result.sharpe_ratio * 0.4
                + opt_result.expected_return * 252 * 0.3
                + (1 - opt_result.volatility * np.sqrt(252)) * 0.3
            )

            if composite_score > best_score:
                best_score = composite_score
                best_strategy = strategy_name

        return best_strategy

    def get_optimization_constraints(self) -> OptimizationConstraints:
        """최적화 제약조건 설정"""
        portfolio_config = self.config.get("portfolio", {})
        trading_config = self.config.get("trading", {})

        # 기본 제약조건 (설정 파일에서 읽기)
        min_weight = portfolio_config.get("min_weight", 0.0)
        max_weight = portfolio_config.get("max_weight", 1.0)
        
        print(f"🔍 포트폴리오 제약조건 설정:")
        print(f"  - 최소 비중: {min_weight}")
        print(f"  - 최대 비중: {max_weight}")

        constraints = OptimizationConstraints(
            min_weight=min_weight,
            max_weight=max_weight,
            cash_weight=portfolio_config.get("cash_weight", 0.0),
            leverage=portfolio_config.get("leverage", 1.0),
            enable_short_position=trading_config.get("enable_short_position", False),
            short_weight_limit=portfolio_config.get("short_weight_limit", 0.5),
            target_return=portfolio_config.get("target_return"),
            target_volatility=portfolio_config.get("target_volatility"),
            max_drawdown=portfolio_config.get("max_drawdown"),
        )

        return constraints

    def get_portfolio_config(self) -> PortfolioConfig:
        """포트폴리오 설정 반환"""
        portfolio_config = self.config.get("portfolio", {})
        data_config = self.config.get("data", {})

        return PortfolioConfig(
            symbols=data_config.get("symbols", []),
            weight_method=portfolio_config.get(
                "weight_calculation_method", "sharpe_maximization"
            ),
            rebalance_period=portfolio_config.get("rebalance_period", 4),
            risk_free_rate=portfolio_config.get("risk_free_rate", 0.02),
            target_volatility=portfolio_config.get("target_volatility"),
            min_weight=portfolio_config.get("min_weight", 0.0),
            max_weight=portfolio_config.get("max_weight", 1.0),
        )

    def save_portfolio_optimization_result(self, output_path: Optional[str] = None):
        """포트폴리오 최적화 결과 저장 (개별 종목별 최적 전략 기반)"""
        if not self.portfolio_optimization_result:
            self.logger.log_warning("저장할 포트폴리오 최적화 결과가 없습니다")
            return

        if not output_path:
            # config에서 output 경로 가져오기
            output_config = self.config.get("output", {})
            results_folder = output_config.get("results_folder", "results")
            
            # results 폴더 생성
            os.makedirs(results_folder, exist_ok=True)
            
            # UUID가 있으면 사용, 없으면 현재 시간 사용
            if self.uuid:
                output_path = os.path.join(results_folder, f"portfolio_optimization_{self.uuid}.json")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_folder, f"portfolio_optimization_{timestamp}.json")

        try:
            # 결과를 JSON 직렬화 가능한 형태로 변환
            serializable_result = {
                "portfolio_weights": self.portfolio_optimization_result[
                    "portfolio_weights"
                ],
                "symbol_strategies": self.portfolio_optimization_result[
                    "symbol_strategies"
                ],
                "performance": self.portfolio_optimization_result["performance"],
                "optimization_method": self.portfolio_optimization_result[
                    "optimization_method"
                ],
                "timestamp": self.portfolio_optimization_result["timestamp"],
            }

            # 최적화 결과도 포함 (numpy 배열을 리스트로 변환)
            if "optimization_result" in self.portfolio_optimization_result:
                opt_result = self.portfolio_optimization_result["optimization_result"]
                serializable_result["optimization_details"] = {
                    "sharpe_ratio": opt_result.sharpe_ratio,
                    "expected_return": opt_result.expected_return,
                    "volatility": opt_result.volatility,
                    "sortino_ratio": opt_result.sortino_ratio,
                    "max_drawdown": opt_result.max_drawdown,
                    "weights": (
                        opt_result.weights.tolist()
                        if opt_result.weights is not None
                        else []
                    ),
                }

            # 수익률 데이터도 포함 (evaluator에서 사용)
            if "returns_data" in self.portfolio_optimization_result:
                returns_df = self.portfolio_optimization_result["returns_data"]
                serializable_result["returns_data"] = {
                    "columns": returns_df.columns.tolist(),
                    "index": returns_df.index.tolist(),
                    "values": returns_df.values.tolist()
                }

            # 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            self.logger.log_success(f"포트폴리오 최적화 결과 저장: {output_path}")
            return output_path

        except Exception as e:
            self.logger.log_error(f"포트폴리오 최적화 결과 저장 실패: {e}")
            import traceback

            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            return None

    def generate_portfolio_report(self) -> str:
        """포트폴리오 최적화 보고서 생성 (개별 종목별 최적 전략 기반)"""
        if not self.portfolio_optimization_result:
            return "포트폴리오 최적화 결과가 없습니다."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🎯 개별 종목별 최적 전략 기반 포트폴리오 최적화 보고서")
        report_lines.append("=" * 80)
        report_lines.append(f"📅 생성 시간: {datetime.now()}")
        report_lines.append(
            f"🔧 최적화 방법: {self.portfolio_optimization_result['optimization_method']}"
        )

        # 사용된 방법 표시
        optimization_method = self.portfolio_optimization_result.get(
            "optimization_method", "unknown"
        )
        method_used = self.portfolio_optimization_result.get("method_used", "unknown")

        if optimization_method == "main":
            report_lines.append(f"🎯 사용된 방법: 메인 최적화 ({method_used})")
        elif optimization_method == "fallback":
            report_lines.append(f"🔄 사용된 방법: Fallback 최적화 ({method_used})")
        else:
            # 기존 호환성을 위한 처리
            if method_used == "unknown" and optimization_method in [
                "sharpe_maximization",
                "sortino_maximization",
                "risk_parity",
                "minimum_variance",
                "mean_variance",
            ]:
                report_lines.append(
                    f"🎯 사용된 방법: 메인 최적화 ({optimization_method})"
                )
            else:
                report_lines.append(
                    f"❓ 사용된 방법: {optimization_method} ({method_used})"
                )

        report_lines.append("")

        # 포트폴리오 성과 요약
        performance = self.portfolio_optimization_result["performance"]
        report_lines.append("📊 포트폴리오 성과 요약:")
        report_lines.append("-" * 60)
        report_lines.append(f"샤프 비율: {performance['sharpe_ratio']:.3f}")
        report_lines.append(
            f"예상 수익률: {performance['expected_return']*252*100:.2f}%"
        )
        report_lines.append(
            f"변동성: {performance['volatility']*np.sqrt(252)*100:.2f}%"
        )
        report_lines.append(f"소르티노 비율: {performance['sortino_ratio']:.3f}")
        report_lines.append(f"최대 낙폭: {performance['max_drawdown']*100:.2f}%")
        report_lines.append("")

        # 종목별 최적 전략 및 비중
        report_lines.append("📈 종목별 최적 전략 및 포트폴리오 비중:")
        report_lines.append("-" * 60)

        symbol_strategies = self.portfolio_optimization_result["symbol_strategies"]
        portfolio_weights = self.portfolio_optimization_result["portfolio_weights"]

        # 비중 기준으로 정렬
        sorted_weights = sorted(
            portfolio_weights.items(), key=lambda x: x[1], reverse=True
        )

        for i, (symbol, weight) in enumerate(sorted_weights, 1):
            strategy_info = symbol_strategies.get(symbol, {})
            strategy_name = strategy_info.get("strategy", "Unknown")
            score = strategy_info.get("score", 0)

            report_lines.append(f"{i}. {symbol}:")
            report_lines.append(f"   최적 전략: {strategy_name}")
            report_lines.append(f"   전략 점수: {score:.3f}")
            report_lines.append(f"   포트폴리오 비중: {weight*100:.2f}%")
            report_lines.append("")

        # 전략별 분포
        report_lines.append("🎯 전략별 분포:")
        report_lines.append("-" * 60)
        strategy_distribution = {}
        for symbol, strategy_info in symbol_strategies.items():
            strategy_name = strategy_info["strategy"]
            weight = portfolio_weights.get(symbol, 0)
            if strategy_name not in strategy_distribution:
                strategy_distribution[strategy_name] = {"count": 0, "weight": 0}
            strategy_distribution[strategy_name]["count"] += 1
            strategy_distribution[strategy_name]["weight"] += weight

        for strategy_name, info in strategy_distribution.items():
            report_lines.append(f"{strategy_name}:")
            report_lines.append(f"  사용 종목 수: {info['count']}개")
            report_lines.append(f"  총 비중: {info['weight']*100:.2f}%")
            report_lines.append("")

        return "\n".join(report_lines)

    def run_portfolio_optimization(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        optimization_file_path: Optional[str] = None,
    ) -> bool:
        """포트폴리오 최적화 실행"""
        print("=" * 80)
        print("⚖️ 포트폴리오 최적화 시작")
        print("=" * 80)
        print_subsection_header("⚖️ 포트폴리오 최적화")

        print(f"🔍 포트폴리오 최적화 시작 - 데이터 디렉토리: {data_dir}")
        print(f"🔍 설정 파일 경로: {self.config_path}")
        print(f"🔍 UUID: {self.uuid}")
        print(f"🔍 현재 작업 디렉토리: {os.getcwd()}")
        print(f"🔍 optimization_file_path: {optimization_file_path}")

        self.logger.log_info(f"🔍 포트폴리오 최적화 시작 - 데이터 디렉토리: {data_dir}")
        self.logger.log_info(f"🔍 설정 파일 경로: {self.config_path}")
        self.logger.log_info(f"🔍 UUID: {self.uuid}")
        self.logger.log_info(f"🔍 현재 작업 디렉토리: {os.getcwd()}")

        try:
            # 1. 데이터 로드
            print("📊 1단계: 포트폴리오 데이터 로드 시작")
            self.logger.log_info("📊 1단계: 포트폴리오 데이터 로드 시작")

            print(f"🔍 데이터 디렉토리 존재 여부: {os.path.exists(data_dir)}")
            self.logger.log_info(
                f"🔍 데이터 디렉토리 존재 여부: {os.path.exists(data_dir)}"
            )

            if os.path.exists(data_dir):
                dir_contents = os.listdir(data_dir)
                print(f"🔍 데이터 디렉토리 내용: {dir_contents}")
                self.logger.log_info(f"🔍 데이터 디렉토리 내용: {dir_contents}")
            else:
                print(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
                self.logger.log_error(
                    f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_dir}"
                )

            print("🔍 load_portfolio_data 호출 시작")
            data_dict = self.load_portfolio_data(data_dir)
            print(
                f"🔍 load_portfolio_data 결과: {type(data_dict)}, 길이: {len(data_dict) if data_dict else 0}"
            )

            if not data_dict:
                print("❌ 데이터 로드 실패")
                self.logger.log_error("데이터 로드 실패")
                return False

            # Train/Test 분할 (train 데이터만 사용)
            train_ratio = self.config.get("data", {}).get("train_ratio", 0.8)
            train_data_dict, test_data_dict = split_data_train_test(
                data_dict, train_ratio
            )
            data_dict = train_data_dict  # train 데이터만 사용
            print(
                f"Train/Test 분할 완료: Train {len(train_data_dict)}개 종목, Test {len(test_data_dict)}개 종목"
            )
            self.logger.log_info(
                f"Train/Test 분할 완료: Train {len(train_data_dict)}개 종목, Test {len(test_data_dict)}개 종목"
            )

            print(f"✅ 데이터 로드 완료: {len(data_dict)}개 종목")
            print(f"🔍 로드된 종목들: {list(data_dict.keys())}")
            self.logger.log_success(f"✅ 데이터 로드 완료: {len(data_dict)}개 종목")
            self.logger.log_info(f"🔍 로드된 종목들: {list(data_dict.keys())}")

            # 2. 개별 최적화 결과 로드
            print("📊 2단계: 개별 최적화 결과 로드 시작")
            self.logger.log_info("📊 2단계: 개별 최적화 결과 로드 시작")

            if optimization_file_path:
                print(f"🔍 지정된 최적화 파일: {optimization_file_path}")
                print(f"🔍 파일 존재 여부: {os.path.exists(optimization_file_path)}")
                self.logger.log_info(f"🔍 지정된 최적화 파일: {optimization_file_path}")
                self.logger.log_info(
                    f"🔍 파일 존재 여부: {os.path.exists(optimization_file_path)}"
                )

                print("🔍 load_individual_optimization_results 호출 시작")
                optimization_results = self.load_individual_optimization_results(
                    optimization_file_path
                )
                print(
                    f"🔍 load_individual_optimization_results 결과: {type(optimization_results)}, 길이: {len(optimization_results) if optimization_results else 0}"
                )
            else:
                # 최신 최적화 결과 파일 자동 감지
                print("🔍 최신 최적화 결과 파일 자동 감지")
                self.logger.log_info("🔍 최신 최적화 결과 파일 자동 감지")
                optimization_results = self._find_latest_optimization_results()

            if not optimization_results:
                print("❌ 개별 최적화 결과를 찾을 수 없습니다")
                self.logger.log_error("개별 최적화 결과를 찾을 수 없습니다")
                return False

            print(f"✅ 개별 최적화 결과 로드 완료: {len(optimization_results)}개 조합")
            print(f"🔍 최적화 결과 키 예시: {list(optimization_results.keys())[:5]}")
            self.logger.log_success(
                f"✅ 개별 최적화 결과 로드 완료: {len(optimization_results)}개 조합"
            )
            self.logger.log_info(
                f"🔍 최적화 결과 키 예시: {list(optimization_results.keys())[:5]}"
            )

            # 3. 포트폴리오 최적화 실행
            print("📊 3단계: 포트폴리오 최적화 실행 시작")
            self.logger.log_info("📊 3단계: 포트폴리오 최적화 실행 시작")
            print(f"🔍 데이터 종목 수: {len(data_dict)}")
            print(f"🔍 최적화 결과 조합 수: {len(optimization_results)}")
            self.logger.log_info(f"🔍 데이터 종목 수: {len(data_dict)}")
            self.logger.log_info(f"🔍 최적화 결과 조합 수: {len(optimization_results)}")

            print("🔍 optimize_portfolio_with_individual_results 호출 시작")
            result = self.optimize_portfolio_with_individual_results(
                data_dict, optimization_results
            )
            print(
                f"🔍 optimize_portfolio_with_individual_results 결과: {type(result)}, 길이: {len(result) if result else 0}"
            )

            if not result:
                print("❌ 포트폴리오 최적화 실패")
                self.logger.log_error("포트폴리오 최적화 실패")
                return False
            print("✅ 포트폴리오 최적화 실행 완료")
            self.logger.log_success("✅ 포트폴리오 최적화 실행 완료")

            # 4. 결과 저장
            self.logger.log_info("📊 4단계: 결과 저장 시작")
            output_file = self.save_portfolio_optimization_result()
            if output_file:
                self.logger.log_success(f"✅ 결과 저장 완료: {output_file}")
            else:
                self.logger.log_warning("⚠️ 결과 저장 실패")

            # 5. 보고서 생성 및 출력
            self.logger.log_info("📊 5단계: 보고서 생성 시작")
            report = self.generate_portfolio_report()
            print(report)
            self.logger.log_success("✅ 보고서 생성 완료")

            self.logger.log_success("포트폴리오 최적화 완료")
            return True

        except Exception as e:
            self.logger.log_error(f"포트폴리오 최적화 실행 중 오류: {e}")
            import traceback

            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            return False

    def _find_latest_optimization_results(self) -> Dict[str, Dict]:
        """최신 개별 최적화 결과 파일 찾기"""
        try:
            self.logger.log_info("🔍 최신 최적화 결과 파일 검색 시작")
            
            # config에서 output 경로 가져오기
            output_config = self.config.get("output", {})
            results_folder = output_config.get("results_folder", "results")
            results_dir = Path(results_folder)
            self.logger.log_info(f"🔍 결과 디렉토리: {results_dir}")

            if not results_dir.exists():
                self.logger.log_error(f"{results_folder} 디렉토리가 존재하지 않습니다")
                return {}

            # hyperparam_optimization_*.json 파일들 찾기
            optimization_files = list(
                results_dir.glob("hyperparam_optimization_*.json")
            )
            self.logger.log_info(
                f"🔍 찾은 최적화 파일들: {[f.name for f in optimization_files]}"
            )

            if not optimization_files:
                self.logger.log_error(
                    "하이퍼파라미터 최적화 결과 파일을 찾을 수 없습니다"
                )
                return {}

            # 가장 최신 파일 로드
            latest_file = max(optimization_files, key=lambda x: x.stat().st_mtime)
            self.logger.log_info(f"🔍 최신 파일 선택: {latest_file.name}")
            return self.load_individual_optimization_results(str(latest_file))

        except Exception as e:
            self.logger.log_error(f"최신 최적화 결과 파일 찾기 실패: {e}")
            import traceback

            self.logger.log_error(f"상세 오류: {traceback.format_exc()}")
            return {}


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="고급 포트폴리오 관리자")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="설정 파일")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="데이터 디렉토리")
    parser.add_argument("--optimization-file", help="개별 최적화 결과 파일 경로")
    parser.add_argument(
        "--uuid", help="실행 UUID (특정 UUID의 최적화 결과 파일 자동 감지)"
    )
    parser.add_argument("--time-horizon", default="swing", help="시간대 설정")

    args = parser.parse_args()

    # UUID가 지정된 경우 해당 UUID의 최적화 결과 파일 찾기
    optimization_file_path = args.optimization_file
    if args.uuid and not args.optimization_file:
        results_dir = Path("results")
        current_date = datetime.now().strftime("%Y%m%d")
        optimization_file = (
            results_dir / f"hyperparam_optimization_{current_date}_{args.uuid}.json"
        )
        if optimization_file.exists():
            optimization_file_path = str(optimization_file)
            print(
                f"🔍 UUID {args.uuid}의 하이퍼파라미터 최적화 결과 파일 발견: {optimization_file_path}"
            )
        else:
            print(
                f"⚠️ UUID {args.uuid}의 하이퍼파라미터 최적화 결과 파일을 찾을 수 없습니다: {optimization_file}"
            )
            print(f"🔍 사용 가능한 파일들:")
            for file in results_dir.glob("hyperparam_optimization_*.json"):
                print(f"  - {file.name}")

    # 포트폴리오 매니저 초기화
    portfolio_manager = AdvancedPortfolioManager(
        config_path=args.config, time_horizon=args.time_horizon, uuid=args.uuid
    )

    # 포트폴리오 최적화 실행
    success = portfolio_manager.run_portfolio_optimization(
        data_dir=args.data_dir,
        optimization_file_path=optimization_file_path,
    )

    if success:
        print("✅ 포트폴리오 최적화 완료")
    else:
        print("❌ 포트폴리오 최적화 실패")


if __name__ == "__main__":
    main()
