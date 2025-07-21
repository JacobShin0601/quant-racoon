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
        # 포트폴리오 매니저 초기화 제거 - portfolio_results_path로 직접 로드

        self.optimization_results_path = optimization_results_path
        self.portfolio_results_path = portfolio_results_path
        self.results = {}
        self.logger = Logger()
        self.evaluation_start_time = datetime.now()
        self.execution_uuid = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        print(f"🔍 데이터 로드 시작 - data_dir: {self.data_dir}")
        print(f"🔍 심볼: {symbols}")

        # data_dir 인자를 직접 사용
        data_path = Path(self.data_dir)

        # data_dir이 존재하는지 확인
        if not data_path.exists():
            print(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_path}")
            return {}, {}

        print(f"🔍 데이터 디렉토리 사용: {data_path}")

        print(f"🔍 최종 검색 경로: {data_path}")

        data_dict = {}
        for symbol in symbols:
            print(f"🔍 {symbol} 데이터 파일 검색 중...")
            # 파일명 패턴 찾기
            pattern = f"{symbol}_*.csv"
            files = list(data_path.glob(pattern))

            if files:
                # 가장 최신 파일 선택
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                print(f"🔍 {symbol} 파일 로드: {latest_file}")
                df = pd.read_csv(latest_file)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
                data_dict[symbol] = df
                print(f"✅ {symbol} 데이터 로드: {latest_file.name} (행: {len(df)})")
            else:
                print(f"⚠️ {symbol} 데이터 파일을 찾을 수 없음")

        if not data_dict:
            self.logger.log_error(f"데이터를 로드할 수 없습니다: {data_path}")
            return {}, {}

        print(f"✅ 데이터 로드 완료: {len(data_dict)}개 종목")

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

    def load_portfolio_weights(self) -> Dict[str, float]:
        """포트폴리오 비중 로드"""
        try:
            weights_file = self._find_latest_weights_file()
            if not weights_file:
                self.logger.log_warning("포트폴리오 비중 파일을 찾을 수 없습니다")
                return {}

            with open(weights_file, "r", encoding="utf-8") as f:
                weights = json.load(f)

            self.logger.log_success(f"포트폴리오 비중 로드 완료: {weights_file}")
            return weights

        except Exception as e:
            self.logger.log_error(f"포트폴리오 비중 로드 실패: {e}")
            return {}

    def _find_latest_weights_file(self) -> Optional[str]:
        """최신 포트폴리오 비중 파일 찾기"""
        try:
            results_dir = Path("results")
            if not results_dir.exists():
                return None

            # portfolio_weights_*.json 파일들 찾기
            weights_files = list(results_dir.glob("portfolio_weights_*.json"))

            if not weights_files:
                self.logger.log_warning("포트폴리오 비중 파일을 찾을 수 없습니다")
                return None

            # 가장 최신 파일 반환
            latest_file = max(weights_files, key=lambda x: x.stat().st_mtime)
            self.logger.log_success(
                f"최신 포트폴리오 비중 파일 발견: {latest_file.name}"
            )
            return str(latest_file)

        except Exception as e:
            self.logger.log_error(f"포트폴리오 비중 파일 찾기 실패: {e}")
            return None

    def load_portfolio_results(self) -> Dict[str, Any]:
        """포트폴리오 최적화 결과 로드"""
        try:
            portfolio_file = self._find_latest_portfolio_file()
            if not portfolio_file:
                self.logger.log_warning(
                    "포트폴리오 최적화 결과 파일을 찾을 수 없습니다"
                )
                return {}

            with open(portfolio_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            # 수익률 데이터 복원 (JSON에서 DataFrame으로)
            if "returns_data" in results:
                returns_data = results["returns_data"]
                returns_df = pd.DataFrame(
                    returns_data["values"],
                    index=returns_data["index"],
                    columns=returns_data["columns"],
                )
                results["returns_data"] = returns_df

            self.logger.log_success(f"포트폴리오 결과 로드 완료: {portfolio_file}")
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
            print(f"🔍 전략 평가 시작: {strategy_name}")
            print(f"🔍 데이터 종목: {list(data_dict.keys())}")
            print(f"🔍 파라미터: {optimized_params}")

            # 전략 인스턴스 생성
            strategy = self.strategy_manager.strategies.get(strategy_name)
            if not strategy:
                print(f"❌ 전략을 찾을 수 없습니다: {strategy_name}")
                self.logger.log_error(f"전략을 찾을 수 없습니다: {strategy_name}")
                return {}

            print(f"✅ 전략 인스턴스 생성 성공: {strategy}")

            # 최적화된 파라미터 적용
            for param_name, param_value in optimized_params.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)
                    print(f"  - 파라미터 설정: {param_name} = {param_value}")

            # 각 종목에 대해 전략 실행
            for symbol, data in data_dict.items():
                try:
                    print(f"  🔍 {symbol} 신호 생성 시작")
                    signals = strategy.generate_signals(data)
                    print(
                        f"  🔍 {symbol} 신호 생성 결과: {type(signals)}, shape: {getattr(signals, 'shape', None) if signals is not None else None}"
                    )

                    if signals is not None and not signals.empty:
                        print(f"  ✅ {symbol} 신호 생성 성공")
                        # 거래 시뮬레이션
                        print(f"  🔍 {symbol} 거래 시뮬레이션 시작")
                        result = self.simulator.simulate_trading(
                            data, signals, strategy_name
                        )
                        print(
                            f"  🔍 {symbol} 시뮬레이션 결과: {type(result)}, keys: {list(result.keys()) if result else None}"
                        )

                        # 시뮬레이션 결과 요약만 출력
                        if result:
                            print(f"  ✅ {symbol} 시뮬레이션 성공")
                            # 성과 지표 계산 - simulate_trading 결과 구조에 맞게 수정
                            results_data = result.get("results", {})
                            total_return = results_data.get("total_return", 0.0)
                            total_trades = results_data.get("total_trades", 0)

                            print(f"  🔍 {symbol} 결과 데이터: {results_data}")
                            print(f"  🔍 {symbol} 총 수익률: {total_return}")
                            print(f"  🔍 {symbol} 총 거래 수: {total_trades}")

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

                                    # 무위험 수익률 고려한 샤프 비율 계산
                                    risk_free_rate = 0.02 / 252  # 일간 무위험 수익률
                                    excess_return = mean_return - risk_free_rate
                                    # 연간화된 샤프 비율: (연간 초과수익률) / (연간 표준편차)
                                    sharpe_ratio = (
                                        (excess_return * 252)
                                        / (std_return * np.sqrt(252))
                                        if std_return > 0
                                        else 0
                                    )

                                    # 소르티노 비율 계산 (무위험 수익률 고려)
                                    negative_returns = returns_series[
                                        returns_series < 0
                                    ]
                                    if len(negative_returns) > 0:
                                        downside_deviation = negative_returns.std()
                                        # 연간화된 소르티노 비율: (연간 초과수익률) / (연간 하방표준편차)
                                        sortino_ratio = (
                                            (excess_return * 252)
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

                            # 누적 수익률 계산 (복리 효과 고려)
                            cumulative_return = self._calculate_cumulative_return(
                                result.get("trades", [])
                            )

                            results[symbol] = {
                                "total_return": total_return,  # 기존 total_return 유지 (거래별 수익률 합계)
                                "cumulative_return": cumulative_return,  # 누적 수익률 추가
                                "sharpe_ratio": sharpe_ratio,
                                "sortino_ratio": sortino_ratio,
                                "max_drawdown": max_drawdown,
                                "volatility": volatility,
                                "beta": beta,
                                "total_trades": total_trades,
                                "trades": result.get("trades", []),  # 거래 내역 추가
                                "strategy": strategy_name,  # 전략 이름 추가
                                "current_position": result.get(
                                    "current_position", 0
                                ),  # 현재 보유 상태 추가
                                "final_price": result.get(
                                    "final_price"
                                ),  # 최종 매수/매도 가격
                                "final_date": result.get(
                                    "final_date"
                                ),  # 최종 매수/매도 시점
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
        portfolio_results: Dict[str, Any] = None,
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

        print(f"🔍 Buy & Hold 데이터 생성:")
        print(f"  - TRAIN: {len(all_results['buy_hold_train'])}개 종목")
        print(f"  - TEST: {len(all_results['buy_hold_test'])}개 종목")
        if all_results["buy_hold_train"]:
            sample_symbol = list(all_results["buy_hold_train"].keys())[0]
            sample_data = all_results["buy_hold_train"][sample_symbol]
            print(
                f"  - 샘플 ({sample_symbol}): {sample_data.get('total_return', 0)*100:.2f}%"
            )

        # 최적화된 전략들 평가
        symbols = list(train_data_dict.keys())
        strategy_scores = []  # 전략별 점수 저장

        for symbol in symbols:
            # 해당 종목의 최적 전략 찾기
            best_strategy = None
            best_params = {}

            # 1. 포트폴리오 결과에서 전략 정보 확인 (우선순위)
            if portfolio_results and "symbol_strategies" in portfolio_results:
                symbol_strategies = portfolio_results["symbol_strategies"]
                if symbol in symbol_strategies:
                    best_strategy = symbol_strategies[symbol].get("strategy")
                    best_params = symbol_strategies[symbol].get("params", {})
                    print(f"🔍 {symbol} 전략 발견 (포트폴리오): {best_strategy}")

            # 2. 최적화 결과에서 해당 종목의 최적 전략 찾기 (fallback)
            if not best_strategy:
                found = False

                # 패턴 1: "strategy_symbol" 형태
                for key, result in optimization_results.items():
                    if key.endswith(f"_{symbol}"):
                        best_strategy = result.get("strategy_name")
                        best_params = result.get("best_params", {})
                        found = True
                        print(f"🔍 {symbol} 전략 발견 (패턴1): {best_strategy}")
                        break

                # 패턴 2: "symbol" 키로 직접 검색
                if not found:
                    for key, result in optimization_results.items():
                        if result.get("symbol") == symbol:
                            best_strategy = result.get("strategy_name")
                            best_params = result.get("best_params", {})
                            found = True
                            print(f"🔍 {symbol} 전략 발견 (패턴2): {best_strategy}")
                            break

                # 패턴 3: 키에 symbol이 포함된 경우
                if not found:
                    for key, result in optimization_results.items():
                        if symbol in key:
                            best_strategy = result.get("strategy_name")
                            best_params = result.get("best_params", {})
                            found = True
                            print(f"🔍 {symbol} 전략 발견 (패턴3): {best_strategy}")
                            break

            if not best_strategy:
                print(f"⚠️ {symbol}의 최적 전략을 찾을 수 없습니다")
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
            # 포트폴리오 비중 가져오기 (별도 파일에서 로드)
            portfolio_weights = self.load_portfolio_weights()
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
                portfolio_performance["train"] = self._calculate_real_portfolio_metrics(
                    individual_results["train"], portfolio_weights
                )

            # Test 포트폴리오 성과
            if individual_results["test"]:
                portfolio_performance["test"] = self._calculate_real_portfolio_metrics(
                    individual_results["test"], portfolio_weights
                )

            # Buy & Hold 포트폴리오 성과
            if individual_results["buy_hold_train"]:
                portfolio_performance["buy_hold_train"] = (
                    self._calculate_buy_hold_portfolio_metrics(
                        individual_results["buy_hold_train"], portfolio_weights
                    )
                )

            if individual_results["buy_hold_test"]:
                portfolio_performance["buy_hold_test"] = (
                    self._calculate_buy_hold_portfolio_metrics(
                        individual_results["buy_hold_test"], portfolio_weights
                    )
                )

        except Exception as e:
            self.logger.log_error(f"포트폴리오 성과 계산 중 오류: {e}")

        return portfolio_performance

    def _calculate_real_portfolio_metrics(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        portfolio_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """실제 포트폴리오 성과 지표 계산 (거래 내역 기반)"""
        try:
            # 1. 포트폴리오 누적 수익률 계산 (복리 효과 고려)
            portfolio_cumulative_return = self._calculate_portfolio_cumulative_return(
                individual_results, portfolio_weights
            )

            # 2. 포트폴리오 일별 수익률 계산
            portfolio_daily_returns = self._calculate_portfolio_daily_returns(
                individual_results, portfolio_weights
            )

            if not portfolio_daily_returns or len(portfolio_daily_returns) == 0:
                return {
                    "total_return": 0.0,
                    "cumulative_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                    "beta": 1.0,
                    "total_trades": 0,
                    "returns": [],
                }

            returns_series = pd.Series(portfolio_daily_returns)

            # 3. 기본 통계 계산
            mean_return = returns_series.mean()
            std_return = returns_series.std()

            # 4. 샤프 비율 계산 (연간화)
            risk_free_rate = 0.02 / 252  # 일간 무위험 수익률
            excess_return = mean_return - risk_free_rate
            sharpe_ratio = (
                (excess_return * 252) / (std_return * np.sqrt(252))
                if std_return > 0
                else 0
            )

            # 5. 소르티노 비율 계산
            negative_returns = returns_series[returns_series < 0]
            sortino_ratio = 0
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std()
                sortino_ratio = (
                    (excess_return * 252) / (downside_deviation * np.sqrt(252))
                    if downside_deviation > 0
                    else 0
                )

            # 6. 최대 낙폭 계산
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())

            # 7. 변동성 계산 (연간화)
            volatility = std_return * np.sqrt(252)

            # 8. 베타 계산 (간단히 1.0으로 설정)
            beta = 1.0

            # 9. 총 거래 수 계산
            total_trades = sum(
                individual_results[symbol].get("total_trades", 0)
                for symbol in individual_results.keys()
            )

            return {
                "total_return": portfolio_cumulative_return,  # 누적 수익률
                "cumulative_return": portfolio_cumulative_return,  # 누적 수익률 (동일)
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "beta": beta,
                "total_trades": total_trades,
                "returns": portfolio_daily_returns,
            }

        except Exception as e:
            self.logger.log_error(f"실제 포트폴리오 지표 계산 중 오류: {e}")
            return {
                "total_return": 0.0,
                "cumulative_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "beta": 1.0,
                "total_trades": 0,
                "returns": [],
            }

    def _calculate_portfolio_daily_returns(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        portfolio_weights: Dict[str, float],
    ) -> List[float]:
        """포트폴리오 일별 수익률 계산 (실제 일별 수익률)"""
        try:
            # 모든 거래 내역을 날짜별로 정렬
            all_trades = []
            for symbol, result in individual_results.items():
                weight = portfolio_weights.get(symbol, 0.0)
                trades = result.get("trades", [])

                for trade in trades:
                    # 거래 가중치 적용
                    weighted_trade = trade.copy()
                    # PnL이 이미 수익률로 저장되어 있음
                    pnl_rate = trade.get("pnl", 0)
                    weighted_trade["pnl_rate"] = pnl_rate * weight
                    weighted_trade["symbol"] = symbol
                    weighted_trade["weight"] = weight
                    weighted_trade["exit_time"] = trade.get("exit_time")
                    all_trades.append(weighted_trade)

            if not all_trades:
                return []

            # 거래를 날짜순으로 정렬 (exit_time이 숫자인 경우 처리)
            def get_sort_key(trade):
                exit_time = trade.get("exit_time")
                if isinstance(exit_time, (int, float)) and exit_time is not None:
                    return exit_time
                elif isinstance(exit_time, str):
                    try:
                        # 문자열 날짜를 숫자로 변환 시도
                        return float(exit_time)
                    except:
                        return 0
                else:
                    return 0

            all_trades.sort(key=get_sort_key)

            # 거래가 발생한 날짜들 추출
            trade_days = set()
            for trade in all_trades:
                exit_time = trade.get("exit_time")
                if isinstance(exit_time, (int, float)) and exit_time is not None:
                    trade_days.add(int(exit_time))

            if not trade_days:
                return []

            # 전체 기간 설정 (최대 거래일 + 1)
            max_day = max(trade_days)
            total_days = max_day + 1

            # 일별 포트폴리오 수익률 계산
            daily_returns = []
            current_portfolio_value = 1.0  # 초기값 1.0 (100%)

            for day in range(total_days):
                if day in trade_days:
                    # 해당 날짜의 모든 거래 찾기
                    day_trades = [
                        t
                        for t in all_trades
                        if isinstance(t.get("exit_time"), (int, float))
                        and t.get("exit_time") is not None
                        and int(t.get("exit_time")) == day
                    ]

                    # 해당 날짜의 총 수익률 계산
                    day_total_pnl = sum(t.get("pnl_rate", 0) for t in day_trades)
                    if day_total_pnl != 0:
                        daily_return = day_total_pnl / current_portfolio_value
                        current_portfolio_value *= 1 + daily_return
                    else:
                        daily_return = 0.0
                else:
                    # 거래가 없는 날은 수익률 0
                    daily_return = 0.0

                daily_returns.append(daily_return)

            return daily_returns

        except Exception as e:
            self.logger.log_error(f"포트폴리오 일별 수익률 계산 중 오류: {e}")
            return []

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

    def _calculate_cumulative_return(self, trades: List[Dict]) -> float:
        """거래 내역에서 누적 수익률 계산 (복리 효과 고려)"""
        if not trades:
            return 0.0

        cumulative_return = 1.0  # 1.0 = 100%
        for trade in trades:
            pnl = trade.get("pnl", 0)
            cumulative_return *= 1 + pnl  # 복리 효과 적용

        return cumulative_return - 1.0  # 백분율로 변환 (0.045 → 0.045 = 4.5%)

    def _calculate_portfolio_cumulative_return(
        self, individual_results: Dict[str, Dict], weights: Dict[str, float]
    ) -> float:
        """포트폴리오 누적 수익률 계산 (복리 효과 고려)"""
        try:
            # 포트폴리오 일별 수익률을 계산하여 누적 수익률 도출
            portfolio_daily_returns = self._calculate_portfolio_daily_returns(
                individual_results, weights
            )

            if not portfolio_daily_returns:
                return 0.0

            # 누적 수익률 계산 (복리 효과 고려)
            cumulative_return = (1 + pd.Series(portfolio_daily_returns)).prod() - 1

            return cumulative_return

        except Exception as e:
            self.logger.log_error(f"포트폴리오 누적 수익률 계산 중 오류: {e}")
            return 0.0

    def _calculate_buy_hold_return(
        self, individual_results: Dict[str, Dict], weights: Dict[str, float]
    ) -> float:
        """BUY&HOLD 수익률 계산 (가격 변화 기반)"""
        if not individual_results:
            return 0.0

        # 각 종목의 BUY&HOLD 수익률 계산
        symbol_returns = {}
        for symbol, data in individual_results.items():
            # BUY&HOLD는 total_return을 사용 (거래 없이 가격 변화만)
            symbol_returns[symbol] = data.get("total_return", 0.0)

        # 포트폴리오 BUY&HOLD 수익률 계산 (가중 평균)
        buy_hold_return = sum(
            symbol_returns[symbol] * weights.get(symbol, 0.0)
            for symbol in individual_results.keys()
        )

        return buy_hold_return

    def _calculate_individual_buy_hold_return(
        self, symbol: str, data_dict: Dict[str, pd.DataFrame]
    ) -> float:
        """개별 종목의 Buy & Hold 수익률 계산"""
        if symbol not in data_dict:
            return 0

        df = data_dict[symbol]
        if len(df) < 2:
            return 0

        # 시작가격과 종가
        start_price = df.iloc[0]["close"]
        end_price = df.iloc[-1]["close"]

        # Buy & Hold 수익률 계산
        buy_hold_return = (end_price - start_price) / start_price
        return buy_hold_return

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
                    # 누적 수익률 사용
                    cumulative_return = (
                        train_result.get(
                            "cumulative_return", train_result.get("total_return", 0)
                        )
                        * 100
                    )
                    f.write(
                        f"{symbol:<15} {strategy:<20} {cumulative_return:>8.2f}% {train_result['sharpe_ratio']:>6.3f} {train_result['sortino_ratio']:>8.3f} {train_result['max_drawdown']*100:>8.2f}% {train_result['volatility']*100:>8.2f}% {train_result.get('beta', 1.0):>5.2f} {train_result['total_trades']:>6} [{composite_score:>6.1f}]\n"
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
                    # 누적 수익률 사용
                    cumulative_return = (
                        test_result.get(
                            "cumulative_return", test_result.get("total_return", 0)
                        )
                        * 100
                    )
                    f.write(
                        f"{symbol:<15} {strategy:<20} {cumulative_return:>8.2f}% {test_result['sharpe_ratio']:>6.3f} {test_result['sortino_ratio']:>8.3f} {test_result['max_drawdown']*100:>8.2f}% {test_result['volatility']*100:>8.2f}% {test_result.get('beta', 1.0):>5.2f} {test_result['total_trades']:>6} [{composite_score:>6.1f}]\n"
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
            print("🔍 1단계: 데이터 로드 및 분할 시작")
            train_data_dict, test_data_dict = self.load_data_and_split(symbols)
            print(
                f"🔍 데이터 로드 결과: train={len(train_data_dict) if train_data_dict else 0}, test={len(test_data_dict) if test_data_dict else 0}"
            )
            if not train_data_dict or not test_data_dict:
                print("❌ 데이터 로드 실패")
                return {}

                # 2. 최적화 결과 로드
            print("🔍 2단계: 최적화 결과 로드 시작")
            optimization_results = self.load_optimization_results()
            print(
                f"🔍 최적화 결과 로드: {len(optimization_results) if optimization_results else 0}개"
            )
            if not optimization_results:
                print("❌ 최적화 결과를 찾을 수 없습니다.")
                return {}

            # 3. 포트폴리오 결과 로드
            print("🔍 3단계: 포트폴리오 결과 로드 시작")
            portfolio_results = self.load_portfolio_results()
            print(
                f"🔍 포트폴리오 결과 로드: {len(portfolio_results) if portfolio_results else 0}개 키"
            )
            if not portfolio_results:
                print("⚠️ 포트폴리오 결과를 찾을 수 없어 기본값 사용")
                portfolio_results = {
                    "portfolio_weights": {},
                    "portfolio_performance": {},
                }

            # 4. 전략별 Train/Test 성과 평가
            individual_results = self.evaluate_all_strategies(
                train_data_dict, test_data_dict, optimization_results, portfolio_results
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

            # 6. 거래 내역 로그 저장
            self.save_transaction_logs(
                individual_results, train_data_dict, test_data_dict
            )

            # 7. 성과 요약 테이블 출력
            self._print_performance_summary(
                individual_results,
                portfolio_performance,
                portfolio_weights,
                train_data_dict,
                test_data_dict,
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
        train_data_dict: Dict[str, pd.DataFrame] = None,
        test_data_dict: Dict[str, pd.DataFrame] = None,
    ):
        """성과 요약 테이블 출력"""
        # 데이터 기간 정보 출력
        self._print_data_period_info(train_data_dict, test_data_dict)

        # TRAIN 포트폴리오 리스크 지표 테이블
        print("\n" + "=" * 100)
        print("📊 TRAIN 포트폴리오 리스크 지표")
        print("=" * 100)
        self._print_portfolio_risk_table(
            "TRAIN",
            portfolio_performance,
            portfolio_weights,
        )

        # TRAIN 종목별 성과 테이블
        print("\n" + "=" * 100)
        print("📊 TRAIN 종목별 성과")
        print("=" * 100)
        self._print_individual_performance_table(
            "TRAIN",
            individual_results,
            portfolio_performance,
            portfolio_weights,
            train_data_dict,
            test_data_dict,
        )

        # TEST 포트폴리오 리스크 지표 테이블
        print("\n" + "=" * 100)
        print("📊 TEST 포트폴리오 리스크 지표")
        print("=" * 100)
        self._print_portfolio_risk_table(
            "TEST",
            portfolio_performance,
            portfolio_weights,
        )

        # TEST 종목별 성과 테이블
        print("\n" + "=" * 100)
        print("📊 TEST 종목별 성과")
        print("=" * 100)
        self._print_individual_performance_table(
            "TEST",
            individual_results,
            portfolio_performance,
            portfolio_weights,
            train_data_dict,
            test_data_dict,
        )

        print("=" * 100)

        # 종목별 end_date 주가 테이블 추가
        self._print_end_date_price_table(test_data_dict, portfolio_weights)

    def save_transaction_logs(
        self,
        individual_results: Dict[str, Any],
        train_data_dict: Dict[str, pd.DataFrame] = None,
        test_data_dict: Dict[str, pd.DataFrame] = None,
    ):
        """개별 종목별 거래 내역을 로그 파일로 저장"""
        try:
            # 로그 디렉토리 생성
            log_dir = Path("log")
            log_dir.mkdir(exist_ok=True)

            today = datetime.now().strftime("%Y%m%d")

            # Train 거래 내역 저장
            if train_data_dict:
                train_log_path = (
                    log_dir
                    / f"transaction_train_swing_{today}_{self.execution_uuid}.log"
                )
                self._save_period_transaction_log(
                    individual_results.get("train", {}),
                    train_data_dict,
                    train_log_path,
                    "TRAIN",
                )

            # Test 거래 내역 저장
            if test_data_dict:
                test_log_path = (
                    log_dir
                    / f"transaction_test_swing_{today}_{self.execution_uuid}.log"
                )
                self._save_period_transaction_log(
                    individual_results.get("test", {}),
                    test_data_dict,
                    test_log_path,
                    "TEST",
                )

        except Exception as e:
            self.logger.log_error(f"거래 내역 로그 저장 실패: {e}")

    def _save_period_transaction_log(
        self,
        period_results: Dict[str, Any],
        data_dict: Dict[str, pd.DataFrame],
        log_path: Path,
        period_name: str,
    ):
        """특정 기간의 거래 내역을 로그 파일로 저장"""
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== {period_name} 거래 내역 로그 ===\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"실행 UUID: {self.execution_uuid}\n")
                f.write("=" * 80 + "\n\n")

                for symbol, data in data_dict.items():
                    if symbol in period_results:
                        result = period_results[symbol]
                        strategy = result.get("strategy", "UNKNOWN")
                        trades = result.get("trades", [])

                        f.write(f"📊 {symbol} ({strategy})\n")
                        f.write("-" * 50 + "\n")

                        if trades:
                            f.write(f"총 거래 수: {len(trades)}\n")
                            f.write(
                                f"수익률: {result.get('cumulative_return', result.get('total_return', 0))*100:.2f}%\n"
                            )
                            f.write(f"샤프 비율: {result.get('sharpe_ratio', 0):.3f}\n")
                            f.write(
                                f"소르티노 비율: {result.get('sortino_ratio', 0):.3f}\n\n"
                            )

                            f.write("거래 내역:\n")
                            f.write(
                                f"{'날짜':<20} {'시간':<10} {'타입':<6} {'가격':<10} {'수량':<8} {'수익률':<10} {'누적수익률':<12}\n"
                            )
                            f.write("-" * 80 + "\n")

                            cumulative_return = 0
                            for trade in trades:
                                entry_time = trade.get("entry_time", "")
                                exit_time = trade.get("exit_time", "")
                                entry_price = trade.get("entry_price", 0)
                                exit_price = trade.get("exit_price", 0)
                                shares = trade.get("shares", 0)
                                pnl = trade.get("pnl", 0)  # pnl 키 사용
                                cumulative_return += pnl

                                # 시간 정보 처리
                                entry_time_str = str(entry_time) if entry_time else ""

                                # exit_time이 숫자인 경우 날짜로 변환
                                if isinstance(exit_time, (int, float)):
                                    # 데이터프레임에서 해당 인덱스의 날짜 가져오기
                                    if (
                                        symbol in data_dict
                                        and not data_dict[symbol].empty
                                    ):
                                        try:
                                            if exit_time < len(data_dict[symbol]):
                                                exit_date = data_dict[symbol].index[
                                                    int(exit_time)
                                                ]
                                                if hasattr(exit_date, "strftime"):
                                                    exit_time_str = exit_date.strftime(
                                                        "%Y-%m-%d"
                                                    )
                                                else:
                                                    exit_time_str = str(exit_date)
                                            else:
                                                exit_time_str = str(exit_time)
                                        except:
                                            exit_time_str = str(exit_time)
                                    else:
                                        exit_time_str = str(exit_time)
                                else:
                                    exit_time_str = str(exit_time) if exit_time else ""

                                # 매수 거래
                                if entry_time:
                                    f.write(
                                        f"{entry_time_str:<20} {'매수':<10} {entry_price:<10.2f} {shares:<8.2f} {'':<10} {'':<12}\n"
                                    )

                                # 매도 거래
                                if exit_time:
                                    f.write(
                                        f"{exit_time_str:<20} {'매도':<10} {exit_price:<10.2f} {shares:<8.2f} {pnl*100:<10.2f}% {cumulative_return*100:<12.2f}%\n"
                                    )
                        else:
                            f.write("거래 내역 없음\n")

                        f.write("\n" + "=" * 80 + "\n\n")

        except Exception as e:
            self.logger.log_error(f"{period_name} 거래 내역 로그 저장 실패: {e}")

    def _print_data_period_info(
        self,
        train_data_dict: Dict[str, pd.DataFrame] = None,
        test_data_dict: Dict[str, pd.DataFrame] = None,
    ):
        """데이터 기간 정보 출력"""
        print("\n" + "=" * 100)
        print("📅 데이터 기간 정보")
        print("=" * 100)

        if train_data_dict:
            train_start = None
            train_end = None
            train_symbols = []

            for symbol, data in train_data_dict.items():
                if not data.empty:
                    symbol_start = data.index[0]
                    symbol_end = data.index[-1]

                    # 인덱스가 datetime인지 확인
                    if hasattr(symbol_start, "strftime"):
                        if train_start is None or symbol_start < train_start:
                            train_start = symbol_start
                        if train_end is None or symbol_end > train_end:
                            train_end = symbol_end
                        train_symbols.append(symbol)

            if train_start and train_end and hasattr(train_start, "strftime"):
                print(
                    f"📊 TRAIN 기간: {train_start.strftime('%Y-%m-%d %H:%M')} ~ {train_end.strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"📊 TRAIN 종목 수: {len(train_symbols)}개")
                print(f"📊 TRAIN 종목: {', '.join(train_symbols)}")
            else:
                print(f"📊 TRAIN 종목 수: {len(train_data_dict)}개")
                print(f"📊 TRAIN 종목: {', '.join(list(train_data_dict.keys()))}")

        if test_data_dict:
            test_start = None
            test_end = None
            test_symbols = []

            for symbol, data in test_data_dict.items():
                if not data.empty:
                    symbol_start = data.index[0]
                    symbol_end = data.index[-1]

                    # 인덱스가 datetime인지 확인
                    if hasattr(symbol_start, "strftime"):
                        if test_start is None or symbol_start < test_start:
                            test_start = symbol_start
                        if test_end is None or symbol_end > test_end:
                            test_end = symbol_end
                        test_symbols.append(symbol)

            if test_start and test_end and hasattr(test_start, "strftime"):
                print(
                    f"📊 TEST 기간: {test_start.strftime('%Y-%m-%d %H:%M')} ~ {test_end.strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"📊 TEST 종목 수: {len(test_symbols)}개")
                print(f"📊 TEST 종목: {', '.join(test_symbols)}")
            else:
                print(f"📊 TEST 종목 수: {len(test_data_dict)}개")
                print(f"📊 TEST 종목: {', '.join(list(test_data_dict.keys()))}")

        print("=" * 100)

    def _print_performance_table(
        self,
        period: str,
        individual_results: Dict[str, Any],
        portfolio_performance: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        train_data_dict: Dict[str, pd.DataFrame] = None,
        test_data_dict: Dict[str, pd.DataFrame] = None,
    ):
        """성과 테이블 출력"""
        # 헤더 출력
        print(
            f"{'종목':<8} {'비중':<6} {'수익률':<8} {'B&H':<6} {'샤프':<6} {'소르티노':<8} {'거래수':<6} {'보유':<4} {'매수/매도가격':<12} {'최종시점':<12} {'전략':<20}"
        )
        print("-" * 138)

        # Buy & Hold 성과 (포트폴리오 비중 기준)
        buy_hold_data = individual_results.get(f"buy_hold_{period.lower()}", {})
        if buy_hold_data:
            # Buy & Hold 수익률 계산 (가격 변화 기반)
            buy_hold_return = self._calculate_buy_hold_return(
                buy_hold_data, portfolio_weights
            )

            total_sharpe = 0
            total_sortino = 0
            total_trades = 0
            symbol_count = 0

            for symbol, weight in portfolio_weights.items():
                if symbol in buy_hold_data:
                    data = buy_hold_data[symbol]
                    total_sharpe += data.get("sharpe_ratio", 0) * weight
                    total_sortino += data.get("sortino_ratio", 0) * weight
                    total_trades += data.get("total_trades", 0)
                    symbol_count += 1

            if symbol_count > 0:
                print(
                    f"{'BUY&HOLD':<8} {'100%':<6} {buy_hold_return*100:>7.2f}% {'':<6} {total_sharpe:>5.3f} {total_sortino:>7.3f} {total_trades:>5} {'Y':<4} {'':<12} {'':<12} {'PASSIVE':<20}"
                )

        # 포트폴리오 성과
        portfolio_data = portfolio_performance.get(period.lower(), {})
        if portfolio_data:
            # 포트폴리오 누적 수익률 계산
            individual_data = individual_results.get(period.lower(), {})
            portfolio_cumulative_return = self._calculate_portfolio_cumulative_return(
                individual_data, portfolio_weights
            )

            portfolio_score = self._calculate_portfolio_score(portfolio_data)
            print(
                f"{'PORTFOLIO':<8} {'100%':<6} {portfolio_cumulative_return*100:>7.2f}% {'':<6} {portfolio_data.get('sharpe_ratio', 0):>5.3f} {portfolio_data.get('sortino_ratio', 0):>7.3f} {portfolio_data.get('total_trades', 0):>5} {'Y':<4} {'':<12} {'':<12} {'OPTIMIZED':<20} [{portfolio_score:>6.1f}]"
            )

        print("-" * 138)

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
                    trades_list = data.get("trades", [])

                    # 누적 수익률 사용 (새로 추가된 필드)
                    cumulative_return = data.get("cumulative_return", 0) * 100

                    sharpe = data.get("sharpe_ratio", 0)
                    sortino = data.get("sortino_ratio", 0)
                    trades = data.get("total_trades", 0)

                    # 현재 보유 상태 판단 (거래 시뮬레이터 결과에서 가져오기)
                    current_position = data.get("current_position", 0)
                    holding = "Y" if current_position > 0 else "N"

                    # 최종 매수/매도 가격 및 시점
                    final_price = data.get("final_price")
                    final_date = data.get("final_date")

                    price_info = ""
                    date_info = ""

                    if final_price is not None:
                        if holding == "Y":
                            price_info = f"매수:{final_price:.2f}"
                        else:
                            price_info = f"매도:{final_price:.2f}"

                    # 날짜 정보 처리 - 거래 내역에서 마지막 거래 날짜 확인
                    trades_list = data.get("trades", [])
                    if trades_list:
                        last_trade = trades_list[-1]

                        # 매도 완료된 경우 exit_time 사용
                        if last_trade.get("exit_time") is not None:
                            exit_time = last_trade.get("exit_time")
                            if hasattr(exit_time, "strftime"):
                                date_info = exit_time.strftime("%Y-%m-%d")
                            elif isinstance(exit_time, pd.Timestamp):
                                date_info = exit_time.strftime("%Y-%m-%d")
                            elif isinstance(exit_time, (int, float)):
                                # 인덱스 번호를 실제 날짜로 변환
                                try:
                                    data_dict = (
                                        train_data_dict
                                        if period.upper() == "TRAIN"
                                        else test_data_dict
                                    )
                                    if symbol in data_dict:
                                        df = data_dict[symbol]
                                        if 0 <= exit_time < len(df):
                                            actual_date = df.iloc[exit_time]["date"]
                                            if hasattr(actual_date, "strftime"):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            elif isinstance(actual_date, pd.Timestamp):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            else:
                                                date_info = str(actual_date)[:10]
                                except Exception as e:
                                    # 디버깅을 위한 로그 (필요시 주석 해제)
                                    # print(f"날짜 변환 오류 ({symbol}): {e}")
                                    date_info = ""
                            else:
                                date_info = str(exit_time)[:10]

                        # 매수만 하고 매도하지 않은 경우 entry_time 사용
                        elif last_trade.get("entry_time") is not None:
                            entry_time = last_trade.get("entry_time")
                            if hasattr(entry_time, "strftime"):
                                date_info = entry_time.strftime("%Y-%m-%d")
                            elif isinstance(entry_time, pd.Timestamp):
                                date_info = entry_time.strftime("%Y-%m-%d")
                            elif isinstance(entry_time, (int, float)):
                                # 인덱스 번호를 실제 날짜로 변환
                                try:
                                    data_dict = (
                                        train_data_dict
                                        if period.upper() == "TRAIN"
                                        else test_data_dict
                                    )
                                    if symbol in data_dict:
                                        df = data_dict[symbol]
                                        if 0 <= entry_time < len(df):
                                            actual_date = df.iloc[entry_time]["date"]
                                            if hasattr(actual_date, "strftime"):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            elif isinstance(actual_date, pd.Timestamp):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            else:
                                                date_info = str(actual_date)[:10]
                                except Exception as e:
                                    # 디버깅을 위한 로그 (필요시 주석 해제)
                                    # print(f"날짜 변환 오류 ({symbol}): {e}")
                                    date_info = ""
                            else:
                                date_info = str(entry_time)[:10]

                    # final_date가 있고 위에서 날짜를 찾지 못한 경우
                    elif final_date is not None:
                        if hasattr(final_date, "strftime"):
                            date_info = final_date.strftime("%Y-%m-%d")
                        elif isinstance(final_date, pd.Timestamp):
                            date_info = final_date.strftime("%Y-%m-%d")
                        else:
                            date_info = str(final_date)[:10]

                    # 날짜 정보가 여전히 비어있는 경우, 시뮬레이션 종료 날짜 사용
                    if not date_info:
                        try:
                            data_dict = (
                                train_data_dict
                                if period.upper() == "TRAIN"
                                else test_data_dict
                            )
                            if symbol in data_dict:
                                df = data_dict[symbol]
                                if len(df) > 0:
                                    # 시뮬레이션 마지막 날짜 (시뮬레이션 종료 시점)
                                    # datetime 또는 date 컬럼 찾기
                                    date_column = None
                                    for col in ["datetime", "date", "Date", "DateTime"]:
                                        if col in df.columns:
                                            date_column = col
                                            break

                                    if date_column:
                                        last_date = df.iloc[-1][date_column]
                                        if hasattr(last_date, "strftime"):
                                            date_info = last_date.strftime("%Y-%m-%d")
                                        elif isinstance(last_date, pd.Timestamp):
                                            date_info = last_date.strftime("%Y-%m-%d")
                                        else:
                                            date_info = str(last_date)[:10]
                        except Exception as e:
                            # 조용히 처리 (오류 로그 제거)
                            date_info = ""

                    # Buy & Hold 수익률 계산
                    data_dict = (
                        train_data_dict if period.upper() == "TRAIN" else test_data_dict
                    )
                    buy_hold_return = (
                        self._calculate_individual_buy_hold_return(symbol, data_dict)
                        * 100
                    )

                    print(
                        f"{symbol:<8} {weight*100:>5.1f}% {cumulative_return:>7.2f}% {buy_hold_return:>5.2f}% {sharpe:>5.3f} {sortino:>7.3f} {trades:>5} {holding:<4} {price_info:<12} {date_info:<12} {strategy:<20}"
                    )

    def _print_portfolio_risk_table(
        self,
        period: str,
        portfolio_performance: Dict[str, Any],
        portfolio_weights: Dict[str, float],
    ):
        """포트폴리오 리스크 지표 테이블 출력"""
        # 헤더 출력
        print(
            f"{'구분':<15} {'수익률':<10} {'샤프':<8} {'소르티노':<10} {'칼마':<8} {'MDD':<8} {'VaR(95%)':<10} {'CVaR(95%)':<12} {'변동성':<10} {'베타':<6} {'정보비율':<10}"
        )
        print("-" * 100)

        # 포트폴리오 성과 데이터
        portfolio_data = portfolio_performance.get(period.lower(), {})
        if portfolio_data:
            # 포트폴리오 누적 수익률 계산 (복리 효과 고려)
            cumulative_return = (
                portfolio_data.get(
                    "cumulative_return", portfolio_data.get("total_return", 0)
                )
                * 100
            )
            sharpe_ratio = portfolio_data.get("sharpe_ratio", 0)
            sortino_ratio = portfolio_data.get("sortino_ratio", 0)
            max_drawdown = portfolio_data.get("max_drawdown", 0) * 100
            volatility = portfolio_data.get("volatility", 0) * 100
            beta = portfolio_data.get("beta", 1.0)

            # 칼마 비율 계산 (누적 수익률 사용)
            calmar_ratio = 0
            if max_drawdown > 0:
                calmar_ratio = cumulative_return / max_drawdown

            # VaR 및 CVaR 계산 (간단한 구현)
            var_95 = self._calculate_var(portfolio_data, 0.95) * 100
            cvar_95 = self._calculate_cvar(portfolio_data, 0.95) * 100

            # 정보 비율 계산 (무위험 수익률 대비 초과수익률)
            risk_free_rate = 0.02  # 2% 연간 무위험 수익률
            excess_return = cumulative_return - (risk_free_rate * 100)
            information_ratio = excess_return / volatility if volatility > 0 else 0

            print(
                f"{'OPTIMIZED':<15} {cumulative_return:>8.2f}% {sharpe_ratio:>6.3f} {sortino_ratio:>8.3f} {calmar_ratio:>6.3f} {max_drawdown:>6.2f}% {var_95:>8.2f}% {cvar_95:>10.2f}% {volatility:>8.2f}% {beta:>5.2f} {information_ratio:>8.3f}"
            )

            # Buy & Hold 성과 데이터
        buy_hold_data = portfolio_performance.get(f"buy_hold_{period.lower()}", {})
        if buy_hold_data:
            cumulative_return = (
                buy_hold_data.get(
                    "cumulative_return", buy_hold_data.get("total_return", 0)
                )
                * 100
            )
            sharpe_ratio = buy_hold_data.get("sharpe_ratio", 0)
            sortino_ratio = buy_hold_data.get("sortino_ratio", 0)
            max_drawdown = buy_hold_data.get("max_drawdown", 0) * 100
            volatility = buy_hold_data.get("volatility", 0) * 100
            beta = buy_hold_data.get("beta", 1.0)

            # 칼마 비율 계산 (누적 수익률 사용)
            calmar_ratio = 0
            if max_drawdown > 0:
                calmar_ratio = cumulative_return / max_drawdown

            # VaR 및 CVaR 계산
            var_95 = self._calculate_var(buy_hold_data, 0.95) * 100
            cvar_95 = self._calculate_cvar(buy_hold_data, 0.95) * 100

            # 정보 비율 계산
            risk_free_rate = 0.02
            excess_return = cumulative_return - (risk_free_rate * 100)
            information_ratio = excess_return / volatility if volatility > 0 else 0

            print(
                f"{'BUY&HOLD':<15} {cumulative_return:>8.2f}% {sharpe_ratio:>6.3f} {sortino_ratio:>8.3f} {calmar_ratio:>6.3f} {max_drawdown:>6.2f}% {var_95:>8.2f}% {cvar_95:>10.2f}% {volatility:>8.2f}% {beta:>5.2f} {information_ratio:>8.3f}"
            )

        print("-" * 100)

    def _print_individual_performance_table(
        self,
        period: str,
        individual_results: Dict[str, Any],
        portfolio_performance: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        train_data_dict: Dict[str, pd.DataFrame] = None,
        test_data_dict: Dict[str, pd.DataFrame] = None,
    ):
        """종목별 성과 테이블 출력"""
        # 헤더 출력
        print(
            f"{'종목':<8} {'비중':<6} {'수익률':<8} {'B&H':<6} {'샤프':<6} {'소르티노':<8} {'거래수':<6} {'보유':<4} {'매수/매도가격':<12} {'최종시점':<12} {'전략':<20}"
        )
        print("-" * 138)

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
                    trades_list = data.get("trades", [])

                    # 누적 수익률 사용 (새로 추가된 필드)
                    cumulative_return = data.get("cumulative_return", 0) * 100

                    sharpe = data.get("sharpe_ratio", 0)
                    sortino = data.get("sortino_ratio", 0)
                    trades = data.get("total_trades", 0)

                    # 현재 보유 상태 판단 (거래 시뮬레이터 결과에서 가져오기)
                    current_position = data.get("current_position", 0)
                    holding = "Y" if current_position > 0 else "N"

                    # 최종 매수/매도 가격 및 시점
                    final_price = data.get("final_price")
                    final_date = data.get("final_date")

                    price_info = ""
                    date_info = ""

                    if final_price is not None:
                        if holding == "Y":
                            price_info = f"매수:{final_price:.2f}"
                        else:
                            price_info = f"매도:{final_price:.2f}"

                    # 날짜 정보 처리 - 거래 내역에서 마지막 거래 날짜 확인
                    trades_list = data.get("trades", [])
                    if trades_list:
                        last_trade = trades_list[-1]

                        # 매도 완료된 경우 exit_time 사용
                        if last_trade.get("exit_time") is not None:
                            exit_time = last_trade.get("exit_time")
                            if hasattr(exit_time, "strftime"):
                                date_info = exit_time.strftime("%Y-%m-%d")
                            elif isinstance(exit_time, pd.Timestamp):
                                date_info = exit_time.strftime("%Y-%m-%d")
                            elif isinstance(exit_time, (int, float)):
                                # 인덱스 번호를 실제 날짜로 변환
                                try:
                                    data_dict = (
                                        train_data_dict
                                        if period.upper() == "TRAIN"
                                        else test_data_dict
                                    )
                                    if symbol in data_dict:
                                        df = data_dict[symbol]
                                        if 0 <= exit_time < len(df):
                                            actual_date = df.iloc[exit_time]["date"]
                                            if hasattr(actual_date, "strftime"):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            elif isinstance(actual_date, pd.Timestamp):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            else:
                                                date_info = str(actual_date)[:10]
                                except Exception as e:
                                    date_info = ""
                            else:
                                date_info = str(exit_time)[:10]

                        # 매수만 하고 매도하지 않은 경우 entry_time 사용
                        elif last_trade.get("entry_time") is not None:
                            entry_time = last_trade.get("entry_time")
                            if hasattr(entry_time, "strftime"):
                                date_info = entry_time.strftime("%Y-%m-%d")
                            elif isinstance(entry_time, pd.Timestamp):
                                date_info = entry_time.strftime("%Y-%m-%d")
                            elif isinstance(entry_time, (int, float)):
                                # 인덱스 번호를 실제 날짜로 변환
                                try:
                                    data_dict = (
                                        train_data_dict
                                        if period.upper() == "TRAIN"
                                        else test_data_dict
                                    )
                                    if symbol in data_dict:
                                        df = data_dict[symbol]
                                        if 0 <= entry_time < len(df):
                                            actual_date = df.iloc[entry_time]["date"]
                                            if hasattr(actual_date, "strftime"):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            elif isinstance(actual_date, pd.Timestamp):
                                                date_info = actual_date.strftime(
                                                    "%Y-%m-%d"
                                                )
                                            else:
                                                date_info = str(actual_date)[:10]
                                except Exception as e:
                                    date_info = ""
                            else:
                                date_info = str(entry_time)[:10]

                    # final_date가 있고 위에서 날짜를 찾지 못한 경우
                    elif final_date is not None:
                        if hasattr(final_date, "strftime"):
                            date_info = final_date.strftime("%Y-%m-%d")
                        elif isinstance(final_date, pd.Timestamp):
                            date_info = final_date.strftime("%Y-%m-%d")
                        else:
                            date_info = str(final_date)[:10]

                    # 날짜 정보가 여전히 비어있는 경우, 시뮬레이션 종료 날짜 사용
                    if not date_info:
                        try:
                            data_dict = (
                                train_data_dict
                                if period.upper() == "TRAIN"
                                else test_data_dict
                            )
                            if symbol in data_dict:
                                df = data_dict[symbol]
                                if len(df) > 0:
                                    # 시뮬레이션 마지막 날짜 (시뮬레이션 종료 시점)
                                    # datetime 또는 date 컬럼 찾기
                                    date_column = None
                                    for col in ["datetime", "date", "Date", "DateTime"]:
                                        if col in df.columns:
                                            date_column = col
                                            break

                                    if date_column:
                                        last_date = df.iloc[-1][date_column]
                                        if hasattr(last_date, "strftime"):
                                            date_info = last_date.strftime("%Y-%m-%d")
                                        elif isinstance(last_date, pd.Timestamp):
                                            date_info = last_date.strftime("%Y-%m-%d")
                                        else:
                                            date_info = str(last_date)[:10]
                        except Exception as e:
                            date_info = ""

                    # Buy & Hold 수익률 계산
                    data_dict = (
                        train_data_dict if period.upper() == "TRAIN" else test_data_dict
                    )
                    buy_hold_return = (
                        self._calculate_individual_buy_hold_return(symbol, data_dict)
                        * 100
                    )

                    print(
                        f"{symbol:<8} {weight*100:>5.1f}% {cumulative_return:>7.2f}% {buy_hold_return:>5.2f}% {sharpe:>5.3f} {sortino:>7.3f} {trades:>5} {holding:<4} {price_info:<12} {date_info:<12} {strategy:<20}"
                    )

    def _calculate_var(
        self, portfolio_data: Dict[str, Any], confidence_level: float
    ) -> float:
        """Value at Risk (VaR) 계산"""
        try:
            returns = portfolio_data.get("returns", [])
            if not returns or len(returns) < 10:  # 최소 데이터 포인트 필요
                return 0.0

            returns_series = pd.Series(returns)

            # 히스토리컬 VaR 계산 (정규분포 가정 없이)
            # 신뢰수준에 해당하는 분위수 계산
            var_percentile = (1 - confidence_level) * 100
            var = returns_series.quantile(var_percentile / 100)

            return abs(var)
        except Exception:
            return 0.0

    def _calculate_cvar(
        self, portfolio_data: Dict[str, Any], confidence_level: float
    ) -> float:
        """Conditional Value at Risk (CVaR) 계산"""
        try:
            returns = portfolio_data.get("returns", [])
            if not returns or len(returns) < 10:  # 최소 데이터 포인트 필요
                return 0.0

            returns_series = pd.Series(returns)

            # VaR 계산
            var = self._calculate_var(portfolio_data, confidence_level)

            # VaR보다 작은 수익률들의 평균 (Expected Shortfall)
            tail_returns = returns_series[returns_series <= -var]
            if len(tail_returns) > 0:
                cvar = abs(tail_returns.mean())
            else:
                # VaR보다 작은 수익률이 없는 경우, 하위 분위수 평균 계산
                var_percentile = (1 - confidence_level) * 100
                tail_threshold = returns_series.quantile(var_percentile / 100)
                tail_returns = returns_series[returns_series <= tail_threshold]
                cvar = abs(tail_returns.mean()) if len(tail_returns) > 0 else var

            return cvar
        except Exception:
            return 0.0

    def _print_end_date_price_table(
        self,
        test_data_dict: Dict[str, pd.DataFrame],
        portfolio_weights: Dict[str, float],
    ):
        """종목별 end_date 주가 테이블 출력"""
        print("\n" + "=" * 100)
        print("📈 종목별 END_DATE 주가 정보")
        print("=" * 100)

        # 헤더 출력
        print(
            f"{'종목':<8} {'비중':<6} {'종료날짜':<12} {'시가':<10} {'고가':<10} {'저가':<10} {'종가':<10} {'거래량':<12} {'변동률':<8}"
        )
        print("-" * 100)

        # 포트폴리오 비중 순으로 정렬
        sorted_symbols = sorted(
            portfolio_weights.items(), key=lambda x: x[1], reverse=True
        )

        for symbol, weight in sorted_symbols:
            if symbol in test_data_dict:
                df = test_data_dict[symbol]
                if not df.empty:
                    # 마지막 데이터 (end_date)
                    last_row = df.iloc[-1]

                    # 날짜 정보 추출
                    end_date = ""
                    if hasattr(df.index[-1], "strftime"):
                        end_date = df.index[-1].strftime("%Y-%m-%d")
                    elif "datetime" in df.columns:
                        end_date = str(last_row["datetime"])[:10]
                    elif "date" in df.columns:
                        end_date = str(last_row["date"])[:10]
                    else:
                        end_date = "N/A"

                    # 가격 정보 추출
                    open_price = last_row.get("open", 0)
                    high_price = last_row.get("high", 0)
                    low_price = last_row.get("low", 0)
                    close_price = last_row.get("close", 0)
                    volume = last_row.get("volume", 0)

                    # 변동률 계산 (전일 대비)
                    if len(df) > 1:
                        prev_close = df.iloc[-2].get("close", close_price)
                        if prev_close > 0:
                            change_rate = (
                                (close_price - prev_close) / prev_close
                            ) * 100
                        else:
                            change_rate = 0.0
                    else:
                        change_rate = 0.0

                    # 거래량 포맷팅 (천 단위)
                    if volume > 1000000:
                        volume_str = f"{volume/1000000:.1f}M"
                    elif volume > 1000:
                        volume_str = f"{volume/1000:.1f}K"
                    else:
                        volume_str = f"{volume:.0f}"

                    print(
                        f"{symbol:<8} {weight*100:>5.1f}% {end_date:<12} {open_price:>9.2f} {high_price:>9.2f} {low_price:>9.2f} {close_price:>9.2f} {volume_str:>11} {change_rate:>7.2f}%"
                    )
                else:
                    print(
                        f"{symbol:<8} {weight*100:>5.1f}% {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<8}"
                    )
            else:
                print(
                    f"{symbol:<8} {weight*100:>5.1f}% {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<8}"
                )

        print("=" * 100)

    def _calculate_buy_hold_portfolio_metrics(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        portfolio_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """BUY&HOLD 포트폴리오 성과 지표 계산 (가격 변화 기반)"""
        try:
            # 각 종목의 BUY&HOLD 수익률 계산
            symbol_returns = {}
            for symbol, data in individual_results.items():
                # BUY&HOLD는 total_return을 사용 (거래 없이 가격 변화만)
                symbol_returns[symbol] = data.get("total_return", 0.0)

            # 포트폴리오 BUY&HOLD 수익률 계산 (가중 평균)
            portfolio_return = sum(
                symbol_returns[symbol] * portfolio_weights.get(symbol, 0.0)
                for symbol in individual_results.keys()
            )

            # BUY&HOLD 포트폴리오 일별 수익률 계산 (실제 가격 데이터 기반)
            # 각 종목의 일별 수익률을 가중 평균하여 포트폴리오 일별 수익률 계산
            daily_returns = []

            # 데이터 로드 (train 데이터 사용)
            try:
                train_data_dict, _ = self.load_data_and_split()

                # 공통 기간 찾기
                common_dates = None
                for symbol in individual_results.keys():
                    if symbol in train_data_dict and not train_data_dict[symbol].empty:
                        symbol_dates = train_data_dict[symbol].index
                        if common_dates is None:
                            common_dates = symbol_dates
                        else:
                            common_dates = common_dates.intersection(symbol_dates)

                if common_dates is not None and len(common_dates) > 1:
                    # 공통 기간의 일별 수익률 계산
                    for i in range(1, len(common_dates)):
                        portfolio_daily_return = 0.0

                        for symbol, weight in portfolio_weights.items():
                            if (
                                symbol in train_data_dict
                                and not train_data_dict[symbol].empty
                            ):
                                df = train_data_dict[symbol]
                                if i < len(df):
                                    # 전일 대비 수익률 계산
                                    prev_close = df.iloc[i - 1]["close"]
                                    curr_close = df.iloc[i]["close"]
                                    if prev_close > 0:
                                        symbol_return = (
                                            curr_close - prev_close
                                        ) / prev_close
                                        portfolio_daily_return += symbol_return * weight

                        daily_returns.append(portfolio_daily_return)

                # 일별 수익률이 충분하지 않으면 균등 분할 사용
                if len(daily_returns) < 10:
                    total_days = 365
                    daily_return = (
                        portfolio_return / total_days if total_days > 0 else 0.0
                    )
                    daily_returns = [daily_return] * total_days

            except Exception as e:
                # 오류 발생 시 균등 분할 사용
                total_days = 365
                daily_return = portfolio_return / total_days if total_days > 0 else 0.0
                daily_returns = [daily_return] * total_days

            # 수정된 일별 수익률로 리스크 지표 재계산
            returns_series = pd.Series(daily_returns)

            # 기본 통계 계산
            mean_return = returns_series.mean()
            std_return = returns_series.std()

            # 샤프 비율 계산 (연간화)
            risk_free_rate = 0.02 / 252  # 일간 무위험 수익률
            excess_return = mean_return - risk_free_rate
            sharpe_ratio = (
                (excess_return * 252) / (std_return * np.sqrt(252))
                if std_return > 0
                else 0
            )

            # 소르티노 비율 계산
            negative_returns = returns_series[returns_series < 0]
            sortino_ratio = 0
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std()
                sortino_ratio = (
                    (excess_return * 252) / (downside_deviation * np.sqrt(252))
                    if downside_deviation > 0
                    else 0
                )

            # 최대 낙폭 계산
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())

            # 변동성 계산 (연간화)
            volatility = std_return * np.sqrt(252)

            # 총 거래 수 (BUY&HOLD는 0)
            total_trades = 0

            return {
                "total_return": portfolio_return,
                "cumulative_return": portfolio_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "beta": 1.0,
                "total_trades": total_trades,
                "returns": daily_returns,  # BUY&HOLD 일별 수익률
            }

        except Exception as e:
            self.logger.log_error(f"BUY&HOLD 포트폴리오 지표 계산 중 오류: {e}")
            return {
                "total_return": 0.0,
                "cumulative_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "beta": 1.0,
                "total_trades": 0,
                "returns": [],
            }

    def _calculate_real_portfolio_metrics(
        self,
        individual_results: Dict[str, Dict[str, Any]],
        portfolio_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """실제 포트폴리오 성과 지표 계산 (거래 내역 기반)"""
        try:
            # 1. 포트폴리오 누적 수익률 계산 (복리 효과 고려)
            portfolio_cumulative_return = self._calculate_portfolio_cumulative_return(
                individual_results, portfolio_weights
            )

            # 2. 포트폴리오 일별 수익률 계산
            portfolio_daily_returns = self._calculate_portfolio_daily_returns(
                individual_results, portfolio_weights
            )

            if not portfolio_daily_returns or len(portfolio_daily_returns) == 0:
                return {
                    "total_return": 0.0,
                    "cumulative_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                    "beta": 1.0,
                    "total_trades": 0,
                    "returns": [],
                }

            returns_series = pd.Series(portfolio_daily_returns)

            # 3. 기본 통계 계산
            mean_return = returns_series.mean()
            std_return = returns_series.std()

            # 4. 샤프 비율 계산 (연간화)
            risk_free_rate = 0.02 / 252  # 일간 무위험 수익률
            excess_return = mean_return - risk_free_rate
            sharpe_ratio = (
                (excess_return * 252) / (std_return * np.sqrt(252))
                if std_return > 0
                else 0
            )

            # 5. 소르티노 비율 계산
            negative_returns = returns_series[returns_series < 0]
            sortino_ratio = 0
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std()
                sortino_ratio = (
                    (excess_return * 252) / (downside_deviation * np.sqrt(252))
                    if downside_deviation > 0
                    else 0
                )

            # 6. 최대 낙폭 계산
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())

            # 7. 변동성 계산 (연간화)
            volatility = std_return * np.sqrt(252)

            # 8. 베타 계산 (간단히 1.0으로 설정)
            beta = 1.0

            # 9. 총 거래 수 계산
            total_trades = sum(
                individual_results[symbol].get("total_trades", 0)
                for symbol in individual_results.keys()
            )

            return {
                "total_return": portfolio_cumulative_return,  # 누적 수익률
                "cumulative_return": portfolio_cumulative_return,  # 누적 수익률 (동일)
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "beta": beta,
                "total_trades": total_trades,
                "returns": portfolio_daily_returns,
            }

        except Exception as e:
            self.logger.log_error(f"실제 포트폴리오 지표 계산 중 오류: {e}")
            return {
                "total_return": 0.0,
                "cumulative_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "beta": 1.0,
                "total_trades": 0,
                "returns": [],
            }


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
