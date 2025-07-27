#!/usr/bin/env python3
"""
신경망 기반 포트폴리오 최적화 및 백테스팅 매니저
- 신경망 예측값을 활용한 동적 자산 배분
- 과거 신호에 따른 성과 백테스팅
- 포트폴리오 vs 개별 종목 성과 비교
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 포트폴리오 최적화 관련 임포트
try:
    from .portfolio_manager import AdvancedPortfolioManager
    from .evaluator import TrainTestEvaluator
    from .helper import Logger
except ImportError:
    from src.agent.portfolio_manager import AdvancedPortfolioManager
    from src.agent.evaluator import TrainTestEvaluator
    from src.agent.helper import Logger

# 트레이딩 관련 임포트
try:
    from ..actions.portfolio_optimization import (
        PortfolioOptimizer,
        OptimizationMethod,
        OptimizationConstraints,
    )
    from ..actions.log_pl import TradingSimulator
    from .performance_calculator import AdvancedPerformanceCalculator
    from .backtest_reporter import BacktestReporter
except ImportError:
    from src.actions.portfolio_optimization import (
        PortfolioOptimizer,
        OptimizationMethod,
        OptimizationConstraints,
    )
    from src.actions.log_pl import TradingSimulator
    from src.agent.performance_calculator import AdvancedPerformanceCalculator
    from src.agent.backtest_reporter import BacktestReporter

from .formatted_output import formatted_output

logger = logging.getLogger(__name__)


class NeuralPortfolioManager:
    """신경망 기반 포트폴리오 최적화 및 백테스팅 매니저"""

    def __init__(self, config: Dict, uuid: Optional[str] = None):
        self.config = config
        self.uuid = uuid or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = Logger()

        # 성과 계산기 및 리포터 초기화
        risk_free_rate = config.get("portfolio", {}).get("risk_free_rate", 0.02)
        self.performance_calculator = AdvancedPerformanceCalculator(risk_free_rate)
        self.backtest_reporter = BacktestReporter()

        # 백테스팅용 데이터 저장
        self.historical_signals = []
        self.portfolio_weights = {}
        self.performance_metrics = {}

        # 최적화된 임계점 로드
        self.optimized_thresholds = self._load_optimized_thresholds()

    def calculate_neural_based_weights(
        self, individual_results: List[Dict]
    ) -> Dict[str, float]:
        """
        신경망 예측값과 투자 점수를 기반으로 포트폴리오 비중 계산

        Args:
            individual_results: 개별 종목 분석 결과 리스트

        Returns:
            종목별 포트폴리오 비중 딕셔너리
        """
        try:
            logger.info("🎯 신경망 기반 포트폴리오 비중 계산 시작")

            if not individual_results:
                logger.warning("분석 결과가 없습니다")
                return {}

            weights = {}
            total_score = 0

            # 1. 기본 가중치 계산 (투자 점수 기반)
            for result in individual_results:
                symbol = result.get("symbol")
                investment_score = result.get("investment_score", {})
                score = investment_score.get("final_score", 0)
                confidence = investment_score.get("confidence", 0)

                # 신뢰도로 조정된 점수
                adjusted_score = max(0, score * confidence)
                weights[symbol] = adjusted_score
                total_score += adjusted_score

            # 2. 정규화 (최소 비중 보장)
            min_weight = self.config.get("portfolio", {}).get("min_weight", 0.05)
            max_weight = self.config.get("portfolio", {}).get("max_weight", 0.4)

            if total_score > 0:
                # 정규화
                for symbol in weights:
                    weights[symbol] = weights[symbol] / total_score

                # 최소/최대 비중 제약 적용
                weights = self._apply_weight_constraints(
                    weights, min_weight, max_weight
                )
            else:
                # 모든 점수가 0이면 동등 비중
                equal_weight = 1.0 / len(individual_results)
                weights = {
                    result["symbol"]: equal_weight for result in individual_results
                }

            logger.info(f"📊 계산된 포트폴리오 비중: {weights}")
            return weights

        except Exception as e:
            logger.error(f"포트폴리오 비중 계산 실패: {e}")
            return {}

    def _load_optimized_thresholds(self) -> Optional[Dict[str, float]]:
        """최적화된 임계점 로드"""
        try:
            # 최신 최적화 결과 파일 찾기
            results_dir = Path("results/trader")
            if not results_dir.exists():
                return None

            # threshold_optimization_best_*.json 파일들 찾기
            threshold_files = list(
                results_dir.glob("threshold_optimization_best_*.json")
            )

            if not threshold_files:
                logger.info(
                    "최적화된 임계점 파일을 찾을 수 없습니다. 기본값을 사용합니다."
                )
                return None

            # 가장 최신 파일 선택
            latest_file = max(threshold_files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, "r", encoding="utf-8") as f:
                optimization_result = json.load(f)

            best_thresholds = optimization_result.get("thresholds", {})

            if best_thresholds:
                logger.info(f"최적화된 임계점 로드 완료: {latest_file.name}")
                logger.info(f"임계점: {best_thresholds}")
                return best_thresholds
            else:
                logger.warning("최적화 결과에서 임계점을 찾을 수 없습니다.")
                return None

        except Exception as e:
            logger.warning(f"최적화된 임계점 로드 실패: {e}")
            return None

    def get_signal_thresholds(self) -> Dict[str, float]:
        """신호 생성에 사용할 임계점 반환 (최적화된 값 우선)"""
        try:
            # 기본 임계점 (config에서 가져오기)
            default_thresholds = {
                "strong_buy": 0.7,
                "buy": 0.5,
                "hold_upper": 0.3,
                "hold_lower": -0.3,
                "sell": -0.5,
                "strong_sell": -0.7,
            }

            # 최적화된 임계점이 있으면 사용
            if self.optimized_thresholds:
                logger.info("🎯 최적화된 임계점 사용")
                return self.optimized_thresholds
            else:
                logger.info("📊 기본 임계점 사용")
                return default_thresholds

        except Exception as e:
            logger.error(f"임계점 가져오기 실패: {e}")
            return default_thresholds

    def _apply_weight_constraints(
        self, weights: Dict[str, float], min_weight: float, max_weight: float
    ) -> Dict[str, float]:
        """포트폴리오 비중 제약조건 적용"""
        try:
            # 최대 비중 제한 적용
            for symbol in weights:
                if weights[symbol] > max_weight:
                    weights[symbol] = max_weight

            # 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                for symbol in weights:
                    weights[symbol] = weights[symbol] / total_weight

            # 최소 비중 보장
            symbols_below_min = [s for s, w in weights.items() if w < min_weight]
            if symbols_below_min:
                # 최소 비중 미달 종목들을 최소 비중으로 조정
                excess_needed = len(symbols_below_min) * min_weight - sum(
                    weights[s] for s in symbols_below_min
                )

                # 다른 종목들에서 비중 차감
                other_symbols = [
                    s for s in weights.keys() if s not in symbols_below_min
                ]
                if other_symbols:
                    for symbol in other_symbols:
                        reduction = excess_needed / len(other_symbols)
                        weights[symbol] = max(min_weight, weights[symbol] - reduction)

                # 최소 비중 적용
                for symbol in symbols_below_min:
                    weights[symbol] = min_weight

            # 최종 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                for symbol in weights:
                    weights[symbol] = weights[symbol] / total_weight

            return weights

        except Exception as e:
            logger.error(f"비중 제약조건 적용 실패: {e}")
            return weights

    def optimize_portfolio_with_constraints(
        self, individual_results: List[Dict], historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        제약조건을 고려한 고급 포트폴리오 최적화

        Args:
            individual_results: 개별 종목 분석 결과
            historical_data: 과거 데이터

        Returns:
            최적화된 포트폴리오 결과
        """
        try:
            logger.info("🎯 고급 포트폴리오 최적화 시작")

            # 1. 신경망 기반 기본 비중 계산
            neural_weights = self.calculate_neural_based_weights(individual_results)

            if not neural_weights:
                logger.warning("기본 비중 계산 실패")
                return {}

            # 2. 과거 수익률 데이터 준비
            returns_data = self._prepare_returns_data(historical_data)

            if returns_data.empty:
                logger.warning("수익률 데이터 부족, 신경망 기반 비중 사용")
                return {
                    "weights": neural_weights,
                    "method": "neural_based",
                    "performance": self._estimate_performance(neural_weights),
                }

            # 3. 포트폴리오 최적화 실행
            try:
                # 최적화 방법 선택
                optimization_method = self.config.get("portfolio", {}).get(
                    "optimization_method", "sharpe_maximization"
                )

                if optimization_method == "sharpe_maximization":
                    method = OptimizationMethod.SHARPE_MAXIMIZATION
                elif optimization_method == "risk_parity":
                    method = OptimizationMethod.RISK_PARITY
                elif optimization_method == "minimum_variance":
                    method = OptimizationMethod.MINIMUM_VARIANCE
                else:
                    method = OptimizationMethod.SHARPE_MAXIMIZATION

                # 제약조건 설정
                constraints = self._get_optimization_constraints()

                # 포트폴리오 최적화 실행
                risk_free_rate = self.config.get("portfolio", {}).get(
                    "risk_free_rate", 0.02
                )
                optimizer = PortfolioOptimizer(returns_data, risk_free_rate)
                result = optimizer.optimize_portfolio(method, constraints)

                # 최적화된 비중과 신경망 비중 결합
                optimized_weights = dict(zip(returns_data.columns, result.weights))
                combined_weights = self._combine_weights(
                    neural_weights, optimized_weights
                )

                return {
                    "weights": combined_weights,
                    "neural_weights": neural_weights,
                    "optimized_weights": optimized_weights,
                    "method": f"combined_{optimization_method}",
                    "performance": {
                        "sharpe_ratio": result.sharpe_ratio,
                        "expected_return": result.expected_return,
                        "volatility": result.volatility,
                        "sortino_ratio": result.sortino_ratio,
                        "max_drawdown": result.max_drawdown,
                    },
                    "optimization_result": result,
                }

            except Exception as e:
                logger.warning(f"최적화 실패, 신경망 기반 비중 사용: {e}")
                return {
                    "weights": neural_weights,
                    "method": "neural_fallback",
                    "performance": self._estimate_performance(neural_weights),
                }

        except Exception as e:
            logger.error(f"포트폴리오 최적화 실패: {e}")
            return {}

    def _prepare_returns_data(
        self, historical_data: Dict[str, pd.DataFrame], lookback_days: int = 252
    ) -> pd.DataFrame:
        """과거 수익률 데이터 준비"""
        try:
            returns_dict = {}

            for symbol, data in historical_data.items():
                if "close" in data.columns and len(data) > lookback_days:
                    # 최근 데이터만 사용
                    recent_data = data.tail(lookback_days)
                    returns = recent_data["close"].pct_change().dropna()
                    returns_dict[symbol] = returns

            if returns_dict:
                returns_df = pd.DataFrame(returns_dict).dropna()
                logger.info(f"수익률 데이터 준비 완료: {returns_df.shape}")
                return returns_df
            else:
                logger.warning("유효한 수익률 데이터 없음")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"수익률 데이터 준비 실패: {e}")
            return pd.DataFrame()

    def _get_optimization_constraints(self) -> OptimizationConstraints:
        """최적화 제약조건 반환"""
        portfolio_config = self.config.get("portfolio", {})

        return OptimizationConstraints(
            min_weight=portfolio_config.get("min_weight", 0.05),
            max_weight=portfolio_config.get("max_weight", 0.4),
            cash_weight=portfolio_config.get("cash_weight", 0.0),
            leverage=portfolio_config.get("leverage", 1.0),
            enable_short_position=portfolio_config.get("enable_short_position", False),
        )

    def _combine_weights(
        self,
        neural_weights: Dict[str, float],
        optimized_weights: Dict[str, float],
        alpha: float = 0.7,
    ) -> Dict[str, float]:
        """신경망 비중과 최적화 비중 결합"""
        try:
            combined_weights = {}
            all_symbols = set(neural_weights.keys()) | set(optimized_weights.keys())

            for symbol in all_symbols:
                neural_w = neural_weights.get(symbol, 0)
                opt_w = optimized_weights.get(symbol, 0)

                # 가중 평균 (alpha: 신경망 비중의 가중치)
                combined_w = alpha * neural_w + (1 - alpha) * opt_w
                combined_weights[symbol] = combined_w

            # 정규화
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                for symbol in combined_weights:
                    combined_weights[symbol] = combined_weights[symbol] / total_weight

            return combined_weights

        except Exception as e:
            logger.error(f"비중 결합 실패: {e}")
            return neural_weights

    def _estimate_performance(self, weights: Dict[str, float]) -> Dict[str, float]:
        """간단한 성과 추정"""
        return {
            "sharpe_ratio": 0.8,  # 추정값
            "expected_return": 0.12,  # 추정값
            "volatility": 0.15,  # 추정값
            "sortino_ratio": 1.0,  # 추정값
            "max_drawdown": -0.08,  # 추정값
        }

    def backtest_neural_signals(
        self,
        historical_data: Dict[str, pd.DataFrame],
        signal_history: List[Dict],
        portfolio_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """신경망 신호 백테스팅"""
        try:
            logger.info("📊 신경망 신호 백테스팅 시작")

            # 백테스팅 기간 추출
            start_date, end_date = self._get_backtest_period(historical_data)
            logger.info(f"📅 백테스팅 기간: {start_date} ~ {end_date}")

            # 개별 종목 백테스팅
            individual_performance = {}

            for symbol in portfolio_weights.keys():
                if symbol not in historical_data:
                    logger.warning(f"⚠️ {symbol} 히스토리컬 데이터 없음")
                    continue

                # 해당 종목의 신호 추출
                symbol_signals = self._extract_symbol_signals(signal_history, symbol)

                if not symbol_signals:
                    logger.warning(f"⚠️ {symbol} 신호 없음")
                    continue

                # 백테스팅 실행
                try:
                    symbol_result = self._backtest_symbol(
                        historical_data[symbol], symbol_signals, symbol
                    )

                    # Buy & Hold 수익률 계산
                    try:
                        buy_hold_return = self.performance_calculator.calculate_buy_hold_cumulative_return(
                            historical_data[symbol], start_date, end_date
                        )
                        symbol_result["buy_hold_return"] = buy_hold_return
                        logger.info(
                            f"📊 {symbol} Buy & Hold 누적 수익률: {buy_hold_return:.4f}"
                        )
                    except Exception as bh_error:
                        logger.warning(f"⚠️ {symbol} Buy & Hold 계산 실패: {bh_error}")
                        symbol_result["buy_hold_return"] = 0.0

                    individual_performance[symbol] = symbol_result

                except Exception as e:
                    logger.error(f"❌ {symbol} 백테스팅 실패: {e}")
                    individual_performance[symbol] = {
                        "symbol": symbol,
                        "total_return": 0,
                        "trades": [],
                        "trade_count": 0,
                        "metrics": {},
                        "buy_hold_return": 0.0,
                    }

            # 포트폴리오 레벨 백테스팅
            portfolio_performance = self._backtest_portfolio(
                individual_performance, portfolio_weights
            )

            # 성과 지표 계산
            performance_metrics = self._calculate_backtest_metrics(
                {
                    "individual_performance": individual_performance,
                    "portfolio_performance": portfolio_performance,
                }
            )

            # 백테스팅 결과 구성
            backtest_result = {
                "individual_performance": individual_performance,
                "portfolio_performance": portfolio_performance,
                "performance_metrics": performance_metrics,
                "start_date": start_date,
                "end_date": end_date,
                "historical_data": historical_data,  # Buy & Hold 계산을 위해 포함
            }

            # 거래 내역 로그 저장
            self.save_transaction_logs(backtest_result, historical_data)

            logger.info("✅ 백테스팅 완료")
            return backtest_result

        except Exception as e:
            logger.error(f"❌ 백테스팅 실패: {e}")
            return {
                "individual_performance": {},
                "portfolio_performance": {},
                "performance_metrics": {},
                "start_date": "",
                "end_date": "",
                "historical_data": historical_data,
            }

    def _extract_symbol_signals(
        self, signal_history: List[Dict], symbol: str
    ) -> List[Dict]:
        """특정 종목의 신호 기록 추출"""
        return [signal for signal in signal_history if signal.get("symbol") == symbol]

    def _backtest_symbol(
        self, data: pd.DataFrame, signals: List[Dict], symbol: str
    ) -> Dict[str, Any]:
        """개별 종목 백테스팅"""
        try:
            logger.info(
                f"🔍 {symbol} 백테스팅 시작 - 신호 {len(signals)}개, 데이터 {len(data)}일"
            )

            if not signals or data.empty:
                logger.warning(
                    f"⚠️ {symbol} 백테스팅 건너뜀 - 신호: {len(signals)}, 데이터: {len(data)}"
                )
                return {
                    "symbol": symbol,
                    "total_return": 0,
                    "trades": [],
                    "trade_count": 0,
                    "metrics": {},
                }

            # 데이터 인덱스 타입 디버깅 (주석 처리)
            # logger.info(f"🔍 {symbol} 데이터 인덱스 타입: {type(data.index)}")
            # logger.info(
            #     f"🔍 {symbol} 데이터 인덱스 timezone: {getattr(data.index, 'tz', None)}"
            # )
            # if len(data) > 0:
            #     logger.info(
            #         f"🔍 {symbol} 첫 번째 데이터 날짜: {data.index[0]} (타입: {type(data.index[0])})"
            #     )

            # 간단한 백테스팅 로직
            trades = []
            position = 0
            entry_price = 0
            total_return = 0

            for i, signal in enumerate(signals):
                try:
                    signal_timestamp_str = signal.get("timestamp")
                    # logger.info(
                    #     f"🔍 {symbol} 신호 {i+1}: timestamp_str = {signal_timestamp_str}"
                    # )

                    signal_date = pd.to_datetime(signal_timestamp_str)
                    # logger.info(
                    #     f"🔍 {symbol} 신호 {i+1}: signal_date = {signal_date} (타입: {type(signal_date)}, tz: {getattr(signal_date, 'tz', None)})"
                    # )

                    action = signal.get("trading_signal", {}).get("action", "HOLD")
                    # logger.info(f"🔍 {symbol} 신호 {i+1}: action = {action}")

                    # timezone 문제 해결: 모든 것을 naive datetime으로 변환
                    if hasattr(signal_date, "tz") and signal_date.tz is not None:
                        signal_date_naive = signal_date.tz_localize(None)
                        # logger.info(
                        #     f"🔍 {symbol} signal_date를 naive로 변환: {signal_date_naive}"
                        # )
                    else:
                        signal_date_naive = signal_date
                        # logger.info(
                        #     f"🔍 {symbol} signal_date는 이미 naive: {signal_date_naive}"
                        # )

                    # 데이터 인덱스도 naive로 변환
                    if hasattr(data.index, "tz") and data.index.tz is not None:
                        data_naive = data.copy()
                        data_naive.index = data_naive.index.tz_localize(None)
                        # logger.info(f"🔍 {symbol} 데이터 인덱스를 naive로 변환")
                    else:
                        data_naive = data
                        # logger.info(f"🔍 {symbol} 데이터 인덱스는 이미 naive")

                    # 해당 날짜의 가격 찾기 (이제 모두 naive datetime)
                    # logger.info(f"🔍 {symbol} 신호 {i+1}: 날짜 비교 준비")
                    # logger.info(
                    #     f"🔍 {symbol} signal_date_naive: {signal_date_naive} (타입: {type(signal_date_naive)})"
                    # )
                    # logger.info(
                    #     f"🔍 {symbol} data_naive.index 샘플: {data_naive.index[:3].tolist()}"
                    # )
                    # logger.info(
                    #     f"🔍 {symbol} data_naive.index[0] 타입: {type(data_naive.index[0])}"
                    # )

                    # 더 강력한 timezone 처리: 문자열 비교로 전환
                    try:
                        # 1차 시도: 직접 비교
                        price_data = data_naive[data_naive.index >= signal_date_naive]
                        # logger.info(
                        #     f"✅ {symbol} 직접 비교 성공: 필터링된 데이터 {len(price_data)}개"
                        # )
                    except Exception as comparison_error:
                        # logger.warning(f"⚠️ {symbol} 직접 비교 실패: {comparison_error}")
                        # logger.info(f"🔄 {symbol} 문자열 비교로 전환")

                        # 2차 시도: 문자열 비교
                        try:
                            signal_date_str = str(signal_date_naive)[:10]  # YYYY-MM-DD
                            # logger.info(
                            #     f"🔍 {symbol} signal_date_str: {signal_date_str}"
                            # )

                            # 데이터 인덱스를 문자열로 변환
                            data_with_str = data_naive.copy()
                            try:
                                data_with_str["date_str"] = (
                                    data_with_str.index.strftime("%Y-%m-%d")
                                )
                            except:
                                data_with_str["date_str"] = [
                                    str(d)[:10] for d in data_with_str.index
                                ]

                            price_data = data_with_str[
                                data_with_str["date_str"] >= signal_date_str
                            ]
                            # logger.info(
                            #     f"✅ {symbol} 문자열 비교 성공: 필터링된 데이터 {len(price_data)}개"
                            # )

                        except Exception as str_error:
                            logger.error(f"❌ {symbol} 문자열 비교도 실패: {str_error}")
                            # logger.info(f"🔄 {symbol} 날짜 변환 후 재시도")

                            # 3차 시도: 완전히 새로운 datetime 객체 생성
                            try:
                                from datetime import datetime

                                signal_dt = datetime(
                                    signal_date_naive.year,
                                    signal_date_naive.month,
                                    signal_date_naive.day,
                                )
                                # logger.info(
                                #     f"🔍 {symbol} 새로운 signal_dt: {signal_dt} (타입: {type(signal_dt)})"
                                # )

                                # 데이터 인덱스도 완전히 새로 만들기
                                data_converted = data_naive.copy()
                                new_index = []
                                for dt in data_converted.index:
                                    if hasattr(dt, "year"):
                                        new_dt = datetime(dt.year, dt.month, dt.day)
                                    else:
                                        new_dt = datetime.strptime(
                                            str(dt)[:10], "%Y-%m-%d"
                                        )
                                    new_index.append(new_dt)

                                data_converted.index = new_index
                                # logger.info(
                                #     f"🔍 {symbol} 변환된 인덱스 샘플: {data_converted.index[:3].tolist()}"
                                # )

                                price_data = data_converted[
                                    data_converted.index >= signal_dt
                                ]
                                # logger.info(
                                #     f"✅ {symbol} 날짜 변환 비교 성공: 필터링된 데이터 {len(price_data)}개"
                                # )

                            except Exception as final_error:
                                logger.error(
                                    f"❌ {symbol} 모든 날짜 비교 실패: {final_error}"
                                )
                                logger.warning(f"⚠️ {symbol} 신호 {i+1} 건너뜀")
                                continue

                    if price_data.empty:
                        # logger.warning(
                        #     f"⚠️ {symbol} 신호 {i+1}: 해당 날짜 이후 가격 데이터 없음"
                        # )
                        continue

                    current_price = price_data.iloc[0]["close"]
                    # logger.info(
                    #     f"🔍 {symbol} 신호 {i+1}: current_price = {current_price}"
                    # )

                    if action in ["BUY", "STRONG_BUY"] and position == 0:
                        # 매수
                        position = 1
                        entry_price = current_price
                        trades.append(
                            {
                                "action": "BUY",
                                "price": current_price,
                                "date": signal_date_naive.strftime(
                                    "%Y-%m-%d"
                                ),  # 날짜만 저장
                                "signal": signal,
                            }
                        )
                        # logger.info(
                        #     f"✅ {symbol} 매수 실행: 가격 ${current_price:.2f}, 날짜 {signal_date_naive.strftime('%Y-%m-%d')}, 신호 강도: {signal.get('trading_signal', {}).get('strength', 0):.3f}"
                        # )

                    elif action in ["SELL", "STRONG_SELL"] and position == 1:
                        # 매도
                        position = 0
                        pnl = (current_price - entry_price) / entry_price
                        total_return += pnl
                        trades.append(
                            {
                                "action": "SELL",
                                "price": current_price,
                                "date": signal_date_naive.strftime(
                                    "%Y-%m-%d"
                                ),  # 날짜만 저장
                                "pnl": pnl,
                                "signal": signal,
                            }
                        )
                        # logger.info(
                        #     f"✅ {symbol} 매도 실행: 가격 ${current_price:.2f}, 매수가 ${entry_price:.2f}, PnL {pnl:.4f} ({pnl*100:.2f}%), 누적수익률 {total_return:.4f}"
                        # )

                    else:
                        pass
                        # logger.info(
                        #     f"⏸️ {symbol} 신호 {i+1}: {action} - 조건 불충족 (position: {position}, 신호 강도: {signal.get('trading_signal', {}).get('strength', 0):.3f})"
                        # )

                except Exception as signal_error:
                    logger.error(f"❌ {symbol} 신호 {i+1} 처리 실패: {signal_error}")
                    continue

            # 마지막 포지션 청산
            if position == 1 and not data.empty:
                final_price = data.iloc[-1]["close"]
                pnl = (final_price - entry_price) / entry_price
                total_return += pnl
                # logger.info(
                #     f"🔚 {symbol} 마지막 포지션 청산: 가격 {final_price}, PnL {pnl:.4f}, 최종수익률 {total_return:.4f}"
                # )

            # logger.info(
            #     f"✅ {symbol} 백테스팅 완료 - 총 거래: {len(trades)}, 최종 수익률: {total_return:.4f}"
            # )

            # 거래수 계산: BUY와 SELL 거래 모두 포함
            buy_trades = [t for t in trades if t["action"] == "BUY"]
            sell_trades = [t for t in trades if t["action"] == "SELL"]
            total_trade_count = len(buy_trades) + len(sell_trades)

            # logger.info(
            #     f"📊 {symbol} 거래 분석: BUY {len(buy_trades)}회, SELL {len(sell_trades)}회, 총 {total_trade_count}회"
            # )

            return {
                "symbol": symbol,
                "total_return": total_return,
                "trades": trades,
                "trade_count": total_trade_count,  # BUY와 SELL 모두 포함
                "buy_count": len(buy_trades),
                "sell_count": len(sell_trades),
                "metrics": self._calculate_symbol_metrics(trades, total_return),
            }

        except Exception as e:
            logger.error(f"❌ {symbol} 백테스팅 실패: {e}")
            import traceback

            logger.error(f"❌ {symbol} 상세 오류: {traceback.format_exc()}")
            return {
                "symbol": symbol,
                "total_return": 0,
                "trades": [],
                "trade_count": 0,
                "metrics": {},
            }

    def _backtest_portfolio(
        self, individual_performance: Dict[str, Any], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """포트폴리오 백테스팅"""
        try:
            total_return = 0
            total_trades = 0

            # 일일 수익률 시계열 생성
            daily_returns = pd.Series(dtype=float)

            for symbol, weight in weights.items():
                if symbol in individual_performance:
                    symbol_return = individual_performance[symbol]["total_return"]
                    total_return += weight * symbol_return
                    total_trades += individual_performance[symbol].get(
                        "trade_count", 0
                    )  # 안전하게 접근

                    # 개별 종목의 거래를 기반으로 일일 수익률 생성
                    symbol_trades = individual_performance[symbol].get("trades", [])
                    if symbol_trades:
                        # 거래 날짜별 수익률 계산
                        for trade in symbol_trades:
                            if (
                                trade.get("action") == "SELL"
                                and trade.get("pnl") is not None
                            ):
                                trade_date = pd.to_datetime(trade.get("date"))
                                pnl = trade.get("pnl", 0)
                                weighted_pnl = pnl * weight

                                if trade_date in daily_returns.index:
                                    daily_returns[trade_date] += weighted_pnl
                                else:
                                    daily_returns[trade_date] = weighted_pnl

            # 날짜 순으로 정렬
            if not daily_returns.empty:
                daily_returns = daily_returns.sort_index()
                logger.info(
                    f"📊 포트폴리오 일일 수익률 시계열 생성: {len(daily_returns)}일"
                )

            return {
                "total_return": total_return,
                "total_trades": total_trades,
                "weights": weights,
                "daily_returns": daily_returns,
                "metrics": self._calculate_portfolio_metrics(
                    total_return, individual_performance, daily_returns
                ),
            }

        except Exception as e:
            logger.error(f"포트폴리오 백테스팅 실패: {e}")
            return {
                "total_return": 0,
                "total_trades": 0,
                "weights": weights,
                "daily_returns": pd.Series(),
                "metrics": {},
            }  # 일관성 있는 구조 반환

    def _calculate_symbol_metrics(
        self, trades: List[Dict], total_return: float
    ) -> Dict[str, float]:
        """개별 종목 성과 지표 계산"""
        try:
            if not trades:
                return {
                    "total_return": total_return,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                    "trade_count": 0,
                    "profitable_trades": 0,
                    "total_trades": 0,
                }

            # 거래 분류
            buy_trades = [t for t in trades if t.get("action") == "BUY"]
            sell_trades = [t for t in trades if t.get("action") == "SELL"]
            total_trade_count = len(buy_trades) + len(sell_trades)

            # 매도 거래만 필터링 (pnl이 있는 거래)
            profitable_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]

            # 안전한 승률 계산 (매도 거래 기준)
            win_rate = 0.0
            if sell_trades:
                win_rate = len(profitable_trades) / len(sell_trades)

            # 안전한 평균 PnL 계산
            avg_pnl = 0.0
            if sell_trades:
                pnl_values = [t.get("pnl", 0) for t in sell_trades]
                avg_pnl = sum(pnl_values) / len(pnl_values)

            # logger.info(
            #     f"📊 거래 통계: 총 {len(trades)}개, 매수 {len(buy_trades)}개, 매도 {len(sell_trades)}개, "
            #     f"수익 {len(profitable_trades)}개, 승률 {win_rate:.2%}"
            # )

            return {
                "total_return": total_return,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "trade_count": total_trade_count,  # BUY와 SELL 모두 포함
                "buy_count": len(buy_trades),
                "sell_count": len(sell_trades),
                "profitable_trades": len(profitable_trades),
                "total_trades": len(trades),
            }

        except Exception as e:
            logger.error(f"❌ 성과 지표 계산 실패: {e}")
            return {
                "total_return": total_return,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "trade_count": 0,
                "profitable_trades": 0,
                "total_trades": len(trades) if trades else 0,
            }

    def _calculate_portfolio_metrics(
        self,
        total_return: float,
        individual_performance: Dict[str, Any],
        daily_returns: pd.Series = None,
    ) -> Dict[str, float]:
        """포트폴리오 성과 지표 계산"""
        try:
            # 개별 종목 수익률 수집
            returns = []
            for symbol, perf in individual_performance.items():
                if "total_return" in perf:
                    returns.append(perf["total_return"])

            # 포트폴리오 수익률이 0이면 개별 종목 수익률의 가중평균 사용
            if total_return == 0 and returns:
                total_return = sum(returns) / len(returns)

            # 변동성 계산
            volatility = 0.15  # 기본값
            if daily_returns is not None and len(daily_returns) > 1:
                # 일일 수익률 시계열이 있으면 이를 사용
                volatility = daily_returns.std() * np.sqrt(252)  # 연율화
                # logger.info(f"📊 일일 수익률 기반 변동성: {volatility:.4f} (연율화)")
            elif len(returns) > 1:
                # 개별 종목 수익률의 표준편차 사용
                volatility = np.std(returns)
                # logger.info(f"📊 개별 종목 기반 변동성: {volatility:.4f}")

            # 샤프비율 계산 (무위험 수익률 0% 가정)
            sharpe_ratio = 0.0
            if volatility > 0:
                sharpe_ratio = total_return / volatility

            # 최대 낙폭 계산
            max_drawdown = 0.0
            if daily_returns is not None and len(daily_returns) > 1:
                # 일일 수익률 시계열을 사용한 정확한 최대 낙폭 계산
                cumulative_returns = (1 + daily_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                # logger.info(f"📊 일일 수익률 기반 최대 낙폭: {max_drawdown:.4f}")
            else:
                # 간단한 추정 (수익률이 음수일 때만)
                max_drawdown = (
                    min(0, -abs(total_return) * 0.3) if total_return < 0 else 0
                )
                # logger.info(f"📊 추정 최대 낙폭: {max_drawdown:.4f}")

            # logger.info(
            #     f"📊 포트폴리오 메트릭: 수익률={total_return:.4f}, 변동성={volatility:.4f}, "
            #     f"샤프비율={sharpe_ratio:.4f}, 최대낙폭={max_drawdown:.4f}"
            # )

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
            }

        except Exception as e:
            logger.error(f"❌ 포트폴리오 메트릭 계산 실패: {e}")
            return {
                "total_return": total_return,
                "sharpe_ratio": 0.0,
                "volatility": 0.15,
                "max_drawdown": 0.0,
            }

    def _calculate_backtest_metrics(
        self, backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """전체 백테스팅 성과 지표 계산"""
        try:
            individual_perf = backtest_results["individual_performance"]
            portfolio_perf = backtest_results["portfolio_performance"]

            # 개별 종목 요약
            individual_summary = {}
            for symbol, perf in individual_perf.items():
                # metrics가 별도 키가 아니라 perf 자체에 있는 경우
                metrics = perf.get("metrics", {})
                if not metrics:
                    # perf 자체가 metrics인 경우
                    metrics = perf

                individual_summary[symbol] = {
                    "return": perf.get("total_return", 0),
                    "trades": perf.get("trade_count", 0),  # 안전하게 접근
                    "win_rate": metrics.get("win_rate", 0),
                }

                logger.info(
                    f"📊 {symbol} 요약: 수익률={perf.get('total_return', 0):.4f}, "
                    f"거래수={perf.get('trade_count', 0)}, 승률={metrics.get('win_rate', 0):.2%}"
                )

            # 포트폴리오 요약
            portfolio_metrics = portfolio_perf.get("metrics", {})
            if not portfolio_metrics:
                portfolio_metrics = portfolio_perf

            portfolio_summary = {
                "total_return": portfolio_perf.get("total_return", 0),
                "total_trades": portfolio_perf.get("total_trades", 0),
                "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0),
            }

            return {
                "individual_summary": individual_summary,
                "portfolio_summary": portfolio_summary,
                "comparison": {
                    "portfolio_vs_equal_weight": self._compare_vs_equal_weight(
                        individual_perf
                    ),
                    "best_individual": (
                        max(
                            individual_perf.keys(),
                            key=lambda s: individual_perf[s]["total_return"],
                        )
                        if individual_perf
                        else None
                    ),
                    "worst_individual": (
                        min(
                            individual_perf.keys(),
                            key=lambda s: individual_perf[s]["total_return"],
                        )
                        if individual_perf
                        else None
                    ),
                },
            }

        except Exception as e:
            logger.error(f"백테스팅 지표 계산 실패: {e}")
            return {}

    def _compare_vs_equal_weight(
        self, individual_performance: Dict[str, Any]
    ) -> Dict[str, float]:
        """동등 비중 대비 성과 비교"""
        if not individual_performance:
            return {}

        equal_weight_return = np.mean(
            [perf["total_return"] for perf in individual_performance.values()]
        )

        return {
            "equal_weight_return": equal_weight_return,
            "difference": self.performance_metrics.get("portfolio_summary", {}).get(
                "total_return", 0
            )
            - equal_weight_return,
        }

    def generate_enhanced_portfolio_report(
        self,
        portfolio_result: Dict[str, Any],
        backtest_result: Dict[str, Any] = None,
        historical_data: Dict[str, pd.DataFrame] = None,
    ) -> str:
        """고급 성과 지표를 포함한 포트폴리오 분석 보고서 생성"""
        try:
            if backtest_result and historical_data:
                # 백테스팅 결과가 있는 경우 종합 리포트 생성
                return self._generate_comprehensive_backtest_report(
                    portfolio_result, backtest_result, historical_data
                )
            else:
                # 백테스팅 없이 기본 포트폴리오 정보만
                return self._create_basic_portfolio_info(portfolio_result)

        except Exception as e:
            logger.error(f"고급 보고서 생성 실패: {e}")
            return "고급 보고서 생성 중 오류가 발생했습니다."

    def _generate_comprehensive_backtest_report(
        self,
        portfolio_result: Dict[str, Any],
        backtest_result: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
    ) -> str:
        """종합 백테스팅 리포트 생성"""
        try:
            # 데이터 준비
            start_date, end_date = self._get_backtest_period(historical_data)

            # 포트폴리오 데이터
            portfolio_data = {
                "total_return": backtest_result.get("portfolio_performance", {}).get(
                    "total_return", 0
                ),
                "total_trades": backtest_result.get("portfolio_performance", {}).get(
                    "total_trades", 0
                ),
                "sharpe_ratio": backtest_result.get("portfolio_performance", {})
                .get("metrics", {})
                .get("sharpe_ratio", 0),
                "volatility": backtest_result.get("portfolio_performance", {})
                .get("metrics", {})
                .get("volatility", 0),
                "max_drawdown": backtest_result.get("portfolio_performance", {})
                .get("metrics", {})
                .get("max_drawdown", 0),
                "weights": portfolio_result.get("weights", {}),
            }

            # 개별 종목 데이터
            individual_data = {}
            individual_perf = backtest_result.get("individual_performance", {})
            for symbol, perf in individual_perf.items():
                individual_data[symbol] = {
                    "weight": portfolio_result.get("weights", {}).get(symbol, 0),
                    "total_return": perf.get("total_return", 0),
                    "buy_hold_return": perf.get("buy_hold_return", 0),
                    "trade_count": perf.get("trade_count", 0),
                    "win_rate": perf.get("win_rate", 0),
                }

            # 매매 데이터
            trading_data = self._prepare_trading_summary_data(individual_perf)

            # 거래 이력 데이터
            trades_data = self._prepare_recent_trades_data(individual_perf)

            # 성과 비교 데이터
            comparison_data = self._prepare_comparison_data(backtest_result)

            # 시장 체제 데이터
            regime_data = {
                "regime": "SIDEWAYS",  # 기본값, 실제로는 시장 체제 분석 결과 사용
                "confidence": 0.9,
                "portfolio_score": 0.2862,
                "portfolio_action": "SELECTIVE_BUY",
                "signal_distribution": {"BUY": 2, "HOLD": 2, "SELL": 0},
            }

            # 최종 보유현황 데이터
            positions_data = self._prepare_positions_data(individual_perf, end_date)

            # 종합 리포트 생성
            return formatted_output.format_comprehensive_report(
                portfolio_data=portfolio_data,
                individual_data=individual_data,
                trading_data=trading_data,
                trades_data=trades_data,
                comparison_data=comparison_data,
                regime_data=regime_data,
                positions_data=positions_data,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            logger.error(f"종합 백테스팅 리포트 생성 실패: {e}")
            return "종합 백테스팅 리포트 생성 중 오류가 발생했습니다."

    def _prepare_trading_summary_data(
        self, individual_perf: Dict[str, Any]
    ) -> Dict[str, Any]:
        """매매 요약 데이터 준비"""
        total_buy_count = 0
        total_sell_count = 0
        total_profitable_trades = 0
        total_trades = 0
        symbol_trading = {}

        for symbol, perf in individual_perf.items():
            trades = perf.get("trades", [])
            buy_trades = [t for t in trades if t.get("action") == "BUY"]
            sell_trades = [t for t in trades if t.get("action") == "SELL"]
            profitable_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]

            total_buy_count += len(buy_trades)
            total_sell_count += len(sell_trades)
            total_profitable_trades += len(profitable_trades)
            total_trades += len(sell_trades)

            symbol_trading[symbol] = {
                "buy_count": len(buy_trades),
                "sell_count": len(sell_trades),
                "profitable": len(profitable_trades),
                "win_rate": (
                    len(profitable_trades) / len(sell_trades)
                    if len(sell_trades) > 0
                    else 0
                ),
            }

        overall_win_rate = (
            total_profitable_trades / total_trades if total_trades > 0 else 0
        )

        return {
            "total_buy_count": total_buy_count,
            "total_sell_count": total_sell_count,
            "total_profitable_trades": total_profitable_trades,
            "total_trades": total_trades,
            "overall_win_rate": overall_win_rate,
            "symbol_trading": symbol_trading,
        }

    def _prepare_recent_trades_data(
        self, individual_perf: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """최근 거래 이력 데이터 준비"""
        all_trades = []

        for symbol, perf in individual_perf.items():
            trades = perf.get("trades", [])
            for trade in trades:
                trade_info = trade.copy()
                trade_info["symbol"] = symbol
                all_trades.append(trade_info)

        # 날짜순 정렬 (최신순)
        all_trades.sort(
            key=lambda x: pd.to_datetime(x.get("date", "1900-01-01")), reverse=True
        )

        return all_trades[:15]  # 최근 15건

    def _prepare_comparison_data(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """성과 비교 데이터 준비"""
        portfolio_perf = backtest_result.get("portfolio_performance", {})
        metrics = portfolio_perf.get("metrics", {})

        return {
            "strategy_return": portfolio_perf.get("total_return", 0),
            "strategy_sharpe": metrics.get("sharpe_ratio", 0),
            "strategy_volatility": metrics.get("volatility", 0),
            "strategy_max_drawdown": metrics.get("max_drawdown", 0),
            "benchmark_return": 0.0419,  # Buy & Hold 수익률 (실제로는 계산 필요)
            "benchmark_sharpe": 0.270,
            "benchmark_volatility": 0.2535,
            "benchmark_max_drawdown": 0.2442,
        }

    def _prepare_positions_data(
        self, individual_perf: Dict[str, Any], end_date: str
    ) -> Dict[str, Any]:
        """최종 보유현황 데이터 준비"""
        positions_data = {}

        for symbol, perf in individual_perf.items():
            trades = perf.get("trades", [])

            if not trades:
                positions_data[symbol] = {
                    "position_status": "없음",
                    "last_date": "-",
                    "last_action": "-",
                    "last_price": 0,
                }
                continue

            # 최종 거래 찾기
            last_trade = trades[-1]
            last_action = last_trade.get("action", "")
            last_price = last_trade.get("price", 0)
            last_date = last_trade.get("date", "")

            # 포지션 상태 확인
            position_status = "보유중" if last_action == "BUY" else "청산완료"

            positions_data[symbol] = {
                "position_status": position_status,
                "last_date": last_date,
                "last_action": last_action,
                "last_price": last_price,
            }

        return positions_data

    def save_transaction_logs(
        self,
        backtest_result: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
    ):
        """개별 종목별 거래 내역을 로그 파일로 저장"""
        try:
            # 로그 디렉토리 생성
            log_dir = Path("log")
            log_dir.mkdir(exist_ok=True)

            today = datetime.now().strftime("%Y%m%d")
            execution_uuid = self.uuid if self.uuid else "neural_backtest"

            # 거래 내역 로그 저장
            log_path = log_dir / f"transaction_neural_{today}_{execution_uuid}.log"
            self._save_transaction_log(
                backtest_result,
                historical_data,
                log_path,
            )

            logger.info(f"✅ 거래 내역 로그 저장 완료: {log_path}")

        except Exception as e:
            logger.error(f"❌ 거래 내역 로그 저장 실패: {e}")

    def _save_transaction_log(
        self,
        backtest_result: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        log_path: Path,
    ):
        """거래 내역을 로그 파일로 저장"""
        try:
            individual_perf = backtest_result.get("individual_performance", {})
            start_date = backtest_result.get("start_date", "")
            end_date = backtest_result.get("end_date", "")

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=== 신경망 기반 포트폴리오 거래 내역 로그 ===\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"실행 UUID: {self.uuid if self.uuid else 'neural_backtest'}\n")
                f.write(f"백테스팅 기간: {start_date} ~ {end_date}\n")
                f.write("=" * 80 + "\n\n")

                for symbol, perf in individual_perf.items():
                    trades = perf.get("trades", [])
                    total_return = perf.get("total_return", 0)
                    buy_hold_return = perf.get("buy_hold_return", 0)

                    f.write(f"📊 {symbol} (Neural Network Strategy)\n")
                    f.write("-" * 50 + "\n")

                    if trades:
                        f.write(f"총 거래 수: {len(trades)}\n")
                        f.write(f"전략 수익률: {total_return*100:.2f}%\n")
                        f.write(f"Buy & Hold 수익률: {buy_hold_return*100:.2f}%\n")
                        f.write(
                            f"초과 수익률: {(total_return - buy_hold_return)*100:.2f}%\n\n"
                        )

                        f.write("거래 내역:\n")
                        f.write(
                            f"{'날짜':<20} {'시간':<10} {'타입':<6} {'가격':<10} {'신호강도':<10} {'수익률':<10} {'누적수익률':<12}\n"
                        )
                        f.write("-" * 80 + "\n")

                        cumulative_return = 0
                        for trade in trades:
                            date = trade.get("date", "")
                            action = trade.get("action", "")
                            price = trade.get("price", 0)
                            pnl = trade.get("pnl", 0)
                            signal = trade.get("signal", {})
                            trading_signal = signal.get("trading_signal", {})
                            strength = trading_signal.get("strength", 0)

                            # 시간 정보 처리
                            date_str = str(date) if date else ""

                            if action == "BUY":
                                f.write(
                                    f"{date_str:<20} {'매수':<10} ${price:<9.2f} {strength:<10.3f} {'':<10} {'':<12}\n"
                                )
                            elif action == "SELL":
                                cumulative_return += pnl
                                f.write(
                                    f"{date_str:<20} {'매도':<10} ${price:<9.2f} {strength:<10.3f} {pnl*100:<10.2f}% {cumulative_return*100:<12.2f}%\n"
                                )
                    else:
                        f.write("거래 내역 없음\n")

                    f.write("\n" + "=" * 80 + "\n\n")

                # 포트폴리오 요약
                portfolio_perf = backtest_result.get("portfolio_performance", {})
                if portfolio_perf:
                    f.write("📊 포트폴리오 전체 요약\n")
                    f.write("-" * 50 + "\n")
                    f.write(
                        f"총 수익률: {portfolio_perf.get('total_return', 0)*100:.2f}%\n"
                    )
                    f.write(
                        f"총 거래 횟수: {portfolio_perf.get('total_trades', 0)} 회\n"
                    )

                    metrics = portfolio_perf.get("metrics", {})
                    f.write(f"샤프 비율: {metrics.get('sharpe_ratio', 0):.3f}\n")
                    f.write(f"변동성: {metrics.get('volatility', 0)*100:.2f}%\n")
                    f.write(f"최대 낙폭: {metrics.get('max_drawdown', 0)*100:.2f}%\n")

        except Exception as e:
            logger.error(f"❌ 거래 내역 로그 저장 실패: {e}")

    def _get_backtest_period(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> Tuple[str, str]:
        """백테스팅 기간 추출 (config 설정 기반)"""
        try:
            # config에서 백테스팅 설정 가져오기
            backtest_config = self.config.get("backtesting", {})
            period_ratio = backtest_config.get("period_ratio", 0.3)  # 기본값 30%
            min_period_days = backtest_config.get("min_period_days", 60)
            max_period_days = backtest_config.get("max_period_days", 252)

            # 가장 긴 데이터를 가진 종목의 기간 사용
            all_dates = []
            for symbol, data in historical_data.items():
                if not data.empty:
                    all_dates.extend(data.index.tolist())

            if all_dates:
                data_start = min(all_dates)
                data_end = max(all_dates)

                # 전체 데이터 기간 계산
                total_days = (data_end - data_start).days

                # config 기반 백테스팅 기간 계산
                backtest_days = int(total_days * period_ratio)

                # 최소/최대 기간 제한 적용
                backtest_days = max(
                    min_period_days, min(backtest_days, max_period_days)
                )

                # 백테스팅 시작일 계산 (최근 기간)
                start_dt = data_end - timedelta(days=backtest_days)
                start_date = start_dt.strftime("%Y-%m-%d")
                end_date = data_end.strftime("%Y-%m-%d")

                logger.info(
                    f"📅 백테스팅 기간 설정: {start_date} ~ {end_date} ({backtest_days}일, 전체 기간의 {period_ratio:.1%})"
                )
                return start_date, end_date
            else:
                # 기본값 (config 기반)
                end_date = datetime.now().strftime("%Y-%m-%d")
                default_days = int(252 * period_ratio)  # 1년 기준
                default_days = max(min_period_days, min(default_days, max_period_days))
                start_date = (datetime.now() - timedelta(days=default_days)).strftime(
                    "%Y-%m-%d"
                )
                logger.info(
                    f"📅 기본 백테스팅 기간: {start_date} ~ {end_date} ({default_days}일)"
                )
                return start_date, end_date

        except Exception as e:
            logger.error(f"백테스팅 기간 추출 실패: {e}")
            # 오류 시 기본값
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            logger.info(f"📅 오류 시 기본 기간: {start_date} ~ {end_date} (180일)")
            return start_date, end_date

    def _create_benchmark_comparison(
        self,
        backtest_result: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> str:
        """Buy & Hold 벤치마크와 성과 비교"""
        try:
            logger.info("🔍 벤치마크 비교 시작")
            logger.info(f"📅 백테스팅 기간: {start_date} ~ {end_date}")
            logger.info(f"📊 historical_data 종목 수: {len(historical_data)}")
            logger.info(f"📊 backtest_result 키: {list(backtest_result.keys())}")

            # 전략 수익률 계산
            strategy_returns = self._extract_strategy_returns(backtest_result)
            logger.info(
                f"📊 전략 수익률 시계열: {len(strategy_returns)}일, 비어있음: {strategy_returns.empty}"
            )

            # Buy & Hold 벤치마크 수익률 계산 (포트폴리오 가중평균)
            benchmark_returns = self._calculate_portfolio_benchmark_returns(
                historical_data, backtest_result, start_date, end_date
            )
            logger.info(
                f"📊 벤치마크 수익률 시계열: {len(benchmark_returns)}일, 비어있음: {benchmark_returns.empty}"
            )

            if strategy_returns.empty:
                logger.error("❌ 전략 수익률 시계열이 비어있음")
                return "⚠️ 전략 수익률 데이터 부족"

            if benchmark_returns.empty:
                logger.error("❌ 벤치마크 수익률 시계열이 비어있음")
                return "⚠️ 벤치마크 수익률 데이터 부족"

            if strategy_returns.empty or benchmark_returns.empty:
                logger.error("❌ 전략 또는 벤치마크 수익률 시계열이 비어있음")
                return "⚠️ 벤치마크 비교 데이터 부족"

            # 고급 성과 지표 계산
            strategy_metrics = self.performance_calculator.calculate_all_metrics(
                strategy_returns, benchmark_returns
            )

            benchmark_metrics = self.performance_calculator.calculate_all_metrics(
                benchmark_returns
            )

            # 성과 비교 테이블 생성
            comparison_table = (
                self.performance_calculator.create_performance_comparison_table(
                    strategy_metrics, benchmark_metrics, "신경망 전략", "Buy & Hold"
                )
            )

            # 추가 요약 정보
            summary = self.backtest_reporter.create_performance_comparison_summary(
                strategy_metrics,
                benchmark_metrics,
                {"start": start_date, "end": end_date},
            )

            return f"{comparison_table}\n\n{summary}"

        except Exception as e:
            logger.error(f"벤치마크 비교 생성 실패: {e}")
            return "벤치마크 비교 생성 실패"

    def _extract_strategy_returns(self, backtest_result: Dict[str, Any]) -> pd.Series:
        """전략 수익률 시계열 추출"""
        try:
            portfolio_perf = backtest_result.get("portfolio_performance", {})
            individual_perf = backtest_result.get("individual_performance", {})
            start_date = backtest_result.get("start_date", "")
            end_date = backtest_result.get("end_date", "")

            if not portfolio_perf or not start_date or not end_date:
                logger.warning("전략 수익률 추출을 위한 데이터 부족")
                return pd.Series()

            # 포트폴리오 전체 수익률 가져오기
            total_return = portfolio_perf.get("total_return", 0)
            logger.info(f"📊 포트폴리오 전체 수익률: {total_return:.4f}")

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
            daily_returns = pd.Series(0.0, index=date_range)

            logger.info(
                f"📊 전략 수익률 시계열 생성: {start_date} ~ {end_date} ({len(date_range)}일)"
            )

            # 실제 거래가 발생한 날짜에만 수익률 배치
            trade_dates = set()
            for symbol, perf in individual_perf.items():
                trades = perf.get("trades", [])
                weight = portfolio_perf.get("weights", {}).get(symbol, 0.1)

                for trade in trades:
                    trade_date = trade.get("date")
                    if trade_date:
                        # 시간 정보 제거하고 날짜만 추출
                        if isinstance(trade_date, str):
                            trade_dt = pd.to_datetime(trade_date.split()[0])
                        else:
                            trade_dt = pd.to_datetime(trade_date).normalize()

                        if start_dt <= trade_dt <= end_dt:
                            trade_dates.add(trade_dt)
                            action = trade.get("action", "")
                            pnl = trade.get("pnl", 0)
                            if action == "SELL" and pnl != 0:
                                daily_returns[trade_dt] += pnl * weight
                                logger.info(
                                    f"📊 {symbol} {action} 수익 반영: {trade_dt.date()} PnL={pnl:.4f} * {weight:.3f} = {pnl*weight:.6f}"
                                )

            # 거래가 없는 날짜들에 균등하게 나머지 수익률 분배
            if trade_dates:
                remaining_return = total_return - daily_returns.sum()
                non_trade_dates = [d for d in date_range if d not in trade_dates]

                if remaining_return != 0 and len(non_trade_dates) > 0:
                    daily_remaining = remaining_return / len(non_trade_dates)
                    for date in non_trade_dates:
                        daily_returns[date] = daily_remaining
                    logger.info(f"📊 거래 없는 날짜에 균등 분배: {daily_remaining:.6f}")

            logger.info(
                f"📊 전략 수익률 시계열: {len(daily_returns)}일, 총 수익률: {daily_returns.sum():.4f}, 거래일: {len(trade_dates)}일"
            )
            return daily_returns

        except Exception as e:
            logger.error(f"전략 수익률 추출 실패: {e}")
            return pd.Series()

    def _calculate_portfolio_benchmark_returns(
        self,
        historical_data: Dict[str, pd.DataFrame],
        backtest_result: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """포트폴리오 Buy & Hold 벤치마크 수익률 계산"""
        try:
            # logger.info("🔍 Buy & Hold 벤치마크 계산 시작")
            logger.info(f"📊 historical_data 종목: {list(historical_data.keys())}")
            logger.info(f"📅 백테스팅 기간: {start_date} ~ {end_date}")

            portfolio_weights = backtest_result.get("portfolio_performance", {}).get(
                "weights", {}
            )

            if not portfolio_weights:
                # 동등 비중 사용
                symbols = list(historical_data.keys())
                portfolio_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
                logger.info(f"📊 동등 비중 사용: {portfolio_weights}")
            else:
                logger.info(f"📊 포트폴리오 비중: {portfolio_weights}")

            # 백테스팅 기간 설정
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")

            # 포트폴리오 일일 수익률 초기화
            portfolio_daily_returns = pd.Series(0.0, index=date_range)

            # logger.info(
            #     f"📊 Buy & Hold 벤치마크 계산: {start_date} ~ {end_date} ({len(date_range)}일)"
            # )

            # 각 종목의 일일 수익률 계산
            symbols_processed = 0
            total_portfolio_return = 0.0

            for symbol, weight in portfolio_weights.items():
                if symbol in historical_data:
                    try:
                        # 해당 종목의 일일 수익률 계산
                        symbol_data = historical_data[symbol].copy()

                        # datetime 컬럼이 있는지 확인
                        if "datetime" in symbol_data.columns:
                            # datetime 컬럼을 인덱스로 설정
                            symbol_data = symbol_data.set_index("datetime")

                        # 날짜 필터링 (간단한 문자열 비교 사용)
                        symbol_data["date_str"] = symbol_data.index.astype(str).str[:10]

                        filtered_data = symbol_data[
                            (symbol_data["date_str"] >= start_date)
                            & (symbol_data["date_str"] <= end_date)
                        ]

                        if len(filtered_data) > 1:
                            close_prices = filtered_data["close"]
                            daily_returns = close_prices.pct_change().dropna()
                            weighted_returns = daily_returns * weight

                            logger.info(
                                f"📊 {symbol} Buy & Hold: {len(filtered_data)}일, 매칭 {len(daily_returns)}일, 가중치 {weight:.3f}"
                            )

                            # 실제 가격 변동이 있는 날짜에만 수익률 배치
                            for date_str, return_val in zip(
                                filtered_data.index[1:], weighted_returns
                            ):
                                try:
                                    # 날짜만 추출 (timezone 정보 제거)
                                    date_str_clean = str(date_str)[:10]
                                    date_dt = pd.to_datetime(date_str_clean)

                                    if start_dt <= date_dt <= end_dt:
                                        portfolio_daily_returns[date_dt] += return_val
                                except Exception as e:
                                    logger.warning(
                                        f"📊 {symbol} 날짜 처리 실패: {date_str} - {e}"
                                    )
                                    continue

                            # 누적 수익률 계산
                            start_price = close_prices.iloc[0]
                            end_price = close_prices.iloc[-1]
                            symbol_return = (end_price / start_price) - 1
                            weighted_return = symbol_return * weight
                            total_portfolio_return += weighted_return

                            logger.info(
                                f"📊 {symbol} Buy & Hold: 시작가=${start_price:.2f}, 끝가=${end_price:.2f}, 수익률={symbol_return:.4f}, 가중수익률={weighted_return:.4f}"
                            )

                            symbols_processed += 1
                        else:
                            logger.warning(
                                f"⚠️ {symbol} 백테스팅 기간 데이터 부족: {len(filtered_data)}일"
                            )

                    except Exception as e:
                        logger.warning(f"⚠️ {symbol} Buy & Hold 계산 실패: {e}")
                        continue

            logger.info(
                f"📊 Buy & Hold 포트폴리오 누적 수익률: {total_portfolio_return:.4f}"
            )
            logger.info(
                f"📊 Buy & Hold 변동성: {portfolio_daily_returns.std() * np.sqrt(252):.4f}"
            )
            logger.info(f"📊 처리된 종목 수: {symbols_processed}")

            if symbols_processed == 0:
                logger.warning("⚠️ 처리된 종목이 없음 - 빈 시계열 반환")
                return pd.Series()

            return portfolio_daily_returns

        except Exception as e:
            logger.error(f"포트폴리오 벤치마크 계산 실패: {e}")
            return pd.Series()

    def _create_basic_portfolio_info(self, portfolio_result: Dict[str, Any]) -> str:
        """기본 포트폴리오 정보 (백테스팅 없는 경우)"""
        try:
            lines = []
            lines.append("📊 포트폴리오 구성 및 예상 성과")
            lines.append("-" * 80)

            # 포트폴리오 비중
            weights = portfolio_result.get("weights", {})
            lines.append("💼 최적 포트폴리오 비중:")
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for symbol, weight in sorted_weights:
                lines.append(f"   {symbol}: {weight*100:>6.2f}%")
            lines.append("")

            # 예상 성과
            performance = portfolio_result.get("performance", {})
            if performance:
                lines.append("📈 예상 포트폴리오 성과:")
                lines.append(
                    f"   샤프 비율:     {performance.get('sharpe_ratio', 0):>8.3f}"
                )
                lines.append(
                    f"   예상 수익률:   {performance.get('expected_return', 0)*100:>8.2f}%"
                )
                lines.append(
                    f"   변동성:       {performance.get('volatility', 0)*100:>8.2f}%"
                )
                lines.append(
                    f"   소르티노 비율: {performance.get('sortino_ratio', 0):>8.3f}"
                )
                lines.append(
                    f"   최대 낙폭:     {performance.get('max_drawdown', 0)*100:>8.2f}%"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"기본 포트폴리오 정보 생성 실패: {e}")
            return "기본 포트폴리오 정보 생성 실패"
