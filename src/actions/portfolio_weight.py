#!/usr/bin/env python3
"""
포트폴리오 비중 산출 시스템
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import logging
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioWeightCalculator:
    """포트폴리오 비중 계산 클래스"""

    def __init__(self, config_path: Optional[str] = None):
        print(f"🔍 PortfolioWeightCalculator 초기화 시작 - config_path: {config_path}")

        self.config = self._load_config(config_path)
        print(f"🔍 설정 로드 완료: {type(self.config)}")

        self.portfolio_config = self.config["portfolio"]
        print(f"🔍 포트폴리오 설정: {self.portfolio_config}")

        # 새로운 설정 구조에 맞게 수정
        self.rebalance_period = self.portfolio_config.get("rebalance_period", 20)
        print(f"🔍 리밸런싱 주기: {self.rebalance_period}")

        # optimization_method 사용 (기존 weight_calculation_method 대신)
        self.method = self.portfolio_config.get(
            "optimization_method", "sharpe_maximization"
        )
        print(f"🔍 선택된 최적화 방법: {self.method}")

        # fallback 현황 기록
        self.fallback_stats = {
            "risk_parity": 0,
            "min_variance": 0,
            "volatility_inverse": 0,
            "equal_weight": 0,
        }
        self.fallback_log = []

        # AdvancedPortfolioManager import (lazy loading)
        self.advanced_manager = None

        print("✅ PortfolioWeightCalculator 초기화 완료")

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """설정 파일 로드"""
        try:
            # config_path가 None이면 기본값 사용
            if config_path is None:
                config_path = "config/config_swing.json"

            # 먼저 절대 경로로 시도
            if os.path.isabs(config_path):
                config_file = config_path
            else:
                # 상대 경로인 경우 여러 위치에서 시도
                possible_paths = [
                    config_path,  # 현재 작업 디렉토리 기준
                    os.path.join(
                        os.path.dirname(__file__), "..", "..", config_path
                    ),  # 프로젝트 루트 기준
                    os.path.join(
                        os.path.dirname(__file__), config_path
                    ),  # actions 디렉토리 기준
                ]

                config_file = None
                for path in possible_paths:
                    if os.path.exists(path):
                        config_file = path
                        break

                if config_file is None:
                    raise FileNotFoundError(
                        f"Config file not found in any of: {possible_paths}"
                    )

            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패, 기본값 사용: {e}")
            return {
                "portfolio": {
                    "initial_capital": 100000,
                    "rebalance_period": 20,
                    "optimization_method": "sharpe_maximization",
                    "risk_free_rate": 0.02,
                    "target_volatility": 0.20,
                    "min_weight": 0.0,
                    "max_weight": 0.8,
                }
            }

    def calculate_optimal_weights(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """최적 비중 계산"""
        print(f"📊 {self.method} 방식으로 포트폴리오 비중 계산 중...")

        # 모든 종목의 공통 기간 찾기
        common_dates = self._get_common_dates(data_dict)
        symbols = list(data_dict.keys())

        # 비중 DataFrame 초기화
        weights_df = pd.DataFrame(index=common_dates)
        weights_df.index.name = "datetime"

        # 각 시점별로 비중 계산
        for i, date in enumerate(common_dates):
            # date가 timezone-naive인지 확인하고 필요시 변환
            if hasattr(date, "tz") and date.tz is not None:
                date_naive = date.tz_localize(None)
            else:
                date_naive = date

            if i % self.rebalance_period == 0:  # 리밸런싱 시점
                # 해당 시점까지의 데이터로 비중 계산
                historical_data = self._get_historical_data(
                    data_dict, date_naive, symbols
                )
                weights = self._calculate_weights_at_date(historical_data, symbols)
            else:
                # 이전 비중 유지 (첫 번째 시점이 아닌 경우)
                if i > 0:
                    weights = {
                        symbol: weights_df.iloc[i - 1][symbol] for symbol in symbols
                    }
                else:
                    # 첫 번째 시점에서는 등가 비중 사용
                    weights = self._equal_weight(symbols)

            # 비중을 DataFrame에 저장
            for symbol in symbols:
                weights_df.loc[date, symbol] = float(weights.get(symbol, 0.0))

            # 현금 비중 계산 (1 - 모든 종목 비중 합)
            total_weight = sum(float(weights.get(symbol, 0.0)) for symbol in symbols)
            weights_df.loc[date, "CASH"] = max(0.0, 1.0 - total_weight)

        return weights_df

    def _get_common_dates(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """모든 종목의 공통 날짜 찾기"""
        if not isinstance(data_dict, dict):
            logger.error(
                f"data_dict 타입 오류: {type(data_dict)}. dict[str, pd.DataFrame]이어야 합니다."
            )
            return pd.DatetimeIndex([])
        date_sets = []
        for symbol, df in data_dict.items():
            if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns:
                logger.warning(
                    f"{symbol} 데이터가 DataFrame이 아니거나 datetime 컬럼이 없습니다."
                )
                continue
            # datetime 컬럼을 안전하게 변환 (문자열, tz-aware 모두 처리)
            dates = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            if hasattr(dates, "dt") and dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
            date_sets.append(set(dates))

        if not date_sets:
            return pd.DatetimeIndex([])
        common_dates = set.intersection(*date_sets)
        return pd.DatetimeIndex(sorted(common_dates))

    def _get_historical_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
        symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """현재 시점까지의 과거 데이터 추출"""
        historical_data = {}

        for symbol in symbols:
            df = data_dict[symbol]

            # datetime 컬럼의 timezone 처리 - 더 강력한 방법
            df_dates = pd.to_datetime(df["datetime"], errors="coerce")
            if hasattr(df_dates, "dt") and df_dates.dt.tz is not None:
                # timezone-aware datetime을 timezone-naive로 변환
                df_dates = df_dates.dt.tz_localize(None)

            # current_date도 timezone-naive로 보장
            if hasattr(current_date, "tz") and current_date.tz is not None:
                current_date_naive = current_date.tz_localize(None)
            else:
                current_date_naive = current_date

            # timezone-naive datetime으로 비교
            historical_df = df[df_dates <= current_date_naive].copy()
            if len(historical_df) > 0:
                historical_data[symbol] = historical_df

        return historical_data

    def _calculate_weights_at_date(
        self, historical_data: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """특정 시점에서의 비중 계산"""
        # AdvancedPortfolioManager가 지원하는 방법들
        advanced_methods = [
            "sharpe_maximization",
            "sortino_maximization",
            "mean_variance",
            "risk_parity",
            "minimum_variance",
            "maximum_diversification",
            "black_litterman",
            "kelly_criterion",
        ]

        if self.method in advanced_methods:
            return self._advanced_optimization_weight(historical_data, symbols)
        elif self.method == "equal_weight":
            return self._equal_weight(symbols)
        elif self.method == "volatility_inverse":
            return self._volatility_inverse_weight(historical_data, symbols)
        elif self.method == "momentum_weight":
            return self._momentum_weight(historical_data, symbols)
        else:
            logger.warning(f"알 수 없는 비중 계산 방식: {self.method}, 등가 비중 사용")
            return self._equal_weight(symbols)

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """등가 비중"""
        weight_per_symbol = 1.0 / len(symbols)
        return {symbol: weight_per_symbol for symbol in symbols}

    def _advanced_optimization_weight(
        self, historical_data: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """AdvancedPortfolioManager를 사용한 고급 최적화 비중 계산"""
        try:
            # Lazy loading of AdvancedPortfolioManager
            if self.advanced_manager is None:
                from agent.portfolio_manager import AdvancedPortfolioManager

                # config_path를 절대 경로로 변환
                config_path = os.path.join(
                    os.path.dirname(__file__), "../../config/config_long.json"
                )
                self.advanced_manager = AdvancedPortfolioManager(config_path)

            # 수익률 데이터 준비
            returns_data = {}
            for symbol in symbols:
                if symbol in historical_data and len(historical_data[symbol]) > 1:
                    returns = historical_data[symbol]["close"].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[symbol] = returns

            if len(returns_data) < 2:
                logger.warning(
                    "Advanced optimization requires at least 2 assets with data"
                )
                return self._equal_weight(symbols)

            # 공통 기간으로 정렬
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if len(returns_df) < 30:  # 최소 데이터 포인트 필요
                logger.warning("Advanced optimization requires at least 30 data points")
                return self._equal_weight(symbols)

            # 최적화 방법을 문자열로 직접 전달 (AdvancedPortfolioManager에서 처리)
            optimization_method = self.method

            # 포트폴리오 최적화 실행 (AdvancedPortfolioManager의 run_advanced_portfolio_management 사용)
            try:
                # 임시로 AdvancedPortfolioManager의 최적화 로직을 직접 호출
                from actions.portfolio_optimization import (
                    PortfolioOptimizer,
                    OptimizationMethod,
                )

                # 수익률 데이터 준비
                returns_data = {}
                for symbol in symbols:
                    if symbol in historical_data and len(historical_data[symbol]) > 1:
                        returns = historical_data[symbol]["close"].pct_change().dropna()
                        if len(returns) > 0:
                            returns_data[symbol] = returns

                if len(returns_data) < 2:
                    return self._equal_weight(symbols)

                returns_df = pd.DataFrame(returns_data)
                returns_df = returns_df.dropna()

                if len(returns_df) < 30:
                    return self._equal_weight(symbols)

                # PortfolioOptimizer 직접 사용
                optimizer = PortfolioOptimizer(returns=returns_df, risk_free_rate=0.02)

                # 최적화 방법 매핑
                method_mapping = {
                    "sharpe_maximization": OptimizationMethod.SHARPE_MAXIMIZATION,
                    "sortino_maximization": OptimizationMethod.SORTINO_MAXIMIZATION,
                    "mean_variance": OptimizationMethod.MEAN_VARIANCE,
                    "risk_parity": OptimizationMethod.RISK_PARITY,
                    "minimum_variance": OptimizationMethod.MINIMUM_VARIANCE,
                    "maximum_diversification": OptimizationMethod.MAXIMUM_DIVERSIFICATION,
                    "black_litterman": OptimizationMethod.BLACK_LITTERMAN,
                    "kelly_criterion": OptimizationMethod.KELLY_CRITERION,
                }

                method = method_mapping.get(
                    self.method, OptimizationMethod.SHARPE_MAXIMIZATION
                )

                # 기본 제약조건
                from actions.portfolio_optimization import OptimizationConstraints

                constraints = OptimizationConstraints(
                    min_weight=0.0, max_weight=1.0, cash_weight=0.0, leverage=1.0
                )

                result = optimizer.optimize_portfolio(method, constraints)

            except Exception as e:
                logger.warning(f"Direct optimization failed: {e}, using equal weight")
                return self._equal_weight(symbols)

            if result and result.weights is not None:
                # 결과를 딕셔너리로 변환
                weights = {}
                for i, symbol in enumerate(symbols):
                    if i < len(result.weights):
                        weights[symbol] = float(result.weights[i])
                    else:
                        weights[symbol] = 0.0
                return weights
            else:
                logger.warning("Advanced optimization failed, using equal weight")
                return self._equal_weight(symbols)

        except Exception as e:
            logger.warning(f"Advanced optimization error: {e}, using equal weight")
            return self._equal_weight(symbols)

    def _volatility_inverse_weight(
        self, historical_data: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """변동성 역비례 비중"""
        lookback = self.weight_methods["volatility_inverse"]["lookback_period"]
        volatilities = {}

        for symbol in symbols:
            if symbol in historical_data and len(historical_data[symbol]) >= lookback:
                returns = historical_data[symbol]["close"].pct_change().dropna()
                if len(returns) >= lookback:
                    volatility = returns.tail(lookback).std()
                    volatilities[symbol] = volatility

        if not volatilities:
            return self._equal_weight(symbols)

        # 변동성 역수 계산
        inv_volatilities = {symbol: 1 / vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_volatilities.values())

        # 비중 계산
        weights = {
            symbol: inv_vol / total_inv_vol
            for symbol, inv_vol in inv_volatilities.items()
        }
        return weights

    def _risk_parity_weight(
        self, historical_data: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """리스크 패리티 비중 (fallback 포함)"""
        returns_data = {}
        min_length = float("inf")
        for symbol in symbols:
            if symbol in historical_data:
                returns = historical_data[symbol]["close"].pct_change().dropna()
                if len(returns) > 0:
                    returns_data[symbol] = returns
                    min_length = min(min_length, len(returns))
        if len(returns_data) < 2:
            self.fallback_stats["equal_weight"] += 1
            self.fallback_log.append("risk_parity→equal_weight: insufficient data")
            return self._equal_weight(symbols)
        aligned_returns = {}
        for symbol in symbols:
            if symbol in returns_data:
                aligned_returns[symbol] = returns_data[symbol].tail(int(min_length))
        if len(aligned_returns) < 2:
            self.fallback_stats["equal_weight"] += 1
            self.fallback_log.append("risk_parity→equal_weight: aligned insufficient")
            return self._equal_weight(symbols)
        returns_df = pd.DataFrame(aligned_returns)
        cov_matrix = returns_df.cov()
        # 공분산 행렬 정규화(특이행렬 방지)
        try:
            if np.linalg.cond(cov_matrix) > 1e8:
                cov_matrix += np.eye(len(cov_matrix)) * 1e-6
        except Exception as e:
            logger.warning(f"공분산 행렬 cond 계산 실패: {e}")
        n_assets = len(aligned_returns)
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        def risk_parity_objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            asset_contributions = (
                weights * (np.dot(cov_matrix, weights)) / portfolio_risk
            )
            return np.sum((asset_contributions - asset_contributions.mean()) ** 2)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        bounds = [(0.0, 1.0)] * n_assets
        try:
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.weight_methods["risk_parity"]["max_iterations"]
                },
            )
            if result.success and np.all(result.x >= 0):
                self.fallback_stats["risk_parity"] += 1
                return {
                    symbol: weight
                    for symbol, weight in zip(aligned_returns.keys(), result.x)
                }
            else:
                logger.warning("리스크 패리티 최적화 실패, 최소분산 fallback")
                self.fallback_stats["min_variance"] += 1
                self.fallback_log.append("risk_parity→min_variance")
                # 최소분산 fallback
                minvar = self._min_variance_weight(historical_data, symbols)
                if sum(minvar.values()) > 0:
                    return minvar
                logger.warning("최소분산도 실패, 변동성 역비례 fallback")
                self.fallback_stats["volatility_inverse"] += 1
                self.fallback_log.append("min_variance→volatility_inverse")
                # 변동성 역비례 fallback
                invvol = self._volatility_inverse_weight(historical_data, symbols)
                if sum(invvol.values()) > 0:
                    return invvol
                logger.warning("변동성 역비례도 실패, 등가 fallback")
                self.fallback_stats["equal_weight"] += 1
                self.fallback_log.append("volatility_inverse→equal_weight")
                return self._equal_weight(symbols)
        except Exception as e:
            logger.error(f"리스크 패리티 최적화 예외: {e}")
            self.fallback_stats["min_variance"] += 1
            self.fallback_log.append(f"risk_parity_exception→min_variance: {e}")
            minvar = self._min_variance_weight(historical_data, symbols)
            if sum(minvar.values()) > 0:
                return minvar
            self.fallback_stats["volatility_inverse"] += 1
            self.fallback_log.append("min_variance→volatility_inverse")
            invvol = self._volatility_inverse_weight(historical_data, symbols)
            if sum(invvol.values()) > 0:
                return invvol
            self.fallback_stats["equal_weight"] += 1
            self.fallback_log.append("volatility_inverse→equal_weight")
            return self._equal_weight(symbols)

    def _momentum_weight(
        self, historical_data: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """모멘텀 기반 비중"""
        momentum_period = self.weight_methods["momentum_weight"]["momentum_period"]
        top_n = self.weight_methods["momentum_weight"]["top_n_symbols"]

        momentums = {}

        for symbol in symbols:
            if (
                symbol in historical_data
                and len(historical_data[symbol]) >= momentum_period
            ):
                prices = historical_data[symbol]["close"]
                if len(prices) >= momentum_period:
                    momentum = (prices.iloc[-1] / prices.iloc[-momentum_period]) - 1
                    momentums[symbol] = momentum

        if not momentums:
            return self._equal_weight(symbols)

        # 상위 N개 종목 선택
        sorted_momentums = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_momentums[:top_n]]

        # 비중 계산 (모멘텀에 비례)
        total_momentum = sum(momentums[symbol] for symbol in top_symbols)
        if total_momentum > 0:
            weights = {
                symbol: momentums[symbol] / total_momentum for symbol in top_symbols
            }
            # 나머지 종목은 0
            for symbol in symbols:
                if symbol not in weights:
                    weights[symbol] = 0.0
        else:
            weights = self._equal_weight(symbols)

        return weights

    def _min_variance_weight(
        self, historical_data: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> Dict[str, float]:
        """최소 분산 포트폴리오 비중"""
        lookback = self.weight_methods["min_variance"]["lookback_period"]

        # 수익률 데이터 준비
        returns_data = {}
        min_length = float("inf")

        for symbol in symbols:
            if symbol in historical_data:
                returns = historical_data[symbol]["close"].pct_change().dropna()
                if len(returns) >= lookback:
                    returns_data[symbol] = returns.tail(lookback)
                    min_length = min(min_length, len(returns))

        if len(returns_data) < 2:
            return self._equal_weight(symbols)

        # 공분산 행렬 계산
        returns_df = pd.DataFrame(returns_data)
        cov_matrix = returns_df.cov()

        # 최소 분산 최적화
        n_assets = len(returns_data)
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # 제약조건: 비중 합 = 1, 비중 >= 0
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        bounds = [(0.0, 1.0)] * n_assets

        # 최적화 실행
        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = result.x
            return {
                symbol: weight for symbol, weight in zip(returns_data.keys(), weights)
            }
        else:
            logger.warning("최소 분산 최적화 실패, 등가 비중 사용")
            return self._equal_weight(symbols)

    def print_weight_summary(self, weights_df: pd.DataFrame):
        """비중 요약 출력 및 fallback 현황 출력"""
        print(f"\n📊 포트폴리오 비중 요약")
        print(f"리밸런싱 주기: {self.rebalance_period}회 거래마다")
        print(f"비중 계산 방식: {self.method}")
        print("-" * 50)
        # 평균 비중
        avg_weights = weights_df.mean()
        print("평균 비중:")
        if isinstance(avg_weights, pd.Series):
            for symbol, weight in avg_weights.items():
                print(f"  {symbol}: {weight*100:.1f}%")
        else:
            print("  평균 비중 계산 불가")
        # 비중 변동성
        weight_volatility = weights_df.std()
        print(f"\n비중 변동성 (표준편차):")
        if isinstance(weight_volatility, pd.Series):
            for symbol, vol in weight_volatility.items():
                print(f"  {symbol}: {vol*100:.1f}%")
        else:
            print("  비중 변동성 계산 불가")
        # 리밸런싱 횟수
        rebalance_count = len(weights_df) // self.rebalance_period
        print(f"\n총 리밸런싱 횟수: {rebalance_count}회")
        # fallback 현황
        print(f"\nFallback 현황:")
        for k, v in self.fallback_stats.items():
            print(f"  {k}: {v}회")
        if self.fallback_log:
            print("  최근 fallback 로그:")
            for log in self.fallback_log[-5:]:
                print(f"    - {log}")


def main():
    """테스트용 메인 함수"""
    print("PortfolioWeightCalculator 테스트")

    # 샘플 데이터 생성
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    symbols = ["NVDL", "TSLL", "CONL"]

    data_dict = {}
    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)
        close_prices = [100.0]  # float로 초기화
        for i in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)
            close_prices.append(close_prices[-1] * (1 + change))

        df = pd.DataFrame(
            {
                "datetime": dates,
                "close": close_prices,
                "open": [p * (1 + np.random.normal(0, 0.005)) for p in close_prices],
                "high": [
                    p * (1 + abs(np.random.normal(0, 0.01))) for p in close_prices
                ],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in close_prices],
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            }
        )
        data_dict[symbol] = df

    # 비중 계산
    calculator = PortfolioWeightCalculator()
    weights_df = calculator.calculate_optimal_weights(data_dict)

    # 결과 출력
    print(f"\n비중 DataFrame 형태: {weights_df.shape}")
    print(f"컬럼: {list(weights_df.columns)}")
    print(f"\n처음 5개 시점의 비중:")
    print(weights_df.head())

    # 비중 요약
    calculator.print_weight_summary(weights_df)


if __name__ == "__main__":
    main()
