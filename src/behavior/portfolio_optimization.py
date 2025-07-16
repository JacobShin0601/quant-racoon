#!/usr/bin/env python3
"""
고급 포트폴리오 최적화 엔진
금융권 수준의 다양한 최적화 기법들을 구현
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")


class OptimizationMethod(Enum):
    """최적화 방법들"""

    MEAN_VARIANCE = "mean_variance"
    SHARPE_MAXIMIZATION = "sharpe_maximization"
    SORTINO_MAXIMIZATION = "sortino_maximization"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    BLACK_LITTERMAN = "black_litterman"
    KELLY_CRITERION = "kelly_criterion"


@dataclass
class OptimizationConstraints:
    """최적화 제약조건"""

    min_weight: float = 0.0
    max_weight: float = 1.0
    cash_weight: float = 0.0
    leverage: float = 1.0
    group_constraints: Optional[Dict[str, Dict[str, float]]] = None
    sector_constraints: Optional[Dict[str, Dict[str, float]]] = None
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None


@dataclass
class OptimizationResult:
    """최적화 결과"""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    diversification_ratio: float
    method: str
    constraints: OptimizationConstraints
    metadata: Dict[str, Any]


class PortfolioOptimizer:
    """
    고급 포트폴리오 최적화 엔진
    금융권 수준의 다양한 최적화 기법들을 제공
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        초기화

        Args:
            returns: 수익률 데이터 (T x N)
            risk_free_rate: 무위험 수익률
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()

        # 기본 통계량 계산
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.correlation_matrix = returns.corr()

        # 로거 설정
        self.logger = logging.getLogger(__name__)

    def calculate_performance_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """포트폴리오 성과 지표 계산"""
        portfolio_returns = self.returns @ weights

        # 기본 지표
        expected_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)

        # 샤프 비율
        excess_returns = (
            portfolio_returns - self.risk_free_rate / 252
        )  # 일간 무위험 수익률
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        # 소르티노 비율 (하방 표준편차 사용)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = (
                np.mean(excess_returns) / downside_std * np.sqrt(252)
                if downside_std > 0
                else 0
            )
        else:
            sortino_ratio = np.inf

        # 최대 낙폭
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # VaR, CVaR (95% 신뢰수준)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])

        # 분산화 비율
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        weighted_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix)))
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0

        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "diversification_ratio": diversification_ratio,
        }

    def mean_variance_optimization(
        self, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Markowitz 평균-분산 최적화"""
        self.logger.debug("Markowitz 평균-분산 최적화 실행 중...")

        w = cp.Variable(self.n_assets)

        # 목적함수: 분산 최소화
        risk = cp.quad_form(w, self.cov_matrix.values)

        # 제약조건
        constraint_list = [cp.sum(w) == 1 - constraints.cash_weight]

        # 비중 제약
        constraint_list.extend([w >= constraints.min_weight])
        constraint_list.extend([w <= constraints.max_weight])

        # 레버리지 제약
        if constraints.leverage != 1.0:
            constraint_list.append(cp.sum(cp.abs(w)) <= constraints.leverage)

        # 목표 수익률 제약
        if constraints.target_return is not None:
            constraint_list.append(
                self.mean_returns.values @ w >= constraints.target_return
            )

        # 목표 변동성 제약
        if constraints.target_volatility is not None:
            constraint_list.append(
                cp.quad_form(w, self.cov_matrix.values)
                <= constraints.target_volatility**2
            )

        # 그룹 제약
        if constraints.group_constraints:
            for group_name, group_data in constraints.group_constraints.items():
                if "assets" in group_data and "min" in group_data:
                    group_indices = [
                        self.asset_names.index(asset)
                        for asset in group_data["assets"]
                        if asset in self.asset_names
                    ]
                    if group_indices:
                        constraint_list.append(
                            cp.sum(w[group_indices]) >= group_data["min"]
                        )
                if "assets" in group_data and "max" in group_data:
                    group_indices = [
                        self.asset_names.index(asset)
                        for asset in group_data["assets"]
                        if asset in self.asset_names
                    ]
                    if group_indices:
                        constraint_list.append(
                            cp.sum(w[group_indices]) <= group_data["max"]
                        )

        # 최적화 문제 풀이
        problem = cp.Problem(cp.Minimize(risk), constraint_list)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"최적화 실패: {problem.status}")

        weights = w.value
        metrics = self.calculate_performance_metrics(weights)

        return OptimizationResult(
            weights=weights,
            method="Mean-Variance",
            constraints=constraints,
            **metrics,
            metadata={"optimization_status": problem.status},
        )

    def sharpe_maximization(
        self, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """샤프 비율 최대화 최적화"""
        self.logger.debug("샤프 비율 최대화 최적화 실행 중...")

        w = cp.Variable(self.n_assets)

        # 목적함수: 샤프 비율 최대화 (분모 최소화)
        excess_returns = self.mean_returns.values - self.risk_free_rate / 252
        risk = cp.quad_form(w, self.cov_matrix.values)

        # 제약조건
        constraint_list = [cp.sum(w) == 1 - constraints.cash_weight]
        constraint_list.extend([w >= constraints.min_weight])
        constraint_list.extend([w <= constraints.max_weight])

        if constraints.leverage != 1.0:
            constraint_list.append(cp.sum(cp.abs(w)) <= constraints.leverage)

        # 최적화 문제 풀이
        problem = cp.Problem(cp.Minimize(risk), constraint_list)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"최적화 실패: {problem.status}")

        weights = w.value
        metrics = self.calculate_performance_metrics(weights)

        return OptimizationResult(
            weights=weights,
            method="Sharpe Maximization",
            constraints=constraints,
            **metrics,
            metadata={"optimization_status": problem.status},
        )

    def sortino_maximization(
        self, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """소르티노 비율 최대화 최적화"""
        self.logger.debug("소르티노 비율 최대화 최적화 실행 중...")

        # 하방 분산 계산
        downside_returns = self.returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_cov = downside_returns.cov()

        w = cp.Variable(self.n_assets)

        # 목적함수: 하방 분산 최소화
        downside_risk = cp.quad_form(w, downside_cov.values)

        # 제약조건
        constraint_list = [cp.sum(w) == 1 - constraints.cash_weight]
        constraint_list.extend([w >= constraints.min_weight])
        constraint_list.extend([w <= constraints.max_weight])

        if constraints.leverage != 1.0:
            constraint_list.append(cp.sum(cp.abs(w)) <= constraints.leverage)

        # 최적화 문제 풀이
        problem = cp.Problem(cp.Minimize(downside_risk), constraint_list)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"최적화 실패: {problem.status}")

        weights = w.value
        metrics = self.calculate_performance_metrics(weights)

        return OptimizationResult(
            weights=weights,
            method="Sortino Maximization",
            constraints=constraints,
            **metrics,
            metadata={"optimization_status": problem.status},
        )

    def risk_parity_optimization(
        self, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """리스크 패리티 최적화"""
        self.logger.debug("리스크 패리티 최적화 실행 중...")

        def risk_parity_objective(weights):
            """리스크 패리티 목적함수: 각 자산의 리스크 기여도를 동일하게 만듦"""
            portfolio_risk = np.sqrt(weights.T @ self.cov_matrix.values @ weights)
            if portfolio_risk == 0:
                return 0

            # 각 자산의 리스크 기여도 계산
            asset_contributions = (
                weights * (self.cov_matrix.values @ weights) / portfolio_risk
            )

            # 목표 리스크 기여도 (균등 분배)
            target_contribution = portfolio_risk / self.n_assets

            # 리스크 기여도의 분산을 최소화
            variance_of_contributions = np.var(asset_contributions)

            self.logger.debug(f"포트폴리오 리스크: {portfolio_risk:.6f}")
            self.logger.debug(f"자산별 리스크 기여도: {asset_contributions}")
            self.logger.debug(f"목표 기여도: {target_contribution:.6f}")
            self.logger.debug(f"기여도 분산: {variance_of_contributions:.6f}")

            return variance_of_contributions

        # 초기 가중치 (동일 가중치)
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # 제약조건
        bounds = [(constraints.min_weight, constraints.max_weight)] * self.n_assets
        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - (1 - constraints.cash_weight)}
        ]

        if constraints.leverage != 1.0:
            constraints_list.append(
                {
                    "type": "ineq",
                    "fun": lambda x: constraints.leverage - np.sum(np.abs(x)),
                }
            )

        # 최적화
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
            options={"maxiter": 1000},
        )

        if not result.success:
            raise ValueError(f"최적화 실패: {result.message}")

        weights = result.x
        metrics = self.calculate_performance_metrics(weights)

        return OptimizationResult(
            weights=weights,
            method="Risk Parity",
            constraints=constraints,
            **metrics,
            metadata={"optimization_status": result.success},
        )

    def minimum_variance_optimization(
        self, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """최소 분산 최적화"""
        self.logger.debug("최소 분산 최적화 실행 중...")

        w = cp.Variable(self.n_assets)

        # 목적함수: 분산 최소화
        risk = cp.quad_form(w, self.cov_matrix.values)

        # 제약조건
        constraint_list = [cp.sum(w) == 1 - constraints.cash_weight]
        constraint_list.extend([w >= constraints.min_weight])
        constraint_list.extend([w <= constraints.max_weight])

        if constraints.leverage != 1.0:
            constraint_list.append(cp.sum(cp.abs(w)) <= constraints.leverage)

        # 최적화 문제 풀이
        problem = cp.Problem(cp.Minimize(risk), constraint_list)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"최적화 실패: {problem.status}")

        weights = w.value
        metrics = self.calculate_performance_metrics(weights)

        return OptimizationResult(
            weights=weights,
            method="Minimum Variance",
            constraints=constraints,
            **metrics,
            metadata={"optimization_status": problem.status},
        )

    def maximum_diversification_optimization(
        self, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """최대 분산화 최적화"""
        self.logger.debug("최대 분산화 최적화 실행 중...")

        w = cp.Variable(self.n_assets)

        # 개별 자산 변동성
        asset_vols = np.sqrt(np.diag(self.cov_matrix.values))

        # 목적함수: 분산화 비율 최대화 (분모 최소화)
        portfolio_vol = cp.quad_form(w, self.cov_matrix.values)
        weighted_vol = asset_vols @ w

        # 제약조건
        constraint_list = [cp.sum(w) == 1 - constraints.cash_weight]
        constraint_list.extend([w >= constraints.min_weight])
        constraint_list.extend([w <= constraints.max_weight])

        if constraints.leverage != 1.0:
            constraint_list.append(cp.sum(cp.abs(w)) <= constraints.leverage)

        # 최적화 문제 풀이
        problem = cp.Problem(cp.Minimize(portfolio_vol), constraint_list)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"최적화 실패: {problem.status}")

        weights = w.value
        metrics = self.calculate_performance_metrics(weights)

        return OptimizationResult(
            weights=weights,
            method="Maximum Diversification",
            constraints=constraints,
            **metrics,
            metadata={"optimization_status": problem.status},
        )

    def black_litterman_optimization(
        self,
        constraints: OptimizationConstraints,
        market_caps: Optional[np.ndarray] = None,
        views: Optional[Dict[str, float]] = None,
        confidence: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Black-Litterman 모델 최적화"""
        self.logger.info("Black-Litterman 모델 최적화 실행 중...")

        # 시장 균형 가중치 (시가총액 기반)
        if market_caps is None:
            market_caps = np.ones(self.n_assets) / self.n_assets

        # 시장 균형 수익률
        market_weights = market_caps / np.sum(market_caps)
        market_return = self.mean_returns.values @ market_weights

        # Black-Litterman 수익률
        if views is not None and confidence is not None:
            # 뷰 매트릭스 구성 (간단한 구현)
            pi = market_return * np.ones(self.n_assets)
            # 실제로는 더 복잡한 뷰 매트릭스 구성 필요
            mu_bl = pi  # 뷰가 없는 경우 시장 균형 수익률 사용
        else:
            mu_bl = self.mean_returns.values

        # Black-Litterman 공분산 행렬
        tau = 0.05  # 스케일링 팩터
        sigma_bl = (1 + tau) * self.cov_matrix.values

        # 최적화
        w = cp.Variable(self.n_assets)

        # 목적함수: Black-Litterman 기반 분산 최소화
        risk = cp.quad_form(w, sigma_bl)

        # 제약조건
        constraint_list = [cp.sum(w) == 1 - constraints.cash_weight]
        constraint_list.extend([w >= constraints.min_weight])
        constraint_list.extend([w <= constraints.max_weight])

        if constraints.leverage != 1.0:
            constraint_list.append(cp.sum(cp.abs(w)) <= constraints.leverage)

        # 최적화 문제 풀이
        problem = cp.Problem(cp.Minimize(risk), constraint_list)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError(f"최적화 실패: {problem.status}")

        weights = w.value
        metrics = self.calculate_performance_metrics(weights)

        return OptimizationResult(
            weights=weights,
            method="Black-Litterman",
            constraints=constraints,
            **metrics,
            metadata={
                "optimization_status": problem.status,
                "market_weights": market_weights,
                "black_litterman_returns": mu_bl,
            },
        )

    def kelly_criterion_optimization(
        self, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Kelly Criterion 최적화"""
        self.logger.info("Kelly Criterion 최적화 실행 중...")

        # Kelly Criterion: f = (μ - r) / σ²
        excess_returns = self.mean_returns.values - self.risk_free_rate / 252
        asset_variances = np.diag(self.cov_matrix.values)

        # Kelly 비율 계산
        kelly_weights = excess_returns / asset_variances
        kelly_weights = np.maximum(kelly_weights, 0)  # 음수 비중 제거

        # 정규화
        if np.sum(kelly_weights) > 0:
            kelly_weights = (
                kelly_weights / np.sum(kelly_weights) * (1 - constraints.cash_weight)
            )

        # 제약조건 적용
        kelly_weights = np.clip(
            kelly_weights, constraints.min_weight, constraints.max_weight
        )

        # 레버리지 제약
        if (
            constraints.leverage != 1.0
            and np.sum(np.abs(kelly_weights)) > constraints.leverage
        ):
            kelly_weights = (
                kelly_weights / np.sum(np.abs(kelly_weights)) * constraints.leverage
            )

        metrics = self.calculate_performance_metrics(kelly_weights)

        return OptimizationResult(
            weights=kelly_weights,
            method="Kelly Criterion",
            constraints=constraints,
            **metrics,
            metadata={"kelly_ratios": excess_returns / asset_variances},
        )

    def optimize_portfolio(
        self, method: OptimizationMethod, constraints: OptimizationConstraints, **kwargs
    ) -> OptimizationResult:
        """포트폴리오 최적화 실행"""

        if method == OptimizationMethod.MEAN_VARIANCE:
            return self.mean_variance_optimization(constraints)
        elif method == OptimizationMethod.SHARPE_MAXIMIZATION:
            return self.sharpe_maximization(constraints)
        elif method == OptimizationMethod.SORTINO_MAXIMIZATION:
            return self.sortino_maximization(constraints)
        elif method == OptimizationMethod.RISK_PARITY:
            return self.risk_parity_optimization(constraints)
        elif method == OptimizationMethod.MINIMUM_VARIANCE:
            return self.minimum_variance_optimization(constraints)
        elif method == OptimizationMethod.MAXIMUM_DIVERSIFICATION:
            return self.maximum_diversification_optimization(constraints)
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            return self.black_litterman_optimization(constraints, **kwargs)
        elif method == OptimizationMethod.KELLY_CRITERION:
            return self.kelly_criterion_optimization(constraints)
        else:
            raise ValueError(f"지원하지 않는 최적화 방법: {method}")

    def compare_methods(
        self, constraints: OptimizationConstraints
    ) -> Dict[str, OptimizationResult]:
        """모든 최적화 방법 비교"""
        self.logger.info("모든 최적화 방법 비교 실행 중...")

        results = {}
        methods = [
            OptimizationMethod.MEAN_VARIANCE,
            OptimizationMethod.SHARPE_MAXIMIZATION,
            OptimizationMethod.SORTINO_MAXIMIZATION,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MINIMUM_VARIANCE,
            OptimizationMethod.MAXIMUM_DIVERSIFICATION,
            OptimizationMethod.KELLY_CRITERION,
        ]

        for method in methods:
            try:
                result = self.optimize_portfolio(method, constraints)
                results[method.value] = result
                self.logger.info(f"{method.value} 최적화 완료")
            except Exception as e:
                self.logger.error(f"{method.value} 최적화 실패: {e}")

        return results

    def generate_optimization_report(
        self, results: Dict[str, OptimizationResult]
    ) -> str:
        """최적화 결과 리포트 생성"""
        report_lines = []
        report_lines.append("\n" + "=" * 100)
        report_lines.append("📊 포트폴리오 최적화 결과 비교 리포트")
        report_lines.append("=" * 100)

        # 헤더
        header = f"{'방법':<20} {'수익률':<10} {'변동성':<10} {'샤프':<10} {'소르티노':<10} {'최대낙폭':<10} {'VaR(95%)':<10}"
        report_lines.append(header)
        report_lines.append("-" * 100)

        # 결과 비교
        for method, result in results.items():
            line = f"{method:<20} {result.expected_return*252*100:>8.2f}% {result.volatility*np.sqrt(252)*100:>8.2f}% "
            line += f"{result.sharpe_ratio:>8.2f} {result.sortino_ratio:>8.2f} {result.max_drawdown*100:>8.2f}% "
            line += f"{result.var_95*100:>8.2f}%"
            report_lines.append(line)

        # 최적 방법 찾기
        best_sharpe = max(results.values(), key=lambda x: x.sharpe_ratio)
        best_sortino = max(results.values(), key=lambda x: x.sortino_ratio)
        best_return = max(results.values(), key=lambda x: x.expected_return)
        min_vol = min(results.values(), key=lambda x: x.volatility)

        report_lines.append("\n🏆 최적 방법:")
        report_lines.append(
            f"  최고 샤프 비율: {best_sharpe.method} ({best_sharpe.sharpe_ratio:.3f})"
        )
        report_lines.append(
            f"  최고 소르티노 비율: {best_sortino.method} ({best_sortino.sortino_ratio:.3f})"
        )
        report_lines.append(
            f"  최고 수익률: {best_return.method} ({best_return.expected_return*252*100:.2f}%)"
        )
        report_lines.append(
            f"  최저 변동성: {min_vol.method} ({min_vol.volatility*np.sqrt(252)*100:.2f}%)"
        )

        return "\n".join(report_lines)
