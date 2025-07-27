#!/usr/bin/env python3
"""
고급 성과 지표 계산기
- 샤프비율, 소르티노비율, 칼마비율
- MDD, VaR, CVaR
- Buy & Hold 벤치마크 비교
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AdvancedPerformanceCalculator:
    """고급 성과 지표 계산기"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252

    def calculate_all_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series = None
    ) -> Dict[str, float]:
        """
        모든 성과 지표 계산

        Args:
            returns: 전략 수익률 시계열
            benchmark_returns: 벤치마크 수익률 시계열

        Returns:
            모든 성과 지표 딕셔너리
        """
        try:
            if returns.empty:
                return self._get_empty_metrics()

            metrics = {}

            # 기본 수익률 지표
            metrics.update(self._calculate_return_metrics(returns))

            # 위험 조정 수익률 지표
            metrics.update(self._calculate_risk_adjusted_metrics(returns))

            # 리스크 지표
            metrics.update(self._calculate_risk_metrics(returns))

            # 벤치마크 비교 지표
            if benchmark_returns is not None and not benchmark_returns.empty:
                metrics.update(
                    self._calculate_benchmark_metrics(returns, benchmark_returns)
                )

            return metrics

        except Exception as e:
            logger.error(f"성과 지표 계산 실패: {e}")
            return self._get_empty_metrics()

    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """수익률 관련 지표 계산"""
        try:
            # 누적 수익률
            cumulative_return = (1 + returns).prod() - 1

            # 연율화 수익률
            n_periods = len(returns)
            annualized_return = (1 + cumulative_return) ** (
                self.trading_days_per_year / n_periods
            ) - 1

            # 변동성 (연율화)
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)

            return {
                "cumulative_return": cumulative_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "total_periods": n_periods,
            }

        except Exception as e:
            logger.error(f"수익률 지표 계산 실패: {e}")
            return {}

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """위험 조정 수익률 지표 계산"""
        try:
            if returns.empty or returns.std() == 0:
                return {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0}

            # 초과 수익률
            excess_returns = returns - (
                self.risk_free_rate / self.trading_days_per_year
            )

            # 샤프 비율
            sharpe_ratio = (
                excess_returns.mean()
                / returns.std()
                * np.sqrt(self.trading_days_per_year)
            )

            # 소르티노 비율 (하향 편차만 고려)
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(
                    self.trading_days_per_year
                )
                sortino_ratio = (
                    excess_returns.mean()
                    * self.trading_days_per_year
                    / downside_deviation
                    if downside_deviation > 0
                    else 0
                )
            else:
                sortino_ratio = float("inf") if excess_returns.mean() > 0 else 0

            # 칼마 비율 (연율화 수익률 / 최대낙폭)
            max_drawdown = self._calculate_max_drawdown(returns)
            calmar_ratio = (
                (excess_returns.mean() * self.trading_days_per_year) / abs(max_drawdown)
                if max_drawdown < 0
                else 0
            )

            return {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
            }

        except Exception as e:
            logger.error(f"위험 조정 지표 계산 실패: {e}")
            return {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0}

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """리스크 지표 계산"""
        try:
            # 최대 낙폭 (MDD)
            max_drawdown = self._calculate_max_drawdown(returns)

            # VaR (95% 신뢰수준)
            var_95 = self._calculate_var(returns, 0.05)
            var_99 = self._calculate_var(returns, 0.01)

            # CVaR (조건부 VaR)
            cvar_95 = self._calculate_cvar(returns, 0.05)
            cvar_99 = self._calculate_cvar(returns, 0.01)

            # 스큐니스와 커토시스
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            # 최대 연속 손실일
            max_consecutive_losses = self._calculate_max_consecutive_losses(returns)

            return {
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "cvar_99": cvar_99,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "max_consecutive_losses": max_consecutive_losses,
            }

        except Exception as e:
            logger.error(f"리스크 지표 계산 실패: {e}")
            return {}

    def _calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """벤치마크 대비 성과 지표"""
        try:
            # 공통 기간으로 맞춤
            common_index = returns.index.intersection(benchmark_returns.index)
            if len(common_index) == 0:
                return {}

            strategy_returns = returns.loc[common_index]
            bench_returns = benchmark_returns.loc[common_index]

            # 알파 (초과 수익률)
            alpha = strategy_returns.mean() - bench_returns.mean()
            alpha_annualized = alpha * self.trading_days_per_year

            # 베타 (시장 민감도)
            beta = np.cov(strategy_returns, bench_returns)[0, 1] / np.var(bench_returns)

            # 트래킹 에러 (초과 수익률의 변동성)
            tracking_error = (strategy_returns - bench_returns).std() * np.sqrt(
                self.trading_days_per_year
            )

            # 정보 비율 (알파 / 트래킹 에러)
            information_ratio = (
                alpha_annualized / tracking_error if tracking_error > 0 else 0
            )

            # 상승/하락 시장에서의 성과
            up_market = bench_returns > 0
            down_market = bench_returns <= 0

            up_capture = (
                strategy_returns[up_market].mean() / bench_returns[up_market].mean()
                if up_market.sum() > 0 and bench_returns[up_market].mean() != 0
                else 1
            )

            down_capture = (
                strategy_returns[down_market].mean() / bench_returns[down_market].mean()
                if down_market.sum() > 0 and bench_returns[down_market].mean() != 0
                else 1
            )

            return {
                "alpha": alpha_annualized,
                "beta": beta,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "up_capture": up_capture,
                "down_capture": down_capture,
            }

        except Exception as e:
            logger.error(f"벤치마크 지표 계산 실패: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭 계산"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()

        except Exception as e:
            logger.error(f"최대 낙폭 계산 실패: {e}")
            return 0.0

    def _calculate_var(self, returns: pd.Series, alpha: float) -> float:
        """Value at Risk 계산"""
        try:
            return returns.quantile(alpha)

        except Exception as e:
            logger.error(f"VaR 계산 실패: {e}")
            return 0.0

    def _calculate_cvar(self, returns: pd.Series, alpha: float) -> float:
        """Conditional Value at Risk 계산"""
        try:
            var = self._calculate_var(returns, alpha)
            cvar = returns[returns <= var].mean()
            return cvar if not np.isnan(cvar) else 0.0

        except Exception as e:
            logger.error(f"CVaR 계산 실패: {e}")
            return 0.0

    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """최대 연속 손실일 계산"""
        try:
            losses = returns < 0
            max_consecutive = 0
            current_consecutive = 0

            for loss in losses:
                if loss:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            return max_consecutive

        except Exception as e:
            logger.error(f"연속 손실일 계산 실패: {e}")
            return 0

    def _get_empty_metrics(self) -> Dict[str, float]:
        """빈 지표 딕셔너리 반환"""
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "cvar_99": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "max_consecutive_losses": 0,
            "total_periods": 0,
        }

    def calculate_buy_hold_benchmark(
        self, price_data: pd.DataFrame, start_date: str = None, end_date: str = None
    ) -> pd.Series:
        """Buy & Hold 벤치마크 수익률 계산"""
        try:
            if "close" not in price_data.columns:
                logger.error("가격 데이터에 'close' 컬럼이 없습니다")
                return pd.Series()

            # 기간 필터링 (timezone 문제를 피하기 위해 문자열 비교 사용)
            data = price_data.copy()

            # 인덱스를 날짜 문자열로 변환 (UTC 변환으로 timezone 문제 해결)
            try:
                # timezone-aware 데이터를 UTC로 변환 후 날짜 문자열 생성
                if hasattr(data.index, "tz") and data.index.tz is not None:
                    data["date_str"] = data.index.tz_convert("UTC").strftime("%Y-%m-%d")
                else:
                    data["date_str"] = pd.to_datetime(data.index, utc=True).strftime(
                        "%Y-%m-%d"
                    )
            except Exception:
                # fallback: 인덱스를 직접 문자열로 변환
                data["date_str"] = [str(d)[:10] for d in data.index]

            # 날짜 필터링 (문자열 비교)
            if start_date:
                data = data[data["date_str"] >= start_date]
            if end_date:
                data = data[data["date_str"] <= end_date]

            if len(data) < 2:
                logger.error("Buy & Hold 계산을 위한 데이터가 부족합니다")
                return pd.Series()

            # 일일 수익률 계산
            returns = data["close"].pct_change().dropna()

            return returns

        except Exception as e:
            logger.error(f"Buy & Hold 벤치마크 계산 실패: {e}")
            return pd.Series()

    def calculate_buy_hold_cumulative_return(
        self, price_data: pd.DataFrame, start_date: str = None, end_date: str = None
    ) -> float:
        """Buy & Hold 누적 수익률 계산 (스칼라 값 반환)"""
        try:
            if "close" not in price_data.columns:
                logger.error("가격 데이터에 'close' 컬럼이 없습니다")
                return 0.0

            # 기간 필터링 (timezone 문제를 피하기 위해 문자열 비교 사용)
            data = price_data.copy()

            # 인덱스를 날짜 문자열로 변환
            try:
                if hasattr(data.index, "tz") and data.index.tz is not None:
                    data["date_str"] = data.index.tz_convert("UTC").strftime("%Y-%m-%d")
                else:
                    data["date_str"] = pd.to_datetime(data.index, utc=True).strftime(
                        "%Y-%m-%d"
                    )
            except Exception:
                data["date_str"] = [str(d)[:10] for d in data.index]

            # 날짜 필터링
            if start_date:
                data = data[data["date_str"] >= start_date]
            if end_date:
                data = data[data["date_str"] <= end_date]

            if len(data) < 2:
                logger.error("Buy & Hold 계산을 위한 데이터가 부족합니다")
                return 0.0

            # 시작가와 종료가 추출
            start_price = data["close"].iloc[0]
            end_price = data["close"].iloc[-1]

            # 누적 수익률 계산
            cumulative_return = (end_price / start_price) - 1

            logger.info(
                f"📊 Buy & Hold 계산: 시작가={start_price:.2f}, 종료가={end_price:.2f}, "
                f"수익률={cumulative_return:.4f} ({start_date} ~ {end_date})"
            )

            return cumulative_return

        except Exception as e:
            logger.error(f"Buy & Hold 누적 수익률 계산 실패: {e}")
            return 0.0

    def create_performance_comparison_table(
        self,
        strategy_metrics: Dict[str, float],
        benchmark_metrics: Dict[str, float],
        strategy_name: str = "전략",
        benchmark_name: str = "Buy & Hold",
    ) -> str:
        """성과 비교 테이블 생성"""
        try:
            table_lines = []
            table_lines.append("=" * 80)
            table_lines.append(
                f"📊 성과 지표 비교: {strategy_name} vs {benchmark_name}"
            )
            table_lines.append("=" * 80)

            # 헤더
            table_lines.append(
                f"{'지표':<20} {strategy_name:<15} {benchmark_name:<15} {'차이':<15}"
            )
            table_lines.append("-" * 80)

            # 수익률 지표
            table_lines.append("【 수익률 지표 】")
            self._add_comparison_row(
                table_lines,
                "누적수익률",
                strategy_metrics,
                benchmark_metrics,
                "cumulative_return",
                "percent",
            )
            self._add_comparison_row(
                table_lines,
                "연율화수익률",
                strategy_metrics,
                benchmark_metrics,
                "annualized_return",
                "percent",
            )
            self._add_comparison_row(
                table_lines,
                "변동성",
                strategy_metrics,
                benchmark_metrics,
                "volatility",
                "percent",
            )

            # 위험조정 수익률
            table_lines.append("\n【 위험조정 수익률 】")
            self._add_comparison_row(
                table_lines,
                "샤프비율",
                strategy_metrics,
                benchmark_metrics,
                "sharpe_ratio",
                "ratio",
            )
            self._add_comparison_row(
                table_lines,
                "소르티노비율",
                strategy_metrics,
                benchmark_metrics,
                "sortino_ratio",
                "ratio",
            )
            self._add_comparison_row(
                table_lines,
                "칼마비율",
                strategy_metrics,
                benchmark_metrics,
                "calmar_ratio",
                "ratio",
            )

            # 리스크 지표
            table_lines.append("\n【 리스크 지표 】")
            self._add_comparison_row(
                table_lines,
                "최대낙폭(MDD)",
                strategy_metrics,
                benchmark_metrics,
                "max_drawdown",
                "percent",
            )
            self._add_comparison_row(
                table_lines,
                "VaR(95%)",
                strategy_metrics,
                benchmark_metrics,
                "var_95",
                "percent",
            )
            self._add_comparison_row(
                table_lines,
                "CVaR(95%)",
                strategy_metrics,
                benchmark_metrics,
                "cvar_95",
                "percent",
            )

            # 분포 특성
            table_lines.append("\n【 분포 특성 】")
            self._add_comparison_row(
                table_lines,
                "왜도(Skewness)",
                strategy_metrics,
                benchmark_metrics,
                "skewness",
                "ratio",
            )
            self._add_comparison_row(
                table_lines,
                "첨도(Kurtosis)",
                strategy_metrics,
                benchmark_metrics,
                "kurtosis",
                "ratio",
            )

            table_lines.append("=" * 80)

            return "\n".join(table_lines)

        except Exception as e:
            logger.error(f"성과 비교 테이블 생성 실패: {e}")
            return "성과 비교 테이블 생성 중 오류가 발생했습니다."

    def _add_comparison_row(
        self,
        table_lines: List[str],
        label: str,
        strategy_metrics: Dict[str, float],
        benchmark_metrics: Dict[str, float],
        key: str,
        format_type: str,
    ):
        """비교 테이블 행 추가"""
        try:
            strategy_val = strategy_metrics.get(key, 0)
            benchmark_val = benchmark_metrics.get(key, 0)
            diff = strategy_val - benchmark_val

            if format_type == "percent":
                strategy_str = f"{strategy_val*100:>6.2f}%"
                benchmark_str = f"{benchmark_val*100:>6.2f}%"
                diff_str = f"{diff*100:>+6.2f}%"
            else:  # ratio
                strategy_str = f"{strategy_val:>8.3f}"
                benchmark_str = f"{benchmark_val:>8.3f}"
                diff_str = f"{diff:>+8.3f}"

            table_lines.append(
                f"{label:<20} {strategy_str:<15} {benchmark_str:<15} {diff_str:<15}"
            )

        except Exception as e:
            logger.error(f"테이블 행 추가 실패: {e}")
            table_lines.append(f"{label:<20} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
