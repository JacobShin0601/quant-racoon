#!/usr/bin/env python3
"""
고급 포트폴리오 관리자 - 금융권 수준의 포트폴리오 최적화 및 관리
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavior.portfolio_optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationConstraints,
    OptimizationResult,
)
from behavior.portfolio_weight import PortfolioWeightCalculator
from behavior.calculate_index import StrategyParams
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
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
)


class AdvancedPortfolioManager:
    """고급 포트폴리오 관리자 클래스 - 금융권 수준"""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.weight_calculator = PortfolioWeightCalculator(config_path)
        self.params = StrategyParams()
        self.logger = Logger()
        self.optimizer = None

    def load_portfolio_data(
        self, data_dir: str = DEFAULT_DATA_DIR
    ) -> Dict[str, pd.DataFrame]:
        """포트폴리오 데이터 로드 및 전처리"""
        # config에서 심볼 목록 가져오기
        config_symbols = self.config.get("data", {}).get("symbols", [])

        return load_and_preprocess_data(data_dir, config_symbols)

    def prepare_returns_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """수익률 데이터 준비"""
        returns_data = {}

        for symbol, df in data_dict.items():
            if "close" in df.columns:
                # 수익률 계산
                returns = df["close"].pct_change().dropna()
                returns_data[symbol] = returns

        # 공통 기간으로 정렬
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        return returns_df

    def get_optimization_constraints(self) -> OptimizationConstraints:
        """최적화 제약조건 설정"""
        portfolio_config = self.config.get("portfolio", {})

        # 기본 제약조건
        constraints = OptimizationConstraints(
            min_weight=portfolio_config.get("min_weight", 0.0),
            max_weight=portfolio_config.get("max_weight", 1.0),
            cash_weight=portfolio_config.get("cash_weight", 0.0),
            leverage=portfolio_config.get("leverage", 1.0),
            target_return=portfolio_config.get("target_return"),
            target_volatility=portfolio_config.get("target_volatility"),
            max_drawdown=portfolio_config.get("max_drawdown"),
        )

        # 그룹 제약조건 (예: 섹터별, 자산군별)
        group_constraints = portfolio_config.get("group_constraints", {})
        if group_constraints:
            constraints.group_constraints = group_constraints

        # 섹터 제약조건
        sector_constraints = portfolio_config.get("sector_constraints", {})
        if sector_constraints:
            constraints.sector_constraints = sector_constraints

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

    def calculate_advanced_portfolio_weights(
        self,
        data_dict: Dict[str, pd.DataFrame],
        method: OptimizationMethod = OptimizationMethod.SHARPE_MAXIMIZATION,
    ) -> OptimizationResult:
        """고급 포트폴리오 비중 계산"""
        # 로거 설정
        symbols = list(data_dict.keys())
        self.logger.setup_logger(
            strategy="advanced_portfolio_optimization",
            symbols=symbols,
            mode="portfolio",
        )

        # 수익률 데이터 준비
        returns_df = self.prepare_returns_data(data_dict)

        # 포트폴리오 최적화 엔진 초기화
        portfolio_config = self.get_portfolio_config()
        self.optimizer = PortfolioOptimizer(
            returns=returns_df, risk_free_rate=portfolio_config.risk_free_rate
        )

        # 최적화 제약조건 설정
        constraints = self.get_optimization_constraints()

        # 포트폴리오 최적화 실행
        self.logger.log_info(f"포트폴리오 최적화 실행 중... ({method.value})")
        result = self.optimizer.optimize_portfolio(method, constraints)

        # 주요 결과만 로그
        self.logger.log_info(
            f"최적화 완료 - 샤프: {result.sharpe_ratio:.3f}, 수익률: {result.expected_return*252*100:.2f}%"
        )

        # 개별 종목 비중 요약
        weight_summary = ", ".join(
            [
                f"{symbol}: {result.weights[i]*100:.1f}%"
                for i, symbol in enumerate(symbols)
            ]
        )
        self.logger.log_info(f"비중 분배: {weight_summary}")

        return result

    def compare_all_optimization_methods(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, OptimizationResult]:
        """모든 최적화 방법 비교"""
        self.logger.log_section("🔍 모든 최적화 방법 비교")

        # 수익률 데이터 준비
        returns_df = self.prepare_returns_data(data_dict)

        # 포트폴리오 최적화 엔진 초기화
        portfolio_config = self.get_portfolio_config()
        self.optimizer = PortfolioOptimizer(
            returns=returns_df, risk_free_rate=portfolio_config.risk_free_rate
        )

        # 최적화 제약조건 설정
        constraints = self.get_optimization_constraints()

        # 모든 방법 비교
        results = self.optimizer.compare_methods(constraints)

        # 비교 리포트 생성
        report = self.optimizer.generate_optimization_report(results)
        print(report)

        # 결과 로그
        self.logger.log_info(f"총 {len(results)}개 최적화 방법 비교 완료")

        # 최적 방법 찾기
        best_sharpe = max(results.values(), key=lambda x: x.sharpe_ratio)
        best_sortino = max(results.values(), key=lambda x: x.sortino_ratio)

        self.logger.log_subsection("🏆 최적 방법")
        self.logger.log_info(
            f"최고 샤프 비율: {best_sharpe.method} ({best_sharpe.sharpe_ratio:.3f})"
        )
        self.logger.log_info(
            f"최고 소르티노 비율: {best_sortino.method} ({best_sortino.sortino_ratio:.3f})"
        )

        return results

    def validate_optimization_result(self, result: OptimizationResult) -> bool:
        """최적화 결과 유효성 검증"""
        self.logger.log_info("최적화 결과 유효성 검증 시작...")

        # 기본 검증
        if result.weights is None or len(result.weights) == 0:
            self.logger.log_error("비중이 비어있습니다")
            return False

        # 비중 합계 검증
        total_weight = np.sum(result.weights)
        if abs(total_weight - (1 - result.constraints.cash_weight)) > 1e-6:
            self.logger.log_error(f"비중 합계 오류: {total_weight:.6f}")
            return False

        # 비중 범위 검증
        if np.any(result.weights < result.constraints.min_weight - 1e-6):
            self.logger.log_error("최소 비중 제약 위반")
            return False

        if np.any(result.weights > result.constraints.max_weight + 1e-6):
            self.logger.log_error("최대 비중 제약 위반")
            return False

        # 성과 지표 검증
        if result.sharpe_ratio < -10 or result.sharpe_ratio > 10:
            self.logger.log_warning(f"샤프 비율이 비정상적: {result.sharpe_ratio}")

        if result.sortino_ratio < -10 or result.sortino_ratio > 10:
            self.logger.log_warning(f"소르티노 비율이 비정상적: {result.sortino_ratio}")

        self.logger.log_success("최적화 결과 유효성 검증 통과")
        return True

    def save_optimization_result(
        self,
        result: OptimizationResult,
        output_path: str = "log/optimization_result.json",
    ):
        """최적화 결과를 JSON 파일로 저장"""
        # 결과 데이터 구성
        result_data = {
            "calculation_date": datetime.now().isoformat(),
            "method": result.method,
            "asset_names": self.optimizer.asset_names if self.optimizer else [],
            "weights": result.weights.tolist(),
            "performance_metrics": {
                "expected_return": result.expected_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "var_95": result.var_95,
                "cvar_95": result.cvar_95,
                "diversification_ratio": result.diversification_ratio,
            },
            "constraints": {
                "min_weight": result.constraints.min_weight,
                "max_weight": result.constraints.max_weight,
                "cash_weight": result.constraints.cash_weight,
                "leverage": result.constraints.leverage,
                "target_return": result.constraints.target_return,
                "target_volatility": result.constraints.target_volatility,
            },
            "metadata": result.metadata,
        }

        save_json_data(result_data, output_path, "포트폴리오 최적화 결과")

    def generate_advanced_portfolio_report(self, result: OptimizationResult) -> str:
        """고급 포트폴리오 리포트 생성"""
        report_lines = []
        report_lines.append("\n" + "=" * 100)
        report_lines.append("🚀 고급 포트폴리오 최적화 리포트")
        report_lines.append("=" * 100)

        report_lines.append(f"\n📅 계산 일시: {datetime.now()}")
        report_lines.append(f"🎯 최적화 방법: {result.method}")
        report_lines.append(
            f"📈 구성 종목: {', '.join(self.optimizer.asset_names) if self.optimizer else 'N/A'}"
        )

        # 성과 지표
        report_lines.append(f"\n📊 성과 지표:")
        report_lines.append("-" * 50)
        report_lines.append(
            f"예상 수익률 (연간): {result.expected_return*252*100:>8.2f}%"
        )
        report_lines.append(
            f"변동성 (연간):      {result.volatility*np.sqrt(252)*100:>8.2f}%"
        )
        report_lines.append(f"샤프 비율:          {result.sharpe_ratio:>8.3f}")
        report_lines.append(f"소르티노 비율:      {result.sortino_ratio:>8.3f}")
        report_lines.append(f"최대 낙폭:          {result.max_drawdown*100:>8.2f}%")
        report_lines.append(f"VaR (95%):         {result.var_95*100:>8.2f}%")
        report_lines.append(f"CVaR (95%):        {result.cvar_95*100:>8.2f}%")
        report_lines.append(f"분산화 비율:        {result.diversification_ratio:>8.3f}")

        # 개별 종목 비중
        report_lines.append(f"\n📋 개별 종목 비중:")
        report_lines.append("-" * 50)
        for i, symbol in enumerate(
            self.optimizer.asset_names if self.optimizer else []
        ):
            weight = result.weights[i]
            report_lines.append(f"{symbol:<10}: {weight*100:>8.2f}%")

        # 제약조건
        report_lines.append(f"\n⚙️ 제약조건:")
        report_lines.append("-" * 30)
        report_lines.append(f"최소 비중: {result.constraints.min_weight}")
        report_lines.append(f"최대 비중: {result.constraints.max_weight}")
        report_lines.append(f"현금 비중: {result.constraints.cash_weight}")
        report_lines.append(f"레버리지: {result.constraints.leverage}")
        if result.constraints.target_return:
            report_lines.append(
                f"목표 수익률: {result.constraints.target_return*252*100:.2f}%"
            )
        if result.constraints.target_volatility:
            report_lines.append(
                f"목표 변동성: {result.constraints.target_volatility*np.sqrt(252)*100:.2f}%"
            )

        # 메타데이터
        if result.metadata:
            report_lines.append(f"\n🔧 메타데이터:")
            report_lines.append("-" * 30)
            for key, value in result.metadata.items():
                if isinstance(value, float):
                    report_lines.append(f"{key}: {format_number(value)}")
                else:
                    report_lines.append(f"{key}: {value}")

        return "\n".join(report_lines)

    def run_advanced_portfolio_management(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        method: OptimizationMethod = OptimizationMethod.SHARPE_MAXIMIZATION,
        compare_methods: bool = False,
        save_result: bool = True,
    ) -> OptimizationResult:
        """고급 포트폴리오 관리 전체 프로세스 실행"""
        print_section_header("🚀 고급 포트폴리오 관리 시스템 시작")
        print(f"📁 데이터 디렉토리: {data_dir}")
        print(f"🎯 최적화 방법: {method.value}")

        # 1. 데이터 로드
        self.logger.log_info("데이터 로드 시작...")
        data_dict = self.load_portfolio_data(data_dir)
        symbols = list(data_dict.keys())
        self.logger.log_success(f"데이터 로드 완료: {symbols}")

        if compare_methods:
            # 모든 방법 비교
            results = self.compare_all_optimization_methods(data_dict)
            return results
        else:
            # 단일 방법 최적화
            result = self.calculate_advanced_portfolio_weights(data_dict, method)

            # 2. 결과 유효성 검증
            if not self.validate_optimization_result(result):
                self.logger.log_error("최적화 결과 유효성 검증 실패")
                return None

            # 3. 리포트 생성 및 출력
            report = self.generate_advanced_portfolio_report(result)
            print(report)

            # 4. 결과 저장 (선택사항)
            if save_result:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    "log", f"optimization_result_{timestamp}.json"
                )
                self.save_optimization_result(result, output_path)
                self.logger.log_success(f"최적화 결과 저장 완료: {output_path}")

            # JSON 로그 저장
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "optimization_method": method.value,
                "performance_metrics": {
                    "expected_return": result.expected_return,
                    "volatility": result.volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown": result.max_drawdown,
                },
                "constraints": {
                    "min_weight": result.constraints.min_weight,
                    "max_weight": result.constraints.max_weight,
                    "cash_weight": result.constraints.cash_weight,
                    "leverage": result.constraints.leverage,
                },
            }
            self.logger.save_json_log(log_data, f"advanced_portfolio_{timestamp}.json")

            return result


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="고급 포트폴리오 관리자")
    parser.add_argument("--data_dir", default="data", help="데이터 디렉토리 경로")
    parser.add_argument(
        "--method",
        default="sharpe_maximization",
        choices=[m.value for m in OptimizationMethod],
        help="최적화 방법",
    )
    parser.add_argument("--compare", action="store_true", help="모든 방법 비교")
    parser.add_argument("--save_result", action="store_true", help="결과를 파일로 저장")

    args = parser.parse_args()

    # 고급 포트폴리오 매니저 초기화
    portfolio_manager = AdvancedPortfolioManager()

    # 최적화 방법 선택
    method = OptimizationMethod(args.method)

    # 포트폴리오 관리 실행
    result = portfolio_manager.run_advanced_portfolio_management(
        data_dir=args.data_dir,
        method=method,
        compare_methods=args.compare,
        save_result=args.save_result,
    )


if __name__ == "__main__":
    main()
