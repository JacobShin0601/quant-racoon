#!/usr/bin/env python3
"""
통합 분석 시스템
- QuantAnalyst: 기술적 지표 기반 분석
- FundamentalAnalyst: 재무지표 기반 분석
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
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.correlation import CorrelationAnalyzer
from actions.linear_regression import LinearRegressionAnalyzer
from actions.lasso_regression import LassoRegressionAnalyzer
from actions.random_forest import RandomForestAnalyzer
from actions.multi_layer_perceptron import MLPAnalyzer
from actions.bayesian_distribution import BayesianDistributionAnalyzer
from actions.financial_analysis import FinancialAnalyzer
from agent.helper import (
    Logger,
    load_config,
    load_and_preprocess_data,
    save_analysis_results,
    create_analysis_folder_structure,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
)


class QuantAnalyst:
    """기술적 지표 기반 정량 분석 시스템"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        config_path: str = DEFAULT_CONFIG_PATH,
        return_type: str = "percentage",  # "percentage" or "log"
        top_features: int = 10,
        analysis_dir: str = "analysis",  # 분석 결과 저장 디렉토리
    ):
        self.data_dir = data_dir
        self.config = load_config(config_path)
        self.return_type = return_type
        self.top_features = top_features
        self.analysis_dir = analysis_dir
        self.logger = Logger()
        self.analysis_start_time = datetime.now()
        self.execution_uuid = None  # UUID 초기화

        # 분석 폴더 구조 생성
        create_analysis_folder_structure(analysis_dir)

        # 분석기들 초기화
        self.correlation_analyzer = CorrelationAnalyzer()
        self.linear_regression_analyzer = LinearRegressionAnalyzer()
        self.lasso_regression_analyzer = LassoRegressionAnalyzer()
        self.random_forest_analyzer = RandomForestAnalyzer()
        self.mlp_analyzer = MLPAnalyzer()
        self.bayesian_analyzer = BayesianDistributionAnalyzer()

        # 분석 결과 저장
        self.analysis_results = {}
        self.prepared_data = {}

    def prepare_data(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """데이터 전처리 및 수익률 계산 (기술적 지표만 사용)"""
        self.logger.log_info("📊 기술적 지표 데이터 전처리 및 수익률 계산 중...")

        # 기술적 지표 관련 컬럼들 (재무지표 제외)
        technical_columns = {
            "datetime", "date", "time", "timestamp", "open", "high", "low", "close", "volume",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
            "stoch_k", "stoch_d", "stoch_slow_k", "stoch_slow_d",
            "atr", "adx", "cci", "williams_r", "mfi", "obv",
        }

        # 제외할 컬럼들 (재무지표 및 기타)
        excluded_columns = {
            "datetime", "date", "time", "timestamp", "open", "high", "low", "close", "volume",
        }

        prepared_data = {}

        for symbol, data in data_dict.items():
            # 데이터 복사
            df = data.copy()

            # 기존 returns 컬럼이 있는지 확인
            if "returns" in df.columns:
                # 기존 returns 컬럼을 target으로 사용
                df["return"] = df["returns"]
                self.logger.log_info(f"  {symbol}: 기존 returns 컬럼 사용")
            else:
                # 수익률 계산 (target 변수)
                if self.return_type == "log":
                    df["return"] = np.log(df["close"] / df["close"].shift(1))
                else:  # percentage
                    df["return"] = df["close"].pct_change() * 100
                self.logger.log_info(f"  {symbol}: 새로운 수익률 계산")

            # NaN 제거
            df = df.dropna()

            # 기술적 지표 컬럼들만 선택 (재무지표 제외)
            feature_columns = []
            for col in df.columns:
                if col not in excluded_columns and col != "return" and col != "returns":
                    # 재무지표가 아닌 컬럼만 선택 (pe_ratio, market_cap 등으로 시작하지 않는 컬럼)
                    if not any(col.startswith(prefix) for prefix in [
                        "pe_", "market_", "enterprise_", "return_on_", "debt_", "current_",
                        "profit_", "operating_", "ebitda_", "revenue_", "earnings_",
                        "dividend_", "payout_", "book_", "cash_", "total_", "quarterly_",
                        "calculated_", "latest_", "beta", "fifty_", "two_hundred_",
                        "shares_", "held_", "institutional_", "short_", "float_"
                    ]):
                        feature_columns.append(col)

            # 숫자형 데이터만 선택
            numeric_columns = []
            for col in feature_columns:
                try:
                    pd.to_numeric(df[col], errors="raise")
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    self.logger.log_info(f"    {symbol}: {col} 컬럼 제외 (숫자가 아님)")

            # 수익률을 마지막 컬럼으로 이동
            columns_order = numeric_columns + ["return"]
            df = df[columns_order]

            # 모든 컬럼을 숫자형으로 변환
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # NaN 제거 (숫자 변환 후)
            df = df.dropna()

            prepared_data[symbol] = df
            self.logger.log_info(
                f"  {symbol}: {len(df)}개 데이터 포인트, {len(numeric_columns)}개 기술적 지표"
            )

        self.prepared_data = prepared_data
        return prepared_data

    def run_correlation_analysis(self, symbol: str) -> Dict[str, Any]:
        """상관관계 분석 실행"""
        self.logger.log_info(f"🔍 {symbol} 상관관계 분석 실행...")

        data = self.prepared_data[symbol]
        result = self.correlation_analyzer.analyze(
            data, target_column="return", top_n=self.top_features
        )

        self.analysis_results[symbol] = {"correlation": result}
        return result

    def run_linear_regression_analysis(
        self, symbol: str, top_features: int = 5
    ) -> Dict[str, Any]:
        """선형회귀 분석 실행"""
        self.logger.log_info(f"📈 {symbol} 선형회귀 분석 실행...")

        data = self.prepared_data[symbol]

        # 상위 특성 선택
        if (
            symbol in self.analysis_results
            and "correlation" in self.analysis_results[symbol]
        ):
            top_corr_features = self.analysis_results[symbol]["correlation"][
                "top_features"
            ]
            selected_features = top_corr_features[
                : min(top_features, len(top_corr_features))
            ]
        else:
            # 상관관계 분석이 없으면 모든 특성 사용
            feature_columns = [col for col in data.columns if col != "return"]
            selected_features = feature_columns[:top_features]

        result = self.linear_regression_analyzer.analyze(
            data, target_column="return", feature_columns=selected_features
        )

        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        self.analysis_results[symbol]["linear_regression"] = result
        return result

    def run_lasso_regression_analysis(self, symbol: str) -> Dict[str, Any]:
        """Lasso 회귀 분석 실행"""
        self.logger.log_info(f"🎯 {symbol} Lasso 회귀 분석 실행...")

        data = self.prepared_data[symbol]
        feature_columns = [col for col in data.columns if col != "return"]

        result = self.lasso_regression_analyzer.analyze(
            data, target_column="return", feature_columns=feature_columns
        )

        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        self.analysis_results[symbol]["lasso_regression"] = result
        return result

    def run_random_forest_analysis(self, symbol: str) -> Dict[str, Any]:
        """랜덤 포레스트 분석 실행"""
        self.logger.log_info(f"🌲 {symbol} 랜덤 포레스트 분석 실행...")

        data = self.prepared_data[symbol]
        feature_columns = [col for col in data.columns if col != "return"]

        result = self.random_forest_analyzer.analyze(
            data, target_column="return", feature_columns=feature_columns
        )

        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        self.analysis_results[symbol]["random_forest"] = result
        return result

    def run_mlp_analysis(self, symbol: str) -> Dict[str, Any]:
        """MLP 분석 실행"""
        self.logger.log_info(f"🧠 {symbol} MLP 분석 실행...")

        data = self.prepared_data[symbol]
        feature_columns = [col for col in data.columns if col != "return"]

        result = self.mlp_analyzer.analyze(
            data, target_column="return", feature_columns=feature_columns
        )

        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        self.analysis_results[symbol]["mlp"] = result
        return result

    def run_bayesian_distribution_analysis(self, symbol: str) -> Dict[str, Any]:
        """베이지안 분포 분석 실행"""
        self.logger.log_info(f"🔮 {symbol} 베이지안 분포 분석 실행...")

        data = self.prepared_data[symbol]

        # 1. 수익률 분포 분석
        dist_result = self.bayesian_analyzer.analyze_return_distribution(
            data, target_column="return"
        )

        # 2. 베이지안 회귀 분석
        reg_result = self.bayesian_analyzer.analyze_bayesian_regression(
            data, target_column="return"
        )

        # 3. 변동성 분석
        vol_result = self.bayesian_analyzer.analyze_volatility(
            data, target_column="return", window=20
        )

        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        self.analysis_results[symbol]["bayesian_distribution"] = dist_result
        self.analysis_results[symbol]["bayesian_regression"] = reg_result
        self.analysis_results[symbol]["volatility"] = vol_result
        return {
            "distribution": dist_result,
            "regression": reg_result,
            "volatility": vol_result,
        }

    def run_full_analysis(self, symbols: List[str] = None) -> Dict[str, Any]:
        """전체 분석 실행"""
        # 로거 설정
        self.logger.setup_logger(
            strategy="quant_analysis", symbols=symbols or [], mode="analysis"
        )

        # 종합 요약 로거 설정
        self.logger.setup_summary_logger(
            symbols=symbols or [], timestamp=self.analysis_start_time
        )

        self.logger.log_section("🎯 기술적 지표 기반 정량 분석 시스템 시작")
        self.logger.log_info(f"📁 데이터 디렉토리: {self.data_dir}")
        self.logger.log_info(f"📊 수익률 타입: {self.return_type}")
        self.logger.log_info(f"🔝 상위 특성 수: {self.top_features}")

        # 데이터 로드
        self.logger.log_info("📂 데이터 로딩 중...")
        data_dict = load_and_preprocess_data(self.data_dir, symbols)
        self.logger.log_success(f"✅ 데이터 로딩 완료 ({len(data_dict)}개 종목)")

        # 데이터 전처리
        prepared_data = self.prepare_data(data_dict)

        # 각 종목별 분석 실행
        for symbol in prepared_data.keys():
            self.logger.log_info(f"🔄 {symbol} 분석 시작...")

            try:
                # 1. 상관관계 분석
                corr_result = self.run_correlation_analysis(symbol)

                # 2. 선형회귀 분석
                lr_result = self.run_linear_regression_analysis(symbol)

                # 3. Lasso 회귀 분석
                lasso_result = self.run_lasso_regression_analysis(symbol)

                # 4. 랜덤 포레스트 분석
                rf_result = self.run_random_forest_analysis(symbol)

                # 5. MLP 분석
                mlp_result = self.run_mlp_analysis(symbol)

                # 6. 베이지안 분포 분석
                bayesian_result = self.run_bayesian_distribution_analysis(symbol)

                self.logger.log_success(f"✅ {symbol} 분석 완료")

            except Exception as e:
                self.logger.log_error(f"❌ {symbol} 분석 중 오류: {str(e)}")

        # 종합 요약 생성
        self.generate_analysis_summary()

        return self.analysis_results

    def generate_analysis_summary(self):
        """분석 결과 종합 요약"""
        if not self.analysis_results:
            return

        self.logger.log_summary_section("📊 기술적 지표 기반 정량 분석 종합 요약 리포트")

        # 분석 설정
        self.logger.log_summary_subsection("📋 분석 설정")
        self.logger.log_summary_info(f"분석 유형: 기술적 지표 기반")
        self.logger.log_summary_info(f"수익률 타입: {self.return_type}")
        self.logger.log_summary_info(f"상위 특성 수: {self.top_features}")
        self.logger.log_summary_info(f"분석 종목 수: {len(self.analysis_results)}")

        # 종목별 요약
        self.logger.log_summary_subsection("📈 종목별 분석 요약")

        for symbol, results in self.analysis_results.items():
            self.logger.log_summary_info(f"\n{symbol}:")

            if "correlation" in results:
                top_features = results["correlation"]["top_features"][:3]
                self.logger.log_summary_info(
                    f"  상관관계 상위: {', '.join(top_features)}"
                )

            if "linear_regression" in results:
                r2 = results["linear_regression"]["r_squared"]
                self.logger.log_summary_info(f"  선형회귀 R²: {r2:.4f}")

            if "random_forest" in results:
                rf_r2 = results["random_forest"]["r_squared"]
                self.logger.log_summary_info(f"  랜덤포레스트 R²: {rf_r2:.4f}")

            if "mlp" in results:
                mlp_r2 = results["mlp"]["r_squared"]
                self.logger.log_summary_info(f"  MLP R²: {mlp_r2:.4f}")

            if "bayesian_regression" in results:
                br_r2 = results["bayesian_regression"]["bayesian_ridge"]["metrics"][
                    "br_test_r2"
                ]
                ard_r2 = results["bayesian_regression"]["ard_regression"]["metrics"][
                    "ard_test_r2"
                ]
                self.logger.log_summary_info(f"  베이지안 Ridge R²: {br_r2:.4f}")
                self.logger.log_summary_info(f"  ARD R²: {ard_r2:.4f}")

            if "bayesian_distribution" in results:
                best_dist = results["bayesian_distribution"]["best_distribution"]
                var_95 = results["bayesian_distribution"]["var_95"]
                self.logger.log_summary_info(f"  최적 분포: {best_dist}")
                self.logger.log_summary_info(f"  VaR 95%: {var_95:.4f}")

        # 모델 성능 비교
        self.logger.log_summary_subsection("🏆 모델 성능 비교")

        model_performance = {}
        for symbol, results in self.analysis_results.items():
            for model_name, result in results.items():
                if "r_squared" in result:
                    if model_name not in model_performance:
                        model_performance[model_name] = []
                    model_performance[model_name].append(result["r_squared"])

        for model_name, r2_scores in model_performance.items():
            avg_r2 = np.mean(r2_scores)
            max_r2 = np.max(r2_scores)
            self.logger.log_summary_info(
                f"{model_name}: 평균 R² = {avg_r2:.4f}, 최고 R² = {max_r2:.4f}"
            )

        # 종료 메시지
        self.logger.log_summary_section("🎉 분석 완료")
        self.logger.log_summary_success(
            f"총 {len(self.analysis_results)}개 종목 분석 완료"
        )
        self.logger.log_summary_info(f"종합 요약 로그: {self.logger.summary_log_file}")

    def save_results(self, output_path: str = None):
        """분석 결과 저장"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid_suffix = f"_{self.execution_uuid}" if self.execution_uuid else ""
            output_path = f"quant_analysis_results_{timestamp}{uuid_suffix}.json"

        # JSON 직렬화 가능한 형태로 변환
        def make_serializable(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif hasattr(obj, "__name__") and not isinstance(obj, str):
                return str(obj.__name__)
            elif hasattr(obj, "__class__") and not isinstance(obj, str):
                return str(obj.__class__.__name__)
            else:
                try:
                    import json

                    json.dumps(obj)
                    return obj
                except Exception:
                    return str(obj)

        # NaN/inf를 안전한 값으로 변환
        def clean_nan_inf(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: clean_nan_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_inf(v) for v in obj]
            elif isinstance(obj, pd.DataFrame):
                return clean_nan_inf(obj.to_dict())
            elif isinstance(obj, pd.Series):
                return clean_nan_inf(obj.to_dict())
            else:
                return obj

        serializable_results = {}
        for symbol, results in self.analysis_results.items():
            serializable_results[symbol] = {}
            for model_name, result in results.items():
                serializable_results[symbol][model_name] = clean_nan_inf(
                    make_serializable(result)
                )

        # analysis 폴더에 저장
        saved_path = save_analysis_results(
            serializable_results, 
            "quant_analysis", 
            output_path,
            self.analysis_dir
        )

        self.logger.log_success(f"✅ 분석 결과 저장: {saved_path}")
        return saved_path


class FundamentalAnalyst:
    """재무지표 기반 분석 시스템"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        config_path: str = DEFAULT_CONFIG_PATH,
        return_type: str = "percentage",  # "percentage" or "log"
        top_features: int = 10,
        analysis_dir: str = "analysis",  # 분석 결과 저장 디렉토리
    ):
        self.data_dir = data_dir
        self.config = load_config(config_path)
        self.return_type = return_type
        self.top_features = top_features
        self.analysis_dir = analysis_dir
        self.logger = Logger()
        self.analysis_start_time = datetime.now()
        self.execution_uuid = None  # UUID 초기화

        # 분석 폴더 구조 생성
        create_analysis_folder_structure(analysis_dir)

        # 재무분석기 초기화
        self.financial_analyzer = FinancialAnalyzer()

        # 분석 결과 저장
        self.analysis_results = {}
        self.prepared_data = {}

    def prepare_data(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """데이터 전처리 및 수익률 계산 (재무지표만 사용)"""
        self.logger.log_info("📊 재무지표 데이터 전처리 및 수익률 계산 중...")

        # 재무지표 관련 컬럼들
        financial_columns = {
            "pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "price_to_sales",
            "ev_to_ebitda", "ev_to_revenue", "price_to_cashflow", "price_to_free_cashflow",
            "return_on_equity", "return_on_assets", "return_on_capital", "return_on_invested_capital",
            "profit_margin", "operating_margin", "gross_margin", "ebitda_margin", "net_income_margin",
            "revenue_growth", "earnings_growth", "earnings_quarterly_growth", "revenue_quarterly_growth",
            "earnings_annual_growth", "revenue_annual_growth", "revenue_per_employee", "revenue_per_share",
            "debt_to_equity", "debt_to_assets", "current_ratio", "quick_ratio", "cash_ratio",
            "interest_coverage", "total_cash", "total_debt", "net_debt", "cash_per_share",
            "book_value", "tangible_book_value", "operating_cashflow", "free_cashflow",
            "free_cashflow_yield", "operating_cashflow_per_share", "free_cashflow_per_share",
            "cashflow_to_debt", "dividend_yield", "dividend_rate", "payout_ratio",
            "dividend_payout_ratio", "five_year_avg_dividend_yield", "forward_dividend_yield",
            "forward_dividend_rate", "earnings_ttm", "earnings_forward", "earnings_quarterly",
            "earnings_annual", "eps_ttm", "eps_forward", "eps_quarterly", "eps_annual",
            "total_revenue", "revenue_ttm", "revenue_forward", "revenue_quarterly", "revenue_annual",
            "gross_profits", "ebitda", "ebit", "net_income", "net_income_ttm",
            "shares_outstanding", "float_shares", "shares_short", "shares_short_prior_month",
            "short_ratio", "short_percent_of_float", "shares_percent_shares_out",
            "held_percent_insiders", "held_percent_institutions", "institutional_ownership",
            "beta", "fifty_two_week_change", "fifty_day_average", "two_hundred_day_average",
            "fifty_two_week_high", "fifty_two_week_low", "day_high", "day_low", "volume",
            "average_volume", "market_cap", "enterprise_value",
            # 분기별 데이터
            "quarterly_revenue", "quarterly_net_income", "quarterly_operating_income",
            "quarterly_ebitda", "quarterly_eps", "quarterly_gross_profit", "quarterly_ebit",
            "quarterly_operating_expense", "quarterly_research_development",
            "quarterly_selling_general_admin", "quarterly_total_assets", "quarterly_total_liabilities",
            "quarterly_total_equity", "quarterly_cash", "quarterly_debt", "quarterly_current_assets",
            "quarterly_current_liabilities", "quarterly_inventory", "quarterly_accounts_receivable",
            "quarterly_accounts_payable", "quarterly_short_term_debt", "quarterly_long_term_debt",
            "quarterly_goodwill", "quarterly_intangible_assets", "quarterly_property_plant_equipment",
            "quarterly_operating_cashflow", "quarterly_investing_cashflow", "quarterly_financing_cashflow",
            "quarterly_free_cashflow", "quarterly_capital_expenditure", "quarterly_dividends_paid",
            "quarterly_net_income_cashflow", "quarterly_depreciation", "quarterly_change_in_cash",
            "quarterly_change_in_receivables", "quarterly_change_in_inventory", "quarterly_change_in_payables",
            # 배당 및 기업 행동 데이터
            "latest_dividend_amount", "dividend_frequency", "latest_split_ratio", "split_frequency",
            "latest_capital_gain",
            # 계산된 재무비율들
            "calculated_roe", "calculated_roa", "calculated_debt_to_assets", "calculated_current_ratio",
            "calculated_operating_margin", "calculated_net_margin", "calculated_ebitda_margin",
            "calculated_asset_turnover", "calculated_inventory_turnover", "calculated_receivables_turnover",
            "calculated_cashflow_to_debt", "calculated_fcf_yield", "calculated_dividend_payout"
        }

        # 제외할 컬럼들 (기술적 지표 및 기본 데이터)
        excluded_columns = {
            "datetime", "date", "time", "timestamp", "open", "high", "low", "close", "volume",
        }

        prepared_data = {}

        for symbol, data in data_dict.items():
            # 데이터 복사
            df = data.copy()

            # 기존 returns 컬럼이 있는지 확인
            if "returns" in df.columns:
                # 기존 returns 컬럼을 target으로 사용
                df["return"] = df["returns"]
                self.logger.log_info(f"  {symbol}: 기존 returns 컬럼 사용")
            else:
                # 수익률 계산 (target 변수)
                if self.return_type == "log":
                    df["return"] = np.log(df["close"] / df["close"].shift(1))
                else:  # percentage
                    df["return"] = df["close"].pct_change() * 100
                self.logger.log_info(f"  {symbol}: 새로운 수익률 계산")

            # NaN 제거
            df = df.dropna()

            # 재무지표 컬럼들만 선택
            feature_columns = []
            for col in df.columns:
                if col not in excluded_columns and col != "return" and col != "returns":
                    # 재무지표 컬럼만 선택
                    if col in financial_columns or any(col.startswith(prefix) for prefix in [
                        "pe_", "market_", "return_on_", "debt_", "current_",
                        "profit_", "operating_", "ebitda_", "revenue_", "earnings_",
                        "dividend_", "payout_", "book_", "cash_", "total_", "quarterly_",
                        "calculated_", "latest_", "beta", "fifty_", "two_hundred_",
                        "shares_", "held_", "institutional_", "short_", "float_"
                    ]):
                        feature_columns.append(col)

            # 숫자형 데이터만 선택
            numeric_columns = []
            for col in feature_columns:
                try:
                    pd.to_numeric(df[col], errors="raise")
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    self.logger.log_info(f"    {symbol}: {col} 컬럼 제외 (숫자가 아님)")

            if len(numeric_columns) == 0:
                self.logger.log_info(f"  {symbol}: 재무지표 없음 → 재무분석 제외")
                continue

            # 수익률을 마지막 컬럼으로 이동
            columns_order = numeric_columns + ["return"]
            df = df[columns_order]

            # 모든 컬럼을 숫자형으로 변환
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # NaN 제거 (숫자 변환 후)
            df = df.dropna()

            prepared_data[symbol] = df
            self.logger.log_info(
                f"  {symbol}: {len(df)}개 데이터 포인트, {len(numeric_columns)}개 재무지표"
            )

        self.prepared_data = prepared_data
        return prepared_data

    def run_financial_analysis(self, symbol: str) -> Dict[str, Any]:
        """재무분석 실행"""
        self.logger.log_info(f"💰 {symbol} 재무분석 실행...")

        data = self.prepared_data[symbol]
        result = self.financial_analyzer.analyze_comprehensive(
            data, target_column="return", symbol=symbol
        )

        if symbol not in self.analysis_results:
            self.analysis_results[symbol] = {}
        self.analysis_results[symbol]["financial_analysis"] = result
        return result

    def run_full_analysis(self, symbols: List[str] = None) -> Dict[str, Any]:
        """전체 분석 실행"""
        # 로거 설정
        self.logger.setup_logger(
            strategy="fundamental_analysis", symbols=symbols or [], mode="analysis"
        )

        # 종합 요약 로거 설정
        self.logger.setup_summary_logger(
            symbols=symbols or [], timestamp=self.analysis_start_time
        )

        self.logger.log_section("💰 재무지표 기반 분석 시스템 시작")
        self.logger.log_info(f"📁 데이터 디렉토리: {self.data_dir}")
        self.logger.log_info(f"📊 수익률 타입: {self.return_type}")
        self.logger.log_info(f"🔝 상위 특성 수: {self.top_features}")

        # 데이터 로드
        self.logger.log_info("📂 데이터 로딩 중...")
        data_dict = load_and_preprocess_data(self.data_dir, symbols)
        self.logger.log_success(f"✅ 데이터 로딩 완료 ({len(data_dict)}개 종목)")

        # 데이터 전처리
        prepared_data = self.prepare_data(data_dict)
        if not prepared_data:
            self.logger.log_warning("재무지표가 있는 종목이 없어 재무분석을 건너뜁니다.")
            return {}

        # 각 종목별 분석 실행
        for symbol in prepared_data.keys():
            self.logger.log_info(f"🔄 {symbol} 분석 시작...")

            try:
                # 재무분석 실행
                financial_result = self.run_financial_analysis(symbol)
                self.logger.log_success(f"✅ {symbol} 분석 완료")

            except Exception as e:
                self.logger.log_error(f"❌ {symbol} 분석 중 오류: {str(e)}")

        # 종합 요약 생성
        self.generate_analysis_summary()

        return self.analysis_results

    def generate_analysis_summary(self):
        """분석 결과 종합 요약"""
        if not self.analysis_results:
            return

        self.logger.log_summary_section("💰 재무지표 기반 분석 종합 요약 리포트")

        # 분석 설정
        self.logger.log_summary_subsection("📋 분석 설정")
        self.logger.log_summary_info(f"분석 유형: 재무지표 기반")
        self.logger.log_summary_info(f"수익률 타입: {self.return_type}")
        self.logger.log_summary_info(f"상위 특성 수: {self.top_features}")
        self.logger.log_summary_info(f"분석 종목 수: {len(self.analysis_results)}")

        # 종목별 요약
        self.logger.log_summary_subsection("📈 종목별 분석 요약")

        for symbol, results in self.analysis_results.items():
            self.logger.log_summary_info(f"\n{symbol}:")

            if "financial_analysis" in results:
                financial_result = results["financial_analysis"]
                
                # 주요 재무지표 요약
                if "key_metrics" in financial_result:
                    metrics = financial_result["key_metrics"]
                    self.logger.log_summary_info(f"  P/E 비율: {metrics.get('pe_ratio', 'N/A')}")
                    self.logger.log_summary_info(f"  ROE: {metrics.get('roe', 'N/A')}")
                    self.logger.log_summary_info(f"  부채비율: {metrics.get('debt_to_equity', 'N/A')}")
                    self.logger.log_summary_info(f"  배당수익률: {metrics.get('dividend_yield', 'N/A')}")

                # 상관관계 분석 결과
                if "correlation_analysis" in financial_result:
                    corr_result = financial_result["correlation_analysis"]
                    if "top_features" in corr_result:
                        top_features = corr_result["top_features"][:3]
                        self.logger.log_summary_info(f"  상관관계 상위: {', '.join(top_features)}")

                # 예측 모델 결과
                if "prediction_models" in financial_result:
                    pred_result = financial_result["prediction_models"]
                    for model_name, model_result in pred_result.items():
                        if "r_squared" in model_result:
                            r2 = model_result["r_squared"]
                            self.logger.log_summary_info(f"  {model_name} R²: {r2:.4f}")

        # 종료 메시지
        self.logger.log_summary_section("🎉 분석 완료")
        self.logger.log_summary_success(
            f"총 {len(self.analysis_results)}개 종목 분석 완료"
        )
        self.logger.log_summary_info(f"종합 요약 로그: {self.logger.summary_log_file}")

    def save_results(self, output_path: str = None):
        """분석 결과 저장"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid_suffix = f"_{self.execution_uuid}" if self.execution_uuid else ""
            output_path = f"fundamental_analysis_results_{timestamp}{uuid_suffix}.json"

        # JSON 직렬화 가능한 형태로 변환
        def make_serializable(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif hasattr(obj, "__name__") and not isinstance(obj, str):
                return str(obj.__name__)
            elif hasattr(obj, "__class__") and not isinstance(obj, str):
                return str(obj.__class__.__name__)
            else:
                try:
                    import json

                    json.dumps(obj)
                    return obj
                except Exception:
                    return str(obj)

        # NaN/inf를 안전한 값으로 변환
        def clean_nan_inf(obj):
            import numpy as np
            import pandas as pd

            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: clean_nan_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_inf(v) for v in obj]
            elif isinstance(obj, pd.DataFrame):
                return clean_nan_inf(obj.to_dict())
            elif isinstance(obj, pd.Series):
                return clean_nan_inf(obj.to_dict())
            else:
                return obj

        serializable_results = {}
        for symbol, results in self.analysis_results.items():
            serializable_results[symbol] = {}
            for model_name, result in results.items():
                serializable_results[symbol][model_name] = clean_nan_inf(
                    make_serializable(result)
                )

        # analysis 폴더에 저장
        saved_path = save_analysis_results(
            serializable_results, 
            "fundamental_analysis", 
            output_path,
            self.analysis_dir
        )

        self.logger.log_success(f"✅ 분석 결과 저장: {saved_path}")
        return saved_path


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 분석 시스템")
    parser.add_argument("--data_dir", default="data", help="데이터 디렉토리 경로")
    parser.add_argument("--symbols", nargs="+", help="분석할 종목 목록")
    parser.add_argument(
        "--return_type",
        choices=["percentage", "log"],
        default="percentage",
        help="수익률 계산 방식",
    )
    parser.add_argument("--top_features", type=int, default=10, help="상위 특성 수")
    parser.add_argument("--output", help="결과 저장 경로")
    parser.add_argument("--uuid", help="실행 UUID")
    parser.add_argument(
        "--analysis_type",
        choices=["quant", "fundamental", "both"],
        default="both",
        help="분석 유형 (기술적/재무적/둘 다)",
    )

    args = parser.parse_args()

    # 분석기 초기화
    quant_analyst = QuantAnalyst(
        data_dir=args.data_dir,
        return_type=args.return_type,
        top_features=args.top_features,
    )
    
    fundamental_analyst = FundamentalAnalyst(
        data_dir=args.data_dir,
        return_type=args.return_type,
        top_features=args.top_features,
    )
    
    # UUID 설정
    if args.uuid:
        quant_analyst.execution_uuid = args.uuid
        fundamental_analyst.execution_uuid = args.uuid
        print(f"🆔 분석 UUID 설정: {args.uuid}")

    results = {}

    # 분석 실행
    if args.analysis_type in ["quant", "both"]:
        print("🎯 기술적 지표 기반 분석 시작...")
        quant_results = quant_analyst.run_full_analysis(symbols=args.symbols)
        quant_analyst.save_results()
        results["quant_analysis"] = quant_results

    if args.analysis_type in ["fundamental", "both"]:
        print("💰 재무지표 기반 분석 시작...")
        fundamental_results = fundamental_analyst.run_full_analysis(symbols=args.symbols)
        fundamental_analyst.save_results()
        results["fundamental_analysis"] = fundamental_results

    print(f"\n✅ 분석 완료!")
    if "quant_analysis" in results:
        print(f"   기술적 분석: {len(results['quant_analysis'])}개 종목")
    if "fundamental_analysis" in results:
        print(f"   재무적 분석: {len(results['fundamental_analysis'])}개 종목")


if __name__ == "__main__":
    main()
