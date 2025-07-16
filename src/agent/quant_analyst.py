#!/usr/bin/env python3
"""
정량 분석 시스템
종가 기준 수익률을 종속변수로 하여 다양한 요인들의 영향을 분석
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

from behavior.correlation import CorrelationAnalyzer
from behavior.linear_regression import LinearRegressionAnalyzer
from behavior.lasso_regression import LassoRegressionAnalyzer
from behavior.random_forest import RandomForestAnalyzer
from behavior.multi_layer_perceptron import MLPAnalyzer
from behavior.bayesian_distribution import BayesianDistributionAnalyzer
from agent.helper import (
    Logger,
    load_config,
    load_and_preprocess_data,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
)


class QuantAnalyst:
    """정량 분석 시스템 메인 클래스"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        config_path: str = DEFAULT_CONFIG_PATH,
        return_type: str = "percentage",  # "percentage" or "log"
        top_features: int = 10,
    ):
        self.data_dir = data_dir
        self.config = load_config(config_path)
        self.return_type = return_type
        self.top_features = top_features
        self.logger = Logger()
        self.analysis_start_time = datetime.now()

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
        """데이터 전처리 및 수익률 계산"""
        self.logger.log_info("📊 데이터 전처리 및 수익률 계산 중...")

        # 시계열 관련 컬럼들과 중복 특성들 제외 리스트
        excluded_columns = {
            "datetime",
            "date",
            "time",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
            "dividend_amount",
            "split_coefficient",
            "returns",  # 이전 수익률 - target과 중복되어 제외
        }

        prepared_data = {}

        for symbol, data in data_dict.items():
            # 데이터 복사
            df = data.copy()

            # 수익률 계산 (target 변수)
            if self.return_type == "log":
                df["return"] = np.log(df["close"] / df["close"].shift(1))
            else:  # percentage
                df["return"] = df["close"].pct_change() * 100

            # NaN 제거
            df = df.dropna()

            # 제외 컬럼들을 제외한 특성 컬럼들 선택
            feature_columns = [
                col
                for col in df.columns
                if col not in excluded_columns and col != "return"
            ]

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
                f"  {symbol}: {len(df)}개 데이터 포인트, {len(numeric_columns)}개 특성"
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

        self.logger.log_section("🎯 정량 분석 시스템 시작")
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

        self.logger.log_summary_section("📊 정량 분석 종합 요약 리포트")

        # 분석 설정
        self.logger.log_summary_subsection("📋 분석 설정")
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
            output_path = f"log/quant_analysis_results_{timestamp}.json"

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

        with open(output_path, "w", encoding="utf-8") as f:
            import json

            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        self.logger.log_success(f"✅ 분석 결과 저장: {output_path}")
        return output_path


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="정량 분석 시스템")
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

    args = parser.parse_args()

    # 분석기 초기화
    analyst = QuantAnalyst(
        data_dir=args.data_dir,
        return_type=args.return_type,
        top_features=args.top_features,
    )

    # 분석 실행
    results = analyst.run_full_analysis(symbols=args.symbols)

    # 결과 저장
    analyst.save_results(args.output)

    print(f"\n✅ 정량 분석 완료! {len(results)}개 종목 분석됨")


if __name__ == "__main__":
    main()
