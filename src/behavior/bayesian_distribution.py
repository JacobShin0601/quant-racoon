#!/usr/bin/env python3
"""
베이지안 분포 분석 모듈
수익률 분포 분석, 베이지안 선형회귀, 변동성 모델링
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import norm, t, skewnorm, jarque_bera, shapiro
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class BayesianDistributionAnalyzer:
    """베이지안 분포 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.scaler = StandardScaler()

    def analyze_return_distribution(
        self, data: pd.DataFrame, target_column: str = "return"
    ) -> Dict[str, Any]:
        """
        수익률 분포 분석

        Args:
            data: 분석할 데이터프레임
            target_column: 수익률 컬럼명

        Returns:
            분포 분석 결과 딕셔너리
        """
        returns = data[target_column].dropna()

        if len(returns) < 10:
            raise ValueError("분석에 충분한 데이터가 없습니다.")

        # 기본 통계량
        basic_stats = {
            "mean": returns.mean(),
            "std": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "min": returns.min(),
            "max": returns.max(),
            "q25": returns.quantile(0.25),
            "q50": returns.quantile(0.50),
            "q75": returns.quantile(0.75),
            "n_samples": len(returns),
        }

        # 정규성 검정
        normality_tests = {}

        # Shapiro-Wilk 검정
        shapiro_stat, shapiro_p = shapiro(returns)
        normality_tests["shapiro_wilk"] = {
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "is_normal": shapiro_p > 0.05,
        }

        # Jarque-Bera 검정
        jb_stat, jb_p = jarque_bera(returns)
        normality_tests["jarque_bera"] = {
            "statistic": jb_stat,
            "p_value": jb_p,
            "is_normal": jb_p > 0.05,
        }

        # 분포 피팅
        distribution_fits = {}

        try:
            # 정규분포 피팅
            norm_params = norm.fit(returns)
            norm_aic = self._calculate_aic(returns, norm, norm_params)
            distribution_fits["normal"] = {
                "params": norm_params,
                "aic": norm_aic,
                "log_likelihood": norm.logpdf(returns, *norm_params).sum(),
            }
        except Exception as e:
            print(f"정규분포 피팅 오류: {e}")
            distribution_fits["normal"] = {
                "params": (returns.mean(), returns.std()),
                "aic": float("inf"),
                "log_likelihood": 0,
            }

        try:
            # t-분포 피팅
            t_params = t.fit(returns)
            t_aic = self._calculate_aic(returns, t, t_params)
            distribution_fits["t"] = {
                "params": t_params,
                "aic": t_aic,
                "log_likelihood": t.logpdf(returns, *t_params).sum(),
            }
        except Exception as e:
            print(f"t-분포 피팅 오류: {e}")
            distribution_fits["t"] = {
                "params": (returns.mean(), returns.std(), 3),
                "aic": float("inf"),
                "log_likelihood": 0,
            }

        try:
            # 스큐드 정규분포 피팅
            skewnorm_params = skewnorm.fit(returns)
            skewnorm_aic = self._calculate_aic(returns, skewnorm, skewnorm_params)
            distribution_fits["skew_normal"] = {
                "params": skewnorm_params,
                "aic": skewnorm_aic,
                "log_likelihood": skewnorm.logpdf(returns, *skewnorm_params).sum(),
            }
        except Exception as e:
            print(f"스큐드 정규분포 피팅 오류: {e}")
            distribution_fits["skew_normal"] = {
                "params": (returns.mean(), returns.std(), 0),
                "aic": float("inf"),
                "log_likelihood": 0,
            }

        # 최적 분포 선택 (AIC 기준)
        best_dist = min(distribution_fits.items(), key=lambda x: x[1]["aic"])

        # VaR 계산 (95%, 99% 신뢰수준)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # 정규분포 가정하의 VaR
        norm_var_95 = norm.ppf(0.05, norm_params[0], norm_params[1])
        norm_var_99 = norm.ppf(0.01, norm_params[0], norm_params[1])

        result = {
            "target_column": target_column,
            "basic_statistics": basic_stats,
            "normality_tests": normality_tests,
            "distribution_fits": distribution_fits,
            "best_distribution": best_dist[0],
            "best_distribution_aic": best_dist[1]["aic"],
            "var_95": var_95,
            "var_99": var_99,
            "norm_var_95": norm_var_95,
            "norm_var_99": norm_var_99,
            "returns": returns.values,
        }

        self.results["distribution"] = result
        return result

    def _calculate_aic(self, data: pd.Series, dist, params) -> float:
        """AIC 계산"""
        try:
            # params가 튜플이 아닌 경우 튜플로 변환
            if not isinstance(params, tuple):
                params = (params,)

            log_likelihood = dist.logpdf(data, *params).sum()
            n_params = len(params)
            return 2 * n_params - 2 * log_likelihood
        except Exception as e:
            # 오류 발생 시 기본값 반환
            print(f"AIC 계산 오류: {e}")
            return float("inf")

    def analyze_bayesian_regression(
        self,
        data: pd.DataFrame,
        target_column: str = "return",
        feature_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        베이지안 선형회귀 분석

        Args:
            data: 분석할 데이터프레임
            target_column: 종속변수 컬럼명
            feature_columns: 독립변수 컬럼명 리스트
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드

        Returns:
            베이지안 회귀 분석 결과 딕셔너리
        """
        if feature_columns is None:
            # 시계열 관련 컬럼들과 중복 특성들 제외
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
            feature_columns = [
                col
                for col in data.columns
                if col != target_column and col not in excluded_columns
            ]

        # 데이터 준비
        analysis_data = data[feature_columns + [target_column]].dropna()

        if len(analysis_data) < 10:
            raise ValueError("분석에 충분한 데이터가 없습니다.")

        X = analysis_data[feature_columns]
        y = analysis_data[target_column]

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Bayesian Ridge Regression
        bayesian_ridge = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
        )
        bayesian_ridge.fit(X_train_scaled, y_train)

        # ARD Regression (Automatic Relevance Determination)
        ard = ARDRegression(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        ard.fit(X_train_scaled, y_train)

        # 예측
        y_pred_train_br = bayesian_ridge.predict(X_train_scaled)
        y_pred_test_br = bayesian_ridge.predict(X_test_scaled)
        y_pred_train_ard = ard.predict(X_train_scaled)
        y_pred_test_ard = ard.predict(X_test_scaled)

        # 성능 지표
        def calculate_metrics(y_true, y_pred, prefix):
            return {
                f"{prefix}_r2": r2_score(y_true, y_pred),
                f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
            }

        br_metrics = calculate_metrics(y_train, y_pred_train_br, "br_train")
        br_metrics.update(calculate_metrics(y_test, y_pred_test_br, "br_test"))
        ard_metrics = calculate_metrics(y_train, y_pred_train_ard, "ard_train")
        ard_metrics.update(calculate_metrics(y_test, y_pred_test_ard, "ard_test"))

        # 특성 중요도 (ARD의 alpha_ 값)
        if isinstance(ard.alpha_, float) or np.isscalar(ard.alpha_):
            alpha_arr = np.array([ard.alpha_])
        else:
            alpha_arr = np.array(ard.alpha_)
        feature_importance_ard = dict(zip(feature_columns, 1.0 / alpha_arr))
        sorted_importance = sorted(
            feature_importance_ard.items(), key=lambda x: x[1], reverse=True
        )

        # 베이지안 Ridge 계수 정보
        br_coefficients = dict(zip(feature_columns, bayesian_ridge.coef_))
        br_intercept = bayesian_ridge.intercept_

        # ARD 계수 정보
        ard_coefficients = dict(zip(feature_columns, ard.coef_))
        ard_intercept = ard.intercept_

        # 중요 특성 선택 (ARD alpha_ 기준)
        important_features_ard = [
            f
            for f, imp in feature_importance_ard.items()
            if imp > np.mean(list(feature_importance_ard.values()))
        ]

        result = {
            "model_type": "Bayesian Linear Regression",
            "target_column": target_column,
            "feature_columns": feature_columns,
            "n_samples": len(analysis_data),
            "n_features": len(feature_columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
            # Bayesian Ridge 결과
            "bayesian_ridge": {
                "coefficients": br_coefficients,
                "intercept": br_intercept,
                "alpha_": bayesian_ridge.alpha_,
                "lambda_": bayesian_ridge.lambda_,
                "metrics": br_metrics,
                "y_pred_train": y_pred_train_br,
                "y_pred_test": y_pred_test_br,
            },
            # ARD 결과
            "ard_regression": {
                "coefficients": ard_coefficients,
                "intercept": ard_intercept,
                "alpha_": ard.alpha_,
                "lambda_": ard.lambda_,
                "feature_importance": feature_importance_ard,
                "sorted_importance": sorted_importance,
                "important_features": important_features_ard,
                "metrics": ard_metrics,
                "y_pred_train": y_pred_train_ard,
                "y_pred_test": y_pred_test_ard,
            },
            # 모델 객체
            "bayesian_ridge_model": bayesian_ridge,
            "ard_model": ard,
            "scaler": self.scaler,
            # 실제값
            "y_train": y_train.values,
            "y_test": y_test.values,
        }

        self.results["bayesian_regression"] = result
        return result

    def analyze_volatility(
        self, data: pd.DataFrame, target_column: str = "return", window: int = 20
    ) -> Dict[str, Any]:
        """
        변동성 분석

        Args:
            data: 분석할 데이터프레임
            target_column: 수익률 컬럼명
            window: 이동평균 윈도우 크기

        Returns:
            변동성 분석 결과 딕셔너리
        """
        returns = data[target_column].dropna()

        if len(returns) < window:
            raise ValueError("분석에 충분한 데이터가 없습니다.")

        # 변동성 계산
        volatility = returns.rolling(window=window).std()
        volatility_annualized = volatility * np.sqrt(252)  # 연율화

        # 변동성 클러스터링 (변동성의 변동성)
        vol_of_vol = volatility.rolling(window=window).std()

        # 변동성 분포 분석
        vol_stats = {
            "mean": volatility.mean(),
            "std": volatility.std(),
            "skewness": volatility.skew(),
            "kurtosis": volatility.kurtosis(),
            "min": volatility.min(),
            "max": volatility.max(),
        }

        # 변동성과 수익률의 관계
        vol_return_corr = returns.corr(volatility)

        # 변동성 예측 모델 (간단한 AR 모델)
        vol_lag1 = volatility.shift(1)
        vol_lag2 = volatility.shift(2)

        # 변동성 예측 회귀
        vol_pred_data = pd.DataFrame(
            {"vol": volatility, "vol_lag1": vol_lag1, "vol_lag2": vol_lag2}
        ).dropna()

        if len(vol_pred_data) > 10:
            from sklearn.linear_model import LinearRegression

            vol_model = LinearRegression()
            vol_model.fit(vol_pred_data[["vol_lag1", "vol_lag2"]], vol_pred_data["vol"])
            vol_pred_r2 = vol_model.score(
                vol_pred_data[["vol_lag1", "vol_lag2"]], vol_pred_data["vol"]
            )
        else:
            vol_pred_r2 = None

        result = {
            "target_column": target_column,
            "window": window,
            "volatility": volatility.values,
            "volatility_annualized": volatility_annualized.values,
            "volatility_of_volatility": vol_of_vol.values,
            "volatility_statistics": vol_stats,
            "volatility_return_correlation": vol_return_corr,
            "volatility_prediction_r2": vol_pred_r2,
            "returns": returns.values,
        }

        self.results["volatility"] = result
        return result

    def plot_distribution_analysis(
        self, figsize: Tuple[int, int] = (15, 10), save_path: str = None
    ) -> plt.Figure:
        """분포 분석 플롯"""
        if "distribution" not in self.results:
            raise ValueError("먼저 분포 분석을 실행해주세요.")

        result = self.results["distribution"]
        returns = result["returns"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. 히스토그램과 분포 피팅
        ax1.hist(
            returns,
            bins=30,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        # 최적 분포 플롯
        best_dist_name = result["best_distribution"]
        best_params = result["distribution_fits"][best_dist_name]["params"]

        x = np.linspace(returns.min(), returns.max(), 100)
        if best_dist_name == "normal":
            ax1.plot(
                x,
                norm.pdf(x, *best_params),
                "r-",
                lw=2,
                label=f"Best: {best_dist_name}",
            )
        elif best_dist_name == "t":
            ax1.plot(
                x, t.pdf(x, *best_params), "r-", lw=2, label=f"Best: {best_dist_name}"
            )
        elif best_dist_name == "skew_normal":
            ax1.plot(
                x,
                skewnorm.pdf(x, *best_params),
                "r-",
                lw=2,
                label=f"Best: {best_dist_name}",
            )

        ax1.set_xlabel("Returns")
        ax1.set_ylabel("Density")
        ax1.set_title("Return Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Q-Q 플롯
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal)")
        ax2.grid(True, alpha=0.3)

        # 3. 분포 비교 (AIC 기준)
        dist_names = list(result["distribution_fits"].keys())
        aic_values = [result["distribution_fits"][d]["aic"] for d in dist_names]

        bars = ax3.bar(
            dist_names, aic_values, color=["blue", "green", "red"], alpha=0.7
        )
        ax3.set_ylabel("AIC")
        ax3.set_title("Distribution Comparison (AIC)")
        ax3.grid(axis="y", alpha=0.3)

        # 값 표시
        for bar, aic in zip(bars, aic_values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{aic:.1f}",
                ha="center",
                va="bottom",
            )

        # 4. VaR 비교
        var_methods = ["Empirical", "Normal"]
        var_95_values = [result["var_95"], result["norm_var_95"]]
        var_99_values = [result["var_99"], result["norm_var_99"]]

        x_pos = np.arange(len(var_methods))
        width = 0.35

        bars1 = ax4.bar(
            x_pos - width / 2, var_95_values, width, label="VaR 95%", alpha=0.7
        )
        bars2 = ax4.bar(
            x_pos + width / 2, var_99_values, width, label="VaR 99%", alpha=0.7
        )

        ax4.set_xlabel("Method")
        ax4.set_ylabel("VaR")
        ax4.set_title("Value at Risk Comparison")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(var_methods)
        ax4.legend()
        ax4.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"분포 분석 플롯 저장: {save_path}")

        return fig

    def plot_bayesian_regression_results(
        self, figsize: Tuple[int, int] = (15, 10), save_path: str = None
    ) -> plt.Figure:
        """베이지안 회귀 결과 플롯"""
        if "bayesian_regression" not in self.results:
            raise ValueError("먼저 베이지안 회귀 분석을 실행해주세요.")

        result = self.results["bayesian_regression"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. ARD 특성 중요도
        ard_importance = result["ard_regression"]["feature_importance"]
        top_features = sorted(ard_importance.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        features = [f[0] for f in top_features]
        importance = [f[1] for f in top_features]

        bars = ax1.barh(features, importance, color="green", alpha=0.7)
        ax1.set_xlabel("Importance (1/alpha_)")
        ax1.set_title("ARD Feature Importance (Top 10)")
        ax1.grid(axis="x", alpha=0.3)

        # 2. Bayesian Ridge vs ARD 계수 비교
        br_coeffs = result["bayesian_ridge"]["coefficients"]
        ard_coeffs = result["ard_regression"]["coefficients"]

        common_features = list(br_coeffs.keys())[:10]  # 상위 10개만
        br_values = [br_coeffs[f] for f in common_features]
        ard_values = [ard_coeffs[f] for f in common_features]

        x_pos = np.arange(len(common_features))
        width = 0.35

        bars1 = ax2.bar(
            x_pos - width / 2, br_values, width, label="Bayesian Ridge", alpha=0.7
        )
        bars2 = ax2.bar(x_pos + width / 2, ard_values, width, label="ARD", alpha=0.7)

        ax2.set_xlabel("Features")
        ax2.set_ylabel("Coefficients")
        ax2.set_title("Coefficient Comparison")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(common_features, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # 3. 실제값 vs 예측값 (Bayesian Ridge)
        y_test = result["y_test"]
        y_pred_br = result["bayesian_ridge"]["y_pred_test"]

        ax3.scatter(y_test, y_pred_br, alpha=0.6, s=20)
        ax3.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", alpha=0.8
        )
        ax3.set_xlabel("Actual")
        ax3.set_ylabel("Predicted")
        ax3.set_title(
            f'Bayesian Ridge: R² = {result["bayesian_ridge"]["metrics"]["br_test_r2"]:.4f}'
        )
        ax3.grid(True, alpha=0.3)

        # 4. 실제값 vs 예측값 (ARD)
        y_pred_ard = result["ard_regression"]["y_pred_test"]

        ax4.scatter(y_test, y_pred_ard, alpha=0.6, s=20)
        ax4.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", alpha=0.8
        )
        ax4.set_xlabel("Actual")
        ax4.set_ylabel("Predicted")
        ax4.set_title(
            f'ARD: R² = {result["ard_regression"]["metrics"]["ard_test_r2"]:.4f}'
        )
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"베이지안 회귀 결과 플롯 저장: {save_path}")

        return fig

    def get_summary(self) -> str:
        """분석 결과 요약 문자열 반환"""
        if not self.results:
            return "분석 결과가 없습니다."

        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("베이지안 분포 분석 결과 요약")
        summary_lines.append("=" * 60)

        # 분포 분석 결과
        if "distribution" in self.results:
            dist_result = self.results["distribution"]
            summary_lines.append("\n📊 수익률 분포 분석:")
            summary_lines.append("-" * 40)
            summary_lines.append(
                f"샘플 수: {dist_result['basic_statistics']['n_samples']}"
            )
            summary_lines.append(f"평균: {dist_result['basic_statistics']['mean']:.4f}")
            summary_lines.append(
                f"표준편차: {dist_result['basic_statistics']['std']:.4f}"
            )
            summary_lines.append(
                f"왜도: {dist_result['basic_statistics']['skewness']:.4f}"
            )
            summary_lines.append(
                f"첨도: {dist_result['basic_statistics']['kurtosis']:.4f}"
            )

            # 정규성 검정
            sw_test = dist_result["normality_tests"]["shapiro_wilk"]
            summary_lines.append(
                f"Shapiro-Wilk: p={sw_test['p_value']:.3e} ({'정규' if sw_test['is_normal'] else '비정규'})"
            )

            # 최적 분포
            summary_lines.append(
                f"최적 분포: {dist_result['best_distribution']} (AIC: {dist_result['best_distribution_aic']:.2f})"
            )

            # VaR
            summary_lines.append(f"VaR 95%: {dist_result['var_95']:.4f}")
            summary_lines.append(f"VaR 99%: {dist_result['var_99']:.4f}")

        # 베이지안 회귀 결과
        if "bayesian_regression" in self.results:
            br_result = self.results["bayesian_regression"]
            summary_lines.append("\n🤖 베이지안 회귀 분석:")
            summary_lines.append("-" * 40)
            summary_lines.append(f"특성 수: {br_result['n_features']}")
            summary_lines.append(f"샘플 수: {br_result['n_samples']}")

            # 성능 비교
            br_r2 = br_result["bayesian_ridge"]["metrics"]["br_test_r2"]
            ard_r2 = br_result["ard_regression"]["metrics"]["ard_test_r2"]
            summary_lines.append(f"Bayesian Ridge R²: {br_r2:.4f}")
            summary_lines.append(f"ARD R²: {ard_r2:.4f}")

            # 중요 특성
            important_features = br_result["ard_regression"]["important_features"][:5]
            summary_lines.append(
                f"중요 특성 (상위 5개): {', '.join(important_features)}"
            )

        # 변동성 분석 결과
        if "volatility" in self.results:
            vol_result = self.results["volatility"]
            summary_lines.append("\n📈 변동성 분석:")
            summary_lines.append("-" * 40)
            summary_lines.append(
                f"평균 변동성: {vol_result['volatility_statistics']['mean']:.4f}"
            )
            summary_lines.append(
                f"변동성-수익률 상관관계: {vol_result['volatility_return_correlation']:.4f}"
            )
            if vol_result["volatility_prediction_r2"]:
                summary_lines.append(
                    f"변동성 예측 R²: {vol_result['volatility_prediction_r2']:.4f}"
                )

        return "\n".join(summary_lines)

    def print_summary(self):
        """분석 결과 요약 출력"""
        print(self.get_summary())
