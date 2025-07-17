import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
import warnings
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class MLVisualizer:
    """머신러닝 분석 결과 시각화 클래스"""

    def __init__(self, analysis_results: Dict[str, Any]):
        self.analysis_results = analysis_results
        self.symbols = list(analysis_results.keys())
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    def safe_numeric_convert(self, value, default=0.0):
        """
        안전하게 숫자로 변환하는 함수
        'int', 'float' 같은 타입 지시자나 문자열을 적절히 처리
        """
        if value is None:
            return default

        # 이미 숫자인 경우
        if isinstance(value, (int, float)):
            return float(value)

        # 문자열인 경우
        if isinstance(value, str):
            # 타입 지시자인 경우 기본값 반환
            if value.lower() in ["int", "float", "n/a", "na", "none", ""]:
                return default

            # 숫자 문자열 변환 시도
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        # 기타 경우 기본값 반환
        return default

    def plot_correlation_heatmap(
        self, symbol: str, figsize: tuple = (12, 10)
    ) -> plt.Figure:
        """상관관계 히트맵 시각화"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Symbol {symbol} not found in analysis results")

        corr_data = self.analysis_results[symbol]["correlation"]["correlation_matrix"]
        corr_df = pd.DataFrame(corr_data)

        fig, ax = plt.subplots(figsize=figsize)

        # 히트맵 생성
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(
            corr_df,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title(
            f"{symbol} - Feature Correlation Heatmap", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)

        plt.tight_layout()
        return fig

    def plot_correlation_bar(self, symbol: str, figsize: tuple = (12, 6)) -> plt.Figure:
        """상관관계 막대 그래프"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Symbol {symbol} not found in analysis results")

        corr_data = self.analysis_results[symbol]["correlation"]["correlations"]
        corr_df = pd.DataFrame(
            list(corr_data.items()), columns=["Feature", "Correlation"]
        )
        corr_df["Abs_Correlation"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("Abs_Correlation", ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 절댓값 기준 정렬된 상관관계
        colors = ["red" if x < 0 else "blue" for x in corr_df["Correlation"]]
        bars1 = ax1.barh(
            corr_df["Feature"], corr_df["Correlation"], color=colors, alpha=0.7
        )
        ax1.set_title(
            f"{symbol} - Feature Correlations with Return",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Correlation Coefficient", fontsize=12)
        ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax1.grid(axis="x", alpha=0.3)

        # 절댓값 기준 상관관계
        bars2 = ax2.barh(
            corr_df["Feature"], corr_df["Abs_Correlation"], color="green", alpha=0.7
        )
        ax2.set_title(
            f"{symbol} - Absolute Correlation Strength", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Absolute Correlation", fontsize=12)
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_linear_regression_results(
        self, symbol: str, figsize: tuple = (15, 10)
    ) -> plt.Figure:
        """선형회귀 결과 시각화"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Symbol {symbol} not found in analysis results")

        lr_data = self.analysis_results[symbol]["linear_regression"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 계수 시각화
        coef_data = {k: v for k, v in lr_data["coefficients"].items() if k != "const"}
        features = list(coef_data.keys())
        coefficients = list(coef_data.values())

        # 안전한 계수 비교 및 변환
        numeric_coefficients = [self.safe_numeric_convert(x, 0.0) for x in coefficients]
        colors = ["red" if x < 0 else "blue" for x in numeric_coefficients]
        bars = ax1.barh(features, numeric_coefficients, color=colors, alpha=0.7)
        ax1.set_title("Linear Regression Coefficients", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Coefficient Value", fontsize=12)
        ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax1.grid(axis="x", alpha=0.3)

        # p-value 시각화
        p_values = [lr_data["p_values"][f] for f in features]

        # p-value를 숫자로 변환하고 색상 결정
        numeric_p_values = [self.safe_numeric_convert(p, 1.0) for p in p_values]
        colors_p = ["red" if p < 0.05 else "gray" for p in numeric_p_values]
        bars_p = ax2.barh(features, numeric_p_values, color=colors_p, alpha=0.7)
        ax2.set_title(
            "P-values (Red: Significant < 0.05)", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("P-value", fontsize=12)
        ax2.axvline(x=0.05, color="red", linestyle="--", alpha=0.7)
        ax2.set_xscale("log")
        ax2.grid(axis="x", alpha=0.3)

        # VIF 점수 시각화
        if "vif_scores" in lr_data and isinstance(lr_data["vif_scores"], dict):
            vif_data = lr_data["vif_scores"]
            vif_features = list(vif_data.keys())
            vif_scores = list(vif_data.values())

            # VIF 점수를 숫자로 변환하고 색상 결정
            numeric_vif_scores = [self.safe_numeric_convert(v, 0.0) for v in vif_scores]

            # 모든 VIF 값이 0인 경우 (즉, 모두 "float" 문자열이었던 경우)
            if all(v == 0.0 for v in numeric_vif_scores):
                # high_vif_features 정보를 활용하여 시각화
                high_vif_features = lr_data.get("high_vif_features", [])

                if high_vif_features:
                    # 높은 VIF 특성들에 대해 시각적 표시
                    demo_vif_scores = []
                    colors_vif = []

                    for feature in vif_features:
                        if feature in high_vif_features:
                            demo_vif_scores.append(12)  # 높은 VIF로 가정
                            colors_vif.append("red")
                        else:
                            demo_vif_scores.append(3)  # 낮은 VIF로 가정
                            colors_vif.append("green")

                    bars_vif = ax3.barh(
                        vif_features, demo_vif_scores, color=colors_vif, alpha=0.7
                    )
                    ax3.set_title(
                        "VIF Analysis (Red: High VIF Features)",
                        fontsize=14,
                        fontweight="bold",
                    )
                    ax3.set_xlabel("VIF Indication (Red>10, Green<5)", fontsize=12)
                    ax3.axvline(x=5, color="orange", linestyle="--", alpha=0.7)
                    ax3.axvline(x=10, color="red", linestyle="--", alpha=0.7)
                    ax3.grid(axis="x", alpha=0.3)

                    # 범례 추가
                    ax3.text(
                        0.02,
                        0.98,
                        f"High VIF features: {', '.join(high_vif_features)}",
                        transform=ax3.transAxes,
                        va="top",
                        ha="left",
                        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                        fontsize=10,
                    )
                else:
                    ax3.text(
                        0.5,
                        0.5,
                        "VIF scores not available\n(All values are type indicators)",
                        ha="center",
                        va="center",
                        transform=ax3.transAxes,
                        fontsize=12,
                    )
                    ax3.set_title("VIF Scores", fontsize=14, fontweight="bold")
            else:
                # 정상적인 VIF 값들이 있는 경우
                colors_vif = [
                    "red" if v > 10 else "orange" if v > 5 else "green"
                    for v in numeric_vif_scores
                ]

                bars_vif = ax3.barh(
                    vif_features, numeric_vif_scores, color=colors_vif, alpha=0.7
                )
                ax3.set_title(
                    "VIF Scores (Green<5, Orange<10, Red>10)",
                    fontsize=14,
                    fontweight="bold",
                )
                ax3.set_xlabel("VIF Score", fontsize=12)
                ax3.axvline(x=5, color="orange", linestyle="--", alpha=0.7)
                ax3.axvline(x=10, color="red", linestyle="--", alpha=0.7)
                ax3.grid(axis="x", alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "VIF data not available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=14,
            )
            ax3.set_title("VIF Scores", fontsize=14, fontweight="bold")

        # 성능 지표
        metrics = {
            "R²": lr_data.get("r_squared", 0),
            "Train R²": lr_data.get("train_r_squared", 0),
            "Test R²": lr_data.get("test_r_squared", 0),
            "Train RMSE": lr_data.get("train_rmse", 0),
            "Test RMSE": lr_data.get("test_rmse", 0),
        }

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # 안전한 숫자 변환
        numeric_values = [self.safe_numeric_convert(val, 0.0) for val in metric_values]

        bars_metric = ax4.bar(metric_names, numeric_values, color="skyblue", alpha=0.7)
        ax4.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Value", fontsize=12)
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(axis="y", alpha=0.3)

        # 값 표시
        for bar, val in zip(bars_metric, numeric_values):
            if val != 0:  # 0이 아닌 경우에만 표시
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        return fig

    def plot_random_forest_results(
        self, symbol: str, figsize: tuple = (15, 8)
    ) -> plt.Figure:
        """랜덤포레스트 결과 시각화"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Symbol {symbol} not found in analysis results")

        rf_data = self.analysis_results[symbol]["random_forest"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 특성 중요도 - 상위 10개만 표시
        importance_data = rf_data["feature_importance"]
        features = list(importance_data.keys())
        importances = list(importance_data.values())

        # 중요도 순으로 정렬하고 상위 10개만 선택
        sorted_indices = np.argsort(importances)[::-1][:10]  # 상위 10개
        top_features = [features[i] for i in sorted_indices]
        top_importances = [importances[i] for i in sorted_indices]

        # 세로 막대 그래프로 변경하고 색상 개선
        bars = ax1.bar(
            range(len(top_features)),
            top_importances,
            color="forestgreen",
            alpha=0.7,
            edgecolor="darkgreen",
            linewidth=1,
        )
        ax1.set_title(
            f"Random Forest Feature Importance (Top 10) - {symbol}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Importance Score", fontsize=12)
        ax1.set_xlabel("Features", fontsize=12)
        ax1.set_xticks(range(len(top_features)))
        ax1.set_xticklabels(top_features, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)

        # 막대 위에 값 표시
        for bar, importance in zip(bars, top_importances):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{importance:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # RMSE 성능 지표만 표시
        rmse_metrics = {
            "Train RMSE": rf_data.get("train_rmse", "N/A"),
            "Test RMSE": rf_data.get("test_rmse", "N/A"),
        }

        metric_names = list(rmse_metrics.keys())
        metric_values = list(rmse_metrics.values())

        # 안전한 숫자 변환
        numeric_values = [self.safe_numeric_convert(val, 0.0) for val in metric_values]

        bars_metric = ax2.bar(
            metric_names,
            numeric_values,
            color=["lightcoral", "salmon"],
            alpha=0.7,
            edgecolor="darkred",
            linewidth=1,
        )
        ax2.set_title(
            f"Random Forest RMSE Metrics - {symbol}", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("RMSE Value", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)

        # 값 표시
        for bar, val in zip(bars_metric, numeric_values):
            if val > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.01,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

        plt.tight_layout()
        return fig

    def plot_bayesian_results(
        self, symbol: str, figsize: tuple = (16, 12)
    ) -> plt.Figure:
        """베이지안 회귀 결과 시각화 (개선된 버전)"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Symbol {symbol} not found in analysis results")

        bayes_data = self.analysis_results[symbol]["bayesian_regression"]

        # 2x3 레이아웃으로 변경하여 더 많은 정보 표시
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize)

        # 1. Bayesian Ridge 계수 (상위 15개)
        br_coef = bayes_data["bayesian_ridge"]["coefficients"]
        br_features = [k for k in br_coef.keys() if k != "const"]
        br_coefficients = [br_coef[f] for f in br_features]

        # 절댓값 기준으로 정렬하여 상위 15개만 표시
        sorted_indices = np.argsort([abs(x) for x in br_coefficients])[::-1][:15]
        top_br_features = [br_features[i] for i in sorted_indices]
        top_br_coefficients = [br_coefficients[i] for i in sorted_indices]

        colors_br = ["red" if x < 0 else "blue" for x in top_br_coefficients]
        bars_br = ax1.barh(
            top_br_features, top_br_coefficients, color=colors_br, alpha=0.7
        )
        ax1.set_title(
            "Bayesian Ridge Coefficients (Top 15)", fontsize=12, fontweight="bold"
        )
        ax1.set_xlabel("Coefficient Value", fontsize=10)
        ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax1.grid(axis="x", alpha=0.3)

        # 2. ARD 계수 (Bayesian Ridge와 비교)
        ard_coef = bayes_data["ard_regression"]["coefficients"]
        ard_features = [k for k in ard_coef.keys() if k != "const"]
        ard_coefficients = [ard_coef[f] for f in ard_features]

        # 같은 특성들에 대해 ARD 계수 표시
        ard_coef_for_top = [ard_coef.get(f, 0) for f in top_br_features]
        colors_ard = ["red" if x < 0 else "blue" for x in ard_coef_for_top]
        bars_ard = ax2.barh(
            top_br_features, ard_coef_for_top, color=colors_ard, alpha=0.7
        )
        ax2.set_title(
            "ARD Coefficients (Same Features)", fontsize=12, fontweight="bold"
        )
        ax2.set_xlabel("Coefficient Value", fontsize=10)
        ax2.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax2.grid(axis="x", alpha=0.3)

        # 3. 계수 비교 (상위 10개 특성)
        top_10_features = top_br_features[:10]
        br_top_10 = top_br_coefficients[:10]
        ard_top_10 = [ard_coef.get(f, 0) for f in top_10_features]

        x = np.arange(len(top_10_features))
        width = 0.35

        bars1 = ax3.bar(
            x - width / 2,
            br_top_10,
            width,
            label="Bayesian Ridge",
            alpha=0.7,
            color="skyblue",
        )
        bars2 = ax3.bar(
            x + width / 2, ard_top_10, width, label="ARD", alpha=0.7, color="lightcoral"
        )

        ax3.set_title("Coefficient Comparison (Top 10)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Coefficient Value", fontsize=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(top_10_features, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)

        # 4. 성능 비교 (RMSE 사용 - 더 신뢰할 수 있는 지표)
        models = ["Bayesian Ridge", "ARD"]

        # RMSE 값들 추출 (이 값들은 실제 숫자로 저장됨)
        train_rmse = [
            self.safe_numeric_convert(
                bayes_data["bayesian_ridge"]["metrics"].get("br_train_rmse", 0)
            ),
            self.safe_numeric_convert(
                bayes_data["ard_regression"]["metrics"].get("ard_train_rmse", 0)
            ),
        ]

        test_rmse = [
            self.safe_numeric_convert(
                bayes_data["bayesian_ridge"]["metrics"].get("br_test_rmse", 0)
            ),
            self.safe_numeric_convert(
                bayes_data["ard_regression"]["metrics"].get("ard_test_rmse", 0)
            ),
        ]

        x_pos = np.arange(len(models))
        width = 0.35

        bars_train = ax4.bar(
            x_pos - width / 2,
            train_rmse,
            width,
            label="Train RMSE",
            alpha=0.7,
            color="lightcoral",
        )
        bars_test = ax4.bar(
            x_pos + width / 2,
            test_rmse,
            width,
            label="Test RMSE",
            alpha=0.7,
            color="lightblue",
        )

        ax4.set_title("Model Performance (RMSE)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("RMSE", fontsize=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(axis="y", alpha=0.3)

        # 값 표시 (RMSE가 낮을수록 좋음)
        for i, (train_val, test_val) in enumerate(zip(train_rmse, test_rmse)):
            if train_val != 0:
                ax4.text(
                    i - width / 2,
                    train_val + max(train_rmse) * 0.02,
                    f"{train_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            if test_val != 0:
                ax4.text(
                    i + width / 2,
                    test_val + max(test_rmse) * 0.02,
                    f"{test_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # 더 나은 모델 표시
        if train_rmse[0] != 0 and train_rmse[1] != 0:
            better_model = "Bayesian Ridge" if train_rmse[0] < train_rmse[1] else "ARD"
            ax4.text(
                0.5,
                0.95,
                f"Better Model: {better_model} (Lower RMSE)",
                transform=ax4.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                fontsize=9,
            )

        # 5. 특성 선택 효과 (0이 아닌 계수 개수)
        br_nonzero = sum(1 for coef in br_coefficients if abs(coef) > 1e-6)
        ard_nonzero = sum(1 for coef in ard_coefficients if abs(coef) > 1e-6)

        feature_counts = [br_nonzero, ard_nonzero]
        bars_count = ax5.bar(
            models, feature_counts, color=["skyblue", "lightcoral"], alpha=0.7
        )
        ax5.set_title("Feature Selection Effect", fontsize=12, fontweight="bold")
        ax5.set_ylabel("Non-zero Coefficients", fontsize=10)
        ax5.grid(axis="y", alpha=0.3)

        # 값 표시
        for bar, count in zip(bars_count, feature_counts):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 6. 리스크 메트릭 (개선된 버전)
        if "bayesian_distribution" in self.analysis_results[symbol]:
            dist_data = self.analysis_results[symbol]["bayesian_distribution"]

            # 여러 VaR 메트릭 수집
            risk_metrics = {
                "VaR 95%": self.safe_numeric_convert(dist_data.get("var_95", 0)),
                "VaR 99%": self.safe_numeric_convert(dist_data.get("var_99", 0)),
                "Normal VaR 95%": self.safe_numeric_convert(
                    dist_data.get("norm_var_95", 0)
                ),
                "Normal VaR 99%": self.safe_numeric_convert(
                    dist_data.get("norm_var_99", 0)
                ),
            }

            # 기본 통계도 추가
            if "basic_statistics" in dist_data:
                basic_stats = dist_data["basic_statistics"]
                risk_metrics.update(
                    {
                        "Mean Return": self.safe_numeric_convert(
                            basic_stats.get("mean", 0)
                        ),
                        "Volatility": self.safe_numeric_convert(
                            basic_stats.get("std", 0)
                        ),
                    }
                )

            # 0이 아닌 값들만 표시
            filtered_metrics = {k: v for k, v in risk_metrics.items() if v != 0}

            if len(filtered_metrics) >= 2:  # 최소 2개 이상의 메트릭이 있어야 의미있음
                risk_names = list(filtered_metrics.keys())
                risk_values = list(filtered_metrics.values())

                # 색상 매핑 (음수는 빨간색, 양수는 파란색 계열)
                colors_risk = []
                for val in risk_values:
                    if val < 0:
                        colors_risk.append(
                            "red"
                            if "VaR" in risk_names[risk_values.index(val)]
                            else "darkred"
                        )
                    else:
                        colors_risk.append(
                            "blue"
                            if "Mean" in risk_names[risk_values.index(val)]
                            else "green"
                        )

                bars_risk = ax6.bar(
                    risk_names, risk_values, color=colors_risk, alpha=0.7
                )
                ax6.set_title("Risk & Return Metrics", fontsize=12, fontweight="bold")
                ax6.set_ylabel("Value", fontsize=10)
                ax6.tick_params(axis="x", rotation=45)
                ax6.grid(axis="y", alpha=0.3)

                # 0 라인 추가
                ax6.axhline(y=0, color="black", linestyle="-", alpha=0.3)

                # 값 표시
                for bar, val, name in zip(bars_risk, risk_values, risk_names):
                    # 값이 음수면 아래쪽에, 양수면 위쪽에 표시
                    if val < 0:
                        y_pos = val - abs(val) * 0.1
                        va = "top"
                    else:
                        y_pos = val + abs(val) * 0.1
                        va = "bottom"

                    ax6.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_pos,
                        f"{val:.3f}",
                        ha="center",
                        va=va,
                        fontsize=8,
                        fontweight="bold",
                    )

                # 해석 텍스트 추가
                var_95 = filtered_metrics.get("VaR 95%", 0)
                if var_95 < 0:
                    ax6.text(
                        0.02,
                        0.98,
                        f"95% confidence: Max loss ≤ {abs(var_95):.2f}%",
                        transform=ax6.transAxes,
                        va="top",
                        ha="left",
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
                        fontsize=9,
                    )
            else:
                # 대신 기본 통계 요약 표시
                if "basic_statistics" in dist_data:
                    stats = dist_data["basic_statistics"]
                    summary_text = f"""
분포 분석 요약:

• 평균 수익률: {self.safe_numeric_convert(stats.get('mean', 0)):.4f}
• 표준편차: {self.safe_numeric_convert(stats.get('std', 0)):.4f}
• 왜도: {self.safe_numeric_convert(stats.get('skewness', 0)):.2f}
• 첨도: {self.safe_numeric_convert(stats.get('kurtosis', 0)):.2f}

• 최솟값: {self.safe_numeric_convert(stats.get('min', 0)):.2f}
• 최댓값: {self.safe_numeric_convert(stats.get('max', 0)):.2f}
                    """

                    ax6.text(
                        0.05,
                        0.95,
                        summary_text.strip(),
                        ha="left",
                        va="top",
                        transform=ax6.transAxes,
                        fontsize=9,
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8
                        ),
                    )
                    ax6.set_title(
                        "Distribution Summary", fontsize=12, fontweight="bold"
                    )
                    ax6.axis("off")
                else:
                    ax6.text(
                        0.5,
                        0.5,
                        "Risk metrics not available\n(Distribution data missing)",
                        ha="center",
                        va="center",
                        transform=ax6.transAxes,
                        fontsize=11,
                    )
                    ax6.set_title("Risk Metrics", fontsize=12, fontweight="bold")
        else:
            # 대신 모델 비교 요약 정보 표시 (RMSE 기반)
            # RMSE 값들 다시 계산
            br_train_rmse = self.safe_numeric_convert(
                bayes_data["bayesian_ridge"]["metrics"].get("br_train_rmse", 0)
            )
            br_test_rmse = self.safe_numeric_convert(
                bayes_data["bayesian_ridge"]["metrics"].get("br_test_rmse", 0)
            )
            ard_train_rmse = self.safe_numeric_convert(
                bayes_data["ard_regression"]["metrics"].get("ard_train_rmse", 0)
            )
            ard_test_rmse = self.safe_numeric_convert(
                bayes_data["ard_regression"]["metrics"].get("ard_test_rmse", 0)
            )

            # 더 나은 모델 결정
            if br_test_rmse != 0 and ard_test_rmse != 0:
                better_model = (
                    "Bayesian Ridge" if br_test_rmse < ard_test_rmse else "ARD"
                )
                performance_diff = abs(br_test_rmse - ard_test_rmse)
            else:
                better_model = "Unable to determine"
                performance_diff = 0

            summary_text = f"""
베이지안 분석 요약:

• Bayesian Ridge:
  - 비영계수: {br_nonzero}개
  - Train RMSE: {br_train_rmse:.3f}
  - Test RMSE: {br_test_rmse:.3f}

• ARD (Sparse Bayesian):
  - 비영계수: {ard_nonzero}개  
  - Train RMSE: {ard_train_rmse:.3f}
  - Test RMSE: {ard_test_rmse:.3f}

• 결론:
  - 더 나은 모델: {better_model}
  - 특성 선택: ARD가 {abs(br_nonzero - ard_nonzero)}개 
    {'더 적은' if ard_nonzero < br_nonzero else '더 많은'} 특성 사용
  - 성능 차이: {performance_diff:.3f} RMSE
            """

            ax6.text(
                0.05,
                0.95,
                summary_text.strip(),
                ha="left",
                va="top",
                transform=ax6.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )
            ax6.set_title("Analysis Summary", fontsize=12, fontweight="bold")
            ax6.axis("off")

        plt.tight_layout()
        return fig

    def plot_lasso_results(self, symbol: str, figsize: tuple = (16, 12)) -> plt.Figure:
        """Lasso 회귀 결과 시각화 (개선된 버전)"""
        if symbol not in self.analysis_results:
            raise ValueError(f"Symbol {symbol} not found in analysis results")

        lasso_data = self.analysis_results[symbol]["lasso_regression"]

        # 2x3 레이아웃으로 변경하여 더 많은 정보 표시
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize)

        # 1. 중요한 계수만 표시 (0이 아닌 계수)
        coef_data = lasso_data["coefficients"]
        all_features = [k for k in coef_data.keys() if k != "const"]
        all_coefficients = [
            self.safe_numeric_convert(coef_data[f], 0.0) for f in all_features
        ]

        # 0이 아닌 계수만 필터링 (임계값 사용)
        threshold = 1e-10
        important_indices = [
            i for i, coef in enumerate(all_coefficients) if abs(coef) > threshold
        ]

        if important_indices:
            important_features = [all_features[i] for i in important_indices]
            important_coefficients = [all_coefficients[i] for i in important_indices]

            # 절댓값 기준으로 정렬하여 상위 15개만 표시
            sorted_indices = sorted(
                range(len(important_coefficients)),
                key=lambda i: abs(important_coefficients[i]),
                reverse=True,
            )[:15]

            top_features = [important_features[i] for i in sorted_indices]
            top_coefficients = [important_coefficients[i] for i in sorted_indices]

            colors = ["red" if x < 0 else "blue" for x in top_coefficients]
            bars = ax1.barh(top_features, top_coefficients, color=colors, alpha=0.7)

            # 계수 값 표시
            for i, (bar, coef) in enumerate(zip(bars, top_coefficients)):
                ax1.text(
                    coef
                    + (
                        0.01 * max(abs(c) for c in top_coefficients)
                        if coef >= 0
                        else -0.01 * max(abs(c) for c in top_coefficients)
                    ),
                    bar.get_y() + bar.get_height() / 2,
                    f"{coef:.4f}",
                    va="center",
                    ha="left" if coef >= 0 else "right",
                    fontsize=8,
                )

        ax1.set_title(
            "Non-Zero Lasso Coefficients (Top 15)", fontsize=12, fontweight="bold"
        )
        ax1.set_xlabel("Coefficient Value", fontsize=10)
        ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax1.grid(axis="x", alpha=0.3)

        # 2. 성능 메트릭 비교 (개선된 버전)
        # RMSE 사용 (더 직관적)
        train_rmse = self.safe_numeric_convert(lasso_data.get("train_rmse", 0))
        test_rmse = self.safe_numeric_convert(lasso_data.get("test_rmse", 0))

        models = ["Train RMSE", "Test RMSE"]
        rmse_values = [train_rmse, test_rmse]

        colors_rmse = ["lightblue", "lightcoral"]
        bars_rmse = ax2.bar(models, rmse_values, color=colors_rmse, alpha=0.7)

        ax2.set_title("RMSE Comparison", fontsize=12, fontweight="bold")
        ax2.set_ylabel("RMSE", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

        # RMSE 값 표시
        for bar, val in zip(bars_rmse, rmse_values):
            if val > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # 3. 특성 선택 정보
        total_features = len(all_features)
        important_features_count = len(
            [x for x in all_coefficients if abs(x) > threshold]
        )
        zero_features_count = total_features - important_features_count
        sparsity_ratio = (
            zero_features_count / total_features if total_features > 0 else 0
        )

        categories = ["Selected", "Eliminated"]
        counts = [important_features_count, zero_features_count]
        colors_selection = ["green", "gray"]

        bars_selection = ax3.bar(categories, counts, color=colors_selection, alpha=0.7)
        ax3.set_title("Feature Selection", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Number of Features", fontsize=10)
        ax3.grid(axis="y", alpha=0.3)

        # 특성 개수 표시
        for bar, count in zip(bars_selection, counts):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # 스파스성 비율 표시
        ax3.text(
            0.5,
            0.95,
            f"Sparsity: {sparsity_ratio:.1%}",
            transform=ax3.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        # 4. 성능 메트릭 (CV R² 및 MAE 사용)
        # R² 값들이 타입 지시자로 저장된 경우 CV 값 사용
        train_r2 = self.safe_numeric_convert(lasso_data.get("train_r_squared", 0))
        test_r2 = self.safe_numeric_convert(lasso_data.get("test_r_squared", 0))
        cv_r2_mean = self.safe_numeric_convert(lasso_data.get("cv_r2_mean", 0))

        # R² 값이 유효하지 않으면 CV R²와 MAE 표시
        if train_r2 == 0 and test_r2 == 0 and cv_r2_mean > 0:
            # CV R²와 MAE 표시
            train_mae = self.safe_numeric_convert(lasso_data.get("train_mae", 0))
            test_mae = self.safe_numeric_convert(lasso_data.get("test_mae", 0))

            metrics = ["CV R²", "Train MAE", "Test MAE"]
            values = [cv_r2_mean, train_mae, test_mae]
            colors_metrics = ["green", "lightblue", "lightcoral"]

            # MAE는 더 작은 스케일이므로 정규화
            if train_mae > 0 and test_mae > 0:
                max_mae = max(train_mae, test_mae)
                normalized_values = [
                    cv_r2_mean,
                    train_mae / max_mae * cv_r2_mean,
                    test_mae / max_mae * cv_r2_mean,
                ]
            else:
                normalized_values = values

            bars_r2 = ax4.bar(
                metrics, normalized_values, color=colors_metrics, alpha=0.7
            )
            ax4.set_title("Performance Metrics", fontsize=12, fontweight="bold")
            ax4.set_ylabel("Normalized Score", fontsize=10)
            ax4.grid(axis="y", alpha=0.3)

            # 실제 값 표시
            actual_values = [f"{cv_r2_mean:.4f}", f"{train_mae:.4f}", f"{test_mae:.4f}"]
            for bar, val in zip(bars_r2, actual_values):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    val,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            # 기존 R² 표시
            r2_models = ["Train R²", "Test R²"]
            r2_values = [train_r2, test_r2]
            colors_r2 = ["skyblue", "orange"]

            bars_r2 = ax4.bar(r2_models, r2_values, color=colors_r2, alpha=0.7)
            ax4.set_title("R² Performance", fontsize=12, fontweight="bold")
            ax4.set_ylabel("R² Score", fontsize=10)
            ax4.set_ylim(0, max(max(r2_values) * 1.1, 0.1))
            ax4.grid(axis="y", alpha=0.3)

            # R² 값 표시
            for bar, val in zip(bars_r2, r2_values):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # 5. Alpha 정규화 파라미터 정보
        best_alpha = self.safe_numeric_convert(lasso_data.get("best_alpha", 0))
        cv_r2_mean = self.safe_numeric_convert(lasso_data.get("cv_r2_mean", 0))
        cv_r2_std = self.safe_numeric_convert(lasso_data.get("cv_r2_std", 0))

        # Alpha 값과 CV 성능을 텍스트로 표시
        alpha_info = [
            f"Best Alpha: {best_alpha:.6f}",
            f"CV R² Mean: {cv_r2_mean:.4f}",
            f"CV R² Std: {cv_r2_std:.4f}",
            "",
            f"Features: {total_features} → {important_features_count}",
            f"Reduction: {(1-sparsity_ratio):.1%}",
        ]

        ax5.text(
            0.05,
            0.95,
            "\n".join(alpha_info),
            transform=ax5.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )
        ax5.set_title("Regularization Info", fontsize=12, fontweight="bold")
        ax5.axis("off")

        # 6. 모델 요약 및 해석
        # 유효한 성능 지표 확인
        if train_r2 > 0 and test_r2 > 0:
            # R² 값이 유효한 경우
            overfitting_gap = abs(train_r2 - test_r2)
            overfitting_status = (
                "Good"
                if overfitting_gap < 0.1
                else "Moderate" if overfitting_gap < 0.2 else "High"
            )
            performance_metric = f"• Test R²: {test_r2:.4f}"
            assessment_lines = [
                f"🔍 Model Assessment:",
                f"• Overfitting: {overfitting_status}",
                f"  (Gap: {overfitting_gap:.4f})",
            ]
        else:
            # CV R² 사용
            performance_metric = f"• CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}"
            cv_reliability = (
                "High" if cv_r2_std < 0.05 else "Moderate" if cv_r2_std < 0.1 else "Low"
            )
            assessment_lines = [
                f"🔍 Model Assessment:",
                f"• CV Reliability: {cv_reliability}",
                f"  (Std: {cv_r2_std:.4f})",
            ]

        # RMSE 개선도 (train 대비 test)
        rmse_degradation = (
            (test_rmse - train_rmse) / train_rmse * 100 if train_rmse > 0 else 0
        )

        summary_lines = (
            [
                "🎯 Lasso Analysis Summary",
                "─" * 25,
                f"• Selected Features: {important_features_count}/{total_features}",
                f"• Feature Reduction: {sparsity_ratio:.1%}",
                performance_metric,
                f"• Test RMSE: {test_rmse:.4f}",
                "",
            ]
            + assessment_lines
            + [
                f"• RMSE Change: +{rmse_degradation:.1f}%",
                "",
                f"🏆 Best Alpha: {best_alpha:.2e}",
            ]
        )

        ax6.text(
            0.05,
            0.95,
            "\n".join(summary_lines),
            transform=ax6.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )
        ax6.set_title("Analysis Summary", fontsize=12, fontweight="bold")
        ax6.axis("off")

        plt.tight_layout()
        return fig

    def plot_model_comparison(self, figsize: tuple = (15, 10)) -> plt.Figure:
        """모든 모델의 성능 비교"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 각 심볼별 모델 성능 수집 (RMSE 기반)
        all_models = []
        all_rmse_scores = []
        all_symbols = []

        for symbol in self.symbols:
            if "linear_regression" in self.analysis_results[symbol]:
                all_models.append("Linear")
                rmse_val = self.safe_numeric_convert(
                    self.analysis_results[symbol]["linear_regression"].get(
                        "test_rmse", 0
                    ),
                    1.0,
                )
                all_rmse_scores.append(rmse_val)
                all_symbols.append(symbol)

            if "random_forest" in self.analysis_results[symbol]:
                all_models.append("Random Forest")
                rmse_val = self.safe_numeric_convert(
                    self.analysis_results[symbol]["random_forest"].get("test_rmse", 0),
                    1.0,
                )
                all_rmse_scores.append(rmse_val)
                all_symbols.append(symbol)

            if "bayesian_regression" in self.analysis_results[symbol]:
                all_models.append("Bayesian Ridge")
                rmse_val = self.safe_numeric_convert(
                    self.analysis_results[symbol]["bayesian_regression"][
                        "bayesian_ridge"
                    ]["metrics"].get("br_test_rmse", 0),
                    1.0,
                )
                all_rmse_scores.append(rmse_val)
                all_symbols.append(symbol)

                all_models.append("ARD")
                rmse_val = self.safe_numeric_convert(
                    self.analysis_results[symbol]["bayesian_regression"][
                        "ard_regression"
                    ]["metrics"].get("ard_test_rmse", 0),
                    1.0,
                )
                all_rmse_scores.append(rmse_val)
                all_symbols.append(symbol)

            if "lasso_regression" in self.analysis_results[symbol]:
                all_models.append("Lasso")
                rmse_val = self.safe_numeric_convert(
                    self.analysis_results[symbol]["lasso_regression"].get(
                        "test_rmse", 0
                    ),
                    1.0,
                )
                all_rmse_scores.append(rmse_val)
                all_symbols.append(symbol)

        # 모델별 평균 성능 (RMSE)
        model_performance = {}
        for model, score in zip(all_models, all_rmse_scores):
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(score)

        avg_rmse = {
            model: np.mean(scores) for model, scores in model_performance.items()
        }

        # RMSE를 성능 점수로 변환 (1/(1+RMSE) - 높을수록 좋음)
        performance_scores = {
            model: 1 / (1 + rmse) if rmse > 0 else 0 for model, rmse in avg_rmse.items()
        }

        # 평균 성능 시각화
        models = list(performance_scores.keys())
        scores = list(performance_scores.values())

        bars = ax1.bar(models, scores, color=self.colors[: len(models)], alpha=0.7)
        ax1.set_title(
            "Average Model Performance (1/(1+RMSE))", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Performance Score", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        # 값과 실제 RMSE 표시
        for bar, score, model in zip(bars, scores, models):
            actual_rmse = avg_rmse[model]
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}\n(RMSE: {actual_rmse:.3f})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 심볼별 최고 모델 (RMSE 기반 - 낮을수록 좋음)
        symbol_best_models = {}
        for symbol in self.symbols:
            best_model = "None"
            best_rmse = float("inf")  # 초기값을 무한대로 설정

            if "linear_regression" in self.analysis_results[symbol]:
                rmse = self.safe_numeric_convert(
                    self.analysis_results[symbol]["linear_regression"].get(
                        "test_rmse", 0
                    ),
                    float("inf"),
                )
                if rmse < best_rmse and rmse > 0:
                    best_rmse = rmse
                    best_model = "Linear"

            if "random_forest" in self.analysis_results[symbol]:
                rmse = self.safe_numeric_convert(
                    self.analysis_results[symbol]["random_forest"].get("test_rmse", 0),
                    float("inf"),
                )
                if rmse < best_rmse and rmse > 0:
                    best_rmse = rmse
                    best_model = "Random Forest"

            if "bayesian_regression" in self.analysis_results[symbol]:
                br_rmse = self.safe_numeric_convert(
                    self.analysis_results[symbol]["bayesian_regression"][
                        "bayesian_ridge"
                    ]["metrics"].get("br_test_rmse", 0),
                    float("inf"),
                )
                if br_rmse < best_rmse and br_rmse > 0:
                    best_rmse = br_rmse
                    best_model = "Bayesian Ridge"

                ard_rmse = self.safe_numeric_convert(
                    self.analysis_results[symbol]["bayesian_regression"][
                        "ard_regression"
                    ]["metrics"].get("ard_test_rmse", 0),
                    float("inf"),
                )
                if ard_rmse < best_rmse and ard_rmse > 0:
                    best_rmse = ard_rmse
                    best_model = "ARD"

            if "lasso_regression" in self.analysis_results[symbol]:
                rmse = self.safe_numeric_convert(
                    self.analysis_results[symbol]["lasso_regression"].get(
                        "test_rmse", 0
                    ),
                    float("inf"),
                )
                if rmse < best_rmse and rmse > 0:
                    best_rmse = rmse
                    best_model = "Lasso"

            symbol_best_models[symbol] = best_model

        # 심볼별 최고 모델 시각화
        best_models = list(symbol_best_models.values())
        symbols = list(symbol_best_models.keys())

        model_counts = {}
        for model in best_models:
            model_counts[model] = model_counts.get(model, 0) + 1

        if model_counts:
            best_model_names = list(model_counts.keys())
            best_model_counts = list(model_counts.values())

            bars_best = ax2.pie(
                best_model_counts,
                labels=best_model_names,
                autopct="%1.1f%%",
                colors=self.colors[: len(best_model_names)],
                startangle=90,
            )
            ax2.set_title(
                "Best Model Distribution by Symbol", fontsize=14, fontweight="bold"
            )

        # 상관관계 강도 비교
        correlation_strengths = {}
        for symbol in self.symbols:
            if "correlation" in self.analysis_results[symbol]:
                corr_data = self.analysis_results[symbol]["correlation"][
                    "abs_correlations"
                ]
                avg_corr = np.mean(list(corr_data.values()))
                correlation_strengths[symbol] = avg_corr

        if correlation_strengths:
            corr_symbols = list(correlation_strengths.keys())
            corr_strengths = list(correlation_strengths.values())

            bars_corr = ax3.bar(
                corr_symbols, corr_strengths, color="lightblue", alpha=0.7
            )
            ax3.set_title(
                "Average Correlation Strength by Symbol", fontsize=14, fontweight="bold"
            )
            ax3.set_ylabel("Average Absolute Correlation", fontsize=12)
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(axis="y", alpha=0.3)

            # 값 표시
            for bar, strength in zip(bars_corr, corr_strengths):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{strength:.3f}",
                    ha="center",
                    va="bottom",
                )

        # 특성 중요도 비교 (Random Forest 기준)
        feature_importance_avg = {}
        for symbol in self.symbols:
            if "random_forest" in self.analysis_results[symbol]:
                rf_importance = self.analysis_results[symbol]["random_forest"][
                    "feature_importance"
                ]
                for feature, importance in rf_importance.items():
                    if feature not in feature_importance_avg:
                        feature_importance_avg[feature] = []
                    feature_importance_avg[feature].append(importance)

        if feature_importance_avg:
            # 평균 중요도 계산
            avg_importance = {
                feature: np.mean(importances)
                for feature, importances in feature_importance_avg.items()
            }

            # 상위 10개 특성만 선택
            sorted_features = sorted(
                avg_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            top_features = [f[0] for f in sorted_features]
            top_importances = [f[1] for f in sorted_features]

            bars_feat = ax4.barh(
                top_features, top_importances, color="lightgreen", alpha=0.7
            )
            ax4.set_title(
                "Top 10 Average Feature Importance (RF)", fontsize=14, fontweight="bold"
            )
            ax4.set_xlabel("Average Importance Score", fontsize=12)
            ax4.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return fig

    def create_interactive_dashboard(self) -> go.Figure:
        """인터랙티브 대시보드 생성 (Plotly)"""
        # 서브플롯 생성
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Model Performance Comparison",
                "Correlation Heatmap (TSLL)",
                "Feature Importance (Random Forest)",
                "Linear Regression Coefficients",
                "Bayesian Model Comparison",
                "Risk Metrics",
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # 1. 모델 성능 비교
        model_performance = {}
        for symbol in self.symbols:
            if "linear_regression" in self.analysis_results[symbol]:
                if "Linear" not in model_performance:
                    model_performance["Linear"] = []
                model_performance["Linear"].append(
                    self.analysis_results[symbol]["linear_regression"].get(
                        "r_squared", 0
                    )
                )

            if "random_forest" in self.analysis_results[symbol]:
                if "Random Forest" not in model_performance:
                    model_performance["Random Forest"] = []
                model_performance["Random Forest"].append(
                    self.analysis_results[symbol]["random_forest"].get("r_squared", 0)
                )

        for model, scores in model_performance.items():
            fig.add_trace(
                go.Bar(name=model, x=self.symbols, y=scores, showlegend=True),
                row=1,
                col=1,
            )

        # 2. 상관관계 히트맵 (TSLL)
        if "TSLL" in self.analysis_results:
            corr_data = self.analysis_results["TSLL"]["correlation"][
                "correlation_matrix"
            ]
            corr_df = pd.DataFrame(corr_data)

            fig.add_trace(
                go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale="RdBu",
                    zmid=0,
                ),
                row=1,
                col=2,
            )

        # 3. 특성 중요도 (Random Forest)
        if (
            "TSLL" in self.analysis_results
            and "random_forest" in self.analysis_results["TSLL"]
        ):
            rf_importance = self.analysis_results["TSLL"]["random_forest"][
                "feature_importance"
            ]
            features = list(rf_importance.keys())
            importances = list(rf_importance.values())

            fig.add_trace(
                go.Bar(
                    x=importances, y=features, orientation="h", name="RF Importance"
                ),
                row=2,
                col=1,
            )

        # 4. 선형회귀 계수
        if (
            "TSLL" in self.analysis_results
            and "linear_regression" in self.analysis_results["TSLL"]
        ):
            lr_coef = self.analysis_results["TSLL"]["linear_regression"]["coefficients"]
            coef_features = [k for k in lr_coef.keys() if k != "const"]
            coef_values = [lr_coef[f] for f in coef_features]

            fig.add_trace(
                go.Bar(x=coef_features, y=coef_values, name="LR Coefficients"),
                row=2,
                col=2,
            )

        # 5. 베이지안 모델 비교
        if (
            "TSLL" in self.analysis_results
            and "bayesian_regression" in self.analysis_results["TSLL"]
        ):
            bayes_data = self.analysis_results["TSLL"]["bayesian_regression"]
            models = ["Bayesian Ridge", "ARD"]
            r2_scores = [
                bayes_data["bayesian_ridge"].get("r_squared", 0),
                bayes_data["ard_regression"].get("r_squared", 0),
            ]

            fig.add_trace(
                go.Bar(x=models, y=r2_scores, name="Bayesian Models"), row=3, col=1
            )

        # 6. 리스크 메트릭
        if (
            "TSLL" in self.analysis_results
            and "bayesian_distribution" in self.analysis_results["TSLL"]
        ):
            dist_data = self.analysis_results["TSLL"]["bayesian_distribution"]
            risk_metrics = ["VaR (95%)", "CVaR (95%)", "Volatility Mean"]
            risk_values = [
                dist_data.get("var_95", 0),
                dist_data.get("cvar_95", 0),
                dist_data.get("volatility_mean", 0),
            ]

            fig.add_trace(
                go.Bar(x=risk_metrics, y=risk_values, name="Risk Metrics"), row=3, col=2
            )

        fig.update_layout(height=1200, title_text="Quantitative Analysis Dashboard")
        return fig

    def generate_summary_report(self) -> str:
        """분석 결과 요약 리포트 생성"""
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")

        for symbol in self.symbols:
            report.append(f"📊 {symbol} ANALYSIS")
            report.append("-" * 40)

            # 상관관계 분석
            if "correlation" in self.analysis_results[symbol]:
                corr_data = self.analysis_results[symbol]["correlation"]
                top_features = corr_data.get("top_features", [])
                correlations = corr_data.get("correlations", {})

                report.append("🔗 CORRELATION ANALYSIS:")
                for feature in top_features[:5]:
                    corr_val = correlations.get(feature, 0)
                    report.append(f"  • {feature}: {corr_val:.4f}")
                report.append("")

            # 모델 성능 비교 (RMSE 기반)
            model_performance = {}

            if "linear_regression" in self.analysis_results[symbol]:
                lr_data = self.analysis_results[symbol]["linear_regression"]
                rmse_val = self.safe_numeric_convert(
                    lr_data.get("test_rmse", 0), float("inf")
                )
                if rmse_val < float("inf"):
                    model_performance["Linear Regression"] = rmse_val

            if "random_forest" in self.analysis_results[symbol]:
                rf_data = self.analysis_results[symbol]["random_forest"]
                rmse_val = self.safe_numeric_convert(
                    rf_data.get("test_rmse", 0), float("inf")
                )
                if rmse_val < float("inf"):
                    model_performance["Random Forest"] = rmse_val

            if "bayesian_regression" in self.analysis_results[symbol]:
                bayes_data = self.analysis_results[symbol]["bayesian_regression"]
                br_rmse = self.safe_numeric_convert(
                    bayes_data["bayesian_ridge"]["metrics"].get("br_test_rmse", 0),
                    float("inf"),
                )
                if br_rmse < float("inf"):
                    model_performance["Bayesian Ridge"] = br_rmse

                ard_rmse = self.safe_numeric_convert(
                    bayes_data["ard_regression"]["metrics"].get("ard_test_rmse", 0),
                    float("inf"),
                )
                if ard_rmse < float("inf"):
                    model_performance["ARD"] = ard_rmse

            if "lasso_regression" in self.analysis_results[symbol]:
                lasso_data = self.analysis_results[symbol]["lasso_regression"]
                rmse_val = self.safe_numeric_convert(
                    lasso_data.get("test_rmse", 0), float("inf")
                )
                if rmse_val < float("inf"):
                    model_performance["Lasso"] = rmse_val

            if model_performance:
                report.append("🤖 MODEL PERFORMANCE (RMSE - Lower is Better):")
                for model, rmse in sorted(
                    model_performance.items(), key=lambda x: x[1]  # 낮은 RMSE가 좋음
                ):
                    performance_score = 1 / (1 + rmse)  # 성능 점수로 변환
                    report.append(
                        f"  • {model}: {rmse:.4f} RMSE (Score: {performance_score:.3f})"
                    )
                report.append("")

                # 최고 성능 모델 (가장 낮은 RMSE)
                best_model = min(model_performance.items(), key=lambda x: x[1])
                best_score = 1 / (1 + best_model[1])
                report.append(
                    f"🏆 BEST MODEL: {best_model[0]} (RMSE: {best_model[1]:.4f}, Score: {best_score:.3f})"
                )
                report.append("")
            else:
                report.append("🤖 MODEL PERFORMANCE: No valid RMSE data available")
                report.append("")

            # 리스크 메트릭
            if "bayesian_distribution" in self.analysis_results[symbol]:
                dist_data = self.analysis_results[symbol]["bayesian_distribution"]
                report.append("⚠️ RISK METRICS:")

                var_95 = self.safe_numeric_convert(dist_data.get("var_95", 0), 0.0)
                cvar_95 = self.safe_numeric_convert(dist_data.get("cvar_95", 0), 0.0)
                vol_mean = self.safe_numeric_convert(
                    dist_data.get("volatility_mean", 0), 0.0
                )

                if var_95 != 0:
                    report.append(f"  • VaR (95%): {var_95:.4f}")
                if cvar_95 != 0:
                    report.append(f"  • CVaR (95%): {cvar_95:.4f}")
                if vol_mean != 0:
                    report.append(f"  • Volatility Mean: {vol_mean:.4f}")

                if var_95 == 0 and cvar_95 == 0 and vol_mean == 0:
                    report.append("  • Risk metrics: No valid data available")

                report.append("")

            report.append("")

        # 전체 요약
        report.append("📈 OVERALL SUMMARY")
        report.append("-" * 40)

        # 전체 모델 성능 수집 (RMSE 기반)
        all_rmse_values = []
        all_performance_scores = []

        for symbol in self.symbols:
            for model_type in [
                "linear_regression",
                "random_forest",
                "bayesian_regression",
                "lasso_regression",
            ]:
                if model_type in self.analysis_results[symbol]:
                    if model_type == "bayesian_regression":
                        # Bayesian Ridge
                        br_rmse = self.safe_numeric_convert(
                            self.analysis_results[symbol][model_type]["bayesian_ridge"][
                                "metrics"
                            ].get("br_test_rmse", 0),
                            float("inf"),
                        )
                        if br_rmse < float("inf"):
                            all_rmse_values.append(br_rmse)
                            all_performance_scores.append(1 / (1 + br_rmse))

                        # ARD
                        ard_rmse = self.safe_numeric_convert(
                            self.analysis_results[symbol][model_type]["ard_regression"][
                                "metrics"
                            ].get("ard_test_rmse", 0),
                            float("inf"),
                        )
                        if ard_rmse < float("inf"):
                            all_rmse_values.append(ard_rmse)
                            all_performance_scores.append(1 / (1 + ard_rmse))
                    else:
                        # 다른 모델들
                        rmse_val = self.safe_numeric_convert(
                            self.analysis_results[symbol][model_type].get(
                                "test_rmse", 0
                            ),
                            float("inf"),
                        )
                        if rmse_val < float("inf"):
                            all_rmse_values.append(rmse_val)
                            all_performance_scores.append(1 / (1 + rmse_val))

        if all_rmse_values and all_performance_scores:
            avg_rmse = np.mean(all_rmse_values)
            best_rmse = np.min(all_rmse_values)
            worst_rmse = np.max(all_rmse_values)

            avg_score = np.mean(all_performance_scores)
            best_score = np.max(all_performance_scores)
            worst_score = np.min(all_performance_scores)

            report.append(f"📊 Average RMSE: {avg_rmse:.4f} (Score: {avg_score:.3f})")
            report.append(f"📊 Best RMSE: {best_rmse:.4f} (Score: {best_score:.3f})")
            report.append(f"📊 Worst RMSE: {worst_rmse:.4f} (Score: {worst_score:.3f})")
            report.append(f"📊 Total Models Analyzed: {len(all_rmse_values)}")
        else:
            report.append("📊 No valid performance data available")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)
