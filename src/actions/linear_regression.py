#!/usr/bin/env python3
"""
선형회귀 분석 모듈
statsmodels를 사용한 선형회귀 분석 및 VIF 다중공선성 점검
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class LinearRegressionAnalyzer:
    """선형회귀 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.model = None

    def calculate_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """VIF(분산팽창인자) 계산"""
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]

        return dict(zip(vif_data["Variable"], vif_data["VIF"]))

    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str = "return",
        feature_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        선형회귀 분석 실행

        Args:
            data: 분석할 데이터프레임
            target_column: 종속변수 컬럼명
            feature_columns: 독립변수 컬럼명 리스트
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드

        Returns:
            분석 결과 딕셔너리
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

        # VIF 계산
        vif_scores = self.calculate_vif(X_train)

        # 상수항 추가 (statsmodels용)
        X_train_with_const = sm.add_constant(X_train)
        X_test_with_const = sm.add_constant(X_test)

        # statsmodels로 회귀 분석
        model = sm.OLS(y_train, X_train_with_const).fit()

        # 예측
        y_pred_train = model.predict(X_train_with_const)
        y_pred_test = model.predict(X_test_with_const)

        # 성능 지표 계산
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # 계수 정보
        coefficients = {}
        p_values = {}
        t_values = {}
        std_errors = {}

        for i, feature in enumerate(feature_columns):
            coefficients[feature] = model.params[i + 1]  # +1 for constant
            p_values[feature] = model.pvalues[i + 1]
            t_values[feature] = model.tvalues[i + 1]
            std_errors[feature] = model.bse[i + 1]

        # 상수항 정보
        coefficients["const"] = model.params[0]
        p_values["const"] = model.pvalues[0]
        t_values["const"] = model.tvalues[0]
        std_errors["const"] = model.bse[0]

        # 결과 구성
        result = {
            "model_type": "Linear Regression (statsmodels)",
            "target_column": target_column,
            "feature_columns": feature_columns,
            "n_samples": len(analysis_data),
            "n_features": len(feature_columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
            # 모델 성능
            "r_squared": test_r2,
            "train_r_squared": train_r2,
            "test_r_squared": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            # 계수 정보
            "coefficients": coefficients,
            "p_values": p_values,
            "t_values": t_values,
            "std_errors": std_errors,
            # VIF 정보
            "vif_scores": vif_scores,
            "high_vif_features": [f for f, vif in vif_scores.items() if vif > 10],
            # 모델 객체
            "model": model,
            # 예측값
            "y_train": y_train.values,
            "y_test": y_test.values,
            "y_pred_train": y_pred_train.values,
            "y_pred_test": y_pred_test.values,
            # 모델 요약
            "model_summary": str(model.summary()),
            "aic": model.aic,
            "bic": model.bic,
            "adj_r_squared": model.rsquared_adj,
        }

        self.results = result
        self.model = model
        return result

    def plot_coefficients(
        self, figsize: Tuple[int, int] = (12, 8), save_path: str = None
    ) -> plt.Figure:
        """계수 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. 계수 크기 (상수항 제외)
        feature_coeffs = {
            k: v for k, v in result["coefficients"].items() if k != "const"
        }
        features = list(feature_coeffs.keys())
        coeffs = list(feature_coeffs.values())

        colors = ["red" if c < 0 else "blue" for c in coeffs]
        bars = ax1.barh(features, coeffs, color=colors, alpha=0.7)
        ax1.set_xlabel("계수")
        ax1.set_title("특성별 계수")
        ax1.grid(axis="x", alpha=0.3)

        # 값 표시
        for bar, coeff in zip(bars, coeffs):
            ax1.text(
                coeff + (0.01 if coeff >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{coeff:.4f}",
                va="center",
                fontsize=9,
            )

        # 2. p-value
        feature_pvals = {k: v for k, v in result["p_values"].items() if k != "const"}
        pvals = list(feature_pvals.values())

        colors = ["red" if p < 0.05 else "gray" for p in pvals]
        bars = ax2.barh(features, pvals, color=colors, alpha=0.7)
        ax2.set_xlabel("p-value")
        ax2.set_title("특성별 p-value")
        ax2.axvline(x=0.05, color="red", linestyle="--", alpha=0.7)
        ax2.grid(axis="x", alpha=0.3)

        # 3. VIF
        vif_scores = result["vif_scores"]
        vif_features = list(vif_scores.keys())
        vif_values = list(vif_scores.values())

        colors = ["red" if v > 10 else "blue" for v in vif_values]
        bars = ax3.barh(vif_features, vif_values, color=colors, alpha=0.7)
        ax3.set_xlabel("VIF")
        ax3.set_title("특성별 VIF")
        ax3.axvline(x=10, color="red", linestyle="--", alpha=0.7)
        ax3.grid(axis="x", alpha=0.3)

        # 4. 실제값 vs 예측값
        ax4.scatter(result["y_test"], result["y_pred_test"], alpha=0.6, s=20)
        ax4.plot(
            [result["y_test"].min(), result["y_test"].max()],
            [result["y_test"].min(), result["y_test"].max()],
            "r--",
            alpha=0.8,
        )
        ax4.set_xlabel("실제값")
        ax4.set_ylabel("예측값")
        ax4.set_title(f'실제값 vs 예측값 (R² = {result["r_squared"]:.4f})')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"계수 플롯 저장: {save_path}")

        return fig

    def plot_residuals(
        self, figsize: Tuple[int, int] = (12, 8), save_path: str = None
    ) -> plt.Figure:
        """잔차 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        # 잔차 계산
        residuals_train = result["y_train"] - result["y_pred_train"]
        residuals_test = result["y_test"] - result["y_pred_test"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. 잔차 vs 예측값 (훈련)
        ax1.scatter(result["y_pred_train"], residuals_train, alpha=0.6, s=20)
        ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax1.set_xlabel("예측값")
        ax1.set_ylabel("잔차")
        ax1.set_title("잔차 vs 예측값 (훈련)")
        ax1.grid(True, alpha=0.3)

        # 2. 잔차 vs 예측값 (테스트)
        ax2.scatter(result["y_pred_test"], residuals_test, alpha=0.6, s=20)
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("예측값")
        ax2.set_ylabel("잔차")
        ax2.set_title("잔차 vs 예측값 (테스트)")
        ax2.grid(True, alpha=0.3)

        # 3. 잔차 히스토그램 (훈련)
        ax3.hist(residuals_train, bins=30, alpha=0.7, color="blue", edgecolor="black")
        ax3.set_xlabel("잔차")
        ax3.set_ylabel("빈도")
        ax3.set_title("잔차 분포 (훈련)")
        ax3.grid(True, alpha=0.3)

        # 4. Q-Q 플롯 (훈련)
        from scipy import stats

        stats.probplot(residuals_train, dist="norm", plot=ax4)
        ax4.set_title("Q-Q 플롯 (훈련)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"잔차 플롯 저장: {save_path}")

        return fig

    def get_summary(self) -> str:
        """분석 결과 요약 문자열 반환"""
        if not self.results:
            return "분석 결과가 없습니다."

        result = self.results
        summary_lines = []

        summary_lines.append("=" * 60)
        summary_lines.append("선형회귀 분석 결과 요약")
        summary_lines.append("=" * 60)
        summary_lines.append(f"모델 타입: {result['model_type']}")
        summary_lines.append(f"종속변수: {result['target_column']}")
        summary_lines.append(f"특성 수: {result['n_features']}")
        summary_lines.append(f"샘플 수: {result['n_samples']}")
        summary_lines.append(
            f"훈련/테스트: {result['train_size']}/{result['test_size']}"
        )

        summary_lines.append(f"\n모델 성능:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"R² (테스트): {result['r_squared']:.4f}")
        summary_lines.append(f"R² (훈련): {result['train_r_squared']:.4f}")
        summary_lines.append(f"조정 R²: {result['adj_r_squared']:.4f}")
        summary_lines.append(f"RMSE (테스트): {result['test_rmse']:.4f}")
        summary_lines.append(f"MAE (테스트): {result['test_mae']:.4f}")
        summary_lines.append(f"AIC: {result['aic']:.2f}")
        summary_lines.append(f"BIC: {result['bic']:.2f}")

        summary_lines.append(f"\n계수 정보:")
        summary_lines.append("-" * 40)
        for feature in result["feature_columns"]:
            coeff = result["coefficients"][feature]
            p_val = result["p_values"][feature]
            vif = result["vif_scores"].get(feature, 0)
            significance = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            vif_warning = " (VIF>10)" if vif > 10 else ""
            summary_lines.append(
                f"{feature:<20} {coeff:>8.4f} {significance} (p={p_val:.3e}, VIF={vif:.2f}){vif_warning}"
            )

        if result["high_vif_features"]:
            summary_lines.append(f"\n⚠️ 높은 VIF 특성들 (VIF > 10):")
            summary_lines.append("-" * 40)
            for feature in result["high_vif_features"]:
                vif = result["vif_scores"][feature]
                summary_lines.append(f"  {feature}: VIF = {vif:.2f}")

        return "\n".join(summary_lines)

    def print_summary(self):
        """분석 결과 요약 출력"""
        print(self.get_summary())

    def print_model_summary(self):
        """statsmodels 모델 요약 출력"""
        if self.model:
            print(self.model.summary())
        else:
            print("모델이 없습니다. 먼저 분석을 실행해주세요.")
