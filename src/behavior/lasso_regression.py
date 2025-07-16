#!/usr/bin/env python3
"""
Lasso 회귀 분석 모듈
변수 중요도 판단 및 정규화를 통한 특성 선택
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class LassoRegressionAnalyzer:
    """Lasso 회귀 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.model = None
        self.scaler = StandardScaler()

    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str = "return",
        feature_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        max_iter: int = 2000,
    ) -> Dict[str, Any]:
        """
        Lasso 회귀 분석 실행

        Args:
            data: 분석할 데이터프레임
            target_column: 종속변수 컬럼명
            feature_columns: 독립변수 컬럼명 리스트
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
            cv_folds: 교차 검증 폴드 수
            max_iter: 최대 반복 횟수

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

        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # LassoCV로 최적 alpha 찾기
        lasso_cv = LassoCV(
            cv=cv_folds,
            random_state=random_state,
            max_iter=max_iter,
            alphas=np.logspace(-4, 1, 50),
        )
        lasso_cv.fit(X_train_scaled, y_train)

        # 최적 alpha로 Lasso 모델 학습
        best_alpha = lasso_cv.alpha_
        lasso_model = Lasso(
            alpha=best_alpha, max_iter=max_iter, random_state=random_state
        )
        lasso_model.fit(X_train_scaled, y_train)

        # 예측
        y_pred_train = lasso_model.predict(X_train_scaled)
        y_pred_test = lasso_model.predict(X_test_scaled)

        # 성능 지표 계산
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # 교차 검증 성능
        cv_scores = cross_val_score(
            lasso_model, X_train_scaled, y_train, cv=cv_folds, scoring="r2"
        )

        # 계수 정보
        coefficients = dict(zip(feature_columns, lasso_model.coef_))
        intercept = lasso_model.intercept_

        # 중요 특성 선택 (계수가 0이 아닌 것들)
        important_features = [f for f, c in coefficients.items() if abs(c) > 1e-10]
        zero_features = [f for f, c in coefficients.items() if abs(c) <= 1e-10]

        # 계수 절댓값 기준 정렬
        sorted_coefficients = sorted(
            coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # 결과 구성
        result = {
            "model_type": "Lasso Regression",
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
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            # 모델 파라미터
            "best_alpha": best_alpha,
            "intercept": intercept,
            # 계수 정보
            "coefficients": coefficients,
            "sorted_coefficients": sorted_coefficients,
            "important_features": important_features,
            "zero_features": zero_features,
            "n_important_features": len(important_features),
            "n_zero_features": len(zero_features),
            "sparsity_ratio": len(zero_features) / len(feature_columns),
            # 모델 객체
            "model": lasso_model,
            "cv_model": lasso_cv,
            "scaler": self.scaler,
            # 예측값
            "y_train": y_train.values,
            "y_test": y_test.values,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            # 교차 검증 결과
            "cv_scores": cv_scores,
            "cv_alphas": lasso_cv.alphas_,
            "cv_mse_path": lasso_cv.mse_path_,
        }

        self.results = result
        self.model = lasso_model
        return result

    def plot_coefficient_path(
        self, figsize: Tuple[int, int] = (12, 8), save_path: str = None
    ) -> plt.Figure:
        """계수 경로 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. 계수 크기 (중요한 특성들만)
        important_coeffs = {
            k: v
            for k, v in result["coefficients"].items()
            if k in result["important_features"]
        }

        if important_coeffs:
            features = list(important_coeffs.keys())
            coeffs = list(important_coeffs.values())

            colors = ["red" if c < 0 else "blue" for c in coeffs]
            bars = ax1.barh(features, coeffs, color=colors, alpha=0.7)
            ax1.set_xlabel("계수")
            ax1.set_title(f'중요 특성 계수 (α={result["best_alpha"]:.4f})')
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

        # 2. 교차 검증 MSE 경로
        alphas = result["cv_alphas"]
        mse_path = result["cv_mse_path"]
        mse_mean = mse_path.mean(axis=1)
        mse_std = mse_path.std(axis=1)

        ax2.semilogx(alphas, mse_mean, "b-", label="Mean MSE")
        ax2.fill_between(alphas, mse_mean - mse_std, mse_mean + mse_std, alpha=0.3)
        ax2.axvline(
            result["best_alpha"],
            color="red",
            linestyle="--",
            label=f'Best α = {result["best_alpha"]:.4f}',
        )
        ax2.set_xlabel("Alpha")
        ax2.set_ylabel("Mean Squared Error")
        ax2.set_title("교차 검증 MSE 경로")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 실제값 vs 예측값
        ax3.scatter(result["y_test"], result["y_pred_test"], alpha=0.6, s=20)
        ax3.plot(
            [result["y_test"].min(), result["y_test"].max()],
            [result["y_test"].min(), result["y_test"].max()],
            "r--",
            alpha=0.8,
        )
        ax3.set_xlabel("실제값")
        ax3.set_ylabel("예측값")
        ax3.set_title(f'실제값 vs 예측값 (R² = {result["r_squared"]:.4f})')
        ax3.grid(True, alpha=0.3)

        # 4. 특성 선택 요약
        categories = ["중요 특성", "제거된 특성"]
        counts = [result["n_important_features"], result["n_zero_features"]]
        colors = ["blue", "gray"]

        bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
        ax4.set_ylabel("특성 수")
        ax4.set_title(f'특성 선택 결과 (스파스성: {result["sparsity_ratio"]:.1%})')
        ax4.grid(axis="y", alpha=0.3)

        # 값 표시
        for bar, count in zip(bars, counts):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"계수 경로 플롯 저장: {save_path}")

        return fig

    def plot_feature_importance(
        self, top_n: int = 10, figsize: Tuple[int, int] = (10, 8), save_path: str = None
    ) -> plt.Figure:
        """특성 중요도 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        # 상위 n개 특성 선택
        top_features = result["sorted_coefficients"][:top_n]
        features = [f[0] for f in top_features]
        coeffs = [f[1] for f in top_features]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. 계수 크기
        colors = ["red" if c < 0 else "blue" for c in coeffs]
        bars = ax1.barh(features, coeffs, color=colors, alpha=0.7)
        ax1.set_xlabel("계수")
        ax1.set_title(f"상위 {top_n}개 특성 계수")
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

        # 2. 절댓값 기준
        abs_coeffs = [abs(c) for c in coeffs]
        bars = ax2.barh(features, abs_coeffs, color="green", alpha=0.7)
        ax2.set_xlabel("절댓값 계수")
        ax2.set_title(f"상위 {top_n}개 특성 중요도")
        ax2.grid(axis="x", alpha=0.3)

        # 값 표시
        for bar, abs_coeff in zip(bars, abs_coeffs):
            ax2.text(
                abs_coeff + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{abs_coeff:.4f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"특성 중요도 플롯 저장: {save_path}")

        return fig

    def get_summary(self) -> str:
        """분석 결과 요약 문자열 반환"""
        if not self.results:
            return "분석 결과가 없습니다."

        result = self.results
        summary_lines = []

        summary_lines.append("=" * 60)
        summary_lines.append("Lasso 회귀 분석 결과 요약")
        summary_lines.append("=" * 60)
        summary_lines.append(f"모델 타입: {result['model_type']}")
        summary_lines.append(f"종속변수: {result['target_column']}")
        summary_lines.append(f"전체 특성 수: {result['n_features']}")
        summary_lines.append(f"샘플 수: {result['n_samples']}")
        summary_lines.append(
            f"훈련/테스트: {result['train_size']}/{result['test_size']}"
        )

        summary_lines.append(f"\n모델 성능:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"R² (테스트): {result['r_squared']:.4f}")
        summary_lines.append(f"R² (훈련): {result['train_r_squared']:.4f}")
        summary_lines.append(
            f"교차검증 R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}"
        )
        summary_lines.append(f"RMSE (테스트): {result['test_rmse']:.4f}")
        summary_lines.append(f"MAE (테스트): {result['test_mae']:.4f}")

        summary_lines.append(f"\n모델 파라미터:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"최적 Alpha: {result['best_alpha']:.6f}")
        summary_lines.append(f"절편: {result['intercept']:.4f}")

        summary_lines.append(f"\n특성 선택 결과:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"중요 특성 수: {result['n_important_features']}")
        summary_lines.append(f"제거된 특성 수: {result['n_zero_features']}")
        summary_lines.append(f"스파스성 비율: {result['sparsity_ratio']:.1%}")

        if result["important_features"]:
            summary_lines.append(f"\n중요 특성들 (계수 순):")
            summary_lines.append("-" * 40)
            for i, (feature, coeff) in enumerate(result["sorted_coefficients"][:10], 1):
                summary_lines.append(f"{i:2d}. {feature:<20} {coeff:>8.4f}")

        if result["zero_features"]:
            summary_lines.append(f"\n제거된 특성들:")
            summary_lines.append("-" * 40)
            for feature in result["zero_features"][:10]:  # 최대 10개만 표시
                summary_lines.append(f"  {feature}")
            if len(result["zero_features"]) > 10:
                summary_lines.append(f"  ... 외 {len(result['zero_features']) - 10}개")

        return "\n".join(summary_lines)

    def print_summary(self):
        """분석 결과 요약 출력"""
        print(self.get_summary())

    def get_important_features(self, threshold: float = 1e-10) -> List[str]:
        """중요 특성 목록 반환"""
        if not self.results:
            return []

        return [
            f for f, c in self.results["coefficients"].items() if abs(c) > threshold
        ]

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """특성 중요도 순위 반환"""
        if not self.results:
            return []

        return self.results["sorted_coefficients"]
