#!/usr/bin/env python3
"""
랜덤 포레스트 분석 모듈
특성 중요도 분석 및 예측 수행
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class RandomForestAnalyzer:
    """랜덤 포레스트 분석 클래스"""

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
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        cv_folds: int = 5,
        tune_hyperparameters: bool = False,
    ) -> Dict[str, Any]:
        """
        랜덤 포레스트 분석 실행

        Args:
            data: 분석할 데이터프레임
            target_column: 종속변수 컬럼명
            feature_columns: 독립변수 컬럼명 리스트
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
            n_estimators: 트리 개수
            max_depth: 최대 깊이
            min_samples_split: 분할 최소 샘플 수
            min_samples_leaf: 리프 최소 샘플 수
            cv_folds: 교차 검증 폴드 수
            tune_hyperparameters: 하이퍼파라미터 튜닝 여부

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

        # 특성 스케일링 (랜덤 포레스트는 스케일링이 필요 없지만 일관성을 위해)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 하이퍼파라미터 튜닝
        if tune_hyperparameters:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

            rf_cv = RandomForestRegressor(random_state=random_state)
            grid_search = GridSearchCV(
                rf_cv, param_grid, cv=cv_folds, scoring="r2", n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            # 최적 파라미터로 모델 학습
            rf_model = RandomForestRegressor(**best_params, random_state=random_state)
        else:
            # 기본 파라미터로 모델 학습
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            best_params = rf_model.get_params()
            best_score = None

        rf_model.fit(X_train_scaled, y_train)

        # 예측
        y_pred_train = rf_model.predict(X_train_scaled)
        y_pred_test = rf_model.predict(X_test_scaled)

        # 성능 지표 계산
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # 교차 검증 성능
        cv_scores = cross_val_score(
            rf_model, X_train_scaled, y_train, cv=cv_folds, scoring="r2"
        )

        # 특성 중요도
        feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
        sorted_importance = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # 상위 특성들
        top_features = [f[0] for f in sorted_importance[:10]]
        top_importance = [f[1] for f in sorted_importance[:10]]

        # 결과 구성
        result = {
            "model_type": "Random Forest Regressor",
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
            "best_params": best_params,
            "best_cv_score": best_score,
            "n_estimators": rf_model.n_estimators,
            "max_depth": rf_model.max_depth,
            # 특성 중요도
            "feature_importance": feature_importance,
            "sorted_importance": sorted_importance,
            "top_features": top_features,
            "top_importance": top_importance,
            # 모델 객체
            "model": rf_model,
            "scaler": self.scaler,
            # 예측값
            "y_train": y_train.values,
            "y_test": y_test.values,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            # 교차 검증 결과
            "cv_scores": cv_scores,
            # 추가 정보
            "n_trees": rf_model.n_estimators,
            "avg_tree_depth": np.mean(
                [tree.get_depth() for tree in rf_model.estimators_]
            ),
        }

        self.results = result
        self.model = rf_model
        return result

    def plot_feature_importance(
        self, top_n: int = 15, figsize: Tuple[int, int] = (12, 8), save_path: str = None
    ) -> plt.Figure:
        """특성 중요도 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        # 상위 n개 특성 선택
        top_features = result["sorted_importance"][:top_n]
        features = [f[0] for f in top_features]
        importance = [f[1] for f in top_features]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. 특성 중요도 바 차트
        bars = ax1.barh(features, importance, color="skyblue", alpha=0.7)
        ax1.set_xlabel("중요도")
        ax1.set_title(f"상위 {top_n}개 특성 중요도")
        ax1.grid(axis="x", alpha=0.3)

        # 값 표시
        for bar, imp in zip(bars, importance):
            ax1.text(
                imp + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}",
                va="center",
                fontsize=9,
            )

        # 2. 누적 중요도
        cumulative_importance = np.cumsum(importance)
        ax2.plot(range(1, len(importance) + 1), cumulative_importance, "b-o", alpha=0.7)
        ax2.set_xlabel("특성 수")
        ax2.set_ylabel("누적 중요도")
        ax2.set_title("누적 특성 중요도")
        ax2.grid(True, alpha=0.3)

        # 80% 임계선 표시
        threshold_80 = 0.8
        idx_80 = np.argmax(cumulative_importance >= threshold_80) + 1
        ax2.axhline(y=threshold_80, color="red", linestyle="--", alpha=0.7)
        ax2.axvline(x=idx_80, color="red", linestyle="--", alpha=0.7)
        ax2.text(
            idx_80 + 0.5,
            threshold_80 + 0.02,
            f"{idx_80}개 특성",
            fontsize=10,
            color="red",
        )

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

        # 4. 잔차 플롯
        residuals = result["y_test"] - result["y_pred_test"]
        ax4.scatter(result["y_pred_test"], residuals, alpha=0.6, s=20)
        ax4.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax4.set_xlabel("예측값")
        ax4.set_ylabel("잔차")
        ax4.set_title("잔차 vs 예측값")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"특성 중요도 플롯 저장: {save_path}")

        return fig

    def plot_tree_analysis(
        self, figsize: Tuple[int, int] = (12, 8), save_path: str = None
    ) -> plt.Figure:
        """트리 분석 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. 트리별 깊이 분포
        tree_depths = [tree.get_depth() for tree in result["model"].estimators_]
        ax1.hist(tree_depths, bins=20, alpha=0.7, color="green", edgecolor="black")
        ax1.set_xlabel("트리 깊이")
        ax1.set_ylabel("빈도")
        ax1.set_title(f'트리 깊이 분포 (평균: {result["avg_tree_depth"]:.1f})')
        ax1.grid(True, alpha=0.3)

        # 2. 교차 검증 성능
        cv_scores = result["cv_scores"]
        ax2.hist(cv_scores, bins=10, alpha=0.7, color="orange", edgecolor="black")
        ax2.set_xlabel("R² Score")
        ax2.set_ylabel("빈도")
        ax2.set_title(f"교차 검증 성능 (평균: {cv_scores.mean():.4f})")
        ax2.grid(True, alpha=0.3)

        # 3. 훈련 vs 테스트 성능 비교
        performance_metrics = ["R²", "RMSE", "MAE"]
        train_scores = [
            result["train_r_squared"],
            result["train_rmse"],
            result["train_mae"],
        ]
        test_scores = [
            result["test_r_squared"],
            result["test_rmse"],
            result["test_mae"],
        ]

        x = np.arange(len(performance_metrics))
        width = 0.35

        bars1 = ax3.bar(x - width / 2, train_scores, width, label="훈련", alpha=0.7)
        bars2 = ax3.bar(x + width / 2, test_scores, width, label="테스트", alpha=0.7)

        ax3.set_xlabel("성능 지표")
        ax3.set_ylabel("점수")
        ax3.set_title("훈련 vs 테스트 성능 비교")
        ax3.set_xticks(x)
        ax3.set_xticklabels(performance_metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # 4. 특성 중요도 분포
        importance_values = list(result["feature_importance"].values())
        ax4.hist(
            importance_values, bins=20, alpha=0.7, color="purple", edgecolor="black"
        )
        ax4.set_xlabel("중요도")
        ax4.set_ylabel("빈도")
        ax4.set_title("특성 중요도 분포")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"트리 분석 플롯 저장: {save_path}")

        return fig

    def get_summary(self) -> str:
        """분석 결과 요약 문자열 반환"""
        if not self.results:
            return "분석 결과가 없습니다."

        result = self.results
        summary_lines = []

        summary_lines.append("=" * 60)
        summary_lines.append("랜덤 포레스트 분석 결과 요약")
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
        summary_lines.append(
            f"교차검증 R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}"
        )
        summary_lines.append(f"RMSE (테스트): {result['test_rmse']:.4f}")
        summary_lines.append(f"MAE (테스트): {result['test_mae']:.4f}")

        summary_lines.append(f"\n모델 구조:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"트리 개수: {result['n_estimators']}")
        summary_lines.append(f"최대 깊이: {result['max_depth']}")
        summary_lines.append(f"평균 트리 깊이: {result['avg_tree_depth']:.1f}")

        if result["best_cv_score"]:
            summary_lines.append(f"최적 CV 점수: {result['best_cv_score']:.4f}")

        summary_lines.append(f"\n상위 특성 중요도:")
        summary_lines.append("-" * 40)
        for i, (feature, importance) in enumerate(result["sorted_importance"][:10], 1):
            summary_lines.append(f"{i:2d}. {feature:<20} {importance:>8.4f}")

        # 누적 중요도 계산
        cumulative_importance = np.cumsum(
            [imp for _, imp in result["sorted_importance"]]
        )
        threshold_80 = 0.8
        idx_80 = np.argmax(cumulative_importance >= threshold_80) + 1

        summary_lines.append(f"\n특성 선택 가이드:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"80% 중요도 달성에 필요한 특성 수: {idx_80}")
        summary_lines.append(
            f"상위 {idx_80}개 특성의 누적 중요도: {cumulative_importance[idx_80-1]:.1%}"
        )

        return "\n".join(summary_lines)

    def print_summary(self):
        """분석 결과 요약 출력"""
        print(self.get_summary())

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """상위 n개 특성 반환"""
        if not self.results:
            return []

        return self.results["sorted_importance"][:n]

    def get_important_features(self, threshold: float = 0.01) -> List[str]:
        """중요도 임계값 이상의 특성 반환"""
        if not self.results:
            return []

        return [
            f
            for f, imp in self.results["feature_importance"].items()
            if imp >= threshold
        ]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """새로운 데이터에 대한 예측"""
        if self.model is None:
            raise ValueError("먼저 모델을 학습해주세요.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
