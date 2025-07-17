#!/usr/bin/env python3
"""
상관관계 분석 모듈
종가 수익률과 다른 요인들 간의 상관관계를 분석하여 상위 특성들을 추출
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings("ignore")


class CorrelationAnalyzer:
    """상관관계 분석 클래스"""

    def __init__(self):
        self.results = {}

    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str = "return",
        top_n: int = 10,
        method: str = "pearson",  # 'pearson' or 'spearman'
    ) -> Dict[str, Any]:
        """
        상관관계 분석 실행

        Args:
            data: 분석할 데이터프레임
            target_column: 종속변수 컬럼명
            top_n: 상위 상관관계 특성 수
            method: 상관관계 계산 방법 ('pearson' 또는 'spearman')

        Returns:
            분석 결과 딕셔너리
        """
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

        # 특성 컬럼들 (target과 제외 컬럼들 제외)
        feature_columns = [
            col
            for col in data.columns
            if col != target_column and col not in excluded_columns
        ]

        # 상관관계 계산
        correlations = {}
        p_values = {}

        for feature in feature_columns:
            # NaN 제거
            valid_data = data[[feature, target_column]].dropna()

            # 데이터 유효성 검사
            if len(valid_data) < 10:  # 최소 데이터 포인트 확인
                continue

            # 상수열 또는 너무 적은 고유값 체크
            if valid_data[feature].nunique() < 2:
                continue

            if method == "pearson":
                corr, p_val = pearsonr(valid_data[feature], valid_data[target_column])
            else:  # spearman
                corr, p_val = spearmanr(valid_data[feature], valid_data[target_column])

            # NaN/inf 체크
            if np.isnan(corr) or np.isinf(corr) or np.isnan(p_val) or np.isinf(p_val):
                continue

            correlations[feature] = corr
            p_values[feature] = p_val

        # 절댓값 기준으로 정렬
        abs_correlations = {k: abs(v) for k, v in correlations.items()}
        sorted_features = sorted(
            abs_correlations.items(), key=lambda x: x[1], reverse=True
        )

        # 상위 n개 특성 선택
        top_features = [feature for feature, _ in sorted_features[:top_n]]

        # 결과 구성
        result = {
            "method": method,
            "target_column": target_column,
            "total_features": len(feature_columns),
            "analyzed_features": len(correlations),
            "top_n": top_n,
            "top_features": top_features,
            "correlations": {
                feature: correlations[feature] for feature in top_features
            },
            "abs_correlations": {
                feature: abs_correlations[feature] for feature in top_features
            },
            "p_values": {feature: p_values[feature] for feature in top_features},
            "significant_features": [
                feature for feature in top_features if p_values[feature] < 0.05
            ],
        }

        # 상관관계 매트릭스 계산 (상위 특성들만)
        if len(top_features) > 1:
            top_data = data[top_features + [target_column]].dropna()
            correlation_matrix = top_data.corr(method=method)
            result["correlation_matrix"] = correlation_matrix

        self.results = result
        return result

    def plot_correlation_heatmap(
        self,
        data: pd.DataFrame,
        target_column: str = "return",
        top_n: int = 10,
        method: str = "pearson",
        figsize: Tuple[int, int] = (12, 8),
        save_path: str = None,
    ) -> plt.Figure:
        """상관관계 히트맵 플롯"""

        # 분석 실행
        result = self.analyze(data, target_column, top_n, method)

        # 상위 특성들만 선택
        top_features = result["top_features"]
        plot_data = data[top_features + [target_column]].dropna()

        # 상관관계 매트릭스 계산
        corr_matrix = plot_data.corr(method=method)

        # 플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. 히트맵
        sns.heatmap(
            corr_matrix, annot=True, cmap="RdBu_r", center=0, square=True, ax=ax1
        )
        ax1.set_title(f"상관관계 히트맵 ({method})")

        # 2. 상위 특성들의 절댓값 상관관계 바 차트
        abs_corr = result["abs_correlations"]
        features = list(abs_corr.keys())
        values = list(abs_corr.values())

        colors = ["red" if result["correlations"][f] < 0 else "blue" for f in features]

        bars = ax2.barh(features, values, color=colors, alpha=0.7)
        ax2.set_xlabel("절댓값 상관관계")
        ax2.set_title(f"상위 {top_n}개 특성 상관관계")
        ax2.grid(axis="x", alpha=0.3)

        # 값 표시
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax2.text(
                value + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"상관관계 히트맵 저장: {save_path}")

        return fig

    def plot_correlation_scatter(
        self,
        data: pd.DataFrame,
        target_column: str = "return",
        top_n: int = 5,
        method: str = "pearson",
        figsize: Tuple[int, int] = (15, 10),
        save_path: str = None,
    ) -> plt.Figure:
        """상위 특성들과 종속변수의 산점도 플롯"""

        # 분석 실행
        result = self.analyze(data, target_column, top_n, method)
        top_features = result["top_features"]

        # 서브플롯 개수 계산
        n_features = len(top_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # 각 특성별 산점도
        for i, feature in enumerate(top_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # 데이터 준비
            plot_data = data[[feature, target_column]].dropna()

            # 산점도
            ax.scatter(plot_data[feature], plot_data[target_column], alpha=0.6, s=20)

            # 회귀선
            z = np.polyfit(plot_data[feature], plot_data[target_column], 1)
            p = np.poly1d(z)
            ax.plot(plot_data[feature], p(plot_data[feature]), "r--", alpha=0.8)

            # 상관관계 정보
            corr = result["correlations"][feature]
            p_val = result["p_values"][feature]
            ax.set_xlabel(feature)
            ax.set_ylabel(target_column)
            ax.set_title(f"{feature}\nCorr: {corr:.3f}, p: {p_val:.3e}")
            ax.grid(True, alpha=0.3)

        # 빈 서브플롯 숨기기
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"상관관계 산점도 저장: {save_path}")

        return fig

    def get_summary(self) -> str:
        """분석 결과 요약 문자열 반환"""
        if not self.results:
            return "분석 결과가 없습니다."

        result = self.results
        summary_lines = []

        summary_lines.append("=" * 60)
        summary_lines.append("상관관계 분석 결과 요약")
        summary_lines.append("=" * 60)
        summary_lines.append(f"분석 방법: {result['method']}")
        summary_lines.append(f"종속변수: {result['target_column']}")
        summary_lines.append(f"전체 특성 수: {result['total_features']}")
        summary_lines.append(f"분석된 특성 수: {result['analyzed_features']}")
        summary_lines.append(f"상위 특성 수: {result['top_n']}")
        summary_lines.append(f"유의한 특성 수: {len(result['significant_features'])}")

        summary_lines.append("\n상위 특성들:")
        summary_lines.append("-" * 40)
        for i, feature in enumerate(result["top_features"], 1):
            corr = result["correlations"][feature]
            p_val = result["p_values"][feature]
            significance = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            summary_lines.append(f"{i:2d}. {feature:<20} {corr:>8.4f} {significance}")

        if result["significant_features"]:
            summary_lines.append(f"\n유의한 특성들 (p < 0.05):")
            summary_lines.append("-" * 40)
            for feature in result["significant_features"]:
                corr = result["correlations"][feature]
                p_val = result["p_values"][feature]
                summary_lines.append(f"  {feature:<20} {corr:>8.4f} (p={p_val:.3e})")

        return "\n".join(summary_lines)

    def print_summary(self):
        """분석 결과 요약 출력"""
        print(self.get_summary())
