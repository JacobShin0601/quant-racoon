#!/usr/bin/env python3
"""
다층 퍼셉트론(MLP) 분석 모듈
PyTorch를 활용한 신경망 모델 구축 및 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

warnings.filterwarnings("ignore")


class MLP:
    """다층 퍼셉트론 모델 (PyTorch 없을 때 대체 클래스)"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        dropout_rate: float = 0.2,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLP model")

        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # 히든 레이어들
        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        # 출력 레이어
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPAnalyzer:
    """MLP 분석 클래스"""

    def __init__(self):
        self.results = {}
        self.model = None
        self.scaler = StandardScaler()

        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. MLP analysis will be skipped.")
            self.available = False
        else:
            self.available = True

    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str = "return",
        feature_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        hidden_sizes: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = None,
    ) -> Dict[str, Any]:
        """
        MLP 분석 실행

        Args:
            data: 분석할 데이터프레임
            target_column: 종속변수 컬럼명
            feature_columns: 독립변수 컬럼명 리스트
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
            hidden_sizes: 히든 레이어 크기 리스트
            dropout_rate: 드롭아웃 비율
            learning_rate: 학습률
            batch_size: 배치 크기
            epochs: 에포크 수
            early_stopping_patience: 조기 종료 인내심
            device: 사용할 디바이스 ('cpu' 또는 'cuda')

        Returns:
            분석 결과 딕셔너리
        """
        if not self.available:
            return {
                "model_type": "Multi-Layer Perceptron (PyTorch) - SKIPPED",
                "error": "PyTorch not available",
                "r_squared": 0.0,
                "train_r_squared": 0.0,
                "test_r_squared": 0.0,
                "train_rmse": 0.0,
                "test_rmse": 0.0,
                "train_mae": 0.0,
                "test_mae": 0.0,
            }

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

        # 디바이스 설정
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

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

        # PyTorch 텐서로 변환
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

        # 데이터 로더 생성
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 모델 초기화
        model = MLP(
            input_size=len(feature_columns),
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
        ).to(device)

        # 손실 함수와 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 학습 과정
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # 훈련
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 검증
            model.eval()
            with torch.no_grad():
                X_test_device = X_test_tensor.to(device)
                y_test_device = y_test_tensor.to(device)
                val_outputs = model(X_test_device)
                val_loss = criterion(val_outputs, y_test_device).item()
                val_losses.append(val_loss)

            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 최적 모델 로드
        model.load_state_dict(best_model_state)

        # 예측
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor.to(device)).cpu().numpy().flatten()
            y_pred_test = model(X_test_tensor.to(device)).cpu().numpy().flatten()

        # 성능 지표 계산
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # 결과 구성
        result = {
            "model_type": "Multi-Layer Perceptron (PyTorch)",
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
            # 모델 파라미터
            "hidden_sizes": hidden_sizes,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "actual_epochs": len(train_losses),
            "device": device,
            # 모델 객체
            "model": model,
            "scaler": self.scaler,
            # 예측값
            "y_train": y_train.values,
            "y_test": y_test.values,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            # 학습 과정
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

        self.results = result
        self.model = model
        return result

    def plot_training_history(
        self, figsize: Tuple[int, int] = (12, 8), save_path: str = None
    ) -> plt.Figure:
        """학습 과정 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        epochs = range(1, len(result["train_losses"]) + 1)

        # 1. 손실 함수
        ax1.plot(epochs, result["train_losses"], "b-", label="훈련 손실", alpha=0.7)
        ax1.plot(epochs, result["val_losses"], "r-", label="검증 손실", alpha=0.7)
        ax1.set_xlabel("에포크")
        ax1.set_ylabel("손실 (MSE)")
        ax1.set_title("학습 과정")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 실제값 vs 예측값
        ax2.scatter(result["y_test"], result["y_pred_test"], alpha=0.6, s=20)
        ax2.plot(
            [result["y_test"].min(), result["y_test"].max()],
            [result["y_test"].min(), result["y_test"].max()],
            "r--",
            alpha=0.8,
        )
        ax2.set_xlabel("실제값")
        ax2.set_ylabel("예측값")
        ax2.set_title(f'실제값 vs 예측값 (R² = {result["r_squared"]:.4f})')
        ax2.grid(True, alpha=0.3)

        # 3. 잔차 플롯
        residuals = result["y_test"] - result["y_pred_test"]
        ax3.scatter(result["y_pred_test"], residuals, alpha=0.6, s=20)
        ax3.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax3.set_xlabel("예측값")
        ax3.set_ylabel("잔차")
        ax3.set_title("잔차 vs 예측값")
        ax3.grid(True, alpha=0.3)

        # 4. 잔차 분포
        ax4.hist(residuals, bins=30, alpha=0.7, color="green", edgecolor="black")
        ax4.set_xlabel("잔차")
        ax4.set_ylabel("빈도")
        ax4.set_title("잔차 분포")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"학습 과정 플롯 저장: {save_path}")

        return fig

    def plot_model_architecture(
        self, figsize: Tuple[int, int] = (10, 8), save_path: str = None
    ) -> plt.Figure:
        """모델 아키텍처 플롯"""
        if not self.results:
            raise ValueError("먼저 분석을 실행해주세요.")

        result = self.results

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. 모델 구조 시각화
        layer_sizes = [result["n_features"]] + result["hidden_sizes"] + [1]
        layer_names = (
            ["입력"]
            + [f"히든 {i+1}" for i in range(len(result["hidden_sizes"]))]
            + ["출력"]
        )

        y_positions = np.linspace(0, 1, len(layer_sizes))

        for i, (size, name, y_pos) in enumerate(
            zip(layer_sizes, layer_names, y_positions)
        ):
            ax1.text(
                0.2,
                y_pos,
                f"{name}\n({size}개)",
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            )

        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_title("모델 아키텍처")
        ax1.axis("off")

        # 2. 훈련 vs 테스트 성능 비교
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

        bars1 = ax2.bar(x - width / 2, train_scores, width, label="훈련", alpha=0.7)
        bars2 = ax2.bar(x + width / 2, test_scores, width, label="테스트", alpha=0.7)

        ax2.set_xlabel("성능 지표")
        ax2.set_ylabel("점수")
        ax2.set_title("훈련 vs 테스트 성능 비교")
        ax2.set_xticks(x)
        ax2.set_xticklabels(performance_metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # 3. 모델 파라미터 요약
        param_info = [
            f"입력 특성: {result['n_features']}개",
            f"히든 레이어: {result['hidden_sizes']}",
            f"드롭아웃: {result['dropout_rate']}",
            f"학습률: {result['learning_rate']}",
            f"배치 크기: {result['batch_size']}",
            f"에포크: {result['actual_epochs']}/{result['epochs']}",
            f"디바이스: {result['device']}",
        ]

        ax3.text(
            0.1,
            0.9,
            "\n".join(param_info),
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
        )
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title("모델 파라미터")
        ax3.axis("off")

        # 4. 손실 함수 로그 스케일
        epochs = range(1, len(result["train_losses"]) + 1)
        ax4.semilogy(epochs, result["train_losses"], "b-", label="훈련 손실", alpha=0.7)
        ax4.semilogy(epochs, result["val_losses"], "r-", label="검증 손실", alpha=0.7)
        ax4.set_xlabel("에포크")
        ax4.set_ylabel("손실 (MSE) - 로그 스케일")
        ax4.set_title("학습 과정 (로그 스케일)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"모델 아키텍처 플롯 저장: {save_path}")

        return fig

    def get_summary(self) -> str:
        """분석 결과 요약 문자열 반환"""
        if not self.results:
            return "분석 결과가 없습니다."

        result = self.results
        summary_lines = []

        summary_lines.append("=" * 60)
        summary_lines.append("MLP 분석 결과 요약")
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
        summary_lines.append(f"RMSE (테스트): {result['test_rmse']:.4f}")
        summary_lines.append(f"MAE (테스트): {result['test_mae']:.4f}")

        summary_lines.append(f"\n모델 구조:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"입력 레이어: {result['n_features']}개 특성")
        for i, size in enumerate(result["hidden_sizes"]):
            summary_lines.append(f"히든 레이어 {i+1}: {size}개 뉴런")
        summary_lines.append(f"출력 레이어: 1개 뉴런")
        summary_lines.append(f"드롭아웃 비율: {result['dropout_rate']}")

        summary_lines.append(f"\n학습 파라미터:")
        summary_lines.append("-" * 40)
        summary_lines.append(f"학습률: {result['learning_rate']}")
        summary_lines.append(f"배치 크기: {result['batch_size']}")
        summary_lines.append(f"에포크: {result['actual_epochs']}/{result['epochs']}")
        summary_lines.append(f"디바이스: {result['device']}")
        summary_lines.append(f"최종 검증 손실: {result['best_val_loss']:.6f}")

        return "\n".join(summary_lines)

    def print_summary(self):
        """분석 결과 요약 출력"""
        print(self.get_summary())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """새로운 데이터에 대한 예측"""
        if self.model is None:
            raise ValueError("먼저 모델을 학습해주세요.")

        # 특성 스케일링
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        # 예측
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        return predictions
