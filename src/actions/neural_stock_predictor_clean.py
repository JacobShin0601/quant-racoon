"""
신경망 기반 주식 예측 시스템 (정리된 버전)
- 동적 피처 생성 지원
- FeatureEngineeringPipeline 통합
- NaN 처리 개선
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler

# FeatureEngineeringPipeline 임포트
try:
    from .feature_engineering import FeatureEngineeringPipeline
except ImportError:
    try:
        sys.path.append(os.path.dirname(__file__))
        from feature_engineering import FeatureEngineeringPipeline
    except ImportError:
        FeatureEngineeringPipeline = None
        print(
            "FeatureEngineeringPipeline을 임포트할 수 없습니다. 기본 피처 생성 방식을 사용합니다."
        )

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class SimpleStockPredictor(nn.Module):
    """단순한 주식 예측 신경망"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [32, 16],
        dropout_rate: float = 0.2,
        output_size: int = 4,
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class StockDataset(Dataset):
    """주식 데이터셋"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class StockPredictionNetwork:
    """신경망 기반 주식 예측 시스템"""

    def __init__(self, config: Dict):
        self.config = config
        self.neural_config = config.get("neural_network", {})

        # 앙상블 설정
        self.ensemble_config = self.neural_config.get(
            "ensemble",
            {
                "universal_weight": 0.7,
                "individual_weight": 0.3,
                "enable_individual_models": True,
            },
        )

        # 모델들
        self.universal_model = None
        self.individual_models = {}
        self.universal_scaler = StandardScaler()
        self.individual_scalers = {}

        # 피처 정보
        self.feature_names = None
        self.target_columns = None

        # FeatureEngineeringPipeline 초기화
        self.feature_pipeline = (
            FeatureEngineeringPipeline(config) if FeatureEngineeringPipeline else None
        )

        # 피처 정보 저장
        self.feature_info = {
            "universal_features": {},
            "individual_features": {},
            "macro_features": {},
            "feature_dimensions": {},
            "created_at": datetime.now().isoformat(),
        }

        # 앙상블 가중치
        self.universal_weight = self.ensemble_config.get("universal_weight", 0.7)
        self.individual_weight = self.ensemble_config.get("individual_weight", 0.3)
        self.enable_individual_models = self.ensemble_config.get(
            "enable_individual_models", True
        )

        logger.info(
            f"StockPredictionNetwork 초기화 완료 - 앙상블 모드 (Universal: {self.universal_weight}, Individual: {self.individual_weight})"
        )

    def _build_model(self, input_dim: int, output_size: int = 4) -> nn.Module:
        """신경망 모델 구축"""
        architecture = self.neural_config.get("architecture", {})
        hidden_sizes = architecture.get("hidden_layers", [32, 16])
        dropout_rate = architecture.get("dropout_rate", 0.2)

        model = SimpleStockPredictor(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            output_size=output_size,
        )

        return model

    def create_features(
        self,
        stock_data: pd.DataFrame,
        symbol: str,
        market_regime: Dict,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """동적 피처 생성"""
        try:
            logger.info(f"🔍 {symbol} 동적 피처 생성 시작...")
            logger.info(f"   - 주식 데이터: {stock_data.shape}")
            logger.info(f"   - 사용 가능한 컬럼: {len(stock_data.columns)}개")

            # FeatureEngineeringPipeline 사용
            if self.feature_pipeline:
                logger.info(f"   📊 FeatureEngineeringPipeline 사용...")
                try:
                    all_features, feature_metadata = (
                        self.feature_pipeline.create_features(
                            stock_data,
                            symbol,
                            market_regime,
                            macro_data,
                            mode="individual",
                        )
                    )

                    # 피처 정보 저장
                    self.feature_info["individual_features"][symbol] = {
                        "total_features": len(all_features.columns),
                        "feature_names": list(all_features.columns),
                        "feature_stats": feature_metadata.get("feature_stats", {}),
                        "created_at": datetime.now().isoformat(),
                    }

                    logger.info(f"   ✅ 동적 피처 생성 완료: {all_features.shape}")
                    logger.info(
                        f"   📊 피처 통계: {feature_metadata.get('feature_stats', {})}"
                    )

                    return all_features

                except Exception as e:
                    logger.error(f"   ❌ FeatureEngineeringPipeline 실패: {e}")

            # 기본 피처 생성 (호환성)
            logger.info(f"   📊 기본 피처 생성...")
            features = pd.DataFrame(index=stock_data.index)

            # 기본 기술적 지표들
            if all(
                col in stock_data.columns
                for col in ["open", "high", "low", "close", "volume"]
            ):
                features["dual_momentum"] = stock_data["close"].pct_change(
                    5
                ) - stock_data["close"].pct_change(20)
                features["volatility_breakout"] = (
                    stock_data["close"] - stock_data["close"].rolling(20).mean()
                ) / stock_data["close"].rolling(20).std()
                features["swing_ema"] = (
                    stock_data["close"].ewm(span=12).mean()
                    - stock_data["close"].ewm(span=26).mean()
                ) / stock_data["close"].ewm(span=26).mean()

            # 시장 체제 피처
            regimes = ["BULLISH", "BEARISH", "SIDEWAYS", "VOLATILE"]
            current_regime = market_regime.get("current_regime", "SIDEWAYS")

            for regime in regimes:
                features[f"regime_{regime.lower()}"] = int(current_regime == regime)

            # NaN 처리
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

            logger.info(f"   ✅ 기본 피처 생성 완료: {features.shape}")
            return features

        except Exception as e:
            logger.error(f"❌ {symbol} 피처 생성 실패: {e}")
            return pd.DataFrame(index=stock_data.index)

    def prepare_training_data(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, pd.DataFrame, np.ndarray],
        lookback_days: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 준비"""
        try:
            # 피처 정규화
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # 시퀀스 데이터 생성
            X, y = [], []

            for i in range(lookback_days, len(features_scaled)):
                X.append(features_scaled[i - lookback_days : i])
                if isinstance(target, pd.DataFrame):
                    y.append(target.iloc[i].values)
                else:
                    y.append(target[i])

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"훈련 데이터 준비 실패: {e}")
            return np.array([]), np.array([])

    def _train_universal_model(self, training_data: Dict) -> bool:
        """Universal 모델 훈련"""
        try:
            logger.info("🧠 Universal 모델 훈련 시작...")

            # 모든 종목의 피처 통합
            all_features = []
            all_targets = []

            for symbol, data in training_data.items():
                features = data["features"]
                targets = data["targets"]

                if features is not None and targets is not None:
                    all_features.append(features)
                    all_targets.append(targets)

            if not all_features:
                logger.error("훈련할 피처가 없습니다.")
                return False

            # 피처 통합
            combined_features = pd.concat(all_features, axis=0)
            combined_targets = pd.concat(all_targets, axis=0)

            # 피처 정규화
            self.universal_scaler.fit(combined_features)
            features_scaled = self.universal_scaler.transform(combined_features)

            # 모델 구축
            input_dim = features_scaled.shape[1]
            output_size = (
                combined_targets.shape[1] if len(combined_targets.shape) > 1 else 1
            )

            self.universal_model = self._build_model(input_dim, output_size)
            self.feature_names = list(combined_features.columns)
            self.target_columns = (
                list(combined_targets.columns)
                if hasattr(combined_targets, "columns")
                else None
            )

            # 훈련 설정
            epochs = self.neural_config.get("epochs", 200)
            batch_size = self.neural_config.get("batch_size", 32)
            learning_rate = self.neural_config.get("learning_rate", 0.001)
            early_stopping_patience = self.neural_config.get(
                "early_stopping_patience", 20
            )

            # 데이터셋 생성
            dataset = StockDataset(features_scaled, combined_targets.values)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 옵티마이저 및 손실 함수
            optimizer = optim.Adam(self.universal_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # 훈련 루프
            best_loss = float("inf")
            patience_counter = 0
            actual_epochs = 0

            for epoch in range(epochs):
                self.universal_model.train()
                total_loss = 0

                for batch_features, batch_targets in dataloader:
                    optimizer.zero_grad()
                    predictions = self.universal_model(batch_features)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                actual_epochs = epoch + 1

            logger.info(
                f"✅ Universal 모델 훈련 완료: {actual_epochs} epochs, 최종 손실: {best_loss:.6f}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Universal 모델 훈련 실패: {e}")
            return False

    def _train_individual_models(self, training_data: Dict) -> bool:
        """개별 모델 훈련"""
        try:
            logger.info("🧠 개별 모델 훈련 시작...")

            from tqdm import tqdm

            symbol_iter = tqdm(training_data.items(), desc="개별 모델 훈련")

            for symbol, data in symbol_iter:
                features = data["features"]
                targets = data["targets"]

                if features is None or targets is None:
                    continue

                try:
                    # 피처 정규화
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)

                    # 모델 구축
                    input_dim = features_scaled.shape[1]
                    output_size = targets.shape[1] if len(targets.shape) > 1 else 1

                    model = self._build_model(input_dim, output_size)

                    # 훈련 설정
                    epochs = self.neural_config.get("epochs", 200)
                    batch_size = self.neural_config.get("batch_size", 32)
                    learning_rate = self.neural_config.get("learning_rate", 0.001)
                    early_stopping_patience = self.neural_config.get(
                        "early_stopping_patience", 20
                    )

                    # 데이터셋 생성
                    dataset = StockDataset(features_scaled, targets.values)
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True
                    )

                    # 옵티마이저 및 손실 함수
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()

                    # 훈련 루프
                    best_loss = float("inf")
                    patience_counter = 0
                    actual_epochs = 0

                    for epoch in range(epochs):
                        model.train()
                        total_loss = 0

                        for batch_features, batch_targets in dataloader:
                            optimizer.zero_grad()
                            predictions = model(batch_features)
                            loss = criterion(predictions, batch_targets)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        avg_loss = total_loss / len(dataloader)

                        # Early stopping
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= early_stopping_patience:
                            break

                        actual_epochs = epoch + 1

                    # 모델 저장
                    self.individual_models[symbol] = model
                    self.individual_scalers[symbol] = scaler

                    symbol_iter.set_postfix(
                        {
                            "symbol": symbol,
                            "final_train": f"{best_loss:.6f}",
                            "epochs": actual_epochs,
                        }
                    )

                except Exception as e:
                    logger.error(f"❌ {symbol} 개별 모델 훈련 실패: {e}")
                    continue

            logger.info(f"✅ 개별 모델 훈련 완료: {len(self.individual_models)}개 모델")
            return True

        except Exception as e:
            logger.error(f"❌ 개별 모델 훈련 실패: {e}")
            return False

    def fit(self, training_data: Dict) -> bool:
        """모델 훈련"""
        try:
            logger.info("🚀 신경망 모델 훈련 시작...")

            # Universal 모델 훈련
            if not self._train_universal_model(training_data):
                return False

            # 개별 모델 훈련
            if self.enable_individual_models:
                if not self._train_individual_models(training_data):
                    logger.warning("개별 모델 훈련 실패, Universal 모델만 사용")

            logger.info("✅ 신경망 모델 훈련 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 모델 훈련 실패: {e}")
            return False

    def predict(
        self, features: pd.DataFrame, symbol: str
    ) -> Union[float, Dict[str, float]]:
        """예측"""
        try:
            if self.universal_model is None:
                logger.error("모델이 훈련되지 않았습니다.")
                return 0.0

            # 피처 정규화
            features_scaled = self.universal_scaler.transform(features)

            # 예측
            self.universal_model.eval()
            with torch.no_grad():
                predictions = self.universal_model(torch.FloatTensor(features_scaled))
                latest_pred = predictions[-1].numpy()

            # 결과 반환
            if self.target_columns:
                return {
                    col: float(val)
                    for col, val in zip(self.target_columns, latest_pred)
                }
            else:
                return {f"target_{i}": float(val) for i, val in enumerate(latest_pred)}

        except Exception as e:
            logger.error(f"❌ 예측 실패: {e}")
            return 0.0

    def save_model(self, filepath: str) -> bool:
        """모델 저장"""
        try:
            # Universal 모델 저장
            torch.save(
                self.universal_model.state_dict(), f"{filepath}_pytorch_universal.pth"
            )

            # 피처 정보 저장
            if self.feature_info:
                feature_info_path = f"{filepath}_feature_info.json"
                with open(feature_info_path, "w", encoding="utf-8") as f:
                    json.dump(self.feature_info, f, indent=2, ensure_ascii=False)
                logger.info(f"피처 정보 저장 완료: {feature_info_path}")

            # 개별 모델 저장
            for symbol, model in self.individual_models.items():
                torch.save(
                    model.state_dict(), f"{filepath}_pytorch_individual_{symbol}.pth"
                )

            # 메타데이터 저장
            model_data = {
                "feature_names": self.feature_names,
                "target_columns": self.target_columns,
                "config": self.config,
                "is_fitted": self.enable_individual_models,
                "feature_info": self.feature_info,
            }

            joblib.dump(model_data, f"{filepath}_meta.pkl")
            logger.info(f"✅ 모델 저장 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ 모델 저장 실패: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """모델 로드"""
        try:
            # 메타데이터 로드
            model_data = joblib.load(f"{filepath}_meta.pkl")
            self.feature_names = model_data["feature_names"]
            self.target_columns = model_data["target_columns"]
            self.enable_individual_models = model_data["is_fitted"]

            # 피처 정보 로드
            if "feature_info" in model_data:
                self.feature_info = model_data["feature_info"]
                logger.info("피처 정보 로드 완료")

            # PyTorch 모델 로드
            if self.enable_individual_models:
                # Universal 모델 로드
                universal_state_dict = torch.load(
                    f"{filepath}_pytorch_universal.pth",
                    map_location=torch.device("cpu"),
                )
                actual_input_dim = universal_state_dict["network.0.weight"].shape[1]
                actual_output_dim = universal_state_dict["network.6.weight"].shape[0]

                self.universal_model = self._build_model(
                    actual_input_dim, actual_output_dim
                )
                self.universal_model.load_state_dict(universal_state_dict)

                # 개별 모델 로드
                for symbol in self.individual_scalers.keys():
                    individual_state_dict = torch.load(
                        f"{filepath}_pytorch_individual_{symbol}.pth",
                        map_location=torch.device("cpu"),
                    )
                    model = self._build_model(actual_input_dim, actual_output_dim)
                    model.load_state_dict(individual_state_dict)
                    self.individual_models[symbol] = model

            logger.info(f"✅ 모델 로드 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            return False


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="신경망 기반 주식 예측 시스템")
    parser.add_argument("--train", action="store_true", help="모델 훈련")
    parser.add_argument("--force", action="store_true", help="강제 재훈련")
    parser.add_argument("--data-dir", default="data/trader", help="데이터 디렉토리")

    args = parser.parse_args()

    # 설정 로드
    config_path = "config/config_trader.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "neural_network": {
                "epochs": 200,
                "batch_size": 32,
                "learning_rate": 0.001,
                "early_stopping_patience": 20,
                "architecture": {"hidden_layers": [32, 16], "dropout_rate": 0.2},
                "ensemble": {
                    "universal_weight": 0.7,
                    "individual_weight": 0.3,
                    "enable_individual_models": True,
                },
            }
        }

    # 모델 초기화
    model = StockPredictionNetwork(config)

    if args.train:
        # 데이터 로드 및 훈련
        training_data = {}

        # 데이터 파일들 로드
        data_files = [f for f in os.listdir(args.data_dir) if f.endswith(".csv")]

        for file in data_files:
            symbol = file.split("_")[0]
            filepath = os.path.join(args.data_dir, file)

            try:
                df = pd.read_csv(filepath)
                df = df.fillna(method="ffill").fillna(method="bfill")

                # 피처 생성
                market_regime = {"current_regime": "SIDEWAYS"}
                features = model.create_features(df, symbol, market_regime)

                # 타겟 생성 (22일, 66일 수익률 + 표준편차)
                close = df["close"]
                target_22d = close.pct_change(22).shift(-22)
                target_66d = close.pct_change(66).shift(-66)

                # 표준편차 계산
                rolling_std_22d = (
                    close.pct_change(22)
                    .rolling(window=22, min_periods=1)
                    .std()
                    .shift(-22)
                )
                rolling_std_66d = (
                    close.pct_change(66)
                    .rolling(window=66, min_periods=1)
                    .std()
                    .shift(-66)
                )

                # NaN 처리
                target_22d = target_22d.fillna(method="ffill").fillna(method="bfill")
                target_66d = target_66d.fillna(method="ffill").fillna(method="bfill")
                rolling_std_22d = rolling_std_22d.fillna(method="ffill").fillna(
                    method="bfill"
                )
                rolling_std_66d = rolling_std_66d.fillna(method="ffill").fillna(
                    method="bfill"
                )

                targets = pd.DataFrame(
                    {
                        "target_22d": target_22d,
                        "sigma_22d": rolling_std_22d,
                        "target_66d": target_66d,
                        "sigma_66d": rolling_std_66d,
                    }
                )

                training_data[symbol] = {"features": features, "targets": targets}

                print(f"✅ {symbol} 데이터 로드 완료: {len(df)}행")
                print(
                    f"   📊 피처 차원: {features.shape if features is not None else 'None'}"
                )

            except Exception as e:
                print(f"❌ {symbol} 데이터 로드 실패: {e}")

        print(f"📊 총 {len(training_data)}개 종목 데이터 로드 완료")

        # 모델 훈련
        if model.fit(training_data):
            # 모델 저장
            model_path = "models/trader/neural_predictor"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_model(model_path)

            # 예측 테스트
            if training_data:
                symbol = list(training_data.keys())[0]
                features = training_data[symbol]["features"]
                prediction = model.predict(features.tail(1), symbol)
                print(f"📈 {symbol} 예측 결과: {prediction}")

    else:
        # 모델 로드 및 예측
        model_path = "models/trader/neural_predictor"
        if model.load_model(model_path):
            print("✅ 모델 로드 완료")
        else:
            print("❌ 모델 로드 실패")


if __name__ == "__main__":
    main()
