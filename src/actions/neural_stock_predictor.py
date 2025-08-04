"""
신경망 기반 주식 예측 시스템
- 피처 엔지니어링은 feature_engineering.py 사용
- 신경망 모델 학습 및 예측에 집중
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import warnings

# tqdm import (진행바 표시용)
try:
    from tqdm import trange
except ImportError:
    # tqdm이 없으면 기본 range 사용
    trange = range

# feature_engineering.py에서 피처 엔지니어링 기능 import
try:
    from .feature_engineering import FeatureEngineeringPipeline
except ImportError:
    # 상대 임포트가 실패하면 절대 경로로 시도
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    try:
        from feature_engineering import FeatureEngineeringPipeline
    except ImportError:
        FeatureEngineeringPipeline = None
        logger.warning(
            "FeatureEngineeringPipeline을 임포트할 수 없습니다. 기본 피처 생성 방식을 사용합니다."
        )

# FeatureEngineeringPipeline import 시도
try:
    from ..agent.enhancements.feature_engineering_pipeline import (
        FeatureEngineeringPipeline,
    )
except ImportError:
    FeatureEngineeringPipeline = None

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class EnsembleWeightLearner(nn.Module):
    """
    앙상블 가중치를 학습하는 메타-학습 신경망

    입력: 모델별 예측값, 시장 상황, 종목 특성
    출력: 최적 앙상블 가중치
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32, 16]):
        super(EnsembleWeightLearner, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)]
            )
            prev_size = hidden_size

        # 출력층: 2개 가중치 (universal, individual)
        layers.append(nn.Linear(prev_size, 2))
        layers.append(nn.Softmax(dim=1))  # 가중치 합이 1이 되도록

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EnsembleWeightDataset(Dataset):
    """
    앙상블 가중치 학습을 위한 데이터셋
    """

    def __init__(self, ensemble_inputs: np.ndarray, optimal_weights: np.ndarray):
        self.ensemble_inputs = torch.FloatTensor(ensemble_inputs)
        self.optimal_weights = torch.FloatTensor(optimal_weights)

    def __len__(self):
        return len(self.ensemble_inputs)

    def __getitem__(self, idx):
        return self.ensemble_inputs[idx], self.optimal_weights[idx]


class SimpleStockPredictor(nn.Module):
    """
    간단한 주식 예측 신경망
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [32, 16],
        dropout_rate: float = 0.2,
        output_size: int = 1,  # 멀티타겟을 위해 출력 크기 파라미터 추가
    ):
        super(SimpleStockPredictor, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        # 출력층을 동적으로 설정
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class StockDataset(Dataset):
    """
    주식 데이터를 위한 PyTorch Dataset
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class StockPredictionNetwork:
    """
    신경망 기반 개별 종목 예측기

    Features:
    - 18개 기존 스윙 전략 신호
    - 시장 체제 정보 (HMM 결과)
    - 매크로 환경 피처
    - 기술적 지표들
    """

    def __init__(self, config: Dict):
        """
        신경망 기반 주식 예측 시스템 (앙상블 방식)

        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.neural_config = config.get("neural_network", {})

        # Train-test 분할 설정
        self.train_ratio = self.neural_config.get("train_ratio", 0.8)
        logger.info(f"📊 Train-test 분할 비율: {self.train_ratio:.1%}")

        # 앙상블 설정
        self.ensemble_config = self.neural_config.get(
            "ensemble",
            {
                "universal_weight": 0.7,
                "individual_weight": 0.3,
                "enable_individual_models": True,
                "enable_weight_learning": True,  # 가중치 학습 활성화
                "weight_learning_config": {
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "batch_size": 32,
                    "validation_split": 0.2,
                    "min_samples_for_weight_learning": 50,
                },
            },
        )

        # 모델들
        self.universal_model = None  # 통합 모델
        self.individual_models = {}  # 종목별 모델들
        self.universal_scaler = StandardScaler()
        self.individual_scalers = {}

        # 앙상블 가중치 학습기
        self.weight_learner = None
        self.weight_learner_scaler = StandardScaler()
        self.enable_weight_learning = self.ensemble_config.get(
            "enable_weight_learning", True
        )
        self.weight_learning_config = self.ensemble_config.get(
            "weight_learning_config", {}
        )

        # 피처 정보
        self.feature_names = None
        self.target_columns = None

        # 피처 엔지니어링 파이프라인 초기화
        if FeatureEngineeringPipeline:
            try:
                self.feature_pipeline = FeatureEngineeringPipeline(config)
                logger.info("✅ FeatureEngineeringPipeline 초기화 완료")
            except Exception as e:
                logger.warning(f"FeatureEngineeringPipeline 초기화 실패: {e}")
                self.feature_pipeline = None
        else:
            self.feature_pipeline = None
            logger.info("ℹ️ 기본 피처 생성 방식 사용")

        # 피처 정보 저장
        self.feature_info = {
            "universal_features": {},
            "individual_features": {},
            "macro_features": {},
            "feature_dimensions": {},
            "created_at": datetime.now().isoformat(),
        }

        # 앙상블 가중치 (기본값, 학습 후 업데이트됨)
        self.universal_weight = self.ensemble_config.get("universal_weight", 0.7)
        self.individual_weight = self.ensemble_config.get("individual_weight", 0.3)
        self.enable_individual_models = self.ensemble_config.get(
            "enable_individual_models", True
        )

        # 가중치 학습 데이터 저장
        self.weight_training_data = {
            "ensemble_inputs": [],
            "optimal_weights": [],
            "performance_metrics": [],
        }

        logger.info(
            f"StockPredictionNetwork 초기화 완료 - 앙상블 모드 (Universal: {self.universal_weight}, Individual: {self.individual_weight})"
        )
        if self.enable_weight_learning:
            logger.info("✅ 앙상블 가중치 학습 활성화")

    def _build_model(self, input_dim: int, output_size: int = 1) -> nn.Module:
        """
        신경망 모델 구축

        Args:
            input_dim: 입력 차원
            output_size: 출력 차원 (멀티타겟 개수)

        Returns:
            PyTorch 모델
        """
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

    def extract_swing_features(
        self, stock_data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """
        기존 스윙 전략 피처 추출 (호환성 유지)

        Args:
            stock_data: 주식 데이터 (OHLCV)
            symbol: 종목 코드

        Returns:
            스윙 전략 피처 데이터프레임
        """
        # feature_engineering.py 사용
        if self.feature_pipeline:
            try:
                features, _ = self.feature_pipeline.create_dynamic_features(
                    stock_data, symbol, {}, None, "individual"
                )
                return features
            except Exception as e:
                logger.warning(f"FeatureEngineeringPipeline 사용 실패: {e}")

        # 기본 피처 생성 (fallback)
        return self._create_basic_swing_features(stock_data, symbol)

    def _create_basic_swing_features(
        self, stock_data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """기본 스윙 피처 생성 (fallback)"""
        features = pd.DataFrame(index=stock_data.index)

        try:
            # 기본 가격 데이터 확인
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in stock_data.columns:
                    logger.warning(f"{symbol}: {col} 컬럼 없음")
                    stock_data[col] = stock_data.get("close", 100)  # 기본값

            # 확장된 기본 피처들 생성
            features["dual_momentum"] = self._calculate_dual_momentum(stock_data)
            features["volatility_breakout"] = self._calculate_volatility_breakout(stock_data)
            features["swing_ema"] = self._calculate_swing_ema(stock_data)
            features["swing_rsi"] = self._calculate_swing_rsi(stock_data)
            
            # 추가 기술적 지표 피처들
            features["swing_donchian"] = self._calculate_swing_donchian(stock_data)
            features["stoch_donchian"] = self._calculate_stoch_donchian(stock_data)
            features["whipsaw_prevention"] = self._calculate_whipsaw_prevention(stock_data)
            features["donchian_rsi_whipsaw"] = self._calculate_donchian_rsi_whipsaw(stock_data)
            features["volatility_filtered_breakout"] = self._calculate_volatility_filtered_breakout(stock_data)
            features["multi_timeframe_whipsaw"] = self._calculate_multi_timeframe_whipsaw(stock_data)
            features["adaptive_whipsaw"] = self._calculate_adaptive_whipsaw(stock_data)
            features["cci_bollinger"] = self._calculate_cci_bollinger(stock_data)
            features["mean_reversion"] = self._calculate_mean_reversion(stock_data)
            features["swing_breakout"] = self._calculate_swing_breakout(stock_data)
            features["swing_pullback_entry"] = self._calculate_swing_pullback_entry(stock_data)
            features["swing_candle_pattern"] = self._calculate_swing_candle_pattern(stock_data)
            features["swing_bollinger_band"] = self._calculate_swing_bollinger_band(stock_data)
            features["swing_macd"] = self._calculate_swing_macd(stock_data)
            
            # 추가 가격 기반 피처들
            features["atr_normalized"] = self._calculate_atr(stock_data) / stock_data["close"]
            features["rsi_14"] = self._calculate_rsi(stock_data, 14) / 100 - 0.5  # -0.5 ~ 0.5 정규화
            features["macd_signal"] = self._calculate_macd(stock_data) / stock_data["close"]
            
            # 새로운 고급 피처들
            features["volume_price_trend"] = self._calculate_volume_price_trend(stock_data)
            features["price_volume_oscillator"] = self._calculate_price_volume_oscillator(stock_data)
            features["trend_strength"] = self._calculate_trend_strength(stock_data)
            features["volatility_regime"] = self._calculate_volatility_regime(stock_data)
            features["support_resistance"] = self._calculate_support_resistance_strength(stock_data)
            features["momentum_divergence"] = self._calculate_momentum_divergence(stock_data)
            features["volatility_skew"] = self._calculate_volatility_skew(stock_data)
            features["market_microstructure"] = self._calculate_market_microstructure(stock_data)

            # NaN 처리
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

        except Exception as e:
            logger.error(f"기본 스윙 피처 생성 실패: {e}")

        return features

    # 기본 기술적 지표 계산 메서드들 (feature_engineering.py와 중복되지 않는 것들만 유지)
    def _calculate_dual_momentum(self, data: pd.DataFrame) -> pd.Series:
        """듀얼 모멘텀 계산"""
        close = data["close"]
        momentum_short = close.pct_change(5)
        momentum_long = close.pct_change(20)
        return momentum_short - momentum_long

    def _calculate_volatility_breakout(self, data: pd.DataFrame) -> pd.Series:
        """변동성 브레이크아웃 계산"""
        close = data["close"]
        volatility = close.pct_change().rolling(20).std()
        return (close - close.rolling(20).mean()) / (
            volatility * close.rolling(20).mean()
        )

    def _calculate_swing_ema(self, data: pd.DataFrame) -> pd.Series:
        """스윙 EMA 계산"""
        close = data["close"]
        ema_short = close.ewm(span=12).mean()
        ema_long = close.ewm(span=26).mean()
        return (ema_short - ema_long) / ema_long

    def _calculate_swing_rsi(self, data: pd.DataFrame) -> pd.Series:
        """스윙 RSI 계산"""
        close = data["close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50  # -1 ~ 1 정규화

    def _calculate_swing_donchian(self, data: pd.DataFrame) -> pd.Series:
        """Swing Donchian 계산"""
        try:
            period = 20
            donchian_high = data["high"].rolling(period).max()
            donchian_low = data["low"].rolling(period).min()
            donchian_mid = (donchian_high + donchian_low) / 2
            return (data["close"] - donchian_mid) / (donchian_high - donchian_low)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_stoch_donchian(self, data: pd.DataFrame) -> pd.Series:
        """Stochastic Donchian 계산"""
        try:
            period = 14
            lowest_low = data["low"].rolling(period).min()
            highest_high = data["high"].rolling(period).max()
            stoch = (data["close"] - lowest_low) / (highest_high - lowest_low)
            return (stoch - 0.5) * 2  # -1 ~ 1 정규화
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_whipsaw_prevention(self, data: pd.DataFrame) -> pd.Series:
        """Whipsaw Prevention 계산"""
        try:
            ma = data["close"].rolling(20).mean()
            volatility = data["close"].rolling(20).std()
            signal = np.where(
                data["close"] > ma + volatility,
                1,
                np.where(data["close"] < ma - volatility, -1, 0),
            )
            return pd.Series(signal, index=data.index)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_donchian_rsi_whipsaw(self, data: pd.DataFrame) -> pd.Series:
        """Donchian RSI Whipsaw 계산"""
        try:
            donchian = self._calculate_swing_donchian(data)
            rsi = self._calculate_swing_rsi(data)
            return (donchian + rsi) / 2
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_volatility_filtered_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Volatility Filtered Breakout 계산"""
        try:
            volatility = data["close"].rolling(20).std()
            vol_threshold = volatility.rolling(60).quantile(0.3)  # 낮은 변동성 기준
            breakout = self._calculate_volatility_breakout(data)
            return np.where(volatility < vol_threshold, breakout, 0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_multi_timeframe_whipsaw(self, data: pd.DataFrame) -> pd.Series:
        """Multi Timeframe Whipsaw 계산"""
        try:
            short_trend = data["close"].rolling(5).mean()
            medium_trend = data["close"].rolling(20).mean()
            long_trend = data["close"].rolling(50).mean()

            short_signal = np.where(data["close"] > short_trend, 1, -1)
            medium_signal = np.where(short_trend > medium_trend, 1, -1)
            long_signal = np.where(medium_trend > long_trend, 1, -1)

            return (short_signal + medium_signal + long_signal) / 3
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_adaptive_whipsaw(self, data: pd.DataFrame) -> pd.Series:
        """Adaptive Whipsaw 계산"""
        try:
            volatility = data["close"].rolling(20).std()
            adaptive_period = np.clip(20 / (volatility + 0.01), 10, 50).astype(int)

            signals = []
            for i, period in enumerate(adaptive_period):
                if i < period:
                    signals.append(0)
                else:
                    ma = data["close"].iloc[i - period : i].mean()
                    signal = 1 if data["close"].iloc[i] > ma else -1
                    signals.append(signal)

            return pd.Series(signals, index=data.index)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_cci_bollinger(self, data: pd.DataFrame) -> pd.Series:
        """CCI Bollinger 계산"""
        try:
            # CCI 계산
            tp = (data["high"] + data["low"] + data["close"]) / 3
            ma = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (tp - ma) / (0.015 * mad)

            # 정규화 (-1 ~ 1)
            return np.clip(cci / 100, -1, 1)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_mean_reversion(self, data: pd.DataFrame) -> pd.Series:
        """Mean Reversion 계산"""
        try:
            ma = data["close"].rolling(20).mean()
            std = data["close"].rolling(20).std()
            z_score = (data["close"] - ma) / std
            return -z_score  # 평균 회귀 신호 (현재가가 높으면 음의 신호)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_swing_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Swing Breakout 계산"""
        try:
            period = 20
            resistance = data["high"].rolling(period).max()
            support = data["low"].rolling(period).min()

            breakout_up = data["close"] > resistance.shift(1)
            breakout_down = data["close"] < support.shift(1)

            return np.where(breakout_up, 1, np.where(breakout_down, -1, 0))
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_swing_pullback_entry(self, data: pd.DataFrame) -> pd.Series:
        """Swing Pullback Entry 계산"""
        try:
            ma = data["close"].rolling(50).mean()
            recent_high = data["high"].rolling(10).max()

            uptrend = data["close"] > ma
            pullback = (data["close"] / recent_high - 1) < -0.05  # 5% 하락

            return np.where(uptrend & pullback, 1, 0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_swing_candle_pattern(self, data: pd.DataFrame) -> pd.Series:
        """Swing Candle Pattern 계산 (단순 버전)"""
        try:
            body = abs(data["close"] - data["open"])
            range_size = data["high"] - data["low"]

            # 도지 패턴 (작은 몸통)
            doji = body < (range_size * 0.1)

            # 망치 패턴 (아래꼬리가 긴 패턴)
            hammer = (data["close"] > data["open"]) & (
                (data["open"] - data["low"]) > body * 2
            )

            return np.where(hammer, 1, np.where(doji, 0, -1))
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_swing_bollinger_band(self, data: pd.DataFrame) -> pd.Series:
        """Swing Bollinger Band 계산"""
        try:
            ma = data["close"].rolling(20).mean()
            std = data["close"].rolling(20).std()
            upper = ma + (std * 2)
            lower = ma - (std * 2)

            bb_position = (data["close"] - lower) / (upper - lower)
            return (bb_position - 0.5) * 2  # -1 ~ 1 정규화
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_swing_macd(self, data: pd.DataFrame) -> pd.Series:
        """Swing MACD 계산"""
        try:
            ema_12 = data["close"].ewm(span=12).mean()
            ema_26 = data["close"].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal

            return histogram / data["close"] * 100  # 정규화
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI (Relative Strength Index) 계산"""
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.Series:
        """MACD (Moving Average Convergence Divergence) 계산"""
        ema_fast = data["close"].ewm(span=fast).mean()
        ema_slow = data["close"].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_bollinger_bands(
        self, data: pd.DataFrame, period: int = 20, std: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        sma = data["close"].rolling(window=period).mean()
        std_dev = data["close"].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band

    def _calculate_volume_price_trend(self, data: pd.DataFrame) -> pd.Series:
        """볼륨-가격 추세 계산"""
        try:
            price_change = data["close"].pct_change()
            volume_norm = data["volume"] / data["volume"].rolling(20).mean()
            vpt = (price_change * volume_norm).rolling(10).sum()
            return vpt.fillna(0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_price_volume_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """가격-볼륨 오실레이터"""
        try:
            volume_sma_short = data["volume"].rolling(12).mean()
            volume_sma_long = data["volume"].rolling(26).mean()
            pvo = (volume_sma_short - volume_sma_long) / volume_sma_long
            return pvo.fillna(0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """추세 강도 계산"""
        try:
            close = data["close"]
            ma_5 = close.rolling(5).mean()
            ma_10 = close.rolling(10).mean()
            ma_20 = close.rolling(20).mean()
            ma_50 = close.rolling(50).mean()
            
            # 이동평균선 정렬도 계산
            trend_alignment = (
                (ma_5 > ma_10).astype(int) +
                (ma_10 > ma_20).astype(int) +
                (ma_20 > ma_50).astype(int)
            ) / 3.0 - 0.5  # -0.5 ~ 0.5 정규화
            
            return trend_alignment.fillna(0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """변동성 체제 분류"""
        try:
            returns = data["close"].pct_change()
            volatility = returns.rolling(20).std()
            vol_percentile = volatility.rolling(252).rank(pct=True)
            
            # 변동성 체제 (0: 낮음, 1: 높음)
            vol_regime = (vol_percentile > 0.7).astype(float) - (vol_percentile < 0.3).astype(float)
            return vol_regime.fillna(0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_support_resistance_strength(self, data: pd.DataFrame) -> pd.Series:
        """지지/저항 강도 계산"""
        try:
            close = data["close"]
            high = data["high"]
            low = data["low"]
            
            # 최근 20일 고점/저점 대비 현재 위치
            period_high = high.rolling(20).max()
            period_low = low.rolling(20).min()
            
            # 고점/저점 근접도 (-1: 저점 근처, 1: 고점 근처)
            position = (close - period_low) / (period_high - period_low) * 2 - 1
            return position.fillna(0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_momentum_divergence(self, data: pd.DataFrame) -> pd.Series:
        """모멘텀 다이버전스 감지"""
        try:
            close = data["close"]
            rsi = self._calculate_rsi(data, 14)
            
            # 가격과 RSI의 상관관계 변화로 다이버전스 감지
            price_momentum = close.pct_change(5)
            rsi_momentum = rsi.diff(5)
            
            # 상관계수 기반 다이버전스 신호
            correlation = price_momentum.rolling(10).corr(rsi_momentum)
            divergence = (1 - correlation).fillna(0)  # 낮은 상관관계 = 다이버전스
            
            return divergence
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_volatility_skew(self, data: pd.DataFrame) -> pd.Series:
        """변동성 비대칭성 (스큐) 계산"""
        try:
            returns = data["close"].pct_change()
            
            # 롤링 스큐 계산
            skewness = returns.rolling(20).skew()
            
            # 정규화 (-1 ~ 1)
            normalized_skew = np.tanh(skewness / 2)
            return normalized_skew.fillna(0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def _calculate_market_microstructure(self, data: pd.DataFrame) -> pd.Series:
        """시장 미시구조 지표"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            open_price = data["open"]
            
            # 일중 가격 효율성 측정
            intraday_range = (high - low) / close
            gap = abs(open_price - close.shift(1)) / close.shift(1)
            
            # 미시구조 신호 (갭 대비 일중 레인지)
            microstructure = (intraday_range / (gap + 0.001)).rolling(5).mean()
            
            # 로그 변환 후 정규화
            microstructure_norm = np.tanh(np.log1p(microstructure))
            return microstructure_norm.fillna(0)
        except:
            return pd.Series([0.0] * len(data), index=data.index)

    def create_features(
        self,
        stock_data: pd.DataFrame,
        symbol: str,
        market_regime: Dict,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        전체 피처 생성 (feature_engineering.py 사용)

        Args:
            stock_data: 주식 데이터
            symbol: 종목 코드
            market_regime: HMM에서 예측한 시장 체제
            macro_data: 전체 매크로 데이터 (선택사항)

        Returns:
            통합 피처 데이터프레임
        """
        try:
            logger.info(f"🔍 {symbol} 피처 생성 시작...")
            logger.info(f"   - 주식 데이터: {stock_data.shape}")
            logger.info(f"   - 사용 가능한 컬럼 수: {len(stock_data.columns)}개")
            logger.info(
                f"   - 매크로 데이터: {macro_data.shape if macro_data is not None else 'None'}"
            )

            # feature_engineering.py의 동적 피처 생성 사용
            if self.feature_pipeline:
                logger.info(f"   🔧 {symbol} 동적 피처 생성 중...")
                try:
                    # individual 모드에서는 매크로 데이터 제외 (저장된 모델과 일치)
                    features, metadata = self.feature_pipeline.create_dynamic_features(
                        stock_data=stock_data,
                        symbol=symbol,
                        market_regime=market_regime,
                        macro_data=None,  # 개별 모델은 매크로 피처 제외
                        mode="individual",
                    )

                    # 피처 카테고리 요약
                    feature_categories = metadata.get("feature_categories", {})
                    total_features = (
                        sum(feature_categories.values())
                        if feature_categories
                        else len(features.columns)
                    )
                    category_summary = " | ".join(
                        [f"{k}:{v}" for k, v in feature_categories.items() if v > 0]
                    )

                    logger.info(f"   ✅ 동적 피처 생성 완료: {features.shape}")
                    logger.info(
                        f"   📊 복잡도: {metadata.get('data_complexity', 'unknown')} | 카테고리: {category_summary}"
                    )

                    # 피처 정보 저장
                    self.feature_info["individual_features"][symbol] = {
                        "total_features": len(features.columns),
                        "feature_names": list(features.columns),
                        "data_complexity": metadata.get("data_complexity", "unknown"),
                        "created_at": datetime.now().isoformat(),
                    }

                    return features

                except Exception as e:
                    logger.error(f"   ❌ 동적 피처 생성 실패: {e}")
                    logger.info("   🔄 기본 피처 생성 방식으로 전환...")

            # 기본 피처 생성 방식 (fallback)
            logger.info(f"   📊 기본 피처 생성 중...")
            return self._create_basic_features(
                stock_data, symbol, market_regime, macro_data
            )

        except Exception as e:
            logger.error(f"피처 생성 실패 ({symbol}): {e}")
            return pd.DataFrame(index=stock_data.index)

    def _create_basic_features(
        self,
        stock_data: pd.DataFrame,
        symbol: str,
        market_regime: Dict,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """기본 피처 생성 (fallback)"""
        features = pd.DataFrame(index=stock_data.index)

        try:
            # 기본 스윙 피처
            swing_features = self.extract_swing_features(stock_data, symbol)
            features = pd.concat([features, swing_features], axis=1)

            # 시장 체제 피처
            regime_features = pd.DataFrame(index=stock_data.index)
            regimes = ["TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE"]
            current_regime = market_regime.get("current_regime", "SIDEWAYS")

            for regime in regimes:
                regime_features[f"regime_{regime.lower()}"] = pd.Series(
                    int(current_regime == regime), index=stock_data.index
                )

            features = pd.concat([features, regime_features], axis=1)

            # NaN 처리
            features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)

        except Exception as e:
            logger.error(f"기본 피처 생성 실패: {e}")

        return features

    def prepare_training_data(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, pd.DataFrame, np.ndarray],
        lookback_days: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 윈도우 데이터 준비

        Args:
            features: 피처 데이터
            target: 타겟 데이터 (단일 또는 멀티타겟)
            lookback_days: 윈도우 크기

        Returns:
            (X, y) 튜플
        """
        try:
            # 타겟을 numpy array로 변환
            if isinstance(target, pd.Series):
                target_array = target.values
            elif isinstance(target, pd.DataFrame):
                target_array = target.values
            else:
                target_array = target

            # 1D인 경우 2D로 변환
            if len(target_array.shape) == 1:
                target_array = target_array.reshape(-1, 1)

            # 피처를 numpy array로 변환
            if isinstance(features, pd.DataFrame):
                features_array = features.values
            else:
                features_array = features

            # 데이터 타입 검증 및 변환
            try:
                # 피처 데이터를 숫자형으로 변환
                if isinstance(features_array, np.ndarray):
                    # 문자열이나 객체 타입이 있는지 확인
                    if features_array.dtype.kind in [
                        "O",
                        "U",
                        "S",
                    ]:  # object, unicode, bytes
                        logger.warning(
                            "피처에 문자열 데이터가 발견되어 숫자형으로 변환합니다"
                        )
                        # 숫자형으로 변환 가능한 컬럼만 선택
                        numeric_features = []
                        for i in range(features_array.shape[1]):
                            try:
                                numeric_features.append(
                                    pd.to_numeric(features_array[:, i], errors="coerce")
                                )
                            except:
                                # 변환 불가능한 컬럼은 0으로 채움
                                numeric_features.append(
                                    np.zeros(features_array.shape[0])
                                )
                        features_array = np.column_stack(numeric_features)

                # 타겟 데이터를 숫자형으로 변환
                if isinstance(target_array, np.ndarray):
                    if target_array.dtype.kind in ["O", "U", "S"]:
                        logger.warning(
                            "타겟에 문자열 데이터가 발견되어 숫자형으로 변환합니다"
                        )
                        target_array = pd.to_numeric(
                            target_array.flatten(), errors="coerce"
                        ).reshape(target_array.shape)

                # NaN 검증 및 처리
                if np.isnan(features_array).any():
                    logger.warning("피처에 NaN이 발견되어 처리합니다")
                    # NaN을 0으로 채우기
                    features_array = np.nan_to_num(features_array, nan=0.0)

                if np.isnan(target_array).any():
                    logger.warning("타겟에 NaN이 발견되어 처리합니다")
                    # NaN을 0으로 채우기
                    target_array = np.nan_to_num(target_array, nan=0.0)

            except Exception as e:
                logger.error(f"데이터 타입 변환 중 오류: {e}")
                # 오류 발생 시 모든 값을 0으로 설정
                features_array = np.zeros_like(features_array, dtype=float)
                target_array = np.zeros_like(target_array, dtype=float)

            # 윈도우 데이터 생성
            X, y = [], []

            for i in range(lookback_days, len(features_array)):
                # 입력 윈도우
                X.append(features_array[i - lookback_days : i].flatten())
                # 타겟 (현재 시점의 미래 수익률)
                y.append(target_array[i])

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"시계열 데이터 준비 실패: {e}")
            return np.array([]), np.array([])

    def _prepare_prediction_data(
        self, features: pd.DataFrame, lookback_days: int = 20
    ) -> np.ndarray:
        """
        예측용 윈도우 데이터 생성 (타겟 데이터 없이)

        Args:
            features: 피처 데이터
            lookback_days: 룩백 기간

        Returns:
            윈도우 형태의 피처 배열
        """
        try:
            # 피처를 numpy array로 변환
            if isinstance(features, pd.DataFrame):
                features_array = features.values
            else:
                features_array = features

            # 데이터 타입 검증 및 변환
            if features_array.dtype.kind in ["O", "U", "S"]:  # object, unicode, bytes
                logger.warning("피처에 문자열 데이터가 발견되어 숫자형으로 변환합니다")
                numeric_features = []
                for i in range(features_array.shape[1]):
                    try:
                        numeric_features.append(
                            pd.to_numeric(features_array[:, i], errors="coerce")
                        )
                    except:
                        numeric_features.append(np.zeros(features_array.shape[0]))
                features_array = np.column_stack(numeric_features)

            # NaN 처리
            if np.isnan(features_array).any():
                logger.warning("피처에 NaN이 발견되어 처리합니다")
                features_array = np.nan_to_num(features_array, nan=0.0)

            # 데이터 길이 확인
            if len(features_array) < lookback_days:
                logger.warning(
                    f"데이터 길이({len(features_array)})가 룩백 기간({lookback_days})보다 짧습니다"
                )
                # 부족한 데이터는 첫 번째 행으로 패딩
                padding_rows = lookback_days - len(features_array)
                if len(features_array) > 0:
                    padding = np.repeat(features_array[0:1], padding_rows, axis=0)
                    features_array = np.vstack([padding, features_array])
                else:
                    logger.error("피처 데이터가 비어있습니다")
                    return np.array([])

            # 최근 lookback_days 만큼의 데이터로 윈도우 생성
            if len(features_array) >= lookback_days:
                # 가장 최근 데이터의 윈도우만 생성 (예측용)
                window = features_array[-lookback_days:].flatten()
                return np.array([window])  # 배치 차원 추가
            else:
                logger.error(
                    f"윈도우 생성 불가: 데이터 길이({len(features_array)}) < 룩백 기간({lookback_days})"
                )
                return np.array([])

        except Exception as e:
            logger.error(f"예측용 데이터 준비 실패: {e}")
            return np.array([])

    def _train_universal_model(self, training_data: Dict) -> bool:
        """
        통합 모델 학습 (모든 종목 데이터를 합쳐서 학습)
        """
        try:
            # 전체 학습 데이터 수집
            all_X, all_y = [], []
            target_columns = None
            symbol_features = {}  # 종목별 피처 저장

            for symbol, data in training_data.items():
                logger.info(f"🔍 {symbol} 통합 모델용 데이터 처리 중...")

                try:
                    features = data["features"]
                    target = data["target"]
                except KeyError as e:
                    logger.error(
                        f"❌ {symbol}: {e} 키가 없습니다. 사용 가능한 키: {list(data.keys())}"
                    )
                    continue

                # 멀티타겟인 경우 컬럼명 저장
                if isinstance(target, pd.DataFrame):
                    if target_columns is None:
                        target_columns = list(target.columns)
                    target = target.values
                elif isinstance(target, pd.Series):
                    target = target.values

                if len(features) < 50 or len(target) < 50:
                    logger.warning(
                        f"{symbol}: 학습 데이터 부족 (features: {len(features)}, target: {len(target)})"
                    )
                    continue

                # 종목별 피처 저장
                symbol_features[symbol] = features

                # 시계열 윈도우 데이터 생성
                features_config = self.neural_config.get("features", {})
                lookback = features_config.get("lookback_days", 20)
                X, y = self.prepare_training_data(features, target, lookback)

                if len(X) > 0 and len(y) > 0:
                    all_X.append(X)
                    all_y.append(y)
                else:
                    logger.warning(
                        f"{symbol}: 윈도우 데이터 생성 실패 (X: {len(X)}, y: {len(y)})"
                    )

            if not all_X:
                logger.error("통합 모델 학습 데이터가 없습니다")
                return False

            # 종목별 피처를 통합하여 새로운 피처 생성
            logger.info("🔧 종목별 피처 통합 중...")
            symbols = list(symbol_features.keys())

            # 첫 번째 종목의 피처 구조를 기준으로 통합 피처 생성
            base_features = symbol_features[symbols[0]]
            combined_features = pd.DataFrame(index=base_features.index)

            # 각 종목의 스윙 전략 피처를 종목별로 구분하여 추가
            swing_feature_names = [
                "dual_momentum",
                "volatility_breakout",
                "swing_ema",
                "swing_rsi",
                "swing_donchian",
                "stoch_donchian",
                "whipsaw_prevention",
                "donchian_rsi_whipsaw",
                "volatility_filtered_breakout",
                "multi_timeframe_whipsaw",
                "adaptive_whipsaw",
                "cci_bollinger",
                "mean_reversion",
                "swing_breakout",
                "swing_pullback_entry",
                "swing_candle_pattern",
                "swing_bollinger_band",
                "swing_macd",
            ]

            for symbol in symbols:
                features = symbol_features[symbol]
                for feature_name in swing_feature_names:
                    if feature_name in features.columns:
                        combined_features[f"{symbol}_{feature_name}"] = features[
                            feature_name
                        ]
                    else:
                        # 피처가 없으면 0으로 채움
                        combined_features[f"{symbol}_{feature_name}"] = 0.0

            # 공통 피처 추가 (체제, 매크로 등)
            for symbol in symbols:
                features = symbol_features[symbol]
                for col in features.columns:
                    if (
                        col not in swing_feature_names
                        and col not in combined_features.columns
                    ):
                        combined_features[col] = features[col]

            logger.info(f"✅ 통합 피처 생성 완료: {combined_features.shape}")
            logger.info(
                f"📊 종목별 스윙 피처: {len(symbols)} × {len(swing_feature_names)} = {len(symbols) * len(swing_feature_names)}개"
            )

            # 통합된 피처로 새로운 윈도우 데이터 생성
            all_X_combined, all_y_combined = [], []

            for symbol, data in training_data.items():
                target = data["target"]
                if isinstance(target, pd.DataFrame):
                    target = target.values
                elif isinstance(target, pd.Series):
                    target = target.values

                features_config = self.neural_config.get("features", {})
                lookback = features_config.get("lookback_days", 20)
                X, y = self.prepare_training_data(combined_features, target, lookback)

                if len(X) > 0 and len(y) > 0:
                    all_X_combined.append(X)
                    all_y_combined.append(y)

            if not all_X_combined:
                logger.error("통합 피처로 생성된 학습 데이터가 없습니다")
                return False

            # 데이터 결합
            X_combined = np.vstack(all_X_combined)
            y_combined = (
                np.vstack(all_y_combined)
                if len(all_y_combined[0].shape) > 1
                else np.hstack(all_y_combined)
            )

            # NaN 검증 및 제거
            if np.isnan(X_combined).any():
                logger.warning("X 데이터에 NaN이 발견되어 제거합니다")
                valid_mask = ~np.isnan(X_combined).any(axis=1)
                X_combined = X_combined[valid_mask]
                y_combined = (
                    y_combined[valid_mask]
                    if len(y_combined.shape) > 1
                    else y_combined[valid_mask]
                )

            if np.isnan(y_combined).any():
                logger.warning("y 데이터에 NaN이 발견되어 제거합니다")
                valid_mask = (
                    ~np.isnan(y_combined).any(axis=1)
                    if len(y_combined.shape) > 1
                    else ~np.isnan(y_combined)
                )
                X_combined = X_combined[valid_mask]
                y_combined = y_combined[valid_mask]

            # 데이터 크기 재확인
            if len(X_combined) < 50:
                logger.error(f"유효한 학습 데이터가 부족합니다: {len(X_combined)}행")
                return False

            logger.info(
                f"통합 모델 학습 데이터 크기: {X_combined.shape}, 타겟: {y_combined.shape}"
            )
            if target_columns:
                logger.info(f"멀티타겟 컬럼: {target_columns}")
                self.target_columns = target_columns

            # 피처 이름 저장
            if isinstance(combined_features, pd.DataFrame):
                self.feature_names = list(combined_features.columns)
            else:
                self.feature_names = [
                    f"feature_{i}" for i in range(X_combined.shape[1])
                ]

            # 데이터 스케일링
            X_scaled = self.universal_scaler.fit_transform(X_combined)

            # 타겟 클리핑 (-1 ~ 1) 및 NaN 검증
            y_clipped = np.clip(y_combined, -1, 1)

            # 최종 NaN 검증
            if np.isnan(X_scaled).any() or np.isnan(y_clipped).any():
                logger.error("스케일링 후에도 NaN이 존재합니다. 학습을 중단합니다.")
                return False

            # 모델 구축
            output_size = y_clipped.shape[1] if len(y_clipped.shape) > 1 else 1
            self.universal_model = self._build_model(X_scaled.shape[1], output_size)

            # 학습/검증 데이터 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_clipped, test_size=0.2, random_state=42
            )

            # 학습 설정
            training_config = self.neural_config.get("training", {})
            batch_size = training_config.get("batch_size", 32)
            train_dataset = StockDataset(X_train, y_train)
            val_dataset = StockDataset(X_val, y_val)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 손실 함수 및 옵티마이저
            criterion = nn.MSELoss()
            learning_rate = training_config.get("learning_rate", 0.001)
            optimizer = optim.Adam(self.universal_model.parameters(), lr=learning_rate)

            # 학습 파라미터
            epochs = training_config.get("epochs", 200)
            best_val_loss = float("inf")
            patience = training_config.get("early_stopping_patience", 20)
            patience_counter = 0

            logger.info(f"통합 모델 학습 시작 - Epochs: {epochs}, Patience: {patience}")

            # tqdm 진행바
            try:
                epoch_iter = trange(epochs, desc="Universal Model")
            except Exception:
                epoch_iter = range(epochs)

            actual_epochs = 0
            for epoch in epoch_iter:
                actual_epochs = epoch + 1
                self.universal_model.train()
                train_losses = []
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.universal_model(X_batch.float())
                    loss = criterion(outputs.squeeze(), y_batch.float())
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                train_loss = np.mean(train_losses)

                # 검증
                self.universal_model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.universal_model(X_batch.float())
                        loss = criterion(outputs.squeeze(), y_batch.float())
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)

                # tqdm bar에 loss 표시
                if hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(
                        {
                            "train_loss": f"{train_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                            "best_val": f"{best_val_loss:.4f}",
                            "patience": patience_counter,
                        }
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val_loss:.4f} | Patience: {patience_counter}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {actual_epochs} (val_loss did not improve for {patience} epochs)"
                    )
                    print(
                        f"   ⏹️  Early stopping at epoch {actual_epochs} (patience: {patience})"
                    )
                    break

            logger.info(f"통합 모델 학습 완료 - Best Val Loss: {best_val_loss:.4f}")

            # 최종 학습 결과 요약 출력
            print(f"✅ 통합 모델 학습 완료:")
            print(f"   📊 최종 Train Loss: {train_loss:.6f}")
            print(f"   📈 최종 Val Loss: {val_loss:.6f}")
            print(f"   🎯 Best Val Loss: {best_val_loss:.6f}")
            print(
                f"   📈 학습 Epochs: {actual_epochs}/{epochs} (Early stopping: {actual_epochs < epochs})"
            )

            return True

        except Exception as e:
            logger.error(f"통합 모델 학습 실패: {e}")
            return False

    def _train_individual_models(self, training_data: Dict) -> bool:
        """
        종목별 모델 학습 (각 종목별로 개별 모델 생성)
        """
        try:
            success_count = 0
            total_count = len(training_data)

            logger.info(f"🎯 종목별 모델 학습 시작: {total_count}개 종목")
            logger.info(f"📊 학습할 종목들: {list(training_data.keys())}")

            # tqdm 진행바로 종목별 학습 현황 표시
            try:
                symbol_iter = trange(total_count, desc="Individual Models")
                symbols = list(training_data.keys())
            except Exception as e:
                symbol_iter = range(total_count)
                symbols = list(training_data.keys())

            for idx, symbol in enumerate(symbols):
                data = training_data[symbol]

                if hasattr(symbol_iter, "set_description"):
                    symbol_iter.set_description(f"Training {symbol}")

                logger.info(
                    f"🎯 {symbol} 개별 모델 학습 시작... ({idx+1}/{total_count})"
                )

                try:
                    features = data["features"]
                    target = data["target"]

                except KeyError as e:
                    print(f"❌ {symbol}: {e} 키가 없습니다.")
                    logger.error(f"❌ {symbol}: {e} 키가 없습니다.")
                    if hasattr(symbol_iter, "update"):
                        symbol_iter.update(1)
                    continue

                # 타겟 처리
                if isinstance(target, pd.DataFrame):
                    target = target.values
                elif isinstance(target, pd.Series):
                    target = target.values

                if len(features) < 50 or len(target) < 50:
                    logger.warning(
                        f"{symbol}: 개별 모델 학습 데이터 부족 (features: {len(features)}, target: {len(target)})"
                    )
                    if hasattr(symbol_iter, "update"):
                        symbol_iter.update(1)
                    continue

                # 시계열 윈도우 데이터 생성
                features_config = self.neural_config.get("features", {})
                lookback = features_config.get("lookback_days", 20)
                X, y = self.prepare_training_data(features, target, lookback)

                if len(X) == 0 or len(y) == 0:
                    logger.warning(f"{symbol}: 윈도우 데이터 생성 실패")
                    if hasattr(symbol_iter, "update"):
                        symbol_iter.update(1)
                    continue

                # 데이터 스케일링
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # NaN 검증
                if np.isnan(X_scaled).any() or np.isnan(y).any():
                    logger.warning(f"{symbol}: NaN이 발견되어 건너뜀")
                    if hasattr(symbol_iter, "update"):
                        symbol_iter.update(1)
                    continue

                # 타겟 클리핑 (-1 ~ 1)
                y_clipped = np.clip(y, -1, 1)

                # 모델 구축
                output_size = y_clipped.shape[1] if len(y_clipped.shape) > 1 else 1
                model = self._build_model(X_scaled.shape[1], output_size)

                # 학습/검증 데이터 분할
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_clipped, test_size=0.2, random_state=42
                )

                # 학습 설정
                training_config = self.neural_config.get("training", {})
                batch_size = training_config.get("batch_size", 32)
                train_dataset = StockDataset(X_train, y_train)
                val_dataset = StockDataset(X_val, y_val)

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                # 손실 함수 및 옵티마이저
                criterion = nn.MSELoss()
                learning_rate = training_config.get("learning_rate", 0.001)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # 학습 파라미터
                epochs = training_config.get("epochs", 200)
                best_val_loss = float("inf")
                patience = training_config.get("early_stopping_patience", 20)
                patience_counter = 0

                # 학습 과정 추적
                train_losses = []
                val_losses = []

                # 학습 루프
                actual_epochs = 0
                for epoch in range(epochs):
                    actual_epochs = epoch + 1
                    # 학습 모드
                    model.train()
                    epoch_train_losses = []

                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        outputs = model(X_batch.float())
                        loss = criterion(outputs.squeeze(), y_batch.float())
                        loss.backward()
                        optimizer.step()
                        epoch_train_losses.append(loss.item())

                    # 검증 모드
                    model.eval()
                    epoch_val_losses = []

                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            outputs = model(X_batch.float())
                            loss = criterion(outputs.squeeze(), y_batch.float())
                            epoch_val_losses.append(loss.item())

                    # 평균 손실 계산
                    train_loss = np.mean(epoch_train_losses)
                    val_loss = np.mean(epoch_val_losses)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    # tqdm 진행바에 실시간 로스 정보 표시
                    if hasattr(symbol_iter, "set_postfix"):
                        symbol_iter.set_postfix(
                            {
                                "success": success_count,
                                "total": total_count,
                                "current": symbol,
                                "epoch": f"{epoch+1}/{epochs}",
                                "train_loss": f"{train_loss:.4f}",
                                "val_loss": f"{val_loss:.4f}",
                                "best_val": f"{best_val_loss:.4f}",
                                "patience": patience_counter,
                            }
                        )

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(
                                f"   ⏹️  Early stopping at epoch {actual_epochs} (patience: {patience})"
                            )
                            break

                # 모델 저장
                self.individual_models[symbol] = model
                self.individual_scalers[symbol] = scaler

                # 상세한 학습 결과 출력
                final_train_loss = train_losses[-1] if train_losses else float("inf")
                final_val_loss = val_losses[-1] if val_losses else float("inf")
                min_train_loss = min(train_losses) if train_losses else float("inf")
                min_val_loss = min(val_losses) if val_losses else float("inf")

                logger.info(f"✅ {symbol} 개별 모델 학습 완료:")
                logger.info(
                    f"   📊 최종 Train Loss: {final_train_loss:.6f} | 최종 Val Loss: {final_val_loss:.6f}"
                )
                logger.info(
                    f"   🎯 최고 Train Loss: {min_train_loss:.6f} | 최고 Val Loss: {min_val_loss:.6f}"
                )
                logger.info(
                    f"   📈 학습 Epochs: {len(train_losses)} | 데이터 크기: {X_scaled.shape}"
                )

                # 개별 모델 학습 완료 요약 출력 (간소화)
                print(
                    f"✅ {symbol}: {actual_epochs}/{epochs} epochs, Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}, Best: {min_val_loss:.4f}"
                )

                success_count += 1

                # tqdm 진행바 업데이트 (최종 로스 정보 포함)
                if hasattr(symbol_iter, "set_postfix"):
                    final_train_loss = (
                        train_losses[-1] if train_losses else float("inf")
                    )
                    final_val_loss = val_losses[-1] if val_losses else float("inf")
                    symbol_iter.set_postfix(
                        {
                            "success": success_count,
                            "total": total_count,
                            "current": symbol,
                            "final_train": f"{final_train_loss:.4f}",
                            "final_val": f"{final_val_loss:.4f}",
                        }
                    )

                # tqdm 진행바 업데이트 (성공한 경우)
                if hasattr(symbol_iter, "update"):
                    symbol_iter.update(1)

            logger.info(
                f"🎯 종목별 모델 학습 완료: {success_count}/{total_count}개 성공"
            )
            logger.info(f"📊 생성된 개별 모델들: {list(self.individual_models.keys())}")

            # 전체 개별 모델 학습 요약 출력 (간소화)
            print(f"🎯 개별 모델 학습 완료: {success_count}/{total_count}개 성공")

            return success_count > 0

        except Exception as e:
            logger.error(f"종목별 모델 학습 실패: {e}")
            return False

    def fit(self, training_data: Dict) -> bool:
        """
        앙상블 신경망 모델 학습 (통합 모델 + 종목별 모델 + 가중치 학습기)

        Args:
            training_data: {symbol: {'features': DataFrame, 'target': Series}} 형태

        Returns:
            학습 성공 여부
        """
        try:
            logger.info("앙상블 신경망 모델 학습 시작...")
            logger.info(f"📊 training_data 구조: {list(training_data.keys())}")

            # 1. Train-test 분할
            logger.info("📊 Train-test 데이터 분할 시작...")
            train_data, test_data = self._split_train_test_data(training_data)

            if not test_data:
                logger.warning("⚠️ Test 데이터가 없어 검증을 건너뜁니다.")
                test_data = {}

            # 2. 통합 모델 학습 (Train set만 사용)
            logger.info("🌐 통합 모델 학습 시작...")
            if not self._train_universal_model(train_data):
                logger.error("통합 모델 학습 실패")
                return False

            # 3. 종목별 모델 학습 (옵션, Train set만 사용)
            if self.enable_individual_models:
                logger.info("🎯 종목별 모델 학습 시작...")
                logger.info(f"   📈 학습할 종목 수: {len(train_data)}")
                if not self._train_individual_models(train_data):
                    logger.warning("일부 종목별 모델 학습 실패")

            # 4. 앙상블 가중치 학습기 훈련 (Train set만 사용)
            if self.enable_weight_learning:
                print("\n" + "="*70)
                print("⚖️  앙상블 가중치 학습 시작")
                print("="*70)
                logger.info("⚖️ 앙상블 가중치 학습기 훈련 시작...")
                logger.info(
                    f"   📊 enable_weight_learning: {self.enable_weight_learning}"
                )
                logger.info(f"   📈 train_data 종목 수: {len(train_data)}")
                
                # 앙상블 모델 구성 정보 출력
                print("\n📋 앙상블 모델 구성:")
                print(f"   - Universal 모델: {'✅ 활성화' if self.universal_model else '❌ 비활성화'}")
                print(f"   - Individual 모델: {'✅ 활성화' if self.enable_individual_models else '❌ 비활성화'}")
                print(f"   - 가중치 학습기: {'✅ 활성화' if self.enable_weight_learning else '❌ 비활성화'}")
                print(f"\n📊 초기 앙상블 가중치:")
                print(f"   - Universal: {self.universal_weight:.1%}")
                print(f"   - Individual: {self.individual_weight:.1%}")
                
                weight_learning_success = self._train_ensemble_weight_learner(train_data)
                if not weight_learning_success:
                    print("\n❌ 앙상블 가중치 학습기 훈련 실패")
                    print(f"   → 기본 가중치 사용: Universal {self.universal_weight:.1%}, Individual {self.individual_weight:.1%}")
                    logger.warning("앙상블 가중치 학습기 훈련 실패")
                else:
                    print("\n✅ 앙상블 가중치 학습기 훈련 완료!")
                    
                    # 동적 가중치 테스트 (전체 종목)
                    print(f"\n🎯 전체 종목 앙상블 가중치:")
                    print("─" * 80)
                    print(f"{'종목':^8} {'Universal':^12} {'Individual':^12} {'Universal 변화':^15} {'Individual 변화':^15}")
                    print("─" * 80)
                    
                    weights_calculated = False
                    for symbol in train_data.keys():
                        if len(train_data[symbol]['features']) > 20:
                            symbol_features = train_data[symbol]['features'].tail(20)
                            try:
                                dynamic_universal, dynamic_individual = self._update_ensemble_weights(symbol, symbol_features)
                                universal_change = (dynamic_universal - self.universal_weight) * 100
                                individual_change = (dynamic_individual - self.individual_weight) * 100
                                
                                print(f"{symbol:^8} {dynamic_universal:^11.1%} {dynamic_individual:^11.1%} {universal_change:^14.1f}%p {individual_change:^14.1f}%p")
                                weights_calculated = True
                            except Exception as e:
                                print(f"{symbol:^8} {'오류':^11} {'오류':^11} {'N/A':^14} {'N/A':^14}")
                    
                    print("─" * 80)
                    print(f"📌 기본 가중치: Universal {self.universal_weight:.1%}, Individual {self.individual_weight:.1%}")
                    
                    if not weights_calculated:
                        print(f"\n⚠️ 동적 가중치 계산 실패 - 기본 가중치 사용")
                        print(f"   • Universal: {self.universal_weight:.1%}")
                        print(f"   • Individual: {self.individual_weight:.1%}")
                        print(f"   • 동적 가중치는 예측 시 실시간 계산됨")
                    print("="*70 + "\n")
            else:
                print("⏩ 앙상블 가중치 학습 건너뛰기 (비활성화)")
                logger.info(
                    "⏩ 앙상블 가중치 학습 건너뛰기 (enable_weight_learning: False)"
                )

            # 5. 22일 예측 검증 (Test set 사용)
            if test_data:
                print("\n" + "="*70)
                print("🔍 22일 예측 검증 (Test Set)")
                print("="*70)
                logger.info("🔍 22일 예측 검증 시작...")
                logger.info(f"   📊 test_data 종목 수: {len(test_data)}")
                logger.info(f"   📈 test_data 종목들: {list(test_data.keys())}")
                
                # Train/Test 분할 정보 출력
                print(f"\n📊 Train/Test 분할 정보:")
                print(f"   - Train 비율: {self.train_ratio:.1%}")
                print(f"   - Test 비율: {1-self.train_ratio:.1%}")
                
                for symbol in list(test_data.keys())[:3]:  # 처음 3개 종목만 출력
                    if symbol in training_data:
                        total_len = len(training_data[symbol]['features'])
                        train_len = len(train_data[symbol]['features']) if symbol in train_data else 0
                        test_len = len(test_data[symbol]['features'])
                        print(f"   - {symbol}: 전체 {total_len}일 → Train {train_len}일, Test {test_len}일")
                
                validation_results = self._validate_22d_predictions(test_data)

                # 검증 결과 요약
                print("\n📊 22일 예측 검증 결과:")
                print("─" * 50)
                print(f"{'종목':^10} {'RMSE':^10} {'예측수':^10} {'평균오차':^10}")
                print("─" * 50)
                
                all_predictions = []
                all_actuals = []
                
                for symbol, result in validation_results.items():
                    if result["num_predictions"] > 0:
                        # 평균 오차 계산
                        predictions = result['predictions']
                        actuals = result['actual_values']
                        all_predictions.extend(predictions)
                        all_actuals.extend(actuals)
                        
                        mean_error = np.mean([p - a for p, a in zip(predictions, actuals)])
                        
                        print(f"{symbol:^10} {result['rmse']:^10.4f} {result['num_predictions']:^10d} {mean_error:^10.4f}")
                        logger.info(
                            f"   {symbol}: RMSE = {result['rmse']:.4f} ({result['num_predictions']}개 예측, 평균오차 = {mean_error:.4f})"
                        )
                    else:
                        print(f"{symbol:^10} {'N/A':^10} {0:^10d} {'N/A':^10}")
                        logger.warning(f"   {symbol}: 검증 데이터 부족")
                
                print("─" * 50)
                
                # 전체 평균 RMSE 계산
                valid_rmses = [
                    result["rmse"]
                    for result in validation_results.values()
                    if result["num_predictions"] > 0
                ]
                if valid_rmses:
                    avg_rmse = sum(valid_rmses) / len(valid_rmses)
                    overall_mean_error = np.mean([p - a for p, a in zip(all_predictions, all_actuals)])
                    overall_mae = np.mean([abs(p - a) for p, a in zip(all_predictions, all_actuals)])
                    
                    print(f"\n📊 전체 성능 지표:")
                    print(f"   - 평균 RMSE: {avg_rmse:.4f}")
                    print(f"   - 평균 오차 (ME): {overall_mean_error:.4f}")
                    print(f"   - 평균 절대 오차 (MAE): {overall_mae:.4f}")
                    print(f"   - 총 예측 수: {len(all_predictions)}개")
                    
                    logger.info(f"📊 전체 평균 RMSE: {avg_rmse:.4f}")
                else:
                    print("\n⚠️ 유효한 RMSE가 없습니다.")
                    logger.warning("⚠️ 유효한 RMSE가 없습니다.")
                
                print("="*70 + "\n")

            print("\n" + "="*60)
            print("🎉 앙상블 신경망 모델 학습 완료!")
            print("="*60)
            logger.info("✅ 앙상블 신경망 모델 학습 완료")
            return True

        except Exception as e:
            logger.error(f"앙상블 신경망 모델 학습 실패: {e}")
            return False

    def predict(
        self, features: pd.DataFrame, symbol: str
    ) -> Union[float, Dict[str, float]]:
        """
        앙상블 예측 (통합 모델 + 종목별 모델)

        Args:
            features: 입력 피처
            symbol: 종목 심볼

        Returns:
            예측 결과 (단일값 또는 멀티타겟 딕셔너리)
        """
        try:
            if self.universal_model is None:
                logger.error("통합 모델이 학습되지 않았습니다")
                return None

            # 피처 전처리 - 개별 종목 예측 시에는 피처명 필터링 스킵
            # 통합 모델용 feature_names는 모든 종목 피처를 포함하므로 개별 예측 시 사용 불가
            logger.info(f"예측용 피처 입력: {features.shape}")
            logger.info(f"피처 컬럼 샘플: {list(features.columns[:5])}")

            # 시계열 윈도우 데이터 생성
            features_config = self.neural_config.get("features", {})
            lookback = features_config.get("lookback_days", 20)

            # 최근 데이터만 사용 (예측용)
            if len(features) > lookback:
                features = features.tail(lookback)

            # 개별 모델 예측을 위한 피처 필터링
            if symbol in self.individual_models and hasattr(self, "feature_info"):
                individual_features = self.feature_info.get("individual_features", {})
                if symbol in individual_features:
                    saved_feature_names = individual_features[symbol].get(
                        "feature_names", []
                    )

                    # 저장된 피처명과 현재 피처명의 교집합만 사용
                    available_features = [
                        col for col in saved_feature_names if col in features.columns
                    ]

                    if len(available_features) != len(saved_feature_names):
                        logger.warning(
                            f"{symbol} 피처 불일치: 저장된({len(saved_feature_names)}) vs 사용가능({len(available_features)})"
                        )
                        missing_features = [
                            col
                            for col in saved_feature_names
                            if col not in features.columns
                        ]
                        extra_features = [
                            col
                            for col in features.columns
                            if col not in saved_feature_names
                            and not col.startswith("macro_")
                        ]
                        logger.info(f"누락된 피처: {missing_features[:3]}...")
                        logger.info(f"추가된 피처: {extra_features[:3]}...")

                    # 저장된 피처명 순서대로 필터링
                    if available_features:
                        features = features[available_features]
                        logger.info(
                            f"{symbol} 모델용 피처 필터링 완료: {features.shape}"
                        )
                    else:
                        logger.error(f"{symbol} 사용 가능한 피처가 없습니다!")
                        return None

            # 예측용 윈도우 데이터 생성 (타겟 없이)
            try:
                X = self._prepare_prediction_data(features, lookback)
                if len(X) == 0:
                    logger.error("예측용 윈도우 데이터 생성 실패")
                    return None
            except Exception as e:
                logger.error(f"예측용 윈도우 데이터 생성 실패: {e}")
                return None

            # 개별 종목 예측 시 우선순위: 개별 모델 > 통합 모델
            individual_pred = None
            universal_pred = None

            # 1. 개별 모델 예측 (우선 시도)
            if (
                self.enable_individual_models
                and symbol in self.individual_models
                and symbol in self.individual_scalers
            ):
                try:
                    X_individual = self.individual_scalers[symbol].transform(X)
                    individual_pred = self._predict_with_model(
                        self.individual_models[symbol], X_individual
                    )
                    logger.info(f"✅ {symbol} 개별 모델 예측 성공: {individual_pred}")
                except Exception as e:
                    logger.warning(f"❌ {symbol} 개별 모델 예측 실패: {e}")

            # 2. 통합 모델 예측 (개별 모델이 없거나 실패한 경우)
            if individual_pred is None:
                try:
                    # 통합 모델도 개별 종목 예측에 사용 가능하도록 수정
                    if (
                        self.universal_model is not None
                        and self.universal_scaler is not None
                    ):
                        # 통합 모델용 피처 준비 (차원 매핑 필요)
                        try:
                            # 통합 모델의 feature_names와 현재 피처 매핑
                            if self.feature_names is not None:
                                # 통합 모델 피처에 맞게 피처 선택
                                available_features = [
                                    col
                                    for col in self.feature_names
                                    if col in features.columns
                                ]

                                if len(available_features) > 0:
                                    # 사용 가능한 피처만 선택
                                    features_for_universal = features[
                                        available_features
                                    ]

                                    # 부족한 피처는 0으로 채움
                                    if len(available_features) < len(
                                        self.feature_names
                                    ):
                                        missing_features = [
                                            col
                                            for col in self.feature_names
                                            if col not in available_features
                                        ]
                                        for col in missing_features:
                                            features_for_universal[col] = 0.0

                                    # 피처 순서 맞추기
                                    features_for_universal = features_for_universal[
                                        self.feature_names
                                    ]

                                    # 윈도우 데이터 생성
                                    X_universal = self._prepare_prediction_data(
                                        features_for_universal, lookback
                                    )

                                    # 차원 검증
                                    expected_dim = len(self.feature_names) * lookback
                                    actual_dim = (
                                        X_universal.shape[1]
                                        if len(X_universal.shape) > 1
                                        else X_universal.shape[0]
                                    )

                                    if actual_dim != expected_dim:
                                        logger.warning(
                                            f"{symbol} 통합 모델 차원 불일치: 예상 {expected_dim}, 실제 {actual_dim}"
                                        )
                                        # 차원을 맞추기 위해 조정
                                        if actual_dim < expected_dim:
                                            # 부족한 차원을 0으로 채움
                                            padding = np.zeros(
                                                (
                                                    X_universal.shape[0],
                                                    expected_dim - actual_dim,
                                                )
                                            )
                                            X_universal = np.hstack(
                                                [X_universal, padding]
                                            )
                                        else:
                                            # 초과 차원을 잘라냄
                                            X_universal = X_universal[:, :expected_dim]

                                    X_universal_scaled = (
                                        self.universal_scaler.transform(X_universal)
                                    )

                                    universal_pred = self._predict_with_model(
                                        self.universal_model, X_universal_scaled
                                    )
                                    logger.info(
                                        f"🌐 {symbol} 통합 모델 예측 성공: {universal_pred}"
                                    )
                                else:
                                    logger.warning(
                                        f"{symbol} 통합 모델용 피처가 없습니다."
                                    )
                                    universal_pred = None
                            else:
                                logger.warning(f"통합 모델의 feature_names가 없습니다.")
                                universal_pred = None

                        except Exception as e:
                            logger.warning(f"{symbol} 통합 모델 피처 매핑 실패: {e}")
                            universal_pred = None
                    else:
                        logger.warning(f"통합 모델이 로드되지 않았습니다.")
                        universal_pred = None
                except Exception as e:
                    logger.warning(f"❌ {symbol} 통합 모델 예측 실패: {e}")
                    universal_pred = None

            # 3. 앙상블 조합 - 동적 가중치 사용
            if individual_pred is not None and universal_pred is not None:
                # 두 모델 모두 예측 성공한 경우 앙상블 조합
                try:
                    # 동적 가중치 계산
                    dynamic_universal_weight, dynamic_individual_weight = (
                        self._update_ensemble_weights(symbol, features)
                    )

                    # 앙상블 예측 계산
                    ensemble_pred = (
                        dynamic_universal_weight * universal_pred
                        + dynamic_individual_weight * individual_pred
                    )

                    logger.info(
                        f"🎯 {symbol} 앙상블 예측: Universal({dynamic_universal_weight:.3f}) + Individual({dynamic_individual_weight:.3f}) = {ensemble_pred:.4f}"
                    )
                    return ensemble_pred

                except Exception as e:
                    logger.warning(f"앙상블 조합 실패: {e}, 개별 모델 예측 사용")
                    return individual_pred

            elif individual_pred is not None:
                # 개별 모델 예측이 성공한 경우 우선 사용
                logger.info(f"✅ {symbol} 개별 모델 예측 사용: {individual_pred}")
                return individual_pred
            elif universal_pred is not None:
                # 개별 모델이 없으면 통합 모델 사용
                logger.info(f"🌐 {symbol} 통합 모델 예측 사용: {universal_pred}")
                return universal_pred
            else:
                # 모든 예측이 실패한 경우 기본값 반환
                logger.warning(f"⚠️ {symbol} 모든 예측 실패, 기본값 반환")
                # 멀티타겟 형식으로 기본값 반환 (이전 결과와 유사한 값)
                return {
                    "target_22d": 0.05,  # 5% 예상 수익률
                    "target_66d": 0.15,  # 15% 예상 수익률
                    "sigma_22d": 0.02,  # 2% 변동성
                    "sigma_66d": 0.03,  # 3% 변동성
                }

        except Exception as e:
            logger.error(f"앙상블 예측 실패: {e}")
            return None

    def _predict_with_model(
        self, model: nn.Module, X: np.ndarray
    ) -> Union[float, Dict[str, float]]:
        """
        개별 모델로 예측 수행
        """
        try:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)
                predictions = outputs.numpy()

                # 최근 예측값 사용
                latest_pred = predictions[-1]

                # 멀티타겟인 경우 딕셔너리로 변환
                if len(latest_pred.shape) > 0 and len(latest_pred) > 1:
                    if self.target_columns:
                        return {
                            col: float(val)
                            for col, val in zip(self.target_columns, latest_pred)
                        }
                    else:
                        return {
                            f"target_{i}": float(val)
                            for i, val in enumerate(latest_pred)
                        }
                else:
                    return float(latest_pred)

        except Exception as e:
            logger.error(f"모델 예측 실패: {e}")
            return None

    def save_model(self, filepath: str) -> bool:
        """
        모델 저장
        """
        try:
            # PyTorch 모델 저장
            if self.universal_model is not None:
                torch.save(
                    self.universal_model.state_dict(),
                    f"{filepath}_pytorch_universal.pth",
                )

            # 개별 모델 저장
            for symbol, model in self.individual_models.items():
                if model is not None:
                    torch.save(
                        model.state_dict(),
                        f"{filepath}_pytorch_individual_{symbol}.pth",
                    )

            # 가중치 학습기 저장
            if self.weight_learner is not None:
                torch.save(
                    self.weight_learner.state_dict(),
                    f"{filepath}_pytorch_weight_learner.pth",
                )
                logger.info(
                    f"가중치 학습기 저장 완료: {filepath}_pytorch_weight_learner.pth"
                )

            # 피처 정보 저장
            if self.feature_info:
                feature_info_path = f"{filepath}_feature_info.json"
                with open(feature_info_path, "w", encoding="utf-8") as f:
                    json.dump(self.feature_info, f, indent=2, ensure_ascii=False)
                logger.info(f"피처 정보 저장 완료: {feature_info_path}")

            # 기타 정보 저장
            model_data = {
                "universal_scaler": self.universal_scaler,
                "individual_scalers": self.individual_scalers,
                "weight_learner_scaler": self.weight_learner_scaler,
                "feature_names": self.feature_names,
                "target_columns": self.target_columns,
                "config": self.config,
                "is_fitted": self.enable_individual_models,  # 개별 모델 학습 여부 저장
                "feature_info": self.feature_info,  # 피처 정보도 함께 저장
                "ensemble_config": self.ensemble_config,  # 앙상블 설정 저장
                "weight_training_data": self.weight_training_data,  # 가중치 학습 데이터 저장
            }

            joblib.dump(model_data, f"{filepath}_meta.pkl")

            logger.info(f"신경망 모델 저장 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        모델 로드
        """
        try:
            # 메타 데이터 로드
            meta_path = f"{filepath}_meta.pkl"
            if os.path.exists(meta_path):
                model_data = joblib.load(meta_path)

                self.universal_scaler = model_data.get("universal_scaler")
                self.individual_scalers = model_data.get("individual_scalers", {})
                self.weight_learner_scaler = model_data.get("weight_learner_scaler")
                self.feature_names = model_data.get("feature_names")
                self.target_columns = model_data.get("target_columns")
                self.ensemble_config = model_data.get(
                    "ensemble_config", self.ensemble_config
                )
                self.weight_training_data = model_data.get(
                    "weight_training_data", self.weight_training_data
                )

                logger.info("메타 데이터 로드 완료")
            else:
                logger.warning(f"메타 데이터 파일이 없습니다: {meta_path}")

            # 통합 모델 로드
            universal_path = f"{filepath}_pytorch_universal.pth"
            if os.path.exists(universal_path):
                try:
                    # 저장된 모델의 실제 차원 확인
                    universal_state_dict = torch.load(
                        universal_path, map_location=torch.device("cpu")
                    )
                    actual_input_dim = universal_state_dict["network.0.weight"].shape[1]
                    actual_output_dim = universal_state_dict["network.6.weight"].shape[
                        0
                    ]

                    logger.info(
                        f"통합 모델 차원: 입력 {actual_input_dim}, 출력 {actual_output_dim}"
                    )

                    # 저장된 차원으로 모델 재생성
                    self.universal_model = self._build_model(
                        actual_input_dim, actual_output_dim
                    )

                    # 가중치 로드
                    self.universal_model.load_state_dict(universal_state_dict)
                    self.universal_model.eval()
                    logger.info("통합 모델 로드 완료")
                except Exception as e:
                    logger.error(f"통합 모델 로드 실패: {e}")

            # 개별 모델들 로드
            for symbol in self.individual_scalers.keys():
                individual_path = f"{filepath}_pytorch_individual_{symbol}.pth"
                if os.path.exists(individual_path):
                    try:
                        # 저장된 모델의 실제 차원 확인
                        individual_state_dict = torch.load(
                            individual_path, map_location=torch.device("cpu")
                        )
                        individual_input_dim = individual_state_dict[
                            "network.0.weight"
                        ].shape[1]
                        individual_output_dim = individual_state_dict[
                            "network.6.weight"
                        ].shape[0]

                        logger.info(
                            f"{symbol} 개별 모델 차원: 입력 {individual_input_dim}, 출력 {individual_output_dim}"
                        )

                        # 저장된 차원으로 모델 재생성
                        model = self._build_model(
                            individual_input_dim, individual_output_dim
                        )
                        model.load_state_dict(individual_state_dict)
                        model.eval()
                        self.individual_models[symbol] = model
                        logger.info(f"{symbol} 개별 모델 로드 완료")
                    except Exception as e:
                        logger.error(f"{symbol} 개별 모델 로드 실패: {e}")

            # 가중치 학습기 로드
            weight_learner_path = f"{filepath}_pytorch_weight_learner.pth"
            if os.path.exists(weight_learner_path):
                try:
                    # 가중치 학습기 구조 재생성 (입력 크기는 앙상블 입력 피처에 따라 결정)
                    input_size = (
                        20  # 기본값 (실제로는 앙상블 입력 피처 크기에 따라 결정)
                    )
                    self.weight_learner = EnsembleWeightLearner(input_size)

                    # 가중치 로드
                    self.weight_learner.load_state_dict(torch.load(weight_learner_path))
                    self.weight_learner.eval()
                    logger.info("가중치 학습기 로드 완료")
                except Exception as e:
                    logger.error(f"가중치 학습기 로드 실패: {e}")

            # 피처 정보 로드
            feature_info_path = f"{filepath}_feature_info.json"
            if os.path.exists(feature_info_path):
                try:
                    with open(feature_info_path, "r", encoding="utf-8") as f:
                        self.feature_info = json.load(f)
                    logger.info("피처 정보 로드 완료")
                except Exception as e:
                    logger.error(f"피처 정보 로드 실패: {e}")

            logger.info(f"신경망 모델 로드 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def _get_default_signal(self, symbol: str) -> Dict:
        """기본 신호 (오류시)"""
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "action": "HOLD",
            "action_strength": 0.5,
            "score": 0.0,
            "confidence": 0.3,
            "position_size": 0.05,
            "execution_priority": 5,
            "timing": {"entry": {"type": "WAIT"}, "exit": {"stop_loss": -0.08}},
            "risk_management": {"risk_level": "MEDIUM"},
            "recommendations": {"primary_recommendation": "분석 오류로 인한 기본 신호"},
        }

    def _split_train_test_data(self, training_data: Dict) -> Tuple[Dict, Dict]:
        """
        Train-test 데이터 분할

        Args:
            training_data: 전체 학습 데이터

        Returns:
            (train_data, test_data) 튜플
        """
        try:
            train_data = {}
            test_data = {}

            for symbol, data in training_data.items():
                features = data["features"]
                target = data["target"]

                # 날짜 기준으로 분할
                total_days = len(features)
                train_end_idx = int(total_days * self.train_ratio)

                # Train set
                train_features = features.iloc[:train_end_idx]
                train_target = (
                    target.iloc[:train_end_idx]
                    if hasattr(target, "iloc")
                    else target[:train_end_idx]
                )

                # Test set
                test_features = features.iloc[train_end_idx:]
                test_target = (
                    target.iloc[train_end_idx:]
                    if hasattr(target, "iloc")
                    else target[train_end_idx:]
                )

                train_data[symbol] = {
                    "features": train_features,
                    "target": train_target,
                }

                test_data[symbol] = {
                    "features": test_features, 
                    "target": test_target,
                    "full_features": features,  # 연속 예측용 전체 피처
                    "train_end_idx": train_end_idx  # 학습 종료 인덱스
                }

                logger.info(
                    f"📊 {symbol} 데이터 분할: Train {len(train_features)}일, Test {len(test_features)}일"
                )

            return train_data, test_data

        except Exception as e:
            logger.error(f"Train-test 분할 실패: {e}")
            return training_data, {}

    def _validate_22d_predictions(self, test_data: Dict) -> Dict:
        """
        22일 예측 검증 수행

        Args:
            test_data: 테스트 데이터

        Returns:
            검증 결과 딕셔너리
        """
        try:
            logger.info(f"🔍 22일 예측 검증 시작 - 총 {len(test_data)}개 종목")
            validation_results = {}

            for symbol, data in test_data.items():
                features = data["features"]
                target = data["target"]

                logger.info(f"🔍 {symbol} 22일 예측 검증 시작...")
                logger.info(f"   📊 features 크기: {features.shape}")
                logger.info(
                    f"   📈 target 크기: {target.shape if hasattr(target, 'shape') else len(target)}"
                )

                predictions = []
                actual_values = []
                dates = []

                # 22일 전부터 예측 시작 (22일 후 예측을 위해)
                for i in range(22, len(features)):
                    # 현재 시점의 피처로 22일 후 예측
                    current_features = features.iloc[: i + 1]  # 현재까지의 데이터

                    try:
                        # 22일 후 예측 수행
                        prediction = self.predict(current_features, symbol)

                        # 예측값 추출 (22일 수익률)
                        if isinstance(prediction, dict):
                            pred_22d = prediction.get("target_22d", 0.0)
                        elif isinstance(prediction, (int, float)):
                            pred_22d = float(prediction)
                        else:
                            logger.warning(
                                f"❌ {symbol} {i}일차 예측값 타입 오류: {type(prediction)}"
                            )
                            continue

                        # 실제 22일 후 값
                        if i + 22 < len(target):
                            # target이 DataFrame인 경우 target_22d 컬럼 선택
                            if isinstance(target, pd.DataFrame):
                                if "target_22d" in target.columns:
                                    actual_22d = target["target_22d"].iloc[i + 22]
                                else:
                                    # target_22d 컬럼이 없으면 첫 번째 컬럼 사용
                                    actual_22d = target.iloc[i + 22, 0]
                            elif hasattr(target, "iloc"):
                                actual_22d = target.iloc[i + 22]
                            else:
                                actual_22d = target[i + 22]

                            # 실제값을 숫자로 변환
                            if isinstance(actual_22d, (pd.Series, pd.DataFrame)):
                                actual_22d = (
                                    actual_22d.iloc[0] if len(actual_22d) > 0 else 0.0
                                )
                            actual_22d = float(actual_22d)

                            predictions.append(pred_22d)
                            actual_values.append(actual_22d)

                            # 날짜 정보 처리
                            if hasattr(features, "index"):
                                current_date = features.index[i]
                                if hasattr(current_date, "strftime"):
                                    date_str = current_date.strftime("%Y-%m-%d")
                                else:
                                    date_str = str(current_date)
                            else:
                                date_str = f"Day_{i}"

                            dates.append(date_str)

                            # Line-by-line 로그 출력
                            logger.info(
                                f"📅 {symbol} {date_str}: 예측 {pred_22d:.4f} vs 실제 {actual_22d:.4f} (차이: {pred_22d - actual_22d:.4f})"
                            )

                    except Exception as e:
                        logger.warning(f"❌ {symbol} {i}일차 예측 실패: {e}")
                        continue

                if predictions and actual_values:
                    # RMSE 계산
                    rmse = self._calculate_test_rmse(predictions, actual_values)

                    validation_results[symbol] = {
                        "rmse": rmse,
                        "predictions": predictions,
                        "actual_values": actual_values,
                        "dates": dates,
                        "num_predictions": len(predictions),
                    }

                    logger.info(
                        f"✅ {symbol} 검증 완료: RMSE = {rmse:.4f} ({len(predictions)}개 예측)"
                    )
                else:
                    logger.warning(f"⚠️ {symbol} 검증 데이터 부족")
                    validation_results[symbol] = {
                        "rmse": float("inf"),
                        "predictions": [],
                        "actual_values": [],
                        "dates": [],
                        "num_predictions": 0,
                    }

            return validation_results

        except Exception as e:
            logger.error(f"22일 예측 검증 실패: {e}")
            return {}

    def _calculate_test_rmse(
        self, predictions: List[float], actuals: List[float]
    ) -> float:
        """
        RMSE 계산

        Args:
            predictions: 예측값 리스트
            actuals: 실제값 리스트

        Returns:
            RMSE 값
        """
        try:
            if len(predictions) != len(actuals) or len(predictions) == 0:
                return float("inf")

            # RMSE 계산
            squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
            mse = sum(squared_errors) / len(squared_errors)
            rmse = mse**0.5

            return rmse

        except Exception as e:
            logger.error(f"RMSE 계산 실패: {e}")
            return float("inf")

    def _train_ensemble_weight_learner(self, training_data: Dict) -> bool:
        """
        앙상블 가중치 학습기 훈련

        Args:
            training_data: 훈련 데이터

        Returns:
            학습 성공 여부
        """
        try:
            if not self.enable_weight_learning:
                logger.info("앙상블 가중치 학습이 비활성화되어 있습니다")
                return True

            logger.info("🎯 앙상블 가중치 학습기 훈련 시작...")

            # 가중치 학습 데이터 생성
            ensemble_inputs, optimal_weights = self._prepare_weight_learning_data(
                training_data
            )

            if len(ensemble_inputs) < self.weight_learning_config.get(
                "min_samples_for_weight_learning", 100
            ):
                logger.warning(
                    f"가중치 학습 데이터 부족: {len(ensemble_inputs)}개 (최소 {self.weight_learning_config.get('min_samples_for_weight_learning', 100)}개 필요)"
                )
                return False

            # 데이터 스케일링
            ensemble_inputs_scaled = self.weight_learner_scaler.fit_transform(
                ensemble_inputs
            )

            # 학습/검증 분할
            X_train, X_val, y_train, y_val = train_test_split(
                ensemble_inputs_scaled,
                optimal_weights,
                test_size=self.weight_learning_config.get("validation_split", 0.2),
                random_state=42,
            )

            # 데이터셋 생성
            train_dataset = EnsembleWeightDataset(X_train, y_train)
            val_dataset = EnsembleWeightDataset(X_val, y_val)

            # 데이터로더 생성
            batch_size = self.weight_learning_config.get("batch_size", 32)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 가중치 학습기 모델 생성
            input_size = ensemble_inputs.shape[1]
            self.weight_learner = EnsembleWeightLearner(input_size)

            # 손실 함수 및 옵티마이저
            criterion = nn.MSELoss()
            learning_rate = self.weight_learning_config.get("learning_rate", 0.001)
            optimizer = optim.Adam(self.weight_learner.parameters(), lr=learning_rate)

            # 학습 파라미터
            epochs = self.weight_learning_config.get("epochs", 100)
            best_val_loss = float("inf")
            patience = 15
            patience_counter = 0

            logger.info(
                f"가중치 학습기 훈련 시작 - Epochs: {epochs}, Input Size: {input_size}"
            )

            # 훈련 루프
            for epoch in range(epochs):
                # 훈련
                self.weight_learner.train()
                train_losses = []
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.weight_learner(X_batch.float())
                    loss = criterion(outputs, y_batch.float())
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                train_loss = np.mean(train_losses)

                # 검증
                self.weight_learner.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.weight_learner(X_batch.float())
                        loss = criterion(outputs, y_batch.float())
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)

                # 로깅
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            print(f"\n✅ 앙상블 가중치 학습기 훈련 완료!")
            print(f"   • 최종 검증 손실: {best_val_loss:.6f}")
            print(f"   • 훈련 에포크: {epochs}회")
            print(f"   • 학습률: {learning_rate}")
            print(f"   • 배치 크기: {batch_size}")
            logger.info(
                f"✅ 앙상블 가중치 학습기 훈련 완료 - Best Val Loss: {best_val_loss:.6f}"
            )
            return True

        except Exception as e:
            logger.error(f"앙상블 가중치 학습기 훈련 실패: {e}")
            return False

    def _prepare_weight_learning_data(
        self, training_data: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        가중치 학습을 위한 데이터 준비

        Args:
            training_data: 훈련 데이터

        Returns:
            (앙상블 입력 피처, 최적 가중치) 튜플
        """
        try:
            ensemble_inputs = []
            optimal_weights = []

            for symbol, data in training_data.items():
                logger.info(f"🔍 {symbol} 가중치 학습 데이터 준비 중...")

                try:
                    features = data["features"]
                    target = data["target"]

                    # 실제 수익률 계산 (타겟이 수익률인 경우)
                    if isinstance(target, pd.DataFrame):
                        actual_returns = target.values
                    elif isinstance(target, pd.Series):
                        actual_returns = target.values
                    else:
                        actual_returns = target

                    # 시계열 데이터의 여러 시점에서 가중치 학습 데이터 생성
                    # 22일 후 예측을 위해 22일 전부터 시작
                    start_idx = 22
                    end_idx = len(features) - 22  # 22일 후 데이터가 있는 지점까지만

                    logger.info(
                        f"📊 {symbol} 시계열 데이터: {len(features)}일, 가중치 학습 구간: {start_idx}~{end_idx}"
                    )

                    for i in range(start_idx, end_idx, 5):  # 5일 간격으로 샘플링
                        try:
                            # 현재 시점까지의 데이터로 예측
                            current_features = features.iloc[: i + 1]
                            current_returns = actual_returns[: i + 1]

                            # 22일 후 실제 수익률 (목표값)
                            if i + 22 < len(actual_returns):
                                future_returns = actual_returns[
                                    i : i + 22
                                ]  # 22일간의 수익률
                            else:
                                continue

                            # 개별 모델 예측 시도
                            individual_pred = None
                            if symbol in self.individual_models:
                                try:
                                    # 개별 모델용 피처 준비
                                    individual_features = current_features.copy()
                                    if symbol in self.individual_scalers:
                                        X_individual = self.individual_scalers[
                                            symbol
                                        ].transform(
                                            self._prepare_prediction_data(
                                                individual_features, 20
                                            )
                                        )
                                        individual_pred = self._predict_with_model(
                                            self.individual_models[symbol], X_individual
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"{symbol} {i}일차 개별 모델 예측 실패: {e}"
                                    )

                            # 통합 모델 예측 시도
                            universal_pred = None
                            if self.universal_model is not None:
                                try:
                                    # 통합 모델용 피처 준비
                                    if self.feature_names is not None:
                                        available_features = [
                                            col
                                            for col in self.feature_names
                                            if col in current_features.columns
                                        ]

                                        if len(available_features) > 0:
                                            features_for_universal = current_features[
                                                available_features
                                            ]

                                            # 부족한 피처는 0으로 채움
                                            if len(available_features) < len(
                                                self.feature_names
                                            ):
                                                missing_features = [
                                                    col
                                                    for col in self.feature_names
                                                    if col not in available_features
                                                ]
                                                for col in missing_features:
                                                    features_for_universal[col] = 0.0

                                            # 피처 순서 맞추기
                                            features_for_universal = (
                                                features_for_universal[
                                                    self.feature_names
                                                ]
                                            )

                                            # 윈도우 데이터 생성
                                            X_universal = self._prepare_prediction_data(
                                                features_for_universal, 20
                                            )

                                            # 차원 검증 및 조정
                                            expected_dim = len(self.feature_names) * 20
                                            actual_dim = (
                                                X_universal.shape[1]
                                                if len(X_universal.shape) > 1
                                                else X_universal.shape[0]
                                            )

                                            if actual_dim != expected_dim:
                                                if actual_dim < expected_dim:
                                                    padding = np.zeros(
                                                        (
                                                            X_universal.shape[0],
                                                            expected_dim - actual_dim,
                                                        )
                                                    )
                                                    X_universal = np.hstack(
                                                        [X_universal, padding]
                                                    )
                                                else:
                                                    X_universal = X_universal[
                                                        :, :expected_dim
                                                    ]

                                            X_universal_scaled = (
                                                self.universal_scaler.transform(
                                                    X_universal
                                                )
                                            )
                                            universal_pred = self._predict_with_model(
                                                self.universal_model, X_universal_scaled
                                            )
                                except Exception as e:
                                    logger.debug(
                                        f"{symbol} {i}일차 통합 모델 예측 실패: {e}"
                                    )

                            # 메타 피처 생성 (순환 의존성 제거)
                            meta_features = self._create_meta_features_for_weight_learning(
                                current_features, symbol
                            )

                            if meta_features is not None and len(meta_features) == 20:
                                # 최적 가중치 계산 (예측값 기반)
                                optimal_weight = self._calculate_optimal_weights(
                                    individual_pred, universal_pred, future_returns
                                )

                                if optimal_weight is not None and len(optimal_weight) == 2:
                                    ensemble_inputs.append(meta_features)
                                    optimal_weights.append(optimal_weight)

                        except Exception as e:
                            logger.debug(
                                f"{symbol} {i}일차 가중치 학습 데이터 생성 실패: {e}"
                            )
                            continue

                except Exception as e:
                    logger.error(f"{symbol} 가중치 학습 데이터 준비 실패: {e}")
                    continue

            logger.info(f"📊 생성된 가중치 학습 데이터: {len(ensemble_inputs)}개")

            if not ensemble_inputs:
                logger.error("가중치 학습 데이터가 생성되지 않았습니다")
                return np.array([]), np.array([])

            return np.array(ensemble_inputs), np.array(optimal_weights)

        except Exception as e:
            logger.error(f"가중치 학습 데이터 준비 실패: {e}")
            return np.array([]), np.array([])

    def _create_ensemble_input_features(
        self,
        features: pd.DataFrame,
        symbol: str,
        individual_pred: Optional[float],
        universal_pred: Optional[float],
        actual_returns: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        앙상블 가중치 학습을 위한 입력 피처 생성

        Args:
            features: 피처 데이터
            symbol: 종목 심볼
            individual_pred: 개별 모델 예측값
            universal_pred: 통합 모델 예측값
            actual_returns: 실제 수익률

        Returns:
            앙상블 입력 피처 (20차원)
        """
        try:
            # 예측값을 숫자로 변환
            if isinstance(individual_pred, dict):
                individual_pred = individual_pred.get("target_22d", 0.0)
            elif individual_pred is None:
                individual_pred = 0.0

            if isinstance(universal_pred, dict):
                universal_pred = universal_pred.get("target_22d", 0.0)
            elif universal_pred is None:
                universal_pred = 0.0

            # 예측값 차이
            pred_diff = abs(individual_pred - universal_pred)

            # 예측값 크기
            individual_magnitude = abs(individual_pred)
            universal_magnitude = abs(universal_pred)

            # 예측값 비율
            if universal_magnitude > 0:
                pred_ratio = individual_magnitude / universal_magnitude
            else:
                pred_ratio = 1.0

            # 실제 수익률 통계
            if len(actual_returns) > 0:
                actual_mean = np.mean(actual_returns)
                actual_std = np.std(actual_returns)
                actual_min = np.min(actual_returns)
                actual_max = np.max(actual_returns)
            else:
                actual_mean = actual_std = actual_min = actual_max = 0.0

            # 피처 통계 (최근 20일)
            recent_features = features.tail(20)
            if len(recent_features) > 0:
                feature_mean = recent_features.mean().mean()
                feature_std = recent_features.std().mean()
                feature_min = recent_features.min().min()
                feature_max = recent_features.max().max()
            else:
                feature_mean = feature_std = feature_min = feature_max = 0.0

            # 시장 상황 피처 (기본값)
            market_volatility = 0.02  # 기본 변동성
            market_trend = 0.0  # 기본 트렌드
            regime_stability = 0.5  # 기본 안정성

            # 종목 해시 생성
            symbol_hash = hash(symbol) % 1000 / 1000.0  # 0-1 범위로 정규화

            # 20차원 앙상블 입력 피처 생성
            ensemble_features = np.array(
                [
                    individual_pred,  # 1. 개별 모델 예측값
                    universal_pred,  # 2. 통합 모델 예측값
                    pred_diff,  # 3. 예측값 차이
                    individual_magnitude,  # 4. 개별 예측 크기
                    universal_magnitude,  # 5. 통합 예측 크기
                    pred_ratio,  # 6. 예측값 비율
                    actual_mean,  # 7. 실제 수익률 평균
                    actual_std,  # 8. 실제 수익률 표준편차
                    actual_min,  # 9. 실제 수익률 최소값
                    actual_max,  # 10. 실제 수익률 최대값
                    feature_mean,  # 11. 피처 평균
                    feature_std,  # 12. 피처 표준편차
                    feature_min,  # 13. 피처 최소값
                    feature_max,  # 14. 피처 최대값
                    market_volatility,  # 15. 시장 변동성
                    market_trend,  # 16. 시장 트렌드
                    regime_stability,  # 17. 체제 안정성
                    len(features),  # 18. 데이터 길이
                    symbol_hash,  # 19. 종목 해시 (고정값)
                    1.0,  # 20. 바이어스 항
                ]
            )

            return ensemble_features

        except Exception as e:
            logger.error(f"앙상블 입력 피처 생성 실패: {e}")
            return None

    def _calculate_optimal_weights(
        self,
        individual_pred: Optional[float],
        universal_pred: Optional[float],
        actual_returns: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        실제 수익률을 기반으로 최적 앙상블 가중치 계산

        Args:
            individual_pred: 개별 모델 예측값
            universal_pred: 통합 모델 예측값
            actual_returns: 실제 수익률

        Returns:
            최적 가중치 [universal_weight, individual_weight]
        """
        try:
            # 예측값을 숫자로 변환
            if isinstance(individual_pred, dict):
                individual_pred = individual_pred.get("target_22d", 0.0)
            elif individual_pred is None:
                individual_pred = 0.0

            if isinstance(universal_pred, dict):
                universal_pred = universal_pred.get("target_22d", 0.0)
            elif universal_pred is None:
                universal_pred = 0.0

            # 실제 수익률이 없으면 기본 가중치 반환
            if len(actual_returns) == 0:
                return np.array([0.7, 0.3])  # 기본 가중치

            # 실제 수익률 평균
            actual_return = np.mean(actual_returns)

            # 각 모델의 예측 오차 계산
            individual_error = abs(individual_pred - actual_return)
            universal_error = abs(universal_pred - actual_return)

            # 오차의 역수를 가중치로 사용 (오차가 작을수록 높은 가중치)
            total_error = individual_error + universal_error
            if total_error > 0:
                individual_weight = universal_error / total_error
                universal_weight = individual_error / total_error
            else:
                # 오차가 0이면 기본 가중치 사용
                individual_weight = 0.3
                universal_weight = 0.7

            # 가중치 정규화 (합이 1이 되도록)
            total_weight = individual_weight + universal_weight
            individual_weight /= total_weight
            universal_weight /= total_weight

            return np.array([universal_weight, individual_weight])

        except Exception as e:
            logger.error(f"최적 가중치 계산 실패: {e}")
            return np.array([0.7, 0.3])  # 기본 가중치 반환

    def _update_ensemble_weights(
        self, symbol: str, features: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        동적으로 앙상블 가중치 업데이트 (개선된 버전)

        Args:
            symbol: 종목 심볼
            features: 종목 피처

        Returns:
            (universal_weight, individual_weight) 튜플
        """
        try:
            if self.weight_learner is None:
                logger.warning(f"{symbol} 가중치 학습기가 없어 기본값 사용")
                return self.universal_weight, self.individual_weight

            # 메타 피처 생성 (예측값 제외, 시장 상황 기반)
            meta_features = self._create_meta_features_for_weight_learning(features, symbol)
            
            if meta_features is not None and len(meta_features) > 0:
                try:
                    # 가중치 학습기로 동적 가중치 예측
                    self.weight_learner.eval()
                    with torch.no_grad():
                        # 차원 맞추기
                        if len(meta_features.shape) == 1:
                            meta_features = meta_features.reshape(1, -1)
                        
                        # 스케일링
                        X_scaled = self.weight_learner_scaler.transform(meta_features)
                        X_tensor = torch.FloatTensor(X_scaled)
                        
                        # 가중치 예측
                        weights = self.weight_learner(X_tensor)
                        weights_np = weights.numpy()[0]

                        dynamic_universal_weight = float(weights_np[0])
                        dynamic_individual_weight = float(weights_np[1])

                        # 유효성 검증
                        if dynamic_universal_weight < 0 or dynamic_individual_weight < 0:
                            logger.warning(f"{symbol} 음수 가중치 감지, 기본값 사용")
                            return self.universal_weight, self.individual_weight

                        # 가중치 정규화 (합이 1이 되도록)
                        total_weight = dynamic_universal_weight + dynamic_individual_weight
                        if total_weight > 0:
                            dynamic_universal_weight /= total_weight
                            dynamic_individual_weight /= total_weight
                        else:
                            return self.universal_weight, self.individual_weight

                        logger.info(
                            f"🎯 {symbol} 동적 가중치: Universal={dynamic_universal_weight:.3f}, Individual={dynamic_individual_weight:.3f}"
                        )

                        return dynamic_universal_weight, dynamic_individual_weight

                except Exception as e:
                    logger.error(f"{symbol} 가중치 학습기 실행 실패: {e}")
                    return self.universal_weight, self.individual_weight
            
            # 메타 피처 생성 실패시 기본값
            logger.warning(f"{symbol} 메타 피처 생성 실패, 기본값 사용")
            return self.universal_weight, self.individual_weight

        except Exception as e:
            logger.error(f"앙상블 가중치 업데이트 실패: {e}")
            return self.universal_weight, self.individual_weight

    def _create_meta_features_for_weight_learning(
        self, features: pd.DataFrame, symbol: str
    ) -> Optional[np.ndarray]:
        """
        가중치 학습기를 위한 메타 피처 생성 (순환 의존성 제거)
        
        Args:
            features: 종목 피처 데이터
            symbol: 종목 심볼
            
        Returns:
            메타 피처 배열 (예측값 의존성 없음)
        """
        try:
            # 최근 20일 데이터만 사용
            recent_features = features.tail(20)
            if len(recent_features) < 10:
                logger.warning(f"{symbol} 메타 피처 생성용 데이터 부족")
                return None
            
            meta_features = []
            
            # 1. 시장 변동성 지표
            if 'volatility_regime' in recent_features.columns:
                volatility_level = recent_features['volatility_regime'].mean()
                meta_features.append(volatility_level)
            else:
                meta_features.append(0.0)
            
            # 2. 추세 강도
            if 'trend_strength' in recent_features.columns:
                trend_strength = recent_features['trend_strength'].mean()
                meta_features.append(trend_strength)
            else:
                meta_features.append(0.0)
            
            # 3. 모멘텀 다이버전스
            if 'momentum_divergence' in recent_features.columns:
                divergence_level = recent_features['momentum_divergence'].mean()
                meta_features.append(divergence_level)
            else:
                meta_features.append(0.0)
            
            # 4. 변동성 스큐
            if 'volatility_skew' in recent_features.columns:
                skew_level = recent_features['volatility_skew'].mean()
                meta_features.append(skew_level)
            else:
                meta_features.append(0.0)
            
            # 5. 지지/저항 강도
            if 'support_resistance' in recent_features.columns:
                support_resistance = recent_features['support_resistance'].mean()
                meta_features.append(support_resistance)
            else:
                meta_features.append(0.0)
            
            # 6. 시장 미시구조
            if 'market_microstructure' in recent_features.columns:
                microstructure = recent_features['market_microstructure'].mean()
                meta_features.append(microstructure)
            else:
                meta_features.append(0.0)
            
            # 7. 볼륨-가격 추세
            if 'volume_price_trend' in recent_features.columns:
                vpt = recent_features['volume_price_trend'].mean()
                meta_features.append(vpt)
            else:
                meta_features.append(0.0)
            
            # 8. 가격-볼륨 오실레이터
            if 'price_volume_oscillator' in recent_features.columns:
                pvo = recent_features['price_volume_oscillator'].mean()
                meta_features.append(pvo)
            else:
                meta_features.append(0.0)
            
            # 9-12. 시장 체제 원핫 인코딩
            regime_features = []
            for regime in ['bullish', 'bearish', 'sideways', 'volatile']:
                regime_col = f'regime_{regime}'
                if regime_col in recent_features.columns:
                    regime_value = recent_features[regime_col].iloc[-1]  # 최신 체제
                    regime_features.append(regime_value)
                else:
                    regime_features.append(0.0)
            meta_features.extend(regime_features)
            
            # 13-20. 기본 기술적 지표 통계
            technical_indicators = [
                'dual_momentum', 'volatility_breakout', 'swing_ema', 'swing_rsi',
                'swing_donchian', 'stoch_donchian', 'whipsaw_prevention', 'swing_macd'
            ]
            
            for indicator in technical_indicators:
                if indicator in recent_features.columns:
                    # 최근 값과 과거 평균의 차이
                    recent_mean = recent_features[indicator].tail(5).mean()
                    historical_mean = recent_features[indicator].mean()
                    diff = recent_mean - historical_mean
                    meta_features.append(diff)
                else:
                    meta_features.append(0.0)
            
            # 최종 20차원으로 맞추기
            if len(meta_features) < 20:
                meta_features.extend([0.0] * (20 - len(meta_features)))
            elif len(meta_features) > 20:
                meta_features = meta_features[:20]
            
            return np.array(meta_features)
            
        except Exception as e:
            logger.error(f"{symbol} 메타 피처 생성 실패: {e}")
            return None


def main():
    """명령행 인터페이스"""
    import argparse
    import json
    import os
    import sys
    import glob
    from pathlib import Path

    parser = argparse.ArgumentParser(description="신경망 개별 종목 예측기")
    parser.add_argument("--train", action="store_true", help="모델 학습")
    parser.add_argument("--force", action="store_true", help="강제 재학습")
    parser.add_argument(
        "--data-dir", type=str, default="data/trader", help="종목 데이터 디렉토리"
    )
    parser.add_argument(
        "--config", type=str, default="config/config_trader.json", help="설정 파일"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/trader", help="모델 저장 디렉토리"
    )
    parser.add_argument("--predict", action="store_true", help="종목 예측")
    parser.add_argument("--symbol", type=str, help="예측할 종목 코드")
    parser.add_argument("--experiment", action="store_true", help="다양한 신경망 구조 실험")
    parser.add_argument(
        "--experiment-config", 
        type=str, 
        default="config/neural_experiments.json", 
        help="실험 설정 파일"
    )

    args = parser.parse_args()

    # 설정 파일 로드
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)

    # 신경망 모델 초기화
    neural_predictor = StockPredictionNetwork(config)

    # 모델 디렉토리 생성
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "neural_predictor"
    meta_path = str(model_path) + "_meta.pkl"

    if args.train:
        print("🧠 신경망 개별 종목 예측 모델 학습 시작")

        # --force 옵션이 있으면 무조건 재학습
        if args.force:
            print("⚡ 강제 재학습 모드 - 기존 모델 무시하고 새로 학습")
        else:
            # 기존 모델 확인
            if os.path.exists(meta_path):
                try:
                    if neural_predictor.load_model(str(model_path)):
                        print("✅ 기존 모델 로드 완료")
                        print(
                            "📊 기존 모델 사용 - 앙상블 가중치 학습 및 22일 검증은 건너뜀"
                        )
                        return
                except Exception as e:
                    print(f"⚠️  기존 모델 로드 실패: {e}")
                    print("🔄 새로 학습을 진행합니다")

        # 종목 데이터 로드
        print(f"📊 종목 데이터 로드: {args.data_dir}")
        trader_files = glob.glob(f"{args.data_dir}/*.csv")

        if not trader_files:
            print("❌ 종목 데이터를 찾을 수 없습니다.")
            print(f"  확인 경로: {args.data_dir}")
            sys.exit(1)

        # 모든 종목 데이터 로드
        training_data = {}

        # 매크로 데이터 로드
        print("📊 매크로 데이터 로드 중...")
        macro_data = None
        try:
            # data/macro 디렉토리에서 모든 매크로 CSV 파일 로드
            macro_dir = Path("data/macro")
            if macro_dir.exists():
                macro_files = list(macro_dir.glob("*.csv"))
                if macro_files:
                    print(f"📈 매크로 파일 {len(macro_files)}개 발견")
                    macro_data_list = []

                    # 중복 파일 처리: _data.csv 파일 우선, 그 다음 일반 .csv 파일
                    processed_symbols = set()

                    # 1단계: _data.csv 파일들 먼저 처리
                    for macro_file in macro_files:
                        if macro_file.name.endswith("_data.csv"):
                            try:
                                df = pd.read_csv(macro_file)
                                if not df.empty and "datetime" in df.columns:
                                    # 파일명에서 심볼 추출
                                    symbol = macro_file.stem.replace(
                                        "_data", ""
                                    ).replace("^", "")

                                    if symbol not in processed_symbols:
                                        df["macro_symbol"] = symbol
                                        df["datetime"] = pd.to_datetime(df["datetime"])
                                        df = df.fillna(method="ffill").fillna(
                                            method="bfill"
                                        )
                                        macro_data_list.append(df)
                                        processed_symbols.add(symbol)
                                        print(
                                            f"   ✅ {symbol}: {len(df)}행 로드 ({macro_file.name})"
                                        )
                            except Exception as e:
                                print(f"   ⚠️ {macro_file.name} 로드 실패: {e}")

                    # 2단계: 일반 .csv 파일들 처리 (이미 처리된 심볼 제외)
                    for macro_file in macro_files:
                        if not macro_file.name.endswith("_data.csv"):
                            try:
                                df = pd.read_csv(macro_file)
                                if not df.empty and "datetime" in df.columns:
                                    # 파일명에서 심볼 추출
                                    symbol = macro_file.stem.replace("^", "")

                                    if symbol not in processed_symbols:
                                        df["macro_symbol"] = symbol
                                        df["datetime"] = pd.to_datetime(df["datetime"])
                                        df = df.fillna(method="ffill").fillna(
                                            method="bfill"
                                        )
                                        macro_data_list.append(df)
                                        processed_symbols.add(symbol)
                                        print(
                                            f"   ✅ {symbol}: {len(df)}행 로드 ({macro_file.name})"
                                        )
                            except Exception as e:
                                print(f"   ⚠️ {macro_file.name} 로드 실패: {e}")

                    if macro_data_list:
                        # 모든 매크로 데이터 병합
                        macro_data = pd.concat(macro_data_list, ignore_index=True)
                        print(f"📊 매크로 데이터 병합 완료: {macro_data.shape}")

                        # 공통 datetime 기준으로 정렬
                        macro_data = macro_data.sort_values("datetime")
                        print(
                            f"📅 매크로 데이터 datetime 범위: {macro_data['datetime'].min()} ~ {macro_data['datetime'].max()}"
                        )
                    else:
                        print("⚠️ 유효한 매크로 데이터가 없습니다")
                else:
                    print("⚠️ 매크로 파일을 찾을 수 없습니다")
            else:
                print("⚠️ data/macro 디렉토리가 없습니다")
        except Exception as e:
            print(f"❌ 매크로 데이터 로드 실패: {e}")

        for file_path in trader_files:
            symbol = os.path.basename(file_path).split("_")[0]  # 파일명에서 종목명 추출
            print(f"📈 {symbol} 데이터 로드 중...")

            try:
                stock_data = pd.read_csv(file_path)
                if len(stock_data) < 50:
                    print(f"⚠️  {symbol}: 데이터 부족 ({len(stock_data)}행), 건너뜀")
                    continue

                # 피처 생성 (매크로 데이터 포함)
                features = neural_predictor.create_features(
                    stock_data,
                    symbol,
                    {"regime": "NEUTRAL", "confidence": 0.5},
                    macro_data,
                )

                # 피처 차원 로깅 (간소화)
                if features is not None:
                    lookback_days = (
                        config.get("neural_network", {})
                        .get("features", {})
                        .get("lookback_days", 20)
                    )
                    input_dim = len(features.columns) * lookback_days
                    print(
                        f"   📊 {symbol}: {len(features.columns)}차원 → {input_dim}차원 입력 (lookback: {lookback_days}일)"
                    )
                else:
                    print(f"   ❌ {symbol}: 피처 생성 실패")

                # 타겟 생성 (22일, 66일 후 수익률 + 각각의 표준편차)
                target_forward_days = config.get("neural_network", {}).get(
                    "target_forward_days", [22, 66]
                )
                if isinstance(target_forward_days, list):
                    targets = {}
                    for days in target_forward_days:
                        # 미래 수익률 계산
                        future_returns = (
                            stock_data["close"].pct_change(days).shift(-days)
                        )
                        # NaN 처리: ffill과 bfill 사용
                        future_returns = future_returns.fillna(method="ffill").fillna(
                            method="bfill"
                        )
                        targets[f"target_{days}d"] = future_returns

                        # 해당 기간의 수익률 표준편차 계산
                        rolling_returns = stock_data["close"].pct_change()
                        rolling_std = rolling_returns.rolling(
                            window=days, min_periods=1
                        ).std()
                        # NaN 처리: ffill과 bfill 사용
                        rolling_std = rolling_std.fillna(method="ffill").fillna(
                            method="bfill"
                        )
                        targets[f"sigma_{days}d"] = rolling_std

                    target = pd.DataFrame(targets)
                else:
                    # 단일 타겟 (22일 수익률 + 표준편차)
                    future_returns = stock_data["close"].pct_change(22).shift(-22)
                    future_returns = future_returns.fillna(method="ffill").fillna(
                        method="bfill"
                    )

                    rolling_returns = stock_data["close"].pct_change()
                    rolling_std = rolling_returns.rolling(
                        window=22, min_periods=1
                    ).std()
                    rolling_std = rolling_std.fillna(method="ffill").fillna(
                        method="bfill"
                    )

                    target = pd.DataFrame(
                        {"target_22d": future_returns, "sigma_22d": rolling_std}
                    )

                training_data[symbol] = {
                    "features": features,
                    "target": target,
                    "data": stock_data,
                }

                print(f"✅ {symbol} 데이터 로드 완료: {len(stock_data)}행")

            except Exception as e:
                print(f"❌ {symbol} 데이터 로드 실패: {e}")
                continue

        if not training_data:
            print("❌ 학습 가능한 종목 데이터가 없습니다.")
            sys.exit(1)

        print(f"📊 총 {len(training_data)}개 종목 데이터 로드 완료")
        print(f"📈 종목들: {list(training_data.keys())}")

        # 타겟 정보 한 번만 표시 (첫 번째 종목 기준)
        if training_data:
            first_symbol = list(training_data.keys())[0]
            first_target = training_data[first_symbol]["target"]
            print(
                f"📊 타겟 구조: {list(first_target.columns)} (크기: {first_target.shape})"
            )

            # 피처 차원 요약 (간소화)
            lookback_days = (
                config.get("neural_network", {})
                .get("features", {})
                .get("lookback_days", 20)
            )
            feature_summary = []
            for symbol, data in training_data.items():
                features = data["features"]
                if features is not None:
                    feature_summary.append(f"{symbol}:{len(features.columns)}")

            print(
                f"\n🔍 피처 요약: {' | '.join(feature_summary)} (lookback: {lookback_days}일)"
            )

        # 신경망 학습
        print("🧠 신경망 학습 시작...")
        print(
            f"📊 enable_individual_models: {neural_predictor.enable_individual_models}"
        )
        print(f"📈 학습할 종목 수: {len(training_data)}")

        if neural_predictor.fit(training_data):
            print("✅ 신경망 학습 성공")
            neural_predictor.save_model(str(model_path))
            print("✅ 모델 저장 완료")
        else:
            print("❌ 신경망 학습 실패")
            sys.exit(1)

    elif args.predict:
        print("🔮 종목 예측")

        if not args.symbol:
            print("❌ --symbol 인자가 필요합니다.")
            sys.exit(1)

        # 모델 로드
        if not os.path.exists(meta_path):
            print("❌ 학습된 모델이 없습니다. 먼저 --train 옵션으로 학습하세요.")
            sys.exit(1)

        if not neural_predictor.load_model(str(model_path)):
            print("❌ 모델 로드 실패")
            sys.exit(1)

        # 종목 데이터 로드
        symbol_file = f"{args.data_dir}/{args.symbol}_*.csv"
        symbol_files = glob.glob(symbol_file)

        if not symbol_files:
            print(f"❌ {args.symbol} 데이터를 찾을 수 없습니다.")
            sys.exit(1)

        stock_data = pd.read_csv(symbol_files[0])
        print(f"✅ {args.symbol} 데이터 로드 완료: {len(stock_data)}행")

        # 예측 실행 (간단한 피처 생성)
        features = pd.DataFrame(
            {
                "close": stock_data["close"],
                "volume": stock_data["volume"],
                "high": stock_data["high"],
                "low": stock_data["low"],
            }
        )

        prediction = neural_predictor.predict(features, args.symbol)
        
        # 최종 시점 기준 예측 결과 상세 출력
        current_date = features.index[-1] if hasattr(features, 'index') else "현재"
        print("\n" + "="*60)
        print(f"🔮 {args.symbol} 최종 예측 결과 (기준일: {current_date})")
        print("="*60)
        
        if isinstance(prediction, dict):
            for key, value in prediction.items():
                if 'target_22d' in key:
                    print(f"🎯 22일 후 예상 수익률: {value:.4f} ({value*100:.2f}%)")
                elif 'target_66d' in key:
                    print(f"🎯 66일 후 예상 수익률: {value:.4f} ({value*100:.2f}%)")
                else:
                    print(f"   {key}: {value:.4f}")
        else:
            print(f"🎯 22일 후 예상 수익률: {prediction:.4f} ({prediction*100:.2f}%)")
        
        print("="*60 + "\n")

    elif args.experiment:
        print("\n" + "="*70)
        print("🧪 신경망 구조 실험 모드")
        print("="*70)
        
        # 실험을 위한 함수 임포트
        from .neural_experiment import run_neural_experiments
        
        # 실험 실행
        experiment_results = run_neural_experiments(
            config_path=args.config,
            experiment_config_path=args.experiment_config,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            force_retrain=args.force
        )
        
        # 결과 출력
        print("\n🏆 실험 결과 요약:")
        print("="*70)
        for symbol, results in experiment_results.items():
            print(f"\n📊 {symbol}:")
            for model_name, performance in results.items():
                print(f"   - {model_name}: RMSE = {performance['rmse']:.4f}")
        
    else:
        print("사용법:")
        print("  --train --data-dir data/trader     # 모델 학습")
        print("  --predict --symbol AAPL            # 종목 예측")
        print("  --experiment                       # 다양한 신경망 구조 실험")
        print("  --force                            # 강제 재학습")


if __name__ == "__main__":
    main()
