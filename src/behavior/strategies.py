import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import os
import json
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")

# calculate_index에서 import
from .calculate_index import StrategyParams, TechnicalIndicators

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """전략 기본 클래스"""

    def __init__(self, params: StrategyParams):
        self.params = params
        self.positions = []
        self.trades = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 신호 생성 (하위 클래스에서 구현)"""
        pass

    def calculate_position_size(self, df: pd.DataFrame, signal: int) -> float:
        """포지션 사이즈 계산"""
        # 기본 포지션 사이즈 (변동성 조정)
        base_size = 1.0

        if "volatility" in df.columns and not df["volatility"].isna().all():
            current_vol = df["volatility"].iloc[-1]
            if current_vol > 0:
                # 변동성 타겟에 맞춰 포지션 사이즈 조정
                vol_adjustment = self.params.volatility_target / current_vol
                base_size *= min(vol_adjustment, 2.0)  # 최대 2배로 제한

        return base_size * signal

    def calculate_stop_loss(
        self, df: pd.DataFrame, entry_price: float, position: int
    ) -> float:
        """손절가 계산 (ATR 기반)"""
        if "atr" in df.columns and not df["atr"].isna().all():
            atr = df["atr"].iloc[-1]
            if position > 0:  # 롱 포지션
                return entry_price - (self.params.stop_loss_atr_multiplier * atr)
            else:  # 숏 포지션
                return entry_price + (self.params.stop_loss_atr_multiplier * atr)
        return None

    def calculate_take_profit(
        self, df: pd.DataFrame, entry_price: float, position: int
    ) -> float:
        """익절가 계산 (ATR 기반)"""
        if "atr" in df.columns and not df["atr"].isna().all():
            atr = df["atr"].iloc[-1]
            if position > 0:  # 롱 포지션
                return entry_price + (self.params.take_profit_atr_multiplier * atr)
            else:  # 숏 포지션
                return entry_price - (self.params.take_profit_atr_multiplier * atr)
        return None


class DualMomentumStrategy(BaseStrategy):
    """추세⇄평균회귀 듀얼 모멘텀 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Donchian Channel 하이퍼파라미터
        self.donchian_period = getattr(params, "donchian_period", 20)
        # RSI 하이퍼파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)
        # 모멘텀 하이퍼파라미터
        self.momentum_period = getattr(params, "momentum_period", 10)
        self.momentum_threshold = getattr(params, "momentum_threshold", 0.02)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """듀얼 모멘텀 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # Donchian Channel 계산 (하이퍼파라미터 적용)
        if "donchian_upper" not in df.columns or "donchian_lower" not in df.columns:
            df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
            df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 1. Donchian Channel 돌파 신호 (추세 추종)
        df.loc[df["close"] > df["donchian_upper"], "signal"] = 1  # 롱 신호
        df.loc[df["close"] < df["donchian_lower"], "signal"] = -1  # 숏 신호

        # 2. 횡보구간에서 RSI 기반 반대 포지션 (평균회귀)
        # Donchian Channel 내부에 있을 때만 RSI 신호 적용
        channel_inside = (df["close"] <= df["donchian_upper"]) & (
            df["close"] >= df["donchian_lower"]
        )

        # RSI 계산 (하이퍼파라미터 적용)
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # RSI 과매수/과매도 신호 (하이퍼파라미터 적용)
        df.loc[channel_inside & (df["rsi"] < self.rsi_oversold), "signal"] = 1
        df.loc[channel_inside & (df["rsi"] > self.rsi_overbought), "signal"] = -1

        # 3. 모멘텀 필터링 (하이퍼파라미터 적용)
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        momentum_filter = abs(df["momentum"]) > self.momentum_threshold

        # 모멘텀 필터 적용
        df.loc[~momentum_filter, "signal"] = 0

        return df


class VolatilityAdjustedBreakoutStrategy(BaseStrategy):
    """변동성 조정 채널 브레이크아웃 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Keltner Channel 하이퍼파라미터
        self.keltner_period = getattr(params, "keltner_period", 20)
        self.keltner_multiplier = getattr(params, "keltner_multiplier", 2.0)
        # 볼륨 필터 하이퍼파라미터
        self.volume_period = getattr(params, "volume_period", 20)
        self.volume_threshold = getattr(params, "volume_threshold", 0.8)
        # 변동성 필터 하이퍼파라미터
        self.volatility_period = getattr(params, "volatility_period", 20)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 조정 브레이크아웃 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # Keltner Channel 계산 (하이퍼파라미터 적용)
        if "keltner_upper" not in df.columns or "keltner_lower" not in df.columns:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            atr = df["high"] - df["low"]  # 간단한 ATR 계산
            atr_ma = atr.rolling(window=self.keltner_period).mean()
            typical_price_ma = typical_price.rolling(window=self.keltner_period).mean()

            df["keltner_upper"] = typical_price_ma + (self.keltner_multiplier * atr_ma)
            df["keltner_lower"] = typical_price_ma - (self.keltner_multiplier * atr_ma)

        # Keltner Channel 돌파 신호
        df.loc[df["close"] > df["keltner_upper"], "signal"] = 1  # 롱 신호
        df.loc[df["close"] < df["keltner_lower"], "signal"] = -1  # 숏 신호

        # 볼륨 필터 (하이퍼파라미터 적용)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)

        # 변동성 필터 (하이퍼파라미터 적용)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=self.volatility_period).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.volatility_threshold
        else:
            volatility_filter = pd.Series([True] * len(df), index=df.index)

        # 필터 적용
        valid_signals = volume_filter & volatility_filter
        df.loc[~valid_signals, "signal"] = 0

        # 변동성 조정 포지션 사이즈
        df["position_size"] = df.apply(
            lambda row: self.calculate_position_size(df.loc[: row.name], row["signal"]),
            axis=1,
        )

        return df


class RiskParityLeverageStrategy(BaseStrategy):
    """퀀트 리스크 패리티 + 레버리지 ETF 전략"""

    def __init__(self, params: StrategyParams, symbols: List[str]):
        super().__init__(params)
        self.symbols = symbols
        self.portfolio_weights = {}
        # EMA 하이퍼파라미터
        self.ema_short_period = getattr(params, "ema_short_period", 20)
        self.ema_long_period = getattr(params, "ema_long_period", 50)

    def calculate_correlation_matrix(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """종목 간 상관관계 행렬 계산"""
        returns_dict = {}
        for symbol, df in data_dict.items():
            if "returns" in df.columns:
                returns_dict[symbol] = df["returns"].dropna()

        # 공통 기간으로 맞추기
        min_length = min(len(returns) for returns in returns_dict.values())
        aligned_returns = {}
        for symbol, returns in returns_dict.items():
            aligned_returns[symbol] = returns.tail(min_length)

        returns_df = pd.DataFrame(aligned_returns)
        return returns_df.corr()

    def calculate_risk_parity_weights(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """리스크 패리티 가중치 계산"""
        volatilities = {}
        for symbol, df in data_dict.items():
            if "volatility" in df.columns and not df["volatility"].isna().all():
                volatilities[symbol] = df["volatility"].iloc[-1]

        if not volatilities:
            # 균등 가중치
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

        # 변동성의 역수로 가중치 계산
        inv_vol = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        weights = {symbol: inv_vol[symbol] / total_inv_vol for symbol in self.symbols}
        return weights

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """리스크 패리티 기반 신호 생성"""
        signals = {}

        # 리스크 패리티 가중치 계산
        self.portfolio_weights = self.calculate_risk_parity_weights(data_dict)

        for symbol, df in data_dict.items():
            df = df.copy()
            df["signal"] = 0
            df["weight"] = self.portfolio_weights.get(symbol, 1.0 / len(self.symbols))

            # EMA 계산 (하이퍼파라미터 적용)
            df["ema_short"] = df["close"].ewm(span=self.ema_short_period).mean()
            df["ema_long"] = df["close"].ewm(span=self.ema_long_period).mean()

            # 기본 추세 추종 신호 (EMA 크로스오버)
            df.loc[df["ema_short"] > df["ema_long"], "signal"] = 1
            df.loc[df["ema_short"] < df["ema_long"], "signal"] = -1

            # 가중치 적용
            df["position_size"] = df["signal"] * df["weight"]

            signals[symbol] = df

        return signals


class SwingEMACrossoverStrategy(BaseStrategy):
    """중기 이동평균 돌파 스윙 트레이딩 전략 (1-3주 홀딩)"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.ema_short_period = getattr(params, "ema_short_period", 20)
        self.ema_long_period = getattr(params, "ema_long_period", 50)
        self.min_holding_days = getattr(params, "min_holding_days", 5)
        self.max_holding_days = getattr(params, "max_holding_days", 15)
        # 추가 하이퍼파라미터
        self.slope_period = getattr(params, "slope_period", 5)
        self.volume_threshold = getattr(params, "volume_threshold", 0.8)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA 크로스오버 스윙 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # EMA 계산
        df["ema_short"] = df["close"].ewm(span=self.ema_short_period).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long_period).mean()

        # 크로스오버 신호 생성
        df["ema_cross"] = 0
        df.loc[df["ema_short"] > df["ema_long"], "ema_cross"] = 1
        df.loc[df["ema_short"] < df["ema_long"], "ema_cross"] = -1

        # 신호 변화점 감지 (크로스오버 발생 시점)
        df["signal_change"] = df["ema_cross"].diff()

        # 매수 신호: 상향 크로스오버
        df.loc[df["signal_change"] == 2, "signal"] = 1
        # 매도 신호: 하향 크로스오버
        df.loc[df["signal_change"] == -2, "signal"] = -1

        # 스윙 트레이딩을 위한 추가 필터링
        # 1. 추세 강도 확인 (EMA 기울기) - 하이퍼파라미터 적용
        df["ema_short_slope"] = df["ema_short"].diff(self.slope_period)
        df["ema_long_slope"] = df["ema_long"].diff(self.slope_period)

        # 2. 볼륨 확인 (하이퍼파라미터 적용)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)

        # 3. 변동성 필터 (ATR 기반) - 하이퍼파라미터 적용
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.volatility_threshold
        else:
            volatility_filter = pd.Series([True] * len(df), index=df.index)

        # 필터 적용
        valid_signals = volume_filter & volatility_filter
        df.loc[~valid_signals, "signal"] = 0

        return df


class SwingRSIReversalStrategy(BaseStrategy):
    """RSI 리버설 스윙 트레이딩 전략 (1-2주 홀딩)"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)
        self.min_holding_days = getattr(params, "min_holding_days", 5)
        self.max_holding_days = getattr(params, "max_holding_days", 10)
        # 추가 하이퍼파라미터
        self.rsi_momentum_period = getattr(params, "rsi_momentum_period", 3)
        self.price_momentum_period = getattr(params, "price_momentum_period", 5)
        self.volume_threshold = getattr(params, "volume_threshold", 0.7)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI 리버설 스윙 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # RSI 과매도/과매수 신호
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < self.rsi_oversold, "rsi_signal"] = 1  # 매수 신호
        df.loc[df["rsi"] > self.rsi_overbought, "rsi_signal"] = -1  # 매도 신호

        # 신호 변화점 감지
        df["rsi_signal_change"] = df["rsi_signal"].diff()

        # 매수 신호: RSI가 과매도에서 벗어날 때
        df.loc[df["rsi_signal_change"] == 1, "signal"] = 1
        # 매도 신호: RSI가 과매수에서 벗어날 때
        df.loc[df["rsi_signal_change"] == -1, "signal"] = -1

        # 추가 필터링
        # 1. RSI 모멘텀 확인 (하이퍼파라미터 적용)
        df["rsi_momentum"] = df["rsi"].diff(self.rsi_momentum_period)

        # 2. 가격 모멘텀 확인 (하이퍼파라미터 적용)
        df["price_momentum"] = df["close"].pct_change(self.price_momentum_period)

        # 3. 볼륨 확인 (하이퍼파라미터 적용)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)

        # 필터 적용
        # RSI 매수: RSI 모멘텀 양수이고 가격 모멘텀 양수일 때
        rsi_buy_filter = (df["rsi_momentum"] > 0) & (df["price_momentum"] > 0)
        # RSI 매도: RSI 모멘텀 음수이고 가격 모멘텀 음수일 때
        rsi_sell_filter = (df["rsi_momentum"] < 0) & (df["price_momentum"] < 0)

        # 필터 적용
        df.loc[(df["signal"] == 1) & ~rsi_buy_filter, "signal"] = 0
        df.loc[(df["signal"] == -1) & ~rsi_sell_filter, "signal"] = 0
        df.loc[~volume_filter, "signal"] = 0

        return df


class DonchianSwingBreakoutStrategy(BaseStrategy):
    """Donchian Channel 돌파 스윙 트레이딩 전략 (1-3주 홀딩)"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.donchian_period = getattr(params, "donchian_period", 20)
        self.min_holding_days = getattr(params, "min_holding_days", 7)
        self.max_holding_days = getattr(params, "max_holding_days", 15)
        # 추가 하이퍼파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.8)
        self.breakout_strength_threshold = getattr(
            params, "breakout_strength_threshold", 0.005
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Donchian Channel 돌파 스윙 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # Donchian Channel 계산
        if "donchian_upper" not in df.columns or "donchian_lower" not in df.columns:
            df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
            df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 돌파 신호 생성
        df["breakout_signal"] = 0
        # 상향 돌파: 현재가가 이전 고점을 돌파
        df.loc[df["close"] > df["donchian_upper"].shift(1), "breakout_signal"] = 1
        # 하향 돌파: 현재가가 이전 저점을 돌파
        df.loc[df["close"] < df["donchian_lower"].shift(1), "breakout_signal"] = -1

        # 신호 변화점 감지
        df["breakout_change"] = df["breakout_signal"].diff()

        # 매수 신호: 상향 돌파
        df.loc[df["breakout_change"] == 1, "signal"] = 1
        # 매도 신호: 하향 돌파
        df.loc[df["breakout_change"] == -1, "signal"] = -1

        # 추가 필터링
        # 1. 돌파 강도 확인 (하이퍼파라미터 적용)
        df["breakout_strength"] = 0.0
        # 상향 돌파 강도
        df.loc[df["signal"] == 1, "breakout_strength"] = (
            df.loc[df["signal"] == 1, "close"]
            - df.loc[df["signal"] == 1, "donchian_upper"].shift(1)
        ) / df.loc[df["signal"] == 1, "donchian_upper"].shift(1)
        # 하향 돌파 강도
        df.loc[df["signal"] == -1, "breakout_strength"] = (
            df.loc[df["signal"] == -1, "donchian_lower"].shift(1)
            - df.loc[df["signal"] == -1, "close"]
        ) / df.loc[df["signal"] == -1, "donchian_lower"].shift(1)

        # 2. 볼륨 확인 (하이퍼파라미터 적용)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = (
                df["volume"] > df["volume_ma"] * self.volume_threshold
            )  # 돌파 시 볼륨 증가 필요
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)

        # 3. 변동성 확인 (하이퍼파라미터 적용)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.volatility_threshold
        else:
            volatility_filter = pd.Series([True] * len(df), index=df.index)

        # 필터 적용 (하이퍼파라미터 적용)
        strength_filter = (
            abs(df["breakout_strength"]) > self.breakout_strength_threshold
        )

        # 필터 적용
        df.loc[~strength_filter, "signal"] = 0
        df.loc[~volume_filter, "signal"] = 0
        df.loc[~volatility_filter, "signal"] = 0

        return df


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator 기반 전략"""

    def __init__(
        self,
        params: StrategyParams,
        k_period: int = 14,
        d_period: int = 3,
        low_threshold: float = 20,
        high_threshold: float = 80,
    ):
        super().__init__(params)
        self.k_period = k_period
        self.d_period = d_period
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        # stoch_k, stoch_d가 없으면 계산
        if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
            low_min = df["low"].rolling(window=self.k_period).min()
            high_max = df["high"].rolling(window=self.k_period).max()
            df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
            df["stoch_d"] = df["stoch_k"].rolling(window=self.d_period).mean()
        # 신호: %K가 low_threshold 아래에서 상향 돌파 → 매수, high_threshold 위에서 하향 돌파 → 매도
        buy = (
            (df["stoch_k"].shift(1) < self.low_threshold)
            & (df["stoch_k"] > self.low_threshold)
            & (df["stoch_k"] > df["stoch_d"])
            & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
        )
        sell = (
            (df["stoch_k"].shift(1) > self.high_threshold)
            & (df["stoch_k"] < self.high_threshold)
            & (df["stoch_k"] < df["stoch_d"])
            & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
        )
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


class WilliamsRStrategy(BaseStrategy):
    """Williams %R 기반 전략"""

    def __init__(
        self,
        params: StrategyParams,
        period: int = 14,
        low_threshold: float = -80,
        high_threshold: float = -20,
    ):
        super().__init__(params)
        self.period = period
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        # williams_r이 없으면 계산
        if "williams_r" not in df.columns:
            high_max = df["high"].rolling(window=self.period).max()
            low_min = df["low"].rolling(window=self.period).min()
            df["williams_r"] = -100 * (high_max - df["close"]) / (high_max - low_min)
        # 신호: low_threshold 아래에서 위로 돌파 → 매수, high_threshold 위에서 아래로 돌파 → 매도
        buy = (df["williams_r"].shift(1) < self.low_threshold) & (
            df["williams_r"] > self.low_threshold
        )
        sell = (df["williams_r"].shift(1) > self.high_threshold) & (
            df["williams_r"] < self.high_threshold
        )
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


class CCIStrategy(BaseStrategy):
    """CCI 기반 전략"""

    def __init__(
        self, params: StrategyParams, period: int = 20, threshold: float = 100
    ):
        super().__init__(params)
        self.period = period
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        # cci가 없으면 계산
        if "cci" not in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            ma = tp.rolling(window=self.period).mean()
            md = tp.rolling(window=self.period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            df["cci"] = (tp - ma) / (0.015 * md)
        # 신호: -threshold 아래에서 상향 돌파 → 매수, +threshold 위에서 하향 돌파 → 매도
        buy = (df["cci"].shift(1) < -self.threshold) & (df["cci"] > -self.threshold)
        sell = (df["cci"].shift(1) > self.threshold) & (df["cci"] < self.threshold)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


class CCIBollingerStrategy(BaseStrategy):
    """CCI + Bollinger Band 결합 전략 - CCI 과매도 + 밴드 이탈 = 매수 신호"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # CCI 파라미터
        self.cci_period = getattr(params, "cci_period", 20)
        self.cci_oversold = getattr(params, "cci_oversold", -100)
        self.cci_overbought = getattr(params, "cci_overbought", 100)

        # Bollinger Band 파라미터
        self.bb_period = getattr(params, "bb_period", 20)
        self.bb_std = getattr(params, "bb_std", 2.0)

        # 휩쏘 방지 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI + Bollinger Band 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. CCI 계산
        if "cci" not in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            ma = tp.rolling(window=self.cci_period).mean()
            md = tp.rolling(window=self.cci_period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            df["cci"] = (tp - ma) / (0.015 * md)

        # 2. Bollinger Band 계산
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * bb_std)
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * bb_std)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. 신호 생성
        # 매수 신호: CCI 과매도 + 밴드 하단 이탈
        buy_signal = (
            (df["cci"] < self.cci_oversold)  # CCI 과매도
            & (df["close"] < df["bb_lower"])  # 밴드 하단 이탈
            & (df["close"] > df["close"].shift(1))  # 가격 반등 시작
        )

        # 매도 신호: CCI 과매수 + 밴드 상단 이탈
        sell_signal = (
            (df["cci"] > self.cci_overbought)  # CCI 과매수
            & (df["close"] > df["bb_upper"])  # 밴드 상단 이탈
            & (df["close"] < df["close"].shift(1))  # 가격 하락 시작
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 5. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 6. 최종 신호 생성
        final_filter = volume_filter & volatility_filter
        df.loc[~final_filter, "signal"] = 0

        # 7. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class StochDonchianStrategy(BaseStrategy):
    """Stoch %K 교차 + Donchian 채널 전략 - 추세 전환 감지"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Stochastic 파라미터
        self.stoch_k_period = getattr(params, "stoch_k_period", 14)
        self.stoch_d_period = getattr(params, "stoch_d_period", 3)
        self.stoch_low_threshold = getattr(params, "stoch_low_threshold", 20)
        self.stoch_high_threshold = getattr(params, "stoch_high_threshold", 80)

        # Donchian Channel 파라미터
        self.donchian_period = getattr(params, "donchian_period", 20)

        # 휩쏘 방지 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 3)
        self.confirmation_period = getattr(params, "confirmation_period", 2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stoch %K 교차 + Donchian 채널 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. Stochastic 계산
        if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
            low_min = df["low"].rolling(window=self.stoch_k_period).min()
            high_max = df["high"].rolling(window=self.stoch_k_period).max()
            df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
            df["stoch_d"] = df["stoch_k"].rolling(window=self.stoch_d_period).mean()

        # 2. Donchian Channel 계산
        if "donchian_upper" not in df.columns or "donchian_lower" not in df.columns:
            df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
            df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. Stochastic 교차 신호
        df["stoch_cross"] = 0
        # %K가 %D를 상향 돌파 (골든 크로스)
        golden_cross = (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (
            df["stoch_k"] > df["stoch_d"]
        )
        # %K가 %D를 하향 돌파 (데드 크로스)
        dead_cross = (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (
            df["stoch_k"] < df["stoch_d"]
        )

        df.loc[golden_cross, "stoch_cross"] = 1
        df.loc[dead_cross, "stoch_cross"] = -1

        # 5. Donchian Channel 신호
        df["donchian_signal"] = 0
        # 상향 돌파
        df.loc[df["close"] > df["donchian_upper"].shift(1), "donchian_signal"] = 1
        # 하향 돌파
        df.loc[df["close"] < df["donchian_lower"].shift(1), "donchian_signal"] = -1

        # 6. 신호 결합 및 확인
        df["combined_signal"] = 0

        # 매수 신호: Stochastic 골든 크로스 + Donchian 상향 돌파
        buy_condition = (
            (df["stoch_cross"] == 1)
            & (df["donchian_signal"] == 1)
            & (df["stoch_k"] > self.stoch_low_threshold)  # 과매도 구간 제외
        )

        # 매도 신호: Stochastic 데드 크로스 + Donchian 하향 돌파
        sell_condition = (
            (df["stoch_cross"] == -1)
            & (df["donchian_signal"] == -1)
            & (df["stoch_k"] < self.stoch_high_threshold)  # 과매수 구간 제외
        )

        df.loc[buy_condition, "combined_signal"] = 1
        df.loc[sell_condition, "combined_signal"] = -1

        # 7. 신호 확인 (휩쏘 방지)
        df["signal_confirmed"] = False
        for i in range(self.confirmation_period, len(df)):
            if df["combined_signal"].iloc[i] != 0:
                # 신호 후 일정 기간 동안 방향 유지 확인
                recent_signals = df["combined_signal"].iloc[
                    i - self.confirmation_period : i + 1
                ]
                if len(set(recent_signals)) == 1 and recent_signals.iloc[0] != 0:
                    df.loc[df.index[i], "signal_confirmed"] = True

        # 8. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 9. 최종 신호 생성
        final_filter = volume_filter & volatility_filter & df["signal_confirmed"]
        df.loc[final_filter & (df["combined_signal"] == 1), "signal"] = 1
        df.loc[final_filter & (df["combined_signal"] == -1), "signal"] = -1

        # 10. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class StrategyManager:
    """전략 관리 및 백테스팅 클래스"""

    def __init__(self):
        self.strategies = {}
        self.results = {}

    def add_strategy(self, name: str, strategy: BaseStrategy):
        """전략 추가"""
        self.strategies[name] = strategy

    def load_data(self, symbol: str, file_path: str) -> pd.DataFrame:
        """CSV 파일에서 데이터 로드"""
        try:
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"데이터 로드 오류: {e}")
            raise

    def run_strategy(
        self, strategy_name: str, df: pd.DataFrame, params: StrategyParams
    ) -> Dict[str, Any]:
        """전략 실행"""
        if strategy_name not in self.strategies:
            raise ValueError(f"전략 '{strategy_name}'이 등록되지 않았습니다.")

        strategy = self.strategies[strategy_name]

        # 기술적 지표 계산
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, params)

        # 신호 생성
        if strategy_name == "risk_parity_leverage":
            # 리스크 패리티 전략은 별도 처리
            signals = strategy.generate_signals({symbol: df_with_indicators})
        else:
            signals = strategy.generate_signals(df_with_indicators)

        return {
            "strategy_name": strategy_name,
            "data": df_with_indicators,
            "signals": signals,
            "params": params,
        }

    def optimize_parameters(
        self, strategy_name: str, df: pd.DataFrame, param_ranges: Dict[str, List]
    ) -> StrategyParams:
        """하이퍼파라미터 최적화"""
        # 간단한 그리드 서치 구현
        best_params = None
        best_score = float("-inf")

        # 파라미터 조합 생성
        import itertools

        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        for combination in itertools.product(*param_values):
            params_dict = dict(zip(param_names, combination))
            params = StrategyParams(**params_dict)

            try:
                result = self.run_strategy(strategy_name, df, params)
                # 간단한 성과 지표 (수익률 기준)
                if "signals" in result and isinstance(result["signals"], pd.DataFrame):
                    signals = result["signals"]
                    if "signal" in signals.columns:
                        # 신호 기반 수익률 계산
                        returns = df["close"].pct_change()
                        strategy_returns = signals["signal"].shift(1) * returns
                        total_return = strategy_returns.sum()

                        if total_return > best_score:
                            best_score = total_return
                            best_params = params
            except Exception as e:
                logger.warning(f"파라미터 조합 {combination} 실행 중 오류: {e}")
                continue

        return best_params if best_params else StrategyParams()


class WhipsawPreventionStrategy(BaseStrategy):
    """휩쏘 방지 전략 - 다중 필터링으로 거짓 신호 제거"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 기본 지표 파라미터
        self.ema_short = getattr(params, "ema_short", 10)
        self.ema_long = getattr(params, "ema_long", 20)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.atr_period = getattr(params, "atr_period", 14)

        # 휩쏘 방지 파라미터
        self.signal_confirmation_period = getattr(
            params, "signal_confirmation_period", 3
        )
        self.volume_threshold = getattr(params, "volume_threshold", 1.5)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.8)
        self.trend_strength_threshold = getattr(
            params, "trend_strength_threshold", 0.02
        )
        self.min_holding_period = getattr(params, "min_holding_period", 5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """휩쏘 방지 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 기본 지표 계산
        df["ema_short"] = df["close"].ewm(span=self.ema_short).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long).mean()

        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.atr_period)

        # 2. 기본 신호 생성 (EMA 크로스오버)
        df["ema_cross"] = 0
        df.loc[df["ema_short"] > df["ema_long"], "ema_cross"] = 1
        df.loc[df["ema_short"] < df["ema_long"], "ema_cross"] = -1

        # 3. 휩쏘 방지 필터링

        # A. 신호 지속성 확인
        df["signal_strength"] = 0
        for i in range(self.signal_confirmation_period, len(df)):
            recent_signals = df["ema_cross"].iloc[
                i - self.signal_confirmation_period : i + 1
            ]
            if len(set(recent_signals)) == 1 and recent_signals.iloc[0] != 0:
                df.loc[df.index[i], "signal_strength"] = recent_signals.iloc[0]

        # B. 거래량 확인
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # C. 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.volatility_threshold

        # D. 추세 강도 확인
        df["trend_strength"] = abs(df["ema_short"] - df["ema_long"]) / df["close"]
        trend_filter = df["trend_strength"] > self.trend_strength_threshold

        # E. RSI 필터 (과매수/과매도 구간 제외)
        rsi_filter = (df["rsi"] > 20) & (df["rsi"] < 80)

        # 4. 최종 신호 생성
        final_filter = volume_filter & volatility_filter & trend_filter & rsi_filter
        df.loc[final_filter, "signal"] = df.loc[final_filter, "signal_strength"]

        # 5. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용으로 과도한 신호 변경 방지"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                # 새로운 신호 시작
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    # 최소 홀딩 기간 미달 시 신호 무시
                    df.loc[df.index[i], "signal"] = 0

        return df


class DonchianRSIWhipsawStrategy(BaseStrategy):
    """Donchian + RSI 듀얼 모멘텀 휩쏘 방지 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Donchian Channel 파라미터
        self.donchian_period = getattr(params, "donchian_period", 20)

        # RSI 파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)

        # 휩쏘 방지 파라미터
        self.breakout_confirmation_period = getattr(
            params, "breakout_confirmation_period", 2
        )
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Donchian + RSI 듀얼 모멘텀 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. Donchian Channel 계산
        df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
        df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 2. RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. Donchian 돌파 신호
        df["donchian_breakout"] = 0
        df.loc[df["close"] > df["donchian_upper"].shift(1), "donchian_breakout"] = 1
        df.loc[df["close"] < df["donchian_lower"].shift(1), "donchian_breakout"] = -1

        # 5. RSI 신호
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < self.rsi_oversold, "rsi_signal"] = 1
        df.loc[df["rsi"] > self.rsi_overbought, "rsi_signal"] = -1

        # 6. 돌파 확인 (휩쏘 방지)
        df["breakout_confirmed"] = False
        for i in range(self.breakout_confirmation_period, len(df)):
            if df["donchian_breakout"].iloc[i] != 0:
                # 돌파 후 일정 기간 동안 돌파 상태 유지 확인
                recent_breakout = df["donchian_breakout"].iloc[
                    i - self.breakout_confirmation_period : i + 1
                ]
                if len(set(recent_breakout)) == 1 and recent_breakout.iloc[0] != 0:
                    df.loc[df.index[i], "breakout_confirmed"] = True

        # 7. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 8. 최종 신호 생성
        # Donchian 돌파가 확인되고 필터를 통과한 경우
        donchian_signals = df["breakout_confirmed"] & volume_filter & volatility_filter
        df.loc[donchian_signals & (df["donchian_breakout"] == 1), "signal"] = 1
        df.loc[donchian_signals & (df["donchian_breakout"] == -1), "signal"] = -1

        # RSI 신호는 Donchian Channel 내부에서만 적용
        channel_inside = (df["close"] <= df["donchian_upper"]) & (
            df["close"] >= df["donchian_lower"]
        )
        rsi_signals = channel_inside & volume_filter & volatility_filter
        df.loc[rsi_signals & (df["rsi_signal"] == 1), "signal"] = 1
        df.loc[rsi_signals & (df["rsi_signal"] == -1), "signal"] = -1

        # 9. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class VolatilityFilteredBreakoutStrategy(BaseStrategy):
    """변동성 필터링 브레이크아웃 휩쏘 방지 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 브레이크아웃 파라미터
        self.breakout_period = getattr(params, "breakout_period", 20)
        self.breakout_threshold = getattr(
            params, "breakout_threshold", 0.01
        )  # 2% → 1%로 완화

        # 변동성 필터 파라미터
        self.atr_period = getattr(params, "atr_period", 14)
        self.volatility_lookback = getattr(params, "volatility_lookback", 50)
        self.volatility_quantile = getattr(
            params, "volatility_quantile", 0.5
        )  # 0.7 → 0.5로 완화

        # 거래량 필터 파라미터
        self.volume_period = getattr(params, "volume_period", 20)
        self.volume_threshold = getattr(
            params, "volume_threshold", 1.2
        )  # 1.5 → 1.2로 완화

        # 신호 확인 파라미터
        self.confirmation_period = getattr(
            params, "confirmation_period", 2
        )  # 3 → 2로 완화
        self.min_holding_period = getattr(
            params, "min_holding_period", 3
        )  # 5 → 3으로 완화

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 필터링 브레이크아웃 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 브레이크아웃 레벨 계산
        df["breakout_high"] = df["high"].rolling(window=self.breakout_period).max()
        df["breakout_low"] = df["low"].rolling(window=self.breakout_period).min()

        # 2. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.atr_period)

        # 3. 변동성 필터 계산
        df["volatility_rank"] = (
            df["atr"].rolling(window=self.volatility_lookback).rank(pct=True)
        )
        volatility_filter = df["volatility_rank"] > self.volatility_quantile

        # 4. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 5. 브레이크아웃 신호 생성
        df["breakout_signal"] = 0

        # 상향 브레이크아웃 (조건 완화)
        upward_breakout = (df["close"] > df["breakout_high"].shift(1)) | (  # 고점 돌파
            df["close"] > df["close"].shift(1) * (1 + self.breakout_threshold)
        )  # 가격 상승

        # 하향 브레이크아웃 (조건 완화)
        downward_breakout = (df["close"] < df["breakout_low"].shift(1)) | (  # 저점 돌파
            df["close"] < df["close"].shift(1) * (1 - self.breakout_threshold)
        )  # 가격 하락

        df.loc[upward_breakout, "breakout_signal"] = 1
        df.loc[downward_breakout, "breakout_signal"] = -1

        # 6. 브레이크아웃 확인 (휩쏘 방지)
        df["breakout_confirmed"] = False
        for i in range(self.confirmation_period, len(df)):
            if df["breakout_signal"].iloc[i] != 0:
                # 브레이크아웃 후 일정 기간 동안 방향 유지 확인
                recent_signals = df["breakout_signal"].iloc[
                    i - self.confirmation_period : i + 1
                ]
                if len(set(recent_signals)) == 1 and recent_signals.iloc[0] != 0:
                    df.loc[df.index[i], "breakout_confirmed"] = True

        # 7. 최종 신호 생성
        final_filter = volatility_filter & volume_filter & df["breakout_confirmed"]
        df.loc[final_filter & (df["breakout_signal"] == 1), "signal"] = 1
        df.loc[final_filter & (df["breakout_signal"] == -1), "signal"] = -1

        # 디버깅 정보 (선택적)
        if len(df) > 0:
            total_signals = len(df[df["breakout_signal"] != 0])
            confirmed_signals = len(df[df["breakout_confirmed"] == True])
            volatility_passed = len(df[volatility_filter])
            volume_passed = len(df[volume_filter])
            final_signals = len(df[df["signal"] != 0])

            # 로깅 (필요시 주석 해제)
            # print(f"VolatilityFilteredBreakout Debug:")
            # print(f"  Total breakout signals: {total_signals}")
            # print(f"  Confirmed signals: {confirmed_signals}")
            # print(f"  Volatility filter passed: {volatility_passed}")
            # print(f"  Volume filter passed: {volume_passed}")
            # print(f"  Final signals: {final_signals}")

        # 8. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class MultiTimeframeWhipsawStrategy(BaseStrategy):
    """다중 시간 프레임 휩쏘 방지 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 다중 시간 프레임 파라미터
        self.short_period = getattr(params, "short_period", 5)
        self.medium_period = getattr(params, "medium_period", 10)
        self.long_period = getattr(params, "long_period", 20)

        # RSI 파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)

        # 필터링 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.3)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """다중 시간 프레임 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 다중 시간 프레임 EMA 계산
        df["ema_short"] = df["close"].ewm(span=self.short_period).mean()
        df["ema_medium"] = df["close"].ewm(span=self.medium_period).mean()
        df["ema_long"] = df["close"].ewm(span=self.long_period).mean()

        # 2. RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. 다중 시간 프레임 신호 생성
        df["short_trend"] = np.where(df["close"] > df["ema_short"], 1, -1)
        df["medium_trend"] = np.where(df["close"] > df["ema_medium"], 1, -1)
        df["long_trend"] = np.where(df["close"] > df["ema_long"], 1, -1)

        # 5. 트렌드 일치도 계산
        df["trend_alignment"] = (
            df["short_trend"] + df["medium_trend"] + df["long_trend"]
        )

        # 6. 신호 생성 (모든 시간 프레임이 일치할 때만)
        df["trend_signal"] = 0
        df.loc[df["trend_alignment"] == 3, "trend_signal"] = 1  # 모든 시간 프레임 상승
        df.loc[df["trend_alignment"] == -3, "trend_signal"] = (
            -1
        )  # 모든 시간 프레임 하락

        # 7. RSI 필터 (과매수/과매도 구간에서 반대 신호)
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < self.rsi_oversold, "rsi_signal"] = 1
        df.loc[df["rsi"] > self.rsi_overbought, "rsi_signal"] = -1

        # 8. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 9. 최종 신호 생성
        # 트렌드 신호 우선, RSI 신호는 보조
        trend_signals = volume_filter & volatility_filter & (df["trend_signal"] != 0)
        df.loc[trend_signals, "signal"] = df.loc[trend_signals, "trend_signal"]

        # RSI 신호는 트렌드가 불분명할 때만 적용
        unclear_trend = (
            (df["trend_alignment"].abs() <= 1) & volume_filter & volatility_filter
        )
        df.loc[unclear_trend & (df["rsi_signal"] != 0), "signal"] = df.loc[
            unclear_trend & (df["rsi_signal"] != 0), "rsi_signal"
        ]

        # 10. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class AdaptiveWhipsawStrategy(BaseStrategy):
    """적응형 휩쏘 방지 전략 - 시장 상황에 따라 동적 조정"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 기본 지표 파라미터
        self.ema_period = getattr(params, "ema_period", 20)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.atr_period = getattr(params, "atr_period", 14)

        # 적응형 파라미터
        self.volatility_lookback = getattr(params, "volatility_lookback", 50)
        self.trend_lookback = getattr(params, "trend_lookback", 20)
        self.volume_lookback = getattr(params, "volume_lookback", 20)

        # 동적 임계값
        self.base_volume_threshold = getattr(params, "base_volume_threshold", 1.2)
        self.base_volatility_threshold = getattr(
            params, "base_volatility_threshold", 0.8
        )
        self.base_holding_period = getattr(params, "base_holding_period", 5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """적응형 휩쏘 방지 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 기본 지표 계산
        df["ema"] = df["close"].ewm(span=self.ema_period).mean()

        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.atr_period)

        # 2. 시장 상황 분석
        df = self._analyze_market_conditions(df)

        # 3. 동적 임계값 계산
        df = self._calculate_adaptive_thresholds(df)

        # 4. 기본 신호 생성
        df["ema_signal"] = np.where(df["close"] > df["ema"], 1, -1)

        # 5. 적응형 필터링
        df = self._apply_adaptive_filters(df)

        # 6. 최종 신호 생성
        df["signal"] = df["filtered_signal"]

        return df

    def _analyze_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """시장 상황 분석"""
        # 변동성 레벨
        df["volatility_level"] = (
            df["atr"].rolling(window=self.volatility_lookback).rank(pct=True)
        )

        # 트렌드 강도
        df["trend_strength"] = abs(df["close"] - df["ema"]) / df["close"]
        df["trend_strength_rank"] = (
            df["trend_strength"].rolling(window=self.trend_lookback).rank(pct=True)
        )

        # 거래량 레벨
        if "volume" in df.columns:
            df["volume_level"] = (
                df["volume"].rolling(window=self.volume_lookback).rank(pct=True)
            )
        else:
            df["volume_level"] = 0.5

        # 시장 상태 분류
        df["market_state"] = "normal"
        df.loc[df["volatility_level"] > 0.8, "market_state"] = "high_volatility"
        df.loc[df["trend_strength_rank"] < 0.2, "market_state"] = "sideways"
        df.loc[
            (df["volatility_level"] > 0.8) & (df["trend_strength_rank"] < 0.2),
            "market_state",
        ] = "whipsaw"

        return df

    def _calculate_adaptive_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """동적 임계값 계산"""
        # 변동성에 따른 임계값 조정
        df["adaptive_volume_threshold"] = self.base_volume_threshold
        df["adaptive_volatility_threshold"] = self.base_volatility_threshold
        df["adaptive_holding_period"] = self.base_holding_period

        # 휩쏘 상황에서는 더 엄격한 필터 적용
        whipsaw_mask = df["market_state"] == "whipsaw"
        df.loc[whipsaw_mask, "adaptive_volume_threshold"] *= 1.5
        df.loc[whipsaw_mask, "adaptive_volatility_threshold"] *= 1.3
        df.loc[whipsaw_mask, "adaptive_holding_period"] *= 2

        # 고변동성 상황에서는 중간 정도 필터
        high_vol_mask = df["market_state"] == "high_volatility"
        df.loc[high_vol_mask, "adaptive_volume_threshold"] *= 1.2
        df.loc[high_vol_mask, "adaptive_volatility_threshold"] *= 1.1
        df.loc[high_vol_mask, "adaptive_holding_period"] *= 1.5

        return df

    def _apply_adaptive_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """적응형 필터 적용"""
        df["filtered_signal"] = 0

        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = (
                df["volume"] > df["volume_ma"] * df["adaptive_volume_threshold"]
            )

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = (
                df["atr"] > df["atr_ma"] * df["adaptive_volatility_threshold"]
            )

        # RSI 필터
        rsi_filter = (df["rsi"] > 20) & (df["rsi"] < 80)

        # 최종 필터 적용
        final_filter = volume_filter & volatility_filter & rsi_filter
        df.loc[final_filter, "filtered_signal"] = df.loc[final_filter, "ema_signal"]

        # 적응형 홀딩 기간 적용
        df = self._apply_adaptive_holding_period(df)

        return df

    def _apply_adaptive_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """적응형 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if (
                df["filtered_signal"].iloc[i] != 0
                and df["filtered_signal"].iloc[i] != current_signal
            ):
                holding_period = int(df["adaptive_holding_period"].iloc[i])
                if i - signal_start >= holding_period or current_signal == 0:
                    current_signal = df["filtered_signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "filtered_signal"] = 0

        return df


class VWAPMACDScalpingStrategy(BaseStrategy):
    """VWAP + MACD 스켈핑 전략 - 1분 차트 기반 빠른 진입/청산"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # VWAP 파라미터
        self.vwap_period = getattr(params, "vwap_period", 20)

        # MACD 파라미터 (1분 스켈핑용)
        self.macd_fast = getattr(params, "macd_fast", 12)
        self.macd_slow = getattr(params, "macd_slow", 26)
        self.macd_signal = getattr(params, "macd_signal", 9)

        # 스켈핑 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.5)
        self.atr_multiplier = getattr(params, "atr_multiplier", 1.0)
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 10)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP + MACD 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. VWAP 계산
        df["vwap"] = self._calculate_vwap(df)

        # 2. MACD 계산
        df["macd"], df["macd_signal"], df["macd_histogram"] = self._calculate_macd(df)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. 신호 생성
        # 매수 신호: VWAP 상향 돌파 + MACD 골든 크로스
        buy_signal = (
            (df["close"] > df["vwap"])  # VWAP 상향 돌파
            & (df["macd"] > df["macd_signal"])  # MACD 골든 크로스
            & (
                df["macd"].shift(1) <= df["macd_signal"].shift(1)
            )  # 이전 봉에서는 크로스하지 않음
            & (df["close"] > df["close"].shift(1))  # 가격 상승
        )

        # 매도 신호: VWAP 하향 돌파 + MACD 데드 크로스
        sell_signal = (
            (df["close"] < df["vwap"])  # VWAP 하향 돌파
            & (df["macd"] < df["macd_signal"])  # MACD 데드 크로스
            & (
                df["macd"].shift(1) >= df["macd_signal"].shift(1)
            )  # 이전 봉에서는 크로스하지 않음
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 5. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 6. 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * 0.8

        # 7. 최종 신호 생성
        final_filter = volume_filter & volatility_filter
        df.loc[~final_filter, "signal"] = 0

        # 8. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """VWAP 계산"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        volume = (
            df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
        )

        vwap = (typical_price * volume).rolling(
            window=self.vwap_period
        ).sum() / volume.rolling(window=self.vwap_period).sum()
        return vwap

    def _calculate_macd(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산"""
        ema_fast = df["close"].ewm(span=self.macd_fast).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class KeltnerRSIScalpingStrategy(BaseStrategy):
    """Keltner Channels + RSI 스켈핑 전략 - 변동성 기반 빠른 진입"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Keltner Channel 파라미터
        self.keltner_period = getattr(params, "keltner_period", 20)
        self.keltner_multiplier = getattr(params, "keltner_multiplier", 2.0)

        # RSI 파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)

        # 스켈핑 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.3)
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 8)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keltner Channels + RSI 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. Keltner Channels 계산
        df["keltner_middle"] = df["close"].ewm(span=self.keltner_period).mean()
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.keltner_period)
        df["keltner_upper"] = df["keltner_middle"] + (
            self.keltner_multiplier * df["atr"]
        )
        df["keltner_lower"] = df["keltner_middle"] - (
            self.keltner_multiplier * df["atr"]
        )

        # 2. RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 3. 신호 생성
        # 매수 신호: Keltner 하단 이탈 + RSI 과매도에서 반등
        buy_signal = (
            (df["close"] < df["keltner_lower"])  # Keltner 하단 이탈
            & (df["rsi"] < self.rsi_oversold)  # RSI 과매도
            & (df["rsi"] > df["rsi"].shift(1))  # RSI 반등 시작
            & (df["close"] > df["close"].shift(1))  # 가격 반등
        )

        # 매도 신호: Keltner 상단 이탈 + RSI 과매수에서 하락
        sell_signal = (
            (df["close"] > df["keltner_upper"])  # Keltner 상단 이탈
            & (df["rsi"] > self.rsi_overbought)  # RSI 과매수
            & (df["rsi"] < df["rsi"].shift(1))  # RSI 하락 시작
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 4. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 5. 최종 신호 생성
        df.loc[~volume_filter, "signal"] = 0

        # 6. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class AbsorptionScalpingStrategy(BaseStrategy):
    """흡수(Absorption) 패턴 기반 스켈핑 전략 - 오더플로우 분석"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 흡수 패턴 파라미터
        self.absorption_lookback = getattr(params, "absorption_lookback", 5)
        self.volume_ratio_threshold = getattr(params, "volume_ratio_threshold", 2.0)
        self.price_rejection_threshold = getattr(
            params, "price_rejection_threshold", 0.005
        )

        # 확정 조건 파라미터
        self.fibonacci_levels = getattr(
            params, "fibonacci_levels", [0.236, 0.382, 0.618]
        )
        self.ema_short = getattr(params, "ema_short", 5)
        self.ema_long = getattr(params, "ema_long", 10)
        self.macd_fast = getattr(params, "macd_fast", 12)
        self.macd_slow = getattr(params, "macd_slow", 26)

        # 스켈핑 파라미터
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 6)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """흡수 패턴 기반 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 흡수 패턴 감지
        df["absorption_pattern"] = self._detect_absorption_pattern(df)

        # 2. 확정 조건 계산
        df["fibonacci_confirmed"] = self._check_fibonacci_levels(df)
        df["ema_cross_confirmed"] = self._check_ema_crossing(df)
        df["macd_confirmed"] = self._check_macd_convergence(df)

        # 3. 신호 생성
        # 매수 신호: 흡수 패턴 + 확정 조건
        buy_signal = (
            (df["absorption_pattern"] == 1)  # 흡수 패턴 감지
            & (
                df["fibonacci_confirmed"]
                | df["ema_cross_confirmed"]
                | df["macd_confirmed"]
            )  # 확정 조건 중 하나
            & (df["close"] > df["close"].shift(1))  # 가격 상승
        )

        # 매도 신호: 반대 흡수 패턴
        sell_signal = (
            (df["absorption_pattern"] == -1)  # 반대 흡수 패턴
            & (
                df["fibonacci_confirmed"]
                | df["ema_cross_confirmed"]
                | df["macd_confirmed"]
            )  # 확정 조건 중 하나
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 4. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _detect_absorption_pattern(self, df: pd.DataFrame) -> pd.Series:
        """흡수 패턴 감지"""
        absorption = pd.Series(0, index=df.index)

        for i in range(self.absorption_lookback, len(df)):
            # 최근 봉들의 패턴 분석
            recent_highs = df["high"].iloc[i - self.absorption_lookback : i + 1]
            recent_lows = df["low"].iloc[i - self.absorption_lookback : i + 1]
            recent_volumes = (
                df["volume"].iloc[i - self.absorption_lookback : i + 1]
                if "volume" in df.columns
                else pd.Series(1, index=recent_highs.index)
            )

            # 매도 압력 감지 (높은 고가, 높은 거래량)
            selling_pressure = (recent_highs.max() > recent_highs.mean() * 1.01) and (
                recent_volumes.mean()
                > recent_volumes.rolling(window=20).mean().iloc[-1]
                * self.volume_ratio_threshold
            )

            # 가격 거부 패턴 (고가에서 하락)
            price_rejection = (
                df["close"].iloc[i]
                < df["high"].iloc[i] * (1 - self.price_rejection_threshold)
            ) and (df["close"].iloc[i] > df["close"].iloc[i - 1])

            # 흡수 패턴: 매도 압력이 있지만 가격이 상승
            if selling_pressure and price_rejection:
                absorption.iloc[i] = 1

            # 반대 흡수 패턴: 매수 압력이 있지만 가격이 하락
            buying_pressure = (recent_lows.min() < recent_lows.mean() * 0.99) and (
                recent_volumes.mean()
                > recent_volumes.rolling(window=20).mean().iloc[-1]
                * self.volume_ratio_threshold
            )

            reverse_rejection = (
                df["close"].iloc[i]
                > df["low"].iloc[i] * (1 + self.price_rejection_threshold)
            ) and (df["close"].iloc[i] < df["close"].iloc[i - 1])

            if buying_pressure and reverse_rejection:
                absorption.iloc[i] = -1

        return absorption

    def _check_fibonacci_levels(self, df: pd.DataFrame) -> pd.Series:
        """피보나치 레벨 확인"""
        confirmed = pd.Series(False, index=df.index)

        for i in range(20, len(df)):
            # 최근 고점/저점 찾기
            recent_high = df["high"].iloc[i - 20 : i + 1].max()
            recent_low = df["low"].iloc[i - 20 : i + 1].min()
            current_price = df["close"].iloc[i]

            # 피보나치 레벨 계산
            price_range = recent_high - recent_low
            for level in self.fibonacci_levels:
                fib_level = recent_low + (price_range * level)
                if abs(current_price - fib_level) / current_price < 0.01:  # 1% 이내
                    confirmed.iloc[i] = True
                    break

        return confirmed

    def _check_ema_crossing(self, df: pd.DataFrame) -> pd.Series:
        """EMA 크로싱 확인"""
        df["ema_short"] = df["close"].ewm(span=self.ema_short).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long).mean()

        # 골든 크로스 또는 데드 크로스
        golden_cross = (df["ema_short"] > df["ema_long"]) & (
            df["ema_short"].shift(1) <= df["ema_long"].shift(1)
        )
        dead_cross = (df["ema_short"] < df["ema_long"]) & (
            df["ema_short"].shift(1) >= df["ema_long"].shift(1)
        )

        return golden_cross | dead_cross

    def _check_macd_convergence(self, df: pd.DataFrame) -> pd.Series:
        """MACD 0 수렴 확인"""
        ema_fast = df["close"].ewm(span=self.macd_fast).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow).mean()
        macd = ema_fast - ema_slow

        # MACD가 0에 수렴하는 패턴
        convergence = abs(macd) < abs(macd).rolling(window=10).mean() * 0.5
        return convergence

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class RSIBollingerScalpingStrategy(BaseStrategy):
    """RSI + Bollinger Bands 스켈핑 전략 - 1분 차트 최적화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # RSI 파라미터 (1분 스켈핑용)
        self.rsi_period = getattr(params, "rsi_period", 4)  # 짧은 기간
        self.rsi_oversold = getattr(params, "rsi_oversold", 20)
        self.rsi_overbought = getattr(params, "rsi_overbought", 80)

        # Bollinger Bands 파라미터
        self.bb_period = getattr(params, "bb_period", 20)
        self.bb_std = getattr(params, "bb_std", 2.0)

        # 스켈핑 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 8)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI + Bollinger Bands 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. RSI 계산 (짧은 기간)
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 2. Bollinger Bands 계산
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * bb_std)
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * bb_std)

        # 3. 신호 생성
        # 매수 신호: RSI 과매도 이탈 + Bollinger 하단 터치
        buy_signal = (
            (df["rsi"].shift(1) < self.rsi_oversold)  # 이전에 과매도
            & (df["rsi"] > self.rsi_oversold)  # 현재 과매도 이탈
            & (df["close"] <= df["bb_lower"])  # Bollinger 하단 터치
            & (df["close"] > df["close"].shift(1))  # 가격 반등
        )

        # 매도 신호: RSI 과매수 이탈 + Bollinger 상단 터치
        sell_signal = (
            (df["rsi"].shift(1) > self.rsi_overbought)  # 이전에 과매수
            & (df["rsi"] < self.rsi_overbought)  # 현재 과매수 이탈
            & (df["close"] >= df["bb_upper"])  # Bollinger 상단 터치
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 4. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 5. 최종 신호 생성
        df.loc[~volume_filter, "signal"] = 0

        # 6. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band 기반 Mean Reversion(평균회귀) 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.bb_period = getattr(params, "bb_period", 20)
        self.bb_std = getattr(params, "bb_std", 2.0)
        self.entry_zscore = getattr(
            params, "entry_zscore", 2.0
        )  # 진입 임계값 (표준편차)
        self.exit_zscore = getattr(params, "exit_zscore", 0.5)  # 청산 임계값 (표준편차)
        self.min_holding_period = getattr(params, "min_holding_period", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        # 볼린저 밴드 계산
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * bb_std)
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * bb_std)

        # z-score 계산 (현재가와 밴드 중심의 표준편차 단위 거리)
        df["zscore"] = (df["close"] - df["bb_middle"]) / bb_std

        # 진입 신호: 하단 밴드 이탈(과매도) → 매수, 상단 밴드 이탈(과매수) → 매도
        df.loc[df["zscore"] < -self.entry_zscore, "signal"] = 1
        df.loc[df["zscore"] > self.entry_zscore, "signal"] = -1

        # 청산 신호: zscore가 0에 근접(평균 복귀)
        exit_long = (df["signal"].shift(1) == 1) & (df["zscore"] > -self.exit_zscore)
        exit_short = (df["signal"].shift(1) == -1) & (df["zscore"] < self.exit_zscore)
        df.loc[exit_long | exit_short, "signal"] = 0

        # 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)
        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        current_signal = 0
        signal_start = 0
        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0
            # 평균 복귀 후 신호 종료
            if current_signal != 0 and (
                (current_signal == 1 and df["zscore"].iloc[i] > -self.exit_zscore)
                or (current_signal == -1 and df["zscore"].iloc[i] < self.exit_zscore)
            ):
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0
        return df


def main():
    """메인 실행 함수"""
    # 전략 매니저 초기화
    manager = StrategyManager()

    # 기본 파라미터
    params = StrategyParams()

    # 전략들 등록
    manager.add_strategy("dual_momentum", DualMomentumStrategy(params))
    manager.add_strategy(
        "volatility_breakout", VolatilityAdjustedBreakoutStrategy(params)
    )

    # 리스크 패리티 전략 (여러 종목 필요)
    risk_parity_params = StrategyParams()
    manager.add_strategy(
        "risk_parity_leverage",
        RiskParityLeverageStrategy(risk_parity_params, ["NVDL", "TSLL", "CONL"]),
    )

    # 추가 전략 등록
    manager.add_strategy("stochastic", StochasticStrategy(params))
    manager.add_strategy("williams_r", WilliamsRStrategy(params))
    manager.add_strategy("cci", CCIStrategy(params))
    manager.add_strategy("whipsaw_prevention", WhipsawPreventionStrategy(params))
    manager.add_strategy("donchian_rsi_whipsaw", DonchianRSIWhipsawStrategy(params))
    manager.add_strategy(
        "volatility_filtered_breakout", VolatilityFilteredBreakoutStrategy(params)
    )
    manager.add_strategy(
        "multi_timeframe_whipsaw", MultiTimeframeWhipsawStrategy(params)
    )
    manager.add_strategy("adaptive_whipsaw", AdaptiveWhipsawStrategy(params))
    manager.add_strategy("cci_bollinger", CCIBollingerStrategy(params))
    manager.add_strategy("stoch_donchian", StochDonchianStrategy(params))
    manager.add_strategy("vwap_macd", VWAPMACDScalpingStrategy(params))
    manager.add_strategy("keltner_rsi", KeltnerRSIScalpingStrategy(params))
    manager.add_strategy("absorption", AbsorptionScalpingStrategy(params))
    manager.add_strategy("rsi_bollinger", RSIBollingerScalpingStrategy(params))
    manager.add_strategy("mean_reversion", MeanReversionStrategy(params))

    print("퀀트 전략 클래스가 성공적으로 초기화되었습니다.")
    print("사용 가능한 전략:")
    print("1. dual_momentum - 추세⇄평균회귀 듀얼 모멘텀")
    print("2. volatility_breakout - 변동성 조정 채널 브레이크아웃")
    print("3. risk_parity_leverage - 퀀트 리스크 패리티 + 레버리지 ETF")
    print("4. swing_ema - 중기 이동평균 돌파 스윙 트레이딩")
    print("5. swing_rsi - RSI 리버설 스윙 트레이딩")
    print("6. swing_donchian - Donchian Channel 돌파 스윙 트레이딩")
    print("7. stochastic - Stochastic Oscillator 기반 전략")
    print("8. williams_r - Williams %R 기반 전략")
    print("9. cci - CCI 기반 전략")
    print("10. whipsaw_prevention - 휩쏘 방지 전략")
    print("11. donchian_rsi_whipsaw - Donchian + RSI 듀얼 모멘텀 휩쏘 방지 전략")
    print(
        "12. volatility_filtered_breakout - 변동성 필터링 브레이크아웃 휩쏘 방지 전략"
    )
    print("13. multi_timeframe_whipsaw - 다중 시간 프레임 휩쏘 방지 전략")
    print("14. adaptive_whipsaw - 적응형 휩쏘 방지 전략")
    print("15. cci_bollinger - CCI + Bollinger Band 결합 전략")
    print("16. stoch_donchian - Stoch %K 교차 + Donchian 채널 전략")
    print("17. vwap_macd - VWAP + MACD 스켈핑 전략")
    print("18. keltner_rsi - Keltner Channels + RSI 스켈핑 전략")
    print("19. absorption - 흡수(Absorption) 패턴 기반 스켈핑 전략")
    print("20. rsi_bollinger - RSI + Bollinger Bands 스켈핑 전략")
    print("21. mean_reversion - Bollinger Band 기반 Mean Reversion(평균회귀) 전략")


if __name__ == "__main__":
    main()
