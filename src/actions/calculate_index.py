import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import os
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    """전략 하이퍼파라미터를 저장하는 데이터클래스"""

    # 최적화용 lookback 기간
    lookback_period: int = 60

    # 포트폴리오 최적화 파라미터
    rebalance_period: int = 60
    max_leverage: float = 2.0

    # 리스크 관리 파라미터
    risk_free_rate: float = 0.02
    min_weight: float = 0.0
    max_weight: float = 1.0

    # 추가 최적화 파라미터들
    momentum_period: int = 20
    top_n_symbols: int = 3
    max_iterations: int = 1000

    # Dual Momentum 전략 파라미터
    donchian_period: int = 20
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    momentum_threshold: float = 0.02

    # Volatility Breakout 전략 파라미터
    keltner_period: int = 20
    keltner_multiplier: float = 2.0
    volume_period: int = 20
    volume_threshold: float = 1.0
    volatility_period: int = 20
    volatility_threshold: float = 0.5

    # Swing EMA 전략 파라미터
    ema_short_period: int = 20
    ema_long_period: int = 50
    min_holding_days: int = 5
    max_holding_days: int = 15
    slope_period: int = 5

    # Swing RSI 전략 파라미터
    rsi_momentum_period: int = 3
    price_momentum_period: int = 5

    # Swing Donchian 전략 파라미터
    breakout_strength_threshold: float = 0.005

    # Stochastic 전략 파라미터
    k_period: int = 14
    d_period: int = 3
    low_threshold: int = 20
    high_threshold: int = 80

    # Williams %R 전략 파라미터
    period: int = 14
    williams_r_period: int = 14  # 추가
    low_threshold: int = -80
    high_threshold: int = -20

    # CCI 전략 파라미터
    threshold: int = 100

    # ADX 전략 파라미터
    adx_period: int = 14  # 추가

    # OBV 전략 파라미터
    obv_smooth_period: int = 20  # 추가

    # 리스크 관리 파라미터 (추가)
    volatility_target: float = 0.15  # 추가
    stop_loss_atr_multiplier: float = 2.0  # 추가
    take_profit_atr_multiplier: float = 3.0  # 추가

    # Whipsaw Prevention 전략 파라미터
    ema_short: int = 10
    ema_long: int = 20
    atr_period: int = 14
    signal_confirmation_period: int = 3
    trend_strength_threshold: float = 0.02
    min_holding_period: int = 5

    # Donchian RSI Whipsaw 전략 파라미터
    breakout_confirmation_period: int = 2
    atr_threshold: float = 1.0

    # Volatility Filtered Breakout 전략 파라미터
    breakout_period: int = 20
    breakout_threshold: float = 0.01
    volatility_lookback: int = 50
    volatility_quantile: float = 0.5
    confirmation_period: int = 2

    # Multi Timeframe Whipsaw 전략 파라미터
    short_period: int = 5
    medium_period: int = 10
    long_period: int = 20

    # Adaptive Whipsaw 전략 파라미터
    ema_period: int = 20
    trend_lookback: int = 20
    volume_lookback: int = 20
    base_volume_threshold: float = 1.2
    base_volatility_threshold: float = 0.8
    base_holding_period: int = 5

    # CCI Bollinger 전략 파라미터
    cci_period: int = 20
    cci_oversold: int = -100
    cci_overbought: int = 100
    bb_period: int = 20
    bb_std: float = 2.0

    # Stoch Donchian 전략 파라미터
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_low_threshold: int = 20
    stoch_high_threshold: int = 80

    # VWAP MACD Scalping 전략 파라미터
    vwap_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    max_holding_period: int = 10

    # Keltner RSI Scalping 전략 파라미터

    # Absorption Scalping 전략 파라미터
    price_momentum_period: int = 5
    momentum_threshold: float = 0.01

    # RSI Bollinger Scalping 전략 파라미터

    # Buy Hold 전략 파라미터 (없음)

    # Fixed Weight Rebalance 전략 파라미터
    weight_method: str = "equal"

    # ETF Momentum Rotation 전략 파라미터
    top_n: int = 5

    # Trend Following MA20 전략 파라미터
    ma_period: int = 20
    atr_multiplier: float = 2.0

    # Return Stacking 전략 파라미터
    volatility_period: int = 60

    # Risk Parity Leverage 전략 파라미터
    target_volatility: float = 0.15

    # Fibonacci 레벨 파라미터
    fibonacci_levels: List[float] = None

    def __post_init__(self):
        """초기화 후 실행되는 메서드"""
        if self.fibonacci_levels is None:
            self.fibonacci_levels = [0.236, 0.382, 0.618]


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""

    @staticmethod
    def _calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range (ATR) 계산"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def _calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average (EMA) 계산"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (RSI) 계산"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def _calculate_donchian_channels(
        high: pd.Series, low: pd.Series, period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels 계산"""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2

        return upper, middle, lower

    @staticmethod
    def _calculate_keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        multiplier: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels 계산"""
        typical_price = (high + low + close) / 3
        atr = TechnicalIndicators._calculate_atr(high, low, close, period)

        middle = typical_price.rolling(window=period).mean()
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)

        return upper, middle, lower

    @staticmethod
    def _calculate_macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence) 계산"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def _calculate_bollinger_bands(
        close: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands 계산"""
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    @staticmethod
    def _calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator 계산"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    @staticmethod
    def _calculate_williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Williams %R 계산"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r

    @staticmethod
    def _calculate_cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> pd.Series:
        """CCI (Commodity Channel Index) 계산"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )

        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci

    @staticmethod
    def _calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX (Average Directional Index) 계산"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def _calculate_obv(
        close: pd.Series, volume: pd.Series, smooth_period: int = 20
    ) -> pd.Series:
        """OBV (On-Balance Volume) 계산"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        # Smoothing
        obv_smooth = obv.rolling(window=smooth_period).mean()
        return obv_smooth

    @staticmethod
    def _calculate_volatility(returns: pd.Series, period: int = 30) -> pd.Series:
        """실현 변동성 계산"""
        return returns.rolling(window=period).std() * np.sqrt(252)  # 연간화

    @staticmethod
    def calculate_all_indicators(
        df: pd.DataFrame, params: StrategyParams
    ) -> pd.DataFrame:
        """모든 기술적 지표를 계산하여 DataFrame에 추가"""
        df = df.copy()

        # 기본 지표들
        df["atr"] = TechnicalIndicators._calculate_atr(
            df["high"], df["low"], df["close"], params.atr_period
        )
        df["ema_short"] = TechnicalIndicators._calculate_ema(
            df["close"], params.ema_short
        )
        df["ema_long"] = TechnicalIndicators._calculate_ema(
            df["close"], params.ema_long
        )
        df["rsi"] = TechnicalIndicators._calculate_rsi(df["close"], params.rsi_period)

        # MACD
        df["macd"], df["macd_signal"], df["macd_histogram"] = (
            TechnicalIndicators._calculate_macd(
                df["close"], params.macd_fast, params.macd_slow, params.macd_signal
            )
        )

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = (
            TechnicalIndicators._calculate_bollinger_bands(
                df["close"], params.bb_period, params.bb_std
            )
        )

        # Stochastic
        df["stoch_k"], df["stoch_d"] = TechnicalIndicators._calculate_stochastic(
            df["high"],
            df["low"],
            df["close"],
            params.stoch_k_period,
            params.stoch_d_period,
        )

        # Williams %R
        df["williams_r"] = TechnicalIndicators._calculate_williams_r(
            df["high"], df["low"], df["close"], params.williams_r_period
        )

        # CCI
        df["cci"] = TechnicalIndicators._calculate_cci(
            df["high"], df["low"], df["close"], params.cci_period
        )

        # ADX
        df["adx"], df["plus_di"], df["minus_di"] = TechnicalIndicators._calculate_adx(
            df["high"], df["low"], df["close"], params.adx_period
        )

        # OBV
        df["obv"] = TechnicalIndicators._calculate_obv(
            df["close"], df["volume"], params.obv_smooth_period
        )

        # Donchian Channels
        df["donchian_upper"], df["donchian_middle"], df["donchian_lower"] = (
            TechnicalIndicators._calculate_donchian_channels(
                df["high"], df["low"], params.donchian_period
            )
        )

        # Keltner Channels
        df["keltner_upper"], df["keltner_middle"], df["keltner_lower"] = (
            TechnicalIndicators._calculate_keltner_channels(
                df["high"],
                df["low"],
                df["close"],
                params.keltner_period,
                params.keltner_multiplier,
            )
        )

        # 수익률 및 변동성
        df["returns"] = df["close"].pct_change()
        df["volatility"] = TechnicalIndicators._calculate_volatility(
            df["returns"], params.volatility_period
        )

        return df
