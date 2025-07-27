"""
피처 엔지니어링 프로세스 관리
- 피처 생성 로직 통합
- 피처 차원 정보 저장/로드
- 재사용 가능한 피처 엔지니어링 파이프라인
- 동적 피처 생성 시스템
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """피처 엔지니어링 파이프라인"""

    def __init__(self, config: Dict):
        self.config = config
        self.feature_info = {
            "universal_features": {},
            "individual_features": {},
            "macro_features": {},
            "feature_dimensions": {},
            "created_at": datetime.now().isoformat(),
        }
        # 동적 피처 생성을 위한 컬럼 패턴 정의
        self.column_patterns = self._define_column_patterns()

    def _define_column_patterns(self) -> Dict[str, Dict]:
        """컬럼 패턴 정의"""
        return {
            "basic_ohlcv": {
                "required": ["open", "high", "low", "close", "volume"],
                "optional": ["adj_close", "vwap"],
            },
            "financial_metrics": {
                "basic": ["pe_ratio", "pb_ratio", "market_cap", "dividend_yield"],
                "advanced": [
                    "roe",
                    "roa",
                    "debt_to_equity",
                    "current_ratio",
                    "quick_ratio",
                    "gross_margin",
                    "operating_margin",
                    "net_margin",
                    "ebitda_margin",
                    "revenue_growth",
                    "earnings_growth",
                    "free_cash_flow",
                    "book_value",
                    "tangible_book_value",
                    "cash_per_share",
                    "debt_per_share",
                    "working_capital",
                    "total_assets",
                    "total_liabilities",
                    "shareholders_equity",
                ],
            },
            "technical_indicators": {
                "momentum": ["rsi", "stoch", "williams_r", "cci", "mfi"],
                "trend": ["sma", "ema", "macd", "adx", "aroon"],
                "volatility": ["bbands", "atr", "natr", "keltner"],
                "volume": ["obv", "vwap", "volume_sma", "volume_ratio"],
            },
        }

    def analyze_data_structure(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        데이터 구조 분석

        Args:
            data: 분석할 데이터프레임
            symbol: 종목 심볼

        Returns:
            데이터 구조 분석 결과
        """
        analysis = {
            "symbol": symbol,
            "total_columns": len(data.columns),
            "available_columns": list(data.columns),
            "column_categories": {},
            "missing_columns": {},
            "data_complexity": "basic",
        }

        # 기본 OHLCV 확인
        basic_ohlcv = self.column_patterns["basic_ohlcv"]["required"]
        available_ohlcv = [col for col in basic_ohlcv if col in data.columns]
        analysis["column_categories"]["ohlcv"] = {
            "available": available_ohlcv,
            "missing": [col for col in basic_ohlcv if col not in data.columns],
        }

        # 재무 지표 확인
        financial_basic = self.column_patterns["financial_metrics"]["basic"]
        financial_advanced = self.column_patterns["financial_metrics"]["advanced"]

        available_financial_basic = [
            col for col in financial_basic if col in data.columns
        ]
        available_financial_advanced = [
            col for col in financial_advanced if col in data.columns
        ]

        analysis["column_categories"]["financial"] = {
            "basic": {
                "available": available_financial_basic,
                "missing": [col for col in financial_basic if col not in data.columns],
            },
            "advanced": {
                "available": available_financial_advanced,
                "missing": [
                    col for col in financial_advanced if col not in data.columns
                ],
            },
        }

        # 기술적 지표 확인
        tech_momentum = self.column_patterns["technical_indicators"]["momentum"]
        tech_trend = self.column_patterns["technical_indicators"]["trend"]
        tech_volatility = self.column_patterns["technical_indicators"]["volatility"]
        tech_volume = self.column_patterns["technical_indicators"]["volume"]

        analysis["column_categories"]["technical"] = {
            "momentum": {
                "available": [col for col in tech_momentum if col in data.columns],
                "missing": [col for col in tech_momentum if col not in data.columns],
            },
            "trend": {
                "available": [col for col in tech_trend if col in data.columns],
                "missing": [col for col in tech_trend if col not in data.columns],
            },
            "volatility": {
                "available": [col for col in tech_volatility if col in data.columns],
                "missing": [col for col in tech_volatility if col not in data.columns],
            },
            "volume": {
                "available": [col for col in tech_volume if col in data.columns],
                "missing": [col for col in tech_volume if col not in data.columns],
            },
        }

        # 데이터 복잡도 평가
        total_financial = len(available_financial_basic) + len(
            available_financial_advanced
        )
        total_technical = sum(
            len(cat["available"])
            for cat in analysis["column_categories"]["technical"].values()
        )

        if total_financial > 10 and total_technical > 15:
            analysis["data_complexity"] = "advanced"
        elif total_financial > 5 or total_technical > 10:
            analysis["data_complexity"] = "intermediate"
        else:
            analysis["data_complexity"] = "basic"

        return analysis

    def create_dynamic_features(
        self,
        stock_data: pd.DataFrame,
        symbol: str,
        market_regime: Dict,
        macro_data: Optional[pd.DataFrame] = None,
        mode: str = "individual",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        동적 피처 생성 - 데이터 구조에 따라 적응적 피처 생성

        Args:
            stock_data: 주식 데이터
            symbol: 종목 심볼
            market_regime: 시장 체제 정보
            macro_data: 매크로 데이터
            mode: 피처 생성 모드

        Returns:
            (features_df, feature_metadata)
        """
        try:
            # 1. 데이터 구조 분석
            structure_analysis = self.analyze_data_structure(stock_data, symbol)
            logger.info(
                f"데이터 구조 분석 완료 ({symbol}): {structure_analysis['data_complexity']} 복잡도"
            )

            # 2. 동적 스윙 피처 생성
            swing_features = self._create_dynamic_swing_features(
                stock_data, symbol, structure_analysis
            )

            # 3. 동적 재무 피처 생성
            financial_features = self._create_dynamic_financial_features(
                stock_data, symbol, structure_analysis
            )

            # 4. 동적 기술적 지표 피처 생성
            technical_features = self._create_dynamic_technical_features(
                stock_data, symbol, structure_analysis
            )

            # 5. 매크로 피처 생성
            macro_features = (
                self._create_macro_features(macro_data, symbol)
                if macro_data is not None
                else pd.DataFrame()
            )

            # 6. 시장 체제 피처 생성
            regime_features = self._create_regime_features(
                market_regime, symbol, stock_data
            )

            # 7. 피처 통합
            if mode == "universal":
                combined_features = self._combine_universal_features(
                    swing_features,
                    financial_features,
                    technical_features,
                    macro_features,
                    regime_features,
                    symbol,
                )
            else:
                combined_features = self._combine_individual_features(
                    swing_features,
                    financial_features,
                    technical_features,
                    macro_features,
                    regime_features,
                    symbol,
                )

            # 8. 피처 메타데이터 생성
            feature_metadata = self._create_dynamic_feature_metadata(
                combined_features, symbol, mode, structure_analysis
            )

            return combined_features, feature_metadata

        except Exception as e:
            logger.error(f"동적 피처 생성 실패 ({symbol}, {mode}): {e}")
            return pd.DataFrame(), {}

    def _create_dynamic_swing_features(
        self, stock_data: pd.DataFrame, symbol: str, structure_analysis: Dict
    ) -> pd.DataFrame:
        """동적 스윙 피처 생성 - 기본 OHLCV 기반"""
        features = {}

        # 기본 OHLCV가 있는지 확인
        ohlcv_available = structure_analysis["column_categories"]["ohlcv"]["available"]
        if len(ohlcv_available) >= 4:  # 최소 open, high, low, close 필요
            features[f"{symbol}_dual_momentum"] = self._calculate_dual_momentum(
                stock_data
            )
            features[f"{symbol}_volatility_breakout"] = (
                self._calculate_volatility_breakout(stock_data)
            )
            features[f"{symbol}_swing_ema"] = self._calculate_swing_ema(stock_data)
            features[f"{symbol}_swing_rsi"] = self._calculate_swing_rsi(stock_data)
            features[f"{symbol}_swing_donchian"] = self._calculate_swing_donchian(
                stock_data
            )
            features[f"{symbol}_stoch_donchian"] = self._calculate_stoch_donchian(
                stock_data
            )
            features[f"{symbol}_whipsaw_prevention"] = (
                self._calculate_whipsaw_prevention(stock_data)
            )
            features[f"{symbol}_donchian_rsi_whipsaw"] = (
                self._calculate_donchian_rsi_whipsaw(stock_data)
            )
            features[f"{symbol}_volatility_filtered_breakout"] = (
                self._calculate_volatility_filtered_breakout(stock_data)
            )
            features[f"{symbol}_multi_timeframe_whipsaw"] = (
                self._calculate_multi_timeframe_whipsaw(stock_data)
            )

        # 인덱스가 있는 DataFrame 생성
        if features:
            return pd.DataFrame(features, index=stock_data.index)
        else:
            return pd.DataFrame(index=stock_data.index)

    def _create_dynamic_financial_features(
        self, stock_data: pd.DataFrame, symbol: str, structure_analysis: Dict
    ) -> pd.DataFrame:
        """동적 재무 피처 생성"""
        features = {}

        # 기본 재무 지표
        basic_financial = structure_analysis["column_categories"]["financial"]["basic"][
            "available"
        ]
        for metric in basic_financial:
            if metric in stock_data.columns:
                features[f"{symbol}_financial_{metric}"] = stock_data[metric]

        # 고급 재무 지표
        advanced_financial = structure_analysis["column_categories"]["financial"][
            "advanced"
        ]["available"]
        for metric in advanced_financial:
            if metric in stock_data.columns:
                features[f"{symbol}_financial_{metric}"] = stock_data[metric]

        # 재무 지표 조합 피처 생성
        if len(basic_financial) >= 2:
            features.update(
                self._create_financial_combination_features(
                    stock_data, symbol, basic_financial
                )
            )

        if len(advanced_financial) >= 3:
            features.update(
                self._create_advanced_financial_features(
                    stock_data, symbol, advanced_financial
                )
            )

        # 인덱스가 있는 DataFrame 생성
        if features:
            return pd.DataFrame(features, index=stock_data.index)
        else:
            return pd.DataFrame(index=stock_data.index)

    def _create_dynamic_technical_features(
        self, stock_data: pd.DataFrame, symbol: str, structure_analysis: Dict
    ) -> pd.DataFrame:
        """동적 기술적 지표 피처 생성"""
        features = {}

        # 모멘텀 지표
        momentum_indicators = structure_analysis["column_categories"]["technical"][
            "momentum"
        ]["available"]
        for indicator in momentum_indicators:
            if indicator in stock_data.columns:
                features[f"{symbol}_tech_{indicator}"] = stock_data[indicator]

        # 트렌드 지표
        trend_indicators = structure_analysis["column_categories"]["technical"][
            "trend"
        ]["available"]
        for indicator in trend_indicators:
            if indicator in stock_data.columns:
                features[f"{symbol}_tech_{indicator}"] = stock_data[indicator]

        # 변동성 지표
        volatility_indicators = structure_analysis["column_categories"]["technical"][
            "volatility"
        ]["available"]
        for indicator in volatility_indicators:
            if indicator in stock_data.columns:
                features[f"{symbol}_tech_{indicator}"] = stock_data[indicator]

        # 거래량 지표
        volume_indicators = structure_analysis["column_categories"]["technical"][
            "volume"
        ]["available"]
        for indicator in volume_indicators:
            if indicator in stock_data.columns:
                features[f"{symbol}_tech_{indicator}"] = stock_data[indicator]

        # 기술적 지표 조합 피처 생성
        if len(momentum_indicators) >= 2:
            features.update(
                self._create_momentum_combination_features(
                    stock_data, symbol, momentum_indicators
                )
            )

        if len(trend_indicators) >= 2:
            features.update(
                self._create_trend_combination_features(
                    stock_data, symbol, trend_indicators
                )
            )

        # 인덱스가 있는 DataFrame 생성
        if features:
            return pd.DataFrame(features, index=stock_data.index)
        else:
            return pd.DataFrame(index=stock_data.index)

    def _create_financial_combination_features(
        self, stock_data: pd.DataFrame, symbol: str, available_metrics: List[str]
    ) -> Dict:
        """재무 지표 조합 피처 생성"""
        features = {}

        # P/E와 P/B 비율 조합
        if "pe_ratio" in available_metrics and "pb_ratio" in available_metrics:
            pe_ratio = stock_data["pe_ratio"]
            pb_ratio = stock_data["pb_ratio"]
            features[f"{symbol}_financial_pe_pb_ratio"] = pe_ratio / (pb_ratio + 1e-8)

        # 시가총액과 배당수익률 조합
        if "market_cap" in available_metrics and "dividend_yield" in available_metrics:
            market_cap = stock_data["market_cap"]
            dividend_yield = stock_data["dividend_yield"]
            features[f"{symbol}_financial_market_cap_dividend"] = (
                market_cap * dividend_yield
            )

        return features

    def _create_advanced_financial_features(
        self, stock_data: pd.DataFrame, symbol: str, available_metrics: List[str]
    ) -> Dict:
        """고급 재무 피처 생성"""
        features = {}

        # 수익성 지표 조합
        if "roe" in available_metrics and "roa" in available_metrics:
            roe = stock_data["roe"]
            roa = stock_data["roa"]
            features[f"{symbol}_financial_roe_roa_ratio"] = roe / (roa + 1e-8)

        # 마진 지표 조합
        if (
            "gross_margin" in available_metrics
            and "operating_margin" in available_metrics
        ):
            gross_margin = stock_data["gross_margin"]
            operating_margin = stock_data["operating_margin"]
            features[f"{symbol}_financial_margin_efficiency"] = operating_margin / (
                gross_margin + 1e-8
            )

        # 성장성 지표 조합
        if (
            "revenue_growth" in available_metrics
            and "earnings_growth" in available_metrics
        ):
            revenue_growth = stock_data["revenue_growth"]
            earnings_growth = stock_data["earnings_growth"]
            features[f"{symbol}_financial_growth_quality"] = earnings_growth / (
                revenue_growth + 1e-8
            )

        return features

    def _create_momentum_combination_features(
        self, stock_data: pd.DataFrame, symbol: str, available_indicators: List[str]
    ) -> Dict:
        """모멘텀 지표 조합 피처 생성"""
        features = {}

        # RSI와 스토캐스틱 조합
        if "rsi" in available_indicators and "stoch" in available_indicators:
            rsi = stock_data["rsi"]
            stoch = stock_data["stoch"]
            features[f"{symbol}_tech_rsi_stoch_divergence"] = rsi - stoch

        # CCI와 MFI 조합
        if "cci" in available_indicators and "mfi" in available_indicators:
            cci = stock_data["cci"]
            mfi = stock_data["mfi"]
            features[f"{symbol}_tech_cci_mfi_signal"] = (cci > 0).astype(float) * (
                mfi > 50
            ).astype(float)

        return features

    def _create_trend_combination_features(
        self, stock_data: pd.DataFrame, symbol: str, available_indicators: List[str]
    ) -> Dict:
        """트렌드 지표 조합 피처 생성"""
        features = {}

        # MACD와 ADX 조합
        if "macd" in available_indicators and "adx" in available_indicators:
            macd = stock_data["macd"]
            adx = stock_data["adx"]
            features[f"{symbol}_tech_macd_adx_strength"] = macd * (adx / 100)

        # SMA와 EMA 조합
        if "sma" in available_indicators and "ema" in available_indicators:
            sma = stock_data["sma"]
            ema = stock_data["ema"]
            features[f"{symbol}_tech_sma_ema_divergence"] = (ema - sma) / sma

        return features

    def _create_macro_features(
        self, macro_data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """매크로 피처 생성"""
        if macro_data is None or macro_data.empty:
            return pd.DataFrame()

        # 매크로 데이터에서 공통 피처 추출
        macro_features = {}

        # 주요 매크로 지표들
        macro_indicators = ["vix", "tnx", "irx", "spy", "uup", "gld", "qqq", "tlt"]

        for indicator in macro_indicators:
            if f"{indicator}_close" in macro_data.columns:
                macro_features[f"macro_{indicator}_close"] = macro_data[
                    f"{indicator}_close"
                ]
            if f"{indicator}_rsi" in macro_data.columns:
                macro_features[f"macro_{indicator}_rsi"] = macro_data[
                    f"{indicator}_rsi"
                ]
            if f"{indicator}_volatility" in macro_data.columns:
                macro_features[f"macro_{indicator}_volatility"] = macro_data[
                    f"{indicator}_volatility"
                ]

        # 인덱스가 있는 DataFrame 생성
        if macro_features:
            return pd.DataFrame(macro_features, index=macro_data.index)
        else:
            return pd.DataFrame(index=macro_data.index)

    def _create_regime_features(
        self, market_regime: Dict, symbol: str, stock_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """시장 체제 피처 생성"""
        regime_features = {}

        # 체제별 원핫 인코딩
        regimes = ["BULLISH", "BEARISH", "SIDEWAYS", "VOLATILE"]
        current_regime = market_regime.get("regime", "NEUTRAL")

        for regime in regimes:
            regime_features[f"{symbol}_regime_{regime.lower()}"] = (
                1.0 if current_regime == regime else 0.0
            )

        # 체제 신뢰도
        regime_features[f"{symbol}_regime_confidence"] = market_regime.get(
            "confidence", 0.5
        )

        # 인덱스가 있는 DataFrame 생성
        if regime_features:
            if stock_data is not None:
                # stock_data의 인덱스 사용
                return pd.DataFrame(regime_features, index=stock_data.index)
            else:
                # 기본 인덱스 사용
                temp_index = pd.date_range(start="2020-01-01", periods=1000, freq="D")
                return pd.DataFrame(regime_features, index=temp_index)
        else:
            return pd.DataFrame()

    def _combine_universal_features(
        self,
        swing_features: pd.DataFrame,
        financial_features: pd.DataFrame,
        technical_features: pd.DataFrame,
        macro_features: pd.DataFrame,
        regime_features: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Universal 모드: 모든 종목 피처 통합"""
        combined = swing_features.copy()

        if not financial_features.empty:
            combined = pd.concat([combined, financial_features], axis=1)

        if not technical_features.empty:
            combined = pd.concat([combined, technical_features], axis=1)

        if not macro_features.empty:
            combined = pd.concat([combined, macro_features], axis=1)

        if not regime_features.empty:
            combined = pd.concat([combined, regime_features], axis=1)

        return combined

    def _combine_individual_features(
        self,
        swing_features: pd.DataFrame,
        financial_features: pd.DataFrame,
        technical_features: pd.DataFrame,
        macro_features: pd.DataFrame,
        regime_features: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Individual 모드: 개별 종목 피처만"""
        combined = swing_features.copy()

        if not financial_features.empty:
            combined = pd.concat([combined, financial_features], axis=1)

        if not technical_features.empty:
            combined = pd.concat([combined, technical_features], axis=1)

        if not macro_features.empty:
            combined = pd.concat([combined, macro_features], axis=1)

        if not regime_features.empty:
            combined = pd.concat([combined, regime_features], axis=1)

        return combined

    def _create_feature_metadata(
        self, features: pd.DataFrame, symbol: str, mode: str
    ) -> Dict:
        """동적 피처 메타데이터 생성"""
        metadata = {
            "symbol": symbol,
            "mode": mode,
            "feature_count": len(features.columns),
            "feature_names": list(features.columns),
            "data_shape": features.shape,
            "created_at": datetime.now().isoformat(),
        }

        # 피처 카테고리별 상세 분석
        feature_categories = {
            "swing_features": [
                col
                for col in features.columns
                if col.startswith(f"{symbol}_")
                and not col.startswith(f"{symbol}_regime_")
            ],
            "financial_features": [
                col
                for col in features.columns
                if col.startswith(f"{symbol}_financial_")
            ],
            "technical_features": [
                col
                for col in features.columns
                if col.startswith(f"{symbol}_technical_")
            ],
            "macro_features": [
                col for col in features.columns if col.startswith("macro_")
            ],
            "regime_features": [
                col for col in features.columns if col.startswith(f"{symbol}_regime_")
            ],
        }

        metadata["feature_categories"] = {
            category: len(features) for category, features in feature_categories.items()
        }

        # 피처 통계 정보
        metadata["feature_stats"] = {
            "total_features": len(features.columns),
            "swing_features": len(feature_categories["swing_features"]),
            "financial_features": len(feature_categories["financial_features"]),
            "technical_features": len(feature_categories["technical_features"]),
            "macro_features": len(feature_categories["macro_features"]),
            "regime_features": len(feature_categories["regime_features"]),
        }

        return metadata

    def create_features(
        self,
        stock_data: pd.DataFrame,
        symbol: str,
        market_regime: Dict,
        macro_data: Optional[pd.DataFrame] = None,
        mode: str = "individual",  # "individual" or "universal"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        피처 생성 및 메타데이터 반환 (기존 호환성 유지)

        Args:
            stock_data: 주식 데이터
            symbol: 종목 심볼
            market_regime: 시장 체제 정보
            macro_data: 매크로 데이터
            mode: 피처 생성 모드 ("individual" 또는 "universal")

        Returns:
            (features_df, feature_metadata)
        """
        # 동적 피처 생성 시스템 사용
        return self.create_dynamic_features(
            stock_data, symbol, market_regime, macro_data, mode
        )

    def _create_dynamic_feature_metadata(
        self, features: pd.DataFrame, symbol: str, mode: str, structure_analysis: Dict
    ) -> Dict:
        """동적 피처 메타데이터 생성"""
        metadata = {
            "symbol": symbol,
            "mode": mode,
            "data_complexity": structure_analysis["data_complexity"],
            "feature_count": len(features.columns),
            "feature_names": list(features.columns),
            "data_shape": features.shape,
            "structure_analysis": structure_analysis,
            "created_at": datetime.now().isoformat(),
        }

        # 피처 카테고리별 분석
        feature_categories = {
            "swing_features": [
                col
                for col in features.columns
                if col.startswith(f"{symbol}_")
                and "financial" not in col
                and "tech" not in col
            ],
            "financial_features": [
                col for col in features.columns if "financial" in col
            ],
            "technical_features": [col for col in features.columns if "tech" in col],
            "macro_features": [
                col for col in features.columns if col.startswith("macro_")
            ],
            "regime_features": [
                col for col in features.columns if col.startswith(f"{symbol}_regime_")
            ],
        }

        metadata["feature_categories"] = {
            category: len(features) for category, features in feature_categories.items()
        }

        return metadata

    def save_feature_info(self, filepath: str) -> bool:
        """피처 정보를 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.feature_info, f, indent=2, ensure_ascii=False)

            logger.info(f"피처 정보 저장 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"피처 정보 저장 실패: {e}")
            return False

    def load_feature_info(self, filepath: str) -> bool:
        """피처 정보를 파일에서 로드"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"피처 정보 파일이 없습니다: {filepath}")
                return False

            with open(filepath, "r", encoding="utf-8") as f:
                self.feature_info = json.load(f)

            logger.info(f"피처 정보 로드 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"피처 정보 로드 실패: {e}")
            return False

    # 기술적 지표 계산 메서드들 (기존 neural_stock_predictor.py에서 가져옴)
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
        """스윙 돈치안 채널 계산"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        upper = high.rolling(20).max()
        lower = low.rolling(20).min()
        middle = (upper + lower) / 2

        return (close - middle) / (upper - lower)

    def _calculate_stoch_donchian(self, data: pd.DataFrame) -> pd.Series:
        """스토캐스틱 돈치안 계산"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        upper = high.rolling(20).max()
        lower = low.rolling(20).min()

        stoch = (close - lower) / (upper - lower)
        return stoch.rolling(14).mean()

    def _calculate_whipsaw_prevention(self, data: pd.DataFrame) -> pd.Series:
        """휩쏘우 방지 계산"""
        close = data["close"]
        atr = self._calculate_atr(data)

        # ATR 기반 필터
        atr_filter = atr > atr.rolling(20).mean()

        # 모멘텀 기반 필터
        momentum = close.pct_change(5)
        momentum_filter = abs(momentum) > momentum.rolling(20).std()

        return (atr_filter & momentum_filter).astype(float)

    def _calculate_donchian_rsi_whipsaw(self, data: pd.DataFrame) -> pd.Series:
        """돈치안 RSI 휩쏘우 계산"""
        donchian = self._calculate_swing_donchian(data)
        rsi = self._calculate_swing_rsi(data)

        # 돈치안과 RSI의 조합
        return donchian * rsi

    def _calculate_volatility_filtered_breakout(self, data: pd.DataFrame) -> pd.Series:
        """변동성 필터링 브레이크아웃 계산"""
        breakout = self._calculate_volatility_breakout(data)
        volatility = data["close"].pct_change().rolling(20).std()

        # 변동성 필터 적용
        volatility_filter = volatility > volatility.rolling(50).mean()

        return breakout * volatility_filter

    def _calculate_multi_timeframe_whipsaw(self, data: pd.DataFrame) -> pd.Series:
        """멀티 타임프레임 휩쏘우 계산"""
        # 단기 모멘텀
        short_momentum = data["close"].pct_change(5)

        # 중기 모멘텀
        medium_momentum = data["close"].pct_change(20)

        # 장기 모멘텀
        long_momentum = data["close"].pct_change(60)

        # 모멘텀 일치도
        momentum_agreement = (
            (short_momentum > 0) & (medium_momentum > 0) & (long_momentum > 0)
        ) | ((short_momentum < 0) & (medium_momentum < 0) & (long_momentum < 0))

        return momentum_agreement.astype(float)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def _calculate_whipsaw_prevention(self, data: pd.DataFrame) -> pd.Series:
        """휩쏘우 방지 계산"""
        close = data["close"]
        atr = self._calculate_atr(data)

        # ATR 기반 필터
        atr_filter = atr > atr.rolling(20).mean()

        # 모멘텀 기반 필터
        momentum = close.pct_change(5)
        momentum_filter = abs(momentum) > momentum.rolling(20).std()

        return (atr_filter & momentum_filter).astype(float)

    def _calculate_donchian_rsi_whipsaw(self, data: pd.DataFrame) -> pd.Series:
        """돈치안 RSI 휩쏘우 계산"""
        donchian = self._calculate_swing_donchian(data)
        rsi = self._calculate_swing_rsi(data)

        # 돈치안과 RSI의 조합
        return donchian * rsi

    def _calculate_volatility_filtered_breakout(self, data: pd.DataFrame) -> pd.Series:
        """변동성 필터링 브레이크아웃 계산"""
        breakout = self._calculate_volatility_breakout(data)
        volatility = data["close"].pct_change().rolling(20).std()

        # 변동성 필터 적용
        volatility_filter = volatility > volatility.rolling(50).mean()

        return breakout * volatility_filter

    def _calculate_multi_timeframe_whipsaw(self, data: pd.DataFrame) -> pd.Series:
        """멀티 타임프레임 휩쏘우 계산"""
        # 단기 모멘텀
        short_momentum = data["close"].pct_change(5)

        # 중기 모멘텀
        medium_momentum = data["close"].pct_change(20)

        # 장기 모멘텀
        long_momentum = data["close"].pct_change(60)

        # 모멘텀 일치도
        momentum_agreement = (
            (short_momentum > 0) & (medium_momentum > 0) & (long_momentum > 0)
        ) | ((short_momentum < 0) & (medium_momentum < 0) & (long_momentum < 0))

        return momentum_agreement.astype(float)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr
