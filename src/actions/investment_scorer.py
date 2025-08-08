"""
투자추천 지수 생성기
신경망 예측값 + 시장 체제 정보를 종합하여 -1~1 스케일의 투자 점수 생성
- 시장 체제별 가중치 적용
- 변동성 페널티 적용 (config_swing.json의 volatility_penalty 활용)
- 리스크 조정 메커니즘
- 포지션 사이징 권고
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class InvestmentScoreGenerator:
    """
    투자추천 지수 생성기

    기능:
    - 신경망 예측값을 시장 체제에 따라 조정
    - 변동성 기반 리스크 페널티 적용
    - 최종 투자 점수 (-1 ~ 1) 생성
    - 포지션 사이징 권고
    """

    def __init__(self, config: Dict):
        self.config = config
        self.scoring_config = config.get("scoring", {})

        # 시장 체제별 승수 (config에서 가져오거나 기본값 사용)
        self.regime_multipliers = self.scoring_config.get(
            "regime_multipliers",
            {"BULLISH": 1.2, "BEARISH": 0.3, "SIDEWAYS": 0.8, "VOLATILE": 0.6},
        )

        # 변동성 페널티 (config_swing.json에서 가져옴)
        self.volatility_penalty = self.scoring_config.get("volatility_penalty", 0.3)

        # 리스크 관리 파라미터
        self.risk_config = self.scoring_config.get("risk_management", {})

        # 자동 스케일링 설정
        self.auto_scaling_config = self.scoring_config.get("auto_scaling", {})
        self.enable_auto_scaling = self.auto_scaling_config.get("enable", True)
        self.target_range = self.auto_scaling_config.get("target_range", 1.6)  # -0.8 ~ 0.8
        self.min_score_spread = self.auto_scaling_config.get("min_score_spread", 0.1)  # 최소 점수 폭
        
        logger.info(f"InvestmentScoreGenerator 초기화 완료")
        logger.info(f"체제별 승수: {self.regime_multipliers}")
        logger.info(f"변동성 페널티: {self.volatility_penalty}")
        logger.info(f"자동 스케일링: {'활성화' if self.enable_auto_scaling else '비활성화'}")

    def calculate_stock_volatility(
        self, stock_data: pd.DataFrame, period: int = 20
    ) -> float:
        """
        개별 종목 변동성 계산

        Args:
            stock_data: 주식 데이터
            period: 계산 기간

        Returns:
            연율화 변동성
        """
        try:
            if "close" not in stock_data.columns or len(stock_data) < period:
                return 0.25  # 기본 변동성 25%

            # 일일 수익률
            returns = stock_data["close"].pct_change().dropna()

            if len(returns) < period:
                return 0.25

            # 최근 period일 변동성
            recent_vol = returns.tail(period).std()

            # 연율화 (252 거래일 기준)
            annualized_vol = recent_vol * np.sqrt(252)

            final_vol = float(np.clip(annualized_vol, 0.05, 2.0))  # 5% ~ 200% 클리핑

            # 디버깅: 변동성 계산 결과 로깅
            logger.debug(
                f"변동성 계산 - 최근 {period}일 변동성: {recent_vol:.4f}, 연율화: {annualized_vol:.4f}, 최종: {final_vol:.4f}"
            )

            return final_vol

        except Exception as e:
            logger.error(f"변동성 계산 실패: {e}")
            return 0.25

    def calculate_momentum_factor(self, stock_data: pd.DataFrame) -> float:
        """
        모멘텀 팩터 계산

        Args:
            stock_data: 주식 데이터

        Returns:
            모멘텀 점수 (-1 ~ 1)
        """
        try:
            if "close" not in stock_data.columns or len(stock_data) < 60:
                return 0.0

            close = stock_data["close"]

            # 다양한 기간 모멘텀
            momentum_1m = (
                close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
            )
            momentum_3m = (
                close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0
            )
            momentum_6m = (
                close.iloc[-1] / close.iloc[-126] - 1 if len(close) >= 126 else 0
            )

            # 가중 평균 모멘텀
            momentum_score = momentum_1m * 0.5 + momentum_3m * 0.3 + momentum_6m * 0.2

            # -1 ~ 1 정규화
            return float(np.clip(momentum_score * 2, -1, 1))

        except Exception as e:
            logger.error(f"모멘텀 계산 실패: {e}")
            return 0.0

    def calculate_technical_strength(self, stock_data: pd.DataFrame) -> float:
        """
        기술적 강도 계산

        Args:
            stock_data: 주식 데이터

        Returns:
            기술적 강도 (-1 ~ 1)
        """
        try:
            if "close" not in stock_data.columns or len(stock_data) < 50:
                return 0.0

            close = stock_data["close"]

            # 이동평균 대비 위치
            ma_20 = close.rolling(20).mean()
            ma_50 = close.rolling(50).mean()

            if ma_20.isna().all() or ma_50.isna().all():
                return 0.0

            # 현재가의 이동평균 대비 위치
            current_price = close.iloc[-1]
            ma_20_current = ma_20.iloc[-1]
            ma_50_current = ma_50.iloc[-1]

            # 이동평균 배열
            ma_alignment = 0
            if current_price > ma_20_current > ma_50_current:
                ma_alignment = 1  # 상승 배열
            elif current_price < ma_20_current < ma_50_current:
                ma_alignment = -1  # 하락 배열

            # RSI 계산
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_current = rsi.iloc[-1] if not rsi.isna().all() else 50

            # RSI를 -1 ~ 1로 정규화
            rsi_normalized = (rsi_current - 50) / 50

            # 기술적 강도 = 이동평균 배열 + RSI
            technical_strength = ma_alignment * 0.6 + rsi_normalized * 0.4

            return float(np.clip(technical_strength, -1, 1))

        except Exception as e:
            logger.error(f"기술적 강도 계산 실패: {e}")
            return 0.0

    def calculate_liquidity_factor(self, stock_data: pd.DataFrame) -> float:
        """
        유동성 팩터 계산

        Args:
            stock_data: 주식 데이터 (volume 포함)

        Returns:
            유동성 점수 (0 ~ 1)
        """
        try:
            if "volume" not in stock_data.columns or len(stock_data) < 20:
                return 0.5  # 기본값

            volume = stock_data["volume"]
            recent_volume = volume.tail(20)

            if recent_volume.sum() == 0:
                return 0.1  # 거래량 없음

            # 최근 거래량의 변화
            volume_ma = recent_volume.mean()
            current_volume = volume.iloc[-1]

            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1

            # 0 ~ 1 정규화
            liquidity_score = min(1.0, volume_ratio / 2)

            return float(liquidity_score)

        except Exception as e:
            logger.error(f"유동성 계산 실패: {e}")
            return 0.5

    def auto_scale_scores(self, raw_scores: List[Dict]) -> List[Dict]:
        """
        점수 분포를 분석해서 자동 스케일링
        
        Args:
            raw_scores: 원본 투자 점수 리스트 (각 딕셔너리는 final_score 키를 포함)
            
        Returns:
            스케일링된 투자 점수 리스트
        """
        try:
            if not raw_scores or not self.enable_auto_scaling:
                return raw_scores
                
            # 원본 점수들 추출
            original_scores = [score.get("final_score", 0.0) for score in raw_scores]
            
            if not original_scores:
                return raw_scores
            
            # 현재 분포 분석
            current_min = min(original_scores)
            current_max = max(original_scores)
            current_range = current_max - current_min
            current_center = (current_max + current_min) / 2
            
            logger.debug(f"점수 분포 분석 - Min: {current_min:.4f}, Max: {current_max:.4f}, Range: {current_range:.4f}, Center: {current_center:.4f}")
            
            # 점수 범위가 너무 작으면 스케일링 건너뛰기
            if current_range < self.min_score_spread:
                logger.debug(f"점수 범위가 너무 작음 ({current_range:.4f} < {self.min_score_spread}), 스케일링 건너뛰기")
                return raw_scores
            
            # 스케일 팩터 계산
            scale_factor = self.target_range / current_range
            
            # 너무 극단적인 스케일링 방지
            scale_factor = np.clip(scale_factor, 0.5, 3.0)
            
            logger.info(f"📊 자동 스케일링 적용 - 팩터: {scale_factor:.2f}, 목표 범위: {self.target_range:.1f}")
            
            # 스케일링된 점수 리스트 생성
            scaled_scores = []
            for i, score_dict in enumerate(raw_scores):
                original_score = original_scores[i]
                
                # 중심을 0으로 이동 후 스케일링
                centered = original_score - current_center
                scaled = centered * scale_factor
                
                # -1 ~ 1 범위로 클리핑
                final_scaled_score = float(np.clip(scaled, -1, 1))
                
                # 새로운 딕셔너리 생성 (기존 정보는 유지)
                scaled_dict = score_dict.copy()
                scaled_dict["final_score"] = final_scaled_score
                scaled_dict["original_score"] = original_score
                scaled_dict["scaling_info"] = {
                    "scale_factor": scale_factor,
                    "original_range": current_range,
                    "target_range": self.target_range,
                    "original_center": current_center
                }
                
                scaled_scores.append(scaled_dict)
                
                logger.debug(f"{score_dict.get('symbol', 'UNKNOWN')}: {original_score:.4f} → {final_scaled_score:.4f}")
            
            # 스케일링 후 분포 확인
            new_scores = [s["final_score"] for s in scaled_scores]
            new_min, new_max = min(new_scores), max(new_scores)
            logger.info(f"✅ 스케일링 완료 - 새 범위: [{new_min:.3f}, {new_max:.3f}]")
            
            return scaled_scores
            
        except Exception as e:
            logger.error(f"자동 스케일링 실패: {e}")
            return raw_scores

    def apply_risk_adjustments(
        self, base_score: float, stock_data: pd.DataFrame, market_regime: Dict
    ) -> float:
        """
        리스크 조정 적용

        Args:
            base_score: 기본 점수
            stock_data: 주식 데이터
            market_regime: 시장 체제 정보

        Returns:
            리스크 조정된 점수
        """
        try:
            adjusted_score = base_score

            # 1. 변동성 페널티
            volatility = self.calculate_stock_volatility(stock_data)
            volatility_threshold = self.risk_config.get("volatility_threshold", 0.30)

            if volatility > volatility_threshold:
                vol_penalty = min(
                    self.volatility_penalty, (volatility - volatility_threshold) * 2
                )
                adjusted_score *= 1 - vol_penalty
                logger.debug(
                    f"변동성 페널티 적용: {vol_penalty:.3f} (변동성: {volatility:.3f})"
                )

            # 2. 시장 체제 변동성 조정
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_confidence = market_regime.get("confidence", 0.5)

            if regime == "VOLATILE" and regime_confidence > 0.7:
                volatile_penalty = 0.2
                adjusted_score *= 1 - volatile_penalty
                logger.debug(f"변동성 시장 페널티 적용: {volatile_penalty}")

            # 3. 유동성 조정
            liquidity = self.calculate_liquidity_factor(stock_data)
            if liquidity < 0.3:  # 낮은 유동성
                liquidity_penalty = (0.3 - liquidity) * 0.5
                adjusted_score *= 1 - liquidity_penalty
                logger.debug(f"유동성 페널티 적용: {liquidity_penalty:.3f}")

            # 4. 최대 드로우다운 리스크 조정
            max_dd_threshold = self.risk_config.get("max_drawdown_threshold", 0.25)
            recent_dd = self.calculate_recent_drawdown(stock_data)

            if recent_dd > max_dd_threshold:
                dd_penalty = min(0.5, (recent_dd - max_dd_threshold) * 2)
                adjusted_score *= 1 - dd_penalty
                logger.debug(
                    f"드로우다운 페널티 적용: {dd_penalty:.3f} (DD: {recent_dd:.3f})"
                )

            return float(np.clip(adjusted_score, -1, 1))

        except Exception as e:
            logger.error(f"리스크 조정 실패: {e}")
            return base_score

    def calculate_recent_drawdown(
        self, stock_data: pd.DataFrame, period: int = 60
    ) -> float:
        """
        최근 드로우다운 계산

        Args:
            stock_data: 주식 데이터
            period: 계산 기간

        Returns:
            최대 드로우다운 (0 ~ 1)
        """
        try:
            if "close" not in stock_data.columns or len(stock_data) < period:
                return 0.0

            close = stock_data["close"].tail(period)

            # 누적 최고점
            cum_max = close.expanding().max()

            # 드로우다운 계산
            drawdown = (close - cum_max) / cum_max

            # 최대 드로우다운
            max_drawdown = abs(drawdown.min())

            return float(max_drawdown)

        except Exception as e:
            logger.error(f"드로우다운 계산 실패: {e}")
            return 0.0

    def generate_investment_score(
        self,
        neural_prediction: float,
        stock_data: pd.DataFrame,
        symbol: str,
        market_regime: Dict,
    ) -> Dict:
        """
        최종 투자 추천 지수 생성

        Args:
            neural_prediction: 신경망 예측값 (-1 ~ 1)
            stock_data: 주식 데이터
            symbol: 종목 코드
            market_regime: 시장 체제 정보

        Returns:
            투자 점수 및 상세 정보
        """
        try:
            logger.debug(f"{symbol} 투자 점수 생성 시작...")

            # neural_prediction None 체크 및 멀티타겟 처리
            if neural_prediction is None:
                logger.warning(f"{symbol} 신경망 예측값이 None입니다. 기본값 0.0 사용")
                neural_prediction = 0.0
            elif isinstance(neural_prediction, dict):
                # 멀티타겟 예측의 경우 주요 타겟값 사용
                if "target_22d" in neural_prediction:
                    neural_prediction = neural_prediction["target_22d"]
                    logger.debug(
                        f"{symbol} 멀티타겟 예측 → 22일 타겟 사용: {neural_prediction:.4f}"
                    )
                else:
                    # 첫 번째 값 사용
                    neural_prediction = list(neural_prediction.values())[0]
                    logger.debug(
                        f"{symbol} 멀티타겟 예측 → 첫 번째 값 사용: {neural_prediction:.4f}"
                    )

            # 1. 기본 점수 (신경망 예측값)
            base_score = float(np.clip(neural_prediction, -1, 1))

            # 2. 시장 체제 조정
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_multiplier = self.regime_multipliers.get(regime, 0.8)
            regime_confidence = market_regime.get("confidence", 0.5)

            # 체제 신뢰도가 낮으면 승수 효과 감소
            adjusted_multiplier = 1 + (regime_multiplier - 1) * regime_confidence
            regime_adjusted_score = base_score * adjusted_multiplier

            # 3. 추가 팩터들 계산
            momentum_factor = self.calculate_momentum_factor(stock_data)
            technical_strength = self.calculate_technical_strength(stock_data)
            liquidity_factor = self.calculate_liquidity_factor(stock_data)

            # 4. 팩터 가중 통합
            factor_weights = self.scoring_config.get(
                "factor_weights",
                {"neural": 0.4, "momentum": 0.25, "technical": 0.25, "liquidity": 0.1},
            )

            integrated_score = (
                regime_adjusted_score * factor_weights["neural"]
                + momentum_factor * factor_weights["momentum"]
                + technical_strength * factor_weights["technical"]
                + liquidity_factor * factor_weights["liquidity"]
            )

            # 5. 리스크 조정 적용
            final_score = self.apply_risk_adjustments(
                integrated_score, stock_data, market_regime
            )

            # 6. 포지션 사이징 계산
            position_size = self.calculate_position_size(
                final_score, stock_data, market_regime
            )

            # 7. 홀딩 기간 추정
            holding_period = self.estimate_holding_period(final_score, market_regime)

            # 8. 신뢰도 계산
            confidence = self.calculate_confidence(
                final_score, regime_confidence, stock_data
            )

            result = {
                "symbol": symbol,
                "final_score": final_score,
                "confidence": confidence,
                "position_size": position_size,
                "holding_period": holding_period,
                "components": {
                    "neural_prediction": neural_prediction,
                    "base_score": base_score,
                    "regime_adjusted_score": regime_adjusted_score,
                    "momentum_factor": momentum_factor,
                    "technical_strength": technical_strength,
                    "liquidity_factor": liquidity_factor,
                    "integrated_score": integrated_score,
                },
                "market_info": {
                    "regime": regime,
                    "regime_confidence": regime_confidence,
                    "regime_multiplier": adjusted_multiplier,
                },
                "risk_metrics": {
                    "volatility": self.calculate_stock_volatility(stock_data),
                    "recent_drawdown": self.calculate_recent_drawdown(stock_data),
                    "liquidity": liquidity_factor,
                },
                "timestamp": datetime.now().isoformat(),
            }

            logger.debug(
                f"{symbol} 최종 점수: {final_score:.4f} (신뢰도: {confidence:.3f})"
            )

            return result

        except Exception as e:
            logger.error(f"{symbol} 투자 점수 생성 실패: {e}")
            return self._get_default_score(symbol, neural_prediction, market_regime)

    def calculate_position_size(
        self, score: float, stock_data: pd.DataFrame, market_regime: Dict
    ) -> float:
        """
        포지션 사이징 계산

        Args:
            score: 투자 점수
            stock_data: 주식 데이터
            market_regime: 시장 체제

        Returns:
            권장 포지션 크기 (0 ~ 1)
        """
        try:
            # 기본 포지션 크기 (점수 절댓값 기반)
            base_position = abs(score) * 0.1  # 최대 10%

            # 변동성 조정
            volatility = self.calculate_stock_volatility(stock_data)
            vol_adjustment = max(
                0.5, 1 - (volatility - 0.2) * 2
            )  # 변동성이 높으면 줄임

            # 시장 체제 조정
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_adjustments = {
                "BULLISH": 1.2,
                "BEARISH": 0.5,
                "SIDEWAYS": 0.8,
                "VOLATILE": 0.6,
            }

            regime_adj = regime_adjustments.get(regime, 0.8)

            # 최종 포지션 크기
            position_size = base_position * vol_adjustment * regime_adj

            # 클리핑 (최대 15%)
            return float(np.clip(position_size, 0, 0.15))

        except Exception as e:
            logger.error(f"포지션 사이징 실패: {e}")
            return 0.05  # 기본값

    def estimate_holding_period(self, score: float, market_regime: Dict) -> int:
        """
        홀딩 기간 추정 (일수)

        Args:
            score: 투자 점수
            market_regime: 시장 체제

        Returns:
            권장 홀딩 기간 (일)
        """
        try:
            # 기본 홀딩 기간 (점수 강도에 반비례)
            base_period = 30  # 30일 기본

            score_intensity = abs(score)
            if score_intensity > 0.8:
                period_multiplier = 0.5  # 강한 신호는 짧게
            elif score_intensity > 0.5:
                period_multiplier = 0.7
            else:
                period_multiplier = 1.2  # 약한 신호는 길게

            # 시장 체제별 조정
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_periods = {
                "BULLISH": 1.5,  # 상승장에서는 길게
                "BEARISH": 0.5,  # 하락장에서는 짧게
                "SIDEWAYS": 1.0,  # 횡보장에서는 보통
                "VOLATILE": 0.7,  # 변동성 장에서는 짧게
            }

            regime_multiplier = regime_periods.get(regime, 1.0)

            # 최종 홀딩 기간
            holding_period = base_period * period_multiplier * regime_multiplier

            return int(np.clip(holding_period, 5, 90))  # 5일 ~ 90일

        except Exception as e:
            logger.error(f"홀딩 기간 추정 실패: {e}")
            return 30  # 기본값

    def calculate_confidence(
        self, score: float, regime_confidence: float, stock_data: pd.DataFrame
    ) -> float:
        """
        전체 신뢰도 계산

        Args:
            score: 최종 점수
            regime_confidence: 시장 체제 신뢰도
            stock_data: 주식 데이터

        Returns:
            종합 신뢰도 (0 ~ 1)
        """
        try:
            # 점수 강도 기반 신뢰도
            score_confidence = abs(score)

            # 데이터 품질 기반 신뢰도
            data_quality = min(1.0, len(stock_data) / 252)  # 1년 데이터 기준

            # 변동성 기반 신뢰도 (낮은 변동성이 높은 신뢰도)
            volatility = self.calculate_stock_volatility(stock_data)
            vol_confidence = max(0.3, 1 - (volatility - 0.2) / 0.5)

            # 종합 신뢰도
            overall_confidence = (
                score_confidence * 0.4
                + regime_confidence * 0.3
                + data_quality * 0.15
                + vol_confidence * 0.15
            )

            return float(np.clip(overall_confidence, 0, 1))

        except Exception as e:
            logger.error(f"신뢰도 계산 실패: {e}")
            return 0.5

    def _get_default_score(
        self, symbol: str, neural_prediction: float, market_regime: Dict
    ) -> Dict:
        """
        기본 점수 (오류시)
        """
        return {
            "symbol": symbol,
            "final_score": 0.0,
            "confidence": 0.3,
            "position_size": 0.05,
            "holding_period": 30,
            "components": {
                "neural_prediction": neural_prediction,
                "base_score": 0.0,
                "regime_adjusted_score": 0.0,
                "momentum_factor": 0.0,
                "technical_strength": 0.0,
                "liquidity_factor": 0.5,
                "integrated_score": 0.0,
            },
            "market_info": {
                "regime": market_regime.get("regime", "SIDEWAYS"),
                "regime_confidence": market_regime.get("confidence", 0.5),
                "regime_multiplier": 1.0,
            },
            "risk_metrics": {
                "volatility": 0.25,
                "recent_drawdown": 0.0,
                "liquidity": 0.5,
            },
            "timestamp": datetime.now().isoformat(),
        }


class PortfolioScoreAggregator:
    """
    포트폴리오 레벨에서 개별 종목 점수들을 집계하고 관리
    """

    def __init__(self, config: Dict):
        self.config = config
        self.portfolio_config = config.get("portfolio", {})
        
        # 자동 스케일링을 위한 InvestmentScoreGenerator 인스턴스
        self.score_generator = None

    def set_score_generator(self, score_generator):
        """자동 스케일링을 위한 score_generator 설정"""
        self.score_generator = score_generator

    def aggregate_scores(self, individual_scores: List[Dict]) -> Dict:
        """
        개별 종목 점수들을 포트폴리오 레벨로 집계 (자동 스케일링 포함)

        Args:
            individual_scores: 개별 종목 점수 리스트

        Returns:
            포트폴리오 집계 결과
        """
        try:
            if not individual_scores:
                return {"portfolio_score": 0.0, "total_positions": 0}
            
            # 자동 스케일링 적용 (활성화된 경우)
            if self.score_generator:
                logger.debug(f"자동 스케일링 적용 전 점수 수: {len(individual_scores)}")
                individual_scores = self.score_generator.auto_scale_scores(individual_scores)
                logger.debug(f"자동 스케일링 적용 후 점수 수: {len(individual_scores)}")
            else:
                logger.debug("자동 스케일링 건너뛰기 (score_generator가 설정되지 않음)")

            # 점수별 분류
            strong_buy = [s for s in individual_scores if s["final_score"] > 0.6]
            buy = [s for s in individual_scores if 0.3 < s["final_score"] <= 0.6]
            hold = [s for s in individual_scores if -0.3 <= s["final_score"] <= 0.3]
            sell = [s for s in individual_scores if -0.6 <= s["final_score"] < -0.3]
            strong_sell = [s for s in individual_scores if s["final_score"] < -0.6]

            # 포트폴리오 메트릭
            total_position_size = sum(s["position_size"] for s in individual_scores)
            weighted_score = sum(
                s["final_score"] * s["position_size"] for s in individual_scores
            )

            portfolio_score = (
                weighted_score / total_position_size if total_position_size > 0 else 0
            )

            # 신뢰도 가중 평균
            total_confidence = sum(
                s["confidence"] * s["position_size"] for s in individual_scores
            )
            portfolio_confidence = (
                total_confidence / total_position_size if total_position_size > 0 else 0
            )

            return {
                "portfolio_score": portfolio_score,
                "portfolio_confidence": portfolio_confidence,
                "total_position_size": total_position_size,
                "position_counts": {
                    "strong_buy": len(strong_buy),
                    "buy": len(buy),
                    "hold": len(hold),
                    "sell": len(sell),
                    "strong_sell": len(strong_sell),
                },
                "top_recommendations": sorted(
                    individual_scores, key=lambda x: x["final_score"], reverse=True
                )[:5],
                "bottom_recommendations": sorted(
                    individual_scores, key=lambda x: x["final_score"]
                )[:5],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"포트폴리오 집계 실패: {e}")
            return {"portfolio_score": 0.0, "total_positions": 0}
