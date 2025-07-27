"""
최종 매매 신호 생성기
투자 점수를 바탕으로 구체적인 매수/매도/보유 신호와 실행 권고사항을 생성합니다.
- 구체적 액션 권고 (강력매수/매수/보유/매도/강력매도)
- 진입/청산 타이밍 최적화
- 리스크 관리 권고
- 포트폴리오 레벨 권고사항
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class TradingSignalGenerator:
    """
    최종 매매 신호 생성기

    기능:
    - 투자 점수를 매수/매도 신호로 변환
    - 진입/청산 타이밍 최적화
    - 구체적 실행 권고사항 생성
    - 리스크 관리 권고
    """

    def __init__(self, config: Dict):
        self.config = config
        self.signal_config = config.get("signal_generation", {})

        # 신호 임계값 설정 (더 보수적으로 조정)
        self.signal_thresholds = self.signal_config.get(
            "thresholds",
            {
                "strong_buy": 0.7,
                "buy": 0.5,
                "hold_upper": 0.5,
                "hold_lower": -0.5,
                "sell": -0.5,
                "strong_sell": -0.7,
            },
        )

        # 신뢰도 최소 기준 (더 낮게, 0.3)
        self.min_confidence = self.signal_config.get("min_confidence", 0.3)

        # 포지션 관리 설정
        self.position_config = self.signal_config.get("position_management", {})

        logger.info(f"TradingSignalGenerator 초기화 완료")
        logger.info(f"신호 임계값: {self.signal_thresholds}")

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """임계점 업데이트"""
        try:
            self.signal_thresholds.update(new_thresholds)
            logger.info(f"임계점 업데이트 완료: {self.signal_thresholds}")
        except Exception as e:
            logger.error(f"임계점 업데이트 실패: {e}")

    def generate_signal(self, investment_score: Dict) -> Dict:
        """
        개별 종목 매매 신호 생성

        Args:
            investment_score: InvestmentScoreGenerator에서 생성한 투자 점수

        Returns:
            매매 신호 딕셔너리
        """
        try:
            symbol = investment_score["symbol"]
            score = investment_score.get("final_score", 0.0)
            confidence = investment_score.get("confidence", 0.3)

            # None 값 처리
            if score is None:
                logger.warning(f"{symbol} 점수가 None입니다. 기본값 0.0 사용")
                score = 0.0
            if confidence is None:
                logger.warning(f"{symbol} 신뢰도가 None입니다. 기본값 0.3 사용")
                confidence = 0.3

            logger.info(
                f"{symbol} 매매 신호 생성 시작 - 점수: {score:.4f}, 신뢰도: {confidence:.3f}"
            )

            # 1. 기본 신호 결정
            action, action_strength = self._determine_action(score, confidence)

            # 2. 진입/청산 타이밍 최적화
            entry_timing = self._optimize_entry_timing(investment_score)
            exit_timing = self._optimize_exit_timing(investment_score)

            # 3. 포지션 사이징 조정
            adjusted_position_size = self._adjust_position_size(
                investment_score, action
            )

            # 4. 리스크 관리 권고
            risk_management = self._generate_risk_management(investment_score)

            # 5. 실행 우선순위 계산
            execution_priority = self._calculate_execution_priority(
                score, confidence, action
            )

            # 6. 상세 권고사항 생성
            recommendations = self._generate_recommendations(investment_score, action)

            signal = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "action_strength": action_strength,
                "score": score,
                "confidence": confidence,
                "position_size": adjusted_position_size,
                "execution_priority": execution_priority,
                "timing": {"entry": entry_timing, "exit": exit_timing},
                "risk_management": risk_management,
                "recommendations": recommendations,
                "holding_period": investment_score["holding_period"],
                "market_regime": investment_score["market_info"]["regime"],
                "regime_confidence": investment_score["market_info"][
                    "regime_confidence"
                ],
            }

            logger.info(
                f"{symbol} 신호: {action} (강도: {action_strength:.2f}, 우선순위: {execution_priority})"
            )

            return signal

        except Exception as e:
            logger.error(f"매매 신호 생성 실패: {e}")
            return self._get_default_signal(investment_score.get("symbol", "UNKNOWN"))

    def _determine_action(self, score: float, confidence: float) -> Tuple[str, float]:
        """
        매매 액션 결정

        Args:
            score: 투자 점수 (-1 ~ 1)
            confidence: 신뢰도 (0 ~ 1)

        Returns:
            (액션, 강도) 튜플
        """
        try:
            # 신뢰도가 너무 낮으면 보유
            if confidence < self.min_confidence:
                return "HOLD", 0.5

            # 점수 기반 액션 결정
            if score >= self.signal_thresholds["strong_buy"]:
                action = "STRONG_BUY"
                strength = min(1.0, score * confidence)
            elif score >= self.signal_thresholds["buy"]:
                action = "BUY"
                strength = score * confidence * 0.8
            elif score >= self.signal_thresholds["hold_lower"]:
                action = "HOLD"
                strength = 0.5
            elif score >= self.signal_thresholds["sell"]:
                action = "SELL"
                strength = abs(score) * confidence * 0.8
            else:
                action = "STRONG_SELL"
                strength = min(1.0, abs(score) * confidence)

            return action, float(strength)

        except Exception as e:
            logger.error(f"액션 결정 실패: {e}")
            return "HOLD", 0.5

    def _optimize_entry_timing(self, investment_score: Dict) -> Dict:
        """
        진입 타이밍 최적화

        Args:
            investment_score: 투자 점수 정보

        Returns:
            진입 타이밍 권고
        """
        try:
            score = investment_score["final_score"]
            confidence = investment_score["confidence"]
            regime = investment_score["market_info"]["regime"]
            volatility = investment_score["risk_metrics"]["volatility"]

            # 즉시 진입 vs 분할 진입 vs 대기
            if confidence > 0.8 and abs(score) > 0.7:
                if volatility < 0.25:
                    timing_type = "IMMEDIATE"
                    urgency = "HIGH"
                else:
                    timing_type = "GRADUAL"
                    urgency = "MEDIUM"
            elif confidence > 0.6 and abs(score) > 0.5:
                timing_type = "GRADUAL"
                urgency = "MEDIUM"
            elif confidence > 0.4 and abs(score) > 0.3:
                timing_type = "PATIENT"
                urgency = "LOW"
            else:
                timing_type = "WAIT"
                urgency = "NONE"

            # 체제별 조정
            regime_adjustments = {
                "VOLATILE": {"wait_factor": 1.5, "gradual_preference": True},
                "SIDEWAYS": {"wait_factor": 1.2, "patience_required": True},
                "BULLISH": {"immediate_ok": True, "wait_factor": 0.8},
                "BEARISH": {"caution_required": True, "wait_factor": 1.3},
            }

            regime_adj = regime_adjustments.get(regime, {})

            # 분할 진입 세부사항
            if timing_type == "GRADUAL":
                entry_phases = self._calculate_entry_phases(score, volatility)
            else:
                entry_phases = None

            return {
                "type": timing_type,
                "urgency": urgency,
                "entry_phases": entry_phases,
                "regime_considerations": regime_adj,
                "optimal_time_windows": self._get_optimal_time_windows(
                    regime, volatility
                ),
                "risk_factors": {
                    "volatility_level": volatility,
                    "regime_stability": investment_score["market_info"][
                        "regime_confidence"
                    ],
                },
            }

        except Exception as e:
            logger.error(f"진입 타이밍 최적화 실패: {e}")
            return {"type": "WAIT", "urgency": "NONE"}

    def _optimize_exit_timing(self, investment_score: Dict) -> Dict:
        """
        청산 타이밍 최적화

        Args:
            investment_score: 투자 점수 정보

        Returns:
            청산 타이밍 권고
        """
        try:
            score = investment_score["final_score"]
            holding_period = investment_score["holding_period"]
            volatility = investment_score["risk_metrics"]["volatility"]
            recent_dd = investment_score["risk_metrics"]["recent_drawdown"]

            # 손절선 설정
            if volatility < 0.2:
                stop_loss = -0.05  # 5% 손절
            elif volatility < 0.3:
                stop_loss = -0.08  # 8% 손절
            else:
                stop_loss = -0.12  # 12% 손절

            # 이익실현선 설정
            if abs(score) > 0.8:
                take_profit_levels = [0.08, 0.15, 0.25]  # 강한 신호
            elif abs(score) > 0.5:
                take_profit_levels = [0.06, 0.12, 0.20]  # 중간 신호
            else:
                take_profit_levels = [0.04, 0.08, 0.15]  # 약한 신호

            # 디버깅: 실제 값들 로깅
            logger.debug(
                f"청산 타이밍 계산 - 점수: {score:.4f}, 변동성: {volatility:.4f}, 손절: {stop_loss:.1%}, 이익실현: {take_profit_levels}"
            )

            # 시간 기반 청산
            max_holding_days = min(90, holding_period * 1.5)

            # 트레일링 스탑 설정
            if volatility < 0.25:
                trailing_stop = 0.03  # 3% 트레일링
            else:
                trailing_stop = 0.05  # 5% 트레일링

            return {
                "stop_loss": stop_loss,
                "take_profit_levels": take_profit_levels,
                "max_holding_days": max_holding_days,
                "trailing_stop": trailing_stop,
                "exit_signals": {
                    "score_reversal_threshold": -score * 0.5,  # 점수 반전시 청산
                    "drawdown_limit": max(0.15, recent_dd + 0.05),
                    "volatility_spike_threshold": volatility * 1.5,
                },
                "partial_exit_strategy": {
                    "first_target_ratio": 0.3,  # 첫 목표가에서 30% 청산
                    "second_target_ratio": 0.5,  # 두번째 목표가에서 50% 청산
                    "final_target_ratio": 1.0,  # 마지막 목표가에서 전량 청산
                },
            }

        except Exception as e:
            logger.error(f"청산 타이밍 최적화 실패: {e}")
            return {"stop_loss": -0.08, "take_profit_levels": [0.06, 0.12]}

    def _adjust_position_size(self, investment_score: Dict, action: str) -> float:
        """
        액션에 따른 포지션 사이징 조정

        Args:
            investment_score: 투자 점수 정보
            action: 매매 액션

        Returns:
            조정된 포지션 크기
        """
        try:
            base_position = investment_score["position_size"]
            confidence = investment_score["confidence"]

            # 액션별 조정
            action_multipliers = {
                "STRONG_BUY": 1.2,
                "BUY": 1.0,
                "HOLD": 0.5,
                "SELL": 0.0,
                "STRONG_SELL": 0.0,
            }

            multiplier = action_multipliers.get(action, 0.5)
            adjusted_position = base_position * multiplier

            # 신뢰도 조정
            confidence_adjustment = 0.5 + (confidence * 0.5)
            adjusted_position *= confidence_adjustment

            # 최종 클리핑
            return float(np.clip(adjusted_position, 0, 0.15))

        except Exception as e:
            logger.error(f"포지션 사이징 조정 실패: {e}")
            return 0.05

    def _generate_risk_management(self, investment_score: Dict) -> Dict:
        """
        리스크 관리 권고 생성

        Args:
            investment_score: 투자 점수 정보

        Returns:
            리스크 관리 권고사항
        """
        try:
            volatility = investment_score["risk_metrics"]["volatility"]
            recent_dd = investment_score["risk_metrics"]["recent_drawdown"]
            liquidity = investment_score["risk_metrics"]["liquidity"]
            regime = investment_score["market_info"]["regime"]

            risk_level = "LOW"
            warnings = []

            # 변동성 체크
            if volatility > 0.4:
                risk_level = "HIGH"
                warnings.append("높은 변동성 - 포지션 크기 축소 권장")
            elif volatility > 0.3:
                risk_level = "MEDIUM"
                warnings.append("중간 변동성 - 신중한 진입 필요")

            # 드로우다운 체크
            if recent_dd > 0.2:
                risk_level = "HIGH"
                warnings.append(f"최근 드로우다운 {recent_dd:.1%} - 추가 하락 리스크")

            # 유동성 체크
            if liquidity < 0.3:
                warnings.append("낮은 유동성 - 청산시 슬리피지 주의")

            # 시장 체제별 리스크
            regime_risks = {
                "VOLATILE": "변동성 시장 - 급격한 방향 전환 가능",
                "BEARISH": "하락 시장 - 전반적 하락 압력 존재",
                "SIDEWAYS": "횡보 시장 - 방향성 부족으로 위험",
            }

            if regime in regime_risks:
                warnings.append(regime_risks[regime])

            # 리스크 완화 방안
            mitigation_strategies = []

            if volatility > 0.3:
                mitigation_strategies.append("변동성 기반 포지션 사이징 적용")

            if recent_dd > 0.15:
                mitigation_strategies.append("엄격한 손절선 적용")

            if liquidity < 0.5:
                mitigation_strategies.append("분할 진입/청산 전략 사용")

            return {
                "risk_level": risk_level,
                "warnings": warnings,
                "mitigation_strategies": mitigation_strategies,
                "monitoring_points": {
                    "volatility_threshold": volatility * 1.3,
                    "drawdown_alert": recent_dd + 0.05,
                    "liquidity_minimum": max(0.2, liquidity * 0.8),
                },
                "emergency_exit_conditions": [
                    f"변동성 {volatility * 1.5:.1%} 초과시",
                    f"드로우다운 {min(0.25, recent_dd + 0.1):.1%} 초과시",
                    "시장 체제 급변시",
                ],
            }

        except Exception as e:
            logger.error(f"리스크 관리 생성 실패: {e}")
            return {"risk_level": "MEDIUM", "warnings": ["분석 오류 발생"]}

    def _calculate_execution_priority(
        self, score: float, confidence: float, action: str
    ) -> int:
        """
        실행 우선순위 계산 (1=최고, 10=최저)

        Args:
            score: 투자 점수
            confidence: 신뢰도
            action: 매매 액션

        Returns:
            우선순위 (1-10)
        """
        try:
            # 기본 우선순위 (점수 * 신뢰도)
            base_priority = abs(score) * confidence

            # 액션별 가중치
            action_weights = {
                "STRONG_BUY": 1.0,
                "STRONG_SELL": 1.0,
                "BUY": 0.8,
                "SELL": 0.8,
                "HOLD": 0.3,
            }

            weight = action_weights.get(action, 0.5)
            final_priority = base_priority * weight

            # 1-10 스케일로 변환 (높은 값 = 낮은 순위)
            priority_rank = int(np.clip(11 - (final_priority * 10), 1, 10))

            return priority_rank

        except Exception as e:
            logger.error(f"우선순위 계산 실패: {e}")
            return 5

    def _generate_recommendations(self, investment_score: Dict, action: str) -> Dict:
        """
        상세 권고사항 생성

        Args:
            investment_score: 투자 점수 정보
            action: 매매 액션

        Returns:
            상세 권고사항
        """
        try:
            symbol = investment_score["symbol"]
            score = investment_score["final_score"]
            regime = investment_score["market_info"]["regime"]

            # 액션별 구체적 권고
            action_recommendations = {
                "STRONG_BUY": f"{symbol} 강력 매수 권장 - 적극적 포지션 구축",
                "BUY": f"{symbol} 매수 권장 - 점진적 포지션 증대",
                "HOLD": f"{symbol} 현 포지션 유지 - 추가 신호 대기",
                "SELL": f"{symbol} 매도 권장 - 포지션 축소",
                "STRONG_SELL": f"{symbol} 강력 매도 권장 - 즉시 포지션 청산",
            }

            primary_recommendation = action_recommendations.get(action, "신호 불분명")

            # 시장 체제별 추가 권고
            regime_advice = {
                "BULLISH": "상승 추세 지속 예상 - 길게 보유 고려",
                "BEARISH": "하락 압력 존재 - 신중한 접근 필요",
                "SIDEWAYS": "횡보 구간 - 단기 매매 전략 고려",
                "VOLATILE": "변동성 확대 - 리스크 관리 강화",
            }

            # 타이밍 권고
            timing_advice = []
            components = investment_score["components"]

            if components["momentum_factor"] > 0.5:
                timing_advice.append("모멘텀 강화 중 - 추세 추종 전략 유효")
            elif components["momentum_factor"] < -0.5:
                timing_advice.append("모멘텀 약화 중 - 신중한 진입 필요")

            if components["technical_strength"] > 0.5:
                timing_advice.append("기술적 강세 - 차트상 긍정적")
            elif components["technical_strength"] < -0.5:
                timing_advice.append("기술적 약세 - 차트상 부정적")

            # 주의사항
            cautions = []

            if investment_score["confidence"] < 0.6:
                cautions.append("신뢰도 보통 - 포지션 크기 조절 필요")

            if investment_score["risk_metrics"]["volatility"] > 0.3:
                cautions.append("높은 변동성 - 손절선 엄격 적용")

            if investment_score["risk_metrics"]["recent_drawdown"] > 0.15:
                cautions.append("최근 하락 - 추가 하락 리스크 존재")

            return {
                "primary_recommendation": primary_recommendation,
                "regime_advice": regime_advice.get(regime, ""),
                "timing_advice": timing_advice,
                "cautions": cautions,
                "key_factors": {
                    "score_drivers": self._identify_score_drivers(components),
                    "risk_factors": self._identify_risk_factors(investment_score),
                    "opportunity_factors": self._identify_opportunities(
                        investment_score
                    ),
                },
            }

        except Exception as e:
            logger.error(f"권고사항 생성 실패: {e}")
            return {"primary_recommendation": "분석 오류 발생"}

    def _identify_score_drivers(self, components: Dict) -> List[str]:
        """점수 주요 동인 식별"""
        drivers = []

        if components["momentum_factor"] > 0.3:
            drivers.append("모멘텀 강화")
        elif components["momentum_factor"] < -0.3:
            drivers.append("모멘텀 약화")

        if components["technical_strength"] > 0.3:
            drivers.append("기술적 강세")
        elif components["technical_strength"] < -0.3:
            drivers.append("기술적 약세")

        # neural_prediction이 딕셔너리일 수 있으므로 처리
        neural_pred = components["neural_prediction"]
        if isinstance(neural_pred, dict):
            neural_pred = neural_pred.get("target_22d", 0.0)

        if neural_pred > 0.5:
            drivers.append("AI 모델 긍정적")
        elif neural_pred < -0.5:
            drivers.append("AI 모델 부정적")

        return drivers

    def _identify_risk_factors(self, investment_score: Dict) -> List[str]:
        """리스크 요인 식별"""
        risks = []

        risk_metrics = investment_score["risk_metrics"]

        if risk_metrics["volatility"] > 0.3:
            risks.append("높은 변동성")

        if risk_metrics["recent_drawdown"] > 0.15:
            risks.append("최근 하락")

        if risk_metrics["liquidity"] < 0.4:
            risks.append("낮은 유동성")

        if investment_score["market_info"]["regime"] == "VOLATILE":
            risks.append("변동성 시장")

        return risks

    def _identify_opportunities(self, investment_score: Dict) -> List[str]:
        """기회 요인 식별"""
        opportunities = []

        if investment_score["confidence"] > 0.7:
            opportunities.append("높은 신뢰도")

        if investment_score["market_info"]["regime"] == "BULLISH":
            opportunities.append("상승 시장")

        if investment_score["risk_metrics"]["liquidity"] > 0.7:
            opportunities.append("높은 유동성")

        components = investment_score["components"]
        if (
            components["momentum_factor"] > 0.5
            and components["technical_strength"] > 0.5
        ):
            opportunities.append("기술적/모멘텀 동조")

        return opportunities

    def _calculate_entry_phases(self, score: float, volatility: float) -> List[Dict]:
        """분할 진입 계획 수립"""
        try:
            phases = []

            if abs(score) > 0.7:
                # 강한 신호 - 3단계 분할
                phases = [
                    {"phase": 1, "ratio": 0.4, "timing": "immediate"},
                    {"phase": 2, "ratio": 0.35, "timing": "1-2 days"},
                    {"phase": 3, "ratio": 0.25, "timing": "3-5 days"},
                ]
            elif abs(score) > 0.5:
                # 중간 신호 - 2단계 분할
                phases = [
                    {"phase": 1, "ratio": 0.6, "timing": "immediate"},
                    {"phase": 2, "ratio": 0.4, "timing": "2-3 days"},
                ]
            else:
                # 약한 신호 - 단일 진입
                phases = [{"phase": 1, "ratio": 1.0, "timing": "patient"}]

            return phases

        except Exception as e:
            logger.error(f"분할 진입 계획 수립 실패: {e}")
            return [{"phase": 1, "ratio": 1.0, "timing": "immediate"}]

    def _get_optimal_time_windows(self, regime: str, volatility: float) -> List[str]:
        """최적 진입 시간대 권고"""
        try:
            if regime == "VOLATILE":
                return ["장 초반 30분 후", "장 마감 30분 전"]
            elif regime == "SIDEWAYS":
                return ["장 중반 (11:00-14:00)", "장 마감 1시간 전"]
            else:
                return ["장 시작 1시간 후", "장 중반", "장 마감 전"]

        except Exception as e:
            logger.error(f"최적 시간대 계산 실패: {e}")
            return ["장중 언제든지"]

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


class PortfolioSignalAggregator:
    """
    포트폴리오 레벨에서 개별 종목 신호들을 집계하고 전체 포트폴리오 권고 생성
    """

    def __init__(self, config: Dict):
        self.config = config
        self.portfolio_config = config.get("portfolio", {})

    def aggregate_portfolio_signals(
        self, individual_signals: List[Dict], market_regime: Dict
    ) -> Dict:
        """
        포트폴리오 레벨 신호 집계

        Args:
            individual_signals: 개별 종목 신호 리스트
            market_regime: 시장 체제 정보

        Returns:
            포트폴리오 종합 권고
        """
        try:
            if not individual_signals:
                return self._get_empty_portfolio_signal()

            logger.info(f"포트폴리오 집계 시작: {len(individual_signals)}개 신호")

            # 입력 데이터 구조 확인
            for i, signal in enumerate(individual_signals):
                logger.info(f"신호 {i}: keys = {list(signal.keys())}")
                if "trading_signal" in signal:
                    trading_signal_keys = list(signal["trading_signal"].keys())
                    logger.info(f"  trading_signal keys = {trading_signal_keys}")

            # 신호별 분류
            signal_counts = {
                "STRONG_BUY": 0,
                "BUY": 0,
                "HOLD": 0,
                "SELL": 0,
                "STRONG_SELL": 0,
            }

            total_position_size = 0
            weighted_score = 0
            high_priority_signals = []

            for i, signal in enumerate(individual_signals):
                try:
                    # 신호 구조에서 trading_signal 부분 추출
                    trading_signal = signal.get("trading_signal", {})
                    action = trading_signal.get("action", "HOLD")

                    signal_counts[action] = signal_counts.get(action, 0) + 1

                    position_size = trading_signal.get("position_size", 0.05)
                    score = trading_signal.get("score", 0.0)
                    execution_priority = trading_signal.get("execution_priority", 5)

                    total_position_size += position_size
                    weighted_score += score * position_size

                    if execution_priority <= 3:  # 높은 우선순위
                        high_priority_signals.append(trading_signal)

                    logger.info(f"신호 {i} 처리 완료: {action}, score={score:.3f}")

                except Exception as e:
                    logger.error(f"신호 {i} 처리 실패: {e}")
                    logger.error(f"신호 구조: {signal}")
                    continue

            # 포트폴리오 종합 점수
            portfolio_score = (
                weighted_score / total_position_size if total_position_size > 0 else 0
            )

            # 포트폴리오 액션 결정
            portfolio_action = self._determine_portfolio_action(
                signal_counts, portfolio_score
            )

            # 리스크 수준 평가
            portfolio_risk = self._assess_portfolio_risk(
                individual_signals, market_regime
            )

            # 집중도 분석
            concentration_risk = self._analyze_concentration_risk(individual_signals)

            # 실행 계획 수립
            execution_plan = self._create_execution_plan(
                high_priority_signals, market_regime
            )

            # 포트폴리오 권고사항
            portfolio_recommendations = self._generate_portfolio_recommendations(
                signal_counts, portfolio_score, market_regime, portfolio_risk
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "market_regime": market_regime["regime"],
                "regime_confidence": market_regime["confidence"],
                "portfolio_score": portfolio_score,
                "portfolio_action": portfolio_action,
                "signal_distribution": signal_counts,
                "total_position_size": total_position_size,
                "high_priority_count": len(high_priority_signals),
                "risk_assessment": portfolio_risk,
                "concentration_risk": concentration_risk,
                "execution_plan": execution_plan,
                "recommendations": portfolio_recommendations,
                "top_opportunities": sorted(
                    [s.get("trading_signal", {}) for s in individual_signals],
                    key=lambda x: x.get("score", 0),
                    reverse=True,
                )[:5],
                "immediate_actions": [
                    s.get("trading_signal", {})
                    for s in individual_signals
                    if s.get("trading_signal", {}).get("execution_priority", 5) <= 2
                ],
            }

        except Exception as e:
            import traceback

            logger.error(f"포트폴리오 신호 집계 실패: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return self._get_empty_portfolio_signal()

    def _determine_portfolio_action(
        self, signal_counts: Dict, portfolio_score: float
    ) -> str:
        """포트폴리오 전체 액션 결정"""
        total_signals = sum(signal_counts.values())

        if total_signals == 0:
            return "HOLD"

        # 비율 계산
        buy_ratio = (signal_counts["STRONG_BUY"] + signal_counts["BUY"]) / total_signals
        sell_ratio = (
            signal_counts["STRONG_SELL"] + signal_counts["SELL"]
        ) / total_signals

        if buy_ratio > 0.6 and portfolio_score > 0.3:
            return "PORTFOLIO_BUY"
        elif sell_ratio > 0.6 and portfolio_score < -0.3:
            return "PORTFOLIO_SELL"
        elif buy_ratio > 0.4 and portfolio_score > 0.1:
            return "SELECTIVE_BUY"
        elif sell_ratio > 0.4 and portfolio_score < -0.1:
            return "SELECTIVE_SELL"
        else:
            return "PORTFOLIO_HOLD"

    def _assess_portfolio_risk(self, signals: List[Dict], market_regime: Dict) -> Dict:
        """포트폴리오 리스크 평가"""
        try:
            # 개별 종목 리스크 집계
            high_vol_count = sum(
                1
                for s in signals
                if s.get("risk_management", {}).get("risk_level") == "HIGH"
            )

            total_count = len(signals)
            high_risk_ratio = high_vol_count / total_count if total_count > 0 else 0

            # 시장 체제 리스크
            regime_risk_levels = {
                "BULLISH": "LOW",
                "SIDEWAYS": "MEDIUM",
                "BEARISH": "HIGH",
                "VOLATILE": "HIGH",
            }

            regime_risk = regime_risk_levels.get(market_regime["regime"], "MEDIUM")

            # 종합 리스크 레벨
            if high_risk_ratio > 0.5 or regime_risk == "HIGH":
                overall_risk = "HIGH"
            elif high_risk_ratio > 0.3 or regime_risk == "MEDIUM":
                overall_risk = "MEDIUM"
            else:
                overall_risk = "LOW"

            return {
                "overall_risk": overall_risk,
                "regime_risk": regime_risk,
                "high_risk_stock_ratio": high_risk_ratio,
                "risk_factors": [
                    f"고위험 종목 비율: {high_risk_ratio:.1%}",
                    f"시장 체제: {market_regime['regime']}",
                    f"체제 신뢰도: {market_regime['confidence']:.1%}",
                ],
            }

        except Exception as e:
            logger.error(f"포트폴리오 리스크 평가 실패: {e}")
            return {"overall_risk": "MEDIUM"}

    def _analyze_concentration_risk(self, signals: List[Dict]) -> Dict:
        """집중도 리스크 분석"""
        try:
            if not signals:
                return {"concentration_risk": "LOW"}

            # 포지션 크기 분석
            position_sizes = [s["position_size"] for s in signals]
            max_position = max(position_sizes) if position_sizes else 0
            total_position = sum(position_sizes)

            # 상위 3개 종목 집중도
            top_3_concentration = sum(sorted(position_sizes, reverse=True)[:3])

            if max_position > 0.15 or top_3_concentration > 0.4:
                risk_level = "HIGH"
                warnings = ["포지션 집중도 높음 - 분산 필요"]
            elif max_position > 0.1 or top_3_concentration > 0.3:
                risk_level = "MEDIUM"
                warnings = ["적정 수준의 집중도"]
            else:
                risk_level = "LOW"
                warnings = ["양호한 분산 수준"]

            return {
                "concentration_risk": risk_level,
                "max_position_size": max_position,
                "top_3_concentration": top_3_concentration,
                "total_position_size": total_position,
                "warnings": warnings,
            }

        except Exception as e:
            logger.error(f"집중도 리스크 분석 실패: {e}")
            return {"concentration_risk": "MEDIUM"}

    def _create_execution_plan(
        self, high_priority_signals: List[Dict], market_regime: Dict
    ) -> Dict:
        """실행 계획 수립"""
        try:
            if not high_priority_signals:
                return {"immediate_actions": [], "planned_actions": []}

            # 우선순위별 분류
            immediate = [
                s for s in high_priority_signals if s.get("execution_priority", 5) == 1
            ]
            urgent = [
                s for s in high_priority_signals if s.get("execution_priority", 5) == 2
            ]
            planned = [
                s for s in high_priority_signals if s.get("execution_priority", 5) == 3
            ]

            return {
                "immediate_actions": len(immediate),
                "urgent_actions": len(urgent),
                "planned_actions": len(planned),
                "execution_sequence": self._create_execution_sequence(
                    high_priority_signals
                ),
                "timing_considerations": {
                    "market_regime": market_regime["regime"],
                    "regime_confidence": market_regime["confidence"],
                    "recommended_execution_window": self._get_execution_window(
                        market_regime
                    ),
                },
            }

        except Exception as e:
            logger.error(f"실행 계획 수립 실패: {e}")
            return {"immediate_actions": 0}

    def _create_execution_sequence(self, signals: List[Dict]) -> List[Dict]:
        """실행 순서 결정"""
        # 우선순위와 점수를 종합하여 정렬
        sorted_signals = sorted(
            signals,
            key=lambda x: (x.get("execution_priority", 5), -abs(x.get("score", 0))),
        )

        return [
            {
                "symbol": s.get("symbol", "UNKNOWN"),
                "action": s.get("action", "HOLD"),
                "priority": s.get("execution_priority", 5),
            }
            for s in sorted_signals[:10]
        ]  # 상위 10개만

    def _get_execution_window(self, market_regime: Dict) -> str:
        """최적 실행 시간대"""
        regime = market_regime["regime"]
        confidence = market_regime["confidence"]

        if regime == "VOLATILE" and confidence > 0.7:
            return "시장 변동성 고려하여 분할 실행"
        elif regime == "BULLISH" and confidence > 0.7:
            return "장 초반 적극적 실행"
        elif regime == "BEARISH" and confidence > 0.7:
            return "신중한 분할 실행"
        else:
            return "일반적 실행 (장중 균등 분할)"

    def _generate_portfolio_recommendations(
        self,
        signal_counts: Dict,
        portfolio_score: float,
        market_regime: Dict,
        portfolio_risk: Dict,
    ) -> Dict:
        """포트폴리오 권고사항 생성"""
        try:
            recommendations = []

            # 전체 포지션 권고
            total_signals = sum(signal_counts.values())
            buy_signals = signal_counts["STRONG_BUY"] + signal_counts["BUY"]
            sell_signals = signal_counts["STRONG_SELL"] + signal_counts["SELL"]

            if buy_signals > sell_signals and portfolio_score > 0.2:
                recommendations.append(
                    f"매수 신호 우세 ({buy_signals}/{total_signals}) - 포지션 확대 고려"
                )
            elif sell_signals > buy_signals and portfolio_score < -0.2:
                recommendations.append(
                    f"매도 신호 우세 ({sell_signals}/{total_signals}) - 포지션 축소 고려"
                )
            else:
                recommendations.append("혼재된 신호 - 선별적 접근 필요")

            # 시장 체제별 권고
            regime = market_regime["regime"]
            if regime == "BULLISH":
                recommendations.append("상승 시장 - 적극적 매수 전략 고려")
            elif regime == "BEARISH":
                recommendations.append("하락 시장 - 방어적 포지션 유지")
            elif regime == "VOLATILE":
                recommendations.append("변동성 시장 - 리스크 관리 강화")
            else:
                recommendations.append("횡보 시장 - 단기 매매 전략 고려")

            # 리스크 관리 권고
            if portfolio_risk["overall_risk"] == "HIGH":
                recommendations.append("높은 리스크 - 포지션 크기 축소 및 손절선 강화")
            elif portfolio_risk["overall_risk"] == "MEDIUM":
                recommendations.append("중간 리스크 - 신중한 포지션 관리")

            return {
                "primary_recommendations": recommendations,
                "action_summary": f"매수 {buy_signals}개, 매도 {sell_signals}개, 보유 {signal_counts['HOLD']}개",
                "risk_guidance": f"포트폴리오 리스크: {portfolio_risk['overall_risk']}",
                "regime_guidance": f"시장 체제: {regime} (신뢰도: {market_regime['confidence']:.1%})",
            }

        except Exception as e:
            logger.error(f"포트폴리오 권고사항 생성 실패: {e}")
            return {"primary_recommendations": ["분석 오류 발생"]}

    def _get_empty_portfolio_signal(self) -> Dict:
        """빈 포트폴리오 신호"""
        return {
            "timestamp": datetime.now().isoformat(),
            "market_regime": "UNKNOWN",
            "portfolio_score": 0.0,
            "portfolio_action": "HOLD",
            "signal_distribution": {
                "STRONG_BUY": 0,
                "BUY": 0,
                "HOLD": 0,
                "SELL": 0,
                "STRONG_SELL": 0,
            },
            "recommendations": {"primary_recommendations": ["분석 대상 없음"]},
        }
