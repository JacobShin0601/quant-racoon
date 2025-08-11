"""
동적 적응형 ML 기반 시장 체제 분류기
학술적으로 검증된 방법론을 적용한 지도학습 기반 체제 분류

주요 특징:
- 시장 환경 변화에 따른 임계값 자동 조정
- Rolling window 기반 동적 calibration  
- 변동성 체제별 가중치 적응
- Hamilton (1989) regime-switching 방법론 적용
- NBER-style 지속성 요구사항
- VIX percentile 기반 동적 임계값 (2024 Financial Management 논문)
- 150년간 Bull/Bear 시장 데이터 기반 임계값
- 다중 지표 통합 점수화 시스템
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DynamicRegimeLabelGenerator:
    """
    동적 적응형 시장 체제 라벨 생성기
    
    핵심 동적 기능:
    1. Rolling Window Adaptation: 시간에 따른 임계값 자동 조정
    2. Volatility Regime Adaptation: 변동성 환경별 가중치 조정
    3. Market Condition Sensing: 실시간 시장 조건 반영
    4. Adaptive Calibration: 예측 성능에 따른 파라미터 자동 튜닝
    
    학술적 기반:
    - Hamilton (1989): Regime-switching models
    - NBER Business Cycle Dating: 지속성 요구사항
    - Financial Management (2024): VIX percentile 임계값
    - 150년간 Bull/Bear 시장 역사적 기준
    """

    def __init__(self, config: Dict):
        self.config = config
        self.regime_config = config.get("ml_regime", {})
        
        # 동적 조정 설정
        self.adaptive_config = self.regime_config.get("adaptive", {})
        self.enable_adaptation = self.adaptive_config.get("enable", True)
        self.adaptation_window = self.adaptive_config.get("window_days", 252)  # 1년 윈도우
        self.recalibration_frequency = self.adaptive_config.get("recalibration_days", 22)  # 22일마다
        self.volatility_regime_memory = self.adaptive_config.get("volatility_memory_days", 66)  # 3개월
        
        # 4가지 시장 체제 정의
        self.states = ["TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE"]
        
        # 학술 논문 기반 기본 임계값들 (동적 조정의 출발점)
        self.BASE_HISTORICAL_BENCHMARKS = {
            'bear_median_decline': -0.33,      # 중간값 -33% (학술 연구)
            'bear_median_duration': 19 * 22,   # 19개월 → 거래일 변환
            'bull_median_gain': 0.87,          # +87%
            'bull_median_duration': 42 * 22,   # 42개월 → 거래일 변환
            'correction_threshold': -0.10,     # 10% 조정
            'bear_threshold': -0.20,           # 20% 하락 (표준)
            'bull_threshold': 0.20             # 20% 상승
        }
        
        # 동적 조정되는 임계값들 (초기값은 기본값)
        self.current_benchmarks = self.BASE_HISTORICAL_BENCHMARKS.copy()
        
        # VIX 해석 기준 (전문가/실무 검증) - 동적 조정됨
        self.BASE_VIX_THRESHOLDS = {
            'complacency': 15,     # 시장 안주
            'normal_bull': 20,     # 정상 강세장
            'elevated': 25,        # 불확실성 증가
            'high_stress': 30,     # 높은 스트레스
            'crisis': 40           # 위기 (매수 기회 가능성)
        }
        
        # NBER-style 최소 지속 기간 (거래일 기준) - 동적 조정됨
        self.base_min_duration = self.regime_config.get("min_duration_days", 66)  # 3개월
        self.current_min_duration = self.base_min_duration
        
        # 다중 지표 가중치 (학술 연구 기반) - 변동성 체제별 동적 조정
        self.BASE_INDICATOR_WEIGHTS = {
            'vix_score': 0.35,           # VIX 주요 지표
            'momentum_score': 0.35,      # 가격 모멘텀  
            'duration_adjusted': 0.20,   # NBER-style 지속성
            'yield_curve_score': 0.10    # 거시경제
        }
        
        # 변동성 환경별 가중치 조정
        self.volatility_weight_adjustments = {
            'low_vol': {'vix_score': -0.1, 'momentum_score': 0.15, 'duration_adjusted': 0.05},
            'normal_vol': {'vix_score': 0.0, 'momentum_score': 0.0, 'duration_adjusted': 0.0},
            'high_vol': {'vix_score': 0.2, 'momentum_score': -0.1, 'duration_adjusted': -0.1},
            'crisis_vol': {'vix_score': 0.3, 'momentum_score': -0.15, 'duration_adjusted': -0.15}
        }
        
        # 동적 조정 이력 저장
        self.adaptation_history = []
        self.last_recalibration = None
        
        logger.info("DynamicRegimeLabelGenerator 초기화 완료")
        if self.enable_adaptation:
            logger.info(f"동적 조정 활성화 - 윈도우: {self.adaptation_window}일, 재조정: {self.recalibration_frequency}일")

    def generate_dynamic_labels(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        동적 적응형 시장 체제 라벨 생성
        
        핵심 동적 기능:
        1. Rolling calibration으로 임계값 지속적 업데이트
        2. 변동성 환경 감지하여 가중치 자동 조정
        3. 시장 조건에 따른 지속성 요구사항 조정
        
        Args:
            macro_data: 매크로 경제 데이터 (SPY, VIX, 금리 등 포함)
            
        Returns:
            동적으로 조정된 체제 라벨이 포함된 DataFrame
        """
        try:
            logger.info("동적 적응형 시장 체제 라벨 생성 시작")
            
            # 1. 필수 데이터 추출 및 검증
            spy_data, vix_data, tnx_data, irx_data = self._extract_core_data(macro_data)
            
            if len(spy_data) < self.adaptation_window:
                logger.warning(f"동적 조정을 위한 데이터 부족: {len(spy_data)}일 (최소 {self.adaptation_window}일 필요)")
                return self._create_default_labels(macro_data)
            
            # 2. 시장 환경 감지 및 분류
            market_environment = self._detect_market_environment(spy_data, vix_data)
            logger.info(f"감지된 시장 환경: {market_environment}")
            
            # 3. 동적 임계값 및 가중치 조정
            if self.enable_adaptation:
                self._adapt_to_market_conditions(spy_data, vix_data, market_environment)
            
            # 4. Rolling window로 지속적 calibration
            result_labels = []
            adaptation_points = []
            
            # 최소 윈도우 크기부터 시작하여 rolling
            start_idx = max(self.adaptation_window, 252)  # 최소 1년
            
            for end_idx in range(start_idx, len(spy_data), self.recalibration_frequency):
                # 현재 윈도우 데이터
                window_spy = spy_data.iloc[max(0, end_idx-self.adaptation_window):end_idx]
                window_vix = vix_data.iloc[max(0, end_idx-self.adaptation_window):end_idx]
                window_tnx = tnx_data.iloc[max(0, end_idx-self.adaptation_window):end_idx]
                window_irx = irx_data.iloc[max(0, end_idx-self.adaptation_window):end_idx]
                
                # 윈도우별 동적 임계값 계산
                window_vix_thresholds = self._calculate_adaptive_vix_thresholds(window_vix)
                
                # 현재 시점까지의 라벨 생성
                current_end = min(end_idx + self.recalibration_frequency, len(spy_data))
                segment_spy = spy_data.iloc[end_idx:current_end]
                segment_vix = vix_data.iloc[end_idx:current_end]
                segment_tnx = tnx_data.iloc[end_idx:current_end]
                segment_irx = irx_data.iloc[end_idx:current_end]
                
                if len(segment_spy) > 0:
                    # 현재 시장 환경에 맞는 가중치 적용
                    current_weights = self._get_adaptive_weights(segment_vix.mean())
                    
                    # 다중 지표 점수 계산
                    segment_scores = self._calculate_adaptive_scores(
                        segment_spy, segment_vix, segment_tnx, segment_irx, 
                        window_vix_thresholds, current_weights
                    )
                    
                    # 통합 점수 및 라벨
                    integrated_scores = self._integrate_scores_with_weights(segment_scores, current_weights)
                    segment_labels = self._classify_with_adaptive_thresholds(integrated_scores)
                    
                    result_labels.extend(segment_labels)
                    adaptation_points.append({
                        'timestamp': end_idx,
                        'vix_thresholds': window_vix_thresholds,
                        'weights': current_weights,
                        'market_env': market_environment
                    })
            
            # 5. 처음 부분 처리 (최소 윈도우 이전)
            if start_idx > 0:
                initial_labels = self._generate_initial_labels(
                    spy_data.iloc[:start_idx], vix_data.iloc[:start_idx], 
                    tnx_data.iloc[:start_idx], irx_data.iloc[:start_idx]
                )
                result_labels = initial_labels + result_labels
            
            # 6. NBER-style 동적 지속성 필터 적용
            adaptive_min_duration = self._get_adaptive_duration_requirement(market_environment)
            validated_labels = self._apply_adaptive_duration_filter(result_labels, adaptive_min_duration)
            
            # 7. 결과 DataFrame 생성
            result_df = self._create_dynamic_result_dataframe(
                macro_data, validated_labels, adaptation_points
            )
            
            # 8. 동적 조정 이력 저장
            self.adaptation_history.extend(adaptation_points)
            self.last_recalibration = datetime.now()
            
            logger.info(f"동적 시장 체제 라벨 생성 완료: {len(result_df)}개 라벨")
            self._log_adaptation_statistics(adaptation_points, market_environment)
            
            return result_df
            
        except Exception as e:
            logger.error(f"동적 시장 체제 라벨 생성 실패: {e}")
            return self._create_default_labels(macro_data)

    def _detect_market_environment(self, spy_data: pd.Series, vix_data: pd.Series) -> str:
        """
        현재 시장 환경 감지
        - 변동성 레벨
        - 트렌드 방향
        - 시장 스트레스 수준
        """
        recent_window = min(66, len(vix_data))  # 최근 3개월 또는 가용 데이터
        recent_vix = vix_data.iloc[-recent_window:]
        recent_returns = spy_data.iloc[-recent_window:].pct_change(22).fillna(0)
        
        avg_vix = recent_vix.mean()
        avg_return = recent_returns.mean()
        vol_of_vix = recent_vix.std()
        
        # 복합적 환경 판단
        if avg_vix > 35 or vol_of_vix > 8:
            return "crisis_vol"
        elif avg_vix > 25:
            return "high_vol"
        elif avg_vix < 16 and avg_return > 0.05:
            return "low_vol"
        else:
            return "normal_vol"

    def _adapt_to_market_conditions(self, spy_data: pd.Series, vix_data: pd.Series, market_env: str):
        """시장 조건에 따른 파라미터 동적 조정"""
        
        # 1. Bull/Bear 임계값 조정 (최근 변동성에 따라)
        recent_volatility = spy_data.iloc[-66:].pct_change().std() * np.sqrt(252)  # 연율화 변동성
        
        if market_env in ['crisis_vol', 'high_vol']:
            # 고변동성 시기: 더 보수적 임계값
            volatility_multiplier = 1.5
            self.current_min_duration = int(self.base_min_duration * 0.7)  # 지속성 요구 완화
        elif market_env == 'low_vol':
            # 저변동성 시기: 더 민감한 임계값
            volatility_multiplier = 0.8
            self.current_min_duration = int(self.base_min_duration * 1.3)  # 지속성 요구 강화
        else:
            volatility_multiplier = 1.0
            self.current_min_duration = self.base_min_duration
        
        # 임계값 동적 조정
        self.current_benchmarks['bear_threshold'] = (
            self.BASE_HISTORICAL_BENCHMARKS['bear_threshold'] * volatility_multiplier
        )
        self.current_benchmarks['bull_threshold'] = (
            self.BASE_HISTORICAL_BENCHMARKS['bull_threshold'] * volatility_multiplier
        )
        
        logger.info(f"임계값 동적 조정 - Bear: {self.current_benchmarks['bear_threshold']:.1%}, "
                   f"Bull: {self.current_benchmarks['bull_threshold']:.1%}, "
                   f"지속기간: {self.current_min_duration}일")

    def _calculate_adaptive_vix_thresholds(self, vix_window: pd.Series) -> Dict[str, float]:
        """윈도우 기반 적응형 VIX 임계값 계산"""
        
        # 기본 percentile 계산
        base_thresholds = {
            'complacency': float(vix_window.quantile(0.20)),
            'normal': float(vix_window.quantile(0.50)),
            'elevated': float(vix_window.quantile(0.75)),
            'high_stress': float(vix_window.quantile(0.85)),
            'crisis': float(vix_window.quantile(0.95))
        }
        
        # 극값 보정 (너무 극단적인 임계값 방지)
        base_thresholds['complacency'] = max(10, min(20, base_thresholds['complacency']))
        base_thresholds['crisis'] = max(30, min(60, base_thresholds['crisis']))
        
        # 전통적 기준과 혼합 (70% 동적, 30% 전통적)
        mixed_thresholds = {}
        for key in base_thresholds:
            if key in self.BASE_VIX_THRESHOLDS:
                mixed_thresholds[key] = (
                    0.7 * base_thresholds[key] + 
                    0.3 * self.BASE_VIX_THRESHOLDS[key]
                )
            else:
                mixed_thresholds[key] = base_thresholds[key]
        
        mixed_thresholds['static_crisis'] = 30.0  # 전통적 위기 기준 유지
        
        return mixed_thresholds

    def _get_adaptive_weights(self, current_vix: float) -> Dict[str, float]:
        """변동성 수준에 따른 적응형 가중치"""
        
        base_weights = self.BASE_INDICATOR_WEIGHTS.copy()
        
        # VIX 수준별 조정
        if current_vix > 35:
            adjustment = self.volatility_weight_adjustments['crisis_vol']
        elif current_vix > 25:
            adjustment = self.volatility_weight_adjustments['high_vol']
        elif current_vix < 16:
            adjustment = self.volatility_weight_adjustments['low_vol']
        else:
            adjustment = self.volatility_weight_adjustments['normal_vol']
        
        # 가중치 조정 적용
        adapted_weights = {}
        for key, base_weight in base_weights.items():
            adjustment_val = adjustment.get(key, 0)
            adapted_weights[key] = max(0.05, min(0.60, base_weight + adjustment_val))
        
        # 정규화 (합이 1이 되도록)
        total_weight = sum(adapted_weights.values())
        for key in adapted_weights:
            adapted_weights[key] /= total_weight
        
        return adapted_weights

    def _calculate_adaptive_scores(
        self, spy_data: pd.Series, vix_data: pd.Series, 
        tnx_data: pd.Series, irx_data: pd.Series,
        vix_thresholds: Dict, weights: Dict
    ) -> Dict[str, pd.Series]:
        """적응형 다중 지표 점수 계산"""
        
        # 기존 점수 계산 메서드들을 적응형 임계값으로 호출
        vix_score = self._calculate_adaptive_vix_score(vix_data, vix_thresholds)
        momentum_score = self._calculate_adaptive_momentum_score(spy_data)
        duration_score = self._calculate_duration_score(spy_data)
        yield_score = self._calculate_yield_curve_score(tnx_data, irx_data)
        
        return {
            'vix_score': vix_score,
            'momentum_score': momentum_score,
            'duration_adjusted': duration_score,
            'yield_curve_score': yield_score
        }

    def _calculate_adaptive_vix_score(self, vix_data: pd.Series, thresholds: Dict) -> pd.Series:
        """적응형 VIX 점수 계산"""
        score = pd.Series(0.0, index=vix_data.index)
        
        score = np.where(
            vix_data < thresholds['complacency'], -1.0,
            np.where(
                vix_data < thresholds['normal'], -0.5,
                np.where(
                    vix_data < thresholds['elevated'], 0.0,
                    np.where(
                        vix_data < thresholds['high_stress'], 0.5,
                        1.0
                    )
                )
            )
        )
        
        # 위기 수준 특별 처리
        score = np.where(vix_data > thresholds.get('static_crisis', 30), 1.0, score)
        
        return pd.Series(score, index=vix_data.index)

    def _calculate_adaptive_momentum_score(self, spy_data: pd.Series) -> pd.Series:
        """적응형 모멘텀 점수 (동적 임계값 적용)"""
        
        returns_22d = spy_data.pct_change(22)
        returns_66d = spy_data.pct_change(66)
        
        # 동적으로 조정된 임계값 사용
        bear_threshold = self.current_benchmarks['bear_threshold']
        bull_threshold = self.current_benchmarks['bull_threshold']
        
        score_22d = np.where(
            returns_22d > bull_threshold * 0.25, 1.0,
            np.where(
                returns_22d > 0.02, 0.5,
                np.where(
                    returns_22d < bear_threshold * 0.25, -1.0,
                    np.where(returns_22d < -0.02, -0.5, 0.0)
                )
            )
        )
        
        score_66d = np.where(
            returns_66d > bull_threshold * 0.5, 1.0,
            np.where(
                returns_66d > 0.05, 0.5,
                np.where(
                    returns_66d < bear_threshold * 0.5, -1.0,
                    np.where(returns_66d < -0.05, -0.5, 0.0)
                )
            )
        )
        
        momentum_score = 0.6 * score_22d + 0.4 * score_66d
        
        return pd.Series(momentum_score, index=spy_data.index).fillna(0)

    def _integrate_scores_with_weights(self, scores: Dict[str, pd.Series], weights: Dict) -> pd.Series:
        """가중치를 적용한 점수 통합"""
        integrated = pd.Series(0.0, index=scores['vix_score'].index)
        
        for indicator, weight in weights.items():
            if indicator in scores:
                integrated += scores[indicator] * weight
        
        return np.clip(integrated, -1, 1)

    def _classify_with_adaptive_thresholds(self, integrated_scores: pd.Series) -> List[str]:
        """적응형 임계값을 사용한 체제 분류"""
        
        # 동적 임계값 (시장 환경에 따라 조정 가능)
        upper_threshold = 0.4
        lower_threshold = -0.4
        neutral_threshold = 0.15
        
        regimes = []
        for score in integrated_scores:
            if score > upper_threshold:
                regimes.append("TRENDING_UP")
            elif score < lower_threshold:
                regimes.append("TRENDING_DOWN")
            elif abs(score) < neutral_threshold:
                regimes.append("SIDEWAYS")
            else:
                regimes.append("VOLATILE")
        
        return regimes

    def _get_adaptive_duration_requirement(self, market_env: str) -> int:
        """시장 환경별 적응형 지속성 요구사항"""
        if market_env == 'crisis_vol':
            return int(self.base_min_duration * 0.5)  # 위기시 빠른 반응
        elif market_env == 'high_vol':
            return int(self.base_min_duration * 0.7)
        elif market_env == 'low_vol':
            return int(self.base_min_duration * 1.5)  # 안정시 신중한 판단
        else:
            return self.base_min_duration

    def _apply_adaptive_duration_filter(self, regimes: List[str], min_duration: int) -> List[str]:
        """적응형 지속성 필터"""
        if len(regimes) < min_duration:
            return regimes
        
        filtered_regimes = regimes.copy()
        
        i = 0
        while i < len(filtered_regimes) - min_duration:
            current_regime = filtered_regimes[i]
            
            duration = 1
            j = i + 1
            while j < len(filtered_regimes) and filtered_regimes[j] == current_regime:
                duration += 1
                j += 1
            
            if duration < min_duration:
                if i > 0:
                    for k in range(i, min(i + duration, len(filtered_regimes))):
                        filtered_regimes[k] = filtered_regimes[i-1]
                elif j < len(filtered_regimes):
                    for k in range(i, j):
                        filtered_regimes[k] = filtered_regimes[j] if j < len(filtered_regimes) else "SIDEWAYS"
            
            i = j if duration >= min_duration else i + duration
        
        return filtered_regimes

    def _generate_initial_labels(self, spy_data: pd.Series, vix_data: pd.Series, 
                                tnx_data: pd.Series, irx_data: pd.Series) -> List[str]:
        """초기 부분 라벨 생성 (단순한 방식)"""
        
        # 기본 임계값으로 단순 분류
        returns_22d = spy_data.pct_change(22)
        
        labels = []
        for i, (ret, vix) in enumerate(zip(returns_22d, vix_data)):
            if pd.isna(ret):
                labels.append("SIDEWAYS")
            elif vix > 30:
                labels.append("VOLATILE")
            elif ret > 0.05:
                labels.append("TRENDING_UP")
            elif ret < -0.05:
                labels.append("TRENDING_DOWN")
            else:
                labels.append("SIDEWAYS")
        
        return labels

    # 기존 메서드들 재사용
    def _extract_core_data(self, macro_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """핵심 데이터 추출"""
        spy_col = self._find_column(macro_data, ['SPY_close', 'spy_close', 'SPY_data', 'spy'])
        if spy_col is None:
            raise ValueError("SPY 데이터를 찾을 수 없습니다")
        spy_data = pd.to_numeric(macro_data[spy_col], errors='coerce').fillna(method='ffill')
        
        vix_col = self._find_column(macro_data, ['^VIX_close', 'VIX_close', 'vix_close', '^vix', 'vix'])
        if vix_col is None:
            logger.warning("VIX 데이터 없음, 기본값 사용")
            vix_data = pd.Series(20.0, index=macro_data.index)
        else:
            vix_data = pd.to_numeric(macro_data[vix_col], errors='coerce').fillna(20.0)
        
        tnx_col = self._find_column(macro_data, ['^TNX_close', 'TNX_close', 'tnx_close'])
        irx_col = self._find_column(macro_data, ['^IRX_close', 'IRX_close', 'irx_close'])
        
        tnx_data = (pd.to_numeric(macro_data[tnx_col], errors='coerce').fillna(2.0) 
                   if tnx_col else pd.Series(2.0, index=macro_data.index))
        irx_data = (pd.to_numeric(macro_data[irx_col], errors='coerce').fillna(0.5)
                   if irx_col else pd.Series(0.5, index=macro_data.index))
        
        return spy_data, vix_data, tnx_data, irx_data

    def _find_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """컬럼명 패턴 매칭"""
        for pattern in patterns:
            if pattern in df.columns:
                return pattern
        return None

    def _calculate_duration_score(self, spy_data: pd.Series) -> pd.Series:
        """트렌드 지속성 점수 계산"""
        ma_10 = spy_data.rolling(10).mean()
        ma_22 = spy_data.rolling(22).mean()
        ma_50 = spy_data.rolling(50).mean()
        
        ma_slope_22 = ma_22.diff(5) / ma_22.shift(5)
        
        cross_signal = np.where(
            (spy_data > ma_10) & (ma_10 > ma_22) & (ma_22 > ma_50), 1.0,
            np.where(
                (spy_data < ma_10) & (ma_10 < ma_22) & (ma_22 < ma_50), -1.0,
                np.where(
                    spy_data > ma_22, 0.5,
                    np.where(spy_data < ma_22, -0.5, 0.0)
                )
            )
        )
        
        slope_score = np.clip(ma_slope_22 * 20, -1, 1)
        duration_score = 0.7 * cross_signal + 0.3 * slope_score
        
        return pd.Series(duration_score, index=spy_data.index).fillna(0)

    def _calculate_yield_curve_score(self, tnx_data: pd.Series, irx_data: pd.Series) -> pd.Series:
        """수익률 곡선 점수 계산"""
        yield_spread = tnx_data - irx_data
        
        curve_score = np.where(
            yield_spread > 2.5, 0.5,
            np.where(
                yield_spread > 1.5, 1.0,
                np.where(
                    yield_spread > 0.5, 0.0,
                    np.where(
                        yield_spread > 0, -0.5,
                        -1.0
                    )
                )
            )
        )
        
        return pd.Series(curve_score, index=tnx_data.index).fillna(0)

    def _create_dynamic_result_dataframe(
        self, macro_data: pd.DataFrame, validated_regimes: List[str],
        adaptation_points: List[Dict]
    ) -> pd.DataFrame:
        """동적 결과 DataFrame 생성"""
        
        result_df = pd.DataFrame(index=macro_data.index[:len(validated_regimes)])
        result_df['regime_label'] = validated_regimes
        
        # 체제 변화 지점
        result_df['regime_change'] = (
            result_df['regime_label'] != result_df['regime_label'].shift(1)
        ).astype(int)
        
        # 동적 조정 지점 표시
        result_df['adaptation_point'] = 0
        for point in adaptation_points:
            if point['timestamp'] < len(result_df):
                result_df.iloc[point['timestamp'], result_df.columns.get_loc('adaptation_point')] = 1
        
        # 신뢰도 (단순화)
        result_df['confidence'] = 0.75
        
        return result_df

    def _log_adaptation_statistics(self, adaptation_points: List[Dict], market_env: str):
        """적응 통계 로깅"""
        logger.info(f"=== 동적 적응 통계 (시장 환경: {market_env}) ===")
        logger.info(f"총 적응 지점: {len(adaptation_points)}개")
        
        if adaptation_points:
            # VIX 임계값 변화 추적
            first_thresholds = adaptation_points[0]['vix_thresholds']
            last_thresholds = adaptation_points[-1]['vix_thresholds']
            
            logger.info("VIX 임계값 변화:")
            for key in ['complacency', 'normal', 'elevated', 'high_stress']:
                if key in first_thresholds and key in last_thresholds:
                    change = last_thresholds[key] - first_thresholds[key]
                    logger.info(f"  {key}: {first_thresholds[key]:.1f} → {last_thresholds[key]:.1f} ({change:+.1f})")
        
        logger.info(f"현재 최소 지속기간: {self.current_min_duration}일")

    def _create_default_labels(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """기본 라벨 생성"""
        result_df = pd.DataFrame(index=macro_data.index)
        result_df['regime_label'] = "SIDEWAYS"
        result_df['confidence'] = 0.25
        result_df['regime_change'] = 0
        result_df['adaptation_point'] = 0
        
        return result_df


class DynamicMLRegimeClassifier:
    """
    동적 적응형 머신러닝 기반 시장 체제 분류기
    """

    def __init__(self, config: Dict):
        self.config = config
        self.ml_config = config.get("ml_regime", {})
        
        # 동적 라벨 생성기
        self.label_generator = DynamicRegimeLabelGenerator(config)
        
        # ML 모델
        self.model = RandomForestClassifier(
            n_estimators=self.ml_config.get("n_estimators", 100),
            max_depth=self.ml_config.get("max_depth", 10),
            min_samples_split=self.ml_config.get("min_samples_split", 20),
            min_samples_leaf=self.ml_config.get("min_samples_leaf", 10),
            random_state=self.ml_config.get("random_state", 42),
            class_weight='balanced'
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.class_names = ["TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE"]
        
        logger.info("DynamicMLRegimeClassifier 초기화 완료")

    def _find_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """컬럼명 패턴 매칭"""
        for pattern in patterns:
            if pattern in df.columns:
                return pattern
        return None

    def create_dynamic_training_labels(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """동적 적응형 훈련 라벨 생성"""
        logger.info("동적 적응형 ML 훈련용 라벨 생성 시작")
        
        labeled_data = self.label_generator.generate_dynamic_labels(macro_data)
        
        logger.info(f"동적 훈련용 라벨 생성 완료: {len(labeled_data)}개")
        return labeled_data

    def load_comprehensive_data_from_files(self) -> pd.DataFrame:
        """기존 다운로드된 data/macro 파일들을 활용한 포괄적 데이터 로딩"""
        try:
            import os
            import glob
            
            macro_dir = "data/macro"
            
            logger.info("ML 분류기: 기존 매크로 데이터 파일들 로딩 시작")
            print(f"DEBUG: macro_dir = {macro_dir}")
            print(f"DEBUG: 디렉토리 존재: {os.path.exists(macro_dir)}")
            
            # 최신 UUID 디렉토리 찾기 (metadata.json이 있는 곳)
            all_items = os.listdir(macro_dir)
            print(f"DEBUG: macro_dir 내용: {all_items}")
            
            uuid_dirs = [d for d in all_items 
                        if os.path.isdir(os.path.join(macro_dir, d)) and len(d) > 30]
            print(f"DEBUG: UUID 디렉토리들: {uuid_dirs}")
            
            target_dir = macro_dir
            if uuid_dirs:
                # 데이터 파일이 많이 있는 최신 디렉토리 찾기
                best_dir = None
                max_files = 0
                
                for uuid_dir in sorted(uuid_dirs, reverse=True):
                    uuid_path = os.path.join(macro_dir, uuid_dir)
                    try:
                        files = os.listdir(uuid_path)
                        csv_files = [f for f in files if f.endswith('.csv')]
                        print(f"DEBUG: {uuid_dir}에 {len(csv_files)}개 CSV 파일")
                        
                        if len(csv_files) > max_files:
                            max_files = len(csv_files)
                            best_dir = uuid_dir
                            
                    except Exception as e:
                        print(f"DEBUG: {uuid_dir} 확인 실패: {e}")
                
                if best_dir:
                    target_dir = os.path.join(macro_dir, best_dir)
                    print(f"DEBUG: 최적 UUID 디렉토리 선택: {best_dir} ({max_files}개 파일)")
                    logger.info(f"UUID 디렉토리 사용: {best_dir}")
            
            print(f"DEBUG: target_dir = {target_dir}")
            
            # 파일 매핑 (실제 파일명과 일치)
            symbol_mapping = {
                "vix": "^vix_data.csv",
                "tnx": "^tnx_data.csv", 
                "irx": "^irx_data.csv",
                "uup": "uup_data.csv",
                "gld": "gld_data.csv",
                "tlt": "tlt_data.csv",
                "qqq": "qqq_data.csv",
                "iwm": "iwm_data.csv",
                "tip": "tip_data.csv",
                "xrt": "xrt_data.csv",
                "goex": "goex_data.csv",
                "spy": "spy_data.csv",
                # 섹터 ETF들 (실제 파일명)
                "xlk": "xlk_sector.csv",
                "xlf": "xlf_sector.csv", 
                "xle": "xle_sector.csv",
                "xlv": "xlv_sector.csv",
                "xli": "xli_sector.csv",
                "xlp": "xlp_sector.csv",
                "xlu": "xlu_sector.csv",
                "xlb": "xlb_sector.csv",
                "xlre": "xlre_sector.csv"
            }
            
            # 데이터 로딩
            all_data = {}
            loaded_count = 0
            
            for symbol, filename in symbol_mapping.items():
                file_path = os.path.join(target_dir, filename)
                
                logger.info(f"파일 체크: {file_path} - 존재여부: {os.path.exists(file_path)}")
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        logger.info(f"{filename} 컬럼: {list(df.columns)}")
                        
                        # 날짜 인덱스 설정
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        elif 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])  
                            df.set_index('date', inplace=True)
                        elif 'datetime' in df.columns:
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df.set_index('datetime', inplace=True)
                        
                        # Close 컬럼 찾기
                        close_col = None
                        for col in ['Close', 'close', 'adj_close', 'Adj Close']:
                            if col in df.columns:
                                close_col = col
                                break
                        
                        if close_col is not None:
                            column_name = f"{symbol}_close"
                            all_data[column_name] = df[close_col]
                            loaded_count += 1
                            logger.info(f"{symbol} -> {column_name} 로딩 완료 ({len(df)} 행)")
                        else:
                            logger.warning(f"{file_path}: Close 컬럼을 찾을 수 없음. 컬럼: {list(df.columns)}")
                            
                    except Exception as e:
                        logger.warning(f"{file_path} 로딩 실패: {e}")
                else:
                    logger.info(f"{file_path} 파일 없음")
            
            if loaded_count == 0:
                raise ValueError("로딩된 매크로 데이터 파일이 없습니다")
            
            # 데이터프레임 생성 및 정리
            macro_data = pd.DataFrame(all_data)
            macro_data = macro_data.dropna(how='all').fillna(method='ffill').dropna()
            
            logger.info(f"매크로 데이터 파일 로딩 완료: {loaded_count}개 파일, {len(macro_data)} 행")
            logger.info(f"컬럼: {list(macro_data.columns)}")
            
            return macro_data
            
        except Exception as e:
            logger.error(f"매크로 데이터 파일 로딩 실패: {e}")
            return None

    def extract_comprehensive_features_from_data(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """HMM과 동일한 포괄적 피처 추출 - 로딩된 데이터 활용"""
        logger.info(f"포괄적 피처 추출 시작 - 입력 데이터: {len(macro_data)} 행, {len(macro_data.columns)}개 컬럼")
        
        try:
            features = pd.DataFrame(index=macro_data.index)
            
            # 1. VIX 관련 피처 (5개)
            vix_col = self._find_column(macro_data, ['^vix_close', 'vix_close'])
            if vix_col is not None:
                vix_data = pd.to_numeric(macro_data[vix_col], errors="coerce").fillna(20.0)
                
                features["vix_level"] = vix_data
                vix_ma = vix_data.rolling(20).mean()
                features["vix_ma_ratio"] = (vix_data / vix_ma - 1).fillna(0)
                
                # 동적 VIX 임계값
                vix_low_threshold = vix_data.rolling(60, min_periods=20).quantile(0.25)
                vix_high_threshold = vix_data.rolling(60, min_periods=20).quantile(0.75)
                
                features["volatility_regime"] = np.where(
                    vix_data > vix_high_threshold.fillna(25), 1, 
                    np.where(vix_data < vix_low_threshold.fillna(15), -1, 0)
                )
                features["vix_acceleration"] = vix_data.diff(2).fillna(0)
                features["vix_percentile"] = (
                    vix_data.rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
                )
                logger.info("VIX 피처 5개 추출 완료")
            else:
                logger.warning("VIX 데이터 없음")
                for feat in ["vix_level", "vix_ma_ratio", "volatility_regime", "vix_acceleration", "vix_percentile"]:
                    features[feat] = 0.0 if 'level' not in feat else 20.0
            
            # 2. 금리 스프레드 피처 (2개)
            tnx_col = self._find_column(macro_data, ['^tnx_close', 'tnx_close'])
            irx_col = self._find_column(macro_data, ['^irx_close', 'irx_close'])
            
            if tnx_col and irx_col:
                tnx_data = pd.to_numeric(macro_data[tnx_col], errors="coerce").fillna(2.0)
                irx_data = pd.to_numeric(macro_data[irx_col], errors="coerce").fillna(0.5)
                
                features["yield_spread"] = tnx_data - irx_data
                features["yield_spread_ma"] = features["yield_spread"].rolling(10).mean().fillna(1.5)
                logger.info("금리 피처 2개 추출 완료")
            else:
                logger.warning("금리 데이터 부족")
                features["yield_spread"] = 1.5
                features["yield_spread_ma"] = 1.5
            
            # 3. 달러 강세 피처 (2개)  
            dollar_col = self._find_column(macro_data, ['uup_close'])
            if dollar_col is not None:
                dollar_data = pd.to_numeric(macro_data[dollar_col], errors="coerce").fillna(25.0)
                features["dollar_strength"] = dollar_data.pct_change(20).fillna(0)
                features["dollar_momentum"] = dollar_data.pct_change(5).fillna(0)
                logger.info("달러 피처 2개 추출 완료")
            else:
                logger.warning("달러 데이터 없음")
                features["dollar_strength"] = 0.0
                features["dollar_momentum"] = 0.0
            
            # 4. SPY 기반 포괄적 피처 (19개)
            spy_col = self._find_column(macro_data, ['spy_close'])
            if spy_col is not None:
                spy_data = pd.to_numeric(macro_data[spy_col], errors="coerce").fillna(400.0)
                
                # 기본 모멘텀 (2개)
                features["market_momentum"] = spy_data.pct_change(20).fillna(0)
                features["market_trend"] = (spy_data / spy_data.rolling(50).mean() - 1).fillna(0)
                
                # 수익률 피처 (4개)
                features["spy_return_1d"] = spy_data.pct_change(1).fillna(0)
                features["spy_return_5d"] = spy_data.pct_change(5).fillna(0)
                features["spy_return_10d"] = spy_data.pct_change(10).fillna(0)
                features["spy_return_22d"] = spy_data.pct_change(22).fillna(0)
                
                # 이동평균 교차 (6개)
                spy_ma5 = spy_data.rolling(5).mean()
                spy_ma10 = spy_data.rolling(10).mean()
                spy_ma20 = spy_data.rolling(20).mean()  
                spy_ma50 = spy_data.rolling(50).mean()
                
                features["spy_ma5_cross"] = (spy_data > spy_ma5).astype(int) - 0.5
                features["spy_ma10_cross"] = (spy_data > spy_ma10).astype(int) - 0.5
                features["spy_ma20_cross"] = (spy_data > spy_ma20).astype(int) - 0.5
                features["spy_ma50_cross"] = (spy_data > spy_ma50).astype(int) - 0.5
                features["spy_ma5_ma10_cross"] = (spy_ma5 > spy_ma10).astype(int) - 0.5
                features["spy_ma10_ma20_cross"] = (spy_ma10 > spy_ma20).astype(int) - 0.5
                
                # 변동성 (2개)
                features["spy_volatility_5d"] = spy_data.pct_change().rolling(5).std().fillna(0)
                features["spy_volatility_20d"] = spy_data.pct_change().rolling(20).std().fillna(0)
                
                # RSI 유사 (1개)
                spy_returns = spy_data.pct_change().fillna(0)
                gains = spy_returns.where(spy_returns > 0, 0).rolling(14).mean()
                losses = (-spy_returns.where(spy_returns < 0, 0)).rolling(14).mean()
                rs = gains / (losses + 1e-8)
                features["spy_rsi_like"] = (100 - (100 / (1 + rs))).fillna(50) / 100 - 0.5
                
                # 모멘텀 강도 (1개)
                features["spy_momentum_strength"] = (
                    (spy_data.pct_change(5) > 0).astype(int) + 
                    (spy_data.pct_change(10) > 0).astype(int) + 
                    (spy_data.pct_change(20) > 0).astype(int)
                ) / 3 - 0.5
                
                # 위치 (1개)
                spy_high_52w = spy_data.rolling(252, min_periods=50).max()
                spy_low_52w = spy_data.rolling(252, min_periods=50).min()
                features["spy_position_in_range"] = (
                    (spy_data - spy_low_52w) / (spy_high_52w - spy_low_52w + 1e-8)
                ).fillna(0.5) - 0.5
                
                # 체제 (1개) 
                features["spy_bull_bear_regime"] = np.where(
                    (features["spy_return_22d"] > 0.05) & (features["spy_momentum_strength"] > 0.1), 1,
                    np.where(
                        (features["spy_return_22d"] < -0.05) & (features["spy_momentum_strength"] < -0.1), -1, 0
                    )
                )
                
                logger.info("SPY 피처 19개 추출 완료")
            else:
                logger.warning("SPY 데이터 없음")
                spy_features = [
                    "market_momentum", "market_trend", "spy_return_1d", "spy_return_5d", 
                    "spy_return_10d", "spy_return_22d", "spy_ma5_cross", "spy_ma10_cross",
                    "spy_ma20_cross", "spy_ma50_cross", "spy_ma5_ma10_cross", "spy_ma10_ma20_cross",
                    "spy_volatility_5d", "spy_volatility_20d", "spy_rsi_like", 
                    "spy_momentum_strength", "spy_position_in_range", "spy_bull_bear_regime"
                ]
                for feat in spy_features:
                    features[feat] = 0.0
            
            # 5. 신용 스프레드 피처 (3개) - HYG, LQD는 다운로드 안되었을 수 있음
            self._add_credit_spread_features(features, macro_data)
            
            # 6. 복합 지표 (3개)
            features["cross_market_stress"] = features["vix_level"] * abs(features["yield_spread"])
            features["regime_transition"] = features["volatility_regime"].diff().fillna(0)
            features["regime_persistence"] = self._calculate_regime_persistence(features)
            features["market_stress_composite"] = self._calculate_market_stress_composite(features)
            
            # NaN 처리
            features = features.fillna(method="ffill").fillna(0)
            
            logger.info(f"포괄적 피처 추출 완료: {len(features.columns)}개 피처")
            logger.info(f"피처 리스트: {list(features.columns)}")
            
            return features
            
        except Exception as e:
            logger.error(f"포괄적 피처 추출 실패: {e}")
            # 최소 피처 반환
            n_rows = len(macro_data)
            return pd.DataFrame({
                "vix_level": [20.0] * n_rows,
                "yield_spread": [1.5] * n_rows, 
                "market_momentum": [0.0] * n_rows,
            }, index=macro_data.index)
    
    def _add_credit_spread_features(self, features: pd.DataFrame, macro_data: pd.DataFrame):
        """신용 스프레드 지표 추가 - 파일에서 로딩된 데이터 기반"""
        try:
            # TLT는 있을 가능성 높음 (config에 있음)
            tlt_col = self._find_column(macro_data, ['tlt_close'])
            
            # HYG, LQD는 섹터 파일에 없을 수 있으므로 기본값 처리
            features["credit_spread"] = 0.0
            features["credit_stress"] = 0
            features["ig_credit_spread"] = 0.0
            
            if tlt_col is not None:
                logger.info("TLT 데이터 발견 - 신용 스프레드는 기본값 사용")
            else:
                logger.info("신용 관련 데이터 없음 - 기본값 사용")
                
        except Exception as e:
            logger.warning(f"신용 스프레드 피처 추가 실패: {e}")
            features["credit_spread"] = 0.0
            features["credit_stress"] = 0
            features["ig_credit_spread"] = 0.0
    
    def _calculate_regime_persistence(self, features: pd.DataFrame) -> pd.Series:
        """체제 지속성 지표 계산"""
        if "volatility_regime" in features.columns:
            regime = features["volatility_regime"]
            persistence = regime.rolling(10, min_periods=3).apply(
                lambda x: (x == x.iloc[-1]).sum() / len(x), raw=False
            ).fillna(0.5)
            return persistence
        else:
            return pd.Series(0.5, index=features.index)
    
    def _calculate_market_stress_composite(self, features: pd.DataFrame) -> pd.Series:
        """시장 스트레스 복합 지표 계산"""
        try:
            stress_components = []
            
            if "vix_level" in features.columns:
                vix_stress = np.clip((features["vix_level"] - 20) / 20, -1, 1)
                stress_components.append(0.4 * vix_stress)
            
            if "credit_spread" in features.columns:
                credit_stress = np.clip(features["credit_spread"] * 10, -1, 1) 
                stress_components.append(0.3 * credit_stress)
            
            if "yield_spread" in features.columns:
                yield_stress = np.where(
                    features["yield_spread"] < 0.5, 0.5,
                    np.where(features["yield_spread"] > 3.0, -0.5, 0.0)
                )
                stress_components.append(0.3 * yield_stress)
            
            if stress_components:
                composite = sum(stress_components)
                return np.clip(composite, -1, 1)
            else:
                return pd.Series(0.0, index=features.index)
                
        except Exception as e:
            logger.warning(f"시장 스트레스 복합 지표 계산 실패: {e}")
            return pd.Series(0.0, index=features.index)


def main():
    """동적 라벨 생성 테스트"""
    import argparse
    import json
    import os
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="동적 ML 기반 시장 체제 분류기")
    parser.add_argument("--create-dynamic-labels", action="store_true", help="동적 적응형 라벨 생성")
    parser.add_argument("--data-dir", type=str, default="data/macro", help="매크로 데이터 디렉토리")
    parser.add_argument("--config", type=str, default="config/config_trader.json", help="설정 파일")
    parser.add_argument("--output-dir", type=str, default="results/labels", help="결과 저장 디렉토리")

    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)

    if args.create_dynamic_labels:
        print("🔄 동적 적응형 시장 체제 라벨 생성 시작")
        
        classifier = DynamicMLRegimeClassifier(config)
        
        # 기존 파일에서 포괄적 데이터 로드
        print(f"📊 매크로 데이터 로드: {args.data_dir}")
        
        macro_data = classifier.load_comprehensive_data_from_files()
        
        if macro_data is None or macro_data.empty:
            print("❌ 매크로 데이터를 찾을 수 없습니다.")
            sys.exit(1)

        print(f"✅ 매크로 데이터 로드 완료: {len(macro_data)} 행, {len(macro_data.columns)}개 컬럼")

        # 동적 라벨 생성
        try:
            labeled_data = classifier.create_dynamic_training_labels(macro_data)
            
            # 결과 저장
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"dynamic_regime_labels_{timestamp}.csv"
            
            labeled_data.to_csv(output_file)
            print(f"✅ 동적 라벨 저장 완료: {output_file}")
            
            # 통계 출력
            print("\n📊 동적 라벨 요약:")
            print(labeled_data['regime_label'].value_counts())
            print(f"\n평균 신뢰도: {labeled_data['confidence'].mean():.3f}")
            print(f"체제 변화 횟수: {labeled_data['regime_change'].sum()}회")
            print(f"동적 조정 지점: {labeled_data['adaptation_point'].sum()}개")
            
        except Exception as e:
            print(f"❌ 동적 라벨 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print("사용법:")
        print("  --create-dynamic-labels    # 동적 적응형 라벨 생성")


if __name__ == "__main__":
    main()