"""
HMM 기반 시장 체제 분류기
Hidden Markov Models를 사용하여 시장을 4가지 체제로 분류
- BULLISH: 상승 추세
- BEARISH: 하락 추세
- SIDEWAYS: 횡보
- VOLATILE: 고변동성
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class MarketRegimeHMM:
    """
    Hidden Markov Model을 사용한 시장 체제 분류기

    Features:
    - VIX 수준 (변동성 지표)
    - 수익률 곡선 기울기 (TNX - IRX)
    - 달러 강세 지수
    - 섹터 회전율
    - 모멘텀 지표들
    """

    def __init__(self, config: Dict):
        self.config = config
        self.hmm_config = config.get("hmm_regime", {})

        # 시장 체제 정의
        self.states = ["BULLISH", "BEARISH", "SIDEWAYS", "VOLATILE"]
        self.n_states = len(self.states)

        # HMM 모델 초기화
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.hmm_config.get("covariance_type", "diag"),
            n_iter=self.hmm_config.get("n_iter", 100),
            random_state=42,
        )

        # 스케일러 초기화
        self.scaler = StandardScaler()

        # 모델 학습 상태
        self.is_fitted = False
        self.feature_names = []

        logger.info(f"MarketRegimeHMM 초기화 완료 - States: {self.states}")

    def extract_macro_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        매크로 데이터에서 HMM 피처 추출 (개선된 버전)
        
        새로운 기능:
        - 동적 VIX 임계값
        - 신용 스프레드 지표
        - 개선된 체제 감지

        Args:
            macro_data: 매크로 경제 데이터

        Returns:
            피처 데이터프레임
        """
        features = pd.DataFrame(index=macro_data.index)

        try:
            # 매크로 데이터 컬럼명 확인 및 정규화 (간소화)
            unique_symbols = set()
            for col in macro_data.columns:
                symbol = col.split("_")[0]
                unique_symbols.add(symbol)
            logger.info(
                f"매크로 데이터: {len(macro_data.columns)}개 컬럼, 심볼: {sorted(unique_symbols)}"
            )

            # 1. VIX 수준 (변동성) - 다양한 컬럼명 시도
            vix_col = None
            for col in ["vix", "vix_close", "^vix", "vix_data"]:
                if col in macro_data.columns:
                    vix_col = col
                    break

            if vix_col is not None:
                # 숫자형으로 변환
                vix_data = pd.to_numeric(macro_data[vix_col], errors="coerce").fillna(
                    20.0
                )
                features["vix_level"] = vix_data
                features["vix_ma_ratio"] = vix_data / vix_data.rolling(
                    20
                ).mean().fillna(1.0)
            else:
                logger.warning("VIX 데이터 없음")
                features["vix_level"] = 20.0  # 기본값
                features["vix_ma_ratio"] = 1.0

            # 2. 수익률 곡선 기울기 - 다양한 컬럼명 시도
            tnx_col = None
            irx_col = None
            for col in ["tnx", "tnx_close", "^tnx", "tnx_data"]:
                if col in macro_data.columns:
                    tnx_col = col
                    break
            for col in ["irx", "irx_close", "^irx", "irx_data"]:
                if col in macro_data.columns:
                    irx_col = col
                    break

            if tnx_col is not None and irx_col is not None:
                # 숫자형으로 변환
                tnx_data = pd.to_numeric(macro_data[tnx_col], errors="coerce").fillna(
                    2.0
                )
                irx_data = pd.to_numeric(macro_data[irx_col], errors="coerce").fillna(
                    0.5
                )
                features["yield_spread"] = tnx_data - irx_data
                features["yield_spread_ma"] = (
                    features["yield_spread"].rolling(10).mean().fillna(1.5)
                )
            else:
                logger.warning("금리 데이터 없음")
                features["yield_spread"] = 1.5  # 기본값
                features["yield_spread_ma"] = 1.5

            # 3. 달러 강세 (UUP 또는 DXY 대용) - 다양한 컬럼명 시도
            dollar_col = None
            dollar_patterns = ["uup_close", "uup", "dxy_close", "dxy", "uup_data"]
            for pattern in dollar_patterns:
                if pattern in macro_data.columns:
                    dollar_col = pattern
                    break

            if dollar_col is not None:
                # 숫자형으로 변환
                dollar_data = pd.to_numeric(
                    macro_data[dollar_col], errors="coerce"
                ).fillna(25.0)
                features["dollar_strength"] = dollar_data.pct_change(20).fillna(0)
                features["dollar_momentum"] = dollar_data.pct_change(5).fillna(0)
                logger.info(f"달러 강세 데이터 사용: {dollar_col}")
            else:
                logger.warning("달러 강세 데이터 없음")
                features["dollar_strength"] = 0.0
                features["dollar_momentum"] = 0.0

            # 4. 개선된 변동성 체제 (동적 VIX 임계값)
            if vix_col is not None:
                vix_data = pd.to_numeric(macro_data[vix_col], errors="coerce").fillna(
                    20.0
                )
                # 동적 임계값 계산 (60일 롤링 백분위수)
                vix_low_threshold = vix_data.rolling(60, min_periods=20).quantile(0.25)
                vix_high_threshold = vix_data.rolling(60, min_periods=20).quantile(0.75)
                
                # 동적 체제 분류
                features["volatility_regime"] = np.where(
                    vix_data > vix_high_threshold.fillna(25), 1, 
                    np.where(vix_data < vix_low_threshold.fillna(15), -1, 0)
                )
                features["vix_acceleration"] = vix_data.diff(2).fillna(0)
                features["vix_percentile"] = (
                    vix_data.rolling(252, min_periods=60)
                    .rank(pct=True).fillna(0.5)
                )
                logger.info("동적 VIX 임계값 적용 완료")
            else:
                features["volatility_regime"] = 0
                features["vix_acceleration"] = 0
                features["vix_percentile"] = 0.5

            # 5. 모멘텀 지표 (SPY 기반) - 개선된 검색
            spy_col = None
            spy_patterns = ["spy_close", "spy", "spy_data"]
            for pattern in spy_patterns:
                if pattern in macro_data.columns:
                    spy_col = pattern
                    break

            if spy_col is not None:
                # 숫자형으로 변환
                spy_data = pd.to_numeric(macro_data[spy_col], errors="coerce").fillna(
                    400.0
                )
                features["market_momentum"] = spy_data.pct_change(20).fillna(0)
                features["market_trend"] = (
                    spy_data / spy_data.rolling(50).mean() - 1
                ).fillna(0)
                logger.info(f"SPY 데이터 사용: {spy_col}")
            else:
                logger.warning("SPY 데이터 없음")
                features["market_momentum"] = 0.0
                features["market_trend"] = 0.0

            # 6. 신용 스프레드 지표 추가
            self._add_credit_spread_features(features, macro_data)
            
            # 7. 추가 피처들
            features["cross_market_stress"] = features["vix_level"] * abs(
                features["yield_spread"]
            )
            features["regime_transition"] = (
                features["volatility_regime"].diff().fillna(0)
            )
            
            # 8. 체제 지속성 지표
            features["regime_persistence"] = self._calculate_regime_persistence(features)
            
            # 9. 시장 스트레스 복합 지표
            features["market_stress_composite"] = self._calculate_market_stress_composite(features)

            # NaN 처리
            features = features.fillna(method="ffill").fillna(0)

            self.feature_names = list(features.columns)
            logger.info(f"추출된 피처 ({len(self.feature_names)}개): {self.feature_names}")

            return features

        except Exception as e:
            logger.error(f"매크로 피처 추출 실패: {e}")
            # 기본 피처 반환
            n_rows = len(macro_data)
            default_features = pd.DataFrame(
                {
                    "vix_level": [20.0] * n_rows,
                    "yield_spread": [1.5] * n_rows,
                    "dollar_strength": [0.0] * n_rows,
                    "volatility_regime": [0] * n_rows,
                    "market_momentum": [0.0] * n_rows,
                },
                index=macro_data.index,
            )

            self.feature_names = list(default_features.columns)
            return default_features

    def _add_credit_spread_features(self, features: pd.DataFrame, macro_data: pd.DataFrame):
        """신용 스프레드 지표 추가"""
        try:
            # HYG (고수익 회사채 ETF) 검색
            hyg_col = None
            for col in ["hyg_close", "hyg", "hyg_data"]:
                if col in macro_data.columns:
                    hyg_col = col
                    break
            
            # LQD (투자등급 회사채 ETF) 검색
            lqd_col = None
            for col in ["lqd_close", "lqd", "lqd_data"]:
                if col in macro_data.columns:
                    lqd_col = col
                    break
                    
            # TLT (장기 국채 ETF) 검색
            tlt_col = None
            for col in ["tlt_close", "tlt", "tlt_data"]:
                if col in macro_data.columns:
                    tlt_col = col
                    break
            
            if hyg_col and tlt_col:
                hyg_data = pd.to_numeric(macro_data[hyg_col], errors="coerce").fillna(100.0)
                tlt_data = pd.to_numeric(macro_data[tlt_col], errors="coerce").fillna(120.0)
                
                # HYG-TLT 스프레드 (신용 위험 지표)
                features["credit_spread"] = (hyg_data / tlt_data).pct_change(20).fillna(0)
                features["credit_stress"] = np.where(features["credit_spread"] < -0.05, 1, 0)
                logger.info("HYG-TLT 신용 스프레드 지표 추가")
            else:
                features["credit_spread"] = 0.0
                features["credit_stress"] = 0
                
            if lqd_col and tlt_col:
                lqd_data = pd.to_numeric(macro_data[lqd_col], errors="coerce").fillna(110.0)
                tlt_data = pd.to_numeric(macro_data[tlt_col], errors="coerce").fillna(120.0)
                
                # LQD-TLT 스프레드 (투자등급 신용 스프레드)
                features["ig_credit_spread"] = (lqd_data / tlt_data).pct_change(20).fillna(0)
                logger.info("LQD-TLT 투자등급 스프레드 지표 추가")
            else:
                features["ig_credit_spread"] = 0.0
                
        except Exception as e:
            logger.warning(f"신용 스프레드 지표 추가 실패: {e}")
            features["credit_spread"] = 0.0
            features["credit_stress"] = 0
            features["ig_credit_spread"] = 0.0
    
    def _calculate_regime_persistence(self, features: pd.DataFrame) -> pd.Series:
        """체제 지속성 지표 계산"""
        try:
            if "volatility_regime" not in features.columns:
                return pd.Series(0.5, index=features.index)
                
            vol_regime = features["volatility_regime"]
            
            # 동일 체제 지속 기간 계산
            regime_changes = vol_regime != vol_regime.shift(1)
            regime_groups = regime_changes.cumsum()
            
            persistence = []
            for i, group in enumerate(regime_groups):
                if i == 0:
                    persistence.append(1)
                else:
                    # 현재 체제가 지속된 기간
                    same_regime_count = (regime_groups[:i+1] == group).sum()
                    # 최대 20일로 정규화
                    normalized_persistence = min(same_regime_count / 20.0, 1.0)
                    persistence.append(normalized_persistence)
            
            return pd.Series(persistence, index=features.index)
            
        except Exception as e:
            logger.warning(f"체제 지속성 계산 실패: {e}")
            return pd.Series(0.5, index=features.index)
    
    def _calculate_market_stress_composite(self, features: pd.DataFrame) -> pd.Series:
        """시장 스트레스 복합 지표 계산"""
        try:
            components = []
            weights = []
            
            # VIX 컴포넌트
            if "vix_level" in features.columns:
                vix_stress = np.clip(features["vix_level"] / 40.0, 0, 1)
                components.append(vix_stress)
                weights.append(0.3)
            
            # 신용 스프레드 컴포넌트
            if "credit_spread" in features.columns:
                credit_stress = np.clip(-features["credit_spread"] * 5, 0, 1)
                components.append(credit_stress)
                weights.append(0.25)
            
            # 수익률 곡선 컴포넌트
            if "yield_spread" in features.columns:
                # 역전된 수익률 곡선은 스트레스 신호
                yield_stress = np.clip(-features["yield_spread"] + 2, 0, 1)
                components.append(yield_stress)
                weights.append(0.2)
            
            # 달러 강세 컴포넌트
            if "dollar_strength" in features.columns:
                dollar_stress = np.clip(abs(features["dollar_strength"]) * 2, 0, 1)
                components.append(dollar_stress)
                weights.append(0.15)
            
            # 모멘텀 컴포넌트  
            if "market_momentum" in features.columns:
                momentum_stress = np.clip(-features["market_momentum"] * 3, 0, 1)
                components.append(momentum_stress)
                weights.append(0.1)
            
            if components:
                # 가중 평균 계산
                weights = np.array(weights) / sum(weights)  # 정규화
                composite = sum(w * comp for w, comp in zip(weights, components))
                return composite.fillna(0.5)
            else:
                return pd.Series(0.5, index=features.index)
                
        except Exception as e:
            logger.warning(f"시장 스트레스 복합 지표 계산 실패: {e}")
            return pd.Series(0.5, index=features.index)

    def _walk_forward_validation(self, features: pd.DataFrame, n_splits: int = 5) -> float:
        """
        워크포워드 검증 수행
        
        Args:
            features: 피처 데이터
            n_splits: 검증 분할 수
            
        Returns:
            평균 검증 점수
        """
        try:
            if len(features) < 100:
                logger.warning("워크포워드 검증을 위한 데이터 부족")
                return 0.5
                
            scores = []
            min_train_size = max(50, len(features) // (n_splits + 1))
            
            for i in range(n_splits):
                # 분할 지점 계산
                train_end = min_train_size + i * (len(features) - min_train_size) // n_splits
                test_start = train_end
                test_end = min(test_start + 20, len(features))  # 20일 테스트 윈도우
                
                if test_end <= test_start:
                    continue
                    
                # 훈련/테스트 데이터 분할
                train_features = features.iloc[:train_end]
                test_features = features.iloc[test_start:test_end]
                
                # 임시 모델 생성 및 학습
                temp_model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.model.covariance_type,
                    n_iter=100,
                    random_state=42,
                )
                
                temp_scaler = StandardScaler()
                train_scaled = temp_scaler.fit_transform(train_features)
                
                try:
                    temp_model.fit(train_scaled)
                    
                    # 테스트 데이터 예측
                    test_scaled = temp_scaler.transform(test_features)
                    predicted_states = temp_model.predict(test_scaled)
                    
                    # 예측 일관성 점수 (연속된 예측의 안정성)
                    if len(predicted_states) > 1:
                        stability_score = 1 - (np.diff(predicted_states) != 0).mean()
                        scores.append(stability_score)
                        
                except Exception as e:
                    logger.warning(f"워크포워드 검증 {i+1}번째 분할 실패: {e}")
                    continue
            
            if scores:
                avg_score = np.mean(scores)
                logger.info(f"워크포워드 검증 완료: {len(scores)}개 분할, 평균 점수: {avg_score:.3f}")
                return avg_score
            else:
                logger.warning("워크포워드 검증 실패")
                return 0.5
                
        except Exception as e:
            logger.error(f"워크포워드 검증 오류: {e}")
            return 0.5

    def fit(self, macro_data: pd.DataFrame) -> bool:
        """
        HMM 모델 학습

        Args:
            macro_data: 학습용 매크로 데이터

        Returns:
            학습 성공 여부
        """
        try:
            logger.info("HMM 모델 학습 시작...")

            # 피처 추출
            features = self.extract_macro_features(macro_data)

            if len(features) < 200:  # 최소 데이터 요구량 (개선)
                logger.warning(f"학습 데이터 부족: {len(features)}개 (최소 200개 필요)")
                return False

            # 데이터 스케일링
            scaled_features = self.scaler.fit_transform(features)

            # HMM 모델 학습
            self.model.fit(scaled_features)

            # 학습된 상태 해석
            self._interpret_states(scaled_features, features)

            # 워크포워드 검증 수행
            validation_score = self._walk_forward_validation(features)
            logger.info(f"워크포워드 검증 점수: {validation_score:.3f}")

            self.is_fitted = True
            logger.info("HMM 모델 학습 완료")

            return True

        except Exception as e:
            logger.error(f"HMM 모델 학습 실패: {e}")
            return False

    def _interpret_states(
        self, scaled_features: np.ndarray, original_features: pd.DataFrame
    ):
        """
        학습된 HMM 상태들을 해석하여 BULLISH, BEARISH 등으로 매핑
        """
        try:
            # 상태 예측
            states = self.model.predict(scaled_features)

            # 각 상태별 특성 분석
            state_characteristics = {}

            for state_idx in range(self.n_states):
                mask = states == state_idx
                if mask.sum() == 0:
                    continue

                state_data = original_features[mask]

                characteristics = {
                    "vix_mean": (
                        state_data["vix_level"].mean()
                        if "vix_level" in state_data.columns
                        else 20
                    ),
                    "volatility_regime": (
                        state_data["volatility_regime"].mean()
                        if "volatility_regime" in state_data.columns
                        else 0
                    ),
                    "market_momentum": (
                        state_data["market_momentum"].mean()
                        if "market_momentum" in state_data.columns
                        else 0
                    ),
                    "yield_spread": (
                        state_data["yield_spread"].mean()
                        if "yield_spread" in state_data.columns
                        else 1.5
                    ),
                    "frequency": mask.sum(),
                }

                state_characteristics[state_idx] = characteristics

            # 상태 매핑 규칙
            state_mapping = {}

            for state_idx, chars in state_characteristics.items():
                vix = chars["vix_mean"]
                momentum = chars["market_momentum"]
                vol_regime = chars["volatility_regime"]

                # 개선된 상태 분류 로직
                # 1차: 변동성 기준
                if vix > 28 or vol_regime > 0.6:
                    regime = "VOLATILE"
                # 2차: 모멘텀 기준 (더 보수적 임계값)
                elif momentum > 0.015:
                    regime = "BULLISH"
                elif momentum < -0.015:
                    regime = "BEARISH"
                # 3차: 복합 지표 고려
                else:
                    # 신용 스프레드나 기타 스트레스 지표 확인
                    if hasattr(chars, 'credit_stress') and chars.get('credit_stress', 0) > 0.5:
                        regime = "VOLATILE"
                    else:
                        regime = "SIDEWAYS"

                state_mapping[state_idx] = regime

            self.state_mapping = state_mapping
            logger.info(f"상태 매핑: {state_mapping}")

        except Exception as e:
            logger.error(f"상태 해석 실패: {e}")
            # 기본 매핑
            self.state_mapping = {i: self.states[i] for i in range(self.n_states)}

    def predict_regime(self, macro_data: pd.DataFrame) -> Dict:
        """
        현재 시장 체제 예측

        Args:
            macro_data: 예측용 매크로 데이터 (최근 데이터)

        Returns:
            예측 결과 딕셔너리
        """
        if not self.is_fitted:
            logger.error("모델이 학습되지 않았습니다")
            return self._get_default_prediction()

        try:
            # 피처 추출
            features = self.extract_macro_features(macro_data)

            if len(features) == 0:
                return self._get_default_prediction()

            # 최근 데이터만 사용 (마지막 1개)
            recent_features = features.iloc[-1:].values
            scaled_features = self.scaler.transform(recent_features)

            # 상태 예측
            predicted_state_idx = self.model.predict(scaled_features)[0]

            # 상태 확률 계산
            log_probs = self.model.score_samples(scaled_features)[0]
            state_probs = self.model.predict_proba(scaled_features)[0]

            # 매핑된 체제명
            predicted_regime = self.state_mapping.get(predicted_state_idx, "SIDEWAYS")

            # 신뢰도 조정 (과도한 확신 방지)
            raw_confidence = float(state_probs[predicted_state_idx])
            # 신뢰도를 0.3~0.9 범위로 제한하고 불확실성 추가
            confidence = min(0.9, max(0.3, raw_confidence * 0.8 + 0.1))

            # 체제 강도 계산
            regime_strength = self._calculate_regime_strength(features.iloc[-1])

            # 추가 분석
            current_features = features.iloc[-1]

            result = {
                "regime": predicted_regime,
                "confidence": confidence,
                "regime_strength": regime_strength,
                "state_probabilities": {
                    self.state_mapping.get(i, f"State_{i}"): float(prob)
                    for i, prob in enumerate(state_probs)
                },
                "features": {
                    "vix_level": float(current_features.get("vix_level", 20)),
                    "yield_spread": float(current_features.get("yield_spread", 1.5)),
                    "market_momentum": float(
                        current_features.get("market_momentum", 0)
                    ),
                    "volatility_regime": float(
                        current_features.get("volatility_regime", 0)
                    ),
                },
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "n_states": self.n_states,
                    "state_mapping": self.state_mapping,
                    "is_fitted": self.is_fitted,
                },
            }

            logger.info(
                f"예측된 시장 체제: {predicted_regime} (신뢰도: {confidence:.3f})"
            )

            return result

        except Exception as e:
            logger.error(f"시장 체제 예측 실패: {e}")
            return self._get_default_prediction()

    def _calculate_regime_strength(self, features: pd.Series) -> float:
        """
        체제 강도 계산 (0~1 스케일)
        """
        try:
            vix = features.get("vix_level", 20)
            momentum = abs(features.get("market_momentum", 0))
            yield_spread = abs(features.get("yield_spread", 1.5))

            # 정규화된 강도 계산
            vix_strength = min(1.0, vix / 30)  # VIX 30을 최대로
            momentum_strength = min(1.0, momentum * 10)  # 10% 모멘텀을 최대로
            yield_strength = min(1.0, yield_spread / 3)  # 3% 스프레드를 최대로

            # 가중 평균
            strength = (
                vix_strength * 0.4 + momentum_strength * 0.4 + yield_strength * 0.2
            )

            return float(np.clip(strength, 0, 1))

        except Exception as e:
            logger.error(f"체제 강도 계산 실패: {e}")
            return 0.5

    def _get_default_prediction(self) -> Dict:
        """
        기본 예측 결과 (모델 실패시)
        """
        return {
            "regime": "SIDEWAYS",
            "confidence": 0.25,  # 낮은 신뢰도
            "regime_strength": 0.5,
            "state_probabilities": {regime: 0.25 for regime in self.states},
            "features": {
                "vix_level": 20.0,
                "yield_spread": 1.5,
                "market_momentum": 0.0,
                "volatility_regime": 0,
            },
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "n_states": self.n_states,
                "state_mapping": {i: regime for i, regime in enumerate(self.states)},
                "is_fitted": False,
            },
        }

    def save_model(self, filepath: str) -> bool:
        """
        모델 저장
        """
        try:
            model_data = {
                "hmm_model": self.model,
                "scaler": self.scaler,
                "state_mapping": self.state_mapping,
                "feature_names": self.feature_names,
                "config": self.config,
                "is_fitted": self.is_fitted,
            }

            joblib.dump(model_data, filepath)
            logger.info(f"HMM 모델 저장 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        모델 로드
        """
        try:
            model_data = joblib.load(filepath)

            self.model = model_data["hmm_model"]
            self.scaler = model_data["scaler"]
            self.state_mapping = model_data["state_mapping"]
            self.feature_names = model_data["feature_names"]
            self.is_fitted = model_data["is_fitted"]

            logger.info(f"HMM 모델 로드 완료: {filepath}")
            return True

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False


class RegimeTransitionAnalyzer:
    """
    시장 체제 전환 분석기
    체제 변화의 지속성과 전환 확률을 분석
    """

    def __init__(self, config: Dict):
        self.config = config
        self.transition_history = []

    def analyze_transition_probability(self, regime_history: List[Dict]) -> Dict:
        """
        체제 전환 확률 분석
        """
        try:
            if len(regime_history) < 10:
                return {"transition_probability": 0.5, "stability": "unknown"}

            regimes = [r["regime"] for r in regime_history]
            transitions = 0

            for i in range(1, len(regimes)):
                if regimes[i] != regimes[i - 1]:
                    transitions += 1

            transition_rate = transitions / (len(regimes) - 1)

            # 안정성 평가
            if transition_rate < 0.1:
                stability = "very_stable"
            elif transition_rate < 0.2:
                stability = "stable"
            elif transition_rate < 0.4:
                stability = "moderate"
            else:
                stability = "volatile"

            return {
                "transition_probability": transition_rate,
                "stability": stability,
                "recent_transitions": transitions,
                "analysis_period": len(regimes),
            }

        except Exception as e:
            logger.error(f"체제 전환 분석 실패: {e}")
            return {"transition_probability": 0.5, "stability": "unknown"}


def main():
    """명령행 인터페이스"""
    import argparse
    import json
    import os
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="HMM 시장 체제 분류기")
    parser.add_argument("--train", action="store_true", help="모델 학습")
    parser.add_argument("--force", action="store_true", help="강제 재학습")
    parser.add_argument(
        "--data-dir", type=str, default="data/macro", help="매크로 데이터 디렉토리"
    )
    parser.add_argument(
        "--config", type=str, default="config/config_trader.json", help="설정 파일"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/trader", help="모델 저장 디렉토리"
    )
    parser.add_argument("--predict", action="store_true", help="현재 시장 체제 예측")

    args = parser.parse_args()

    # 설정 파일 로드
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)

    # HMM 모델 초기화
    hmm = MarketRegimeHMM(config)

    # 모델 디렉토리 생성
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hmm_regime_model.pkl"

    if args.train:
        print("🎭 HMM 시장 체제 분류 모델 학습 시작")

        # 기존 모델 확인
        if not args.force and model_path.exists():
            try:
                if hmm.load_model(str(model_path)):
                    print("✅ 기존 모델 로드 완료")
                    return
            except Exception as e:
                print(f"⚠️  기존 모델 로드 실패: {e}")

        # 매크로 데이터 로드 (개선된 버전)
        print(f"📊 매크로 데이터 로드: {args.data_dir}")
        macro_files = {
            "vix": f"{args.data_dir}/^vix_data.csv",
            "tnx": f"{args.data_dir}/^tnx_data.csv",
            "irx": f"{args.data_dir}/^irx_data.csv",
            "uup": f"{args.data_dir}/uup_data.csv",
            "spy": f"{args.data_dir}/spy_data.csv",
            # 신용 스프레드 데이터 추가
            "hyg": f"{args.data_dir}/hyg_data.csv",
            "lqd": f"{args.data_dir}/lqd_data.csv", 
            "tlt": f"{args.data_dir}/tlt_data.csv",
        }

        macro_data = pd.DataFrame()
        for key, filepath in macro_files.items():
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                if not df.empty and "close" in df.columns:
                    macro_data[key] = df["close"]
                elif not df.empty and "Close" in df.columns:
                    macro_data[key] = df["Close"]

        if macro_data.empty:
            print("❌ 매크로 데이터를 찾을 수 없습니다.")
            print(f"  확인 경로: {args.data_dir}")
            sys.exit(1)

        print(f"✅ 매크로 데이터 로드 완료: {len(macro_data)}행")

        # 모델 학습
        if hmm.fit(macro_data):
            # 모델 저장
            if hmm.save_model(str(model_path)):
                print(f"✅ 모델 저장 완료: {model_path}")
            else:
                print("❌ 모델 저장 실패")
                sys.exit(1)
        else:
            print("❌ 모델 학습 실패")
            sys.exit(1)

    elif args.predict:
        print("🔮 현재 시장 체제 예측")

        # 모델 로드
        if not model_path.exists():
            print("❌ 학습된 모델이 없습니다. 먼저 --train 옵션으로 학습하세요.")
            sys.exit(1)

        if not hmm.load_model(str(model_path)):
            print("❌ 모델 로드 실패")
            sys.exit(1)

        # 최근 매크로 데이터로 예측 (개선된 버전)
        macro_files = {
            "vix": f"{args.data_dir}/^vix_data.csv",
            "tnx": f"{args.data_dir}/^tnx_data.csv",
            "irx": f"{args.data_dir}/^irx_data.csv",
            "uup": f"{args.data_dir}/uup_data.csv",
            "spy": f"{args.data_dir}/spy_data.csv",
            # 신용 스프레드 데이터 추가
            "hyg": f"{args.data_dir}/hyg_data.csv",
            "lqd": f"{args.data_dir}/lqd_data.csv",
            "tlt": f"{args.data_dir}/tlt_data.csv",
        }

        macro_data = pd.DataFrame()
        for key, filepath in macro_files.items():
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                if not df.empty and "close" in df.columns:
                    macro_data[key] = df["close"]
                elif not df.empty and "Close" in df.columns:
                    macro_data[key] = df["Close"]

        if macro_data.empty:
            print("❌ 매크로 데이터를 찾을 수 없습니다.")
            sys.exit(1)

        # 예측 실행
        result = hmm.predict_regime(macro_data)
        print(f"📊 예측 결과: {result['regime']} (신뢰도: {result['confidence']:.3f})")
        print(f"📈 체제 강도: {result['regime_strength']:.3f}")

        # 상태별 확률
        print("📊 상태별 확률:")
        for state, prob in result["state_probabilities"].items():
            print(f"  - {state}: {prob:.3f}")

    else:
        print("사용법:")
        print("  --train --data-dir data/macro    # 모델 학습")
        print("  --predict --data-dir data/macro  # 현재 체제 예측")
        print("  --force                          # 강제 재학습")


if __name__ == "__main__":
    main()
