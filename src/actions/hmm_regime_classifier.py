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
        매크로 데이터에서 HMM 피처 추출

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

            # 4. 변동성 체제 (VIX 기반) - 개선된 검색
            if vix_col is not None:
                vix_data = pd.to_numeric(macro_data[vix_col], errors="coerce").fillna(
                    20.0
                )
                features["volatility_regime"] = np.where(
                    vix_data > 25, 1, np.where(vix_data < 15, -1, 0)
                )
                features["vix_acceleration"] = vix_data.diff(2).fillna(0)
            else:
                features["volatility_regime"] = 0
                features["vix_acceleration"] = 0

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

            # 6. 추가 피처들
            features["cross_market_stress"] = features["vix_level"] * abs(
                features["yield_spread"]
            )
            features["regime_transition"] = (
                features["volatility_regime"].diff().fillna(0)
            )

            # NaN 처리
            features = features.fillna(method="ffill").fillna(0)

            self.feature_names = list(features.columns)
            logger.info(f"추출된 피처: {self.feature_names}")

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

            if len(features) < 50:  # 최소 데이터 요구량
                logger.warning(f"학습 데이터 부족: {len(features)}개 (최소 50개 필요)")
                return False

            # 데이터 스케일링
            scaled_features = self.scaler.fit_transform(features)

            # HMM 모델 학습
            self.model.fit(scaled_features)

            # 학습된 상태 해석
            self._interpret_states(scaled_features, features)

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

                if vix > 30 or vol_regime > 0.5:
                    regime = "VOLATILE"
                elif momentum > 0.02:
                    regime = "BULLISH"
                elif momentum < -0.02:
                    regime = "BEARISH"
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

        # 매크로 데이터 로드
        print(f"📊 매크로 데이터 로드: {args.data_dir}")
        macro_files = {
            "vix": f"{args.data_dir}/^vix_data.csv",
            "tnx": f"{args.data_dir}/^tnx_data.csv",
            "irx": f"{args.data_dir}/^irx_data.csv",
            "uup": f"{args.data_dir}/uup_data.csv",
            "spy": f"{args.data_dir}/spy_data.csv",
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

        # 최근 매크로 데이터로 예측
        macro_files = {
            "vix": f"{args.data_dir}/^vix_data.csv",
            "tnx": f"{args.data_dir}/^tnx_data.csv",
            "irx": f"{args.data_dir}/^irx_data.csv",
            "uup": f"{args.data_dir}/uup_data.csv",
            "spy": f"{args.data_dir}/spy_data.csv",
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
