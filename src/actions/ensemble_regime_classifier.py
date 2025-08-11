"""
RF-XGBoost 앙상블 보팅 시장 체제 분류기
Random Forest와 XGBoost를 결합한 투표 방식 체제 분류

주요 특징:
- Soft Voting: 각 분류기의 확률적 예측을 결합
- Hard Voting: 각 분류기의 최종 예측을 다수결로 결정  
- Weighted Voting: 모델 성능에 따른 가중치 적용
- 동적 가중치 조정: 최근 성능에 따른 실시간 가중치 업데이트
- Cross Validation 기반 성능 평가 및 가중치 결정
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
import pickle
import os

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("⚠️ XGBoost not found. Using LightGBM as fallback.")
    HAS_XGBOOST = False
    try:
        import lightgbm as lgb
        HAS_LIGHTGBM = True
    except ImportError:
        print("⚠️ LightGBM not found. Using only RandomForest.")
        HAS_LIGHTGBM = False

try:
    from .ml_regime_classifier import DynamicRegimeLabelGenerator
except ImportError:
    from ml_regime_classifier import DynamicRegimeLabelGenerator

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class RFXGBEnsembleRegimeClassifier:
    """
    RF-XGBoost 앙상블 보팅 시장 체제 분류기
    
    Random Forest와 XGBoost(또는 LightGBM)을 결합하여 더 안정적이고 정확한 체제 분류 제공
    
    주요 앙상블 전략:
    1. Soft Voting: 확률 분포 결합
    2. Hard Voting: 다수결 투표
    3. Weighted Voting: CV 성능 기반 가중치
    4. Stacking: 메타 모델을 통한 앙상블
    5. Dynamic Weighting: 적응적 가중치 조정
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.ensemble_config = config.get("ensemble_regime", {})
        self.ml_config = config.get("ml_regime", {})
        
        # Random Forest 모델 초기화
        self.rf_model = RandomForestClassifier(
            n_estimators=self.ml_config.get("n_estimators", 100),
            max_depth=self.ml_config.get("max_depth", 10),
            min_samples_split=self.ml_config.get("min_samples_split", 20),
            min_samples_leaf=self.ml_config.get("min_samples_leaf", 10),
            random_state=self.ml_config.get("random_state", 42),
            class_weight='balanced',
            n_jobs=-1
        )
        
        # XGBoost 또는 LightGBM 모델 초기화
        if HAS_XGBOOST:
            self.gbm_model = xgb.XGBClassifier(
                n_estimators=self.ml_config.get("n_estimators", 100),
                max_depth=self.ml_config.get("max_depth", 6),
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.ml_config.get("random_state", 42),
                n_jobs=-1,
                eval_metric='logloss'
            )
            self.gbm_name = "XGBoost"
        elif HAS_LIGHTGBM:
            self.gbm_model = lgb.LGBMClassifier(
                n_estimators=self.ml_config.get("n_estimators", 100),
                max_depth=self.ml_config.get("max_depth", 6),
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.ml_config.get("random_state", 42),
                n_jobs=-1,
                verbosity=-1
            )
            self.gbm_name = "LightGBM"
        else:
            # GBM 없이 RF만 사용 (두 번째 RF를 다른 설정으로)
            self.gbm_model = RandomForestClassifier(
                n_estimators=self.ml_config.get("n_estimators", 150),
                max_depth=self.ml_config.get("max_depth", 15),
                min_samples_split=self.ml_config.get("min_samples_split", 10),
                min_samples_leaf=self.ml_config.get("min_samples_leaf", 5),
                random_state=self.ml_config.get("random_state", 42) + 1,
                class_weight='balanced',
                n_jobs=-1
            )
            self.gbm_name = "RandomForest2"
        
        # 피처 스케일러
        self.scaler = StandardScaler()
        
        # 라벨 생성기
        self.label_generator = DynamicRegimeLabelGenerator(config)
        
        # 앙상블 설정
        self.voting_strategy = self.ensemble_config.get("voting_strategy", "weighted")
        self.enable_dynamic_weights = self.ensemble_config.get("enable_dynamic_weights", True)
        
        # 기본 가중치 (RF, XGB/LightGBM)
        self.base_weights = {
            "rf": self.ensemble_config.get("rf_weight", 0.5),
            "gbm": self.ensemble_config.get("gbm_weight", 0.5)
        }
        
        # 동적 가중치 (초기값은 기본 가중치)
        self.current_weights = self.base_weights.copy()
        
        # 성능 추적
        self.performance_history = {
            "rf": [],
            "gbm": [],
            "ensemble": []
        }
        
        # CV 점수 저장
        self.cv_scores = {"rf": None, "gbm": None}
        
        # 예측 이력 저장
        self.prediction_history = []
        
        # 클래스 매핑
        self.class_names = ["TRENDING_UP", "TRENDING_DOWN", "SIDEWAYS", "VOLATILE"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
        # 피처명 저장
        self.feature_names = []
        
        # 학습 상태
        self.is_fitted = False
        
        logger.info("RFXGBEnsembleRegimeClassifier 초기화 완료")
        logger.info(f"모델: RandomForest + {self.gbm_name}")
        logger.info(f"투표 전략: {self.voting_strategy}")
        logger.info(f"기본 가중치 - RF: {self.base_weights['rf']:.2f}, {self.gbm_name}: {self.base_weights['gbm']:.2f}")

    def fit(self, macro_data: pd.DataFrame, force_retrain: bool = False):
        """RF-XGBoost 앙상블 분류기 학습"""
        logger.info("RF-XGBoost 앙상블 분류기 학습 시작")
        
        try:
            # 1. 동적 라벨 생성
            logger.info("동적 시장 체제 라벨 생성")
            labeled_data = self.label_generator.generate_dynamic_labels(macro_data)
            
            if len(labeled_data) < 100:
                logger.warning(f"학습 데이터 부족: {len(labeled_data)}개 (최소 100개 권장)")
            
            # 2. 포괄적 피처 추출
            logger.info("포괄적 피처 추출")
            features = self._extract_comprehensive_features(macro_data)
            
            # 3. 라벨 정렬 (인덱스 매칭)
            common_index = features.index.intersection(labeled_data.index)
            features_aligned = features.loc[common_index]
            labels_aligned = labeled_data.loc[common_index]['regime_label']
            
            if len(features_aligned) < 50:
                logger.error(f"정렬된 학습 데이터 부족: {len(features_aligned)}개")
                return False
            
            # 4. 데이터 준비
            X = features_aligned.values
            y = self.label_encoder.transform(labels_aligned.values)
            
            # 피처 스케일링
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = list(features_aligned.columns)
            
            logger.info(f"학습 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 피처")
            
            # 5. 교차 검증을 통한 개별 모델 성능 평가
            logger.info("교차 검증을 통한 모델 성능 평가")
            
            rf_cv_scores = cross_val_score(self.rf_model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)
            gbm_cv_scores = cross_val_score(self.gbm_model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)
            
            self.cv_scores["rf"] = rf_cv_scores
            self.cv_scores["gbm"] = gbm_cv_scores
            
            logger.info(f"RF CV 점수: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
            logger.info(f"{self.gbm_name} CV 점수: {gbm_cv_scores.mean():.4f} ± {gbm_cv_scores.std():.4f}")
            
            # 6. CV 성능 기반 가중치 계산
            if self.voting_strategy == "weighted":
                self._update_weights_from_cv_scores()
            
            # 7. 개별 모델 학습
            logger.info("개별 모델 학습")
            self.rf_model.fit(X_scaled, y)
            self.gbm_model.fit(X_scaled, y)
            
            # 8. 앙상블 학습 완료
            self.is_fitted = True
            logger.info("RF-XGBoost 앙상블 분류기 학습 완료")
            
            # 9. 피처 중요도 로깅
            self._log_feature_importance()
            
            return True
            
        except Exception as e:
            logger.error(f"앙상블 분류기 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_comprehensive_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """포괄적 피처 추출"""
        try:
            from ml_regime_classifier import DynamicMLRegimeClassifier
            ml_classifier = DynamicMLRegimeClassifier(self.config)
            return ml_classifier.extract_comprehensive_features_from_data(macro_data)
        except Exception as e:
            logger.error(f"피처 추출 실패: {e}")
            # 기본 피처 반환
            return pd.DataFrame({
                "vix_level": [20.0] * len(macro_data),
                "yield_spread": [1.5] * len(macro_data),
                "market_momentum": [0.0] * len(macro_data),
            }, index=macro_data.index)

    def _update_weights_from_cv_scores(self):
        """CV 점수 기반 가중치 업데이트"""
        rf_score = self.cv_scores["rf"].mean()
        gbm_score = self.cv_scores["gbm"].mean()
        
        # 성능 기반 가중치 계산
        total_score = rf_score + gbm_score
        if total_score > 0:
            self.current_weights["rf"] = rf_score / total_score
            self.current_weights["gbm"] = gbm_score / total_score
            
            logger.info(f"CV 기반 가중치 업데이트 - RF: {self.current_weights['rf']:.3f}, "
                       f"{self.gbm_name}: {self.current_weights['gbm']:.3f}")

    def _log_feature_importance(self):
        """피처 중요도 로깅"""
        if len(self.feature_names) == 0:
            return
            
        try:
            # RF 피처 중요도
            rf_importance = self.rf_model.feature_importances_
            
            # GBM 피처 중요도
            if hasattr(self.gbm_model, 'feature_importances_'):
                gbm_importance = self.gbm_model.feature_importances_
            else:
                gbm_importance = np.zeros_like(rf_importance)
            
            # 상위 10개 피처 출력
            rf_top_features = sorted(zip(self.feature_names, rf_importance), 
                                   key=lambda x: x[1], reverse=True)[:10]
            
            logger.info("=== RF 상위 피처 중요도 ===")
            for name, importance in rf_top_features:
                logger.info(f"  {name}: {importance:.4f}")
                
        except Exception as e:
            logger.warning(f"피처 중요도 로깅 실패: {e}")

    def predict(self, macro_data: pd.DataFrame) -> Dict:
        """RF-XGBoost 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 분류기가 학습되지 않았습니다")
        
        try:
            logger.info("앙상블 예측 시작")
            
            # 1. 피처 추출
            features = self._extract_comprehensive_features(macro_data)
            if len(features) == 0:
                logger.error("예측용 피처 추출 실패")
                return self._fallback_prediction()
            
            # 2. 최신 데이터로 예측
            latest_features = features.iloc[-1:].values
            X_scaled = self.scaler.transform(latest_features)
            
            # 3. 개별 모델 예측
            rf_pred = self.rf_model.predict(X_scaled)[0]
            rf_proba = self.rf_model.predict_proba(X_scaled)[0]
            rf_confidence = float(np.max(rf_proba))
            rf_pred_label = self.label_encoder.inverse_transform([rf_pred])[0]
            
            gbm_pred = self.gbm_model.predict(X_scaled)[0]
            gbm_proba = self.gbm_model.predict_proba(X_scaled)[0]
            gbm_confidence = float(np.max(gbm_proba))
            gbm_pred_label = self.label_encoder.inverse_transform([gbm_pred])[0]
            
            # 4. 앙상블 예측 수행
            ensemble_result = self._ensemble_predict(
                rf_pred_label, rf_proba, rf_confidence,
                gbm_pred_label, gbm_proba, gbm_confidence
            )
            
            # 5. 예측 이력 저장
            prediction_record = {
                'timestamp': datetime.now(),
                'rf_prediction': rf_pred_label,
                'gbm_prediction': gbm_pred_label,
                'ensemble_prediction': ensemble_result['predicted_regime'],
                'rf_confidence': rf_confidence,
                'gbm_confidence': gbm_confidence,
                'ensemble_confidence': ensemble_result['confidence'],
                'weights_used': self.current_weights.copy()
            }
            self.prediction_history.append(prediction_record)
            
            # 6. 동적 가중치 업데이트 (옵션)
            if self.enable_dynamic_weights and len(self.prediction_history) > 10:
                self._update_dynamic_weights()
            
            logger.info(f"앙상블 예측 완료: {ensemble_result['predicted_regime']} "
                       f"(신뢰도: {ensemble_result['confidence']:.3f})")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"앙상블 예측 실패: {e}")
            return self._fallback_prediction()

    def _ensemble_predict(self, rf_pred: str, rf_proba: np.ndarray, rf_conf: float,
                         gbm_pred: str, gbm_proba: np.ndarray, gbm_conf: float) -> Dict:
        """앙상블 투표 수행"""
        
        if self.voting_strategy == "soft":
            # Soft Voting: 확률 분포 가중 결합
            ensemble_proba = (
                self.current_weights['rf'] * rf_proba + 
                self.current_weights['gbm'] * gbm_proba
            )
            ensemble_idx = np.argmax(ensemble_proba)
            ensemble_pred = self.class_names[ensemble_idx]
            ensemble_conf = float(ensemble_proba[ensemble_idx])
            
        elif self.voting_strategy == "hard":
            # Hard Voting: 다수결
            if rf_pred == gbm_pred:
                ensemble_pred = rf_pred
                ensemble_conf = (rf_conf + gbm_conf) / 2
            else:
                if rf_conf >= gbm_conf:
                    ensemble_pred, ensemble_conf = rf_pred, rf_conf
                else:
                    ensemble_pred, ensemble_conf = gbm_pred, gbm_conf
                    
        elif self.voting_strategy == "weighted":
            # Weighted Voting: 성능 기반 가중치
            rf_score = self.current_weights['rf'] * rf_conf
            gbm_score = self.current_weights['gbm'] * gbm_conf
            
            if rf_pred == gbm_pred:
                ensemble_pred = rf_pred
                ensemble_conf = (rf_score + gbm_score)
            else:
                if rf_score >= gbm_score:
                    ensemble_pred, ensemble_conf = rf_pred, rf_score
                else:
                    ensemble_pred, ensemble_conf = gbm_pred, gbm_score
                    
        else:
            # 기본값: Weighted
            rf_score = self.current_weights['rf'] * rf_conf
            gbm_score = self.current_weights['gbm'] * gbm_conf
            
            if rf_score >= gbm_score:
                ensemble_pred, ensemble_conf = rf_pred, rf_score
            else:
                ensemble_pred, ensemble_conf = gbm_pred, gbm_score
        
        return {
            'predicted_regime': ensemble_pred,
            'confidence': ensemble_conf,
            'individual_predictions': {
                'rf': {'regime': rf_pred, 'confidence': rf_conf},
                'gbm': {'regime': gbm_pred, 'confidence': gbm_conf}
            },
            'voting_strategy': self.voting_strategy,
            'weights': self.current_weights.copy()
        }

    def _update_dynamic_weights(self):
        """최근 성능에 따른 동적 가중치 업데이트"""
        if len(self.prediction_history) < 20:
            return
        
        try:
            recent_predictions = self.prediction_history[-20:]
            
            # 각 모델의 일관성 점수
            rf_consistency = self._calculate_consistency([p['rf_prediction'] for p in recent_predictions])
            gbm_consistency = self._calculate_consistency([p['gbm_prediction'] for p in recent_predictions])
            
            # 신뢰도 평균
            rf_avg_conf = np.mean([p['rf_confidence'] for p in recent_predictions])
            gbm_avg_conf = np.mean([p['gbm_confidence'] for p in recent_predictions])
            
            # 종합 점수
            rf_score = (rf_consistency * 0.6 + rf_avg_conf * 0.4)
            gbm_score = (gbm_consistency * 0.6 + gbm_avg_conf * 0.4)
            
            # 정규화하여 가중치 업데이트
            total_score = rf_score + gbm_score
            if total_score > 0:
                new_rf_weight = rf_score / total_score
                new_gbm_weight = gbm_score / total_score
                
                # 관성 적용
                inertia = 0.7
                self.current_weights['rf'] = (
                    inertia * self.current_weights['rf'] + (1 - inertia) * new_rf_weight
                )
                self.current_weights['gbm'] = (
                    inertia * self.current_weights['gbm'] + (1 - inertia) * new_gbm_weight
                )
                
                logger.info(f"동적 가중치 업데이트 - RF: {self.current_weights['rf']:.3f}, "
                           f"{self.gbm_name}: {self.current_weights['gbm']:.3f}")
                
        except Exception as e:
            logger.warning(f"동적 가중치 업데이트 실패: {e}")

    def _calculate_consistency(self, predictions: List[str]) -> float:
        """예측 일관성 점수 계산"""
        if len(predictions) <= 1:
            return 0.5
        
        consistent_count = 0
        for i in range(1, len(predictions)):
            if predictions[i] == predictions[i-1]:
                consistent_count += 1
        
        return consistent_count / (len(predictions) - 1)

    def _fallback_prediction(self) -> Dict:
        """오류 시 대체 예측"""
        logger.warning("대체 예측 모드 활성화")
        
        return {
            'predicted_regime': 'SIDEWAYS',
            'confidence': 0.25,
            'individual_predictions': {
                'rf': {'regime': 'SIDEWAYS', 'confidence': 0.25},
                'gbm': {'regime': 'SIDEWAYS', 'confidence': 0.25}
            },
            'voting_strategy': 'fallback',
            'weights': {'rf': 0.5, 'gbm': 0.5}
        }

    def get_prediction_summary(self, n_recent: int = 20) -> Dict:
        """최근 예측 요약 통계"""
        
        if len(self.prediction_history) == 0:
            return {"message": "예측 이력 없음"}
        
        recent = self.prediction_history[-n_recent:] if n_recent > 0 else self.prediction_history
        
        # 각 모델별 예측 분포
        rf_preds = [p['rf_prediction'] for p in recent]
        gbm_preds = [p['gbm_prediction'] for p in recent]  
        ensemble_preds = [p['ensemble_prediction'] for p in recent]
        
        # 일치율 계산
        agreements = sum(1 for p in recent if p['rf_prediction'] == p['gbm_prediction'])
        agreement_rate = agreements / len(recent)
        
        # 평균 신뢰도
        avg_rf_conf = np.mean([p['rf_confidence'] for p in recent])
        avg_gbm_conf = np.mean([p['gbm_confidence'] for p in recent])
        avg_ensemble_conf = np.mean([p['ensemble_confidence'] for p in recent])
        
        return {
            'total_predictions': len(recent),
            'agreement_rate': agreement_rate,
            'average_confidence': {
                'rf': avg_rf_conf,
                'gbm': avg_gbm_conf,
                'ensemble': avg_ensemble_conf
            },
            'prediction_distribution': {
                'rf': {regime: rf_preds.count(regime) for regime in self.class_names},
                'gbm': {regime: gbm_preds.count(regime) for regime in self.class_names},
                'ensemble': {regime: ensemble_preds.count(regime) for regime in self.class_names}
            },
            'current_weights': self.current_weights.copy(),
            'cv_scores': {
                'rf': self.cv_scores["rf"].mean() if self.cv_scores["rf"] is not None else None,
                'gbm': self.cv_scores["gbm"].mean() if self.cv_scores["gbm"] is not None else None
            }
        }


def main():
    """RF-XGBoost 앙상블 분류기 테스트"""
    import argparse
    import json
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="RF-XGBoost 앙상블 시장 체제 분류기")
    parser.add_argument("--train", action="store_true", help="앙상블 모델 학습")
    parser.add_argument("--predict", action="store_true", help="앙상블 예측")
    parser.add_argument("--config", type=str, default="config/config_trader.json", help="설정 파일")
    parser.add_argument("--model-path", type=str, default="models/trader/rf_xgb_ensemble_model.pkl", help="모델 저장/로딩 경로")
    parser.add_argument("--force-retrain", action="store_true", help="강제 재학습")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)
    
    # 앙상블 설정 확인 (없는 경우 기본값)
    if "ensemble_regime" not in config:
        config["ensemble_regime"] = {
            "voting_strategy": "weighted",
            "enable_dynamic_weights": True,
            "rf_weight": 0.5,
            "gbm_weight": 0.5
        }
    
    classifier = RFXGBEnsembleRegimeClassifier(config)
    
    if args.train:
        print("🔄 RF-XGBoost 앙상블 분류기 학습 시작")
        
        # 데이터 로딩
        from ml_regime_classifier import DynamicMLRegimeClassifier
        ml_classifier = DynamicMLRegimeClassifier(config)
        macro_data = ml_classifier.load_comprehensive_data_from_files()
        
        if macro_data is None or macro_data.empty:
            print("❌ 매크로 데이터 로딩 실패")
            sys.exit(1)
        
        print(f"📊 매크로 데이터: {len(macro_data)} 행, {len(macro_data.columns)}개 컬럼")
        
        # 학습 실행
        success = classifier.fit(macro_data, force_retrain=args.force_retrain)
        
        if success:
            # 모델 저장
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            with open(args.model_path, 'wb') as f:
                pickle.dump(classifier, f)
            print(f"✅ RF-XGBoost 앙상블 모델 학습 및 저장 완료: {args.model_path}")
        else:
            print("❌ 앙상블 모델 학습 실패")
            sys.exit(1)
    
    elif args.predict:
        print("🔮 RF-XGBoost 앙상블 예측 시작")
        
        # 모델 로딩
        if os.path.exists(args.model_path):
            with open(args.model_path, 'rb') as f:
                classifier = pickle.load(f)
        else:
            print(f"❌ 모델 파일을 찾을 수 없음: {args.model_path}")
            sys.exit(1)
        
        # 데이터 로딩
        from ml_regime_classifier import DynamicMLRegimeClassifier
        ml_classifier = DynamicMLRegimeClassifier(config)
        macro_data = ml_classifier.load_comprehensive_data_from_files()
        
        if macro_data is None or macro_data.empty:
            print("❌ 매크로 데이터 로딩 실패")
            sys.exit(1)
        
        # 예측 실행
        result = classifier.predict(macro_data)
        
        print(f"\n📊 RF-XGBoost 앙상블 예측 결과:")
        print(f"🎯 예측 체제: {result['predicted_regime']}")
        print(f"📈 신뢰도: {result['confidence']:.3f}")
        print(f"🗳️ 투표 전략: {result['voting_strategy']}")
        
        print(f"\n🔍 개별 모델 예측:")
        for model_name, pred_info in result['individual_predictions'].items():
            model_display = "RandomForest" if model_name == "rf" else classifier.gbm_name
            print(f"  {model_display}: {pred_info['regime']} (신뢰도: {pred_info['confidence']:.3f})")
        
        print(f"\n⚖️ 현재 가중치:")
        for model_name, weight in result['weights'].items():
            model_display = "RandomForest" if model_name == "rf" else classifier.gbm_name
            print(f"  {model_display}: {weight:.3f}")
        
        # 예측 요약 통계
        summary = classifier.get_prediction_summary()
        if 'total_predictions' in summary:
            print(f"\n📊 예측 통계 (최근 {summary['total_predictions']}회):")
            print(f"  모델 일치율: {summary['agreement_rate']:.1%}")
            print(f"  평균 신뢰도: RF {summary['average_confidence']['rf']:.3f}, "
                  f"{classifier.gbm_name} {summary['average_confidence']['gbm']:.3f}, "
                  f"앙상블 {summary['average_confidence']['ensemble']:.3f}")
            
            if summary['cv_scores']['rf'] is not None:
                print(f"  CV 점수: RF {summary['cv_scores']['rf']:.4f}, "
                      f"{classifier.gbm_name} {summary['cv_scores']['gbm']:.4f}")
    
    else:
        print("사용법:")
        print("  --train          # RF-XGBoost 앙상블 모델 학습")
        print("  --predict        # 앙상블 예측 실행")
        print("  --force-retrain  # 강제 재학습")


if __name__ == "__main__":
    main()
            latest_features = features.iloc[-1:].values
            X_scaled = self.ml_classifier.scaler.transform(latest_features)
            
            ml_proba = self.ml_classifier.model.predict_proba(X_scaled)[0]
            ml_pred = self.ml_classifier.model.predict(X_scaled)[0]
            ml_pred_label = self.label_encoder.inverse_transform([ml_pred])[0]
            
            # 2. 앙상블 예측 수행
            ensemble_result = self._ensemble_predict(hmm_results, ml_pred_label, ml_proba)
            
            # 3. 예측 이력 저장
            prediction_record = {
                'timestamp': datetime.now(),
                'hmm_prediction': hmm_results.get('predicted_regime', 'SIDEWAYS'),
                'ml_prediction': ml_pred_label,
                'ensemble_prediction': ensemble_result['predicted_regime'],
                'hmm_confidence': hmm_results.get('confidence', 0.5),
                'ml_confidence': float(np.max(ml_proba)),
                'ensemble_confidence': ensemble_result['confidence'],
                'weights_used': self.current_weights.copy()
            }
            self.prediction_history.append(prediction_record)
            
            # 4. 동적 가중치 업데이트 (옵션)
            if self.enable_dynamic_weights and len(self.prediction_history) > 10:
                self._update_dynamic_weights()
            
            logger.info(f"앙상블 예측 완료: {ensemble_result['predicted_regime']} "
                       f"(신뢰도: {ensemble_result['confidence']:.3f})")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"앙상블 예측 실패: {e}")
            return self._fallback_prediction(macro_data)

    def _ensemble_predict(self, hmm_results: Dict, ml_pred: str, ml_proba: np.ndarray) -> Dict:
        """앙상블 투표 수행"""
        
        hmm_pred = hmm_results.get('predicted_regime', 'SIDEWAYS')
        hmm_confidence = hmm_results.get('confidence', 0.5)
        ml_confidence = float(np.max(ml_proba))
        
        if self.voting_strategy == "soft":
            # Soft Voting: 확률 분포 결합
            ensemble_pred, ensemble_conf = self._soft_voting(
                hmm_pred, hmm_confidence, ml_pred, ml_proba
            )
            
        elif self.voting_strategy == "hard":
            # Hard Voting: 다수결
            ensemble_pred, ensemble_conf = self._hard_voting(
                hmm_pred, hmm_confidence, ml_pred, ml_confidence
            )
            
        elif self.voting_strategy == "weighted":
            # Weighted Voting: 성능 기반 가중치
            ensemble_pred, ensemble_conf = self._weighted_voting(
                hmm_pred, hmm_confidence, ml_pred, ml_confidence
            )
            
        elif self.voting_strategy == "confidence":
            # Confidence-based: 더 확신하는 모델 선택
            ensemble_pred, ensemble_conf = self._confidence_based_voting(
                hmm_pred, hmm_confidence, ml_pred, ml_confidence
            )
            
        else:
            # 기본값: Weighted
            ensemble_pred, ensemble_conf = self._weighted_voting(
                hmm_pred, hmm_confidence, ml_pred, ml_confidence
            )
        
        return {
            'predicted_regime': ensemble_pred,
            'confidence': ensemble_conf,
            'individual_predictions': {
                'hmm': {'regime': hmm_pred, 'confidence': hmm_confidence},
                'ml': {'regime': ml_pred, 'confidence': ml_confidence}
            },
            'voting_strategy': self.voting_strategy,
            'weights': self.current_weights.copy()
        }

    def _soft_voting(self, hmm_pred: str, hmm_conf: float, ml_pred: str, ml_proba: np.ndarray) -> Tuple[str, float]:
        """Soft Voting: 확률 분포 가중 결합"""
        
        # HMM 결과를 확률 분포로 변환 (간단한 방식)
        hmm_proba = np.zeros(len(self.class_names))
        hmm_idx = self.class_names.index(hmm_pred)
        hmm_proba[hmm_idx] = hmm_conf
        
        # 나머지 확률을 균등하게 분배
        remaining_prob = 1.0 - hmm_conf
        for i in range(len(self.class_names)):
            if i != hmm_idx:
                hmm_proba[i] = remaining_prob / (len(self.class_names) - 1)
        
        # 가중 결합
        ensemble_proba = (
            self.current_weights['hmm'] * hmm_proba + 
            self.current_weights['ml'] * ml_proba
        )
        
        # 최종 예측
        ensemble_idx = np.argmax(ensemble_proba)
        ensemble_pred = self.class_names[ensemble_idx]
        ensemble_conf = float(ensemble_proba[ensemble_idx])
        
        return ensemble_pred, ensemble_conf

    def _hard_voting(self, hmm_pred: str, hmm_conf: float, ml_pred: str, ml_conf: float) -> Tuple[str, float]:
        """Hard Voting: 다수결 투표"""
        
        # 간단한 경우: 2개 모델이므로 일치/불일치만 확인
        if hmm_pred == ml_pred:
            # 일치하는 경우: 평균 신뢰도
            return hmm_pred, (hmm_conf + ml_conf) / 2
        else:
            # 불일치하는 경우: 더 높은 신뢰도를 가진 모델 선택
            if hmm_conf >= ml_conf:
                return hmm_pred, hmm_conf
            else:
                return ml_pred, ml_conf

    def _weighted_voting(self, hmm_pred: str, hmm_conf: float, ml_pred: str, ml_conf: float) -> Tuple[str, float]:
        """Weighted Voting: 성능 기반 가중치 투표"""
        
        # 각 예측에 가중치 적용
        hmm_score = self.current_weights['hmm'] * hmm_conf
        ml_score = self.current_weights['ml'] * ml_conf
        
        # 같은 예측인 경우
        if hmm_pred == ml_pred:
            return hmm_pred, (hmm_score + ml_score)
        
        # 다른 예측인 경우: 가중 점수가 높은 것 선택
        if hmm_score >= ml_score:
            return hmm_pred, hmm_score
        else:
            return ml_pred, ml_score

    def _confidence_based_voting(self, hmm_pred: str, hmm_conf: float, ml_pred: str, ml_conf: float) -> Tuple[str, float]:
        """Confidence-based: 더 확신하는 모델의 예측 사용"""
        
        # 신뢰도 임계값 이상인 경우만 고려
        confidence_threshold = 0.6
        
        if hmm_conf >= confidence_threshold and ml_conf >= confidence_threshold:
            # 둘 다 확신하는 경우: 가중 투표
            return self._weighted_voting(hmm_pred, hmm_conf, ml_pred, ml_conf)
        elif hmm_conf >= confidence_threshold:
            # HMM만 확신하는 경우
            return hmm_pred, hmm_conf
        elif ml_conf >= confidence_threshold:
            # ML만 확신하는 경우  
            return ml_pred, ml_conf
        else:
            # 둘 다 확신하지 못하는 경우: 가중 투표
            return self._weighted_voting(hmm_pred, hmm_conf, ml_pred, ml_conf)

    def _update_dynamic_weights(self):
        """최근 성능에 따른 동적 가중치 업데이트"""
        
        if len(self.prediction_history) < 20:
            return
        
        try:
            # 최근 20개 예측 분석
            recent_predictions = self.prediction_history[-20:]
            
            # 각 모델의 일관성 점수 계산 (임시적으로 단순한 방식 사용)
            hmm_consistency = self._calculate_consistency([p['hmm_prediction'] for p in recent_predictions])
            ml_consistency = self._calculate_consistency([p['ml_prediction'] for p in recent_predictions])
            
            # 신뢰도 평균
            hmm_avg_conf = np.mean([p['hmm_confidence'] for p in recent_predictions])
            ml_avg_conf = np.mean([p['ml_confidence'] for p in recent_predictions])
            
            # 종합 점수 (일관성 + 신뢰도)
            hmm_score = (hmm_consistency * 0.6 + hmm_avg_conf * 0.4)
            ml_score = (ml_consistency * 0.6 + ml_avg_conf * 0.4)
            
            # 정규화하여 가중치 업데이트
            total_score = hmm_score + ml_score
            if total_score > 0:
                new_hmm_weight = hmm_score / total_score
                new_ml_weight = ml_score / total_score
                
                # 급격한 변화 방지 (관성 적용)
                inertia = 0.7
                self.current_weights['hmm'] = (
                    inertia * self.current_weights['hmm'] + 
                    (1 - inertia) * new_hmm_weight
                )
                self.current_weights['ml'] = (
                    inertia * self.current_weights['ml'] + 
                    (1 - inertia) * new_ml_weight
                )
                
                logger.info(f"동적 가중치 업데이트 - HMM: {self.current_weights['hmm']:.3f}, "
                           f"ML: {self.current_weights['ml']:.3f}")
                
        except Exception as e:
            logger.warning(f"동적 가중치 업데이트 실패: {e}")

    def _calculate_consistency(self, predictions: List[str]) -> float:
        """예측 일관성 점수 계산"""
        if len(predictions) <= 1:
            return 0.5
        
        # 연속된 예측이 같은 경우를 일관성으로 평가
        consistent_count = 0
        for i in range(1, len(predictions)):
            if predictions[i] == predictions[i-1]:
                consistent_count += 1
        
        return consistent_count / (len(predictions) - 1)

    def _fallback_prediction(self, macro_data: pd.DataFrame) -> Dict:
        """오류 시 대체 예측"""
        logger.warning("대체 예측 모드 활성화")
        
        try:
            # HMM만 사용
            if self.hmm_classifier.is_fitted:
                return self.hmm_classifier.predict_regime(macro_data)
        except:
            pass
        
        # 완전한 대체: 기본 예측
        return {
            'predicted_regime': 'SIDEWAYS',
            'confidence': 0.25,
            'individual_predictions': {
                'hmm': {'regime': 'SIDEWAYS', 'confidence': 0.25},
                'ml': {'regime': 'SIDEWAYS', 'confidence': 0.25}
            },
            'voting_strategy': 'fallback',
            'weights': {'hmm': 0.5, 'ml': 0.5}
        }

    def save_model(self, filepath: str):
        """앙상블 모델 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'config': self.config,
                'ensemble_config': self.ensemble_config,
                'voting_strategy': self.voting_strategy,
                'base_weights': self.base_weights,
                'current_weights': self.current_weights,
                'performance_history': self.performance_history,
                'prediction_history': self.prediction_history[-100:],  # 최근 100개만 저장
                'class_names': self.class_names,
                'is_fitted': self.is_fitted,
                'ml_model': self.ml_classifier.model if self.ml_classifier.is_fitted else None,
                'ml_scaler': self.ml_classifier.scaler if self.ml_classifier.is_fitted else None,
                'ml_feature_names': self.ml_classifier.feature_names
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"앙상블 모델 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"앙상블 모델 저장 실패: {e}")

    def load_model(self, filepath: str):
        """앙상블 모델 로딩"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.ensemble_config = model_data['ensemble_config']
            self.voting_strategy = model_data['voting_strategy']
            self.base_weights = model_data['base_weights']
            self.current_weights = model_data['current_weights']
            self.performance_history = model_data['performance_history']
            self.prediction_history = model_data.get('prediction_history', [])
            self.class_names = model_data['class_names']
            self.is_fitted = model_data['is_fitted']
            
            # ML 모델 복원
            if model_data.get('ml_model'):
                self.ml_classifier.model = model_data['ml_model']
                self.ml_classifier.scaler = model_data['ml_scaler'] 
                self.ml_classifier.feature_names = model_data['ml_feature_names']
                self.ml_classifier.is_fitted = True
            
            logger.info(f"앙상블 모델 로딩 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"앙상블 모델 로딩 실패: {e}")

    def get_prediction_summary(self, n_recent: int = 20) -> Dict:
        """최근 예측 요약 통계"""
        
        if len(self.prediction_history) == 0:
            return {"message": "예측 이력 없음"}
        
        recent = self.prediction_history[-n_recent:] if n_recent > 0 else self.prediction_history
        
        # 각 모델별 예측 분포
        hmm_preds = [p['hmm_prediction'] for p in recent]
        ml_preds = [p['ml_prediction'] for p in recent]  
        ensemble_preds = [p['ensemble_prediction'] for p in recent]
        
        # 일치율 계산
        agreements = sum(1 for p in recent if p['hmm_prediction'] == p['ml_prediction'])
        agreement_rate = agreements / len(recent)
        
        # 평균 신뢰도
        avg_hmm_conf = np.mean([p['hmm_confidence'] for p in recent])
        avg_ml_conf = np.mean([p['ml_confidence'] for p in recent])
        avg_ensemble_conf = np.mean([p['ensemble_confidence'] for p in recent])
        
        return {
            'total_predictions': len(recent),
            'agreement_rate': agreement_rate,
            'average_confidence': {
                'hmm': avg_hmm_conf,
                'ml': avg_ml_conf,
                'ensemble': avg_ensemble_conf
            },
            'prediction_distribution': {
                'hmm': {regime: hmm_preds.count(regime) for regime in self.class_names},
                'ml': {regime: ml_preds.count(regime) for regime in self.class_names},
                'ensemble': {regime: ensemble_preds.count(regime) for regime in self.class_names}
            },
            'current_weights': self.current_weights.copy()
        }


def main():
    """앙상블 분류기 테스트"""
    import argparse
    import json
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="앙상블 시장 체제 분류기")
    parser.add_argument("--train", action="store_true", help="앙상블 모델 학습")
    parser.add_argument("--predict", action="store_true", help="앙상블 예측")
    parser.add_argument("--config", type=str, default="config/config_trader.json", help="설정 파일")
    parser.add_argument("--model-path", type=str, default="models/trader/ensemble_regime_model.pkl", help="모델 저장/로딩 경로")
    parser.add_argument("--force-retrain", action="store_true", help="강제 재학습")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        sys.exit(1)
    
    # 앙상블 설정 추가 (없는 경우)
    if "ensemble_regime" not in config:
        config["ensemble_regime"] = {
            "voting_strategy": "weighted",
            "enable_dynamic_weights": True,
            "hmm_weight": 0.6,
            "ml_weight": 0.4
        }
    
    classifier = EnsembleRegimeClassifier(config)
    
    if args.train:
        print("🔄 앙상블 분류기 학습 시작")
        
        # 데이터 로딩
        macro_data = classifier.ml_classifier.load_comprehensive_data_from_files()
        if macro_data is None or macro_data.empty:
            print("❌ 매크로 데이터 로딩 실패")
            sys.exit(1)
        
        print(f"📊 매크로 데이터: {len(macro_data)} 행, {len(macro_data.columns)}개 컬럼")
        
        # 학습 실행
        success = classifier.fit(macro_data, force_retrain=args.force_retrain)
        
        if success:
            # 모델 저장
            classifier.save_model(args.model_path)
            print(f"✅ 앙상블 모델 학습 및 저장 완료: {args.model_path}")
        else:
            print("❌ 앙상블 모델 학습 실패")
            sys.exit(1)
    
    elif args.predict:
        print("🔮 앙상블 예측 시작")
        
        # 모델 로딩
        if os.path.exists(args.model_path):
            classifier.load_model(args.model_path)
        else:
            print(f"❌ 모델 파일을 찾을 수 없음: {args.model_path}")
            sys.exit(1)
        
        # 데이터 로딩
        macro_data = classifier.ml_classifier.load_comprehensive_data_from_files()
        if macro_data is None or macro_data.empty:
            print("❌ 매크로 데이터 로딩 실패")
            sys.exit(1)
        
        # 예측 실행
        result = classifier.predict(macro_data)
        
        print(f"\n📊 앙상블 예측 결과:")
        print(f"🎯 예측 체제: {result['predicted_regime']}")
        print(f"📈 신뢰도: {result['confidence']:.3f}")
        print(f"🗳️ 투표 전략: {result['voting_strategy']}")
        
        print(f"\n🔍 개별 모델 예측:")
        for model_name, pred_info in result['individual_predictions'].items():
            print(f"  {model_name.upper()}: {pred_info['regime']} (신뢰도: {pred_info['confidence']:.3f})")
        
        print(f"\n⚖️ 현재 가중치:")
        for model_name, weight in result['weights'].items():
            print(f"  {model_name.upper()}: {weight:.3f}")
        
        # 예측 요약 통계
        summary = classifier.get_prediction_summary()
        if 'total_predictions' in summary:
            print(f"\n📊 예측 통계 (최근 {summary['total_predictions']}회):")
            print(f"  모델 일치율: {summary['agreement_rate']:.1%}")
            print(f"  평균 신뢰도: HMM {summary['average_confidence']['hmm']:.3f}, "
                  f"ML {summary['average_confidence']['ml']:.3f}, "
                  f"앙상블 {summary['average_confidence']['ensemble']:.3f}")
    
    else:
        print("사용법:")
        print("  --train          # 앙상블 모델 학습")
        print("  --predict        # 앙상블 예측 실행")
        print("  --force-retrain  # 강제 재학습")


if __name__ == "__main__":
    main()