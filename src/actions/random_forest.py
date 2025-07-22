#!/usr/bin/env python3
"""
Random Forest 기반 시장 상태 분류기
실제 머신러닝 모델을 사용하여 시장 상태 확률을 예측
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from actions.y_finance import YahooFinanceDataCollector
from actions.calculate_index import TechnicalIndicators


class MarketRegimeRF:
    """Random Forest 기반 시장 상태 분류기"""
    
    def __init__(self, verbose: bool = True):
        """
        MarketRegimeRF 초기화
        
        Args:
            verbose: 상세 로그 출력 여부
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # 모델 관련 변수들
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # 데이터 수집기
        self.data_collector = YahooFinanceDataCollector()
        self.tech_indicators = TechnicalIndicators()
        
        # 모델 저장 경로
        self.model_dir = Path("models/market_regime")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 시장 상태 매핑
        self.regime_mapping = {
            0: 'TRENDING_UP',
            1: 'TRENDING_DOWN', 
            2: 'VOLATILE',
            3: 'SIDEWAYS'
        }
        
        self.regime_mapping_reverse = {v: k for k, v in self.regime_mapping.items()}
    
    def _print(self, *args, level="info", **kwargs):
        """로그 출력"""
        if self.verbose:
            if level == "info":
                self.logger.info(*args, **kwargs)
            elif level == "warning":
                self.logger.warning(*args, **kwargs)
            elif level == "error":
                self.logger.error(*args, **kwargs)
            else:
                print(*args, **kwargs)
    
    def collect_training_data(self, start_date: str = "2020-01-01", end_date: str = None, data_dir: str = "data/macro") -> pd.DataFrame:
        """
        학습 데이터 수집 (저장된 데이터 사용)
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜 (None이면 현재)
            data_dir: 데이터 디렉토리 경로
            
        Returns:
            학습용 데이터프레임
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        self._print(f"저장된 데이터 로드 중: {start_date} ~ {end_date}")
        
        # SPY 데이터 로드
        spy_path = f"{data_dir}/spy_data.csv"
        if not os.path.exists(spy_path):
            raise ValueError(f"SPY 데이터 파일이 없습니다: {spy_path}")
        
        spy_data = pd.read_csv(spy_path, index_col=0, parse_dates=False)
        
        self._print(f"원본 SPY 데이터 크기: {len(spy_data)}개")
        self._print(f"SPY 데이터 컬럼: {list(spy_data.columns)}")
        
        # 컬럼명 정규화
        spy_data.columns = spy_data.columns.str.lower()
        
        # 날짜 필터링
        if 'datetime' in spy_data.columns:
            self._print("datetime 컬럼 발견, 날짜 필터링 시작")
            try:
                # 안전한 datetime 변환
                spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], errors='coerce')
                # NaN 값 제거
                spy_data = spy_data.dropna(subset=['datetime'])
                
                # 타임존 정보가 있으면 제거
                if spy_data['datetime'].dt.tz is not None:
                    spy_data['datetime'] = spy_data['datetime'].dt.tz_localize(None)
                
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                self._print(f"필터링 전 데이터 크기: {len(spy_data)}개")
                self._print(f"시작 날짜: {start_dt}, 종료 날짜: {end_dt}")
                self._print(f"데이터 날짜 범위: {spy_data['datetime'].min()} ~ {spy_data['datetime'].max()}")
                
                spy_data = spy_data[(spy_data['datetime'] >= start_dt) & (spy_data['datetime'] <= end_dt)]
                self._print(f"필터링 후 데이터 크기: {len(spy_data)}개")
                
                spy_data.set_index('datetime', inplace=True)
                self._print(f"날짜 필터링 완료: {start_date} ~ {end_date}")
            except Exception as e:
                self._print(f"날짜 필터링 중 오류: {e}", level="error")
                # 오류 발생 시 전체 데이터 사용
                self._print("날짜 필터링 실패로 전체 데이터 사용")
        else:
            # 인덱스가 숫자인 경우, 전체 데이터 사용
            self._print("날짜 컬럼이 없어 전체 데이터 사용")
        
        if spy_data.empty:
            raise ValueError("필터링된 SPY 데이터가 없습니다.")
        
        # 최소 데이터 포인트 확인
        if len(spy_data) < 50:
            raise ValueError(f"데이터 포인트가 너무 적습니다: {len(spy_data)}개 (최소 50개 필요)")
        
        self._print(f"SPY 데이터 로드 완료: {len(spy_data)}개")
        
        # 매크로 데이터 로드
        macro_symbols = ['^VIX', '^TNX', '^TYX', '^DXY', 'GC=F', '^TLT', '^TIP']
        macro_data = {}
        
        for symbol in macro_symbols:
            macro_path = f"{data_dir}/{symbol.lower()}_data.csv"
            if os.path.exists(macro_path):
                try:
                    data = pd.read_csv(macro_path, index_col=0, parse_dates=False)
                    if 'datetime' in data.columns:
                        data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
                        data.set_index('datetime', inplace=True)
                    
                    # 날짜 필터링
                    try:
                        if 'datetime' in data.columns:
                            # 안전한 datetime 변환
                            data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
                            data = data.dropna(subset=['datetime'])
                            
                            # 타임존 정보가 있으면 제거
                            if data['datetime'].dt.tz is not None:
                                data['datetime'] = data['datetime'].dt.tz_localize(None)
                            
                            start_dt = pd.to_datetime(start_date)
                            end_dt = pd.to_datetime(end_date)
                            data = data[(data['datetime'] >= start_dt) & (data['datetime'] <= end_dt)]
                            data.set_index('datetime', inplace=True)
                        else:
                            # 인덱스가 날짜인 경우
                            data.index = pd.to_datetime(data.index, errors='coerce')
                            data = data.dropna()
                            
                            if data.index.tz is not None:
                                data.index = data.index.tz_localize(None)
                            
                            start_dt = pd.to_datetime(start_date)
                            end_dt = pd.to_datetime(end_date)
                            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                    except Exception as e:
                        self._print(f"{symbol} 날짜 필터링 중 오류: {e}", level="warning")
                        continue
                    
                    if not data.empty:
                        macro_data[symbol] = data
                        self._print(f"  {symbol} 데이터 로드 완료: {len(data)}개")
                except Exception as e:
                    self._print(f"  {symbol} 데이터 로드 실패: {e}", level="warning")
        
        # 기술적 지표 계산 (기본 파라미터 사용)
        from ..actions.calculate_index import StrategyParams
        
        default_params = StrategyParams(
            atr_period=14,
            ema_short=20,
            ema_long=50,
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            bb_period=20,
            bb_std=2.0,
            stoch_k_period=14,
            stoch_d_period=3,
            williams_r_period=14,
            cci_period=20,
            adx_period=14,
            obv_smooth_period=20,
            donchian_period=20,
            keltner_period=20,
            keltner_multiplier=2.0,
            volatility_period=20
        )
        self._print(f"기술적 지표 계산 시작 (데이터 크기: {len(spy_data)}개)")
        tech_data = self.tech_indicators.calculate_all_indicators(spy_data, default_params)
        self._print(f"기술적 지표 계산 완료 (데이터 크기: {len(tech_data)}개)")
        
        # 매크로 데이터 병합
        self._print(f"매크로 데이터 병합 시작 (매크로 데이터 개수: {len(macro_data)}개)")
        for symbol, data in macro_data.items():
            if 'close' in data.columns:
                tech_data[f'{symbol}_close'] = data['close']
            elif 'Close' in data.columns:
                tech_data[f'{symbol}_close'] = data['Close']
        
        self._print(f"매크로 데이터 병합 완료 (데이터 크기: {len(tech_data)}개)")
        
        # 추가 특성 생성
        self._print("고급 특성 생성 시작")
        tech_data = self._create_advanced_features(tech_data)
        self._print(f"고급 특성 생성 완료 (데이터 크기: {len(tech_data)}개)")
        
        # NaN 값 처리 (모든 컬럼이 NaN인 행만 제거)
        self._print(f"NaN 값 처리 전 데이터 크기: {len(tech_data)}개")
        # NaN 비율 확인
        nan_ratio = tech_data.isnull().sum() / len(tech_data)
        self._print(f"NaN 비율이 높은 컬럼들: {nan_ratio[nan_ratio > 0.5].index.tolist()}")
        
        # 모든 컬럼이 NaN인 행만 제거
        tech_data = tech_data.dropna(how='all')
        self._print(f"모든 컬럼이 NaN인 행 제거 후 데이터 크기: {len(tech_data)}개")
        
        # 나머지 NaN 값은 0으로 채움
        tech_data = tech_data.fillna(0)
        self._print(f"NaN 값을 0으로 채운 후 데이터 크기: {len(tech_data)}개")
        
        self._print(f"학습 데이터 준비 완료: {len(tech_data)}개 샘플")
        return tech_data
    
    def _create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """고급 특성 생성"""
        # 컬럼명 매핑
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 기본 가격 특성들
        data['returns_1d'] = data[close_col].pct_change()
        data['returns_5d'] = data[close_col].pct_change(5)
        data['returns_20d'] = data[close_col].pct_change(20)
        data['volatility_20d'] = data['returns_1d'].rolling(20).std()
        # 데이터 크기에 맞게 rolling window 조정
        max_window = min(60, len(data) // 2)
        data['volatility_60d'] = data['returns_1d'].rolling(max_window).std()
        
        # 이동평균 관련 특성들
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            data['sma_ratio'] = data['sma_20'] / data['sma_50']
            data['price_sma20_ratio'] = data[close_col] / data['sma_20']
            data['price_sma50_ratio'] = data[close_col] / data['sma_50']
        
        # RSI 관련 특성들
        if 'rsi' in data.columns:
            data['rsi_ma'] = data['rsi'].rolling(14).mean()
            data['rsi_std'] = data['rsi'].rolling(14).std()
            data['rsi_zscore'] = (data['rsi'] - data['rsi_ma']) / data['rsi_std']
        
        # MACD 관련 특성들
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            data['macd_ratio'] = data['macd'] / (data['macd_signal'] + 1e-8)
        
        # 볼린저 밴드 관련 특성들
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data[close_col]
            data['bb_position'] = (data[close_col] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # ATR 관련 특성들
        if 'atr' in data.columns:
            data['atr_ratio'] = data['atr'] / data[close_col]
            data['atr_ma'] = data['atr'].rolling(14).mean()
            data['atr_ratio_ma'] = data['atr_ratio'].rolling(14).mean()
        
        # 거래량 관련 특성들
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            data['volume_price_trend'] = (data['volume'] * data['returns_1d']).rolling(20).sum()
        
        # 매크로 특성들
        macro_features = ['^VIX_close', '^TNX_close', '^TYX_close', '^DXY_close', 'GC=F_close', '^TLT_close', '^TIP_close']
        
        for feature in macro_features:
            if feature in data.columns:
                # 매크로 지표의 변화율
                data[f'{feature}_change'] = data[feature].pct_change()
                data[f'{feature}_ma'] = data[feature].rolling(20).mean()
                data[f'{feature}_ratio'] = data[feature] / data[f'{feature}_ma']
                
                # VIX 특별 처리
                if feature == '^VIX_close':
                    data['vix_volatility'] = data[feature].rolling(20).std()
                    data['vix_percentile'] = data[feature].rolling(252).rank(pct=True)
                
                # 금리 관련 특성들
                if feature in ['^TNX_close', '^TYX_close']:
                    data[f'{feature}_spread'] = data[feature] - data['^TNX_close'] if feature == '^TYX_close' else 0
        
        # 복합 특성들
        if 'rsi' in data.columns and 'volatility_20d' in data.columns:
            data['rsi_volatility'] = data['rsi'] * data['volatility_20d']
        
        if 'macd' in data.columns and 'volume_ratio' in data.columns:
            data['macd_volume'] = data['macd'] * data['volume_ratio']
        
        if '^VIX_close' in data.columns and 'returns_1d' in data.columns:
            data['vix_return_correlation'] = data['^VIX_close'].rolling(20).corr(data['returns_1d'])
        
        return data
    
    def _create_labels(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
        """시장 상태 라벨 생성 (미래 수익률 기반)"""
        # 기본 파라미터 설정
        if params is None:
            params = {
                'sma_short': 20, 'sma_long': 50, 'rsi_period': 14,
                'rsi_overbought': 70, 'rsi_oversold': 30, 'atr_period': 14,
                'trend_weight': 0.4, 'momentum_weight': 0.3,
                'volatility_weight': 0.2, 'macro_weight': 0.1,
                'base_position': 0.8, 'trending_boost': 1.2, 'volatile_reduction': 0.5
            }
        
        # 미래 수익률 기반 라벨 생성
        labels = []
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 미래 수익률 계산 (5일 후)
        future_returns = data[close_col].pct_change(5).shift(-5)
        
        for i in range(len(data)):
            # 기본값은 SIDEWAYS (3)
            label = 3
            
            # 미래 수익률이 있는 경우에만 라벨 생성
            if i < len(data) - 5 and not pd.isna(future_returns.iloc[i]):
                future_return = future_returns.iloc[i]
                
                # 수익률 기준으로 라벨 분류
                if future_return > 0.02:  # 2% 이상 상승
                    label = 0  # TRENDING_UP
                elif future_return < -0.02:  # 2% 이상 하락
                    label = 1  # TRENDING_DOWN
                elif abs(future_return) > 0.01:  # 1% 이상 변동
                    label = 2  # VOLATILE
                else:
                    label = 3  # SIDEWAYS (1% 미만 변동)
            
            labels.append(label)
        
        data['regime_label'] = labels
        self._print(f"미래 수익률 기반 라벨 생성 완료: {len(labels)}개")
        
        # 라벨 분포 확인
        label_counts = pd.Series(labels).value_counts().sort_index()
        self._print(f"라벨 분포: {dict(label_counts)}")
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """특성 준비"""
        # 사용할 특성들 선택 (라벨 생성에 사용된 지표들 제외)
        feature_columns = [
            # 기본 가격 특성들 (미래 수익률과 독립적)
            'returns_1d', 'returns_5d', 'returns_20d', 'volatility_20d', 'volatility_60d',
            
            # 이동평균 특성들 (미래 수익률과 독립적)
            'sma_ratio', 'price_sma20_ratio', 'price_sma50_ratio',
            
            # MACD 특성들 (미래 수익률과 독립적)
            'macd', 'macd_signal', 'macd_histogram', 'macd_ratio',
            
            # 볼린저 밴드 특성들 (미래 수익률과 독립적)
            'bb_width', 'bb_position',
            
            # 거래량 특성들 (미래 수익률과 독립적)
            'volume_ratio', 'volume_price_trend',
            
            # 매크로 특성들 (미래 수익률과 독립적)
            '^VIX_close', '^VIX_close_change', '^VIX_close_ratio', 'vix_volatility', 'vix_percentile',
            '^TNX_close', '^TNX_close_change', '^TNX_close_ratio',
            '^TYX_close', '^TYX_close_change', '^TYX_close_ratio',
            '^DXY_close', '^DXY_close_change', '^DXY_close_ratio',
            'GC=F_close', 'GC=F_close_change', 'GC=F_close_ratio',
            '^TLT_close', '^TLT_close_change', '^TLT_close_ratio',
            '^TIP_close', '^TIP_close_change', '^TIP_close_ratio',
            
            # 복합 특성들 (미래 수익률과 독립적)
            'macd_volume', 'vix_return_correlation'
        ]
        
        # 제외된 특성들: RSI, ATR 관련 (라벨 생성에 사용됨)
        # 'rsi', 'rsi_ma', 'rsi_std', 'rsi_zscore', 'atr', 'atr_ratio', 'atr_ma', 'atr_ratio_ma', 'rsi_volatility'
        
        # 실제 존재하는 특성들만 선택
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 특성 데이터 준비
        X = data[available_features].copy()
        
        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill').fillna(0)
        
        return X, available_features
    
    def train_model(self, data: pd.DataFrame = None, params: Dict[str, Any] = None, save_model: bool = True) -> Dict[str, Any]:
        """
        Random Forest 모델 학습
        
        Args:
            data: 학습 데이터 (None이면 자동 수집)
            params: 시장 상태 분류에 사용할 파라미터 (None이면 기본값 사용)
            save_model: 모델 저장 여부
            
        Returns:
            학습 결과
        """
        if data is None:
            data = self.collect_training_data()
        
        # 라벨 생성 시 파라미터 전달
        data = self._create_labels(data, params)
        
        # 특성과 라벨 준비
        X, feature_names = self.prepare_features(data)
        y = data['regime_label'].dropna()
        
        # 인덱스 맞추기
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 100:
            raise ValueError(f"학습 데이터가 부족합니다: {len(X)}개")
        
        self._print(f"모델 학습 시작: {len(X)}개 샘플, {len(feature_names)}개 특성")
        
        # 시간적 분할 (과거 70%로 학습, 최근 30%로 테스트)
        split_idx = int(len(X) * 0.7)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        self._print(f"시간적 분할: 학습 {len(X_train)}개, 테스트 {len(X_test)}개")
        self._print(f"학습 기간: {X_train.index[0]} ~ {X_train.index[-1]}")
        self._print(f"테스트 기간: {X_test.index[0]} ~ {X_test.index[-1]}")
        
        # 테스트 데이터의 라벨 분포 확인
        test_label_counts = y_test.value_counts().sort_index()
        self._print(f"테스트 데이터 라벨 분포: {dict(test_label_counts)}")
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest 모델 생성 및 학습
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 모델 성능 평가
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # 교차 검증
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # 특성 중요도
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 결과 저장
        self.feature_names = feature_names
        self.is_trained = True
        
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_features': len(feature_names)
        }
        
        self._print(f"모델 학습 완료:")
        self._print(f"  훈련 정확도: {train_score:.4f}")
        self._print(f"  테스트 정확도: {test_score:.4f}")
        self._print(f"  교차 검증: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 모델 저장
        if save_model:
            self.save_model()
        
        return results
    
    def predict_probabilities(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        현재 시장 상태 확률 예측
        
        Args:
            data: 예측할 데이터
            
        Returns:
            시장 상태별 확률
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train_model()을 먼저 실행하세요.")
        
        # 특성 준비
        X, _ = self.prepare_features(data)
        
        # 최신 데이터만 사용
        latest_data = X.iloc[-1:].copy()
        
        # 스케일링
        latest_scaled = self.scaler.transform(latest_data)
        
        # 확률 예측
        probabilities = self.model.predict_proba(latest_scaled)[0]
        
        # 결과 매핑
        result = {}
        for i, prob in enumerate(probabilities):
            regime_name = self.regime_mapping[i].lower()
            result[regime_name] = float(prob)
        
        return result
    
    def save_model(self, filepath: str = None):
        """모델 저장"""
        if filepath is None:
            filepath = self.model_dir / "market_regime_rf_model.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'regime_mapping': self.regime_mapping,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self._print(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str = None):
        """모델 로드"""
        if filepath is None:
            filepath = self.model_dir / "market_regime_rf_model.pkl"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.regime_mapping = model_data['regime_mapping']
        self.is_trained = True
        
        self._print(f"모델 로드 완료: {filepath}")
    
    def get_current_market_probabilities(self, data_dir: str = "data/macro") -> Dict[str, float]:
        """
        현재 시장 상태 확률 계산
        
        Args:
            data_dir: 데이터 디렉토리 경로
            
        Returns:
            현재 시장 상태별 확률
        """
        # 최근 데이터 로드
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        
        data = self.collect_training_data(start_date, end_date, data_dir)
        
        # 확률 예측
        probabilities = self.predict_probabilities(data)
        
        return probabilities


def main():
    """메인 실행 함수"""
    # Random Forest 모델 초기화
    rf_model = MarketRegimeRF(verbose=True)
    
    try:
        # 기존 모델 로드 시도
        rf_model.load_model()
        print("기존 모델을 로드했습니다.")
    except FileNotFoundError:
        print("기존 모델이 없습니다. 새로 학습을 시작합니다.")
        # 모델 학습
        results = rf_model.train_model()
        print(f"모델 학습 완료: 테스트 정확도 {results['test_score']:.4f}")
    
    # 현재 시장 상태 확률 예측
    probabilities = rf_model.get_current_market_probabilities()
    
    print("\n📊 현재 시장 상태 확률 (ML 기반):")
    for regime, prob in probabilities.items():
        print(f"  {regime.upper()}: {prob:.1%}")


if __name__ == "__main__":
    main()
