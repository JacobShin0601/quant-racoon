#!/usr/bin/env python3
"""
Random Forest 기반 시장 상태 분류기
실제 머신러닝 모델을 사용하여 시장 상태 확률을 예측
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
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
    
    def __init__(self, verbose: bool = True, config_path: str = "config/config_macro.json"):
        """
        MarketRegimeRF 초기화
        
        Args:
            verbose: 상세 로그 출력 여부
            config_path: 설정 파일 경로
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # 설정 파일 로드
        self.config = self._load_config(config_path)
        
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
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._print(f"설정 파일 로드 완료: {config_path}")
            return config
        except Exception as e:
            self._print(f"설정 파일 로드 실패: {e}", level="error")
            return {}
    
    def _get_days_back(self, collection_type: str = "default") -> int:
        """설정에서 데이터 수집 기간 가져오기"""
        try:
            data_collection = self.config.get('data_collection', {})
            days_back = data_collection.get(f'{collection_type}_days_back', 
                                          data_collection.get('default_days_back', 730))
            return days_back
        except Exception as e:
            self._print(f"설정에서 데이터 수집 기간 로드 실패: {e}", level="error")
            return 730  # 기본값 2년
    
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
    
    def collect_training_data(self, start_date: str = None, end_date: str = None, data_dir: str = "data/macro") -> pd.DataFrame:
        """
        학습 데이터 수집 (저장된 데이터 사용)
        
        Args:
            start_date: 시작 날짜 (None이면 설정에서 계산)
            end_date: 종료 날짜 (None이면 현재)
            data_dir: 데이터 디렉토리 경로
            
        Returns:
            학습용 데이터프레임
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # 설정에서 모델 학습용 데이터 수집 기간 가져오기
            days_back = self._get_days_back("model_training")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            self._print(f"설정 기반 시작 날짜 설정: {start_date} ({days_back}일)")
        
        self._print(f"저장된 데이터 로드 중: {start_date} ~ {end_date}")
        
        # 절대 경로로 변환
        if not os.path.isabs(data_dir):
            # 현재 작업 디렉토리 기준으로 상대 경로 해결
            current_dir = os.getcwd()
            data_dir = os.path.join(current_dir, data_dir)
        
        self._print(f"데이터 디렉토리: {data_dir}")
        
        # SPY 데이터 로드
        spy_path = os.path.join(data_dir, "spy_data.csv")
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
                # 안전한 datetime 변환 (타임존 처리 포함)
                spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], utc=True, errors='coerce')
                # UTC에서 로컬로 변환하고 타임존 정보 제거
                spy_data['datetime'] = spy_data['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
                # NaN 값 제거
                spy_data = spy_data.dropna(subset=['datetime'])
                
                # datetime 타입 확인
                if pd.api.types.is_datetime64_any_dtype(spy_data['datetime']):
                    
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    self._print(f"필터링 전 데이터 크기: {len(spy_data)}개")
                    self._print(f"시작 날짜: {start_dt}, 종료 날짜: {end_dt}")
                    self._print(f"데이터 날짜 범위: {spy_data['datetime'].min()} ~ {spy_data['datetime'].max()}")
                    
                    spy_data = spy_data[(spy_data['datetime'] >= start_dt) & (spy_data['datetime'] <= end_dt)]
                    self._print(f"필터링 후 데이터 크기: {len(spy_data)}개")
                    
                    spy_data.set_index('datetime', inplace=True)
                    self._print(f"날짜 필터링 완료: {start_date} ~ {end_date}")
                else:
                    self._print("datetime 컬럼이 datetime 타입이 아님, 전체 데이터 사용")
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
            # 파일명 변환 (특수문자 처리)
            filename = symbol.lower().replace('^', '').replace('=', '') + '_data.csv'
            macro_path = os.path.join(data_dir, filename)
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
        from actions.calculate_index import StrategyParams
        
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
        
        # 라벨 생성
        self._print("라벨 생성 시작")
        tech_data = self._create_labels(tech_data)
        
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
        """시장 상태 라벨 생성 - 완전히 새로운 Quant 기반 로직"""
        self._print("새로운 Quant 기반 라벨 생성 시작")
        
        # 컬럼명 매핑
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 1. 핵심 지표 계산
        # 수익률 계산 (다양한 기간)
        returns_1d = data[close_col].pct_change()
        returns_5d = data[close_col].pct_change(5)
        returns_20d = data[close_col].pct_change(20)
        
        # 변동성 계산
        volatility_20d = returns_1d.rolling(20).std()
        
        # 2. 트렌드 강도 계산 (ADX 기반)
        trend_strength = np.zeros(len(data))
        if 'adx' in data.columns:
            adx = data['adx']
            trend_strength = np.where(~adx.isna(), adx / 100.0, 0)  # 0-1 범위로 정규화
        
        # 3. 이동평균 기반 트렌드 방향
        trend_direction = np.zeros(len(data))
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            sma_20 = data['sma_20']
            sma_50 = data['sma_50']
            price = data[close_col]
            valid_mask = ~(sma_20.isna() | sma_50.isna())
            
            # 트렌드 방향 (-1: 하락, 0: 중립, 1: 상승)
            trend_direction = np.where(valid_mask,
                np.where((price > sma_20) & (sma_20 > sma_50), 1,  # 강한 상승
                np.where((price < sma_20) & (sma_20 < sma_50), -1,  # 강한 하락
                0)),  # 중립
                0
            )
        
        # 4. RSI 기반 모멘텀
        momentum = np.zeros(len(data))
        if 'rsi' in data.columns:
            rsi = data['rsi']
            momentum = np.where(~rsi.isna(),
                np.where(rsi > 70, -0.5,  # 과매수
                np.where(rsi < 30, 0.5,   # 과매도
                np.where(rsi > 60, -0.2,  # 약한 과매수
                np.where(rsi < 40, 0.2,   # 약한 과매도
                0)))),  # 중립
                0
            )
        
        # 5. 변동성 레벨
        volatility_level = np.zeros(len(data))
        if 'atr_ratio' in data.columns:
            atr_ratio = data['atr_ratio']
            volatility_level = np.where(~atr_ratio.isna(),
                np.where(atr_ratio > 0.03, 1.0,    # 높은 변동성
                np.where(atr_ratio > 0.02, 0.5,    # 중간 변동성
                0)),  # 낮은 변동성
                0
            )
        
        # 6. 거래량 신호
        volume_signal = np.zeros(len(data))
        if 'volume_ratio' in data.columns:
            volume_ratio = data['volume_ratio']
            volume_signal = np.where(~volume_ratio.isna(),
                np.where(volume_ratio > 1.5, 0.3,   # 높은 거래량
                np.where(volume_ratio < 0.5, -0.3,  # 낮은 거래량
                0)),  # 중간 거래량
                0
            )
        
        # 7. 새로운 라벨링 로직 (통계적 분위수 기반)
        labels = np.zeros(len(data), dtype=int)
        
        for i in range(len(data)):
            # 각 지표별 점수 계산
            trend_score = trend_direction[i] * trend_strength[i]
            momentum_score = momentum[i]
            volatility_score = volatility_level[i]
            volume_score = volume_signal[i]
            
            # 최근 수익률 기반 점수
            recent_return_score = 0
            if i >= 20:
                recent_return = returns_20d.iloc[i]
                if not pd.isna(recent_return):
                    if recent_return > 0.05:  # 5% 이상 상승
                        recent_return_score = 0.5
                    elif recent_return < -0.05:  # 5% 이상 하락
                        recent_return_score = -0.5
                    elif recent_return > 0.02:  # 2% 이상 상승
                        recent_return_score = 0.2
                    elif recent_return < -0.02:  # 2% 이상 하락
                        recent_return_score = -0.2
            
            # 종합 점수 계산
            total_score = (trend_score * 0.3 + 
                          momentum_score * 0.2 + 
                          recent_return_score * 0.3 + 
                          volume_score * 0.1 + 
                          volatility_score * 0.1)
            
            # 라벨 분류 (통계적 분위수 기반)
            if i >= 50:  # 충분한 데이터가 있는 경우
                # 전체 점수 분포의 분위수 계산
                all_scores = []
                for j in range(max(0, i-50), i+1):
                    if j < len(data):
                        trend_s = trend_direction[j] * trend_strength[j]
                        momentum_s = momentum[j]
                        vol_s = volume_signal[j]
                        vol_level_s = volatility_level[j]
                        
                        recent_ret_s = 0
                        if j >= 20:
                            recent_ret = returns_20d.iloc[j]
                            if not pd.isna(recent_ret):
                                if recent_ret > 0.05:
                                    recent_ret_s = 0.5
                                elif recent_ret < -0.05:
                                    recent_ret_s = -0.5
                                elif recent_ret > 0.02:
                                    recent_ret_s = 0.2
                                elif recent_ret < -0.02:
                                    recent_ret_s = -0.2
                        
                        score = (trend_s * 0.3 + momentum_s * 0.2 + recent_ret_s * 0.3 + 
                                vol_s * 0.1 + vol_level_s * 0.1)
                        all_scores.append(score)
                
                if len(all_scores) > 0:
                    all_scores = np.array(all_scores)
                    q25 = np.percentile(all_scores, 25)
                    q75 = np.percentile(all_scores, 75)
                    
                    # 분위수 기반 라벨링 (VOLATILE 우선순위 높임)
                    if volatility_score > 0.3:  # 변동성 임계값 낮춤
                        labels[i] = 2  # VOLATILE (높은 변동성 우선)
                    elif total_score > q75:
                        labels[i] = 0  # TRENDING_UP (상위 25%)
                    elif total_score < q25:
                        labels[i] = 1  # TRENDING_DOWN (하위 25%)
                    else:
                        labels[i] = 3  # SIDEWAYS (중간 50%)
                else:
                    labels[i] = 3  # 기본값
            else:
                # 초기 데이터는 기본 라벨링 (VOLATILE 우선순위 높임)
                if volatility_score > 0.3:  # 변동성 임계값 낮춤
                    labels[i] = 2  # VOLATILE (높은 변동성 우선)
                elif total_score > 0.1:
                    labels[i] = 0  # TRENDING_UP
                elif total_score < -0.1:
                    labels[i] = 1  # TRENDING_DOWN
                else:
                    labels[i] = 3  # SIDEWAYS
        
        data['regime_label'] = labels
        
        # 점수 저장 (디버깅용)
        total_scores = []
        for i in range(len(data)):
            trend_score = trend_direction[i] * trend_strength[i]
            momentum_score = momentum[i]
            volatility_score = volatility_level[i]
            volume_score = volume_signal[i]
            
            recent_return_score = 0
            if i >= 20:
                recent_return = returns_20d.iloc[i]
                if not pd.isna(recent_return):
                    if recent_return > 0.05:
                        recent_return_score = 0.5
                    elif recent_return < -0.05:
                        recent_return_score = -0.5
                    elif recent_return > 0.02:
                        recent_return_score = 0.2
                    elif recent_return < -0.02:
                        recent_return_score = -0.2
            
            total_score = (trend_score * 0.3 + 
                          momentum_score * 0.2 + 
                          recent_return_score * 0.3 + 
                          volume_score * 0.1 + 
                          volatility_score * 0.1)
            total_scores.append(total_score)
        
        data['regime_score'] = total_scores
        
        self._print(f"새로운 Quant 기반 라벨 생성 완료: {len(labels)}개")
        
        # 라벨 분포 확인
        label_counts = pd.Series(labels).value_counts().sort_index()
        self._print(f"라벨 분포: {dict(label_counts)}")
        
        # 라벨 불균형 확인
        total_samples = len(labels)
        for label, count in label_counts.items():
            percentage = count / total_samples * 100
            self._print(f"  라벨 {label} ({self.regime_mapping[label]}): {count}개 ({percentage:.1f}%)")
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """특성 준비 - 개선된 버전"""
        self._print("특성 준비 시작")
        self._print(f"데이터 컬럼 수: {len(data.columns)}")
        
        # 기본 특성들 (항상 존재해야 하는 것들)
        base_features = [
            'returns_1d', 'returns_5d', 'returns_20d', 'volatility_20d'
        ]
        
        # 선택적 특성들 (존재하면 사용)
        optional_features = [
            # 이동평균 관련
            'sma_20', 'sma_50', 'sma_ratio', 'price_sma20_ratio', 'price_sma50_ratio',
            
            # 기술적 지표들
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'adx', 'obv',
            
            # 거래량 관련
            'volume', 'volume_ratio', 'volume_price_trend',
            
            # 매크로 지표들 (다양한 명명 규칙 지원)
            'vix_close', '^vix_close', 'vix_close_change', 'vix_close_ratio', 'vix_volatility',
            'tnx_close', '^tnx_close', 'tnx_close_change', 'tnx_close_ratio',
            'tyx_close', '^tyx_close', 'tyx_close_change', 'tyx_close_ratio',
            'dxy_close', '^dxy_close', 'dxy_close_change', 'dxy_close_ratio',
            'gc=f_close', 'gc_close', 'gc_close_change', 'gc_close_ratio',
            'tlt_close', '^tlt_close', 'tlt_close_change', 'tlt_close_ratio',
            'tip_close', '^tip_close', 'tip_close_change', 'tip_close_ratio',
            
            # 복합 특성들
            'macd_volume', 'vix_return_correlation', 'rsi_volatility'
        ]
        
        # 실제 존재하는 특성들 찾기
        available_base = [col for col in base_features if col in data.columns]
        available_optional = [col for col in optional_features if col in data.columns]
        
        # 특성 우선순위 설정 (중요한 특성들을 먼저 포함)
        priority_features = [
            'returns_1d', 'returns_5d', 'returns_20d', 'volatility_20d',
            'rsi', 'macd', 'bb_width', 'volume_ratio'
        ]
        
        # 우선순위 특성들을 먼저 포함
        selected_features = [col for col in priority_features if col in data.columns]
        
        # 나머지 특성들 추가 (중복 제거)
        remaining_features = [col for col in available_base + available_optional 
                            if col not in selected_features]
        selected_features.extend(remaining_features)
        
        self._print(f"선택된 특성 수: {len(selected_features)}")
        self._print(f"선택된 특성들: {selected_features[:10]}...")  # 처음 10개만 출력
        
        # 특성 데이터 준비
        X = data[selected_features].copy()
        
        # 데이터 품질 검사
        self._print(f"특성 데이터 크기: {X.shape}")
        self._print(f"NaN 값 비율: {X.isnull().sum().sum() / (X.shape[0] * X.shape[1]):.2%}")
        
        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # NaN 값 처리 개선
        # 각 특성별로 적절한 방법으로 처리
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                # 시계열 데이터이므로 forward fill 사용
                X[col] = X[col].fillna(method='ffill')
                # 남은 NaN 값은 0으로 채움
                X[col] = X[col].fillna(0)
        
        # 최종 검증
        if X.isnull().sum().sum() > 0:
            self._print("경고: 여전히 NaN 값이 남아있습니다.", level="warning")
            X = X.fillna(0)
        
        self._print(f"최종 특성 데이터 크기: {X.shape}")
        self._print(f"최종 NaN 값 수: {X.isnull().sum().sum()}")
        
        return X, selected_features
    
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
        
        # Random Forest 모델 생성 및 학습 (개선된 하이퍼파라미터)
        self.model = RandomForestClassifier(
            n_estimators=200,  # 트리 수 증가
            max_depth=15,      # 깊이 증가
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',  # 특성 수 제한
            bootstrap=True,
            oob_score=True,    # Out-of-bag 점수 활성화
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # 클래스 불균형 처리
        )
        
        self._print("모델 학습 시작...")
        self.model.fit(X_train_scaled, y_train)
        
        # 모델 성능 평가
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        oob_score = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        
        # 교차 검증
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # 예측 및 상세 평가
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # 분류 보고서
        from sklearn.metrics import classification_report, confusion_matrix
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # 혼동 행렬
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        
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
            'oob_score': oob_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'train_report': train_report,
            'test_report': test_report,
            'train_confusion_matrix': train_cm.tolist(),
            'test_confusion_matrix': test_cm.tolist(),
            'n_samples': len(X),
            'n_features': len(feature_names),
            'class_distribution': y.value_counts().to_dict()
        }
        
        self._print(f"모델 학습 완료:")
        self._print(f"  훈련 정확도: {train_score:.4f}")
        self._print(f"  테스트 정확도: {test_score:.4f}")
        if oob_score:
            self._print(f"  OOB 정확도: {oob_score:.4f}")
        self._print(f"  교차 검증: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 클래스별 성능 출력
        self._print("클래스별 성능 (테스트):")
        for class_label in sorted(y_test.unique()):
            class_name = self.regime_mapping[class_label]
            precision = test_report[str(class_label)]['precision']
            recall = test_report[str(class_label)]['recall']
            f1 = test_report[str(class_label)]['f1-score']
            self._print(f"  {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # 상위 특성 중요도 출력
        self._print("상위 10개 특성 중요도:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            self._print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # 모델 저장 (자동 저장)
        if save_model:
            self.save_model()
        
        return results
    
    def predict_probabilities(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        현재 시장 상태 확률 예측 - 개선된 버전
        
        Args:
            data: 예측할 데이터
            
        Returns:
            시장 상태별 확률
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train_model()을 먼저 실행하세요.")
        
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")
        
        try:
            # 특성 준비
            X, feature_names = self.prepare_features(data)
            
            if X.empty:
                raise ValueError("예측할 데이터가 없습니다.")
            
            # 최신 데이터만 사용
            latest_data = X.iloc[-1:].copy()
            
            # 특성 수 검증
            if len(latest_data.columns) != len(self.feature_names):
                self._print(f"경고: 특성 수 불일치 (예측: {len(latest_data.columns)}, 모델: {len(self.feature_names)})", level="warning")
                
                # 누락된 특성들을 0으로 채움
                missing_features = set(self.feature_names) - set(latest_data.columns)
                for feature in missing_features:
                    latest_data[feature] = 0
                
                # 모델의 특성 순서에 맞춰 재정렬
                latest_data = latest_data[self.feature_names]
            
            # 스케일링
            latest_scaled = self.scaler.transform(latest_data)
            
            # 확률 예측
            probabilities = self.model.predict_proba(latest_scaled)[0]
            
            # 결과 매핑
            result = {}
            for i, prob in enumerate(probabilities):
                if i in self.regime_mapping:
                    regime_name = self.regime_mapping[i].lower()
                    result[regime_name] = float(prob)
                else:
                    self._print(f"경고: 알 수 없는 라벨 {i}", level="warning")
            
            # 확률 합계 검증
            total_prob = sum(result.values())
            if abs(total_prob - 1.0) > 0.01:
                self._print(f"경고: 확률 합계가 1이 아닙니다: {total_prob:.4f}", level="warning")
                # 정규화
                for key in result:
                    result[key] /= total_prob
            
            self._print(f"예측 완료: {result}")
            return result
            
        except Exception as e:
            self._print(f"예측 중 오류 발생: {e}", level="error")
            # 기본값 반환
            default_result = {regime.lower(): 0.25 for regime in self.regime_mapping.values()}
            return default_result
    
    def save_model(self, filepath: str = None):
        """모델 저장 - 개선된 버전"""
        if filepath is None:
            filepath = self.model_dir / "market_regime_rf_model.pkl"
        
        # 모델 저장 전 검증
        if not self.is_trained or self.model is None:
            raise ValueError("학습된 모델이 없습니다. train_model()을 먼저 실행하세요.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'regime_mapping': self.regime_mapping,
            'trained_at': datetime.now().isoformat(),
            'model_version': '1.1',  # 버전 정보 추가
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'model_type': 'RandomForestClassifier'
        }
        
        # 모델 디렉토리 생성
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 안전한 저장 (임시 파일 사용)
        import tempfile
        import shutil
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            joblib.dump(model_data, temp_path)
            shutil.move(temp_path, filepath)
            self._print(f"모델 저장 완료: {filepath}")
            
            # 모델 정보 출력
            self._print(f"  모델 버전: {model_data['model_version']}")
            self._print(f"  특성 수: {model_data['n_features']}")
            self._print(f"  학습 시간: {model_data['trained_at']}")
            
        except Exception as e:
            # 임시 파일 정리
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    def load_model(self, filepath: str = None):
        """모델 로드 - 개선된 버전"""
        if filepath is None:
            filepath = self.model_dir / "market_regime_rf_model.pkl"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            
            # 필수 키 확인
            required_keys = ['model', 'scaler', 'feature_names', 'regime_mapping']
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"모델 파일에 필수 키가 없습니다: {key}")
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.regime_mapping = model_data['regime_mapping']
            self.is_trained = True
            
            self._print(f"모델 로드 완료: {filepath}")
            self._print(f"  모델 버전: {model_data.get('model_version', 'unknown')}")
            self._print(f"  특성 수: {len(self.feature_names)}")
            self._print(f"  학습 시간: {model_data.get('trained_at', 'unknown')}")
            
            # 모델 유효성 검사
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("로드된 모델이 예측 기능을 지원하지 않습니다.")
            
        except Exception as e:
            self._print(f"모델 로드 실패: {e}", level="error")
            self.is_trained = False
            raise e
    
    def get_current_market_probabilities(self, data_dir: str = "data/macro") -> Dict[str, float]:
        """
        현재 시장 상태 확률 계산 (저장된 모델 우선 사용)
        
        Args:
            data_dir: 데이터 디렉토리 경로
            
        Returns:
            현재 시장 상태별 확률
        """
        # 저장된 모델이 있는지 확인하고 로드 시도
        if not self.is_trained:
            try:
                self.load_model()
                self._print("저장된 모델을 로드했습니다.")
            except FileNotFoundError:
                self._print("저장된 모델이 없습니다. 새로 학습을 시작합니다.")
                # 모델 학습
                self.train_model(save_model=True)
        
        # 최근 데이터 로드
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        
        data = self.collect_training_data(start_date, end_date, data_dir)
        
        # 확률 예측
        probabilities = self.predict_probabilities(data)
        
        return probabilities


def main():
    """메인 실행 함수 - 개선된 버전"""
    print("🚀 Random Forest 시장 상태 분류기 시작")
    
    # Random Forest 모델 초기화
    rf_model = MarketRegimeRF(verbose=True)
    
    try:
        # 기존 모델 로드 시도
        print("기존 모델 로드 시도 중...")
        rf_model.load_model()
        print("✅ 기존 모델을 성공적으로 로드했습니다.")
        
        # 모델 정보 출력
        print(f"  - 특성 수: {len(rf_model.feature_names)}")
        print(f"  - 시장 상태 수: {len(rf_model.regime_mapping)}")
        
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"❌ 기존 모델 로드 실패: {e}")
        print("🔄 새로 학습을 시작합니다.")
        
        try:
            # 모델 학습
            print("데이터 수집 및 모델 학습 중...")
            results = rf_model.train_model()
            
            print(f"✅ 모델 학습 완료!")
            print(f"  - 훈련 정확도: {results['train_score']:.4f}")
            print(f"  - 테스트 정확도: {results['test_score']:.4f}")
            if results.get('oob_score'):
                print(f"  - OOB 정확도: {results['oob_score']:.4f}")
            print(f"  - 교차 검증: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            
            # 클래스 분포 출력
            if 'class_distribution' in results:
                print("  - 클래스 분포:")
                for label, count in results['class_distribution'].items():
                    regime_name = rf_model.regime_mapping[label]
                    print(f"    {regime_name}: {count}개")
            
        except Exception as e:
            print(f"❌ 모델 학습 실패: {e}")
            return
    
    # 현재 시장 상태 확률 예측
    try:
        print("\n🔮 현재 시장 상태 예측 중...")
        probabilities = rf_model.get_current_market_probabilities()
        
        print("\n📊 현재 시장 상태 확률 (ML 기반):")
        # 확률 순으로 정렬
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for regime, prob in sorted_probs:
            percentage = prob * 100
            if percentage > 50:
                print(f"  🎯 {regime.upper()}: {percentage:.1f}% (주요 상태)")
            elif percentage > 25:
                print(f"  📈 {regime.upper()}: {percentage:.1f}% (보조 상태)")
            else:
                print(f"  📊 {regime.upper()}: {percentage:.1f}%")
        
        # 최고 확률 상태 출력
        max_regime, max_prob = max(probabilities.items(), key=lambda x: x[1])
        print(f"\n🏆 현재 주요 시장 상태: {max_regime.upper()} ({max_prob:.1%})")
        
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        print("기본 확률 반환:")
        for regime in ['trending_up', 'trending_down', 'volatile', 'sideways']:
            print(f"  {regime.upper()}: 25.0%")
    
    print("\n✅ Random Forest 시장 상태 분류 완료!")


if __name__ == "__main__":
    main()
