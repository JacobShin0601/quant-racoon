#!/usr/bin/env python3
"""
시장 환경 분류기 (Market Sensor)
통합 시장 분석 시스템 - 실행 인터페이스
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import os
import optuna
from dataclasses import dataclass
from enum import Enum
import uuid

from ..actions.global_macro import (
    GlobalMacroDataCollector, 
    HyperparamTuner, 
    MacroSectorAnalyzer,
    MarketRegime, 
    MarketCondition, 
    SectorStrength,
    MarketClassification,
    MacroAnalysis
)
from ..actions.random_forest import MarketRegimeRF


class MarketSensor:
    """통합 시장 분석 시스템 - 실행 인터페이스"""
    
    def __init__(self, data_dir: str = "data/macro", config_path: str = "config/config_macro.json"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        
        # 세션 UUID 생성
        self.session_uuid = str(uuid.uuid4())
        self.logger.info(f"MarketSensor 초기화 - Session UUID: {self.session_uuid}")
        
        # 핵심 컴포넌트들 초기화 (UUID 전달)
        self.macro_collector = GlobalMacroDataCollector(self.session_uuid)
        self.hyperparam_tuner = HyperparamTuner(config_path, self.session_uuid)
        self.macro_analyzer = MacroSectorAnalyzer(data_dir, self.session_uuid)
        
        # Random Forest 모델 초기화
        self.rf_model = MarketRegimeRF(verbose=True)
        
        # 최적 파라미터 (백테스팅으로 찾은 값)
        self.optimal_params = None
        
    def load_macro_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """매크로 데이터 로드"""
        try:
            # pandas 경고 억제
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
            warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
            
            # SPY 데이터 로드
            spy_path = f"{self.data_dir}/spy_data.csv"
            if os.path.exists(spy_path):
                spy_data = pd.read_csv(spy_path, index_col=0, parse_dates=False)
                # datetime 컬럼을 인덱스로 설정
                if 'datetime' in spy_data.columns:
                    spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], utc=True)
                    spy_data.set_index('datetime', inplace=True)
            else:
                self.logger.warning("SPY 데이터가 없습니다. 새로 수집합니다.")
                return self._collect_fresh_data()
            
            # 매크로 지표 로드
            macro_data = {}
            for symbol in self.macro_collector.macro_symbols.keys():
                macro_path = f"{self.data_dir}/{symbol.lower()}_data.csv"
                if os.path.exists(macro_path):
                    df = pd.read_csv(macro_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    macro_data[symbol] = df
            
            # 섹터 데이터 로드
            sector_data = {}
            for symbol in self.macro_collector.sector_etfs.keys():
                sector_path = f"{self.data_dir}/{symbol.lower()}_sector.csv"
                if os.path.exists(sector_path):
                    df = pd.read_csv(sector_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    sector_data[symbol] = df
            
            self.logger.info(f"매크로 데이터 로드 완료: SPY({len(spy_data)}), 매크로({len(macro_data)}), 섹터({len(sector_data)})")
            return spy_data, macro_data, sector_data
            
        except Exception as e:
            self.logger.error(f"매크로 데이터 로드 중 오류: {e}")
            return pd.DataFrame(), {}, {}
    
    def load_macro_data_only(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """저장된 매크로 데이터만 로드 (다운로드 없음)"""
        try:
            # pandas 경고 억제
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
            warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
            
            # SPY 데이터 로드
            spy_path = f"{self.data_dir}/spy_data.csv"
            if os.path.exists(spy_path):
                spy_data = pd.read_csv(spy_path, index_col=0, parse_dates=False)
                # datetime 컬럼을 인덱스로 설정
                if 'datetime' in spy_data.columns:
                    spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], utc=True)
                    spy_data.set_index('datetime', inplace=True)
            else:
                self.logger.warning("SPY 데이터가 없습니다.")
                return pd.DataFrame(), {}, {}
            
            # 매크로 지표 로드
            macro_data = {}
            for symbol in self.macro_collector.macro_symbols.keys():
                macro_path = f"{self.data_dir}/{symbol.lower()}_data.csv"
                if os.path.exists(macro_path):
                    df = pd.read_csv(macro_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    macro_data[symbol] = df
            
            # 섹터 데이터 로드
            sector_data = {}
            for symbol in self.macro_collector.sector_etfs.keys():
                sector_path = f"{self.data_dir}/{symbol.lower()}_sector.csv"
                if os.path.exists(sector_path):
                    df = pd.read_csv(sector_path, index_col=0, parse_dates=False)
                    # datetime 컬럼을 인덱스로 설정
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                        df.set_index('datetime', inplace=True)
                    sector_data[symbol] = df
            
            self.logger.info(f"저장된 매크로 데이터 로드 완료: SPY({len(spy_data)}), 매크로({len(macro_data)}), 섹터({len(sector_data)})")
            return spy_data, macro_data, sector_data
            
        except Exception as e:
            self.logger.error(f"매크로 데이터 로드 중 오류: {e}")
            return pd.DataFrame(), {}, {}
    
    def _collect_fresh_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """새로운 매크로 데이터 수집"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        spy_data = self.macro_collector.collect_spy_data(start_date, end_date)
        macro_data = self.macro_collector.collect_macro_indicators(start_date, end_date)
        sector_data = self.macro_collector.collect_sector_data(start_date, end_date)
        
        self.macro_collector.save_macro_data(spy_data, macro_data, sector_data, self.data_dir, start_date, end_date)
        
        return spy_data, macro_data, sector_data
    
    def get_macro_sector_analysis(self, start_date: str = None, end_date: str = None) -> MacroAnalysis:
        """매크로 & 섹터 분석 - MacroSectorAnalyzer 위임"""
        return self.macro_analyzer.get_comprehensive_analysis(start_date, end_date)
    
    def optimize_hyperparameters_optuna(self, start_date: str, end_date: str, n_trials: int = None) -> Dict[str, Any]:
        """하이퍼파라미터 최적화 - HyperparamTuner 위임"""
        return self.hyperparam_tuner.optimize_hyperparameters(start_date, end_date, n_trials)
    
    def save_optimization_results(self, results: Dict[str, Any], output_dir: str = "results/market_sensor_optimization"):
        """최적화 결과 저장 - HyperparamTuner 위임"""
        self.hyperparam_tuner.save_results(results, output_dir)
        
        # MarketSensor 전용 메타데이터 추가
        try:
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            market_sensor_metadata = {
                'session_uuid': self.session_uuid,
                'execution_type': 'market_sensor_optimization',
                'created_at': datetime.now().isoformat(),
                'data_dir': self.data_dir,
                'optimization_completed': True
            }
            
            with open(f"{session_dir}/market_sensor_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(market_sensor_metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"MarketSensor 메타데이터 저장 중 오류: {e}")
    
    def save_macro_analysis_results(self, analysis: MacroAnalysis, output_dir: str = "results/macro_sector_analysis"):
        """매크로 분석 결과 저장 - MacroSectorAnalyzer 위임"""
        self.macro_analyzer.save_analysis_results(analysis, output_dir)
        
        # MarketSensor 전용 메타데이터 추가
        try:
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            market_sensor_metadata = {
                'session_uuid': self.session_uuid,
                'execution_type': 'market_sensor_macro_analysis',
                'created_at': datetime.now().isoformat(),
                'data_dir': self.data_dir,
                'analysis_completed': True
            }
            
            with open(f"{session_dir}/market_sensor_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(market_sensor_metadata, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"MarketSensor 메타데이터 저장 중 오류: {e}")
    
    def load_optimal_params(self, input_path: str = "results/market_sensor_optimization/best_params.json") -> Dict[str, Any]:
        """최적 파라미터 로드"""
        try:
            if os.path.exists(input_path):
                with open(input_path, 'r', encoding='utf-8') as f:
                    params_data = json.load(f)
                
                # best_params.json은 직접 파라미터 딕셔너리이므로 optimal_params 키가 없음
                self.optimal_params = params_data
                self.logger.info(f"최적 파라미터 로드 완료: {input_path}")
                return self.optimal_params
            else:
                self.logger.warning(f"최적 파라미터 파일이 없습니다: {input_path}")
                return {}
                
        except Exception as e:
            self.logger.error(f"최적 파라미터 로드 중 오류: {e}")
            return {}
    
    def get_current_market_analysis(self, use_optimized_params: bool = True) -> Dict[str, Any]:
        """현재 시장 분석 결과 반환 (기술적 + 매크로 종합 분석)"""
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if spy_data.empty:
                return {'error': 'SPY 데이터를 로드할 수 없습니다.'}
            
            # 1. 기술적 분석 (기존)
            if use_optimized_params and self.optimal_params:
                params = self.optimal_params
            else:
                params = {
                    'sma_short': 20, 'sma_long': 50, 'rsi_period': 14,
                    'rsi_overbought': 70, 'rsi_oversold': 30, 'atr_period': 14,
                    'trend_weight': 0.4, 'momentum_weight': 0.3,
                    'volatility_weight': 0.2, 'macro_weight': 0.1,
                    'base_position': 0.8, 'trending_boost': 1.2, 'volatile_reduction': 0.5
                }
            
            # 파생 변수 계산
            data_with_features = self.hyperparam_tuner._calculate_derived_features(spy_data, params)
            
            # 매크로 데이터 병합
            if '^VIX' in macro_data:
                vix_df = macro_data['^VIX']
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 시장 상태 분류
            regime = self.hyperparam_tuner._classify_market_regime(data_with_features, params)
            current_regime = regime.iloc[-1]
            
            # 전략 수익률 계산
            strategy_returns = self.hyperparam_tuner._calculate_strategy_returns(data_with_features, regime, params)
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            buy_hold_returns = spy_data[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self.hyperparam_tuner._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            # 2. 매크로 분석 (새로 추가)
            macro_analysis = None
            sector_analysis = None
            
            if macro_data and sector_data:
                try:
                    # 매크로 환경 분석
                    macro_analysis = self.macro_analyzer.analyze_macro_environment(macro_data)
                    
                    # 섹터 로테이션 분석
                    sector_analysis = self.macro_analyzer.analyze_sector_rotation(sector_data)
                    
                    # 시장 조건 분류
                    market_condition = self.macro_analyzer.classify_market_condition(macro_analysis)
                    
                except Exception as e:
                    self.logger.warning(f"매크로 분석 중 오류: {e}")
            
            # 3. 종합 전략 추천
            recommendation = self.recommend_strategy(MarketClassification(
                regime=MarketRegime(current_regime),
                confidence=0.8,
                features={},
                timestamp=datetime.now(),
                metadata={}
            ))
            
            # 매크로 분석 결과가 있으면 전략에 반영
            if macro_analysis and sector_analysis:
                recommendation = self._enhance_recommendation_with_macro(
                    recommendation, macro_analysis, sector_analysis
                )
            
            return {
                'current_regime': current_regime,
                'performance_metrics': metrics,
                'recommendation': recommendation,
                'macro_analysis': macro_analysis,
                'sector_analysis': sector_analysis,
                'last_update': datetime.now().isoformat(),
                'data_period': f"{spy_data.index[0].strftime('%Y-%m-%d')} ~ {spy_data.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            self.logger.error(f"현재 시장 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def _enhance_recommendation_with_macro(self, base_recommendation: Dict[str, Any], 
                                         macro_analysis: Dict[str, Any], 
                                         sector_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """매크로 분석 결과를 기반으로 전략 추천 강화"""
        enhanced_recommendation = base_recommendation.copy()
        
        # 매크로 환경에 따른 포지션 크기 조정
        if 'inflation_risk' in macro_analysis:
            inflation_risk = macro_analysis['inflation_risk']
            if inflation_risk > 0.7:  # 높은 인플레이션 위험
                enhanced_recommendation['position_size'] *= 0.8
                enhanced_recommendation['description'] += " (인플레이션 위험으로 인한 포지션 축소)"
            elif inflation_risk < 0.3:  # 낮은 인플레이션 위험
                enhanced_recommendation['position_size'] *= 1.1
                enhanced_recommendation['description'] += " (낮은 인플레이션으로 인한 포지션 확대)"
        
        # 금리 환경에 따른 조정
        if 'rate_environment' in macro_analysis:
            rate_env = macro_analysis['rate_environment']
            if rate_env == 'high_rates':
                enhanced_recommendation['stop_loss'] *= 1.2  # 손절폭 확대
                enhanced_recommendation['description'] += " (고금리 환경으로 인한 손절폭 확대)"
            elif rate_env == 'low_rates':
                enhanced_recommendation['take_profit'] *= 1.1  # 익절폭 확대
                enhanced_recommendation['description'] += " (저금리 환경으로 인한 익절폭 확대)"
        
        # 섹터 추천 추가
        if sector_analysis:
            leading_sectors = [sector for sector, strength in sector_analysis.items() 
                             if strength == SectorStrength.LEADING]
            defensive_sectors = [sector for sector, strength in sector_analysis.items() 
                               if strength == SectorStrength.DEFENSIVE]
            
            if leading_sectors:
                enhanced_recommendation['leading_sectors'] = leading_sectors
            if defensive_sectors:
                enhanced_recommendation['defensive_sectors'] = defensive_sectors
        
        return enhanced_recommendation
    
    def get_comprehensive_macro_analysis(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """종합 매크로 분석 (MacroSectorAnalyzer 활용)"""
        try:
            analysis = self.macro_analyzer.get_comprehensive_analysis(start_date, end_date)
            
            if analysis is None:
                return {'error': '매크로 분석을 수행할 수 없습니다.'}
            
            return {
                'market_condition': analysis.market_condition.value,
                'confidence': analysis.confidence,
                'key_indicators': analysis.key_indicators,
                'sector_rotation': {sector: strength.value for sector, strength in analysis.sector_rotation.items()},
                'recommendations': analysis.recommendations,
                'timestamp': analysis.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"종합 매크로 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def get_technical_analysis(self, use_optimized_params: bool = True, use_ml_model: bool = False) -> Dict[str, Any]:
        """기술적 분석 (시장 상태 분류 + 확률 분석)"""
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if spy_data.empty:
                return {'error': 'SPY 데이터를 로드할 수 없습니다.'}
            
            # 파라미터 선택
            if use_optimized_params and self.optimal_params:
                params = self.optimal_params
            else:
                params = {
                    'sma_short': 20, 'sma_long': 50, 'rsi_period': 14,
                    'rsi_overbought': 70, 'rsi_oversold': 30, 'atr_period': 14,
                    'trend_weight': 0.4, 'momentum_weight': 0.3,
                    'volatility_weight': 0.2, 'macro_weight': 0.1,
                    'base_position': 0.8, 'trending_boost': 1.2, 'volatile_reduction': 0.5
                }
            
            # 파생 변수 계산
            data_with_features = self.hyperparam_tuner._calculate_derived_features(spy_data, params)
            
            # 매크로 데이터 병합 (VIX 등)
            if '^VIX' in macro_data:
                vix_df = macro_data['^VIX']
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 시장 상태 분류 (두 가지 방식 지원)
            if use_ml_model:
                # ML 기반 분석 (Random Forest)
                try:
                    # 기존 모델 로드 시도
                    self.rf_model.load_model()
                    self.logger.info("ML 모델 로드 완료")
                except FileNotFoundError:
                    self.logger.warning("ML 모델이 없습니다. Quant 기반 분석을 사용합니다.")
                    use_ml_model = False
            
            if use_ml_model:
                # ML 기반 확률 계산
                current_probabilities = self.rf_model.get_current_market_probabilities(data_dir=self.data_dir)
                analysis_method = "ML (Random Forest)"
                # ML 모델에서는 regime을 확률이 가장 높은 것으로 설정
                regime = None  # 나중에 설정됨
            else:
                # Quant 기반 분석 (기존 방식)
                regime_analysis = self.hyperparam_tuner._classify_market_regime_with_probabilities(data_with_features, params)
                regime = regime_analysis['regime']
                probabilities = regime_analysis['probabilities']
                
                current_probabilities = {
                    'trending_up': float(probabilities['trending_up'][-1]),
                    'trending_down': float(probabilities['trending_down'][-1]),
                    'volatile': float(probabilities['volatile'][-1]),
                    'sideways': float(probabilities['sideways'][-1])
                }
                analysis_method = "Quant (Rule-based)"
            
            # 확률 순서대로 정렬하여 시장 상태와 두 번째 가능성 결정
            sorted_probs = sorted(current_probabilities.items(), key=lambda x: x[1], reverse=True)
            primary_regime = sorted_probs[0][0]
            secondary_regime = sorted_probs[1][0] if len(sorted_probs) > 1 else None
            secondary_probability = sorted_probs[1][1] if len(sorted_probs) > 1 else 0
            
            # 시장 상태를 확률이 가장 높은 상태로 설정 (일관성 유지)
            current_regime = primary_regime.upper()
            
            # ML 모델을 사용하는 경우 regime 변수 설정
            if use_ml_model and regime is None:
                # ML 모델에서는 regime을 시리즈로 변환 (모든 행에 동일한 값)
                regime = pd.Series([current_regime] * len(data_with_features), index=data_with_features.index)
            
            # 전략 수익률 계산
            strategy_returns = self.hyperparam_tuner._calculate_strategy_returns(data_with_features, regime, params)
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            buy_hold_returns = spy_data[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self.hyperparam_tuner._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            # 전략 추천
            recommendation = self.recommend_strategy(MarketClassification(
                regime=MarketRegime(current_regime),
                confidence=current_probabilities.get(primary_regime, 0.8),
                features=current_probabilities,
                timestamp=datetime.now(),
                metadata={'secondary_regime': secondary_regime, 'secondary_probability': secondary_probability}
            ))
            
            return {
                'analysis_type': 'technical',
                'analysis_method': analysis_method,
                'current_regime': current_regime,
                'regime_probabilities': current_probabilities,
                'primary_regime': primary_regime,
                'secondary_regime': secondary_regime,
                'secondary_probability': secondary_probability,
                'performance_metrics': metrics,
                'recommendation': recommendation,
                'technical_indicators': {
                    'rsi': data_with_features['rsi'].iloc[-1] if 'rsi' in data_with_features.columns else None,
                    'macd': data_with_features['macd'].iloc[-1] if 'macd' in data_with_features.columns else None,
                    'sma_short': data_with_features[f'sma_{params.get("sma_short", 20)}'].iloc[-1] if f'sma_{params.get("sma_short", 20)}' in data_with_features.columns else None,
                    'sma_long': data_with_features[f'sma_{params.get("sma_long", 50)}'].iloc[-1] if f'sma_{params.get("sma_long", 50)}' in data_with_features.columns else None,
                    'atr': data_with_features['atr'].iloc[-1] if 'atr' in data_with_features.columns else None,
                },
                'last_update': datetime.now().isoformat(),
                'data_period': f"{spy_data.index[0].strftime('%Y-%m-%d')} ~ {spy_data.index[-1].strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            self.logger.error(f"기술적 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def get_macro_analysis_only(self) -> Dict[str, Any]:
        """매크로 분석만 수행 (VIX, TIPS 등 상세 분석 포함)"""
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if not macro_data:
                return {'error': '매크로 데이터를 로드할 수 없습니다.'}
            
            # 매크로 환경 분석
            macro_analysis = self.macro_analyzer.analyze_macro_environment(macro_data)
            
            # 시장 조건 분류
            market_condition = self.macro_analyzer.classify_market_condition(macro_analysis)
            
            # VIX 상세 분석
            vix_analysis = {}
            if '^VIX' in macro_data:
                vix_data = macro_data['^VIX']
                close_col = 'close' if 'close' in vix_data.columns else 'Close'
                if close_col in vix_data.columns:
                    vix_series = vix_data[close_col]
                    # 52주 백분위 계산 (NaN 방지)
                    rolling_max = vix_series.rolling(252).max()
                    percentile_52w = 0.0
                    if not rolling_max.isna().all() and rolling_max.iloc[-1] > 0:
                        percentile_52w = float((vix_series.iloc[-1] / rolling_max.iloc[-1]) * 100)
                    
                    # 20일 평균 계산 (NaN 방지)
                    ma_20 = vix_series.rolling(20).mean().iloc[-1]
                    if pd.isna(ma_20):
                        ma_20 = vix_series.iloc[-1]  # 평균이 NaN이면 현재값 사용
                    
                    vix_analysis = {
                        'current_level': float(vix_series.iloc[-1]),
                        'ma_20': float(ma_20),
                        'percentile_52w': percentile_52w,
                        'volatility_regime': 'high' if vix_series.iloc[-1] > 25 else 'normal',
                        'trend': 'increasing' if vix_series.iloc[-1] > ma_20 else 'decreasing'
                    }
            
            # TIPS Spread 상세 분석
            tips_analysis = {}
            tips_spread_indicators = [key for key in macro_analysis.keys() if 'tips_spread' in key]
            if tips_spread_indicators:
                tips_analysis = {
                    'composite_spread': macro_analysis.get('tips_spread_composite', 0),
                    'composite_ma_50': macro_analysis.get('tips_spread_composite_ma_50', 0),
                    'inflation_expectation': macro_analysis.get('inflation_expectation', 'unknown'),
                    'inflation_trend': macro_analysis.get('inflation_trend', 'unknown'),
                    'tip_tlt_ratio': macro_analysis.get('tips_spread_tip_tlt', 0),
                    'short_long_ratio': macro_analysis.get('tips_spread_short_long', 0),
                    'schp_tlt_ratio': macro_analysis.get('tips_spread_schp_tlt', 0)
                }
            
            # 국채 스프레드 분석
            yield_analysis = {}
            if 'yield_spread' in macro_analysis:
                yield_analysis = {
                    'current_spread': macro_analysis.get('yield_spread', 0),
                    'spread_ma_20': macro_analysis.get('spread_ma_20', 0),
                    'recession_risk': macro_analysis.get('recession_risk', 'unknown'),
                    'spread_trend': 'inverting' if macro_analysis.get('yield_spread', 0) < macro_analysis.get('spread_ma_20', 0) else 'normal'
                }
            
            # 달러 강도 분석
            dollar_analysis = {}
            if 'dollar_strength' in macro_analysis:
                dollar_analysis = {
                    'current_level': macro_analysis.get('dollar_strength', 0),
                    'ma_50': macro_analysis.get('dollar_ma_50', 0),
                    'trend': macro_analysis.get('dollar_trend', 'unknown'),
                    'strength_level': 'strong' if macro_analysis.get('dollar_strength', 0) > macro_analysis.get('dollar_ma_50', 0) else 'weak'
                }
            
            # 금 가격 분석
            gold_analysis = {}
            if 'gold_price' in macro_analysis:
                gold_analysis = {
                    'current_price': macro_analysis.get('gold_price', 0),
                    'ma_50': macro_analysis.get('gold_ma_50', 0),
                    'trend': macro_analysis.get('gold_trend', 'unknown'),
                    'safe_haven_demand': 'high' if macro_analysis.get('gold_trend', 'unknown') == 'bullish' else 'low'
                }
            
            # 국채 가격 분석
            bond_analysis = {}
            if 'bond_price' in macro_analysis:
                bond_analysis = {
                    'current_price': macro_analysis.get('bond_price', 0),
                    'ma_50': macro_analysis.get('bond_ma_50', 0),
                    'trend': macro_analysis.get('bond_trend', 'unknown'),
                    'flight_to_quality': 'high' if macro_analysis.get('bond_trend', 'unknown') == 'bullish' else 'low'
                }
            
            return {
                'analysis_type': 'macro',
                'market_condition': market_condition.value,
                'macro_analysis': macro_analysis,
                'vix_analysis': vix_analysis,
                'tips_analysis': tips_analysis,
                'yield_analysis': yield_analysis,
                'dollar_analysis': dollar_analysis,
                'gold_analysis': gold_analysis,
                'bond_analysis': bond_analysis,
                'key_indicators': {
                    'inflation_risk': macro_analysis.get('inflation_risk', 0),
                    'rate_environment': macro_analysis.get('rate_environment', 'unknown'),
                    'growth_outlook': macro_analysis.get('growth_outlook', 'unknown'),
                    'vix_level': macro_analysis.get('vix_current', 0),
                    'yield_curve': macro_analysis.get('yield_spread', 0)
                },
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"매크로 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def get_sector_analysis_only(self) -> Dict[str, Any]:
        """섹터 분석만 수행"""
        try:
            # 데이터 로드
            spy_data, macro_data, sector_data = self.load_macro_data()
            
            if not sector_data:
                return {'error': '섹터 데이터를 로드할 수 없습니다.'}
            
            # 섹터 로테이션 분석
            sector_analysis = self.macro_analyzer.analyze_sector_rotation(sector_data)
            
            # 섹터별 분류
            leading_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.LEADING]
            lagging_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.LAGGING]
            defensive_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.DEFENSIVE]
            cyclical_sectors = [sector for sector, strength in sector_analysis.items() if strength == SectorStrength.CYCLICAL]
            
            return {
                'analysis_type': 'sector',
                'sector_analysis': {sector: strength.value for sector, strength in sector_analysis.items()},
                'sector_categories': {
                    'leading': leading_sectors,
                    'lagging': lagging_sectors,
                    'defensive': defensive_sectors,
                    'cyclical': cyclical_sectors
                },
                'recommendations': {
                    'overweight': leading_sectors + defensive_sectors,
                    'underweight': lagging_sectors,
                    'neutral': cyclical_sectors
                },
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"섹터 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def run_analysis_by_type(self, analysis_type: str, use_optimized_params: bool = True, use_ml_model: bool = False) -> Dict[str, Any]:
        """분석 유형에 따른 분석 실행"""
        if analysis_type == 'technical':
            return self.get_technical_analysis(use_optimized_params, use_ml_model)
        elif analysis_type == 'macro':
            return self.get_macro_analysis_only()
        elif analysis_type == 'sector':
            return self.get_sector_analysis_only()
        elif analysis_type == 'comprehensive':
            return self.get_current_market_analysis(use_optimized_params)
        elif analysis_type == 'all':
            # 모든 분석 수행 (Quant 기반과 ML 기반 모두)
            results = {
                'technical_quant': self.get_technical_analysis(use_optimized_params, use_ml_model=False),
                'technical_ml': self.get_technical_analysis(use_optimized_params, use_ml_model=True),
                'macro': self.get_macro_analysis_only(),
                'sector': self.get_sector_analysis_only(),
                'comprehensive': self.get_current_market_analysis(use_optimized_params),
                'timestamp': datetime.now().isoformat()
            }
            return results
        else:
            return {'error': f'지원하지 않는 분석 유형: {analysis_type}'}
    
    def recommend_strategy(self, classification: MarketClassification) -> Dict[str, Any]:
        """분류 결과에 따른 전략 추천"""
        recommendations = {
            MarketRegime.TRENDING_UP: {
                'primary_strategy': 'momentum_following',
                'secondary_strategy': 'buy_hold',
                'position_size': 1.0,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'description': '상승 추세 - 모멘텀 추종 전략 권장'
            },
            MarketRegime.TRENDING_DOWN: {
                'primary_strategy': 'cash_heavy',
                'secondary_strategy': 'inverse_momentum',
                'position_size': 0.3,
                'stop_loss': 0.03,
                'take_profit': 0.08,
                'description': '하락 추세 - 현금 비중 확대 권장'
            },
            MarketRegime.SIDEWAYS: {
                'primary_strategy': 'swing_trading',
                'secondary_strategy': 'mean_reversion',
                'position_size': 0.7,
                'stop_loss': 0.04,
                'take_profit': 0.10,
                'description': '횡보장 - 스윙 트레이딩 전략 권장'
            },
            MarketRegime.VOLATILE: {
                'primary_strategy': 'reduced_position',
                'secondary_strategy': 'volatility_breakout',
                'position_size': 0.5,
                'stop_loss': 0.06,
                'take_profit': 0.12,
                'description': '변동성 높음 - 포지션 크기 축소 권장'
            },
            MarketRegime.UNCERTAIN: {
                'primary_strategy': 'wait_and_watch',
                'secondary_strategy': 'minimal_position',
                'position_size': 0.2,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'description': '불확실 - 관망 후 진입 권장'
            }
        }
        
        return recommendations[classification.regime]
    
    def generate_transaction_log(self, results: Dict[str, Any], output_dir: str = "log/market_sensor") -> str:
        """거래 로그 생성 (swing 로그와 유사한 형식)"""
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 로그 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"transaction_market_sensor_{timestamp}.log"
            log_path = os.path.join(output_dir, log_filename)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("=== MARKET SENSOR 거래 내역 로그 ===\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"실행 UUID: {self.session_uuid}\n")
                f.write("=" * 80 + "\n\n")
                
                # SPY 거래 내역
                f.write("📊 SPY (market_sensor)\n")
                f.write("-" * 50 + "\n")
                
                if 'test_performance' in results and results['test_performance']:
                    test_metrics = results['test_performance']
                    
                    # 거래 통계
                    total_return = test_metrics.get('total_return', 0)
                    sharpe_ratio = test_metrics.get('sharpe_ratio', 0)
                    max_drawdown = test_metrics.get('max_drawdown', 0)
                    win_rate = test_metrics.get('win_rate', 0)
                    
                    f.write(f"총 거래 수: {test_metrics.get('total_trades', 0)}\n")
                    f.write(f"수익률: {total_return:.2%}\n")
                    f.write(f"샤프 비율: {sharpe_ratio:.3f}\n")
                    f.write(f"최대 낙폭: {max_drawdown:.2%}\n")
                    f.write(f"승률: {win_rate:.2%}\n\n")
                    
                    # 거래 내역 (시뮬레이션된 거래)
                    f.write("거래 내역:\n")
                    f.write("날짜                   시간         타입     가격         수량       수익률        누적수익률       \n")
                    f.write("-" * 80 + "\n")
                    
                    # 시뮬레이션된 거래 내역 생성
                    if 'strategy_returns' in results:
                        strategy_returns = results['strategy_returns']
                        if isinstance(strategy_returns, pd.Series):
                            # 거래 시뮬레이션
                            trades = self._simulate_trades_from_returns(strategy_returns)
                            
                            for i, trade in enumerate(trades):
                                date_str = trade['date'].strftime('%Y-%m-%d')
                                time_str = trade['time'] if 'time' in trade else ''
                                trade_type = trade['type']
                                price = trade['price']
                                quantity = trade['quantity']
                                profit_rate = trade.get('profit_rate', '')
                                cumulative_profit = trade.get('cumulative_profit', '')
                                
                                f.write(f"{date_str:<20} {time_str:<12} {trade_type:<10} {price:<10.2f} {quantity:<10.2f} {profit_rate:<10} {cumulative_profit:<10}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("📋 거래 요약\n")
                f.write("-" * 50 + "\n")
                
                if 'best_params' in results:
                    best_params = results['best_params']
                    f.write("최적 파라미터:\n")
                    for param, value in best_params.items():
                        if isinstance(value, float):
                            f.write(f"  {param}: {value:.4f}\n")
                        else:
                            f.write(f"  {param}: {value}\n")
                
                f.write(f"\n실험 ID: {self.session_uuid}\n")
                f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            return log_path
            
        except Exception as e:
            self.logger.error(f"거래 로그 생성 중 오류: {e}")
            return ""
    
    def _simulate_trades_from_returns(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """수익률 시리즈에서 거래 내역 시뮬레이션"""
        trades = []
        cumulative_return = 0
        position = 0
        entry_price = 100  # 초기 가격
        
        for date, daily_return in returns.items():
            if abs(daily_return) > 0.01:  # 1% 이상 변동 시 거래로 간주
                if position == 0 and daily_return > 0:
                    # 매수
                    position = 1
                    entry_price = 100 * (1 + cumulative_return)
                    trades.append({
                        'date': date,
                        'type': '매수',
                        'price': entry_price,
                        'quantity': 1.0,
                        'profit_rate': '',
                        'cumulative_profit': ''
                    })
                elif position == 1 and daily_return < -0.005:
                    # 매도
                    position = 0
                    exit_price = entry_price * (1 + daily_return)
                    profit_rate = (exit_price - entry_price) / entry_price
                    cumulative_return += profit_rate
                    
                    trades.append({
                        'date': date,
                        'type': '매도',
                        'price': exit_price,
                        'quantity': 1.0,
                        'profit_rate': f"{profit_rate:.2f} %",
                        'cumulative_profit': f"{cumulative_return:.2f} %"
                    })
        
        return trades
    

def main():
    """실험 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Sensor - 통합 시장 분석 시스템')
    parser.add_argument('--mode', choices=['experiment', 'analyze', 'optimize', 'collect', 'macro_sector', 'comprehensive'], default='experiment',
                       help='실행 모드: experiment (종합 실험), analyze (기본 분석), optimize (하이퍼파라미터 튜닝), collect (데이터 수집), macro_sector (매크로&섹터 분석), comprehensive (기술적+매크로 종합 분석)')
    parser.add_argument('--analysis', choices=['technical', 'macro', 'sector', 'comprehensive', 'all'], 
                       help='분석 유형: technical (기술적 분석), macro (매크로 분석), sector (섹터 분석), comprehensive (종합 분석), all (모든 분석)')
    parser.add_argument('--start_date', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna 시도 횟수')
    parser.add_argument('--use_optimized', action='store_true', help='최적화된 파라미터 사용')
    parser.add_argument('--use_ml_model', action='store_true', help='ML 모델 사용 (Random Forest)')
    parser.add_argument('--train_ml_model', action='store_true', help='ML 모델 학습')
    parser.add_argument('--save_results', action='store_true', help='결과 저장')
    parser.add_argument('--download_data', action='store_true', help='새로운 데이터 다운로드')
    parser.add_argument('--force_download', action='store_true', help='기존 데이터 무시하고 강제 다운로드')
    parser.add_argument('--use_saved_data', action='store_true', help='저장된 데이터만 사용 (새로운 데이터 수집 안함)')
    
    args = parser.parse_args()
    
    # 기본 날짜 설정
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
        
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2년치 데이터
    else:
        start_date = args.start_date
    
    print(f"🎯 Market Sensor 시작: {start_date} ~ {end_date}")
    
    sensor = MarketSensor()
    print(f"🆔 Session UUID: {sensor.session_uuid}")
    
    # --analysis argument가 있으면 분석 모드로 실행
    if args.analysis:
        print(f"🔍 분석 모드: {args.analysis}")
        print("=" * 60)
        
        try:
            # 데이터 준비
            if args.use_saved_data:
                print("📂 저장된 데이터만 사용 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data_only()
                if spy_data.empty:
                    print("❌ 저장된 데이터가 없습니다. --use_saved_data 옵션을 제거하고 다시 실행하세요.")
                    return
            else:
                print("📂 기존 데이터 로드 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data()
                if spy_data.empty:
                    print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                    spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            
            print(f"✅ 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
            
            # 분석 모드에서는 기본적으로 최적화된 파라미터 사용
            use_optimized = args.use_optimized if args.use_optimized is not None else True
            if use_optimized:
                print("🔧 최적화된 파라미터 로드 중...")
                sensor.load_optimal_params()
                print("✅ 최적화된 파라미터 로드 완료")
            
            # ML 모델 학습 (요청된 경우)
            if args.train_ml_model:
                print("🤖 ML 모델 학습 시작...")
                try:
                    # 최적화된 파라미터 사용 (있는 경우)
                    training_params = None
                    if use_optimized and sensor.optimal_params:
                        training_params = sensor.optimal_params
                        print("  최적화된 파라미터로 라벨 생성")
                    else:
                        print("  기본 파라미터로 라벨 생성")
                    
                    training_results = sensor.rf_model.train_model(params=training_params)
                    print(f"✅ ML 모델 학습 완료: 테스트 정확도 {training_results['test_score']:.4f}")
                except Exception as e:
                    print(f"❌ ML 모델 학습 실패: {e}")
                    return
            
            # 분석 실행 (ML 모델 사용 여부 전달)
            results = sensor.run_analysis_by_type(args.analysis, use_optimized, args.use_ml_model)
            
            if 'error' in results:
                print(f"❌ 분석 오류: {results['error']}")
                return
            
            # 분석 결과 출력
            if args.analysis == 'all':
                print("\n" + "=" * 80)
                print("📊 모든 분석 결과")
                print("=" * 80)
                
                # Quant 기반 기술적 분석 결과
                if 'technical_quant' in results and 'error' not in results['technical_quant']:
                    tech_quant = results['technical_quant']
                    print(f"\n📈 기술적 분석 (Quant 기반):")
                    print(f"  시장 상태: {tech_quant['current_regime']}")
                    
                    # 확률 정보 표시
                    if 'regime_probabilities' in tech_quant:
                        probs = tech_quant['regime_probabilities']
                        print(f"  확률 분포:")
                        print(f"    TRENDING_UP: {probs.get('trending_up', 0):.1%}")
                        print(f"    TRENDING_DOWN: {probs.get('trending_down', 0):.1%}")
                        print(f"    VOLATILE: {probs.get('volatile', 0):.1%}")
                        print(f"    SIDEWAYS: {probs.get('sideways', 0):.1%}")
                    
                    if 'secondary_regime' in tech_quant and tech_quant['secondary_regime']:
                        print(f"  두 번째 가능성: {tech_quant['secondary_regime']} ({tech_quant.get('secondary_probability', 0):.1%})")
                    
                    print(f"  총 수익률: {tech_quant['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {tech_quant['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"  승률: {tech_quant['performance_metrics'].get('win_rate', 0):.2%}")
                    print(f"  수익 팩터: {tech_quant['performance_metrics'].get('profit_factor', 0):.2f}")
                    print(f"  RSI: {tech_quant['technical_indicators'].get('rsi', 0):.2f}")
                    print(f"  MACD: {tech_quant['technical_indicators'].get('macd', 0):.4f}")
                
                # ML 기반 기술적 분석 결과
                if 'technical_ml' in results and 'error' not in results['technical_ml']:
                    tech_ml = results['technical_ml']
                    print(f"\n🤖 기술적 분석 (ML 기반):")
                    print(f"  시장 상태: {tech_ml['current_regime']}")
                    
                    # 확률 정보 표시
                    if 'regime_probabilities' in tech_ml:
                        probs = tech_ml['regime_probabilities']
                        print(f"  확률 분포:")
                        print(f"    TRENDING_UP: {probs.get('trending_up', 0):.1%}")
                        print(f"    TRENDING_DOWN: {probs.get('trending_down', 0):.1%}")
                        print(f"    VOLATILE: {probs.get('volatile', 0):.1%}")
                        print(f"    SIDEWAYS: {probs.get('sideways', 0):.1%}")
                    
                    if 'secondary_regime' in tech_ml and tech_ml['secondary_regime']:
                        print(f"  두 번째 가능성: {tech_ml['secondary_regime']} ({tech_ml.get('secondary_probability', 0):.1%})")
                    
                    print(f"  총 수익률: {tech_ml['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {tech_ml['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"  승률: {tech_ml['performance_metrics'].get('win_rate', 0):.2%}")
                    print(f"  수익 팩터: {tech_ml['performance_metrics'].get('profit_factor', 0):.2f}")
                    print(f"  RSI: {tech_ml['technical_indicators'].get('rsi', 0):.2f}")
                    print(f"  MACD: {tech_ml['technical_indicators'].get('macd', 0):.4f}")
                
                # 매크로 분석 결과
                if 'macro' in results and 'error' not in results['macro']:
                    macro = results['macro']
                    print(f"\n🌍 매크로 분석:")
                    print(f"  시장 조건: {macro['market_condition']}")
                    
                    # VIX 분석
                    if 'vix_analysis' in macro and macro['vix_analysis']:
                        vix = macro['vix_analysis']
                        print(f"  📊 VIX 분석:")
                        print(f"    현재 레벨: {vix.get('current_level', 0):.2f}")
                        print(f"    20일 평균: {vix.get('ma_20', 0):.2f}")
                        print(f"    52주 백분위: {vix.get('percentile_52w', 0):.1f}%")
                        print(f"    변동성 상태: {vix.get('volatility_regime', 'unknown')}")
                        print(f"    추세: {vix.get('trend', 'unknown')}")
                    
                    # TIPS 분석
                    if 'tips_analysis' in macro and macro['tips_analysis']:
                        tips = macro['tips_analysis']
                        print(f"  💰 TIPS Spread 분석:")
                        print(f"    종합 스프레드: {tips.get('composite_spread', 0):.4f}")
                        print(f"    50일 평균: {tips.get('composite_ma_50', 0):.4f}")
                        print(f"    인플레이션 기대: {tips.get('inflation_expectation', 'unknown')}")
                        print(f"    인플레이션 추세: {tips.get('inflation_trend', 'unknown')}")
                    
                    # 국채 스프레드 분석
                    if 'yield_analysis' in macro and macro['yield_analysis']:
                        yield_curve = macro['yield_analysis']
                        print(f"  📈 국채 스프레드 분석:")
                        print(f"    현재 스프레드: {yield_curve.get('current_spread', 0):.4f}")
                        print(f"    20일 평균: {yield_curve.get('spread_ma_20', 0):.4f}")
                        print(f"    경기침체 위험: {yield_curve.get('recession_risk', 'unknown')}")
                        print(f"    스프레드 추세: {yield_curve.get('spread_trend', 'unknown')}")
                    
                    # 달러 강도 분석
                    if 'dollar_analysis' in macro and macro['dollar_analysis']:
                        dollar = macro['dollar_analysis']
                        print(f"  💵 달러 강도 분석:")
                        print(f"    현재 레벨: {dollar.get('current_level', 0):.4f}")
                        print(f"    50일 평균: {dollar.get('ma_50', 0):.4f}")
                        print(f"    강도: {dollar.get('strength_level', 'unknown')}")
                        print(f"    추세: {dollar.get('trend', 'unknown')}")
                    
                    # 금 가격 분석
                    if 'gold_analysis' in macro and macro['gold_analysis']:
                        gold = macro['gold_analysis']
                        print(f"  🥇 금 가격 분석:")
                        print(f"    현재 가격: {gold.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {gold.get('ma_50', 0):.2f}")
                        print(f"    안전자산 수요: {gold.get('safe_haven_demand', 'unknown')}")
                        print(f"    추세: {gold.get('trend', 'unknown')}")
                    
                    # 국채 가격 분석
                    if 'bond_analysis' in macro and macro['bond_analysis']:
                        bond = macro['bond_analysis']
                        print(f"  📋 국채 가격 분석:")
                        print(f"    현재 가격: {bond.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {bond.get('ma_50', 0):.2f}")
                        print(f"    품질 선호도: {bond.get('flight_to_quality', 'unknown')}")
                        print(f"    추세: {bond.get('trend', 'unknown')}")
                    
                    print(f"  인플레이션 위험: {macro['key_indicators'].get('inflation_risk', 0):.2%}")
                    print(f"  금리 환경: {macro['key_indicators'].get('rate_environment', 'unknown')}")
                    print(f"  성장 전망: {macro['key_indicators'].get('growth_outlook', 'unknown')}")
                
                # 섹터 분석 결과
                if 'sector' in results and 'error' not in results['sector']:
                    sector = results['sector']
                    print(f"\n🏭 섹터 분석:")
                    
                    # 섹터 분류 출력
                    if sector['sector_categories']['leading']:
                        print(f"  선도 섹터: {', '.join(sector['sector_categories']['leading'])}")
                    if sector['sector_categories']['lagging']:
                        print(f"  후행 섹터: {', '.join(sector['sector_categories']['lagging'])}")
                    if sector['sector_categories']['defensive']:
                        print(f"  방어적 섹터: {', '.join(sector['sector_categories']['defensive'])}")
                    if sector['sector_categories']['cyclical']:
                        print(f"  순환적 섹터: {', '.join(sector['sector_categories']['cyclical'])}")
                    
                    # 투자 추천 출력
                    print(f"\n💡 투자 추천:")
                    if sector['recommendations']['overweight']:
                        print(f"  과중 배치: {', '.join(sector['recommendations']['overweight'])}")
                    if sector['recommendations']['underweight']:
                        print(f"  과소 배치: {', '.join(sector['recommendations']['underweight'])}")
                    if sector['recommendations']['neutral']:
                        print(f"  중립 배치: {', '.join(sector['recommendations']['neutral'])}")
                
                # 종합 분석 결과
                if 'comprehensive' in results and 'error' not in results['comprehensive']:
                    comp = results['comprehensive']
                    print(f"\n🎯 종합 전략:")
                    print(f"  주요 전략: {comp['recommendation']['primary_strategy']}")
                    print(f"  포지션 크기: {comp['recommendation']['position_size']:.1%}")
                    print(f"  설명: {comp['recommendation']['description']}")
                
            else:
                # 단일 분석 결과 출력
                if args.analysis == 'technical':
                    tech = results
                    print(f"\n📈 기술적 분석 결과:")
                    print(f"  시장 상태: {tech['current_regime']}")
                    
                    # 확률 정보 표시
                    if 'regime_probabilities' in tech:
                        probs = tech['regime_probabilities']
                        print(f"  확률 분포:")
                        print(f"    TRENDING_UP: {probs.get('trending_up', 0):.1%}")
                        print(f"    TRENDING_DOWN: {probs.get('trending_down', 0):.1%}")
                        print(f"    VOLATILE: {probs.get('volatile', 0):.1%}")
                        print(f"    SIDEWAYS: {probs.get('sideways', 0):.1%}")
                    
                    if 'secondary_regime' in tech and tech['secondary_regime']:
                        print(f"  두 번째 가능성: {tech['secondary_regime']} ({tech.get('secondary_probability', 0):.1%})")
                    
                    print(f"  총 수익률: {tech['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {tech['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"  승률: {tech['performance_metrics'].get('win_rate', 0):.2%}")
                    print(f"  수익 팩터: {tech['performance_metrics'].get('profit_factor', 0):.2f}")
                    print(f"  평균 승: {tech['performance_metrics'].get('avg_win', 0):.4%}")
                    print(f"  평균 패: {tech['performance_metrics'].get('avg_loss', 0):.4%}")
                    print(f"  최대 연속 승: {tech['performance_metrics'].get('max_consecutive_wins', 0)}")
                    print(f"  최대 연속 패: {tech['performance_metrics'].get('max_consecutive_losses', 0)}")
                    print(f"  거래 빈도: {tech['performance_metrics'].get('trade_frequency', 0):.2%}")
                    print(f"  총 거래 수: {tech['performance_metrics'].get('total_trades', 0)}")
                    print(f"  최대 낙폭: {tech['performance_metrics'].get('max_drawdown', 0):.4%}")
                    print(f"\n📊 기술적 지표:")
                    print(f"  RSI: {tech['technical_indicators'].get('rsi', 0):.2f}")
                    print(f"  MACD: {tech['technical_indicators'].get('macd', 0):.4f}")
                    print(f"  SMA(20): {tech['technical_indicators'].get('sma_short', 0):.2f}")
                    print(f"  SMA(50): {tech['technical_indicators'].get('sma_long', 0):.2f}")
                    print(f"  ATR: {tech['technical_indicators'].get('atr', 0):.4f}")
                
                elif args.analysis == 'macro':
                    macro = results
                    print(f"\n🌍 매크로 분석 결과:")
                    print(f"  시장 조건: {macro['market_condition']}")
                    
                    # VIX 분석
                    if 'vix_analysis' in macro and macro['vix_analysis']:
                        vix = macro['vix_analysis']
                        print(f"  📊 VIX 분석:")
                        print(f"    현재 레벨: {vix.get('current_level', 0):.2f}")
                        print(f"    20일 평균: {vix.get('ma_20', 0):.2f}")
                        print(f"    52주 백분위: {vix.get('percentile_52w', 0):.1f}%")
                        print(f"    변동성 상태: {vix.get('volatility_regime', 'unknown')}")
                        print(f"    추세: {vix.get('trend', 'unknown')}")
                    
                    # TIPS 분석
                    if 'tips_analysis' in macro and macro['tips_analysis']:
                        tips = macro['tips_analysis']
                        print(f"  💰 TIPS Spread 분석:")
                        print(f"    종합 스프레드: {tips.get('composite_spread', 0):.4f}")
                        print(f"    50일 평균: {tips.get('composite_ma_50', 0):.4f}")
                        print(f"    인플레이션 기대: {tips.get('inflation_expectation', 'unknown')}")
                        print(f"    인플레이션 추세: {tips.get('inflation_trend', 'unknown')}")
                    
                    # 국채 스프레드 분석
                    if 'yield_analysis' in macro and macro['yield_analysis']:
                        yield_curve = macro['yield_analysis']
                        print(f"  📈 국채 스프레드 분석:")
                        print(f"    현재 스프레드: {yield_curve.get('current_spread', 0):.4f}")
                        print(f"    20일 평균: {yield_curve.get('spread_ma_20', 0):.4f}")
                        print(f"    경기침체 위험: {yield_curve.get('recession_risk', 'unknown')}")
                        print(f"    스프레드 추세: {yield_curve.get('spread_trend', 'unknown')}")
                    
                    # 달러 강도 분석
                    if 'dollar_analysis' in macro and macro['dollar_analysis']:
                        dollar = macro['dollar_analysis']
                        print(f"  💵 달러 강도 분석:")
                        print(f"    현재 레벨: {dollar.get('current_level', 0):.4f}")
                        print(f"    50일 평균: {dollar.get('ma_50', 0):.4f}")
                        print(f"    강도: {dollar.get('strength_level', 'unknown')}")
                        print(f"    추세: {dollar.get('trend', 'unknown')}")
                    
                    # 금 가격 분석
                    if 'gold_analysis' in macro and macro['gold_analysis']:
                        gold = macro['gold_analysis']
                        print(f"  🥇 금 가격 분석:")
                        print(f"    현재 가격: {gold.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {gold.get('ma_50', 0):.2f}")
                        print(f"    안전자산 수요: {gold.get('safe_haven_demand', 'unknown')}")
                        print(f"    추세: {gold.get('trend', 'unknown')}")
                    
                    # 국채 가격 분석
                    if 'bond_analysis' in macro and macro['bond_analysis']:
                        bond = macro['bond_analysis']
                        print(f"  📋 국채 가격 분석:")
                        print(f"    현재 가격: {bond.get('current_price', 0):.2f}")
                        print(f"    50일 평균: {bond.get('ma_50', 0):.2f}")
                        print(f"    품질 선호도: {bond.get('flight_to_quality', 'unknown')}")
                        print(f"    추세: {bond.get('trend', 'unknown')}")
                    
                    print(f"\n📊 주요 지표:")
                    print(f"  인플레이션 위험: {macro['key_indicators'].get('inflation_risk', 0):.2%}")
                    print(f"  금리 환경: {macro['key_indicators'].get('rate_environment', 'unknown')}")
                    print(f"  성장 전망: {macro['key_indicators'].get('growth_outlook', 'unknown')}")
                    print(f"  VIX 레벨: {macro['key_indicators'].get('vix_level', 0):.2f}")
                    print(f"  수익률 곡선: {macro['key_indicators'].get('yield_curve', 'unknown')}")
                
                elif args.analysis == 'sector':
                    sector = results
                    print(f"\n🏭 섹터 분석 결과:")
                    print(f"\n📊 섹터 분류:")
                    if sector['sector_categories']['leading']:
                        print(f"  선도 섹터: {', '.join(sector['sector_categories']['leading'])}")
                    if sector['sector_categories']['lagging']:
                        print(f"  후행 섹터: {', '.join(sector['sector_categories']['lagging'])}")
                    if sector['sector_categories']['defensive']:
                        print(f"  방어적 섹터: {', '.join(sector['sector_categories']['defensive'])}")
                    if sector['sector_categories']['cyclical']:
                        print(f"  순환적 섹터: {', '.join(sector['sector_categories']['cyclical'])}")
                    
                    print(f"\n💡 투자 추천:")
                    if sector['recommendations']['overweight']:
                        print(f"  과중 배치: {', '.join(sector['recommendations']['overweight'])}")
                    if sector['recommendations']['underweight']:
                        print(f"  과소 배치: {', '.join(sector['recommendations']['underweight'])}")
                    if sector['recommendations']['neutral']:
                        print(f"  중립 배치: {', '.join(sector['recommendations']['neutral'])}")
                
                elif args.analysis == 'comprehensive':
                    comp = results
                    print(f"\n🎯 종합 분석 결과:")
                    print(f"  시장 상태: {comp['current_regime']}")
                    print(f"  총 수익률: {comp['performance_metrics'].get('total_return', 0):.4%}")
                    print(f"  샤프 비율: {comp['performance_metrics'].get('sharpe_ratio', 0):.4f}")
                    print(f"\n💡 전략 추천:")
                    print(f"  주요 전략: {comp['recommendation']['primary_strategy']}")
                    print(f"  보조 전략: {comp['recommendation']['secondary_strategy']}")
                    print(f"  포지션 크기: {comp['recommendation']['position_size']:.1%}")
                    print(f"  설명: {comp['recommendation']['description']}")
            
            # 결과 저장
            if args.save_results:
                output_dir = f"results/analysis_{args.analysis}"
                os.makedirs(output_dir, exist_ok=True)
                
                with open(f"{output_dir}/analysis_{args.analysis}_{sensor.session_uuid}.json", 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"\n✅ 분석 결과 저장 완료: {output_dir}/analysis_{args.analysis}_{sensor.session_uuid}.json")
            
            print(f"\n✅ {args.analysis} 분석 완료!")
            return
            
        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
            return
    
    if args.mode == 'experiment':
        # 종합 실험 모드
        print("🚀 종합 실험 시작...")
        print("=" * 80)
        
        # 1. 데이터 준비
        print("📊 1단계: 데이터 준비")
        print("-" * 40)
        
        if args.use_saved_data:
            print("📂 저장된 데이터만 사용 중...")
            spy_data, macro_data, sector_data = sensor.load_macro_data_only()
            if spy_data.empty:
                print("❌ 저장된 데이터가 없습니다. --use_saved_data 옵션을 제거하고 다시 실행하세요.")
                return
            print(f"✅ 저장된 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
        elif args.force_download or (args.download_data and not os.path.exists(f"{sensor.data_dir}/spy_data.csv")):
            print("📥 새로운 데이터 다운로드 중...")
            spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            print(f"✅ 데이터 다운로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
        else:
            print("📂 기존 데이터 로드 중...")
            spy_data, macro_data, sector_data = sensor.load_macro_data()
            if spy_data.empty:
                print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            print(f"✅ 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
        
        print(f"📅 데이터 기간: {spy_data.index[0].strftime('%Y-%m-%d')} ~ {spy_data.index[-1].strftime('%Y-%m-%d')}")
        print()
        
        # 2. 하이퍼파라미터 최적화
        print("🔧 2단계: 하이퍼파라미터 최적화")
        print("-" * 40)
        print(f"🎯 최적화 시도 횟수: {args.n_trials}")
        print(f"📊 Train/Test 분할: 80%/20%")
        
        try:
            results = sensor.optimize_hyperparameters_optuna(start_date, end_date, args.n_trials)
            
            print(f"\n📈 최적화 결과:")
            print(f"   🏆 최적 목적 함수 값: {results['best_value']:.6f}")
            print(f"   ⚙️ 최적 파라미터 개수: {len(results['best_params'])}")
            print(f"   🎯 최적화 목표: {sensor.hyperparam_tuner.config.get('optimization', {}).get('objective', 'total_return')}")
            
            # 최적 파라미터 요약
            print(f"\n🔍 최적 파라미터 요약:")
            best_params = results['best_params']
            
            # 기술적 지표 파라미터
            tech_params = {k: v for k, v in best_params.items() if any(x in k for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'volume'])}
            print(f"   📊 기술적 지표: {len(tech_params)}개")
            for param, value in tech_params.items():
                if isinstance(value, float):
                    print(f"      {param}: {value:.3f}")
                else:
                    print(f"      {param}: {value}")
            
            # 가중치 파라미터
            weight_params = {k: v for k, v in best_params.items() if 'weight' in k}
            print(f"   ⚖️ 분류 가중치: {len(weight_params)}개")
            for param, value in weight_params.items():
                print(f"      {param}: {value:.3f}")
            
            # 거래 전략 파라미터
            strategy_params = {k: v for k, v in best_params.items() if any(x in k for x in ['position', 'boost', 'reduction', 'stop', 'profit'])}
            print(f"   💼 거래 전략: {len(strategy_params)}개")
            for param, value in strategy_params.items():
                if isinstance(value, float):
                    print(f"      {param}: {value:.3f}")
                else:
                    print(f"      {param}: {value}")
            
            print()
                    
        except Exception as e:
            print(f"❌ 최적화 중 오류: {e}")
            return
        
        # 3. 성과 분석
        print("📊 3단계: 성과 분석")
        print("-" * 40)
        
        if results['test_performance']:
            test_metrics = results['test_performance']
            
            print("🧪 Test 성과 지표:")
            print(f"   📈 총 수익률: {test_metrics.get('total_return', 0):.4%}")
            print(f"   📊 로그 수익률: {test_metrics.get('log_return', 0):.4%}")
            print(f"   🏠 Buy & Hold 수익률: {test_metrics.get('buy_hold_return', 0):.4%}")
            print(f"   🏠 Buy & Hold 로그 수익률: {test_metrics.get('buy_hold_log_return', 0):.4%}")
            print(f"   ⚡ 초과 수익률: {test_metrics.get('excess_return', 0):.4%}")
            print(f"   📊 샤프 비율: {test_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"   📊 소르티노 비율: {test_metrics.get('sortino_ratio', 0):.4f}")
            print(f"   📊 칼마 비율: {test_metrics.get('calmar_ratio', 0):.4f}")
            print(f"   📉 최대 낙폭: {test_metrics.get('max_drawdown', 0):.4%}")
            print(f"   🎯 승률: {test_metrics.get('win_rate', 0):.2%}")
            print(f"   💰 평균 승: {test_metrics.get('avg_win', 0):.4%}")
            print(f"   💸 평균 패: {test_metrics.get('avg_loss', 0):.4%}")
            print(f"   📈 수익 팩터: {test_metrics.get('profit_factor', 0):.2f}")
            print(f"   🔥 최대 연속 승: {test_metrics.get('max_consecutive_wins', 0)}")
            print(f"   ❄️ 최대 연속 패: {test_metrics.get('max_consecutive_losses', 0)}")
            print(f"   📊 거래 빈도: {test_metrics.get('trade_frequency', 0):.2%}")
            print(f"   📋 총 거래 수: {test_metrics.get('total_trades', 0)}")
            
            # 성과 평가
            print(f"\n🏆 성과 평가:")
            if test_metrics.get('total_return', 0) > test_metrics.get('buy_hold_return', 0):
                print("   ✅ 전략이 Buy & Hold를 상회")
            else:
                print("   ❌ 전략이 Buy & Hold에 미달")
            
            if test_metrics.get('sharpe_ratio', 0) > 1.0:
                print("   ✅ 샤프 비율 양호 (> 1.0)")
            elif test_metrics.get('sharpe_ratio', 0) > 0:
                print("   ⚠️ 샤프 비율 보통 (0 ~ 1.0)")
            else:
                print("   ❌ 샤프 비율 불량 (< 0)")
            
            if test_metrics.get('max_drawdown', 0) > -0.1:
                print("   ✅ 최대 낙폭 양호 (< 10%)")
            elif test_metrics.get('max_drawdown', 0) > -0.2:
                print("   ⚠️ 최대 낙폭 보통 (10% ~ 20%)")
            else:
                print("   ❌ 최대 낙폭 불량 (> 20%)")
            
            print()
        
        # 4. 결과 저장
        print("💾 4단계: 결과 저장")
        print("-" * 40)
        
        try:
            sensor.save_optimization_results(results)
            print(f"✅ 결과 저장 완료: results/market_sensor_optimization/{sensor.session_uuid}/")
            print(f"   📄 best_params.json - 최적 파라미터")
            print(f"   📊 performance_summary.json - 성과 지표")
            print(f"   📈 optuna_study.json - 최적화 과정")
            print(f"   📋 metadata.json - 메타데이터")
            
            # 거래 로그 생성
            log_path = sensor.generate_transaction_log(results)
            if log_path:
                print(f"   📋 transaction_log.log - 거래 내역 로그")
            print()
        except Exception as e:
            print(f"❌ 결과 저장 중 오류: {e}")
        
        # 5. 실험 요약
        print("📋 5단계: 실험 요약")
        print("-" * 40)
        print(f"🆔 실험 ID: {sensor.session_uuid}")
        print(f"📅 실험 기간: {start_date} ~ {end_date}")
        print(f"🔧 최적화 시도: {args.n_trials}회")
        print(f"📊 데이터 포인트: {len(spy_data)}개")
        print(f"🎯 최적화 목표: {sensor.hyperparam_tuner.config.get('optimization', {}).get('objective', 'total_return')}")
        
        if results['test_performance']:
            test_metrics = results['test_performance']
            print(f"📈 최종 성과: {test_metrics.get('total_return', 0):.4%} (vs Buy & Hold: {test_metrics.get('buy_hold_return', 0):.4%})")
            print(f"📊 위험 조정 수익률: {test_metrics.get('sharpe_ratio', 0):.4f}")
        
        print("=" * 80)
        print("🎉 종합 실험 완료!")
        
    elif args.mode == 'collect':
        # 데이터 수집 모드
        print("📊 데이터 수집 모드")
        print("=" * 60)
        
        if args.force_download:
            print("🔄 강제 데이터 다운로드 중...")
        else:
            print("📥 새로운 데이터 다운로드 중...")
        
        spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        
        print(f"✅ 데이터 수집 완료:")
        print(f"   📈 SPY: {len(spy_data)}개 데이터 포인트")
        print(f"   🌍 매크로 지표: {len(macro_data)}개 심볼")
        print(f"   🏭 섹터 ETF: {len(sector_data)}개 심볼")
        print(f"   📅 기간: {spy_data.index[0].strftime('%Y-%m-%d')} ~ {spy_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   💾 저장 위치: {sensor.data_dir}/")
        
    elif args.mode == 'optimize':
        # 하이퍼파라미터 튜닝 모드
        print("🔧 하이퍼파라미터 튜닝 모드")
        print("=" * 60)
        
        # 데이터 준비
        if args.force_download or (args.download_data and not os.path.exists(f"{sensor.data_dir}/spy_data.csv")):
            print("📥 새로운 데이터 다운로드 중...")
            spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        else:
            print("📂 기존 데이터 로드 중...")
            spy_data, macro_data, sector_data = sensor.load_macro_data()
            if spy_data.empty:
                print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                spy_data, macro_data, sector_data = sensor._collect_fresh_data()
        
        print(f"📊 데이터 준비 완료: {len(spy_data)}개 포인트")
        print(f"🔧 최적화 시작 (n_trials={args.n_trials})...")
        
        try:
            results = sensor.optimize_hyperparameters_optuna(start_date, end_date, args.n_trials)
            
            print(f"\n📈 최적화 결과:")
            print(f"   🏆 최적 목적 함수 값: {results['best_value']:.6f}")
            print(f"   ⚙️ 최적 파라미터 개수: {len(results['best_params'])}")
            
            if results['test_performance']:
                print(f"\n🧪 Test 성과:")
                test_metrics = results['test_performance']
                print(f"   📈 총 수익률: {test_metrics.get('total_return', 0):.4%}")
                print(f"   🏠 Buy & Hold 수익률: {test_metrics.get('buy_hold_return', 0):.4%}")
                print(f"   ⚡ 초과 수익률: {test_metrics.get('excess_return', 0):.4%}")
                print(f"   📊 샤프 비율: {test_metrics.get('sharpe_ratio', 0):.4f}")
                print(f"   📉 최대 낙폭: {test_metrics.get('max_drawdown', 0):.4%}")
                print(f"   🎯 승률: {test_metrics.get('win_rate', 0):.2%}")
            
            # 결과 저장
            if args.save_results:
                sensor.save_optimization_results(results)
                print(f"\n✅ 결과 저장 완료: results/market_sensor_optimization/{sensor.session_uuid}/")
                
                # 거래 로그 생성
                log_path = sensor.generate_transaction_log(results)
                if log_path:
                    print(f"   📋 transaction_log.log - 거래 내역 로그")
            
            print("✅ 하이퍼파라미터 튜닝 완료!")
            
        except Exception as e:
            print(f"❌ 최적화 중 오류: {e}")
    
    elif args.mode == 'macro_sector':
        # 매크로 & 섹터 분석 모드
        print("🔍 매크로 & 섹터 분석 중...")
        
        try:
            analysis = sensor.get_macro_sector_analysis(start_date, end_date)
            
            if analysis is None:
                print("❌ 분석 실패")
                return
            
            print(f"\n🎯 시장 조건: {analysis.market_condition.value}")
            print(f"📊 신뢰도: {analysis.confidence:.2%}")
            
            print(f"\n📈 주요 지표:")
            for indicator, value in analysis.key_indicators.items():
                if isinstance(value, float):
                    print(f"  {indicator}: {value:.4f}")
                else:
                    print(f"  {indicator}: {value}")
            
            print(f"\n🏭 섹터 강도:")
            for sector, strength in analysis.sector_rotation.items():
                sector_name = sensor.macro_analyzer.sector_classification.get(sector, {}).get('name', sector)
                print(f"  {sector_name} ({sector}): {strength.value}")
            
            print(f"\n💡 투자 추천:")
            print(f"  전략: {analysis.recommendations['strategy']}")
            print(f"  위험도: {analysis.recommendations['risk_level']}")
            
            if analysis.recommendations['overweight_sectors']:
                print(f"  과중 배치 섹터: {', '.join(analysis.recommendations['overweight_sectors'])}")
            if analysis.recommendations['underweight_sectors']:
                print(f"  과소 배치 섹터: {', '.join(analysis.recommendations['underweight_sectors'])}")
            
            # 결과 저장
            if args.save_results:
                sensor.save_macro_analysis_results(analysis)
                print(f"\n✅ 분석 결과 저장 완료")
            
            print("✅ 매크로 & 섹터 분석 완료!")
                
        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
    
    elif args.mode == 'analyze':
        # 기본 시장 분석 모드
        print("🔍 기본 시장 분석 중...")
        
        try:
            # 최적 파라미터 로드 시도
            if args.use_optimized:
                sensor.load_optimal_params()
            
            # 현재 시장 분석
            analysis = sensor.get_current_market_analysis(use_optimized_params=args.use_optimized)
            
            if 'error' in analysis:
                print(f"❌ 분석 오류: {analysis['error']}")
                return
            
            print(f"\n🎯 현재 시장 환경: {analysis['current_regime']}")
            print(f"📅 데이터 기간: {analysis['data_period']}")
            print(f"🕒 마지막 업데이트: {analysis['last_update']}")
            
            print(f"\n📊 성과 지표:")
            for metric, value in analysis['performance_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\n💡 전략 추천:")
            print(f"  주요 전략: {analysis['recommendation']['primary_strategy']}")
            print(f"  보조 전략: {analysis['recommendation']['secondary_strategy']}")
            print(f"  포지션 크기: {analysis['recommendation']['position_size']:.1%}")
            print(f"  설명: {analysis['recommendation']['description']}")
            
            # 매크로 분석 결과 출력
            if 'macro_analysis' in analysis and analysis['macro_analysis']:
                print(f"\n🌍 매크로 분석:")
                macro = analysis['macro_analysis']
                if 'inflation_risk' in macro:
                    print(f"  인플레이션 위험: {macro['inflation_risk']:.2%}")
                if 'rate_environment' in macro:
                    print(f"  금리 환경: {macro['rate_environment']}")
                if 'growth_outlook' in macro:
                    print(f"  성장 전망: {macro['growth_outlook']}")
            
            # 섹터 분석 결과 출력
            if 'sector_analysis' in analysis and analysis['sector_analysis']:
                print(f"\n🏭 섹터 분석:")
                sector = analysis['sector_analysis']
                leading_sectors = [s for s, strength in sector.items() if strength == SectorStrength.LEADING]
                defensive_sectors = [s for s, strength in sector.items() if strength == SectorStrength.DEFENSIVE]
                
                if leading_sectors:
                    print(f"  선도 섹터: {', '.join(leading_sectors)}")
                if defensive_sectors:
                    print(f"  방어적 섹터: {', '.join(defensive_sectors)}")
            
            # 강화된 전략 추천 출력
            if 'leading_sectors' in analysis['recommendation']:
                print(f"  추천 선도 섹터: {', '.join(analysis['recommendation']['leading_sectors'])}")
            if 'defensive_sectors' in analysis['recommendation']:
                print(f"  추천 방어적 섹터: {', '.join(analysis['recommendation']['defensive_sectors'])}")
            
            print("✅ 기본 시장 분석 완료!")
            
        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
    
    elif args.mode == 'comprehensive':
        # 기술적 + 매크로 종합 분석 모드
        print("🔍 기술적 + 매크로 종합 분석 중...")
        
        try:
            # 데이터 준비
            if args.use_saved_data:
                print("📂 저장된 데이터만 사용 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data()
                if spy_data.empty:
                    print("❌ 저장된 데이터가 없습니다. --use_saved_data 옵션을 제거하고 다시 실행하세요.")
                    return
            else:
                print("📂 기존 데이터 로드 중...")
                spy_data, macro_data, sector_data = sensor.load_macro_data()
                if spy_data.empty:
                    print("⚠️ 기존 데이터가 없어 새로 다운로드합니다.")
                    spy_data, macro_data, sector_data = sensor._collect_fresh_data()
            
            print(f"✅ 데이터 로드 완료: SPY({len(spy_data)}개), 매크로({len(macro_data)}개), 섹터({len(sector_data)}개)")
            
            # 최적 파라미터 로드 시도
            if args.use_optimized:
                sensor.load_optimal_params()
            
            # 종합 분석 실행
            analysis = sensor.get_current_market_analysis(use_optimized_params=args.use_optimized)
            
            if 'error' in analysis:
                print(f"❌ 분석 오류: {analysis['error']}")
                return
            
            print(f"\n🎯 현재 시장 환경: {analysis['current_regime']}")
            print(f"📅 데이터 기간: {analysis['data_period']}")
            print(f"🕒 마지막 업데이트: {analysis['last_update']}")
            
            print(f"\n📊 성과 지표:")
            for metric, value in analysis['performance_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\n💡 전략 추천:")
            print(f"  주요 전략: {analysis['recommendation']['primary_strategy']}")
            print(f"  보조 전략: {analysis['recommendation']['secondary_strategy']}")
            print(f"  포지션 크기: {analysis['recommendation']['position_size']:.1%}")
            print(f"  설명: {analysis['recommendation']['description']}")
            
            # 매크로 분석 결과 출력
            if 'macro_analysis' in analysis and analysis['macro_analysis']:
                print(f"\n🌍 매크로 분석:")
                macro = analysis['macro_analysis']
                if 'inflation_risk' in macro:
                    print(f"  인플레이션 위험: {macro['inflation_risk']:.2%}")
                if 'rate_environment' in macro:
                    print(f"  금리 환경: {macro['rate_environment']}")
                if 'growth_outlook' in macro:
                    print(f"  성장 전망: {macro['growth_outlook']}")
            
            # 섹터 분석 결과 출력
            if 'sector_analysis' in analysis and analysis['sector_analysis']:
                print(f"\n🏭 섹터 분석:")
                sector = analysis['sector_analysis']
                leading_sectors = [s for s, strength in sector.items() if strength == SectorStrength.LEADING]
                defensive_sectors = [s for s, strength in sector.items() if strength == SectorStrength.DEFENSIVE]
                
                if leading_sectors:
                    print(f"  선도 섹터: {', '.join(leading_sectors)}")
                if defensive_sectors:
                    print(f"  방어적 섹터: {', '.join(defensive_sectors)}")
            
            # 강화된 전략 추천 출력
            if 'leading_sectors' in analysis['recommendation']:
                print(f"  추천 선도 섹터: {', '.join(analysis['recommendation']['leading_sectors'])}")
            if 'defensive_sectors' in analysis['recommendation']:
                print(f"  추천 방어적 섹터: {', '.join(analysis['recommendation']['defensive_sectors'])}")
            
            # 결과 저장
            if args.save_results:
                # 종합 분석 결과 저장
                comprehensive_results = {
                    'session_uuid': sensor.session_uuid,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'data_period': analysis['data_period']
                }
                
                output_dir = "results/comprehensive_analysis"
                os.makedirs(output_dir, exist_ok=True)
                
                with open(f"{output_dir}/comprehensive_analysis_{sensor.session_uuid}.json", 'w', encoding='utf-8') as f:
                    json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"\n✅ 종합 분석 결과 저장 완료: {output_dir}/comprehensive_analysis_{sensor.session_uuid}.json")
            
            print("✅ 기술적 + 매크로 종합 분석 완료!")
            
        except Exception as e:
            print(f"❌ 종합 분석 중 오류: {e}")
    
    print("\n🎉 Market Sensor 실행 완료!")


if __name__ == "__main__":
    main()
