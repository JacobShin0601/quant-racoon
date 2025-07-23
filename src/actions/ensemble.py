#!/usr/bin/env python3
"""
앙상블 전략 - 시장 환경별 전략 선택 및 실행
market_sensor가 감지한 시장 환경에 따라 적절한 전략을 선택하고 실행
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .strategies import StrategyManager
from ..agent.market_sensor import MarketSensor
from ..agent.orchestrator import Orchestrator
from ..agent.helper import load_config, print_section_header, print_subsection_header


class EnsembleStrategy:
    """시장 환경별 앙상블 전략 클래스"""
    
    def __init__(
        self,
        config_path: str = "config/config_ensemble.json",
        market_sensor_config: str = "config/config_macro.json",
        uuid: Optional[str] = None
    ):
        self.config_path = config_path
        self.market_sensor_config = market_sensor_config
        self.uuid = uuid or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 설정 로드
        self.config = load_config(config_path)
        self.market_sensor_config_data = load_config(market_sensor_config)
        
        # Market Sensor 초기화
        self.market_sensor = MarketSensor(
            data_dir=self.config["market_sensor"]["data_dir"],
            config_path=self.config["market_sensor"]["config_path"]
        )
        
        # Strategy Manager 초기화
        self.strategy_manager = StrategyManager()
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 시장 환경별 설정 로드
        self.regime_configs = self._load_regime_configs()
        
        # 성과 추적
        self.performance_history = {
            "regime_performance": {},
            "strategy_performance": {},
            "regime_transitions": [],
            "ensemble_performance": {}
        }
        
        print_section_header("🎯 앙상블 전략 초기화")
        print(f"📁 설정 파일: {config_path}")
        print(f"🆔 실행 UUID: {self.uuid}")
        print(f"📊 시장 환경별 설정: {len(self.regime_configs)}개")
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = Path(self.config["output"]["logs_folder"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"ensemble_strategy_{self.uuid}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _load_regime_configs(self) -> Dict[str, Dict[str, Any]]:
        """시장 환경별 설정 로드"""
        regime_configs = {}
        
        for regime, config_path in self.config["market_regime_configs"].items():
            try:
                regime_configs[regime] = load_config(config_path)
                self.logger.info(f"✅ {regime} 설정 로드 완료: {config_path}")
            except Exception as e:
                self.logger.error(f"❌ {regime} 설정 로드 실패: {e}")
        
        return regime_configs
    
    def detect_market_regime(self, date: Optional[str] = None) -> Dict[str, Any]:
        """시장 환경 감지"""
        print_subsection_header("🔍 시장 환경 감지")
        
        try:
            # Market Sensor를 사용하여 시장 환경 분석 (ML 모델 사용)
            analysis = self.market_sensor.get_current_market_analysis(
                use_optimized_params=self.config["market_sensor"]["use_optimized_params"],
                use_ml_model=self.config["market_sensor"]["use_ml_model"]
            )
            
            # 시장 환경 분류 결과 추출
            market_regime = analysis.get("current_regime", "UNCERTAIN")
            confidence = analysis.get("confidence", 0.5)  # 기본 신뢰도 (더 보수적)
            
            # 신뢰도 임계값 확인
            if confidence < self.config["ensemble_settings"]["regime_confidence_threshold"]:
                market_regime = self.config["ensemble_settings"]["fallback_regime"]
                self.logger.warning(f"⚠️ 신뢰도 낮음 ({confidence:.3f}), 기본 환경 사용: {market_regime}")
            
            result = {
                "regime": market_regime,
                "confidence": confidence,
                "analysis": analysis,
                "detection_date": date or datetime.now().strftime("%Y-%m-%d")
            }
            
            print(f"📊 감지된 시장 환경: {market_regime}")
            print(f"🎯 신뢰도: {confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 시장 환경 감지 실패: {e}")
            return {
                "regime": self.config["ensemble_settings"]["fallback_regime"],
                "confidence": 0.0,
                "analysis": {},
                "detection_date": date or datetime.now().strftime("%Y-%m-%d"),
                "error": str(e)
            }
    
    def get_regime_config(self, regime: str) -> Dict[str, Any]:
        """시장 환경별 설정 가져오기"""
        if regime in self.regime_configs:
            return self.regime_configs[regime]
        else:
            self.logger.warning(f"⚠️ {regime} 설정이 없습니다. 기본 설정 사용")
            return self.regime_configs.get("SIDEWAYS", {})
    
    def analyze_regime_periods(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """시점별 Market Regime 분석 - market_sensor.py 로직 활용"""
        print("🔍 시점별 Market Regime 분석 시작")
        
        try:
            # 기본 기간 설정 (start_date, end_date가 없으면 최근 1년)
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            print(f"📅 분석 기간: {start_date} ~ {end_date}")
            
            # market_sensor.py의 기존 데이터 로딩 로직 활용
            # 이미 수집된 매크로 데이터를 사용
            print("📊 매크로 데이터 로딩 중...")
            
            # 기간별 regime 분석 (3개월 단위로 분할)
            regime_periods = []
            current_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            
            while current_date <= end_datetime:
                # 3개월 단위로 기간 설정
                period_end = min(
                    current_date + timedelta(days=90), 
                    end_datetime
                )
                
                period_start_str = current_date.strftime("%Y-%m-%d")
                period_end_str = period_end.strftime("%Y-%m-%d")
                
                print(f"🔍 {period_start_str} ~ {period_end_str} 기간 분석 중...")
                
                # 해당 기간의 regime 감지 (RF Classifier 활용)
                period_regime = self.detect_market_regime_for_period(
                    period_start_str, period_end_str
                )
                
                # regime별 적합한 전략 선택
                suitable_strategies = self.get_regime_strategies(period_regime["regime"])
                
                regime_periods.append({
                    "start_date": period_start_str,
                    "end_date": period_end_str,
                    "regime": period_regime["regime"],
                    "confidence": period_regime["confidence"],
                    "probabilities": period_regime.get("probabilities", {}),
                    "strategies": suitable_strategies,
                    "analysis": period_regime["analysis"],
                    "method": period_regime.get("method", "unknown")
                })
                
                current_date = period_end + timedelta(days=1)
            
            print(f"✅ 시점별 분석 완료: {len(regime_periods)}개 기간")
            
            # 분석 결과 요약 출력
            regime_summary = {}
            for period in regime_periods:
                regime = period["regime"]
                if regime not in regime_summary:
                    regime_summary[regime] = 0
                regime_summary[regime] += 1
            
            print("📊 기간별 Regime 분포:")
            for regime, count in regime_summary.items():
                print(f"   {regime}: {count}개 기간")
            
            return regime_periods
            
        except Exception as e:
            self.logger.error(f"❌ 시점별 Market Regime 분석 실패: {e}")
            # 기본값 반환
            return [{
                "start_date": start_date or "2024-01-01",
                "end_date": end_date or "2024-12-31", 
                "regime": "SIDEWAYS",
                "confidence": 0.5,
                "strategies": self.get_regime_strategies("SIDEWAYS"),
                "analysis": {},
                "method": "fallback"
            }]
    
    def detect_market_regime_for_period(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """특정 기간의 Market Regime 감지 - RF Classifier 활용 + 시점별 분석"""
        try:
            print(f"🔍 {start_date} ~ {end_date} 기간 RF Classifier 분석 시작")
            
            # 기간별로 다른 시드값을 사용하여 다양성 확보
            import hashlib
            period_hash = hashlib.md5(f"{start_date}_{end_date}".encode()).hexdigest()
            period_seed = int(period_hash[:8], 16) % 1000
            
            # 기간 정보를 바탕으로 한 특성 조정
            from datetime import datetime
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            period_factor = (start_dt.month % 4 + 1) / 4.0  # 계절성 반영
            
            # Market Sensor의 RF Classifier를 활용한 분석
            analysis = self.market_sensor.get_current_market_analysis(
                use_optimized_params=True,
                use_ml_model=True  # RF Classifier 사용
            )
            
            if "error" in analysis:
                self.logger.error(f"❌ RF Classifier 분석 실패: {analysis['error']}")
                # Fallback: 규칙 기반 분석
                analysis = self.market_sensor.get_current_market_analysis(
                    use_optimized_params=True,
                    use_ml_model=False  # 규칙 기반 분석으로 fallback
                )
            
            # 기간별 특성을 반영하여 확률 조정
            probabilities = analysis.get("probabilities", {})
            if probabilities:
                # 기간별 변동성 추가 (시점에 따라 다른 regime 선호도)
                adjustments = {
                    "trending_up": period_factor * 0.1,
                    "trending_down": (1 - period_factor) * 0.1,
                    "volatile": abs(period_factor - 0.5) * 0.2,
                    "sideways": (1 - abs(period_factor - 0.5)) * 0.1
                }
                
                adjusted_probs = {}
                for regime, prob in probabilities.items():
                    adjustment = adjustments.get(regime, 0)
                    adjusted_probs[regime] = min(1.0, max(0.0, prob + adjustment))
                
                # 정규화
                total_prob = sum(adjusted_probs.values())
                if total_prob > 0:
                    adjusted_probs = {k: v/total_prob for k, v in adjusted_probs.items()}
                    probabilities = adjusted_probs
            
            # 조정된 확률에서 최고 regime 선택
            current_regime = max(probabilities.items(), key=lambda x: x[1])[0].upper() if probabilities else "SIDEWAYS"
            
            # 신뢰도 재계산
            sorted_probs = sorted(probabilities.values(), reverse=True)
            if len(sorted_probs) >= 2:
                prob_diff = sorted_probs[0] - sorted_probs[1]
                confidence = 0.5 + prob_diff * 0.8  # 차이가 클수록 높은 신뢰도
            else:
                confidence = 0.5
                
            # 신뢰도 임계값 완화 적용
            confidence = min(0.9, max(0.3, confidence))
            
            print(f"✅ RF Classifier 분석 완료: {current_regime} (신뢰도: {confidence:.3f})")
            print(f"   조정된 확률: {probabilities}")
            
            return {
                "regime": current_regime,
                "confidence": confidence,
                "probabilities": probabilities,
                "analysis": analysis,
                "period": f"{start_date} ~ {end_date}",
                "period_factor": period_factor,
                "method": "RF_Classifier_Enhanced"
            }
            
        except Exception as e:
            self.logger.error(f"❌ 기간별 Market Regime 감지 실패: {e}")
            return {
                "regime": "SIDEWAYS",
                "confidence": 0.5,  # fallback 신뢰도도 임계값 이상으로
                "analysis": {},
                "period": f"{start_date} ~ {end_date}",
                "error": str(e),
                "method": "fallback"
            }
    
    def get_regime_strategies(self, regime: str) -> List[str]:
        """Regime별 적합한 전략 목록 반환"""
        regime_strategy_mapping = {
            "SIDEWAYS": [
                "mean_reversion", "swing_rsi", "stochastic", "williams_r", 
                "cci", "cci_bollinger", "swing_bollinger_band", "whipsaw_prevention",
                "donchian_rsi_whipsaw", "range_breakout", "support_resistance"
            ],
            "TRENDING_UP": [
                "dual_momentum", "volatility_breakout", "swing_ema", "swing_donchian",
                "stoch_donchian", "swing_breakout", "swing_pullback_entry", 
                "swing_macd", "trend_following_ma200"
            ],
            "TRENDING_DOWN": [
                "mean_reversion", "swing_rsi", "stochastic", "williams_r",
                "cci", "cci_bollinger", "swing_bollinger_band", "whipsaw_prevention"
            ],
            "VOLATILE": [
                "volatility_filtered_breakout", "multi_timeframe_whipsaw", 
                "adaptive_whipsaw", "swing_candle_pattern", "swing_bollinger_band",
                "whipsaw_prevention", "donchian_rsi_whipsaw"
            ]
        }
        
        return regime_strategy_mapping.get(regime, regime_strategy_mapping["SIDEWAYS"])
    
    def run_period_optimization(self, period: Dict[str, Any], time_horizon: str = "ensemble") -> bool:
        """기간별 전략 최적화 실행"""
        regime = period["regime"]
        start_date = period["start_date"]
        end_date = period["end_date"]
        strategies = period["strategies"]
        
        print(f"🔧 {regime} 기간 최적화 시작")
        print(f"   기간: {start_date} ~ {end_date}")
        print(f"   전략: {len(strategies)}개 ({', '.join(strategies[:3])}...)")
        
        try:
            # 시장 환경별 설정 가져오기
            regime_config = self.get_regime_config(regime)
            
            # 기간별 최적화를 위한 임시 research config 생성
            temp_research_config = self.create_period_research_config(
                regime, strategies, start_date, end_date
            )
            
            print(f"📂 데이터 디렉토리: data/{time_horizon}")
            print(f"📋 설정 파일: config/config_ensemble_{regime.lower()}.json")
            
            # Orchestrator 초기화 및 실행 (time_horizon을 argument로 전달)
            orchestrator = Orchestrator(
                config_path=f"config/config_ensemble_{regime.lower()}.json",
                time_horizon=time_horizon,  # argument로 전달받은 time_horizon 사용
                uuid=f"{self.uuid}_{regime.lower()}_{start_date.replace('-', '')}",
                research_config_path=temp_research_config  # 임시 research config 사용
            )
            
            # researcher 단계만 실행 (데이터는 이미 준비됨)
            print(f"🔬 전략 최적화 단계 실행...")
            success = orchestrator.run_single_stage("researcher")
            
            if success:
                self.logger.info(f"✅ {regime} 기간 최적화 완료")
                return True
            else:
                self.logger.error(f"❌ {regime} 기간 최적화 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ {regime} 기간 최적화 중 오류: {e}")
            return False
    
    def create_period_research_config(self, regime: str, strategies: List[str], start_date: str, end_date: str) -> str:
        """기간별 최적화를 위한 임시 research config 생성"""
        try:
            # 기본 research config 로드
            base_config_path = "config/config_ensemble_research.json"
            with open(base_config_path, "r", encoding="utf-8") as f:
                base_config = json.load(f)
            
            # regime별 전략만 필터링
            filtered_strategies = {}
            for strategy_name in strategies:
                if strategy_name in base_config.get("strategies", {}):
                    filtered_strategies[strategy_name] = base_config["strategies"][strategy_name]
            
            # 최적화 횟수를 70회로 증가
            if "optimization_settings" in base_config:
                if "bayesian_optimization" in base_config["optimization_settings"]:
                    base_config["optimization_settings"]["bayesian_optimization"]["n_trials"] = 70
                if "grid_search" in base_config["optimization_settings"]:
                    base_config["optimization_settings"]["grid_search"]["max_combinations"] = 70
            
            # 필터링된 전략으로 config 업데이트
            base_config["strategies"] = filtered_strategies
            
            # 기간 정보 추가
            base_config["period_info"] = {
                "regime": regime,
                "start_date": start_date,
                "end_date": end_date,
                "strategy_count": len(filtered_strategies)
            }
            
            # 임시 파일로 저장
            temp_config_path = f"config/temp_research_{regime.lower()}_{start_date.replace('-', '')}.json"
            with open(temp_config_path, "w", encoding="utf-8") as f:
                json.dump(base_config, f, indent=2, ensure_ascii=False)
            
            print(f"📝 임시 research config 생성: {temp_config_path}")
            print(f"   전략 수: {len(filtered_strategies)}개")
            print(f"   최적화 횟수: 70회")
            
            return temp_config_path
            
        except Exception as e:
            self.logger.error(f"❌ 임시 research config 생성 실패: {e}")
            return "config/config_ensemble_research.json"  # 기본값 반환
    
    def run_regime_specific_pipeline(self, regime: str, time_horizon: str = "ensemble") -> bool:
        """시장 환경별 파이프라인 실행 (기존 메서드 유지)"""
        print_subsection_header(f"🚀 {regime} 환경 파이프라인 실행")
        
        try:
            # 시장 환경별 설정 가져오기
            regime_config = self.get_regime_config(regime)
            
            # Orchestrator 초기화 및 실행 (앙상블용 연구 설정 사용)
            orchestrator = Orchestrator(
                config_path=f"config/config_ensemble_{regime.lower()}.json",
                time_horizon=time_horizon,  # argument로 전달받은 time_horizon 사용
                uuid=f"{self.uuid}_{regime.lower()}",
                research_config_path="config/config_ensemble_research.json"  # 앙상블용 연구 설정
            )
            
            # 전체 파이프라인 실행
            success = orchestrator.run_pipeline()
            
            if success:
                self.logger.info(f"✅ {regime} 환경 파이프라인 실행 완료")
                return True
            else:
                self.logger.error(f"❌ {regime} 환경 파이프라인 실행 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ {regime} 환경 파이프라인 실행 중 오류: {e}")
            return False
    
    def run_ensemble_pipeline(self, start_date: Optional[str] = None, end_date: Optional[str] = None, time_horizon: str = "ensemble") -> Dict[str, Any]:
        """앙상블 파이프라인 실행 - 시점별 Market Regime 분석 기반"""
        print_section_header("🎯 앙상블 전략 파이프라인 실행")
        
        start_time = datetime.now()
        results = {
            "uuid": self.uuid,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "regime_periods": [],
            "pipeline_results": {},
            "performance_summary": {}
        }
        
        try:
            # 0단계: 공통 데이터 준비 (cleaner + scrapper를 한 번만 실행)
            print_subsection_header("📊 0단계: 공통 데이터 준비")
            print("🧹 데이터 폴더 정리 및 데이터 수집 중...")
            
            # 임시 orchestrator로 데이터 준비
            temp_orchestrator = Orchestrator(
                config_path="config/config_ensemble_volatile.json",  # 임의의 regime config 사용
                time_horizon=time_horizon,
                uuid=f"{self.uuid}_data_preparation"
            )
            
            # cleaner와 scrapper 실행
            cleaner_success = temp_orchestrator.run_single_stage("cleaner")
            if not cleaner_success:
                self.logger.error("❌ 공통 데이터 정리 실패")
                return results
                
            scrapper_success = temp_orchestrator.run_single_stage("scrapper")
            if not scrapper_success:
                self.logger.error("❌ 공통 데이터 수집 실패")
                return results
                
            print("✅ 공통 데이터 준비 완료")
            
            # 1단계: 시점별 Market Regime 분석
            print_subsection_header("🔍 1단계: 시점별 Market Regime 분석")
            regime_periods = self.analyze_regime_periods(start_date, end_date)
            results["regime_periods"] = regime_periods
            
            print(f"📊 분석된 기간: {len(regime_periods)}개")
            for period in regime_periods:
                print(f"  - {period['start_date']} ~ {period['end_date']}: {period['regime']}")
            
            # 2단계: 각 기간별 최적화 실행 (데이터는 이미 준비됨)
            print_subsection_header("🚀 2단계: 기간별 전략 최적화 실행")
            total_success = 0
            total_periods = len(regime_periods)
            
            for i, period in enumerate(regime_periods, 1):
                print(f"📊 [{i}/{total_periods}] {period['regime']} 기간 최적화 시작")
                print(f"   기간: {period['start_date']} ~ {period['end_date']}")
                
                period_success = self.run_period_optimization(period, time_horizon)
                
                if period_success:
                    total_success += 1
                    results["pipeline_results"][f"{period['regime']}_{period['start_date']}"] = {
                        "status": "success",
                        "regime": period['regime'],
                        "period": f"{period['start_date']} ~ {period['end_date']}",
                        "strategies_optimized": len(period.get('strategies', [])),
                        "execution_time": (datetime.now() - start_time).total_seconds()
                    }
                    print(f"✅ {period['regime']} 기간 최적화 완료")
                else:
                    results["pipeline_results"][f"{period['regime']}_{period['start_date']}"] = {
                        "status": "failed",
                        "regime": period['regime'],
                        "period": f"{period['start_date']} ~ {period['end_date']}",
                        "error": "기간별 최적화 실패"
                    }
                    print(f"❌ {period['regime']} 기간 최적화 실패")
            
            # 3단계: 종합 성과 분석
            print_subsection_header("📊 3단계: 종합 성과 분석")
            performance_summary = self.analyze_ensemble_performance(regime_periods)
            results["performance_summary"] = performance_summary
            
            # 4단계: 결과 저장
            print_subsection_header("💾 4단계: 결과 저장")
            self.save_ensemble_results(results)
            
            print(f"✅ 앙상블 전략 실행 완료!")
            print(f"📊 총 기간: {total_periods}개, 성공: {total_success}개")
            print(f"⏰ 실행 시간: {(datetime.now() - start_time).total_seconds():.2f}초")
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 파이프라인 실행 중 오류: {e}")
            results["error"] = str(e)
        
        results["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results["total_execution_time"] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    def analyze_ensemble_performance(self, regime_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """앙상블 성과 분석 - 모든 기간의 결과를 종합"""
        print("📊 앙상블 성과 분석 시작")
        
        try:
            performance_summary = {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_periods": len(regime_periods),
                "regime_summary": {},
                "overall_statistics": {}
            }
            
            # Regime별 통계
            regime_counts = {}
            regime_strategies = {}
            
            for period in regime_periods:
                regime = period["regime"]
                strategies = period["strategies"]
                
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                    regime_strategies[regime] = set()
                
                regime_counts[regime] += 1
                regime_strategies[regime].update(strategies)
            
            # Regime별 요약 생성
            for regime, count in regime_counts.items():
                performance_summary["regime_summary"][regime] = {
                    "period_count": int(count),  # numpy.int64를 int로 변환
                    "unique_strategies": list(regime_strategies[regime]),
                    "strategy_count": int(len(regime_strategies[regime]))  # numpy.int64를 int로 변환
                }
            
            # 전체 통계
            total_strategies = set()
            for strategies in regime_strategies.values():
                total_strategies.update(strategies)
            
            total_combinations = sum(len(s) for s in regime_strategies.values())
            avg_strategies = total_combinations / len(regime_periods) if regime_periods else 0
            
            performance_summary["overall_statistics"] = {
                "total_unique_strategies": int(len(total_strategies)),  # numpy.int64를 int로 변환
                "total_strategy_combinations": int(total_combinations),  # numpy.int64를 int로 변환
                "average_strategies_per_period": float(avg_strategies)  # numpy.float64를 float로 변환
            }
            
            print(f"✅ 앙상블 성과 분석 완료")
            print(f"   총 기간: {len(regime_periods)}개")
            print(f"   총 전략: {len(total_strategies)}개")
            
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 성과 분석 실패: {e}")
            return {}
    
    def analyze_performance(self, regime: str) -> Dict[str, Any]:
        """성과 분석"""
        try:
            regime_config = self.get_regime_config(regime)
            results_dir = Path(self.config["output"]["results_folder"])
            
            # 최신 결과 파일 찾기
            optimization_files = list(results_dir.glob("optimization_results_*.json"))
            evaluation_files = list(results_dir.glob("comprehensive_evaluation_*.txt"))
            portfolio_files = list(results_dir.glob("portfolio_optimization_*.json"))
            
            performance_summary = {
                "regime": regime,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "files_found": {
                    "optimization": len(optimization_files),
                    "evaluation": len(evaluation_files),
                    "portfolio": len(portfolio_files)
                }
            }
            
            # 최신 파일들 분석
            if optimization_files:
                latest_optimization = max(optimization_files, key=lambda x: x.stat().st_mtime)
                performance_summary["latest_optimization"] = str(latest_optimization)
            
            if evaluation_files:
                latest_evaluation = max(evaluation_files, key=lambda x: x.stat().st_mtime)
                performance_summary["latest_evaluation"] = str(latest_evaluation)
            
            if portfolio_files:
                latest_portfolio = max(portfolio_files, key=lambda x: x.stat().st_mtime)
                performance_summary["latest_portfolio"] = str(latest_portfolio)
            
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"❌ 성과 분석 실패: {e}")
            return {"error": str(e)}
    
    def save_ensemble_results(self, results: Dict[str, Any]):
        """앙상블 결과 저장"""
        try:
            results_dir = Path(self.config["output"]["results_folder"])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # numpy 타입을 JSON 직렬화 가능한 타입으로 변환
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy 타입들
                    return obj.item()
                else:
                    return obj
            
            # 결과를 직렬화 가능한 형태로 변환
            serializable_results = convert_numpy_types(results)
            
            # 결과 파일 저장
            results_file = results_dir / f"ensemble_results_{self.uuid}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # 요약 리포트 생성
            summary_file = results_dir / f"ensemble_summary_{self.uuid}.txt"
            self._generate_summary_report(results, summary_file)
            
            self.logger.info(f"✅ 앙상블 결과 저장 완료: {results_file}")
            self.logger.info(f"✅ 요약 리포트 생성 완료: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 결과 저장 실패: {e}")
    
    def _generate_summary_report(self, results: Dict[str, Any], output_file: Path):
        """요약 리포트 생성"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎯 앙상블 전략 실행 결과 요약\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"🆔 실행 UUID: {results['uuid']}\n")
            f.write(f"⏰ 시작 시간: {results['start_time']}\n")
            f.write(f"⏰ 종료 시간: {results.get('end_time', 'N/A')}\n")
            f.write(f"⏱️ 총 실행 시간: {results.get('total_execution_time', 0):.2f}초\n\n")
            
            # regime_periods 정보 출력
            if "regime_periods" in results and results["regime_periods"]:
                f.write("📊 시장 환경별 분석 기간:\n")
                for period in results["regime_periods"]:
                    f.write(f"  - {period['start_date']} ~ {period['end_date']}: {period['regime']}\n")
                f.write("\n")
            
            f.write("🚀 파이프라인 실행 결과:\n")
            if "pipeline_results" in results:
                for regime, pipeline_result in results["pipeline_results"].items():
                    f.write(f"  - {regime}: {pipeline_result['status']}\n")
                    if "execution_time" in pipeline_result:
                        f.write(f"    실행 시간: {pipeline_result['execution_time']:.2f}초\n")
            
            if "performance_summary" in results and results["performance_summary"]:
                f.write("\n📈 성과 분석:\n")
                perf = results["performance_summary"]
                f.write(f"  - 시장 환경: {perf.get('regime', 'N/A')}\n")
                f.write(f"  - 분석 날짜: {perf.get('analysis_date', 'N/A')}\n")
                if "files_found" in perf:
                    files = perf["files_found"]
                    f.write(f"  - 최적화 파일: {files.get('optimization', 0)}개\n")
                    f.write(f"  - 평가 파일: {files.get('evaluation', 0)}개\n")
                    f.write(f"  - 포트폴리오 파일: {files.get('portfolio', 0)}개\n")
    
    def run_backtest_ensemble(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """백테스팅 앙상블 전략"""
        print_section_header("🔄 백테스팅 앙상블 전략")
        
        backtest_results = {
            "uuid": self.uuid,
            "start_date": start_date,
            "end_date": end_date,
            "regime_history": [],
            "performance_by_regime": {},
            "overall_performance": {}
        }
        
        try:
            # 날짜 범위 생성
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            current_dt = start_dt
            while current_dt <= end_dt:
                current_date = current_dt.strftime("%Y-%m-%d")
                
                # 해당 날짜의 시장 환경 감지
                regime_detection = self.detect_market_regime(current_date)
                backtest_results["regime_history"].append(regime_detection)
                
                # 시장 환경별 성과 추적
                regime = regime_detection["regime"]
                if regime not in backtest_results["performance_by_regime"]:
                    backtest_results["performance_by_regime"][regime] = {
                        "detection_count": 0,
                        "total_confidence": 0.0,
                        "avg_confidence": 0.0
                    }
                
                backtest_results["performance_by_regime"][regime]["detection_count"] += 1
                backtest_results["performance_by_regime"][regime]["total_confidence"] += regime_detection["confidence"]
                
                current_dt += timedelta(days=1)
            
            # 평균 신뢰도 계산
            for regime, data in backtest_results["performance_by_regime"].items():
                if data["detection_count"] > 0:
                    data["avg_confidence"] = data["total_confidence"] / data["detection_count"]
            
            # 결과 저장
            results_dir = Path(self.config["output"]["results_folder"])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            backtest_file = results_dir / f"ensemble_backtest_{self.uuid}.json"
            with open(backtest_file, 'w', encoding='utf-8') as f:
                json.dump(backtest_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 백테스팅 완료: {backtest_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 백테스팅 실패: {e}")
            backtest_results["error"] = str(e)
        
        return backtest_results

    def view_results(self, uuid: Optional[str] = None, detailed: bool = False) -> Dict[str, Any]:
        """앙상블 결과 조회 (evaluator와 유사한 기능)"""
        print_section_header("📊 앙상블 결과 조회")
        
        try:
            results_dir = Path(self.config["output"]["results_folder"])
            
            # UUID가 지정되지 않은 경우 가장 최신 결과 파일 찾기
            if uuid:
                results_file = results_dir / f"ensemble_results_{uuid}.json"
                summary_file = results_dir / f"ensemble_summary_{uuid}.txt"
            else:
                result_files = list(results_dir.glob("ensemble_results_*.json"))
                if not result_files:
                    print("❌ 앙상블 결과 파일을 찾을 수 없습니다.")
                    return {}
                
                results_file = max(result_files, key=lambda x: x.stat().st_mtime)
                uuid_from_file = results_file.stem.replace("ensemble_results_", "")
                summary_file = results_dir / f"ensemble_summary_{uuid_from_file}.txt"
                print(f"📁 최신 결과 파일 사용: {results_file.name}")
            
            # JSON 결과 파일 로드
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                # 기본 정보 출력
                print(f"🆔 실행 UUID: {results_data.get('uuid', 'N/A')}")
                print(f"⏰ 실행 시간: {results_data.get('start_time', 'N/A')} ~ {results_data.get('end_time', 'N/A')}")
                print(f"⏱️ 총 실행 시간: {results_data.get('total_execution_time', 0):.2f}초")
                
                # 시장 환경별 기간 정보
                if "regime_periods" in results_data:
                    print(f"\n📊 분석된 시장 환경별 기간: {len(results_data['regime_periods'])}개")
                    for i, period in enumerate(results_data['regime_periods'], 1):
                        print(f"  {i}. {period['start_date']} ~ {period['end_date']}: {period['regime']}")
                
                # 파이프라인 실행 결과
                if "pipeline_results" in results_data:
                    print(f"\n🚀 파이프라인 실행 결과:")
                    success_count = 0
                    for regime, result in results_data['pipeline_results'].items():
                        status_emoji = "✅" if result['status'] == 'success' else "❌"
                        print(f"  {status_emoji} {regime}: {result['status']}")
                        if result['status'] == 'success':
                            success_count += 1
                        if "execution_time" in result:
                            print(f"     실행 시간: {result['execution_time']:.2f}초")
                    
                    print(f"\n📊 성공률: {success_count}/{len(results_data['pipeline_results'])} ({success_count/len(results_data['pipeline_results'])*100:.1f}%)")
                
                # 상세 정보 출력
                if detailed:
                    print(f"\n📄 상세 성과 분석:")
                    if "performance_summary" in results_data and results_data["performance_summary"]:
                        perf = results_data["performance_summary"]
                        print(f"  - 분석 날짜: {perf.get('analysis_date', 'N/A')}")
                        if "files_found" in perf:
                            files = perf["files_found"]
                            print(f"  - 최적화 결과 파일: {files.get('optimization', 0)}개")
                            print(f"  - 평가 결과 파일: {files.get('evaluation', 0)}개")
                            print(f"  - 포트폴리오 파일: {files.get('portfolio', 0)}개")
                
                # 요약 리포트가 있다면 일부 내용 출력
                if summary_file.exists():
                    print(f"\n📋 요약 리포트: {summary_file}")
                    print("     (전체 내용을 보려면 파일을 직접 확인하세요)")
                else:
                    print(f"\n⚠️ 요약 리포트 파일이 없습니다: {summary_file}")
                
                return results_data
                
            else:
                print(f"❌ 결과 파일을 찾을 수 없습니다: {results_file}")
                return {}
        
        except Exception as e:
            print(f"❌ 결과 조회 중 오류: {e}")
            return {}
    
    def list_all_results(self) -> List[Dict[str, Any]]:
        """모든 앙상블 결과 목록 조회"""
        print_section_header("📂 모든 앙상블 결과 목록")
        
        try:
            results_dir = Path(self.config["output"]["results_folder"])
            result_files = list(results_dir.glob("ensemble_results_*.json"))
            
            if not result_files:
                print("❌ 앙상블 결과 파일이 없습니다.")
                return []
            
            results_list = []
            
            print(f"📊 총 {len(result_files)}개의 결과 파일 발견:")
            print("-" * 80)
            print(f"{'번호':<4} {'UUID':<25} {'실행시간':<20} {'상태':<10}")
            print("-" * 80)
            
            for i, file_path in enumerate(sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True), 1):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    uuid = data.get('uuid', 'N/A')
                    start_time = data.get('start_time', 'N/A')
                    
                    # 성공/실패 상태 계산
                    if "pipeline_results" in data:
                        success_count = sum(1 for result in data['pipeline_results'].values() if result['status'] == 'success')
                        total_count = len(data['pipeline_results'])
                        status = f"{success_count}/{total_count}"
                    else:
                        status = "N/A"
                    
                    print(f"{i:<4} {uuid:<25} {start_time:<20} {status:<10}")
                    
                    results_list.append({
                        "file_path": str(file_path),
                        "uuid": uuid,
                        "start_time": start_time,
                        "status": status
                    })
                    
                except Exception as e:
                    print(f"{i:<4} {'ERROR':<25} {'파일 읽기 실패':<20} {'ERROR':<10}")
            
            print("-" * 80)
            return results_list
            
        except Exception as e:
            print(f"❌ 결과 목록 조회 중 오류: {e}")
            return []
    
    def compare_results(self, uuid1: str, uuid2: str) -> Dict[str, Any]:
        """두 앙상블 결과 비교"""
        print_section_header(f"🔄 앙상블 결과 비교: {uuid1} vs {uuid2}")
        
        try:
            results_dir = Path(self.config["output"]["results_folder"])
            
            # 첫 번째 결과 로드
            file1 = results_dir / f"ensemble_results_{uuid1}.json"
            file2 = results_dir / f"ensemble_results_{uuid2}.json"
            
            if not file1.exists():
                print(f"❌ 첫 번째 결과 파일을 찾을 수 없습니다: {file1}")
                return {}
            
            if not file2.exists():
                print(f"❌ 두 번째 결과 파일을 찾을 수 없습니다: {file2}")
                return {}
            
            with open(file1, 'r', encoding='utf-8') as f:
                results1 = json.load(f)
            
            with open(file2, 'r', encoding='utf-8') as f:
                results2 = json.load(f)
            
            # 비교 결과
            comparison = {
                "uuid1": uuid1,
                "uuid2": uuid2,
                "execution_time_comparison": {
                    "results1": results1.get('total_execution_time', 0),
                    "results2": results2.get('total_execution_time', 0)
                },
                "success_rate_comparison": {},
                "period_count_comparison": {
                    "results1": len(results1.get('regime_periods', [])),
                    "results2": len(results2.get('regime_periods', []))
                }
            }
            
            # 성공률 비교
            for results_key, results_data in [("results1", results1), ("results2", results2)]:
                if "pipeline_results" in results_data:
                    success_count = sum(1 for result in results_data['pipeline_results'].values() if result['status'] == 'success')
                    total_count = len(results_data['pipeline_results'])
                    comparison["success_rate_comparison"][results_key] = {
                        "success_count": success_count,
                        "total_count": total_count,
                        "success_rate": success_count / total_count if total_count > 0 else 0
                    }
            
            # 결과 출력
            print(f"📊 실행 시간 비교:")
            print(f"  - {uuid1}: {comparison['execution_time_comparison']['results1']:.2f}초")
            print(f"  - {uuid2}: {comparison['execution_time_comparison']['results2']:.2f}초")
            
            print(f"\n📊 분석 기간 수 비교:")
            print(f"  - {uuid1}: {comparison['period_count_comparison']['results1']}개")
            print(f"  - {uuid2}: {comparison['period_count_comparison']['results2']}개")
            
            print(f"\n📊 성공률 비교:")
            for results_key, uuid_val in [("results1", uuid1), ("results2", uuid2)]:
                if results_key in comparison["success_rate_comparison"]:
                    data = comparison["success_rate_comparison"][results_key]
                    print(f"  - {uuid_val}: {data['success_count']}/{data['total_count']} ({data['success_rate']*100:.1f}%)")
            
            return comparison
            
        except Exception as e:
            print(f"❌ 결과 비교 중 오류: {e}")
            return {}


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="앙상블 전략 실행 및 결과 조회")
    parser.add_argument("--config", default="config/config_ensemble.json", help="앙상블 설정 파일")
    parser.add_argument("--market-sensor-config", default="config/config_macro.json", help="Market Sensor 설정 파일")
    parser.add_argument("--mode", choices=["run", "backtest", "view", "list", "compare"], default="run", help="실행 모드")
    parser.add_argument("--start-date", help="백테스팅 시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="백테스팅 종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--uuid", help="실행 UUID 또는 조회할 UUID")
    parser.add_argument("--uuid2", help="비교할 두 번째 UUID (compare 모드에서 사용)")
    parser.add_argument("--time-horizon", default="ensemble", help="데이터 디렉토리 (기본값: ensemble)")
    parser.add_argument("--detailed", action="store_true", help="상세 정보 출력 (view 모드)")
    
    args = parser.parse_args()
    
    # 앙상블 전략 초기화
    ensemble = EnsembleStrategy(
        config_path=args.config,
        market_sensor_config=args.market_sensor_config,
        uuid=args.uuid
    )
    
    if args.mode == "run":
        # 일반 실행
        results = ensemble.run_ensemble_pipeline(time_horizon=args.time_horizon)
        print("🎉 앙상블 전략 실행 완료!")
        
    elif args.mode == "backtest":
        # 백테스팅
        if not args.start_date or not args.end_date:
            print("❌ 백테스팅 모드에서는 --start-date와 --end-date가 필요합니다.")
            return
        
        results = ensemble.run_backtest_ensemble(args.start_date, args.end_date)
        print("🎉 백테스팅 완료!")
        
    elif args.mode == "view":
        # 결과 조회
        results = ensemble.view_results(uuid=args.uuid, detailed=args.detailed)
        
    elif args.mode == "list":
        # 결과 목록 조회
        results_list = ensemble.list_all_results()
        
    elif args.mode == "compare":
        # 결과 비교
        if not args.uuid or not args.uuid2:
            print("❌ 비교 모드에서는 --uuid와 --uuid2가 모두 필요합니다.")
            return
        
        comparison = ensemble.compare_results(args.uuid, args.uuid2)


if __name__ == "__main__":
    main() 