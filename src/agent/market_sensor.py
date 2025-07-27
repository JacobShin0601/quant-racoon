#!/usr/bin/env python3
"""
시장 센서 - 통합 시장 분석 시스템
기본 분석, 고도화된 분석, LLM 분석을 통합하여 제공
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import os
import optuna
from dataclasses import dataclass
from enum import Enum
import uuid
import joblib
from pathlib import Path
import warnings

# yfinance 디버그 로그 억제
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# 프로젝트 루트 경로 추가
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from actions.global_macro import (
    GlobalMacroDataCollector,
    MacroSectorAnalyzer,
    HyperparamTuner,
    MarketRegimeValidator,
    MarketRegime,
    MarketCondition,
    MacroAnalysis,
    SectorStrength,
)
from actions.random_forest import MarketRegimeRF  # Random Forest 모델 추가
from .enhancements import (
    RLMFRegimeAdaptation,
    MultiLayerConfidenceSystem,
    DynamicRegimeSwitchingDetector,
    LLMPrivilegedInformationSystem,
    LLMAPIIntegration,
)


@dataclass
class MarketAnalysisResult:
    """시장 분석 결과 통합 데이터 클래스"""

    # 기본 분석 결과
    current_regime: MarketRegime
    confidence: float
    probabilities: Dict[str, float]

    # 매크로 분석 결과
    macro_analysis: MacroAnalysis

    # 하이퍼파라미터 최적화 결과
    optimal_params: Dict[str, Any]
    optimization_performance: Dict[str, float]

    # 검증 결과
    validation_results: Dict[str, Any]

    # 고급 분석 결과
    rlmf_analysis: Dict[str, Any]
    confidence_analysis: Dict[str, Any]
    regime_detection: Dict[str, Any]
    llm_insights: Dict[str, Any]
    llm_api_insights: Dict[str, Any]

    # 최종 신뢰도 및 추천
    final_confidence: Dict[str, Any]
    enhanced_recommendations: Dict[str, Any]

    # 메타데이터
    session_uuid: str
    timestamp: datetime
    data_period: str
    analysis_type: str

    # 원본 분류 결과 (고급 분석 포함)
    classification_result: Dict[str, Any] = None


class MarketSensor:
    """통합 시장 분석 시스템 - 실행 인터페이스 (고도화된 버전)"""

    def __init__(
        self,
        data_dir: str = "data/macro",
        config_path: str = "config/config_macro.json",
        enable_llm_api: bool = False,
        llm_config: Dict[str, Any] = None,
        use_cached_data: bool = False,
        use_cached_optimization: bool = False,
        cache_days: int = 1,
        use_random_forest: bool = True,  # Random Forest 모델 사용 여부
        retrain_rf_model: bool = False,  # Random Forest 모델 재학습 여부
    ):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.config_path = config_path

        # 세션 UUID 생성
        self.session_uuid = str(uuid.uuid4())
        self.logger.info(f"MarketSensor 초기화 - Session UUID: {self.session_uuid}")

        # 핵심 컴포넌트들 초기화 (UUID 전달)
        self.macro_collector = GlobalMacroDataCollector(self.session_uuid, config_path)
        self.hyperparam_tuner = HyperparamTuner(config_path, self.session_uuid)
        self.macro_analyzer = MacroSectorAnalyzer(data_dir, self.session_uuid)
        self.regime_validator = MarketRegimeValidator(self.session_uuid)

        # 고도화된 시스템 컴포넌트들 초기화
        self.rlmf_adaptation = RLMFRegimeAdaptation()
        self.confidence_system = MultiLayerConfidenceSystem()
        self.regime_detector = DynamicRegimeSwitchingDetector()
        self.llm_privileged_system = LLMPrivilegedInformationSystem()

        # LLM API 통합 시스템 (선택적 활성화)
        self.llm_api_system = None
        self.llm_config = llm_config
        if enable_llm_api and llm_config:
            try:
                self.llm_api_system = LLMAPIIntegration(llm_config)
            except Exception as e:
                pass

        # Random Forest 모델 초기화
        self.rf_model = None
        self.use_random_forest = use_random_forest
        self.retrain_rf_model = retrain_rf_model

        if use_random_forest:
            try:
                self.rf_model = MarketRegimeRF(verbose=True, config_path=config_path)
            except Exception as e:
                self.use_random_forest = False

        # 캐시 설정
        self.use_cached_data = use_cached_data
        self.use_cached_optimization = use_cached_optimization
        self.cache_days = cache_days

        # 최적화 파라미터 저장 변수
        self.optimal_params = None
        self.optimization_performance = None

        # 경고 무시 설정
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

        # 초기화 완료
        pass

    def _load_cached_data(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """캐시된 매크로 데이터 로드"""
        try:
            # 가장 최근 캐시 파일 찾기
            cache_dir = self.data_dir
            if not os.path.exists(cache_dir):
                return pd.DataFrame(), {}, {}

            # 세션별 캐시 디렉토리 찾기
            session_dirs = [
                d
                for d in os.listdir(cache_dir)
                if os.path.isdir(os.path.join(cache_dir, d))
            ]
            if not session_dirs:
                return pd.DataFrame(), {}, {}

            # 가장 최근 세션 찾기
            latest_session = max(
                session_dirs, key=lambda x: os.path.getctime(os.path.join(cache_dir, x))
            )
            session_path = os.path.join(cache_dir, latest_session)

            # 캐시 유효성 검사
            cache_time = datetime.fromtimestamp(os.path.getctime(session_path))
            if (datetime.now() - cache_time).days > self.cache_days:
                return pd.DataFrame(), {}, {}

            # 데이터 파일들 로드
            spy_data = pd.DataFrame()
            macro_data = {}
            sector_data = {}

            # SPY 데이터 로드
            spy_file = os.path.join(session_path, "spy_data.csv")
            if os.path.exists(spy_file):
                spy_data = pd.read_csv(spy_file, index_col=0, parse_dates=True)

            # 매크로 데이터 로드
            for file in os.listdir(session_path):
                if (
                    file.endswith(".csv")
                    and not file.startswith("spy_")
                    and not file.endswith("_sector.csv")
                ):
                    symbol = file.replace(".csv", "")
                    file_path = os.path.join(session_path, file)
                    macro_data[symbol] = pd.read_csv(
                        file_path, index_col=0, parse_dates=True
                    )

            # 섹터 데이터 로드
            for file in os.listdir(session_path):
                if file.endswith("_sector.csv"):
                    sector = file.replace("_sector.csv", "")
                    file_path = os.path.join(session_path, file)
                    sector_data[sector] = pd.read_csv(
                        file_path, index_col=0, parse_dates=True
                    )

            return spy_data, macro_data, sector_data

        except Exception as e:
            return pd.DataFrame(), {}, {}

    def _load_cached_optimization(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """캐시된 최적화 결과 로드"""
        try:
            # 최적화 결과 디렉토리들 확인
            optimization_dirs = [
                "results/macro_optimization",
                "results/macro/basic",
                "results/macro/enhanced",
            ]

            best_params = {}
            test_performance = {}
            latest_time = 0
            latest_file = None

            for optimization_dir in optimization_dirs:
                if not os.path.exists(optimization_dir):
                    continue

                # UUID 기반 디렉토리에서 찾기
                for item in os.listdir(optimization_dir):
                    item_path = os.path.join(optimization_dir, item)
                    if os.path.isdir(item_path):
                        # best_params.json 파일 찾기
                        best_params_file = os.path.join(item_path, "best_params.json")
                        performance_file = os.path.join(
                            item_path, "performance_summary.json"
                        )

                        if os.path.exists(best_params_file):
                            file_time = os.path.getctime(best_params_file)

                            # 캐시 유효성 검사
                            if (
                                datetime.now() - datetime.fromtimestamp(file_time)
                            ).days <= self.cache_days:
                                # JSON 파일 유효성 검사 및 로드
                                try:
                                    with open(
                                        best_params_file, "r", encoding="utf-8"
                                    ) as f:
                                        temp_best_params = json.load(f)

                                    # performance_summary.json도 확인
                                    if os.path.exists(performance_file):
                                        with open(
                                            performance_file, "r", encoding="utf-8"
                                        ) as f:
                                            performance_data = json.load(f)
                                        temp_test_performance = performance_data.get(
                                            "test_performance", {}
                                        )

                                        # 유효한 파일인 경우에만 업데이트
                                        if (
                                            temp_best_params
                                            and temp_test_performance
                                            and file_time > latest_time
                                        ):
                                            latest_time = file_time
                                            latest_file = best_params_file
                                            best_params = temp_best_params
                                            test_performance = temp_test_performance
                                            self.logger.info(
                                                f"best_params.json 로드 성공: {len(best_params)}개 파라미터"
                                            )
                                            self.logger.info(
                                                f"performance_summary.json 로드 성공: {len(test_performance)}개 지표"
                                            )
                                            self.logger.info(
                                                f"Sharpe Ratio: {test_performance.get('sharpe_ratio', 'N/A')}"
                                            )
                                    else:
                                        self.logger.warning(
                                            f"performance_summary.json 파일이 없습니다: {performance_file}"
                                        )
                                except Exception as e:
                                    self.logger.error(f"JSON 파일 로드 실패: {e}")
                                    self.logger.error(f"파일 경로: {best_params_file}")
                                    continue

                # 일반 JSON 파일에서도 찾기 (이전 버전 호환성 + 폴백 저장 파일)
                optimization_files = [
                    f
                    for f in os.listdir(optimization_dir)
                    if f.endswith(".json")
                    and ("optimization_results" in f or "hyperparam_optimization" in f)
                ]
                for file in optimization_files:
                    file_path = os.path.join(optimization_dir, file)
                    file_time = os.path.getctime(file_path)

                    # 캐시 유효성 검사
                    if (
                        datetime.now() - datetime.fromtimestamp(file_time)
                    ).days <= self.cache_days:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                optimization_data = json.load(f)

                            # best_params가 있는 유효한 파일인지 확인
                            if (
                                "best_params" in optimization_data
                                and optimization_data["best_params"]
                            ):
                                temp_best_params = optimization_data.get(
                                    "best_params", {}
                                )
                                temp_test_performance = optimization_data.get(
                                    "test_performance", {}
                                )

                                # 더 최신 파일인 경우 업데이트
                                if file_time > latest_time and temp_best_params:
                                    latest_time = file_time
                                    latest_file = file_path
                                    best_params = temp_best_params
                                    test_performance = temp_test_performance
                                    self.logger.info(
                                        f"폴백 최적화 결과 파일 로드: {file}"
                                    )
                                    self.logger.info(f"파라미터 수: {len(best_params)}")
                                    self.logger.info(
                                        f"최고 점수: {optimization_data.get('best_value', 'N/A')}"
                                    )
                                    self.logger.info(
                                        f"Trial 수: {optimization_data.get('n_trials', 'N/A')}"
                                    )
                                    if test_performance:
                                        self.logger.info(
                                            f"Test 성과: {list(test_performance.keys())}"
                                        )
                        except Exception as e:
                            self.logger.warning(f"JSON 파일 로드 실패: {file} - {e}")
                            continue

            if best_params and test_performance:
                self.logger.info(f"캐시된 최적화 결과 로드 완료: {latest_file}")
                self.logger.info(f"로드된 성과 지표: {list(test_performance.keys())}")
                return best_params, test_performance
            else:
                self.logger.info("유효한 캐시된 최적화 결과를 찾을 수 없습니다.")
                return {}, {}

        except Exception as e:
            self.logger.warning(f"캐시된 최적화 결과 로드 실패: {e}")
            return {}, {}

    def _copy_data_to_macro_dir(self):
        """UUID 디렉토리의 데이터를 data/macro로 복사"""
        try:
            import shutil

            # 세션 UUID 디렉토리 경로
            session_dir = f"data/macro/{self.session_uuid}"
            if not os.path.exists(session_dir):
                self.logger.warning(f"세션 디렉토리가 존재하지 않습니다: {session_dir}")
                return

            # CSV 파일들 복사
            csv_count = 0
            for file in os.listdir(session_dir):
                if file.endswith(".csv"):
                    src_path = os.path.join(session_dir, file)
                    dst_path = os.path.join("data/macro", file)
                    shutil.copy2(src_path, dst_path)
                    csv_count += 1
                    self.logger.info(f"📄 복사됨: {file}")

            # JSON 메타데이터 파일 복사
            metadata_src = os.path.join(session_dir, "metadata.json")
            metadata_dst = os.path.join("data/macro", "metadata.json")
            if os.path.exists(metadata_src):
                shutil.copy2(metadata_src, metadata_dst)
                self.logger.info("📄 복사됨: metadata.json")

            self.logger.info(
                f"✅ 매크로 데이터 복사 완료 ({csv_count}개 CSV 파일 + 메타데이터)"
            )

        except Exception as e:
            self.logger.error(f"매크로 데이터 복사 중 오류: {e}")

    def run_basic_analysis(
        self, output_dir: str = "results/macro/basic", verbose: bool = True
    ) -> MarketAnalysisResult:
        """
        기본 시장 분석 실행 (GlobalMacroDataCollector 기반)
        - 매크로 데이터 수집
        - 매크로 & 섹터 분석
        - 하이퍼파라미터 최적화
        - 시장 체제 분류
        - 검증
        """
        if verbose:
            print("📊 기본 시장 분석 시작 (GlobalMacroDataCollector 기반)")

        try:
            # 1. 매크로 데이터 수집 또는 캐시 로드
            if self.use_cached_data:
                spy_data, macro_data, sector_data = self._load_cached_data()

                if spy_data.empty or not macro_data:
                    spy_data, macro_data, sector_data = (
                        self.macro_collector.collect_all_data()
                    )
            else:
                spy_data, macro_data, sector_data = (
                    self.macro_collector.collect_all_data()
                )

                # 데이터 다운로드 직후 즉시 data/macro로 파일 복사
                if not spy_data.empty and macro_data:
                    self._copy_data_to_macro_dir()

            if spy_data.empty or not macro_data:
                raise ValueError("매크로 데이터 수집 실패")

            # 2. 매크로 & 섹터 종합 분석
            macro_analysis = self.macro_analyzer.get_comprehensive_analysis(
                spy_data=spy_data, macro_data=macro_data, sector_data=sector_data
            )

            # 2-1. 상세 매크로 환경 분석
            detailed_macro_analysis = self.macro_analyzer.analyze_macro_environment(
                macro_data
            )

            # 2-2. 섹터 로테이션 분석
            sector_analysis = self.macro_analyzer.analyze_sector_rotation(sector_data)

            # 2-3. 섹터 추천 생성
            sector_recommendations = (
                self.macro_analyzer.generate_sector_recommendations(
                    macro_analysis.market_condition, sector_analysis
                )
            )

            # 매크로 분석에 상세 정보 추가
            macro_analysis.key_indicators.update(detailed_macro_analysis)
            macro_analysis.sector_rotation.update(sector_analysis)
            macro_analysis.recommendations.update(sector_recommendations)

            # 3. 하이퍼파라미터 최적화 또는 캐시 로드
            if self.use_cached_optimization:
                self.optimal_params, self.optimization_performance = (
                    self._load_cached_optimization()
                )

                if not self.optimal_params:
                    raise ValueError(
                        "캐시된 최적화 결과가 없습니다. 새로 최적화하려면 --use-cached-optimization 옵션을 제거하세요."
                    )
            else:
                optimization_results = self.hyperparam_tuner.optimize_hyperparameters(
                    spy_data=spy_data, macro_data=macro_data
                )
                self.optimal_params = optimization_results["best_params"]
                self.optimization_performance = optimization_results["test_performance"]

                # Buy & Hold 성과 계산 추가
                buyhold_performance = self._calculate_buyhold_performance(spy_data)
                self.optimization_performance.update(buyhold_performance)

            # 4. 시장 체제 분류 (최적화된 파라미터 사용)
            regime_classification = self._classify_market_regime_with_optimal_params(
                spy_data, macro_data, self.optimal_params
            )

            current_regime = regime_classification["current_regime"]
            base_confidence = regime_classification["confidence"]
            probabilities = regime_classification["probabilities"]

            # 4-1. RLMF 분석 (Reinforcement Learning Market Feedback)
            rlmf_analysis = self._perform_rlmf_analysis(macro_analysis, current_regime)

            # 4-2. 다층 신뢰도 계산 (confidence_system.py 사용)
            technical_confidence = base_confidence
            macro_confidence = macro_analysis.confidence
            stat_arb_confidence = rlmf_analysis.get("statistical_arbitrage", {}).get(
                "confidence", 0.5
            )
            rlmf_confidence = (
                np.mean(list(rlmf_analysis.get("market_feedback", {}).values()))
                if rlmf_analysis.get("market_feedback")
                else 0.5
            )
            cross_val_confidence = self._calculate_cross_validation_confidence(
                spy_data, regime_classification
            )

            comprehensive_confidence = (
                self.confidence_system.calculate_comprehensive_confidence(
                    technical_confidence,
                    macro_confidence,
                    stat_arb_confidence,
                    rlmf_confidence,
                    cross_val_confidence,
                )
            )

            # 최종 신뢰도는 다층 신뢰도 시스템의 결과 사용
            confidence = comprehensive_confidence.get(
                "adjusted_confidence", base_confidence
            )

            # 5. 검증 수행
            validation_results = self._perform_validation(
                spy_data, macro_data, regime_classification
            )

            # 6. 결과 생성
            result = MarketAnalysisResult(
                current_regime=current_regime,
                confidence=confidence,
                probabilities=probabilities,
                macro_analysis=macro_analysis,
                optimal_params=self.optimal_params,
                optimization_performance=self.optimization_performance,
                validation_results=validation_results,
                rlmf_analysis=rlmf_analysis,
                confidence_analysis=comprehensive_confidence,
                regime_detection={},
                llm_insights={},
                llm_api_insights={},
                final_confidence={"final_confidence": confidence},
                enhanced_recommendations=self._generate_basic_recommendations(
                    macro_analysis, current_regime
                ),
                session_uuid=self.session_uuid,
                timestamp=datetime.now(),
                data_period="2_years",
                classification_result=regime_classification,
                analysis_type="basic",
            )

            # 7. 결과 저장
            self._save_analysis_result(result, output_dir, verbose)

            if verbose:
                self._print_basic_summary(result)

            return result

        except Exception as e:
            self.logger.error(f"기본 분석 실패: {e}")
            if verbose:
                print(f"❌ 기본 분석 실패: {e}")
            return None

    def run_enhanced_analysis(
        self, output_dir: str = "results/macro/enhanced", verbose: bool = True
    ) -> MarketAnalysisResult:
        """
        고도화된 시장 분석 실행 (기본 분석 + LLM + 고급 기능)
        """
        if verbose:
            print("🚀 고도화된 시장 분석 시작")

        try:
            # 1. 기본 분석 수행
            basic_result = self.run_basic_analysis(output_dir, verbose=False)
            if basic_result is None:
                raise ValueError("기본 분석 실패")

            # 2. 고급 분석 수행
            if verbose:
                print("🧠 고급 분석 수행 중...")

            # RLMF 적응 분석 (매크로 데이터 전달)
            rlmf_analysis = self._perform_rlmf_analysis(
                basic_result.macro_analysis, basic_result.current_regime
            )

            # 다층 신뢰도 분석
            confidence_analysis = self._perform_confidence_analysis(
                basic_result, rlmf_analysis
            )

            # Regime 전환 감지
            regime_detection = self._perform_regime_detection(
                basic_result.macro_analysis
            )

            # LLM 특권 정보 분석
            llm_insights = self._perform_llm_analysis(
                basic_result.macro_analysis, basic_result.current_regime
            )

            # LLM API 통합 분석 (활성화된 경우)
            llm_api_insights = {}
            if self.llm_api_system:
                # 기존 분석 결과를 딕셔너리로 변환
                analysis_results = {
                    "current_regime": basic_result.current_regime.value,
                    "confidence": basic_result.confidence,
                    "probabilities": basic_result.probabilities,
                    "macro_analysis": basic_result.macro_analysis,
                    "optimal_params": basic_result.optimal_params,
                    "optimization_performance": basic_result.optimization_performance,
                    "validation_results": basic_result.validation_results,
                    "rlmf_analysis": rlmf_analysis,
                    "confidence_analysis": confidence_analysis,
                    "regime_detection": regime_detection,
                    "llm_insights": llm_insights,
                }

                llm_api_insights = self._perform_llm_api_analysis(
                    basic_result.macro_analysis,
                    basic_result.current_regime,
                    analysis_results,
                )

            # 3. 종합 신뢰도 계산
            final_confidence = self._calculate_final_confidence(
                basic_result, rlmf_analysis, confidence_analysis, regime_detection
            )

            # 4. 고도화된 추천 생성
            enhanced_recommendations = self._generate_enhanced_recommendations(
                basic_result, rlmf_analysis, regime_detection, llm_insights
            )

            # 5. 고도화된 결과 생성
            enhanced_result = MarketAnalysisResult(
                current_regime=basic_result.current_regime,
                confidence=basic_result.confidence,
                probabilities=basic_result.probabilities,
                macro_analysis=basic_result.macro_analysis,
                optimal_params=basic_result.optimal_params,
                optimization_performance=basic_result.optimization_performance,
                validation_results=basic_result.validation_results,
                rlmf_analysis=rlmf_analysis,
                confidence_analysis=confidence_analysis,
                regime_detection=regime_detection,
                llm_insights=llm_insights,
                llm_api_insights=llm_api_insights,
                final_confidence=final_confidence,
                enhanced_recommendations=enhanced_recommendations,
                session_uuid=self.session_uuid,
                timestamp=datetime.now(),
                data_period="2_years",
                analysis_type="enhanced",
            )

            # 6. 결과 저장
            self._save_analysis_result(enhanced_result, output_dir, verbose)

            if verbose:
                print("✅ 고도화된 시장 분석 완료!")
                self._print_enhanced_summary(enhanced_result)

            return enhanced_result

        except Exception as e:
            self.logger.error(f"고도화된 분석 실패: {e}")
            if verbose:
                print(f"❌ 고도화된 분석 실패: {e}")
            return None

    def _print_detailed_regime_analysis(self, classification_result: Dict[str, Any]):
        """시장 체제 분석 결과를 상세히 출력 (분리된 점수와 확률)"""
        print("\n" + "=" * 80)
        print("🔍 상세 시장 체제 분석 결과")
        print("=" * 80)

        # 1. 기술적 지표 점수 분석
        print("\n📊 1. 기술적 지표 점수 분석")
        print("-" * 50)

        if "scores" in classification_result:
            scores = classification_result["scores"]
            total_score = classification_result.get("total_score", scores.sum(axis=1))

            # 최근 5일간의 점수 추이
            recent_scores = scores.tail(5)
            recent_total = total_score.tail(5)

            print("📈 최근 5일간 점수 추이:")
            for i, (date, row) in enumerate(recent_scores.iterrows()):
                date_str = (
                    date.strftime("%Y-%m-%d")
                    if hasattr(date, "strftime")
                    else str(date)
                )
                total = recent_total.iloc[i] if i < len(recent_total) else 0
                print(f"  {date_str}:")
                print(f"    • 트렌드 점수: {row.get('trend_score', 0):.3f}")
                print(f"    • 모멘텀 점수: {row.get('momentum_score', 0):.3f}")
                print(f"    • 변동성 점수: {row.get('volatility_score', 0):.3f}")
                print(f"    • 매크로 점수: {row.get('macro_score', 0):.3f}")
                print(f"    • 거래량 점수: {row.get('volume_score', 0):.3f}")
                print(f"    • 지지/저항 점수: {row.get('sr_score', 0):.3f}")
                print(f"    • 총점: {total:.3f}")
                print()

            # 점수 통계
            print("📊 점수 통계 (전체 기간):")
            for col in scores.columns:
                if col in scores:
                    mean_score = scores[col].mean()
                    std_score = scores[col].std()
                    min_score = scores[col].min()
                    max_score = scores[col].max()
                    print(
                        f"  • {col}: 평균={mean_score:.3f}, 표준편차={std_score:.3f}, 범위=[{min_score:.3f}, {max_score:.3f}]"
                    )

        # 2. Random Forest 확률 분석
        print("\n🎯 2. Random Forest 확률 분석")
        print("-" * 50)

        if "probabilities_series" in classification_result:
            prob_series = classification_result["probabilities_series"]

            # 최근 5일간의 확률 추이
            print("📈 최근 5일간 확률 추이:")
            for i in range(min(5, len(prob_series["trending_up"]))):
                idx = -(i + 1)  # 최근부터 역순
                print(f"  {i+1}일 전:")
                print(f"    • TRENDING_UP: {prob_series['trending_up'][idx]:.1%}")
                print(f"    • TRENDING_DOWN: {prob_series['trending_down'][idx]:.1%}")
                print(f"    • VOLATILE: {prob_series['volatile'][idx]:.1%}")
                print(f"    • SIDEWAYS: {prob_series['sideways'][idx]:.1%}")
                print()

            # 확률 통계
            print("📊 확률 통계 (전체 기간):")
            for regime, probs in prob_series.items():
                mean_prob = np.mean(probs)
                std_prob = np.std(probs)
                max_prob = np.max(probs)
                min_prob = np.min(probs)
                print(
                    f"  • {regime.upper()}: 평균={mean_prob:.1%}, 표준편차={std_prob:.1%}, 범위=[{min_prob:.1%}, {max_prob:.1%}]"
                )

        # 3. 현재 상태 분석
        print("\n🎯 3. 현재 시장 상태 분석")
        print("-" * 50)

        current_regime = classification_result.get("current_regime", "UNKNOWN")
        confidence = classification_result.get("confidence", 0)
        probabilities = classification_result.get("probabilities", {})

        print(f"현재 체제: {current_regime}")
        print(f"신뢰도: {confidence:.3f}")
        print("\n체제별 확률:")
        for regime, prob in probabilities.items():
            print(f"  • {regime}: {prob:.1%}")

        # 4. 신뢰도 분석
        print("\n🔍 4. 신뢰도 분석")
        print("-" * 50)

        if confidence < 0.3:
            confidence_level = "매우 낮음"
            recommendation = "분류 결과를 신중하게 해석하세요"
        elif confidence < 0.5:
            confidence_level = "낮음"
            recommendation = "추가 지표 확인이 필요합니다"
        elif confidence < 0.7:
            confidence_level = "보통"
            recommendation = "일반적인 신뢰도입니다"
        elif confidence < 0.9:
            confidence_level = "높음"
            recommendation = "신뢰할 만한 분류입니다"
        else:
            confidence_level = "매우 높음"
            recommendation = "매우 신뢰할 만한 분류입니다"

        print(f"신뢰도 수준: {confidence_level}")
        print(f"권장사항: {recommendation}")

        # 5. 개선 제안
        print("\n💡 5. 개선 제안")
        print("-" * 50)

        if "scores" in classification_result:
            scores = classification_result["scores"]

            # 가장 낮은 점수 지표 찾기
            if not scores.empty:
                recent_scores = scores.tail(1).iloc[0]
                min_score_key = recent_scores.idxmin()
                min_score_value = recent_scores[min_score_key]

                print(f"가장 낮은 점수 지표: {min_score_key} ({min_score_value:.3f})")

                if min_score_value < -0.5:
                    print("→ 해당 지표의 데이터 품질이나 파라미터 조정이 필요합니다")
                elif min_score_value < 0:
                    print("→ 해당 지표의 성능 개선이 필요합니다")
                else:
                    print("→ 모든 지표가 양호한 상태입니다")

        print("\n" + "=" * 80)

    def _classify_market_regime_with_optimal_params(
        self,
        spy_data: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame],
        optimal_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """최적화된 파라미터로 시장 체제 분류 (고급 Quant 분석 통합)"""
        try:
            # Random Forest 모델 사용 여부 확인
            if self.use_random_forest and self.rf_model:
                return self._classify_with_random_forest(spy_data, macro_data)
            else:
                # 하이퍼파라미터 튜너의 분류 메서드 사용
                classification_result = (
                    self.hyperparam_tuner._classify_market_regime_with_probabilities(
                        spy_data, optimal_params
                    )
                )

            # 고급 Quant 분석 수행
            enhanced_quant_analysis = self._perform_enhanced_quant_analysis(
                spy_data, macro_data, classification_result, optimal_params
            )

            # SPY 진입/매도 포인트 계산
            spy_entry_exit_points = self._calculate_spy_entry_exit_points(
                spy_data, current_regime
            )

            # 기존 결과와 고급 분석 결과 통합
            classification_result.update(enhanced_quant_analysis)
            classification_result["spy_entry_exit_points"] = spy_entry_exit_points

            # 상세 분석 결과 출력
            self._print_detailed_regime_analysis(classification_result)

            # 키 존재 여부 확인
            if "current_regime" not in classification_result:
                raise KeyError("'current_regime' 키가 분류 결과에 없습니다")

            if "confidence" not in classification_result:
                classification_result["confidence"] = 0.5

            if "probabilities" not in classification_result:
                classification_result["probabilities"] = {
                    "TRENDING_UP": 0.25,
                    "TRENDING_DOWN": 0.25,
                    "SIDEWAYS": 0.25,
                    "VOLATILE": 0.25,
                }

            return {
                "current_regime": MarketRegime(classification_result["current_regime"]),
                "confidence": classification_result["confidence"],
                "probabilities": classification_result["probabilities"],
                "enhanced_quant_analysis": enhanced_quant_analysis,
            }

        except Exception as e:
            return {
                "current_regime": MarketRegime.UNCERTAIN,
                "confidence": 0.5,
                "probabilities": {
                    "TRENDING_UP": 0.25,
                    "TRENDING_DOWN": 0.25,
                    "SIDEWAYS": 0.25,
                    "VOLATILE": 0.25,
                },
            }

    def _classify_with_random_forest(
        self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Random Forest 모델을 사용한 시장 체제 분류"""
        try:
            # 모델 학습/로드 상태 확인
            if self.retrain_rf_model:
                # 학습 데이터 준비
                training_data = self.rf_model.collect_training_data()
                # 모델 학습
                training_results = self.rf_model.train_model(
                    training_data, save_model=True
                )
                self.logger.info(
                    f"모델 학습 완료 - 테스트 정확도: {training_results['test_score']:.4f}"
                )
            else:
                # 저장된 모델 로드 시도
                try:
                    self.rf_model.load_model()
                    self.logger.info("저장된 Random Forest 모델 로드 완료")
                except FileNotFoundError:
                    self.logger.info("저장된 모델이 없어 새로 학습을 시작합니다")
                    training_data = self.rf_model.collect_training_data()
                    training_results = self.rf_model.train_model(
                        training_data, save_model=True
                    )
                    self.logger.info(
                        f"모델 학습 완료 - 테스트 정확도: {training_results['test_score']:.4f}"
                    )

            # 현재 시장 상태 확률 예측
            probabilities = self.rf_model.get_current_market_probabilities(
                self.data_dir
            )
            self.logger.info(f"Random Forest 예측 확률: {probabilities}")

            # 가장 높은 확률의 체제 선택
            current_regime = max(probabilities.items(), key=lambda x: x[1])[0].upper()

            # 신뢰도 계산 (최고 확률 값)
            confidence = max(probabilities.values())

            # MarketRegime 열거형으로 변환
            regime_mapping = {
                "TRENDING_UP": MarketRegime.TRENDING_UP,
                "TRENDING_DOWN": MarketRegime.TRENDING_DOWN,
                "VOLATILE": MarketRegime.VOLATILE,
                "SIDEWAYS": MarketRegime.SIDEWAYS,
            }

            current_regime_enum = regime_mapping.get(
                current_regime, MarketRegime.UNCERTAIN
            )

            # 확률을 대문자 키로 변환
            probabilities_upper = {k.upper(): v for k, v in probabilities.items()}

            result = {
                "current_regime": current_regime_enum,
                "confidence": confidence,
                "probabilities": probabilities_upper,
            }

            # SPY 진입/매도 포인트 계산
            spy_entry_exit_points = self._calculate_spy_entry_exit_points(
                spy_data, current_regime_enum
            )
            result["spy_entry_exit_points"] = spy_entry_exit_points

            # 고급 Quant 분석 수행 및 통합
            enhanced_quant_analysis = self._perform_enhanced_quant_analysis(
                spy_data, macro_data, result, {}
            )

            # 결과 통합
            result.update(enhanced_quant_analysis)

            return result

        except Exception as e:
            # 실패 시 규칙 기반으로 폴백
            return {
                "current_regime": MarketRegime.UNCERTAIN,
                "confidence": 0.5,
                "probabilities": {
                    "TRENDING_UP": 0.25,
                    "TRENDING_DOWN": 0.25,
                    "SIDEWAYS": 0.25,
                    "VOLATILE": 0.25,
                },
            }

    def _perform_enhanced_quant_analysis(
        self,
        spy_data: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame],
        classification_result: Dict[str, Any],
        optimal_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """고급 Quant 분석 수행 (RLMF, 신뢰도 시스템, Regime 감지 통합)"""
        try:

            # 1. 동적 Regime Switching 감지
            regime_detection = self.regime_detector.detect_regime_shifts(
                spy_data, macro_data
            )
            regime_stability = self.regime_detector.analyze_regime_stability(spy_data)
            regime_persistence = self.regime_detector.calculate_regime_persistence(
                spy_data
            )

            # 2. RLMF (Reinforcement Learning from Market Feedback) 분석
            current_regime = classification_result.get("current_regime", "SIDEWAYS")
            spy_returns = spy_data["close"].pct_change().dropna()

            # RF 모델 예측 기록
            rf_probabilities = classification_result.get("probabilities", {})
            confidence = classification_result.get("confidence", 0.5)
            self.rlmf_adaptation.record_prediction(
                current_regime.value, rf_probabilities, confidence
            )

            # Market feedback 계산 (개선된 버전)
            if hasattr(self, "last_prediction") and hasattr(self, "last_returns"):
                # 이전 예측이 있으면 실제 피드백 계산
                market_feedback = self.rlmf_adaptation.calculate_market_feedback(
                    self.last_prediction,
                    spy_returns.tail(len(self.last_returns)),
                    spy_data,
                    macro_data,
                )

                # RF 모델 비교 피드백 추가
                rf_comparison_feedback = (
                    self.rlmf_adaptation.calculate_rf_comparison_feedback(
                        spy_returns.tail(5)  # 최근 5일 수익률
                    )
                )
                market_feedback.update(rf_comparison_feedback)

                # 적응 가중치 업데이트
                self.rlmf_adaptation.update_adaptation_weights(market_feedback)
            else:
                # 첫 번째 실행이거나 이전 예측이 없는 경우
                # 현재 시장 상황 기반 기본 피드백 계산
                market_feedback = self.rlmf_adaptation.calculate_market_feedback(
                    (
                        current_regime.value
                        if hasattr(current_regime, "value")
                        else str(current_regime)
                    ),
                    spy_returns.tail(20) if len(spy_returns) >= 20 else spy_returns,
                    spy_data,
                    macro_data,
                )

                # RF 모델 비교 피드백 추가 (현재 데이터 기반)
                rf_comparison_feedback = (
                    self.rlmf_adaptation.calculate_rf_comparison_feedback(
                        spy_returns.tail(5) if len(spy_returns) >= 5 else spy_returns
                    )
                )
                market_feedback.update(rf_comparison_feedback)

                # 적응 가중치 초기화
                self.rlmf_adaptation.update_adaptation_weights(market_feedback)

            # 3. Statistical Arbitrage 신호 계산
            stat_arb_signals = (
                self.rlmf_adaptation.calculate_statistical_arbitrage_signal(macro_data)
            )

            # 4. 다층 신뢰도 계산 (confidence_system.py 사용)
            technical_confidence = classification_result.get("confidence", 0.5)
            macro_confidence = self._calculate_macro_confidence(macro_data)
            stat_arb_confidence = stat_arb_signals.get("confidence", 0.5)
            rlmf_confidence = (
                np.mean(list(market_feedback.values())) if market_feedback else 0.5
            )
            cross_val_confidence = self._calculate_cross_validation_confidence(
                spy_data, classification_result
            )

            comprehensive_confidence = (
                self.confidence_system.calculate_comprehensive_confidence(
                    technical_confidence,
                    macro_confidence,
                    stat_arb_confidence,
                    rlmf_confidence,
                    cross_val_confidence,
                )
            )

            # 5. 고급 Quant 지표 계산
            advanced_indicators = self._calculate_advanced_quant_indicators(
                spy_data, macro_data, optimal_params
            )

            # 6. 결과 통합
            enhanced_analysis = {
                # Regime 감지 결과
                "regime_detection": {
                    "shift_detected": regime_detection.get(
                        "regime_shift_detected", False
                    ),
                    "shift_score": regime_detection.get("shift_score", 0.0),
                    "stability_score": regime_stability.get("stability_score", 0.5),
                    "persistence_score": regime_persistence.get(
                        "persistence_score", 0.5
                    ),
                    "expected_duration": regime_persistence.get(
                        "expected_duration", "unknown"
                    ),
                },
                # RLMF 분석 결과
                "rlmf_analysis": {
                    "market_feedback": market_feedback,
                    "adaptation_status": self.rlmf_adaptation.get_adaptation_status(),
                    "statistical_arbitrage": stat_arb_signals,
                },
                # 신뢰도 분석 결과
                "confidence_analysis": comprehensive_confidence,
                # 고급 지표
                "advanced_indicators": advanced_indicators,
                # 종합 평가
                "quant_score": self._calculate_quant_score(
                    classification_result,
                    regime_detection,
                    market_feedback,
                    stat_arb_signals,
                    comprehensive_confidence,
                ),
            }

            # 현재 예측 저장 (다음 분석에서 feedback 계산용)
            self.last_prediction = current_regime
            self.last_returns = spy_returns.tail(20).copy()

            self.logger.info("✅ 고급 Quant 분석 완료")
            return enhanced_analysis

        except Exception as e:
            self.logger.error(f"고급 Quant 분석 실패: {e}")
            return {
                "regime_detection": {"shift_detected": False, "shift_score": 0.0},
                "rlmf_analysis": {"market_feedback": {}, "adaptation_status": {}},
                "confidence_analysis": {"adjusted_confidence": 0.5},
                "advanced_indicators": {},
                "quant_score": 0.5,
            }

    def _calculate_macro_confidence(self, macro_data: Dict[str, pd.DataFrame]) -> float:
        """매크로 데이터 기반 신뢰도 계산"""
        try:
            confidence_scores = []

            # VIX 기반 신뢰도
            if "^VIX" in macro_data and not macro_data["^VIX"].empty:
                vix_data = macro_data["^VIX"]
                close_col = "close" if "close" in vix_data.columns else "Close"
                current_vix = vix_data[close_col].iloc[-1]

                # VIX가 15-25 범위일 때 신뢰도 높음
                if 15 <= current_vix <= 25:
                    confidence_scores.append(0.8)
                elif 10 <= current_vix <= 30:
                    confidence_scores.append(0.6)
                else:
                    confidence_scores.append(0.4)

            # 금리 기반 신뢰도
            if "^TNX" in macro_data and not macro_data["^TNX"].empty:
                tnx_data = macro_data["^TNX"]
                close_col = "close" if "close" in tnx_data.columns else "Close"
                if len(tnx_data) > 5:
                    rate_volatility = tnx_data[close_col].pct_change().std()
                    # 금리 변동성이 낮을 때 신뢰도 높음
                    if rate_volatility < 0.02:
                        confidence_scores.append(0.7)
                    elif rate_volatility < 0.05:
                        confidence_scores.append(0.5)
                    else:
                        confidence_scores.append(0.3)

            return np.mean(confidence_scores) if confidence_scores else 0.5

        except Exception as e:
            self.logger.warning(f"매크로 신뢰도 계산 중 오류: {e}")
            return 0.5

    def _calculate_cross_validation_confidence(
        self, spy_data: pd.DataFrame, classification_result: Dict[str, Any]
    ) -> float:
        """교차 검증 기반 신뢰도 계산"""
        try:
            if len(spy_data) < 60:
                return 0.5

            # 최근 60일 데이터로 간단한 교차 검증
            recent_data = spy_data.tail(60)
            returns = recent_data["close"].pct_change().dropna()

            # 변동성 일관성
            volatility_consistency = (
                1.0 - (returns.std() / returns.mean()) if returns.mean() != 0 else 0.5
            )
            volatility_consistency = max(0.0, min(1.0, volatility_consistency))

            # 트렌드 일관성
            positive_ratio = np.mean(returns > 0)
            trend_consistency = (
                abs(positive_ratio - 0.5) * 2
            )  # 0.5에서 멀수록 일관된 트렌드

            # 예측과 실제의 일관성 (간단한 검증)
            predicted_regime = classification_result.get("current_regime", "SIDEWAYS")
            actual_trend = returns.mean()

            if predicted_regime == "TRENDING_UP" and actual_trend > 0.001:
                prediction_consistency = 0.8
            elif predicted_regime == "TRENDING_DOWN" and actual_trend < -0.001:
                prediction_consistency = 0.8
            elif predicted_regime == "SIDEWAYS" and abs(actual_trend) < 0.001:
                prediction_consistency = 0.8
            else:
                prediction_consistency = 0.4

            # 종합 신뢰도
            cross_val_confidence = (
                volatility_consistency + trend_consistency + prediction_consistency
            ) / 3
            return cross_val_confidence

        except Exception as e:
            self.logger.warning(f"교차 검증 신뢰도 계산 중 오류: {e}")
            return 0.5

    def _calculate_buyhold_performance(
        self, spy_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Buy & Hold 성과 계산"""
        try:
            if spy_data.empty or len(spy_data) < 2:
                return {
                    "buyhold_return": 0.0,
                    "buyhold_sharpe": 0.0,
                    "buyhold_drawdown": 0.0,
                }

            # 컬럼명 확인
            close_col = "close" if "close" in spy_data.columns else "Close"

            # Buy & Hold 수익률 계산
            initial_price = spy_data[close_col].iloc[0]
            final_price = spy_data[close_col].iloc[-1]
            buyhold_return = (final_price - initial_price) / initial_price

            # 일별 수익률 계산
            daily_returns = spy_data[close_col].pct_change().dropna()

            # 샤프 비율 계산 (연간화)
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            risk_free_rate = 0.02 / 252  # 일간 무위험 수익률
            excess_return = mean_return - risk_free_rate
            buyhold_sharpe = (
                (excess_return * 252) / (std_return * np.sqrt(252))
                if std_return > 0
                else 0.0
            )

            # 최대 낙폭 계산
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            buyhold_drawdown = abs(drawdown.min())

            return {
                "buyhold_return": buyhold_return,
                "buyhold_sharpe": buyhold_sharpe,
                "buyhold_drawdown": buyhold_drawdown,
            }

        except Exception as e:
            self.logger.warning(f"Buy & Hold 성과 계산 중 오류: {e}")
            return {
                "buyhold_return": 0.0,
                "buyhold_sharpe": 0.0,
                "buyhold_drawdown": 0.0,
            }

    def _calculate_spy_entry_exit_points(
        self, spy_data: pd.DataFrame, current_regime: MarketRegime
    ) -> Dict[str, Any]:
        """SPY 기반 구체적인 진입/매도 포인트 계산"""
        try:
            if spy_data.empty:
                return {}

            current_price = spy_data["Close"].iloc[-1]
            current_volume = spy_data["Volume"].iloc[-1]

            # 기술적 지표 계산
            spy_data = spy_data.copy()

            # 이동평균
            spy_data["SMA_20"] = spy_data["Close"].rolling(window=20).mean()
            spy_data["SMA_50"] = spy_data["Close"].rolling(window=50).mean()
            spy_data["EMA_12"] = spy_data["Close"].ewm(span=12).mean()
            spy_data["EMA_26"] = spy_data["Close"].ewm(span=26).mean()

            # RSI
            delta = spy_data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            spy_data["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            spy_data["MACD"] = spy_data["EMA_12"] - spy_data["EMA_26"]
            spy_data["MACD_Signal"] = spy_data["MACD"].ewm(span=9).mean()
            spy_data["MACD_Histogram"] = spy_data["MACD"] - spy_data["MACD_Signal"]

            # 볼린저 밴드
            spy_data["BB_Middle"] = spy_data["Close"].rolling(window=20).mean()
            bb_std = spy_data["Close"].rolling(window=20).std()
            spy_data["BB_Upper"] = spy_data["BB_Middle"] + (bb_std * 2)
            spy_data["BB_Lower"] = spy_data["BB_Middle"] - (bb_std * 2)

            # 지지/저항선 계산
            recent_highs = spy_data["High"].rolling(window=20).max()
            recent_lows = spy_data["Low"].rolling(window=20).min()

            # 현재 값들
            current_sma_20 = spy_data["SMA_20"].iloc[-1]
            current_sma_50 = spy_data["SMA_50"].iloc[-1]
            current_rsi = spy_data["RSI"].iloc[-1]
            current_macd = spy_data["MACD"].iloc[-1]
            current_macd_signal = spy_data["MACD_Signal"].iloc[-1]
            current_bb_upper = spy_data["BB_Upper"].iloc[-1]
            current_bb_lower = spy_data["BB_Lower"].iloc[-1]
            current_bb_middle = spy_data["BB_Middle"].iloc[-1]

            # 체제별 진입/매도 전략
            entry_exit_points = {
                "current_price": current_price,
                "support_levels": [],
                "resistance_levels": [],
                "entry_signals": [],
                "exit_signals": [],
                "stop_loss_levels": [],
                "take_profit_levels": [],
                "technical_indicators": {
                    "rsi": current_rsi,
                    "macd": current_macd,
                    "macd_signal": current_macd_signal,
                    "sma_20": current_sma_20,
                    "sma_50": current_sma_50,
                    "bb_upper": current_bb_upper,
                    "bb_lower": current_bb_lower,
                    "bb_middle": current_bb_middle,
                },
            }

            # 체제별 전략 설정
            if current_regime == MarketRegime.TRENDING_UP:
                # 강세장: 모멘텀 팔로잉
                entry_exit_points["support_levels"] = [
                    current_sma_20,
                    current_sma_50,
                    current_bb_middle,
                ]
                entry_exit_points["resistance_levels"] = [
                    current_bb_upper,
                    current_price * 1.02,  # 2% 상승
                    current_price * 1.05,  # 5% 상승
                ]
                entry_exit_points["entry_signals"] = [
                    f"RSI < 70 (현재: {current_rsi:.1f})",
                    f"MACD > Signal (현재: {current_macd:.2f} > {current_macd_signal:.2f})",
                    f"가격 > SMA20 (현재: {current_price:.2f} > {current_sma_20:.2f})",
                ]
                entry_exit_points["exit_signals"] = [
                    f"RSI > 80 (현재: {current_rsi:.1f})",
                    f"가격 < SMA20 (현재: {current_price:.2f} < {current_sma_20:.2f})",
                    f"볼린저 상단 돌파 후 하락",
                ]
                entry_exit_points["stop_loss_levels"] = [
                    current_sma_50,
                    current_price * 0.95,  # 5% 손절
                    current_bb_lower,
                ]
                entry_exit_points["take_profit_levels"] = [
                    current_price * 1.05,  # 5% 익절
                    current_price * 1.10,  # 10% 익절
                    current_bb_upper,
                ]

            elif current_regime == MarketRegime.TRENDING_DOWN:
                # 약세장: 방어적 전략
                entry_exit_points["support_levels"] = [
                    current_bb_lower,
                    current_price * 0.95,  # 5% 하락
                    current_price * 0.90,  # 10% 하락
                ]
                entry_exit_points["resistance_levels"] = [
                    current_sma_20,
                    current_sma_50,
                    current_bb_middle,
                ]
                entry_exit_points["entry_signals"] = [
                    f"RSI < 30 (현재: {current_rsi:.1f})",
                    f"가격 < 볼린저 하단 (현재: {current_price:.2f} < {current_bb_lower:.2f})",
                    f"반등 신호 확인",
                ]
                entry_exit_points["exit_signals"] = [
                    f"RSI > 50 (현재: {current_rsi:.1f})",
                    f"가격 > SMA20 (현재: {current_price:.2f} > {current_sma_20:.2f})",
                    f"저항선 돌파 실패",
                ]
                entry_exit_points["stop_loss_levels"] = [
                    current_price * 0.97,  # 3% 손절
                    current_price * 0.95,  # 5% 손절
                    current_bb_lower * 0.98,
                ]
                entry_exit_points["take_profit_levels"] = [
                    current_price * 1.03,  # 3% 익절
                    current_price * 1.05,  # 5% 익절
                    current_sma_20,
                ]

            elif current_regime == MarketRegime.VOLATILE:
                # 변동성 높은 시장: 범위 거래
                entry_exit_points["support_levels"] = [
                    current_bb_lower,
                    current_sma_50,
                    current_price * 0.98,
                ]
                entry_exit_points["resistance_levels"] = [
                    current_bb_upper,
                    current_sma_20,
                    current_price * 1.02,
                ]
                entry_exit_points["entry_signals"] = [
                    f"RSI < 40 (현재: {current_rsi:.1f})",
                    f"가격 < 볼린저 하단 (현재: {current_price:.2f} < {current_bb_lower:.2f})",
                    f"변동성 수축 시점",
                ]
                entry_exit_points["exit_signals"] = [
                    f"RSI > 60 (현재: {current_rsi:.1f})",
                    f"가격 > 볼린저 상단 (현재: {current_price:.2f} > {current_bb_upper:.2f})",
                    f"변동성 확대 시점",
                ]
                entry_exit_points["stop_loss_levels"] = [
                    current_price * 0.96,  # 4% 손절
                    current_bb_lower * 0.99,
                    current_sma_50 * 0.98,
                ]
                entry_exit_points["take_profit_levels"] = [
                    current_price * 1.04,  # 4% 익절
                    current_bb_upper * 1.01,
                    current_sma_20 * 1.02,
                ]

            elif current_regime == MarketRegime.SIDEWAYS:
                # 횡보장: 범위 내 거래
                entry_exit_points["support_levels"] = [
                    current_bb_lower,
                    current_sma_50,
                    current_price * 0.985,
                ]
                entry_exit_points["resistance_levels"] = [
                    current_bb_upper,
                    current_sma_20,
                    current_price * 1.015,
                ]
                entry_exit_points["entry_signals"] = [
                    f"RSI < 35 (현재: {current_rsi:.1f})",
                    f"지지선 근처 매수 (현재: {current_price:.2f})",
                    f"볼린저 하단 지지",
                ]
                entry_exit_points["exit_signals"] = [
                    f"RSI > 65 (현재: {current_rsi:.1f})",
                    f"저항선 근처 매도 (현재: {current_price:.2f})",
                    f"볼린저 상단 저항",
                ]
                entry_exit_points["stop_loss_levels"] = [
                    current_price * 0.965,  # 3.5% 손절
                    current_bb_lower * 0.995,
                    current_sma_50 * 0.985,
                ]
                entry_exit_points["take_profit_levels"] = [
                    current_price * 1.035,  # 3.5% 익절
                    current_bb_upper * 1.005,
                    current_sma_20 * 1.015,
                ]

            return entry_exit_points

        except Exception as e:
            self.logger.warning(f"SPY 진입/매도 포인트 계산 실패: {e}")
            return {}

    def _calculate_advanced_quant_indicators(
        self,
        spy_data: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame],
        optimal_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """고급 Quant 지표 계산"""
        try:
            indicators = {}

            # 1. 변동성 구조 분석
            returns = spy_data["close"].pct_change().dropna()
            indicators["volatility_structure"] = {
                "realized_vol": returns.std() * np.sqrt(252),
                "vol_of_vol": returns.rolling(20).std().std(),
                "vol_regime": (
                    "high"
                    if returns.std() > returns.rolling(60).std().mean()
                    else "normal"
                ),
            }

            # 2. 상관관계 분석
            if "^VIX" in macro_data and not macro_data["^VIX"].empty:
                vix_data = macro_data["^VIX"]
                close_col = "close" if "close" in vix_data.columns else "Close"
                vix_returns = vix_data[close_col].pct_change().dropna()

                # SPY와 VIX의 상관관계
                correlation = returns.tail(len(vix_returns)).corr(vix_returns)
                indicators["correlation_analysis"] = {
                    "spy_vix_correlation": correlation,
                    "correlation_regime": (
                        "negative"
                        if correlation < -0.3
                        else "positive" if correlation > 0.3 else "neutral"
                    ),
                }

            # 3. 모멘텀 구조 분석 (고도화)
            short_momentum = returns.tail(5).mean()
            medium_momentum = returns.tail(20).mean()
            long_momentum = returns.tail(60).mean()

            # 모멘텀 가속도 계산
            momentum_acceleration = short_momentum - medium_momentum
            momentum_trend = medium_momentum - long_momentum

            # RSI 모멘텀
            rsi_momentum = 0
            if "rsi" in spy_data.columns:
                rsi = spy_data["rsi"].tail(5)
                rsi_momentum = (rsi.iloc[-1] - rsi.iloc[0]) / 100  # RSI 변화율

            # MACD 모멘텀
            macd_momentum = 0
            if "macd" in spy_data.columns and "macd_signal" in spy_data.columns:
                macd = spy_data["macd"].tail(5)
                macd_signal = spy_data["macd_signal"].tail(5)
                macd_momentum = (
                    (macd.iloc[-1] - macd_signal.iloc[-1]) / abs(macd_signal.iloc[-1])
                    if abs(macd_signal.iloc[-1]) > 0
                    else 0
                )

            # 종합 모멘텀 정렬 판단
            momentum_scores = []
            if short_momentum > medium_momentum > long_momentum:
                momentum_scores.append(1)  # 상승 정렬
            elif short_momentum < medium_momentum < long_momentum:
                momentum_scores.append(-1)  # 하락 정렬
            else:
                momentum_scores.append(0)  # 혼재

            if momentum_acceleration > 0:
                momentum_scores.append(1)
            elif momentum_acceleration < 0:
                momentum_scores.append(-1)
            else:
                momentum_scores.append(0)

            if rsi_momentum > 0:
                momentum_scores.append(1)
            elif rsi_momentum < 0:
                momentum_scores.append(-1)
            else:
                momentum_scores.append(0)

            if macd_momentum > 0:
                momentum_scores.append(1)
            elif macd_momentum < 0:
                momentum_scores.append(-1)
            else:
                momentum_scores.append(0)

            # 모멘텀 정렬 결정
            avg_momentum_score = np.mean(momentum_scores)
            if avg_momentum_score > 0.3:
                momentum_alignment = "bullish"
            elif avg_momentum_score < -0.3:
                momentum_alignment = "bearish"
            else:
                momentum_alignment = "mixed"

            indicators["momentum_structure"] = {
                "short_momentum": short_momentum,
                "medium_momentum": medium_momentum,
                "long_momentum": long_momentum,
                "momentum_acceleration": momentum_acceleration,
                "momentum_trend": momentum_trend,
                "rsi_momentum": rsi_momentum,
                "macd_momentum": macd_momentum,
                "momentum_alignment": momentum_alignment,
                "momentum_score": avg_momentum_score,
            }

            # 4. 거래량 분석
            if "volume" in spy_data.columns:
                volume = spy_data["volume"]
                indicators["volume_analysis"] = {
                    "volume_trend": volume.tail(20).mean() / volume.tail(60).mean(),
                    "volume_regime": (
                        "high"
                        if volume.tail(20).mean() > volume.tail(60).mean() * 1.2
                        else "normal"
                    ),
                }

            # 5. 시장 미세구조 분석
            indicators["market_microstructure"] = {
                "bid_ask_spread_proxy": returns.abs().mean(),  # 간단한 스프레드 프록시
                "price_efficiency": (
                    1.0 - abs(returns.autocorr()) if len(returns) > 1 else 0.5
                ),
                "liquidity_regime": "high" if returns.abs().mean() < 0.01 else "normal",
            }

            return indicators

        except Exception as e:
            self.logger.warning(f"고급 Quant 지표 계산 중 오류: {e}")
            return {}

    def _calculate_quant_score(
        self,
        classification_result: Dict[str, Any],
        regime_detection: Dict[str, Any],
        market_feedback: Dict[str, float],
        stat_arb_signals: Dict[str, Any],
        comprehensive_confidence: Dict[str, Any],
    ) -> float:
        """종합 Quant 점수 계산"""
        try:
            scores = []

            # 1. 기본 분류 신뢰도
            scores.append(classification_result.get("confidence", 0.5))

            # 2. Regime 안정성
            stability_score = regime_detection.get("stability_score", 0.5)
            scores.append(stability_score)

            # 3. Market Feedback 성과
            feedback_score = np.mean(list(market_feedback.values()))
            scores.append(feedback_score)

            # 4. Statistical Arbitrage 신호 강도
            arb_signal_strength = stat_arb_signals.get("signal_strength", 0.0)
            scores.append(min(1.0, arb_signal_strength * 2))  # 0.5를 1.0으로 스케일링

            # 5. 종합 신뢰도
            comprehensive_conf = comprehensive_confidence.get(
                "adjusted_confidence", 0.5
            )
            scores.append(comprehensive_conf)

            # 가중 평균 계산
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # 기본 분류에 가장 높은 가중치
            quant_score = np.average(scores, weights=weights)

            return max(0.0, min(1.0, quant_score))

        except Exception as e:
            self.logger.warning(f"Quant 점수 계산 중 오류: {e}")
            return 0.5

    def _perform_validation(
        self,
        spy_data: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame],
        regime_classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """분석 결과 검증"""
        try:
            # 실제 수익률과 예측 비교
            actual_returns = spy_data["close"].pct_change().dropna()

            # 간단한 검증 (실제로는 더 복잡한 검증 로직 필요)
            validation_results = {
                "data_quality": {
                    "spy_data_points": len(spy_data),
                    "macro_indicators": len(macro_data),
                    "data_completeness": 0.95,  # 예시
                },
                "regime_consistency": {
                    "confidence_threshold_met": regime_classification["confidence"]
                    > 0.6,
                    "probability_sum": sum(
                        regime_classification["probabilities"].values()
                    ),
                },
            }

            return validation_results

        except Exception as e:
            self.logger.error(f"검증 실패: {e}")
            return {}

    def _perform_rlmf_analysis(
        self, macro_analysis: MacroAnalysis, current_regime: MarketRegime
    ) -> Dict[str, Any]:
        """RLMF 적응 분석 수행"""
        try:
            # 매크로 데이터를 가져오기 위해 캐시된 데이터 로드
            spy_data, macro_data, sector_data = self._load_cached_data()

            if spy_data.empty or not macro_data:
                self.logger.warning(
                    "매크로 데이터를 찾을 수 없어 RLMF 분석을 건너뜁니다."
                )
                return {
                    "statistical_arbitrage": {
                        "overall_signal": 0.0,
                        "confidence": 0.0,
                        "individual_signals": {},
                        "signal_strength": 0.0,
                        "direction": "NEUTRAL",
                    },
                    "market_feedback": {
                        "prediction_accuracy": 0.5,
                        "return_alignment": 0.5,
                        "volatility_prediction": 0.5,
                        "regime_persistence": 0.5,
                        "macro_consistency": 0.5,
                    },
                    "adaptation_status": self.rlmf_adaptation.get_adaptation_status(),
                    "learning_rate": self.rlmf_adaptation.learning_rate,
                }

            # Statistical Arbitrage 신호 계산 (실제 매크로 데이터 사용)
            stat_arb_signal = (
                self.rlmf_adaptation.calculate_statistical_arbitrage_signal(macro_data)
            )

            # Market feedback 계산 (실제 데이터 사용)
            spy_returns = spy_data["close"].pct_change().dropna()
            market_feedback = self.rlmf_adaptation.calculate_market_feedback(
                current_regime.value,
                spy_returns.tail(20) if len(spy_returns) >= 20 else spy_returns,
                spy_data,
                macro_data,
            )

            return {
                "statistical_arbitrage": stat_arb_signal,
                "market_feedback": market_feedback,
                "adaptation_status": self.rlmf_adaptation.get_adaptation_status(),
                "learning_rate": self.rlmf_adaptation.learning_rate,
            }

        except Exception as e:
            self.logger.warning(f"RLMF 분석 실패: {e}")
            return {
                "statistical_arbitrage": {
                    "overall_signal": 0.0,
                    "confidence": 0.0,
                    "individual_signals": {},
                    "signal_strength": 0.0,
                    "direction": "NEUTRAL",
                },
                "market_feedback": {
                    "prediction_accuracy": 0.5,
                    "return_alignment": 0.5,
                    "volatility_prediction": 0.5,
                    "regime_persistence": 0.5,
                    "macro_consistency": 0.5,
                },
                "adaptation_status": self.rlmf_adaptation.get_adaptation_status(),
                "learning_rate": self.rlmf_adaptation.learning_rate,
            }

    def _perform_confidence_analysis(
        self, basic_result: MarketAnalysisResult, rlmf_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """다층 신뢰도 분석 수행"""
        try:
            # 기본 신뢰도 구성요소들
            technical_conf = basic_result.confidence
            macro_conf = basic_result.macro_analysis.confidence

            # RLMF 기반 신뢰도
            rlmf_conf = 0.5
            if rlmf_analysis and "market_feedback" in rlmf_analysis:
                feedback = rlmf_analysis["market_feedback"]
                rlmf_conf = feedback.get("prediction_accuracy", 0.5)

            # 종합 신뢰도 계산
            confidence_result = (
                self.confidence_system.calculate_comprehensive_confidence(
                    technical_conf, macro_conf, 0.5, rlmf_conf, 0.5
                )
            )

            return {
                "confidence_result": confidence_result,
                "component_breakdown": {
                    "technical": technical_conf,
                    "macro": macro_conf,
                    "rlmf_feedback": rlmf_conf,
                },
            }

        except Exception as e:
            self.logger.warning(f"신뢰도 분석 실패: {e}")
            return {"confidence_result": {"adjusted_confidence": 0.5}}

    def _perform_regime_detection(
        self, macro_analysis: MacroAnalysis
    ) -> Dict[str, Any]:
        """Regime 전환 감지 수행"""
        try:
            # Regime shift 감지 (예시 데이터)
            regime_shift = {
                "regime_shift_detected": False,
                "confidence": 0.5,
                "last_change_date": None,
            }

            return {
                "regime_shift_detection": regime_shift,
                "stability_analysis": {"stability_score": 0.7},
                "persistence_analysis": {"persistence_score": 0.8},
            }

        except Exception as e:
            self.logger.warning(f"Regime 감지 실패: {e}")
            return {}

    def _perform_llm_analysis(
        self, macro_analysis: MacroAnalysis, current_regime: MarketRegime
    ) -> Dict[str, Any]:
        """LLM 특권 정보 분석 수행"""
        try:
            # 시장 메트릭 준비
            market_metrics = {
                "current_regime": current_regime.value,
                "market_condition": macro_analysis.market_condition.value,
                "confidence": macro_analysis.confidence,
            }

            # LLM 특권 정보 획득
            llm_insights = self.llm_privileged_system.get_privileged_insights(
                current_regime.value, {}, market_metrics
            )

            return llm_insights

        except Exception as e:
            self.logger.warning(f"LLM 분석 실패: {e}")
            return {}

    def _perform_llm_api_analysis(
        self,
        macro_analysis: MacroAnalysis,
        current_regime: MarketRegime,
        analysis_results: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """LLM API 통합 분석 수행"""
        try:
            # 시장 메트릭 준비
            market_metrics = {
                "current_regime": current_regime.value,
                "market_condition": macro_analysis.market_condition.value,
                "confidence": macro_analysis.confidence,
            }

            # LLM API 통합 인사이트 획득 (기존 분석 결과 포함)
            llm_api_insights = self.llm_api_system.get_enhanced_insights(
                current_regime.value, {}, market_metrics, analysis_results
            )

            # API 통계 추가
            api_stats = self.llm_api_system.get_api_stats()
            llm_api_insights["api_stats"] = api_stats

            return llm_api_insights

        except Exception as e:
            self.logger.warning(f"LLM API 분석 실패: {e}")
            return {}

    def _calculate_final_confidence(
        self,
        basic_result: MarketAnalysisResult,
        rlmf_analysis: Dict[str, Any],
        confidence_analysis: Dict[str, Any],
        regime_detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """최종 신뢰도 계산"""
        try:
            # 기본 신뢰도
            base_confidence = basic_result.confidence

            # 고급 분석 신뢰도들
            confidence_result = confidence_analysis.get("confidence_result", {})
            adjusted_confidence = confidence_result.get(
                "adjusted_confidence", base_confidence
            )

            # Regime 전환 감지 영향
            regime_shift = regime_detection.get("regime_shift_detection", {})
            shift_confidence = regime_shift.get("confidence", 1.0)

            # 최종 신뢰도 계산
            final_confidence = adjusted_confidence * shift_confidence
            final_confidence = max(0.0, min(1.0, final_confidence))

            return {
                "final_confidence": final_confidence,
                "base_confidence": base_confidence,
                "adjusted_confidence": adjusted_confidence,
                "regime_shift_impact": shift_confidence,
                "confidence_level": self._get_confidence_level(final_confidence),
            }

        except Exception as e:
            self.logger.warning(f"최종 신뢰도 계산 실패: {e}")
            return {"final_confidence": 0.5, "confidence_level": "MEDIUM"}

    def _get_confidence_level(self, confidence: float) -> str:
        """신뢰도 수준 분류"""
        if confidence >= 0.8:
            return "VERY_HIGH"
        elif confidence >= 0.6:
            return "HIGH"
        elif confidence >= 0.4:
            return "MEDIUM"
        elif confidence >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"

    def _generate_basic_recommendations(
        self, macro_analysis: MacroAnalysis, current_regime: MarketRegime
    ) -> Dict[str, Any]:
        """기본 전략적 추천 생성"""
        try:
            self.logger.info(f"기본 추천 생성 시작 - 현재 체제: {current_regime}")
            recommendations = {
                "primary_strategy": "",
                "risk_level": "medium",
                "position_sizing": "MODERATE",
                "time_horizon": "MEDIUM",
                "overweight_sectors": [],
                "underweight_sectors": [],
                "neutral_sectors": [],
                "key_actions": [],
                "risk_warnings": [],
                "sector_rotation": [],
                # 구체적인 투자 전략 추가
                "position_size_percentage": 0.0,
                "stop_loss_percentage": 0.0,
                "take_profit_percentage": 0.0,
                "trailing_stop_percentage": 0.0,
                "hedging_strategy": "",
                "entry_points": [],
                "exit_strategy": "",
                "risk_management": {},
                "portfolio_allocation": {},
            }

            # 체제별 기본 전략
            if current_regime == MarketRegime.TRENDING_UP:
                recommendations["primary_strategy"] = "BULLISH"
                recommendations["risk_level"] = "high"
                recommendations["position_sizing"] = "AGGRESSIVE"
                recommendations["position_size_percentage"] = 80.0
                recommendations["stop_loss_percentage"] = 5.0
                recommendations["take_profit_percentage"] = 15.0
                recommendations["trailing_stop_percentage"] = 3.0
                recommendations["hedging_strategy"] = "PUT 옵션 헤지 (10% 비중)"
                recommendations["entry_points"] = ["지지선 돌파", "모멘텀 확인"]
                recommendations["exit_strategy"] = "트레일링 스탑 + 익절"
                recommendations["risk_management"] = {
                    "max_drawdown": 8.0,
                    "position_limit": 25.0,
                    "correlation_limit": 0.7,
                }
                # 레이 달리오 올웨더 포트폴리오 기반 (강세장 조정)
                recommendations["portfolio_allocation"] = {
                    "equity": 40.0,  # 30% 기본 + 10% 추가
                    "bonds_short": 5.0,
                    "bonds_intermediate": 15.0,  # 7-10년 국채
                    "bonds_long": 25.0,  # 20-25년 국채
                    "gold": 7.5,
                    "commodities": 7.5,
                    "cash": 0.0,
                }
                recommendations["key_actions"].append("모멘텀 팔로잉 전략")
                recommendations["key_actions"].append("순환적 섹터 중 선도 섹터 집중")

            elif current_regime == MarketRegime.TRENDING_DOWN:
                recommendations["primary_strategy"] = "BEARISH"
                recommendations["risk_level"] = "low"
                recommendations["position_sizing"] = "CONSERVATIVE"
                recommendations["position_size_percentage"] = 30.0
                recommendations["stop_loss_percentage"] = 3.0
                recommendations["take_profit_percentage"] = 8.0
                recommendations["trailing_stop_percentage"] = 2.0
                recommendations["hedging_strategy"] = (
                    "SHORT 포지션 + PUT 옵션 (20% 비중)"
                )
                recommendations["entry_points"] = [
                    "저항선 하향 돌파",
                    "기술적 약세 확인",
                ]
                recommendations["exit_strategy"] = "보수적 익절 + 스탑로스"
                recommendations["risk_management"] = {
                    "max_drawdown": 5.0,
                    "position_limit": 15.0,
                    "correlation_limit": 0.5,
                }
                # 레이 달리오 올웨더 포트폴리오 기반 (약세장 조정)
                recommendations["portfolio_allocation"] = {
                    "equity": 20.0,  # 30% 기본 - 10% 감소
                    "bonds_short": 10.0,
                    "bonds_intermediate": 20.0,  # 7-10년 국채 증가
                    "bonds_long": 30.0,  # 20-25년 국채 증가
                    "gold": 10.0,  # 인플레이션 헤지 증가
                    "commodities": 10.0,
                    "cash": 0.0,
                }
                recommendations["key_actions"].append("방어적 포지셔닝")
                recommendations["key_actions"].append("방어적 섹터 집중")

            elif current_regime == MarketRegime.VOLATILE:
                recommendations["primary_strategy"] = "VOLATILITY_HEDGE"
                recommendations["risk_level"] = "medium"
                recommendations["position_sizing"] = "CAUTIOUS"
                recommendations["position_size_percentage"] = 50.0
                recommendations["stop_loss_percentage"] = 4.0
                recommendations["take_profit_percentage"] = 10.0
                recommendations["trailing_stop_percentage"] = 2.5
                recommendations["hedging_strategy"] = (
                    "VIX ETF + PUT 스프레드 (15% 비중)"
                )
                recommendations["entry_points"] = ["변동성 수축 시점", "지지선 근처"]
                recommendations["exit_strategy"] = "빠른 익절 + 변동성 헤지"
                recommendations["risk_management"] = {
                    "max_drawdown": 6.0,
                    "position_limit": 20.0,
                    "correlation_limit": 0.6,
                }
                # 레이 달리오 올웨더 포트폴리오 기반 (변동성 높은 시장)
                recommendations["portfolio_allocation"] = {
                    "equity": 25.0,  # 30% 기본 - 5% 감소
                    "bonds_short": 7.5,
                    "bonds_intermediate": 17.5,  # 7-10년 국채
                    "bonds_long": 27.5,  # 20-25년 국채
                    "gold": 8.75,  # 인플레이션 헤지
                    "commodities": 8.75,
                    "cash": 5.0,  # 유동성 보유
                }
                recommendations["key_actions"].append("변동성 헤지 전략")
                recommendations["key_actions"].append("옵션 스프레드 활용")

            elif current_regime == MarketRegime.SIDEWAYS:
                recommendations["primary_strategy"] = "RANGE_TRADING"
                recommendations["risk_level"] = "medium"
                recommendations["position_sizing"] = "MODERATE"
                recommendations["position_size_percentage"] = 60.0
                recommendations["stop_loss_percentage"] = 3.5
                recommendations["take_profit_percentage"] = 7.0
                recommendations["trailing_stop_percentage"] = 2.0
                recommendations["hedging_strategy"] = "CALL/PUT 스프레드 (10% 비중)"
                recommendations["entry_points"] = [
                    "지지선 근처 매수",
                    "저항선 근처 매도",
                ]
                recommendations["exit_strategy"] = "범위 내 익절 + 스탑로스"
                recommendations["risk_management"] = {
                    "max_drawdown": 5.5,
                    "position_limit": 20.0,
                    "correlation_limit": 0.6,
                }
                # 레이 달리오 올웨더 포트폴리오 기반 (횡보장)
                recommendations["portfolio_allocation"] = {
                    "equity": 30.0,  # 기본 30%
                    "bonds_short": 7.5,
                    "bonds_intermediate": 15.0,  # 7-10년 국채
                    "bonds_long": 25.0,  # 20-25년 국채
                    "gold": 7.5,
                    "commodities": 7.5,
                    "cash": 7.5,  # 횡보장에서 현금 비중
                }
                recommendations["key_actions"].append("범위 내 거래")
                recommendations["key_actions"].append("지지/저항선 활용")

            # 매크로 조건별 추가 전략
            market_condition = macro_analysis.market_condition
            if market_condition == MarketCondition.RECESSION_FEAR:
                recommendations["risk_warnings"].append(
                    "경기침체 우려 - 현금 비중 확대 고려"
                )
                recommendations["key_actions"].append("국채 비중 확대")
            elif market_condition == MarketCondition.INFLATION_FEAR:
                recommendations["risk_warnings"].append(
                    "인플레이션 우려 - 실물자산 비중 확대"
                )
                recommendations["key_actions"].append("TIPS 및 실물자산 투자")

            # 섹터 로테이션 추천
            for sector, strength in macro_analysis.sector_rotation.items():
                if strength == SectorStrength.LEADING:
                    recommendations["overweight_sectors"].append(sector)
                    recommendations["sector_rotation"].append(f"OVERWEIGHT: {sector}")
                elif strength == SectorStrength.LAGGING:
                    recommendations["underweight_sectors"].append(sector)
                    recommendations["sector_rotation"].append(f"UNDERWEIGHT: {sector}")
                else:
                    recommendations["neutral_sectors"].append(sector)

            # 매크로 분석의 추천 정보 추가
            if (
                hasattr(macro_analysis, "recommendations")
                and macro_analysis.recommendations
            ):
                recommendations.update(macro_analysis.recommendations)

            return recommendations

        except Exception as e:
            self.logger.warning(f"기본 추천 생성 실패: {e}")
            return {
                "primary_strategy": "BALANCED",
                "risk_level": "medium",
                "position_sizing": "MODERATE",
                "time_horizon": "MEDIUM",
                "position_size_percentage": 50.0,
                "stop_loss_percentage": 4.0,
                "take_profit_percentage": 10.0,
                "trailing_stop_percentage": 2.5,
                "hedging_strategy": "기본 헤지 없음",
                "entry_points": ["기술적 지지선", "모멘텀 확인"],
                "exit_strategy": "스탑로스 + 익절",
                "risk_management": {
                    "max_drawdown": 6.0,
                    "position_limit": 20.0,
                    "correlation_limit": 0.6,
                },
                "portfolio_allocation": {
                    "equity": 30.0,  # 레이 달리오 올웨더 기본
                    "bonds_short": 7.5,
                    "bonds_intermediate": 15.0,  # 7-10년 국채
                    "bonds_long": 25.0,  # 20-25년 국채
                    "gold": 7.5,
                    "commodities": 7.5,
                    "cash": 7.5,
                },
            }

    def _generate_enhanced_recommendations(
        self,
        basic_result: MarketAnalysisResult,
        rlmf_analysis: Dict[str, Any],
        regime_detection: Dict[str, Any],
        llm_insights: Dict[str, Any],
    ) -> Dict[str, Any]:
        """고도화된 전략적 추천 생성"""
        try:
            recommendations = basic_result.enhanced_recommendations.copy()
            recommendations["advanced_insights"] = []
            recommendations["llm_recommendations"] = []

            # RLMF 분석 기반 추천
            if rlmf_analysis:
                stat_arb = rlmf_analysis.get("statistical_arbitrage", {})
                if stat_arb.get("direction") == "BULLISH":
                    recommendations["key_considerations"].append(
                        "Statistical Arbitrage 신호: 강세"
                    )
                elif stat_arb.get("direction") == "BEARISH":
                    recommendations["key_considerations"].append(
                        "Statistical Arbitrage 신호: 약세"
                    )

            # Regime 전환 감지 기반 추천
            if regime_detection.get("regime_shift_detection", {}).get(
                "regime_shift_detected", False
            ):
                recommendations["key_considerations"].append(
                    "⚠️ 시장 체제 전환 감지됨 - 신중한 접근 필요"
                )

            # LLM 인사이트 기반 추천
            if llm_insights:
                strategic_recs = llm_insights.get("strategic_recommendations", [])
                recommendations["llm_recommendations"].extend(strategic_recs[:3])

            # 신뢰도 기반 포지션 사이징 조정
            final_confidence = basic_result.final_confidence.get(
                "final_confidence", 0.5
            )
            if final_confidence >= 0.7:
                recommendations["position_sizing"] = "AGGRESSIVE"
            elif final_confidence <= 0.3:
                recommendations["position_sizing"] = "CONSERVATIVE"

            return recommendations

        except Exception as e:
            self.logger.warning(f"고도화된 추천 생성 실패: {e}")
            return {"primary_strategy": "BALANCED", "position_sizing": "MODERATE"}

    def _save_analysis_result(
        self, result: MarketAnalysisResult, output_dir: str, verbose: bool = True
    ):
        """분석 결과 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # JSON 형태로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{output_dir}/analysis_results_{timestamp}.json"

            # Random Forest 모델 정보 추가
            rf_info = {
                "model_used": self.use_random_forest,
                "accuracy": None,
                "trained_at": None,
            }

            if self.use_random_forest and self.rf_model:
                try:
                    # 모델 정보 가져오기
                    model_file = self.rf_model.model_dir / "market_regime_rf_model.pkl"
                    if model_file.exists():
                        import joblib

                        model_data = joblib.load(model_file)
                        rf_info["trained_at"] = model_data.get("trained_at", "Unknown")
                        # 정확도는 학습 시에만 계산되므로 None으로 유지
                except Exception as e:
                    self.logger.warning(f"Random Forest 모델 정보 로드 실패: {e}")

            # 데이터클래스를 딕셔너리로 변환
            result_dict = {
                "session_uuid": result.session_uuid,
                "timestamp": result.timestamp.isoformat(),
                "analysis_type": result.analysis_type,
                "data_period": result.data_period,
                "current_regime": result.current_regime.value,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "macro_analysis": {
                    "market_condition": result.macro_analysis.market_condition.value,
                    "confidence": result.macro_analysis.confidence,
                    "key_indicators": result.macro_analysis.key_indicators,
                    "sector_rotation": {
                        k: v.value
                        for k, v in result.macro_analysis.sector_rotation.items()
                    },
                    "recommendations": result.macro_analysis.recommendations,
                },
                "optimal_params": result.optimal_params,
                "optimization_performance": result.optimization_performance,
                "validation_results": result.validation_results,
                "rlmf_analysis": result.rlmf_analysis,
                "confidence_analysis": result.confidence_analysis,
                "regime_detection": result.regime_detection,
                "llm_insights": result.llm_insights,
                "llm_api_insights": result.llm_api_insights,
                "final_confidence": result.final_confidence,
                "enhanced_recommendations": result.enhanced_recommendations,
                "random_forest_info": rf_info,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

            if verbose:
                print(f"✅ 결과 저장: {output_file}")

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")

    def _print_basic_summary(self, result: MarketAnalysisResult):
        """기본 분석 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 상세 시장 분석 결과")
        print("=" * 80)

        # 1. 시장 체제 분석 (Quant 기반 + ML 기반)
        print("\n🎯 1. 시장 체제 분석")
        print("-" * 40)

        # Quant 기반 분석 (고급)
        print("📊 Quant 기반 분석 (고급):")
        print(f"  현재 체제: {result.current_regime.value}")

        # 기본 신뢰도 (Random Forest)
        base_confidence = (
            result.classification_result.get("confidence", 0.5)
            if result.classification_result
            else 0.5
        )
        print(f"  기본 신뢰도: {base_confidence:.3f}")

        # 다층 신뢰도 시스템 결과
        if result.confidence_analysis:
            confidence_analysis = result.confidence_analysis
            adjusted_confidence = confidence_analysis.get(
                "adjusted_confidence", base_confidence
            )
            consistency_score = confidence_analysis.get("consistency_score", 0.5)

            print(f"  🎯 다층 신뢰도: {adjusted_confidence:.3f}")
            print(f"  🔗 일관성 점수: {consistency_score:.3f}")

            # 구성요소별 신뢰도
            component_confidences = confidence_analysis.get("component_confidences", {})
            if component_confidences:
                print(f"  📊 구성요소별 신뢰도:")
                component_names = {
                    "technical": "기술적 분석",
                    "macro": "매크로 환경",
                    "statistical_arb": "통계적 차익거래",
                    "rlmf_feedback": "RLMF 피드백",
                    "cross_validation": "교차 검증",
                }
                for component, value in component_confidences.items():
                    component_name = component_names.get(component, component)
                    print(f"    • {component_name}: {value:.3f}")
        else:
            print(f"  🎯 최종 신뢰도: {result.confidence:.3f}")

        # 고급 Quant 분석 결과 표시 (classification_result에서 직접 접근)
        enhanced_keys = [
            "quant_score",
            "regime_detection",
            "rlmf_analysis",
            "confidence_analysis",
            "advanced_indicators",
        ]

        # MarketAnalysisResult 객체에서 classification_result 접근
        classification_result = (
            result.classification_result if result.classification_result else {}
        )
        has_enhanced = any(
            key in classification_result and classification_result[key]
            for key in enhanced_keys
        )

        if has_enhanced:
            # Quant 점수
            if (
                "quant_score" in classification_result
                and classification_result["quant_score"]
            ):
                print(f"  🎯 Quant 점수: {classification_result['quant_score']:.3f}")

            # Regime 감지 결과
            if (
                "regime_detection" in classification_result
                and classification_result["regime_detection"]
            ):
                regime_det = classification_result["regime_detection"]
                print(f"  🔄 Regime 안정성: {regime_det.get('stability_score', 0):.3f}")
                print(
                    f"  ⏱️ 예상 지속기간: {regime_det.get('expected_duration', 'unknown')}"
                )
                if regime_det.get("shift_detected", False):
                    print(
                        f"  ⚠️ Regime 변화 감지! (점수: {regime_det.get('shift_score', 0):.3f})"
                    )

            # RLMF 분석 결과
            if (
                "rlmf_analysis" in classification_result
                and classification_result["rlmf_analysis"]
            ):
                rlmf = classification_result["rlmf_analysis"]
                if "market_feedback" in rlmf:
                    feedback = rlmf["market_feedback"]
                    avg_feedback = np.mean(list(feedback.values())) if feedback else 0.5
                    print(f"  🧠 RLMF 피드백: {avg_feedback:.3f}")

                if "statistical_arbitrage" in rlmf:
                    stat_arb = rlmf["statistical_arbitrage"]

                    direction = stat_arb.get("direction", "NEUTRAL")
                    signal_strength = stat_arb.get("signal_strength", 0)
                    confidence = stat_arb.get("confidence", 0)
                    print(
                        f"  📈 Statistical Arbitrage: {direction} (강도: {signal_strength:.3f}, 신뢰도: {confidence:.3f})"
                    )

                    # 개별 신호 표시
                    individual_signals = stat_arb.get("individual_signals", {})
                    if individual_signals:
                        print(f"    └─ 개별 신호:")
                        for metric, signal_data in individual_signals.items():
                            signal = signal_data.get("signal", 0)
                            ret = signal_data.get("return", 0)
                            weight = signal_data.get("weight", 0)
                            print(
                                f"      • {metric}: {signal:+.1f} (수익률: {ret:+.1%}, 가중치: {weight:.0%})"
                            )

            # 신뢰도 분석 결과
            if (
                "confidence_analysis" in classification_result
                and classification_result["confidence_analysis"]
            ):
                conf_analysis = classification_result["confidence_analysis"]
                adjusted_conf = conf_analysis.get("adjusted_confidence", 0.5)
                consistency = conf_analysis.get("consistency_score", 0.5)
                print(f"  🎯 조정된 신뢰도: {adjusted_conf:.3f}")
                print(f"  🔗 일관성 점수: {consistency:.3f}")

            # 고급 지표
            if (
                "advanced_indicators" in classification_result
                and classification_result["advanced_indicators"]
            ):
                indicators = classification_result["advanced_indicators"]
                if "volatility_structure" in indicators:
                    vol_struct = indicators["volatility_structure"]
                    vol_regime = vol_struct.get("vol_regime", "normal")
                    print(f"  📊 변동성 체제: {vol_regime}")

                if "momentum_structure" in indicators:
                    mom_struct = indicators["momentum_structure"]
                    mom_alignment = mom_struct.get("momentum_alignment", "mixed")
                    momentum_score = mom_struct.get("momentum_score", 0)
                    print(
                        f"  🚀 모멘텀 정렬: {mom_alignment} (점수: {momentum_score:.3f})"
                    )

                    # 상세 모멘텀 정보
                    short_mom = mom_struct.get("short_momentum", 0)
                    medium_mom = mom_struct.get("medium_momentum", 0)
                    long_mom = mom_struct.get("long_momentum", 0)
                    accel = mom_struct.get("momentum_acceleration", 0)
                    print(
                        f"    └─ 단기: {short_mom:+.1%}, 중기: {medium_mom:+.1%}, 장기: {long_mom:+.1%}, 가속도: {accel:+.1%}"
                    )

        # ML 기반 분석 (Random Forest)
        if hasattr(self, "use_random_forest") and self.use_random_forest:
            print("\n🤖 ML 기반 분석 (Random Forest):")
            if result.probabilities:
                # ML 확률을 내림차순으로 정렬
                sorted_probs = sorted(
                    result.probabilities.items(), key=lambda x: x[1], reverse=True
                )
                for i, (regime, prob) in enumerate(sorted_probs):
                    regime_name = regime.replace("_", " ").title()
                    rank_emoji = (
                        "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                    )
                    print(f"  {rank_emoji} {regime_name}: {prob:.1%}")

                # 최고 확률 체제 표시
                top_regime, top_prob = sorted_probs[0]
                top_regime_name = top_regime.replace("_", " ").title()
                print(f"  🎯 ML 예측 최고 확률: {top_regime_name} ({top_prob:.1%})")

            # Quant vs ML 비교
            if result.probabilities:
                quant_regime = result.current_regime.value
                ml_regime = sorted_probs[0][0].replace("_", " ").title()
                if quant_regime.upper() == ml_regime.upper():
                    print(f"  ✅ Quant와 ML 예측 일치: {quant_regime}")
                else:
                    print(
                        f"  ⚠️ Quant와 ML 예측 차이: Quant={quant_regime}, ML={ml_regime}"
                    )
        else:
            print("\n📈 체제별 확률 (Quant 기반):")
            if result.probabilities:
                for regime, prob in result.probabilities.items():
                    regime_name = regime.replace("_", " ").title()
                    print(f"  • {regime_name}: {prob:.1%}")

        # 2. 매크로 환경 분석
        print("\n🌍 2. 매크로 환경 분석")
        print("-" * 40)
        print(f"시장 조건: {result.macro_analysis.market_condition.value}")

        # 주요 지표 분석
        if result.macro_analysis.key_indicators:
            print("\n📊 주요 매크로 지표:")
            for indicator, value in result.macro_analysis.key_indicators.items():
                if isinstance(value, float):
                    print(f"  • {indicator}: {value:.3f}")
                else:
                    print(f"  • {indicator}: {value}")

        # 3. 섹터 분석
        print("\n🏭 3. 섹터 분석")
        print("-" * 40)

        # 섹터 로테이션 분석
        if result.macro_analysis.sector_rotation:
            print("📊 섹터별 강도 분석:")
            leading_sectors = []
            lagging_sectors = []
            neutral_sectors = []

            for sector, strength in result.macro_analysis.sector_rotation.items():
                if strength.value == "LEADING":
                    leading_sectors.append(sector)
                elif strength.value == "LAGGING":
                    lagging_sectors.append(sector)
                else:
                    neutral_sectors.append(sector)

            if leading_sectors:
                print(f"  🚀 선도 섹터: {', '.join(leading_sectors)}")
            if lagging_sectors:
                print(f"  📉 후행 섹터: {', '.join(lagging_sectors)}")
            if neutral_sectors:
                print(f"  ➡️ 중립 섹터: {', '.join(neutral_sectors)}")
        else:
            print("  📊 섹터 데이터 없음")
            print("  🔍 섹터 데이터 수집 중...")

            # 섹터 데이터 수집 시도
            try:
                spy_data, macro_data, sector_data = self._load_cached_data()
                if sector_data:
                    print(f"  ✅ 캐시된 섹터 데이터 발견: {list(sector_data.keys())}")
                    # 섹터 분석 재수행
                    sector_analysis = self.macro_analyzer.analyze_sector_rotation(
                        sector_data
                    )
                    if sector_analysis:
                        print("  📈 섹터 분석 결과:")
                        for sector, strength in sector_analysis.items():
                            strength_emoji = (
                                "🚀"
                                if strength.value == "LEADING"
                                else "📉" if strength.value == "LAGGING" else "➡️"
                            )
                            print(f"    {strength_emoji} {sector}: {strength.value}")
                    else:
                        print("  ⚠️ 섹터 분석 실패")
                else:
                    print("  ⚠️ 섹터 데이터를 찾을 수 없음")
            except Exception as e:
                print(f"  ❌ 섹터 데이터 로드 실패: {e}")

        # 섹터별 상세 정보 (가능한 경우)
        if (
            hasattr(result.macro_analysis, "sector_performance")
            and result.macro_analysis.sector_performance
        ):
            print("\n📈 섹터별 성과:")
            for sector, perf in result.macro_analysis.sector_performance.items():
                if isinstance(perf, dict):
                    return_val = perf.get("return", 0)
                    volatility = perf.get("volatility", 0)
                    print(
                        f"  • {sector}: 수익률 {return_val:+.2%}, 변동성 {volatility:.2%}"
                    )

        # 섹터별 상세 분석 (추가 정보)
        if result.macro_analysis.sector_rotation:
            print("\n🔍 섹터별 상세 분석:")

            # 섹터 분류 정보 (소문자로 통일)
            sector_classification = {
                "xlk": "Technology (순환적)",
                "xlf": "Financials (순환적)",
                "xle": "Energy (순환적)",
                "xlv": "Healthcare (방어적)",
                "xli": "Industrials (순환적)",
                "xlp": "Consumer Staples (방어적)",
                "xlu": "Utilities (방어적)",
                "xlb": "Materials (순환적)",
                "xlre": "Real Estate (순환적)",
            }

            for sector, strength in result.macro_analysis.sector_rotation.items():
                sector_name = sector_classification.get(sector, f"{sector} (미분류)")
                strength_emoji = (
                    "🚀"
                    if strength.value == "LEADING"
                    else "📉" if strength.value == "LAGGING" else "➡️"
                )
                print(f"  {strength_emoji} {sector}: {sector_name} - {strength.value}")

            # 섹터 로테이션 패턴 분석 (개선된 버전)
            leading_count = sum(
                1
                for s in result.macro_analysis.sector_rotation.values()
                if hasattr(s, "value") and s.value == "leading"
            )
            lagging_count = sum(
                1
                for s in result.macro_analysis.sector_rotation.values()
                if hasattr(s, "value") and s.value == "lagging"
            )
            neutral_count = sum(
                1
                for s in result.macro_analysis.sector_rotation.values()
                if hasattr(s, "value")
                and s.value in ["neutral", "cyclical", "defensive"]
            )
            total_sectors = len(result.macro_analysis.sector_rotation)

            if total_sectors > 0:
                leading_ratio = leading_count / total_sectors
                lagging_ratio = lagging_count / total_sectors
                neutral_ratio = neutral_count / total_sectors

                print(f"\n📊 섹터 로테이션 패턴:")
                print(
                    f"  • 선도 섹터 비율: {leading_ratio:.1%} ({leading_count}/{total_sectors})"
                )
                print(
                    f"  • 후행 섹터 비율: {lagging_ratio:.1%} ({lagging_count}/{total_sectors})"
                )
                print(
                    f"  • 중립 섹터 비율: {neutral_ratio:.1%} ({neutral_count}/{total_sectors})"
                )

                # 섹터 강도별 분류 개선
                if leading_count > 0:
                    print(
                        f"  🚀 선도 섹터: {', '.join([s for s, strength in result.macro_analysis.sector_rotation.items() if hasattr(strength, 'value') and strength.value == 'leading'])}"
                    )
                if lagging_count > 0:
                    print(
                        f"  📉 후행 섹터: {', '.join([s for s, strength in result.macro_analysis.sector_rotation.items() if hasattr(strength, 'value') and strength.value == 'lagging'])}"
                    )
                if neutral_count > 0:
                    print(
                        f"  ➡️ 중립 섹터: {', '.join([s for s, strength in result.macro_analysis.sector_rotation.items() if hasattr(strength, 'value') and strength.value in ['neutral', 'cyclical', 'defensive']])}"
                    )

                # 시장 전망 판단 개선
                if leading_ratio > 0.4:
                    print(f"  🎯 시장 전망: 순환적 섹터 선호 (강세장 신호)")
                elif lagging_ratio > 0.4:
                    print(f"  ⚠️ 시장 전망: 방어적 섹터 선호 (약세장 신호)")
                elif neutral_ratio > 0.6:
                    print(f"  ➡️ 시장 전망: 중립적 섹터 우세 (횡보장 신호)")
                else:
                    print(f"  🔄 시장 전망: 혼재된 신호 (방향성 불명확)")

                # 섹터 배치 추천
                print(f"\n📈 과중 배치 섹터:")
                leading_sectors = [
                    s
                    for s, strength in result.macro_analysis.sector_rotation.items()
                    if hasattr(strength, "value") and strength.value == "leading"
                ]
                for i, sector in enumerate(leading_sectors, 1):
                    print(f"  {i}. {sector}")

                print(f"\n📉 과소 배치 섹터:")
                lagging_sectors = [
                    s
                    for s, strength in result.macro_analysis.sector_rotation.items()
                    if hasattr(strength, "value") and strength.value == "lagging"
                ]
                for i, sector in enumerate(lagging_sectors, 1):
                    print(f"  {i}. {sector}")

                print(f"\n🔄 섹터 로테이션:")
                rotation_count = 1
                for sector in leading_sectors:
                    print(f"  {rotation_count}. OVERWEIGHT: {sector}")
                    rotation_count += 1
                for sector in lagging_sectors:
                    print(f"  {rotation_count}. UNDERWEIGHT: {sector}")
                    rotation_count += 1
            else:
                print(f"\n📊 섹터 로테이션 패턴:")
                print(f"  ⚠️ 섹터 데이터 없음")

        # 4. 하이퍼파라미터 최적화
        print("\n⚙️ 4. 하이퍼파라미터 최적화")
        print("-" * 40)

        # 최적화 성과
        opt_perf = result.optimization_performance
        strategy_sharpe = opt_perf.get("sharpe_ratio", 0)
        strategy_return = opt_perf.get("total_return", 0)
        strategy_drawdown = opt_perf.get("max_drawdown", 0)

        print(f"📊 최적화 전략 성과:")
        print(f"  • Sharpe Ratio: {strategy_sharpe:.4f}")
        print(f"  • Total Return: {strategy_return:.2%}")
        print(f"  • Max Drawdown: {strategy_drawdown:.2%}")

        # Buy & Hold 비교
        buyhold_return = opt_perf.get("buyhold_return", 0)
        buyhold_sharpe = opt_perf.get("buyhold_sharpe", 0)
        buyhold_drawdown = opt_perf.get("buyhold_drawdown", 0)

        if buyhold_return != 0:
            print(f"\n📈 Buy & Hold 비교:")
            print(f"  • Buy & Hold Return: {buyhold_return:.2%}")
            print(f"  • Buy & Hold Sharpe: {buyhold_sharpe:.4f}")
            print(f"  • Buy & Hold Max DD: {buyhold_drawdown:.2%}")

            # 성과 비교
            excess_return = strategy_return - buyhold_return
            excess_sharpe = strategy_sharpe - buyhold_sharpe
            print(f"\n🎯 성과 비교:")
            print(f"  • 초과 수익률: {excess_return:+.2%}")
            print(f"  • 초과 Sharpe: {excess_sharpe:+.4f}")

            if excess_return > 0:
                print(f"  ✅ 최적화 전략이 Buy & Hold 대비 우수")
            else:
                print(f"  ⚠️ 최적화 전략이 Buy & Hold 대비 열위")

        # 5. 전략 추천
        print("\n💡 5. 전략 추천")
        print("-" * 40)

        # 전략 추천 생성 (enhanced_recommendations가 없으면 기본 추천 생성)
        if result.enhanced_recommendations:
            rec = result.enhanced_recommendations
        else:
            # 기본 추천 생성
            rec = self._generate_basic_recommendations(
                result.macro_analysis, result.current_regime
            )

        # 주요 전략
        primary_strategy = rec.get("primary_strategy", "N/A")
        risk_level = rec.get("risk_level", "N/A")
        position_sizing = rec.get("position_sizing", "N/A")
        time_horizon = rec.get("time_horizon", "N/A")

        print(f"🎯 주요 전략: {primary_strategy}")
        print(f"⚠️ 위험 수준: {risk_level}")
        print(f"💰 포지션 사이징: {position_sizing}")
        print(f"⏱️ 투자 기간: {time_horizon}")

        # 구체적인 투자 전략 정보
        position_size = rec.get("position_size_percentage", 0.0)
        stop_loss = rec.get("stop_loss_percentage", 0.0)
        take_profit = rec.get("take_profit_percentage", 0.0)
        trailing_stop = rec.get("trailing_stop_percentage", 0.0)
        hedging = rec.get("hedging_strategy", "N/A")
        entry_points = rec.get("entry_points", [])
        exit_strategy = rec.get("exit_strategy", "N/A")

        print(f"\n📊 구체적인 투자 전략:")
        print(f"  💰 포지션 크기: {position_size:.1f}%")
        print(f"  🛑 손절: -{stop_loss:.1f}%")
        print(f"  🎯 익절: +{take_profit:.1f}%")
        print(f"  📈 트레일링 스탑: {trailing_stop:.1f}%")
        print(f"  🛡️ 헷징 전략: {hedging}")
        print(f"  🚪 진입 전략: {', '.join(entry_points) if entry_points else 'N/A'}")
        print(f"  🚪 청산 전략: {exit_strategy}")

        # 포트폴리오 배분
        portfolio_allocation = rec.get("portfolio_allocation", {})
        if portfolio_allocation:
            print(f"\n📈 포트폴리오 배분:")
            for asset, percentage in portfolio_allocation.items():
                asset_name = {
                    "equity": "주식",
                    "bonds_short": "단기채권",
                    "bonds_intermediate": "중기채권",
                    "bonds_long": "장기채권",
                    "gold": "금",
                    "commodities": "상품",
                    "cash": "현금",
                }.get(asset, asset)
                print(f"  • {asset_name}: {percentage:.1f}%")

        # 리스크 관리
        risk_management = rec.get("risk_management", {})
        if risk_management:
            print(f"\n⚠️ 리스크 관리:")
            max_dd = risk_management.get("max_drawdown", 0.0)
            pos_limit = risk_management.get("position_limit", 0.0)
            corr_limit = risk_management.get("correlation_limit", 0.0)
            print(f"  • 최대 손실: {max_dd:.1f}%")
            print(f"  • 포지션 한도: {pos_limit:.1f}%")
            print(f"  • 상관관계 한도: {corr_limit:.1f}")

        # SPY 진입/매도 포인트 (classification_result에서 가져오기)
        if (
            result.classification_result
            and "spy_entry_exit_points" in result.classification_result
        ):
            spy_points = result.classification_result["spy_entry_exit_points"]
            if spy_points:
                current_price = spy_points.get("current_price", 0.0)
                tech_indicators = spy_points.get("technical_indicators", {})

                print(f"\n📊 SPY 진입/매도 포인트 (현재가: ${current_price:.2f}):")

                # 기술적 지표
                if tech_indicators:
                    rsi = tech_indicators.get("rsi", 0.0)
                    macd = tech_indicators.get("macd", 0.0)
                    macd_signal = tech_indicators.get("macd_signal", 0.0)
                    sma_20 = tech_indicators.get("sma_20", 0.0)
                    sma_50 = tech_indicators.get("sma_50", 0.0)
                    bb_upper = tech_indicators.get("bb_upper", 0.0)
                    bb_lower = tech_indicators.get("bb_lower", 0.0)

                    print(f"  📈 기술적 지표:")
                    print(f"    • RSI: {rsi:.1f}")
                    print(f"    • MACD: {macd:.2f} (Signal: {macd_signal:.2f})")
                    print(f"    • SMA20: ${sma_20:.2f}")
                    print(f"    • SMA50: ${sma_50:.2f}")
                    print(f"    • 볼린저 상단: ${bb_upper:.2f}")
                    print(f"    • 볼린저 하단: ${bb_lower:.2f}")

                # 지지/저항선
                support_levels = spy_points.get("support_levels", [])
                resistance_levels = spy_points.get("resistance_levels", [])

                if support_levels:
                    print(f"  🛡️ 지지선:")
                    for i, level in enumerate(support_levels[:3], 1):
                        print(f"    {i}. ${level:.2f}")

                if resistance_levels:
                    print(f"  🚧 저항선:")
                    for i, level in enumerate(resistance_levels[:3], 1):
                        print(f"    {i}. ${level:.2f}")

                # 진입/매도 신호
                entry_signals = spy_points.get("entry_signals", [])
                exit_signals = spy_points.get("exit_signals", [])

                if entry_signals:
                    print(f"  🚪 진입 신호:")
                    for signal in entry_signals[:3]:
                        print(f"    • {signal}")

                if exit_signals:
                    print(f"  🚪 매도 신호:")
                    for signal in exit_signals[:3]:
                        print(f"    • {signal}")

                # 손절/익절 레벨
                stop_loss_levels = spy_points.get("stop_loss_levels", [])
                take_profit_levels = spy_points.get("take_profit_levels", [])

                if stop_loss_levels:
                    print(f"  🛑 손절 레벨:")
                    for i, level in enumerate(stop_loss_levels[:3], 1):
                        print(f"    {i}. ${level:.2f}")

                if take_profit_levels:
                    print(f"  🎯 익절 레벨:")
                    for i, level in enumerate(take_profit_levels[:3], 1):
                        print(f"    {i}. ${level:.2f}")

        # 전략 상세 설명
        strategy_explanations = {
            "RANGE_TRADING": "지지/저항선을 활용한 범위 내 거래 전략",
            "MOMENTUM_FOLLOWING": "상승 모멘텀을 따라가는 추세 추종 전략",
            "MEAN_REVERSION": "평균 회귀를 활용한 반대 포지션 전략",
            "DEFENSIVE_POSITIONING": "위험 회피를 위한 방어적 포지션 전략",
            "INFLATION_HEDGE": "인플레이션 헤지를 위한 실물자산 투자 전략",
            "RECESSION_HEDGE": "경기침체 헤지를 위한 안전자산 투자 전략",
        }

        if primary_strategy in strategy_explanations:
            print(f"📖 전략 설명: {strategy_explanations[primary_strategy]}")

        # 시장 상황별 전략 적합성
        current_regime = result.current_regime.value
        regime_strategy_fit = {
            "SIDEWAYS": "RANGE_TRADING",
            "TRENDING_UP": "MOMENTUM_FOLLOWING",
            "TRENDING_DOWN": "DEFENSIVE_POSITIONING",
            "VOLATILE": "DEFENSIVE_POSITIONING",
        }

        if current_regime in regime_strategy_fit:
            recommended_strategy = regime_strategy_fit[current_regime]
            if primary_strategy == recommended_strategy:
                print(f"✅ 전략 적합성: 현재 시장 체제({current_regime})와 전략이 일치")
            else:
                print(
                    f"⚠️ 전략 적합성: 현재 시장 체제({current_regime})에는 {recommended_strategy} 전략이 더 적합할 수 있음"
                )

        # 핵심 액션
        if "key_actions" in rec and rec["key_actions"]:
            print(f"\n🚀 핵심 액션:")
            for i, action in enumerate(rec["key_actions"], 1):
                print(f"  {i}. {action}")

        # 위험 경고
        if "risk_warnings" in rec and rec["risk_warnings"]:
            print(f"\n⚠️ 위험 경고:")
            for i, warning in enumerate(rec["risk_warnings"], 1):
                print(f"  {i}. {warning}")

        # 추가 전략적 고려사항
        print(f"\n🔍 추가 전략적 고려사항:")

        # 신뢰도 기반 포지션 사이징
        confidence = result.confidence
        if confidence > 0.7:
            print(f"  • 높은 신뢰도({confidence:.1%}) - 공격적 포지션 사이징 고려")
        elif confidence < 0.4:
            print(f"  • 낮은 신뢰도({confidence:.1%}) - 보수적 포지션 사이징 권장")
        else:
            print(f"  • 중간 신뢰도({confidence:.1%}) - 중립적 포지션 사이징")

        # 변동성 기반 리스크 관리
        if "volatility_regime" in result.macro_analysis.key_indicators:
            vol_regime = result.macro_analysis.key_indicators["volatility_regime"]
            if vol_regime == "high":
                print(f"  • 높은 변동성 환경 - 스탑로스 설정 및 포지션 크기 축소 권장")
            elif vol_regime == "normal":
                print(f"  • 정상 변동성 환경 - 표준 리스크 관리 적용")

        # 매크로 환경 기반 추가 고려사항
        market_condition = result.macro_analysis.market_condition.value
        if market_condition == "bull_market":
            print(f"  • 강세장 환경 - 모멘텀 전략 및 레버리지 활용 고려")
        elif market_condition == "recession_fear":
            print(f"  • 경기침체 우려 - 현금 비중 확대 및 안전자산 비중 확대")
        elif market_condition == "inflation_fear":
            print(f"  • 인플레이션 우려 - 실물자산 및 TIPS 비중 확대")

        else:
            print("  📊 추천 데이터 없음")

        print("\n" + "=" * 80)

    def _print_enhanced_summary(self, result: MarketAnalysisResult):
        """고도화된 분석 요약 출력"""
        print("\n🚀 고도화된 분석 결과 요약:")

        # Quant vs ML 비교
        if (
            hasattr(self, "use_random_forest")
            and self.use_random_forest
            and result.probabilities
        ):
            quant_regime = result.current_regime.value
            sorted_probs = sorted(
                result.probabilities.items(), key=lambda x: x[1], reverse=True
            )
            ml_regime = sorted_probs[0][0].replace("_", " ").title()

            print(f"📊 Quant 기반 체제: {quant_regime}")
            print(f"🤖 ML 기반 체제: {ml_regime}")

            if quant_regime.upper() == ml_regime.upper():
                print(f"✅ Quant와 ML 예측 일치")
            else:
                print(f"⚠️ Quant와 ML 예측 차이")
        else:
            print(f"현재 체제: {result.current_regime.value}")

        print(f"최종 신뢰도: {result.final_confidence.get('final_confidence', 0):.3f}")
        print(f"매크로 조건: {result.macro_analysis.market_condition.value}")

        if result.llm_api_insights:
            api_stats = result.llm_api_insights.get("api_stats", {})
            print(f"LLM API 성공률: {api_stats.get('success_rate', 0):.2%}")

        print(
            f"주요 전략: {result.enhanced_recommendations.get('primary_strategy', 'N/A')}"
        )
        print(
            f"포지션 사이징: {result.enhanced_recommendations.get('position_sizing', 'N/A')}"
        )

    def enable_llm_api(self, llm_config: Dict[str, Any]):
        """LLM API 시스템 활성화"""
        try:
            self.llm_api_system = LLMAPIIntegration(llm_config)
            self.llm_config = llm_config
            self.logger.info("LLM API 통합 시스템 활성화됨")
        except Exception as e:
            self.logger.error(f"LLM API 시스템 활성화 실패: {e}")

    def disable_llm_api(self):
        """LLM API 시스템 비활성화"""
        self.llm_api_system = None
        self.llm_config = None
        self.logger.info("LLM API 통합 시스템 비활성화됨")

    def get_llm_api_stats(self) -> Dict[str, Any]:
        """LLM API 통계 반환"""
        if self.llm_api_system:
            return self.llm_api_system.get_api_stats()
        return {"status": "disabled"}


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="고도화된 Market Sensor 시스템")
    parser.add_argument(
        "--analysis-type",
        choices=["basic", "enhanced"],
        default="enhanced",
        help="분석 유형 선택 (기본값: enhanced)",
    )
    parser.add_argument(
        "--output-dir", default="results/macro/enhanced", help="결과 출력 디렉토리"
    )
    parser.add_argument(
        "--use-cached-data", action="store_true", help="저장된 매크로 데이터 사용"
    )
    parser.add_argument(
        "--use-cached-optimization", action="store_true", help="저장된 최적화 결과 사용"
    )
    parser.add_argument("--cache-days", type=int, default=1, help="캐시 유효기간 (일)")
    parser.add_argument("--enable-llm", action="store_true", help="LLM API 활성화")
    parser.add_argument(
        "--use-random-forest",
        action="store_true",
        default=True,
        help="Random Forest 모델 사용 (기본값: True)",
    )
    parser.add_argument(
        "--retrain-rf-model", action="store_true", help="Random Forest 모델 재학습"
    )
    parser.add_argument(
        "--no-random-forest",
        action="store_true",
        help="Random Forest 모델 사용 안함 (규칙 기반 사용)",
    )

    args = parser.parse_args()

    # Random Forest 옵션 처리
    use_random_forest = args.use_random_forest and not args.no_random_forest
    retrain_rf_model = args.retrain_rf_model

    print("🚀 고도화된 Market Sensor 시스템 시작")
    print(f"분석 유형: {args.analysis_type}")
    print(
        f"캐시 설정: 데이터={args.use_cached_data}, 최적화={args.use_cached_optimization}"
    )
    print(f"Random Forest: {'사용' if use_random_forest else '사용 안함'}")
    if use_random_forest and retrain_rf_model:
        print("Random Forest 모델 재학습 모드")

    # LLM API 설정 (선택사항)
    llm_config = None
    if args.enable_llm:
        llm_config = {
            "provider": "hybrid",
            "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "fallback_to_rules": True,
        }

    # Market Sensor 초기화
    sensor = MarketSensor(
        enable_llm_api=args.enable_llm,
        llm_config=llm_config,
        use_cached_data=args.use_cached_data,
        use_cached_optimization=args.use_cached_optimization,
        cache_days=args.cache_days,
        use_random_forest=use_random_forest,
        retrain_rf_model=retrain_rf_model,
    )

    # 분석 수행
    if args.analysis_type == "basic":
        result = sensor.run_basic_analysis(output_dir=args.output_dir, verbose=True)
    else:
        result = sensor.run_enhanced_analysis(output_dir=args.output_dir, verbose=True)

    if result:
        print("\n🎉 분석 완료!")
    else:
        print("\n❌ 분석 실패!")


if __name__ == "__main__":
    main()
