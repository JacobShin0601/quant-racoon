"""
HMM-Neural 하이브리드 트레이더
모든 컴포넌트를 통합하여 실행하는 메인 트레이더 클래스

컴포넌트 순서:
1. 데이터 수집 (매크로 + 개별 종목)
2. HMM 시장 체제 분류
3. 신경망 개별 종목 예측
4. 투자 점수 생성
5. 매매 신호 생성
6. 포트폴리오 권고 종합
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
import glob
import uuid

warnings.filterwarnings("ignore")

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

# Actions 컴포넌트 임포트
from src.actions.hmm_regime_classifier import MarketRegimeHMM, RegimeTransitionAnalyzer
from src.actions.neural_stock_predictor import StockPredictionNetwork
from src.actions.investment_scorer import (
    InvestmentScoreGenerator,
    PortfolioScoreAggregator,
)
from src.actions.trading_signal_generator import (
    TradingSignalGenerator,
    PortfolioSignalAggregator,
)

# 포트폴리오 최적화 및 백테스팅 임포트
try:
    from .neural_portfolio_manager import NeuralPortfolioManager
    from .evaluator import TrainTestEvaluator
    from .formatted_output import formatted_output
except ImportError:
    from src.agent.neural_portfolio_manager import NeuralPortfolioManager
    from src.agent.evaluator import TrainTestEvaluator
    from src.agent.formatted_output import formatted_output

# 기존 액션들 임포트
from src.actions.y_finance import YahooFinanceDataCollector
from src.actions.global_macro import GlobalMacroDataCollector, MacroSectorAnalyzer

# 에이전트 헬퍼 임포트
try:
    from src.agent.helper import ConfigLoader, DataValidator, ResultSaver
except ImportError:
    # helper.py가 없으면 기본 구현 사용
    pass

logger = logging.getLogger(__name__)


class HybridTrader:
    """
    HMM + Neural Network 하이브리드 트레이더

    주요 기능:
    - 시장 체제 분류 (HMM)
    - 개별 종목 예측 (Neural Network)
    - 투자 점수 생성 (-1~1 스케일)
    - 매매 신호 생성
    - 포트폴리오 종합 권고
    """

    def __init__(
        self,
        config_path: str = "config/config_trader.json",
        analysis_mode: bool = False,
    ):
        """
        트레이더 초기화

        Args:
            config_path: 설정 파일 경로
            analysis_mode: 분석 모드 (5단계용, 불필요한 초기화 건너뛰기)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.analysis_mode = analysis_mode

        # 로깅 설정
        self._setup_logging()

        if not analysis_mode:
            # 전체 초기화 (1-4단계용)
            self._full_initialization()
        else:
            # 분석 모드 초기화 (5단계용)
            self._analysis_mode_initialization()

        # 상태 관리
        self.is_initialized = False
        self.last_run_time = None
        self.regime_history = []

        # 캐시 설정
        self.use_cached_data = self.config.get("data", {}).get("use_cached_data", True)
        self.model_version = "v1.0"  # 모델 버전 정보

    def _full_initialization(self):
        """전체 초기화 (1-4단계용)"""
        # 컴포넌트 초기화
        self.regime_classifier = MarketRegimeHMM(self.config)
        self.regime_analyzer = RegimeTransitionAnalyzer(self.config)
        self.neural_predictor = StockPredictionNetwork(self.config)
        self.score_generator = InvestmentScoreGenerator(self.config)
        self.signal_generator = TradingSignalGenerator(self.config)
        self.portfolio_aggregator = PortfolioSignalAggregator(self.config)

        # 포트폴리오 최적화 및 백테스팅 매니저
        self.portfolio_manager = NeuralPortfolioManager(self.config)
        self.evaluator = None  # 필요시 초기화

        # 최적화된 임계점을 신호 생성기에 적용
        optimized_thresholds = self.portfolio_manager.get_signal_thresholds()
        if optimized_thresholds:
            logger.info(f"🎯 최적화된 임계점 적용: {optimized_thresholds}")
            # 신호 생성기의 임계점 업데이트
            self.signal_generator.update_thresholds(optimized_thresholds)

        # 데이터 소스
        self.data_loader = YahooFinanceDataCollector()
        self.macro_collector = GlobalMacroDataCollector()
        self.macro_analyzer = MacroSectorAnalyzer()

    def _analysis_mode_initialization(self):
        """분석 모드 초기화 (5단계용) - 최소한의 컴포넌트만 초기화"""
        logger.info("🔧 분석 모드 초기화 시작...")

        # 필수 컴포넌트만 초기화 (설정만 로드, 실제 초기화는 나중에)
        self.regime_classifier = None
        self.neural_predictor = None
        self.score_generator = None

        # signal_generator는 threshold 로드를 위해 초기화
        self.signal_generator = TradingSignalGenerator(self.config)

        self.portfolio_aggregator = None

        # 포트폴리오 관리자도 None으로 초기화 (필요시 동적 초기화됨)
        self.portfolio_manager = None
        self.evaluator = None

        # 데이터 소스도 None으로 초기화
        self.data_loader = None
        self.macro_collector = None
        self.macro_analyzer = None

        logger.info("✅ 분석 모드 초기화 완료")

    def _load_config(self) -> Dict:
        """설정 파일 로드 - config_trader.json과 config_swing.json 통합"""
        try:
            # config_trader.json 로드
            with open(self.config_path, "r", encoding="utf-8") as f:
                trader_config = json.load(f)

            # config_swing.json 로드 (추가 설정)
            swing_config_path = "config/config_swing.json"
            swing_config = {}
            try:
                with open(swing_config_path, "r", encoding="utf-8") as f:
                    swing_config = json.load(f)
                logger.info(f"스윙 설정 로드 완료: {swing_config_path}")
            except Exception as e:
                logger.warning(f"스윙 설정 로드 실패 (기본값 사용): {e}")

            # 설정 통합 (trader_config 우선, swing_config로 보완)
            merged_config = self._merge_configs(trader_config, swing_config)

            logger.info(f"설정 로드 완료: {self.config_path}")
            return merged_config

        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            # 기본 설정 반환
            return self._get_default_config()

    def _merge_configs(self, trader_config: Dict, swing_config: Dict) -> Dict:
        """trader_config와 swing_config 통합"""
        merged = trader_config.copy()

        # data 섹션 통합
        if "data" in swing_config:
            if "data" not in merged:
                merged["data"] = {}
            merged["data"].update(swing_config["data"])

        # strategies 섹션 추가
        if "strategies" in swing_config:
            merged["strategies"] = swing_config["strategies"]

        # trading 섹션 통합
        if "trading" in swing_config:
            if "trading" not in merged:
                merged["trading"] = {}
            merged["trading"].update(swing_config["trading"])

        # portfolio 섹션 통합
        if "portfolio" in swing_config:
            if "portfolio" not in merged:
                merged["portfolio"] = {}
            merged["portfolio"].update(swing_config["portfolio"])

        # evaluator 섹션 추가
        if "evaluator" in swing_config:
            merged["evaluator"] = swing_config["evaluator"]

        # automation 섹션 통합
        if "automation" in swing_config:
            if "automation" not in merged:
                merged["automation"] = {}
            merged["automation"].update(swing_config["automation"])

        # output 섹션 통합
        if "output" in swing_config:
            if "output" not in merged:
                merged["output"] = {}
            merged["output"].update(swing_config["output"])

        return merged

    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            "data": {
                "symbols": ["AAPL", "QQQ", "SPY"],
                "interval": "1d",
                "lookback_days": 360,
            },
            "logging": {"level": "INFO"},
            "hmm_regime": {"n_states": 4},
            "neural_network": {"hidden_layers": [32, 16]},
            "scoring": {"volatility_penalty": 0.3},
            "signal_generation": {"min_confidence": 0.4},
        }

    def _setup_logging(self):
        """로깅 설정"""
        try:
            log_config = self.config.get("logging", {})
            log_level = getattr(logging, log_config.get("level", "INFO"))

            # 기본 로거 설정
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler("log/trader.log", encoding="utf-8"),
                ],
            )

            logger.info("로깅 설정 완료")

        except Exception as e:
            print(f"로깅 설정 실패: {e}")

    def initialize_models(self, force_retrain: bool = False) -> bool:
        """
        모델 초기화 및 로드 (재학습 없음)

        Args:
            force_retrain: 강제 재학습 여부 (현재는 사용하지 않음)

        Returns:
            초기화 성공 여부
        """
        try:
            logger.info("모델 초기화 시작...")

            # 기존 모델 로드만 시도 (재학습 없음)
            if self._load_existing_models():
                logger.info("기존 모델 로드 성공")
                self.is_initialized = True
                return True
            else:
                logger.error("기존 모델 로드 실패. 먼저 1-3단계에서 모델을 학습하세요.")
                return False

        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            return False

    def _get_latest_macro_session_dir(self, macro_dir: str) -> Optional[str]:
        """가장 최근 세션 디렉토리 경로 반환"""
        if not os.path.exists(macro_dir):
            return None
        session_dirs = [
            d
            for d in os.listdir(macro_dir)
            if os.path.isdir(os.path.join(macro_dir, d))
        ]
        if not session_dirs:
            return None
        latest_session = max(
            session_dirs, key=lambda x: os.path.getctime(os.path.join(macro_dir, x))
        )
        return os.path.join(macro_dir, latest_session)

    def _load_cached_macro_data(self) -> Optional[pd.DataFrame]:
        """market_sensor.py 방식: 세션별 디렉토리에서 매크로 데이터 로드"""
        try:
            macro_dir = "data/macro"
            session_path = self._get_latest_macro_session_dir(macro_dir)
            if not session_path:
                logger.warning("매크로 데이터 세션 디렉토리가 없습니다.")
                return None

            # 모든 매크로 파일 로드
            macro_data = {}
            for file in os.listdir(session_path):
                if file.endswith(".csv") and not file.endswith("_sector.csv"):
                    symbol = file.replace(".csv", "")
                    file_path = os.path.join(session_path, file)
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    macro_data[symbol] = df

            # 주요 매크로 심볼들 (모든 기술적 지표 포함)
            required_symbols = [
                "^VIX",
                "^TNX",
                "^IRX",
                "SPY",
                "UUP",
                "GLD",
                "QQQ",
                "^DXY",
            ]
            combined_macro = pd.DataFrame()

            for symbol in required_symbols:
                if symbol in macro_data and not macro_data[symbol].empty:
                    symbol_data = macro_data[symbol].copy()

                    # 컬럼명 소문자로 통일
                    symbol_data.columns = [col.lower() for col in symbol_data.columns]

                    # 모든 컬럼에 심볼 접두사 추가
                    symbol_prefix = symbol.lower().replace("^", "")
                    for col in symbol_data.columns:
                        if col not in ["date", "time", "timestamp"]:  # 메타데이터 제외
                            combined_macro[f"{symbol_prefix}_{col}"] = symbol_data[col]

            if not combined_macro.empty:
                logger.info(
                    f"세션 캐시 매크로 데이터 로드 완료: {len(combined_macro)}개 행, {len(combined_macro.columns)}개 컬럼"
                )
                return combined_macro
            else:
                logger.warning("세션 캐시에서 유효한 매크로 데이터가 없습니다.")
                return None

        except Exception as e:
            logger.error(f"세션 캐시 매크로 데이터 로드 실패: {e}")
            return None

    def _save_macro_data_to_session(self, macro_data: Dict[str, pd.DataFrame]):
        """market_sensor.py 방식: 세션별 디렉토리 및 data/macro 루트에 저장"""
        try:
            macro_dir = "data/macro"
            session_uuid = str(uuid.uuid4())
            session_dir = os.path.join(macro_dir, session_uuid)
            os.makedirs(session_dir, exist_ok=True)
            for symbol, df in macro_data.items():
                file_name = f"{symbol}.csv"
                file_path = os.path.join(session_dir, file_name)
                df.to_csv(file_path)
                # 루트에도 저장
                root_path = os.path.join(macro_dir, file_name)
                df.to_csv(root_path)
            logger.info(f"매크로 데이터 세션 저장 완료: {session_dir}")
        except Exception as e:
            logger.error(f"매크로 데이터 세션 저장 실패: {e}")

    def _collect_macro_data(self) -> Optional[pd.DataFrame]:
        """캐시된 매크로 데이터 우선 사용, 없을 때만 새로 수집"""
        try:
            logger.info("매크로 데이터 로드 시작...")

            # 1. 캐시된 데이터 우선 시도
            cached = self._load_cached_macro_data()
            if cached is not None and len(cached) > 100:
                logger.info(
                    f"✅ 캐시된 매크로 데이터 사용: {len(cached)}개 행, {len(cached.columns)}개 컬럼"
                )
                return cached

            # 2. 캐시가 없거나 부족한 경우에만 새로 수집
            logger.warning("캐시된 매크로 데이터가 없어 새로 수집합니다.")

            lookback_days = self.config["data"].get("lookback_days", 360)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            spy_data, macro_data, sector_data = self.macro_collector.collect_all_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            # 모든 매크로 데이터를 하나의 통합 데이터프레임으로 결합
            combined_macro = pd.DataFrame()

            # 주요 매크로 심볼들 (모든 기술적 지표 포함)
            required_symbols = [
                "^VIX",
                "^TNX",
                "^IRX",
                "SPY",
                "UUP",
                "GLD",
                "QQQ",
                "^DXY",
            ]

            for symbol in required_symbols:
                if symbol in macro_data and not macro_data[symbol].empty:
                    symbol_data = macro_data[symbol].copy()

                    # 컬럼명 소문자로 통일
                    symbol_data.columns = [col.lower() for col in symbol_data.columns]

                    # datetime 컬럼이 있으면 인덱스로 설정
                    if "datetime" in symbol_data.columns:
                        symbol_data["datetime"] = pd.to_datetime(
                            symbol_data["datetime"]
                        )
                        symbol_data.set_index("datetime", inplace=True)

                    # 모든 컬럼에 심볼 접두사 추가
                    symbol_prefix = symbol.lower().replace("^", "")
                    for col in symbol_data.columns:
                        if col not in ["date", "time", "timestamp"]:  # 메타데이터 제외
                            combined_macro[f"{symbol_prefix}_{col}"] = symbol_data[col]

            if not combined_macro.empty:
                # 세션 저장
                self._save_macro_data_to_session(macro_data)
                logger.info(
                    f"새로 수집한 매크로 데이터 사용: {len(combined_macro)}개 행, {len(combined_macro.columns)}개 컬럼"
                )
                return combined_macro

            logger.warning("매크로 데이터 없음 - 기본 데이터 생성")
            dates = pd.date_range(start=start_date, end=end_date, freq="D")

            # 기본 매크로 데이터 (기술적 지표 포함)
            basic_indicators = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "macd",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "stoch_k",
                "stoch_d",
                "williams_r",
                "cci",
                "adx",
                "plus_di",
                "minus_di",
                "obv",
                "donchian_upper",
                "donchian_middle",
                "donchian_lower",
                "keltner_upper",
                "keltner_middle",
                "keltner_lower",
                "atr",
                "ema_short",
                "ema_long",
                "returns",
                "volatility",
            ]

            # 기본 데이터 생성
            for symbol in required_symbols:
                symbol_prefix = symbol.lower().replace("^", "")
                for indicator in basic_indicators:
                    combined_macro[f"{symbol_prefix}_{indicator}"] = np.random.normal(
                        0, 1, len(dates)
                    )

            combined_macro.index = dates
            logger.info(f"기본 매크로 데이터 생성: {len(combined_macro)}개 행")
            return combined_macro

        except Exception as e:
            logger.error(f"매크로 데이터 수집 실패: {e}")
            return None

    def _collect_stock_training_data(self) -> Dict:
        """개별 종목 학습 데이터 수집 - scrapper.py 방식 참고"""
        try:
            logger.info("🚀 _collect_stock_training_data 메서드 시작")

            # 설정에서 심볼과 설정 가져오기
            data_config = self.config.get("data", {})
            symbols = data_config.get("symbols", [])

            if not symbols:
                logger.error("수집할 심볼이 설정되지 않았습니다.")
                return {}

            training_data = {}
            success_count = 0

            logger.info(f"📈 {len(symbols)}개 종목 데이터 수집 중...")
            logger.info(f"📋 수집할 심볼들: {symbols}")

            # 현재 시장 체제 예측 (학습용)
            macro_data = self._collect_macro_data()
            market_regime = self.regime_classifier.predict_regime(macro_data)

            for symbol in symbols:
                try:
                    # logger.info(f"  🔍 {symbol} 데이터 수집 중...")

                    # 캐시된 데이터 사용 옵션 확인
                    if self.use_cached_data:
                        cached_stock_data = self._load_cached_stock_data(symbol)
                        if cached_stock_data is not None:
                            logger.info(f"    📋 {symbol} 캐시된 데이터 사용")
                            stock_data = cached_stock_data
                        else:
                            logger.warning(
                                f"    ⚠️ {symbol} 캐시된 데이터 없음, 새로 수집"
                            )
                            stock_data = self._get_stock_data_from_api(symbol)
                    else:
                        stock_data = self._get_stock_data_from_api(symbol)

                    if (
                        stock_data is not None
                        and not stock_data.empty
                        and len(stock_data) > 50
                    ):
                        # 컬럼명 소문자로 통일
                        stock_data.columns = [col.lower() for col in stock_data.columns]

                        # 피처 생성 (매크로 데이터 포함)
                        logger.info(f"    🔧 {symbol} 피처 생성 시작...")
                        features = self.neural_predictor.create_features(
                            stock_data, symbol, market_regime, macro_data
                        )
                        logger.info(
                            f"    📊 {symbol} 피처 생성 결과: {features.shape if features is not None else 'None'}"
                        )

                        # config에서 미래 수익률 기간 읽기
                        forward_days_config = self.config.get("neural_network", {}).get(
                            "target_forward_days", [22, 66]
                        )

                        # 단일 값인 경우 리스트로 변환
                        if isinstance(forward_days_config, int):
                            forward_days_list = [forward_days_config]
                        else:
                            forward_days_list = forward_days_config

                        # 타겟 생성 (미래 수익률)
                        logger.info(
                            f"    🎯 {symbol} 멀티타겟 생성 시작... (forward_days={forward_days_list})"
                        )
                        target = self._create_multi_target_variable(
                            stock_data, forward_days_list
                        )
                        logger.info(
                            f"    📈 {symbol} 멀티타겟 생성 결과: {target.shape if target is not None else 'None'}"
                        )

                        if len(features) > 0 and len(target) > 0:
                            training_data[symbol] = {
                                "features": features,
                                "target": target,
                                "data": stock_data,
                            }
                            logger.info(
                                f"    ✅ {symbol} 학습 데이터 준비 완료: {len(stock_data)}개 포인트"
                            )
                            success_count += 1
                        else:
                            logger.warning(
                                f"    ❌ {symbol}: 피처/타겟 생성 실패 (features: {len(features) if features is not None else 'None'}, target: {len(target) if target is not None else 'None'})"
                            )
                    else:
                        logger.warning(f"    ❌ {symbol}: 데이터 부족")

                except Exception as e:
                    logger.error(f"    ❌ {symbol} 데이터 수집 실패: {e}")
                    continue

            logger.info(
                f"✅ 데이터 수집 완료: {success_count}/{len(symbols)}개 종목 성공"
            )
            logger.info(
                f"🏁 _collect_stock_training_data 메서드 완료, 반환 데이터: {len(training_data)}개 종목"
            )
            return training_data

        except Exception as e:
            logger.error(f"종목 데이터 수집 실패: {e}")
            logger.error(f"❌ _collect_stock_training_data 메서드 실패")
            return {}

    def _get_stock_data_from_api(self, symbol: str) -> Optional[pd.DataFrame]:
        """API에서 종목 데이터 수집"""
        try:
            data_config = self.config.get("data", {})

            # 종목 정보 가져오기
            info = self.data_loader.get_stock_info(symbol)
            logger.info(f"    📋 {info['name']} ({info['sector']})")

            # 기본 데이터 수집
            stock_data = self.data_loader.get_candle_data(
                symbol=symbol,
                interval=data_config.get("interval", "1d"),
                start_date=data_config.get("start_date"),
                end_date=data_config.get("end_date"),
                days_back=data_config.get("lookback_days", 360),
            )

            # 데이터 수집 성공 시 캐시에 저장
            if stock_data is not None and not stock_data.empty and len(stock_data) > 50:
                self._save_stock_data_to_cache(symbol, stock_data)

            return stock_data

        except Exception as e:
            logger.error(f"API에서 {symbol} 데이터 수집 실패: {e}")
            return None

    def _load_cached_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """캐시된 종목 데이터 로드"""
        try:
            # data/trader/ 폴더에서 캐시된 데이터 찾기
            trader_data_dir = "data/trader"
            if not os.path.exists(trader_data_dir):
                return None

            # 가장 최근 파일 찾기 (패턴: SYMBOL_daily_*.csv)
            pattern = f"{symbol}_daily_*.csv"
            matching_files = glob.glob(os.path.join(trader_data_dir, pattern))

            if not matching_files:
                return None

            # 가장 최근 파일 선택
            latest_file = max(matching_files, key=os.path.getctime)

            logger.info(
                f"    📁 {symbol} 캐시 파일 로드: {os.path.basename(latest_file)}"
            )

            # CSV 파일 로드
            stock_data = pd.read_csv(latest_file, index_col=0, parse_dates=True)

            if stock_data.empty or len(stock_data) < 50:
                logger.warning(
                    f"    ⚠️ {symbol} 캐시 데이터 부족: {len(stock_data)}개 행"
                )
                return None

            return stock_data

        except Exception as e:
            logger.error(f"캐시된 {symbol} 데이터 로드 실패: {e}")
            return None

    def _save_stock_data_to_cache(self, symbol: str, stock_data: pd.DataFrame) -> bool:
        """종목 데이터를 캐시에 저장"""
        try:
            # data/trader/ 폴더 생성
            trader_data_dir = "data/trader"
            os.makedirs(trader_data_dir, exist_ok=True)

            # 파일명 생성 (타임스탬프 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_daily_{timestamp}.csv"
            filepath = os.path.join(trader_data_dir, filename)

            # CSV로 저장
            stock_data.to_csv(filepath)

            logger.info(f"    💾 {symbol} 데이터 캐시 저장: {filename}")
            return True

        except Exception as e:
            logger.error(f"{symbol} 데이터 캐시 저장 실패: {e}")
            return False

    def _create_target_variable(
        self, stock_data: pd.DataFrame, forward_days: int = 5
    ) -> pd.Series:
        """
        타겟 변수 생성 (미래 수익률을 -1~1로 정규화)

        Args:
            stock_data: 주식 데이터
            forward_days: 미래 며칠 수익률을 사용할지

        Returns:
            정규화된 타겟 시리즈
        """
        try:
            close = stock_data["close"]

            # 미래 수익률 계산
            future_returns = close.shift(-forward_days) / close - 1

            # -1 ~ 1 정규화 (tanh 함수 사용)
            normalized_returns = np.tanh(future_returns * 5)  # 5배 스케일링 후 tanh

            return normalized_returns.dropna()

        except Exception as e:
            logger.error(f"타겟 변수 생성 실패: {e}")
            return pd.Series()

    def _create_multi_target_variable(
        self, stock_data: pd.DataFrame, forward_days_list: List[int]
    ) -> pd.DataFrame:
        """
        여러 기간의 미래 수익률을 동시에 생성 (멀티타겟)

        Args:
            stock_data: 주식 데이터
            forward_days_list: 예측할 미래 기간 리스트 (예: [22, 66])

        Returns:
            멀티타겟 DataFrame (각 컬럼이 하나의 미래 기간)
        """
        try:
            targets = {}

            for days in forward_days_list:
                target = self._create_target_variable(stock_data, forward_days=days)
                targets[f"target_{days}d"] = target

            return pd.DataFrame(targets, index=stock_data.index)

        except Exception as e:
            logger.error(f"멀티타겟 생성 실패: {e}")
            return pd.DataFrame()

    def run_analysis(self) -> Dict:
        """
        전체 분석 실행

        Returns:
            분석 결과 종합
        """
        if not self.is_initialized:
            logger.error("모델이 초기화되지 않음")
            return {"status": "error", "message": "모델 초기화 필요"}

        try:
            logger.info("전체 분석 실행 시작...")

            # 1. 현재 매크로 환경 분석
            macro_data = self._collect_macro_data()
            market_regime = self.regime_classifier.predict_regime(macro_data)

            # 체제 히스토리 업데이트
            self.regime_history.append(market_regime)
            if len(self.regime_history) > 100:  # 최근 100개만 유지
                self.regime_history = self.regime_history[-100:]

            # 2. 개별 종목 분석 (매크로 데이터 포함)
            individual_results = []
            symbols = self.config["data"]["symbols"]

            for symbol in symbols:
                try:
                    result = self._analyze_individual_stock(
                        symbol, market_regime, macro_data
                    )
                    if result:
                        individual_results.append(result)
                except Exception as e:
                    logger.error(f"{symbol} 분석 실패: {e}")
                    continue

            # 3. 포트폴리오 레벨 집계
            portfolio_signals = self.portfolio_aggregator.aggregate_portfolio_signals(
                individual_results, market_regime
            )

            # 4. 예측 결과 표 생성
            prediction_table = self._create_prediction_table(
                individual_results, market_regime
            )

            # 5. 결과 종합
            final_result = {
                "prediction_table": prediction_table,
                "timestamp": datetime.now().isoformat(),
                "market_regime": market_regime,
                "individual_signals": individual_results,
                "portfolio_summary": portfolio_signals,
                "analysis_metadata": {
                    "symbols_analyzed": len(individual_results),
                    "model_version": self.model_version,
                    "config_version": self.config.get("version", "1.0.0"),
                },
            }

            # 5. 결과 저장
            self._save_results(final_result)

            # 6. 멀티타겟 예측 결과 출력
            self._print_multi_target_predictions(individual_results)

            logger.info("전체 분석 완료")

            # 7. 포트폴리오 최적화 및 백테스팅 (옵션)
            enhanced_result = self._enhance_with_portfolio_analysis(
                final_result, individual_results, market_regime
            )

            # individual_results를 enhanced_result에 추가 (상세 표 출력용)
            enhanced_result["individual_results"] = individual_results

            return enhanced_result

        except Exception as e:
            logger.error(f"전체 분석 실패: {e}")
            return {"status": "error", "message": str(e)}

    def _enhance_with_portfolio_analysis(
        self, basic_result: Dict, individual_results: List[Dict], market_regime: Dict
    ) -> Dict:
        """기본 분석 결과에 포트폴리오 최적화 및 백테스팅 추가"""
        try:
            logger.info("🎯 포트폴리오 고도화 분석 시작")

            # 포트폴리오 최적화 실행 여부 확인
            portfolio_config = self.config.get("portfolio", {})
            optimization_config = portfolio_config.get("optimization", {})
            enable_portfolio = optimization_config.get("enable_optimization", False)

            logger.info(
                f"🔍 포트폴리오 설정 확인: optimization 섹션 존재={bool(optimization_config)}"
            )
            logger.info(f"🔍 포트폴리오 최적화 활성화: {enable_portfolio}")

            if not enable_portfolio:
                logger.warning("⚠️ 포트폴리오 최적화 비활성화 - 기본 결과만 반환")
                return basic_result

            # portfolio_manager가 없으면 동적으로 초기화 (analysis_mode 대응)
            if not hasattr(self, "portfolio_manager") or self.portfolio_manager is None:
                logger.info("🔧 portfolio_manager 동적 초기화 중...")
                try:
                    self.portfolio_manager = NeuralPortfolioManager(self.config)
                    logger.info("✅ portfolio_manager 동적 초기화 완료")
                except Exception as e:
                    logger.error(f"❌ portfolio_manager 초기화 실패: {e}")
                    logger.warning(
                        "⚠️ 포트폴리오 기능을 사용할 수 없습니다 - 기본 결과만 반환"
                    )
                    return basic_result

            # 1. 신경망 기반 포트폴리오 최적화
            logger.info("📊 신경망 기반 포트폴리오 최적화 실행")

            # 과거 데이터 로드 (캐시된 데이터 활용)
            historical_data = self._load_historical_data()

            # 포트폴리오 최적화
            portfolio_result = (
                self.portfolio_manager.optimize_portfolio_with_constraints(
                    individual_results, historical_data
                )
            )

            if portfolio_result:
                logger.info("✅ 포트폴리오 최적화 완료")
                basic_result["portfolio_optimization"] = portfolio_result

                # 2. 백테스팅 실행 (옵션)
                backtest_config = self.config.get("backtesting", {})
                enable_backtest = backtest_config.get("enable", False)
                if enable_backtest and historical_data:
                    logger.info("📊 포트폴리오 백테스팅 실행")

                    # 과거 신호 생성 (시뮬레이션)
                    signal_history = self._simulate_historical_signals(
                        historical_data, market_regime
                    )

                    # 백테스팅 실행
                    backtest_result = self.portfolio_manager.backtest_neural_signals(
                        historical_data, signal_history, portfolio_result["weights"]
                    )

                    if backtest_result:
                        logger.info("✅ 백테스팅 완료")
                        basic_result["backtest_analysis"] = backtest_result

                        # 통합 리포트 생성기 사용
                        try:
                            from .unified_reporter import UnifiedReporter

                            unified_reporter = UnifiedReporter()

                            # 시장 체제 정보 추출
                            market_regime = {
                                "current_regime": analysis_results.get(
                                    "market_regime", {}
                                ).get("current_regime", "UNKNOWN"),
                                "confidence": analysis_results.get(
                                    "market_regime", {}
                                ).get("confidence", 0),
                            }

                            # 통합 리포트 생성
                            comprehensive_report = (
                                unified_reporter.generate_comprehensive_report(
                                    analysis_results=analysis_results,
                                    backtest_results=backtest_result,
                                    market_regime=market_regime,
                                )
                            )

                            basic_result["comprehensive_report"] = comprehensive_report

                            # 콘솔 출력
                            print("\n" + comprehensive_report)

                        except ImportError:
                            # 기존 방식으로 폴백
                            performance_report = self.portfolio_manager.generate_enhanced_portfolio_report(
                                portfolio_result, backtest_result, historical_data
                            )
                            basic_result["portfolio_report"] = performance_report

                            print("\n" + "=" * 80)
                            print("🎯 포트폴리오 고도화 분석 결과")
                            print("=" * 80)
                            print(performance_report)

            return basic_result

        except Exception as e:
            logger.error(f"포트폴리오 고도화 분석 실패: {e}")
            # 기본 결과 반환 (포트폴리오 분석 실패해도 기본 분석은 유지)
            return basic_result

    def _load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """과거 데이터 로드 (캐시 활용)"""
        try:
            data_config = self.config.get("data", {})
            symbols = data_config.get("symbols", [])
            historical_data = {}

            for symbol in symbols:
                # 캐시된 데이터 파일 찾기
                cache_files = glob.glob(f"data/trader/{symbol}_*.csv")
                if cache_files:
                    # 가장 최신 파일 사용
                    latest_file = max(cache_files, key=os.path.getmtime)
                    df = pd.read_csv(latest_file)

                    # datetime 인덱스 설정
                    if "datetime" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        df.set_index("datetime", inplace=True)

                    historical_data[symbol] = df
                    logger.info(f"📊 {symbol} 과거 데이터 로드: {len(df)}일")

            return historical_data

        except Exception as e:
            logger.error(f"과거 데이터 로드 실패: {e}")
            return {}

    def _simulate_historical_signals(
        self, historical_data: Dict[str, pd.DataFrame], current_regime: Dict
    ) -> List[Dict]:
        """과거 신호 시뮬레이션 (간단한 버전)"""
        try:
            signal_history = []

            # 최근 30일간의 가상 신호 생성
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            for i in range(30):
                signal_date = start_date + timedelta(days=i)

                for symbol in historical_data.keys():
                    try:
                        # 간단한 모멘텀 기반 신호 생성
                        data = historical_data[symbol]

                        # timezone 문제를 완전히 피하기 위해 날짜 문자열로 비교
                        signal_date_str = signal_date.strftime("%Y-%m-%d")

                        # 데이터 인덱스를 날짜 문자열로 변환하여 비교 (pandas Index 호환)
                        data_copy = data.copy()
                        try:
                            # timezone-aware 데이터를 UTC로 변환 후 날짜 문자열 생성
                            if (
                                hasattr(data_copy.index, "tz")
                                and data_copy.index.tz is not None
                            ):
                                data_copy["date_str"] = data_copy.index.tz_convert(
                                    "UTC"
                                ).strftime("%Y-%m-%d")
                            else:
                                data_copy["date_str"] = pd.to_datetime(
                                    data_copy.index, utc=True
                                ).strftime("%Y-%m-%d")
                        except Exception:
                            # fallback: 인덱스를 직접 문자열로 변환
                            data_copy["date_str"] = [
                                str(d)[:10] for d in data_copy.index
                            ]

                        # 날짜 필터링 (문자열 비교로 timezone 문제 완전 회피)
                        price_data = data_copy[data_copy["date_str"] <= signal_date_str]

                        if len(price_data) >= 5:
                            # 5일 모멘텀 계산
                            recent_return = (
                                price_data.iloc[-1]["close"]
                                - price_data.iloc[-5]["close"]
                            ) / price_data.iloc[-5]["close"]

                            if recent_return > 0.02:
                                action = "BUY"
                            elif recent_return < -0.02:
                                action = "SELL"
                            else:
                                action = "HOLD"

                            signal_history.append(
                                {
                                    "symbol": symbol,
                                    "timestamp": signal_date.isoformat(),
                                    "trading_signal": {
                                        "action": action,
                                        "score": abs(recent_return),
                                        "confidence": min(1.0, abs(recent_return) * 10),
                                    },
                                }
                            )
                    except Exception as symbol_error:
                        logger.warning(
                            f"⚠️ {symbol} 신호 생성 중 오류 (건너뜀): {symbol_error}"
                        )
                        continue

            logger.info(f"📊 과거 신호 시뮬레이션 완료: {len(signal_history)}개 신호")
            return signal_history

        except Exception as e:
            logger.error(f"과거 신호 시뮬레이션 실패: {e}")
            return []

    def _print_multi_target_predictions(self, individual_results: List[Dict]):
        """
        멀티타겟 예측 결과를 종목별로 출력
        """
        try:
            print("\n" + "=" * 80)
            print("🎯 멀티타겟 신경망 예측 결과")
            print("=" * 80)

            # config에서 예측 기간 읽기
            forward_days_config = self.config.get("neural_network", {}).get(
                "target_forward_days", [22, 66]
            )
            if isinstance(forward_days_config, int):
                forward_days_list = [forward_days_config]
            else:
                forward_days_list = forward_days_config

            print(f"📊 예측 기간: {forward_days_list}일 후")
            print(f"📈 분석 종목: {len(individual_results)}개")
            print("-" * 80)

            for result in individual_results:
                symbol = result.get("symbol", "UNKNOWN")
                neural_prediction = result.get("neural_prediction", {})

                print(f"🔍 {symbol}:")

                if neural_prediction is None:
                    # 예측 실패
                    print(
                        f"   ❌ 예측 실패: 신경망 모델에서 예측값을 생성하지 못했습니다"
                    )
                elif isinstance(neural_prediction, dict):
                    # 멀티타겟 예측 결과
                    for target_name, prediction in neural_prediction.items():
                        if prediction is not None:
                            days = target_name.replace("target_", "").replace("d", "")
                            percentage = prediction * 100
                            direction = (
                                "📈"
                                if prediction > 0
                                else "📉" if prediction < 0 else "➡️"
                            )
                            print(
                                f"   {direction} {days}일 후: {percentage:+.2f}% ({prediction:.4f})"
                            )
                        else:
                            print(f"   ❌ {target_name}: 예측 실패")
                else:
                    # 단일 타겟 예측 결과
                    if neural_prediction is not None:
                        percentage = neural_prediction * 100
                        direction = (
                            "📈"
                            if neural_prediction > 0
                            else "📉" if neural_prediction < 0 else "➡️"
                        )
                        print(
                            f"   {direction} 예측: {percentage:+.2f}% ({neural_prediction:.4f})"
                        )
                    else:
                        print(f"   ❌ 예측 실패: 값이 None입니다")

                # 추가 정보 (신뢰도, 시장 체제 등)
                confidence = result.get("confidence", 0.0)
                regime = result.get("market_regime", {}).get(
                    "current_regime", "UNKNOWN"
                )
                print(f"   🎯 신뢰도: {confidence:.2f}")
                print(f"   🎭 시장 체제: {regime}")
                print()

            print("=" * 80)
            print("💡 해석:")
            print("   - 양수: 상승 예측, 음수: 하락 예측")
            print("   - 값의 크기: 예측 강도 (-1 ~ +1)")
            print("   - 신뢰도: 모델의 예측 확신도")
            print("=" * 80)

        except Exception as e:
            logger.error(f"예측 결과 출력 실패: {e}")
            print("❌ 예측 결과 출력 중 오류 발생")

    def _analyze_individual_stock(
        self,
        symbol: str,
        market_regime: Dict,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        """
        개별 종목 분석

        Args:
            symbol: 종목 코드
            market_regime: 시장 체제 정보
            macro_data: 전체 매크로 데이터 (선택사항)

        Returns:
            개별 종목 분석 결과
        """
        try:
            logger.info(f"{symbol} 개별 분석 시작...")

            # 1. 종목 데이터 수집 (캐시 옵션 고려)
            if self.use_cached_data:
                stock_data = self._load_cached_stock_data(symbol)
                if stock_data is None:
                    logger.warning(f"{symbol}: 캐시된 데이터 없음, API에서 수집")
                    stock_data = self._get_stock_data_from_api(symbol)
            else:
                stock_data = self._get_stock_data_from_api(symbol)

            if stock_data is None or len(stock_data) < 50:
                logger.warning(f"{symbol}: 데이터 부족")
                return None

            # 컬럼명 소문자로 통일
            stock_data.columns = [col.lower() for col in stock_data.columns]

            # 2. 피처 생성 (매크로 데이터 포함)
            features = self.neural_predictor.create_features(
                stock_data, symbol, market_regime, macro_data
            )

            # 3. 신경망 예측
            neural_prediction = self.neural_predictor.predict(features, symbol)

            # 4. 투자 점수 생성
            investment_score = self.score_generator.generate_investment_score(
                neural_prediction, stock_data, symbol, market_regime
            )

            # 5. 매매 신호 생성
            trading_signal = self.signal_generator.generate_signal(investment_score)

            # 6. 결과에 멀티타겟 예측과 추가 정보 포함
            result = {
                "symbol": symbol,
                "neural_prediction": neural_prediction,  # 멀티타겟 예측 결과
                "investment_score": investment_score,
                "trading_signal": trading_signal,
                "market_regime": market_regime,
                "confidence": investment_score.get("confidence", 0.0),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"{symbol} 분석 완료 - 점수: {investment_score['final_score']:.4f}, "
                f"신호: {trading_signal['action']}"
            )

            return result

        except Exception as e:
            logger.error(f"{symbol} 개별 분석 실패: {e}")
            return None

    def _create_prediction_table(
        self, individual_results: List[Dict], market_regime: Dict
    ) -> Dict:
        """
        예측 결과 표 생성

        Args:
            individual_results: 개별 종목 분석 결과
            market_regime: 시장 체제 정보

        Returns:
            예측 결과 표 데이터
        """
        try:
            # Universal 모델 예측 (통합 예측)
            universal_predictions = {}

            # 개별 종목 예측 수집
            individual_predictions = {}
            prediction_summary = []

            for result in individual_results:
                symbol = result["symbol"]
                neural_pred = result.get("neural_prediction")

                if neural_pred is not None:
                    individual_predictions[symbol] = neural_pred

                    # 예측 요약 정리
                    if isinstance(neural_pred, dict):
                        # 멀티타겟 예측
                        summary = {
                            "symbol": symbol,
                            "target_22d": neural_pred.get("target_22d", 0.0),
                            "sigma_22d": neural_pred.get("sigma_22d", 0.0),
                            "target_66d": neural_pred.get("target_66d", 0.0),
                            "sigma_66d": neural_pred.get("sigma_66d", 0.0),
                            "investment_score": result["investment_score"][
                                "final_score"
                            ],
                            "confidence": result["investment_score"]["confidence"],
                            "action": result["trading_signal"]["action"],
                        }
                    else:
                        # 단일 예측
                        summary = {
                            "symbol": symbol,
                            "prediction": neural_pred,
                            "investment_score": result["investment_score"][
                                "final_score"
                            ],
                            "confidence": result["investment_score"]["confidence"],
                            "action": result["trading_signal"]["action"],
                        }

                    prediction_summary.append(summary)

            # Universal 모델 예측 시도 (모든 종목 데이터 통합)
            try:
                # 통합 예측은 개별 예측과 구조가 다를 수 있으므로 생략하거나 별도 처리
                universal_predictions = {
                    "note": "Universal 모델은 개별 예측과 차원이 달라 표시하지 않음"
                }
            except Exception as e:
                logger.warning(f"Universal 모델 예측 실패: {e}")
                universal_predictions = {"error": "Universal 모델 예측 불가"}

            return {
                "timestamp": datetime.now().isoformat(),
                "market_regime": {
                    "regime": market_regime.get("regime", "UNKNOWN"),
                    "confidence": market_regime.get("confidence", 0.0),
                },
                "universal_predictions": universal_predictions,
                "individual_predictions": individual_predictions,
                "summary_table": prediction_summary,
                "table_headers": {
                    "multitarget": [
                        "Symbol",
                        "22일 수익률",
                        "22일 변동성",
                        "66일 수익률",
                        "66일 변동성",
                        "투자점수",
                        "신뢰도",
                        "액션",
                    ],
                    "single": ["Symbol", "예측값", "투자점수", "신뢰도", "액션"],
                },
            }

        except Exception as e:
            logger.error(f"예측 표 생성 실패: {e}")
            return {"error": str(e)}

    def _print_prediction_table(self, prediction_table: Dict) -> None:
        """
        예측 결과 표를 콘솔에 출력

        Args:
            prediction_table: 예측 표 데이터
        """
        try:
            print("\n" + "=" * 80)
            print("🎯 멀티타겟 신경망 예측 결과")
            print("=" * 80)

            # 시장 체제 정보
            market_info = prediction_table.get("market_regime", {})
            print(
                f"📊 시장 체제: {market_info.get('regime', 'N/A')} (신뢰도: {market_info.get('confidence', 0.0):.1%})"
            )

            # 예측 기간 정보
            summary_table = prediction_table.get("summary_table", [])
            if summary_table:
                # 멀티타겟인지 확인
                first_item = summary_table[0]
                is_multitarget = "target_22d" in first_item

                if is_multitarget:
                    print(f"📈 예측 기간: [22, 66]일 후")
                    print(f"📈 분석 종목: {len(summary_table)}개")
                    print("-" * 80)

                    # 헤더 출력
                    print(
                        f"{'종목':<8} {'22일수익률':<10} {'22일변동성':<10} {'66일수익률':<10} {'66일변동성':<10} {'투자점수':<8} {'신뢰도':<8} {'액션':<12}"
                    )
                    print("-" * 80)

                    # 데이터 출력
                    for item in summary_table:
                        print(
                            f"{item['symbol']:<8} "
                            f"{item['target_22d']:>9.1%} "
                            f"{item['sigma_22d']:>9.1%} "
                            f"{item['target_66d']:>9.1%} "
                            f"{item['sigma_66d']:>9.1%} "
                            f"{item['investment_score']:>7.3f} "
                            f"{item['confidence']:>7.1%} "
                            f"{item['action']:<12}"
                        )
                else:
                    print(f"📈 분석 종목: {len(summary_table)}개")
                    print("-" * 60)

                    # 단일 예측 헤더
                    print(
                        f"{'종목':<8} {'예측값':<10} {'투자점수':<8} {'신뢰도':<8} {'액션':<12}"
                    )
                    print("-" * 60)

                    # 데이터 출력
                    for item in summary_table:
                        print(
                            f"{item['symbol']:<8} "
                            f"{item['prediction']:>9.3f} "
                            f"{item['investment_score']:>7.3f} "
                            f"{item['confidence']:>7.1%} "
                            f"{item['action']:<12}"
                        )

            print("=" * 80)

            # Universal 모델 정보
            universal_info = prediction_table.get("universal_predictions", {})
            if "note" in universal_info:
                print(f"📝 Universal 모델: {universal_info['note']}")
            elif "error" in universal_info:
                print(f"⚠️ Universal 모델: {universal_info['error']}")

        except Exception as e:
            print(f"❌ 예측 결과 출력 중 오류 발생: {e}")

    def get_recommendations(self, symbol: Optional[str] = None) -> Dict:
        """
        투자 권고사항 조회

        Args:
            symbol: 특정 종목 (None이면 전체)

        Returns:
            권고사항
        """
        try:
            if not self.is_initialized:
                return {"status": "error", "message": "모델 초기화 필요"}

            if symbol:
                # 특정 종목만 분석
                return self._analyze_single_stock(symbol)
            else:
                # 전체 포트폴리오 분석
                analysis_result = self.run_analysis()
                return {
                    "portfolio_recommendations": analysis_result["portfolio_summary"],
                    "top_picks": analysis_result["portfolio_summary"][
                        "top_opportunities"
                    ],
                    "market_context": analysis_result["market_regime"],
                    "immediate_actions": analysis_result["portfolio_summary"][
                        "immediate_actions"
                    ],
                }

        except Exception as e:
            logger.error(f"권고사항 조회 실패: {e}")
            return {"status": "error", "message": str(e)}

    def _analyze_single_stock(self, symbol: str) -> Dict:
        """특정 종목만 분석"""
        try:
            logger.info(f"🔍 {symbol} 단일 종목 분석 시작...")

            # 1. 매크로 데이터 수집 (캐시 옵션 고려)
            macro_data = self._collect_macro_data()
            if macro_data is None or len(macro_data) < 100:
                return {"status": "error", "message": "매크로 데이터 부족"}

            # 2. 시장 체제 예측
            market_regime = self.regime_classifier.predict_regime(macro_data)

            # 3. 개별 종목 분석
            result = self._analyze_individual_stock(symbol, market_regime)
            if result is None:
                return {"status": "error", "message": f"{symbol} 분석 실패"}

            return {
                "symbol": symbol,
                "recommendation": result,
                "market_context": market_regime,
            }

        except Exception as e:
            logger.error(f"{symbol} 단일 종목 분석 실패: {e}")
            return {"status": "error", "message": str(e)}

    def _load_existing_models(self) -> bool:
        """기존 모델 로드"""
        try:
            model_dir = self.config.get("model_persistence", {}).get(
                "model_directory", "models/trader"
            )

            # 분석 모드에서는 컴포넌트를 실제로 초기화
            if self.analysis_mode:
                self.regime_classifier = MarketRegimeHMM(self.config)
                self.neural_predictor = StockPredictionNetwork(self.config)
                self.score_generator = InvestmentScoreGenerator(self.config)
                self.signal_generator = TradingSignalGenerator(self.config)
                self.portfolio_aggregator = PortfolioSignalAggregator(self.config)

                # 데이터 소스도 초기화 (매크로 데이터 수집 등에 필요)
                self.data_loader = YahooFinanceDataCollector()
                self.macro_collector = GlobalMacroDataCollector()
                self.macro_analyzer = MacroSectorAnalyzer()

            # HMM 모델 로드
            hmm_path = os.path.join(model_dir, "hmm_regime_model.pkl")
            if os.path.exists(hmm_path):
                if not self.regime_classifier.load_model(hmm_path):
                    return False
            else:
                return False

            # 신경망 모델 로드
            neural_path = os.path.join(model_dir, "neural_predictor")
            if os.path.exists(f"{neural_path}_meta.pkl"):
                if not self.neural_predictor.load_model(neural_path):
                    return False
            else:
                logger.warning(
                    f"신경망 모델 파일을 찾을 수 없습니다: {neural_path}_meta.pkl"
                )
                return False

            logger.info("기존 모델 로드 성공")
            return True

        except Exception as e:
            logger.error(f"기존 모델 로드 실패: {e}")
            return False

    def _save_models(self):
        """모델 저장"""
        try:
            model_dir = self.config.get("model_persistence", {}).get(
                "model_directory", "models/trader"
            )
            os.makedirs(model_dir, exist_ok=True)

            # HMM 모델 저장
            hmm_path = os.path.join(model_dir, "hmm_regime_model.pkl")
            self.regime_classifier.save_model(hmm_path)

            # 신경망 모델 저장
            neural_path = os.path.join(model_dir, "neural_predictor")
            self.neural_predictor.save_model(neural_path)

            logger.info("모델 저장 완료")

        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")

    def _save_results(self, results: Dict):
        """결과 저장"""
        try:
            output_dir = self.config.get("output", {}).get(
                "results_folder", "results/trader"
            )
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trader_results_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"결과 저장 완료: {filepath}")

        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")

    def analyze_macro_assets(self) -> Dict:
        """매크로 자산 분석 (주식/채권/금/원자재 포지션 비율)"""
        try:
            logger.info("🔍 매크로 자산 분석 시작...")

            # 1. 매크로 데이터 수집
            macro_data = self._collect_macro_data()
            if macro_data is None or len(macro_data) < 100:
                return {"status": "error", "message": "매크로 데이터 부족"}

            # 2. 시장 체제 예측
            market_regime = self.regime_classifier.predict_regime(macro_data)

            # 3. 매크로 자산 심볼 정의
            macro_symbols = {
                "주식": ["SPY", "QQQ", "IWM"],
                "채권": ["TLT", "TIP", "VTIP"],
                "금": ["GLD", "GTX"],
                "달러": ["UUP"],
                "변동성": ["^VIX"],
            }

            # 4. 개별 매크로 자산 분석
            macro_assets = []
            asset_scores = {}

            for category, symbols in macro_symbols.items():
                for symbol in symbols:
                    try:
                        # 자산 데이터 수집
                        if self.use_cached_data:
                            asset_data = self._load_cached_macro_asset_data(symbol)
                            if asset_data is None:
                                asset_data = self._get_macro_asset_data_from_api(symbol)
                        else:
                            asset_data = self._get_macro_asset_data_from_api(symbol)

                        if asset_data is not None and len(asset_data) > 50:
                            # 컬럼명 소문자로 통일
                            asset_data.columns = [
                                col.lower() for col in asset_data.columns
                            ]

                            # 피처 생성
                            features = self.neural_predictor.create_features(
                                asset_data, symbol, market_regime
                            )

                            # 신경망 예측
                            neural_prediction = self.neural_predictor.predict(
                                features, symbol
                            )

                            # 투자 점수 생성
                            investment_score = (
                                self.score_generator.generate_investment_score(
                                    neural_prediction, asset_data, symbol, market_regime
                                )
                            )

                            # 매매 신호 생성
                            trading_signal = self.signal_generator.generate_signal(
                                investment_score
                            )

                            asset_result = {
                                "symbol": symbol,
                                "category": category,
                                "action": trading_signal["action"],
                                "score": investment_score["final_score"],
                                "confidence": investment_score["confidence"],
                                "strength": trading_signal.get("action_strength", 0.5),
                                "priority": trading_signal["execution_priority"],
                            }

                            macro_assets.append(asset_result)
                            asset_scores[symbol] = investment_score["final_score"]

                            logger.info(
                                f"    ✅ {symbol} ({category}) 분석 완료: {trading_signal['action']}"
                            )

                    except Exception as e:
                        logger.error(f"    ❌ {symbol} 분석 실패: {e}")
                        continue

            # 5. 자산별 포지션 비율 계산
            asset_allocation = self._calculate_macro_asset_allocation(
                macro_assets, market_regime
            )

            # 6. 전략 요약 생성
            strategy_summary = self._generate_macro_strategy_summary(
                macro_assets, asset_allocation, market_regime
            )

            return {
                "market_regime": market_regime,
                "asset_allocation": asset_allocation,
                "macro_assets": macro_assets,
                "strategy_summary": strategy_summary,
            }

        except Exception as e:
            logger.error(f"매크로 자산 분석 실패: {e}")
            return {"status": "error", "message": str(e)}

    def _get_macro_asset_data_from_api(self, symbol: str) -> Optional[pd.DataFrame]:
        """API에서 매크로 자산 데이터 수집"""
        try:
            data_config = self.config.get("data", {})

            # 매크로 자산 데이터 수집
            asset_data = self.data_loader.get_candle_data(
                symbol=symbol,
                interval=data_config.get("interval", "1d"),
                start_date=data_config.get("start_date"),
                end_date=data_config.get("end_date"),
                days_back=data_config.get("lookback_days", 360),
            )

            # 데이터 수집 성공 시 캐시에 저장
            if asset_data is not None and not asset_data.empty and len(asset_data) > 50:
                self._save_macro_asset_data_to_cache(symbol, asset_data)

            return asset_data

        except Exception as e:
            logger.error(f"API에서 {symbol} 매크로 자산 데이터 수집 실패: {e}")
            return None

    def _load_cached_macro_asset_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """캐시된 매크로 자산 데이터 로드"""
        try:
            # data/macro/ 폴더에서 캐시된 데이터 찾기
            macro_data_dir = "data/macro"
            if not os.path.exists(macro_data_dir):
                return None

            # 특수 문자 처리 (^VIX -> VIX)
            clean_symbol = symbol.replace("^", "")

            # 여러 패턴으로 파일 검색
            patterns = [
                f"{symbol}_*.csv",  # 원본 심볼
                f"{clean_symbol}_*.csv",  # 특수 문자 제거된 심볼
                f"{symbol.lower()}_*.csv",  # 소문자
                f"{clean_symbol.lower()}_*.csv",  # 특수 문자 제거 + 소문자
            ]

            matching_files = []
            for pattern in patterns:
                files = glob.glob(os.path.join(macro_data_dir, pattern))
                matching_files.extend(files)

            # 중복 제거
            matching_files = list(set(matching_files))

            if not matching_files:
                logger.info(
                    f"    📁 {symbol} 매크로 자산 캐시 파일을 찾을 수 없습니다."
                )
                return None

            # 가장 최근 파일 선택
            latest_file = max(matching_files, key=os.path.getctime)

            logger.info(
                f"    📁 {symbol} 매크로 자산 캐시 파일 로드: {os.path.basename(latest_file)}"
            )

            # CSV 파일 로드
            asset_data = pd.read_csv(latest_file, index_col=0, parse_dates=True)

            if asset_data.empty or len(asset_data) < 50:
                logger.warning(
                    f"    ⚠️ {symbol} 매크로 자산 캐시 데이터 부족: {len(asset_data)}개 행"
                )
                return None

            return asset_data

        except Exception as e:
            logger.error(f"캐시된 {symbol} 매크로 자산 데이터 로드 실패: {e}")
            return None

    def _save_macro_asset_data_to_cache(
        self, symbol: str, asset_data: pd.DataFrame
    ) -> bool:
        """매크로 자산 데이터를 캐시에 저장"""
        try:
            # data/macro/ 폴더 생성
            macro_data_dir = "data/macro"
            os.makedirs(macro_data_dir, exist_ok=True)

            # 파일명 생성 (타임스탬프 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_macro_{timestamp}.csv"
            filepath = os.path.join(macro_data_dir, filename)

            # CSV로 저장
            asset_data.to_csv(filepath)

            logger.info(f"    💾 {symbol} 매크로 자산 데이터 캐시 저장: {filename}")
            return True

        except Exception as e:
            logger.error(f"{symbol} 매크로 자산 데이터 캐시 저장 실패: {e}")
            return False

    def _calculate_macro_asset_allocation(
        self, macro_assets: List[Dict], market_regime: Dict
    ) -> Dict:
        """매크로 자산별 포지션 비율 계산"""
        try:
            # 자산 카테고리별 점수 집계
            category_scores = {
                "주식": [],
                "채권": [],
                "금": [],
                "달러": [],
                "변동성": [],
            }

            for asset in macro_assets:
                category = asset["category"]
                score = asset["score"]
                if category in category_scores:
                    category_scores[category].append(score)

            # 카테고리별 평균 점수 계산
            category_avg_scores = {}
            for category, scores in category_scores.items():
                if scores:
                    category_avg_scores[category] = sum(scores) / len(scores)
                else:
                    category_avg_scores[category] = 0.0

            # 시장 체제에 따른 가중치 조정
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_weights = {
                "BULLISH": {
                    "주식": 1.2,
                    "채권": 0.8,
                    "금": 0.6,
                    "달러": 0.7,
                    "변동성": 0.5,
                },
                "BEARISH": {
                    "주식": 0.6,
                    "채권": 1.3,
                    "금": 1.1,
                    "달러": 1.2,
                    "변동성": 1.4,
                },
                "SIDEWAYS": {
                    "주식": 1.0,
                    "채권": 1.0,
                    "금": 1.0,
                    "달러": 1.0,
                    "변동성": 1.0,
                },
                "VOLATILE": {
                    "주식": 0.8,
                    "채권": 1.1,
                    "금": 1.2,
                    "달러": 1.1,
                    "변동성": 1.3,
                },
            }

            weights = regime_weights.get(regime, regime_weights["SIDEWAYS"])

            # 최종 포지션 비율 계산
            total_weight = 0
            asset_allocation = {}

            for category, avg_score in category_avg_scores.items():
                # 점수를 0~1 범위로 정규화하고 가중치 적용
                normalized_score = max(
                    0, min(1, (avg_score + 1) / 2)
                )  # -1~1을 0~1로 변환
                weighted_score = normalized_score * weights.get(category, 1.0)
                total_weight += weighted_score

                # 액션 결정
                if avg_score > 0.3:
                    action = "BUY"
                elif avg_score < -0.3:
                    action = "SELL"
                else:
                    action = "HOLD"

                asset_allocation[category] = {
                    "weight": weighted_score,
                    "action": action,
                    "score": avg_score,
                }

            # 비율 정규화 (총합이 100%가 되도록)
            if total_weight > 0:
                for category in asset_allocation:
                    asset_allocation[category]["weight"] = (
                        asset_allocation[category]["weight"] / total_weight
                    )

            return asset_allocation

        except Exception as e:
            logger.error(f"매크로 자산 배분 계산 실패: {e}")
            return {}

    def _generate_macro_strategy_summary(
        self, macro_assets: List[Dict], asset_allocation: Dict, market_regime: Dict
    ) -> Dict:
        """매크로 전략 요약 생성"""
        try:
            # 전체 전략 결정
            buy_count = sum(1 for asset in macro_assets if asset["action"] == "BUY")
            sell_count = sum(1 for asset in macro_assets if asset["action"] == "SELL")
            hold_count = sum(1 for asset in macro_assets if asset["action"] == "HOLD")

            if buy_count > sell_count and buy_count > hold_count:
                overall_strategy = "공격적 매수"
                risk_level = "높음"
                recommended_leverage = 1.2
            elif sell_count > buy_count and sell_count > hold_count:
                overall_strategy = "방어적 포지션"
                risk_level = "낮음"
                recommended_leverage = 0.8
            else:
                overall_strategy = "중립적 관망"
                risk_level = "보통"
                recommended_leverage = 1.0

            # 시장 체제에 따른 조정
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "VOLATILE":
                risk_level = "높음"
                recommended_leverage *= 0.8
            elif regime == "BEARISH":
                risk_level = "높음"
                recommended_leverage *= 0.7

            return {
                "overall_strategy": overall_strategy,
                "risk_level": risk_level,
                "recommended_leverage": recommended_leverage,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
            }

        except Exception as e:
            logger.error(f"매크로 전략 요약 생성 실패: {e}")
            return {
                "overall_strategy": "N/A",
                "risk_level": "N/A",
                "recommended_leverage": 1.0,
            }

    def load_optimized_thresholds(self) -> bool:
        """저장된 최적화 결과를 로드하여 threshold 설정 업데이트"""
        try:
            results_dir = Path("results/trader")

            # 새로운 형식의 optimized_thresholds.json 파일 먼저 확인
            new_threshold_file = results_dir / "optimized_thresholds.json"
            if new_threshold_file.exists():
                with open(new_threshold_file, "r", encoding="utf-8") as f:
                    threshold_data = json.load(f)

                thresholds = threshold_data.get("thresholds", {})
                timestamp = threshold_data.get("timestamp", "unknown")

                if thresholds:
                    if hasattr(self, "signal_generator") and self.signal_generator:
                        self.signal_generator.update_thresholds(thresholds)
                        logger.info(f"✅ 새로운 형식 최적화된 threshold 로드 완료:")
                        logger.info(f"   - 생성 시간: {timestamp}")
                        logger.info(f"   - Threshold: {thresholds}")
                        return True
                    else:
                        logger.warning("신호 생성기가 초기화되지 않았습니다.")
                        return False

            # 기존 형식의 threshold_optimization_final_*.json 파일 확인
            optimization_files = list(
                results_dir.glob("threshold_optimization_final_*.json")
            )
            if not optimization_files:
                logger.info("저장된 최적화 결과를 찾을 수 없습니다. 기본 설정 사용.")
                return False

            # 가장 최근 파일 선택
            latest_file = max(optimization_files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, "r", encoding="utf-8") as f:
                optimization_result = json.load(f)

            # 최적 threshold 추출
            best_thresholds = optimization_result.get("best_thresholds", {})
            if not best_thresholds:
                logger.warning("최적화 결과에서 threshold를 찾을 수 없습니다.")
                return False

            # 신호 생성기의 threshold 업데이트
            if hasattr(self, "signal_generator") and self.signal_generator:
                self.signal_generator.update_thresholds(best_thresholds)
                logger.info(
                    f"✅ 기존 형식 최적화된 threshold 로드 완료: {best_thresholds}"
                )
                return True
            else:
                logger.warning("신호 생성기가 초기화되지 않았습니다.")
                return False

        except Exception as e:
            logger.error(f"최적화 결과 로드 실패: {e}")
            return False

    def _print_detailed_trading_signals_table(self, analysis_results: Dict) -> None:
        """
        상세한 매매신호 표를 콘솔에 출력

        Args:
            analysis_results: run_analysis()에서 반환된 전체 결과
        """
        try:
            print("\n" + "=" * 150)
            print("🚀 상세 매매신호 분석 결과")
            print("=" * 150)

            # 시장 체제 정보
            market_regime = analysis_results.get("market_regime", {})
            print(
                f"📊 시장 체제: {market_regime.get('regime', 'N/A')} (신뢰도: {market_regime.get('confidence', 0.0):.1%})"
            )

            # 포트폴리오 요약
            portfolio_summary = analysis_results.get("portfolio_summary", {})
            print(
                f"📈 포트폴리오 점수: {portfolio_summary.get('portfolio_score', 0.0):.4f}"
            )
            print(
                f"🎯 포트폴리오 액션: {portfolio_summary.get('portfolio_action', 'N/A')}"
            )

            # 개별 종목 신호들
            individual_results = analysis_results.get("individual_results", [])
            if not individual_results:
                print("❌ 개별 종목 신호 데이터가 없습니다.")
                return

            print(f"\n📋 개별 종목 상세 신호 ({len(individual_results)}개 종목)")
            print("-" * 150)

            # 헤더 출력
            header = (
                f"{'종목':<6} {'액션':<12} {'강도':<6} {'점수':<7} {'신뢰도':<7} {'포지션':<7} "
                f"{'우선순위':<8} {'진입타이밍':<10} {'손절선':<7} {'이익실현':<12} {'리스크':<8} {'체제':<8}"
            )
            print(header)
            print("-" * 150)

            # 실행 우선순위 순으로 정렬
            sorted_results = sorted(
                individual_results,
                key=lambda x: x.get("trading_signal", {}).get("execution_priority", 10),
            )

            # 데이터 출력
            for result in sorted_results:
                trading_signal = result.get("trading_signal", {})
                symbol = trading_signal.get("symbol", "N/A")
                action = trading_signal.get("action", "HOLD")
                action_strength = trading_signal.get("action_strength", 0.0)
                score = trading_signal.get("score", 0.0)
                confidence = trading_signal.get("confidence", 0.0)
                position_size = trading_signal.get("position_size", 0.0)
                execution_priority = trading_signal.get("execution_priority", 10)

                # 타이밍 정보
                timing = trading_signal.get("timing", {})
                entry_timing = timing.get("entry", {}).get("type", "WAIT")

                # 청산 정보
                exit_timing = timing.get("exit", {})
                stop_loss = exit_timing.get("stop_loss", 0.0)
                take_profit_levels = exit_timing.get("take_profit_levels", [])
                take_profit_str = (
                    f"{take_profit_levels[0]:.1%}" if take_profit_levels else "N/A"
                )

                # 리스크 정보
                risk_management = trading_signal.get("risk_management", {})
                risk_level = risk_management.get("risk_level", "MEDIUM")

                # 시장 체제
                regime = trading_signal.get("market_regime", "N/A")

                print(
                    f"{symbol:<6} {action:<12} {action_strength:<6.2f} {score:<7.3f} {confidence:<7.1%} {position_size:<7.1%} "
                    f"{execution_priority:<8} {entry_timing:<10} {stop_loss:<7.1%} {take_profit_str:<12} {risk_level:<8} {regime:<8}"
                )

            print("-" * 150)

            # 액션별 통계
            print(f"\n📊 액션별 통계:")
            signal_distribution = portfolio_summary.get("signal_distribution", {})
            for action, count in signal_distribution.items():
                if count > 0:
                    print(f"   {action}: {count}개")

            # 고우선순위 액션들
            high_priority_signals = [
                result
                for result in sorted_results
                if result.get("trading_signal", {}).get("execution_priority", 10) <= 3
            ]

            if high_priority_signals:
                print(f"\n⚡ 고우선순위 액션 ({len(high_priority_signals)}개):")
                for result in high_priority_signals[:5]:  # 상위 5개만
                    trading_signal = result.get("trading_signal", {})
                    symbol = trading_signal.get("symbol", "N/A")
                    action = trading_signal.get("action", "HOLD")
                    score = trading_signal.get("score", 0.0)
                    priority = trading_signal.get("execution_priority", 10)

                    # 권고사항
                    recommendations = trading_signal.get("recommendations", {})
                    primary_rec = recommendations.get("primary_recommendation", "N/A")

                    print(
                        f"   {symbol}: {action} (점수: {score:.3f}, 우선순위: {priority})"
                    )
                    print(f"      💡 {primary_rec}")

            # 리스크 경고
            portfolio_risk = analysis_results.get("portfolio_summary", {}).get(
                "risk_assessment", {}
            )
            overall_risk = portfolio_risk.get("overall_risk", "MEDIUM")

            if overall_risk == "HIGH":
                print(f"\n⚠️  포트폴리오 리스크 경고: {overall_risk}")
                risk_factors = portfolio_risk.get("risk_factors", [])
                for factor in risk_factors:
                    print(f"   • {factor}")

            print("\n" + "=" * 150)

        except Exception as e:
            logger.error(f"상세 매매신호 표 출력 실패: {e}")
            print(f"❌ 상세 표 출력 실패: {e}")

    def _print_individual_signal_details(self, trading_signal: Dict) -> None:
        """
        개별 종목의 상세한 매매신호 정보 출력

        Args:
            trading_signal: 개별 종목의 매매신호 딕셔너리
        """
        try:
            symbol = trading_signal.get("symbol", "N/A")
            print(f"\n📋 {symbol} 상세 신호 분석")
            print("-" * 80)

            # 기본 신호 정보
            action = trading_signal.get("action", "HOLD")
            action_strength = trading_signal.get("action_strength", 0.0)
            score = trading_signal.get("score", 0.0)
            confidence = trading_signal.get("confidence", 0.0)

            print(f"🎯 매매액션: {action} (강도: {action_strength:.2f})")
            print(f"📊 투자점수: {score:.4f} (신뢰도: {confidence:.1%})")
            print(f"💰 포지션크기: {trading_signal.get('position_size', 0.0):.1%}")
            print(f"⚡ 실행우선순위: {trading_signal.get('execution_priority', 10)}")

            # 타이밍 정보
            timing = trading_signal.get("timing", {})
            entry_timing = timing.get("entry", {})
            exit_timing = timing.get("exit", {})

            print(f"\n⏰ 진입 타이밍:")
            print(f"   타입: {entry_timing.get('type', 'WAIT')}")
            print(f"   긴급도: {entry_timing.get('urgency', 'NONE')}")

            # 분할 진입 계획
            entry_phases = entry_timing.get("entry_phases")
            if entry_phases:
                print(f"   분할 진입 계획:")
                for phase in entry_phases:
                    print(
                        f"     {phase['phase']}단계: {phase['ratio']:.1%} ({phase['timing']})"
                    )

            print(f"\n🚪 청산 타이밍:")
            print(f"   손절선: {exit_timing.get('stop_loss', 0.0):.1%}")

            take_profit_levels = exit_timing.get("take_profit_levels", [])
            if take_profit_levels:
                print(
                    f"   이익실현: {' → '.join([f'{tp:.1%}' for tp in take_profit_levels])}"
                )

            print(f"   트레일링스탑: {exit_timing.get('trailing_stop', 0.0):.1%}")
            print(f"   최대보유기간: {exit_timing.get('max_holding_days', 0)}일")

            # 리스크 관리
            risk_management = trading_signal.get("risk_management", {})
            print(f"\n⚠️  리스크 관리:")
            print(f"   리스크 레벨: {risk_management.get('risk_level', 'MEDIUM')}")

            warnings = risk_management.get("warnings", [])
            if warnings:
                print(f"   경고사항:")
                for warning in warnings:
                    print(f"     • {warning}")

            mitigation_strategies = risk_management.get("mitigation_strategies", [])
            if mitigation_strategies:
                print(f"   완화전략:")
                for strategy in mitigation_strategies:
                    print(f"     • {strategy}")

            # 권고사항
            recommendations = trading_signal.get("recommendations", {})
            primary_rec = recommendations.get("primary_recommendation", "")
            if primary_rec:
                print(f"\n💡 주요 권고사항:")
                print(f"   {primary_rec}")

            regime_advice = recommendations.get("regime_advice", "")
            if regime_advice:
                print(f"   시장체제 조언: {regime_advice}")

            timing_advice = recommendations.get("timing_advice", [])
            if timing_advice:
                print(f"   타이밍 조언:")
                for advice in timing_advice:
                    print(f"     • {advice}")

            cautions = recommendations.get("cautions", [])
            if cautions:
                print(f"   주의사항:")
                for caution in cautions:
                    print(f"     • {caution}")

            print("-" * 80)

        except Exception as e:
            logger.error(f"{symbol} 개별 신호 상세 출력 실패: {e}")
            print(f"❌ {symbol} 상세 정보 출력 실패: {e}")

    def _create_trading_signals_dataframe(self, analysis_results: Dict) -> pd.DataFrame:
        """
        매매신호를 pandas DataFrame으로 변환

        Args:
            analysis_results: run_analysis()에서 반환된 전체 결과

        Returns:
            매매신호 데이터프레임
        """
        try:
            individual_results = analysis_results.get("individual_results", [])
            if not individual_results:
                return pd.DataFrame()

            # 데이터 추출
            data = []
            for result in individual_results:
                trading_signal = result.get("trading_signal", {})

                # 기본 정보
                symbol = trading_signal.get("symbol", "N/A")
                action = trading_signal.get("action", "HOLD")
                action_strength = trading_signal.get("action_strength", 0.0)
                score = trading_signal.get("score", 0.0)
                confidence = trading_signal.get("confidence", 0.0)
                position_size = trading_signal.get("position_size", 0.0)
                execution_priority = trading_signal.get("execution_priority", 10)

                # 타이밍 정보
                timing = trading_signal.get("timing", {})
                entry_timing = timing.get("entry", {}).get("type", "WAIT")
                entry_urgency = timing.get("entry", {}).get("urgency", "NONE")

                # 청산 정보
                exit_timing = timing.get("exit", {})
                stop_loss = exit_timing.get("stop_loss", 0.0)
                take_profit_levels = exit_timing.get("take_profit_levels", [])
                take_profit_1 = take_profit_levels[0] if take_profit_levels else 0.0
                trailing_stop = exit_timing.get("trailing_stop", 0.0)
                max_holding_days = exit_timing.get("max_holding_days", 0)

                # 리스크 정보
                risk_management = trading_signal.get("risk_management", {})
                risk_level = risk_management.get("risk_level", "MEDIUM")

                # 시장 정보
                market_regime = trading_signal.get("market_regime", "N/A")
                regime_confidence = trading_signal.get("regime_confidence", 0.0)

                # 권고사항 (간단하게)
                recommendations = trading_signal.get("recommendations", {})
                primary_rec = recommendations.get("primary_recommendation", "")

                # 데이터 행 추가
                data.append(
                    {
                        "종목": symbol,
                        "액션": action,
                        "액션강도": f"{action_strength:.2f}",
                        "투자점수": f"{score:.3f}",
                        "신뢰도": f"{confidence:.1%}",
                        "포지션": f"{position_size:.1%}",
                        "우선순위": execution_priority,
                        "진입타이밍": entry_timing,
                        "진입긴급도": entry_urgency,
                        "손절선": f"{stop_loss:.1%}",
                        "이익실현1": f"{take_profit_1:.1%}",
                        "트레일링": f"{trailing_stop:.1%}",
                        "최대보유일": f"{max_holding_days}일",
                        "리스크": risk_level,
                        "시장체제": market_regime,
                        "체제신뢰도": f"{regime_confidence:.1%}",
                        "주요권고": (
                            primary_rec[:50] + "..."
                            if len(primary_rec) > 50
                            else primary_rec
                        ),
                    }
                )

            # DataFrame 생성
            df = pd.DataFrame(data)

            # 우선순위 순으로 정렬
            df = df.sort_values("우선순위").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"매매신호 DataFrame 생성 실패: {e}")
            return pd.DataFrame()

    def _print_trading_signals_dataframe(
        self, analysis_results: Dict, max_rows: int = 20
    ) -> None:
        """
        pandas DataFrame을 이용한 깔끔한 매매신호 표 출력

        Args:
            analysis_results: 분석 결과
            max_rows: 최대 출력 행 수
        """
        try:
            df = self._create_trading_signals_dataframe(analysis_results)

            if df.empty:
                print("❌ 출력할 매매신호 데이터가 없습니다.")
                return

            print("\n" + "=" * 120)
            print("📊 매매신호 종합표 (pandas DataFrame)")
            print("=" * 120)

            # 중요한 컬럼들만 먼저 출력
            essential_cols = [
                "종목",
                "액션",
                "투자점수",
                "신뢰도",
                "포지션",
                "우선순위",
                "진입타이밍",
                "손절선",
                "리스크",
            ]
            essential_df = df[essential_cols].head(max_rows)

            # pandas의 깔끔한 출력 사용
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", 15)

            print(f"\n🎯 핵심 정보 ({len(essential_df)}개 종목)")
            print(essential_df.to_string(index=False))

            # 상세 정보 표
            detail_cols = [
                "종목",
                "액션강도",
                "이익실현1",
                "트레일링",
                "최대보유일",
                "시장체제",
                "주요권고",
            ]
            detail_df = df[detail_cols].head(max_rows)

            print(f"\n📋 상세 정보")
            print(detail_df.to_string(index=False))

            # 통계 요약
            print(f"\n📈 통계 요약:")
            print(f"   총 종목 수: {len(df)}")

            action_counts = df["액션"].value_counts()
            print(f"   액션 분포:")
            for action, count in action_counts.items():
                print(f"     {action}: {count}개")

            risk_counts = df["리스크"].value_counts()
            print(f"   리스크 분포:")
            for risk, count in risk_counts.items():
                print(f"     {risk}: {count}개")

            # 고우선순위 종목
            high_priority = df[df["우선순위"] <= 3]
            if not high_priority.empty:
                print(f"\n⚡ 고우선순위 종목 ({len(high_priority)}개):")
                for _, row in high_priority.iterrows():
                    print(
                        f"   {row['종목']}: {row['액션']} (점수: {row['투자점수']}, 우선순위: {row['우선순위']})"
                    )

            print("\n" + "=" * 120)

        except Exception as e:
            logger.error(f"DataFrame 매매신호 표 출력 실패: {e}")
            print(f"❌ DataFrame 표 출력 실패: {e}")

    def _save_trading_signals_to_csv(
        self, analysis_results: Dict, output_dir: str = "results/trader"
    ) -> str:
        """매매 신호를 CSV 파일로 저장"""
        try:
            # 결과 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)

            # 데이터프레임 생성
            df = self._create_trading_signals_dataframe(analysis_results)

            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_signals_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)

            # CSV 저장
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

            # 요약 정보도 함께 저장
            summary_info = {
                "생성시간": datetime.now().isoformat(),
                "총종목수": len(df),
                "액션분포": df["액션"].value_counts().to_dict(),
                "리스크분포": df["리스크"].value_counts().to_dict(),
                "고우선순위종목수": len(df[df["우선순위"] <= 3]),
                "평균투자점수": df["투자점수"].astype(str).astype(float).mean(),
                "시장체제": analysis_results.get("market_regime", {}).get(
                    "regime", "N/A"
                ),
                "포트폴리오점수": analysis_results.get("portfolio_summary", {}).get(
                    "portfolio_score", 0.0
                ),
            }

            summary_filename = f"trading_signals_summary_{timestamp}.json"
            summary_filepath = os.path.join(output_dir, summary_filename)

            import json

            with open(summary_filepath, "w", encoding="utf-8") as f:
                json.dump(summary_info, f, indent=2, ensure_ascii=False)

            logger.info(f"매매신호 CSV 저장 완료: {filepath}")
            logger.info(f"요약 정보 JSON 저장 완료: {summary_filepath}")

            return filepath

        except Exception as e:
            logger.error(f"매매신호 CSV 저장 실패: {e}")
            return ""

    def _generate_comprehensive_report(
        self,
        portfolio_result: Dict[str, Any],
        backtest_result: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
    ) -> str:
        """종합 백테스팅 리포트 생성"""
        try:
            # NeuralPortfolioManager를 사용하여 종합 리포트 생성
            portfolio_manager = NeuralPortfolioManager(self.config)
            return portfolio_manager._generate_comprehensive_backtest_report(
                portfolio_result, backtest_result, historical_data
            )
        except Exception as e:
            logger.error(f"종합 리포트 생성 실패: {e}")
            return "종합 리포트 생성 중 오류가 발생했습니다."


def main():
    """메인 실행 함수"""
    try:
        # 트레이더 초기화 (분석 모드)
        trader = HybridTrader(analysis_mode=True)

        # 모델 초기화
        print("모델 초기화 중...")
        if not trader.initialize_models():
            print("모델 초기화 실패")
            return

        print("모델 초기화 완료")

        # 분석 실행
        print("분석 실행 중...")
        results = trader.run_analysis()

        if "status" in results and results["status"] == "error":
            print(f"분석 실패: {results['message']}")
            return

        print("분석 완료!")

        # 1. 통합 매매 리포트 출력 (모든 정보 포함)
        market_regime = results.get("market_regime", {})
        portfolio_summary = results.get("portfolio_summary", {})
        individual_results = results.get("individual_results", [])

        comprehensive_report = formatted_output.format_comprehensive_trading_report(
            market_regime=market_regime,
            portfolio_summary=portfolio_summary,
            individual_results=individual_results,
        )
        print(f"\n{comprehensive_report}")

        # 2. CSV 파일로 저장
        csv_filepath = trader._save_trading_signals_to_csv(results)
        if csv_filepath:
            print(f"📄 매매신호 CSV 저장됨: {csv_filepath}")

        # 3. 빠른 요약 출력
        regime_data = {
            "regime": market_regime.get("regime", "UNKNOWN"),
            "confidence": market_regime.get("confidence", 0),
            "portfolio_score": portfolio_summary.get("portfolio_score", 0),
            "portfolio_action": portfolio_summary.get("portfolio_action", "UNKNOWN"),
            "signal_distribution": portfolio_summary.get("signal_distribution", {}),
        }
        quick_summary = formatted_output.format_quick_summary(regime_data)
        print(f"\n{quick_summary}")

        # 4. 백테스팅 결과가 있는 경우 추가 리포트 출력
        if "backtest_result" in results:
            backtest_result = results["backtest_result"]
            portfolio_result = results.get("portfolio_result", {})
            historical_data = results.get("historical_data", {})

            # 종합 백테스팅 리포트 생성
            backtest_report = trader._generate_comprehensive_report(
                portfolio_result, backtest_result, historical_data
            )
            print(f"\n{backtest_report}")

    except Exception as e:
        print(f"실행 실패: {e}")
        logger.error(f"메인 실행 실패: {e}")

    # 최종 완료 메시지
    print("\n[INFO] 🎉 전체 프로세스 완료!")
    print("[INFO] 📊 결과 파일: results/trader/")
    print("[INFO] 📝 로그 파일: log/trader.log")
    print("\n[INFO] 🔍 생성된 리포트:")
    print("[INFO]    • 개별 종목 예측 결과")
    print("[INFO]    • 포트폴리오 최적 비중")
    print("[INFO]    • 백테스팅 상세 분석")
    print("[INFO]    • Buy & Hold 대비 성과 비교")
    print("[INFO]    • 매매 내역 및 최종 보유 현황")


if __name__ == "__main__":
    main()
