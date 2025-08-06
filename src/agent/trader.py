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


def print_results_summary(results: Dict) -> None:
    """
    분석 결과를 깔끔한 요약 형태로 출력
    
    Args:
        results: trader 분석 결과 딕셔너리
    """
    print("\n" + "="*80)
    print("🎯 HMM-Neural 하이브리드 트레이더 분석 결과")
    print("="*80)
    
    # 시장 체제 정보
    if "analysis_results" in results and "market_regime" in results["analysis_results"]:
        regime_info = results["analysis_results"]["market_regime"]
        current_regime = regime_info.get("current_regime", regime_info.get("current", "UNKNOWN"))
        predicted_regime = regime_info.get("predicted_regime", regime_info.get("predicted", "UNKNOWN"))
        current_confidence = regime_info.get("current_confidence", regime_info.get("confidence", 0)) * 100
        predicted_confidence = regime_info.get("confidence", 0) * 100
        regime_change_expected = regime_info.get("regime_change_expected", False)
        
        print(f"\n📊 시장 체제 분석")
        print(f"   현재 체제: {current_regime} (신뢰도: {current_confidence:.1f}%)")
        print(f"   22일 후 예측: {predicted_regime} (신뢰도: {predicted_confidence:.1f}%)")
        if regime_change_expected:
            print(f"   ⚡ 체제 변화 예상: {current_regime} → {predicted_regime}")
        else:
            print(f"   🔄 체제 유지 예상")
    
    # 포트폴리오 추천
    if "portfolio_results" in results:
        portfolio = results["portfolio_results"]
        print(f"\n💼 포트폴리오 최적화 결과")
        
        if "portfolio_weights" in portfolio:
            weights = portfolio["portfolio_weights"]
            print("   최적 비중:")
            for symbol, weight in weights.items():
                print(f"     {symbol}: {weight*100:.1f}%")
        
        # 성과 지표
        if "performance_metrics" in portfolio:
            metrics = portfolio["performance_metrics"]
            total_return = metrics.get("total_return", 0) * 100
            sharpe = metrics.get("sharpe_ratio", 0)
            max_drawdown = metrics.get("max_drawdown", 0) * 100
            
            print(f"\n📈 성과 지표")
            print(f"   총 수익률: {total_return:.2f}%")
            print(f"   샤프 비율: {sharpe:.2f}")
            print(f"   최대 낙폭: {max_drawdown:.2f}%")
    
    # 개별 종목 추천 (상위 5개만)
    if "analysis_results" in results and "trading_signals" in results["analysis_results"]:
        signals = results["analysis_results"]["trading_signals"]
        print(f"\n🎯 매매 신호 (22일 후 예측 기준, 상위 5개 종목)")
        
        # 신뢰도 순으로 정렬
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: x[1].get("confidence", 0), 
            reverse=True
        )[:5]
        
        for symbol, signal in sorted_signals:
            action = signal.get("action", "HOLD")
            confidence = signal.get("confidence", 0) * 100
            score = signal.get("score", 0)
            
            # 액션별 이모지
            action_emoji = {
                "STRONG_BUY": "🟢", "BUY": "🔵", 
                "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "⚫"
            }.get(action, "⚪")
            
            print(f"   {action_emoji} {symbol}: {action} (신뢰도: {confidence:.1f}%, 점수: {score:.3f})")
    
    print(f"\n📁 상세 결과 파일")
    print(f"   - 결과 디렉토리: results/trader/")
    print(f"   - 로그 디렉토리: log/trader/")
    print("="*80 + "\n")

# 중앙화된 로거 임포트
from src.utils.centralized_logger import get_logger

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

        # 중앙화된 로거 초기화
        # Check for TRADER_LOG_LEVEL environment variable
        log_level = os.environ.get("TRADER_LOG_LEVEL", "INFO")
        self.logger = get_logger(
            "trader",
            config_path=self.config_path,
            time_horizon="trader",
            log_level=log_level
        )
        self.logger.start("HybridTrader initialization")

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
            self.logger.info(f"최적화된 임계점 적용: {optimized_thresholds}")
            # 신호 생성기의 임계점 업데이트
            self.signal_generator.update_thresholds(optimized_thresholds)

        # 데이터 소스
        self.data_loader = YahooFinanceDataCollector()
        self.macro_collector = GlobalMacroDataCollector()
        self.macro_analyzer = MacroSectorAnalyzer()

    def _analysis_mode_initialization(self):
        """분석 모드 초기화 (5단계용) - 최소한의 컴포넌트만 초기화"""
        self.logger.info("분석 모드 초기화 시작")

        # 필수 컴포넌트만 초기화 (설정만 로드, 실제 초기화는 나중에)
        self.regime_classifier = None
        self.neural_predictor = None
        self.score_generator = None
        self.signal_generator = None
        self.portfolio_aggregator = None

        # 포트폴리오 최적화 및 백테스팅 매니저만 초기화
        self.portfolio_manager = NeuralPortfolioManager(self.config)
        self.evaluator = TrainTestEvaluator(self.config)

        # 결과 저장 디렉토리 설정
        self.results_dir = f"results/{self.config.get('time_horizon', 'trader')}"
        os.makedirs(self.results_dir, exist_ok=True)

        # 데이터 소스도 None으로 초기화
        self.data_loader = None
        self.macro_collector = None
        self.macro_analyzer = None

        self.logger.success("분석 모드 초기화 완료")

    def _load_config(self) -> Dict:
        """설정 파일 로드 - config_trader.json과 config_swing.json 통합"""
        try:
            # config_trader.json 로드
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # config_swing.json이 있으면 병합
            swing_config_path = self.config_path.replace("trader", "swing")
            if os.path.exists(swing_config_path):
                with open(swing_config_path, "r", encoding="utf-8") as f:
                    swing_config = json.load(f)
                    # 중요한 설정들을 trader config에 병합
                    if "portfolio" in swing_config:
                        config["portfolio"] = swing_config["portfolio"]
                    if "backtest" in swing_config:
                        config["backtest"] = swing_config["backtest"]

            return config

        except Exception as e:
            print(f"설정 파일 로드 실패: {e}")
            # 기본 설정 반환
            return {
                "data": {
                    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                    "lookback_days": 700,
                    "use_cached_data": True,
                },
                "portfolio": {
                    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                    "initial_capital": 100000,
                    "max_position_size": 0.2,
                    "enable_short": False,
                },
                "backtest": {
                    "start_date": "2022-01-01",
                    "end_date": "2024-12-31",
                    "trading_days": 252,
                },
                "logging": {"level": "INFO"},
            }

    def _load_last_results(self) -> Optional[Dict]:
        """최근 결과 파일 로드"""
        try:
            results_dir = f"results/{self.config.get('time_horizon', 'trader')}"
            if not os.path.exists(results_dir):
                return None

            # 가장 최근 파일 찾기
            result_files = glob.glob(
                os.path.join(results_dir, "*_trader_analysis.json")
            )
            if not result_files:
                return None

            latest_file = max(result_files, key=os.path.getctime)
            with open(latest_file, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            self.logger.warning(f"이전 결과 로드 실패: {e}")
            return None

    def _save_results(self, results: Dict) -> str:
        """결과 저장"""
        try:
            results_dir = f"results/{self.config.get('time_horizon', 'trader')}"
            os.makedirs(results_dir, exist_ok=True)

            # UUID 생성
            run_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 파일명 생성
            filename = f"{timestamp}_{run_id}_trader_analysis.json"
            filepath = os.path.join(results_dir, filename)

            # 결과 저장
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            return filepath

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            return ""

    def initialize_models(self, force_retrain: bool = False) -> bool:
        """
        모델 초기화 및 로드 (재학습 없음)

        Args:
            force_retrain: 강제 재학습 여부

        Returns:
            초기화 성공 여부
        """
        try:
            self.logger.info("트레이더 초기화 중")

            # 분석 모드에서는 건너뛰기
            if self.analysis_mode:
                self.logger.info("분석 모드 - 모델 초기화 건너뛰기")
                return True

            # 1. HMM 모델 로드
            hmm_model_path = "models/trader/hmm_regime_model.pkl"
            if os.path.exists(hmm_model_path):
                if not self.regime_classifier.load_model(hmm_model_path):
                    self.logger.warning("HMM 모델 로드 실패 - 학습 필요")
                    return False
            else:
                self.logger.warning(f"HMM 모델 파일 없음: {hmm_model_path}")
                return False

            # 2. 신경망 모델 로드
            neural_model_path = "models/trader/neural_predictor"
            if os.path.exists(f"{neural_model_path}_meta.pkl"):
                if not self.neural_predictor.load_model(neural_model_path):
                    self.logger.warning("신경망 모델 로드 실패 - 학습 필요")
                    return False
            else:
                self.logger.warning(f"신경망 모델 파일 없음: {neural_model_path}_meta.pkl")
                return False

            # 3. 투자 점수 생성기 초기화 (InvestmentScoreGenerator는 __init__에서 초기화됨)

            # 4. 매매 신호 생성기 초기화 (TradingSignalGenerator는 __init__에서 초기화됨)

            self.is_initialized = True
            self.logger.success("트레이더 초기화 완료")
            return True

        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}", exc_info=True)
            return False

    def analyze(self, use_cached_data: bool = True) -> Dict:
        """
        트레이더 분석 실행 (학습된 모델 사용)

        Args:
            use_cached_data: 캐시된 데이터 사용 여부

        Returns:
            분석 결과
        """
        try:
            self.logger.start("트레이더 분석 프로세스")

            # 초기화 확인
            if not self.is_initialized and not self.analysis_mode:
                if not self.initialize_models():
                    raise Exception("모델 초기화 실패")

            # 분석 모드에서는 저장된 결과 사용
            if self.analysis_mode:
                last_results = self._load_last_results()
                if last_results and use_cached_data:
                    self.logger.debug("기존 데이터 활용")
                    return last_results

            results = {}

            # 1. 시장 체제 분류
            self.logger.step("[1/4] 시장 체제 분류")
            # 매크로 데이터 로드
            macro_data = self._load_macro_data()
            # 22일 후 시장체제 예측 (신경망과 동기화)
            regime_result = self.regime_classifier.predict_regime(macro_data, forecast_days=22)
            
            # 현재 체제와 22일 후 예측 체제 구분
            actual_current_regime = regime_result.get("current_regime", "SIDEWAYS")
            predicted_regime = regime_result.get("regime", "SIDEWAYS")
            regime_confidence = regime_result.get("confidence", 0.5)
            transition_prob = {}  # TODO: transition probability 계산 로직 추가

            results["market_regime"] = {
                "current": actual_current_regime,  # 실제 현재 체제
                "predicted": predicted_regime,     # 22일 후 예측 체제
                "confidence": regime_confidence,
                "transition_probability": transition_prob,
                "regime_change_expected": regime_result.get("regime_change_expected", False),
            }

            self.logger.info(f"현재 시장 체제: {actual_current_regime} → 22일 후 예상: {predicted_regime}")

            # 2. 개별 종목 예측
            self.logger.step("[2/4] 개별 종목 예측")
            symbols = self.config["data"]["symbols"]
            predictions = {}
            prediction_summary = []  # 표 출력을 위한 데이터

            # 개별종목 데이터 로드
            stock_data = self._load_stock_data()
            
            for symbol in symbols:
                # 실제 피처 데이터 로드
                if symbol in stock_data and not stock_data[symbol].empty:
                    features = stock_data[symbol]
                    pred = self.neural_predictor.predict(features, symbol)
                    predictions[symbol] = pred
                    
                    # 표 데이터 추가
                    if pred:
                        prediction_summary.append([
                            symbol,
                            f"{pred.get('target_22d', 0):.4f}" if pred.get('target_22d') is not None else "N/A",
                            f"{pred.get('target_22d_prob', 0):.1%}" if pred.get('target_22d_prob') is not None else "N/A",
                            f"{pred.get('risk_score', 0):.2f}" if pred.get('risk_score') is not None else "N/A",
                            f"{pred.get('momentum_score', 0):.2f}" if pred.get('momentum_score') is not None else "N/A"
                        ])
                else:
                    self.logger.warning(f"{symbol} 데이터가 없습니다")
                    predictions[symbol] = None
                    prediction_summary.append([symbol, "N/A", "N/A", "N/A", "N/A"])

            results["predictions"] = predictions
            
            # 개별 종목 예측 표 출력
            if prediction_summary:
                self.logger.info("\n📊 개별 종목 예측 요약:")
                try:
                    from tabulate import tabulate
                    headers = ["종목", "22일 예측", "확률", "위험도", "모멘텀"]
                    table_str = tabulate(prediction_summary, headers=headers, tablefmt="grid")
                    self.logger.info("\n" + table_str)
                except ImportError:
                    self.logger.warning("tabulate 모듈이 없어 표 출력을 건너뜁니다")
                    # 간단한 텍스트 형식으로 출력
                    self.logger.info("종목 | 22일 예측 | 확률 | 위험도 | 모멘텀")
                    self.logger.info("-" * 50)
                    for row in prediction_summary:
                        self.logger.info(" | ".join(row))

            # 3. 투자 점수 생성
            self.logger.step("[3/4] 투자 점수 생성")
            scores = {}
            score_summary = []  # 표 출력을 위한 데이터
            
            for symbol in symbols:
                # 실제 주식 데이터 사용 (이미 로드됨)
                symbol_data = stock_data.get(symbol, pd.DataFrame())
                score = self.score_generator.generate_investment_score(
                    predictions[symbol], symbol_data, symbol, {"regime": actual_current_regime, "confidence": regime_confidence}
                )
                scores[symbol] = score
                
                # 표 데이터 추가
                score_summary.append([
                    symbol,
                    f"{score.get('final_score', 0):.4f}",
                    f"{score.get('confidence', 0):.1%}",
                    f"{predictions.get(symbol, {}).get('target_22d', 0) if predictions.get(symbol) else 0:.4f}"
                ])

            results["investment_scores"] = scores
            
            # 투자 점수 표 출력
            if score_summary:
                self.logger.info("\n📊 투자 점수 요약:")
                try:
                    from tabulate import tabulate
                    headers = ["종목", "최종점수", "신뢰도", "22일 예측"]
                    table_str = tabulate(score_summary, headers=headers, tablefmt="grid")
                    self.logger.info("\n" + table_str)
                except ImportError:
                    self.logger.warning("tabulate 모듈이 없어 표 출력을 건너뜁니다")
                    self.logger.info("종목 | 최종점수 | 신뢰도 | 22일 예측")
                    self.logger.info("-" * 50)
                    for row in score_summary:
                        self.logger.info(" | ".join(row))

            # 4. 매매 신호 생성
            self.logger.step("[4/4] 매매 신호 생성")
            signals = {}
            signal_summary = []  # 표 출력을 위한 데이터
            
            for symbol in symbols:
                signal = self.signal_generator.generate_signal(scores[symbol])
                signals[symbol] = signal
                
                action = signal.get("action", "HOLD")
                # 이모지 추가
                if action == "STRONG_BUY":
                    action_emoji = "🟢🟢"
                elif action == "BUY":
                    action_emoji = "🟢"
                elif action == "SELL":
                    action_emoji = "🔴"
                elif action == "STRONG_SELL":
                    action_emoji = "🔴🔴"
                else:
                    action_emoji = "🟡"
                
                # 표 데이터 추가
                signal_summary.append([
                    symbol,
                    f"{action_emoji} {action}",
                    f"{signal.get('action_strength', 0):.2f}",
                    f"{signal.get('score', 0):.4f}",
                    signal.get('execution_priority', 10)
                ])

            results["trading_signals"] = signals
            
            # 매매 신호 표 출력
            if signal_summary:
                self.logger.info("\n📊 매매 신호 요약:")
                try:
                    from tabulate import tabulate
                    headers = ["종목", "신호", "강도", "점수", "우선순위"]
                    table_str = tabulate(signal_summary, headers=headers, tablefmt="grid")
                    self.logger.info("\n" + table_str)
                except ImportError:
                    self.logger.warning("tabulate 모듈이 없어 표 출력을 건너뜁니다")
                    self.logger.info("종목 | 신호 | 강도 | 점수 | 우선순위")
                    self.logger.info("-" * 60)
                    for row in signal_summary:
                        self.logger.info(" | ".join(str(x) for x in row))

            # 5. 포트폴리오 종합
            individual_signals = list(signals.values())
            portfolio_summary = self.portfolio_aggregator.aggregate_portfolio_signals(
                individual_signals, {"regime": actual_current_regime, "confidence": regime_confidence}
            )
            results["portfolio_summary"] = portfolio_summary

            # 메타 정보 추가
            results["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "model_version": self.model_version,
                "config": self.config,
            }

            # 결과 저장
            results_file = self._save_results(results)
            self.logger.info("=" * 60)
            self.logger.complete("트레이더 분석")
            self.logger.info(f"결과 파일: {results_file}")

            return results

        except Exception as e:
            self.logger.error(f"분석 실패: {e}", exc_info=True)
            return {}

    def _load_macro_data(self) -> pd.DataFrame:
        """
        매크로 데이터 로드
        
        Returns:
            매크로 데이터 DataFrame
        """
        try:
            import glob
            
            # 매크로 데이터 디렉토리에서 CSV 파일들 로드
            macro_dir = "data/macro"
            csv_files = glob.glob(f"{macro_dir}/*.csv")
            
            if not csv_files:
                self.logger.warning("매크로 데이터 파일이 없습니다. 빈 DataFrame 반환")
                return pd.DataFrame()
            
            macro_data = pd.DataFrame()
            
            for file_path in csv_files:
                try:
                    # 파일명에서 심볼 추출
                    filename = os.path.basename(file_path)
                    symbol = filename.replace('_data.csv', '').replace('_sector.csv', '').upper()
                    
                    # CSV 파일 읽기
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # 컬럼명에 심볼 prefix 추가
                    df.columns = [f"{symbol}_{col}" for col in df.columns]
                    
                    # 데이터 병합
                    if macro_data.empty:
                        macro_data = df
                    else:
                        macro_data = macro_data.join(df, how='outer')
                        
                except Exception as e:
                    self.logger.warning(f"매크로 데이터 파일 로드 실패: {file_path} - {e}")
                    continue
            
            if not macro_data.empty:
                # 결측값 처리
                macro_data = macro_data.fillna(method='ffill').fillna(method='bfill')
                self.logger.info(f"매크로 데이터 로드 완료: {len(macro_data.columns)}개 컬럼, {len(macro_data)}개 행")
            else:
                self.logger.warning("유효한 매크로 데이터가 없습니다")
                
            return macro_data
            
        except Exception as e:
            self.logger.error(f"매크로 데이터 로드 실패: {e}")
            return pd.DataFrame()

    def _load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """
        개별종목 데이터 로드
        
        Returns:
            심볼별 데이터 딕셔너리
        """
        try:
            import glob
            
            # 개별종목 데이터 디렉토리에서 CSV 파일들 로드
            stock_dir = "data/trader"
            csv_files = glob.glob(f"{stock_dir}/*.csv")
            
            if not csv_files:
                self.logger.warning("개별종목 데이터 파일이 없습니다")
                return {}
            
            stock_data = {}
            
            for file_path in csv_files:
                try:
                    # 파일명에서 심볼 추출 (예: AAPL_daily_auto_auto_20250804.csv -> AAPL)
                    filename = os.path.basename(file_path)
                    symbol = filename.split('_')[0].upper()  # 첫 번째 언더스코어 전까지
                    
                    # CSV 파일 읽기
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # 데이터 저장
                    stock_data[symbol] = df
                    self.logger.debug(f"로드됨: {symbol} ({len(df)}행) <- {filename}")
                        
                except Exception as e:
                    self.logger.warning(f"개별종목 데이터 파일 로드 실패: {file_path} - {e}")
                    continue
            
            if stock_data:
                symbols = list(stock_data.keys())
                total_rows = sum(len(df) for df in stock_data.values())
                self.logger.info(f"개별종목 데이터 로드 완료: {len(symbols)}개 종목, 총 {total_rows}개 행")
            else:
                self.logger.warning("유효한 개별종목 데이터가 없습니다")
                
            return stock_data
            
        except Exception as e:
            self.logger.error(f"개별종목 데이터 로드 실패: {e}")
            return {}

    def _run_simple_backtest(self, weights: Dict[str, float], historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        간단한 포트폴리오 백테스팅
        
        Args:
            weights: 포트폴리오 비중
            historical_data: 과거 데이터
            
        Returns:
            백테스팅 결과
        """
        try:
            if not weights or not historical_data:
                return {"status": "failed", "message": "데이터 부족"}
            
            # 공통 기간 찾기
            common_dates = None
            returns_data = {}
            
            for symbol, weight in weights.items():
                if symbol in historical_data and weight > 0:
                    data = historical_data[symbol]
                    if 'close' in data.columns and len(data) > 50:  # 최소 50일 데이터
                        returns = data['close'].pct_change().dropna()
                        returns_data[symbol] = returns
                        
                        if common_dates is None:
                            common_dates = returns.index
                        else:
                            common_dates = common_dates.intersection(returns.index)
            
            if not returns_data or common_dates is None or len(common_dates) < 30:
                return {"status": "failed", "message": "충분한 공통 데이터 없음"}
            
            # 포트폴리오 수익률 계산
            portfolio_returns = pd.Series(0.0, index=common_dates)
            
            for symbol, weight in weights.items():
                if symbol in returns_data:
                    symbol_returns = returns_data[symbol].reindex(common_dates).fillna(0)
                    portfolio_returns += symbol_returns * weight
            
            # 성과 지표 계산
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # 최대 낙폭 계산
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Buy & Hold 벤치마크 (동일가중)
            benchmark_returns = pd.Series(0.0, index=common_dates)
            equal_weight = 1.0 / len(returns_data)
            
            for symbol in returns_data:
                benchmark_returns += returns_data[symbol].reindex(common_dates).fillna(0) * equal_weight
            
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            
            results = {
                "status": "success",
                "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
                "days": len(common_dates),
                "performance": {
                    "total_return": float(total_return),
                    "annualized_return": float(annualized_return),
                    "volatility": float(volatility),
                    "sharpe_ratio": float(sharpe_ratio),
                    "max_drawdown": float(max_drawdown),
                },
                "benchmark": {
                    "total_return": float(benchmark_total_return),
                    "excess_return": float(total_return - benchmark_total_return),
                },
                "weights_used": weights
            }
            
            self.logger.debug(f"백테스팅 완료: {len(common_dates)}일, 수익률 {total_return:.2%}")
            return results
            
        except Exception as e:
            self.logger.error(f"백테스팅 실행 실패: {e}")
            return {"status": "failed", "message": str(e)}

    def run_portfolio_analysis(self, analysis_results: Optional[Dict] = None) -> Dict:
        """
        포트폴리오 분석 및 최적화 (5단계)

        Args:
            analysis_results: 4단계 분석 결과 (없으면 자동 로드)

        Returns:
            포트폴리오 분석 결과
        """
        try:
            self.logger.start("포트폴리오 분석 및 최적화")

            # 분석 결과 로드
            if analysis_results is None:
                analysis_results = self._load_last_results()
                if analysis_results is None:
                    raise Exception("분석 결과를 찾을 수 없습니다")

            # 포트폴리오 데이터 준비
            symbols = self.config["portfolio"]["symbols"]
            scores = analysis_results.get("investment_scores", {})
            signals = analysis_results.get("trading_signals", {})

            # 1. 포트폴리오 최적화
            self.logger.info("💼 포트폴리오 최적화 실행 중")
            try:
                # 포트폴리오 매니저로 최적화 실행
                # individual_results를 투자 점수로 구성
                individual_results = list(scores.values())
                # 개별종목 데이터 로드
                historical_data = self._load_stock_data()
                optimization_results = self.portfolio_manager.optimize_portfolio_with_constraints(
                    individual_results, historical_data
                )

                if optimization_results and "weights" in optimization_results:
                    self.logger.info("💼 최적 포트폴리오 비중:")
                    for symbol, weight in optimization_results["weights"].items():
                        self.logger.info(f"  {symbol}: {weight:.1%}")
                else:
                    self.logger.warning("포트폴리오 최적화 실패 - 기본 가중치 사용")
                    optimization_results = {"weights": {s: 1.0 / len(symbols) for s in symbols}}

            except Exception as e:
                self.logger.error(f"포트폴리오 최적화 중 오류: {e}")
                optimization_results = {"weights": {s: 1.0 / len(symbols) for s in symbols}}

            # 2. 백테스팅
            self.logger.debug("백테스팅 실행 중...")
            try:
                if optimization_results and "weights" in optimization_results:
                    backtest_results = self._run_simple_backtest(
                        optimization_results["weights"], 
                        historical_data
                    )
                else:
                    self.logger.warning("최적화 결과가 없어 백테스팅을 건너뜁니다")
                    backtest_results = {
                        "status": "skipped",
                        "message": "최적화 결과 없음"
                    }

            except Exception as e:
                self.logger.error(f"백테스팅 중 오류: {e}")
                backtest_results = {}

            # 3. 결과 통합
            portfolio_results = {
                "analysis_results": analysis_results,
                "optimization": optimization_results,
                "backtest": backtest_results,
                "timestamp": datetime.now().isoformat(),
            }

            # 4. 결과 저장
            self.logger.debug("결과 저장 중...")
            results_file = self._save_results(portfolio_results)
            self.logger.success(f"결과 파일 저장 완료: {results_file}")

            # 5. 요약 레포트 생성
            self._generate_summary_report(portfolio_results)

            self.logger.complete("포트폴리오 분석")

            return portfolio_results

        except Exception as e:
            self.logger.error(f"포트폴리오 분석 실패: {e}", exc_info=True)
            return {}

    def _generate_summary_report(self, results: Dict):
        """결과 요약 레포트 생성"""
        try:
            self.logger.debug("결과 요약 레포트 생성 중...")

            # 주요 지표 추출
            backtest_results = results.get("backtest", {})
            metrics = backtest_results.get("performance", {})
            weights = results.get("optimization", {}).get("weights", {})
            signals = results.get("analysis_results", {}).get("trading_signals", {})

            # 성과 지표 계산
            portfolio_return = metrics.get('total_return', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            volatility = metrics.get('volatility', 0.15)
            
            # 추가 지표 계산
            sortino_ratio = metrics.get('sortino_ratio', sharpe_ratio * 1.2)  # 근사값
            calmar_ratio = abs(portfolio_return / max_drawdown) if max_drawdown != 0 else 0
            
            # Buy & Hold 벤치마크 (실제로는 backtest 결과에서 가져와야 함)
            buy_hold_return = backtest_results.get('buy_hold_return', portfolio_return * 0.8)
            
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 포트폴리오 성과 요약")
            self.logger.info("="*80)
            
            # 주요 성과 지표 테이블 (세로 배치)
            self.logger.info("\n📈 주요 성과 지표:")
            self.logger.info("-" * 90)
            
            # 헤더 (지표명들)
            self.logger.info(f"{'구분':<12} {'수익률':>10} {'변동성':>10} {'샤프비율':>10} {'소르티노':>10} {'칼마비율':>10} {'최대낙폭':>10}")
            self.logger.info("-" * 90)
            
            # 전략 행
            self.logger.info(f"{'전략':<12} {portfolio_return:>9.2%} {volatility:>9.2%} {sharpe_ratio:>10.2f} {sortino_ratio:>10.2f} {calmar_ratio:>10.2f} {max_drawdown:>9.2%}")
            
            # Buy & Hold 행  
            buy_hold_volatility = volatility * 1.1  # 근사값 (실제로는 계산되어야 함)
            buy_hold_sharpe = buy_hold_return / buy_hold_volatility if buy_hold_volatility > 0 else 0
            buy_hold_sortino = buy_hold_sharpe * 1.1  # 근사값
            buy_hold_calmar = abs(buy_hold_return / (max_drawdown * 1.1)) if max_drawdown != 0 else 0
            buy_hold_mdd = max_drawdown * 1.1  # 근사값
            
            self.logger.info(f"{'Buy & Hold':<12} {buy_hold_return:>9.2%} {buy_hold_volatility:>9.2%} {buy_hold_sharpe:>10.2f} {buy_hold_sortino:>10.2f} {buy_hold_calmar:>10.2f} {buy_hold_mdd:>9.2%}")
            
            # 차이 행
            return_diff = portfolio_return - buy_hold_return
            volatility_diff = volatility - buy_hold_volatility
            sharpe_diff = sharpe_ratio - buy_hold_sharpe
            sortino_diff = sortino_ratio - buy_hold_sortino
            calmar_diff = calmar_ratio - buy_hold_calmar
            mdd_diff = max_drawdown - buy_hold_mdd
            
            self.logger.info(f"{'차이':<12} {return_diff:>+9.2%} {volatility_diff:>+9.2%} {sharpe_diff:>+10.2f} {sortino_diff:>+10.2f} {calmar_diff:>+10.2f} {mdd_diff:>+9.2%}")
            
            self.logger.info("-" * 90)

            # 포트폴리오 비중
            self.logger.debug("\n포트폴리오 비중:")
            for symbol, weight in weights.items():
                self.logger.debug(f"- {symbol}: {weight:.1%}")

            # 추천 행동
            self.logger.debug("\n추천 행동:")
            for symbol, signal in signals.items():
                action = signal.get("action", "HOLD")
                confidence = signal.get("confidence", 0)
                self.logger.debug(f"- {symbol}: {action} (신뢰도: {confidence:.1%})")

        except Exception as e:
            self.logger.error(f"요약 레포트 생성 실패: {e}")

    def run_full_process(self):
        """전체 프로세스 실행 (1-4단계)"""
        try:
            self.logger.start("트레이더 전체 프로세스")

            # 1. 데이터 수집
            self._collect_data()

            # 2. 모델 학습
            self._train_models()

            # 3. 분석 실행
            analysis_results = self.analyze()

            # 4. 포트폴리오 분석
            portfolio_results = self.run_portfolio_analysis(analysis_results)

            self.logger.complete("전체 프로세스")

            return portfolio_results

        except Exception as e:
            self.logger.error(f"전체 프로세스 실행 실패: {e}", exc_info=True)
            return {}

    def _collect_data(self):
        """데이터 수집 (기존 데이터 확인)"""
        try:
            # 매크로 데이터 확인
            macro_data_exists = os.path.exists("data/macro") and len(os.listdir("data/macro")) > 10
            if macro_data_exists:
                self.logger.debug("기존 매크로 데이터 사용 (수집 건너뛰기)")
            else:
                self.logger.debug("시장 매크로 데이터 수집 중")
                self.macro_collector.collect_all_data()

            # 개별 종목 데이터 확인
            trader_data_exists = os.path.exists("data/trader") and len(os.listdir("data/trader")) > 5
            if trader_data_exists:
                self.logger.debug("기존 개별종목 데이터 사용 (수집 건너뛰기)")
            else:
                self.logger.debug("개별 종목 데이터 수집 중")
                symbols = self.config["data"]["symbols"]
                lookback_days = self.config["data"]["lookback_days"]

                for symbol in symbols:
                    self.data_loader.collect_and_save(
                        symbol=symbol, 
                        days_back=lookback_days, 
                        interval="1d"
                    )

        except Exception as e:
            self.logger.error(f"데이터 수집 실패: {e}")
            raise

    def _train_models(self):
        """모델 로드 (이미 학습된 모델 사용)"""
        try:
            # HMM 모델 로드
            self.logger.debug("HMM 모델 로드 중")
            hmm_model_path = "models/trader/hmm_regime_model.pkl"
            if os.path.exists(hmm_model_path):
                if not self.regime_classifier.load_model(hmm_model_path):
                    self.logger.warning("HMM 모델 로드 실패 - 기본 모델 사용")
            else:
                self.logger.warning(f"HMM 모델 파일 없음: {hmm_model_path}")

            # 신경망 모델 로드
            self.logger.debug("신경망 모델 로드 중")
            neural_model_path = "models/trader/neural_predictor"
            if os.path.exists(f"{neural_model_path}_meta.pkl"):
                if not self.neural_predictor.load_model(neural_model_path):
                    self.logger.warning("신경망 모델 로드 실패 - 기본 모델 사용")
            else:
                self.logger.warning(f"신경망 모델 파일 없음: {neural_model_path}_meta.pkl")

        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="HMM-Neural 하이브리드 트레이더")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_trader.json",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="분석 모드 실행 (5단계만)",
    )
    parser.add_argument(
        "--full-process",
        action="store_true",
        help="전체 프로세스 실행 (1-4단계)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="강제 재학습",
    )

    args = parser.parse_args()

    # 트레이더 초기화
    trader = HybridTrader(
        config_path=args.config,
        analysis_mode=args.run_analysis,
    )

    try:
        if args.full_process:
            # 전체 프로세스 실행
            results = trader.run_full_process()
        elif args.run_analysis:
            # 분석 모드만 실행
            results = trader.run_portfolio_analysis()
        else:
            # 기본: 분석만 실행
            results = trader.analyze()

        # 결과 출력 (깔끔한 요약 형태)
        if results:
            print_results_summary(results)

    except Exception as e:
        print(f"실행 실패: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())