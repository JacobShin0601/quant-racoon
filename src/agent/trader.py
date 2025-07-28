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
        self.logger = get_logger(
            "trader",
            config_path=self.config_path,
            time_horizon="trader"
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
            if not self.regime_classifier.load_model():
                self.logger.warning("HMM 모델 로드 실패 - 학습 필요")
                return False

            # 2. 신경망 모델 로드
            if not self.neural_predictor.load_model():
                self.logger.warning("신경망 모델 로드 실패 - 학습 필요")
                return False

            # 3. 투자 점수 생성기 초기화
            self.score_generator.initialize()

            # 4. 매매 신호 생성기 초기화
            self.signal_generator.initialize()

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
                    self.logger.data_info("기존 데이터 활용")
                    return last_results

            results = {}

            # 1. 시장 체제 분류
            self.logger.step("[1/4] 시장 체제 분류")
            current_regime = self.regime_classifier.get_current_regime()
            regime_confidence = self.regime_classifier.get_regime_confidence()
            transition_prob = self.regime_analyzer.get_transition_probability()

            results["market_regime"] = {
                "current": current_regime,
                "confidence": regime_confidence,
                "transition_probability": transition_prob,
            }

            self.logger.info(f"현재 시장 체제: {current_regime}")

            # 2. 개별 종목 예측
            self.logger.step("[2/4] 개별 종목 예측")
            symbols = self.config["data"]["symbols"]
            predictions = {}

            for symbol in symbols:
                pred = self.neural_predictor.predict(symbol)
                predictions[symbol] = pred

            results["predictions"] = predictions

            # 3. 투자 점수 생성
            self.logger.step("[3/4] 투자 점수 생성")
            scores = {}
            for symbol in symbols:
                score = self.score_generator.generate_score(
                    symbol, predictions[symbol], current_regime
                )
                scores[symbol] = score

            results["investment_scores"] = scores

            # 4. 매매 신호 생성
            self.logger.step("[4/4] 매매 신호 생성")
            signals = {}
            for symbol in symbols:
                signal = self.signal_generator.generate_signal(
                    symbol, scores[symbol], current_regime
                )
                signals[symbol] = signal

            results["trading_signals"] = signals

            # 5. 포트폴리오 종합
            portfolio_summary = self.portfolio_aggregator.aggregate(
                signals, scores, current_regime
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
            self.logger.portfolio_info("포트폴리오 최적화 실행 중")
            try:
                # 포트폴리오 매니저로 최적화 실행
                optimization_results = self.portfolio_manager.optimize_portfolio(
                    symbols, scores, signals
                )

                if optimization_results and "weights" in optimization_results:
                    self.logger.portfolio_info("최적 포트폴리오 비중:")
                    for symbol, weight in optimization_results["weights"].items():
                        self.logger.info(f"  {symbol}: {weight:.1%}")
                else:
                    self.logger.warning("포트폴리오 최적화 실패 - 기본 가중치 사용")
                    optimization_results = {"weights": {s: 1.0 / len(symbols) for s in symbols}}

            except Exception as e:
                self.logger.error(f"포트폴리오 최적화 중 오류: {e}")
                optimization_results = {"weights": {s: 1.0 / len(symbols) for s in symbols}}

            # 2. 백테스팅
            self.logger.info("백테스팅 실행 중...")
            try:
                backtest_results = self.evaluator.backtest_portfolio(
                    symbols, optimization_results["weights"], signals
                )

                if backtest_results:
                    self.logger.info("백테스팅 결과:")
                    metrics = backtest_results.get("metrics", {})
                    self.logger.log_metrics(metrics, "성과 지표")

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
            self.logger.info("결과 저장 중...")
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
            self.logger.info("결과 요약 레포트 생성 중...")

            # 주요 지표 추출
            metrics = results.get("backtest", {}).get("metrics", {})
            weights = results.get("optimization", {}).get("weights", {})
            signals = results.get("analysis_results", {}).get("trading_signals", {})

            self.logger.info("결과 요약:")
            self.logger.info(f"- 총 수익률: {metrics.get('total_return', 0):.2%}")
            self.logger.info(f"- 샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"- 최대 낙폭: {metrics.get('max_drawdown', 0):.2%}")

            # 포트폴리오 비중
            self.logger.info("\n포트폴리오 비중:")
            for symbol, weight in weights.items():
                self.logger.info(f"- {symbol}: {weight:.1%}")

            # 추천 행동
            self.logger.info("\n추천 행동:")
            for symbol, signal in signals.items():
                action = signal.get("action", "HOLD")
                confidence = signal.get("confidence", 0)
                self.logger.info(f"- {symbol}: {action} (신뢰도: {confidence:.1%})")

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
        """데이터 수집"""
        try:
            # 매크로 데이터 수집
            self.logger.data_info("시장 매크로 데이터 수집 중")
            self.macro_collector.collect_all_data()

            # 개별 종목 데이터 수집
            self.logger.data_info("개별 종목 데이터 수집 중")
            symbols = self.config["data"]["symbols"]
            lookback_days = self.config["data"]["lookback_days"]

            for symbol in symbols:
                self.data_loader.download_data(
                    symbol, period_days=lookback_days, interval="1d"
                )

        except Exception as e:
            self.logger.error(f"데이터 수집 실패: {e}")
            raise

    def _train_models(self):
        """모델 학습"""
        try:
            # HMM 학습
            self.logger.model_info("HMM 모델 학습 중")
            self.regime_classifier.train()

            # 신경망 학습
            self.logger.model_info("신경망 학습 중")
            self.neural_predictor.train()

        except Exception as e:
            self.logger.error(f"모델 학습 실패: {e}")
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

        # 결과 출력
        if results:
            print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    except Exception as e:
        print(f"실행 실패: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())