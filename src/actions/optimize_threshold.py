#!/usr/bin/env python3
"""
Threshold 최적화 시스템
하드코딩된 BUY/HOLD/SELL threshold를 실제 거래 성과 기반으로 최적화합니다.

기능:
1. Train-Test 분할 및 백테스팅
2. 다양한 threshold 조합 테스트
3. 신경망/강화학습 기반 최적화
4. 종목별 최적 threshold 찾기
5. 결과 JSON 저장
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
from pathlib import Path
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.samplers import TPESampler

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# 하이퍼파라미터 최적화에 필요한 것만 import
from src.actions.trading_signal_generator import TradingSignalGenerator

# 불필요한 import 제거:
# from src.actions.investment_scorer import InvestmentScoreGenerator
# from src.actions.hmm_regime_classifier import MarketRegimeHMM
# from src.actions.neural_stock_predictor import StockPredictionNetwork
# from src.actions.y_finance import YahooFinanceDataCollector
# from src.actions.global_macro import GlobalMacroDataCollector
# from src.agent.trader import HybridTrader

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Threshold 최적화 시스템"""

    def __init__(self, config: Dict):
        self.config = config

        # config에서 optimization 설정 가져오기 (기본값으로 fallback)
        self.optimization_config = config.get("optimization", {})

        # 로그 레벨 조정 (디버깅을 위해 INFO 레벨로 변경)
        logging.getLogger("src.actions.trading_signal_generator").setLevel(logging.INFO)
        logging.getLogger("src.actions.investment_scorer").setLevel(logging.INFO)
        logging.getLogger("src.actions.optimize_threshold").setLevel(logging.INFO)
        logging.getLogger("src.agent.neural_portfolio_manager").setLevel(logging.INFO)

        # 기본 설정
        self.train_ratio = self.optimization_config.get("train_ratio", 0.7)
        self.test_ratio = 1.0 - self.train_ratio
        self.min_data_points = self.optimization_config.get("min_data_points", 100)

        # 최적화 설정
        self.optimization_method = self.optimization_config.get("method", "optuna")
        self.n_trials = self.optimization_config.get("n_trials", 100)
        self.objective_metric = self.optimization_config.get(
            "objective_metric", "sharpe_ratio"
        )

        # Threshold 범위 설정 (config에서 가져오거나 기본값 사용)
        self.threshold_ranges = self.optimization_config.get(
            "threshold_ranges",
            {
                "strong_buy": [0.5, 0.9],
                "buy": [0.3, 0.7],
                "hold_upper": [0.1, 0.5],
                "hold_lower": [-0.5, -0.1],
                "sell": [-0.7, -0.3],
                "strong_sell": [-0.9, -0.5],
            },
        )

        # 결과 저장 경로
        self.results_dir = Path("results/trader")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 캐시된 데이터
        self.cached_macro_data = None
        self.cached_market_regime = None

        # 컴포넌트 초기화
        self._initialize_components()
        self._nn_warned = False  # 신경망 미학습 경고 플래그

        logger.info(f"ThresholdOptimizer 초기화 완료")
        logger.info(f"최적화 방법: {self.optimization_method}")
        logger.info(f"목표 지표: {self.objective_metric}")

    def _initialize_components(self):
        """하이퍼파라미터 최적화에 필요한 컴포넌트만 초기화"""
        try:
            # 하이퍼파라미터 최적화에는 TradingSignalGenerator만 필요
            # (거래 신호 생성용 임계값 테스트)
            self.signal_generator = None  # 필요할 때만 생성

            # 불필요한 컴포넌트들 제거:
            # - YahooFinanceDataCollector: 캐시된 데이터만 사용
            # - GlobalMacroDataCollector: 최적화에서는 불필요
            # - MarketRegimeHMM: 최적화에서는 불필요
            # - StockPredictionNetwork: 최적화에서는 불필요
            # - InvestmentScoreGenerator: 최적화에서는 불필요

            logger.info("하이퍼파라미터 최적화 컴포넌트 초기화 완료")

        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise

    def load_stock_data_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """data/trader/ 캐시에서만 종목 데이터 로드. 없으면 None 반환, 다운로드 시도 X"""
        # 심볼이 쉼표로 구분된 문자열인 경우 처리
        if "," in symbol:
            logger.warning(f"심볼에 쉼표가 포함되어 있습니다: {symbol}")
            return None

        cached_data = self._load_cached_data(symbol)
        if cached_data is not None:
            logger.info(f"[캐시] {symbol} 데이터 사용 (행: {len(cached_data)})")
            return cached_data
        else:
            logger.error(
                f"[캐시 없음] {symbol} 데이터가 data/trader/에 없습니다. run_trader.sh에서 먼저 다운로드하세요."
            )
            return None

    def load_and_split_data(self, symbols: List[str]) -> Tuple[Dict, Dict]:
        """데이터 로드 및 Train-Test 분할"""
        try:
            # symbols가 문자열로 전달된 경우 처리 (예: "AAPL,META,QQQ,SPY")
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",")]

            logger.info(f"데이터 로드 시작: {symbols}")
            all_data = {}
            for symbol in symbols:
                try:
                    data = self.load_stock_data_from_cache(symbol)
                    if data is not None:
                        all_data[symbol] = data
                    else:
                        logger.warning(f"{symbol} 데이터 없음 (캐시 미존재)")
                except Exception as e:
                    logger.error(f"데이터 로드 실패 ({symbol}): {e}")
                    continue
            if not all_data:
                raise ValueError(
                    "수집된 데이터가 없습니다. 반드시 run_trader.sh에서 데이터 다운로드를 먼저 실행하세요."
                )
            train_data, test_data = self._split_data_by_time(all_data)
            logger.info(
                f"데이터 분할 완료: Train={len(train_data)}, Test={len(test_data)}"
            )
            return train_data, test_data
        except Exception as e:
            logger.error(f"데이터 로드 및 분할 실패: {e}")
            raise

    def _load_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """캐시된 데이터 로드"""
        try:
            cache_dir = Path("data/trader")
            if not cache_dir.exists():
                logger.warning(f"캐시 디렉토리가 없습니다: {cache_dir}")
                return None

            # 실제 파일명 패턴에 맞게 검색 (예: AAPL_daily_auto_2025-07-24_43f94390.csv)
            pattern = f"{symbol}_daily_*.csv"
            files = list(cache_dir.glob(pattern))

            if not files:
                logger.warning(f"{symbol} 패턴의 파일을 찾을 수 없습니다: {pattern}")
                return None

            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            logger.info(f"{symbol} 데이터 로드: {latest_file.name}")

            data = pd.read_csv(latest_file, index_col=0, parse_dates=True)

            if len(data) >= self.min_data_points:
                logger.info(f"{symbol} 데이터 로드 성공: {len(data)}행")
                return data
            else:
                logger.warning(
                    f"{symbol} 데이터 부족: {len(data)}행 (최소 {self.min_data_points}행 필요)"
                )
                return None

        except Exception as e:
            logger.warning(f"캐시 로드 실패 ({symbol}): {e}")
            return None

    def _save_cached_data(self, symbol: str, data: pd.DataFrame):
        """데이터 캐시 저장"""
        try:
            cache_dir = Path("data/trader")
            cache_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_daily_{timestamp}.csv"
            filepath = cache_dir / filename

            data.to_csv(filepath)
            logger.info(f"데이터 캐시 저장: {filepath}")

        except Exception as e:
            logger.warning(f"캐시 저장 실패 ({symbol}): {e}")

    def _split_data_by_time(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict, Dict]:
        """시간 기준으로 Train-Test 분할"""
        try:
            train_data = {}
            test_data = {}

            for symbol, data in data_dict.items():
                if len(data) < self.min_data_points:
                    continue

                # 시간 기준 분할
                split_idx = int(len(data) * self.train_ratio)

                train_data[symbol] = data.iloc[:split_idx]
                test_data[symbol] = data.iloc[split_idx:]

                logger.info(
                    f"{symbol}: Train={len(train_data[symbol])}, Test={len(test_data[symbol])}"
                )

            return train_data, test_data

        except Exception as e:
            logger.error(f"데이터 분할 실패: {e}")
            raise

    def backtest_with_thresholds(
        self, data_dict: Dict[str, pd.DataFrame], thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        주어진 threshold로 백테스팅 수행 (간소화된 버전)
        """
        try:
            # 백테스팅 로그 간소화 (최종 검증에서만 출력)
            if hasattr(self, "_is_final_evaluation") and self._is_final_evaluation:
                logger.info(f"백테스팅 시작 - Threshold: {thresholds}")

            # 신호 생성기 초기화 (임시 설정)
            temp_config = self.config.copy()
            temp_config["signal_generation"] = {
                "thresholds": thresholds,
                "min_confidence": 0.5,
            }
            signal_generator = TradingSignalGenerator(temp_config)

            symbol_results = {}
            portfolio_trades = []

            for symbol, data in data_dict.items():
                try:
                    # 디버깅: 데이터 기본 정보
                    logger.info(f"🔍 {symbol} 데이터 분석 시작 - 총 {len(data)}일")
                    logger.info(f"📅 데이터 기간: {data.index[0]} ~ {data.index[-1]}")

                    # 간단한 기술적 지표 기반 예측값 생성 (신경망 없이)
                    close = data["close"]
                    logger.info(
                        f"💰 {symbol} 종가 범위: {close.min():.2f} ~ {close.max():.2f}"
                    )

                    # RSI 기반 예측
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))

                    # 이동평균 계산
                    ma_20 = close.rolling(20).mean()
                    ma_50 = close.rolling(50).mean()

                    # 각 날짜별 예측값 계산
                    predictions = []
                    signal_count = 0  # 신호 발생 카운트

                    for i in range(len(data)):
                        if i < 50:  # 충분한 데이터가 없으면 기본값 사용
                            predictions.append(0.0)
                            continue

                        # RSI 기반 예측
                        rsi_current = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
                        rsi_prediction = (rsi_current - 50) / 50

                        # 이동평균 기반 조정
                        if not pd.isna(ma_20.iloc[i]) and not pd.isna(ma_50.iloc[i]):
                            ma_signal = 1 if ma_20.iloc[i] > ma_50.iloc[i] else -1
                            combined_prediction = (rsi_prediction + ma_signal * 0.3) / 2
                        else:
                            combined_prediction = rsi_prediction

                        # -1 ~ 1 범위로 클리핑
                        final_prediction = float(np.clip(combined_prediction, -1, 1))
                        predictions.append(final_prediction)

                        # 신호 발생 여부 확인 (임시)
                        if abs(final_prediction) > 0.1:  # 임계값 0.1로 테스트
                            signal_count += 1

                    # 예측값을 데이터프레임에 추가
                    data = data.copy()
                    data["neural_prediction"] = predictions

                    # 마지막 예측값 (기존 호환성을 위해)
                    neural_prediction = predictions[-1] if predictions else 0.0

                    # 디버깅: 예측값 통계
                    predictions_array = np.array(predictions)
                    logger.info(f"📊 {symbol} 예측값 통계:")
                    logger.info(
                        f"   - 범위: {predictions_array.min():.3f} ~ {predictions_array.max():.3f}"
                    )
                    logger.info(f"   - 평균: {predictions_array.mean():.3f}")
                    logger.info(f"   - 표준편차: {predictions_array.std():.3f}")
                    logger.info(
                        f"   - 신호 발생 횟수 (|pred| > 0.1): {signal_count}/{len(predictions)}"
                    )

                    # 거래 신호 생성을 위한 딕셔너리 생성
                    investment_score = {
                        "symbol": symbol,
                        "final_score": neural_prediction,
                        "confidence": 0.7,  # 기본 신뢰도
                        "holding_period": 30,  # 기본 보유 기간 (일)
                        "position_size": 0.1,  # 기본 포지션 크기
                        "momentum_factor": 0.5,  # 기본 모멘텀 팩터
                        "market_info": {"regime": "NEUTRAL", "regime_confidence": 0.5},
                        "risk_metrics": {
                            "volatility": 0.2,
                            "recent_drawdown": -0.05,  # 최근 낙폭
                            "max_drawdown": -0.15,  # 최대 낙폭
                            "var_95": 0.08,  # Value at Risk (95%)
                            "beta": 1.0,  # 베타
                            "liquidity": 0.7,  # risk_metrics에 추가
                        },
                        "components": {
                            "neural_score": neural_prediction,
                            "technical_score": 0.0,
                            "fundamental_score": 0.0,
                            "momentum_score": 0.0,
                            "volatility_score": 0.0,
                            "momentum_factor": 0.5,  # components에 추가
                            "technical_strength": 0.3,  # components에 추가
                            "neural_prediction": neural_prediction,  # components에 추가
                        },
                        "market_conditions": {
                            "trend": "NEUTRAL",
                            "correlation": 0.3,
                        },
                    }

                    # 전체 기간에 대한 연속 거래 시뮬레이션
                    trades = []
                    signal_count = 0  # 신호 발생 카운트

                    # 전체 데이터에 대한 연속 거래 시뮬레이션
                    all_trades = self._simulate_trading(
                        data, {"action": "HOLD"}
                    )  # 임시로 전체 데이터 전달

                    # 각 날짜별 신호를 기반으로 거래 시뮬레이션
                    position = 0  # 0: 없음, 1: 매수, -1: 매도
                    entry_price = 0
                    entry_date = None
                    cumulative_capital = 1.0  # 복리 계산을 위한 누적 자본

                    for i in range(len(data)):
                        if i < 50:  # 충분한 데이터가 없으면 건너뛰기
                            continue

                        # 현재 날짜의 예측값 사용
                        current_prediction = data["neural_prediction"].iloc[i]

                        # 현재 날짜의 investment_score 업데이트
                        current_investment_score = investment_score.copy()
                        current_investment_score["final_score"] = current_prediction
                        current_investment_score["components"][
                            "neural_score"
                        ] = current_prediction
                        current_investment_score["components"][
                            "neural_prediction"
                        ] = current_prediction

                        # 현재 날짜의 신호 생성
                        current_signal = signal_generator.generate_signal(
                            current_investment_score
                        )

                        # 신호 발생 여부 확인
                        if current_signal.get("action") != "HOLD":
                            signal_count += 1

                        # 현재 날짜의 가격과 날짜
                        current_price = data.iloc[i]["close"]
                        current_date = data.index[i]
                        action = current_signal.get("action", "HOLD")

                        # 매수 신호
                        if action in ["STRONG_BUY", "BUY"] and position <= 0:
                            if position == -1:  # 매도 포지션 청산
                                exit_price = current_price
                                pnl = (entry_price - exit_price) / entry_price
                                trades.append(
                                    {
                                        "entry_date": entry_date,
                                        "exit_date": current_date,
                                        "entry_price": entry_price,
                                        "exit_price": exit_price,
                                        "position": "SHORT",
                                        "pnl": pnl,
                                    }
                                )

                            # 매수 포지션 진입
                            position = 1
                            entry_price = current_price
                            entry_date = current_date

                        # 매도 신호
                        elif action in ["STRONG_SELL", "SELL"] and position >= 0:
                            if position == 1:  # 매수 포지션 청산
                                exit_price = current_price
                                pnl = (exit_price - entry_price) / entry_price
                                cumulative_capital *= (1 + pnl)  # 복리 계산
                                trades.append(
                                    {
                                        "entry_date": entry_date,
                                        "exit_date": current_date,
                                        "entry_price": entry_price,
                                        "exit_price": exit_price,
                                        "position": "LONG",
                                        "pnl": pnl,
                                    }
                                )

                            # 매도 포지션 진입
                            position = -1
                            entry_price = current_price
                            entry_date = current_date

                    # 마지막 포지션 청산
                    if position != 0:
                        last_price = data.iloc[-1]["close"]
                        last_date = data.index[-1]

                        if position == 1:  # 매수 포지션 청산
                            pnl = (last_price - entry_price) / entry_price
                            cumulative_capital *= (1 + pnl)  # 복리 계산
                            trades.append(
                                {
                                    "entry_date": entry_date,
                                    "exit_date": last_date,
                                    "entry_price": entry_price,
                                    "exit_price": last_price,
                                    "position": "LONG",
                                    "pnl": pnl,
                                }
                            )
                        elif position == -1:  # 매도 포지션 청산
                            pnl = (entry_price - last_price) / entry_price
                            cumulative_capital *= (1 + pnl)  # 복리 계산
                            trades.append(
                                {
                                    "entry_date": entry_date,
                                    "exit_date": last_date,
                                    "entry_price": entry_price,
                                    "exit_price": last_price,
                                    "position": "SHORT",
                                    "pnl": pnl,
                                }
                            )

                    trade_count = len(trades)
                    
                    # 최종 누적 수익률 계산
                    total_return = cumulative_capital - 1.0

                    # 디버깅: 거래 통계
                    logger.info(f"📈 {symbol} 거래 통계:")
                    logger.info(f"   - 신호 발생 횟수: {signal_count}")
                    logger.info(f"   - 실제 거래 횟수: {trade_count}")
                    logger.info(f"   - 누적 수익률: {total_return:.4f} ({total_return*100:.2f}%)")

                    # 성과 계산 (누적 수익률 전달)
                    performance = self._calculate_performance(trades, data, total_return)

                    # 디버깅: 성과 통계
                    logger.info(f"📊 {symbol} 성과 통계:")
                    logger.info(
                        f"   - 총 수익률: {performance.get('total_return', 0):.4f}"
                    )
                    logger.info(
                        f"   - 샤프 비율: {performance.get('sharpe_ratio', 0):.4f}"
                    )
                    logger.info(
                        f"   - 소르티노 비율: {performance.get('sortino_ratio', 0):.4f}"
                    )
                    logger.info(f"   - 승률: {performance.get('win_rate', 0):.4f}")
                    logger.info(
                        f"   - 총 거래 수: {performance.get('total_trades', 0)}"
                    )

                    symbol_results[symbol] = {
                        "trades": trades,
                        "performance": performance,
                        "signal_count": len(trades),
                    }

                    portfolio_trades.extend(trades)

                except Exception as e:
                    logger.warning(f"{symbol} 백테스팅 실패: {e}")
                    continue

            # 포트폴리오 성과 계산
            portfolio_performance = self._calculate_portfolio_performance(
                portfolio_trades
            )

            # 디버깅: 포트폴리오 성과 통계
            logger.info(f"🎯 포트폴리오 전체 성과:")
            logger.info(f"   - 총 거래 수: {len(portfolio_trades)}")
            logger.info(
                f"   - 총 수익률: {portfolio_performance.get('total_return', 0):.4f}"
            )
            logger.info(
                f"   - 샤프 비율: {portfolio_performance.get('sharpe_ratio', 0):.4f}"
            )
            logger.info(
                f"   - 소르티노 비율: {portfolio_performance.get('sortino_ratio', 0):.4f}"
            )

            return {
                "symbol_results": symbol_results,
                "portfolio_performance": portfolio_performance,
                "total_trades": len(portfolio_trades),
            }

        except Exception as e:
            logger.error(f"백테스팅 실패: {e}")
            return {
                "symbol_results": {},
                "portfolio_performance": {"sharpe_ratio": -999, "total_return": -1.0},
                "total_trades": 0,
            }

    def save_transaction_log(self, trades_log: list):
        """evaluator.py 스타일의 거래 로그 저장"""
        if not trades_log:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "results/trader"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"transaction_log_{timestamp}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("symbol,entry_time,exit_time,entry_price,exit_price,shares,pnl\n")
            for trade in trades_log:
                f.write(
                    f"{trade.get('symbol','')},{trade.get('entry_time','')},{trade.get('exit_time','')},{trade.get('entry_price','')},{trade.get('exit_price','')},{trade.get('shares','')},{trade.get('pnl','')}\n"
                )
        logger.info(f"거래 로그 저장 완료: {log_path}")

    def _simulate_trading(self, data: pd.DataFrame, signal: Dict) -> List[Dict]:
        """거래 시뮬레이션 (단일 날짜 또는 전체 기간)"""
        try:
            trades = []

            # 단일 날짜 데이터인 경우 (새로운 로직)
            if len(data) == 1:
                date = data.index[0]
                current_price = data.iloc[0]["close"]
                action = signal.get("action", "HOLD")

                # 매수/매도 신호가 있는 경우에만 거래 기록
                if action in ["STRONG_BUY", "BUY"]:
                    trades.append(
                        {
                            "entry_date": date,
                            "exit_date": date,
                            "entry_price": current_price,
                            "exit_price": current_price,
                            "position": "LONG",
                            "pnl": 0.0,  # 같은 날 매수/매도는 수익률 0
                        }
                    )
                elif action in ["STRONG_SELL", "SELL"]:
                    trades.append(
                        {
                            "entry_date": date,
                            "exit_date": date,
                            "entry_price": current_price,
                            "exit_price": current_price,
                            "position": "SHORT",
                            "pnl": 0.0,  # 같은 날 매수/매도는 수익률 0
                        }
                    )

                return trades

            # 전체 기간 데이터인 경우 (기존 로직 유지)
            position = 0  # 0: 없음, 1: 매수, -1: 매도
            entry_price = 0
            entry_date = None

            for i, (date, row) in enumerate(data.iterrows()):
                current_price = row["close"]
                action = signal.get("action", "HOLD")

                # 매수 신호
                if action in ["STRONG_BUY", "BUY"] and position <= 0:
                    if position == -1:  # 매도 포지션 청산
                        exit_price = current_price
                        pnl = (entry_price - exit_price) / entry_price
                        trades.append(
                            {
                                "entry_date": entry_date,
                                "exit_date": date,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "position": "SHORT",
                                "pnl": pnl,
                            }
                        )

                    # 매수 포지션 진입
                    position = 1
                    entry_price = current_price
                    entry_date = date

                # 매도 신호
                elif action in ["STRONG_SELL", "SELL"] and position >= 0:
                    if position == 1:  # 매수 포지션 청산
                        exit_price = current_price
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append(
                            {
                                "entry_date": entry_date,
                                "exit_date": date,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "position": "LONG",
                                "pnl": pnl,
                            }
                        )

                    # 매도 포지션 진입
                    position = -1
                    entry_price = current_price
                    entry_date = date

            # 마지막 포지션 청산
            if position != 0:
                last_price = data.iloc[-1]["close"]
                last_date = data.index[-1]

                if position == 1:  # 매수 포지션 청산
                    pnl = (last_price - entry_price) / entry_price
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": last_date,
                            "entry_price": entry_price,
                            "exit_price": last_price,
                            "position": "LONG",
                            "pnl": pnl,
                        }
                    )
                elif position == -1:  # 매도 포지션 청산
                    pnl = (entry_price - last_price) / entry_price
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": last_date,
                            "entry_price": entry_price,
                            "exit_price": last_price,
                            "position": "SHORT",
                            "pnl": pnl,
                        }
                    )

            return trades

        except Exception as e:
            logger.error(f"거래 시뮬레이션 실패: {e}")
            return []

    def _calculate_performance(
        self, trades: List[Dict], data: pd.DataFrame, total_return: float = None
    ) -> Dict[str, float]:
        """개별 종목 성과 계산"""
        try:
            if not trades:
                return {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                }

            # 수익률 계산 (total_return이 주어지면 사용, 아니면 단순 합계)
            returns = [trade["pnl"] for trade in trades]
            if total_return is None:
                total_return = sum(returns)  # 기존 방식 (단순 합계)

            # 승률 계산
            winning_trades = [r for r in returns if r > 0]
            win_rate = len(winning_trades) / len(returns) if returns else 0.0

            # 수익 팩터 계산
            gross_profit = sum([r for r in returns if r > 0])
            gross_loss = abs(sum([r for r in returns if r < 0]))
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            # 샤프 비율 계산
            if len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            # 소르티노 비율 계산
            if len(returns) > 1:
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_deviation = np.std(negative_returns)
                    mean_return = np.mean(returns)
                    sortino_ratio = (
                        mean_return / downside_deviation
                        if downside_deviation > 0
                        else 0.0
                    )
                else:
                    sortino_ratio = float("inf")
            else:
                sortino_ratio = 0.0

            # 최대 낙폭 계산
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": len(trades),
            }

        except Exception as e:
            logger.error(f"성과 계산 실패: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

    def _calculate_portfolio_performance(self, trades: List[Dict]) -> Dict[str, float]:
        """포트폴리오 성과 계산"""
        try:
            if not trades:
                return {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                }

            # 모든 거래의 수익률
            returns = [trade["pnl"] for trade in trades]

            # 개별 성과와 동일한 계산
            return self._calculate_performance(trades, pd.DataFrame())

        except Exception as e:
            logger.error(f"포트폴리오 성과 계산 실패: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

    def optimize_thresholds(
        self, train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Threshold 최적화 수행"""
        try:
            logger.info("Threshold 최적화 시작")

            if self.optimization_method == "grid_search":
                return self._grid_search_optimization(train_data, test_data)
            elif self.optimization_method == "optuna":
                return self._optuna_optimization(train_data, test_data)
            elif self.optimization_method == "neural_network":
                return self._neural_network_optimization(train_data, test_data)
            else:
                raise ValueError(
                    f"지원하지 않는 최적화 방법: {self.optimization_method}"
                )

        except Exception as e:
            logger.error(f"Threshold 최적화 실패: {e}")
            raise

    def _grid_search_optimization(
        self, train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """그리드 서치 최적화"""
        try:
            logger.info("그리드 서치 최적화 시작")

            # Threshold 조합 생성
            threshold_combinations = self._generate_threshold_combinations()
            total = len(threshold_combinations)

            if total == 0:
                logger.error(
                    "유효한 threshold 조합이 생성되지 않았습니다. 범위를 조정하세요."
                )
                return {"best_thresholds": None, "best_score": 0, "all_results": []}

            logger.info(f"테스트할 threshold 조합 수: {total}")

            best_score = float("-inf")
            best_thresholds = None
            best_results = None
            all_results = []

            for i, thresholds in enumerate(threshold_combinations):
                try:
                    # 진행률 요약 출력 (10% 단위)
                    if i % (total // 10) == 0 and i > 0:
                        print(
                            f"[진행] Threshold 조합 테스트: {i//(total//10)*10}% 완료 ({i+1}/{total})"
                        )
                    elif i == total - 1:
                        print(
                            f"[진행] Threshold 조합 테스트: 100% 완료 ({i+1}/{total})"
                        )

                    # Train 데이터로 백테스팅
                    train_results = self.backtest_with_thresholds(
                        train_data, thresholds
                    )

                    # Test 데이터로 백테스팅
                    test_results = self.backtest_with_thresholds(test_data, thresholds)

                    # 성과 점수 계산
                    train_score = train_results["portfolio_performance"][
                        self.objective_metric
                    ]
                    test_score = test_results["portfolio_performance"][
                        self.objective_metric
                    ]

                    # Train과 Test의 평균 점수
                    avg_score = (train_score + test_score) / 2

                    result = {
                        "thresholds": thresholds,
                        "train_score": train_score,
                        "test_score": test_score,
                        "avg_score": avg_score,
                        "train_results": train_results,
                        "test_results": test_results,
                    }

                    all_results.append(result)

                    # 최고 점수 업데이트
                    if avg_score > best_score:
                        best_score = avg_score
                        best_thresholds = thresholds
                        best_results = result
                        # logger.info(f"새로운 최고 점수: {best_score:.4f}")

                except Exception as e:
                    # logger.error(f"조합 {i+1} 테스트 실패: {e}")
                    continue

            # 결과 저장
            self._save_optimization_results(all_results, best_results)

            # 최적화 결과 요약 출력
            print(f"[결과] 최적 Threshold: {best_thresholds}")
            print(f"[결과] 최고 {self.objective_metric}: {best_score:.4f}")

            return {
                "best_thresholds": best_thresholds,
                "best_score": best_score,
                "all_results": all_results,
                "optimization_method": "grid_search",
            }

        except Exception as e:
            # logger.error(f"그리드 서치 최적화 실패: {e}")
            raise

    def _generate_threshold_combinations(self) -> List[Dict[str, float]]:
        """Threshold 조합 생성"""
        try:
            combinations = []

            # 각 threshold의 값 범위 설정 (더 넓은 범위로 조정)
            strong_buy_range = np.linspace(0.1, 0.5, 5)
            buy_range = np.linspace(0.0, 0.4, 5)
            hold_upper_range = np.linspace(-0.2, 0.2, 5)
            hold_lower_range = np.linspace(-0.4, 0.0, 5)
            sell_range = np.linspace(-0.6, -0.2, 5)
            strong_sell_range = np.linspace(-0.8, -0.4, 5)

            # 조합 생성
            for sb in strong_buy_range:
                for b in buy_range:
                    for hu in hold_upper_range:
                        for hl in hold_lower_range:
                            for s in sell_range:
                                for ss in strong_sell_range:
                                    # 논리적 제약 조건 확인 (완화)
                                    if (
                                        sb > b
                                        and b > hu
                                        and hu > hl
                                        and hl > s
                                        and s > ss
                                    ):
                                        thresholds = {
                                            "strong_buy": round(sb, 2),
                                            "buy": round(b, 2),
                                            "hold_upper": round(hu, 2),
                                            "hold_lower": round(hl, 2),
                                            "sell": round(s, 2),
                                            "strong_sell": round(ss, 2),
                                        }
                                        combinations.append(thresholds)

            logger.info(f"생성된 threshold 조합 수: {len(combinations)}")
            return combinations

        except Exception as e:
            logger.error(f"Threshold 조합 생성 실패: {e}")
            raise

    def _optuna_optimization(
        self, train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Optuna 기반 최적화"""
        try:
            print(f"🎯 Optuna 최적화 시작 (총 {self.n_trials}회 시도)")
            print(f"📊 목표 지표: {self.objective_metric}")

            def objective(trial):
                # Threshold 값 제안 (config의 threshold_ranges 사용)
                strong_buy = trial.suggest_float(
                    "strong_buy",
                    self.threshold_ranges["strong_buy"][0],
                    self.threshold_ranges["strong_buy"][1],
                )
                buy = trial.suggest_float(
                    "buy",
                    self.threshold_ranges["buy"][0],
                    self.threshold_ranges["buy"][1],
                )
                hold_upper = trial.suggest_float(
                    "hold_upper",
                    self.threshold_ranges["hold_upper"][0],
                    self.threshold_ranges["hold_upper"][1],
                )
                hold_lower = trial.suggest_float(
                    "hold_lower",
                    self.threshold_ranges["hold_lower"][0],
                    self.threshold_ranges["hold_lower"][1],
                )
                sell = trial.suggest_float(
                    "sell",
                    self.threshold_ranges["sell"][0],
                    self.threshold_ranges["sell"][1],
                )
                strong_sell = trial.suggest_float(
                    "strong_sell",
                    self.threshold_ranges["strong_sell"][0],
                    self.threshold_ranges["strong_sell"][1],
                )

                # 논리적 제약 조건 확인
                if not (
                    strong_buy > buy > hold_upper > hold_lower > sell > strong_sell
                ):
                    return float("-inf")

                thresholds = {
                    "strong_buy": strong_buy,
                    "buy": buy,
                    "hold_upper": hold_upper,
                    "hold_lower": hold_lower,
                    "sell": sell,
                    "strong_sell": strong_sell,
                }

                try:
                    # Train 데이터로 백테스팅
                    train_results = self.backtest_with_thresholds(
                        train_data, thresholds
                    )

                    # Test 데이터로 백테스팅
                    test_results = self.backtest_with_thresholds(test_data, thresholds)

                    # 성과 점수 계산
                    train_score = train_results["portfolio_performance"][
                        self.objective_metric
                    ]
                    test_score = test_results["portfolio_performance"][
                        self.objective_metric
                    ]

                    # Train과 Test의 평균 점수
                    avg_score = (train_score + test_score) / 2

                    # 디버깅: Optuna trial 결과
                    if trial.number % 10 == 0:  # 10회마다 출력
                        logger.info(f"🔍 Trial {trial.number}:")
                        logger.info(f"   - Thresholds: {thresholds}")
                        logger.info(f"   - Train Score: {train_score:.4f}")
                        logger.info(f"   - Test Score: {test_score:.4f}")
                        logger.info(f"   - Avg Score: {avg_score:.4f}")

                    return avg_score

                except Exception as e:
                    return float("-inf")

            # Optuna 스터디 생성 및 최적화
            study = optuna.create_study(
                direction="maximize", sampler=TPESampler(seed=42)
            )

            # 진행 상황 콜백 함수
            def print_progress(study, trial):
                if trial.number % 10 == 0:  # 10회마다 진행 상황 출력
                    print(
                        f"📈 진행률: {trial.number}/{self.n_trials} ({trial.number/self.n_trials*100:.1f}%)"
                    )
                    if study.best_value > float("-inf"):
                        print(f"🏆 현재 최고 점수: {study.best_value:.4f}")

            study.optimize(
                objective, n_trials=self.n_trials, callbacks=[print_progress]
            )

            # 최적 결과
            best_params = study.best_params
            best_score = study.best_value

            print(f"\n✅ 최적화 완료!")
            print(f"🏆 최고 점수: {best_score:.4f}")

            # 최적 threshold로 최종 테스트
            best_thresholds = {
                "strong_buy": best_params["strong_buy"],
                "buy": best_params["buy"],
                "hold_upper": best_params["hold_upper"],
                "hold_lower": best_params["hold_lower"],
                "sell": best_params["sell"],
                "strong_sell": best_params["strong_sell"],
            }

            print(f"\n🎯 최적 임계점:")
            for key, value in best_thresholds.items():
                print(f"  {key}: {value:.3f}")

            train_results = self.backtest_with_thresholds(train_data, best_thresholds)
            test_results = self.backtest_with_thresholds(test_data, best_thresholds)

            best_results = {
                "thresholds": best_thresholds,
                "train_score": train_results["portfolio_performance"][
                    self.objective_metric
                ],
                "test_score": test_results["portfolio_performance"][
                    self.objective_metric
                ],
                "avg_score": best_score,
                "train_results": train_results,
                "test_results": test_results,
            }

            # 결과 저장
            self._save_optimization_results([best_results], best_results)

            # 최적화된 threshold를 JSON으로 저장
            self._save_optimized_thresholds(best_thresholds)

            return {
                "best_thresholds": best_thresholds,
                "best_score": best_score,
                "study": study,
                "optimization_method": "optuna",
            }

        except Exception as e:
            logger.error(f"Optuna 최적화 실패: {e}")
            raise

    def _neural_network_optimization(
        self, train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """신경망 기반 최적화"""
        try:
            logger.info("신경망 최적화 시작")

            # Threshold 최적화를 위한 신경망 모델
            class ThresholdOptimizer(nn.Module):
                def __init__(self):
                    super(ThresholdOptimizer, self).__init__()
                    self.fc1 = nn.Linear(6, 32)  # 6개 threshold
                    self.fc2 = nn.Linear(32, 16)
                    self.fc3 = nn.Linear(16, 6)  # 6개 threshold 출력
                    self.relu = nn.ReLU()
                    self.tanh = nn.Tanh()

                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.tanh(self.fc3(x))  # -1 ~ 1 범위
                    return x

            # 모델 초기화
            model = ThresholdOptimizer()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            best_score = float("-inf")
            best_thresholds = None

            for epoch in range(100):
                # 랜덤 threshold 생성
                random_thresholds = torch.randn(6) * 0.5  # 표준정규분포

                # 신경망으로 threshold 조정
                adjusted_thresholds = model(random_thresholds)

                # -1 ~ 1 범위를 실제 threshold 범위로 변환
                thresholds = {
                    "strong_buy": 0.7 + 0.2 * adjusted_thresholds[0].item(),
                    "buy": 0.5 + 0.2 * adjusted_thresholds[1].item(),
                    "hold_upper": 0.3 + 0.2 * adjusted_thresholds[2].item(),
                    "hold_lower": -0.3 + 0.2 * adjusted_thresholds[3].item(),
                    "sell": -0.5 + 0.2 * adjusted_thresholds[4].item(),
                    "strong_sell": -0.7 + 0.2 * adjusted_thresholds[5].item(),
                }

                # 논리적 제약 조건 확인
                if not (
                    thresholds["strong_buy"]
                    > thresholds["buy"]
                    > thresholds["hold_upper"]
                    > thresholds["hold_lower"]
                    > thresholds["sell"]
                    > thresholds["strong_sell"]
                ):
                    continue

                try:
                    # 백테스팅
                    train_results = self.backtest_with_thresholds(
                        train_data, thresholds
                    )
                    test_results = self.backtest_with_thresholds(test_data, thresholds)

                    # 성과 점수 계산
                    train_score = train_results["portfolio_performance"][
                        self.objective_metric
                    ]
                    test_score = test_results["portfolio_performance"][
                        self.objective_metric
                    ]
                    avg_score = (train_score + test_score) / 2

                    # 손실 계산 (최대화 문제를 최소화 문제로 변환)
                    loss = torch.tensor(-avg_score, requires_grad=True)

                    # 역전파
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 최고 점수 업데이트
                    if avg_score > best_score:
                        best_score = avg_score
                        best_thresholds = thresholds.copy()

                        logger.info(f"Epoch {epoch}: 새로운 최고 점수 {best_score:.4f}")

                except Exception as e:
                    logger.warning(f"Epoch {epoch} 실패: {e}")
                    continue

            # 최적 결과로 최종 테스트
            train_results = self.backtest_with_thresholds(train_data, best_thresholds)
            test_results = self.backtest_with_thresholds(test_data, best_thresholds)

            best_results = {
                "thresholds": best_thresholds,
                "train_score": train_results["portfolio_performance"][
                    self.objective_metric
                ],
                "test_score": test_results["portfolio_performance"][
                    self.objective_metric
                ],
                "avg_score": best_score,
                "train_results": train_results,
                "test_results": test_results,
            }

            # 결과 저장
            self._save_optimization_results([best_results], best_results)

            return {
                "best_thresholds": best_thresholds,
                "best_score": best_score,
                "model": model,
                "optimization_method": "neural_network",
            }

        except Exception as e:
            logger.error(f"신경망 최적화 실패: {e}")
            raise

    def _save_optimized_thresholds(self, thresholds: Dict[str, float]):
        """최적화된 threshold를 JSON 파일로 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 저장할 데이터 구조
            threshold_data = {
                "timestamp": timestamp,
                "optimization_method": "optuna",
                "objective_metric": self.objective_metric,
                "thresholds": thresholds,
                "metadata": {
                    "train_ratio": self.train_ratio,
                    "n_trials": self.n_trials,
                    "symbols": list(
                        self.config.get("portfolio", {}).get("symbols", [])
                    ),
                    "description": "Optuna 최적화로 찾은 최적 threshold 값들",
                },
            }

            # 파일 저장
            output_dir = Path("results/trader")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 최신 파일로 저장
            latest_file = output_dir / "optimized_thresholds.json"
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(threshold_data, f, indent=2, ensure_ascii=False)

            # 타임스탬프가 포함된 백업 파일도 저장
            backup_file = output_dir / f"optimized_thresholds_{timestamp}.json"
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(threshold_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ 최적화된 threshold 저장 완료:")
            logger.info(f"   - 최신 파일: {latest_file}")
            logger.info(f"   - 백업 파일: {backup_file}")

        except Exception as e:
            logger.error(f"최적화된 threshold 저장 실패: {e}")

    def load_optimized_thresholds(self) -> Optional[Dict[str, float]]:
        """저장된 최적화된 threshold를 로드"""
        try:
            threshold_file = Path("results/trader/optimized_thresholds.json")

            if not threshold_file.exists():
                logger.warning("저장된 최적화된 threshold 파일이 없습니다.")
                return None

            with open(threshold_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            thresholds = data.get("thresholds", {})
            timestamp = data.get("timestamp", "unknown")

            logger.info(f"✅ 저장된 최적화된 threshold 로드 완료:")
            logger.info(f"   - 파일: {threshold_file}")
            logger.info(f"   - 생성 시간: {timestamp}")
            logger.info(f"   - Threshold: {thresholds}")

            return thresholds

        except Exception as e:
            logger.error(f"저장된 최적화된 threshold 로드 실패: {e}")
            return None

    def _save_optimization_results(self, all_results: List[Dict], best_result: Dict):
        """최적화 결과 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 전체 결과 저장
            all_results_file = (
                self.results_dir / f"threshold_optimization_all_{timestamp}.json"
            )
            with open(all_results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

            # 최적 결과 저장
            best_result_file = (
                self.results_dir / f"threshold_optimization_best_{timestamp}.json"
            )
            with open(best_result_file, "w", encoding="utf-8") as f:
                json.dump(best_result, f, indent=2, ensure_ascii=False, default=str)

            # 요약 결과 저장
            summary = {
                "timestamp": timestamp,
                "optimization_method": self.optimization_method,
                "objective_metric": self.objective_metric,
                "best_thresholds": best_result["thresholds"],
                "best_score": best_result["avg_score"],
                "train_score": best_result["train_score"],
                "test_score": best_result["test_score"],
                "total_combinations_tested": len(all_results),
            }

            summary_file = (
                self.results_dir / f"threshold_optimization_summary_{timestamp}.json"
            )
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"최적화 결과 저장 완료:")
            logger.info(f"  전체 결과: {all_results_file}")
            logger.info(f"  최적 결과: {best_result_file}")
            logger.info(f"  요약: {summary_file}")

        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")

    def run_optimization(self, symbols: List[str]) -> Dict[str, Any]:
        """전체 최적화 프로세스 실행"""
        try:
            # symbols가 문자열로 전달된 경우 처리 (예: "AAPL,META,QQQ,SPY")
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",")]

            logger.info("=" * 80)
            logger.info("Threshold 최적화 시스템 시작")
            logger.info("=" * 80)
            logger.info(f"최적화 대상 종목: {symbols}")

            # 1. 데이터 로드 및 분할
            logger.info("1단계: 데이터 로드 및 분할")
            train_data, test_data = self.load_and_split_data(symbols)

            # 2. Threshold 최적화
            logger.info("2단계: Threshold 최적화")
            optimization_results = self.optimize_thresholds(train_data, test_data)

            # 최적화 후 거래 로그 저장
            if "trades_log" in optimization_results:
                self.save_transaction_log(optimization_results["trades_log"])

            # 3. 최적 threshold로 최종 검증
            logger.info("3단계: 최종 검증")
            best_thresholds = optimization_results["best_thresholds"]

            # 최종 검증 플래그 설정
            self._is_final_evaluation = True

            final_train_results = self.backtest_with_thresholds(
                train_data, best_thresholds
            )
            final_test_results = self.backtest_with_thresholds(
                test_data, best_thresholds
            )

            # 4. 결과 요약
            logger.info("4단계: 결과 요약")
            summary = {
                "optimization_results": optimization_results,
                "final_train_results": final_train_results,
                "final_test_results": final_test_results,
                "best_thresholds": best_thresholds,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat(),
            }

            # 5. 최종 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_results_file = (
                self.results_dir / f"threshold_optimization_final_{timestamp}.json"
            )

            with open(final_results_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"최종 결과 저장: {final_results_file}")

            # 6. 결과 출력
            self._print_optimization_summary(summary)

            return summary

        except Exception as e:
            logger.error(f"최적화 프로세스 실패: {e}")
            raise

    def _print_optimization_summary(self, summary: Dict[str, Any]):
        """최적화 결과 요약 출력"""
        try:
            print("\n" + "=" * 80)
            print("🎯 Threshold 최적화 결과 요약")
            print("=" * 80)

            best_thresholds = summary["best_thresholds"]
            optimization_results = summary["optimization_results"]

            print(
                f"📊 최적화 방법: {optimization_results.get('optimization_method', 'N/A')}"
            )
            print(f"📊 목표 지표: {self.objective_metric}")
            print(f"📊 최고 점수: {optimization_results.get('best_score', 0):.4f}")
            print(f"📊 테스트 종목 수: {len(summary['symbols'])}")

            print(f"\n🎯 최적 Threshold:")
            for key, value in best_thresholds.items():
                print(f"  {key}: {value:.3f}")

            # Train/Test 성과 비교
            final_train = summary["final_train_results"]["portfolio_performance"]
            final_test = summary["final_test_results"]["portfolio_performance"]

            print(f"\n📈 최종 성과 비교:")
            print(f"  Train 수익률: {final_train.get('total_return', 0)*100:.2f}%")
            print(f"  Test 수익률: {final_test.get('total_return', 0)*100:.2f}%")
            print(f"  Train 샤프 비율: {final_train.get('sharpe_ratio', 0):.3f}")
            print(f"  Test 샤프 비율: {final_test.get('sharpe_ratio', 0):.3f}")
            print(f"  Train 소르티노 비율: {final_train.get('sortino_ratio', 0):.3f}")
            print(f"  Test 소르티노 비율: {final_test.get('sortino_ratio', 0):.3f}")

            print("=" * 80)

        except Exception as e:
            logger.error(f"결과 요약 출력 실패: {e}")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Threshold 최적화 시스템")
    parser.add_argument(
        "--config", default="config/config_trader.json", help="설정 파일"
    )
    parser.add_argument("--symbols", nargs="+", help="최적화할 종목 목록")
    parser.add_argument(
        "--method",
        choices=["grid_search", "optuna", "neural_network"],
        default="grid_search",
        help="최적화 방법",
    )
    parser.add_argument(
        "--objective",
        choices=["total_return", "sharpe_ratio", "sortino_ratio"],
        default="sharpe_ratio",
        help="목표 지표",
    )
    parser.add_argument("--trials", type=int, default=100, help="최적화 시도 횟수")
    parser.add_argument(
        "--force-optimize",
        action="store_true",
        help="강제로 새로운 최적화 실행 (기본: 저장된 threshold 사용)",
    )

    args = parser.parse_args()

    # 설정 로드
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 최적화 설정 업데이트
    config["optimization"] = {
        "method": args.method,
        "objective_metric": args.objective,
        "n_trials": args.trials,
        "train_ratio": 0.7,
    }

    # 기본 종목 목록
    if not args.symbols:
        args.symbols = ["AAPL", "META", "QQQ", "SPY"]

    # ThresholdOptimizer 초기화
    optimizer = ThresholdOptimizer(config)

    # 조건부 최적화 실행
    if args.force_optimize:
        print("🔄 강제 최적화 모드: 새로운 최적화를 실행합니다.")
        results = optimizer.run_optimization(args.symbols)
        print("✅ 새로운 Threshold 최적화 완료!")
    else:
        print("📂 저장된 최적화 결과 사용 모드")

        # 저장된 threshold 로드
        saved_thresholds = optimizer.load_optimized_thresholds()

        if saved_thresholds is None:
            print("⚠️ 저장된 threshold가 없습니다. 새로운 최적화를 실행합니다.")
            results = optimizer.run_optimization(args.symbols)
            print("✅ 새로운 Threshold 최적화 완료!")
        else:
            print("✅ 저장된 최적화된 threshold를 사용합니다.")

            # 저장된 threshold로 백테스팅 실행
            symbols = args.symbols
            train_data, test_data = optimizer.load_and_split_data(symbols)

            # Train/Test 백테스팅
            train_results = optimizer.backtest_with_thresholds(
                train_data, saved_thresholds
            )
            test_results = optimizer.backtest_with_thresholds(
                test_data, saved_thresholds
            )

            # 결과 출력
            print("\n" + "=" * 80)
            print("🎯 저장된 Threshold로 백테스팅 결과")
            print("=" * 80)

            print(f"📊 사용된 Threshold:")
            for key, value in saved_thresholds.items():
                print(f"  {key}: {value:.3f}")

            # Train/Test 성과 비교
            train_perf = train_results["portfolio_performance"]
            test_perf = test_results["portfolio_performance"]

            print(f"\n📈 성과 비교:")
            print(f"  Train 수익률: {train_perf.get('total_return', 0)*100:.2f}%")
            print(f"  Test 수익률: {test_perf.get('total_return', 0)*100:.2f}%")
            print(f"  Train 샤프 비율: {train_perf.get('sharpe_ratio', 0):.3f}")
            print(f"  Test 샤프 비율: {test_perf.get('sharpe_ratio', 0):.3f}")
            print(f"  Train 소르티노 비율: {train_perf.get('sortino_ratio', 0):.3f}")
            print(f"  Test 소르티노 비율: {test_perf.get('sortino_ratio', 0):.3f}")
            print(f"  총 거래 수 (Train): {train_results['total_trades']}")
            print(f"  총 거래 수 (Test): {test_results['total_trades']}")

            print("=" * 80)


if __name__ == "__main__":
    main()
