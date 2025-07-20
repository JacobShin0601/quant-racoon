#!/usr/bin/env python3
"""
Agent 공통 유틸리티 및 헬퍼 함수들
"""

import os
import json
import sys
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from contextlib import contextmanager


# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PortfolioConfig:
    """포트폴리오 설정 데이터 클래스"""

    symbols: List[str]
    weight_method: str
    rebalance_period: int
    risk_free_rate: float
    target_volatility: Optional[float] = None
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class PortfolioWeights:
    """포트폴리오 비중 데이터 클래스"""

    weights: pd.DataFrame
    method: str
    calculation_date: datetime
    symbols: List[str]
    cash_weight: float
    metadata: Dict[str, Any]


@dataclass
class StrategyResult:
    """전략 결과 데이터 클래스"""

    name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sqn: float
    total_trades: int
    avg_hold_duration: float
    trades: List[Dict]
    portfolio_values: List[float]
    signals: pd.DataFrame
    risk_analysis: Optional[Dict[str, Any]] = None  # 리스크 분석 결과 추가


class Logger:
    """구조화된 로깅 클래스"""

    def __init__(self, log_dir: str = "log"):
        self.log_dir = log_dir
        self.ensure_log_dir()
        self.logger = None
        self.log_file = None
        self.summary_logger = None
        self.summary_log_file = None
        self.evaluation_results = []
        self._file_handlers = []  # 파일 핸들러 추적

    def ensure_log_dir(self):
        """로그 디렉토리 생성"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"[Logger] 로그 디렉토리 생성: {self.log_dir}")

    def set_log_dir(self, new_log_dir: str):
        """로그 디렉토리 변경"""
        self.log_dir = new_log_dir
        self.ensure_log_dir()
        print(f"[Logger] 로그 디렉토리 변경: {self.log_dir}")

    def _cleanup_handlers(self, logger_instance):
        """로거의 핸들러들을 정리하고 파일 핸들러를 닫습니다"""
        if logger_instance:
            for handler in logger_instance.handlers[:]:
                # 파일 핸들러인 경우 close() 호출
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    if handler in self._file_handlers:
                        self._file_handlers.remove(handler)
                logger_instance.removeHandler(handler)

    @contextmanager
    def _managed_file_handler(self, log_path: str, logger_instance):
        """파일 핸들러를 관리하는 context manager"""
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # 포맷터 설정
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        try:
            logger_instance.addHandler(file_handler)
            self._file_handlers.append(file_handler)
            yield file_handler
        finally:
            if file_handler in logger_instance.handlers:
                logger_instance.removeHandler(file_handler)
            file_handler.close()
            if file_handler in self._file_handlers:
                self._file_handlers.remove(file_handler)

    def setup_logger(
        self,
        strategy: str = None,
        symbols: List[str] = None,
        mode: str = "general",
        timestamp: datetime = None,
        uuid: str = None,
    ) -> str:
        """로거 설정 및 로그 파일명 생성"""
        if timestamp is None:
            timestamp = datetime.now()

        # 로그 디렉토리 확인 및 생성
        self.ensure_log_dir()

        # 로그 파일명 생성
        filename_parts = []

        if mode:
            filename_parts.append(mode)

        if strategy:
            filename_parts.append(strategy)

        if symbols:
            symbols_str = "_".join(symbols[:3])  # 최대 3개 심볼만
            if len(symbols) > 3:
                symbols_str += f"_etc{len(symbols)-3}"
            filename_parts.append(symbols_str)

        # UUID가 있으면 파일명에 포함
        if uuid:
            filename_parts.append(uuid)

        filename_parts.append(timestamp.strftime("%Y%m%d_%H%M%S"))

        filename = "_".join(filename_parts) + ".log"
        log_path = os.path.join(self.log_dir, filename)

        # 로거 설정
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.INFO)

        # 기존 핸들러 정리 (파일 핸들러 닫기 포함)
        self._cleanup_handlers(self.logger)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 포맷터
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # 콘솔 핸들러 추가
        self.logger.addHandler(console_handler)

        # 🔥 핵심 수정: 파일 핸들러를 직접 추가하고 유지
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self._file_handlers.append(file_handler)

        self.log_file = log_path
        print(f"[Logger] 로그 파일 생성: {log_path}")

        return log_path

    def setup_summary_logger(
        self, symbols: List[str] = None, timestamp: datetime = None
    ) -> str:
        """종합 요약 로거 설정"""
        if timestamp is None:
            timestamp = datetime.now()

        # 로그 디렉토리 확인 및 생성
        self.ensure_log_dir()

        # 종합 요약 로그 파일명 생성
        filename_parts = ["summary"]

        if symbols:
            symbols_str = "_".join(symbols[:3])  # 최대 3개 심볼만
            if len(symbols) > 3:
                symbols_str += f"_etc{len(symbols)-3}"
            filename_parts.append(symbols_str)

        filename_parts.append(timestamp.strftime("%Y%m%d_%H%M%S"))
        filename = "_".join(filename_parts) + ".log"
        summary_log_path = os.path.join(self.log_dir, filename)

        # 종합 요약 로거 설정
        self.summary_logger = logging.getLogger(f"summary_{filename}")
        self.summary_logger.setLevel(logging.INFO)

        # 기존 핸들러 정리 (파일 핸들러 닫기 포함)
        self._cleanup_handlers(self.summary_logger)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 포맷터
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # 콘솔 핸들러 추가
        self.summary_logger.addHandler(console_handler)

        # 🔥 핵심 수정: 파일 핸들러를 직접 추가하고 유지
        file_handler = logging.FileHandler(summary_log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.summary_logger.addHandler(file_handler)
        self._file_handlers.append(file_handler)

        self.summary_log_file = summary_log_path
        print(f"[Logger] 종합 요약 로그 파일 생성: {summary_log_path}")

        return summary_log_path

    def cleanup(self):
        """로거 정리 - 모든 핸들러를 닫습니다"""
        if self.logger:
            self._cleanup_handlers(self.logger)
        if self.summary_logger:
            self._cleanup_handlers(self.summary_logger)

        # 남은 파일 핸들러들 정리
        for handler in self._file_handlers[:]:
            try:
                handler.close()
            except:
                pass
        self._file_handlers.clear()

    def __del__(self):
        """소멸자에서도 정리 수행"""
        self.cleanup()

    def log_info(self, message: str):
        """정보 로그"""
        if self.logger:
            self.logger.info(f"ℹ️ {message}")

    def log_success(self, message: str):
        """성공 로그"""
        if self.logger:
            self.logger.info(f"✅ {message}")

    def log_warning(self, message: str):
        """경고 로그"""
        if self.logger:
            self.logger.warning(f"⚠️ {message}")

    def log_error(self, message: str):
        """에러 로그"""
        if self.logger:
            self.logger.error(f"❌ {message}")

    def log_section(self, title: str):
        """섹션 헤더 로그"""
        if self.logger:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"{title}")
            self.logger.info(f"{'='*60}")

    def log_subsection(self, title: str):
        """서브섹션 헤더 로그"""
        if self.logger:
            self.logger.info(f"\n{'-'*50}")
            self.logger.info(f"{title}")
            self.logger.info(f"{'-'*50}")

    def log_config(self, config: Dict[str, Any], title: str = "설정 정보"):
        """설정 정보 로그"""
        if self.logger:
            self.log_subsection(title)
            for key, value in config.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                elif isinstance(value, list):
                    self.logger.info(f"  {key}: {', '.join(map(str, value))}")
                else:
                    self.logger.info(f"  {key}: {value}")

    def add_evaluation_result(self, strategy_name: str, result: Dict[str, Any]):
        """평가 결과 추가 (종합 요약용)"""
        self.evaluation_results.append({"strategy": strategy_name, "result": result})

    def log_summary_section(self, title: str):
        """종합 요약 섹션 헤더"""
        if self.summary_logger:
            self.summary_logger.info(f"\n{'='*80}")
            self.summary_logger.info(f"{title}")
            self.summary_logger.info(f"{'='*80}")

    def log_summary_subsection(self, title: str):
        """종합 요약 서브섹션 헤더"""
        if self.summary_logger:
            self.summary_logger.info(f"\n{'-'*60}")
            self.summary_logger.info(f"{title}")
            self.summary_logger.info(f"{'-'*60}")

    def log_summary_info(self, message: str):
        """종합 요약 정보 로그"""
        if self.summary_logger:
            self.summary_logger.info(f"ℹ️ {message}")

    def log_summary_success(self, message: str):
        """종합 요약 성공 로그"""
        if self.summary_logger:
            self.summary_logger.info(f"✅ {message}")

    def log_summary_warning(self, message: str):
        """종합 요약 경고 로그"""
        if self.summary_logger:
            self.summary_logger.warning(f"⚠️ {message}")

    def log_summary_error(self, message: str):
        """종합 요약 에러 로그"""
        if self.summary_logger:
            self.summary_logger.error(f"❌ {message}")

    def generate_final_summary(
        self, portfolio_mode: bool = False, portfolio_method: str = "fixed"
    ):
        """최종 종합 요약 로그 생성"""
        if not self.summary_logger or not self.evaluation_results:
            return

        self.log_summary_section("🎯 전략 평가 종합 요약 리포트")

        # 분석 설정 정보
        self.log_summary_subsection("📋 분석 설정")
        self.log_summary_info(
            f"분석 모드: {'포트폴리오' if portfolio_mode else '단일 종목'}"
        )
        if portfolio_mode:
            self.log_summary_info(f"포트폴리오 방법: {portfolio_method}")

        # 전략별 성과 요약
        self.log_summary_subsection("📊 전략별 성과 요약")

        # 성과 지표 테이블 헤더
        header = f"{'전략명':<20} {'수익률':<10} {'샤프비율':<10} {'최대낙폭':<10} {'승률':<8} {'거래횟수':<8}"
        self.summary_logger.info(header)
        self.summary_logger.info("-" * 80)

        # 전략별 결과
        for eval_result in self.evaluation_results:
            strategy_name = eval_result["strategy"]
            result = eval_result["result"]

            row = (
                f"{strategy_name:<20} "
                f"{result.get('total_return', 0)*100:>8.2f}% "
                f"{result.get('sharpe_ratio', 0):>8.2f} "
                f"{result.get('max_drawdown', 0)*100:>8.2f}% "
                f"{result.get('win_rate', 0)*100:>6.1f}% "
                f"{result.get('total_trades', 0):>6d}"
            )
            self.summary_logger.info(row)

        # 최고 성과 분석
        self.log_summary_subsection("🏆 최고 성과 분석")

        # 최고 수익률
        best_return = max(
            self.evaluation_results, key=lambda x: x["result"].get("total_return", 0)
        )
        self.log_summary_success(
            f"최고 수익률: {best_return['strategy']} "
            f"({best_return['result'].get('total_return', 0)*100:.2f}%)"
        )

        # 최고 샤프 비율
        best_sharpe = max(
            self.evaluation_results, key=lambda x: x["result"].get("sharpe_ratio", 0)
        )
        self.log_summary_success(
            f"최고 샤프비율: {best_sharpe['strategy']} "
            f"({best_sharpe['result'].get('sharpe_ratio', 0):.2f})"
        )

        # 최저 수익률
        worst_return = min(
            self.evaluation_results, key=lambda x: x["result"].get("total_return", 0)
        )
        self.log_summary_warning(
            f"최저 수익률: {worst_return['strategy']} "
            f"({worst_return['result'].get('total_return', 0)*100:.2f}%)"
        )

        # 평균 성과
        avg_return = np.mean(
            [r["result"].get("total_return", 0) for r in self.evaluation_results]
        )
        avg_sharpe = np.mean(
            [r["result"].get("sharpe_ratio", 0) for r in self.evaluation_results]
        )

        self.log_summary_info(f"평균 수익률: {avg_return*100:.2f}%")
        self.log_summary_info(f"평균 샤프비율: {avg_sharpe:.2f}")

        # 성과 차이 분석
        self.log_summary_subsection("📈 성과 차이 분석")
        return_range = best_return["result"].get("total_return", 0) - worst_return[
            "result"
        ].get("total_return", 0)
        self.log_summary_info(f"최고-최저 수익률 차이: {return_range*100:.2f}%")

        if return_range > 0.05:  # 5% 이상 차이
            self.log_summary_success("✅ 전략별 성과 차이가 뚜렷함 - 전략 선택이 중요")
        elif return_range > 0.02:  # 2-5% 차이
            self.log_summary_warning("⚠️ 전략별 성과 차이가 보통 - 추가 최적화 필요")
        else:
            self.log_summary_warning(
                "⚠️ 전략별 성과 차이가 미미함 - 포트폴리오 최적화 검토 필요"
            )

        # 종료 메시지
        self.log_summary_section("🎉 평가 완료")
        self.log_summary_success(f"총 {len(self.evaluation_results)}개 전략 평가 완료")
        self.log_summary_info(f"종합 요약 로그: {self.summary_log_file}")

    def log_portfolio_weights(self, portfolio_weights: PortfolioWeights):
        """포트폴리오 비중 로그"""
        if self.logger:
            self.log_subsection("포트폴리오 비중 정보")
            self.logger.info(f"계산 일시: {portfolio_weights.calculation_date}")
            self.logger.info(f"비중 계산 방법: {portfolio_weights.method}")
            self.logger.info(f"구성 종목: {', '.join(portfolio_weights.symbols)}")
            self.logger.info(
                f"평균 현금 비중: {format_percentage(portfolio_weights.cash_weight)}"
            )

            # 개별 종목 비중
            weights = portfolio_weights.weights
            self.logger.info(f"\n개별 종목 비중:")
            for col in weights.columns:
                if col == "CASH":
                    continue
                avg_weight = weights[col].mean()
                self.logger.info(f"  {col}: {format_percentage(avg_weight)}")

    def log_strategy_result(self, result: StrategyResult):
        """전략 결과 로그"""
        if self.logger:
            self.log_subsection(f"{result.name} 전략 결과")
            self.logger.info(f"총 수익률: {format_percentage(result.total_return)}")
            self.logger.info(f"샤프 비율: {result.sharpe_ratio:.2f}")
            self.logger.info(f"최대 낙폭: {format_percentage(result.max_drawdown)}")
            self.logger.info(f"승률: {format_percentage(result.win_rate)}")
            self.logger.info(f"거래 횟수: {result.total_trades}회")
            self.logger.info(f"평균 보유기간: {result.avg_hold_duration:.1f}시간")

    def log_trade(self, trade: Dict[str, Any]):
        """개별 거래 로그"""
        if self.logger:
            self.logger.info(
                f"거래: {trade.get('action', 'N/A')} | "
                f"가격: {trade.get('price', 0):.2f} | "
                f"수량: {trade.get('quantity', 0)} | "
                f"수익: {format_percentage(trade.get('profit', 0))} | "
                f"시간: {trade.get('datetime', 'N/A')}"
            )

    def log_performance_metrics(
        self, metrics: Dict[str, Any], title: str = "성과 지표"
    ):
        """성과 지표 로그"""
        if self.logger:
            self.log_subsection(title)
            for key, value in metrics.items():
                if isinstance(value, float):
                    if "return" in key.lower() or "rate" in key.lower():
                        self.logger.info(f"  {key}: {format_percentage(value)}")
                    else:
                        self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")

    def save_json_log(self, data: Dict[str, Any], filename: str = None):
        """JSON 형태로 로그 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"log_data_{timestamp}.json"

        log_path = os.path.join(self.log_dir, filename)
        save_json_data(data, log_path, "로그 데이터")

        if self.logger:
            self.logger.info(f"JSON 로그 저장: {log_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """통합 설정 파일 로드 (agent 전용)"""
    try:
        # config_path가 절대 경로인지 확인
        if os.path.isabs(config_path):
            # 절대 경로인 경우 그대로 사용
            config_file = config_path
        else:
            # 상대 경로인 경우 프로젝트 루트 기준으로 계산
            current_dir = os.path.dirname(__file__)  # src/agent/
            project_root = os.path.dirname(
                os.path.dirname(current_dir)
            )  # 프로젝트 루트
            config_file = os.path.join(project_root, config_path)

        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[helper] 설정 파일 로드 실패, 기본값 사용: {e}")
        return {
            "trading": {"initial_capital": 100000},
            "portfolio": {
                "rebalance_period": 4,
                "weight_calculation_method": "equal_weight",
                "risk_free_rate": 0.02,
                "target_volatility": None,
                "min_weight": 0.0,
                "max_weight": 1.0,
            },
            "data": {"symbols": []},
            "backtest": {"strategies": [], "symbols": []},
            "simulation_settings": {},
        }


def ensure_dir_exists(path: str):
    """디렉토리 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[helper] 디렉토리 생성: {path}")


def parse_symbol_from_filename(filename: str) -> str:
    """파일명에서 심볼 추출 (예: TSLA_1m.csv → TSLA)"""
    return filename.split("_")[0]


def get_csv_files_from_dir(data_dir: str, symbols: List[str] = None) -> List[str]:
    """데이터 디렉토리에서 CSV 파일 목록 가져오기"""

    if not os.path.exists(data_dir):
        print(f"❌ 디렉토리가 존재하지 않음: {data_dir}")
        raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")

    # 디렉토리 내 모든 파일 확인
    try:
        all_files = os.listdir(data_dir)
        csv_files_in_dir = [f for f in all_files if f.endswith(".csv")]
    except Exception as e:
        print(f"❌ os.listdir 오류: {e}")
        raise

    if symbols is None or len(symbols) == 0:
        # symbols가 None이거나 빈 리스트면 모든 CSV 파일 반환
        return csv_files_in_dir

    csv_files = []
    for sym in symbols:
        # 심볼명으로 시작하는 파일만 찾기 (예: AAPL_daily_auto_auto_20250717_6ff345ef.csv)
        matching_files = [f for f in csv_files_in_dir if f.startswith(f"{sym}_")]
        csv_files.extend(matching_files)

    if not csv_files:
        raise FileNotFoundError("CSV 파일을 찾을 수 없습니다.")

    return csv_files


def load_and_preprocess_data(
    data_dir: str,
    symbols: List[str] = None,
    symbol_filter: str = None,
    calculate_indicators: bool = False,  # 기본값을 False로 변경 (이미 계산된 지표가 있을 수 있음)
) -> Dict[str, pd.DataFrame]:
    """데이터 로드 및 전처리 (기술적 지표 계산 포함)"""
    from actions.calculate_index import TechnicalIndicators, StrategyParams

    # CSV 파일 목록 가져오기
    csv_files = get_csv_files_from_dir(data_dir, symbols)

    data_dict = {}
    params = StrategyParams()

    for file in csv_files:
        if symbol_filter and symbol_filter not in file:
            continue

        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # 기술적 지표가 이미 있는지 확인
        existing_indicators = [
            "atr",
            "ema_short",
            "ema_long",
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
            "returns",
            "volatility",
        ]

        has_indicators = any(
            indicator in df.columns for indicator in existing_indicators
        )

        # 기술적 지표 계산 (옵션)
        if calculate_indicators and not has_indicators:
            print(f"기술적 지표 계산 중: {file}")
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(
                df, params
            )
        else:
            df_with_indicators = df

        # 파일명에서 심볼 추출
        symbol_name = parse_symbol_from_filename(file)
        data_dict[symbol_name] = df_with_indicators

    return data_dict


def validate_portfolio_weights(
    weights: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 1.0
) -> bool:
    """포트폴리오 비중 유효성 검증"""
    print(f"\n🔍 포트폴리오 비중 유효성 검증...")

    # 1. 각 시점별 총 비중이 1.0인지 확인
    row_sums = weights.sum(axis=1)
    if not (np.allclose(row_sums, 1.0, atol=1e-3)):
        print(f"❌ 일부 시점의 총 비중이 1.0이 아님 (예시: {row_sums.head()})")
        return False

    # 2. 개별 비중이 범위 내인지 확인
    for col in weights.columns:
        if col == "CASH":
            continue
        col_weights = weights[col].dropna()
        if (col_weights < min_weight).any() or (col_weights > max_weight).any():
            print(
                f"❌ {col} 비중 범위 오류: {col_weights.min():.4f} ~ {col_weights.max():.4f}"
            )
            return False

    # 3. 음수 비중이 없는지 확인
    if (weights < 0).any().any():
        print(f"❌ 음수 비중 발견")
        return False

    print(f"✅ 포트폴리오 비중 유효성 검증 통과")
    return True


def save_json_data(data: Dict[str, Any], output_path: str, description: str = "데이터"):
    """JSON 데이터 저장"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ {description}이 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"❌ {description} 저장 실패: {e}")


def load_json_data(
    input_path: str, description: str = "데이터"
) -> Optional[Dict[str, Any]]:
    """JSON 데이터 로드"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ {description}이 {input_path}에서 로드되었습니다.")
        return data
    except Exception as e:
        print(f"❌ {description} 로드 실패: {e}")
        return None


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """퍼센트 형식으로 변환"""
    return f"{value * 100:.{decimal_places}f}%"


def format_number(value: float, decimal_places: int = 4) -> str:
    """숫자 형식으로 변환"""
    return f"{value:.{decimal_places}f}"


def print_section_header(title: str, width: int = 60):
    """섹션 헤더 출력"""
    print(f"\n{'='*width}")
    print(f"{title}")
    print(f"{'='*width}")


def print_subsection_header(title: str, width: int = 50):
    """서브섹션 헤더 출력"""
    print(f"\n{'-'*width}")
    print(f"{title}")
    print(f"{'-'*width}")


# 공통 상수
DEFAULT_CONFIG_PATH = "config/config_default.json"
DEFAULT_DATA_DIR = "data"
DEFAULT_REBALANCE_PERIOD = 4
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_WEIGHT_METHOD = "equal_weight"


def load_analysis_results(
    analysis_type: str,
    symbol: str = None,
    strategy: str = None,
    timestamp: str = None,
    analysis_dir: str = "analysis",
) -> Optional[Dict[str, Any]]:
    """분석 결과 로드"""
    try:
        # 분석 타입별 경로 설정
        if analysis_type == "quant_analysis":
            base_path = os.path.join(analysis_dir, "quant_analysis")
        elif analysis_type == "fundamental_analysis":
            base_path = os.path.join(analysis_dir, "fundamental_analysis")
        elif analysis_type == "researcher_results":
            base_path = os.path.join(analysis_dir, "researcher_results")
        elif analysis_type == "strategy_optimization":
            base_path = os.path.join(analysis_dir, "strategy_optimization")
        else:
            raise ValueError(f"지원하지 않는 분석 타입: {analysis_type}")

        # 파일 패턴 생성
        if timestamp:
            pattern = f"*{timestamp}*.json"
        elif symbol and strategy:
            pattern = f"*{strategy}*{symbol}*.json"
        elif symbol:
            pattern = f"*{symbol}*.json"
        elif strategy:
            pattern = f"*{strategy}*.json"
        else:
            pattern = "*.json"

        # 파일 검색
        import glob

        files = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)

        if not files:
            print(f"⚠️ {analysis_type} 결과 파일을 찾을 수 없습니다: {pattern}")
            return None

        # 가장 최근 파일 선택
        latest_file = max(files, key=os.path.getctime)

        # JSON 파일 로드
        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        print(f"✅ 분석 결과 로드: {latest_file}")
        return results

    except Exception as e:
        print(f"❌ 분석 결과 로드 중 오류: {e}")
        return None


def save_analysis_results(
    data: Dict[str, Any],
    analysis_type: str,
    filename: str = None,
    analysis_dir: str = "analysis",
) -> str:
    """분석 결과 저장"""
    try:
        # analysis_dir이 이미 시간대별 하위폴더를 포함하고 있는지 확인
        # (예: analysis/long, analysis/swing 등)
        if os.path.basename(analysis_dir) in ["long", "swing", "scalping"]:
            # 이미 시간대별 하위폴더가 포함된 경우
            base_path = os.path.join(analysis_dir, analysis_type)
        else:
            # 기본 analysis 폴더인 경우
            base_path = os.path.join(analysis_dir, analysis_type)

        # 디렉토리 생성 (상위 폴더까지)
        os.makedirs(base_path, exist_ok=True)
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{analysis_type}_{timestamp}.json"
        filepath = os.path.join(base_path, filename)
        # JSON 파일 저장
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 분석 결과 저장: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ 분석 결과 저장 중 오류: {e}")
        return ""


def load_optimization_results(
    strategy: str,
    symbol: str = None,
    optimization_method: str = None,
    analysis_dir: str = "analysis",
) -> Optional[Dict[str, Any]]:
    """최적화 결과 로드"""
    try:
        base_path = os.path.join(analysis_dir, "researcher_results")

        # 파일 패턴 생성
        if optimization_method:
            pattern = f"*{strategy}*{symbol}*{optimization_method}*.json"
        elif symbol:
            pattern = f"*{strategy}*{symbol}*.json"
        else:
            pattern = f"*{strategy}*.json"

        # 파일 검색
        import glob

        files = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)

        if not files:
            print(f"⚠️ {strategy} 최적화 결과를 찾을 수 없습니다: {pattern}")
            return None

        # 가장 최근 파일 선택
        latest_file = max(files, key=os.path.getctime)

        # JSON 파일 로드
        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        print(f"✅ 최적화 결과 로드: {latest_file}")
        return results

    except Exception as e:
        print(f"❌ 최적화 결과 로드 중 오류: {e}")
        return None


def get_latest_analysis_file(
    analysis_type: str,
    symbol: str = None,
    strategy: str = None,
    analysis_dir: str = "analysis",
) -> Optional[str]:
    """최신 분석 파일 경로 반환"""
    try:
        # 분석 타입별 경로 설정
        if analysis_type == "quant_analysis":
            base_path = os.path.join(analysis_dir, "quant_analysis")
        elif analysis_type == "fundamental_analysis":
            base_path = os.path.join(analysis_dir, "fundamental_analysis")
        elif analysis_type == "researcher_results":
            base_path = os.path.join(analysis_dir, "researcher_results")
        elif analysis_type == "strategy_optimization":
            base_path = os.path.join(analysis_dir, "strategy_optimization")
        else:
            raise ValueError(f"지원하지 않는 분석 타입: {analysis_type}")

        # 파일 패턴 생성
        if symbol and strategy:
            pattern = f"*{strategy}*{symbol}*.json"
        elif symbol:
            pattern = f"*{symbol}*.json"
        elif strategy:
            pattern = f"*{strategy}*.json"
        else:
            pattern = "*.json"

        # 파일 검색
        import glob

        files = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)

        if not files:
            return None

        # 가장 최근 파일 반환
        return max(files, key=os.path.getctime)

    except Exception as e:
        print(f"❌ 최신 분석 파일 검색 중 오류: {e}")
        return None


def create_analysis_folder_structure(analysis_dir: str = "analysis"):
    """분석 폴더 구조 생성"""
    try:
        # 메인 분석 폴더들
        folders = [
            os.path.join(analysis_dir, "quant_analysis", "correlation"),
            os.path.join(analysis_dir, "quant_analysis", "regression"),
            os.path.join(analysis_dir, "quant_analysis", "bayesian"),
            os.path.join(analysis_dir, "quant_analysis", "summary"),
            os.path.join(analysis_dir, "researcher_results", "grid_search"),
            os.path.join(analysis_dir, "researcher_results", "bayesian_opt"),
            os.path.join(analysis_dir, "researcher_results", "genetic_alg"),
            os.path.join(analysis_dir, "researcher_results", "comparison"),
            os.path.join(analysis_dir, "strategy_optimization"),
            os.path.join(analysis_dir, "archive"),
            os.path.join(analysis_dir, "important"),
        ]

        # 전략별 폴더들
        strategies = [
            "dual_momentum",
            "volatility_breakout",
            "swing_ema",
            "swing_rsi",
            "swing_donchian",
            "stochastic",
            "williams_r",
            "cci",
            "whipsaw_prevention",
            "donchian_rsi_whipsaw",
            "volatility_filtered_breakout",
            "multi_timeframe_whipsaw",
            "adaptive_whipsaw",
            "cci_bollinger",
            "stoch_donchian",
            "vwap_macd_scalping",
            "keltner_rsi_scalping",
            "absorption_scalping",
            "rsi_bollinger_scalping",
        ]

        for strategy in strategies:
            folders.append(
                os.path.join(analysis_dir, "strategy_optimization", strategy)
            )

        # 폴더 생성
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        print(f"✅ 분석 폴더 구조 생성 완료: {analysis_dir}")
        return True

    except Exception as e:
        print(f"❌ 분석 폴더 구조 생성 중 오류: {e}")
        return False


def create_results_folder_structure(results_dir: str = "results"):
    """결과 폴더 구조 생성 (long 등 하위 폴더 포함)"""
    try:
        # 메인 결과 폴더들
        folders = [
            os.path.join(results_dir, "long"),
            os.path.join(results_dir, "short"),
            os.path.join(results_dir, "scalping"),
            os.path.join(results_dir, "swing"),
            os.path.join(results_dir, "research"),
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        print(f"✅ 결과 폴더 구조 생성 완료: {results_dir}")
        return True
    except Exception as e:
        print(f"❌ 결과 폴더 구조 생성 중 오류: {e}")
        return False


def split_data_train_test(
    data_dict: Dict[str, pd.DataFrame], train_ratio: float = 0.8
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    데이터를 train과 test로 분할

    Args:
        data_dict: 종목별 데이터 딕셔너리
        train_ratio: train 데이터 비율 (0.0 ~ 1.0)

    Returns:
        train_data_dict, test_data_dict
    """
    train_data = {}
    test_data = {}

    for symbol, data in data_dict.items():
        if data.empty:
            continue

        # 데이터 포인트 수 계산
        total_points = len(data)
        train_points = int(round(total_points * train_ratio))

        # 분할
        train_data[symbol] = data.iloc[:train_points].copy()
        test_data[symbol] = data.iloc[train_points:].copy()

        pass

    return train_data, test_data


def calculate_buy_hold_returns(
    data_dict: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    """
    Buy & Hold 전략의 수익률 계산

    Args:
        data_dict: 종목별 데이터 딕셔너리

    Returns:
        종목별 Buy & Hold 성과 지표
    """
    buy_hold_results = {}

    for symbol, data in data_dict.items():
        if data.empty or "Close" not in data.columns:
            continue

        try:
            # 수익률 계산
            initial_price = data["Close"].iloc[0]
            final_price = data["Close"].iloc[-1]
            total_return = (final_price - initial_price) / initial_price

            # 일별 수익률 계산
            daily_returns = data["Close"].pct_change().dropna()

            # 샤프 비율 계산 (연율화)
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            sharpe_ratio = (
                (mean_return * 252) / (std_return * np.sqrt(252))
                if std_return > 0
                else 0.0
            )

            # 소르티노 비율 계산
            negative_returns = daily_returns[daily_returns < 0]
            downside_std = negative_returns.std()
            sortino_ratio = (
                (mean_return * 252) / (downside_std * np.sqrt(252))
                if downside_std > 0
                else 0.0
            )

            # 최대 낙폭 계산
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())

            # 변동성 (연율화)
            volatility = std_return * np.sqrt(252)

            buy_hold_results[symbol] = {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "total_trades": 1,  # Buy & Hold는 1번의 거래
            }

        except Exception as e:
            print(f"❌ {symbol} Buy & Hold 계산 중 오류: {e}")
            buy_hold_results[symbol] = {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "total_trades": 0,
            }

    return buy_hold_results


def calculate_portfolio_metrics(
    individual_results: Dict[str, Dict[str, float]], weights: Dict[str, float]
) -> Dict[str, float]:
    """
    포트폴리오 전체 성과 지표 계산

    Args:
        individual_results: 종목별 성과 지표
        weights: 포트폴리오 비중

    Returns:
        포트폴리오 전체 성과 지표
    """
    try:
        # 가중 평균 수익률
        portfolio_return = sum(
            individual_results[symbol]["total_return"] * weights.get(symbol, 0.0)
            for symbol in individual_results.keys()
        )

        # 가중 평균 샤프 비율
        portfolio_sharpe = sum(
            individual_results[symbol]["sharpe_ratio"] * weights.get(symbol, 0.0)
            for symbol in individual_results.keys()
        )

        # 가중 평균 소르티노 비율
        portfolio_sortino = sum(
            individual_results[symbol]["sortino_ratio"] * weights.get(symbol, 0.0)
            for symbol in individual_results.keys()
        )

        # 가중 평균 변동성
        portfolio_volatility = sum(
            individual_results[symbol]["volatility"] * weights.get(symbol, 0.0)
            for symbol in individual_results.keys()
        )

        # 최대 낙폭 (가장 큰 값 사용)
        portfolio_max_drawdown = max(
            individual_results[symbol]["max_drawdown"]
            for symbol in individual_results.keys()
        )

        # 총 거래 수
        total_trades = sum(
            individual_results[symbol]["total_trades"]
            for symbol in individual_results.keys()
        )

        return {
            "total_return": portfolio_return,
            "sharpe_ratio": portfolio_sharpe,
            "sortino_ratio": portfolio_sortino,
            "max_drawdown": portfolio_max_drawdown,
            "volatility": portfolio_volatility,
            "total_trades": total_trades,
        }

    except Exception as e:
        print(f"❌ 포트폴리오 지표 계산 중 오류: {e}")
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "total_trades": 0,
        }
