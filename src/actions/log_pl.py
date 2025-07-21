#!/usr/bin/env python3
"""
실제 매매 시뮬레이션 및 P&L 로깅 시스템
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingSimulator:
    """실제 매매 시뮬레이션 클래스"""

    def __init__(
        self,
        config_path: str = "../../config/config_long.json",
    ):
        self.config = self._load_config(config_path)
        self.initial_capital = self.config["trading"]["initial_capital"]
        self.fee_config = self.config["trading"]
        self.simulation_settings = self.config["simulation_settings"]
        self.reset()
        # Short position 지원 추가
        self.enable_short_position = self.config.get("trading", {}).get(
            "enable_short_position", False
        )

    def _load_config(self, config_path: str) -> Dict:
        """통합 설정 파일 로드"""
        try:
            # 절대 경로로 시도
            if os.path.isabs(config_path):
                config_file = config_path
            else:
                # 상대 경로인 경우 프로젝트 루트 기준으로 시도
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(__file__))
                )
                config_file = os.path.join(project_root, config_path)

            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"통합 설정 파일 로드 실패, 기본값 사용: {e}")
            return {
                "trading": {
                    "initial_capital": 100000,
                    "slippage_settings": {
                        "default_slippage": 0.0005,
                        "market_conditions": {
                            "high_volatility": 0.001,
                            "low_volatility": 0.0002,
                            "normal_volatility": 0.0005,
                        },
                    },
                    "commission_schedule": [
                        {
                            "start": "2025-01-01",
                            "end": "2025-12-31",
                            "commission": 0.001,
                        },
                        {
                            "start": "2026-01-01",
                            "end": "2099-12-31",
                            "commission": 0.0025,
                        },
                    ],
                    "additional_fees": {"sec_fee": 0.0000229, "finra_fee": 0.000119},
                },
                "simulation_settings": {
                    "enable_detailed_logging": True,
                    "log_trade_details": True,
                    "calculate_metrics": True,
                },
            }

    def reset(self):
        """시뮬레이션 상태 초기화"""
        self.initial_capital = self.simulation_settings.get("initial_capital", 100000)
        self.cash = self.initial_capital
        self.position = 0  # 롱 포지션 수량
        self.short_position = 0  # 숏 포지션 수량 추가
        self.entry_price = 0
        self.short_entry_price = 0  # 숏 진입가 추가
        self.entry_time = None
        self.short_entry_time = None  # 숏 진입시간 추가
        self.trades = []
        self.portfolio_values = []
        self.peak_value = self.initial_capital
        self.max_drawdown = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0

    def get_commission_rate(self, trade_date: str) -> float:
        """거래 날짜에 따른 수수료율 반환"""
        # commission_schedule이 없으면 기본 commission 사용
        if "commission_schedule" not in self.fee_config:
            return self.fee_config.get("commission", 0.001)

        for rule in self.fee_config["commission_schedule"]:
            # config_swing.json에서는 date_from, date_to를 사용
            start_date = rule.get("start", rule.get("date_from"))
            end_date = rule.get("end", rule.get("date_to"))
            commission_rate = rule.get("commission", rule.get("rate"))

            if start_date <= trade_date <= end_date:
                return commission_rate
        return self.fee_config.get("commission", 0.001)  # 기본값

    def calculate_total_fees(self, price: float, commission_rate: float) -> float:
        """총 수수료 계산 (수수료 + SEC + FINRA)"""
        commission = price * commission_rate

        # additional_fees가 없으면 기본값 사용
        additional_fees = self.fee_config.get("additional_fees", {})
        sec_fee = price * additional_fees.get("sec_fee", 0.0000229)  # SEC 수수료 기본값
        finra_fee = price * additional_fees.get(
            "finra_fee", 0.000119
        )  # FINRA 수수료 기본값

        return commission + sec_fee + finra_fee

    def simulate_trading(
        self, df: pd.DataFrame, signals: pd.DataFrame, strategy_name: str
    ) -> Dict[str, Any]:
        """단일 종목 매매 시뮬레이션 (기존 방식)"""
        return self._simulate_single_asset_trading(df, signals, strategy_name)

    def simulate_portfolio_trading(
        self,
        data_dict: Dict[str, pd.DataFrame],
        weights_df: pd.DataFrame,
        strategy_name: str,
        strategy_signals: Dict[str, pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        포트폴리오 기반 시뮬레이션 실행
        """
        self.reset()
        log_lines = []
        log_lines.append(f"=== {strategy_name} 포트폴리오 시뮬레이션 시작 ===")
        log_lines.append(f"초기 자본: ${self.initial_capital:,.2f}")
        # slippage_settings가 없으면 기본값 사용
        slippage = self.fee_config.get("slippage_settings", {}).get(
            "default_slippage", self.fee_config.get("slippage", 0.0002)
        )
        log_lines.append(f"슬리피지: {slippage*100:.3f}%")
        log_lines.append("=" * 50)

        current_portfolio_value = self.initial_capital  # 항상 초기화
        # 포트폴리오 상태 초기화
        self.portfolio_positions = {symbol: 0 for symbol in data_dict.keys()}
        self.portfolio_entry_prices = {symbol: 0 for symbol in data_dict.keys()}
        self.portfolio_values = []
        self.rebalance_count = 0

        # 공통 날짜 찾기
        common_dates = self._get_common_dates(data_dict, weights_df)

        for i, date in enumerate(common_dates):
            if i == 0:  # 첫 번째 행은 건너뛰기
                self.portfolio_values.append(self.cash)
                continue

            # 현재 시점의 가격과 목표 비중
            current_prices = {}
            for symbol in data_dict.keys():
                symbol_data = data_dict[symbol]
                current_row = symbol_data[symbol_data["datetime"] == date]
                if not current_row.empty:
                    current_prices[symbol] = current_row.iloc[0]["close"]

            # 목표 비중 가져오기
            target_weights = weights_df.loc[date].to_dict()

            # 전략별 신호가 있는 경우 비중 조정
            if strategy_signals:
                target_weights = self._adjust_weights_with_signals(
                    date, target_weights, strategy_signals, current_prices
                )

            # 리밸런싱 실행
            if self._should_rebalance(target_weights, current_prices):
                self._execute_rebalancing(
                    date, target_weights, current_prices, log_lines
                )

            # 포트폴리오 가치 계산
            current_portfolio_value = self._calculate_portfolio_value(current_prices)
            self.portfolio_values.append(current_portfolio_value)

            # 최대 낙폭 업데이트
            if current_portfolio_value > self.peak_value:
                self.peak_value = current_portfolio_value

            drawdown = (current_portfolio_value - self.peak_value) / self.peak_value
            if drawdown < self.max_drawdown:
                self.max_drawdown = drawdown

        # 최종 성과 계산
        results = self._calculate_portfolio_performance_metrics()

        # 최종 로그 출력 전, current_portfolio_value가 정의되어 있는지 확인
        if not hasattr(locals(), "current_portfolio_value"):
            current_portfolio_value = self.initial_capital
        log_lines.append("=" * 50)
        log_lines.append(f"=== {strategy_name} 포트폴리오 시뮬레이션 완료 ===")
        log_lines.append(f"최종 포트폴리오 가치: ${current_portfolio_value:,.2f}")
        log_lines.append(f"총 수익률: {results['total_return']*100:+.2f}%")
        log_lines.append(f"총 리밸런싱 횟수: {self.rebalance_count}")
        log_lines.append(f"최대 낙폭: {results['max_drawdown']*100:.2f}%")
        log_lines.append(f"샤프 비율: {results['sharpe_ratio']:.2f}")
        log_lines.append(f"SQN: {results['sqn']:.2f}")

        return {
            "log_lines": log_lines,
            "results": results,
            "trades": self.trades,
            "portfolio_values": self.portfolio_values,
            "current_position": self.position,  # 현재 보유 상태 추가
            "returns": returns,  # 수익률 리스트 추가
        }

    def _simulate_single_asset_trading(
        self, df: pd.DataFrame, signals: pd.DataFrame, strategy_name: str
    ) -> Dict[str, Any]:
        """실제 매매 시뮬레이션 실행"""
        print(f"🔍 시뮬레이션 시작: {strategy_name}")
        print(f"🔍 데이터 shape: {df.shape}")
        print(f"🔍 신호 shape: {signals.shape}")
        
        self.reset()
        log_lines = []
        returns = []  # 수익률 리스트 추가

        log_lines.append(f"=== {strategy_name} 매매 시뮬레이션 시작 ===")
        log_lines.append(f"초기 자본: ${self.initial_capital:,.2f}")
        # slippage_settings가 없으면 기본값 사용
        slippage = self.fee_config.get("slippage_settings", {}).get(
            "default_slippage", self.fee_config.get("slippage", 0.0002)
        )
        log_lines.append(f"슬리피지: {slippage*100:.3f}%")
        log_lines.append("=" * 50)

        print(f"🔍 시뮬레이션 루프 시작: {len(df)}개 행")
        print(f"🔍 신호 데이터 길이: {len(signals)}개 행")
        
        # 데이터와 신호의 인덱스가 맞는지 확인
        if len(df) != len(signals):
            print(f"⚠️ 데이터와 신호 길이 불일치: 데이터={len(df)}, 신호={len(signals)}")
            # 더 짧은 쪽에 맞춰서 처리
            min_length = min(len(df), len(signals))
            df = df.iloc[:min_length]
            signals = signals.iloc[:min_length]
            print(f"🔍 길이 조정: {min_length}개 행으로 통일")
        
        # 간단한 시뮬레이션 (무한 루프 방지)
        try:
            # 첫 번째 행은 건너뛰기
            self.portfolio_values.append(self.cash)
            
            # 이전 가격 저장 (일별 수익률 계산용)
            prev_price = df.iloc[0]["close"]
            
            # 나머지 행들 처리
            for i in range(1, min(len(df), len(signals))):
                # 현재 행 가져오기
                row = df.iloc[i]
                
                # 날짜 및 가격 정보 (컬럼명 확인)
                if "datetime" in row:
                    trade_date = (
                        row["datetime"].strftime("%Y-%m-%d")
                        if hasattr(row["datetime"], "strftime")
                        else str(row["datetime"])[:10]
                    )
                elif "date" in row:
                    trade_date = (
                        row["date"].strftime("%Y-%m-%d")
                        if hasattr(row["date"], "strftime")
                        else str(row["date"])[:10]
                    )
                else:
                    trade_date = str(i)  # 인덱스를 날짜로 사용
                
                current_price = row["close"]
                
                # 신호 가져오기
                signal = signals.iloc[i]["signal"]
                
                if i % 20 == 0:  # 20개마다 진행상황 출력
                    print(f"  🔍 진행상황: {i}/{len(df)} - 신호: {signal}")
                
                # 일별 수익률 계산 (거래 여부와 관계없이)
                if prev_price > 0:
                    daily_return = (current_price - prev_price) / prev_price
                    returns.append(daily_return)
                else:
                    returns.append(0.0)
                
                # 포트폴리오 가치 업데이트
                self.portfolio_values.append(self.cash)
                
                # 간단한 거래 로직 (무한 루프 방지를 위해 단순화)
                if signal == 1 and self.position == 0:  # 매수 신호
                    # 간단한 매수 로직
                    shares_to_buy = 1
                    self.position = shares_to_buy
                    self.entry_price = current_price
                    if "datetime" in row:
                        self.entry_time = row["datetime"]
                    elif "date" in row:
                        self.entry_time = row["date"]
                    else:
                        self.entry_time = i
                    self.cash -= shares_to_buy * current_price
                    
                    # 거래 기록 추가
                    trade_record = {
                        "entry_time": self.entry_time,
                        "exit_time": None,
                        "entry_price": self.entry_price,
                        "exit_price": None,
                        "shares": shares_to_buy,
                        "position_type": "long",
                        "pnl": 0.0,
                        "pnl_amount": 0.0,
                        "hold_duration": 0.0,
                    }
                    self.trades.append(trade_record)
                    
                elif signal == -1 and self.position > 0:  # 매도 신호
                    # 간단한 매도 로직
                    exit_price = current_price
                    self.cash += self.position * exit_price
                    
                    # 거래 수익률 계산 (거래 기록용)
                    if self.entry_price > 0:
                        pnl = (exit_price - self.entry_price) / self.entry_price
                        pnl_amount = (exit_price - self.entry_price) * self.position
                        
                        # 마지막 거래 기록 업데이트
                        if self.trades:
                            self.trades[-1]["exit_time"] = row["datetime"] if "datetime" in row else i
                            self.trades[-1]["exit_price"] = exit_price
                            self.trades[-1]["pnl"] = pnl
                            self.trades[-1]["pnl_amount"] = pnl_amount
                            self.trades[-1]["hold_duration"] = 1.0  # 간단히 1일로 설정
                    
                    self.position = 0
                    self.entry_price = 0
                    self.entry_time = None
                
                # 이전 가격 업데이트
                prev_price = current_price
                    
        except Exception as e:
            print(f"❌ 시뮬레이션 오류: {e}")
            return {}

            # 수수료율 계산
            commission_rate = self.get_commission_rate(trade_date)
            slippage = self.fee_config.get("slippage_settings", {}).get(
                "default_slippage", self.fee_config.get("slippage", 0.0002)
            )

            # 매수 신호
            if self.position == 0 and self.short_position == 0 and signal == 1:
                # 슬리피지 적용된 체결가
                execution_price = current_price * (1 + slippage)
                total_fees = self.calculate_total_fees(execution_price, commission_rate)
                cost_per_share = execution_price + total_fees

                # 거래 가능한 주식 수 계산 (전체 자본의 95% 사용, 최소 1주 보장)
                available_capital = self.cash * 0.95
                shares_to_buy = max(1, int(available_capital / cost_per_share))

                # 최소 거래 금액 확인 (최소 $1000)
                min_trade_amount = 1000
                if shares_to_buy * cost_per_share < min_trade_amount:
                    shares_to_buy = max(1, int(min_trade_amount / cost_per_share))

                if shares_to_buy > 0:
                    # 롱 포지션 진입
                    self.position = shares_to_buy
                    self.entry_price = cost_per_share
                    self.entry_time = row["datetime"]
                    total_cost = shares_to_buy * cost_per_share
                    self.cash -= total_cost

                log_lines.append(f"[{row['datetime']}] 🔵 롱 포지션 진입")
                log_lines.append(
                    f"    체결가: ${execution_price:.2f} (슬리피지: +{slippage*100:.3f}%)"
                )
                log_lines.append(
                    f"    수수료: ${total_fees:.2f} (수수료율: {commission_rate*100:.2f}%)"
                )
                log_lines.append(
                    f"    매수 주식 수: {shares_to_buy}주 (${shares_to_buy * execution_price:.2f})"
                )
                log_lines.append(f"    총 비용: ${total_cost:.2f}")
                log_lines.append(f"    잔고: ${self.cash:.2f}")
                log_lines.append(
                    f"    거래 비율: {(total_cost/self.initial_capital)*100:.1f}%"
                )

            # 숏 포지션 진입 (Short position 지원이 활성화된 경우)
            elif (
                self.position == 0
                and self.short_position == 0
                and signal == -1
                and self.enable_short_position
            ):
                # 슬리피지 적용된 체결가
                execution_price = current_price * (1 - slippage)
                total_fees = self.calculate_total_fees(execution_price, commission_rate)
                cost_per_share = execution_price + total_fees

                # 거래 가능한 주식 수 계산 (전체 자본의 95% 사용, 최소 1주 보장)
                available_capital = self.cash * 0.95
                shares_to_sell = max(1, int(available_capital / cost_per_share))

                # 최소 거래 금액 확인 (최소 $1000)
                min_trade_amount = 1000
                if shares_to_sell * cost_per_share < min_trade_amount:
                    shares_to_sell = max(1, int(min_trade_amount / cost_per_share))

                if shares_to_sell > 0:
                    # 숏 포지션 진입
                    self.short_position = shares_to_sell
                    self.short_entry_price = cost_per_share
                    self.short_entry_time = row["datetime"]
                    total_revenue = shares_to_sell * execution_price
                    self.cash += total_revenue

                log_lines.append(f"[{row['datetime']}] 🔴 숏 포지션 진입")
                log_lines.append(
                    f"    체결가: ${execution_price:.2f} (슬리피지: -{slippage*100:.3f}%)"
                )
                log_lines.append(
                    f"    수수료: ${total_fees:.2f} (수수료율: {commission_rate*100:.2f}%)"
                )
                log_lines.append(
                    f"    매도 주식 수: {shares_to_sell}주 (${shares_to_sell * execution_price:.2f})"
                )
                log_lines.append(f"    총 수익: ${total_revenue:.2f}")
                log_lines.append(f"    잔고: ${self.cash:.2f}")
                log_lines.append(
                    f"    거래 비율: {(total_revenue/self.initial_capital)*100:.1f}%"
                )

            # 롱 포지션 청산
            elif self.position > 0 and signal == -1:
                # 슬리피지 적용된 체결가
                execution_price = current_price * (1 - slippage)
                total_fees = self.calculate_total_fees(execution_price, commission_rate)
                revenue_per_share = execution_price - total_fees

                # 수익률 계산
                pnl = (revenue_per_share - self.entry_price) / self.entry_price
                pnl_amount = (revenue_per_share - self.entry_price) * self.position

                # 포지션 청산
                total_revenue = self.position * revenue_per_share
                self.cash += total_revenue
                shares_sold = self.position
                self.position = 0

                # 거래 기록
                trade_record = {
                    "entry_time": self.entry_time,
                    "exit_time": row["datetime"],
                    "entry_price": self.entry_price,
                    "exit_price": revenue_per_share,
                    "shares": shares_sold,
                    "position_type": "long",
                    "pnl": pnl,
                    "pnl_amount": pnl_amount,
                    "hold_duration": (row["datetime"] - self.entry_time).total_seconds()
                    / 3600,
                }
                

                self.trades.append(trade_record)

                # 연속 승/패 업데이트
                if pnl > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    self.max_consecutive_wins = max(
                        self.max_consecutive_wins, self.consecutive_wins
                    )
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    self.max_consecutive_losses = max(
                        self.max_consecutive_losses, self.consecutive_losses
                    )

                # 로그 출력
                pnl_symbol = "🟢" if pnl > 0 else "🔴"
                log_lines.append(f"[{row['datetime']}] {pnl_symbol} 롱 포지션 청산")
                log_lines.append(
                    f"    체결가: ${execution_price:.2f} (슬리피지: -{slippage*100:.3f}%)"
                )
                log_lines.append(
                    f"    매도 주식 수: {shares_sold}주 (${shares_sold * execution_price:.2f})"
                )
                log_lines.append(f"    수수료: ${total_fees * shares_sold:.2f}")
                log_lines.append(f"    총 수익: ${total_revenue:.2f}")
                log_lines.append(f"    P&L: {pnl*100:+.2f}% (${pnl_amount:+.2f})")
                log_lines.append(f"    잔고: ${self.cash:.2f}")
                log_lines.append(
                    f"    보유기간: {trade_record['hold_duration']:.1f}시간"
                )
                log_lines.append(
                    f"    거래 비율: {(total_revenue/self.initial_capital)*100:.1f}%"
                )

            # 숏 포지션 청산
            elif self.short_position > 0 and signal == 1:
                # 슬리피지 적용된 체결가
                execution_price = current_price * (1 + slippage)
                total_fees = self.calculate_total_fees(execution_price, commission_rate)
                cost_per_share = execution_price + total_fees

                # 수익률 계산 (숏 포지션은 가격 하락 시 수익)
                pnl = (self.short_entry_price - cost_per_share) / self.short_entry_price
                pnl_amount = (
                    self.short_entry_price - cost_per_share
                ) * self.short_position

                # 포지션 청산
                total_cost = self.short_position * cost_per_share
                self.cash -= total_cost
                shares_bought = self.short_position
                self.short_position = 0

                # 거래 기록
                trade_record = {
                    "entry_time": self.short_entry_time,
                    "exit_time": row["datetime"],
                    "entry_price": self.short_entry_price,
                    "exit_price": cost_per_share,
                    "shares": shares_bought,
                    "position_type": "short",
                    "pnl": pnl,
                    "pnl_amount": pnl_amount,
                    "hold_duration": (
                        row["datetime"] - self.short_entry_time
                    ).total_seconds()
                    / 3600,
                }
                self.trades.append(trade_record)

                # 연속 승/패 업데이트
                if pnl > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    self.max_consecutive_wins = max(
                        self.max_consecutive_wins, self.consecutive_wins
                    )
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    self.max_consecutive_losses = max(
                        self.max_consecutive_losses, self.consecutive_losses
                    )

                # 로그 출력
                pnl_symbol = "🟢" if pnl > 0 else "🔴"
                log_lines.append(f"[{row['datetime']}] {pnl_symbol} 숏 포지션 청산")
                log_lines.append(
                    f"    체결가: ${execution_price:.2f} (슬리피지: +{slippage*100:.3f}%)"
                )
                log_lines.append(
                    f"    매수 주식 수: {shares_bought}주 (${shares_bought * execution_price:.2f})"
                )
                log_lines.append(f"    수수료: ${total_fees * shares_bought:.2f}")
                log_lines.append(f"    총 비용: ${total_cost:.2f}")
                log_lines.append(f"    P&L: {pnl*100:+.2f}% (${pnl_amount:+.2f})")
                log_lines.append(f"    잔고: ${self.cash:.2f}")
                log_lines.append(
                    f"    보유기간: {trade_record['hold_duration']:.1f}시간"
                )
                log_lines.append(
                    f"    거래 비율: {(total_cost/self.initial_capital)*100:.1f}%"
                )

            # 포트폴리오 가치 계산
            current_portfolio_value = self.cash
            if self.position > 0:
                # 미실현 손익 계산
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                current_portfolio_value = (
                    self.cash + self.position * self.entry_price * (1 + unrealized_pnl)
                )
            if self.short_position > 0:
                # 미실현 손익 계산 (숏 포지션은 가격 하락 시 수익)
                unrealized_pnl = (
                    self.short_entry_price - current_price
                ) / self.short_entry_price
                current_portfolio_value = (
                    self.cash
                    + self.short_position
                    * self.short_entry_price
                    * (1 + unrealized_pnl)
                )

            self.portfolio_values.append(current_portfolio_value)

            # 수익률 계산 및 추가
            if len(self.portfolio_values) > 1:
                daily_return = (
                    current_portfolio_value - self.portfolio_values[-2]
                ) / self.portfolio_values[-2]
                returns.append(daily_return)
            else:
                returns.append(0.0)

            # 최대 낙폭 업데이트
            if current_portfolio_value > self.peak_value:
                self.peak_value = current_portfolio_value

            drawdown = (current_portfolio_value - self.peak_value) / self.peak_value
            if drawdown < self.max_drawdown:
                self.max_drawdown = drawdown

        # 최종 성과 계산
        results = self._calculate_performance_metrics()

        # 최종 로그 출력
        log_lines.append("=" * 50)
        log_lines.append(f"=== {strategy_name} 시뮬레이션 완료 ===")
        log_lines.append(f"최종 잔고: ${self.cash:,.2f}")
        log_lines.append(f"총 수익률: {results['total_return']*100:+.2f}%")
        log_lines.append(f"총 거래 횟수: {len(self.trades)}")
        log_lines.append(f"승률: {results['win_rate']*100:.1f}%")
        log_lines.append(f"평균 수익률: {results['avg_return']*100:+.2f}%")
        log_lines.append(f"최대 낙폭: {results['max_drawdown']*100:.2f}%")
        log_lines.append(f"샤프 비율: {results['sharpe_ratio']:.2f}")
        log_lines.append(f"SQN: {results['sqn']:.2f}")
        log_lines.append(f"최대 연속 승: {self.max_consecutive_wins}")
        log_lines.append(f"최대 연속 패: {self.max_consecutive_losses}")
        log_lines.append(f"수익 팩터: {results['profit_factor']:.2f}")
        log_lines.append(f"평균 보유기간: {results['avg_hold_duration']:.1f}시간")

        print(f"✅ 시뮬레이션 완료: {strategy_name}")

        # 최종 거래 정보 추출
        final_position = 0
        final_price = None
        final_date = None
        
        if self.trades:
            last_trade = self.trades[-1]
            # 마지막 거래가 매수이고 아직 매도되지 않았으면 보유 중
            if last_trade.get("exit_time") is None:
                final_position = last_trade.get("shares", 0)
                final_price = last_trade.get("entry_price")
                final_date = last_trade.get("entry_time")
            else:
                # 매도 완료된 경우
                final_price = last_trade.get("exit_price")
                final_date = last_trade.get("exit_time")
        else:
            # 거래가 없는 경우 마지막 데이터의 날짜를 사용
            if len(df) > 0:
                final_date = df.iloc[-1]["datetime"]
        
        return {
            "log_lines": log_lines,
            "results": results,
            "trades": self.trades,
            "portfolio_values": self.portfolio_values,
            "returns": returns,  # 수익률 리스트 추가
            "current_position": final_position,  # 마지막 거래 상태로 보유 여부 판단
            "final_price": final_price,  # 최종 매수/매도 가격
            "final_date": final_date,  # 최종 매수/매도 시점
        }

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """성과 지표 계산"""
        if not self.trades:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sqn": 0.0,
                "profit_factor": 0.0,
                "avg_hold_duration": 0.0,
                "total_trades": 0,
            }

        # 기본 지표
        total_return = (self.cash - self.initial_capital) / self.initial_capital
        
        # 거래별 수익률 (거래 통계용)
        trade_returns = [trade["pnl"] for trade in self.trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r <= 0]

        win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        # 일별 수익률 (샤프/소르티노 비율 계산용)
        daily_returns = []
        if len(self.portfolio_values) > 1:
            for i in range(1, len(self.portfolio_values)):
                daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
                daily_returns.append(daily_return)
        
        # 샤프 비율 (일별 수익률 기준)
        sharpe_ratio = 0.0
        if daily_returns:
            daily_returns_series = pd.Series(daily_returns)
            mean_daily_return = daily_returns_series.mean()
            std_daily_return = daily_returns_series.std()
            
            if std_daily_return > 0:
                risk_free_rate = 0.02 / 252  # 일간 무위험 수익률
                excess_return = mean_daily_return - risk_free_rate
                # 연간화된 샤프 비율: (연간 초과수익률) / (연간 표준편차)
                sharpe_ratio = (excess_return * 252) / (std_daily_return * np.sqrt(252))

        # 소르티노 비율 (일별 수익률 기준)
        sortino_ratio = 0.0
        if daily_returns:
            daily_returns_series = pd.Series(daily_returns)
            mean_daily_return = daily_returns_series.mean()
            negative_returns = daily_returns_series[daily_returns_series < 0]
            
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std()
                if downside_deviation > 0:
                    risk_free_rate = 0.02 / 252  # 일간 무위험 수익률
                    excess_return = mean_daily_return - risk_free_rate
                    # 연간화된 소르티노 비율: (연간 초과수익률) / (연간 하방표준편차)
                    sortino_ratio = (excess_return * 252) / (downside_deviation * np.sqrt(252))

        # SQN (거래별 수익률 기준)
        trade_return_std = np.std(trade_returns) if len(trade_returns) > 1 else 0
        sqn = (avg_trade_return * np.sqrt(len(trade_returns))) / trade_return_std if trade_return_std > 0 else 0

        # 수익 팩터
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # 평균 보유기간
        avg_hold_duration = np.mean([trade["hold_duration"] for trade in self.trades])

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "avg_return": avg_trade_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,  # 소르티노 비율 추가
            "sqn": sqn,
            "profit_factor": profit_factor,
            "avg_hold_duration": avg_hold_duration,
            "total_trades": len(self.trades),
        }

    def _get_common_dates(
        self, data_dict: Dict[str, pd.DataFrame], weights_df: pd.DataFrame
    ) -> pd.DatetimeIndex:
        """공통 날짜 찾기"""
        # weights_df의 인덱스와 각 종목 데이터의 날짜 교집합
        weight_dates = set(weights_df.index)
        common_dates = weight_dates.copy()

        for symbol, df in data_dict.items():
            symbol_dates = set(df["datetime"])
            common_dates = common_dates.intersection(symbol_dates)

        return pd.DatetimeIndex(sorted(common_dates))

    def _should_rebalance(
        self, target_weights: Dict[str, float], current_prices: Dict[str, float]
    ) -> bool:
        """리밸런싱 필요 여부 판단"""
        if not current_prices:
            return False

        # 현재 포트폴리오 가치 계산
        current_portfolio_value = self._calculate_portfolio_value(current_prices)
        if current_portfolio_value == 0:
            return False

        # 현재 비중 계산
        current_weights = {}
        for symbol in current_prices.keys():
            if (
                symbol in self.portfolio_positions
                and self.portfolio_positions[symbol] > 0
            ):
                current_value = (
                    self.portfolio_positions[symbol] * current_prices[symbol]
                )
                current_weights[symbol] = current_value / current_portfolio_value
            else:
                current_weights[symbol] = 0.0

        # 현금 비중
        cash_weight = self.cash / current_portfolio_value
        current_weights["CASH"] = cash_weight

        # 비중 차이 계산
        total_diff = 0
        force_rebalance = False
        for symbol in target_weights.keys():
            target = target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            total_diff += abs(target - current)
            # 목표 비중이 있는데 현재 비중이 0이면 무조건 리밸런싱
            if target > 0 and current == 0:
                force_rebalance = True

        # 실전형 전략 대응: 목표 비중이 있는데 현재 비중이 0인 종목이 있으면 무조건 리밸런싱
        if force_rebalance:
            return True
        # 일반적인 경우: 2% 이상 차이나면 리밸런싱
        return total_diff > 0.02

    def _execute_rebalancing(
        self,
        date: pd.Timestamp,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        log_lines: List[str],
    ):
        """리밸런싱 실행"""
        self.rebalance_count += 1

        # 현재 포트폴리오 가치
        current_portfolio_value = self._calculate_portfolio_value(current_prices)

        log_lines.append(f"[{date}] 🔄 리밸런싱 #{self.rebalance_count}")
        log_lines.append(f"    포트폴리오 가치: ${current_portfolio_value:,.2f}")

        # 각 종목별로 목표 비중에 맞게 조정
        for symbol in current_prices.keys():
            if symbol == "CASH":
                continue

            target_weight = target_weights.get(symbol, 0.0)
            target_value = current_portfolio_value * target_weight
            current_price = current_prices[symbol]

            # 현재 보유 수량
            current_quantity = self.portfolio_positions.get(symbol, 0)
            current_value = current_quantity * current_price

            # 목표 수량
            target_quantity = int(target_value / current_price)

            # 매수/매도 수량 계산
            trade_quantity = target_quantity - current_quantity

            if trade_quantity != 0:
                # 수수료 계산
                commission_rate = self.get_commission_rate(date.strftime("%Y-%m-%d"))
                slippage = self.fee_config["slippage_settings"]["default_slippage"]

                if trade_quantity > 0:  # 매수
                    execution_price = current_price * (1 + slippage)
                    total_fees = self.calculate_total_fees(
                        execution_price, commission_rate
                    )
                    total_cost = trade_quantity * (execution_price + total_fees)

                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        self.portfolio_positions[symbol] = target_quantity

                        log_lines.append(
                            f"    🔵 {symbol} 매수: {trade_quantity}주 @ ${execution_price:.2f}"
                        )
                        log_lines.append(
                            f"        수수료: ${total_fees * trade_quantity:.2f}"
                        )

                else:  # 매도
                    execution_price = current_price * (1 - slippage)
                    total_fees = self.calculate_total_fees(
                        execution_price, commission_rate
                    )
                    total_revenue = abs(trade_quantity) * (execution_price - total_fees)

                    self.cash += total_revenue
                    self.portfolio_positions[symbol] = target_quantity

                    log_lines.append(
                        f"    🔴 {symbol} 매도: {abs(trade_quantity)}주 @ ${execution_price:.2f}"
                    )
                    log_lines.append(
                        f"        수수료: ${total_fees * abs(trade_quantity):.2f}"
                    )

        # 현금 비중 조정
        target_cash_weight = target_weights.get("CASH", 0.0)
        target_cash_value = current_portfolio_value * target_cash_weight

        if abs(self.cash - target_cash_value) > 1000:  # $1000 이상 차이나면 조정
            log_lines.append(
                f"    💰 현금 조정: ${self.cash:,.2f} → ${target_cash_value:,.2f}"
            )

    def _adjust_weights_with_signals(
        self,
        date: pd.Timestamp,
        target_weights: Dict[str, float],
        strategy_signals: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """전략 신호를 기반으로 포트폴리오 비중 조정"""
        adjusted_weights = target_weights.copy()

        for symbol, signal_df in strategy_signals.items():
            if symbol in current_prices and symbol in target_weights:
                # dict 타입이면 DataFrame으로 변환
                if isinstance(signal_df, dict):
                    signal_df = pd.DataFrame(signal_df)
                # 신호가 DataFrame이 아니거나 columns 속성이 없으면 기본값 사용
                if not hasattr(signal_df, "columns") or signal_df is None:
                    continue
                # 해당 날짜의 신호 찾기
                current_signal = (
                    signal_df[signal_df["datetime"] == date]
                    if not signal_df.empty
                    else pd.DataFrame()
                )
                if not current_signal.empty:
                    signal_value = current_signal.iloc[0]["signal"]

                    # 신호에 따른 비중 조정
                    if signal_value > 0:  # 매수 신호
                        adjusted_weights[symbol] = min(
                            target_weights[symbol] * 1.5, target_weights[symbol] + 0.1
                        )
                    elif signal_value < 0:  # 매도 신호
                        adjusted_weights[symbol] = max(
                            target_weights[symbol] * 0.5, target_weights[symbol] - 0.1
                        )

        # 비중 정규화 (총합이 1이 되도록)
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total_weight

        return adjusted_weights

    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """포트폴리오 가치 계산"""
        portfolio_value = self.cash

        for symbol, price in current_prices.items():
            if symbol in self.portfolio_positions:
                quantity = self.portfolio_positions[symbol]
                portfolio_value += quantity * price
            if symbol in self.short_position:
                quantity = self.short_position[symbol]
                portfolio_value += quantity * price

        return portfolio_value

    def _calculate_portfolio_performance_metrics(self) -> Dict[str, float]:
        """포트폴리오 성과 지표 계산"""
        if len(self.portfolio_values) < 2:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sqn": 0.0,
                "profit_factor": 0.0,
                "avg_hold_duration": 0.0,
                "total_trades": 0,
            }

        # 수익률 계산
        portfolio_returns = pd.Series(self.portfolio_values).pct_change().dropna()
        total_return = (
            self.portfolio_values[-1] - self.initial_capital
        ) / self.initial_capital

        # 샤프 비율
        sharpe_ratio = (
            (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252))
            if portfolio_returns.std() > 0
            else 0
        )

        # SQN
        sqn = (
            (portfolio_returns.mean() * np.sqrt(len(portfolio_returns)))
            / portfolio_returns.std()
            if portfolio_returns.std() > 0
            else 0
        )

        # 거래 통계 (리밸런싱 기반)
        profitable_rebalances = 0
        total_rebalances = self.rebalance_count

        # 리밸런싱 후 수익률이 양수인 경우를 승리로 계산
        if len(self.portfolio_values) > 1:
            for i in range(1, len(self.portfolio_values)):
                if self.portfolio_values[i] > self.portfolio_values[i - 1]:
                    profitable_rebalances += 1

        win_rate = (
            profitable_rebalances / total_rebalances if total_rebalances > 0 else 0
        )

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "avg_return": portfolio_returns.mean(),
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sqn": sqn,
            "profit_factor": 1.0,  # 포트폴리오에서는 다름
            "avg_hold_duration": 0.0,  # 포트폴리오에서는 다름
            "total_trades": self.rebalance_count,
        }

    def print_logs(self, log_lines: List[str]):
        """로그를 터미널에 출력"""
        for line in log_lines:
            print(line)
            # 실시간 출력을 위한 flush
            import sys

            sys.stdout.flush()


def main():
    """테스트용 메인 함수"""
    print("TradingSimulator 테스트")

    # 샘플 데이터 생성
    dates = pd.date_range(start="2025-01-01", end="2025-01-31", freq="H")
    np.random.seed(42)

    close_prices = [100]
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 0.01)
        close_prices.append(close_prices[-1] * (1 + change))

    df = pd.DataFrame(
        {
            "datetime": dates,
            "close": close_prices,
            "open": [p * (1 + np.random.normal(0, 0.005)) for p in close_prices],
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in close_prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in close_prices],
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        }
    )

    # 샘플 신호 생성
    signals = pd.DataFrame(
        {
            "datetime": dates,
            "signal": np.random.choice([-1, 0, 1], len(dates), p=[0.3, 0.4, 0.3]),
        }
    )

    # 시뮬레이션 실행
    simulator = TradingSimulator()  # 새로운 설정 파일 사용
    result = simulator.simulate_trading(df, signals, "테스트 전략")

    # 로그 출력
    simulator.print_logs(result["log_lines"])


if __name__ == "__main__":
    main()
