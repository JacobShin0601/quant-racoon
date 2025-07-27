#!/usr/bin/env python3
"""
백테스팅 상세 리포트 생성기
- 백테스팅 기간 명시
- 포트폴리오 및 개별 종목 상세 성과
- 매매 내역 및 최종 보유현황
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BacktestReporter:
    """백테스팅 상세 리포트 생성기"""

    def __init__(self):
        pass

    def generate_detailed_backtest_report(
        self,
        backtest_result: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        start_date: str,
        end_date: str,
        strategy_name: str = "신경망 전략",
    ) -> str:
        """
        상세 백테스팅 리포트 생성

        Args:
            backtest_result: 백테스팅 결과
            portfolio_weights: 포트폴리오 비중
            start_date: 백테스팅 시작일
            end_date: 백테스팅 종료일
            strategy_name: 전략명

        Returns:
            상세 리포트 문자열
        """
        try:
            report_lines = []

            # 헤더
            report_lines.append("=" * 100)
            report_lines.append(f"📊 {strategy_name} 백테스팅 상세 리포트")
            report_lines.append("=" * 100)
            report_lines.append(f"📅 백테스팅 기간: {start_date} ~ {end_date}")

            # 기간 계산
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            total_days = (end_dt - start_dt).days
            report_lines.append(f"📈 분석 기간: {total_days}일")
            report_lines.append("")

            # 1. 포트폴리오 전체 성과 요약
            portfolio_summary = self._create_portfolio_summary_table(
                backtest_result, portfolio_weights
            )
            report_lines.append(portfolio_summary)
            report_lines.append("")

            # 2. 개별 종목 상세 성과
            individual_summary = self._create_individual_performance_table(
                backtest_result, portfolio_weights
            )
            report_lines.append(individual_summary)
            report_lines.append("")

            # 3. 매매 내역 요약
            trading_summary = self._create_trading_summary_table(backtest_result)
            report_lines.append(trading_summary)
            report_lines.append("")

            # 4. 최종 보유 현황
            final_positions = self._create_final_positions_table(
                backtest_result, end_date
            )
            report_lines.append(final_positions)
            report_lines.append("")

            # 5. 상세 거래 이력 (최근 10건)
            recent_trades = self._create_recent_trades_table(backtest_result)
            report_lines.append(recent_trades)

            report_lines.append("=" * 100)

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"백테스팅 리포트 생성 실패: {e}")
            return "백테스팅 리포트 생성 중 오류가 발생했습니다."

    def _create_portfolio_summary_table(
        self, backtest_result: Dict[str, Any], weights: Dict[str, float]
    ) -> str:
        """포트폴리오 전체 성과 요약 테이블"""
        try:
            lines = []
            lines.append("🎯 포트폴리오 전체 성과 요약")
            lines.append("-" * 80)

            portfolio_perf = backtest_result.get("portfolio_performance", {})
            metrics = backtest_result.get("performance_metrics", {})
            portfolio_summary = metrics.get("portfolio_summary", {})

            # 기본 성과 지표
            total_return = portfolio_summary.get("total_return", 0)
            total_trades = portfolio_summary.get("total_trades", 0)
            sharpe_ratio = portfolio_summary.get("sharpe_ratio", 0)

            lines.append(f"📈 총 수익률:        {total_return*100:>8.2f}%")
            lines.append(f"📊 총 거래 횟수:     {total_trades:>8.0f} 회")
            lines.append(f"⚡ 샤프 비율:       {sharpe_ratio:>8.3f}")

            # 추가 지표 (계산 가능한 경우)
            portfolio_metrics = portfolio_perf.get("metrics", {})
            if portfolio_metrics:
                volatility = portfolio_metrics.get("volatility", 0)
                max_dd = portfolio_metrics.get("max_drawdown", 0)
                lines.append(f"📉 변동성:          {volatility*100:>8.2f}%")
                lines.append(f"📉 최대 낙폭:       {max_dd*100:>8.2f}%")

            # 포트폴리오 구성
            lines.append("")
            lines.append("💼 포트폴리오 구성:")
            for symbol, weight in sorted(
                weights.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"   {symbol}: {weight*100:>6.2f}%")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"포트폴리오 요약 테이블 생성 실패: {e}")
            return "포트폴리오 요약 생성 실패"

    def _create_individual_performance_table(
        self, backtest_result: Dict[str, Any], weights: Dict[str, float]
    ) -> str:
        """개별 종목 성과 테이블"""
        try:
            lines = []
            lines.append("📈 개별 종목 상세 성과")
            lines.append("-" * 120)

            # 헤더 (Buy & Hold 컬럼 추가)
            lines.append(
                f"{'종목':<8} {'비중':<8} {'수익률':<10} {'B&H':<10} {'거래수':<8} {'승률':<8} {'기여도':<10} {'평가':<8}"
            )
            lines.append("-" * 120)

            individual_perf = backtest_result.get("individual_performance", {})
            metrics = backtest_result.get("performance_metrics", {})
            individual_summary = metrics.get("individual_summary", {})

            for symbol in weights.keys():
                weight = weights.get(symbol, 0)
                perf = individual_perf.get(symbol, {})
                summary = individual_summary.get(symbol, {})

                # 성과 지표
                total_return = summary.get("return", 0)
                trade_count = summary.get("trades", 0)
                win_rate = summary.get("win_rate", 0)

                # Buy & Hold 수익률 계산
                buy_hold_return = 0.0
                if "buy_hold_return" in perf:
                    buy_hold_return = perf["buy_hold_return"]
                    logger.info(f"📊 {symbol} Buy & Hold (perf): {buy_hold_return:.4f}")
                elif "historical_data" in backtest_result:
                    # historical_data에서 Buy & Hold 계산
                    try:
                        from ..agent.performance_calculator import (
                            AdvancedPerformanceCalculator,
                        )

                        calculator = AdvancedPerformanceCalculator()

                        # 백테스팅 기간 추출
                        start_date = backtest_result.get("start_date", "")
                        end_date = backtest_result.get("end_date", "")

                        logger.info(
                            f"🔍 {symbol} Buy & Hold 계산 시도: {start_date} ~ {end_date}"
                        )

                        if (
                            start_date
                            and end_date
                            and symbol in backtest_result["historical_data"]
                        ):
                            # 새로운 누적 수익률 계산 메서드 사용
                            buy_hold_return = (
                                calculator.calculate_buy_hold_cumulative_return(
                                    backtest_result["historical_data"][symbol],
                                    start_date,
                                    end_date,
                                )
                            )
                            logger.info(
                                f"📊 {symbol} Buy & Hold (누적): {buy_hold_return:.4f}"
                            )
                        else:
                            logger.warning(
                                f"⚠️ {symbol} Buy & Hold 계산 조건 불충족: "
                                f"start_date={start_date}, end_date={end_date}, "
                                f"data_exists={symbol in backtest_result.get('historical_data', {})}"
                            )
                    except Exception as e:
                        logger.warning(f"⚠️ {symbol} Buy & Hold 계산 실패: {e}")
                else:
                    logger.warning(f"⚠️ {symbol} Buy & Hold 데이터 없음")

                # 포트폴리오 기여도
                contribution = total_return * weight

                # 평가
                if total_return > 0.05:
                    evaluation = "우수"
                elif total_return > 0:
                    evaluation = "양호"
                elif total_return > -0.05:
                    evaluation = "보통"
                else:
                    evaluation = "부진"

                logger.info(
                    f"📊 {symbol} 테이블 행: 수익률={total_return:.4f}, B&H={buy_hold_return:.4f}, "
                    f"거래수={trade_count}, 승률={win_rate:.2%}"
                )

                lines.append(
                    f"{symbol:<8} {weight*100:>6.1f}% {total_return*100:>8.2f}% "
                    f"{buy_hold_return*100:>8.2f}% {trade_count:>6.0f}회 {win_rate*100:>6.1f}% "
                    f"{contribution*100:>8.2f}% {evaluation:<8}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"개별 성과 테이블 생성 실패: {e}")
            return "개별 성과 테이블 생성 실패"

    def _create_trading_summary_table(self, backtest_result: Dict[str, Any]) -> str:
        """매매 내역 요약 테이블"""
        try:
            lines = []
            lines.append("📋 매매 내역 요약")
            lines.append("-" * 80)

            individual_perf = backtest_result.get("individual_performance", {})

            # 전체 매매 통계
            total_buy_count = 0
            total_sell_count = 0
            total_profitable_trades = 0
            total_trades = 0

            logger.info("🔍 매매 내역 상세 분석 시작")

            for symbol, perf in individual_perf.items():
                trades = perf.get("trades", [])
                logger.info(f"📊 {symbol} 거래 내역 분석: {len(trades)}개 거래")

                buy_trades = [t for t in trades if t.get("action") == "BUY"]
                sell_trades = [t for t in trades if t.get("action") == "SELL"]

                logger.info(
                    f"📊 {symbol} 거래 분류: 매수 {len(buy_trades)}개, 매도 {len(sell_trades)}개"
                )

                # 매도 거래의 PnL 상세 분석
                logger.info(f"📊 {symbol} 매도 거래 상세 분석:")
                for i, sell_trade in enumerate(sell_trades):
                    pnl = sell_trade.get("pnl", 0)
                    price = sell_trade.get("price", 0)
                    date = sell_trade.get("date", "N/A")
                    is_profitable = pnl > 0
                    logger.info(
                        f"   매도 {i+1}: PnL={pnl:.4f}, 가격=${price:.2f}, 날짜={date}, 수익={is_profitable}"
                    )

                profitable_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
                logger.info(f"📊 {symbol} 수익 거래: {len(profitable_trades)}개")

                # 승률 계산
                win_rate = (
                    len(profitable_trades) / len(sell_trades)
                    if len(sell_trades) > 0
                    else 0
                )
                logger.info(
                    f"📊 {symbol} 승률: {win_rate:.2%} ({len(profitable_trades)}/{len(sell_trades)})"
                )

                total_buy_count += len(buy_trades)
                total_sell_count += len(sell_trades)
                total_profitable_trades += len(profitable_trades)
                total_trades += len(sell_trades)  # 완료된 거래만 계산

            # 승률 계산
            overall_win_rate = (
                total_profitable_trades / total_trades if total_trades > 0 else 0
            )

            logger.info(
                f"📊 전체 통계: 매수 {total_buy_count}회, 매도 {total_sell_count}회, 수익 {total_profitable_trades}회, 승률 {overall_win_rate:.2%}"
            )

            lines.append(f"📊 총 매수 거래:     {total_buy_count:>8.0f} 회")
            lines.append(f"📊 총 매도 거래:     {total_sell_count:>8.0f} 회")
            lines.append(f"💰 수익 거래:       {total_profitable_trades:>8.0f} 회")
            lines.append(
                f"📉 손실 거래:       {total_trades - total_profitable_trades:>8.0f} 회"
            )
            lines.append(f"🎯 전체 승률:       {overall_win_rate*100:>8.1f}%")

            # 종목별 매매 횟수
            lines.append("")
            lines.append("종목별 매매 현황:")
            lines.append(
                f"{'종목':<8} {'매수':<6} {'매도':<6} {'수익거래':<8} {'승률':<8}"
            )
            lines.append("-" * 50)

            for symbol, perf in individual_perf.items():
                trades = perf.get("trades", [])

                buy_count = len([t for t in trades if t.get("action") == "BUY"])
                sell_count = len([t for t in trades if t.get("action") == "SELL"])
                profitable = len(
                    [
                        t
                        for t in trades
                        if t.get("action") == "SELL" and t.get("pnl", 0) > 0
                    ]
                )

                symbol_win_rate = profitable / sell_count if sell_count > 0 else 0

                lines.append(
                    f"{symbol:<8} {buy_count:>4.0f}회 {sell_count:>4.0f}회 "
                    f"{profitable:>6.0f}회 {symbol_win_rate*100:>6.1f}%"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"매매 요약 테이블 생성 실패: {e}")
            return "매매 요약 테이블 생성 실패"

    def _create_final_positions_table(
        self, backtest_result: Dict[str, Any], end_date: str
    ) -> str:
        """최종 보유현황 테이블"""
        try:
            lines = []
            lines.append(f"💼 최종 보유현황 ({end_date} 기준)")
            lines.append("-" * 80)

            lines.append(
                f"{'종목':<8} {'보유여부':<10} {'최종거래일':<12} {'최종거래':<10} {'거래가격':<12}"
            )
            lines.append("-" * 80)

            individual_perf = backtest_result.get("individual_performance", {})

            for symbol, perf in individual_perf.items():
                trades = perf.get("trades", [])

                if not trades:
                    lines.append(
                        f"{symbol:<8} {'없음':<10} {'-':<12} {'-':<10} {'-':<12}"
                    )
                    continue

                # 최종 거래 찾기
                last_trade = trades[-1]
                last_action = last_trade.get("action", "")
                last_price = last_trade.get("price", 0)
                last_date = last_trade.get("date", "")

                # 포지션 상태 확인
                position_status = "보유중" if last_action == "BUY" else "청산완료"

                # 날짜 포맷팅
                if isinstance(last_date, str):
                    try:
                        formatted_date = pd.to_datetime(last_date).strftime("%Y-%m-%d")
                    except:
                        formatted_date = str(last_date)[:10]
                else:
                    formatted_date = str(last_date)[:10]

                lines.append(
                    f"{symbol:<8} {position_status:<10} {formatted_date:<12} "
                    f"{last_action:<10} ${last_price:>9.2f}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"최종 보유현황 테이블 생성 실패: {e}")
            return "최종 보유현황 테이블 생성 실패"

    def _create_recent_trades_table(self, backtest_result: Dict[str, Any]) -> str:
        """최근 거래 이력 테이블"""
        try:
            lines = []
            lines.append("📋 최근 거래 이력 (최대 15건)")
            lines.append("-" * 120)

            lines.append(
                f"{'날짜':<12} {'종목':<8} {'액션':<6} {'가격':<10} {'수익률':<10} {'신뢰도':<8} {'상태':<8}"
            )
            lines.append("-" * 120)

            # 모든 거래를 수집하고 날짜순 정렬
            all_trades = []
            individual_perf = backtest_result.get("individual_performance", {})

            logger.info("🔍 전체 거래 내역 수집 시작")

            for symbol, perf in individual_perf.items():
                trades = perf.get("trades", [])
                logger.info(f"📊 {symbol}: {len(trades)}개 거래 수집")

                for i, trade in enumerate(trades):
                    trade_info = trade.copy()
                    trade_info["symbol"] = symbol
                    all_trades.append(trade_info)

                    # 거래 상세 정보 로깅
                    action = trade.get("action", "")
                    price = trade.get("price", 0)
                    pnl = trade.get("pnl", 0)
                    date = trade.get("date", "")
                    logger.info(
                        f"📊 {symbol} 거래 {i+1}: {action} @ ${price:.2f}, PnL={pnl:.4f}, 날짜={date}"
                    )

            logger.info(f"📊 총 수집된 거래: {len(all_trades)}개")

            # 날짜순 정렬 (최신순)
            all_trades.sort(
                key=lambda x: pd.to_datetime(x.get("date", "1900-01-01")), reverse=True
            )

            # 최근 15건만 표시
            recent_trades = all_trades[:15]

            for trade in recent_trades:
                symbol = trade.get("symbol", "")
                action = trade.get("action", "")
                price = trade.get("price", 0)
                date = trade.get("date", "")
                pnl = trade.get("pnl", 0)

                # 신호 정보에서 신뢰도 추출
                signal = trade.get("signal", {})
                confidence = signal.get("trading_signal", {}).get("confidence", 0)

                # 날짜 포맷팅
                if isinstance(date, str):
                    try:
                        formatted_date = pd.to_datetime(date).strftime("%Y-%m-%d")
                    except:
                        formatted_date = str(date)[:10]
                else:
                    formatted_date = str(date)[:10]

                # 수익률 표시 (매도 시만)
                pnl_str = f"{pnl*100:>+6.2f}%" if action == "SELL" and pnl != 0 else "-"

                # 거래 상태 표시
                if action == "BUY":
                    status = "매수"
                elif action == "SELL":
                    if pnl > 0:
                        status = "수익"
                    elif pnl < 0:
                        status = "손실"
                    else:
                        status = "무손익"
                else:
                    status = "기타"

                lines.append(
                    f"{formatted_date:<12} {symbol:<8} {action:<6} "
                    f"${price:>7.2f} {pnl_str:<10} {confidence*100:>6.1f}% {status:<8}"
                )

            if not recent_trades:
                lines.append("거래 이력이 없습니다.")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"최근 거래 테이블 생성 실패: {e}")
            return "최근 거래 테이블 생성 실패"

    def create_performance_comparison_summary(
        self,
        strategy_metrics: Dict[str, float],
        benchmark_metrics: Dict[str, float],
        period_info: Dict[str, str],
    ) -> str:
        """성과 비교 요약"""
        try:
            lines = []
            lines.append("🆚 성과 비교 요약")
            lines.append("-" * 60)

            strategy_return = strategy_metrics.get("cumulative_return", 0)
            benchmark_return = benchmark_metrics.get("cumulative_return", 0)
            outperformance = strategy_return - benchmark_return

            lines.append(f"📊 전략 수익률:      {strategy_return*100:>8.2f}%")
            lines.append(f"📊 Buy&Hold 수익률:  {benchmark_return*100:>8.2f}%")
            lines.append(f"🎯 초과 수익률:      {outperformance*100:>+8.2f}%")

            # 위험조정 수익률 비교
            strategy_sharpe = strategy_metrics.get("sharpe_ratio", 0)
            benchmark_sharpe = benchmark_metrics.get("sharpe_ratio", 0)

            lines.append(f"⚡ 전략 샤프비율:    {strategy_sharpe:>8.3f}")
            lines.append(f"⚡ 벤치마크 샤프비율: {benchmark_sharpe:>8.3f}")

            # 리스크 비교
            strategy_mdd = strategy_metrics.get("max_drawdown", 0)
            benchmark_mdd = benchmark_metrics.get("max_drawdown", 0)

            lines.append(f"📉 전략 최대낙폭:    {strategy_mdd*100:>8.2f}%")
            lines.append(f"📉 벤치마크 최대낙폭: {benchmark_mdd*100:>8.2f}%")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"성과 비교 요약 생성 실패: {e}")
            return "성과 비교 요약 생성 실패"
