#!/usr/bin/env python3
"""
일관된 출력 포맷팅 시스템
- 모든 출력 결과를 통일된 스타일로 포맷팅
- 테이블 양식 통일 및 가독성 향상
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FormattedOutput:
    """일관된 출력 포맷팅 시스템"""

    def __init__(self):
        self.style = {
            "header_width": 100,
            "section_width": 80,
            "table_width": 120,
            "separator": "=",
            "sub_separator": "-",
            "emoji": {
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
                "info": "📊",
                "money": "💰",
                "chart": "📈",
                "portfolio": "💼",
                "trade": "🔄",
                "risk": "📉",
                "performance": "⚡",
                "summary": "🎯",
                "comparison": "🆚",
                "analysis": "🔍",
                "report": "📋",
                "time": "📅",
                "settings": "🔧",
            },
        }

    def format_header(self, title: str, subtitle: str = None) -> str:
        """헤더 포맷팅"""
        lines = []
        lines.append(self.style["separator"] * self.style["header_width"])
        lines.append(f"🎯 {title}")
        lines.append(self.style["separator"] * self.style["header_width"])

        if subtitle:
            lines.append(f"📅 {subtitle}")
            lines.append("")

        return "\n".join(lines)

    def format_section_header(self, title: str) -> str:
        """섹션 헤더 포맷팅"""
        lines = []
        lines.append(self.style["separator"] * self.style["section_width"])
        lines.append(f"{title}")
        lines.append(self.style["separator"] * self.style["section_width"])
        return "\n".join(lines)

    def format_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> str:
        """포트폴리오 요약 테이블 포맷팅"""
        lines = []
        lines.append("🎯 포트폴리오 전체 성과 요약")
        lines.append(self.style["sub_separator"] * 80)

        # 기본 성과 지표
        total_return = portfolio_data.get("total_return", 0) * 100
        total_trades = portfolio_data.get("total_trades", 0)
        sharpe_ratio = portfolio_data.get("sharpe_ratio", 0)
        volatility = portfolio_data.get("volatility", 0) * 100
        max_drawdown = portfolio_data.get("max_drawdown", 0) * 100

        lines.append(f"📈 총 수익률:             {total_return:>8.2f}%")
        lines.append(f"📊 총 거래 횟수:             {total_trades:>8.0f} 회")
        lines.append(f"⚡ 샤프 비율:           {sharpe_ratio:>8.3f}")
        lines.append(f"📉 변동성:               {volatility:>8.2f}%")
        lines.append(f"📉 최대 낙폭:          {max_drawdown:>8.2f}%")

        # 포트폴리오 구성
        weights = portfolio_data.get("weights", {})
        if weights:
            lines.append("")
            lines.append("💼 포트폴리오 구성:")
            for symbol, weight in sorted(
                weights.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"   {symbol}: {weight*100:>6.2f}%")

        return "\n".join(lines)

    def format_individual_performance_table(
        self, individual_data: Dict[str, Any]
    ) -> str:
        """개별 종목 성과 테이블 포맷팅"""
        lines = []
        lines.append("📈 개별 종목 상세 성과")
        lines.append(self.style["sub_separator"] * 120)

        # 헤더
        lines.append(
            f"{'종목':<8} {'비중':<8} {'수익률':<10} {'B&H':<10} {'거래수':<8} {'승률':<8} "
            f"{'기여도':<10} {'평가':<8}"
        )
        lines.append(self.style["sub_separator"] * 120)

        # 데이터 행
        for symbol, data in individual_data.items():
            weight = data.get("weight", 0) * 100
            total_return = data.get("total_return", 0) * 100
            buy_hold_return = data.get("buy_hold_return", 0) * 100
            trade_count = data.get("trade_count", 0)
            win_rate = data.get("win_rate", 0) * 100
            contribution = total_return * (data.get("weight", 0))

            # 평가
            if total_return > 5:
                evaluation = "우수"
            elif total_return > 0:
                evaluation = "양호"
            elif total_return > -5:
                evaluation = "보통"
            else:
                evaluation = "부진"

            lines.append(
                f"{symbol:<8} {weight:>6.1f}% {total_return:>8.2f}% "
                f"{buy_hold_return:>8.2f}% {trade_count:>6.0f}회 {win_rate:>6.1f}% "
                f"{contribution:>8.2f}% {evaluation:<8}"
            )

        return "\n".join(lines)

    def format_trading_summary_table(self, trading_data: Dict[str, Any]) -> str:
        """매매 내역 요약 테이블 포맷팅"""
        lines = []
        lines.append("📋 매매 내역 요약")
        lines.append(self.style["sub_separator"] * 80)

        # 전체 통계
        total_buy_count = trading_data.get("total_buy_count", 0)
        total_sell_count = trading_data.get("total_sell_count", 0)
        total_profitable_trades = trading_data.get("total_profitable_trades", 0)
        total_trades = trading_data.get("total_trades", 0)
        overall_win_rate = trading_data.get("overall_win_rate", 0) * 100

        lines.append(f"📊 총 매수 거래:             {total_buy_count:>8.0f} 회")
        lines.append(f"📊 총 매도 거래:             {total_sell_count:>8.0f} 회")
        lines.append(f"💰 수익 거래:               {total_profitable_trades:>8.0f} 회")
        lines.append(
            f"📉 손실 거래:               {total_trades - total_profitable_trades:>8.0f} 회"
        )
        lines.append(f"🎯 전체 승률:             {overall_win_rate:>8.1f}%")

        # 종목별 매매 현황
        symbol_trading = trading_data.get("symbol_trading", {})
        if symbol_trading:
            lines.append("")
            lines.append("종목별 매매 현황:")
            lines.append(
                f"{'종목':<8} {'매수':<6} {'매도':<6} {'수익거래':<8} {'승률':<8}"
            )
            lines.append(self.style["sub_separator"] * 50)

            for symbol, data in symbol_trading.items():
                buy_count = data.get("buy_count", 0)
                sell_count = data.get("sell_count", 0)
                profitable = data.get("profitable", 0)
                win_rate = data.get("win_rate", 0) * 100

                lines.append(
                    f"{symbol:<8} {buy_count:>4.0f}회 {sell_count:>4.0f}회 "
                    f"{profitable:>6.0f}회 {win_rate:>6.1f}%"
                )

        return "\n".join(lines)

    def format_recent_trades_table(self, trades_data: List[Dict[str, Any]]) -> str:
        """최근 거래 이력 테이블 포맷팅"""
        lines = []
        lines.append("📋 최근 거래 이력 (최대 15건)")
        lines.append(self.style["sub_separator"] * 120)

        # 헤더
        lines.append(
            f"{'날짜':<12} {'종목':<8} {'액션':<6} {'가격':<10} {'수익률':<10} "
            f"{'신뢰도':<8} {'상태':<8}"
        )
        lines.append(self.style["sub_separator"] * 120)

        # 거래 데이터
        for trade in trades_data[:15]:  # 최대 15건
            date = trade.get("date", "")
            symbol = trade.get("symbol", "")
            action = trade.get("action", "")
            price = trade.get("price", 0)
            pnl = trade.get("pnl", 0)
            confidence = trade.get("confidence", 0) * 100

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
                f"${price:>7.2f} {pnl_str:<10} {confidence:>6.1f}% {status:<8}"
            )

        return "\n".join(lines)

    def format_performance_comparison_table(
        self, comparison_data: Dict[str, Any]
    ) -> str:
        """성과 비교 테이블 포맷팅"""
        lines = []
        lines.append("🆚 성과 비교: 신경망 전략 vs Buy & Hold")
        lines.append(self.style["sub_separator"] * 80)

        # 지표별 비교
        lines.append(
            f"{'지표':<20} {'신경망 전략':<15} {'Buy & Hold':<15} {'차이':<15}"
        )
        lines.append(self.style["sub_separator"] * 80)

        # 수익률 지표
        strategy_return = comparison_data.get("strategy_return", 0) * 100
        benchmark_return = comparison_data.get("benchmark_return", 0) * 100
        return_diff = strategy_return - benchmark_return

        lines.append(
            f"{'누적수익률':<20} {strategy_return:>13.2f}% {benchmark_return:>13.2f}% {return_diff:+13.2f}%"
        )

        # 샤프 비율
        strategy_sharpe = comparison_data.get("strategy_sharpe", 0)
        benchmark_sharpe = comparison_data.get("benchmark_sharpe", 0)
        sharpe_diff = strategy_sharpe - benchmark_sharpe

        lines.append(
            f"{'샤프비율':<20} {strategy_sharpe:>13.3f} {benchmark_sharpe:>13.3f} {sharpe_diff:+13.3f}"
        )

        # 변동성
        strategy_vol = comparison_data.get("strategy_volatility", 0) * 100
        benchmark_vol = comparison_data.get("benchmark_volatility", 0) * 100
        vol_diff = strategy_vol - benchmark_vol

        lines.append(
            f"{'변동성':<20} {strategy_vol:>13.2f}% {benchmark_vol:>13.2f}% {vol_diff:+13.2f}%"
        )

        # 최대낙폭
        strategy_mdd = comparison_data.get("strategy_max_drawdown", 0) * 100
        benchmark_mdd = comparison_data.get("benchmark_max_drawdown", 0) * 100
        mdd_diff = strategy_mdd - benchmark_mdd

        lines.append(
            f"{'최대낙폭':<20} {strategy_mdd:>13.2f}% {benchmark_mdd:>13.2f}% {mdd_diff:+13.2f}%"
        )

        return "\n".join(lines)

    def format_market_regime_info(self, regime_data: Dict[str, Any]) -> str:
        """시장 체제 정보 포맷팅"""
        lines = []
        lines.append("📊 현재 시장 상황")
        lines.append(self.style["sub_separator"] * 50)

        regime = regime_data.get("regime", "UNKNOWN")
        confidence = regime_data.get("confidence", 0) * 100
        portfolio_score = regime_data.get("portfolio_score", 0)
        portfolio_action = regime_data.get("portfolio_action", "UNKNOWN")
        signal_distribution = regime_data.get("signal_distribution", {})

        lines.append(f"시장 체제: {regime} (신뢰도: {confidence:.1f}%)")
        lines.append(f"포트폴리오 점수: {portfolio_score:.4f}")
        lines.append(f"포트폴리오 액션: {portfolio_action}")

        if signal_distribution:
            buy_count = signal_distribution.get("BUY", 0)
            hold_count = signal_distribution.get("HOLD", 0)
            sell_count = signal_distribution.get("SELL", 0)
            lines.append(
                f"신호 분포: BUY: {buy_count}개 | HOLD: {hold_count}개 | SELL: {sell_count}개"
            )

        return "\n".join(lines)

    def format_final_positions_table(
        self, positions_data: Dict[str, Any], end_date: str
    ) -> str:
        """최종 보유현황 테이블 포맷팅"""
        lines = []
        lines.append(f"💼 최종 보유현황 ({end_date} 기준)")
        lines.append(self.style["sub_separator"] * 80)

        # 헤더
        lines.append(
            f"{'종목':<8} {'보유여부':<10} {'최종거래일':<12} {'최종거래':<10} {'거래가격':<12}"
        )
        lines.append(self.style["sub_separator"] * 80)

        # 포지션 데이터
        for symbol, data in positions_data.items():
            position_status = data.get("position_status", "없음")
            last_date = data.get("last_date", "-")
            last_action = data.get("last_action", "-")
            last_price = data.get("last_price", 0)

            # 날짜 포맷팅
            if isinstance(last_date, str) and last_date != "-":
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

    def format_comprehensive_report(
        self,
        portfolio_data: Dict[str, Any],
        individual_data: Dict[str, Any],
        trading_data: Dict[str, Any],
        trades_data: List[Dict[str, Any]],
        comparison_data: Dict[str, Any],
        regime_data: Dict[str, Any],
        positions_data: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> str:
        """종합 리포트 포맷팅"""
        report_sections = []

        # 1. 헤더
        report_sections.append(
            self.format_header(
                "Quant-Racoon 트레이더 실행 결과 요약",
                f"백테스팅 기간: {start_date} ~ {end_date}",
            )
        )

        # 2. 포트폴리오 요약
        report_sections.append(self.format_portfolio_summary(portfolio_data))
        report_sections.append("")

        # 3. 개별 종목 성과
        report_sections.append(
            self.format_individual_performance_table(individual_data)
        )
        report_sections.append("")

        # 4. 매매 내역 요약
        report_sections.append(self.format_trading_summary_table(trading_data))
        report_sections.append("")

        # 5. 최근 거래 이력
        report_sections.append(self.format_recent_trades_table(trades_data))
        report_sections.append("")

        # 6. 성과 비교
        report_sections.append(
            self.format_performance_comparison_table(comparison_data)
        )
        report_sections.append("")

        # 7. 시장 체제 정보
        report_sections.append(self.format_market_regime_info(regime_data))
        report_sections.append("")

        # 8. 최종 보유현황
        report_sections.append(
            self.format_final_positions_table(positions_data, end_date)
        )
        report_sections.append("")

        # 9. 생성된 파일 정보
        report_sections.append(self.format_generated_files_info())

        return "\n".join(report_sections)

    def format_generated_files_info(self) -> str:
        """생성된 파일 정보 포맷팅"""
        lines = []
        lines.append("📄 생성된 파일")
        lines.append(self.style["sub_separator"] * 50)
        lines.append("• 매매신호 CSV: results/trader/trading_signals_*.csv")
        lines.append("• 요약 정보 JSON: results/trader/trading_signals_summary_*.json")
        lines.append("• 로그 파일: log/trader.log")
        return "\n".join(lines)

    def format_quick_summary(self, regime_data: Dict[str, Any]) -> str:
        """빠른 요약 포맷팅"""
        lines = []
        lines.append("=== 빠른 요약 ===")

        regime = regime_data.get("regime", "UNKNOWN")
        confidence = regime_data.get("confidence", 0) * 100
        portfolio_score = regime_data.get("portfolio_score", 0)
        portfolio_action = regime_data.get("portfolio_action", "UNKNOWN")
        signal_distribution = regime_data.get("signal_distribution", {})

        lines.append(f"시장 체제: {regime} (신뢰도: {confidence:.1f}%)")
        lines.append(f"포트폴리오 점수: {portfolio_score:.4f}")
        lines.append(f"포트폴리오 액션: {portfolio_action}")

        if signal_distribution:
            buy_count = signal_distribution.get("BUY", 0)
            hold_count = signal_distribution.get("HOLD", 0)
            sell_count = signal_distribution.get("SELL", 0)
            lines.append(
                f"신호 분포: BUY: {buy_count}개 | HOLD: {hold_count}개 | SELL: {sell_count}개"
            )

        lines.append("")
        lines.append("전체 결과는 results/trader 폴더에 저장되었습니다.")

        return "\n".join(lines)

    def format_detailed_trading_signals_table(
        self, individual_results: List[Dict[str, Any]]
    ) -> str:
        """상세한 매매신호 테이블 포맷팅"""
        lines = []
        lines.append("🚀 상세 매매신호 분석 결과")
        lines.append(self.style["sub_separator"] * 150)

        # 헤더
        header = (
            f"{'종목':<6} {'액션':<12} {'강도':<6} {'점수':<7} {'신뢰도':<7} {'포지션':<7} "
            f"{'우선순위':<8} {'진입타이밍':<10} {'손절선':<7} {'이익실현':<12} {'리스크':<8} {'체제':<8}"
        )
        lines.append(header)
        lines.append(self.style["sub_separator"] * 150)

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

            lines.append(
                f"{symbol:<6} {action:<12} {action_strength:<6.2f} {score:<7.3f} {confidence:<7.1%} {position_size:<7.1%} "
                f"{execution_priority:<8} {entry_timing:<10} {stop_loss:<7.1%} {take_profit_str:<12} {risk_level:<8} {regime:<8}"
            )

        lines.append(self.style["sub_separator"] * 150)

        # 액션별 통계
        signal_counts = {}
        for result in sorted_results:
            action = result.get("trading_signal", {}).get("action", "HOLD")
            signal_counts[action] = signal_counts.get(action, 0) + 1

        lines.append(f"\n📊 액션별 통계:")
        for action, count in signal_counts.items():
            if count > 0:
                lines.append(f"   {action}: {count}개")

        return "\n".join(lines)

    def format_individual_signal_details(
        self, individual_results: List[Dict[str, Any]]
    ) -> str:
        """개별 종목 상세 권고사항 포맷팅"""
        lines = []
        lines.append("📋 개별 종목 상세 권고사항")
        lines.append(self.style["sub_separator"] * 100)

        for result in individual_results:
            trading_signal = result.get("trading_signal", {})
            if not trading_signal:
                continue

            symbol = trading_signal.get("symbol", "N/A")
            lines.append(f"\n📋 {symbol} 상세 신호 분석")
            lines.append("-" * 80)

            # 기본 신호 정보
            action = trading_signal.get("action", "HOLD")
            action_strength = trading_signal.get("action_strength", 0.0)
            score = trading_signal.get("score", 0.0)
            confidence = trading_signal.get("confidence", 0.0)

            lines.append(f"🎯 매매액션: {action} (강도: {action_strength:.2f})")
            lines.append(f"📊 투자점수: {score:.4f} (신뢰도: {confidence:.1%})")
            lines.append(
                f"💰 포지션크기: {trading_signal.get('position_size', 0.0):.1%}"
            )
            lines.append(
                f"⚡ 실행우선순위: {trading_signal.get('execution_priority', 10)}"
            )

            # 타이밍 정보
            timing = trading_signal.get("timing", {})
            entry_timing = timing.get("entry", {})
            exit_timing = timing.get("exit", {})

            lines.append(f"\n⏰ 진입 타이밍:")
            lines.append(f"   타입: {entry_timing.get('type', 'WAIT')}")
            lines.append(f"   긴급도: {entry_timing.get('urgency', 'NONE')}")

            # 분할 진입 계획
            entry_phases = entry_timing.get("entry_phases")
            if entry_phases:
                lines.append(f"   분할 진입 계획:")
                for phase in entry_phases:
                    lines.append(
                        f"     {phase['phase']}단계: {phase['ratio']:.1%} ({phase['timing']})"
                    )

            lines.append(f"\n🚪 청산 타이밍:")
            lines.append(f"   손절선: {exit_timing.get('stop_loss', 0.0):.1%}")

            take_profit_levels = exit_timing.get("take_profit_levels", [])
            if take_profit_levels:
                lines.append(
                    f"   이익실현: {' → '.join([f'{tp:.1%}' for tp in take_profit_levels])}"
                )

            lines.append(
                f"   트레일링스탑: {exit_timing.get('trailing_stop', 0.0):.1%}"
            )
            lines.append(f"   최대보유기간: {exit_timing.get('max_holding_days', 0)}일")

            # 리스크 관리
            risk_management = trading_signal.get("risk_management", {})
            lines.append(f"\n⚠️  리스크 관리:")
            lines.append(
                f"   리스크 레벨: {risk_management.get('risk_level', 'MEDIUM')}"
            )

            warnings = risk_management.get("warnings", [])
            if warnings:
                lines.append(f"   경고사항:")
                for warning in warnings:
                    lines.append(f"     • {warning}")

            mitigation_strategies = risk_management.get("mitigation_strategies", [])
            if mitigation_strategies:
                lines.append(f"   완화전략:")
                for strategy in mitigation_strategies:
                    lines.append(f"     • {strategy}")

            # 권고사항
            recommendations = trading_signal.get("recommendations", {})
            primary_rec = recommendations.get("primary_recommendation", "")
            if primary_rec:
                lines.append(f"\n💡 주요 권고사항:")
                lines.append(f"   {primary_rec}")

            regime_advice = recommendations.get("regime_advice", "")
            if regime_advice:
                lines.append(f"   시장체제 조언: {regime_advice}")

            timing_advice = recommendations.get("timing_advice", [])
            if timing_advice:
                lines.append(f"   타이밍 조언:")
                for advice in timing_advice:
                    lines.append(f"     • {advice}")

            cautions = recommendations.get("cautions", [])
            if cautions:
                lines.append(f"   주의사항:")
                for caution in cautions:
                    lines.append(f"     • {caution}")

            lines.append("-" * 80)

        return "\n".join(lines)

    def format_neural_predictions_table(
        self, individual_results: List[Dict[str, Any]]
    ) -> str:
        """신경망 예측 결과 테이블 포맷팅"""
        lines = []
        lines.append("🎯 멀티타겟 신경망 예측 결과")
        lines.append(self.style["sub_separator"] * 80)

        # 예측 기간 정보
        lines.append("📈 예측 기간: [22, 66]일 후")
        lines.append(f"📈 분석 종목: {len(individual_results)}개")
        lines.append("-" * 80)

        # 헤더
        lines.append(
            f"{'종목':<8} {'22일수익률':<10} {'22일변동성':<10} {'66일수익률':<10} {'66일변동성':<10} {'투자점수':<8} {'신뢰도':<8} {'액션':<12}"
        )
        lines.append("-" * 80)

        # 데이터 출력
        for result in individual_results:
            symbol = result.get("symbol", "N/A")
            neural_prediction = result.get("neural_prediction", {})
            investment_score = result.get("investment_score", {})
            trading_signal = result.get("trading_signal", {})

            # 투자점수는 investment_score에서 가져오기 (final_score)
            final_score = (
                investment_score.get("final_score", 0.0) if investment_score else 0.0
            )
            confidence = result.get("confidence", 0.0)
            action = trading_signal.get("action", "HOLD") if trading_signal else "HOLD"

            # 멀티타겟 예측값 추출
            if isinstance(neural_prediction, dict):
                target_22d = neural_prediction.get("target_22d", 0.0)
                target_66d = neural_prediction.get("target_66d", 0.0)
                sigma_22d = neural_prediction.get("sigma_22d", 0.0)
                sigma_66d = neural_prediction.get("sigma_66d", 0.0)
            else:
                target_22d = target_66d = sigma_22d = sigma_66d = 0.0

            lines.append(
                f"{symbol:<8} {target_22d:>9.1%} {sigma_22d:>9.1%} "
                f"{target_66d:>9.1%} {sigma_66d:>9.1%} "
                f"{final_score:>7.3f} {confidence:>7.1%} {action:<12}"
            )

        lines.append("=" * 80)
        lines.append(
            "📝 Universal 모델: Universal 모델은 개별 예측과 차원이 달라 표시하지 않음"
        )
        lines.append("=" * 80)

        return "\n".join(lines)

    def format_comprehensive_trading_report(
        self,
        market_regime: Dict[str, Any],
        portfolio_summary: Dict[str, Any],
        individual_results: List[Dict[str, Any]],
        start_date: str = None,
        end_date: str = None,
    ) -> str:
        """통합 매매 리포트 포맷팅"""
        lines = []

        # 1. 헤더
        if start_date and end_date:
            lines.append(
                self.format_header(
                    "Quant-Racoon 트레이더 실행 결과 요약",
                    f"백테스팅 기간: {start_date} ~ {end_date}",
                )
            )
        else:
            lines.append(self.format_header("Quant-Racoon 트레이더 실행 결과 요약"))

        # 2. 시장 체제 정보
        regime = market_regime.get("regime", "UNKNOWN")
        confidence = market_regime.get("confidence", 0.0)
        lines.append(f"📊 시장 체제: {regime} (신뢰도: {confidence:.1%})")

        # 3. 포트폴리오 요약
        portfolio_score = portfolio_summary.get("portfolio_score", 0.0)
        portfolio_action = portfolio_summary.get("portfolio_action", "N/A")
        lines.append(f"📈 포트폴리오 점수: {portfolio_score:.4f}")
        lines.append(f"🎯 포트폴리오 액션: {portfolio_action}")

        # 4. 신경망 예측 결과 테이블
        if individual_results:
            lines.append("")
            lines.append(self.format_neural_predictions_table(individual_results))

        # 5. 상세 매매신호 테이블
        if individual_results:
            lines.append("")
            lines.append(self.format_detailed_trading_signals_table(individual_results))

        # 6. 개별 종목 상세 권고사항
        if individual_results:
            lines.append("")
            lines.append(self.format_individual_signal_details(individual_results))

        return "\n".join(lines)


# 전역 인스턴스
formatted_output = FormattedOutput()
