#!/usr/bin/env python3
"""
통합 리포트 생성기
- 포트폴리오와 하이브리드 트레이더 결과를 일관성 있게 정리
- 표 양식 통일 및 가독성 향상
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class UnifiedReporter:
    """통합 리포트 생성기 - 일관성 있는 결과 출력"""

    def __init__(self):
        self.report_style = {
            "header_width": 100,
            "section_width": 80,
            "table_width": 120,
            "separator": "=",
            "sub_separator": "-",
        }

    def generate_comprehensive_report(
        self,
        analysis_results: Dict[str, Any],
        backtest_results: Optional[Dict[str, Any]] = None,
        market_regime: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        종합 리포트 생성

        Args:
            analysis_results: 분석 결과
            backtest_results: 백테스팅 결과 (선택사항)
            market_regime: 시장 체제 정보 (선택사항)

        Returns:
            포맷된 리포트 문자열
        """
        try:
            report_sections = []

            # 1. 헤더 섹션
            report_sections.append(self._create_header_section())

            # 2. 시장 체제 섹션
            if market_regime:
                report_sections.append(
                    self._create_market_regime_section(market_regime)
                )

            # 3. 포트폴리오 요약 섹션
            if "portfolio_analysis" in analysis_results:
                report_sections.append(
                    self._create_portfolio_summary_section(analysis_results)
                )

            # 4. 개별 종목 분석 섹션
            if "individual_results" in analysis_results:
                report_sections.append(
                    self._create_individual_analysis_section(analysis_results)
                )

            # 5. 백테스팅 결과 섹션
            if backtest_results:
                report_sections.append(self._create_backtest_section(backtest_results))

            # 6. 매매 신호 섹션
            if "trading_signals" in analysis_results:
                report_sections.append(
                    self._create_trading_signals_section(analysis_results)
                )

            # 7. 성과 비교 섹션
            if backtest_results and "benchmark_comparison" in backtest_results:
                report_sections.append(
                    self._create_performance_comparison_section(backtest_results)
                )

            # 8. 푸터 섹션
            report_sections.append(self._create_footer_section())

            return "\n\n".join(report_sections)

        except Exception as e:
            logger.error(f"종합 리포트 생성 실패: {e}")
            return f"리포트 생성 중 오류 발생: {e}"

    def _create_header_section(self) -> str:
        """헤더 섹션 생성"""
        lines = []
        lines.append(self.report_style["separator"] * self.report_style["header_width"])
        lines.append("🎯 HMM-Neural 하이브리드 트레이더 종합 분석 리포트")
        lines.append(self.report_style["separator"] * self.report_style["header_width"])
        lines.append(f"📅 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"🔧 분석 버전: v1.0.0")
        return "\n".join(lines)

    def _create_market_regime_section(self, market_regime: Dict[str, Any]) -> str:
        """시장 체제 섹션 생성"""
        lines = []
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )
        lines.append("📊 시장 체제 분석")
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )

        regime = market_regime.get("current_regime", "UNKNOWN")
        confidence = market_regime.get("confidence", 0)

        lines.append(f"🎯 현재 시장 체제: {regime}")
        lines.append(f"📈 신뢰도: {confidence:.1f}%")

        # 시장 체제별 설명
        regime_descriptions = {
            "BULLISH": "📈 상승장 - 적극적 매수 전략 권장",
            "BEARISH": "📉 하락장 - 방어적 포지션 또는 공매도 고려",
            "SIDEWAYS": "↔️ 횡보장 - 단기 매매 또는 현금 보유",
            "VOLATILE": "⚡ 변동성 장 - 리스크 관리 강화",
        }

        if regime in regime_descriptions:
            lines.append(f"💡 전략 방향: {regime_descriptions[regime]}")

        return "\n".join(lines)

    def _create_portfolio_summary_section(
        self, analysis_results: Dict[str, Any]
    ) -> str:
        """포트폴리오 요약 섹션 생성"""
        lines = []
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )
        lines.append("💼 포트폴리오 요약")
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )

        portfolio = analysis_results.get("portfolio_analysis", {})

        # 포트폴리오 점수
        portfolio_score = portfolio.get("portfolio_score", 0)
        portfolio_action = portfolio.get("portfolio_action", "UNKNOWN")

        lines.append(f"📊 포트폴리오 점수: {portfolio_score:.4f}")
        lines.append(f"🎯 포트폴리오 액션: {portfolio_action}")

        # 포트폴리오 구성
        weights = portfolio.get("weights", {})
        if weights:
            lines.append("")
            lines.append("📋 포트폴리오 구성:")
            lines.append(self.report_style["sub_separator"] * 50)

            # 비중 순으로 정렬
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for symbol, weight in sorted_weights:
                lines.append(f"   {symbol:<6} {weight*100:>6.2f}%")

        return "\n".join(lines)

    def _create_individual_analysis_section(
        self, analysis_results: Dict[str, Any]
    ) -> str:
        """개별 종목 분석 섹션 생성"""
        lines = []
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )
        lines.append("📈 개별 종목 상세 분석")
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )

        individual_results = analysis_results.get("individual_results", [])

        if not individual_results:
            lines.append("⚠️ 개별 종목 분석 결과가 없습니다.")
            return "\n".join(lines)

        # 통합 테이블 헤더
        lines.append(
            f"{'종목':<6} {'액션':<6} {'투자점수':<10} {'신뢰도':<8} {'22일예측':<10} {'66일예측':<10} {'우선순위':<8} {'리스크':<6}"
        )
        lines.append(
            self.report_style["sub_separator"] * self.report_style["table_width"]
        )

        # 개별 종목 정보
        for result in individual_results:
            symbol = result.get("symbol", "")
            action = result.get("action", "")
            investment_score = result.get("investment_score", {})
            score = investment_score.get("final_score", 0)
            confidence = investment_score.get("confidence", 0)

            # 예측값
            predictions = result.get("predictions", {})
            pred_22d = predictions.get("22d", {})
            pred_66d = predictions.get("66d", {})

            pred_22d_return = pred_22d.get("return", 0) * 100 if pred_22d else 0
            pred_66d_return = pred_66d.get("return", 0) * 100 if pred_66d else 0

            # 우선순위와 리스크
            priority = result.get("priority", 0)
            risk_level = result.get("risk_level", "MEDIUM")

            lines.append(
                f"{symbol:<6} {action:<6} {score:>8.3f} {confidence:>6.1f}% "
                f"{pred_22d_return:>8.1f}% {pred_66d_return:>8.1f}% {priority:>6} {risk_level:<6}"
            )

        return "\n".join(lines)

    def _create_backtest_section(self, backtest_results: Dict[str, Any]) -> str:
        """백테스팅 결과 섹션 생성"""
        lines = []
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )
        lines.append("📊 백테스팅 결과")
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )

        # 백테스팅 기간
        start_date = backtest_results.get("start_date", "")
        end_date = backtest_results.get("end_date", "")
        lines.append(f"📅 백테스팅 기간: {start_date} ~ {end_date}")

        # 포트폴리오 성과
        portfolio_perf = backtest_results.get("portfolio_performance", {})
        if portfolio_perf:
            lines.append("")
            lines.append("💼 포트폴리오 성과:")
            lines.append(self.report_style["sub_separator"] * 50)

            total_return = portfolio_perf.get("total_return", 0) * 100
            sharpe_ratio = portfolio_perf.get("sharpe_ratio", 0)
            max_drawdown = portfolio_perf.get("max_drawdown", 0) * 100
            volatility = portfolio_perf.get("volatility", 0) * 100
            total_trades = portfolio_perf.get("total_trades", 0)

            lines.append(f"   📈 총 수익률: {total_return:>8.2f}%")
            lines.append(f"   ⚡ 샤프 비율: {sharpe_ratio:>8.3f}")
            lines.append(f"   📉 최대 낙폭: {max_drawdown:>8.2f}%")
            lines.append(f"   📊 변동성: {volatility:>8.2f}%")
            lines.append(f"   🔄 총 거래 수: {total_trades:>8}")

        # 개별 종목 성과
        individual_perf = backtest_results.get("individual_performance", {})
        if individual_perf:
            lines.append("")
            lines.append("📈 개별 종목 성과:")
            lines.append(self.report_style["sub_separator"] * 70)
            lines.append(
                f"{'종목':<6} {'수익률':<10} {'B&H':<10} {'거래수':<8} {'승률':<8} {'평가':<8}"
            )
            lines.append(self.report_style["sub_separator"] * 70)

            for symbol, perf in individual_perf.items():
                total_return = perf.get("total_return", 0) * 100
                buy_hold_return = perf.get("buy_hold_return", 0) * 100
                trade_count = perf.get("trade_count", 0)
                win_rate = perf.get("win_rate", 0) * 100

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
                    f"{symbol:<6} {total_return:>8.2f}% {buy_hold_return:>8.2f}% "
                    f"{trade_count:>6} {win_rate:>6.1f}% {evaluation:<8}"
                )

        return "\n".join(lines)

    def _create_trading_signals_section(self, analysis_results: Dict[str, Any]) -> str:
        """매매 신호 섹션 생성"""
        lines = []
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )
        lines.append("🚀 매매 신호 요약")
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )

        # individual_results에서 trading_signal 추출
        individual_results = analysis_results.get("individual_results", [])
        trading_signals = []

        for result in individual_results:
            trading_signal = result.get("trading_signal", {})
            if trading_signal:
                trading_signals.append(trading_signal)

        if not trading_signals:
            lines.append("⚠️ 매매 신호가 없습니다.")
            return "\n".join(lines)

        # 액션별 통계
        action_counts = {}
        for signal in trading_signals:
            action = signal.get("action", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1

        lines.append("📊 액션별 분포:")
        for action, count in action_counts.items():
            lines.append(f"   {action}: {count}개")

        # 상위 신호 (우선순위 기준)
        lines.append("")
        lines.append("🎯 상위 매매 신호 (우선순위 기준):")
        lines.append(self.report_style["sub_separator"] * 80)
        lines.append(
            f"{'종목':<6} {'액션':<6} {'강도':<6} {'신뢰도':<8} {'투자점수':<10} {'진입타이밍':<10}"
        )
        lines.append(self.report_style["sub_separator"] * 80)

        # 우선순위 순으로 정렬 (상위 5개)
        sorted_signals = sorted(
            trading_signals, key=lambda x: x.get("execution_priority", 10)
        )[:5]

        for signal in sorted_signals:
            symbol = signal.get("symbol", "")
            action = signal.get("action", "")
            strength = signal.get("action_strength", 0)
            confidence = signal.get("confidence", 0)
            score = signal.get("score", 0)
            timing = signal.get("timing", {}).get("entry", {}).get("type", "NORMAL")

            lines.append(
                f"{symbol:<6} {action:<6} {strength:>5.2f} {confidence:>6.1f}% "
                f"{score:>8.3f} {timing:<10}"
            )

        return "\n".join(lines)

    def _create_performance_comparison_section(
        self, backtest_results: Dict[str, Any]
    ) -> str:
        """성과 비교 섹션 생성"""
        lines = []
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )
        lines.append("🆚 성과 비교: 전략 vs Buy & Hold")
        lines.append(
            self.report_style["separator"] * self.report_style["section_width"]
        )

        benchmark_comparison = backtest_results.get("benchmark_comparison", {})

        if not benchmark_comparison:
            lines.append("⚠️ 성과 비교 데이터가 없습니다.")
            return "\n".join(lines)

        # 성과 지표 비교 테이블
        lines.append(f"{'지표':<20} {'전략':<12} {'Buy&Hold':<12} {'차이':<12}")
        lines.append(self.report_style["sub_separator"] * 60)

        # 수익률 지표
        strategy_return = benchmark_comparison.get("strategy_return", 0) * 100
        benchmark_return = benchmark_comparison.get("benchmark_return", 0) * 100
        return_diff = strategy_return - benchmark_return

        lines.append(
            f"{'누적수익률':<20} {strategy_return:>10.2f}% {benchmark_return:>10.2f}% {return_diff:+10.2f}%"
        )

        # 샤프 비율
        strategy_sharpe = benchmark_comparison.get("strategy_sharpe", 0)
        benchmark_sharpe = benchmark_comparison.get("benchmark_sharpe", 0)
        sharpe_diff = strategy_sharpe - benchmark_sharpe

        lines.append(
            f"{'샤프비율':<20} {strategy_sharpe:>10.3f} {benchmark_sharpe:>10.3f} {sharpe_diff:+10.3f}"
        )

        # 최대 낙폭
        strategy_mdd = benchmark_comparison.get("strategy_mdd", 0) * 100
        benchmark_mdd = benchmark_comparison.get("benchmark_mdd", 0) * 100
        mdd_diff = strategy_mdd - benchmark_mdd

        lines.append(
            f"{'최대낙폭':<20} {strategy_mdd:>10.2f}% {benchmark_mdd:>10.2f}% {mdd_diff:+10.2f}%"
        )

        # 변동성
        strategy_vol = benchmark_comparison.get("strategy_vol", 0) * 100
        benchmark_vol = benchmark_comparison.get("benchmark_vol", 0) * 100
        vol_diff = strategy_vol - benchmark_vol

        lines.append(
            f"{'변동성':<20} {strategy_vol:>10.2f}% {benchmark_vol:>10.2f}% {vol_diff:+10.2f}%"
        )

        return "\n".join(lines)

    def _create_footer_section(self) -> str:
        """푸터 섹션 생성"""
        lines = []
        lines.append(self.report_style["separator"] * self.report_style["header_width"])
        lines.append("📝 분석 완료")
        lines.append("💡 결과 파일은 results/trader/ 폴더에 저장되었습니다.")
        lines.append("📊 로그 파일은 log/trader.log에서 확인할 수 있습니다.")
        lines.append(self.report_style["separator"] * self.report_style["header_width"])
        return "\n".join(lines)

    def format_percentage(self, value: float, decimal_places: int = 2) -> str:
        """백분율 포맷팅"""
        return f"{value*100:.{decimal_places}f}%"

    def format_number(self, value: float, decimal_places: int = 3) -> str:
        """숫자 포맷팅"""
        return f"{value:.{decimal_places}f}"
