#!/usr/bin/env python3
"""
Transaction Log 평균 보유일 분석 스크립트
"""

import re
from datetime import datetime
from collections import defaultdict


def analyze_holding_periods(log_file_path):
    """거래 로그에서 평균 보유일 분석"""

    with open(log_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 종목별로 분리
    symbols = re.findall(r"📊 (\w+) \(([^)]+)\)", content)

    print("🔍 평균 보유일 분석")
    print("=" * 60)

    all_holding_periods = []
    symbol_analysis = {}

    for symbol, strategy in symbols:
        print(f"\n📊 {symbol} ({strategy})")
        print("-" * 40)

        # 해당 종목의 거래 내역 추출
        symbol_pattern = rf"📊 {symbol} \([^)]+\)\n.*?거래 내역:\n(.*?)(?=\n📊|\n$)"
        symbol_match = re.search(symbol_pattern, content, re.DOTALL)

        if not symbol_match:
            print("  거래 내역 없음")
            continue

        trades_text = symbol_match.group(1)

        # 매수/매도 라인 추출
        trade_lines = re.findall(
            r"(\d{4}-\d{2}-\d{2}|\d+)\s+매수\s+([\d.]+)\s+[\d.]+\s*\n(\d{4}-\d{2}-\d{2}|\d+)\s+매도\s+([\d.]+)\s+[\d.]+\s+([-\d.]+)",
            trades_text,
        )

        if not trade_lines:
            print("  완료된 거래 없음")
            continue

        symbol_holding_periods = []

        for buy_date, buy_price, sell_date, sell_price, profit in trade_lines:
            try:
                # 날짜 처리
                if buy_date.isdigit():
                    # 인덱스 번호인 경우 (대략적인 날짜 추정)
                    buy_day = int(buy_date)
                    sell_day = int(sell_date)
                    holding_days = sell_day - buy_day
                else:
                    # 실제 날짜인 경우
                    buy_dt = datetime.strptime(buy_date, "%Y-%m-%d")
                    sell_dt = datetime.strptime(sell_date, "%Y-%m-%d")
                    holding_days = (sell_dt - buy_dt).days

                if holding_days > 0:
                    symbol_holding_periods.append(holding_days)
                    all_holding_periods.append(holding_days)

                    print(
                        f"  매수: {buy_date} ({buy_price}) → 매도: {sell_date} ({sell_price}) → 보유일: {holding_days}일 (수익률: {profit}%)"
                    )

            except Exception as e:
                print(f"  날짜 파싱 오류: {e}")
                continue

        if symbol_holding_periods:
            avg_holding = sum(symbol_holding_periods) / len(symbol_holding_periods)
            symbol_analysis[symbol] = {
                "trades": len(symbol_holding_periods),
                "avg_holding": avg_holding,
                "min_holding": min(symbol_holding_periods),
                "max_holding": max(symbol_holding_periods),
                "holding_periods": symbol_holding_periods,
            }
            print(
                f"  📈 평균 보유일: {avg_holding:.1f}일 (거래 수: {len(symbol_holding_periods)})"
            )
        else:
            print("  완료된 거래 없음")

    # 전체 통계
    if all_holding_periods:
        print(f"\n" + "=" * 60)
        print("📊 전체 통계")
        print("=" * 60)
        print(f"총 거래 수: {len(all_holding_periods)}")
        print(
            f"평균 보유일: {sum(all_holding_periods) / len(all_holding_periods):.1f}일"
        )
        print(f"최소 보유일: {min(all_holding_periods)}일")
        print(f"최대 보유일: {max(all_holding_periods)}일")

        # 보유일 분포
        print(f"\n📊 보유일 분포:")
        short_term = len([d for d in all_holding_periods if d <= 7])
        medium_term = len([d for d in all_holding_periods if 7 < d <= 30])
        long_term = len([d for d in all_holding_periods if d > 30])

        print(
            f"  단기 (1-7일): {short_term}건 ({short_term/len(all_holding_periods)*100:.1f}%)"
        )
        print(
            f"  중기 (8-30일): {medium_term}건 ({medium_term/len(all_holding_periods)*100:.1f}%)"
        )
        print(
            f"  장기 (31일+): {long_term}건 ({long_term/len(all_holding_periods)*100:.1f}%)"
        )

        # 종목별 평균 보유일 순위
        print(f"\n📊 종목별 평균 보유일 순위:")
        sorted_symbols = sorted(
            symbol_analysis.items(), key=lambda x: x[1]["avg_holding"], reverse=True
        )
        for i, (symbol, data) in enumerate(sorted_symbols, 1):
            print(
                f"  {i:2d}. {symbol}: {data['avg_holding']:.1f}일 (거래: {data['trades']}건)"
            )

    return symbol_analysis, all_holding_periods


if __name__ == "__main__":
    log_file = "log/transaction_test_swing_20250721_20250721_215748.log"
    analyze_holding_periods(log_file)
