#!/usr/bin/env python3
"""
예측값 범위 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)


def test_predictions():
    """예측값 범위 테스트"""

    # 데이터 로드
    cache_dir = Path("data/trader")
    symbols = ["AAPL", "META", "QQQ", "SPY"]

    for symbol in symbols:
        print(f"\n🔍 {symbol} 예측값 테스트")
        print("=" * 50)

        # 파일 찾기
        pattern = f"{symbol}_daily_*.csv"
        files = list(cache_dir.glob(pattern))

        if not files:
            print(f"❌ {symbol} 데이터 파일 없음")
            continue

        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        print(f"📁 파일: {latest_file.name}")

        # 데이터 로드
        data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        print(f"📊 데이터 기간: {data.index[0]} ~ {data.index[-1]} ({len(data)}일)")

        # 예측값 계산 (optimize_threshold.py와 동일한 로직)
        close = data["close"]

        # RSI 계산
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

        # 예측값 통계
        predictions_array = np.array(predictions)
        print(f"📈 예측값 통계:")
        print(
            f"   - 범위: {predictions_array.min():.4f} ~ {predictions_array.max():.4f}"
        )
        print(f"   - 평균: {predictions_array.mean():.4f}")
        print(f"   - 표준편차: {predictions_array.std():.4f}")

        # Threshold 범위별 신호 발생 횟수
        thresholds = {
            "strong_buy": 0.6,  # 수정된 범위의 중간값
            "buy": 0.35,
            "hold_upper": 0.15,
            "hold_lower": -0.15,
            "sell": -0.35,
            "strong_sell": -0.55,
        }

        print(f"\n🎯 Threshold별 신호 발생 횟수:")
        for name, threshold in thresholds.items():
            if "buy" in name or "strong_buy" in name:
                count = sum(1 for p in predictions if p >= threshold)
            elif "sell" in name or "strong_sell" in name:
                count = sum(1 for p in predictions if p <= threshold)
            else:
                count = sum(1 for p in predictions if p > threshold)

            print(f"   - {name} ({threshold:+.3f}): {count}회")

        # 실제 신호 분포
        print(f"\n📊 신호 분포:")
        strong_buy_count = sum(1 for p in predictions if p >= thresholds["strong_buy"])
        buy_count = sum(
            1 for p in predictions if thresholds["buy"] <= p < thresholds["strong_buy"]
        )
        hold_count = sum(
            1
            for p in predictions
            if thresholds["hold_lower"] < p < thresholds["hold_upper"]
        )
        sell_count = sum(
            1 for p in predictions if thresholds["sell"] < p <= thresholds["hold_lower"]
        )
        strong_sell_count = sum(
            1 for p in predictions if p <= thresholds["strong_sell"]
        )

        print(f"   - STRONG_BUY: {strong_buy_count}회")
        print(f"   - BUY: {buy_count}회")
        print(f"   - HOLD: {hold_count}회")
        print(f"   - SELL: {sell_count}회")
        print(f"   - STRONG_SELL: {strong_sell_count}회")
        print(
            f"   - 총 신호: {strong_buy_count + buy_count + sell_count + strong_sell_count}회"
        )


if __name__ == "__main__":
    test_predictions()
