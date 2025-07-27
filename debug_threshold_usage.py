#!/usr/bin/env python3
"""
Threshold 사용 디버깅 스크립트
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

from src.agent.trader import HybridTrader
from src.actions.optimize_threshold import ThresholdOptimizer


def debug_threshold_usage():
    """Threshold 사용 현황 디버깅"""

    print("🔍 Threshold 사용 현황 디버깅")
    print("=" * 60)

    # 1. 설정 파일 확인
    print("1️⃣ 설정 파일 확인")
    with open("config/config_trader.json", "r") as f:
        config = json.load(f)

    default_thresholds = config.get("signal_generation", {}).get("thresholds", {})
    print(f"   - 기본 threshold: {default_thresholds}")

    # 2. 저장된 최적화 결과 확인
    print("\n2️⃣ 저장된 최적화 결과 확인")

    # 새로운 형식 확인
    new_threshold_file = Path("results/trader/optimized_thresholds.json")
    if new_threshold_file.exists():
        with open(new_threshold_file, "r") as f:
            new_data = json.load(f)
        new_thresholds = new_data.get("thresholds", {})
        timestamp = new_data.get("timestamp", "unknown")
        print(f"   ✅ 새로운 형식 파일 발견:")
        print(f"      - 파일: {new_threshold_file}")
        print(f"      - 생성 시간: {timestamp}")
        print(f"      - Threshold: {new_thresholds}")
    else:
        print(f"   ❌ 새로운 형식 파일 없음: {new_threshold_file}")
        new_thresholds = {}

    # 기존 형식 확인
    results_dir = Path("results/trader")
    old_files = list(results_dir.glob("threshold_optimization_final_*.json"))
    if old_files:
        latest_old_file = max(old_files, key=lambda x: x.stat().st_mtime)
        with open(latest_old_file, "r") as f:
            old_data = json.load(f)
        old_thresholds = old_data.get("best_thresholds", {})
        print(f"   ✅ 기존 형식 파일 발견:")
        print(f"      - 파일: {latest_old_file}")
        print(f"      - Threshold: {old_thresholds}")
    else:
        print(f"   ❌ 기존 형식 파일 없음")
        old_thresholds = {}

    # 3. ThresholdOptimizer로 로드 테스트
    print("\n3️⃣ ThresholdOptimizer 로드 테스트")
    optimizer = ThresholdOptimizer(config)
    loaded_thresholds = optimizer.load_optimized_thresholds()
    print(f"   - 로드된 threshold: {loaded_thresholds}")

    # 4. HybridTrader 초기화 및 threshold 확인
    print("\n4️⃣ HybridTrader threshold 확인")
    try:
        trader = HybridTrader("config/config_trader.json", analysis_mode=True)

        # 신호 생성기 threshold 확인
        if hasattr(trader, "signal_generator") and trader.signal_generator:
            current_thresholds = trader.signal_generator.signal_thresholds
            print(f"   - 신호 생성기 threshold: {current_thresholds}")
        else:
            print(f"   ❌ 신호 생성기가 초기화되지 않음")

        # 최적화된 threshold 로드 시도
        load_success = trader.load_optimized_thresholds()
        print(f"   - 최적화된 threshold 로드 성공: {load_success}")

        # 로드 후 threshold 확인
        if hasattr(trader, "signal_generator") and trader.signal_generator:
            updated_thresholds = trader.signal_generator.signal_thresholds
            print(f"   - 업데이트된 threshold: {updated_thresholds}")

            # 변경 여부 확인
            if current_thresholds != updated_thresholds:
                print(f"   ✅ Threshold가 업데이트됨!")
            else:
                print(f"   ⚠️ Threshold가 변경되지 않음")
        else:
            print(f"   ❌ 신호 생성기를 찾을 수 없음")

    except Exception as e:
        print(f"   ❌ HybridTrader 초기화 실패: {e}")

    # 5. 실제 신호 생성 테스트
    print("\n5️⃣ 실제 신호 생성 테스트")
    try:
        # 테스트용 예측값
        test_predictions = [0.8, 0.3, -0.2, -0.6, 0.1]

        print(f"   - 테스트 예측값: {test_predictions}")

        # 기본 threshold로 신호 생성
        print(f"   - 기본 threshold로 신호 생성:")
        for pred in test_predictions:
            if pred >= default_thresholds.get("strong_buy", 0.6):
                signal = "STRONG_BUY"
            elif pred >= default_thresholds.get("buy", 0.3):
                signal = "BUY"
            elif pred <= default_thresholds.get("strong_sell", -0.6):
                signal = "STRONG_SELL"
            elif pred <= default_thresholds.get("sell", -0.3):
                signal = "SELL"
            else:
                signal = "HOLD"
            print(f"     {pred:+.2f} -> {signal}")

        # 최적화된 threshold로 신호 생성 (새로운 형식)
        if new_thresholds:
            print(f"   - 최적화된 threshold로 신호 생성:")
            for pred in test_predictions:
                if pred >= new_thresholds.get("strong_buy", 0.6):
                    signal = "STRONG_BUY"
                elif pred >= new_thresholds.get("buy", 0.3):
                    signal = "BUY"
                elif pred <= new_thresholds.get("strong_sell", -0.6):
                    signal = "STRONG_SELL"
                elif pred <= new_thresholds.get("sell", -0.3):
                    signal = "SELL"
                else:
                    signal = "HOLD"
                print(f"     {pred:+.2f} -> {signal}")

    except Exception as e:
        print(f"   ❌ 신호 생성 테스트 실패: {e}")

    # 6. 파일 목록 확인
    print("\n6️⃣ results/trader 디렉토리 파일 목록")
    results_dir = Path("results/trader")
    if results_dir.exists():
        files = list(results_dir.glob("*"))
        for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            print(f"   - {file.name} ({file.stat().st_size} bytes)")
    else:
        print(f"   ❌ results/trader 디렉토리가 없음")


if __name__ == "__main__":
    debug_threshold_usage()
