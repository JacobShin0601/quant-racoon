#!/usr/bin/env python3
"""
매크로 분석 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.market_sensor import MarketSensor
from datetime import datetime, timedelta


def test_market_sensor():
    """Market Sensor 테스트"""
    print("🔍 Market Sensor 테스트")
    print("=" * 40)
    
    sensor = MarketSensor()
    
    # 1. 현재 시장 분석
    print("\n1️⃣ 현재 시장 분석:")
    analysis = sensor.get_current_market_analysis(use_optimized_params=False)
    
    if 'error' in analysis:
        print(f"❌ 오류: {analysis['error']}")
        return
    
    print(f"  현재 시장 환경: {analysis['current_regime']}")
    print(f"  데이터 기간: {analysis['data_period']}")
    
    print(f"  성과 지표:")
    for metric, value in analysis['performance_metrics'].items():
        print(f"    {metric}: {value:.4f}")
    
    print(f"  전략 추천:")
    print(f"    주요 전략: {analysis['recommendation']['primary_strategy']}")
    print(f"    보조 전략: {analysis['recommendation']['secondary_strategy']}")
    print(f"    포지션 크기: {analysis['recommendation']['position_size']:.1%}")


def test_macro_sector_analyzer():
    """Macro Sector Analyzer 테스트"""
    print("\n\n🔍 Macro Sector Analyzer 테스트")
    print("=" * 40)
    
    sensor = MarketSensor()
    
    # 종합 분석 실행
    print("\n1️⃣ 종합 분석 실행:")
    analysis = sensor.get_macro_sector_analysis()
    
    if analysis is None:
        print("❌ 분석 실패")
        return
    
    print(f"  시장 조건: {analysis.market_condition.value}")
    print(f"  신뢰도: {analysis.confidence:.2%}")
    
    print(f"  주요 지표:")
    for indicator, value in analysis.key_indicators.items():
        if isinstance(value, float):
            print(f"    {indicator}: {value:.4f}")
        else:
            print(f"    {indicator}: {value}")
    
    print(f"  섹터 강도:")
    for sector, strength in analysis.sector_rotation.items():
        sector_name = sensor.sector_classification.get(sector, {}).get('name', sector)
        print(f"    {sector_name} ({sector}): {strength.value}")
    
    print(f"  투자 추천:")
    print(f"    전략: {analysis.recommendations['strategy']}")
    print(f"    위험도: {analysis.recommendations['risk_level']}")
    
    if analysis.recommendations['overweight_sectors']:
        print(f"    과중 배치 섹터: {', '.join(analysis.recommendations['overweight_sectors'])}")
    if analysis.recommendations['underweight_sectors']:
        print(f"    과소 배치 섹터: {', '.join(analysis.recommendations['underweight_sectors'])}")


def test_data_collection():
    """데이터 수집 테스트"""
    print("\n\n📊 데이터 수집 테스트")
    print("=" * 40)
    
    from src.actions.global_macro import GlobalMacroDataCollector
    
    collector = GlobalMacroDataCollector()
    
    # 날짜 설정
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"데이터 수집 기간: {start_date} ~ {end_date}")
    
    # SPY 데이터 수집
    print("\n1️⃣ SPY 데이터 수집:")
    spy_data = collector.collect_spy_data(start_date, end_date)
    if not spy_data.empty:
        print(f"  ✅ SPY 데이터: {len(spy_data)}개 행, {len(spy_data.columns)}개 컬럼")
        print(f"  컬럼: {list(spy_data.columns)}")
    else:
        print("  ❌ SPY 데이터 수집 실패")
    
    # 매크로 데이터 수집
    print("\n2️⃣ 매크로 데이터 수집:")
    macro_data = collector.collect_macro_indicators(start_date, end_date)
    print(f"  수집된 매크로 지표: {list(macro_data.keys())}")
    for symbol, data in macro_data.items():
        if not data.empty:
            print(f"    {symbol}: {len(data)}개 행")
    
    # 섹터 데이터 수집
    print("\n3️⃣ 섹터 데이터 수집:")
    sector_data = collector.collect_sector_data(start_date, end_date)
    print(f"  수집된 섹터: {list(sector_data.keys())}")
    for symbol, data in sector_data.items():
        if not data.empty:
            print(f"    {symbol}: {len(data)}개 행")


def main():
    """메인 테스트 함수"""
    print("🚀 매크로 분석 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 데이터 수집 테스트
        test_data_collection()
        
        # 2. Market Sensor 테스트
        test_market_sensor()
        
        # 3. Macro Sector Analyzer 테스트
        test_macro_sector_analyzer()
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 