#!/usr/bin/env python3
"""
TIPS Spread 테스트 스크립트
인플레이션 기대치 분석 기능 테스트
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.market_sensor import MarketSensor
from src.actions.global_macro import GlobalMacroDataCollector

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_tips_data_collection():
    """TIPS 데이터 수집 테스트"""
    print("\n🔍 TIPS 데이터 수집 테스트")
    print("=" * 50)
    
    collector = GlobalMacroDataCollector()
    
    # 날짜 설정
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"📅 데이터 수집 기간: {start_date} ~ {end_date}")
    
    # TIPS 관련 심볼들
    tips_symbols = ['TIP', 'SCHP', 'VTIP', 'LTPZ', 'TLT']
    
    for symbol in tips_symbols:
        try:
            print(f"\n📊 {symbol} 데이터 수집 중...")
            df = collector.collector.get_candle_data(
                symbol=symbol,
                interval='1d',
                start_date=start_date,
                end_date=end_date,
                days_back=90
            )
            
            if df is not None and not df.empty:
                print(f"  ✅ {symbol} 데이터 수집 완료: {len(df)}개 행")
                print(f"  📈 최근 가격: {df['close'].iloc[-1]:.2f}")
                print(f"  📊 20일 수익률: {df['close'].pct_change(20).iloc[-1]*100:.2f}%")
            else:
                print(f"  ❌ {symbol} 데이터 수집 실패")
                
        except Exception as e:
            print(f"  ❌ {symbol} 데이터 수집 중 오류: {e}")


def test_tips_spread_calculation():
    """TIPS Spread 계산 테스트"""
    print("\n\n🧮 TIPS Spread 계산 테스트")
    print("=" * 50)
    
    collector = GlobalMacroDataCollector()
    
    # 날짜 설정
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # 매크로 데이터 수집
    print("📊 매크로 데이터 수집 중...")
    macro_data = collector.collect_macro_indicators(start_date, end_date)
    
    # TIPS Spread 계산
    print("🧮 TIPS Spread 계산 중...")
    metrics = collector.calculate_macro_metrics(macro_data)
    
    if not metrics.empty:
        print("✅ TIPS Spread 계산 완료!")
        
        # TIPS 관련 지표들 출력
        tips_columns = [col for col in metrics.columns if 'tips_spread' in col]
        print(f"\n📈 TIPS Spread 지표들 ({len(tips_columns)}개):")
        for col in tips_columns:
            if col in metrics.columns:
                latest_value = metrics[col].iloc[-1]
                print(f"  {col}: {latest_value:.4f}")
        
        # 종합 지표가 있다면 출력
        if 'tips_spread_composite' in metrics.columns:
            composite = metrics['tips_spread_composite'].iloc[-1]
            print(f"\n🎯 종합 TIPS Spread: {composite:.4f}")
            
            if composite > 0.02:
                print("  📈 인플레이션 기대치: 높음")
            elif composite < -0.02:
                print("  📉 인플레이션 기대치: 낮음")
            else:
                print("  ➡️ 인플레이션 기대치: 안정적")
    else:
        print("❌ TIPS Spread 계산 실패")


def test_macro_analysis_with_tips():
    """TIPS를 포함한 매크로 분석 테스트"""
    print("\n\n🔍 TIPS를 포함한 매크로 분석 테스트")
    print("=" * 50)
    
    sensor = MarketSensor()
    
    # 매크로 & 섹터 분석 실행
    print("📊 매크로 & 섹터 분석 중...")
    analysis = sensor.get_macro_sector_analysis()
    
    if analysis is None:
        print("❌ 분석 실패")
        return
    
    print(f"🎯 시장 조건: {analysis.market_condition.value}")
    print(f"📊 신뢰도: {analysis.confidence:.2%}")
    
    # TIPS 관련 지표들 출력
    print(f"\n📈 TIPS 관련 지표:")
    tips_indicators = {k: v for k, v in analysis.key_indicators.items() if 'tips_spread' in k or 'inflation' in k}
    
    if tips_indicators:
        for indicator, value in tips_indicators.items():
            if isinstance(value, float):
                print(f"  {indicator}: {value:.4f}")
            else:
                print(f"  {indicator}: {value}")
    else:
        print("  TIPS 관련 지표가 없습니다.")
    
    # 인플레이션 기대치 출력
    if 'inflation_expectation' in analysis.key_indicators:
        inflation_exp = analysis.key_indicators['inflation_expectation']
        inflation_trend = analysis.key_indicators.get('inflation_trend', 'unknown')
        print(f"\n💰 인플레이션 분석:")
        print(f"  기대치: {inflation_exp}")
        print(f"  추세: {inflation_trend}")
    
    # 투자 추천 출력
    print(f"\n💡 투자 추천:")
    print(f"  전략: {analysis.recommendations['strategy']}")
    print(f"  위험도: {analysis.recommendations['risk_level']}")
    
    if analysis.recommendations['overweight_sectors']:
        print(f"  과중 배치 섹터: {', '.join(analysis.recommendations['overweight_sectors'])}")
    if analysis.recommendations['underweight_sectors']:
        print(f"  과소 배치 섹터: {', '.join(analysis.recommendations['underweight_sectors'])}")


def test_tips_spread_impact():
    """TIPS Spread의 시장 분류 영향 테스트"""
    print("\n\n🎯 TIPS Spread의 시장 분류 영향 테스트")
    print("=" * 50)
    
    sensor = MarketSensor()
    
    # 현재 시장 분석
    print("📊 현재 시장 분석 중...")
    analysis = sensor.get_current_market_analysis()
    
    if 'error' in analysis:
        print(f"❌ 분석 오류: {analysis['error']}")
        return
    
    print(f"🎯 현재 시장 환경: {analysis['current_regime']}")
    print(f"📅 데이터 기간: {analysis['data_period']}")
    
    print(f"\n📊 성과 지표:")
    for metric, value in analysis['performance_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\n💡 전략 추천:")
    print(f"  주요 전략: {analysis['recommendation']['primary_strategy']}")
    print(f"  보조 전략: {analysis['recommendation']['secondary_strategy']}")
    print(f"  포지션 크기: {analysis['recommendation']['position_size']:.1%}")
    print(f"  설명: {analysis['recommendation']['description']}")


def main():
    """메인 테스트 함수"""
    print("🚀 TIPS Spread 테스트 시작")
    print("=" * 60)
    
    try:
        # 1. TIPS 데이터 수집 테스트
        test_tips_data_collection()
        
        # 2. TIPS Spread 계산 테스트
        test_tips_spread_calculation()
        
        # 3. TIPS를 포함한 매크로 분석 테스트
        test_macro_analysis_with_tips()
        
        # 4. TIPS Spread의 시장 분류 영향 테스트
        test_tips_spread_impact()
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        logger.error(f"테스트 중 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main() 