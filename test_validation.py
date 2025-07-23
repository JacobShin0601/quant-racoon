#!/usr/bin/env python3
"""
시장 상태 분류 및 전략 성과 검증 테스트 스크립트
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.market_sensor import MarketSensor


def test_comprehensive_validation():
    """종합 검증 테스트"""
    print("🔍 종합 검증 테스트 시작...")
    
    # MarketSensor 초기화
    sensor = MarketSensor()
    
    # 테스트 기간 설정 (최근 1년)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"📅 테스트 기간: {start_date} ~ {end_date}")
    
    try:
        # 종합 검증 실행
        results = sensor.run_comprehensive_validation(start_date, end_date)
        
        if 'error' in results:
            print(f"❌ 검증 실패: {results['error']}")
            return False
        
        # 검증 결과 출력
        print("\n" + "="*80)
        print("📊 종합 검증 결과")
        print("="*80)
        
        # 1. 데이터 요약
        data_summary = results['data_summary']
        print(f"📈 데이터 요약:")
        print(f"  총 거래일: {data_summary['total_days']}일")
        print(f"  예측 시장 상태 분포:")
        for regime, count in data_summary['regime_distribution'].items():
            percentage = (count / data_summary['total_days']) * 100
            print(f"    {regime}: {count}일 ({percentage:.1f}%)")
        
        print(f"  실제 시장 상태 분포:")
        for regime, count in data_summary['actual_regime_distribution'].items():
            percentage = (count / data_summary['total_days']) * 100
            print(f"    {regime}: {count}일 ({percentage:.1f}%)")
        
        # 2. 분류 정확도 분석
        validation_results = results['validation_results']
        if 'overall_accuracy' in validation_results:
            print(f"\n🎯 분류 정확도 분석:")
            print(f"  전체 정확도: {validation_results['overall_accuracy']:.3f} ({validation_results['overall_accuracy']*100:.1f}%)")
            print(f"  정밀도: {validation_results['precision']:.3f}")
            print(f"  재현율: {validation_results['recall']:.3f}")
            print(f"  F1 점수: {validation_results['f1_score']:.3f}")
            print(f"  상태 변화 정확도: {validation_results['change_accuracy']:.3f}")
            
            # 시장 상태별 정확도
            if 'regime_accuracy' in validation_results:
                print(f"  시장 상태별 정확도:")
                for regime, accuracy in validation_results['regime_accuracy'].items():
                    print(f"    {regime}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # 3. 전략 성과 분석
        performance_results = results['performance_results']
        if 'overall_performance' in performance_results:
            perf = performance_results['overall_performance']
            print(f"\n💰 전략 성과 분석:")
            print(f"  총 수익률: {perf['total_return']:.3f} ({perf['total_return']*100:.1f}%)")
            print(f"  벤치마크 수익률: {perf['benchmark_return']:.3f} ({perf['benchmark_return']*100:.1f}%)")
            print(f"  초과 수익률: {perf['excess_return']:.3f} ({perf['excess_return']*100:.1f}%)")
            print(f"  변동성: {perf['volatility']:.3f} ({perf['volatility']*100:.1f}%)")
            print(f"  샤프 비율: {perf['sharpe_ratio']:.3f}")
            print(f"  최대 낙폭: {perf['max_drawdown']:.3f} ({perf['max_drawdown']*100:.1f}%)")
            print(f"  승률: {perf['win_rate']:.3f} ({perf['win_rate']*100:.1f}%)")
            print(f"  정보 비율: {perf['information_ratio']:.3f}")
            print(f"  VaR (95%): {perf['var_95']:.3f} ({perf['var_95']*100:.1f}%)")
            print(f"  통계적 유의성 (p-value): {perf['p_value']:.4f}")
            
            # 시장 상태별 성과
            if 'regime_performance' in performance_results:
                print(f"  시장 상태별 성과:")
                for regime, regime_perf in performance_results['regime_performance'].items():
                    print(f"    {regime}:")
                    print(f"      수익률: {regime_perf['total_return']:.3f} ({regime_perf['total_return']*100:.1f}%)")
                    print(f"      초과 수익률: {regime_perf['excess_return']:.3f} ({regime_perf['excess_return']*100:.1f}%)")
                    print(f"      샤프 비율: {regime_perf['sharpe_ratio']:.3f}")
                    print(f"      승률: {regime_perf['win_rate']:.3f} ({regime_perf['win_rate']*100:.1f}%)")
                    print(f"      거래일수: {regime_perf['days_count']}일")
        
        # 4. 전략 효과성 순위
        if 'regime_effectiveness' in performance_results:
            print(f"\n🏆 전략 효과성 순위:")
            effectiveness = performance_results['regime_effectiveness']
            sorted_effectiveness = sorted(effectiveness.items(), 
                                        key=lambda x: x[1]['performance_rank'])
            for regime, data in sorted_effectiveness:
                print(f"  {data['performance_rank']}. {regime}: {data['effectiveness_score']:.3f}")
        
        # 5. 검증 요약
        summary = sensor.generate_validation_summary(validation_results)
        print(f"\n{summary}")
        
        # 6. 결과 파일 경로
        if 'results_file' in results and results['results_file']:
            print(f"\n💾 결과 파일 저장됨: {results['results_file']}")
        
        print("\n" + "="*80)
        print("✅ 종합 검증 테스트 완료")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ 검증 테스트 중 오류: {e}")
        return False


def test_backtest_validation():
    """백테스팅 검증 테스트"""
    print("\n🔄 백테스팅 검증 테스트 시작...")
    
    # MarketSensor 초기화
    sensor = MarketSensor()
    
    # 테스트 기간 설정
    test_periods = [
        ('2022-01-01', '2022-06-30'),  # 2022년 상반기
        ('2022-07-01', '2022-12-31'),  # 2022년 하반기
        ('2023-01-01', '2023-06-30'),  # 2023년 상반기
        ('2023-07-01', '2023-12-31'),  # 2023년 하반기
    ]
    
    print(f"📅 백테스트 기간:")
    for i, (start, end) in enumerate(test_periods, 1):
        print(f"  기간 {i}: {start} ~ {end}")
    
    try:
        # 백테스팅 검증 실행
        results = sensor.run_backtest_validation(None, None, test_periods)
        
        if 'error' in results:
            print(f"❌ 백테스팅 실패: {results['error']}")
            return False
        
        # 백테스트 결과 출력
        print("\n" + "="*80)
        print("📊 백테스팅 검증 결과")
        print("="*80)
        
        # 1. 각 기간별 결과
        backtest_results = results['backtest_results']
        print("📈 기간별 검증 결과:")
        
        for period_name, period_data in backtest_results.items():
            if 'error' in period_data:
                print(f"  {period_name}: ❌ {period_data['error']}")
                continue
            
            start_date = period_data['start_date']
            end_date = period_data['end_date']
            
            # 분류 정확도
            val_results = period_data['validation_results']
            accuracy = val_results.get('overall_accuracy', 0)
            
            # 전략 성과
            perf_results = period_data['performance_results']
            perf = perf_results.get('overall_performance', {})
            total_return = perf.get('total_return', 0)
            excess_return = perf.get('excess_return', 0)
            sharpe_ratio = perf.get('sharpe_ratio', 0)
            
            print(f"  {period_name} ({start_date} ~ {end_date}):")
            print(f"    분류 정확도: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"    총 수익률: {total_return:.3f} ({total_return*100:.1f}%)")
            print(f"    초과 수익률: {excess_return:.3f} ({excess_return*100:.1f}%)")
            print(f"    샤프 비율: {sharpe_ratio:.3f}")
        
        # 2. 종합 분석
        overall_analysis = results['overall_analysis']
        if 'error' not in overall_analysis:
            print(f"\n📊 종합 분석:")
            print(f"  성공한 기간: {overall_analysis['successful_periods']}/{overall_analysis['total_periods']}")
            print(f"  평균 분류 정확도: {overall_analysis['average_accuracy']:.3f} ± {overall_analysis['std_accuracy']:.3f}")
            print(f"  평균 총 수익률: {overall_analysis['average_total_return']:.3f} ± {overall_analysis['std_total_return']:.3f}")
            print(f"  평균 초과 수익률: {overall_analysis['average_excess_return']:.3f} ± {overall_analysis['std_excess_return']:.3f}")
            print(f"  평균 샤프 비율: {overall_analysis['average_sharpe_ratio']:.3f} ± {overall_analysis['std_sharpe_ratio']:.3f}")
            print(f"  수익률 범위: {overall_analysis['min_total_return']:.3f} ~ {overall_analysis['max_total_return']:.3f}")
            print(f"  초과 수익률 범위: {overall_analysis['min_excess_return']:.3f} ~ {overall_analysis['max_excess_return']:.3f}")
            
            # 시장 상태별 성과 요약
            if 'regime_performance_summary' in overall_analysis:
                print(f"\n📈 시장 상태별 평균 성과:")
                for regime, perf in overall_analysis['regime_performance_summary'].items():
                    print(f"  {regime}:")
                    print(f"    평균 수익률: {perf['avg_total_return']:.3f} ± {perf['std_total_return']:.3f}")
                    print(f"    평균 초과 수익률: {perf['avg_excess_return']:.3f} ± {perf['std_excess_return']:.3f}")
                    print(f"    평균 샤프 비율: {perf['avg_sharpe_ratio']:.3f} ± {perf['std_sharpe_ratio']:.3f}")
                    print(f"    평균 승률: {perf['avg_win_rate']:.3f} ± {perf['std_win_rate']:.3f}")
        
        print("\n" + "="*80)
        print("✅ 백테스팅 검증 테스트 완료")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ 백테스팅 테스트 중 오류: {e}")
        return False


def main():
    """메인 실행 함수"""
    print("🚀 시장 상태 분류 및 전략 성과 검증 테스트")
    print("="*80)
    
    # 1. 종합 검증 테스트
    success1 = test_comprehensive_validation()
    
    # 2. 백테스팅 검증 테스트
    success2 = test_backtest_validation()
    
    # 3. 최종 결과
    print("\n" + "="*80)
    print("📋 최종 테스트 결과")
    print("="*80)
    
    if success1 and success2:
        print("✅ 모든 테스트가 성공적으로 완료되었습니다!")
        print("📊 검증 시스템이 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트에서 문제가 발생했습니다.")
        if not success1:
            print("  - 종합 검증 테스트 실패")
        if not success2:
            print("  - 백테스팅 검증 테스트 실패")
    
    print("="*80)


if __name__ == "__main__":
    main() 