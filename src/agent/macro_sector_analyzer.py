#!/usr/bin/env python3
"""
매크로 & 섹터 분석기 (Macro Sector Analyzer)
매크로 지표와 섹터 로테이션을 활용한 고급 시장 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import os
from dataclasses import dataclass
from enum import Enum

from ..actions.global_macro import GlobalMacroDataCollector


class MarketCondition(Enum):
    """시장 조건 분류"""
    BULL_MARKET = "bull_market"           # 강세장
    BEAR_MARKET = "bear_market"           # 약세장
    SIDEWAYS_MARKET = "sideways_market"   # 횡보장
    VOLATILE_MARKET = "volatile_market"   # 변동성 장
    RECESSION_FEAR = "recession_fear"     # 경기침체 우려
    INFLATION_FEAR = "inflation_fear"     # 인플레이션 우려


class SectorStrength(Enum):
    """섹터 강도 분류"""
    LEADING = "leading"       # 선도 섹터
    LAGGING = "lagging"       # 후행 섹터
    DEFENSIVE = "defensive"   # 방어적 섹터
    CYCLICAL = "cyclical"     # 순환적 섹터


@dataclass
class MacroAnalysis:
    """매크로 분석 결과"""
    market_condition: MarketCondition
    confidence: float
    key_indicators: Dict[str, float]
    sector_rotation: Dict[str, SectorStrength]
    recommendations: Dict[str, Any]
    timestamp: datetime


class MacroSectorAnalyzer:
    """매크로 & 섹터 분석 시스템"""
    
    def __init__(self, data_dir: str = "data/macro"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.macro_collector = GlobalMacroDataCollector()
        
        # 섹터 분류
        self.sector_classification = {
            'XLK': {'name': 'Technology', 'type': SectorStrength.CYCLICAL},
            'XLF': {'name': 'Financials', 'type': SectorStrength.CYCLICAL},
            'XLE': {'name': 'Energy', 'type': SectorStrength.CYCLICAL},
            'XLV': {'name': 'Healthcare', 'type': SectorStrength.DEFENSIVE},
            'XLI': {'name': 'Industrials', 'type': SectorStrength.CYCLICAL},
            'XLP': {'name': 'Consumer Staples', 'type': SectorStrength.DEFENSIVE},
            'XLU': {'name': 'Utilities', 'type': SectorStrength.DEFENSIVE},
            'XLB': {'name': 'Materials', 'type': SectorStrength.CYCLICAL},
            'XLRE': {'name': 'Real Estate', 'type': SectorStrength.CYCLICAL}
        }
    
    def analyze_macro_environment(self, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """매크로 환경 분석"""
        analysis = {}
        
        try:
            # 1. VIX 기반 변동성 분석
            if '^VIX' in macro_data:
                vix = macro_data['^VIX']['close']
                analysis['vix_current'] = vix.iloc[-1]
                analysis['vix_ma_20'] = vix.rolling(20).mean().iloc[-1]
                analysis['vix_percentile'] = (vix.iloc[-1] / vix.rolling(252).max().iloc[-1]) * 100
                analysis['volatility_regime'] = 'high' if vix.iloc[-1] > 25 else 'normal'
            
            # 2. 국채 스프레드 분석 (경기침체 지표)
            if '^TNX' in macro_data and '^IRX' in macro_data:
                tnx = macro_data['^TNX']['close']
                irx = macro_data['^IRX']['close']
                spread = tnx - irx
                analysis['yield_spread'] = spread.iloc[-1]
                analysis['spread_ma_20'] = spread.rolling(20).mean().iloc[-1]
                analysis['recession_risk'] = 'high' if spread.iloc[-1] < 0 else 'low'
            
            # 3. 달러 강도 분석
            if 'UUP' in macro_data:
                dxy = macro_data['UUP']['close']
                analysis['dollar_strength'] = dxy.iloc[-1]
                analysis['dollar_ma_50'] = dxy.rolling(50).mean().iloc[-1]
                analysis['dollar_trend'] = 'strong' if dxy.iloc[-1] > dxy.rolling(50).mean().iloc[-1] else 'weak'
            
            # 4. 금 가격 분석 (안전자산 선호도)
            if 'GLD' in macro_data:
                gold = macro_data['GLD']['close']
                analysis['gold_price'] = gold.iloc[-1]
                analysis['gold_ma_50'] = gold.rolling(50).mean().iloc[-1]
                analysis['gold_trend'] = 'bullish' if gold.iloc[-1] > gold.rolling(50).mean().iloc[-1] else 'bearish'
            
            # 5. 국채 가격 분석
            if 'TLT' in macro_data:
                tlt = macro_data['TLT']['close']
                analysis['bond_price'] = tlt.iloc[-1]
                analysis['bond_ma_50'] = tlt.rolling(50).mean().iloc[-1]
                analysis['bond_trend'] = 'bullish' if tlt.iloc[-1] > tlt.rolling(50).mean().iloc[-1] else 'bearish'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"매크로 환경 분석 중 오류: {e}")
            return {}
    
    def classify_market_condition(self, macro_analysis: Dict[str, Any]) -> MarketCondition:
        """매크로 분석을 기반으로 시장 조건 분류"""
        scores = {
            MarketCondition.BULL_MARKET: 0,
            MarketCondition.BEAR_MARKET: 0,
            MarketCondition.SIDEWAYS_MARKET: 0,
            MarketCondition.VOLATILE_MARKET: 0,
            MarketCondition.RECESSION_FEAR: 0,
            MarketCondition.INFLATION_FEAR: 0
        }
        
        # VIX 기반 변동성 점수
        if 'volatility_regime' in macro_analysis:
            if macro_analysis['volatility_regime'] == 'high':
                scores[MarketCondition.VOLATILE_MARKET] += 3
                scores[MarketCondition.BEAR_MARKET] += 1
        
        # 국채 스프레드 기반 경기침체 점수
        if 'recession_risk' in macro_analysis:
            if macro_analysis['recession_risk'] == 'high':
                scores[MarketCondition.RECESSION_FEAR] += 4
                scores[MarketCondition.BEAR_MARKET] += 2
        
        # 달러 강도 기반 점수
        if 'dollar_trend' in macro_analysis:
            if macro_analysis['dollar_trend'] == 'strong':
                scores[MarketCondition.BULL_MARKET] += 1
            else:
                scores[MarketCondition.BEAR_MARKET] += 1
        
        # 금 가격 기반 점수
        if 'gold_trend' in macro_analysis:
            if macro_analysis['gold_trend'] == 'bullish':
                scores[MarketCondition.RECESSION_FEAR] += 1
                scores[MarketCondition.VOLATILE_MARKET] += 1
        
        # 국채 가격 기반 점수
        if 'bond_trend' in macro_analysis:
            if macro_analysis['bond_trend'] == 'bullish':
                scores[MarketCondition.RECESSION_FEAR] += 1
            else:
                scores[MarketCondition.INFLATION_FEAR] += 1
        
        # 최고 점수 시장 조건 반환
        return max(scores, key=scores.get)
    
    def analyze_sector_rotation(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, SectorStrength]:
        """섹터 로테이션 분석"""
        sector_analysis = {}
        
        try:
            # 섹터별 수익률 계산
            sector_returns = {}
            for symbol, data in sector_data.items():
                if not data.empty and 'close' in data.columns:
                    sector_returns[symbol] = data['close'].pct_change()
            
            if not sector_returns:
                return sector_analysis
            
            # 20일 상대 강도 계산
            sector_df = pd.DataFrame(sector_returns)
            for symbol in sector_df.columns:
                sector_df[f'{symbol}_rs'] = sector_df[symbol].rolling(20).mean()
            
            # 최근 상대 강도 순위
            rs_columns = [col for col in sector_df.columns if col.endswith('_rs')]
            latest_rs = sector_df[rs_columns].iloc[-1]
            ranked_sectors = latest_rs.rank(ascending=False)
            
            # 섹터 강도 분류
            for symbol in sector_data.keys():
                if symbol in ranked_sectors.index:
                    rank = ranked_sectors[symbol]
                    total_sectors = len(ranked_sectors)
                    
                    if rank <= total_sectors * 0.3:  # 상위 30%
                        sector_analysis[symbol] = SectorStrength.LEADING
                    elif rank >= total_sectors * 0.7:  # 하위 30%
                        sector_analysis[symbol] = SectorStrength.LAGGING
                    else:
                        # 섹터 타입에 따라 분류
                        sector_type = self.sector_classification.get(symbol, {}).get('type', SectorStrength.CYCLICAL)
                        sector_analysis[symbol] = sector_type
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"섹터 로테이션 분석 중 오류: {e}")
            return {}
    
    def generate_sector_recommendations(self, market_condition: MarketCondition, 
                                      sector_analysis: Dict[str, SectorStrength]) -> Dict[str, Any]:
        """시장 조건과 섹터 분석을 기반으로 추천 생성"""
        recommendations = {
            'overweight_sectors': [],
            'underweight_sectors': [],
            'neutral_sectors': [],
            'strategy': '',
            'risk_level': 'medium'
        }
        
        if market_condition == MarketCondition.BULL_MARKET:
            # 강세장: 순환적 섹터 중 선도 섹터 선호
            recommendations['strategy'] = 'Momentum following - 순환적 섹터 중 선도 섹터 집중'
            recommendations['risk_level'] = 'high'
            
            for symbol, strength in sector_analysis.items():
                sector_type = self.sector_classification.get(symbol, {}).get('type')
                if strength == SectorStrength.LEADING and sector_type == SectorStrength.CYCLICAL:
                    recommendations['overweight_sectors'].append(symbol)
                elif strength == SectorStrength.LAGGING:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        elif market_condition == MarketCondition.BEAR_MARKET:
            # 약세장: 방어적 섹터 선호
            recommendations['strategy'] = 'Defensive positioning - 방어적 섹터 집중'
            recommendations['risk_level'] = 'low'
            
            for symbol, strength in sector_analysis.items():
                sector_type = self.sector_classification.get(symbol, {}).get('type')
                if sector_type == SectorStrength.DEFENSIVE:
                    recommendations['overweight_sectors'].append(symbol)
                elif sector_type == SectorStrength.CYCLICAL:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        elif market_condition == MarketCondition.RECESSION_FEAR:
            # 경기침체 우려: 방어적 섹터 + 국채
            recommendations['strategy'] = 'Recession hedge - 방어적 섹터 + 국채 비중 확대'
            recommendations['risk_level'] = 'low'
            
            for symbol, strength in sector_analysis.items():
                sector_type = self.sector_classification.get(symbol, {}).get('type')
                if sector_type == SectorStrength.DEFENSIVE:
                    recommendations['overweight_sectors'].append(symbol)
                elif sector_type == SectorStrength.CYCLICAL:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        elif market_condition == MarketCondition.VOLATILE_MARKET:
            # 변동성 장: 분산 투자 + 현금 비중 확대
            recommendations['strategy'] = 'Diversification - 분산 투자 + 현금 비중 확대'
            recommendations['risk_level'] = 'medium'
            
            # 모든 섹터를 중립으로 설정하고 현금 비중 확대
            recommendations['neutral_sectors'] = list(sector_analysis.keys())
        
        else:  # SIDEWAYS_MARKET
            # 횡보장: 스윙 트레이딩
            recommendations['strategy'] = 'Swing trading - 섹터 로테이션 활용'
            recommendations['risk_level'] = 'medium'
            
            for symbol, strength in sector_analysis.items():
                if strength == SectorStrength.LEADING:
                    recommendations['overweight_sectors'].append(symbol)
                elif strength == SectorStrength.LAGGING:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        return recommendations
    
    def calculate_correlation_matrix(self, sector_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """섹터 간 상관관계 매트릭스 계산"""
        try:
            # 섹터별 수익률 계산
            sector_returns = {}
            for symbol, data in sector_data.items():
                if not data.empty and 'close' in data.columns:
                    sector_returns[symbol] = data['close'].pct_change()
            
            if not sector_returns:
                return pd.DataFrame()
            
            # 상관관계 매트릭스 계산
            returns_df = pd.DataFrame(sector_returns)
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"상관관계 매트릭스 계산 중 오류: {e}")
            return pd.DataFrame()
    
    def analyze_macro_sector_relationship(self, macro_data: Dict[str, pd.DataFrame], 
                                        sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """매크로 지표와 섹터 간 관계 분석"""
        analysis = {}
        
        try:
            # VIX와 섹터 변동성 관계
            if '^VIX' in macro_data:
                vix = macro_data['^VIX']['close']
                sector_volatility = {}
                
                for symbol, data in sector_data.items():
                    if not data.empty and 'close' in data.columns:
                        returns = data['close'].pct_change()
                        sector_volatility[symbol] = returns.rolling(20).std()
                
                if sector_volatility:
                    volatility_df = pd.DataFrame(sector_volatility)
                    vix_correlation = volatility_df.corrwith(vix)
                    analysis['vix_sector_correlation'] = vix_correlation.to_dict()
            
            # 국채 수익률과 섹터 관계
            if 'TLT' in macro_data:
                tlt_returns = macro_data['TLT']['close'].pct_change()
                sector_correlation = {}
                
                for symbol, data in sector_data.items():
                    if not data.empty and 'close' in data.columns:
                        sector_returns = data['close'].pct_change()
                        correlation = sector_returns.corr(tlt_returns)
                        sector_correlation[symbol] = correlation
                
                analysis['bond_sector_correlation'] = sector_correlation
            
            # 달러 강도와 섹터 관계
            if 'UUP' in macro_data:
                dollar_returns = macro_data['UUP']['close'].pct_change()
                sector_correlation = {}
                
                for symbol, data in sector_data.items():
                    if not data.empty and 'close' in data.columns:
                        sector_returns = data['close'].pct_change()
                        correlation = sector_returns.corr(dollar_returns)
                        sector_correlation[symbol] = correlation
                
                analysis['dollar_sector_correlation'] = sector_correlation
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"매크로-섹터 관계 분석 중 오류: {e}")
            return {}
    
    def get_comprehensive_analysis(self, start_date: str = None, end_date: str = None) -> MacroAnalysis:
        """종합 분석 실행"""
        try:
            # 날짜 설정
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # 데이터 수집
            self.logger.info("매크로 & 섹터 데이터 수집 중...")
            spy_data = self.macro_collector.collect_spy_data(start_date, end_date)
            macro_data = self.macro_collector.collect_macro_indicators(start_date, end_date)
            sector_data = self.macro_collector.collect_sector_data(start_date, end_date)
            
            # 매크로 환경 분석
            macro_analysis = self.analyze_macro_environment(macro_data)
            
            # 시장 조건 분류
            market_condition = self.classify_market_condition(macro_analysis)
            
            # 섹터 로테이션 분석
            sector_analysis = self.analyze_sector_rotation(sector_data)
            
            # 추천 생성
            recommendations = self.generate_sector_recommendations(market_condition, sector_analysis)
            
            # 상관관계 분석
            correlation_matrix = self.calculate_correlation_matrix(sector_data)
            macro_sector_relationship = self.analyze_macro_sector_relationship(macro_data, sector_data)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(macro_analysis, sector_analysis)
            
            return MacroAnalysis(
                market_condition=market_condition,
                confidence=confidence,
                key_indicators=macro_analysis,
                sector_rotation=sector_analysis,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"종합 분석 중 오류: {e}")
            return None
    
    def _calculate_confidence(self, macro_analysis: Dict[str, Any], 
                            sector_analysis: Dict[str, SectorStrength]) -> float:
        """분석 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 매크로 지표 개수에 따른 신뢰도 조정
        macro_indicators = len(macro_analysis)
        if macro_indicators >= 5:
            confidence += 0.2
        elif macro_indicators >= 3:
            confidence += 0.1
        
        # 섹터 데이터 개수에 따른 신뢰도 조정
        sector_count = len(sector_analysis)
        if sector_count >= 7:
            confidence += 0.2
        elif sector_count >= 5:
            confidence += 0.1
        
        # VIX 변동성에 따른 신뢰도 조정
        if 'vix_current' in macro_analysis:
            vix = macro_analysis['vix_current']
            if 15 <= vix <= 25:  # 정상 변동성
                confidence += 0.1
        
        return min(confidence, 1.0)  # 최대 1.0
    
    def save_analysis_results(self, analysis: MacroAnalysis, output_dir: str = "results/macro_sector_analysis"):
        """분석 결과 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 분석 결과를 딕셔너리로 변환
            results = {
                'market_condition': analysis.market_condition.value,
                'confidence': analysis.confidence,
                'key_indicators': analysis.key_indicators,
                'sector_rotation': {k: v.value for k, v in analysis.sector_rotation.items()},
                'recommendations': analysis.recommendations,
                'timestamp': analysis.timestamp.isoformat()
            }
            
            # JSON 파일로 저장
            with open(f"{output_dir}/analysis_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"분석 결과 저장 완료: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"분석 결과 저장 중 오류: {e}")


def main():
    """테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Macro Sector Analyzer - 매크로 & 섹터 분석')
    parser.add_argument('--start_date', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--save_results', action='store_true', help='결과 저장')
    
    args = parser.parse_args()
    
    print("🔍 Macro Sector Analyzer 시작")
    print("=" * 50)
    
    analyzer = MacroSectorAnalyzer()
    
    try:
        # 종합 분석 실행
        analysis = analyzer.get_comprehensive_analysis(args.start_date, args.end_date)
        
        if analysis is None:
            print("❌ 분석 실패")
            return
        
        # 결과 출력
        print(f"\n🎯 시장 조건: {analysis.market_condition.value}")
        print(f"📊 신뢰도: {analysis.confidence:.2%}")
        
        print(f"\n📈 주요 지표:")
        for indicator, value in analysis.key_indicators.items():
            if isinstance(value, float):
                print(f"  {indicator}: {value:.4f}")
            else:
                print(f"  {indicator}: {value}")
        
        print(f"\n🏭 섹터 강도:")
        for sector, strength in analysis.sector_rotation.items():
            sector_name = analyzer.sector_classification.get(sector, {}).get('name', sector)
            print(f"  {sector_name} ({sector}): {strength.value}")
        
        print(f"\n💡 투자 추천:")
        print(f"  전략: {analysis.recommendations['strategy']}")
        print(f"  위험도: {analysis.recommendations['risk_level']}")
        
        if analysis.recommendations['overweight_sectors']:
            print(f"  과중 배치 섹터: {', '.join(analysis.recommendations['overweight_sectors'])}")
        if analysis.recommendations['underweight_sectors']:
            print(f"  과소 배치 섹터: {', '.join(analysis.recommendations['underweight_sectors'])}")
        
        # 결과 저장
        if args.save_results:
            analyzer.save_analysis_results(analysis)
            print(f"\n✅ 분석 결과 저장 완료")
        
        print("\n🎉 분석 완료!")
        
    except Exception as e:
        print(f"❌ 분석 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 