#!/usr/bin/env python3
"""
전략 빠른 진단 테스트 코드
각 전략별로 최적화 횟수를 1회로 제한하여 빠르게 문제를 진단
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.append('.')

from src.agent.researcher import IndividualStrategyResearcher
from src.actions.strategies import StrategyManager
from src.actions.calculate_index import StrategyParams
from src.agent.helper import load_and_preprocess_data

class QuickStrategyDiagnostics:
    """빠른 전략 진단 클래스"""
    
    def __init__(self, config_path: str = "config/config_ensemble_research.json"):
        self.config_path = config_path
        self.research_config = self._load_config(config_path)
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            return {}
    
    def test_strategy_registration(self) -> Dict[str, Any]:
        """전략 등록 상태 테스트"""
        print("🔍 1단계: 전략 등록 상태 테스트")
        print("=" * 50)
        
        results = {}
        
        # 연구 설정에서 전략 목록 가져오기
        strategies = list(self.research_config.get("strategies", {}).keys())
        print(f"📊 총 {len(strategies)}개 전략 테스트")
        
        success_count = 0
        fail_count = 0
        
        # 전략 클래스 매핑
        strategy_classes = {
            "dual_momentum": "DualMomentumStrategy",
            "volatility_breakout": "VolatilityAdjustedBreakoutStrategy", 
            "swing_ema": "SwingEMACrossoverStrategy",
            "swing_rsi": "SwingRSIReversalStrategy",
            "swing_donchian": "DonchianSwingBreakoutStrategy",
            "stoch_donchian": "StochDonchianStrategy",
            "whipsaw_prevention": "WhipsawPreventionStrategy",
            "donchian_rsi_whipsaw": "DonchianRSIWhipsawStrategy",
            "volatility_filtered_breakout": "VolatilityFilteredBreakoutStrategy",
            "multi_timeframe_whipsaw": "MultiTimeframeWhipsawStrategy",
            "adaptive_whipsaw": "AdaptiveWhipsawStrategy",
            "cci_bollinger": "CCIBollingerStrategy",
            "mean_reversion": "MeanReversionStrategy",
            "swing_breakout": "SwingBreakoutStrategy",
            "swing_pullback_entry": "SwingPullbackEntryStrategy",
            "swing_candle_pattern": "SwingCandlePatternStrategy",
            "swing_bollinger_band": "SwingBollingerBandStrategy",
            "swing_macd": "SwingMACDStrategy",
            "stochastic": "StochasticStrategy",
            "williams_r": "WilliamsRStrategy",
            "cci": "CCIStrategy",
            "range_breakout": "RangeBreakoutStrategy",
            "support_resistance": "SupportResistanceStrategy",
            "oscillator_convergence": "OscillatorConvergenceStrategy"
        }
        
        for strategy_name in strategies:
            try:
                # 전략 클래스 이름 확인
                if strategy_name in strategy_classes:
                    class_name = strategy_classes[strategy_name]
                    results[strategy_name] = {
                        "status": "✅ 클래스 존재",
                        "class": class_name
                    }
                    success_count += 1
                    print(f"  ✅ {strategy_name}: {class_name}")
                else:
                    results[strategy_name] = {
                        "status": "❌ 클래스 없음",
                        "class": None
                    }
                    fail_count += 1
                    print(f"  ❌ {strategy_name}: 클래스 없음")
                    
            except Exception as e:
                results[strategy_name] = {
                    "status": f"❌ 오류: {str(e)}",
                    "class": None
                }
                fail_count += 1
                print(f"  ❌ {strategy_name}: {e}")
        
        print(f"\n📊 등록 결과: 성공 {success_count}개, 실패 {fail_count}개")
        self.results["registration"] = results
        return results
    
    def test_parameter_ranges(self) -> Dict[str, Any]:
        """파라미터 범위 테스트"""
        print("\n🔍 2단계: 파라미터 범위 테스트")
        print("=" * 50)
        
        results = {}
        strategies = list(self.research_config.get("strategies", {}).keys())
        
        success_count = 0
        fail_count = 0
        
        for strategy_name in strategies:
            param_ranges = (
                self.research_config.get("strategies", {})
                .get(strategy_name, {})
                .get("param_ranges", {})
            )
            
            if param_ranges:
                results[strategy_name] = {
                    "status": "✅ 파라미터 범위 있음",
                    "param_count": len(param_ranges),
                    "params": list(param_ranges.keys())
                }
                success_count += 1
                print(f"  ✅ {strategy_name}: {len(param_ranges)}개 파라미터")
            else:
                results[strategy_name] = {
                    "status": "❌ 파라미터 범위 없음",
                    "param_count": 0,
                    "params": []
                }
                fail_count += 1
                print(f"  ❌ {strategy_name}: 파라미터 범위 없음")
        
        print(f"\n📊 파라미터 결과: 성공 {success_count}개, 실패 {fail_count}개")
        self.results["parameters"] = results
        return results
    
    def test_single_optimization(self, test_strategy: str = "swing_rsi", test_symbol: str = "AAPL") -> Dict[str, Any]:
        """단일 전략 최적화 테스트 (1회만)"""
        print(f"\n🔍 3단계: 단일 전략 최적화 테스트 ({test_strategy} - {test_symbol})")
        print("=" * 50)
        
        results = {}
        
        try:
            # Researcher 초기화
            researcher = IndividualStrategyResearcher(
                research_config_path=self.config_path,
                source_config_path="config/config_ensemble_sideways.json",
                data_dir="data/ensemble_sideways",
                verbose=False  # 로그 최소화
            )
            
            print(f"  🔍 {test_strategy} - {test_symbol} 최적화 테스트...")
            
            # 단일 전략-종목 최적화 테스트
            result = researcher.optimize_single_strategy_for_symbol(test_strategy, test_symbol)
            
            if result:
                results["status"] = "✅ 최적화 성공"
                results["best_score"] = result.best_score
                results["best_params"] = result.best_params
                results["execution_time"] = result.execution_time
                results["n_combinations_tested"] = result.n_combinations_tested
                
                print(f"    ✅ 최적화 성공")
                print(f"      📊 최고 점수: {result.best_score:.2f}")
                print(f"      ⏱️ 실행 시간: {result.execution_time:.1f}초")
                print(f"      🔢 테스트 조합: {result.n_combinations_tested}개")
            else:
                results["status"] = "❌ 최적화 실패"
                results["best_score"] = None
                results["best_params"] = None
                results["execution_time"] = None
                results["n_combinations_tested"] = None
                
                print(f"    ❌ 최적화 실패")
                
        except Exception as e:
            results["status"] = f"❌ 오류: {str(e)}"
            results["best_score"] = None
            results["best_params"] = None
            results["execution_time"] = None
            results["n_combinations_tested"] = None
            
            print(f"    ❌ 오류: {e}")
        
        self.results["single_optimization"] = results
        return results
    
    def test_all_strategies_quick(self, test_symbol: str = "AAPL") -> Dict[str, Any]:
        """모든 전략 빠른 테스트 (각 전략 1회씩만)"""
        print(f"\n🔍 4단계: 모든 전략 빠른 테스트 ({test_symbol})")
        print("=" * 50)
        
        results = {}
        strategies = list(self.research_config.get("strategies", {}).keys())
        
        success_count = 0
        fail_count = 0
        
        print(f"📊 {len(strategies)}개 전략 테스트 시작...")
        
        for i, strategy_name in enumerate(strategies, 1):
            print(f"\n[{i}/{len(strategies)}] {strategy_name} 테스트 중...")
            
            try:
                # Researcher 초기화 (매번 새로)
                researcher = IndividualStrategyResearcher(
                    research_config_path=self.config_path,
                    source_config_path="config/config_ensemble_sideways.json",
                    data_dir="data/ensemble_sideways",
                    verbose=False
                )
                
                # 단일 전략-종목 최적화 테스트
                result = researcher.optimize_single_strategy_for_symbol(strategy_name, test_symbol)
                
                if result:
                    results[strategy_name] = {
                        "status": "✅ 성공",
                        "best_score": result.best_score,
                        "execution_time": result.execution_time,
                        "n_combinations_tested": result.n_combinations_tested
                    }
                    success_count += 1
                    print(f"  ✅ 성공 (점수: {result.best_score:.2f}, 시간: {result.execution_time:.1f}초)")
                else:
                    results[strategy_name] = {
                        "status": "❌ 실패",
                        "best_score": None,
                        "execution_time": None,
                        "n_combinations_tested": None
                    }
                    fail_count += 1
                    print(f"  ❌ 실패")
                    
            except Exception as e:
                results[strategy_name] = {
                    "status": f"❌ 오류: {str(e)}",
                    "best_score": None,
                    "execution_time": None,
                    "n_combinations_tested": None
                }
                fail_count += 1
                print(f"  ❌ 오류: {e}")
        
        print(f"\n📊 전체 결과: 성공 {success_count}개, 실패 {fail_count}개")
        self.results["all_strategies"] = results
        return results
    
    def generate_quick_report(self) -> str:
        """빠른 진단 보고서 생성"""
        print("\n📋 빠른 진단 보고서 생성")
        print("=" * 50)
        
        report = []
        report.append("# 전략 빠른 진단 보고서")
        report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. 전략 등록 상태
        report.append("## 1. 전략 등록 상태")
        if "registration" in self.results:
            success_count = sum(1 for info in self.results["registration"].values() if "✅" in info['status'])
            fail_count = len(self.results["registration"]) - success_count
            report.append(f"- **전체**: {len(self.results['registration'])}개")
            report.append(f"- **성공**: {success_count}개")
            report.append(f"- **실패**: {fail_count}개")
            
            if fail_count > 0:
                report.append("### 실패한 전략들:")
                for strategy, info in self.results["registration"].items():
                    if "❌" in info['status']:
                        report.append(f"- {strategy}: {info['status']}")
        report.append("")
        
        # 2. 파라미터 범위
        report.append("## 2. 파라미터 범위")
        if "parameters" in self.results:
            success_count = sum(1 for info in self.results["parameters"].values() if "✅" in info['status'])
            fail_count = len(self.results["parameters"]) - success_count
            report.append(f"- **전체**: {len(self.results['parameters'])}개")
            report.append(f"- **성공**: {success_count}개")
            report.append(f"- **실패**: {fail_count}개")
            
            if fail_count > 0:
                report.append("### 파라미터 범위 없는 전략들:")
                for strategy, info in self.results["parameters"].items():
                    if "❌" in info['status']:
                        report.append(f"- {strategy}")
        report.append("")
        
        # 3. 단일 전략 최적화
        report.append("## 3. 단일 전략 최적화")
        if "single_optimization" in self.results:
            info = self.results["single_optimization"]
            report.append(f"- **상태**: {info['status']}")
            if info['best_score'] is not None:
                report.append(f"- **최고 점수**: {info['best_score']:.2f}")
                report.append(f"- **실행 시간**: {info['execution_time']:.1f}초")
        report.append("")
        
        # 4. 모든 전략 테스트
        report.append("## 4. 모든 전략 테스트")
        if "all_strategies" in self.results:
            success_count = sum(1 for info in self.results["all_strategies"].values() if "✅" in info['status'])
            fail_count = len(self.results["all_strategies"]) - success_count
            report.append(f"- **전체**: {len(self.results['all_strategies'])}개")
            report.append(f"- **성공**: {success_count}개")
            report.append(f"- **실패**: {fail_count}개")
            
            if fail_count > 0:
                report.append("### 실패한 전략들:")
                for strategy, info in self.results["all_strategies"].items():
                    if "❌" in info['status']:
                        report.append(f"- {strategy}: {info['status']}")
        report.append("")
        
        # 문제점 요약
        report.append("## 🔍 문제점 요약")
        issues = []
        
        if "registration" in self.results:
            for strategy, info in self.results["registration"].items():
                if "❌" in info['status']:
                    issues.append(f"- {strategy}: 등록 실패")
        
        if "parameters" in self.results:
            for strategy, info in self.results["parameters"].items():
                if "❌" in info['status']:
                    issues.append(f"- {strategy}: 파라미터 범위 없음")
        
        if "all_strategies" in self.results:
            for strategy, info in self.results["all_strategies"].items():
                if "❌" in info['status']:
                    issues.append(f"- {strategy}: 최적화 실패")
        
        if issues:
            for issue in issues:
                report.append(issue)
        else:
            report.append("- 발견된 문제점 없음")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quick_diagnostic_report_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 빠른 진단 보고서 저장: {filename}")
        return report_text

def main():
    """메인 함수"""
    print("🚀 전략 빠른 진단 테스트 시작")
    print("=" * 60)
    
    # 진단 객체 생성
    diagnostics = QuickStrategyDiagnostics()
    
    # 1. 전략 등록 상태 테스트
    diagnostics.test_strategy_registration()
    
    # 2. 파라미터 범위 테스트
    diagnostics.test_parameter_ranges()
    
    # 3. 단일 전략 최적화 테스트
    diagnostics.test_single_optimization()
    
    # 4. 모든 전략 빠른 테스트
    diagnostics.test_all_strategies_quick()
    
    # 5. 빠른 진단 보고서 생성
    report = diagnostics.generate_quick_report()
    
    print("\n🎉 전략 빠른 진단 테스트 완료!")
    print("📄 상세 보고서는 생성된 마크다운 파일을 확인하세요.")

if __name__ == "__main__":
    main() 