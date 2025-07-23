#!/usr/bin/env python3
"""
전략 진단 테스트 코드
각 전략의 등록 상태, 파라미터 범위, 데이터 로딩, 시그널 생성 등을 체계적으로 테스트
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
from src.agent.strategy_manager import StrategyManager
from src.agent.strategy_params import StrategyParams
from src.data.data_loader import load_and_preprocess_data
from src.evaluation.trading_simulator import TradingSimulator

class StrategyDiagnostics:
    """전략 진단 클래스"""
    
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
        strategy_manager = StrategyManager()
        
        # 연구 설정에서 전략 목록 가져오기
        strategies = list(self.research_config.get("strategies", {}).keys())
        print(f"📊 총 {len(strategies)}개 전략 테스트")
        
        for strategy_name in strategies:
            print(f"\n🔍 {strategy_name} 전략 테스트 중...")
            
            try:
                # 전략 클래스 가져오기
                strategy_class = strategy_manager.get_strategy_class(strategy_name)
                if strategy_class:
                    # 전략 인스턴스 생성
                    strategy = strategy_class(StrategyParams())
                    results[strategy_name] = {
                        "status": "✅ 등록됨",
                        "class": strategy_class.__name__,
                        "instance": strategy
                    }
                    print(f"  ✅ {strategy_name}: {strategy_class.__name__}")
                else:
                    results[strategy_name] = {
                        "status": "❌ 클래스 없음",
                        "class": None,
                        "instance": None
                    }
                    print(f"  ❌ {strategy_name}: 클래스 없음")
                    
            except Exception as e:
                results[strategy_name] = {
                    "status": f"❌ 오류: {str(e)}",
                    "class": None,
                    "instance": None
                }
                print(f"  ❌ {strategy_name}: {e}")
        
        self.results["registration"] = results
        return results
    
    def test_parameter_ranges(self) -> Dict[str, Any]:
        """파라미터 범위 테스트"""
        print("\n🔍 2단계: 파라미터 범위 테스트")
        print("=" * 50)
        
        results = {}
        strategies = list(self.research_config.get("strategies", {}).keys())
        
        for strategy_name in strategies:
            print(f"\n🔍 {strategy_name} 파라미터 범위 테스트...")
            
            param_ranges = (
                self.research_config.get("strategies", {})
                .get(strategy_name, {})
                .get("param_ranges", {})
            )
            
            if param_ranges:
                results[strategy_name] = {
                    "status": "✅ 파라미터 범위 있음",
                    "param_count": len(param_ranges),
                    "params": list(param_ranges.keys()),
                    "ranges": param_ranges
                }
                print(f"  ✅ {len(param_ranges)}개 파라미터: {list(param_ranges.keys())}")
            else:
                results[strategy_name] = {
                    "status": "❌ 파라미터 범위 없음",
                    "param_count": 0,
                    "params": [],
                    "ranges": {}
                }
                print(f"  ❌ 파라미터 범위 없음")
        
        self.results["parameters"] = results
        return results
    
    def test_data_loading(self, symbols: List[str] = None) -> Dict[str, Any]:
        """데이터 로딩 테스트"""
        print("\n🔍 3단계: 데이터 로딩 테스트")
        print("=" * 50)
        
        if not symbols:
            symbols = ["AAPL", "MSFT", "GOOGL"]
        
        results = {}
        
        try:
            data_dict = load_and_preprocess_data("data/ensemble_sideways", symbols)
            
            if data_dict:
                results["status"] = "✅ 데이터 로드 성공"
                results["symbols_loaded"] = list(data_dict.keys())
                results["data_shape"] = {symbol: data.shape for symbol, data in data_dict.items()}
                results["sample_data"] = {symbol: data.head(3) for symbol, data in data_dict.items()}
                
                print(f"  ✅ {len(data_dict)}개 심볼 데이터 로드 성공")
                for symbol, shape in results["data_shape"].items():
                    print(f"    📊 {symbol}: {shape}")
            else:
                results["status"] = "❌ 데이터 로드 실패"
                results["symbols_loaded"] = []
                results["data_shape"] = {}
                results["sample_data"] = {}
                print("  ❌ 데이터 로드 실패")
                
        except Exception as e:
            results["status"] = f"❌ 오류: {str(e)}"
            results["symbols_loaded"] = []
            results["data_shape"] = {}
            results["sample_data"] = {}
            print(f"  ❌ 데이터 로드 오류: {e}")
        
        self.results["data_loading"] = results
        return results
    
    def test_signal_generation(self, test_symbol: str = "AAPL") -> Dict[str, Any]:
        """시그널 생성 테스트"""
        print(f"\n🔍 4단계: 시그널 생성 테스트 ({test_symbol})")
        print("=" * 50)
        
        results = {}
        
        # 데이터 로드
        try:
            data_dict = load_and_preprocess_data("data/ensemble_sideways", [test_symbol])
            if not data_dict or test_symbol not in data_dict:
                print(f"  ❌ {test_symbol} 데이터 없음")
                return {"status": "❌ 데이터 없음"}
            
            symbol_data = data_dict[test_symbol]
            print(f"  ✅ {test_symbol} 데이터 로드: {symbol_data.shape}")
            
        except Exception as e:
            print(f"  ❌ 데이터 로드 오류: {e}")
            return {"status": f"❌ 데이터 로드 오류: {e}"}
        
        # 각 전략별 시그널 생성 테스트
        strategies = list(self.research_config.get("strategies", {}).keys())
        
        for strategy_name in strategies:
            print(f"\n  🔍 {strategy_name} 시그널 생성 테스트...")
            
            try:
                # 전략 매니저에서 전략 가져오기
                strategy_manager = StrategyManager()
                strategy_class = strategy_manager.get_strategy_class(strategy_name)
                
                if not strategy_class:
                    results[strategy_name] = {
                        "status": "❌ 전략 클래스 없음",
                        "signals_shape": None,
                        "signal_counts": None,
                        "error": "전략 클래스를 찾을 수 없음"
                    }
                    print(f"    ❌ 전략 클래스 없음")
                    continue
                
                # 전략 인스턴스 생성
                strategy = strategy_class(StrategyParams())
                
                # 시그널 생성
                signals = strategy.generate_signals(symbol_data)
                
                if signals is not None and not signals.empty:
                    signal_counts = signals["signal"].value_counts()
                    results[strategy_name] = {
                        "status": "✅ 시그널 생성 성공",
                        "signals_shape": signals.shape,
                        "signal_counts": signal_counts.to_dict(),
                        "error": None
                    }
                    print(f"    ✅ 시그널 생성: {signals.shape}")
                    print(f"      📊 신호 분포: {signal_counts.to_dict()}")
                else:
                    results[strategy_name] = {
                        "status": "❌ 시그널 생성 실패",
                        "signals_shape": None,
                        "signal_counts": None,
                        "error": "시그널이 None이거나 비어있음"
                    }
                    print(f"    ❌ 시그널 생성 실패")
                    
            except Exception as e:
                results[strategy_name] = {
                    "status": f"❌ 오류: {str(e)}",
                    "signals_shape": None,
                    "signal_counts": None,
                    "error": str(e)
                }
                print(f"    ❌ 오류: {e}")
        
        self.results["signal_generation"] = results
        return results
    
    def test_optimization_simulation(self, test_strategy: str = "swing_rsi", test_symbol: str = "AAPL") -> Dict[str, Any]:
        """최적화 시뮬레이션 테스트"""
        print(f"\n🔍 5단계: 최적화 시뮬레이션 테스트 ({test_strategy} - {test_symbol})")
        print("=" * 50)
        
        results = {}
        
        try:
            # Researcher 초기화
            researcher = IndividualStrategyResearcher(
                research_config_path=self.config_path,
                source_config_path="config/config_ensemble_sideways.json",
                data_dir="data/ensemble_sideways",
                verbose=True
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
        
        self.results["optimization_simulation"] = results
        return results
    
    def generate_diagnostic_report(self) -> str:
        """진단 보고서 생성"""
        print("\n📋 진단 보고서 생성")
        print("=" * 50)
        
        report = []
        report.append("# 전략 진단 보고서")
        report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. 전략 등록 상태
        report.append("## 1. 전략 등록 상태")
        if "registration" in self.results:
            for strategy, info in self.results["registration"].items():
                report.append(f"- **{strategy}**: {info['status']}")
        report.append("")
        
        # 2. 파라미터 범위
        report.append("## 2. 파라미터 범위")
        if "parameters" in self.results:
            for strategy, info in self.results["parameters"].items():
                report.append(f"- **{strategy}**: {info['status']} ({info['param_count']}개 파라미터)")
        report.append("")
        
        # 3. 데이터 로딩
        report.append("## 3. 데이터 로딩")
        if "data_loading" in self.results:
            info = self.results["data_loading"]
            report.append(f"- **상태**: {info['status']}")
            if info['symbols_loaded']:
                report.append(f"- **로드된 심볼**: {', '.join(info['symbols_loaded'])}")
        report.append("")
        
        # 4. 시그널 생성
        report.append("## 4. 시그널 생성")
        if "signal_generation" in self.results:
            for strategy, info in self.results["signal_generation"].items():
                report.append(f"- **{strategy}**: {info['status']}")
                if info['error']:
                    report.append(f"  - 오류: {info['error']}")
        report.append("")
        
        # 5. 최적화 시뮬레이션
        report.append("## 5. 최적화 시뮬레이션")
        if "optimization_simulation" in self.results:
            info = self.results["optimization_simulation"]
            report.append(f"- **상태**: {info['status']}")
            if info['best_score'] is not None:
                report.append(f"- **최고 점수**: {info['best_score']:.2f}")
                report.append(f"- **실행 시간**: {info['execution_time']:.1f}초")
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
        
        if "signal_generation" in self.results:
            for strategy, info in self.results["signal_generation"].items():
                if "❌" in info['status']:
                    issues.append(f"- {strategy}: 시그널 생성 실패")
        
        if issues:
            for issue in issues:
                report.append(issue)
        else:
            report.append("- 발견된 문제점 없음")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_report_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 진단 보고서 저장: {filename}")
        return report_text

def main():
    """메인 함수"""
    print("🚀 전략 진단 테스트 시작")
    print("=" * 60)
    
    # 진단 객체 생성
    diagnostics = StrategyDiagnostics()
    
    # 1. 전략 등록 상태 테스트
    diagnostics.test_strategy_registration()
    
    # 2. 파라미터 범위 테스트
    diagnostics.test_parameter_ranges()
    
    # 3. 데이터 로딩 테스트
    diagnostics.test_data_loading()
    
    # 4. 시그널 생성 테스트
    diagnostics.test_signal_generation()
    
    # 5. 최적화 시뮬레이션 테스트
    diagnostics.test_optimization_simulation()
    
    # 6. 진단 보고서 생성
    report = diagnostics.generate_diagnostic_report()
    
    print("\n🎉 전략 진단 테스트 완료!")
    print("📄 상세 보고서는 생성된 마크다운 파일을 확인하세요.")

if __name__ == "__main__":
    main() 