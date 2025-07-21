#!/usr/bin/env python3
"""
Risk Parity 최적화 테스트 스크립트
"""

import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.actions.portfolio_optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationConstraints
)

def test_risk_parity():
    """Risk Parity 최적화 테스트"""
    print("🔍 Risk Parity 최적화 테스트 시작")
    
    # 샘플 수익률 데이터 생성 (12개 종목)
    np.random.seed(42)
    n_days = 200
    n_assets = 12
    asset_names = ["AAPL", "META", "NFLX", "QQQ", "SPY", "NVDA", "TSLA", "MSFT", "PLTR", "NVO", "CONL", "ETHU"]
    
    # 각 자산별로 다른 변동성을 가진 수익률 생성
    returns_data = {}
    for i, name in enumerate(asset_names):
        # 각 자산별로 다른 변동성 설정
        volatility = 0.02 + (i * 0.005)  # 2% ~ 7.5%
        returns = np.random.normal(0.001, volatility, n_days)
        returns_data[name] = returns
    
    returns_df = pd.DataFrame(returns_data)
    print(f"✅ 수익률 데이터 생성: {returns_df.shape}")
    print(f"📊 자산별 변동성:")
    for col in returns_df.columns:
        vol = returns_df[col].std()
        print(f"  {col}: {vol:.4f}")
    
    # PortfolioOptimizer 초기화
    optimizer = PortfolioOptimizer(returns=returns_df, risk_free_rate=0.02)
    
    # 제약조건 설정
    constraints = OptimizationConstraints(
        min_weight=0.01,
        max_weight=0.25,
        cash_weight=0.05,
        leverage=1.0
    )
    
    print(f"🔍 제약조건:")
    print(f"  - 최소 비중: {constraints.min_weight}")
    print(f"  - 최대 비중: {constraints.max_weight}")
    print(f"  - 현금 비중: {constraints.cash_weight}")
    print(f"  - 레버리지: {constraints.leverage}")
    
    # Risk Parity 최적화 실행
    try:
        print("\n🔍 Risk Parity 최적화 실행 중...")
        result = optimizer.optimize_portfolio(OptimizationMethod.RISK_PARITY, constraints)
        
        print(f"✅ Risk Parity 최적화 성공!")
        print(f"📊 최적화 결과:")
        print(f"  - 방법: {result.method}")
        print(f"  - 예상 수익률: {result.expected_return*252*100:.2f}%")
        print(f"  - 변동성: {result.volatility*np.sqrt(252)*100:.2f}%")
        print(f"  - 샤프 비율: {result.sharpe_ratio:.3f}")
        print(f"  - 소르티노 비율: {result.sortino_ratio:.3f}")
        print(f"  - 최대 낙폭: {result.max_drawdown*100:.2f}%")
        
        print(f"\n📊 자산별 비중:")
        for i, (name, weight) in enumerate(zip(asset_names, result.weights)):
            print(f"  {name}: {weight*100:.2f}%")
        
        # 리스크 기여도 확인
        if "risk_contributions" in result.metadata:
            risk_contributions = result.metadata["risk_contributions"]
            print(f"\n📊 리스크 기여도:")
            for i, (name, contribution) in enumerate(zip(asset_names, risk_contributions)):
                print(f"  {name}: {contribution:.6f}")
            
            # 리스크 기여도 표준편차
            contribution_std = result.metadata.get("contribution_std", 0)
            print(f"📊 리스크 기여도 표준편차: {contribution_std:.6f}")
        
        # 동일 가중치와 비교
        equal_weights = np.ones(n_assets) / n_assets
        equal_weights = equal_weights * (1 - constraints.cash_weight)
        
        print(f"\n📊 동일 가중치 vs Risk Parity:")
        print(f"  동일 가중치: {equal_weights*100}")
        print(f"  Risk Parity: {result.weights*100}")
        
        # 비중 차이 계산
        weight_diff = np.abs(result.weights - equal_weights)
        max_diff = np.max(weight_diff)
        avg_diff = np.mean(weight_diff)
        
        print(f"  최대 비중 차이: {max_diff*100:.2f}%")
        print(f"  평균 비중 차이: {avg_diff*100:.2f}%")
        
        if max_diff < 0.001:  # 0.1% 미만
            print("⚠️ Risk Parity가 동일 가중치와 거의 동일합니다. 최적화가 제대로 작동하지 않았을 수 있습니다.")
        else:
            print("✅ Risk Parity가 동일 가중치와 차별화된 비중을 생성했습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk Parity 최적화 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_risk_parity()
    if success:
        print("\n✅ Risk Parity 테스트 완료")
    else:
        print("\n❌ Risk Parity 테스트 실패") 