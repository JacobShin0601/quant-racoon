# Analysis 폴더

이 폴더는 정량 분석 및 연구 결과를 체계적으로 저장하고 관리하는 공간입니다.

## 폴더 구조

```
analysis/
├── quant_analysis/          # 정량 분석 결과
│   ├── correlation/         # 상관관계 분석 결과
│   ├── regression/          # 회귀 분석 결과
│   ├── bayesian/           # 베이지안 분석 결과
│   └── summary/            # 종합 분석 리포트
├── researcher_results/      # 하이퍼파라미터 연구 결과
│   ├── grid_search/        # 그리드 서치 결과
│   ├── bayesian_opt/       # 베이지안 최적화 결과
│   ├── genetic_alg/        # 유전 알고리즘 결과
│   └── comparison/         # 최적화 방법 비교 결과
└── strategy_optimization/   # 전략별 최적화 결과
    ├── dual_momentum/      # 듀얼 모멘텀 전략
    ├── swing_ema/          # 스윙 EMA 전략
    ├── volatility_breakout/ # 변동성 브레이크아웃 전략
    └── ...
```

## 파일 명명 규칙

### 정량 분석 결과
- `quant_analysis_{timestamp}_{symbol}.json`
- `correlation_analysis_{timestamp}_{symbol}.json`
- `regression_analysis_{timestamp}_{symbol}.json`

### 연구 결과
- `research_results_{timestamp}_{strategy}_{symbol}.json`
- `optimization_comparison_{timestamp}.json`
- `best_params_{strategy}_{symbol}_{timestamp}.json`

### 전략 최적화 결과
- `strategy_{strategy_name}_{symbol}_{timestamp}.json`
- `evaluation_{strategy_name}_{symbol}_{timestamp}.json`

## 사용 방법

### 1. 정량 분석 결과 로드
```python
from agent.helper import load_analysis_results

# 최신 정량 분석 결과 로드
analysis_results = load_analysis_results("quant_analysis", symbol="AAPL")
```

### 2. 연구 결과 로드
```python
# 최적화된 파라미터 로드
best_params = load_optimization_results("dual_momentum", symbol="AAPL")
```

### 3. 전략 평가에서 활용
```python
# evaluator에서 최적화된 파라미터 사용
evaluator = StrategyEvaluator(analysis_results_path="analysis/researcher_results/")
```

## 자동 정리

- 30일 이상 된 결과 파일은 자동으로 `archive/` 폴더로 이동
- 매주 일요일 새벽 2시에 자동 정리 실행
- 중요한 결과는 `important/` 폴더에 보관 