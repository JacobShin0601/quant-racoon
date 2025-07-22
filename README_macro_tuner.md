# 매크로 하이퍼파라미터 튜너 (Macro Hyperparameter Tuner)

시장 상태를 분류하고 최적의 하이퍼파라미터를 찾아 SPY 거래 전략을 최적화하는 시스템입니다.

## 🎯 주요 기능

### 시장 상태 분류
- **TRENDING_UP**: Buy & Hold 우선, 스윙 전략 보조
- **TRENDING_DOWN**: 현금 보유 또는 역방향 전략
- **SIDEWAYS**: 스윙 전략 적극 활용
- **VOLATILE**: 포지션 크기 축소 + 단기 전략

### 하이퍼파라미터 최적화
- Optuna를 사용한 베이지안 최적화
- Train/Test 분할을 통한 과적합 방지
- Buy & Hold 대비 성과 비교

## 📁 파일 구조

```
quant-racoon/
├── config/
│   └── config_macro.json          # 하이퍼파라미터 설정
├── src/actions/
│   └── global_macro.py            # 메인 클래스들
├── test_macro_tuner.py            # 테스트 스크립트
└── README_macro_tuner.md          # 이 파일
```

## 🚀 사용법

### 1. 데이터 수집

```bash
# 기본 데이터 수집 (최근 1년)
python src/actions/global_macro.py --mode collect

# 특정 기간 데이터 수집
python src/actions/global_macro.py --mode collect --start_date 2023-01-01 --end_date 2024-01-01
```

### 2. 하이퍼파라미터 튜닝

```bash
# 기본 튜닝 (50회 시도)
python src/actions/global_macro.py --mode optimize

# 더 많은 시도 횟수로 튜닝
python src/actions/global_macro.py --mode optimize --n_trials 100

# 특정 기간으로 튜닝
python src/actions/global_macro.py --mode optimize --start_date 2022-01-01 --end_date 2024-01-01 --n_trials 200
```

### 3. 테스트 실행

```bash
# 전체 테스트 실행
python test_macro_tuner.py
```

## ⚙️ 설정 파일 (config_macro.json)

### 시장 상태 분류 설정

```json
{
  "market_regime_classification": {
    "indicators": {
      "trend_indicators": {
        "sma_short": {"min": 5, "max": 30, "type": "int"},
        "sma_medium": {"min": 20, "max": 60, "type": "int"},
        "sma_long": {"min": 50, "max": 200, "type": "int"}
      },
      "momentum_indicators": {
        "rsi_period": {"min": 10, "max": 30, "type": "int"},
        "rsi_overbought": {"min": 65, "max": 85, "type": "int"},
        "rsi_oversold": {"min": 15, "max": 35, "type": "int"}
      }
    },
    "classification_weights": {
      "trend_weight": {"min": 0.2, "max": 0.6, "type": "float"},
      "momentum_weight": {"min": 0.1, "max": 0.4, "type": "float"},
      "volatility_weight": {"min": 0.1, "max": 0.4, "type": "float"},
      "macro_weight": {"min": 0.1, "max": 0.3, "type": "float"}
    }
  }
}
```

### 거래 전략 설정

```json
{
  "trading_strategy": {
    "position_sizing": {
      "base_position": {"min": 0.5, "max": 1.0, "type": "float"},
      "volatile_reduction": {"min": 0.3, "max": 0.7, "type": "float"},
      "trending_boost": {"min": 1.0, "max": 1.5, "type": "float"}
    }
  }
}
```

### 최적화 설정

```json
{
  "optimization": {
    "n_trials": 100,
    "train_test_split": 0.8,
    "objective": "sharpe_ratio"
  }
}
```

## 📊 성과 지표

- **total_return**: 총 수익률
- **buy_hold_return**: Buy & Hold 수익률
- **excess_return**: 초과 수익률
- **sharpe_ratio**: 샤프 비율
- **max_drawdown**: 최대 낙폭
- **win_rate**: 승률

## 🔧 주요 클래스

### HyperparamTuner

```python
from src.actions.global_macro import HyperparamTuner

# 튜너 초기화
tuner = HyperparamTuner()

# 하이퍼파라미터 최적화
results = tuner.optimize_hyperparameters(
    start_date="2023-01-01",
    end_date="2024-01-01",
    n_trials=100
)

# 결과 저장
tuner.save_results(results)
```

### GlobalMacroDataCollector

```python
from src.actions.global_macro import GlobalMacroDataCollector

# 데이터 수집기 초기화
collector = GlobalMacroDataCollector()

# SPY 데이터 수집
spy_data = collector.collect_spy_data("2023-01-01", "2024-01-01")

# 매크로 지표 수집
macro_data = collector.collect_macro_indicators("2023-01-01", "2024-01-01")
```

## 📈 시장 상태별 전략

### TRENDING_UP
- **전략**: Buy & Hold 우선, 스윙 전략 보조
- **포지션**: 기본 포지션의 120% (trending_boost)
- **적용 시기**: 상승 트렌드가 명확할 때

### TRENDING_DOWN
- **전략**: 현금 보유 또는 역방향 전략
- **포지션**: 기본 포지션의 -50% (역방향)
- **적용 시기**: 하락 트렌드가 명확할 때

### SIDEWAYS
- **전략**: 스윙 전략 적극 활용
- **포지션**: RSI 기반 진입/청산
- **적용 시기**: 횡보장에서 RSI 과매수/과매도 활용

### VOLATILE
- **전략**: 포지션 크기 축소 + 단기 전략
- **포지션**: 기본 포지션의 50% (volatile_reduction)
- **적용 시기**: 높은 변동성 환경

## 🎯 최적화 과정

1. **데이터 수집**: SPY 및 매크로 지표 데이터 수집
2. **Train/Test 분할**: 80% 훈련, 20% 테스트
3. **파생 변수 계산**: 하이퍼파라미터 기반 기술적 지표
4. **시장 상태 분류**: 4가지 시장 상태로 분류
5. **전략 수익률 계산**: 시장 상태별 전략 적용
6. **성과 평가**: Buy & Hold 대비 성과 측정
7. **최적화**: Optuna를 통한 하이퍼파라미터 최적화

## 📝 결과 파일

최적화 완료 후 다음 파일들이 생성됩니다:

```
results/macro_optimization/
├── best_params.json           # 최적 하이퍼파라미터
├── performance_summary.json   # 성과 요약
└── optuna_study.csv          # Optuna 최적화 과정
```

## ⚠️ 주의사항

1. **데이터 의존성**: Yahoo Finance API에 의존하므로 인터넷 연결 필요
2. **최적화 시간**: 많은 시도 횟수는 긴 시간이 소요될 수 있음
3. **과적합 위험**: Train/Test 분할을 통해 완전히 방지할 수는 없음
4. **시장 변화**: 과거 데이터로 최적화된 파라미터가 미래에도 유효하지 않을 수 있음

## 🔄 정기 업데이트

시장 상황 변화에 따라 정기적으로 하이퍼파라미터를 재최적화하는 것을 권장합니다:

```bash
# 월 1회 재최적화
python src/actions/global_macro.py --mode optimize --n_trials 200
``` 