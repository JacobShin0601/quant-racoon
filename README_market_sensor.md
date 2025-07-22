# Market Sensor - 통합 시장 환경 분석 시스템

시장 환경 분류, 하이퍼파라미터 튜닝, 전략 추천을 종합적으로 수행하는 시스템입니다.

## 🎯 주요 기능

### 1. 시장 환경 분류
- **TRENDING_UP**: 상승 추세 - Buy & Hold 우선, 스윙 전략 보조
- **TRENDING_DOWN**: 하락 추세 - 현금 보유 또는 역방향 전략
- **SIDEWAYS**: 횡보장 - 스윙 전략 적극 활용
- **VOLATILE**: 변동성 높음 - 포지션 크기 축소 + 단기 전략
- **UNCERTAIN**: 불확실 - 관망 후 진입 권장

### 2. 하이퍼파라미터 최적화
- Optuna를 사용한 베이지안 최적화
- Train/Test 분할을 통한 과적합 방지
- Buy & Hold 대비 성과 비교

### 3. 전략 추천
- 시장 상태별 최적 전략 자동 추천
- 포지션 크기 및 리스크 관리 설정
- 실시간 시장 분석 결과 제공

## 📁 파일 구조

```
quant-racoon/
├── config/
│   └── config_macro.json                    # 하이퍼파라미터 설정
├── src/agent/
│   └── market_sensor.py                     # 메인 Market Sensor 클래스
├── test_market_sensor.py                    # 통합 테스트 스크립트
└── README_market_sensor.md                  # 이 파일
```

## 🚀 사용법

### 1. 명령행 인터페이스

#### 현재 시장 분석
```bash
# 기본 파라미터로 현재 시장 분석
python src/agent/market_sensor.py --mode analyze

# 최적화된 파라미터로 현재 시장 분석
python src/agent/market_sensor.py --mode analyze --use_optimized
```

#### 하이퍼파라미터 튜닝
```bash
# 기본 튜닝 (50회 시도)
python src/agent/market_sensor.py --mode optimize

# 더 많은 시도 횟수로 튜닝
python src/agent/market_sensor.py --mode optimize --n_trials 100

# 특정 기간으로 튜닝
python src/agent/market_sensor.py --mode optimize --start_date 2023-01-01 --end_date 2024-01-01 --n_trials 200
```

#### 데이터 수집
```bash
# 최신 매크로 데이터 수집
python src/agent/market_sensor.py --mode collect
```

### 2. Python API 사용

#### 기본 사용법
```python
from src.agent.market_sensor import MarketSensor

# Market Sensor 초기화
sensor = MarketSensor()

# 현재 시장 분석
analysis = sensor.get_current_market_analysis()
print(f"현재 시장 환경: {analysis['current_regime']}")
print(f"추천 전략: {analysis['recommendation']['primary_strategy']}")
```

#### 하이퍼파라미터 최적화
```python
# 하이퍼파라미터 최적화
results = sensor.optimize_hyperparameters_optuna(
    start_date="2023-01-01",
    end_date="2024-01-01",
    n_trials=100
)

print(f"최적 샤프 비율: {results['best_value']:.4f}")
print(f"최적 파라미터: {results['best_params']}")

# 결과 저장
sensor.save_optimization_results(results)
```

#### 데이터 수집
```python
# 새로운 데이터 수집
spy_data, macro_data, sector_data = sensor._collect_fresh_data()
print(f"SPY 데이터: {len(spy_data)}개")
```

### 3. 통합 테스트

```bash
# 전체 기능 테스트
python test_market_sensor.py
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

## 📊 성과 지표

- **total_return**: 총 수익률
- **buy_hold_return**: Buy & Hold 수익률
- **excess_return**: 초과 수익률
- **sharpe_ratio**: 샤프 비율
- **max_drawdown**: 최대 낙폭
- **win_rate**: 승률

## 🔧 주요 클래스 및 메서드

### MarketSensor

#### 핵심 메서드
- `get_current_market_analysis()`: 현재 시장 분석
- `optimize_hyperparameters_optuna()`: 하이퍼파라미터 최적화
- `_collect_fresh_data()`: 새로운 데이터 수집
- `save_optimization_results()`: 최적화 결과 저장

#### 내부 메서드
- `_calculate_derived_features()`: 파생 변수 계산
- `_classify_market_regime_optimized()`: 시장 상태 분류
- `_calculate_strategy_returns()`: 전략 수익률 계산
- `_calculate_performance_metrics()`: 성과 지표 계산

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

### UNCERTAIN
- **전략**: 관망 후 진입
- **포지션**: 최소 포지션 (20%)
- **적용 시기**: 불확실한 시장 환경

## 🎯 최적화 과정

1. **데이터 수집**: SPY 및 매크로 지표 데이터 수집
2. **Train/Test 분할**: 80% 훈련, 20% 테스트
3. **파생 변수 계산**: 하이퍼파라미터 기반 기술적 지표
4. **시장 상태 분류**: 5가지 시장 상태로 분류
5. **전략 수익률 계산**: 시장 상태별 전략 적용
6. **성과 평가**: Buy & Hold 대비 성과 측정
7. **최적화**: Optuna를 통한 하이퍼파라미터 최적화

## 📝 결과 파일

최적화 완료 후 다음 파일들이 생성됩니다:

```
results/market_sensor_optimization/
├── best_params.json           # 최적 하이퍼파라미터
├── performance_summary.json   # 성과 요약
└── optuna_study.csv          # Optuna 최적화 과정
```

## 🔄 워크플로우

### 1. 초기 설정
```bash
# 데이터 수집
python src/agent/market_sensor.py --mode collect

# 하이퍼파라미터 최적화
python src/agent/market_sensor.py --mode optimize --n_trials 100
```

### 2. 일상적 사용
```bash
# 현재 시장 분석 (최적화된 파라미터 사용)
python src/agent/market_sensor.py --mode analyze --use_optimized
```

### 3. 정기 업데이트
```bash
# 월 1회 재최적화
python src/agent/market_sensor.py --mode optimize --n_trials 200
```

## ⚠️ 주의사항

1. **데이터 의존성**: Yahoo Finance API에 의존하므로 인터넷 연결 필요
2. **최적화 시간**: 많은 시도 횟수는 긴 시간이 소요될 수 있음
3. **과적합 위험**: Train/Test 분할을 통해 완전히 방지할 수는 없음
4. **시장 변화**: 과거 데이터로 최적화된 파라미터가 미래에도 유효하지 않을 수 있음
5. **메모리 사용량**: 대용량 데이터 처리 시 충분한 메모리 확보 필요

## 🧪 테스트

### 개별 테스트
```bash
# 데이터 수집 테스트
python test_market_sensor.py

# 특정 기능만 테스트하려면 test_market_sensor.py 파일을 수정
```

### 통합 테스트
```bash
# 전체 워크플로우 테스트
python test_market_sensor.py
```

## 🔧 문제 해결

### 일반적인 문제들

1. **데이터 수집 실패**
   - 인터넷 연결 확인
   - Yahoo Finance API 상태 확인
   - 날짜 범위 조정

2. **최적화 실패**
   - 시도 횟수 줄이기 (n_trials 감소)
   - 날짜 범위 축소
   - 메모리 부족 시 데이터 기간 단축

3. **분석 결과 오류**
   - 설정 파일 확인
   - 데이터 파일 존재 여부 확인
   - 로그 파일 확인

## 📞 지원

문제가 발생하면 다음을 확인해주세요:
1. 로그 파일 확인
2. 설정 파일 문법 검사
3. 의존성 패키지 설치 상태 확인
4. 데이터 파일 존재 여부 확인 