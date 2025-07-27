# Threshold 최적화 시스템

하드코딩된 BUY/HOLD/SELL threshold를 실제 거래 성과 기반으로 최적화하는 시스템입니다.

## 🎯 개요

기존 시스템에서는 매매 신호를 결정하는 threshold가 하드코딩되어 있었습니다:
- `strong_buy`: 0.7
- `buy`: 0.5  
- `hold_upper`: 0.5
- `hold_lower`: -0.5
- `sell`: -0.5
- `strong_sell`: -0.7

이 시스템은 실제 거래 성과를 기반으로 이러한 threshold를 최적화하여 더 나은 수익률과 리스크 조정 수익률을 달성합니다.

## 🏗️ 시스템 아키텍처

### 1. Train-Test 분할
- **Train 데이터**: 전체 데이터의 70% (기본값)
- **Test 데이터**: 전체 데이터의 30% (기본값)
- 시간 순서를 유지한 분할로 과적합 방지

### 2. 백테스팅 시스템
- 실제 거래와 유사한 시뮬레이션
- 매수/매도 신호에 따른 포지션 관리
- 수수료, 슬리피지 고려
- 복리 효과를 고려한 수익률 계산

### 3. 최적화 방법

#### A. Grid Search (그리드 서치)
- 모든 가능한 threshold 조합을 체계적으로 테스트
- 장점: 전역 최적해 보장
- 단점: 계산 시간이 오래 걸림

#### B. Optuna (베이지안 최적화)
- 지능적인 하이퍼파라미터 최적화
- 이전 결과를 바탕으로 다음 시도 지점 선택
- 장점: 빠른 수렴, 효율적 탐색
- 단점: 전역 최적해 보장하지 않음

#### C. Neural Network (신경망 최적화)
- 신경망을 사용하여 threshold 조합 학습
- 연속적인 최적화 공간 탐색
- 장점: 복잡한 패턴 학습 가능
- 단점: 학습 안정성 이슈 가능

### 4. 목표 지표
- **total_return**: 총 수익률
- **sharpe_ratio**: 샤프 비율 (위험 조정 수익률)
- **sortino_ratio**: 소르티노 비율 (하방 위험 조정 수익률)

## 🚀 사용법

### 1. 기본 사용법

```bash
# 그리드 서치로 샤프 비율 최적화
./run_threshold_optimizer.sh -m grid_search -o sharpe_ratio

# Optuna로 총 수익률 최적화 (100회 시도)
./run_threshold_optimizer.sh -m optuna -o total_return -t 100

# 신경망으로 소르티노 비율 최적화
./run_threshold_optimizer.sh -m neural_network -o sortino_ratio
```

### 2. 고급 옵션

```bash
# 특정 종목만 최적화
./run_threshold_optimizer.sh -s AAPL,META,NFLX

# 캐시된 데이터 사용
./run_threshold_optimizer.sh --use-cached-data

# 강제 재실행 (기존 모델 무시)
./run_threshold_optimizer.sh -f
```

### 3. Python 직접 실행

```python
from src.actions.threshold_optimizer import ThresholdOptimizer
import json

# 설정 로드
with open('config/config_trader.json', 'r') as f:
    config = json.load(f)

# 최적화 실행
optimizer = ThresholdOptimizer(config)
results = optimizer.run_optimization(['AAPL', 'META', 'NFLX'])

print(f"최적 threshold: {results['best_thresholds']}")
print(f"최고 점수: {results['optimization_results']['best_score']}")
```

## 📊 결과 분석

### 1. 결과 파일 구조

```
results/trader/
├── threshold_optimization_all_20250725_121610.json    # 모든 시도 결과
├── threshold_optimization_best_20250725_121610.json   # 최적 결과
├── threshold_optimization_summary_20250725_121610.json # 요약 정보
└── threshold_optimization_final_20250725_121610.json  # 최종 결과
```

### 2. 결과 해석

```json
{
  "best_thresholds": {
    "strong_buy": 0.6,
    "buy": 0.3,
    "hold_upper": 0.1,
    "hold_lower": -0.1,
    "sell": -0.3,
    "strong_sell": -0.6
  },
  "best_score": 1.234,
  "train_score": 1.156,
  "test_score": 1.312,
  "optimization_method": "optuna"
}
```

### 3. 성과 비교

- **Train 성과**: 과적합 여부 확인
- **Test 성과**: 일반화 성능 평가
- **평균 점수**: Train과 Test의 평균 (최적화 목표)

## ⚙️ 설정 옵션

### config_trader.json 설정

```json
{
  "threshold_optimization": {
    "method": "grid_search",
    "objective_metric": "sharpe_ratio",
    "n_trials": 100,
    "train_ratio": 0.7,
    "min_data_points": 100,
    "threshold_ranges": {
      "strong_buy": [0.5, 0.9],
      "buy": [0.3, 0.7],
      "hold_upper": [0.1, 0.5],
      "hold_lower": [-0.5, -0.1],
      "sell": [-0.7, -0.3],
      "strong_sell": [-0.9, -0.5]
    }
  }
}
```

## 🔧 고급 기능

### 1. 종목별 최적화
각 종목의 특성에 맞는 개별 threshold 최적화

### 2. 시장 체제별 최적화
BULLISH, BEARISH, SIDEWAYS, VOLATILE 체제별 다른 threshold

### 3. 동적 최적화
주기적으로 threshold를 재최적화하여 시장 변화에 적응

### 4. 앙상블 최적화
여러 최적화 방법의 결과를 조합하여 더 안정적인 threshold 도출

## 📈 성능 지표

### 1. 백테스팅 지표
- **총 수익률**: 전체 기간 수익률
- **샤프 비율**: 위험 조정 수익률
- **소르티노 비율**: 하방 위험 조정 수익률
- **최대 낙폭**: 최대 손실 구간
- **승률**: 수익 거래 비율
- **수익 팩터**: 총 수익 / 총 손실

### 2. 검증 지표
- **Train-Test 일관성**: 과적합 여부
- **안정성**: 다양한 시장 환경에서의 성과
- **실용성**: 실제 거래 가능성

## 🚨 주의사항

### 1. 과적합 위험
- Train 데이터에서만 좋은 성과를 보이는 threshold 주의
- Test 데이터에서도 일관된 성과 확인 필요

### 2. 데이터 품질
- 충분한 데이터 포인트 확보 (최소 100일)
- 데이터 품질 검증 필요

### 3. 시장 변화
- 시장 환경 변화에 따른 threshold 재최적화 필요
- 정기적인 성과 모니터링 권장

## 🔄 워크플로우

1. **데이터 수집**: Yahoo Finance에서 주가 데이터 수집
2. **데이터 전처리**: 결측치 처리, 이상치 제거
3. **Train-Test 분할**: 시간 순서 유지한 분할
4. **최적화 실행**: 선택한 방법으로 threshold 최적화
5. **성과 검증**: Test 데이터로 일반화 성능 확인
6. **결과 저장**: JSON 형태로 결과 저장
7. **모델 적용**: 최적 threshold를 실제 거래 시스템에 적용

## 📝 예시 결과

### 최적화 전 (하드코딩)
```
strong_buy: 0.7, buy: 0.5, sell: -0.5, strong_sell: -0.7
평균 샤프 비율: 0.85
```

### 최적화 후
```
strong_buy: 0.6, buy: 0.3, sell: -0.3, strong_sell: -0.6
평균 샤프 비율: 1.23 (44% 개선)
```

## 🤝 기여하기

1. 새로운 최적화 방법 제안
2. 성과 지표 추가
3. 백테스팅 로직 개선
4. 문서화 개선

## 📞 문의

시스템 사용 중 문제가 발생하거나 개선 제안이 있으시면 이슈를 등록해 주세요. 