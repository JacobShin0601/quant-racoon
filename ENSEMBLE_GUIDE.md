# 앙상블 전략 사용 가이드

## 개요

앙상블 전략은 시장 환경(Market Regime)에 따라 최적의 전략을 선택하고 실행하는 고급 시스템입니다. Market Sensor가 감지한 시장 상황에 맞춰 다양한 전략을 기간별로 최적화합니다.

## 1. 앙상블 전략 실행

### 기본 실행
```bash
# 전체 앙상블 파이프라인 실행
./run_ensemble.sh
```

### 수동 실행
```bash
# Python으로 직접 실행
python -m src.actions.ensemble --config config/config_ensemble.json --time-horizon ensemble
```

## 2. 결과 조회 방법

### 2.1 최신 결과 조회
```bash
# 가장 최근 실행 결과 조회
python view_ensemble_results.py

# 상세 정보 포함 조회
python view_ensemble_results.py --detailed
```

### 2.2 특정 UUID 결과 조회
```bash
# 특정 실행 결과 조회 (UUID는 실행 시 생성됨)
python view_ensemble_results.py --uuid 20250123_154722_17b6186e

# 상세 정보 포함
python view_ensemble_results.py --uuid 20250123_154722_17b6186e --detailed
```

### 2.3 모든 결과 목록 조회
```bash
# 실행된 모든 앙상블 결과 목록 확인
python view_ensemble_results.py --list
```

### 2.4 두 결과 비교
```bash
# 두 실행 결과 비교
python view_ensemble_results.py --compare --uuid UUID1 --uuid2 UUID2
```

## 3. Evaluator와의 차이점

### 3.1 왜 Evaluator를 직접 사용하지 않는가?

1. **다중 기간 분석**: 앙상블은 시점별로 다른 시장 환경을 분석하여 기간별 최적화를 수행
2. **환경별 전략**: Market Regime에 따라 다른 전략 조합을 사용
3. **종합적 평가**: 개별 전략이 아닌 시장 환경별 전략 포트폴리오 성과를 평가

### 3.2 내부적으로 Evaluator 사용

앙상블 시스템은 각 기간별 최적화 과정에서 Orchestrator를 통해 내부적으로 Evaluator를 호출합니다:
- `run_period_optimization()` → `Orchestrator` → `researcher` → `evaluator`

## 4. 결과 파일 구조

### 4.1 주요 결과 파일
```
results/ensemble/
├── ensemble_results_UUID.json      # 상세 실행 결과 (JSON)
├── ensemble_summary_UUID.txt       # 요약 리포트 (텍스트)
├── optimization_results_*.json     # 개별 최적화 결과
└── comprehensive_evaluation_*.txt   # 개별 평가 결과
```

### 4.2 백업 파일
```
backup/ensemble/backup_UUID/
├── data/ensemble/                   # 데이터 백업
├── log/ensemble/                    # 로그 백업
├── results/ensemble/                # 결과 백업
├── models/market_regime/            # 모델 백업
└── backup_info.json                # 백업 정보
```

## 5. 시장 환경별 전략

### 5.1 환경 분류
- **TRENDING_UP**: 상승 추세 시장
- **TRENDING_DOWN**: 하락 추세 시장  
- **VOLATILE**: 변동성이 높은 시장
- **SIDEWAYS**: 횡보 시장
- **UNCERTAIN**: 불확실한 시장

### 5.2 환경별 최적 전략
```
TRENDING_UP:
- dual_momentum
- volatility_breakout
- swing_ema

TRENDING_DOWN:
- mean_reversion
- swing_rsi
- stochastic

VOLATILE:
- volatility_filtered_breakout
- multi_timeframe_whipsaw

SIDEWAYS:
- mean_reversion
- swing_rsi
- swing_bollinger_band
```

## 6. 성과 지표

### 6.1 앙상블 성과 지표
- **실행 성공률**: 각 기간별 최적화 성공 비율
- **시장 환경 감지 정확도**: Market Sensor의 환경 분류 정확도
- **기간별 수익률**: 각 시장 환경에서의 전략 성과
- **전체 포트폴리오 성과**: 전 기간 종합 성과

### 6.2 개별 기간 성과
각 기간별로 다음 지표들이 계산됩니다:
- 샤프 비율 (Sharpe Ratio)
- 소르티노 비율 (Sortino Ratio)
- 최대 낙폭 (Max Drawdown)
- 변동성 (Volatility)
- 승률 (Win Rate)

## 7. 문제 해결

### 7.1 일반적인 오류

**오류: 'end_time' 관련 오류**
- 원인: 결과 저장 시 필수 키 누락
- 해결: 최신 코드로 업데이트됨 (안전 처리 추가)

**오류: 결과 파일을 찾을 수 없음**
- 원인: 잘못된 UUID 또는 결과 파일 미생성
- 해결: `python view_ensemble_results.py --list`로 사용 가능한 UUID 확인

**오류: Market Sensor 모델 없음**
- 원인: Random Forest 모델이 학습되지 않음
- 해결: `./run_ensemble.sh`가 자동으로 모델을 학습하거나 로드

### 7.2 성능 최적화

**긴 실행 시간**
- 베이지안 최적화 횟수 조정: `config_ensemble.json`에서 `n_trials` 값 변경
- 전략 수 줄이기: 각 환경별 config에서 `strategies` 목록 축소

**메모리 사용량 높음**
- 심볼 수 줄이기: `config_ensemble.json`에서 `symbols` 목록 축소
- 데이터 기간 단축: `lookback_days` 값 줄이기

## 8. 고급 사용법

### 8.1 커스텀 시장 환경 설정
```json
{
  "market_regime_configs": {
    "CUSTOM_REGIME": "config/config_ensemble_custom.json"
  }
}
```

### 8.2 특정 기간 백테스팅
```bash
python -m src.actions.ensemble --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

### 8.3 결과 데이터 분석
```python
# Python에서 결과 데이터 로드
import json
with open('results/ensemble/ensemble_results_UUID.json', 'r') as f:
    results = json.load(f)

# 기간별 성과 분석
for period in results['regime_periods']:
    print(f"{period['regime']}: {period['start_date']} ~ {period['end_date']}")
```

## 9. 참고사항

- 앙상블 실행은 시간이 오래 걸릴 수 있습니다 (1-2시간)
- 충분한 디스크 공간을 확보하세요 (백업 포함)
- 정기적으로 오래된 백업 파일을 정리하세요
- Market Sensor 모델은 주기적으로 재학습하는 것을 권장합니다 