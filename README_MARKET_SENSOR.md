# Market Sensor - 통합 시장 분석 시스템

## 🎯 개요
Market Sensor는 SPY ETF를 대상으로 한 정량적 트레이딩 전략 시스템입니다. 시장 상태 분류, 하이퍼파라미터 최적화, 매크로/섹터 분석을 통합하여 최적의 트레이딩 전략을 제공합니다.

## 🚀 빠른 시작

### 1. 전체 워크플로우 실행 (권장)
```bash
# 기본 실행 (20회 시도, 종합 분석)
./run_market_analysis.sh

# 커스텀 설정
./run_market_analysis.sh 50 technical    # 50회 시도, 기술적 분석
./run_market_analysis.sh 100 all         # 100회 시도, 모든 분석
```

### 2. 빠른 분석 (기존 데이터/파라미터 사용)
```bash
# 기본 종합 분석
./quick_analysis.sh

# 특정 분석 유형
./quick_analysis.sh technical
./quick_analysis.sh macro
./quick_analysis.sh sector
./quick_analysis.sh comprehensive
./quick_analysis.sh all
```

## 📊 분석 유형

### 1. Technical Analysis (기술적 분석)
- 시장 상태 분류 (TRENDING_UP, TRENDING_DOWN, SIDEWAYS, VOLATILE, UNCERTAIN)
- 기술적 지표 분석 (RSI, MACD, SMA, ATR 등)
- 성과 지표 (수익률, 샤프 비율, 최대 낙폭, 승률)

### 2. Macro Analysis (매크로 분석)
- 시장 조건 분류 (BULL_MARKET, BEAR_MARKET, SIDEWAYS_MARKET 등)
- 인플레이션/금리 환경 분석
- TIPS 스프레드 분석
- 성장 전망 평가

### 3. Sector Analysis (섹터 분석)
- 섹터 로테이션 분석
- 선도/후행/방어적/순환적 섹터 분류
- 과중/과소 배치 추천

### 4. Comprehensive Analysis (종합 분석)
- 기술적 + 매크로 + 섹터 분석 통합
- 최적화된 파라미터 기반 시장 분류
- 전략 추천 (포지션 크기, 손절/익절 등)

### 5. All Analysis (모든 분석)
- 모든 분석 유형을 한 번에 실행
- 각 분석 결과를 개별적으로 제공

## 🔧 수동 실행 옵션

### 데이터 수집
```bash
# 새로운 데이터 다운로드
python -m src.agent.market_sensor --mode collect --force_download

# 저장된 데이터 사용
python -m src.agent.market_sensor --mode collect --use_saved_data
```

### 하이퍼파라미터 튜닝
```bash
# 기본 튜닝 (50회 시도)
python -m src.agent.market_sensor --mode experiment --use_saved_data --save_results

# 커스텀 튜닝
python -m src.agent.market_sensor --mode experiment --use_saved_data --save_results --n_trials 100
```

### 분석 실행
```bash
# 기술적 분석
python -m src.agent.market_sensor --analysis technical --use_saved_data

# 매크로 분석
python -m src.agent.market_sensor --analysis macro --use_saved_data

# 섹터 분석
python -m src.agent.market_sensor --analysis sector --use_saved_data

# 종합 분석
python -m src.agent.market_sensor --analysis comprehensive --use_saved_data

# 모든 분석
python -m src.agent.market_sensor --analysis all --use_saved_data
```

## 📁 결과 파일

### 최적화 결과
- `results/market_sensor_optimization/{session_uuid}/`
  - `best_params.json` - 최적 파라미터
  - `performance_summary.json` - 성과 지표
  - `optuna_study.json` - 최적화 과정
  - `metadata.json` - 메타데이터

### 분석 결과
- `results/analysis_{type}/{session_uuid}/`
  - `analysis_{type}_{timestamp}.json` - 분석 결과

### 거래 로그
- `log/market_sensor/`
  - `transaction_market_sensor_{timestamp}.log` - 거래 내역

## ⚙️ 설정 파일

### 하이퍼파라미터 설정
- `config/config_macro.json` - 최적화 대상 파라미터 및 범위

### 최적화된 파라미터
- `config/optimal_market_params.json` - 튜닝 완료된 최적 파라미터

## 📈 시장 상태 분류

### Market Regime (시장 상태)
- **TRENDING_UP**: 상승 추세 - Buy & Hold 우선, 스윙 전략 보조
- **TRENDING_DOWN**: 하락 추세 - 현금 보유 또는 역방향 전략
- **SIDEWAYS**: 횡보장 - 스윙 전략 적극 활용
- **VOLATILE**: 변동성 높음 - 포지션 크기 축소 + 단기 전략
- **UNCERTAIN**: 불확실 - 관망 후 진입 권장

### Market Condition (매크로 기반)
- **BULL_MARKET**: 강세장
- **BEAR_MARKET**: 약세장
- **SIDEWAYS_MARKET**: 횡보장
- **VOLATILE_MARKET**: 변동성 장
- **RECESSION_FEAR**: 경기침체 우려
- **INFLATION_FEAR**: 인플레이션 우려

## 🔍 주요 기능

### 1. 시장 분류
- 20+ 기술적 지표 기반 시장 상태 분류
- 매크로 지표 통합 분석
- 섹터 로테이션 분석

### 2. 하이퍼파라미터 최적화
- Optuna 기반 베이지안 최적화
- 다중 목적 함수 지원
- 교차 검증 기반 성능 평가

### 3. 거래 전략
- 시장 상태별 포지션 크기 조절
- ATR 기반 손절/익절
- 트레일링 스탑

### 4. 성과 분석
- 총 수익률, 샤프 비율, 최대 낙폭
- Buy & Hold 대비 초과 수익률
- 승률 및 거래 통계

## 🛠️ 기술 스택

- **Python**: 3.8+
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Optuna**: 하이퍼파라미터 최적화
- **Yahoo Finance**: 데이터 수집
- **Matplotlib/Plotly**: 시각화

## 📝 사용 팁

1. **첫 실행**: `./run_market_analysis.sh`로 전체 워크플로우 실행
2. **일상적 분석**: `./quick_analysis.sh`로 빠른 분석
3. **파라미터 조정**: `config/config_macro.json`에서 최적화 범위 수정
4. **결과 확인**: `results/` 디렉토리에서 상세 결과 확인

## 🔄 워크플로우

```
데이터 수집 → 하이퍼파라미터 튜닝 → 시장 분석 → 전략 추천
     ↓              ↓              ↓           ↓
  SPY/매크로/    최적 파라미터   시장 상태    포지션/리스크
   섹터 데이터     도출        분류        관리 전략
``` 