# Quant Racoon - 퀀트 전략 테스트 프로그램

이 프로젝트는 여러 퀀트 전략을 테스트하고 적용하는 프로그램입니다.

## 🚀 새로운 기능

### 📊 Time-Horizon 기반 전략 분류
- **스윙 전략**: 15분봉 데이터, 60일 기간
- **장기 전략**: 1일봉 데이터, 1095일(3년) 기간  
- **초단기 전략**: 1분봉 데이터, 7일 기간

### 🔧 자동화 기능
- **자동 정리**: 오래된 로그 파일 자동 삭제
- **자동 백업**: 실행 결과 자동 백업
- **폴더 분리**: 전략별 결과/로그/백업 폴더 분리

### 📋 로그 관리
- **파일 로테이션**: 로그 파일 크기 제한 및 자동 교체
- **레벨별 로깅**: INFO, WARNING, ERROR 레벨 지원
- **JSON 로그**: 실행 결과 JSON 형태로 저장

### 🔬 연구 파이프라인 통합
- **자동 source config 설정**: 현재 실행 중인 config 파일의 심볼과 설정을 자동으로 research config에 적용
- **조건부 연구 실행**: `enable_research` 설정으로 연구 단계 선택적 실행
- **동적 파라미터 최적화**: 전략별 하이퍼파라미터 자동 최적화
- **논리적 순서**: analyzer → researcher → evaluator 순으로 데이터 분석 → 전략 최적화 → 성능 평가

### 📊 정량 분석 시스템
- **다중 분석 방법**: 상관관계, 선형회귀, Lasso, 랜덤포레스트, MLP, 베이지안 분석
- **자동 특성 선택**: 상관관계 기반 상위 특성 자동 선택
- **종합 리포트**: 모든 분석 결과를 통합한 종합 리포트 생성

### 💼 포트폴리오 최적화
- **다양한 최적화 방법**: 샤프 비율, 소르티노 비율, 최소 분산, 최대 분산화 등
- **고급 제약조건**: 비중 제한, 레버리지, 목표 수익률/변동성 설정
- **리스크 관리**: VaR, CVaR, 최대 낙폭 등 고급 리스크 지표

## 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 API 키들을 설정하세요:

```bash
# .env 파일 생성
echo "FINNHUB_API_KEY=your_finnhub_api_key_here" > .env
echo "ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here" >> .env
```

**API 키 발급 방법:**

**Finnhub API 키:**
1. [Finnhub.io](https://finnhub.io/register)에서 무료 계정을 만드세요
2. API 키를 발급받으세요
3. `.env` 파일에 API 키를 설정하세요

**Alpha Vantage API 키 (권장):**
1. [Alpha Vantage](https://www.alphavantage.co/support/#api-key)에서 무료 API 키를 발급받으세요
2. 무료 계정으로도 캔들스틱 데이터에 접근 가능합니다
3. `.env` 파일에 API 키를 설정하세요

## 🎯 사용법

### 빠른 실행 (권장)

#### 개별 전략 실행 (연구 포함)
```bash
# 스윙 전략 실행 (15분봉) - cleaner → scrapper → analyzer → researcher → evaluator → portfolio_manager
./run_swing.sh

# 장기 전략 실행 (1일봉) - cleaner → scrapper → analyzer → researcher → evaluator → portfolio_manager
./run_long.sh

# 초단기 전략 실행 (1분봉) - cleaner → scrapper → analyzer → researcher → evaluator → portfolio_manager
./run_scalping.sh
```

#### 연구 단계 제외 실행
```bash
# 스윙 전략 (연구 단계 제외) - cleaner → scrapper → analyzer → evaluator → portfolio_manager
./run_swing_no_research.sh

# 또는 --no-research 옵션 사용
python -m src.agent.orchestrator --time-horizon swing --no-research
```

#### 모든 전략 실행
```bash
# 모든 전략 순차 실행
./run_all_strategies.sh
```

#### 폴더 정리
```bash
# 모든 전략별 폴더 정리 (data 폴더는 유지)
./clean_all.sh

# 개별 cleaner 실행
python -m src.agent.cleaner --action clean-and-recreate
```

#### Python 직접 실행
```bash
# time-horizon 기반 실행
python -m src.agent.orchestrator --time-horizon swing
python -m src.agent.orchestrator --time-horizon long
python -m src.agent.orchestrator --time-horizon scalping

# 특정 단계만 실행
python -m src.agent.orchestrator --time-horizon swing --stage evaluator
python -m src.agent.orchestrator --time-horizon swing --stage analyzer
python -m src.agent.orchestrator --time-horizon swing --stage portfolio_manager

# 개별 모듈 직접 실행
python -m src.agent.quant_analyst --symbols AAPL QQQ SPY
python -m src.agent.portfolio_manager --method sharpe_maximization --compare
```

### 📁 결과 폴더 구조

실행 후 다음과 같은 폴더 구조가 생성됩니다:

```
quant-racoon/
├── results/
│   ├── swing/          # 스윙 전략 결과
│   ├── long/           # 장기 전략 결과
│   └── scalping/       # 초단기 전략 결과
├── log/
│   ├── swing/          # 스윙 전략 로그
│   ├── long/           # 장기 전략 로그
│   └── scalping/       # 초단기 전략 로그
├── backup/
│   ├── swing/          # 스윙 전략 백업
│   ├── long/           # 장기 전략 백업
│   └── scalping/       # 초단기 전략 백업
└── data/               # 수집된 데이터 (공통)
```

### 🔧 설정 파일

각 전략별 설정 파일:
- `config_swing.json`: 스윙 전략 설정 (15분봉)
- `config_long.json`: 장기 전략 설정 (1일봉)
- `config_scalping.json`: 초단기 전략 설정 (1분봉)

설정 파일 구조:
```json
{
  "time_horizon": "swing",
  "strategies": ["DualMomentumStrategy", "VolatilityAdjustedBreakoutStrategy"],
  "data": {
    "symbols": ["AAPL", "QQQ", "SPY"],
    "interval": "15m",
    "lookback_days": 60
  },
  "evaluator": {
    "portfolio_mode": false
  },
  "cleaner": {
    "run_cleaner": false,
    "action": "create"
  },
  "automation": {
    "auto_clean": true,
    "auto_backup": true,
    "notification": false
  },
  "logging": {
    "level": "INFO",
    "file_rotation": true,
    "max_file_size": "10MB",
    "backup_count": 5
  },
  "output": {
    "results_folder": "results/swing",
    "logs_folder": "log/swing",
    "backup_folder": "backup/swing",
    "separate_strategy_results": true
  },
  "flow": {
    "stages": ["cleaner", "scrapper", "analyzer", "researcher", "evaluator", "portfolio_manager"],
    "stop_on_error": true,
    "enable_research": true
  }
}
```

### 🔧 Cleaner 설정

- **run_cleaner**: cleaner 단계 실행 여부 (기본값: false)
- **action**: cleaner 동작 방식
  - `create`: 폴더만 생성 (기본값)
  - `clean`: 기존 파일만 삭제
  - `clean-and-recreate`: 파일 삭제 후 폴더 재생성
  - `info`: 폴더 정보만 출력

**중요**: 기본적으로 cleaner는 실행되지 않아 data 폴더의 수집된 데이터가 보존됩니다.

### 🔄 Flow 설정

- **stages**: 실행할 단계 목록 (순서대로 실행)
  - `cleaner`: 폴더 정리 및 생성
  - `scrapper`: 데이터 수집
  - `analyzer`: 정량 분석 (상관관계, 회귀분석, 머신러닝)
  - `researcher`: 하이퍼파라미터 최적화 연구
  - `evaluator`: 전략 평가
  - `portfolio_manager`: 포트폴리오 최적화 및 관리
- **stop_on_error**: 오류 발생 시 파이프라인 중단 여부
- **enable_research**: 연구 단계 활성화 여부 (false로 설정하면 researcher 단계 건너뜀)

### Alpha Vantage API 사용법 (권장)

```python
from src.behavior.alpha_vantage import AlphaVantageDataCollector

# 데이터 수집기 초기화
collector = AlphaVantageDataCollector()

# Apple 주식 데이터 수집 (15분봉, 최근 30일)
filepath = collector.collect_and_save_intraday(
    symbol="AAPL",
    interval="15min",  # 15분봉
    outputsize="full"  # 최근 30일
)

print(f"데이터가 저장되었습니다: {filepath}")
```

### Finnhub API 사용법

```python
from src.behavior.finhub import FinnhubDataCollector

# 데이터 수집기 초기화
collector = FinnhubDataCollector()

# Apple 주식 데이터 수집 (15분봉, 최근 30일)
filepath = collector.collect_and_save(
    symbol="AAPL",
    resolution="15",  # 15분봉
    days_back=30
)

print(f"데이터가 저장되었습니다: {filepath}")
```

### 고급 사용법

**Alpha Vantage:**
```python
# 특정 월의 데이터 수집
filepath = collector.collect_and_save_intraday(
    symbol="MSFT",
    interval="60min",  # 1시간봉
    month="2024-01"    # 2024년 1월
)

# 일봉 데이터 수집
filepath = collector.collect_and_save_daily(
    symbol="AAPL",
    outputsize="full"  # 최대 20년 데이터
)
```

**Finnhub:**
```python
# 특정 기간의 데이터 수집
filepath = collector.collect_and_save(
    symbol="MSFT",
    resolution="60",  # 1시간봉
    start_date="2024-01-01",
    end_date="2024-01-31"
)
```

### 직접 실행
```bash
cd quant_racoon
python src/behavior/alpha_vantage.py  # Alpha Vantage
python src/behavior/finhub.py         # Finnhub
```

## 📊 전략 분류

### 스윙 전략 (Swing Trading)
- **보유기간**: 1일~1주
- **데이터**: 15분봉
- **전략들**: DualMomentum, VolatilityAdjustedBreakout, SwingEMA, SwingRSI, DonchianSwing

### 장기 전략 (Long-term Investment)
- **보유기간**: 1개월~1년
- **데이터**: 1일봉
- **전략들**: RiskParityLeverage, FixedWeightRebalance, ETFMomentumRotation, TrendFollowingMA200, ReturnStacking

### 초단기 전략 (Scalping)
- **보유기간**: 분~시간
- **데이터**: 1분봉
- **전략들**: VWAPMACDScalping, KeltnerRSIScalping, AbsorptionScalping, RSIBollingerScalping

## 파일 구조

```
quant-racoon/
├── data/                    # 수집된 데이터 저장 폴더
├── results/                 # 전략별 결과 폴더
│   ├── swing/              # 스윙 전략 결과
│   ├── long/               # 장기 전략 결과
│   └── scalping/           # 초단기 전략 결과
├── log/                     # 전략별 로그 폴더
│   ├── swing/              # 스윙 전략 로그
│   ├── long/               # 장기 전략 로그
│   └── scalping/           # 초단기 전략 로그
├── backup/                  # 전략별 백업 폴더
│   ├── swing/              # 스윙 전략 백업
│   ├── long/               # 장기 전략 백업
│   └── scalping/           # 초단기 전략 백업
├── config/                  # 설정 파일 폴더
│   ├── config_swing.json   # 스윙 전략 설정
│   ├── config_long.json    # 장기 전략 설정
│   ├── config_scalping.json # 초단기 전략 설정
│   ├── config_default.json # 기본 설정
│   └── config_research.json # 연구용 설정
├── src/
│   ├── agent/              # 에이전트 관련 코드
│   │   ├── orchestrator.py # 파이프라인 오케스트레이터
│   │   ├── cleaner.py      # 폴더 정리 도구
│   │   ├── scrapper.py     # 데이터 수집기
│   │   ├── evaluator.py    # 전략 평가기
│   │   └── helper.py       # 공통 유틸리티
│   └── behavior/           # 전략 및 데이터 수집 코드
│       ├── alpha_vantage.py # Alpha Vantage 데이터 수집기 (권장)
│       ├── finhub.py       # Finnhub 데이터 수집기
│       ├── backtest.py     # 백테스팅 로직
│       └── strategy.py     # 퀀트 전략 구현
├── run_*.sh               # 실행 스크립트
├── requirements.txt        # Python 의존성
└── README.md              # 이 파일
```

## 데이터 형식

수집된 CSV 파일은 다음 컬럼을 포함합니다:
- `datetime`: 전체 날짜시간
- `date`: 날짜
- `time`: 시간
- `timestamp`: Unix 타임스탬프
- `open`: 시가
- `high`: 고가
- `low`: 저가
- `close`: 종가
- `volume`: 거래량

## 파일명 규칙

저장되는 파일명은 다음 형식을 따릅니다:
```
{티커}_{시간단위}_{시작날짜}_{종료날짜}_{오늘날짜}.csv
```

예시: `AAPL_15min_2024-01-01_2024-01-31_20241201.csv` 