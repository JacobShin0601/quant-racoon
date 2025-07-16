# Quant Racoon - 퀀트 전략 테스트 프로그램

이 프로젝트는 여러 퀀트 전략을 테스트하고 적용하는 프로그램입니다.

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

## 사용법

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

## 파일 구조

```
quant_racoon/
├── data/                    # 수집된 데이터 저장 폴더
├── src/
│   ├── agent/              # 에이전트 관련 코드
│   └── behavior/           # 전략 및 데이터 수집 코드
│       ├── alpha_vantage.py # Alpha Vantage 데이터 수집기 (권장)
│       ├── finhub.py       # Finnhub 데이터 수집기
│       ├── backtest.py     # 백테스팅 로직
│       └── strategy.py     # 퀀트 전략 구현
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