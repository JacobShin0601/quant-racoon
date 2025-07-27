# HMM-Neural 하이브리드 트레이더 시스템

## 🚀 **개요**

HMM (Hidden Markov Model) 시장 체제 분류와 신경망 기반 개별 종목 예측을 결합한 하이브리드 트레이딩 시스템입니다. 

### **핵심 기능**
- 🔍 **시장 체제 분류**: HMM으로 BULLISH/BEARISH/SIDEWAYS/VOLATILE 4가지 체제 분류
- 🧠 **신경망 예측**: 18개 스윙 전략 + 매크로 피처를 활용한 개별 종목 예측
- 📊 **투자 점수**: -1~1 스케일의 정량적 투자 추천 지수
- 🎯 **매매 신호**: 구체적인 매수/매도/보유 신호와 실행 권고
- 📈 **포트폴리오 관리**: 전체 포트폴리오 레벨의 종합 권고

---

## 🏗️ **시스템 구조**

```
🔄 데이터 수집 (매크로 + 개별 종목)
    ⬇️
🎭 HMM 시장 체제 분류 모델 학습 (VIX, 금리차, 달러강세 등)
    ⬇️
🧠 신경망 개별 종목 예측 모델 학습 (18개 스윙전략 + 기술적지표)
    ⬇️
⚙️ 하이퍼파라미터 최적화 (학습된 모델들로 threshold 튜닝)
    ⬇️
📊 투자 점수 생성 (-1~1, 리스크 조정 포함)
    ⬇️
🎯 매매 신호 생성 (진입/청산 타이밍 최적화)
    ⬇️
📈 포트폴리오 종합 권고
```

### **핵심 컴포넌트**
```
src/actions/
├── hmm_regime_classifier.py      # HMM 시장 체제 분류
├── neural_stock_predictor.py     # 신경망 종목 예측
├── optimize_threshold.py         # 하이퍼파라미터 최적화
├── investment_scorer.py          # 투자 점수 생성
└── trading_signal_generator.py   # 매매 신호 생성

src/agent/
└── trader.py                     # 통합 실행 관리자

config/
└── config_trader.json           # 시스템 설정

run_trader.sh                     # 실행 스크립트
```

---

## ⚙️ **설치 및 설정**

### **1. 필수 패키지 설치**
```bash
pip install pandas numpy scikit-learn tensorflow hmmlearn yfinance joblib
```

### **2. 디렉토리 구조 확인**
시스템이 자동으로 생성하지만, 수동으로 확인하려면:
```bash
mkdir -p log/trader results/trader backup/trader models/trader data/trader
```

### **3. 설정 파일 수정 (선택사항)**
`config/config_trader.json`에서 다음 항목들을 조정할 수 있습니다:

- **분석 대상 종목**: `data.symbols`
- **데이터 기간**: `data.lookback_days`
- **신경망 구조**: `neural_network.architecture`
- **신호 임계값**: `signal_generation.thresholds`
- **리스크 관리**: `risk_management`

---

## 🚀 **사용법**

### **기본 실행**
```bash
./run_trader.sh
```

### **주요 옵션**
```bash
# 강제 재학습 (모델 새로 생성)
./run_trader.sh -f

# 특정 종목만 분석
./run_trader.sh -s AAPL

# 사용자 정의 설정 파일 사용
./run_trader.sh -c my_config.json

# 디버그 모드
./run_trader.sh -d

# 도움말
./run_trader.sh --help
```

### **Python에서 직접 사용**
```python
from src.agent.trader import HybridTrader

# 트레이더 초기화
trader = HybridTrader('config/config_trader.json')

# 모델 초기화
trader.initialize_models()

# 전체 분석 실행
results = trader.run_analysis()

# 특정 종목 권고
aapl_recommendation = trader.get_recommendations('AAPL')
```

---

## 📊 **출력 결과 해석**

### **시장 체제 분류**
- **BULLISH**: 상승 추세 시장 - 적극적 매수 전략
- **BEARISH**: 하락 추세 시장 - 방어적 포지션
- **SIDEWAYS**: 횡보 시장 - 단기 매매 전략
- **VOLATILE**: 고변동성 시장 - 리스크 관리 강화

### **투자 점수 (-1 ~ 1)**
- **0.6 이상**: 강력 매수 (STRONG_BUY)
- **0.3 ~ 0.6**: 매수 (BUY)
- **-0.3 ~ 0.3**: 보유 (HOLD)
- **-0.6 ~ -0.3**: 매도 (SELL)
- **-0.6 이하**: 강력 매도 (STRONG_SELL)

### **실행 우선순위**
- **1-2**: 즉시 실행 권고
- **3-4**: 급하지 않음
- **5 이상**: 관망

---

## 📈 **실제 활용 예시**

### **시나리오 1: 일반적인 투자 권고**
```bash
./run_trader.sh
```

**출력 예시:**
```
=== 분석 결과 요약 ===
시장 체제: BULLISH (신뢰도: 78.5%)
포트폴리오 점수: 0.234
포트폴리오 액션: SELECTIVE_BUY

=== 신호 분포 ===
STRONG_BUY: 2개
BUY: 4개
HOLD: 5개
SELL: 1개

=== 상위 추천 종목 ===
NVDA: STRONG_BUY (점수: 0.742)
AAPL: BUY (점수: 0.456)
META: BUY (점수: 0.321)

=== 즉시 실행 권고 ===
NVDA: STRONG_BUY (우선순위: 1)
AAPL: BUY (우선순위: 2)
```

### **시나리오 2: 특정 종목 상세 분석**
```bash
./run_trader.sh -s NVDA
```

**출력 예시:**
```
=== NVDA 분석 결과 ===
액션: STRONG_BUY
점수: 0.742
신뢰도: 85.3%
포지션 크기: 12.5%
홀딩 기간: 25일

=== 시장 상황 ===
시장 체제: BULLISH
체제 신뢰도: 78.5%
```

---

## 🔧 **설정 최적화 가이드**

### **보수적 설정 (리스크 회피형)**
```json
{
  "signal_generation": {
    "thresholds": {
      "strong_buy": 0.8,
      "buy": 0.5,
      "sell": -0.5,
      "strong_sell": -0.8
    },
    "min_confidence": 0.6
  },
  "scoring": {
    "volatility_penalty": 0.5
  }
}
```

### **공격적 설정 (수익 추구형)**
```json
{
  "signal_generation": {
    "thresholds": {
      "strong_buy": 0.4,
      "buy": 0.2,
      "sell": -0.2,
      "strong_sell": -0.4
    },
    "min_confidence": 0.3
  },
  "scoring": {
    "volatility_penalty": 0.1
  }
}
```

---

## 📊 **성능 모니터링**

### **로그 파일 위치**
- **실행 로그**: `log/trader.log`
- **결과 파일**: `results/trader/trader_results_YYYYMMDD_HHMMSS.json`
- **모델 파일**: `models/trader/`

### **주요 메트릭 확인**
```bash
# 최근 로그 확인
tail -50 log/trader.log

# 최신 결과 파일 확인
ls -la results/trader/trader_results_*.json | tail -1
```

---

## ⚠️ **주의사항 및 제한사항**

### **데이터 의존성**
- 인터넷 연결 필요 (야후 파이낸스 API 사용)
- 미국 시장 시간 외 데이터 업데이트 지연 가능
- VIX, 금리 데이터 부재시 기본값 사용

### **모델 특성**
- 첫 실행시 모델 학습에 5-10분 소요
- 일봉 데이터 기반으로 단기 변동성에 둔감
- 과거 데이터 기반 예측으로 미래 보장 불가

### **리스크 관리**
- **실제 투자전 충분한 백테스팅 필수**
- 시스템은 참고용이며 최종 판단은 사용자 책임
- 포지션 크기 권고는 참고만 하고 개별 리스크 수준에 맞게 조정

---

## 🔄 **업데이트 및 유지보수**

### **모델 재학습**
```bash
# 강제 재학습 (주 1회 권장)
./run_trader.sh -f
```

### **설정 변경 후 재시작**
```bash
# 설정 변경시에는 재학습 권장
./run_trader.sh -f -c new_config.json
```

### **시스템 상태 확인**
```bash
# 디버그 모드로 문제 진단
./run_trader.sh -d
```

---

## 🛟 **문제 해결**

### **일반적인 오류**

**1. 패키지 없음 오류**
```bash
pip install pandas numpy scikit-learn tensorflow hmmlearn yfinance joblib
```

**2. 데이터 수집 실패**
- 인터넷 연결 확인
- 야후 파이낸스 서비스 상태 확인
- 잠시 후 재시도

**3. 모델 학습 실패**
- 데이터 부족: lookback_days 늘리기
- 메모리 부족: batch_size 줄이기
- 기존 모델 삭제 후 재학습

**4. 권한 오류**
```bash
chmod +x run_trader.sh
```

### **로그 확인**
```bash
# 에러 로그 확인
grep -i "error\|exception" log/trader.log | tail -10

# 전체 로그 확인
tail -100 log/trader.log
```

---

## 📞 **지원 및 커스터마이징**

### **시스템 확장**
- 새로운 기술적 지표 추가: `neural_stock_predictor.py` 수정
- 매크로 지표 추가: `hmm_regime_classifier.py` 수정
- 새로운 신호 로직: `trading_signal_generator.py` 수정

### **성능 튜닝**
- 신경망 구조 조정: `config_trader.json`의 `neural_network` 섹션
- HMM 파라미터 조정: `hmm_regime` 섹션
- 리스크 관리 강화: `scoring.risk_management` 섹션

---

## 🎯 **결론**

이 시스템은 **참고용 투자 도구**입니다. 실제 투자 결정시에는:

1. ✅ **시스템 신호를 참고 자료로 활용**
2. ✅ **개별 리스크 수준에 맞게 포지션 크기 조정**
3. ✅ **다른 분석 도구와 종합적으로 판단**
4. ✅ **충분한 백테스팅 후 소액으로 시작**
5. ⚠️ **과도한 의존 금지**

**성공적인 투자를 위해 신중하게 활용하시기 바랍니다!** 📈 