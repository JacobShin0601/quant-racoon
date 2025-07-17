# Config 폴더

이 폴더는 퀀트 분석 파이프라인의 모든 설정 파일들을 포함합니다.

## 📁 설정 파일 목록

### 🎯 전략별 설정 파일

#### `config_swing.json`
- **용도**: 스윙 전략 설정
- **데이터**: 15분봉, 60일 기간
- **전략들**: DualMomentum, VolatilityAdjustedBreakout, SwingEMA, SwingRSI, DonchianSwing
- **포트폴리오 모드**: 비활성화

#### `config_long.json`
- **용도**: 장기 전략 설정
- **데이터**: 1일봉, 1095일(3년) 기간
- **전략들**: RiskParityLeverage, FixedWeightRebalance, ETFMomentumRotation, TrendFollowingMA200, ReturnStacking
- **포트폴리오 모드**: 활성화

#### `config_scalping.json`
- **용도**: 초단기 전략 설정
- **데이터**: 1분봉, 7일 기간
- **전략들**: VWAPMACDScalping, KeltnerRSIScalping, AbsorptionScalping, RSIBollingerScalping
- **포트폴리오 모드**: 비활성화

### 🔧 기본 설정 파일

#### `config_default.json`
- **용도**: 기본 설정 (time-horizon 미지정 시 사용)
- **특징**: 모든 전략과 기능을 포함한 통합 설정
- **사용**: `python -m src.agent.orchestrator` (time-horizon 옵션 없이)

### 🔬 연구용 설정 파일

#### `config_research.json`
- **용도**: 하이퍼파라미터 최적화 연구
- **특징**: 
  - 전략별 파라미터 범위와 최적화 설정
  - `source_config`를 통해 다른 config 파일의 심볼과 설정을 자동으로 가져옴
  - 포트폴리오 모드, 데이터 간격 등이 source config에 따라 자동 설정
- **사용**: 연구 및 실험 목적
- **source_config 옵션**:
  - `config_long.json`: 장기 전략 심볼 (SPY, TLT, GLD, QQQ, DBMF, SHY)
  - `config_swing.json`: 스윙 전략 심볼 (AAPL, QQQ, SPY)
  - `config_scalping.json`: 초단기 전략 심볼 (BTCUSDT, ETHUSDT, KRW-ETH)

## 🚀 사용 방법

### 개별 전략 실행
```bash
# 스윙 전략 (config_swing.json 자동 로드)
python -m src.agent.orchestrator --time-horizon swing

# 장기 전략 (config_long.json 자동 로드)
python -m src.agent.orchestrator --time-horizon long

# 초단기 전략 (config_scalping.json 자동 로드)
python -m src.agent.orchestrator --time-horizon scalping
```

### 직접 설정 파일 지정
```bash
# 특정 설정 파일 사용
python -m src.agent.orchestrator --config config/config_swing.json

# 연구용 설정 사용
python -m src.agent.orchestrator --config config/config_research.json
```

### 연구용 설정 사용
```bash
# 기본 연구 실행 (config_long.json 심볼 사용)
python -m src.agent.researcher

# 특정 source config 지정
python -m src.agent.researcher --config config/config_research.json

# 스윙 전략 심볼로 연구 (config_research.json에서 source_config를 config_swing.json으로 수정)
python -m src.agent.researcher

# 초단기 전략 심볼로 연구 (config_research.json에서 source_config를 config_scalping.json으로 수정)
python -m src.agent.researcher
```

## 📋 설정 파일 구조

모든 설정 파일은 다음 구조를 따릅니다:

```json
{
  "time_horizon": "전략 타입",
  "strategies": ["전략 목록"],
  "data": {
    "symbols": ["종목 목록"],
    "interval": "데이터 간격",
    "lookback_days": 기간
  },
  "evaluator": {
    "portfolio_mode": true/false
  },
  "cleaner": {
    "run_cleaner": true/false,
    "action": "clean/create/clean-and-recreate"
  },
  "automation": {
    "auto_clean": true/false,
    "auto_backup": true/false
  },
  "logging": {
    "level": "INFO/WARNING/ERROR",
    "file_rotation": true/false
  },
  "output": {
    "results_folder": "결과 폴더 경로",
    "logs_folder": "로그 폴더 경로",
    "backup_folder": "백업 폴더 경로"
  },
  "flow": {
    "stages": ["cleaner", "scrapper", "analyzer", "researcher", "evaluator", "portfolio_manager"],
    "stop_on_error": true,
    "enable_research": true
  }
}
```

## 🔄 설정 파일 수정

설정 파일을 수정할 때 주의사항:

1. **JSON 형식 유지**: 모든 설정은 유효한 JSON 형식이어야 함
2. **경로 설정**: 상대 경로 사용 권장
3. **전략명 일치**: strategies 배열의 전략명은 실제 구현된 전략과 일치해야 함
4. **데이터 기간**: lookback_days는 데이터 수집 가능 범위 내에서 설정

## 🔄 Flow 설정

- **stages**: 실행할 단계 목록 (순서대로 실행)
  - `cleaner`: 폴더 정리 및 생성
  - `scrapper`: 데이터 수집
  - `analyzer`: 정량 분석 (상관관계, 회귀분석, 머신러닝)
  - `researcher`: 하이퍼파라미터 최적화 연구
  - `evaluator`: 전략 평가
  - `portfolio_manager`: 포트폴리오 최적화 및 관리
- **stop_on_error**: 오류 발생 시 파이프라인 중단 여부
- **enable_research**: 연구 단계 활성화 여부 (false로 설정하면 researcher 단계 건너뜀)

## 📝 설정 파일 추가

새로운 전략이나 설정을 추가할 때:

1. 새로운 config 파일 생성
2. orchestrator.py의 config_mapping에 추가
3. 필요한 경우 새로운 time-horizon 옵션 추가
4. README.md 업데이트 