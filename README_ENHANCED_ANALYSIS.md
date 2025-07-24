# Enhanced LLM 분석 시스템

## 🎯 개요

Enhanced LLM 분석 시스템은 기존의 기본 분석 결과를 LLM(Large Language Model)에 전달하여 다면적이고 심층적인 시장 해석을 제공하는 고도화된 분석 시스템입니다.

## 🚀 주요 기능

### 1. 종합 분석 (Comprehensive Analysis)
- **지표 해석**: 각 기술적/매크로 지표의 의미와 현재 시장에서의 중요성 분석
- **다면적 분석**: 기술적, 매크로, 섹터 분석 결과의 일관성과 상충점 평가
- **시장 역학**: 현재 시장의 주요 동인과 변동성 원인 분석

### 2. 위험 평가 (Risk Assessment)
- **단기 위험**: 즉시 주목해야 할 위험 요인
- **중기 위험**: 1-3개월 내 발생 가능한 위험
- **장기 위험**: 3개월 이상의 장기적 위험 요인
- **리스크 완화 전략**: 포트폴리오 헤징 및 손절 전략

### 3. 전략적 추천 (Strategic Recommendations)
- **포트폴리오 배분**: 주식/채권/현금/대체자산 비중 추천
- **섹터 포커스**: 과중/과소 배치 및 회피 섹터 제시
- **트레이딩 전략**: 진입 타이밍, 보유 기간, 청산 전략

### 4. 시나리오 분석 (Scenario Analysis)
- **상승 시나리오**: 확률, 트리거, 대응 방안
- **하락 시나리오**: 확률, 트리거, 대응 방안
- **횡보 시나리오**: 확률, 트리거, 대응 방안

## 📊 사용법

### 1. Enhanced 분석 실행
```bash
# Enhanced 분석 (LLM 종합 분석 포함)
./run_market_analysis.sh --enhanced

# 캐시된 데이터 사용
./run_market_analysis.sh --enhanced --use-cached-data

# 커스텀 출력 디렉토리
./run_market_analysis.sh --enhanced -o results/macro/my_analysis
```

### 2. 테스트 실행
```bash
# Enhanced LLM 분석 테스트
python3 test_enhanced_llm_analysis.py
```

## 🔧 설정 옵션

### LLM 설정
```python
llm_config = {
    'provider': 'hybrid',  # 'bedrock', 'openai', 'hybrid'
    'model_name': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'fallback_to_rules': True,
    'max_tokens': 2000,    # 응답 길이
    'temperature': 0.1     # 일관성 (낮을수록 일관적)
}
```

### 분석 옵션
- `--enhanced`: Enhanced 분석 활성화
- `--use-cached-data`: 캐시된 데이터 사용
- `--use-random-forest`: Random Forest 모델 사용
- `--retrain-rf-model`: Random Forest 모델 재학습

## 📈 출력 결과

### 1. 시장 역학 분석
```json
{
  "market_dynamics": {
    "primary_drivers": ["주요 동인1", "주요 동인2"],
    "volatility_factors": ["변동성 요인1", "변동성 요인2"],
    "trend_strength": "strong/moderate/weak",
    "momentum_quality": "high/medium/low"
  }
}
```

### 2. 지표 해석
```json
{
  "indicator_interpretation": {
    "technical_indicators": {
      "rsi_interpretation": "RSI 해석",
      "macd_interpretation": "MACD 해석",
      "volume_analysis": "거래량 분석"
    },
    "macro_indicators": {
      "yield_curve_analysis": "수익률 곡선 분석",
      "inflation_outlook": "인플레이션 전망",
      "growth_prospects": "성장 전망"
    }
  }
}
```

### 3. 포트폴리오 추천
```json
{
  "strategic_recommendations": {
    "portfolio_allocation": {
      "equity_allocation": "60%",
      "bond_allocation": "30%",
      "cash_allocation": "10%",
      "alternative_allocation": "0%"
    },
    "sector_focus": {
      "overweight_sectors": ["기술", "헬스케어"],
      "underweight_sectors": ["에너지", "금융"],
      "avoid_sectors": ["부동산"]
    }
  }
}
```

### 4. 시나리오 분석
```json
{
  "scenario_analysis": {
    "bull_scenario": {
      "probability": 0.4,
      "triggers": ["금리 인하", "경기 회복"],
      "actions": ["주식 비중 확대", "성장주 집중"]
    },
    "bear_scenario": {
      "probability": 0.3,
      "triggers": ["인플레이션 상승", "경기 침체"],
      "actions": ["현금 비중 확대", "방어적 자산 집중"]
    }
  }
}
```

## 🎯 분석 프로세스

### 1. 기본 분석 수행
- 매크로 데이터 수집
- 기술적 분석
- 섹터 분석
- 하이퍼파라미터 최적화

### 2. 고급 분석 수행
- RLMF (Reinforcement Learning from Market Feedback) 분석
- 다층 신뢰도 분석
- Regime 전환 감지
- LLM 특권 정보 분석

### 3. LLM 종합 분석
- 기존 분석 결과를 LLM에 전달
- 다면적 해석 및 인사이트 생성
- 전략적 추천 및 시나리오 분석

### 4. 결과 통합 및 저장
- 모든 분석 결과 통합
- JSON 형태로 저장
- 상세한 요약 리포트 생성

## 🔍 모니터링 및 통계

### LLM API 통계
- 총 API 호출 수
- 성공률
- 평균 응답 시간
- 캐시 히트율

### 분석 성능 지표
- 분석 완료 시간
- 신뢰도 점수
- 예측 정확도
- 백테스트 성과

## 🛠️ 문제 해결

### LLM API 오류
```bash
# API 키 확인
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"

# 또는 OpenAI API 키
export OPENAI_API_KEY="your_openai_key"
```

### 캐시 문제
```bash
# 캐시 클리어
python3 -c "from src.agent.enhancements.llm_api_integration import LLMAPIIntegration; LLMAPIIntegration().clear_cache()"
```

### 메모리 부족
```bash
# 더 작은 배치 크기로 실행
./run_market_analysis.sh --enhanced --use-cached-data
```

## 📝 업데이트 내역

### v2.0 (현재)
- ✅ 기존 분석 결과를 LLM에 전달하는 종합 분석 기능
- ✅ 다면적 시장 해석 및 지표 분석
- ✅ 위험 평가 및 시나리오 분석
- ✅ 포트폴리오 전략 추천
- ✅ 향상된 프롬프트 엔지니어링

### v1.0 (이전)
- 기본 LLM API 통합
- 규칙 기반 분석
- 단순한 시장 체제 분류

## 🎉 결론

Enhanced LLM 분석 시스템은 기존의 정량적 분석에 AI의 해석 능력을 결합하여 더욱 풍부하고 실용적인 투자 인사이트를 제공합니다. 다면적 분석과 시나리오 기반 전략을 통해 다양한 시장 환경에 대응할 수 있는 강력한 도구입니다. 