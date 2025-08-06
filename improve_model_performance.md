# 신경망 모델 성능 개선 방안

## 현재 문제점
- Train/Val Loss가 0.83~0.94로 매우 높음
- 예측 오차가 11-14% 수준
- 모델이 학습 데이터 패턴을 제대로 포착하지 못함

## 개선 방안

### 1. 단일 타겟 모델로 단순화
```python
# config/config_trader.json 수정
"neural_config": {
    "multi_target": false,  // true -> false로 변경
    "features": {
        "lookback_days": 20
    }
}
```
- 현재 4개 타겟(target_22d, sigma_22d, target_66d, sigma_66d) 동시 학습
- target_22d만 학습하면 복잡도 감소

### 2. 모델 구조 개선
```python
# config/config_trader.json
"neural_config": {
    "architecture": {
        "hidden_layers": [128, 64, 32],  // 더 깊고 큰 모델
        "dropout_rate": 0.3  // 0.2 -> 0.3
    },
    "training": {
        "learning_rate": 0.0001,  // 0.0005 -> 0.0001
        "batch_size": 64,  // 32 -> 64
        "epochs": 300,  // 200 -> 300
        "early_stopping_patience": 30  // 20 -> 30
    }
}
```

### 3. 손실 함수 개선
```python
# src/actions/neural_stock_predictor.py 수정
# MSELoss 대신 HuberLoss 사용 (이상치에 강건)
criterion = nn.HuberLoss(delta=1.0)  # 현재: nn.MSELoss()
```

### 4. 피처 엔지니어링 강화
- 기술적 지표 추가 (RSI, MACD, Bollinger Bands)
- 시장 심리 지표 추가 (VIX, Put/Call ratio)
- 섹터 상대 강도 지표

### 5. 데이터 전처리 개선
```python
# 극단값 처리 강화
y_clipped = np.clip(y, -0.2, 0.2)  # 현재: -0.3, 0.3

# 이상치 제거
from scipy import stats
z_scores = stats.zscore(y)
y_filtered = y[np.abs(z_scores) < 3]
```

### 6. 앙상블 방법 개선
- 현재: 통합 모델 + 개별 모델 단순 가중 평균
- 개선: Stacking 또는 Blending 방식

### 7. 검증 방식 개선
```python
# Walk-forward validation
# 현재: 단순 train/test split
# 개선: 시계열 특성을 고려한 rolling window validation
```

## 즉시 적용 가능한 설정

1. **config/config_trader.json 수정**:
```json
{
    "neural_config": {
        "multi_target": false,
        "architecture": {
            "hidden_layers": [64, 32, 16],
            "dropout_rate": 0.25
        },
        "training": {
            "learning_rate": 0.0002,
            "batch_size": 64,
            "epochs": 300,
            "early_stopping_patience": 25,
            "weight_decay": 0.001
        }
    }
}
```

2. **재학습**:
```bash
./run_trader.sh --force-retrain
```

## 기대 효과
- Loss 감소: 0.83 → 0.3~0.5
- 예측 오차 감소: 11-14% → 5-8%
- 과적합 감소
- 안정적인 예측