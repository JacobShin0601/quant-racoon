#!/bin/bash

# 앙상블 전략 실행 스크립트 (시장 환경별 전략 선택)
# market_sensor가 감지한 시장 환경에 따라 적절한 전략을 선택하고 실행

echo "🎯 앙상블 전략 실행 시작..."
echo "📊 데이터: 일봉, 기간: 365일"
echo "🎯 전략: 시장 환경별 전략 선택 + 개별 최적화 + 포트폴리오 평가"
echo "📁 데이터 디렉토리: data/ensemble"
echo "🔧 평가 모드: 시장 환경 감지 + 환경별 파이프라인"
echo "🤖 ML 모델: 저장된 Random Forest 모델 사용"
echo ""

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# UUID 생성 (실행 시점에 고유한 식별자)
UUID=$(date +"%Y%m%d_%H%M%S")_$(uuidgen | cut -d'-' -f1)
echo "🆔 실행 UUID: $UUID"

# 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 시작 시간: $START_TIME"

# 백업 폴더 생성
BACKUP_DIR="backup/ensemble/backup_${UUID}"
echo "💾 백업 폴더 생성: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# 1단계: 시장 환경 분류 모델 확인/학습
echo "🤖 1단계: 시장 환경 분류 모델 확인..."
echo "📋 모델 위치: models/market_regime/market_regime_rf_model.pkl"

if [ -f "models/market_regime/market_regime_rf_model.pkl" ]; then
    echo "✅ 저장된 모델이 존재합니다."
    MODEL_STATUS="existing"
else
    echo "⚠️ 저장된 모델이 없습니다. 새로 학습을 시작합니다."
    echo "🔄 모델 학습 중..."
    python train_market_model.py --data-dir data/macro
    
    if [ $? -eq 0 ]; then
        echo "✅ 모델 학습 완료!"
        MODEL_STATUS="trained"
    else
        echo "❌ 모델 학습 실패!"
        exit 1
    fi
fi

# 2단계: 앙상블 전략 실행
echo ""
echo "🔄 2단계: 앙상블 전략 파이프라인 실행 중..."
echo "📋 단계: 시점별 시장 환경 감지 → 기간별 최적화 → 성과 분석"
echo "📁 데이터 디렉토리: data/ensemble"
echo "📁 결과 디렉토리: results/ensemble"
echo "🤖 사용 모델: ${MODEL_STATUS} (Random Forest)"
echo "🔧 최적화 횟수: 70회 (Bayesian Optimization)"
python -m src.actions.ensemble --config config/config_ensemble.json --uuid "$UUID" --time-horizon ensemble

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 앙상블 전략 실행 완료!"
    
    # 백업 실행
    echo "💾 백업 시작..."
    echo "📁 백업 대상: data/ensemble, log/ensemble, results/ensemble, models/market_regime"
    echo "📁 백업 위치: $BACKUP_DIR"
    
    # data/ensemble 백업
    if [ -d "data/ensemble" ]; then
        echo "📊 data/ensemble 백업 중..."
        cp -r data/ensemble "$BACKUP_DIR/"
        echo "✅ data/ensemble 백업 완료"
    else
        echo "⚠️ data/ensemble 폴더가 존재하지 않습니다"
    fi
    
    # log/ensemble 백업
    if [ -d "log/ensemble" ]; then
        echo "📋 log/ensemble 백업 중..."
        cp -r log/ensemble "$BACKUP_DIR/"
        echo "✅ log/ensemble 백업 완료"
    else
        echo "⚠️ log/ensemble 폴더가 존재하지 않습니다"
    fi
    
    # results/ensemble 백업
    if [ -d "results/ensemble" ]; then
        echo "📊 results/ensemble 백업 중..."
        cp -r results/ensemble "$BACKUP_DIR/"
        echo "✅ results/ensemble 백업 완료"
    else
        echo "⚠️ results/ensemble 폴더가 존재하지 않습니다"
    fi
    
    # models/market_regime 백업
    if [ -d "models/market_regime" ]; then
        echo "🤖 models/market_regime 백업 중..."
        cp -r models/market_regime "$BACKUP_DIR/"
        echo "✅ models/market_regime 백업 완료"
    else
        echo "⚠️ models/market_regime 폴더가 존재하지 않습니다"
    fi
    
    # 시장 환경별 결과 백업 (새로운 구조에 맞춰 수정)
    for regime in trending_up trending_down volatile sideways; do
        # 기존 regime별 폴더가 있다면 백업 (하위 호환성)
        if [ -d "data/ensemble_${regime}" ]; then
            echo "📊 data/ensemble_${regime} 백업 중..."
            cp -r "data/ensemble_${regime}" "$BACKUP_DIR/"
            echo "✅ data/ensemble_${regime} 백업 완료"
        fi
        
        if [ -d "results/ensemble_${regime}" ]; then
            echo "📊 results/ensemble_${regime} 백업 중..."
            cp -r "results/ensemble_${regime}" "$BACKUP_DIR/"
            echo "✅ results/ensemble_${regime} 백업 완료"
        fi
    done
    
    # 백업 정보 파일 생성
    BACKUP_INFO_FILE="$BACKUP_DIR/backup_info.json"
    cat > "$BACKUP_INFO_FILE" << EOF
{
  "uuid": "$UUID",
  "backup_time": "$(date +"%Y-%m-%d %H:%M:%S")",
  "strategy": "ensemble",
  "model_status": "$MODEL_STATUS",
  "backup_contents": {
    "data_ensemble": "$(if [ -d "data/ensemble" ]; then echo "true"; else echo "false"; fi)",
    "log_ensemble": "$(if [ -d "log/ensemble" ]; then echo "true"; else echo "false"; fi)",
    "results_ensemble": "$(if [ -d "results/ensemble" ]; then echo "true"; else echo "false"; fi)",
    "models_market_regime": "$(if [ -d "models/market_regime" ]; then echo "true"; else echo "false"; fi)",
    "data_trending_up": "$(if [ -d "data/ensemble_trending_up" ]; then echo "true"; else echo "false"; fi)",
    "data_trending_down": "$(if [ -d "data/ensemble_trending_down" ]; then echo "true"; else echo "false"; fi)",
    "data_volatile": "$(if [ -d "data/ensemble_volatile" ]; then echo "true"; else echo "false"; fi)",
    "data_sideways": "$(if [ -d "data/ensemble_sideways" ]; then echo "true"; else echo "false"; fi)"
  },
  "execution_info": {
    "start_time": "$START_TIME",
    "end_time": "$(date +"%Y-%m-%d %H:%M:%S")",
    "script": "run_ensemble.sh"
  }
}
EOF
    echo "✅ 백업 정보 파일 생성: $BACKUP_INFO_FILE"
    
    echo ""
    echo "📁 결과 폴더: results/ensemble/"
    echo "📋 로그 폴더: log/ensemble/"
    echo "💾 백업 폴더: $BACKUP_DIR"
    echo "📊 데이터 폴더: data/ensemble/"
    echo "🤖 모델 폴더: models/market_regime/"
    echo "🔧 시장 환경별 파이프라인 완료"
else
    echo ""
    echo "❌ 앙상블 전략 실행 실패!"
    exit 1
fi

# 실행 시간 계산
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 종료 시간: $END_TIME"

echo ""
echo "🎉 앙상블 전략 분석이 완료되었습니다!"
echo ""
echo "📊 실행 결과 요약:"
echo "  1단계: 시장 환경 분류 모델 확인/학습 (${MODEL_STATUS})"
echo "  2단계: 시점별 시장 환경 감지 (Market Sensor + RF Classifier)"
echo "  3단계: 기간별 전략 최적화 (70회 Bayesian Optimization)"
echo "  4단계: Regime별 특화 설정 적용"
echo "  5단계: 포트폴리오 최적화 (Risk Parity)"
echo "  6단계: 성과 분석 및 백업 생성 (UUID: $UUID)"
echo ""
echo "📁 결과 파일 위치:"
echo "  - 앙상블 결과: results/ensemble/ensemble_results_*.json"
echo "  - 앙상블 요약: results/ensemble/ensemble_summary_*.txt"
echo "  - 시장 환경별 결과: results/ensemble_*/"
echo "  - 모델 파일: models/market_regime/market_regime_rf_model.pkl"
echo "  - 백업 폴더: $BACKUP_DIR"
echo ""
echo "🎯 시장 환경별 전략:"
echo "  - TRENDING_UP: 상승 추세 전략 (dual_momentum, volatility_breakout, swing_ema)"
echo "  - TRENDING_DOWN: 하락 추세 전략 (mean_reversion, swing_rsi, stochastic)"
echo "  - VOLATILE: 변동성 높은 시장 전략 (volatility_filtered_breakout, multi_timeframe_whipsaw)"
echo "  - SIDEWAYS: 횡보장 전략 (mean_reversion, swing_rsi, swing_bollinger_band)"

echo ""
echo "📊 7단계: 실행 결과 자동 조회"
echo "--------------------------------------------------"
echo "🔍 앙상블 실행 결과를 자동으로 조회합니다..."
echo ""

# 앙상블 결과 자동 조회
python view_ensemble_results.py --uuid "$UUID" --detailed

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 결과 조회 완료!"
    echo ""
    echo "📋 추가 결과 조회 명령어:"
    echo "  - 최신 결과 조회: python view_ensemble_results.py"
    echo "  - 모든 결과 목록: python view_ensemble_results.py --list"
    echo "  - 상세 정보: python view_ensemble_results.py --uuid $UUID --detailed"
    echo "  - 요약 리포트: cat results/ensemble/ensemble_summary_$UUID.txt"
else
    echo ""
    echo "⚠️ 자동 결과 조회 실패. 수동으로 확인하세요:"
    echo "  python view_ensemble_results.py --uuid $UUID"
fi 