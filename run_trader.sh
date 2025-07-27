#!/bin/bash

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}
warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

show_help() {
    echo -e "\n${GREEN}🧠 HMM-Neural 하이브리드 트레이더 시스템${NC}"
    echo -e "\n사용법: $0 [옵션]"
    echo -e "\n옵션:"
    echo -e "  --optimize         하이퍼파라미터 최적화 포함 실행"
    echo -e "  --optimize-threshold  임계점 최적화 포함 실행"
    echo -e "  --use-cached-data  캐시된 데이터 사용 (새로 다운로드 안함)"
    echo -e "  --force-retrain    모델 강제 재학습"
    echo -e "  --debug           디버그 모드 (상세 로그)"
    echo -e "  -h, --help        도움말"
    echo -e "\n${BLUE}실행 순서:${NC}"
    echo -e "  1️⃣ 데이터 수집 (매크로 + 개별 종목)"
    echo -e "  2️⃣ HMM 시장 체제 분류 모델 학습"
    echo -e "  3️⃣ 신경망 개별 종목 예측 모델 학습"
    echo -e "  4️⃣ 하이퍼파라미터 최적화 (선택사항)"
    echo -e "  5️⃣ 임계점 최적화 (선택사항)"
    echo -e "  6️⃣ 트레이딩 분석 및 신호 생성"
    echo -e "\n${GREEN}포트폴리오 고급 기능:${NC}"
    echo -e "  🎯 신경망 기반 포트폴리오 최적화"
    echo -e "  📊 샤프비율, 소르티노, 칼마비율, MDD, VaR, CVaR"
    echo -e "  🆚 Buy & Hold 벤치마크 비교"
    echo -e "  📋 상세 백테스팅 리포트 (매매내역, 보유현황)"
    echo ""
}

# 기본 설정
OPTIMIZE=false
OPTIMIZE_THRESHOLD=false
USE_CACHED_DATA=false
FORCE_RETRAIN=false
DEBUG_MODE=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --optimize)
            OPTIMIZE=true
            shift
            ;;
        --optimize-threshold)
            OPTIMIZE_THRESHOLD=true
            shift
            ;;
        --use-cached-data)
            USE_CACHED_DATA=true
            shift
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 디버그 모드 설정
if [[ "$DEBUG_MODE" == true ]]; then
    set -x
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
fi

# 프로젝트 루트 확인
if [[ ! -f "config/config_trader.json" ]]; then
    error "config/config_trader.json 파일을 찾을 수 없습니다."
    error "올바른 디렉토리에서 실행하세요."
    exit 1
fi

log "🚀 HMM-Neural 하이브리드 트레이더 시스템 시작"
log "옵션: optimize=$OPTIMIZE, optimize-threshold=$OPTIMIZE_THRESHOLD, cached=$USE_CACHED_DATA, retrain=$FORCE_RETRAIN"

# ============================================================================
# 1단계: 데이터 수집 (매크로 + 개별 종목)
# ============================================================================
step "1️⃣ 데이터 수집"

if [[ "$USE_CACHED_DATA" == true ]]; then
    log "📂 캐시된 데이터 사용 모드"
    
    # 캐시된 데이터 확인
    if [[ ! -d "data/macro" ]] || [[ ! -d "data/trader" ]]; then
        error "캐시된 데이터 디렉토리가 없습니다."
        error "먼저 데이터 수집을 실행하세요: ./run_trader.sh"
        exit 1
    fi
    
    macro_count=$(find data/macro -name "*.csv" 2>/dev/null | wc -l)
    trader_count=$(find data/trader -name "*.csv" 2>/dev/null | wc -l)
    
    if [[ $macro_count -eq 0 ]] || [[ $trader_count -eq 0 ]]; then
        error "캐시된 데이터가 충분하지 않습니다."
        error "매크로: $macro_count개, 개별종목: $trader_count개"
        error "먼저 데이터 수집을 실행하세요: ./run_trader.sh"
        exit 1
    fi
    
    log "✅ 캐시된 데이터 확인 완료 (매크로: $macro_count개, 개별종목: $trader_count개)"
else
    log "📊 새로운 데이터 수집 시작"
    
    # 매크로 데이터 수집
    log "📈 매크로 데이터 수집 중..."
    python3 src/actions/global_macro.py --mode collect
    if [[ $? -ne 0 ]]; then
        error "매크로 데이터 수집 실패"
        exit 1
    fi
    
    # 개별 종목 데이터 수집
    log "💼 개별 종목 데이터 수집 중..."
    python3 src/agent/scrapper.py --config config/config_trader.json
    if [[ $? -ne 0 ]]; then
        error "개별 종목 데이터 수집 실패"
        exit 1
    fi
    
    log "✅ 데이터 수집 완료"
fi

# ============================================================================
# 2단계: HMM 시장 체제 분류 모델 학습
# ============================================================================
step "2️⃣ HMM 시장 체제 분류 모델 학습"

log "🧠 HMM 시장 체제 모델 학습 중..."
if [[ "$FORCE_RETRAIN" == true ]]; then
    log "⚡ 강제 재학습 모드"
    python3 src/actions/hmm_regime_classifier.py --train --force --data-dir data/macro
else
    log "🔍 기존 모델 확인 후 필요시 학습"
    python3 src/actions/hmm_regime_classifier.py --train --data-dir data/macro
fi

if [[ $? -ne 0 ]]; then
    error "HMM 모델 학습 실패"
    exit 1
fi

log "✅ HMM 시장 체제 분류 모델 학습 완료"

# ============================================================================
# 3단계: 신경망 개별 종목 예측 모델 학습
# ============================================================================
step "3️⃣ 신경망 개별 종목 예측 모델 학습"

log "🧠 신경망 예측 모델 학습 중..."
if [[ "$FORCE_RETRAIN" == true ]]; then
    log "⚡ 강제 재학습 모드"
    python3 src/actions/neural_stock_predictor.py --train --force --data-dir data/trader
else
    log "🔍 기존 모델 확인 후 필요시 학습"
    python3 src/actions/neural_stock_predictor.py --train --data-dir data/trader
fi

if [[ $? -ne 0 ]]; then
    error "신경망 모델 학습 실패"
    exit 1
fi

log "✅ 신경망 개별 종목 예측 모델 학습 완료"

# ============================================================================
# 4단계: 하이퍼파라미터 최적화 (선택사항)
# ============================================================================
if [[ "$OPTIMIZE" == true ]]; then
    step "4️⃣ 하이퍼파라미터 최적화"
    
    log "🎯 신호 임계값 최적화 시작..."
    python3 src/actions/optimize_threshold.py --config config/config_trader.json --symbols AAPL,META,QQQ,SPY
    
    if [[ $? -ne 0 ]]; then
        error "하이퍼파라미터 최적화 실패"
        exit 1
    fi
    
    log "✅ 하이퍼파라미터 최적화 완료"
else
    log "⏩ 하이퍼파라미터 최적화 건너뛰기 (--optimize 옵션 사용시 실행)"
fi

# ============================================================================
# 5단계: 임계점 최적화 (선택사항)
# ============================================================================
if [[ "$OPTIMIZE_THRESHOLD" == true ]]; then
    step "5️⃣ 임계점 최적화"
    
    log "🎯 포트폴리오 임계점 최적화 시작..."
    log "📊 Optuna 기반 최적화 (config/optimization 설정 사용)"
    
    # 포트폴리오 종목 목록 가져오기
    symbols=$(python3 -c "
import json
with open('config/config_trader.json', 'r') as f:
    config = json.load(f)
print(' '.join(config['portfolio']['symbols']))
")
    
    # --optimize-threshold 옵션이 있으면 강제 최적화, 없으면 저장된 결과 사용
    if [[ "$OPTIMIZE_THRESHOLD" == true ]]; then
        log "🔄 새로운 최적화 실행 (--optimize-threshold 옵션)"
        python3 src/actions/optimize_threshold.py --config config/config_trader.json --symbols $symbols --method optuna --force-optimize
    else
        log "📂 저장된 최적화 결과 사용"
        python3 src/actions/optimize_threshold.py --config config/config_trader.json --symbols $symbols --method optuna
    fi
    
    if [[ $? -ne 0 ]]; then
        error "임계점 최적화 실패"
        exit 1
    fi
    
    log "✅ 임계점 최적화 완료"
else
    log "⏩ 임계점 최적화 건너뛰기 (--optimize-threshold 옵션 사용시 실행)"
fi

# ============================================================================
# 6단계: 트레이딩 분석 및 신호 생성
# ============================================================================
step "6️⃣ 트레이딩 분석 및 신호 생성 (고급 포트폴리오 분석 포함)"

log "🎯 학습된 모델을 사용한 트레이딩 분석 실행"
log "📊 포함 기능:"
log "   • 신경망 기반 개별 종목 예측"
log "   • 포트폴리오 최적화 (샤프 최대화, Risk Parity 등)"
log "   • 고급 성과 지표 (샤프, 소르티노, 칼마, MDD, VaR, CVaR)"
log "   • Buy & Hold 벤치마크 비교"
log "   • 상세 백테스팅 리포트 (매매내역, 보유현황)"

python3 src/agent/trader.py --config config/config_trader.json --run-analysis

if [[ $? -eq 0 ]]; then
    echo ""
    log "🎉 전체 프로세스 완료!"
    log "📊 결과 파일: results/trader/"
    log "📝 로그 파일: log/trader.log"
    echo ""
    log "🔍 생성된 리포트:"
    log "   • 개별 종목 예측 결과"
    log "   • 포트폴리오 최적 비중"
    log "   • 백테스팅 상세 분석"
    log "   • Buy & Hold 대비 성과 비교"
    log "   • 매매 내역 및 최종 보유 현황"
else
    error "트레이더 실행 실패"
    exit 1
fi 