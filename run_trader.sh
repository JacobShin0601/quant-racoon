#!/bin/bash

set -e

# Conda 환경 활성화
CONDA_ENV="bedrock_manus"
PYTHON_PATH="/home/yunchae/anaconda3/envs/${CONDA_ENV}/bin/python3"

# Python 경로 확인
if [[ ! -f "$PYTHON_PATH" ]]; then
    echo "Error: Python not found at $PYTHON_PATH"
    echo "Please ensure conda environment '${CONDA_ENV}' is properly installed"
    exit 1
fi

# 로깅 설정
LOG_LEVEL=${LOG_LEVEL:-INFO}  # 환경변수로 로그 레벨 설정 가능
TIMESTAMP_FORMAT="%Y-%m-%d %H:%M:%S"

# Python 로거 헬퍼 스크립트 경로
LOGGER_HELPER="$PYTHON_PATH -m src.utils.centralized_logger"

# 통합 로깅 함수
_log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date +"$TIMESTAMP_FORMAT")
    
    # 레벨별 색상 및 이모지
    case $level in
        INFO)
            echo -e "\033[0;32m[INFO    ]\033[0m $timestamp - $message"
            ;;
        WARN|WARNING)
            echo -e "\033[1;33m[WARNING ]\033[0m $timestamp - $message"
            ;;
        ERROR)
            echo -e "\033[0;31m[ERROR   ]\033[0m $timestamp - $message"
            ;;
        STEP)
            echo -e "\033[0;34m[STEP    ]\033[0m $timestamp - $message"
            ;;
        SUCCESS)
            echo -e "\033[0;92m[SUCCESS ]\033[0m $timestamp - $message"
            ;;
        DEBUG)
            if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
                echo -e "\033[0;36m[DEBUG   ]\033[0m $timestamp - $message"
            fi
            ;;
    esac
}

# 편의 함수들
log() { _log INFO "$@"; }
warn() { _log WARN "$@"; }
error() { _log ERROR "$@"; }
step() { _log STEP "$@"; }
success() { _log SUCCESS "$@"; }
debug() { _log DEBUG "$@"; }

show_help() {
    cat << EOF

🧠 HMM-Neural 하이브리드 트레이더 시스템

사용법: $0 [옵션]

옵션:
  --optimize            하이퍼파라미터 최적화 포함 실행
  --optimize-threshold  임계점 최적화 포함 실행
  --use-cached-data     캐시된 데이터 사용 (새로 다운로드 안함)
  --force-retrain       모델 강제 재학습
  --experiment          다양한 신경망 구조 실험 모드
  --log-level LEVEL     로그 레벨 설정 (DEBUG, INFO, WARN, ERROR)
  --quiet               최소한의 출력만 표시
  -h, --help            도움말

실행 단계:
  1. 데이터 수집 (매크로 + 개별 종목)
  2. HMM 시장 체제 분류 모델 학습
  3. 신경망 개별 종목 예측 모델 학습
  4. 하이퍼파라미터 최적화 (선택사항)
  5. 임계점 최적화 (선택사항)
  6. 트레이딩 분석 및 신호 생성

포트폴리오 기능:
  • 신경망 기반 포트폴리오 최적화
  • 샤프비율, 소르티노, 칼마비율, MDD, VaR, CVaR
  • Buy & Hold 벤치마크 비교
  • 상세 백테스팅 리포트

EOF
}

# 기본 설정
OPTIMIZE=false
OPTIMIZE_THRESHOLD=false
USE_CACHED_DATA=false
FORCE_RETRAIN=false
EXPERIMENT=false
QUIET_MODE=false

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
        --experiment)
            EXPERIMENT=true
            shift
            ;;
        --log-level)
            LOG_LEVEL=$2
            shift 2
            ;;
        --quiet)
            QUIET_MODE=true
            LOG_LEVEL=WARN
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

# 로그 레벨 환경변수 설정
export LOG_LEVEL=$LOG_LEVEL
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Quiet 모드에서는 Python 출력도 제한
if [[ "$QUIET_MODE" == true ]]; then
    export PYTHONWARNINGS="ignore"
fi

# 프로젝트 루트 확인
if [[ ! -f "config/config_trader.json" ]]; then
    error "config/config_trader.json 파일을 찾을 수 없습니다."
    error "올바른 디렉토리에서 실행하세요."
    exit 1
fi

log "HMM-Neural 하이브리드 트레이더 시스템 시작"
debug "옵션: optimize=$OPTIMIZE, optimize-threshold=$OPTIMIZE_THRESHOLD, cached=$USE_CACHED_DATA, retrain=$FORCE_RETRAIN"

# ============================================================================
# 1단계: 데이터 수집
# ============================================================================
step "[1/6] 데이터 수집"

if [[ "$USE_CACHED_DATA" == true ]]; then
    log "캐시된 데이터 사용 모드"
    
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
    
    success "캐시된 데이터 확인 완료 (매크로: $macro_count개, 개별종목: $trader_count개)"
else
    log "새로운 데이터 수집 시작"
    
    # 통합 데이터 관리자를 사용하여 데이터 수집
    log "통합 데이터 관리자를 사용한 데이터 수집"
    
    # 매크로 데이터 수집
    log "매크로 데이터 수집 중..."
    if [[ "$QUIET_MODE" == true ]]; then
        $PYTHON_PATH -m src.agent.data_manager --data-type macro --time-horizon trader > /dev/null 2>&1
    else
        $PYTHON_PATH -m src.agent.data_manager --data-type macro --time-horizon trader
    fi
    if [[ $? -ne 0 ]]; then
        error "매크로 데이터 수집 실패"
        exit 1
    fi
    
    # 개별 종목 데이터 수집
    log "💼 개별 종목 데이터 수집 중..."
    # config_trader.json에서 심볼 목록과 lookback_days 추출
    read_config=$($PYTHON_PATH -c "
import json
with open('config/config_trader.json', 'r') as f:
    config = json.load(f)
symbols = config.get('data', {}).get('symbols', [])
lookback_days = config.get('data', {}).get('lookback_days', 700)
print(' '.join(symbols))
print(lookback_days)
")
    
    symbols=$(echo "$read_config" | head -1)
    lookback_days=$(echo "$read_config" | tail -1)
    
    if [[ -n "$symbols" ]]; then
        log "종목: $symbols (lookback_days: $lookback_days)"
        if [[ "$QUIET_MODE" == true ]]; then
            $PYTHON_PATH -m src.agent.data_manager --data-type stock --time-horizon trader --symbols $symbols --lookback-days $lookback_days > /dev/null 2>&1
        else
            $PYTHON_PATH -m src.agent.data_manager --data-type stock --time-horizon trader --symbols $symbols --lookback-days $lookback_days
        fi
        if [[ $? -ne 0 ]]; then
            error "개별 종목 데이터 수집 실패"
            exit 1
        fi
    else
        error "config_trader.json에 심볼이 정의되지 않았습니다"
        exit 1
    fi
    
    success "데이터 수집 완료"
fi

# ============================================================================
# 2단계: HMM 시장 체제 분류 모델 학습
# ============================================================================
step "[2/6] HMM 시장 체제 분류 모델 학습"

log "HMM 시장 체제 모델 학습 중..."
if [[ "$FORCE_RETRAIN" == true ]]; then
    log "강제 재학습 모드 활성화"
    cmd="$PYTHON_PATH src/actions/hmm_regime_classifier.py --train --force --data-dir data/macro"
else
    log "기존 모델 확인 후 필요시 학습"
    cmd="$PYTHON_PATH src/actions/hmm_regime_classifier.py --train --data-dir data/macro"
fi

if [[ "$QUIET_MODE" == true ]]; then
    $cmd > /dev/null 2>&1
else
    $cmd
fi

if [[ $? -ne 0 ]]; then
    error "HMM 모델 학습 실패"
    exit 1
fi

success "HMM 시장 체제 분류 모델 학습 완료"

# ============================================================================
# 3단계: 신경망 개별 종목 예측 모델 학습
# ============================================================================
step "[3/6] 신경망 개별 종목 예측 모델 학습"

log "신경망 예측 모델 학습 중..."
if [[ "$FORCE_RETRAIN" == true ]]; then
    log "강제 재학습 모드 활성화"
    cmd="$PYTHON_PATH src/actions/neural_stock_predictor.py --train --force --data-dir data/trader"
else
    log "기존 모델 확인 후 필요시 학습"
    cmd="$PYTHON_PATH src/actions/neural_stock_predictor.py --train --data-dir data/trader"
fi

if [[ "$QUIET_MODE" == true ]]; then
    $cmd > /dev/null 2>&1
else
    $cmd
fi

if [[ $? -ne 0 ]]; then
    error "신경망 모델 학습 실패"
    exit 1
fi

success "신경망 개별 종목 예측 모델 학습 완료"

# ============================================================================
# 3-1단계: 신경망 구조 실험 (선택사항)
# ============================================================================
if [[ "$EXPERIMENT" == true ]]; then
    step "[3-1/6] 신경망 구조 실험 모드"
    
    log "다양한 신경망 구조로 실험 시작"
    log "실험 설정 파일: config/neural_experiments.json"
    
    if [[ "$FORCE_RETRAIN" == true ]]; then
        $PYTHON_PATH src/actions/neural_stock_predictor.py --experiment --force --data-dir data/trader
    else
        $PYTHON_PATH src/actions/neural_stock_predictor.py --experiment --data-dir data/trader
    fi
    
    if [[ $? -ne 0 ]]; then
        error "신경망 구조 실험 실패"
        exit 1
    fi
    
    success "신경망 구조 실험 완료"
    log "최적 모델 설정 저장됨: models/trader/best_neural_configs.json"
else
    debug "신경망 구조 실험 건너뛰기 (--experiment 옵션 사용시 실행)"
fi

# ============================================================================
# 4단계: 하이퍼파라미터 최적화 (선택사항)
# ============================================================================
if [[ "$OPTIMIZE" == true ]]; then
    step "[4/6] 하이퍼파라미터 최적화"
    
    log "신호 임계값 최적화 시작"
    $PYTHON_PATH src/actions/optimize_threshold.py --config config/config_trader.json --symbols AAPL,META,QQQ,SPY
    
    if [[ $? -ne 0 ]]; then
        error "하이퍼파라미터 최적화 실패"
        exit 1
    fi
    
    success "하이퍼파라미터 최적화 완료"
else
    debug "하이퍼파라미터 최적화 건너뛰기 (--optimize 옵션 사용시 실행)"
fi

# ============================================================================
# 5단계: 임계점 최적화 (선택사항)
# ============================================================================
if [[ "$OPTIMIZE_THRESHOLD" == true ]]; then
    step "[5/6] 임계점 최적화"
    
    log "포트폴리오 임계점 최적화 시작"
    log "Optuna 기반 최적화 실행"
    
    # 포트폴리오 종목 목록 가져오기
    symbols=$($PYTHON_PATH -c "
import json
with open('config/config_trader.json', 'r') as f:
    config = json.load(f)
print(' '.join(config['portfolio']['symbols']))
")
    
    # --optimize-threshold 옵션이 있으면 강제 최적화, 없으면 저장된 결과 사용
    if [[ "$OPTIMIZE_THRESHOLD" == true ]]; then
        log "새로운 최적화 실행"
        $PYTHON_PATH src/actions/optimize_threshold.py --config config/config_trader.json --symbols $symbols --method optuna --force-optimize
    else
        log "저장된 최적화 결과 사용"
        $PYTHON_PATH src/actions/optimize_threshold.py --config config/config_trader.json --symbols $symbols --method optuna
    fi
    
    if [[ $? -ne 0 ]]; then
        error "임계점 최적화 실패"
        exit 1
    fi
    
    success "임계점 최적화 완료"
else
    debug "임계점 최적화 건너뛰기 (--optimize-threshold 옵션 사용시 실행)"
fi

# ============================================================================
# 6단계: 트레이딩 분석 및 신호 생성
# ============================================================================
step "[6/6] 트레이딩 분석 및 신호 생성"

log "학습된 모델을 사용한 트레이딩 분석 실행"

if [[ "$QUIET_MODE" == false ]]; then
    log "분석 항목:"
    log "  - 신경망 기반 개별 종목 예측"
    log "  - 포트폴리오 최적화 (샤프 최대화, Risk Parity 등)"
    log "  - 고급 성과 지표 계산"
    log "  - Buy & Hold 벤치마크 비교"
    log "  - 상세 백테스팅 리포트 생성"
fi

if [[ "$QUIET_MODE" == true ]]; then
    $PYTHON_PATH src/agent/trader.py --config config/config_trader.json --full-process > /dev/null 2>&1
else
    $PYTHON_PATH src/agent/trader.py --config config/config_trader.json --full-process
fi

if [[ $? -eq 0 ]]; then
    echo ""
    success "전체 프로세스 완료!"
    echo ""
    log "결과 위치:"
    log "  - 결과 파일: results/trader/"
    log "  - 로그 파일: log/trader/"
    
    if [[ "$QUIET_MODE" == false ]]; then
        echo ""
        log "생성된 리포트:"
        log "  - 개별 종목 예측 결과"
        log "  - 포트폴리오 최적 비중"
        log "  - 백테스팅 상세 분석"
        log "  - Buy & Hold 대비 성과 비교"
        log "  - 매매 내역 및 최종 보유 현황"
    fi
else
    error "트레이더 실행 실패"
    exit 1
fi 