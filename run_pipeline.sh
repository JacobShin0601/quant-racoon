#!/bin/bash

# 통합 파이프라인 실행 스크립트
# 데이터 관리가 개선된 버전

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
    echo -e "\n${GREEN}🚀 퀀트 트레이딩 통합 파이프라인${NC}"
    echo -e "\n사용법: $0 [옵션]"
    echo -e "\n옵션:"
    echo -e "  --time-horizon {scalping|swing|long}  시간 지평 설정 (기본값: swing)"
    echo -e "  --stages STAGE1,STAGE2,...            실행할 단계 지정"
    echo -e "  --use-cached-data                     캐시된 데이터 사용"
    echo -e "  --cache-days N                        캐시 유효 기간 (기본값: 1일)"
    echo -e "  --no-research                         연구 단계 건너뛰기"
    echo -e "  --uuid UUID                           실행 UUID 지정"
    echo -e "  -h, --help                            도움말"
    echo -e "\n실행 단계:"
    echo -e "  1. cleaner       - 디렉토리 정리"
    echo -e "  2. scrapper      - 데이터 수집"
    echo -e "  3. researcher    - 하이퍼파라미터 최적화"
    echo -e "  4. portfolio_manager - 포트폴리오 최적화"
    echo -e "  5. evaluator     - 백테스팅 및 평가"
    echo -e "\n예제:"
    echo -e "  $0                                    # 기본 실행 (swing, 모든 단계)"
    echo -e "  $0 --time-horizon long                # 장기 전략"
    echo -e "  $0 --use-cached-data                  # 캐시 데이터 사용"
    echo -e "  $0 --no-research                      # 연구 단계 제외"
    echo -e "  $0 --stages cleaner,scrapper          # 특정 단계만 실행"
    echo ""
}

# 기본 설정
TIME_HORIZON="swing"
USE_CACHED_DATA=""
CACHE_DAYS=1
NO_RESEARCH=false
STAGES=""
UUID=""
CONFIG_FILE=""

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --time-horizon)
            TIME_HORIZON="$2"
            shift 2
            ;;
        --stages)
            STAGES="--stages $(echo $2 | tr ',' ' ')"
            shift 2
            ;;
        --use-cached-data)
            USE_CACHED_DATA="--use-cached-data"
            shift
            ;;
        --cache-days)
            CACHE_DAYS="$2"
            shift 2
            ;;
        --no-research)
            NO_RESEARCH=true
            shift
            ;;
        --uuid)
            UUID="--uuid $2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="--config $2"
            shift 2
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

# 시간대별 설정 파일 확인
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_PATH="config/config_${TIME_HORIZON}.json"
    if [[ -f "$CONFIG_PATH" ]]; then
        CONFIG_FILE="--config $CONFIG_PATH"
        log "시간대별 설정 파일 사용: $CONFIG_PATH"
    else
        warn "시간대별 설정 파일이 없습니다: $CONFIG_PATH"
        log "기본 설정 파일 사용: config/config_default.json"
    fi
fi

# UUID 생성 (지정되지 않은 경우)
if [[ -z "$UUID" ]]; then
    UUID_VALUE=$(date +"%Y%m%d_%H%M%S")_$(uuidgen | cut -d'-' -f1 || echo "random")
    UUID="--uuid $UUID_VALUE"
    log "실행 UUID 생성: $UUID_VALUE"
fi

# 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

log "🚀 퀀트 트레이딩 파이프라인 시작"
log "⏰ 시작 시간: $START_TIME"
log "📊 시간 지평: $TIME_HORIZON"
log "💾 캐시 사용: $(if [[ -n "$USE_CACHED_DATA" ]]; then echo "예 (유효기간: ${CACHE_DAYS}일)"; else echo "아니오"; fi)"
log "🔬 연구 단계: $(if [[ "$NO_RESEARCH" == true ]]; then echo "건너뛰기"; else echo "실행"; fi)"

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# no-research 옵션 처리
if [[ "$NO_RESEARCH" == true && -z "$STAGES" ]]; then
    STAGES="--stages cleaner scrapper portfolio_manager evaluator"
    log "연구 단계를 건너뜁니다"
fi

# 메인 명령어 구성
CMD="python -m src.agent.orchestrator"
CMD="$CMD --time-horizon $TIME_HORIZON"
CMD="$CMD $CONFIG_FILE"
CMD="$CMD $UUID"
CMD="$CMD $USE_CACHED_DATA"
CMD="$CMD --cache-days $CACHE_DAYS"
CMD="$CMD $STAGES"

# 실행
step "파이프라인 실행 명령어:"
echo "  $CMD"
echo ""

# 파이프라인 실행
eval $CMD

# 실행 결과 확인
if [[ $? -eq 0 ]]; then
    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    
    log "✅ 파이프라인 실행 완료!"
    log "⏰ 종료 시간: $END_TIME"
    
    # 결과 디렉토리 안내
    echo ""
    log "📁 결과 위치:"
    log "  - 데이터: data/$TIME_HORIZON/"
    log "  - 결과: results/$TIME_HORIZON/"
    log "  - 로그: log/$TIME_HORIZON/"
    
    # 백업 안내
    if [[ -d "backup/$TIME_HORIZON" ]]; then
        LATEST_BACKUP=$(ls -t backup/$TIME_HORIZON | head -1)
        if [[ -n "$LATEST_BACKUP" ]]; then
            log "  - 백업: backup/$TIME_HORIZON/$LATEST_BACKUP"
        fi
    fi
    
    echo ""
    log "🎉 모든 작업이 성공적으로 완료되었습니다!"
else
    error "파이프라인 실행 실패!"
    exit 1
fi

# 실행 시간 계산 및 출력
if command -v python3 &> /dev/null; then
    DURATION=$(python3 -c "
from datetime import datetime
start = datetime.strptime('$START_TIME', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('$END_TIME', '%Y-%m-%d %H:%M:%S')
duration = end - start
hours, remainder = divmod(duration.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
if hours > 0:
    print(f'{hours}시간 {minutes}분 {seconds}초')
elif minutes > 0:
    print(f'{minutes}분 {seconds}초')
else:
    print(f'{seconds}초')
")
    log "⏱️ 총 실행 시간: $DURATION"
fi