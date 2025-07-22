#!/bin/bash

# Market Sensor 전체 워크플로우 스크립트
# 1. 데이터 다운로드
# 2. 하이퍼파라미터 튜닝
# 3. 매크로 분석

set -e  # 오류 발생 시 스크립트 중단

echo "🚀 Market Sensor 전체 워크플로우 시작"
echo "=" * 60

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수: 로그 출력
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 기본 설정
N_TRIALS=${1:-20}  # 첫 번째 인자로 시도 횟수 받기, 기본값 20
ANALYSIS_TYPE=${2:-"comprehensive"}  # 두 번째 인자로 분석 유형 받기, 기본값 comprehensive

log_info "설정: 시도 횟수=$N_TRIALS, 분석 유형=$ANALYSIS_TYPE"

# 1단계: 데이터 다운로드
echo ""
log_info "1단계: 데이터 다운로드"
echo "-" * 40

if [ -f "data/macro/spy_data.csv" ]; then
    log_warning "기존 데이터가 발견되었습니다."
    read -p "새로운 데이터를 다운로드하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "새로운 데이터 다운로드 중..."
        python -m src.agent.market_sensor --mode collect --force_download
    else
        log_info "기존 데이터 사용"
        # 기존 데이터가 있는지 확인
        if [ ! -f "data/macro/spy_data.csv" ]; then
            log_error "기존 데이터가 없습니다. 데이터를 다운로드하지 않고 종료합니다."
            exit 1
        fi
    fi
else
    log_info "새로운 데이터 다운로드 중..."
    python -m src.agent.market_sensor --mode collect --force_download
fi

log_success "데이터 준비 완료"

# 2단계: 하이퍼파라미터 튜닝
echo ""
log_info "2단계: 하이퍼파라미터 튜닝"
echo "-" * 40

if [ -f "config/optimal_market_params.json" ]; then
    log_warning "기존 최적화된 파라미터가 발견되었습니다."
    read -p "새로 하이퍼파라미터 튜닝을 실행하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "하이퍼파라미터 튜닝 시작 (시도 횟수: $N_TRIALS)..."
        python -m src.agent.market_sensor --mode experiment --use_saved_data --save_results --n_trials $N_TRIALS
    else
        log_info "기존 최적화된 파라미터 사용"
    fi
else
    log_info "하이퍼파라미터 튜닝 시작 (시도 횟수: $N_TRIALS)..."
    python -m src.agent.market_sensor --mode experiment --use_saved_data --save_results --n_trials $N_TRIALS
fi

log_success "하이퍼파라미터 튜닝 완료"

# 3단계: 매크로 분석
echo ""
log_info "3단계: 매크로 분석"
echo "-" * 40

log_info "분석 유형: $ANALYSIS_TYPE"
python -m src.agent.market_sensor --analysis $ANALYSIS_TYPE --use_saved_data --save_results

log_success "매크로 분석 완료"

# 4단계: 결과 요약
echo ""
log_info "4단계: 결과 요약"
echo "-" * 40

echo "📊 분석 완료!"
echo "📁 결과 파일 위치:"
echo "   - 최적화 결과: results/market_sensor_optimization/"
echo "   - 분석 결과: results/analysis_$ANALYSIS_TYPE/"
echo "   - 거래 로그: log/market_sensor/"

# 최신 결과 파일 찾기
LATEST_OPTIMIZATION=$(find results/market_sensor_optimization -name "best_params.json" -type f -exec ls -t {} + | head -1 2>/dev/null || echo "없음")
LATEST_ANALYSIS=$(find results/analysis_$ANALYSIS_TYPE -name "*.json" -type f -exec ls -t {} + | head -1 2>/dev/null || echo "없음")

if [ "$LATEST_OPTIMIZATION" != "없음" ]; then
    echo "🔧 최신 최적화 결과: $LATEST_OPTIMIZATION"
fi

if [ "$LATEST_ANALYSIS" != "없음" ]; then
    echo "📈 최신 분석 결과: $LATEST_ANALYSIS"
fi

echo ""
log_success "🎉 전체 워크플로우 완료!"
echo "=" * 60

# 사용법 안내
echo ""
echo "📋 사용법:"
echo "   ./run_market_analysis.sh [시도횟수] [분석유형]"
echo ""
echo "   예시:"
echo "   ./run_market_analysis.sh                    # 기본값 (20회, comprehensive)"
echo "   ./run_market_analysis.sh 50                 # 50회 시도, comprehensive 분석"
echo "   ./run_market_analysis.sh 30 technical       # 30회 시도, 기술적 분석"
echo "   ./run_market_analysis.sh 100 all            # 100회 시도, 모든 분석"
echo ""
echo "   분석 유형: technical, macro, sector, comprehensive, all" 