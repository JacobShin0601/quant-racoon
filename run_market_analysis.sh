#!/bin/bash

# Market Sensor 전체 워크플로우 스크립트
# 1. 데이터 다운로드 (선택적)
# 2. 하이퍼파라미터 튜닝 (experiment 모드)
# 3. 시장 분석 (analysis 모드)

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
FORCE_DOWNLOAD=${3:-"false"}  # 세 번째 인자로 강제 다운로드 여부 받기, 기본값 false

log_info "설정: 시도 횟수=$N_TRIALS, 분석 유형=$ANALYSIS_TYPE, 강제 다운로드=$FORCE_DOWNLOAD"

# 데이터 존재 여부 확인 함수
check_data_exists() {
    if [ -f "data/macro/spy_data.csv" ] && [ -f "data/macro/macro_data.json" ]; then
        return 0  # 데이터 존재
    else
        return 1  # 데이터 없음
    fi
}

# 1단계: 데이터 준비
echo ""
log_info "1단계: 데이터 준비"
echo "-" * 40

if [ "$FORCE_DOWNLOAD" = "true" ]; then
    log_info "강제 데이터 다운로드 중..."
    python -m src.agent.market_sensor --mode collect --force_download
    log_success "데이터 다운로드 완료"
elif check_data_exists; then
    log_info "기존 데이터가 발견되었습니다."
    read -p "새로운 데이터를 다운로드하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "새로운 데이터 다운로드 중..."
        python -m src.agent.market_sensor --mode collect --force_download
        log_success "데이터 다운로드 완료"
    else
        log_info "기존 데이터 사용"
    fi
else
    log_info "기존 데이터가 없습니다. 새로운 데이터 다운로드 중..."
    python -m src.agent.market_sensor --mode collect --force_download
    log_success "데이터 다운로드 완료"
fi

# 2단계: 하이퍼파라미터 튜닝 (experiment 모드)
echo ""
log_info "2단계: 하이퍼파라미터 튜닝 (experiment 모드)"
echo "-" * 40

if [ -f "config/optimal_market_params.json" ] || [ -f "results/market_sensor_optimization/best_params.json" ]; then
    log_warning "기존 최적화된 파라미터가 발견되었습니다."
    read -p "새로 하이퍼파라미터 튜닝을 실행하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "하이퍼파라미터 튜닝 시작 (시도 횟수: $N_TRIALS)..."
        python -m src.agent.market_sensor --mode experiment --use_saved_data --save_results --n_trials $N_TRIALS
        log_success "하이퍼파라미터 튜닝 완료"
    else
        log_info "기존 최적화된 파라미터 사용"
    fi
else
    log_info "하이퍼파라미터 튜닝 시작 (시도 횟수: $N_TRIALS)..."
    python -m src.agent.market_sensor --mode experiment --use_saved_data --save_results --n_trials $N_TRIALS
    log_success "하이퍼파라미터 튜닝 완료"
fi

# 3단계: 시장 분석 (analysis 모드)
echo ""
log_info "3단계: 시장 분석 (analysis 모드)"
echo "-" * 40

log_info "분석 유형: $ANALYSIS_TYPE"

# 분석 유형에 따른 적절한 모드 선택
case $ANALYSIS_TYPE in
    "technical"|"macro"|"sector"|"comprehensive"|"all")
        # --analysis 옵션을 사용하여 분석 실행
        python -m src.agent.market_sensor --analysis $ANALYSIS_TYPE --use_saved_data --save_results --use_optimized
        ;;
    *)
        log_error "지원하지 않는 분석 유형: $ANALYSIS_TYPE"
        log_info "지원되는 분석 유형: technical, macro, sector, comprehensive, all"
        exit 1
        ;;
esac

log_success "시장 분석 완료"

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
echo "   ./run_market_analysis.sh [시도횟수] [분석유형] [강제다운로드]"
echo ""
echo "   예시:"
echo "   ./run_market_analysis.sh                    # 기본값 (20회, comprehensive, false)"
echo "   ./run_market_analysis.sh 50                 # 50회 시도, comprehensive 분석"
echo "   ./run_market_analysis.sh 30 technical       # 30회 시도, 기술적 분석"
echo "   ./run_market_analysis.sh 100 all true       # 100회 시도, 모든 분석, 강제 다운로드"
echo ""
echo "   분석 유형: technical, macro, sector, comprehensive, all"
echo "   강제 다운로드: true/false (기본값: false)"
echo ""
echo "📝 워크플로우 설명:"
echo "   1. 데이터 준비: 기존 데이터 확인 또는 새로 다운로드"
echo "   2. 하이퍼파라미터 튜닝: --mode experiment로 최적화 실행"
echo "   3. 시장 분석: --analysis 옵션으로 분석 실행"
echo "   4. 결과 저장: 최적화 및 분석 결과를 각각 저장" 