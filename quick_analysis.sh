#!/bin/bash

# 빠른 분석 스크립트 (기존 데이터와 파라미터 사용)

set -e

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 분석 유형 설정 (기본값: comprehensive)
ANALYSIS_TYPE=${1:-"comprehensive"}

echo "🚀 빠른 분석 시작"
echo "=" * 50
log_info "분석 유형: $ANALYSIS_TYPE"

# 데이터 확인
if [ ! -f "data/macro/spy_data.csv" ]; then
    log_warning "데이터가 없습니다. 전체 워크플로우를 실행하세요: ./run_market_analysis.sh"
    exit 1
fi

# 최적화된 파라미터 확인
if [ ! -f "config/optimal_market_params.json" ]; then
    log_warning "최적화된 파라미터가 없습니다. 전체 워크플로우를 실행하세요: ./run_market_analysis.sh"
    exit 1
fi

log_info "기존 데이터와 최적화된 파라미터 사용"
python -m src.agent.market_sensor --analysis $ANALYSIS_TYPE --use_saved_data --save_results

log_success "분석 완료!"
echo "📁 결과: results/analysis_$ANALYSIS_TYPE/" 