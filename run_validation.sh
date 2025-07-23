#!/bin/bash

# Market Sensor 검증 스크립트
# 1. 종합 검증 (분류 정확도 + 전략 성과)
# 2. 백테스팅 검증 (여러 기간에 대한 검증)

set -e  # 오류 발생 시 스크립트 중단

echo "🔍 Market Sensor 검증 스크립트 시작"
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
VALIDATION_TYPE=${1:-"comprehensive"}  # 첫 번째 인자로 검증 유형 받기, 기본값 comprehensive
START_DATE=${2:-"2023-01-01"}  # 두 번째 인자로 시작 날짜 받기
END_DATE=${3:-"2023-12-31"}  # 세 번째 인자로 종료 날짜 받기
USE_OPTIMIZED=${4:-"true"}  # 네 번째 인자로 최적화된 파라미터 사용 여부 받기

log_info "설정: 검증 유형=$VALIDATION_TYPE, 기간=$START_DATE~$END_DATE, 최적화 파라미터=$USE_OPTIMIZED"

# 데이터 존재 여부 확인
if [ ! -f "data/macro/spy_data.csv" ]; then
    log_error "데이터가 없습니다. 먼저 데이터를 다운로드하세요."
    log_info "실행 방법: ./run_market_analysis.sh"
    exit 1
fi

# 검증 유형에 따른 실행
case $VALIDATION_TYPE in
    "comprehensive")
        echo ""
        log_info "🔍 종합 검증 실행"
        echo "-" * 40
        log_info "검증 기간: $START_DATE ~ $END_DATE"
        
        # Python 스크립트로 종합 검증 실행
        python -c "
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.agent.market_sensor import MarketSensor

sensor = MarketSensor()
results = sensor.run_comprehensive_validation('$START_DATE', '$END_DATE', $USE_OPTIMIZED)

if 'error' in results:
    print(f'❌ 검증 실패: {results[\"error\"]}')
    exit(1)

print('✅ 종합 검증 완료!')
print(f'📊 분류 정확도: {results[\"validation_results\"][\"overall_accuracy\"]:.3f}')
print(f'💰 총 수익률: {results[\"performance_results\"][\"overall_performance\"][\"total_return\"]:.3f}')
print(f'📈 초과 수익률: {results[\"performance_results\"][\"overall_performance\"][\"excess_return\"]:.3f}')
print(f'📊 샤프 비율: {results[\"performance_results\"][\"overall_performance\"][\"sharpe_ratio\"]:.3f}')
"
        
        if [ $? -eq 0 ]; then
            log_success "종합 검증 완료"
        else
            log_error "종합 검증 실패"
            exit 1
        fi
        ;;
    
    "backtest")
        echo ""
        log_info "🔄 백테스팅 검증 실행"
        echo "-" * 40
        
        # 백테스팅 검증 실행
        python -c "
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.agent.market_sensor import MarketSensor

sensor = MarketSensor()

# 백테스트 기간 설정
test_periods = [
    ('2022-01-01', '2022-06-30'),  # 2022년 상반기
    ('2022-07-01', '2022-12-31'),  # 2022년 하반기
    ('2023-01-01', '2023-06-30'),  # 2023년 상반기
    ('2023-07-01', '2023-12-31'),  # 2023년 하반기
]

results = sensor.run_backtest_validation(None, None, test_periods)

if 'error' in results:
    print(f'❌ 백테스팅 실패: {results[\"error\"]}')
    exit(1)

print('✅ 백테스팅 검증 완료!')

if 'overall_analysis' in results and 'error' not in results['overall_analysis']:
    analysis = results['overall_analysis']
    print(f'📊 평균 분류 정확도: {analysis[\"average_accuracy\"]:.3f}')
    print(f'💰 평균 총 수익률: {analysis[\"average_total_return\"]:.3f}')
    print(f'📈 평균 초과 수익률: {analysis[\"average_excess_return\"]:.3f}')
    print(f'📊 평균 샤프 비율: {analysis[\"average_sharpe_ratio\"]:.3f}')
    print(f'✅ 성공한 기간: {analysis[\"successful_periods\"]}/{analysis[\"total_periods\"]}')
"
        
        if [ $? -eq 0 ]; then
            log_success "백테스팅 검증 완료"
        else
            log_error "백테스팅 검증 실패"
            exit 1
        fi
        ;;
    
    "both")
        echo ""
        log_info "🔍 종합 검증 + 백테스팅 검증 실행"
        echo "-" * 40
        
        # 1. 종합 검증
        log_info "1단계: 종합 검증"
        python -c "
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.agent.market_sensor import MarketSensor

sensor = MarketSensor()
results = sensor.run_comprehensive_validation('$START_DATE', '$END_DATE', $USE_OPTIMIZED)

if 'error' in results:
    print(f'❌ 종합 검증 실패: {results[\"error\"]}')
    exit(1)

print('✅ 종합 검증 완료!')
print(f'📊 분류 정확도: {results[\"validation_results\"][\"overall_accuracy\"]:.3f}')
print(f'💰 총 수익률: {results[\"performance_results\"][\"overall_performance\"][\"total_return\"]:.3f}')
"
        
        if [ $? -ne 0 ]; then
            log_error "종합 검증 실패"
            exit 1
        fi
        
        # 2. 백테스팅 검증
        log_info "2단계: 백테스팅 검증"
        python -c "
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.agent.market_sensor import MarketSensor

sensor = MarketSensor()

test_periods = [
    ('2022-01-01', '2022-06-30'),
    ('2022-07-01', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
]

results = sensor.run_backtest_validation(None, None, test_periods)

if 'error' in results:
    print(f'❌ 백테스팅 실패: {results[\"error\"]}')
    exit(1)

print('✅ 백테스팅 검증 완료!')

if 'overall_analysis' in results and 'error' not in results['overall_analysis']:
    analysis = results['overall_analysis']
    print(f'📊 평균 분류 정확도: {analysis[\"average_accuracy\"]:.3f}')
    print(f'💰 평균 총 수익률: {analysis[\"average_total_return\"]:.3f}')
    print(f'📈 평균 초과 수익률: {analysis[\"average_excess_return\"]:.3f}')
    print(f'📊 평균 샤프 비율: {analysis[\"average_sharpe_ratio\"]:.3f}')
    print(f'✅ 성공한 기간: {analysis[\"successful_periods\"]}/{analysis[\"total_periods\"]}')
"
        
        if [ $? -ne 0 ]; then
            log_error "백테스팅 검증 실패"
            exit 1
        fi
        
        log_success "모든 검증 완료"
        ;;
    
    *)
        log_error "지원하지 않는 검증 유형: $VALIDATION_TYPE"
        log_info "지원되는 검증 유형: comprehensive, backtest, both"
        exit 1
        ;;
esac

# 결과 요약
echo ""
log_info "📋 검증 결과 요약"
echo "-" * 40

echo "📁 결과 파일 위치:"
echo "   - 검증 결과: results/validation/"
echo "   - 검증 보고서: results/validation/validation_report_*.txt"

# 최신 검증 결과 파일 찾기
LATEST_VALIDATION=$(find results/validation -name "validation_results_*.json" -type f -exec ls -t {} + | head -1 2>/dev/null || echo "없음")
LATEST_REPORT=$(find results/validation -name "validation_report_*.txt" -type f -exec ls -t {} + | head -1 2>/dev/null || echo "없음")

if [ "$LATEST_VALIDATION" != "없음" ]; then
    echo "📊 최신 검증 결과: $LATEST_VALIDATION"
fi

if [ "$LATEST_REPORT" != "없음" ]; then
    echo "📋 최신 검증 보고서: $LATEST_REPORT"
fi

echo ""
log_success "🎉 검증 완료!"
echo "=" * 60

# 사용법 안내
echo ""
echo "📋 사용법:"
echo "   ./run_validation.sh [검증유형] [시작날짜] [종료날짜] [최적화파라미터]"
echo ""
echo "   예시:"
echo "   ./run_validation.sh                                    # 기본값 (comprehensive, 2023-01-01~2023-12-31, true)"
echo "   ./run_validation.sh comprehensive                      # 종합 검증"
echo "   ./run_validation.sh backtest                          # 백테스팅 검증"
echo "   ./run_validation.sh both                              # 종합 + 백테스팅 검증"
echo "   ./run_validation.sh comprehensive 2022-01-01 2022-12-31  # 특정 기간 종합 검증"
echo ""
echo "   검증 유형: comprehensive, backtest, both"
echo "   날짜 형식: YYYY-MM-DD"
echo "   최적화 파라미터: true/false (기본값: true)"
echo ""
echo "📝 검증 내용:"
echo "   - comprehensive: 분류 정확도 + 전략 성과 분석"
echo "   - backtest: 여러 기간에 대한 백테스팅 검증"
echo "   - both: 종합 검증 + 백테스팅 검증 모두 실행" 