#!/bin/bash

# 고도화된 시장 분석 실행 스크립트
# LLM API 통합 및 고급 분석 기능 포함

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 도움말 출력
show_help() {
    echo "🚀 고도화된 시장 분석 시스템"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  -h, --help                    이 도움말을 표시"
    echo "  --basic                       기본 분석 실행 (GlobalMacroDataCollector 기반)"
    echo "  --enhanced                    고도화된 분석 실행 (기본 + LLM + 고급 기능)"
    echo "  --llm-api                     LLM API 통합 분석 실행"
    echo "  --full                        모든 기능 통합 분석 실행"
    echo "  -o, --output DIR              결과 출력 디렉토리"
    echo "  --use-cached-data             저장된 매크로 데이터 사용 (새로 다운로드 안함)"
    echo "  --use-cached-optimization     저장된 최적화 결과 사용 (하이퍼파라미터 튜닝 안함)"
    echo "  --cache-days DAYS             캐시 유효기간 (기본값: 1일)"
    echo "  --use-random-forest           Random Forest 모델 사용 (기본값: True)"
    echo "  --retrain-rf-model            Random Forest 모델 재학습"
    echo "  --no-random-forest            Random Forest 모델 사용 안함 (규칙 기반 사용)"
    echo ""
    echo "예시:"
    echo "  $0 --basic                           # 기본 분석"
    echo "  $0 --enhanced                        # 고도화된 분석"
    echo "  $0 --enhanced -o results/macro/test  # 지정된 디렉토리에 결과 저장"
    echo "  $0 --basic --use-cached-data         # 캐시된 데이터 사용"
    echo "  $0 --enhanced --use-cached-optimization  # 캐시된 최적화 결과 사용"
    echo "  $0 --enhanced --retrain-rf-model     # Random Forest 모델 재학습"
    echo "  $0 --enhanced --no-random-forest     # 규칙 기반 분석만 사용"
    echo ""
}

# 기본 설정
ANALYSIS_TYPE="basic"
OUTPUT_DIR="results/macro/enhanced"
USE_CACHED_DATA=false
USE_CACHED_OPTIMIZATION=false
CACHE_DAYS=1
USE_RANDOM_FOREST=true
RETRAIN_RF_MODEL=false

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --basic)
            ANALYSIS_TYPE="basic"
            OUTPUT_DIR="results/macro/basic"
            shift
            ;;
        --enhanced)
            ANALYSIS_TYPE="enhanced"
            OUTPUT_DIR="results/macro/enhanced"
            shift
            ;;
        --llm-api)
            ANALYSIS_TYPE="llm-api"
            OUTPUT_DIR="results/macro/llm-api"
            shift
            ;;
        --full)
            ANALYSIS_TYPE="full"
            OUTPUT_DIR="results/macro/full"
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use-cached-data)
            USE_CACHED_DATA=true
            shift
            ;;
        --use-cached-optimization)
            USE_CACHED_OPTIMIZATION=true
            shift
            ;;
        --cache-days)
            CACHE_DAYS="$2"
            shift 2
            ;;
        --use-random-forest)
            USE_RANDOM_FOREST=true
            shift
            ;;
        --retrain-rf-model)
            RETRAIN_RF_MODEL=true
            shift
            ;;
        --no-random-forest)
            USE_RANDOM_FOREST=false
            shift
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 분석 유형 검증
case $ANALYSIS_TYPE in
    basic|enhanced|llm-api|full)
        ;;
    *)
        log_error "지원하지 않는 분석 유형: $ANALYSIS_TYPE"
        exit 1
        ;;
esac

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 타임스탬프 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/analysis_${ANALYSIS_TYPE}_${TIMESTAMP}.log"

log_info "🚀 시장 분석 시작"
log_info "분석 유형: $ANALYSIS_TYPE"
log_info "출력 디렉토리: $OUTPUT_DIR"
log_info "캐시 설정: 데이터=$USE_CACHED_DATA, 최적화=$USE_CACHED_OPTIMIZATION, 유효기간=${CACHE_DAYS}일"
log_info "Random Forest 설정: 사용=$USE_RANDOM_FOREST, 재학습=$RETRAIN_RF_MODEL"
log_info "로그 파일: $LOG_FILE"

# Python 모듈 직접 실행
log_info "🐍 Python 분석 실행 중..."

python3 -c "
import sys
import os
import logging
from datetime import datetime

# yfinance 디버그 로그 억제
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# 로그 레벨 설정 (INFO로 변경하여 불필요한 디버그 로그 억제)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath('.'))
sys.path.insert(0, project_root)

from src.agent.market_sensor import MarketSensor

def main():
    print('🚀 시장 분석 시작')
    print(f'분석 유형: $ANALYSIS_TYPE')
    print(f'Random Forest 설정: 사용=$USE_RANDOM_FOREST, 재학습=$RETRAIN_RF_MODEL')
    
    # LLM 설정 (enhanced, llm-api, full에서만 활성화)
    llm_config = None
    enable_llm_api = False
    
    if '$ANALYSIS_TYPE' in ['enhanced', 'llm-api', 'full']:
        llm_config = {
            'provider': 'hybrid',
            'model_name': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'fallback_to_rules': True,
            'max_tokens': 2000,  # 더 긴 응답을 위해 토큰 수 증가
            'temperature': 0.1   # 일관된 분석을 위해 낮은 온도
        }
        enable_llm_api = True
        print('🤖 LLM API 활성화됨 (종합 분석 모드)')
    
    # Market Sensor 초기화
    sensor = MarketSensor(
        enable_llm_api=enable_llm_api,
        llm_config=llm_config,
        use_cached_data='$USE_CACHED_DATA' == 'true',
        use_cached_optimization='$USE_CACHED_OPTIMIZATION' == 'true',
        cache_days=int('$CACHE_DAYS'),
        use_random_forest='$USE_RANDOM_FOREST' == 'true',
        retrain_rf_model='$RETRAIN_RF_MODEL' == 'true'
    )
    
    # 분석 수행
    if '$ANALYSIS_TYPE' == 'basic':
        print('📊 기본 분석 수행 중... (GlobalMacroDataCollector 기반)')
        result = sensor.run_basic_analysis(
            output_dir='$OUTPUT_DIR',
            verbose=True
        )
    else:
        print('🚀 고도화된 분석 수행 중... (기본 + LLM + 고급 기능)')
        result = sensor.run_enhanced_analysis(
            output_dir='$OUTPUT_DIR',
            verbose=True
        )
    
    if result:
        print('✅ 분석 완료!')
        print(f'결과 저장 위치: $OUTPUT_DIR')
        print(f'세션 UUID: {result.session_uuid}')
        # 세션 UUID를 환경변수로 설정하여 쉘에서 사용할 수 있도록 함
        import os
        os.environ['SESSION_UUID'] = result.session_uuid
        return True
    else:
        print('❌ 분석 실패!')
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    log_success "✅ 시장 분석 완료!"
    
    # 매크로 데이터는 Python에서 다운로드 직후 즉시 복사되므로 여기서는 건너뜀
    if [ "$USE_CACHED_DATA" = "false" ]; then
        log_info "📁 매크로 데이터는 다운로드 직후 Python에서 자동으로 복사되었습니다."
    else
        log_info "📁 캐시된 데이터 사용 중 - 파일 복사 건너뜀"
    fi
    
    # 결과 요약 출력
    if [ -f "$OUTPUT_DIR/analysis_results_${TIMESTAMP}.json" ]; then
        log_info "📊 결과 요약:"
        python3 -c "
import json
with open('$OUTPUT_DIR/analysis_results_${TIMESTAMP}.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'현재 체제: {data.get(\"current_regime\", \"N/A\")}')
    print(f'분석 유형: {data.get(\"analysis_type\", \"N/A\")}')
    print(f'데이터 기간: {data.get(\"data_period\", \"N/A\")}')
    print(f'신뢰도: {data.get(\"confidence\", 0):.3f}')
    
    # 매크로 분석 결과
    if 'macro_analysis' in data:
        macro = data['macro_analysis']
        print(f'매크로 조건: {macro.get(\"market_condition\", \"N/A\")}')
        print(f'매크로 신뢰도: {macro.get(\"confidence\", 0):.3f}')
    
    # 최적화 성과
    if 'optimization_performance' in data:
        perf = data['optimization_performance']
        if 'sharpe_ratio' in perf:
            print(f'최적화 Sharpe Ratio: {perf[\"sharpe_ratio\"]:.4f}')
    
    # 최종 신뢰도
    if 'final_confidence' in data:
        conf = data['final_confidence'].get('final_confidence', 0.5)
        print(f'최종 신뢰도: {conf:.3f}')
    
    # 추천
    if 'enhanced_recommendations' in data:
        rec = data['enhanced_recommendations']
        print(f'주요 전략: {rec.get(\"primary_strategy\", \"N/A\")}')
        print(f'포지션 사이징: {rec.get(\"position_sizing\", \"N/A\")}')
    
    # LLM API 통계 및 종합 분석 결과
    if 'llm_api_insights' in data:
        llm = data['llm_api_insights']
        if 'api_stats' in llm:
            stats = llm['api_stats']
            print(f'LLM API 성공률: {stats.get(\"success_rate\", 0):.2%}')
        
        # 종합 분석 결과 출력
        if 'comprehensive_analysis' in llm:
            comp = llm['comprehensive_analysis']
            if 'market_dynamics' in comp:
                dynamics = comp['market_dynamics']
                print(f'시장 동인: {dynamics.get(\"primary_drivers\", [])}')
                print(f'추세 강도: {dynamics.get(\"trend_strength\", \"N/A\")}')
        
        # 위험 평가 결과
        if 'risk_assessment' in llm:
            risk = llm['risk_assessment']
            print(f'단기 위험: {risk.get(\"short_term_risks\", [])}')
            print(f'중기 위험: {risk.get(\"medium_term_risks\", [])}')
        
        # 전략적 추천
        if 'strategic_recommendations' in llm:
            strategy = llm['strategic_recommendations']
            if 'portfolio_allocation' in strategy:
                alloc = strategy['portfolio_allocation']
                print(f'포트폴리오 배분: 주식 {alloc.get(\"equity_allocation\", \"N/A\")}, 채권 {alloc.get(\"bond_allocation\", \"N/A\")}')
        
        # 핵심 인사이트
        if 'key_insights' in llm:
            insights = llm['key_insights']
            print(f'핵심 인사이트: {insights[:3]}')  # 처음 3개만 출력
    
    # Random Forest 모델 정보
    if 'random_forest_info' in data:
        rf_info = data['random_forest_info']
        print(f'RF 모델 사용: {rf_info.get(\"model_used\", False)}')
        if rf_info.get('model_used', False):
            print(f'RF 모델 정확도: {rf_info.get(\"accuracy\", 0):.3f}')
            print(f'RF 모델 학습일: {rf_info.get(\"trained_at\", \"N/A\")}')
"
    fi
else
    log_error "❌ 분석 실행 실패!"
    exit 1
fi

log_info "🎉 모든 작업 완료!" 