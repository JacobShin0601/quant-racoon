#!/bin/bash

# 고도화된 시장 분석 실행 스크립트
# LLM API 통합 및 고급 분석 기능 포함

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_advanced() {
    echo -e "${PURPLE}[ADVANCED]${NC} $1"
}

# 도움말 출력
show_help() {
    echo "🚀 고도화된 시장 분석 시스템"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  -h, --help                    이 도움말을 표시"
    echo "  -t, --type TYPE               분석 유형 선택"
    echo "      basic                     기본 분석 (기본값)"
    echo "      enhanced                  고도화된 분석 (RLMF, 신뢰도, Regime 감지)"
    echo "      llm-api                   LLM API 통합 분석"
    echo "      full                      모든 기능 통합 분석"
    echo ""
    echo "  -p, --provider PROVIDER       LLM API 제공자 선택"
    echo "      bedrock                   AWS Bedrock (기본값)"
    echo "      openai                    OpenAI"
    echo "      hybrid                    하이브리드 (API + 규칙 기반)"
    echo "      rule-only                 규칙 기반만"
    echo ""
    echo "  -m, --model MODEL             LLM 모델 선택"
    echo "      claude-3-sonnet          Claude 3 Sonnet (기본값)"
    echo "      claude-3-haiku           Claude 3 Haiku"
    echo "      gpt-4                    GPT-4"
    echo "      gpt-3.5-turbo            GPT-3.5 Turbo"
    echo ""
    echo "  -k, --api-key KEY             API 키 설정"
    echo "  -r, --region REGION           AWS 리전 설정 (기본값: us-east-1)"
    echo "  -o, --output DIR              결과 출력 디렉토리"
    echo "  -v, --verbose                 상세 로그 출력"
    echo ""
    echo "예시:"
    echo "  $0 --type enhanced                    # 고도화된 분석"
    echo "  $0 --type llm-api --provider openai  # OpenAI API 분석"
    echo "  $0 --type full --provider hybrid     # 모든 기능 통합"
    echo ""
}

# 기본 설정
ANALYSIS_TYPE="basic"
LLM_PROVIDER="bedrock"
LLM_MODEL="claude-3-sonnet"
API_KEY=""
REGION="us-east-1"
OUTPUT_DIR="results/enhanced_analysis"
VERBOSE=false

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            ANALYSIS_TYPE="$2"
            shift 2
            ;;
        -p|--provider)
            LLM_PROVIDER="$2"
            shift 2
            ;;
        -m|--model)
            LLM_MODEL="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
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

# LLM 제공자 검증
case $LLM_PROVIDER in
    bedrock|openai|hybrid|rule-only)
        ;;
    *)
        log_error "지원하지 않는 LLM 제공자: $LLM_PROVIDER"
        exit 1
        ;;
esac

# 모델 매핑
case $LLM_MODEL in
    claude-3-sonnet)
        MODEL_NAME="anthropic.claude-3-sonnet-20240229-v1:0"
        ;;
    claude-3-haiku)
        MODEL_NAME="anthropic.claude-3-haiku-20240307-v1:0"
        ;;
    gpt-4)
        MODEL_NAME="gpt-4"
        ;;
    gpt-3.5-turbo)
        MODEL_NAME="gpt-3.5-turbo"
        ;;
    *)
        log_error "지원하지 않는 모델: $LLM_MODEL"
        exit 1
        ;;
esac

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 타임스탬프 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/analysis_${ANALYSIS_TYPE}_${TIMESTAMP}.log"

# 로그 함수 업데이트
if [ "$VERBOSE" = true ]; then
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
fi

log_info "🚀 고도화된 시장 분석 시작"
log_info "분석 유형: $ANALYSIS_TYPE"
log_info "LLM 제공자: $LLM_PROVIDER"
log_info "모델: $LLM_MODEL"
log_info "출력 디렉토리: $OUTPUT_DIR"
log_info "로그 파일: $LOG_FILE"

# Python 스크립트 생성
create_python_script() {
    local script_file="$OUTPUT_DIR/run_analysis_${TIMESTAMP}.py"
    
    cat > "$script_file" << EOF
#!/usr/bin/env python3
"""
고도화된 시장 분석 실행 스크립트
생성 시간: $(date)
분석 유형: $ANALYSIS_TYPE
LLM 제공자: $LLM_PROVIDER
모델: $LLM_MODEL
"""

import sys
import os
import json
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.market_sensor import MarketSensor
from src.agent.enhancements import LLMConfig

def main():
    print("🚀 고도화된 시장 분석 시작")
    
    # LLM 설정
    llm_config = None
    enable_llm_api = False
    
    if "$ANALYSIS_TYPE" in ["llm-api", "full"]:
        llm_config = LLMConfig(
            provider="$LLM_PROVIDER",
            model_name="$MODEL_NAME",
            api_key="$API_KEY" if "$API_KEY" else None,
            region="$REGION",
            fallback_to_rules=True
        )
        enable_llm_api = True
        print(f"🤖 LLM API 설정: {llm_config.provider} - {llm_config.model_name}")
    
    # Market Sensor 초기화
    sensor = MarketSensor(
        enable_llm_api=enable_llm_api,
        llm_config=llm_config
    )
    
    # 분석 수행
    if "$ANALYSIS_TYPE" == "basic":
        print("📊 기본 분석 수행 중...")
        analysis = sensor.get_current_market_analysis(
            use_optimized_params=True,
            use_ml_model=True
        )
    else:
        print("🚀 고도화된 분석 수행 중...")
        analysis = sensor.get_enhanced_market_analysis(
            use_optimized_params=True,
            use_ml_model=True,
            enable_advanced_features=True
        )
    
    # 결과 저장
    output_file = "$OUTPUT_DIR/analysis_results_${TIMESTAMP}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ 분석 완료! 결과 저장: {output_file}")
    
    # 요약 출력
    print("\\n📊 분석 결과 요약:")
    print(f"현재 체제: {analysis.get('current_regime', 'N/A')}")
    
    if 'final_confidence' in analysis:
        final_conf = analysis['final_confidence'].get('final_confidence', 0.5)
        print(f"최종 신뢰도: {final_conf:.3f}")
    
    if 'rlmf_analysis' in analysis:
        rlmf = analysis['rlmf_analysis']
        if 'statistical_arbitrage' in rlmf:
            stat_arb = rlmf['statistical_arbitrage']
            print(f"Statistical Arbitrage: {stat_arb.get('direction', 'N/A')} (신뢰도: {stat_arb.get('confidence', 0):.3f})")
    
    if 'regime_detection' in analysis:
        regime_det = analysis['regime_detection']
        if 'regime_shift_detection' in regime_det:
            shift_det = regime_det['regime_shift_detection']
            if shift_det.get('regime_shift_detected', False):
                print("⚠️ 시장 체제 전환 감지됨!")
    
    if 'llm_api_insights' in analysis:
        print("🤖 LLM API 분석 완료")
        api_stats = analysis['llm_api_insights'].get('api_stats', {})
        if api_stats:
            print(f"API 성공률: {api_stats.get('success_rate', 0):.2%}")
    
    # LLM API 통계 (활성화된 경우)
    if sensor.llm_api_system:
        stats = sensor.get_llm_api_stats()
        print(f"\\n📈 LLM API 통계:")
        print(f"총 호출: {stats.get('total_calls', 0)}")
        print(f"성공률: {stats.get('success_rate', 0):.2%}")
        print(f"평균 응답시간: {stats.get('avg_response_time', 0):.3f}초")

if __name__ == "__main__":
    main()
EOF

    echo "$script_file"
}

# Python 스크립트 생성 및 실행
log_info "📝 Python 스크립트 생성 중..."
SCRIPT_FILE=$(create_python_script)
chmod +x "$SCRIPT_FILE"

log_info "🐍 Python 스크립트 실행 중..."
python3 "$SCRIPT_FILE"

if [ $? -eq 0 ]; then
    log_success "✅ 고도화된 시장 분석 완료!"
    log_info "결과 파일: $OUTPUT_DIR/analysis_results_${TIMESTAMP}.json"
    log_info "로그 파일: $LOG_FILE"
    
    # 결과 요약 출력
    if [ -f "$OUTPUT_DIR/analysis_results_${TIMESTAMP}.json" ]; then
        log_info "📊 결과 요약:"
        python3 -c "
import json
with open('$OUTPUT_DIR/analysis_results_${TIMESTAMP}.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'현재 체제: {data.get(\"current_regime\", \"N/A\")}')
    if 'final_confidence' in data:
        conf = data['final_confidence'].get('final_confidence', 0.5)
        print(f'최종 신뢰도: {conf:.3f}')
    if 'rlmf_analysis' in data:
        rlmf = data['rlmf_analysis']
        if 'statistical_arbitrage' in rlmf:
            sa = rlmf['statistical_arbitrage']
            print(f'Statistical Arbitrage: {sa.get(\"direction\", \"N/A\")}')
"
    fi
else
    log_error "❌ 분석 실행 실패!"
    exit 1
fi

log_info "🎉 모든 작업 완료!" 