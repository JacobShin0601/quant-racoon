#!/bin/bash

# 스윙 전략 실행 스크립트 (2단계 평가 구조)
# 일일 데이터 기반 스윙 매매 전략

# 옵션 파싱
USE_CACHED_DATA=""
CACHE_DAYS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-cached-data)
            USE_CACHED_DATA="--use-cached-data"
            shift
            ;;
        --cache-days)
            CACHE_DAYS="$2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: $0 [--use-cached-data] [--cache-days N]"
            exit 1
            ;;
    esac
done

echo "🚀 스윙 전략 최적화 실행 시작..."
echo "📊 데이터: 일봉, 기간: 365일"
echo "🎯 전략: 개별 종목별 최적화 + 포트폴리오 평가"
echo "📁 데이터 디렉토리: data/swing"
echo "🔧 평가 모드: 2단계 (개별 + 포트폴리오)"
if [ -n "$USE_CACHED_DATA" ]; then
    echo "💾 캐시 데이터 사용 모드 (캐시 유효 기간: ${CACHE_DAYS}일)"
fi
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
BACKUP_DIR="backup/swing/backup_${UUID}"
echo "💾 백업 폴더 생성: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# 전체 파이프라인 실행 (cleaner → scrapper → researcher → portfolio_manager → evaluator)
echo "🔄 전체 파이프라인 실행 중..."
echo "📋 단계: cleaner → scrapper → researcher → portfolio_manager → evaluator"
echo "📁 데이터 디렉토리: data/swing"
echo "📁 결과 디렉토리: results/swing"
if [ -n "$USE_CACHED_DATA" ]; then
    python3 -m src.agent.orchestrator --time-horizon swing --uuid "$UUID" $USE_CACHED_DATA --cache-days "$CACHE_DAYS"
else
    python3 -m src.agent.orchestrator --time-horizon swing --uuid "$UUID"
fi

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 스윙 전략 실행 완료!"
    
    # 백업 실행
    echo "💾 백업 시작..."
    echo "📁 백업 대상: data/swing, log/swing, results/swing"
    echo "📁 백업 위치: $BACKUP_DIR"
    
    # data/swing 백업
    if [ -d "data/swing" ]; then
        echo "📊 data/swing 백업 중..."
        cp -r data/swing "$BACKUP_DIR/"
        echo "✅ data/swing 백업 완료"
    else
        echo "⚠️ data/swing 폴더가 존재하지 않습니다"
    fi
    
    # log/swing 백업
    if [ -d "log/swing" ]; then
        echo "📋 log/swing 백업 중..."
        cp -r log/swing "$BACKUP_DIR/"
        echo "✅ log/swing 백업 완료"
    else
        echo "⚠️ log/swing 폴더가 존재하지 않습니다"
    fi
    
    # results/swing 백업
    if [ -d "results/swing" ]; then
        echo "📊 results/swing 백업 중..."
        cp -r results/swing "$BACKUP_DIR/"
        echo "✅ results/swing 백업 완료"
    else
        echo "⚠️ results/swing 폴더가 존재하지 않습니다"
    fi
    
    # 백업 정보 파일 생성
    BACKUP_INFO_FILE="$BACKUP_DIR/backup_info.json"
    cat > "$BACKUP_INFO_FILE" << EOF
{
  "uuid": "$UUID",
  "backup_time": "$(date +"%Y-%m-%d %H:%M:%S")",
  "strategy": "swing",
  "backup_contents": {
    "data_swing": "$(if [ -d "data/swing" ]; then echo "true"; else echo "false"; fi)",
    "log_swing": "$(if [ -d "log/swing" ]; then echo "true"; else echo "false"; fi)",
    "results_swing": "$(if [ -d "results/swing" ]; then echo "true"; else echo "false"; fi)"
  },
  "execution_info": {
    "start_time": "$START_TIME",
    "end_time": "$(date +"%Y-%m-%d %H:%M:%S")",
    "script": "run_swing.sh"
  }
}
EOF
    echo "✅ 백업 정보 파일 생성: $BACKUP_INFO_FILE"
    
    echo ""
    echo "📁 결과 폴더: results/swing/"
    echo "📋 로그 폴더: log/swing/"
    echo "💾 백업 폴더: $BACKUP_DIR"
    echo "📊 데이터 폴더: data/swing/"
    echo "🔧 2단계 평가 완료 (개별 + 포트폴리오)"
else
    echo ""
    echo "❌ 스윙 전략 실행 실패!"
    exit 1
fi

# 실행 시간 계산
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 종료 시간: $END_TIME"

echo ""
echo "🎉 스윙 전략 분석이 완료되었습니다!"
echo ""
echo "📊 실행 결과 요약:"
echo "  1단계: 데이터 정리 및 수집"
echo "  2단계: 개별 종목별 전략 최적화"
echo "  3단계: 2단계 평가 (개별 성과 + 포트폴리오 성과)"
echo "  4단계: 포트폴리오 최적화"
echo "  5단계: 백업 생성 (UUID: $UUID)"
echo ""
echo "📁 결과 파일 위치:"
echo "  - 개별 전략 최적화: results/swing/optimization_results_*.json"
echo "  - 2단계 평가 결과: results/swing/comprehensive_evaluation_*.txt"
echo "  - 파이프라인 결과: results/swing/pipeline_results_*.json"
echo "  - 백업 폴더: $BACKUP_DIR" 