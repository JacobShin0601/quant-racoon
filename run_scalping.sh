#!/bin/bash

# 초단기 전략 실행 스크립트
# 1분봉 데이터 기반 스캘핑 전략

echo "🚀 초단기 전략 실행 시작..."
echo "📊 데이터: 1분봉, 기간: 7일"
echo "🎯 전략: VWAPMACDScalping, KeltnerRSIScalping, AbsorptionScalping, RSIBollingerScalping"
echo "⚡ 고빈도 거래 모드"
echo "📁 데이터 디렉토리: data/scalping"
echo ""

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 시작 시간: $START_TIME"

# 전체 파이프라인 실행 (cleaner → scrapper → analyzer → researcher → evaluator → portfolio_manager)
echo "🔄 전체 파이프라인 실행 중..."
echo "📋 단계: cleaner → scrapper → analyzer → researcher → evaluator → portfolio_manager"
echo "📁 데이터 디렉토리: data/scalping"
python -m src.agent.orchestrator --time-horizon scalping

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 초단기 전략 실행 완료!"
    echo "📁 결과 폴더: results/scalping/"
    echo "📋 로그 폴더: log/scalping/"
    echo "💾 백업 폴더: backup/scalping/"
    echo "📊 데이터 폴더: data/scalping/"
else
    echo ""
    echo "❌ 초단기 전략 실행 실패!"
    exit 1
fi

# 실행 시간 계산
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 종료 시간: $END_TIME"

echo ""
echo "🎉 초단기 전략 분석이 완료되었습니다!" 