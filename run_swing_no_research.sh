#!/bin/bash

# 스윙 전략 실행 스크립트 (연구 단계 제외)
# 15분봉 데이터 기반 중기 매매 전략

echo "🚀 스윙 전략 실행 시작 (연구 단계 제외)..."
echo "📊 데이터: 15분봉, 기간: 60일"
echo "🎯 전략: DualMomentum, VolatilityAdjustedBreakout, SwingEMA, SwingRSI, DonchianSwing"
echo ""

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 시작 시간: $START_TIME"

# 연구 단계를 제외한 파이프라인 실행
echo "🔄 파이프라인 실행 중 (cleaner → scrapper → analyzer → evaluator → portfolio_manager)..."
python -m src.agent.orchestrator --time-horizon swing --no-research

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 스윙 전략 실행 완료!"
    echo "📁 결과 폴더: results/swing/"
    echo "📋 로그 폴더: log/swing/"
    echo "💾 백업 폴더: backup/swing/"
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