#!/bin/bash

# 모든 전략 실행 스크립트
# 스윙 → 장기 → 초단기 순서로 실행

echo "🎯 모든 전략 실행 시작..."
echo "📋 실행 순서: 스윙 → 장기 → 초단기"
echo ""

# 전체 시작 시간 기록
TOTAL_START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 전체 시작 시간: $TOTAL_START_TIME"
echo ""

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 실행 결과 추적
SUCCESS_COUNT=0
TOTAL_STRATEGIES=3

# 1. 스윙 전략 실행
echo "🔄 1/3 스윙 전략 실행 중..."
echo "📊 데이터: 15분봉, 기간: 60일"
python -m src.agent.orchestrator --time-horizon swing
if [ $? -eq 0 ]; then
    echo "✅ 스윙 전략 완료"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ 스윙 전략 실패"
fi
echo ""

# 2. 장기 전략 실행
echo "🔄 2/3 장기 전략 실행 중..."
echo "📊 데이터: 1일봉, 기간: 1095일"
python -m src.agent.orchestrator --time-horizon long
if [ $? -eq 0 ]; then
    echo "✅ 장기 전략 완료"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ 장기 전략 실패"
fi
echo ""

# 3. 초단기 전략 실행
echo "🔄 3/3 초단기 전략 실행 중..."
echo "📊 데이터: 1분봉, 기간: 7일"
python -m src.agent.orchestrator --time-horizon scalping
if [ $? -eq 0 ]; then
    echo "✅ 초단기 전략 완료"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ 초단기 전략 실패"
fi
echo ""

# 전체 종료 시간 기록
TOTAL_END_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 결과 요약
echo "📊 전체 실행 결과 요약"
echo "================================"
echo "⏰ 전체 시작 시간: $TOTAL_START_TIME"
echo "⏰ 전체 종료 시간: $TOTAL_END_TIME"
echo "📋 총 전략 수: $TOTAL_STRATEGIES"
echo "✅ 성공: $SUCCESS_COUNT"
echo "❌ 실패: $((TOTAL_STRATEGIES - SUCCESS_COUNT))"
echo ""

# 결과 폴더 정보
echo "📁 생성된 결과 폴더들:"
echo "   📊 results/swing/     - 스윙 전략 결과"
echo "   📊 results/long/      - 장기 전략 결과"
echo "   📊 results/scalping/  - 초단기 전략 결과"
echo ""
echo "📋 생성된 로그 폴더들:"
echo "   📋 log/swing/        - 스윙 전략 로그"
echo "   📋 log/long/         - 장기 전략 로그"
echo "   📋 log/scalping/     - 초단기 전략 로그"
echo ""
echo "💾 생성된 백업 폴더들:"
echo "   💾 backup/swing/      - 스윙 전략 백업"
echo "   💾 backup/long/       - 장기 전략 백업"
echo "   💾 backup/scalping/   - 초단기 전략 백업"
echo ""

if [ $SUCCESS_COUNT -eq $TOTAL_STRATEGIES ]; then
    echo "🎉 모든 전략이 성공적으로 완료되었습니다!"
    exit 0
else
    echo "⚠️ 일부 전략이 실패했습니다. ($SUCCESS_COUNT/$TOTAL_STRATEGIES)"
    exit 1
fi 