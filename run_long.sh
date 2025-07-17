#!/bin/bash

# 장기 전략 실행 스크립트
# 1일봉 데이터 기반 장기 투자 전략

echo "🚀 장기 전략 실행 시작..."
echo "📊 데이터: 1일봉, 기간: 1095일 (3년)"
echo "🎯 전략: RiskParityLeverage, FixedWeightRebalance, ETFMomentumRotation, TrendFollowingMA200, ReturnStacking"
echo "💼 포트폴리오 모드: 활성화"
echo ""

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 시작 시간: $START_TIME"

# 전체 파이프라인 실행 (cleaner → scrapper → analyzer → researcher → evaluator → portfolio_manager)
echo "🔄 전체 파이프라인 실행 중..."
echo "📋 단계: cleaner → scrapper → analyzer → researcher → evaluator → portfolio_manager"
python -m src.agent.orchestrator --time-horizon long

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 장기 전략 실행 완료!"
    echo "📁 결과 폴더: results/long/"
    echo "📋 로그 폴더: log/long/"
    echo "💾 백업 폴더: backup/long/"
else
    echo ""
    echo "❌ 장기 전략 실행 실패!"
    exit 1
fi

# 실행 시간 계산
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 종료 시간: $END_TIME"

echo ""
echo "🎉 장기 전략 분석이 완료되었습니다!" 