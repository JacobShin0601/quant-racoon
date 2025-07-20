#!/bin/bash

# 스윙 전략 실행 스크립트 (2단계 평가 구조)
# 60분봉 데이터 기반 중기 매매 전략

echo "🚀 스윙 전략 실행 시작 (2단계 평가 구조)..."
echo "📊 데이터: 60분봉, 기간: 60일"
echo "🎯 전략: 개별 종목별 최적화 + 포트폴리오 평가"
echo "📁 데이터 디렉토리: data/swing"
echo "🔧 평가 모드: 2단계 (개별 + 포트폴리오)"
echo ""

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "⏰ 시작 시간: $START_TIME"

# 전체 파이프라인 실행 (cleaner → scrapper → researcher → portfolio_manager → evaluator)
echo "🔄 전체 파이프라인 실행 중..."
echo "📋 단계: cleaner → scrapper → researcher → portfolio_manager → evaluator"
echo "📁 데이터 디렉토리: data/swing"
python -m src.agent.orchestrator --time-horizon swing

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 스윙 전략 실행 완료!"
    echo "📁 결과 폴더: results/swing/"
    echo "📋 로그 폴더: log/swing/"
    echo "💾 백업 폴더: backup/swing/"
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
echo ""
echo "📁 결과 파일 위치:"
echo "  - 개별 전략 최적화: results/optimization_results_*.json"
echo "  - 2단계 평가 결과: results/comprehensive_evaluation_*.txt"
echo "  - 파이프라인 결과: results/pipeline_results_*.json" 