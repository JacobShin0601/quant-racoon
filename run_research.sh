#!/bin/bash

# 연구용 실행 스크립트
# 하이퍼파라미터 최적화 연구를 실행합니다.

echo "🔬 하이퍼파라미터 최적화 연구 시작"
echo "=================================="

# 현재 디렉토리를 프로젝트 루트로 설정
cd "$(dirname "$0")"

# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 연구 실행
echo "📊 연구 실행 중..."
python -m src.agent.researcher

echo "✅ 연구 완료!" 