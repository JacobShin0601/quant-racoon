#!/bin/bash

set -e

# Conda 환경 활성화
CONDA_ENV="bedrock_manus"
PYTHON_PATH="python3"

# Conda 환경 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Python 경로 확인
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

# 환경변수 설정
export LOG_LEVEL=WARNING
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export TRADER_LOG_LEVEL="WARNING"

echo "🎯 트레이더 분석 시작..."

# 5단계만 실행 (분석 모드)
$PYTHON_PATH src/agent/trader.py --config config/config_trader.json --full-process 2>&1 | \
    grep -E "(STEP|SUCCESS|ERROR|완료|실패|현재 시장 체제:|결과 파일:|수익률:|샤프|최대 낙폭)" | \
    grep -v "초기화" | \
    grep -v "로드" | \
    grep -v "예측" | \
    grep -v "피처"

echo ""
echo "✅ 트레이더 분석 완료!"