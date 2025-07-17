#!/bin/bash

# 모든 전략별 폴더 정리 스크립트
# data 폴더는 건드리지 않고, 전략별 하위폴더만 정리

echo "🧹 모든 전략별 하위폴더 정리 시작..."
echo "📁 정리 대상: results/*, log/*, backup/* 하위폴더들"
echo "💾 유지 대상: data 폴더"
echo ""

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 정리할 하위폴더 목록
FOLDERS=(
    "results/swing" "results/long" "results/scalping"
    "log/swing" "log/long" "log/scalping"
    "backup/swing" "backup/long" "backup/scalping"
)

# 각 하위폴더 정리
for folder in "${FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        echo "🗑️ $folder 폴더 정리 중..."
        rm -rf "$folder"/*
        echo "✅ $folder 폴더 정리 완료"
    else
        echo "📁 $folder 폴더가 존재하지 않음 - 스킵"
    fi
done

echo ""
echo "🎉 모든 전략별 하위폴더 정리 완료!"
echo "💾 data 폴더는 그대로 유지되었습니다." 