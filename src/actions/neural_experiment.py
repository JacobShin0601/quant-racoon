"""
신경망 구조 실험 모듈
다양한 신경망 구조로 실험하여 종목별 최적 모델 찾기
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import logging
from copy import deepcopy

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.actions.neural_stock_predictor import StockPredictionNetwork

logger = logging.getLogger(__name__)


class NeuralExperiment:
    """신경망 구조 실험 클래스"""
    
    def __init__(
        self,
        base_config: Dict,
        experiment_configs: List[Dict],
        data_dir: str,
        model_dir: str
    ):
        self.base_config = base_config
        self.experiment_configs = experiment_configs
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results = {}
        
    def run_experiments(self, symbols: List[str], force_retrain: bool = False) -> Dict:
        """
        모든 실험 실행
        
        Args:
            symbols: 실험할 종목 리스트
            force_retrain: 강제 재학습 여부
            
        Returns:
            실험 결과 딕셔너리
        """
        print(f"\n🧪 {len(symbols)}개 종목에 대해 {len(self.experiment_configs)}개 모델 구조 실험 시작")
        
        for symbol in symbols:
            print(f"\n{'='*70}")
            print(f"📊 {symbol} 실험 시작")
            print(f"{'='*70}")
            
            self.results[symbol] = {}
            
            # 데이터 로드
            data = self._load_symbol_data(symbol)
            if data is None:
                print(f"⚠️ {symbol} 데이터 로드 실패")
                continue
            
            # 각 실험 구성으로 테스트
            for exp_idx, exp_config in enumerate(self.experiment_configs):
                exp_name = exp_config.get("name", f"experiment_{exp_idx}")
                print(f"\n🔬 실험 {exp_idx+1}: {exp_name}")
                print(f"   - 구조: {exp_config.get('description', 'N/A')}")
                
                # 실험 구성으로 모델 학습 및 평가
                performance = self._run_single_experiment(
                    symbol, data, exp_config, exp_name, force_retrain
                )
                
                self.results[symbol][exp_name] = performance
                
                # 결과 즉시 출력
                if performance['rmse'] < float('inf'):
                    print(f"   ✅ RMSE: {performance['rmse']:.4f}")
                    print(f"   📊 Train/Test 비율: {performance['train_ratio']*100:.0f}%/{(1-performance['train_ratio'])*100:.0f}%")
                    print(f"   🔢 예측 수: {performance['num_predictions']}")
                else:
                    print(f"   ❌ 실험 실패")
        
        # 최적 모델 선택 및 저장
        self._save_best_models()
        
        return self.results
    
    def _load_symbol_data(self, symbol: str) -> Dict:
        """종목 데이터 로드"""
        try:
            # CSV 파일 찾기
            csv_files = list(self.data_dir.glob(f"{symbol}_*.csv"))
            if not csv_files:
                logger.error(f"{symbol} CSV 파일을 찾을 수 없습니다")
                return None
            
            # 가장 최신 파일 사용
            csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            
            # 데이터 로드
            df = pd.read_csv(csv_file)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # 피처와 타겟 분리
            target_columns = [col for col in df.columns if 'target' in col]
            feature_columns = [col for col in df.columns if col not in target_columns]
            
            features = df[feature_columns]
            
            # 타겟이 없으면 22일 수익률 계산
            if 'target_22d' not in df.columns:
                df['target_22d'] = df['close'].pct_change(22).shift(-22)
            
            target = df[['target_22d']]
            
            # NaN 제거
            valid_idx = ~target['target_22d'].isna()
            features = features[valid_idx]
            target = target[valid_idx]
            
            return {
                'features': features,
                'target': target
            }
            
        except Exception as e:
            logger.error(f"{symbol} 데이터 로드 실패: {e}")
            return None
    
    def _run_single_experiment(
        self,
        symbol: str,
        data: Dict,
        exp_config: Dict,
        exp_name: str,
        force_retrain: bool
    ) -> Dict:
        """단일 실험 실행"""
        try:
            # 기본 설정에 실험 설정 병합
            config = deepcopy(self.base_config)
            
            # neural_network 섹션 업데이트
            if 'neural_network' in exp_config:
                config['neural_network'].update(exp_config['neural_network'])
            
            # 모델 생성
            model = StockPredictionNetwork(config)
            
            # 학습 데이터 준비
            training_data = {symbol: data}
            
            # 모델 학습
            print(f"   🏋️ 모델 학습 중...")
            success = model.fit(training_data)
            
            if not success:
                return {
                    'rmse': float('inf'),
                    'train_ratio': config['neural_network'].get('train_ratio', 0.8),
                    'num_predictions': 0,
                    'config': exp_config
                }
            
            # Test set에서 성능 평가
            # fit 메서드 내부에서 이미 train/test 분할하고 검증했으므로
            # 여기서는 모델의 validation_results를 가져와야 함
            
            # 하지만 현재 구조상 validation_results가 저장되지 않으므로
            # 간단히 전체 데이터로 재예측하여 RMSE 계산
            train_ratio = config['neural_network'].get('train_ratio', 0.8)
            train_end_idx = int(len(data['features']) * train_ratio)
            
            test_features = data['features'].iloc[train_end_idx:]
            test_target = data['target'].iloc[train_end_idx:]
            
            # 예측 수행
            predictions = []
            actuals = []
            
            for i in range(22, len(test_features)):
                try:
                    current_features = test_features.iloc[:i+1]
                    pred = model.predict(current_features, symbol)
                    
                    if isinstance(pred, dict):
                        pred_value = pred.get('target_22d', 0.0)
                    else:
                        pred_value = float(pred)
                    
                    if i + 22 < len(test_target):
                        actual_value = test_target.iloc[i + 22]['target_22d']
                        predictions.append(pred_value)
                        actuals.append(actual_value)
                
                except Exception as e:
                    continue
            
            # RMSE 계산
            if predictions:
                rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)]))
            else:
                rmse = float('inf')
            
            return {
                'rmse': rmse,
                'train_ratio': train_ratio,
                'num_predictions': len(predictions),
                'config': exp_config,
                'model': model  # 나중에 최적 모델 저장용
            }
            
        except Exception as e:
            logger.error(f"{symbol} {exp_name} 실험 실패: {e}")
            return {
                'rmse': float('inf'),
                'train_ratio': 0.8,
                'num_predictions': 0,
                'config': exp_config
            }
    
    def _save_best_models(self):
        """각 종목별 최적 모델 설정 저장"""
        best_configs = {}
        
        print("\n" + "="*70)
        print("🏆 종목별 최적 모델")
        print("="*70)
        
        for symbol, experiments in self.results.items():
            # RMSE가 가장 낮은 모델 찾기
            best_exp = min(experiments.items(), key=lambda x: x[1]['rmse'])
            best_name, best_performance = best_exp
            
            if best_performance['rmse'] < float('inf'):
                best_configs[symbol] = {
                    'experiment_name': best_name,
                    'rmse': best_performance['rmse'],
                    'config': best_performance['config']
                }
                
                print(f"\n{symbol}:")
                print(f"   - 최적 모델: {best_name}")
                print(f"   - RMSE: {best_performance['rmse']:.4f}")
                print(f"   - 예측 수: {best_performance['num_predictions']}")
        
        # 결과 저장
        output_file = self.model_dir / "best_neural_configs.json"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(best_configs, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 최적 모델 설정 저장: {output_file}")


def run_neural_experiments(
    config_path: str,
    experiment_config_path: str,
    data_dir: str,
    model_dir: str,
    force_retrain: bool = False
) -> Dict:
    """
    신경망 실험 실행 메인 함수
    
    Args:
        config_path: 기본 설정 파일 경로
        experiment_config_path: 실험 설정 파일 경로
        data_dir: 데이터 디렉토리
        model_dir: 모델 저장 디렉토리
        force_retrain: 강제 재학습 여부
        
    Returns:
        실험 결과
    """
    # 기본 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    # 실험 설정 로드 (파일이 없으면 기본값 사용)
    if os.path.exists(experiment_config_path):
        with open(experiment_config_path, 'r', encoding='utf-8') as f:
            experiment_data = json.load(f)
    else:
        print(f"⚠️ 실험 설정 파일이 없습니다. 기본 실험 구성을 사용합니다.")
        experiment_data = {
            "experiments": [
                {
                    "name": "baseline",
                    "description": "기본 모델 (3층, 128-64-32)",
                    "neural_network": {
                        "hidden_sizes": [128, 64, 32],
                        "dropout_rate": 0.2
                    }
                },
                {
                    "name": "deep",
                    "description": "깊은 모델 (5층, 256-128-64-32-16)",
                    "neural_network": {
                        "hidden_sizes": [256, 128, 64, 32, 16],
                        "dropout_rate": 0.3
                    }
                },
                {
                    "name": "wide",
                    "description": "넓은 모델 (3층, 512-256-128)",
                    "neural_network": {
                        "hidden_sizes": [512, 256, 128],
                        "dropout_rate": 0.2
                    }
                },
                {
                    "name": "shallow",
                    "description": "얕은 모델 (2층, 64-32)",
                    "neural_network": {
                        "hidden_sizes": [64, 32],
                        "dropout_rate": 0.1
                    }
                },
                {
                    "name": "regularized",
                    "description": "강한 정규화 모델 (3층, dropout 0.5)",
                    "neural_network": {
                        "hidden_sizes": [128, 64, 32],
                        "dropout_rate": 0.5
                    }
                }
            ],
            "symbols": base_config.get("data", {}).get("symbols", ["AAPL", "MSFT", "GOOGL"])[:3]
        }
    
    # 실험 객체 생성
    experiment = NeuralExperiment(
        base_config=base_config,
        experiment_configs=experiment_data['experiments'],
        data_dir=data_dir,
        model_dir=model_dir
    )
    
    # 실험 실행
    symbols = experiment_data.get('symbols', base_config.get("data", {}).get("symbols", []))
    results = experiment.run_experiments(symbols, force_retrain)
    
    return results


if __name__ == "__main__":
    # 테스트용
    results = run_neural_experiments(
        config_path="config/config_trader.json",
        experiment_config_path="config/neural_experiments.json",
        data_dir="data/trader",
        model_dir="models/trader",
        force_retrain=True
    )