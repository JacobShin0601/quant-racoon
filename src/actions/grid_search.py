#!/usr/bin/env python3
"""
하이퍼파라미터 최적화 모듈
그리드 서치, 베이지안 최적화, 유전 알고리즘 등 다양한 최적화 방법 제공
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import itertools
import random
import time
import json
import os
from datetime import datetime
import warnings
from dataclasses import dataclass
import logging

warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """최적화 결과를 저장하는 데이터클래스"""

    strategy_name: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_method: str
    execution_time: float
    n_combinations_tested: int
    symbol: str = None


class HyperparameterOptimizer:
    """하이퍼파라미터 최적화 클래스"""

    def __init__(self, config_path: str = "config/config_research.json"):
        self.config = self._load_config(config_path)
        self.results = {}
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"설정 파일 형식이 잘못되었습니다: {config_path}")
            return {}

    def generate_parameter_combinations(
        self,
        param_ranges: Dict[str, List],
        max_combinations: int = 50,
        random_sampling: bool = True,
        sampling_ratio: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """파라미터 조합 생성"""

        # 모든 가능한 조합 생성
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        all_combinations = list(itertools.product(*param_values))
        total_combinations = len(all_combinations)

        self.logger.info(f"총 가능한 조합 수: {total_combinations}")

        if random_sampling and total_combinations > max_combinations:
            # 랜덤 샘플링
            sample_size = min(
                int(total_combinations * sampling_ratio), max_combinations
            )
            selected_combinations = random.sample(all_combinations, sample_size)
            self.logger.info(f"랜덤 샘플링으로 {sample_size}개 조합 선택")
        else:
            # 전체 조합 또는 최대 조합 수만큼
            selected_combinations = all_combinations[:max_combinations]
            self.logger.info(f"전체 조합 중 {len(selected_combinations)}개 사용")

        # 딕셔너리 형태로 변환
        combinations = []
        for combination in selected_combinations:
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)

        return combinations

    def grid_search(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List],
        evaluation_function: Callable,
        max_combinations: int = 50,
        random_sampling: bool = True,
        sampling_ratio: float = 0.3,
        timeout_per_combination: int = 300,
    ) -> OptimizationResult:
        """그리드 서치 최적화"""

        self.logger.info(f"🔍 {strategy_name} 그리드 서치 시작")
        start_time = time.time()

        # 파라미터 조합 생성
        combinations = self.generate_parameter_combinations(
            param_ranges, max_combinations, random_sampling, sampling_ratio
        )

        best_score = float("-inf")
        best_params = None
        all_results = []

        for i, params in enumerate(combinations):
            try:
                self.logger.info(f"  테스트 {i+1}/{len(combinations)}: {params}")

                # 타임아웃 설정
                start_eval = time.time()
                score = evaluation_function(params)
                eval_time = time.time() - start_eval

                result = {
                    "params": params,
                    "score": score,
                    "evaluation_time": eval_time,
                    "combination_index": i,
                }
                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = params
                    self.logger.info(f"    🎯 새로운 최고 점수: {score:.4f}")

            except Exception as e:
                self.logger.warning(
                    f"    ⚠️ 파라미터 조합 {params} 평가 중 오류: {str(e)}"
                )
                continue

        execution_time = time.time() - start_time

        result = OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_method="grid_search",
            execution_time=execution_time,
            n_combinations_tested=len(all_results),
        )

        self.logger.info(f"✅ {strategy_name} 그리드 서치 완료")
        self.logger.info(f"   최고 점수: {best_score:.4f}")
        self.logger.info(f"   최적 파라미터: {best_params}")
        self.logger.info(f"   실행 시간: {execution_time:.2f}초")

        return result

    def bayesian_optimization(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List],
        evaluation_function: Callable,
        n_trials: int = 100,
        n_startup_trials: int = 10,
    ) -> OptimizationResult:
        """베이지안 최적화 (Optuna 사용)"""

        try:
            import optuna
        except ImportError:
            self.logger.error(
                "Optuna가 설치되지 않았습니다. pip install optuna로 설치해주세요."
            )
            return None

        self.logger.info(f"🔍 {strategy_name} 베이지안 최적화 시작")
        start_time = time.time()

        def objective(trial):
            # 파라미터 샘플링
            params = {}
            for param_name, param_values in param_ranges.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )

            try:
                return evaluation_function(params)
            except Exception as e:
                self.logger.warning(f"베이지안 최적화 중 오류: {str(e)}")
                return float("-inf")

        # Optuna 스터디 생성
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_startup_trials=n_startup_trials)

        execution_time = time.time() - start_time

        # 결과 변환
        all_results = []
        for trial in study.trials:
            if trial.value is not None:
                result = {
                    "params": trial.params,
                    "score": trial.value,
                    "evaluation_time": trial.duration,
                    "combination_index": trial.number,
                }
                all_results.append(result)

        result = OptimizationResult(
            strategy_name=strategy_name,
            best_params=study.best_params,
            best_score=study.best_value,
            all_results=all_results,
            optimization_method="bayesian_optimization",
            execution_time=execution_time,
            n_combinations_tested=len(all_results),
        )

        self.logger.info(f"✅ {strategy_name} 베이지안 최적화 완료")
        self.logger.info(f"   최고 점수: {study.best_value:.4f}")
        self.logger.info(f"   최적 파라미터: {study.best_params}")
        self.logger.info(f"   실행 시간: {execution_time:.2f}초")

        return result

    def genetic_algorithm(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List],
        evaluation_function: Callable,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ) -> OptimizationResult:
        """유전 알고리즘 최적화"""

        self.logger.info(f"🔍 {strategy_name} 유전 알고리즘 시작")
        start_time = time.time()

        # 초기 개체군 생성
        population = self._generate_initial_population(param_ranges, population_size)
        best_score = float("-inf")
        best_params = None
        all_results = []

        for generation in range(generations):
            self.logger.info(f"  세대 {generation + 1}/{generations}")

            # 적합도 평가
            fitness_scores = []
            for individual in population:
                try:
                    score = evaluation_function(individual)
                    fitness_scores.append(score)

                    result = {
                        "params": individual,
                        "score": score,
                        "generation": generation,
                        "combination_index": len(all_results),
                    }
                    all_results.append(result)

                    if score > best_score:
                        best_score = score
                        best_params = individual
                        self.logger.info(f"    🎯 새로운 최고 점수: {score:.4f}")

                except Exception as e:
                    fitness_scores.append(float("-inf"))
                    self.logger.warning(f"    ⚠️ 개체 평가 중 오류: {str(e)}")

            # 선택, 교차, 돌연변이
            if generation < generations - 1:  # 마지막 세대가 아니면
                population = self._evolve_population(
                    population,
                    fitness_scores,
                    param_ranges,
                    mutation_rate,
                    crossover_rate,
                )

        execution_time = time.time() - start_time

        result = OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_method="genetic_algorithm",
            execution_time=execution_time,
            n_combinations_tested=len(all_results),
        )

        self.logger.info(f"✅ {strategy_name} 유전 알고리즘 완료")
        self.logger.info(f"   최고 점수: {best_score:.4f}")
        self.logger.info(f"   최적 파라미터: {best_params}")
        self.logger.info(f"   실행 시간: {execution_time:.2f}초")

        return result

    def _generate_initial_population(
        self, param_ranges: Dict[str, List], population_size: int
    ) -> List[Dict[str, Any]]:
        """초기 개체군 생성"""
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, param_values in param_ranges.items():
                individual[param_name] = random.choice(param_values)
            population.append(individual)
        return population

    def _evolve_population(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        param_ranges: Dict[str, List],
        mutation_rate: float,
        crossover_rate: float,
    ) -> List[Dict[str, Any]]:
        """개체군 진화"""
        new_population = []

        # 엘리트 선택 (상위 10% 보존)
        elite_size = max(1, len(population) // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx])

        # 나머지 개체들은 선택, 교차, 돌연변이로 생성
        while len(new_population) < len(population):
            # 토너먼트 선택
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # 교차
            if random.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # 돌연변이
            if random.random() < mutation_rate:
                child = self._mutate(child, param_ranges)

            new_population.append(child)

        return new_population

    def _tournament_selection(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> Dict[str, Any]:
        """토너먼트 선택"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """교차 연산"""
        child = {}
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child

    def _mutate(
        self, individual: Dict[str, Any], param_ranges: Dict[str, List]
    ) -> Dict[str, Any]:
        """돌연변이 연산"""
        mutated = individual.copy()
        param_name = random.choice(list(param_ranges.keys()))
        mutated[param_name] = random.choice(param_ranges[param_name])
        return mutated

    def optimize_strategy(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List],
        evaluation_function: Callable,
        optimization_method: str = "grid_search",
        **kwargs,
    ) -> OptimizationResult:
        """전략 최적화 실행"""

        if optimization_method == "grid_search":
            return self.grid_search(
                strategy_name, param_ranges, evaluation_function, **kwargs
            )
        elif optimization_method == "bayesian_optimization":
            return self.bayesian_optimization(
                strategy_name, param_ranges, evaluation_function, **kwargs
            )
        elif optimization_method == "genetic_algorithm":
            return self.genetic_algorithm(
                strategy_name, param_ranges, evaluation_function, **kwargs
            )
        else:
            raise ValueError(f"지원하지 않는 최적화 방법: {optimization_method}")

    def save_results(
        self, results: List[OptimizationResult], output_dir: str = "results"
    ):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 형태로 저장
        results_dict = []
        for result in results:
            result_dict = {
                "strategy_name": result.strategy_name,
                "best_params": result.best_params,
                "best_score": result.best_score,
                "optimization_method": result.optimization_method,
                "execution_time": result.execution_time,
                "n_combinations_tested": result.n_combinations_tested,
                "symbol": result.symbol,
                "all_results": result.all_results,
            }
            results_dict.append(result_dict)

        json_path = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        # CSV 형태로 저장 (요약)
        summary_data = []
        for result in results:
            summary_data.append(
                {
                    "strategy_name": result.strategy_name,
                    "best_score": result.best_score,
                    "optimization_method": result.optimization_method,
                    "execution_time": result.execution_time,
                    "n_combinations_tested": result.n_combinations_tested,
                    "symbol": result.symbol,
                    "best_params": str(result.best_params),
                }
            )

        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, f"optimization_summary_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")

        self.logger.info(f"결과 저장 완료:")
        self.logger.info(f"  JSON: {json_path}")
        self.logger.info(f"  CSV: {csv_path}")

        return json_path, csv_path

    def generate_optimization_report(self, results: List[OptimizationResult]) -> str:
        """최적화 결과 리포트 생성"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🔬 하이퍼파라미터 최적화 결과 리포트")
        report_lines.append("=" * 80)

        # 전체 요약
        total_time = sum(r.execution_time for r in results)
        total_combinations = sum(r.n_combinations_tested for r in results)

        report_lines.append(f"\n📊 전체 요약:")
        report_lines.append(f"  총 전략 수: {len(results)}")
        report_lines.append(f"  총 실행 시간: {total_time:.2f}초")
        report_lines.append(f"  총 테스트 조합 수: {total_combinations}")

        # 전략별 결과
        report_lines.append(f"\n🏆 전략별 최적화 결과:")
        report_lines.append("-" * 80)
        report_lines.append(
            f"{'전략명':<25} {'최적화방법':<20} {'최고점수':<12} {'실행시간':<10} {'테스트수':<8}"
        )
        report_lines.append("-" * 80)

        # 점수별 정렬
        sorted_results = sorted(results, key=lambda x: x.best_score, reverse=True)

        for result in sorted_results:
            report_lines.append(
                f"{result.strategy_name:<25} {result.optimization_method:<20} "
                f"{result.best_score:<12.4f} {result.execution_time:<10.2f} "
                f"{result.n_combinations_tested:<8}"
            )

        # 최고 성과 전략 상세
        if sorted_results:
            best_result = sorted_results[0]
            report_lines.append(f"\n🥇 최고 성과 전략: {best_result.strategy_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"최적화 방법: {best_result.optimization_method}")
            report_lines.append(f"최고 점수: {best_result.best_score:.4f}")
            report_lines.append(f"실행 시간: {best_result.execution_time:.2f}초")
            report_lines.append(f"테스트 조합 수: {best_result.n_combinations_tested}")
            report_lines.append(f"최적 파라미터:")
            for param, value in best_result.best_params.items():
                report_lines.append(f"  {param}: {value}")

        # 최적화 방법별 통계
        method_stats = {}
        for result in results:
            method = result.optimization_method
            if method not in method_stats:
                method_stats[method] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_score": 0,
                    "scores": [],
                }
            method_stats[method]["count"] += 1
            method_stats[method]["total_time"] += result.execution_time
            method_stats[method]["scores"].append(result.best_score)

        report_lines.append(f"\n📈 최적화 방법별 통계:")
        report_lines.append("-" * 50)
        for method, stats in method_stats.items():
            avg_score = np.mean(stats["scores"])
            avg_time = stats["total_time"] / stats["count"]
            report_lines.append(f"{method}:")
            report_lines.append(f"  전략 수: {stats['count']}")
            report_lines.append(f"  평균 점수: {avg_score:.4f}")
            report_lines.append(f"  평균 실행 시간: {avg_time:.2f}초")

        return "\n".join(report_lines)


def main():
    """테스트용 메인 함수"""

    # 테스트용 평가 함수
    def test_evaluation_function(params):
        # 간단한 테스트 함수 (실제로는 전략 평가 함수가 들어감)
        score = 0
        for param_name, value in params.items():
            if "period" in param_name:
                score += value * 0.1
            elif "threshold" in param_name:
                score += value * 0.2
        return score + random.random() * 0.1  # 약간의 랜덤성 추가

    # 최적화기 초기화
    optimizer = HyperparameterOptimizer()

    # 테스트 파라미터 범위
    test_param_ranges = {
        "period": [10, 20, 30],
        "threshold": [0.1, 0.2, 0.3],
        "multiplier": [1.0, 1.5, 2.0],
    }

    # 그리드 서치 테스트
    result = optimizer.grid_search(
        "test_strategy",
        test_param_ranges,
        test_evaluation_function,
        max_combinations=10,
    )

    print("테스트 완료!")


if __name__ == "__main__":
    main()
