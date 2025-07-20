#!/usr/bin/env python3
"""
오케스트레이터 - 전체 파이프라인 관리
새로운 2단계 구조: cleaner → scrapper → researcher → evaluator → portfolio_manager
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .cleaner import Cleaner
from .scrapper import DataScrapper
from .researcher import IndividualStrategyResearcher
from .evaluator import TrainTestEvaluator
from .portfolio_manager import AdvancedPortfolioManager
from .helper import (
    load_config,
    print_section_header,
    print_subsection_header,
    DEFAULT_CONFIG_PATH,
)

# 환경 변수 설정 (orchestrator 모드)
os.environ["ORCHESTRATOR_MODE"] = "true"


class Orchestrator:
    """전체 파이프라인 오케스트레이터"""

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        time_horizon: str = "swing",
        uuid: Optional[str] = None,
    ):
        self.config_path = config_path
        self.time_horizon = time_horizon
        self.uuid = uuid or datetime.now().strftime("%Y%m%d_%H%M%S")

        # 설정 로드 - 절대 경로로 변환
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.getcwd(), config_path)
        self.config = load_config(config_path)

        # 실행 시간 기록
        self.start_time = datetime.now()

        # 각 단계별 결과 저장
        self.results = {}

        print_section_header("🚀 오케스트레이터 초기화")
        print(f"📁 설정 파일: {config_path}")
        print(f"⏰ 시간대: {time_horizon}")
        print(f"🆔 실행 UUID: {self.uuid}")

    def _get_config_for_horizon(self) -> Dict[str, Any]:
        """시간대별 설정 가져오기"""
        # 시간대별 config 파일 경로
        horizon_config_path = f"config/config_{self.time_horizon}.json"

        if os.path.exists(horizon_config_path):
            print(f"✅ 시간대별 설정 파일 사용: {horizon_config_path}")
            return load_config(horizon_config_path)
        else:
            print(
                f"⚠️ 시간대별 설정 파일이 없습니다. 기본 설정 사용: {self.config_path}"
            )
            return self.config

    def run_cleaner(self) -> bool:
        """1단계: 데이터 정리"""
        print_subsection_header("🧹 1단계: 데이터 정리")

        try:
            horizon_config = self._get_config_for_horizon()

            cleaner = Cleaner()

            # 설정에서 cleaner 액션 확인
            cleaner_config = horizon_config.get("cleaner", {})
            action = cleaner_config.get("action", "clean-and-recreate")
            folders = cleaner_config.get(
                "folders",
                [
                    f"data/{self.time_horizon}",
                    f"log/{self.time_horizon}",
                    f"results/{self.time_horizon}",
                ],
            )

            if action == "clean-and-recreate":
                success = cleaner.clean_and_recreate_folders(folders)
            elif action == "clean-only":
                success = cleaner.clean_folders(folders)
            elif action == "create-only":
                success = cleaner.create_folders(folders)
            else:
                # 기본값: clean-and-recreate
                success = cleaner.clean_and_recreate_folders(folders)

            if success:
                print("✅ 데이터 정리 완료")
                self.results["cleaner"] = {
                    "status": "success",
                    "timestamp": datetime.now(),
                }
            else:
                print("❌ 데이터 정리 실패")
                self.results["cleaner"] = {
                    "status": "failed",
                    "timestamp": datetime.now(),
                }

            return success

        except Exception as e:
            print(f"❌ 데이터 정리 중 오류: {e}")
            self.results["cleaner"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(),
            }
            return False

    def run_scrapper(self) -> bool:
        """2단계: 데이터 수집"""
        print_subsection_header("📊 2단계: 데이터 수집")

        try:
            horizon_config = self._get_config_for_horizon()

            # 시간대별 설정 파일 경로 사용
            horizon_config_path = f"config/config_{self.time_horizon}.json"
            scrapper = DataScrapper(
                config_path=horizon_config_path,
                time_horizon=self.time_horizon,
                uuid=self.uuid,
            )

            success = scrapper.run_scrapper()

            if success:
                print("✅ 데이터 수집 완료")
                self.results["scrapper"] = {
                    "status": "success",
                    "timestamp": datetime.now(),
                }
            else:
                print("❌ 데이터 수집 실패")
                self.results["scrapper"] = {
                    "status": "failed",
                    "timestamp": datetime.now(),
                }

            return success

        except Exception as e:
            print(f"❌ 데이터 수집 중 오류: {e}")
            self.results["scrapper"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(),
            }
            return False

    def run_researcher(self) -> bool:
        """3단계: 개별 종목별 전략 최적화"""
        print_subsection_header("🔬 3단계: 개별 종목별 전략 최적화")

        try:
            horizon_config = self._get_config_for_horizon()

            # 데이터 디렉토리 설정
            data_dir = f"data/{self.time_horizon}"

            # 시간대별 설정 파일 경로 사용
            horizon_config_path = f"config/config_{self.time_horizon}.json"

            researcher = IndividualStrategyResearcher(
                research_config_path="config/config_research.json",
                source_config_path=horizon_config_path,
                data_dir=data_dir,
                results_dir="results",
                log_dir="log",
                analysis_dir="analysis",
                auto_detect_source_config=False,  # 명시적으로 설정된 config 사용
                uuid=self.uuid,  # UUID 전달
            )

            # UUID 설정 - logger를 통해 설정
            if self.uuid:
                researcher.logger.setup_logger(
                    strategy="individual_research", mode="research", uuid=self.uuid
                )

            # 종합 연구 실행
            results = researcher.run_comprehensive_research(
                optimization_method="bayesian_optimization"
            )

            if results:
                print(f"✅ 개별 전략 최적화 완료: {len(results)}개 조합")

                # 결과 저장
                output_file = researcher.save_research_results(results)
                if output_file:
                    print(f"💾 최적화 결과 저장됨: {output_file}")

                # 연구 보고서 생성
                researcher.generate_research_report(results)

                self.results["researcher"] = {
                    "status": "success",
                    "combinations": len(results),
                    "timestamp": datetime.now(),
                }

                # 최적화 결과 파일 경로 저장 (evaluator에서 사용)
                self.results["researcher"]["output_file"] = output_file

                return True
            else:
                print("❌ 개별 전략 최적화 실패")
                self.results["researcher"] = {
                    "status": "failed",
                    "timestamp": datetime.now(),
                }
                return False

        except Exception as e:
            print(f"❌ 개별 전략 최적화 중 오류: {e}")
            self.results["researcher"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(),
            }
            return False

    def run_evaluator(self) -> bool:
        """4단계: 2단계 평가 (개별 + 포트폴리오)"""
        print_subsection_header("📊 4단계: 2단계 전략 평가")

        try:
            horizon_config = self._get_config_for_horizon()

            # 데이터 디렉토리 설정
            data_dir = f"data/{self.time_horizon}"

            # 최적화 결과 파일 경로
            optimization_file = self._find_latest_optimization_file()

            if not optimization_file:
                print("❌ 최적화 결과 파일을 찾을 수 없습니다.")
                return False

            # 시간대별 config 파일 경로 사용
            horizon_config_path = f"config/config_{self.time_horizon}.json"

            evaluator = TrainTestEvaluator(
                data_dir=data_dir,
                log_mode="summary",
                config_path=horizon_config_path,
                optimization_results_path=optimization_file,
            )

            # UUID 설정 (타입 힌트 문제로 주석 처리)
            # if self.uuid and hasattr(evaluator, "execution_uuid"):
            #     evaluator.execution_uuid = self.uuid

            # Train/Test 평가 실행
            results = evaluator.run_train_test_evaluation(save_results=True)

            if results:
                print("✅ 2단계 평가 완료")
                print(f"  📊 개별 종목 평가: {len(results['individual_results'])}개")
                print(
                    f"  🎯 포트폴리오 평가: {len(results['portfolio_results'])}개 전략"
                )

                self.results["evaluator"] = {
                    "status": "success",
                    "individual_symbols": len(results["individual_results"]),
                    "portfolio_methods": len(results["portfolio_results"]),
                    "timestamp": datetime.now(),
                }

                return True
            else:
                print("❌ 2단계 평가 실패")
                self.results["evaluator"] = {
                    "status": "failed",
                    "timestamp": datetime.now(),
                }
                return False

        except Exception as e:
            print(f"❌ 2단계 평가 중 오류: {e}")
            self.results["evaluator"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(),
            }
            return False

    def run_portfolio_manager(self) -> bool:
        """5단계: 포트폴리오 최적화"""
        print_subsection_header("⚖️ 5단계: 포트폴리오 최적화")

        try:
            horizon_config = self._get_config_for_horizon()

            # 포트폴리오 매니저 초기화
            portfolio_manager = AdvancedPortfolioManager(
                config_path=self.config_path,
                time_horizon=self.time_horizon,
                uuid=self.uuid,
            )

            # 포트폴리오 최적화 실행
            success = portfolio_manager.run_portfolio_optimization()

            if success:
                print("✅ 포트폴리오 최적화 완료")
                self.results["portfolio_manager"] = {
                    "status": "success",
                    "timestamp": datetime.now(),
                }
            else:
                print("❌ 포트폴리오 최적화 실패")
                self.results["portfolio_manager"] = {
                    "status": "failed",
                    "timestamp": datetime.now(),
                }

            return success

        except Exception as e:
            print(f"❌ 포트폴리오 최적화 중 오류: {e}")
            self.results["portfolio_manager"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(),
            }
            return False

    def _find_latest_optimization_file(self) -> Optional[str]:
        """최신 최적화 결과 파일 찾기"""
        try:
            # results 디렉토리에서 최적화 결과 파일 찾기
            results_dir = Path("results")
            if not results_dir.exists():
                return None

            # hyperparam_optimization_*.json 파일들 찾기 (researcher가 생성하는 파일명)
            optimization_files = list(
                results_dir.glob("hyperparam_optimization_*.json")
            )

            if not optimization_files:
                print("⚠️ hyperparam_optimization_*.json 파일을 찾을 수 없습니다")
                print(f"🔍 results 디렉토리 내용:")
                for file in results_dir.glob("*.json"):
                    print(f"  - {file.name}")
                return None

            # 가장 최신 파일 반환
            latest_file = max(optimization_files, key=lambda x: x.stat().st_mtime)
            print(f"✅ 최신 최적화 결과 파일 발견: {latest_file.name}")
            return str(latest_file)

        except Exception as e:
            print(f"⚠️ 최적화 결과 파일 찾기 실패: {e}")
            return None

    def run_pipeline(self, stages: Optional[list] = None) -> bool:
        """전체 파이프라인 실행"""
        print_section_header("🚀 전체 파이프라인 실행 시작")

        # 기본 단계 순서 (수정됨)
        if not stages:
            stages = [
                "cleaner",
                "scrapper",
                "researcher",
                "portfolio_manager",
                "evaluator",
            ]

        print(f"📋 실행 단계: {' → '.join(stages)}")
        print(f"⏰ 시작 시간: {self.start_time}")

        success_count = 0
        total_stages = len(stages)

        for i, stage in enumerate(stages, 1):
            print(f"\n🔄 단계 {i}/{total_stages}: {stage}")

            try:
                if stage == "cleaner":
                    success = self.run_cleaner()
                elif stage == "scrapper":
                    success = self.run_scrapper()
                elif stage == "researcher":
                    success = self.run_researcher()
                elif stage == "portfolio_manager":
                    success = self.run_portfolio_manager()
                elif stage == "evaluator":
                    success = self.run_evaluator()
                else:
                    print(f"❌ 알 수 없는 단계: {stage}")
                    success = False

                if success:
                    success_count += 1
                    print(f"✅ {stage} 단계 완료")
                else:
                    print(f"❌ {stage} 단계 실패")

                    # 설정에 따라 오류 시 중단
                    if self.config.get("flow", {}).get("stop_on_error", True):
                        print("⚠️ 오류 발생으로 파이프라인 중단")
                        break

            except Exception as e:
                print(f"❌ {stage} 단계 실행 중 예외 발생: {e}")
                success_count += 1  # 예외는 이미 로깅됨

                if self.config.get("flow", {}).get("stop_on_error", True):
                    print("⚠️ 예외 발생으로 파이프라인 중단")
                    break

        # 최종 요약
        self._generate_final_summary(success_count, total_stages)

        # 결과 저장
        self._save_pipeline_results()

        return success_count == total_stages

    def _generate_final_summary(self, success_count: int, total_stages: int):
        """최종 요약 생성"""
        print_section_header("📊 파이프라인 실행 완료")

        end_time = datetime.now()
        execution_time = end_time - self.start_time

        print(f"⏱️ 총 실행 시간: {execution_time}")
        print(f"✅ 성공한 단계: {success_count}/{total_stages}")
        print(f"📈 성공률: {success_count/total_stages*100:.1f}%")

        # 각 단계별 결과 요약
        print("\n📋 단계별 결과:")
        for stage, result in self.results.items():
            status = result.get("status", "unknown")
            timestamp = result.get("timestamp", "N/A")

            if status == "success":
                print(f"  ✅ {stage}: 성공 ({timestamp})")
            elif status == "failed":
                print(f"  ❌ {stage}: 실패 ({timestamp})")
            elif status == "error":
                error = result.get("error", "Unknown error")
                print(f"  💥 {stage}: 오류 - {error} ({timestamp})")

        if success_count == total_stages:
            print("\n🎉 모든 단계가 성공적으로 완료되었습니다!")
        else:
            print(f"\n⚠️ {total_stages - success_count}개 단계에서 문제가 발생했습니다.")

    def _save_pipeline_results(self):
        """파이프라인 결과 저장"""
        try:
            # UUID가 있으면 사용, 없으면 현재 시간 사용
            if self.uuid:
                timestamp = self.uuid
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_results_{timestamp}.json"
            output_path = os.path.join("results", filename)

            # 결과를 JSON 직렬화 가능한 형태로 변환
            serializable_results = {}
            for stage, result in self.results.items():
                serializable_results[stage] = {
                    "status": result.get("status"),
                    "timestamp": (
                        result.get("timestamp").isoformat()
                        if result.get("timestamp")
                        else None
                    ),
                    "error": result.get("error"),
                }

            pipeline_summary = {
                "uuid": self.uuid,
                "time_horizon": self.time_horizon,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_stages": len(self.results),
                "successful_stages": sum(
                    1 for r in self.results.values() if r.get("status") == "success"
                ),
                "results": serializable_results,
            }

            # 디렉토리 생성
            os.makedirs("results", exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)

            print(f"💾 파이프라인 결과 저장: {output_path}")

        except Exception as e:
            print(f"⚠️ 파이프라인 결과 저장 실패: {e}")

    def run_single_stage(self, stage: str) -> bool:
        """단일 단계 실행"""
        print_section_header(f"🔄 단일 단계 실행: {stage}")

        try:
            if stage == "cleaner":
                return self.run_cleaner()
            elif stage == "scrapper":
                return self.run_scrapper()
            elif stage == "researcher":
                return self.run_researcher()
            elif stage == "portfolio_manager":
                return self.run_portfolio_manager()
            elif stage == "evaluator":
                return self.run_evaluator()
            else:
                print(f"❌ 알 수 없는 단계: {stage}")
                return False

        except Exception as e:
            print(f"❌ {stage} 단계 실행 중 예외 발생: {e}")
            return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="퀀트 트레이딩 파이프라인 오케스트레이터"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="설정 파일 경로")
    parser.add_argument(
        "--time-horizon",
        default="swing",
        choices=["scalping", "swing", "long"],
        help="시간대 설정",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["cleaner", "scrapper", "researcher", "evaluator", "portfolio_manager"],
        help="실행할 단계들 (지정하지 않으면 모든 단계 실행)",
    )
    parser.add_argument(
        "--single-stage",
        choices=["cleaner", "scrapper", "researcher", "evaluator", "portfolio_manager"],
        help="단일 단계만 실행",
    )
    parser.add_argument("--uuid", help="실행 UUID")

    args = parser.parse_args()

    # 오케스트레이터 초기화
    orchestrator = Orchestrator(
        config_path=args.config,
        time_horizon=args.time_horizon,
        uuid=args.uuid,
    )

    # 실행
    if args.single_stage:
        # 단일 단계 실행
        success = orchestrator.run_single_stage(args.single_stage)
        if success:
            print(f"✅ {args.single_stage} 단계 완료")
        else:
            print(f"❌ {args.single_stage} 단계 실패")
    else:
        # 전체 파이프라인 또는 지정된 단계들 실행
        success = orchestrator.run_pipeline(args.stages)
        if success:
            print("🎉 파이프라인 실행 완료")
        else:
            print("⚠️ 파이프라인 실행 중 문제 발생")


if __name__ == "__main__":
    main()
