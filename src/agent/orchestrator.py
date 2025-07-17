#!/usr/bin/env python3
"""
flow 기반 퀀트 분석 파이프라인 오케스트레이터
"""
import os
import sys
import subprocess
import shutil
import json
import uuid
from typing import Dict
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로를 PYTHONPATH에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.helper import Logger, load_config

class FlowOrchestrator:
    """전체 퀀트 분석 파이프라인을 관리하는 클래스"""
    def __init__(self, config_path: str = "../../config/config_default.json", time_horizon: str = None):
        self.config_path = config_path  # config_path를 인스턴스 변수로 저장
        self.config = load_config(config_path)
        self.time_horizon = time_horizon
        self.logger = Logger()
        self.logger.setup_logger(strategy="flow_orchestrator", mode="orchestrator")
        self.execution_results = {}
        
        # 실행 UUID 생성 (한 번의 실행에서 모든 파일이 동일한 UUID 사용)
        self.execution_uuid = str(uuid.uuid4())[:8]  # 8자리로 축약
        self.logger.log_info(f"🆔 실행 UUID 생성: {self.execution_uuid}")
        
        # time-horizon 기반 config 선택
        if time_horizon:
            self._select_config_by_time_horizon(time_horizon)
        
        # 폴더 구조 설정
        self._setup_folders()
        
        # 로깅 설정
        self._setup_logging()

    def _select_config_by_time_horizon(self, time_horizon: str):
        """time-horizon에 따라 적절한 config 파일 선택"""
        config_mapping = {
            "swing": "config/config_swing.json",
            "long": "config/config_long.json", 
            "long-term": "config/config_long.json",
            "scalping": "config/config_scalping.json",
            "short-term": "config/config_scalping.json"
        }
        
        if time_horizon in config_mapping:
            config_file = config_mapping[time_horizon]
            config_path = Path(__file__).parent.parent.parent / config_file
            if config_path.exists():
                self.config = load_config(str(config_path))
                self.logger.log_info(f"✅ {time_horizon} 전략용 config 로드: {config_file}")
            else:
                self.logger.log_warning(f"⚠️ {config_file} 파일을 찾을 수 없습니다. 기본 config 사용")
        else:
            self.logger.log_warning(f"⚠️ 알 수 없는 time-horizon: {time_horizon}. 기본 config 사용")

    def _setup_folders(self):
        """출력 폴더 구조 설정 (cleaner.py를 통해 생성)"""
        output_config = self.config.get("output", {})
        self.results_folder = output_config.get("results_folder", "results")
        self.logs_folder = output_config.get("logs_folder", "logs")
        self.backup_folder = output_config.get("backup_folder", "backup")

        # cleaner.py의 create 기능을 subprocess로 호출
        cmd = [
            sys.executable, "-m", "agent.cleaner",
            "--action", "create",
            "--data-dir", "data",
            "--log-dir", self.logs_folder,
            "--results-dir", self.results_folder
        ]
        self.logger.log_info(f"폴더 생성 명령어: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            self.logger.log_success("✅ cleaner.py를 통한 폴더 생성 완료")
            self.logger.log_info(result.stdout)
        else:
            self.logger.log_error(f"❌ cleaner.py 폴더 생성 실패: {result.stderr}")
            raise RuntimeError("폴더 생성 실패")

        # backup 폴더는 cleaner가 관리하지 않으므로 직접 생성 (상위 폴더까지)
        Path(self.backup_folder).mkdir(parents=True, exist_ok=True)
        self.logger.log_info(f"📁 폴더 확인/생성: {self.backup_folder}")

    def _setup_logging(self):
        """로깅 설정"""
        logging_config = self.config.get("logging", {})
        log_level = logging_config.get("level", "INFO")
        file_rotation = logging_config.get("file_rotation", True)
        
        # 로거 설정 업데이트
        self.logger.setup_logger(
            strategy="flow_orchestrator", 
            mode="orchestrator"
        )

    def _get_current_config_name(self) -> str:
        """현재 사용 중인 config 파일명 반환"""
        if self.time_horizon:
            config_mapping = {
                "swing": "config_swing.json",
                "long": "config_long.json", 
                "long-term": "config_long.json",
                "scalping": "config_scalping.json",
                "short-term": "config_scalping.json"
            }
            return config_mapping.get(self.time_horizon, "config_default.json")
        else:
            # 기본 config 파일명 추출
            config_path = getattr(self, 'config_path', 'config_default.json')
            return os.path.basename(config_path)

    def _update_research_source_config(self, source_config_name: str):
        """research config의 source_config를 동적으로 업데이트"""
        try:
            research_config_path = "config/config_research.json"
            
            # research config 로드
            with open(research_config_path, 'r', encoding='utf-8') as f:
                research_config = json.load(f)
            
            # source_config 업데이트
            research_config["research_config"]["source_config"] = source_config_name
            
            # 백업 생성
            backup_path = f"{research_config_path}.backup"
            shutil.copy2(research_config_path, backup_path)
            
            # 업데이트된 config 저장
            with open(research_config_path, 'w', encoding='utf-8') as f:
                json.dump(research_config, f, indent=2, ensure_ascii=False)
            
            self.logger.log_info(f"📝 Research config 업데이트: source_config = {source_config_name}")
            
        except Exception as e:
            self.logger.log_error(f"❌ Research config 업데이트 실패: {e}")

    def _backup_results(self):
        """결과 백업"""
        automation_config = self.config.get("automation", {})
        if not automation_config.get("auto_backup", False):
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(self.backup_folder) / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            # results 폴더 백업
            if Path(self.results_folder).exists():
                shutil.copytree(self.results_folder, backup_path / "results", dirs_exist_ok=True)
            
            # logs 폴더 백업
            if Path(self.logs_folder).exists():
                shutil.copytree(self.logs_folder, backup_path / "logs", dirs_exist_ok=True)
                
            self.logger.log_success(f"✅ 백업 완료: {backup_path}")
        except Exception as e:
            self.logger.log_error(f"❌ 백업 실패: {e}")

    def _clean_old_files(self):
        """오래된 파일 정리"""
        automation_config = self.config.get("automation", {})
        if not automation_config.get("auto_clean", False):
            return
            
        try:
            # 30일 이상 된 로그 파일 삭제
            log_config = self.config.get("logging", {})
            backup_count = log_config.get("backup_count", 5)
            
            # 로그 파일 정리
            log_dir = Path(self.logs_folder)
            if log_dir.exists():
                log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime)
                if len(log_files) > backup_count:
                    for old_file in log_files[:-backup_count]:
                        old_file.unlink()
                        self.logger.log_info(f"🗑️ 오래된 로그 파일 삭제: {old_file}")
                        
            self.logger.log_success("✅ 파일 정리 완료")
        except Exception as e:
            self.logger.log_error(f"❌ 파일 정리 실패: {e}")

    def run_stage(self, stage_name: str) -> bool:
        self.logger.log_subsection(f"🚀 {stage_name} 단계 실행 시작")
        try:
            if stage_name == "cleaner":
                return self._run_cleaner()
            elif stage_name == "scrapper":
                return self._run_scrapper()
            elif stage_name == "researcher":
                return self._run_researcher()
            elif stage_name == "analyzer":
                return self._run_analyzer()
            elif stage_name == "evaluator":
                return self._run_evaluator()
            elif stage_name == "portfolio_manager":
                return self._run_portfolio_manager()
            else:
                self.logger.log_error(f"❌ 알 수 없는 단계: {stage_name}")
                return False
        except Exception as e:
            self.logger.log_error(f"❌ {stage_name} 단계 실행 중 오류: {e}")
            return False

    def _run_cleaner(self) -> bool:
        try:
            cleaner_config = self.config.get("cleaner", {})
            action = cleaner_config.get("action", "create")  # 기본값을 create로 변경
            run_cleaner = cleaner_config.get("run_cleaner", False)  # cleaner 실행 여부 제어
            folders = cleaner_config.get(
                "folders",
                ["data", "log", "results", "analysis", "researcher_results"]
            )
            
            # cleaner 실행을 건너뛰는 경우
            if not run_cleaner:
                self.logger.log_info("⏭️ Cleaner 단계 건너뛰기 (설정에서 비활성화됨)")
                self.execution_results["cleaner"] = {"status": "skipped", "reason": "disabled in config"}
                return True
            
            # 확장된 cleaner.py 사용 - folders 인자 전달
            cmd = [sys.executable, "-m", "agent.cleaner",
                   "--action", action]
            
            # folders 인자 추가
            if folders:
                cmd.extend(["--folders"] + folders)
            else:
                # 기존 방식 (하위 호환성)
                cmd.extend(["--data-dir", "data",
                           "--log-dir", self.logs_folder,
                           "--results-dir", self.results_folder])
            
            self.logger.log_info(f"실행 명령어: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.log_success("✅ Cleaner 단계 완료")
                self.execution_results["cleaner"] = {"status": "success", "output": result.stdout}
                return True
            else:
                self.logger.log_error(f"❌ Cleaner 단계 실패: {result.stderr}")
                self.execution_results["cleaner"] = {"status": "failed", "error": result.stderr}
                return False
        except Exception as e:
            self.logger.log_error(f"❌ Cleaner 실행 중 오류: {e}")
            return False

    def _run_scrapper(self) -> bool:
        try:
            # 전략별 데이터 경로 설정
            data_dir = f"data/{self.time_horizon}" if self.time_horizon else "data"
            
            # config 파일 경로 설정
            config_file = f"config/config_{self.time_horizon}.json" if self.time_horizon else "config/config_default.json"
            
            cmd = [sys.executable, "-m", "agent.scrapper", "--data-dir", data_dir, "--config", config_file, "--uuid", self.execution_uuid]
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            self.logger.log_info(f"실행 명령어: PYTHONPATH=src {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                self.logger.log_success("✅ Scrapper 단계 완료")
                self.execution_results["scrapper"] = {"status": "success", "output": result.stdout}
                return True
            else:
                self.logger.log_error(f"❌ Scrapper 단계 실패: {result.stderr}")
                self.execution_results["scrapper"] = {"status": "failed", "error": result.stderr}
                return False
        except Exception as e:
            self.logger.log_error(f"❌ Scrapper 실행 중 오류: {e}")
            return False

    def _run_researcher(self) -> bool:
        try:
            # 현재 config 파일명을 기반으로 research config의 source_config 설정
            current_config_name = self._get_current_config_name()
            
            # research config 파일 경로
            research_config_path = "config/config_research.json"
            
            # research config에서 source_config를 현재 config로 업데이트
            self._update_research_source_config(current_config_name)
            
            # 전략과 심볼 정보 가져오기
            strategies = self.config.get("strategies", [])
            symbols = self.config.get("data", {}).get("symbols", [])
            
            cmd = [sys.executable, "-m", "agent.researcher", "--config", research_config_path, "--uuid", self.execution_uuid]
            
            # 전략과 심볼 인자 추가
            if strategies:
                cmd.extend(["--strategies"] + strategies)
            if symbols:
                cmd.extend(["--symbols"] + symbols)
            
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            self.logger.log_info(f"실행 명령어: PYTHONPATH=src {' '.join(cmd)}")
            self.logger.log_info(f"📊 Research source config: {current_config_name}")
            self.logger.log_info(f"📊 Research strategies: {strategies}")
            self.logger.log_info(f"📊 Research symbols: {symbols}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                self.logger.log_success("✅ Researcher 단계 완료")
                self.execution_results["researcher"] = {"status": "success", "output": result.stdout}
                return True
            else:
                self.logger.log_error(f"❌ Researcher 단계 실패: {result.stderr}")
                self.execution_results["researcher"] = {"status": "failed", "error": result.stderr}
                return False
        except Exception as e:
            self.logger.log_error(f"❌ Researcher 실행 중 오류: {e}")
            return False

    def _run_analyzer(self) -> bool:
        try:
            # 전략별 데이터 경로 설정
            data_dir = f"data/{self.time_horizon}" if self.time_horizon else "data"
            
            cmd = [sys.executable, "-m", "agent.analyst", "--data_dir", data_dir, "--uuid", self.execution_uuid]
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            self.logger.log_info(f"실행 명령어: PYTHONPATH=src {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                self.logger.log_success("✅ Analyzer 단계 완료")
                self.execution_results["analyzer"] = {"status": "success", "output": result.stdout}
                return True
            else:
                self.logger.log_error(f"❌ Analyzer 단계 실패: {result.stderr}")
                self.execution_results["analyzer"] = {"status": "failed", "error": result.stderr}
                return False
        except Exception as e:
            self.logger.log_error(f"❌ Analyzer 실행 중 오류: {e}")
            return False

    def _run_evaluator(self) -> bool:
        try:
            evaluator_config = self.config.get("evaluator", {})
            strategies = self.config.get("strategies", [])
            if not strategies:
                self.logger.log_warning("⚠️ 실행할 전략이 없음 - 스킵")
                self.execution_results["evaluator"] = {"status": "skipped", "reason": "no strategies"}
                return True
                
            # 포트폴리오 모드 확인
            portfolio_mode = evaluator_config.get("portfolio_mode", False)
            cmd = [sys.executable, "-m", "agent.evaluator"]
            
            if portfolio_mode:
                cmd.append("--portfolio")
            
            cmd.extend(["--strategies"] + strategies)
            
            # 결과 폴더 지정
            cmd.extend(["--results_dir", self.results_folder])
            
            # UUID 추가
            cmd.extend(["--uuid", self.execution_uuid])
            
            # config 경로 추가
            cmd.extend(["--config", self.config_path])
            
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            self.logger.log_info(f"실행 명령어: PYTHONPATH=src {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                self.logger.log_success("✅ Evaluator 단계 완료")
                self.execution_results["evaluator"] = {"status": "success", "output": result.stdout}
                return True
            else:
                self.logger.log_error(f"❌ Evaluator 단계 실패: {result.stderr}")
                self.execution_results["evaluator"] = {"status": "failed", "error": result.stderr}
                return False
        except Exception as e:
            self.logger.log_error(f"❌ Evaluator 실행 중 오류: {e}")
            return False

    def _run_portfolio_manager(self) -> bool:
        try:
            cmd = [sys.executable, "-m", "agent.portfolio_manager", "--uuid", self.execution_uuid]
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            self.logger.log_info(f"실행 명령어: PYTHONPATH=src {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                self.logger.log_success("✅ Portfolio Manager 단계 완료")
                self.execution_results["portfolio_manager"] = {"status": "success", "output": result.stdout}
                return True
            else:
                self.logger.log_error(f"❌ Portfolio Manager 단계 실패: {result.stderr}")
                self.execution_results["portfolio_manager"] = {"status": "failed", "error": result.stderr}
                return False
        except Exception as e:
            self.logger.log_error(f"❌ Portfolio Manager 실행 중 오류: {e}")
            return False

    def run_full_flow(self) -> bool:
        self.logger.log_section("🎯 퀀트 분석 파이프라인 시작")
        
        # 사전 정리
        self._clean_old_files()
        
        flow_config = self.config.get("flow", {})
        stages = flow_config.get("stages", ["cleaner", "scrapper", "analyzer", "researcher", "evaluator", "portfolio_manager"])
        stop_on_error = flow_config.get("stop_on_error", False)
        
        # researcher 단계 활성화 여부 확인
        enable_research = flow_config.get("enable_research", True)
        if not enable_research and "researcher" in stages:
            stages.remove("researcher")
            self.logger.log_info("⏭️ Researcher 단계 비활성화됨")
        
        self.logger.log_info(f"실행할 단계들: {', '.join(stages)}")
        self.logger.log_info(f"오류 시 중단: {stop_on_error}")
        success_count = 0
        total_stages = len(stages)
        for i, stage in enumerate(stages, 1):
            self.logger.log_info(f"📋 진행률: {i}/{total_stages} ({stage})")
            success = self.run_stage(stage)
            if success:
                success_count += 1
            else:
                self.logger.log_error(f"❌ {stage} 단계 실패")
                if stop_on_error:
                    self.logger.log_error("🚫 오류로 인해 파이프라인 중단")
                    break
        
        # 사후 처리
        self._backup_results()
        
        self.logger.log_section("📊 파이프라인 실행 결과 요약")
        self.logger.log_info(f"총 단계: {total_stages}개")
        self.logger.log_info(f"성공: {success_count}개")
        self.logger.log_info(f"실패: {total_stages - success_count}개")
        for stage, result in self.execution_results.items():
            status = result.get("status", "unknown")
            if status == "success":
                self.logger.log_success(f"✅ {stage}: 성공")
            elif status == "failed":
                self.logger.log_error(f"❌ {stage}: 실패")
            elif status == "skipped":
                self.logger.log_warning(f"⚠️ {stage}: 스킵 ({result.get('reason', 'N/A')})")
        if success_count == total_stages:
            self.logger.log_success("🎉 모든 단계가 성공적으로 완료되었습니다!")
            return True
        else:
            self.logger.log_error(f"⚠️ 일부 단계가 실패했습니다. ({success_count}/{total_stages})")
            return False

    def run_single_stage(self, stage_name: str) -> bool:
        self.logger.log_section(f"🎯 {stage_name} 단계만 실행")
        return self.run_stage(stage_name)

    def get_execution_summary(self) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "time_horizon": self.time_horizon,
            "config_file": getattr(self, 'config_path', 'default'),
            "total_stages": len(self.execution_results),
            "successful_stages": sum(1 for r in self.execution_results.values() if r.get("status") == "success"),
            "failed_stages": sum(1 for r in self.execution_results.values() if r.get("status") == "failed"),
            "skipped_stages": sum(1 for r in self.execution_results.values() if r.get("status") == "skipped"),
            "results": self.execution_results
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="퀀트 분석 파이프라인 Orchestrator")
    parser.add_argument(
        "--stage",
        choices=["cleaner", "scrapper", "researcher", "analyzer", "evaluator", "portfolio_manager"],
        help="실행할 단일 단계"
    )
    parser.add_argument(
        "--config",
        default="../../config/config_default.json",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--time-horizon",
        choices=["swing", "long", "long-term", "scalping", "short-term"],
        help="전략 타임프레임 (config 자동 선택)"
    )
    parser.add_argument(
        "--no-research",
        action="store_true",
        help="연구 단계 제외"
    )
    args = parser.parse_args()
    
    orchestrator = FlowOrchestrator(args.config, args.time_horizon)
    
    # --no-research 옵션 처리
    if args.no_research:
        orchestrator.config.get("flow", {})["enable_research"] = False
    
    if args.stage:
        success = orchestrator.run_single_stage(args.stage)
    else:
        success = orchestrator.run_full_flow()
    
    summary = orchestrator.get_execution_summary()
    orchestrator.logger.save_json_log(
        summary,
        f"flow_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    exit(0 if success else 1)

if __name__ == "__main__":
    main() 