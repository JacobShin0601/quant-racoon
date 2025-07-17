import os
import shutil
import logging
import json
from typing import List, Optional
from datetime import datetime


class Cleaner:
    """
    데이터 폴더, 로그 폴더, 결과 폴더를 정리하는 클래스
    """

    def __init__(self):
        """Cleaner 클래스 초기화"""
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def clean_folder(self, folder: str) -> bool:
        """
        지정한 폴더를 정리합니다.
        """
        try:
            if os.path.exists(folder):
                self.logger.info(f"📁 {folder} 폴더 정리 중...")
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        self.logger.info(f"  삭제: {filename}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        self.logger.info(f"  삭제: {filename}/ (폴더)")
                self.logger.info(f"✅ {folder} 폴더 정리 완료")
                return True
            else:
                self.logger.warning(f"⚠️ {folder} 폴더가 존재하지 않습니다.")
                return False
        except Exception as e:
            self.logger.error(f"❌ {folder} 폴더 정리 중 오류 발생: {e}")
            return False

    def clean_folders(self, folders: List[str]) -> bool:
        """
        여러 폴더를 일괄 정리합니다.
        폴더가 존재하지 않아도 경고만 출력하고 계속 진행합니다.
        """
        self.logger.info(f"🧹 {len(folders)}개 폴더 일괄 정리 시작")
        success_count = 0
        
        for folder in folders:
            if self.clean_folder(folder):
                success_count += 1
            else:
                self.logger.warning(f"⚠️ {folder} 폴더 정리 실패")
        
        if success_count == len(folders):
            self.logger.info(f"✅ 모든 폴더 정리 완료 ({success_count}/{len(folders)})")
            return True
        else:
            self.logger.info(f"ℹ️ 폴더 정리 완료 ({success_count}/{len(folders)}) - 일부 폴더는 존재하지 않음")
            return True  # 폴더가 존재하지 않아도 성공으로 처리

    def create_folder(self, folder: str) -> bool:
        """
        지정한 폴더를 생성합니다.
        """
        try:
            os.makedirs(folder, exist_ok=True)
            self.logger.info(f"✅ {folder} 폴더 생성 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ {folder} 폴더 생성 중 오류 발생: {e}")
            return False

    def create_folders(self, folders: List[str]) -> bool:
        """
        여러 폴더를 일괄 생성합니다.
        """
        self.logger.info(f"📁 {len(folders)}개 폴더 일괄 생성 시작")
        success_count = 0
        
        for folder in folders:
            if self.create_folder(folder):
                success_count += 1
            else:
                self.logger.warning(f"⚠️ {folder} 폴더 생성 실패")
        
        if success_count == len(folders):
            self.logger.info(f"✅ 모든 폴더 생성 완료 ({success_count}/{len(folders)})")
            return True
        else:
            self.logger.warning(f"⚠️ 일부 폴더 생성 실패 ({success_count}/{len(folders)})")
            return False

    def clean_and_recreate_folders(self, folders: List[str]) -> bool:
        """
        여러 폴더를 정리하고 재생성합니다.
        폴더가 존재하지 않아도 경고만 출력하고 계속 진행합니다.
        """
        self.logger.info(f"🔄 {len(folders)}개 폴더 정리 및 재생성 시작")
        
        # 폴더 정리 (존재하지 않는 폴더는 경고만 출력)
        clean_success_count = 0
        for folder in folders:
            if self.clean_folder(folder):
                clean_success_count += 1
        
        # 폴더 재생성
        create_success = self.create_folders(folders)
        if create_success:
            self.logger.info(f"✅ 폴더 정리 및 재생성 완료 (정리: {clean_success_count}/{len(folders)}, 생성: 성공)")
            return True
        else:
            self.logger.error("❌ 폴더 재생성 실패")
            return False

    def clean_data_folder(self, data_dir: str = "data") -> bool:
        return self.clean_folder(data_dir)

    def clean_log_folder(self, log_dir: str = "log") -> bool:
        return self.clean_folder(log_dir)

    def clean_results_folder(self, results_dir: str = "results") -> bool:
        return self.clean_folder(results_dir)

    def clean_all_folders(self, data_dir: str = "data", log_dir: str = "log", results_dir: str = "results") -> bool:
        self.logger.info("🧹 모든 폴더 정리 시작")
        data_success = self.clean_data_folder(data_dir)
        log_success = self.clean_log_folder(log_dir)
        results_success = self.clean_results_folder(results_dir)
        if data_success and log_success and results_success:
            self.logger.info("✅ 모든 폴더 정리 완료")
            return True
        else:
            self.logger.error("❌ 일부 폴더 정리 실패")
            return False

    def create_empty_folders(self, data_dir: str = "data", log_dir: str = "log", results_dir: str = "results") -> bool:
        try:
            self.logger.info("📁 빈 폴더 생성 중...")
            os.makedirs(data_dir, exist_ok=True)
            self.logger.info(f"✅ {data_dir} 폴더 생성 완료")
            os.makedirs(log_dir, exist_ok=True)
            self.logger.info(f"✅ {log_dir} 폴더 생성 완료")
            os.makedirs(results_dir, exist_ok=True)
            self.logger.info(f"✅ {results_dir} 폴더 생성 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ 폴더 생성 중 오류 발생: {e}")
            return False

    def clean_and_recreate_folders_legacy(self, data_dir: str = "data", log_dir: str = "log", results_dir: str = "results") -> bool:
        self.logger.info("🔄 폴더 정리 및 재생성 시작")
        
        # data 폴더는 건드리지 않고, 전략별 폴더만 정리
        clean_success = self.clean_log_folder(log_dir) and self.clean_results_folder(results_dir)
        if clean_success:
            create_success = self.create_empty_folders(data_dir, log_dir, results_dir)
            if create_success:
                self.logger.info("✅ 전략별 폴더 정리 및 재생성 완료 (data 폴더는 유지)")
                return True
            else:
                self.logger.error("❌ 빈 폴더 생성 실패")
                return False
        else:
            self.logger.error("❌ 전략별 폴더 정리 실패")
            return False

    def get_folder_info(self, folders: List[str] = None) -> dict:
        if folders is None:
            folders = ["data", "log", "results"]
            
        info = {}
        for folder in folders:
            info[folder] = {
                "exists": os.path.exists(folder),
                "file_count": 0,
                "files": [],
            }
            if info[folder]["exists"]:
                files = os.listdir(folder)
                info[folder]["file_count"] = len(files)
                info[folder]["files"] = files[:10]
        return info

    def print_folder_info(self, folders: List[str] = None):
        if folders is None:
            folders = ["data", "log", "results"]
            
        info = self.get_folder_info(folders)
        print("\n📊 폴더 정보")
        print("=" * 50)
        
        for folder in folders:
            print(f"📁 {folder} 폴더:")
            if info[folder]["exists"]:
                print(f"  ✅ 존재함")
                print(f"  📄 파일 수: {info[folder]['file_count']}개")
                if info[folder]["files"]:
                    print(f"  📋 파일 목록 (처음 10개):")
                    for file in info[folder]["files"]:
                        print(f"    - {file}")
            else:
                print(f"  ❌ 존재하지 않음")
            print()
        print("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="데이터/로그/결과 폴더 정리 도구")
    parser.add_argument(
        "--action",
        choices=["clean", "create", "clean-and-recreate", "info"],
        default="clean-and-recreate",
        help="수행할 작업",
    )
    parser.add_argument("--data-dir", default="data", help="data 폴더 경로")
    parser.add_argument("--log-dir", default="log", help="log 폴더 경로")
    parser.add_argument("--results-dir", default="results", help="results 폴더 경로")
    parser.add_argument("--folders", nargs="+", help="처리할 폴더 목록 (여러 폴더 지정 가능)")
    parser.add_argument("--folders-json", help="폴더 목록을 JSON 파일로 지정")
    args = parser.parse_args()
    
    cleaner = Cleaner()
    
    # 폴더 목록 결정
    if args.folders:
        # --folders 인자로 직접 지정
        folders = args.folders
    elif args.folders_json:
        # JSON 파일에서 폴더 목록 읽기
        try:
            with open(args.folders_json, 'r', encoding='utf-8') as f:
                folders = json.load(f)
        except Exception as e:
            print(f"❌ JSON 파일 읽기 실패: {e}")
            exit(1)
    else:
        # 기존 방식 (하위 호환성)
        folders = [args.data_dir, args.log_dir, args.results_dir]
    
    print(f"📋 처리할 폴더 목록: {folders}")
    
    if args.action == "clean":
        success = cleaner.clean_folders(folders)
        exit(0 if success else 1)
    elif args.action == "create":
        success = cleaner.create_folders(folders)
        exit(0 if success else 1)
    elif args.action == "clean-and-recreate":
        success = cleaner.clean_and_recreate_folders(folders)
        exit(0 if success else 1)
    elif args.action == "info":
        cleaner.print_folder_info(folders)
        exit(0)


if __name__ == "__main__":
    main()
