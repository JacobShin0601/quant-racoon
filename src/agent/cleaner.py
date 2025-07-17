import os
import shutil
import logging
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

    def clean_and_recreate_folders(self, data_dir: str = "data", log_dir: str = "log", results_dir: str = "results") -> bool:
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

    def get_folder_info(self, data_dir: str = "data", log_dir: str = "log", results_dir: str = "results") -> dict:
        info = {
            "data_folder": {
                "exists": os.path.exists(data_dir),
                "file_count": 0,
                "files": [],
            },
            "log_folder": {
                "exists": os.path.exists(log_dir),
                "file_count": 0,
                "files": [],
            },
            "results_folder": {
                "exists": os.path.exists(results_dir),
                "file_count": 0,
                "files": [],
            },
        }
        if info["data_folder"]["exists"]:
            files = os.listdir(data_dir)
            info["data_folder"]["file_count"] = len(files)
            info["data_folder"]["files"] = files[:10]
        if info["log_folder"]["exists"]:
            files = os.listdir(log_dir)
            info["log_folder"]["file_count"] = len(files)
            info["log_folder"]["files"] = files[:10]
        if info["results_folder"]["exists"]:
            files = os.listdir(results_dir)
            info["results_folder"]["file_count"] = len(files)
            info["results_folder"]["files"] = files[:10]
        return info

    def print_folder_info(self, data_dir: str = "data", log_dir: str = "log", results_dir: str = "results"):
        info = self.get_folder_info(data_dir, log_dir, results_dir)
        print("\n📊 폴더 정보")
        print("=" * 50)
        print(f"📁 {data_dir} 폴더:")
        if info["data_folder"]["exists"]:
            print(f"  ✅ 존재함")
            print(f"  📄 파일 수: {info['data_folder']['file_count']}개")
            if info["data_folder"]["files"]:
                print(f"  📋 파일 목록 (처음 10개):")
                for file in info["data_folder"]["files"]:
                    print(f"    - {file}")
        else:
            print(f"  ❌ 존재하지 않음")
        print()
        print(f"📁 {log_dir} 폴더:")
        if info["log_folder"]["exists"]:
            print(f"  ✅ 존재함")
            print(f"  📄 파일 수: {info['log_folder']['file_count']}개")
            if info["log_folder"]["files"]:
                print(f"  📋 파일 목록 (처음 10개):")
                for file in info["log_folder"]["files"]:
                    print(f"    - {file}")
        else:
            print(f"  ❌ 존재하지 않음")
        print()
        print(f"📁 {results_dir} 폴더:")
        if info["results_folder"]["exists"]:
            print(f"  ✅ 존재함")
            print(f"  📄 파일 수: {info['results_folder']['file_count']}개")
            if info["results_folder"]["files"]:
                print(f"  📋 파일 목록 (처음 10개):")
                for file in info["results_folder"]["files"]:
                    print(f"    - {file}")
        else:
            print(f"  ❌ 존재하지 않음")
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
    args = parser.parse_args()
    cleaner = Cleaner()
    if args.action == "clean":
        success = cleaner.clean_all_folders(args.data_dir, args.log_dir, args.results_dir)
        exit(0 if success else 1)
    elif args.action == "create":
        success = cleaner.create_empty_folders(args.data_dir, args.log_dir, args.results_dir)
        exit(0 if success else 1)
    elif args.action == "clean-and-recreate":
        success = cleaner.clean_and_recreate_folders(args.data_dir, args.log_dir, args.results_dir)
        exit(0 if success else 1)
    elif args.action == "info":
        cleaner.print_folder_info(args.data_dir, args.log_dir, args.results_dir)
        exit(0)


if __name__ == "__main__":
    main()
