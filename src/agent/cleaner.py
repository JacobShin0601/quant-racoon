import os
import shutil
import logging
from typing import List, Optional
from datetime import datetime


class Cleaner:
    """
    데이터 폴더와 로그 폴더를 정리하는 클래스
    """

    def __init__(self):
        """Cleaner 클래스 초기화"""
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def clean_data_folder(self, data_dir: str = "data") -> bool:
        """
        data 폴더를 정리합니다.

        Args:
            data_dir (str): 정리할 data 폴더 경로

        Returns:
            bool: 정리 성공 여부
        """
        try:
            if os.path.exists(data_dir):
                self.logger.info(f"📁 {data_dir} 폴더 정리 중...")

                # 폴더 내 모든 파일 삭제
                for filename in os.listdir(data_dir):
                    file_path = os.path.join(data_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        self.logger.info(f"  삭제: {filename}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        self.logger.info(f"  삭제: {filename}/ (폴더)")

                self.logger.info(f"✅ {data_dir} 폴더 정리 완료")
                return True
            else:
                self.logger.warning(f"⚠️ {data_dir} 폴더가 존재하지 않습니다.")
                return False

        except Exception as e:
            self.logger.error(f"❌ {data_dir} 폴더 정리 중 오류 발생: {e}")
            return False

    def clean_log_folder(self, log_dir: str = "log") -> bool:
        """
        log 폴더를 정리합니다.

        Args:
            log_dir (str): 정리할 log 폴더 경로

        Returns:
            bool: 정리 성공 여부
        """
        try:
            if os.path.exists(log_dir):
                self.logger.info(f"📁 {log_dir} 폴더 정리 중...")

                # 폴더 내 모든 파일 삭제
                for filename in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        self.logger.info(f"  삭제: {filename}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        self.logger.info(f"  삭제: {filename}/ (폴더)")

                self.logger.info(f"✅ {log_dir} 폴더 정리 완료")
                return True
            else:
                self.logger.warning(f"⚠️ {log_dir} 폴더가 존재하지 않습니다.")
                return False

        except Exception as e:
            self.logger.error(f"❌ {log_dir} 폴더 정리 중 오류 발생: {e}")
            return False

    def clean_all_folders(self, data_dir: str = "data", log_dir: str = "log") -> bool:
        """
        data와 log 폴더를 모두 정리합니다.

        Args:
            data_dir (str): 정리할 data 폴더 경로
            log_dir (str): 정리할 log 폴더 경로

        Returns:
            bool: 모든 정리 작업 성공 여부
        """
        self.logger.info("🧹 모든 폴더 정리 시작")

        data_success = self.clean_data_folder(data_dir)
        log_success = self.clean_log_folder(log_dir)

        if data_success and log_success:
            self.logger.info("✅ 모든 폴더 정리 완료")
            return True
        else:
            self.logger.error("❌ 일부 폴더 정리 실패")
            return False

    def create_empty_folders(
        self, data_dir: str = "data", log_dir: str = "log"
    ) -> bool:
        """
        빈 data와 log 폴더를 생성합니다.

        Args:
            data_dir (str): 생성할 data 폴더 경로
            log_dir (str): 생성할 log 폴더 경로

        Returns:
            bool: 폴더 생성 성공 여부
        """
        try:
            self.logger.info("📁 빈 폴더 생성 중...")

            # data 폴더 생성
            os.makedirs(data_dir, exist_ok=True)
            self.logger.info(f"✅ {data_dir} 폴더 생성 완료")

            # log 폴더 생성
            os.makedirs(log_dir, exist_ok=True)
            self.logger.info(f"✅ {log_dir} 폴더 생성 완료")

            return True

        except Exception as e:
            self.logger.error(f"❌ 폴더 생성 중 오류 발생: {e}")
            return False

    def clean_and_recreate_folders(
        self, data_dir: str = "data", log_dir: str = "log"
    ) -> bool:
        """
        폴더를 정리하고 빈 폴더를 다시 생성합니다.

        Args:
            data_dir (str): 정리/생성할 data 폴더 경로
            log_dir (str): 정리/생성할 log 폴더 경로

        Returns:
            bool: 작업 성공 여부
        """
        self.logger.info("🔄 폴더 정리 및 재생성 시작")

        # 폴더 정리
        clean_success = self.clean_all_folders(data_dir, log_dir)

        if clean_success:
            # 빈 폴더 생성
            create_success = self.create_empty_folders(data_dir, log_dir)

            if create_success:
                self.logger.info("✅ 폴더 정리 및 재생성 완료")
                return True
            else:
                self.logger.error("❌ 빈 폴더 생성 실패")
                return False
        else:
            self.logger.error("❌ 폴더 정리 실패")
            return False

    def get_folder_info(self, data_dir: str = "data", log_dir: str = "log") -> dict:
        """
        폴더 정보를 반환합니다.

        Args:
            data_dir (str): 확인할 data 폴더 경로
            log_dir (str): 확인할 log 폴더 경로

        Returns:
            dict: 폴더 정보
        """
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
        }

        # data 폴더 정보
        if info["data_folder"]["exists"]:
            files = os.listdir(data_dir)
            info["data_folder"]["file_count"] = len(files)
            info["data_folder"]["files"] = files[:10]  # 처음 10개만

        # log 폴더 정보
        if info["log_folder"]["exists"]:
            files = os.listdir(log_dir)
            info["log_folder"]["file_count"] = len(files)
            info["log_folder"]["files"] = files[:10]  # 처음 10개만

        return info

    def print_folder_info(self, data_dir: str = "data", log_dir: str = "log"):
        """
        폴더 정보를 출력합니다.

        Args:
            data_dir (str): 확인할 data 폴더 경로
            log_dir (str): 확인할 log 폴더 경로
        """
        info = self.get_folder_info(data_dir, log_dir)

        print("\n📊 폴더 정보")
        print("=" * 50)

        # data 폴더 정보
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

        # log 폴더 정보
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

        print("=" * 50)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="데이터 및 로그 폴더 정리 도구")
    parser.add_argument(
        "--action",
        choices=["clean", "create", "clean-and-recreate", "info"],
        default="clean-and-recreate",
        help="수행할 작업",
    )
    parser.add_argument("--data-dir", default="data", help="data 폴더 경로")
    parser.add_argument("--log-dir", default="log", help="log 폴더 경로")

    args = parser.parse_args()

    cleaner = Cleaner()

    if args.action == "clean":
        success = cleaner.clean_all_folders(args.data_dir, args.log_dir)
        exit(0 if success else 1)

    elif args.action == "create":
        success = cleaner.create_empty_folders(args.data_dir, args.log_dir)
        exit(0 if success else 1)

    elif args.action == "clean-and-recreate":
        success = cleaner.clean_and_recreate_folders(args.data_dir, args.log_dir)
        exit(0 if success else 1)

    elif args.action == "info":
        cleaner.print_folder_info(args.data_dir, args.log_dir)
        exit(0)


if __name__ == "__main__":
    main()
