import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class FinnhubDataCollector:
    """
    Finnhub API를 사용하여 주식 데이터를 수집하고 CSV로 저장하는 클래스
    """

    def __init__(self):
        """Finnhub API 클라이언트 초기화"""
        self.api_key = os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY가 .env 파일에 설정되지 않았습니다.")

        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({"X-Finnhub-Token": self.api_key})

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_candle_data(
        self,
        symbol: str,
        resolution: str = "15",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Finnhub API에서 캔들스틱 데이터를 가져옵니다.

        Args:
            symbol (str): 주식 티커 (예: 'AAPL', 'MSFT')
            resolution (str): 시간 단위 ('1', '5', '15', '30', '60', 'D', 'W', 'M')
            start_date (str): 시작 날짜 (YYYY-MM-DD 형식, None이면 days_back 사용)
            end_date (str): 종료 날짜 (YYYY-MM-DD 형식, None이면 오늘)
            days_back (int): start_date가 None일 때 사용할 과거 일수

        Returns:
            pd.DataFrame: 캔들스틱 데이터
        """
        try:
            # 날짜 설정
            if end_date is None:
                end_date = datetime.now()
            else:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")

            if start_date is None:
                start_date = end_date - timedelta(days=days_back)
            else:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")

            # Unix timestamp로 변환
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())

            # API 요청
            url = f"{self.base_url}/stock/candle"
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": start_timestamp,
                "to": end_timestamp,
            }

            self.logger.info(
                f"{symbol} 데이터 수집 중... (기간: {start_date.date()} ~ {end_date.date()})"
            )

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data["s"] != "ok":
                raise ValueError(f"API 오류: {data.get('s', 'unknown error')}")

            # DataFrame으로 변환
            df = pd.DataFrame(
                {
                    "timestamp": data["t"],
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"],
                }
            )

            # timestamp를 datetime으로 변환
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df["date"] = df["datetime"].dt.date
            df["time"] = df["datetime"].dt.time

            # 컬럼 순서 재정렬
            df = df[
                [
                    "datetime",
                    "date",
                    "time",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]

            self.logger.info(f"{len(df)}개의 데이터 포인트를 수집했습니다.")

            return df

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API 요청 오류: {e}")
            raise
        except Exception as e:
            self.logger.error(f"데이터 처리 오류: {e}")
            raise

    def save_to_csv(
        self,
        df: pd.DataFrame,
        symbol: str,
        resolution: str,
        start_date: str,
        end_date: str,
        output_dir: str = "data",
    ) -> str:
        """
        DataFrame을 CSV 파일로 저장합니다.

        Args:
            df (pd.DataFrame): 저장할 데이터
            symbol (str): 주식 티커
            resolution (str): 시간 단위
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            output_dir (str): 출력 디렉토리

        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)

            # 파일명 생성 (티커_분봉_기간_오늘날짜.csv)
            today = datetime.now().strftime("%Y%m%d")
            resolution_name = self._get_resolution_name(resolution)
            period = f"{start_date}_{end_date}"

            filename = f"{symbol}_{resolution_name}_{period}_{today}.csv"
            filepath = os.path.join(output_dir, filename)

            # CSV 저장
            df.to_csv(filepath, index=False, encoding="utf-8")

            self.logger.info(f"데이터가 {filepath}에 저장되었습니다.")

            return filepath

        except Exception as e:
            self.logger.error(f"CSV 저장 오류: {e}")
            raise

    def _get_resolution_name(self, resolution: str) -> str:
        """해상도 코드를 읽기 쉬운 이름으로 변환"""
        resolution_map = {
            "1": "1min",
            "5": "5min",
            "15": "15min",
            "30": "30min",
            "60": "1hour",
            "D": "daily",
            "W": "weekly",
            "M": "monthly",
        }
        return resolution_map.get(resolution, resolution)

    def collect_and_save(
        self,
        symbol: str,
        resolution: str = "15",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 30,
        output_dir: str = "data",
    ) -> str:
        """
        데이터를 수집하고 CSV로 저장하는 통합 메서드

        Args:
            symbol (str): 주식 티커
            resolution (str): 시간 단위
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            days_back (int): 과거 일수 (start_date가 None일 때)
            output_dir (str): 출력 디렉토리

        Returns:
            str: 저장된 파일 경로
        """
        # 데이터 수집
        df = self.get_candle_data(symbol, resolution, start_date, end_date, days_back)

        # 날짜 범위 계산
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # CSV 저장
        filepath = self.save_to_csv(
            df, symbol, resolution, start_date, end_date, output_dir
        )

        return filepath


def main():
    """메인 실행 함수 - 예제 사용법"""
    try:
        # 데이터 수집기 초기화
        collector = FinnhubDataCollector()

        # 예제: Apple 주식 데이터 수집 (15분봉, 최근 30일)
        symbol = "AAPL"
        resolution = "15"  # 15분봉
        days_back = 30

        print(f"{symbol} 주식 데이터를 수집합니다...")

        # 데이터 수집 및 저장
        filepath = collector.collect_and_save(
            symbol=symbol, resolution=resolution, days_back=days_back
        )

        print(f"데이터가 성공적으로 저장되었습니다: {filepath}")

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
