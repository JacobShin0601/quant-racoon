import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class AlphaVantageDataCollector:
    """
    Alpha Vantage API를 사용하여 주식 데이터를 수집하고 CSV로 저장하는 클래스
    """

    def __init__(self):
        """Alpha Vantage API 클라이언트 초기화"""
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY가 .env 파일에 설정되지 않았습니다.")

        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "15min",
        adjusted: bool = True,
        extended_hours: bool = True,
        outputsize: str = "full",
        month: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Alpha Vantage API에서 intraday 캔들스틱 데이터를 가져옵니다.

        Args:
            symbol (str): 주식 티커 (예: 'AAPL', 'MSFT')
            interval (str): 시간 단위 ('1min', '5min', '15min', '30min', '60min')
            adjusted (bool): 분할/배당 조정 여부
            extended_hours (bool): 장외 시간 포함 여부
            outputsize (str): 출력 크기 ('compact': 최근 100개, 'full': 최근 30일)
            month (str): 특정 월 (YYYY-MM 형식, 예: '2024-01')

        Returns:
            pd.DataFrame: 캔들스틱 데이터
        """
        try:
            # API 요청 파라미터
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "adjusted": "true" if adjusted else "false",
                "extended_hours": "true" if extended_hours else "false",
                "outputsize": outputsize,
                "datatype": "json",
                "apikey": self.api_key,
            }

            # 특정 월이 지정된 경우 추가
            if month:
                params["month"] = month

            self.logger.info(
                f"{symbol} intraday 데이터 수집 중... (interval: {interval})"
            )

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # API 오류 확인
            if "Error Message" in data:
                raise ValueError(f"API 오류: {data['Error Message']}")

            if "Note" in data:
                raise ValueError(f"API 제한: {data['Note']}")

            # 데이터 추출
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                raise ValueError(f"예상치 못한 응답 형식: {list(data.keys())}")

            time_series = data[time_series_key]

            # DataFrame으로 변환
            records = []
            for timestamp, values in time_series.items():
                record = {
                    "datetime": timestamp,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["5. volume"]),
                }
                records.append(record)

            df = pd.DataFrame(records)

            # datetime 파싱 및 추가 컬럼 생성
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = df["datetime"].dt.date
            df["time"] = df["datetime"].dt.time
            df["timestamp"] = df["datetime"].astype("int64") // 10**9  # Unix timestamp

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

            # 시간순 정렬
            df = df.sort_values("datetime").reset_index(drop=True)

            self.logger.info(f"{len(df)}개의 데이터 포인트를 수집했습니다.")

            return df

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API 요청 오류: {e}")
            raise
        except Exception as e:
            self.logger.error(f"데이터 처리 오류: {e}")
            raise

    def get_daily_data(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """
        Alpha Vantage API에서 일봉 데이터를 가져옵니다.

        Args:
            symbol (str): 주식 티커
            outputsize (str): 출력 크기 ('compact': 최근 100개, 'full': 최대 20년)

        Returns:
            pd.DataFrame: 일봉 데이터
        """
        try:
            # API 요청 파라미터
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": outputsize,
                "datatype": "json",
                "apikey": self.api_key,
            }

            self.logger.info(f"{symbol} 일봉 데이터 수집 중...")

            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # API 오류 확인
            if "Error Message" in data:
                raise ValueError(f"API 오류: {data['Error Message']}")

            if "Note" in data:
                raise ValueError(f"API 제한: {data['Note']}")

            # 데이터 추출
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                raise ValueError(f"예상치 못한 응답 형식: {list(data.keys())}")

            time_series = data[time_series_key]

            # DataFrame으로 변환
            records = []
            for date, values in time_series.items():
                record = {
                    "datetime": date,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "adjusted_close": float(values["5. adjusted close"]),
                    "volume": int(values["6. volume"]),
                    "dividend_amount": float(values["7. dividend amount"]),
                    "split_coefficient": float(values["8. split coefficient"]),
                }
                records.append(record)

            df = pd.DataFrame(records)

            # datetime 파싱 및 추가 컬럼 생성
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"] = df["datetime"].dt.date
            df["time"] = df["datetime"].dt.time
            df["timestamp"] = df["datetime"].astype("int64") // 10**9  # Unix timestamp

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
                    "adjusted_close",
                    "volume",
                    "dividend_amount",
                    "split_coefficient",
                ]
            ]

            # 시간순 정렬
            df = df.sort_values("datetime").reset_index(drop=True)

            self.logger.info(f"{len(df)}개의 일봉 데이터 포인트를 수집했습니다.")

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
        interval: str,
        start_date: str,
        end_date: str,
        output_dir: str = "data",
    ) -> str:
        """
        DataFrame을 CSV 파일로 저장합니다.

        Args:
            df (pd.DataFrame): 저장할 데이터
            symbol (str): 주식 티커
            interval (str): 시간 단위
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
            interval_name = self._get_interval_name(interval)
            period = f"{start_date}_{end_date}"

            filename = f"{symbol}_{interval_name}_{period}_{today}.csv"
            filepath = os.path.join(output_dir, filename)

            # CSV 저장
            df.to_csv(filepath, index=False, encoding="utf-8")

            self.logger.info(f"데이터가 {filepath}에 저장되었습니다.")

            return filepath

        except Exception as e:
            self.logger.error(f"CSV 저장 오류: {e}")
            raise

    def _get_interval_name(self, interval: str) -> str:
        """시간 간격 코드를 읽기 쉬운 이름으로 변환"""
        interval_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "1hour",
            "daily": "daily",
            "weekly": "weekly",
            "monthly": "monthly",
        }
        return interval_map.get(interval, interval)

    def collect_and_save_intraday(
        self,
        symbol: str,
        interval: str = "15min",
        adjusted: bool = True,
        extended_hours: bool = True,
        outputsize: str = "full",
        month: Optional[str] = None,
        output_dir: str = "data",
    ) -> str:
        """
        intraday 데이터를 수집하고 CSV로 저장하는 통합 메서드

        Args:
            symbol (str): 주식 티커
            interval (str): 시간 단위
            adjusted (bool): 분할/배당 조정 여부
            extended_hours (bool): 장외 시간 포함 여부
            outputsize (str): 출력 크기
            month (str): 특정 월 (YYYY-MM)
            output_dir (str): 출력 디렉토리

        Returns:
            str: 저장된 파일 경로
        """
        # 데이터 수집
        df = self.get_intraday_data(
            symbol, interval, adjusted, extended_hours, outputsize, month
        )

        # 날짜 범위 계산
        if len(df) > 0:
            start_date = df["date"].min().strftime("%Y-%m-%d")
            end_date = df["date"].max().strftime("%Y-%m-%d")
        else:
            start_date = "unknown"
            end_date = "unknown"

        # CSV 저장
        filepath = self.save_to_csv(
            df, symbol, interval, start_date, end_date, output_dir
        )

        return filepath

    def collect_and_save_daily(
        self, symbol: str, outputsize: str = "full", output_dir: str = "data"
    ) -> str:
        """
        일봉 데이터를 수집하고 CSV로 저장하는 통합 메서드

        Args:
            symbol (str): 주식 티커
            outputsize (str): 출력 크기
            output_dir (str): 출력 디렉토리

        Returns:
            str: 저장된 파일 경로
        """
        # 데이터 수집
        df = self.get_daily_data(symbol, outputsize)

        # 날짜 범위 계산
        if len(df) > 0:
            start_date = df["date"].min().strftime("%Y-%m-%d")
            end_date = df["date"].max().strftime("%Y-%m-%d")
        else:
            start_date = "unknown"
            end_date = "unknown"

        # CSV 저장
        filepath = self.save_to_csv(
            df, symbol, "daily", start_date, end_date, output_dir
        )

        return filepath


def main():
    """메인 실행 함수 - 예제 사용법"""
    try:
        # 데이터 수집기 초기화
        collector = AlphaVantageDataCollector()

        # 예제: Apple 주식 데이터 수집 (15분봉, 최근 30일)
        symbol = "AAPL"
        interval = "15min"  # 15분봉

        print(f"{symbol} 주식 데이터를 수집합니다...")

        # 데이터 수집 및 저장
        filepath = collector.collect_and_save_intraday(
            symbol=symbol, interval=interval, outputsize="full"  # 최근 30일 데이터
        )

        print(f"데이터가 성공적으로 저장되었습니다: {filepath}")

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
