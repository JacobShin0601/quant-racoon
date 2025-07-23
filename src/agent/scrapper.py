import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from actions.y_finance import YahooFinanceDataCollector
from actions.calculate_index import TechnicalIndicators, StrategyParams
from agent.helper import (
    Logger,
    load_config,
    print_section_header,
    print_subsection_header,
)


class DataScrapper:
    """데이터 수집 클래스 - orchestrator에서 사용"""

    def __init__(
        self,
        config_path: str = "config/config_default.json",
        time_horizon: str = "swing",
        uuid: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.config_path = config_path
        self.time_horizon = time_horizon
        self.uuid = uuid
        self.end_date = end_date
        self.config = load_config(config_path)
        self.logger = Logger()

        # 데이터 디렉토리 설정
        self.data_dir = f"data/{time_horizon}"
        self.log_dir = f"log/{time_horizon}"

        # 로거 설정
        if self.log_dir:
            self.logger.set_log_dir(self.log_dir)

        if self.uuid:
            self.logger.setup_logger(
                strategy="data_collection", mode="scrapper", uuid=self.uuid
            )
        else:
            # UUID가 없어도 기본 로거 설정
            self.logger.setup_logger(
                strategy="data_collection", mode="scrapper"
            )

    def run_scrapper(self) -> bool:
        """데이터 수집 실행"""
        try:
            print_subsection_header("📊 데이터 수집 시작")

            # 설정에서 심볼과 설정 가져오기 (data.symbols와 scrapper.symbols 둘 다 확인)
            data_config = self.config.get("data", {})
            scrapper_config = self.config.get("scrapper", {})
            
            # 심볼 우선순위: scrapper.symbols > data.symbols
            symbols = scrapper_config.get("symbols", data_config.get("symbols", []))
            custom_tasks = data_config.get("custom_tasks", [])

            if not symbols and not custom_tasks:
                print("❌ 수집할 심볼이 설정되지 않았습니다.")
                print(f"   설정 파일에서 확인된 내용:")
                print(f"   - data.symbols: {data_config.get('symbols', [])}")
                print(f"   - scrapper.symbols: {scrapper_config.get('symbols', [])}")
                print(f"   - custom_tasks: {custom_tasks}")
                return False

            # 공통 설정 (scrapper 설정 우선, 없으면 data 설정 사용)
            common_settings = scrapper_config if scrapper_config else data_config.get("common_settings", data_config)

            # 데이터 수집기 초기화
            collector = YahooFinanceDataCollector()
            params = StrategyParams()

            success_count = 0
            total_symbols = len(symbols) + len(custom_tasks)

            # 1. 공통 설정을 적용한 symbols 처리
            if symbols:
                print(f"📈 {len(symbols)}개 종목 데이터 수집 중...")

                for symbol in symbols:
                    try:
                        print(f"  🔍 {symbol} 데이터 수집 중...")

                        # 종목 정보 가져오기
                        info = collector.get_stock_info(symbol)
                        print(f"    📋 {info['name']} ({info['sector']})")

                        # 기본 데이터 수집
                        # end_date 우선순위: 인스턴스 변수 > 설정 파일 > None (오늘 날짜)
                        effective_end_date = self.end_date or data_config.get(
                            "end_date"
                        )
                        df = collector.get_candle_data(
                            symbol=symbol,
                            interval=common_settings.get("interval", "60m"),
                            start_date=common_settings.get("start_date"),
                            end_date=effective_end_date,
                            days_back=common_settings.get("lookback_days", 60),
                        )

                        if df is not None and not df.empty:
                            # 기술적 지표 계산
                            df_with_indicators = (
                                TechnicalIndicators.calculate_all_indicators(df, params)
                            )

                            # datetime 컬럼 보장
                            if "datetime" not in df_with_indicators.columns:
                                df_with_indicators = df_with_indicators.reset_index()

                            # CSV 파일로 저장
                            filepath = collector.save_to_csv(
                                df=df_with_indicators,
                                symbol=symbol,
                                interval=common_settings.get("interval", "60m"),
                                start_date=common_settings.get("start_date") or "auto",
                                end_date=common_settings.get("end_date") or "auto",
                                output_dir=self.data_dir,
                                uuid=self.uuid,
                            )

                            print(
                                f"    ✅ {symbol} 데이터 저장 완료: {len(df)}개 포인트"
                            )
                            success_count += 1
                        else:
                            print(f"    ❌ {symbol} 데이터 수집 실패")

                    except Exception as e:
                        print(f"    ❌ {symbol} 처리 중 오류: {e}")
                        continue

            # 2. 개별 설정이 있는 custom_tasks 처리
            if custom_tasks:
                print(f"📈 {len(custom_tasks)}개 개별 설정 종목 처리 중...")

                for task in custom_tasks:
                    symbol = task.get("symbol")
                    try:
                        print(f"  🔍 {symbol} 데이터 수집 중 (개별 설정)...")

                        # 종목 정보 가져오기
                        info = collector.get_stock_info(symbol)
                        print(f"    📋 {info['name']} ({info['sector']})")

                        # 개별 설정으로 데이터 수집
                        # end_date 우선순위: 인스턴스 변수 > 개별 설정 > None (오늘 날짜)
                        effective_end_date = self.end_date or task.get("end_date")
                        df = collector.get_candle_data(
                            symbol=symbol,
                            interval=task.get("interval", "60m"),
                            start_date=task.get("start_date"),
                            end_date=effective_end_date,
                            days_back=task.get("days_back", 60),
                        )

                        if df is not None and not df.empty:
                            # 기술적 지표 계산
                            df_with_indicators = (
                                TechnicalIndicators.calculate_all_indicators(df, params)
                            )

                            # datetime 컬럼 보장
                            if "datetime" not in df_with_indicators.columns:
                                df_with_indicators = df_with_indicators.reset_index()

                            # CSV 파일로 저장
                            filepath = collector.save_to_csv(
                                df=df_with_indicators,
                                symbol=symbol,
                                interval=task.get("interval", "60m"),
                                start_date=task.get("start_date") or "auto",
                                end_date=task.get("end_date") or "auto",
                                output_dir=self.data_dir,
                                uuid=self.uuid,
                            )

                            print(
                                f"    ✅ {symbol} 데이터 저장 완료: {len(df)}개 포인트"
                            )
                            success_count += 1
                        else:
                            print(f"    ❌ {symbol} 데이터 수집 실패")

                    except Exception as e:
                        print(f"    ❌ {symbol} 처리 중 오류: {e}")
                        continue

            print(f"✅ 데이터 수집 완료: {success_count}/{total_symbols}개 종목 성공")
            return success_count > 0

        except Exception as e:
            print(f"❌ 데이터 수집 중 오류: {e}")
            return False


def main():
    import argparse

    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="데이터 수집 시스템")
    parser.add_argument("--data-dir", default="data", help="데이터 저장 디렉토리")
    parser.add_argument(
        "--config", default="../../config/config_default.json", help="설정 파일 경로"
    )
    parser.add_argument("--log-dir", help="로그 디렉토리")
    parser.add_argument("--uuid", help="실행 UUID")
    parser.add_argument(
        "--end-date", help="데이터 수집 종료 날짜 (YYYY-MM-DD 형식, 기본값: 오늘 날짜)"
    )
    args = parser.parse_args()

    # config.json 경로
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)

    # Logger 초기화 - 명령행 인자 우선, 없으면 config에서 로그 디렉토리 가져오기
    log_dir = (
        args.log_dir
        if args.log_dir
        else config.get("output", {}).get("logs_folder", "log")
    )
    logger = Logger(log_dir=log_dir)

    # UUID 설정
    if args.uuid:
        print(f"🆔 스크래퍼 UUID 설정: {args.uuid}")

    # 종료 날짜 설정
    if args.end_date:
        print(f"📅 데이터 수집 종료 날짜 설정: {args.end_date}")
    else:
        print(f"📅 데이터 수집 종료 날짜: 오늘 날짜 (기본값)")

    # 공통 설정 가져오기
    data_config = config.get("data", {})
    # data 섹션에서 직접 설정을 가져오거나, common_settings가 있으면 그것을 사용
    common_settings = data_config.get("common_settings", data_config)
    symbols = data_config.get("symbols", [])
    custom_tasks = data_config.get("custom_tasks", [])

    if not symbols and not custom_tasks:
        print("수집할 작업이 없습니다.")
        return

    # 로거 설정
    all_symbols = symbols + [
        task.get("symbol") for task in custom_tasks if task.get("symbol")
    ]
    logger.setup_logger(
        strategy="data_collection", symbols=all_symbols, mode="scrapper"
    )

    logger.log_section("📊 데이터 수집 시스템 시작")

    collector = YahooFinanceDataCollector()
    params = StrategyParams()

    # 1. 공통 설정을 적용한 symbols 처리
    if symbols:
        logger.log_subsection(f"공통 설정으로 {len(symbols)}개 종목 처리")
        logger.log_config(common_settings, "공통 설정")

        for symbol in symbols:
            logger.log_subsection(f"{symbol} 데이터 수집 시작")
            try:
                info = collector.get_stock_info(symbol)
                logger.log_info(f"주식 정보: {info['name']} ({info['sector']})")
            except Exception as e:
                logger.log_error(f"{symbol} 종목 정보를 가져올 수 없습니다: {e}")

            try:
                # 1단계: 기본 데이터 수집 (재무지표 포함)
                logger.log_info(f"{symbol} 기본 데이터 수집 중...")
                # end_date 우선순위: 명령행 인자 > 설정 파일 > None (오늘 날짜)
                effective_end_date = args.end_date or data_config.get("end_date")
                df = collector.get_candle_data(
                    symbol=symbol,
                    interval=common_settings.get("interval", "15m"),
                    start_date=common_settings.get("start_date"),
                    end_date=effective_end_date,
                    days_back=common_settings.get(
                        "lookback_days", common_settings.get("days_back", 30)
                    ),
                )
                logger.log_success(
                    f"{symbol} 기본 데이터 수집 완료 ({len(df)}개 포인트)"
                )

                # 수집된 재무지표 정보 로깅
                financial_columns = [
                    col
                    for col in df.columns
                    if col
                    not in [
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
                if financial_columns:
                    logger.log_info(
                        f"{symbol} 재무지표 {len(financial_columns)}개 수집됨"
                    )
                    # 주요 재무지표들만 로깅
                    key_indicators = [
                        "pe_ratio",
                        "return_on_equity",
                        "debt_to_equity",
                        "dividend_yield",
                        "free_cashflow",
                        "market_cap",
                        "beta",
                    ]
                    available_indicators = [
                        ind
                        for ind in key_indicators
                        if ind in df.columns and df[ind].iloc[0] is not None
                    ]
                    if available_indicators:
                        indicator_values = {
                            ind: df[ind].iloc[0] for ind in available_indicators
                        }
                        logger.log_info(f"{symbol} 주요 재무지표: {indicator_values}")

                # 2단계: 기술적 지표 계산
                logger.log_info(f"{symbol} 기술적 지표 계산 중...")
                df_with_indicators = TechnicalIndicators.calculate_all_indicators(
                    df, params
                )
                logger.log_success(f"{symbol} 기술적 지표 계산 완료")

                # 3단계: CSV 파일로 저장
                logger.log_info(f"{symbol} CSV 파일 저장 중...")
                filepath = collector.save_to_csv(
                    df=df_with_indicators,
                    symbol=symbol,
                    interval=common_settings.get("interval", "15m"),
                    start_date=common_settings.get("start_date") or "auto",
                    end_date=common_settings.get("end_date") or "auto",
                    output_dir=args.data_dir,
                    uuid=args.uuid,  # UUID 추가
                )
                logger.log_success(f"{symbol} 데이터가 저장되었습니다: {filepath}")

            except Exception as e:
                logger.log_error(f"{symbol} 데이터 수집 실패: {e}")

    # 2. 개별 설정이 있는 custom_tasks 처리
    if custom_tasks:
        logger.log_subsection(f"개별 설정으로 {len(custom_tasks)}개 종목 처리")

        for task in custom_tasks:
            symbol = task.get("symbol")
            logger.log_subsection(f"{symbol} 데이터 수집 시작 (개별 설정)")
            logger.log_config(task, f"{symbol} 개별 설정")

            try:
                info = collector.get_stock_info(symbol)
                logger.log_info(f"주식 정보: {info['name']} ({info['sector']})")
            except Exception as e:
                logger.log_error(f"{symbol} 종목 정보를 가져올 수 없습니다: {e}")

            try:
                # 1단계: 기본 데이터 수집 (재무지표 포함)
                logger.log_info(f"{symbol} 기본 데이터 수집 중...")
                # end_date 우선순위: 명령행 인자 > 개별 설정 > None (오늘 날짜)
                effective_end_date = args.end_date or task.get("end_date")
                df = collector.get_candle_data(
                    symbol=symbol,
                    interval=task.get("interval", "15m"),
                    start_date=task.get("start_date"),
                    end_date=effective_end_date,
                    days_back=task.get("days_back", 30),
                )
                logger.log_success(
                    f"{symbol} 기본 데이터 수집 완료 ({len(df)}개 포인트)"
                )

                # 수집된 재무지표 정보 로깅
                financial_columns = [
                    col
                    for col in df.columns
                    if col
                    not in [
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
                if financial_columns:
                    logger.log_info(
                        f"{symbol} 재무지표 {len(financial_columns)}개 수집됨"
                    )
                    # 주요 재무지표들만 로깅
                    key_indicators = [
                        "pe_ratio",
                        "return_on_equity",
                        "debt_to_equity",
                        "dividend_yield",
                        "free_cashflow",
                        "market_cap",
                        "beta",
                    ]
                    available_indicators = [
                        ind
                        for ind in key_indicators
                        if ind in df.columns and df[ind].iloc[0] is not None
                    ]
                    if available_indicators:
                        indicator_values = {
                            ind: df[ind].iloc[0] for ind in available_indicators
                        }
                        logger.log_info(f"{symbol} 주요 재무지표: {indicator_values}")

                # 2단계: 기술적 지표 계산
                logger.log_info(f"{symbol} 기술적 지표 계산 중...")
                df_with_indicators = TechnicalIndicators.calculate_all_indicators(
                    df, params
                )
                logger.log_success(f"{symbol} 기술적 지표 계산 완료")

                # 3단계: CSV 파일로 저장
                logger.log_info(f"{symbol} CSV 파일 저장 중...")
                filepath = collector.save_to_csv(
                    df=df_with_indicators,
                    symbol=symbol,
                    interval=task.get("interval", "15m"),
                    start_date=task.get("start_date") or "auto",
                    end_date=task.get("end_date") or "auto",
                    output_dir=args.data_dir,
                    uuid=args.uuid,  # UUID 추가
                )
                logger.log_success(f"{symbol} 데이터가 저장되었습니다: {filepath}")

            except Exception as e:
                logger.log_error(f"{symbol} 데이터 수집 실패: {e}")

    logger.log_section("📊 데이터 수집 시스템 완료")


if __name__ == "__main__":
    main()
