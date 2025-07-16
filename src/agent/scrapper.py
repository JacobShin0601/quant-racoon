import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from behavior.y_finance import YahooFinanceDataCollector
from behavior.calculate_index import TechnicalIndicators, StrategyParams
from helper import Logger, load_config, print_section_header, print_subsection_header


def main():
    # Logger 초기화
    logger = Logger()

    # config.json 경로
    config_path = os.path.join(os.path.dirname(__file__), "../../config.json")
    config = load_config(config_path)

    # 공통 설정 가져오기
    data_config = config.get("data", {})
    common_settings = data_config.get("common_settings", {})
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
                df = collector.get_candle_data(
                    symbol=symbol,
                    interval=common_settings.get("interval", "15m"),
                    start_date=common_settings.get("start_date"),
                    end_date=common_settings.get("end_date"),
                    days_back=common_settings.get("days_back", 30),
                )
                logger.log_success(
                    f"{symbol} 기본 데이터 수집 완료 ({len(df)}개 포인트)"
                )

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
                    output_dir="data",
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
                df = collector.get_candle_data(
                    symbol=symbol,
                    interval=task.get("interval", "15m"),
                    start_date=task.get("start_date"),
                    end_date=task.get("end_date"),
                    days_back=task.get("days_back", 30),
                )
                logger.log_success(
                    f"{symbol} 기본 데이터 수집 완료 ({len(df)}개 포인트)"
                )

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
                    output_dir="data",
                )
                logger.log_success(f"{symbol} 데이터가 저장되었습니다: {filepath}")

            except Exception as e:
                logger.log_error(f"{symbol} 데이터 수집 실패: {e}")

    logger.log_success("🎉 모든 데이터 수집이 완료되었습니다!")

    # JSON 로그 저장
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "symbols": all_symbols,
        "common_settings": common_settings,
        "custom_tasks": custom_tasks,
        "total_symbols": len(all_symbols),
    }
    logger.save_json_log(
        log_data, f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )


if __name__ == "__main__":
    main()
