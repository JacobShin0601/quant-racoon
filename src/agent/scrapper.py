import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from actions.y_finance import YahooFinanceDataCollector
from actions.calculate_index import TechnicalIndicators, StrategyParams
from agent.helper import Logger, load_config, print_section_header, print_subsection_header


def main():
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="데이터 수집 시스템")
    parser.add_argument("--data-dir", default="data", help="데이터 저장 디렉토리")
    parser.add_argument("--config", default="../../config/config_default.json", help="설정 파일 경로")
    parser.add_argument("--uuid", help="실행 UUID")
    args = parser.parse_args()
    
    # config.json 경로
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    
    # Logger 초기화 - config에서 로그 디렉토리 가져오기
    log_dir = config.get("output", {}).get("logs_folder", "log")
    logger = Logger(log_dir=log_dir)
    
    # UUID 설정
    if args.uuid:
        print(f"🆔 스크래퍼 UUID 설정: {args.uuid}")

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
                df = collector.get_candle_data(
                    symbol=symbol,
                    interval=common_settings.get("interval", "15m"),
                    start_date=common_settings.get("start_date"),
                    end_date=common_settings.get("end_date"),
                    days_back=common_settings.get("lookback_days", common_settings.get("days_back", 30)),
                )
                logger.log_success(
                    f"{symbol} 기본 데이터 수집 완료 ({len(df)}개 포인트)"
                )
                
                # 수집된 재무지표 정보 로깅
                financial_columns = [col for col in df.columns if col not in [
                    "datetime", "date", "time", "timestamp", "open", "high", "low", "close", "volume"
                ]]
                if financial_columns:
                    logger.log_info(f"{symbol} 재무지표 {len(financial_columns)}개 수집됨")
                    # 주요 재무지표들만 로깅
                    key_indicators = ["pe_ratio", "return_on_equity", "debt_to_equity", "dividend_yield", 
                                    "free_cashflow", "market_cap", "beta"]
                    available_indicators = [ind for ind in key_indicators if ind in df.columns and df[ind].iloc[0] is not None]
                    if available_indicators:
                        indicator_values = {ind: df[ind].iloc[0] for ind in available_indicators}
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
                
                # 수집된 재무지표 정보 로깅
                financial_columns = [col for col in df.columns if col not in [
                    "datetime", "date", "time", "timestamp", "open", "high", "low", "close", "volume"
                ]]
                if financial_columns:
                    logger.log_info(f"{symbol} 재무지표 {len(financial_columns)}개 수집됨")
                    # 주요 재무지표들만 로깅
                    key_indicators = ["pe_ratio", "return_on_equity", "debt_to_equity", "dividend_yield", 
                                    "free_cashflow", "market_cap", "beta"]
                    available_indicators = [ind for ind in key_indicators if ind in df.columns and df[ind].iloc[0] is not None]
                    if available_indicators:
                        indicator_values = {ind: df[ind].iloc[0] for ind in available_indicators}
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

    logger.log_success("🎉 모든 데이터 수집이 완료되었습니다!")

    # JSON 로그 저장
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "symbols": all_symbols,
        "common_settings": common_settings,
        "custom_tasks": custom_tasks,
        "total_symbols": len(all_symbols),
        "uuid": args.uuid,
        "financial_analysis": {
            "description": "포괄적인 재무분석을 위한 확장된 지표들 포함",
            "categories": [
                "기업 가치 지표 (P/E, P/B, EV/EBITDA 등)",
                "수익성 지표 (ROE, ROA, 마진 등)",
                "성장성 지표 (매출성장률, 이익성장률 등)",
                "재무 건전성 지표 (부채비율, 유동비율 등)",
                "현금흐름 지표 (영업현금흐름, 자유현금흐름 등)",
                "배당 관련 지표 (배당수익률, 배당성향 등)",
                "분기별 재무제표 데이터",
                "계산된 재무비율들"
            ]
        }
    }
    logger.save_json_log(
        log_data, f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )


if __name__ == "__main__":
    main()
