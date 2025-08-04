import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import sys
import argparse
import json

# yfinance 디버그 로그 억제
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)


class YahooFinanceDataCollector:
    """
    Yahoo Finance API를 사용하여 주식 데이터를 수집하고 CSV로 저장하는 클래스
    """

    def __init__(self):
        """Yahoo Finance API 클라이언트 초기화"""
        # Yahoo Finance는 API 키가 필요 없음

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_candle_data(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Yahoo Finance API에서 캔들스틱 데이터를 가져옵니다.

        Args:
            symbol (str): 주식 티커 (예: 'AAPL', 'MSFT', 'CONL')
            interval (str): 시간 단위 ('1m', '2m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
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

            # 기간 계산 (Yahoo Finance 형식)
            period = self._calculate_period(start_date, end_date, interval)

            self.logger.info(
                f"{symbol} 데이터 수집 중... (기간: {start_date.date()} ~ {end_date.date()}, interval: {interval})"
            )

            # Yahoo Finance 티커 객체 생성
            ticker = yf.Ticker(symbol)

            # 데이터 수집
            if start_date and end_date:
                # 특정 기간 데이터
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True,
                )
            else:
                # 기간 기반 데이터
                df = ticker.history(
                    period=period, interval=interval, auto_adjust=True, prepost=True
                )

            if df.empty:
                raise ValueError(f"{symbol} 종목에 대한 데이터를 찾을 수 없습니다.")

            # 컬럼명 표준화
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # 인덱스를 datetime 컬럼으로 변환
            df = df.reset_index()
            # Date 컬럼이 있는지 확인하고 이름 변경
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "datetime"})
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "datetime"})
            else:
                # 인덱스가 이미 datetime인 경우
                df["datetime"] = df.index

            # 추가 컬럼 생성
            df["date"] = df["datetime"].dt.date
            df["time"] = df["datetime"].dt.time
            df["timestamp"] = df["datetime"].astype("int64") // 10**9  # Unix timestamp

            # 재무지표 수집 및 추가
            df = self._add_financial_indicators(df, ticker, symbol)

            # 컬럼 순서 재정렬 (기본 컬럼들 먼저, 그 다음 재무지표들)
            basic_columns = [
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

            # 재무지표 컬럼들
            financial_columns = [col for col in df.columns if col not in basic_columns]

            # 최종 컬럼 순서
            final_columns = basic_columns + sorted(financial_columns)
            df = df[final_columns]

            # 시간순 정렬
            df = df.sort_values("datetime").reset_index(drop=True)

            self.logger.info(f"{len(df)}개의 데이터 포인트를 수집했습니다.")

            return df

        except Exception as e:
            self.logger.error(f"데이터 처리 오류: {e}")
            raise

    def _add_financial_indicators(
        self, df: pd.DataFrame, ticker: yf.Ticker, symbol: str
    ) -> pd.DataFrame:
        """
        재무지표를 수집하여 DataFrame에 추가합니다.

        Args:
            df (pd.DataFrame): 기본 캔들스틱 데이터
            ticker (yf.Ticker): Yahoo Finance 티커 객체
            symbol (str): 주식 심볼

        Returns:
            pd.DataFrame: 재무지표가 추가된 DataFrame
        """
        try:
            self.logger.info(f"{symbol} 재무지표 수집 중...")

            # 재무지표 수집
            financial_data = {}

            # 1. 기본 재무정보 (포괄적인 재무분석을 위한 확장된 지표들)
            try:
                info = ticker.info
                if info:
                    # 기업 가치 지표
                    financial_data.update(
                        {
                            "market_cap": info.get("marketCap", None),
                            "enterprise_value": info.get("enterpriseValue", None),
                            "pe_ratio": info.get("trailingPE", None),
                            "forward_pe": info.get("forwardPE", None),
                            "peg_ratio": info.get("pegRatio", None),
                            "price_to_book": info.get("priceToBook", None),
                            "price_to_sales": info.get(
                                "priceToSalesTrailing12Months", None
                            ),
                            "ev_to_ebitda": info.get("enterpriseToEbitda", None),
                            "ev_to_revenue": info.get("enterpriseToRevenue", None),
                            "price_to_cashflow": info.get("priceToCashflow", None),
                            "price_to_free_cashflow": info.get(
                                "priceToFreeCashflow", None
                            ),
                        }
                    )

                    # 수익성 지표
                    financial_data.update(
                        {
                            "return_on_equity": info.get("returnOnEquity", None),
                            "return_on_assets": info.get("returnOnAssets", None),
                            "return_on_capital": info.get("returnOnCapital", None),
                            "return_on_invested_capital": info.get(
                                "returnOnInvestedCapital", None
                            ),
                            "profit_margin": info.get("profitMargins", None),
                            "operating_margin": info.get("operatingMargins", None),
                            "gross_margin": info.get("grossMargins", None),
                            "ebitda_margin": info.get("ebitdaMargins", None),
                            "net_income_margin": info.get("netIncomeToCommon", None),
                        }
                    )

                    # 성장성 지표
                    financial_data.update(
                        {
                            "revenue_growth": info.get("revenueGrowth", None),
                            "earnings_growth": info.get("earningsGrowth", None),
                            "earnings_quarterly_growth": info.get(
                                "earningsQuarterlyGrowth", None
                            ),
                            "revenue_quarterly_growth": info.get(
                                "revenueQuarterlyGrowth", None
                            ),
                            "earnings_annual_growth": info.get(
                                "earningsAnnualGrowth", None
                            ),
                            "revenue_annual_growth": info.get(
                                "revenueAnnualGrowth", None
                            ),
                            "revenue_per_employee": info.get(
                                "revenuePerEmployee", None
                            ),
                            "revenue_per_share": info.get("revenuePerShare", None),
                        }
                    )

                    # 재무 건전성 지표
                    financial_data.update(
                        {
                            "debt_to_equity": info.get("debtToEquity", None),
                            "debt_to_assets": info.get("debtToAssets", None),
                            "current_ratio": info.get("currentRatio", None),
                            "quick_ratio": info.get("quickRatio", None),
                            "cash_ratio": info.get("cashRatio", None),
                            "interest_coverage": info.get("interestCoverage", None),
                            "total_cash": info.get("totalCash", None),
                            "total_debt": info.get("totalDebt", None),
                            "net_debt": info.get("netDebt", None),
                            "cash_per_share": info.get("totalCashPerShare", None),
                            "book_value": info.get("bookValue", None),
                            "tangible_book_value": info.get("tangibleBookValue", None),
                        }
                    )

                    # 현금흐름 지표
                    financial_data.update(
                        {
                            "operating_cashflow": info.get("operatingCashflow", None),
                            "free_cashflow": info.get("freeCashflow", None),
                            "free_cashflow_yield": info.get("freeCashflowYield", None),
                            "operating_cashflow_per_share": info.get(
                                "operatingCashflowPerShare", None
                            ),
                            "free_cashflow_per_share": info.get(
                                "freeCashflowPerShare", None
                            ),
                            "cashflow_to_debt": info.get("cashflowToDebt", None),
                        }
                    )

                    # 배당 관련 지표
                    financial_data.update(
                        {
                            "dividend_yield": info.get("dividendYield", None),
                            "dividend_rate": info.get("dividendRate", None),
                            "payout_ratio": info.get("payoutRatio", None),
                            "dividend_payout_ratio": info.get(
                                "dividendPayoutRatio", None
                            ),
                            "five_year_avg_dividend_yield": info.get(
                                "fiveYearAvgDividendYield", None
                            ),
                            "forward_dividend_yield": info.get(
                                "forwardDividendYield", None
                            ),
                            "forward_dividend_rate": info.get(
                                "forwardDividendRate", None
                            ),
                        }
                    )

                    # 수익성 지표 (EPS 관련)
                    financial_data.update(
                        {
                            "earnings_ttm": info.get("trailingEps", None),
                            "earnings_forward": info.get("forwardEps", None),
                            "earnings_quarterly": info.get("earningsQuarterly", None),
                            "earnings_annual": info.get("earningsAnnual", None),
                            "eps_ttm": info.get("trailingEps", None),
                            "eps_forward": info.get("forwardEps", None),
                            "eps_quarterly": info.get("earningsQuarterly", None),
                            "eps_annual": info.get("earningsAnnual", None),
                        }
                    )

                    # 매출 관련 지표
                    financial_data.update(
                        {
                            "total_revenue": info.get("totalRevenue", None),
                            "revenue_ttm": info.get("trailingRevenue", None),
                            "revenue_forward": info.get("forwardRevenue", None),
                            "revenue_quarterly": info.get("revenueQuarterly", None),
                            "revenue_annual": info.get("revenueAnnual", None),
                            "gross_profits": info.get("grossProfits", None),
                            "ebitda": info.get("ebitda", None),
                            "ebit": info.get("ebit", None),
                            "net_income": info.get("netIncomeToCommon", None),
                            "net_income_ttm": info.get("netIncomeToCommon", None),
                        }
                    )

                    # 주식 관련 지표
                    financial_data.update(
                        {
                            "shares_outstanding": info.get("sharesOutstanding", None),
                            "float_shares": info.get("floatShares", None),
                            "shares_short": info.get("sharesShort", None),
                            "shares_short_prior_month": info.get(
                                "sharesShortPriorMonth", None
                            ),
                            "short_ratio": info.get("shortRatio", None),
                            "short_percent_of_float": info.get(
                                "shortPercentOfFloat", None
                            ),
                            "shares_percent_shares_out": info.get(
                                "sharesPercentSharesOut", None
                            ),
                            "held_percent_insiders": info.get(
                                "heldPercentInsiders", None
                            ),
                            "held_percent_institutions": info.get(
                                "heldPercentInstitutions", None
                            ),
                            "institutional_ownership": info.get(
                                "institutionalOwnershipPercentage", None
                            ),
                        }
                    )

                    # 기타 중요 지표
                    financial_data.update(
                        {
                            "beta": info.get("beta", None),
                            "fifty_two_week_change": info.get(
                                "fiftyTwoWeekChange", None
                            ),
                            "fifty_day_average": info.get("fiftyDayAverage", None),
                            "two_hundred_day_average": info.get(
                                "twoHundredDayAverage", None
                            ),
                            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
                            "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
                            "day_high": info.get("dayHigh", None),
                            "day_low": info.get("dayLow", None),
                            "volume": info.get("volume", None),
                            "average_volume": info.get("averageVolume", None),
                            "market_cap": info.get("marketCap", None),
                            "enterprise_value": info.get("enterpriseValue", None),
                        }
                    )
            except Exception as e:
                self.logger.warning(f"{symbol} 기본 재무정보 수집 실패: {e}")

            # 2. 분기별 재무정보 (포괄적인 재무분석을 위한 확장)
            try:
                # 분기별 손익계산서
                financials = ticker.financials
                if not financials.empty:
                    latest_quarter = financials.columns[0]
                    quarter_data = financials[latest_quarter]

                    financial_data.update(
                        {
                            "quarterly_revenue": quarter_data.get(
                                "Total Revenue", None
                            ),
                            "quarterly_net_income": quarter_data.get(
                                "Net Income", None
                            ),
                            "quarterly_operating_income": quarter_data.get(
                                "Operating Income", None
                            ),
                            "quarterly_ebitda": quarter_data.get("EBITDA", None),
                            "quarterly_eps": quarter_data.get("Basic EPS", None),
                            "quarterly_gross_profit": quarter_data.get(
                                "Gross Profit", None
                            ),
                            "quarterly_ebit": quarter_data.get("EBIT", None),
                            "quarterly_operating_expense": quarter_data.get(
                                "Operating Expense", None
                            ),
                            "quarterly_research_development": quarter_data.get(
                                "Research And Development", None
                            ),
                            "quarterly_selling_general_admin": quarter_data.get(
                                "Selling General And Administration", None
                            ),
                        }
                    )

                # 분기별 대차대조표
                balance_sheet = ticker.balance_sheet
                if not balance_sheet.empty:
                    latest_quarter = balance_sheet.columns[0]
                    quarter_data = balance_sheet[latest_quarter]

                    financial_data.update(
                        {
                            "quarterly_total_assets": quarter_data.get(
                                "Total Assets", None
                            ),
                            "quarterly_total_liabilities": quarter_data.get(
                                "Total Liabilities Net Minority Interest", None
                            ),
                            "quarterly_total_equity": quarter_data.get(
                                "Total Equity Gross Minority Interest", None
                            ),
                            "quarterly_cash": quarter_data.get(
                                "Cash and Cash Equivalents", None
                            ),
                            "quarterly_debt": quarter_data.get("Total Debt", None),
                            "quarterly_current_assets": quarter_data.get(
                                "Total Current Assets", None
                            ),
                            "quarterly_current_liabilities": quarter_data.get(
                                "Total Current Liabilities", None
                            ),
                            "quarterly_inventory": quarter_data.get("Inventory", None),
                            "quarterly_accounts_receivable": quarter_data.get(
                                "Accounts Receivable", None
                            ),
                            "quarterly_accounts_payable": quarter_data.get(
                                "Accounts Payable", None
                            ),
                            "quarterly_short_term_debt": quarter_data.get(
                                "Short Term Debt", None
                            ),
                            "quarterly_long_term_debt": quarter_data.get(
                                "Long Term Debt", None
                            ),
                            "quarterly_goodwill": quarter_data.get("Goodwill", None),
                            "quarterly_intangible_assets": quarter_data.get(
                                "Intangible Assets", None
                            ),
                            "quarterly_property_plant_equipment": quarter_data.get(
                                "Property Plant Equipment Net", None
                            ),
                        }
                    )

                # 분기별 현금흐름표 (새로 추가)
                cashflow = ticker.cashflow
                if not cashflow.empty:
                    latest_quarter = cashflow.columns[0]
                    quarter_data = cashflow[latest_quarter]

                    financial_data.update(
                        {
                            "quarterly_operating_cashflow": quarter_data.get(
                                "Operating Cash Flow", None
                            ),
                            "quarterly_investing_cashflow": quarter_data.get(
                                "Investing Cash Flow", None
                            ),
                            "quarterly_financing_cashflow": quarter_data.get(
                                "Financing Cash Flow", None
                            ),
                            "quarterly_free_cashflow": quarter_data.get(
                                "Free Cash Flow", None
                            ),
                            "quarterly_capital_expenditure": quarter_data.get(
                                "Capital Expenditure", None
                            ),
                            "quarterly_dividends_paid": quarter_data.get(
                                "Dividends Paid", None
                            ),
                            "quarterly_net_income_cashflow": quarter_data.get(
                                "Net Income", None
                            ),
                            "quarterly_depreciation": quarter_data.get(
                                "Depreciation", None
                            ),
                            "quarterly_change_in_cash": quarter_data.get(
                                "Change In Cash", None
                            ),
                            "quarterly_change_in_receivables": quarter_data.get(
                                "Change In Receivables", None
                            ),
                            "quarterly_change_in_inventory": quarter_data.get(
                                "Change In Inventory", None
                            ),
                            "quarterly_change_in_payables": quarter_data.get(
                                "Change In Payables", None
                            ),
                        }
                    )

            except Exception as e:
                self.logger.warning(f"{symbol} 분기별 재무정보 수집 실패: {e}")

            # 3. 배당 및 기업 행동 데이터 (새로 추가)
            try:
                # 배당 데이터
                dividends = ticker.dividends
                if not dividends.empty:
                    latest_dividend = dividends.iloc[-1] if len(dividends) > 0 else None
                    financial_data.update(
                        {
                            "latest_dividend_amount": latest_dividend,
                            "dividend_frequency": (
                                len(dividends) / 4 if len(dividends) > 0 else 0
                            ),  # 연간 배당 횟수 추정
                        }
                    )

                # 주식 분할 데이터
                splits = ticker.splits
                if not splits.empty:
                    latest_split = splits.iloc[-1] if len(splits) > 0 else None
                    financial_data.update(
                        {
                            "latest_split_ratio": latest_split,
                            "split_frequency": len(splits),  # 분할 횟수
                        }
                    )

                # 자본 이득 데이터
                capital_gains = ticker.capital_gains
                if not capital_gains.empty:
                    latest_capital_gain = (
                        capital_gains.iloc[-1] if len(capital_gains) > 0 else None
                    )
                    financial_data.update(
                        {
                            "latest_capital_gain": latest_capital_gain,
                        }
                    )

            except Exception as e:
                self.logger.warning(f"{symbol} 배당 및 기업 행동 데이터 수집 실패: {e}")

            # 4. 재무지표를 DataFrame에 추가
            for key, value in financial_data.items():
                if value is not None:
                    df[key] = value

            # 5. 계산된 재무비율 추가 (새로 추가)
            try:
                # ROE (Return on Equity) 계산
                if financial_data.get("net_income") and financial_data.get(
                    "book_value"
                ):
                    df["calculated_roe"] = (
                        financial_data["net_income"] / financial_data["book_value"]
                    )

                # ROA (Return on Assets) 계산
                if financial_data.get("net_income") and financial_data.get(
                    "quarterly_total_assets"
                ):
                    df["calculated_roa"] = (
                        financial_data["net_income"]
                        / financial_data["quarterly_total_assets"]
                    )

                # 부채비율 계산
                if financial_data.get("total_debt") and financial_data.get(
                    "quarterly_total_assets"
                ):
                    df["calculated_debt_to_assets"] = (
                        financial_data["total_debt"]
                        / financial_data["quarterly_total_assets"]
                    )

                # 유동비율 계산
                if financial_data.get(
                    "quarterly_current_assets"
                ) and financial_data.get("quarterly_current_liabilities"):
                    df["calculated_current_ratio"] = (
                        financial_data["quarterly_current_assets"]
                        / financial_data["quarterly_current_liabilities"]
                    )

                # 영업이익률 계산
                if financial_data.get(
                    "quarterly_operating_income"
                ) and financial_data.get("quarterly_revenue"):
                    df["calculated_operating_margin"] = (
                        financial_data["quarterly_operating_income"]
                        / financial_data["quarterly_revenue"]
                    )

                # 순이익률 계산
                if financial_data.get("quarterly_net_income") and financial_data.get(
                    "quarterly_revenue"
                ):
                    df["calculated_net_margin"] = (
                        financial_data["quarterly_net_income"]
                        / financial_data["quarterly_revenue"]
                    )

                # EBITDA 마진 계산
                if financial_data.get("quarterly_ebitda") and financial_data.get(
                    "quarterly_revenue"
                ):
                    df["calculated_ebitda_margin"] = (
                        financial_data["quarterly_ebitda"]
                        / financial_data["quarterly_revenue"]
                    )

                # 자본회전율 계산
                if financial_data.get("quarterly_revenue") and financial_data.get(
                    "quarterly_total_assets"
                ):
                    df["calculated_asset_turnover"] = (
                        financial_data["quarterly_revenue"]
                        / financial_data["quarterly_total_assets"]
                    )

                # 재고회전율 계산
                if financial_data.get("quarterly_revenue") and financial_data.get(
                    "quarterly_inventory"
                ):
                    df["calculated_inventory_turnover"] = (
                        financial_data["quarterly_revenue"]
                        / financial_data["quarterly_inventory"]
                    )

                # 매출채권회전율 계산
                if financial_data.get("quarterly_revenue") and financial_data.get(
                    "quarterly_accounts_receivable"
                ):
                    df["calculated_receivables_turnover"] = (
                        financial_data["quarterly_revenue"]
                        / financial_data["quarterly_accounts_receivable"]
                    )

                # 현금흐름 대비 부채비율 계산
                if financial_data.get(
                    "quarterly_operating_cashflow"
                ) and financial_data.get("total_debt"):
                    df["calculated_cashflow_to_debt"] = (
                        financial_data["quarterly_operating_cashflow"]
                        / financial_data["total_debt"]
                    )

                # 자유현금흐름 수익률 계산
                if financial_data.get("quarterly_free_cashflow") and financial_data.get(
                    "market_cap"
                ):
                    df["calculated_fcf_yield"] = (
                        financial_data["quarterly_free_cashflow"]
                        / financial_data["market_cap"]
                    )

                # 배당성향 계산
                if financial_data.get(
                    "quarterly_dividends_paid"
                ) and financial_data.get("quarterly_net_income"):
                    df["calculated_dividend_payout"] = (
                        abs(financial_data["quarterly_dividends_paid"])
                        / financial_data["quarterly_net_income"]
                    )

            except Exception as e:
                self.logger.warning(f"{symbol} 계산된 재무비율 생성 실패: {e}")

            # 6. Forward fill을 사용하여 재무지표 데이터 채우기
            # 재무지표는 분기별로 업데이트되므로 forward fill이 적절
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
                # ffill 후 bfill 적용하여 결측치 처리
                df[financial_columns] = (
                    df[financial_columns].fillna(method="ffill").fillna(method="bfill")
                )
                self.logger.info(
                    f"{symbol} 재무지표 {len(financial_columns)}개 추가 완료 (ffill + bfill 적용)"
                )

            return df

        except Exception as e:
            self.logger.error(f"{symbol} 재무지표 추가 중 오류: {e}")
            return df

    def _calculate_period(
        self, start_date: datetime, end_date: datetime, interval: str
    ) -> str:
        """Yahoo Finance 기간 형식으로 변환"""
        days_diff = (end_date - start_date).days

        if interval in ["1m", "2m", "5m", "15m", "30m"]:
            # 분봉 데이터의 경우 기간 제한이 있음
            if days_diff <= 7:
                return "7d"
            elif days_diff <= 60:
                return "60d"
            else:
                return "60d"  # 최대 60일
        elif interval == "1h":
            if days_diff <= 730:
                return "730d"
            else:
                return "730d"  # 최대 730일
        else:
            # 일봉 이상은 최대 데이터
            return "max"

    def save_to_csv(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        output_dir: str = "data",
        uuid: str = None,
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

            # 파일명 생성 (티커_분봉_기간_UUID.csv)
            interval_name = self._get_interval_name(interval)
            period = f"{start_date}_{end_date}"

            if uuid:
                filename = f"{symbol}_{interval_name}_{period}_{uuid}.csv"
            else:
                today = datetime.now().strftime("%Y%m%d")
                filename = f"{symbol}_{interval_name}_{period}_{today}.csv"
            filepath = os.path.join(output_dir, filename)

            # 저장 전 결측치 처리: ffill 후 bfill 적용
            df = df.fillna(method="ffill").fillna(method="bfill")

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
            "1m": "1min",
            "2m": "2min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1hour",
            "1d": "daily",
            "1wk": "weekly",
            "1mo": "monthly",
        }
        return interval_map.get(interval, interval)

    def collect_and_save(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 30,
        output_dir: str = "data",
        uuid: str = None,
    ) -> str:
        """
        데이터를 수집하고 CSV로 저장하는 통합 메서드

        Args:
            symbol (str): 주식 티커
            interval (str): 시간 단위
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            days_back (int): 과거 일수 (start_date가 None일 때)
            output_dir (str): 출력 디렉토리

        Returns:
            str: 저장된 파일 경로
        """
        # 데이터 수집
        df = self.get_candle_data(symbol, interval, start_date, end_date, days_back)

        # 날짜 범위 계산
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # CSV 저장
        filepath = self.save_to_csv(
            df, symbol, interval, start_date, end_date, output_dir, uuid
        )

        return filepath

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        주식 기본 정보를 가져옵니다.

        Args:
            symbol (str): 주식 티커

        Returns:
            Dict[str, Any]: 주식 정보
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # 주요 정보만 추출
            basic_info = {
                "symbol": symbol,
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "current_price": info.get("currentPrice", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "volume": info.get("volume", "N/A"),
                "avg_volume": info.get("averageVolume", "N/A"),
            }

            return basic_info

        except Exception as e:
            self.logger.error(f"주식 정보 조회 오류: {e}")
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Yahoo Finance 데이터 수집기")
    parser.add_argument(
        "--symbol",
        type=str,
        default="CONL",
        help="주식 티커 (예: TSLL, NVDL, PLTR, CONL 등)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        help="캔들 간격 (예: 1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)",
    )
    parser.add_argument(
        "--days_back", type=int, default=30, help="과거 일수 (기본: 30일)"
    )
    parser.add_argument(
        "--start_date", type=str, default=None, help="시작 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, default=None, help="종료 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--from_json",
        type=str,
        default=None,
        help="input.json 파일 경로 (json에서 파라미터 읽기)",
    )
    return parser.parse_args()


def main():
    """메인 실행 함수 - 예제 사용법"""
    try:
        args = parse_args()
        # input.json에서 읽기
        if args.from_json:
            with open(args.from_json, "r") as f:
                config = json.load(f)
            symbol = config.get("symbol", "CONL")
            interval = config.get("interval", "15m")
            days_back = config.get("days_back", 30)
            start_date = config.get("start_date", None)
            end_date = config.get("end_date", None)
        else:
            symbol = args.symbol
            interval = args.interval
            days_back = args.days_back
            start_date = args.start_date
            end_date = args.end_date

        collector = YahooFinanceDataCollector()
        # 주식 데이터 수집 시작
        try:
            info = collector.get_stock_info(symbol)
            # 주식 정보 확인
        except:
            # 종목 정보 오류
            pass
        
        filepath = collector.collect_and_save(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            days_back=days_back,
        )
        # 데이터 저장 완료
    except Exception as e:
        print(f"오류: {e}")


if __name__ == "__main__":
    main()
