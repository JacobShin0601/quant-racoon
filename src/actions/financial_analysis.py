#!/usr/bin/env python3
"""
재무지표 기반 분석 시스템
기존 분석 액션들을 활용하여 재무지표에 특화된 분석 수행
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from .correlation import CorrelationAnalyzer
from .linear_regression import LinearRegressionAnalyzer
from .lasso_regression import LassoRegressionAnalyzer
from .random_forest import RandomForestAnalyzer
from .multi_layer_perceptron import MLPAnalyzer
from .bayesian_distribution import BayesianDistributionAnalyzer


class FinancialAnalyzer:
    """재무지표 기반 종합 분석기"""

    def __init__(self):
        """재무분석기 초기화"""
        self.correlation_analyzer = CorrelationAnalyzer()
        self.linear_regression_analyzer = LinearRegressionAnalyzer()
        self.lasso_regression_analyzer = LassoRegressionAnalyzer()
        self.random_forest_analyzer = RandomForestAnalyzer()
        self.mlp_analyzer = MLPAnalyzer()
        self.bayesian_analyzer = BayesianDistributionAnalyzer()

    def analyze_comprehensive(
        self, 
        data: pd.DataFrame, 
        target_column: str = "return",
        symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        재무지표 기반 종합 분석 수행
        
        Args:
            data (pd.DataFrame): 재무지표 데이터
            target_column (str): 타겟 컬럼명
            symbol (str): 종목 심볼
            
        Returns:
            Dict[str, Any]: 종합 분석 결과
        """
        results = {}
        
        try:
            # 재무지표 중복 제거 및 시계열 처리
            processed_data = self._process_financial_data(data, symbol)
            
            if processed_data is None or len(processed_data) == 0:
                results["error"] = "처리 가능한 재무지표 데이터가 없습니다."
                return results
            
            # 1. 주요 재무지표 요약
            results["key_metrics"] = self._extract_key_metrics(processed_data, symbol)
            
            # 2. 상관관계 분석 (처리된 데이터 사용)
            results["correlation_analysis"] = self._analyze_correlations(processed_data, target_column)
            
            # 3. 예측 모델 분석 (처리된 데이터 사용)
            results["prediction_models"] = self._run_prediction_models(processed_data, target_column)
            
            # 4. 재무건전성 분석
            results["financial_health"] = self._analyze_financial_health(processed_data, symbol)
            
            # 5. 수익성 분석
            results["profitability_analysis"] = self._analyze_profitability(processed_data, symbol)
            
            # 6. 성장성 분석
            results["growth_analysis"] = self._analyze_growth(processed_data, symbol)
            
            # 7. 배당 분석
            results["dividend_analysis"] = self._analyze_dividends(processed_data, symbol)
            
            # 8. 위험도 분석
            results["risk_analysis"] = self._analyze_risk(processed_data, symbol)
            
            # 9. 시계열 분석 결과 추가
            results["time_series_analysis"] = {
                "original_data_points": len(data),
                "processed_data_points": len(processed_data),
                "data_reduction_ratio": len(processed_data) / len(data) if len(data) > 0 else 0
            }
            
        except Exception as e:
            print(f"재무분석 중 오류 발생: {e}")
            results["error"] = str(e)
            
        return results

    def _extract_key_metrics(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """주요 재무지표 추출"""
        metrics = {}
        
        # 최신 데이터에서 주요 지표 추출
        latest_data = data.iloc[-1] if len(data) > 0 else data.iloc[0]
        
        # 기업 가치 지표
        metrics.update({
            "pe_ratio": latest_data.get("pe_ratio"),
            "forward_pe": latest_data.get("forward_pe"),
            "price_to_book": latest_data.get("price_to_book"),
            "price_to_sales": latest_data.get("price_to_sales"),
            "ev_to_ebitda": latest_data.get("ev_to_ebitda"),
            "market_cap": latest_data.get("market_cap"),
            "enterprise_value": latest_data.get("enterprise_value"),
        })
        
        # 수익성 지표
        metrics.update({
            "roe": latest_data.get("return_on_equity"),
            "roa": latest_data.get("return_on_assets"),
            "profit_margin": latest_data.get("profit_margin"),
            "operating_margin": latest_data.get("operating_margin"),
            "gross_margin": latest_data.get("gross_margin"),
        })
        
        # 재무건전성 지표
        metrics.update({
            "debt_to_equity": latest_data.get("debt_to_equity"),
            "current_ratio": latest_data.get("current_ratio"),
            "quick_ratio": latest_data.get("quick_ratio"),
            "total_cash": latest_data.get("total_cash"),
            "total_debt": latest_data.get("total_debt"),
        })
        
        # 배당 지표
        metrics.update({
            "dividend_yield": latest_data.get("dividend_yield"),
            "payout_ratio": latest_data.get("payout_ratio"),
            "dividend_rate": latest_data.get("dividend_rate"),
        })
        
        # 성장성 지표
        metrics.update({
            "revenue_growth": latest_data.get("revenue_growth"),
            "earnings_growth": latest_data.get("earnings_growth"),
            "beta": latest_data.get("beta"),
        })
        
        # 현금흐름 지표
        metrics.update({
            "free_cashflow": latest_data.get("free_cashflow"),
            "operating_cashflow": latest_data.get("operating_cashflow"),
            "free_cashflow_yield": latest_data.get("free_cashflow_yield"),
        })
        
        return metrics

    def _process_financial_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        재무지표 데이터의 시계열 처리 및 분석용 변환
        
        재무지표는 분기별로 공표되므로, 시계열 분석에 적합하도록 처리합니다.
        절대값보다는 변화율, 추세, 상대적 지표를 계산하여 분석합니다.
        
        Args:
            data (pd.DataFrame): 원본 재무지표 데이터
            symbol (str): 종목 심볼
            
        Returns:
            pd.DataFrame: 처리된 재무지표 데이터
        """
        try:
            # 재무지표 컬럼 식별
            financial_columns = []
            for col in data.columns:
                if any(col.startswith(prefix) for prefix in [
                    "pe_", "market_", "return_on_", "debt_", "current_",
                    "profit_", "operating_", "ebitda_", "revenue_", "earnings_",
                    "dividend_", "payout_", "book_", "cash_", "total_", "quarterly_",
                    "calculated_", "latest_", "beta", "fifty_", "two_hundred_",
                    "shares_", "held_", "institutional_", "short_", "float_"
                ]) or col in [
                    "return_on_equity", "return_on_assets", "return_on_capital",
                    "gross_margin", "net_income_margin", "operating_margin",
                    "ev_to_ebitda", "ev_to_revenue", "price_to_cashflow",
                    "free_cashflow_yield", "interest_coverage", "quick_ratio",
                    "cash_ratio", "net_debt", "tangible_book_value"
                ]:
                    financial_columns.append(col)
            
            if not financial_columns:
                print(f"{symbol}: 재무지표 컬럼을 찾을 수 없습니다.")
                return None
            
            # 기본 컬럼 (항상 유지) - 실제 존재하는 컬럼만 선택
            base_columns = []
            for col in ["datetime", "date", "close", "volume"]:
                if col in data.columns:
                    base_columns.append(col)
            
            if "returns" in data.columns:
                base_columns.append("returns")
            if "return" in data.columns:
                base_columns.append("return")
            
            # 처리할 컬럼들
            process_columns = base_columns + financial_columns
            
            # 필요한 컬럼만 선택
            df = data[process_columns].copy()
            
            # datetime을 인덱스로 설정
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime").sort_index()
            
            # 재무지표 변화율 계산 (시계열 분석용)
            for col in financial_columns:
                if col in df.columns:
                    # 변화율 계산 (이전 값 대비)
                    df[f"{col}_change"] = df[col].pct_change()
                    
                    # 이동평균 (추세 분석용)
                    df[f"{col}_ma_20"] = df[col].rolling(window=20).mean()
                    df[f"{col}_ma_60"] = df[col].rolling(window=60).mean()
                    
                    # 표준화된 값 (Z-score)
                    df[f"{col}_zscore"] = (df[col] - df[col].rolling(window=60).mean()) / df[col].rolling(window=60).std()
            
            # 수익률 컬럼 처리
            if "returns" in df.columns and "return" not in df.columns:
                df["return"] = df["returns"]
            elif "return" not in df.columns and "returns" not in df.columns:
                # 수익률 계산
                df["return"] = df["close"].pct_change() * 100
            
            # NaN 제거 (너무 많은 NaN이 있는 행 제거)
            df = df.dropna(thresh=len(df.columns) * 0.3)  # 30% 이상의 값이 있으면 유지
            
            # 분석 가능한 최소 데이터 포인트 확인
            if len(df) < 10:
                print(f"{symbol}: 분석 가능한 데이터 포인트가 부족합니다 ({len(df)}개)")
                return None
            
            print(f"{symbol}: {len(data)}개 → {len(df)}개 데이터 포인트로 처리됨 "
                  f"(유효 데이터율: {len(df)/len(data)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"{symbol} 재무지표 처리 중 오류: {e}")
            return None

    def _analyze_correlations(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """재무지표와 수익률 간의 상관관계 분석"""
        try:
            result = self.correlation_analyzer.analyze(data, target_column, top_n=15)
            return result
        except Exception as e:
            return {"error": f"상관관계 분석 실패: {e}"}

    def _run_prediction_models(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """예측 모델 실행"""
        results = {}
        
        # 특성 컬럼 선택 (타겟 제외)
        feature_columns = [col for col in data.columns if col != target_column]
        
        if len(feature_columns) == 0:
            return {"error": "분석 가능한 특성이 없습니다."}
        
        try:
            # 데이터 전처리
            X = data[feature_columns].fillna(0)
            y = data[target_column].fillna(0)
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 1. 선형회귀
            try:
                lr_result = self.linear_regression_analyzer.analyze(
                    data, target_column, feature_columns
                )
                results["linear_regression"] = lr_result
            except Exception as e:
                results["linear_regression"] = {"error": str(e)}
            
            # 2. Lasso 회귀
            try:
                lasso_result = self.lasso_regression_analyzer.analyze(
                    data, target_column, feature_columns
                )
                results["lasso_regression"] = lasso_result
            except Exception as e:
                results["lasso_regression"] = {"error": str(e)}
            
            # 3. 랜덤 포레스트
            try:
                rf_result = self.random_forest_analyzer.analyze(
                    data, target_column, feature_columns
                )
                results["random_forest"] = rf_result
            except Exception as e:
                results["random_forest"] = {"error": str(e)}
            
            # 4. MLP
            try:
                mlp_result = self.mlp_analyzer.analyze(
                    data, target_column, feature_columns
                )
                results["mlp"] = mlp_result
            except Exception as e:
                results["mlp"] = {"error": str(e)}
            
            # 5. 베이지안 회귀
            try:
                bayesian_result = self.bayesian_analyzer.analyze_bayesian_regression(
                    data, target_column
                )
                results["bayesian_regression"] = bayesian_result
            except Exception as e:
                results["bayesian_regression"] = {"error": str(e)}
                
        except Exception as e:
            results["error"] = f"예측 모델 실행 실패: {e}"
            
        return results

    def _analyze_financial_health(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """재무건전성 분석"""
        health_metrics = {}
        
        try:
            latest_data = data.iloc[-1] if len(data) > 0 else data.iloc[0]
            
            # 부채비율 분석
            debt_to_equity = latest_data.get("debt_to_equity")
            if debt_to_equity is not None:
                if debt_to_equity < 0.5:
                    health_metrics["debt_health"] = "우수"
                elif debt_to_equity < 1.0:
                    health_metrics["debt_health"] = "양호"
                elif debt_to_equity < 2.0:
                    health_metrics["debt_health"] = "주의"
                else:
                    health_metrics["debt_health"] = "위험"
                health_metrics["debt_to_equity"] = debt_to_equity
            
            # 유동비율 분석
            current_ratio = latest_data.get("current_ratio")
            if current_ratio is not None:
                if current_ratio > 2.0:
                    health_metrics["liquidity_health"] = "우수"
                elif current_ratio > 1.5:
                    health_metrics["liquidity_health"] = "양호"
                elif current_ratio > 1.0:
                    health_metrics["liquidity_health"] = "주의"
                else:
                    health_metrics["liquidity_health"] = "위험"
                health_metrics["current_ratio"] = current_ratio
            
            # 현금흐름 대비 부채비율
            cashflow_to_debt = latest_data.get("cashflow_to_debt")
            if cashflow_to_debt is not None:
                if cashflow_to_debt > 0.5:
                    health_metrics["cashflow_health"] = "우수"
                elif cashflow_to_debt > 0.2:
                    health_metrics["cashflow_health"] = "양호"
                elif cashflow_to_debt > 0.1:
                    health_metrics["cashflow_health"] = "주의"
                else:
                    health_metrics["cashflow_health"] = "위험"
                health_metrics["cashflow_to_debt"] = cashflow_to_debt
            
            # 종합 건전성 점수
            health_scores = []
            if "debt_health" in health_metrics:
                health_scores.append(1 if health_metrics["debt_health"] in ["우수", "양호"] else 0)
            if "liquidity_health" in health_metrics:
                health_scores.append(1 if health_metrics["liquidity_health"] in ["우수", "양호"] else 0)
            if "cashflow_health" in health_metrics:
                health_scores.append(1 if health_metrics["cashflow_health"] in ["우수", "양호"] else 0)
            
            if health_scores:
                overall_score = np.mean(health_scores)
                if overall_score >= 0.8:
                    health_metrics["overall_health"] = "우수"
                elif overall_score >= 0.6:
                    health_metrics["overall_health"] = "양호"
                elif overall_score >= 0.4:
                    health_metrics["overall_health"] = "주의"
                else:
                    health_metrics["overall_health"] = "위험"
                health_metrics["health_score"] = overall_score
            
        except Exception as e:
            health_metrics["error"] = f"재무건전성 분석 실패: {e}"
            
        return health_metrics

    def _analyze_profitability(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """수익성 분석"""
        profitability_metrics = {}
        
        try:
            latest_data = data.iloc[-1] if len(data) > 0 else data.iloc[0]
            
            # ROE 분석
            roe = latest_data.get("return_on_equity")
            if roe is not None:
                if roe > 0.15:
                    profitability_metrics["roe_rating"] = "우수"
                elif roe > 0.10:
                    profitability_metrics["roe_rating"] = "양호"
                elif roe > 0.05:
                    profitability_metrics["roe_rating"] = "보통"
                else:
                    profitability_metrics["roe_rating"] = "저조"
                profitability_metrics["roe"] = roe
            
            # ROA 분석
            roa = latest_data.get("return_on_assets")
            if roa is not None:
                if roa > 0.10:
                    profitability_metrics["roa_rating"] = "우수"
                elif roa > 0.05:
                    profitability_metrics["roa_rating"] = "양호"
                elif roa > 0.02:
                    profitability_metrics["roa_rating"] = "보통"
                else:
                    profitability_metrics["roa_rating"] = "저조"
                profitability_metrics["roa"] = roa
            
            # 순이익률 분석
            profit_margin = latest_data.get("profit_margin")
            if profit_margin is not None:
                if profit_margin > 0.20:
                    profitability_metrics["margin_rating"] = "우수"
                elif profit_margin > 0.10:
                    profitability_metrics["margin_rating"] = "양호"
                elif profit_margin > 0.05:
                    profitability_metrics["margin_rating"] = "보통"
                else:
                    profitability_metrics["margin_rating"] = "저조"
                profitability_metrics["profit_margin"] = profit_margin
            
            # 영업이익률 분석
            operating_margin = latest_data.get("operating_margin")
            if operating_margin is not None:
                profitability_metrics["operating_margin"] = operating_margin
            
        except Exception as e:
            profitability_metrics["error"] = f"수익성 분석 실패: {e}"
            
        return profitability_metrics

    def _analyze_growth(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """성장성 분석"""
        growth_metrics = {}
        
        try:
            latest_data = data.iloc[-1] if len(data) > 0 else data.iloc[0]
            
            # 매출성장률 분석
            revenue_growth = latest_data.get("revenue_growth")
            if revenue_growth is not None:
                if revenue_growth > 0.20:
                    growth_metrics["revenue_growth_rating"] = "고성장"
                elif revenue_growth > 0.10:
                    growth_metrics["revenue_growth_rating"] = "성장"
                elif revenue_growth > 0.05:
                    growth_metrics["revenue_growth_rating"] = "안정"
                else:
                    growth_metrics["revenue_growth_rating"] = "저성장"
                growth_metrics["revenue_growth"] = revenue_growth
            
            # 이익성장률 분석
            earnings_growth = latest_data.get("earnings_growth")
            if earnings_growth is not None:
                if earnings_growth > 0.25:
                    growth_metrics["earnings_growth_rating"] = "고성장"
                elif earnings_growth > 0.15:
                    growth_metrics["earnings_growth_rating"] = "성장"
                elif earnings_growth > 0.05:
                    growth_metrics["earnings_growth_rating"] = "안정"
                else:
                    growth_metrics["earnings_growth_rating"] = "저성장"
                growth_metrics["earnings_growth"] = earnings_growth
            
            # 분기별 성장률
            quarterly_revenue_growth = latest_data.get("revenue_quarterly_growth")
            quarterly_earnings_growth = latest_data.get("earnings_quarterly_growth")
            
            if quarterly_revenue_growth is not None:
                growth_metrics["quarterly_revenue_growth"] = quarterly_revenue_growth
            if quarterly_earnings_growth is not None:
                growth_metrics["quarterly_earnings_growth"] = quarterly_earnings_growth
            
        except Exception as e:
            growth_metrics["error"] = f"성장성 분석 실패: {e}"
            
        return growth_metrics

    def _analyze_dividends(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """배당 분석"""
        dividend_metrics = {}
        
        try:
            latest_data = data.iloc[-1] if len(data) > 0 else data.iloc[0]
            
            # 배당수익률 분석
            dividend_yield = latest_data.get("dividend_yield")
            if dividend_yield is not None:
                if dividend_yield > 0.05:
                    dividend_metrics["yield_rating"] = "고배당"
                elif dividend_yield > 0.03:
                    dividend_metrics["yield_rating"] = "배당"
                elif dividend_yield > 0.01:
                    dividend_metrics["yield_rating"] = "저배당"
                else:
                    dividend_metrics["yield_rating"] = "무배당"
                dividend_metrics["dividend_yield"] = dividend_yield
            
            # 배당성향 분석
            payout_ratio = latest_data.get("payout_ratio")
            if payout_ratio is not None:
                if payout_ratio < 0.30:
                    dividend_metrics["payout_rating"] = "보수적"
                elif payout_ratio < 0.60:
                    dividend_metrics["payout_rating"] = "적정"
                elif payout_ratio < 0.80:
                    dividend_metrics["payout_rating"] = "공격적"
                else:
                    dividend_metrics["payout_rating"] = "과도"
                dividend_metrics["payout_ratio"] = payout_ratio
            
            # 5년 평균 배당수익률
            five_year_avg_yield = latest_data.get("five_year_avg_dividend_yield")
            if five_year_avg_yield is not None:
                dividend_metrics["five_year_avg_yield"] = five_year_avg_yield
            
            # 배당 안정성 (최근 배당 데이터)
            latest_dividend = latest_data.get("latest_dividend_amount")
            if latest_dividend is not None:
                dividend_metrics["latest_dividend"] = latest_dividend
            
        except Exception as e:
            dividend_metrics["error"] = f"배당 분석 실패: {e}"
            
        return dividend_metrics

    def _analyze_risk(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """위험도 분석"""
        risk_metrics = {}
        
        try:
            latest_data = data.iloc[-1] if len(data) > 0 else data.iloc[0]
            
            # 베타 분석
            beta = latest_data.get("beta")
            if beta is not None:
                if beta < 0.8:
                    risk_metrics["beta_rating"] = "저위험"
                elif beta < 1.2:
                    risk_metrics["beta_rating"] = "보통"
                else:
                    risk_metrics["beta_rating"] = "고위험"
                risk_metrics["beta"] = beta
            
            # 변동성 분석 (수익률의 표준편차)
            if "return" in data.columns:
                returns = data["return"].dropna()
                if len(returns) > 0:
                    volatility = returns.std()
                    risk_metrics["volatility"] = volatility
                    
                    if volatility < 0.02:
                        risk_metrics["volatility_rating"] = "저변동성"
                    elif volatility < 0.04:
                        risk_metrics["volatility_rating"] = "보통"
                    else:
                        risk_metrics["volatility_rating"] = "고변동성"
            
            # VaR (Value at Risk) 계산
            if "return" in data.columns:
                returns = data["return"].dropna()
                if len(returns) > 0:
                    var_95 = np.percentile(returns, 5)
                    var_99 = np.percentile(returns, 1)
                    risk_metrics["var_95"] = var_95
                    risk_metrics["var_99"] = var_99
            
            # 최대 낙폭 (Maximum Drawdown)
            if "return" in data.columns:
                returns = data["return"].dropna()
                if len(returns) > 0:
                    cumulative_returns = (1 + returns / 100).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown = drawdown.min()
                    risk_metrics["max_drawdown"] = max_drawdown
            
        except Exception as e:
            risk_metrics["error"] = f"위험도 분석 실패: {e}"
            
        return risk_metrics

    def generate_financial_report(self, analysis_results: Dict[str, Any], symbol: str) -> str:
        """재무분석 리포트 생성"""
        report = f"""
=== {symbol} 재무분석 리포트 ===

1. 주요 재무지표
"""
        
        if "key_metrics" in analysis_results:
            metrics = analysis_results["key_metrics"]
            report += f"""
   P/E 비율: {metrics.get('pe_ratio', 'N/A')}
   ROE: {metrics.get('roe', 'N/A')}
   부채비율: {metrics.get('debt_to_equity', 'N/A')}
   배당수익률: {metrics.get('dividend_yield', 'N/A')}
   시가총액: {metrics.get('market_cap', 'N/A')}
"""
        
        if "financial_health" in analysis_results:
            health = analysis_results["financial_health"]
            report += f"""
2. 재무건전성
   종합 건전성: {health.get('overall_health', 'N/A')}
   부채 건전성: {health.get('debt_health', 'N/A')}
   유동성 건전성: {health.get('liquidity_health', 'N/A')}
"""
        
        if "profitability_analysis" in analysis_results:
            profitability = analysis_results["profitability_analysis"]
            report += f"""
3. 수익성
   ROE 등급: {profitability.get('roe_rating', 'N/A')}
   ROA 등급: {profitability.get('roa_rating', 'N/A')}
   순이익률 등급: {profitability.get('margin_rating', 'N/A')}
"""
        
        if "growth_analysis" in analysis_results:
            growth = analysis_results["growth_analysis"]
            report += f"""
4. 성장성
   매출성장 등급: {growth.get('revenue_growth_rating', 'N/A')}
   이익성장 등급: {growth.get('earnings_growth_rating', 'N/A')}
"""
        
        if "dividend_analysis" in analysis_results:
            dividend = analysis_results["dividend_analysis"]
            report += f"""
5. 배당
   배당수익률 등급: {dividend.get('yield_rating', 'N/A')}
   배당성향 등급: {dividend.get('payout_rating', 'N/A')}
"""
        
        if "risk_analysis" in analysis_results:
            risk = analysis_results["risk_analysis"]
            report += f"""
6. 위험도
   베타 등급: {risk.get('beta_rating', 'N/A')}
   변동성 등급: {risk.get('volatility_rating', 'N/A')}
   VaR 95%: {risk.get('var_95', 'N/A')}
"""
        
        return report
