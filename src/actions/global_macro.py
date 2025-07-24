#!/usr/bin/env python3
"""
글로벌 매크로 데이터 수집기
SPY, VIX, 국채 스프레드, 달러 인덱스 등 매크로 지표 수집
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import sys
import json
import optuna
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import uuid

# yfinance 디버그 로그 억제
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from actions.y_finance import YahooFinanceDataCollector
from actions.calculate_index import TechnicalIndicators, StrategyParams


def safe_json_dump(data: Dict[str, Any], filepath: str, description: str = "데이터", logger=None):
    """안전한 JSON 저장 - 임시 파일을 사용하여 원자적 쓰기 보장"""
    import tempfile
    import shutil
    import os
    
    try:
        # JSON 직렬화 가능성 사전 검사 및 변환
        def make_json_serializable(obj):
            """JSON 직렬화 가능한 형태로 변환"""
            import numpy as np
            import pandas as pd
            
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(v) for v in obj]
            elif hasattr(obj, '__name__') and not isinstance(obj, str):
                return str(obj.__name__)
            elif hasattr(obj, '__class__') and not isinstance(obj, str):
                return str(obj.__class__.__name__)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except Exception:
                    return str(obj)
        
        # 데이터를 JSON 직렬화 가능한 형태로 변환
        serializable_data = make_json_serializable(data)
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
        temp_path = temp_file.name
        
        # 임시 파일에 JSON 쓰기
        json.dump(serializable_data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        
        # 원자적 이동 (파일이 완전히 쓰여진 후에만 이동)
        shutil.move(temp_path, filepath)
        
        if logger:
            logger.info(f"✅ {description} 저장 완료: {filepath}")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ {description} 저장 실패: {e}")
        # 임시 파일 정리
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return False


class MarketRegime(Enum):
    """시장 상태 열거형"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    UNCERTAIN = "UNCERTAIN"


class MarketCondition(Enum):
    """시장 조건 분류 (매크로 기반)"""
    BULL_MARKET = "bull_market"           # 강세장
    BEAR_MARKET = "bear_market"           # 약세장
    SIDEWAYS_MARKET = "sideways_market"   # 횡보장
    VOLATILE_MARKET = "volatile_market"   # 변동성 장
    RECESSION_FEAR = "recession_fear"     # 경기침체 우려
    INFLATION_FEAR = "inflation_fear"     # 인플레이션 우려


class SectorStrength(Enum):
    """섹터 강도 분류"""
    LEADING = "leading"       # 선도 섹터
    LAGGING = "lagging"       # 후행 섹터
    DEFENSIVE = "defensive"   # 방어적 섹터
    CYCLICAL = "cyclical"     # 순환적 섹터


@dataclass
class MarketClassification:
    """시장 분류 결과"""
    regime: MarketRegime
    confidence: float
    features: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class MacroAnalysis:
    """매크로 분석 결과"""
    market_condition: MarketCondition
    confidence: float
    key_indicators: Dict[str, float]
    sector_rotation: Dict[str, SectorStrength]
    recommendations: Dict[str, Any]
    timestamp: datetime

class GlobalMacroDataCollector:
    """글로벌 매크로 데이터 수집 클래스"""
    
    def __init__(self, session_uuid: str = None, config_path: str = "config/config_macro.json"):
        self.logger = logging.getLogger(__name__)
        
        # 세션 UUID 설정
        self.session_uuid = session_uuid or str(uuid.uuid4())
        self.logger.info(f"GlobalMacroDataCollector 초기화 - Session UUID: {self.session_uuid}")
        
        # 설정 파일 로드
        self.config = self._load_config(config_path)
        
        # YahooFinanceDataCollector 초기화
        self.collector = YahooFinanceDataCollector()
        self.params = StrategyParams()
        
        # 설정에서 매크로 지표 심볼 정의
        self.macro_symbols = self._get_macro_symbols_from_config()
        self.sector_etfs = self._get_sector_etfs_from_config()
        
        # 기존 매크로 지표 심볼 정의 (하위 호환성)
        self.macro_symbols = {
            'SPY': 'S&P 500 ETF',
            '^VIX': 'CBOE Volatility Index',
            '^TNX': '10-Year Treasury Yield',
            '^IRX': '13-Week Treasury Yield',
            'UUP': 'US Dollar Index ETF',
            'GLD': 'Gold ETF',
            'TLT': '20+ Year Treasury Bond ETF',
            'QQQ': 'NASDAQ-100 ETF',
            'IWM': 'Russell 2000 ETF',
            # TIPS 관련 지표 추가
            'TIP': 'iShares TIPS Bond ETF',  # TIPS ETF
            'SCHP': 'Schwab U.S. TIPS ETF',  # 대안 TIPS ETF
            'VTIP': 'Vanguard Short-Term Inflation-Protected Securities ETF',  # 단기 TIPS
            'LTPZ': 'PIMCO 15+ Year U.S. TIPS ETF',  # 장기 TIPS
            # Statistical Arbitrage용 추가 지표
            'XRT': 'SPDR S&P Retail ETF',  # 소매업
            'GTX': 'Global X Gold Explorers ETF'  # 금광업
        }
        
        # 섹터별 ETF
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate'
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"설정 파일 로드 완료: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            return {}
    
    def _get_macro_symbols_from_config(self) -> Dict[str, str]:
        """설정에서 매크로 심볼 가져오기"""
        try:
            data_sources = self.config.get('data_collection', {}).get('data_sources', {})
            macro_symbols = {}
            for key, value in data_sources.items():
                macro_symbols[value['symbol']] = value['description']
            return macro_symbols
        except Exception as e:
            self.logger.error(f"설정에서 매크로 심볼 로드 실패: {e}")
            return {}
    
    def _get_sector_etfs_from_config(self) -> Dict[str, str]:
        """설정에서 섹터 ETF 심볼 가져오기"""
        try:
            sector_etfs = self.config.get('data_collection', {}).get('sector_etfs', {})
            sector_symbols = {}
            for key, value in sector_etfs.items():
                sector_symbols[value['symbol']] = value['description']
            return sector_symbols
        except Exception as e:
            self.logger.error(f"설정에서 섹터 ETF 심볼 로드 실패: {e}")
            return {}
    
    def _get_days_back(self, collection_type: str = "default") -> int:
        """설정에서 데이터 수집 기간 가져오기"""
        try:
            data_collection = self.config.get('data_collection', {})
            
            # 새로운 period 기반 설정 사용
            data_periods = data_collection.get('data_periods', {})
            period_days = data_collection.get('period_days', {})
            
            # collection_type에 해당하는 period 가져오기
            period_key = f'{collection_type}_period'
            period_name = data_periods.get(period_key, data_periods.get('default_period', '2_years'))
            
            # period_name에 해당하는 일수 가져오기
            days_back = period_days.get(period_name, 730)  # 기본값 2년
            
            self.logger.info(f"데이터 수집 기간: {collection_type} -> {period_name} ({days_back}일)")
            return days_back
            
        except Exception as e:
            self.logger.error(f"설정에서 데이터 수집 기간 로드 실패: {e}")
            return 730  # 기본값 2년
    
    def collect_spy_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """SPY 데이터 수집"""
        try:
            # 설정에서 데이터 수집 기간 가져오기
            days_back = self._get_days_back("macro_analysis")
            
            # YahooFinanceDataCollector를 사용하여 데이터 수집
            df = self.collector.get_candle_data(
                symbol='SPY',
                interval='1d',  # 일봉 데이터
                start_date=start_date,
                end_date=end_date,
                days_back=days_back
            )
            
            if df is None or df.empty:
                self.logger.error("SPY 데이터 수집 실패")
                return pd.DataFrame()
            
            # 기술적 지표 계산
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, self.params)
            
            # datetime 컬럼 보장
            if "datetime" not in df_with_indicators.columns:
                df_with_indicators = df_with_indicators.reset_index()
            
            self.logger.info(f"SPY 데이터 수집 완료: {len(df_with_indicators)}개 데이터, 컬럼: {list(df_with_indicators.columns)}")
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"SPY 데이터 수집 중 오류: {e}")
            return pd.DataFrame()
    
    def collect_macro_indicators(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """매크로 지표들 수집"""
        macro_data = {}
        
        for symbol, description in self.macro_symbols.items():
            try:
                # 설정에서 데이터 수집 기간 가져오기
                days_back = self._get_days_back("macro_analysis")
                
                # YahooFinanceDataCollector를 사용하여 데이터 수집
                df = self.collector.get_candle_data(
                    symbol=symbol,
                    interval='1d',  # 일봉 데이터
                    start_date=start_date,
                    end_date=end_date,
                    days_back=days_back
                )
                
                if df is not None and not df.empty:
                    # 기술적 지표 계산
                    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, self.params)
                    
                    # datetime 컬럼 보장
                    if "datetime" not in df_with_indicators.columns:
                        df_with_indicators = df_with_indicators.reset_index()
                    
                    macro_data[symbol] = df_with_indicators
                    self.logger.info(f"{symbol} ({description}) 데이터 수집 완료: {len(df_with_indicators)}개 데이터")
                else:
                    self.logger.warning(f"{symbol} 데이터가 비어있습니다")
                    
            except Exception as e:
                self.logger.error(f"{symbol} 데이터 수집 중 오류: {e}")
        
        return macro_data
    
    def collect_sector_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """섹터별 ETF 데이터 수집"""
        sector_data = {}
        
        for symbol, sector_name in self.sector_etfs.items():
            try:
                # 설정에서 데이터 수집 기간 가져오기
                days_back = self._get_days_back("sector_analysis")
                
                # YahooFinanceDataCollector를 사용하여 데이터 수집
                df = self.collector.get_candle_data(
                    symbol=symbol,
                    interval='1d',  # 일봉 데이터
                    start_date=start_date,
                    end_date=end_date,
                    days_back=days_back
                )
                
                if df is not None and not df.empty:
                    # 기술적 지표 계산
                    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, self.params)
                    
                    # datetime 컬럼 보장
                    if "datetime" not in df_with_indicators.columns:
                        df_with_indicators = df_with_indicators.reset_index()
                    
                    sector_data[symbol] = df_with_indicators
                    self.logger.info(f"{symbol} ({sector_name}) 섹터 데이터 수집 완료: {len(df_with_indicators)}개 데이터")
                else:
                    self.logger.warning(f"{symbol} 섹터 데이터가 비어있습니다")
                    
            except Exception as e:
                self.logger.error(f"{symbol} 섹터 데이터 수집 중 오류: {e}")
        
        return sector_data
    
    def collect_all_data(self, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """모든 매크로 데이터 수집 (SPY, 매크로 지표, 섹터 데이터)"""
        try:
            # 설정에서 기본 데이터 수집 기간 가져오기
            default_days_back = self._get_days_back("default")
            
            # 날짜 설정 (설정 파일 기반)
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=default_days_back)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.info(f"전체 매크로 데이터 수집 시작: {start_date} ~ {end_date}")
            
            # 1. SPY 데이터 수집
            spy_data = self.collect_spy_data(start_date, end_date)
            self.logger.info(f"SPY 데이터 수집 완료: {len(spy_data)}개 데이터")
            
            # 2. 매크로 지표 데이터 수집
            macro_data = self.collect_macro_indicators(start_date, end_date)
            self.logger.info(f"매크로 지표 데이터 수집 완료: {len(macro_data)}개 지표")
            
            # 3. 섹터 데이터 수집
            sector_data = self.collect_sector_data(start_date, end_date)
            self.logger.info(f"섹터 데이터 수집 완료: {len(sector_data)}개 섹터")
            
            # 4. 데이터 저장 (설정 파일 기반)
            try:
                self.save_macro_data(spy_data, macro_data, sector_data, start_date=start_date, end_date=end_date)
                self.logger.info("매크로 데이터 저장 완료")
            except Exception as e:
                self.logger.warning(f"데이터 저장 실패 (분석은 계속 진행): {e}")
            
            return spy_data, macro_data, sector_data
            
        except Exception as e:
            self.logger.error(f"전체 매크로 데이터 수집 실패: {e}")
            # 빈 데이터 반환
            return pd.DataFrame(), {}, {}
    

    
    def calculate_macro_metrics(self, macro_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """매크로 지표 기반 메트릭 계산"""
        metrics = {}
        
        try:
            # VIX 기반 변동성 지표
            if '^VIX' in macro_data:
                vix = macro_data['^VIX']['Close']
                metrics['vix_level'] = vix
                metrics['vix_ma_20'] = vix.rolling(window=20).mean()
                metrics['vix_volatility'] = vix.rolling(window=20).std()
            
            # 국채 스프레드 (10년 - 13주)
            if '^TNX' in macro_data and '^IRX' in macro_data:
                tnx = macro_data['^TNX']['Close']
                irx = macro_data['^IRX']['Close']
                metrics['yield_spread'] = tnx - irx
                metrics['yield_spread_ma_20'] = metrics['yield_spread'].rolling(window=20).mean()
            
            # TIPS Spread 계산 (인플레이션 기대치)
            metrics = self._calculate_tips_spread(macro_data, metrics)
            
            # 달러 강도
            if 'UUP' in macro_data:
                dxy = macro_data['UUP']['Close']
                metrics['dollar_strength'] = dxy
                metrics['dollar_ma_50'] = dxy.rolling(window=50).mean()
            
            # 금 가격
            if 'GLD' in macro_data:
                gld = macro_data['GLD']['Close']
                metrics['gold_price'] = gld
                metrics['gold_returns'] = gld.pct_change()
            
            # 국채 가격
            if 'TLT' in macro_data:
                tlt = macro_data['TLT']['Close']
                metrics['bond_price'] = tlt
                metrics['bond_returns'] = tlt.pct_change()
            
            return pd.DataFrame(metrics)
            
        except Exception as e:
            self.logger.error(f"매크로 메트릭 계산 중 오류: {e}")
            return pd.DataFrame()
    
    def _calculate_tips_spread(self, macro_data: Dict[str, pd.DataFrame], metrics: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """TIPS Spread 계산 (인플레이션 기대치)"""
        try:
            # TIPS ETF들의 상대적 성과를 기반으로 인플레이션 기대치 추정
            
            # 1. TIP vs TLT 비교 (TIPS vs 일반 국채)
            if 'TIP' in macro_data and 'TLT' in macro_data:
                tip = macro_data['TIP']['Close']
                tlt = macro_data['TLT']['Close']
                tip_tlt_ratio = tip / tlt
                metrics['tips_spread_tip_tlt'] = tip_tlt_ratio.pct_change(20)
                metrics['tips_spread_tip_tlt_ma_50'] = metrics['tips_spread_tip_tlt'].rolling(window=50).mean()
            
            # 2. 단기 vs 장기 TIPS 비교
            if 'VTIP' in macro_data and 'LTPZ' in macro_data:
                vtip = macro_data['VTIP']['Close']
                ltpz = macro_data['LTPZ']['Close']
                vtip_ltpz_ratio = vtip / ltpz
                metrics['tips_spread_short_long'] = vtip_ltpz_ratio.pct_change(20)
                metrics['tips_spread_short_long_ma_50'] = metrics['tips_spread_short_long'].rolling(window=50).mean()
            
            # 3. TIPS vs 일반 국채 ETF 비교 (SCHP vs TLT)
            if 'SCHP' in macro_data and 'TLT' in macro_data:
                schp = macro_data['SCHP']['Close']
                tlt = macro_data['TLT']['Close']
                schp_tlt_ratio = schp / tlt
                metrics['tips_spread_schp_tlt'] = schp_tlt_ratio.pct_change(20)
                metrics['tips_spread_schp_tlt_ma_50'] = metrics['tips_spread_schp_tlt'].rolling(window=50).mean()
            
            # 4. 종합 TIPS Spread 지표 (여러 지표의 평균)
            tips_spread_columns = [col for col in metrics.keys() if 'tips_spread' in col and 'ma_50' not in col]
            if tips_spread_columns:
                tips_spread_df = pd.DataFrame({col: metrics[col] for col in tips_spread_columns})
                metrics['tips_spread_composite'] = tips_spread_df.mean(axis=1)
                metrics['tips_spread_composite_ma_20'] = metrics['tips_spread_composite'].rolling(window=20).mean()
                metrics['tips_spread_composite_ma_50'] = metrics['tips_spread_composite'].rolling(window=50).mean()
                
                # 인플레이션 기대치 분위수
                metrics['inflation_expectation_percentile'] = metrics['tips_spread_composite'].rolling(window=252).rank(pct=True)
            
            self.logger.info(f"TIPS Spread 계산 완료: {len([col for col in metrics.keys() if 'tips_spread' in col])}개 지표")
            return metrics
            
        except Exception as e:
            self.logger.error(f"TIPS Spread 계산 중 오류: {e}")
            return metrics
    
    def calculate_sector_rotation(self, sector_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """섹터 로테이션 분석"""
        sector_returns = {}
        
        try:
            for symbol, data in sector_data.items():
                if not data.empty:
                    sector_returns[symbol] = data['Close'].pct_change()
            
            sector_df = pd.DataFrame(sector_returns)
            
            # 섹터별 상대 강도 계산
            for symbol in sector_df.columns:
                sector_df[f'{symbol}_rs'] = sector_df[symbol].rolling(window=20).mean()
            
            # 섹터 순위 계산
            rs_columns = [col for col in sector_df.columns if col.endswith('_rs')]
            sector_df['sector_rank'] = sector_df[rs_columns].rank(axis=1, ascending=False)
            
            return sector_df
            
        except Exception as e:
            self.logger.error(f"섹터 로테이션 계산 중 오류: {e}")
            return pd.DataFrame()
    
    def save_macro_data(self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame], 
                       sector_data: Dict[str, pd.DataFrame], output_dir: str = "data/macro",
                       start_date: str = None, end_date: str = None):
        """매크로 데이터 저장"""
        import os
        
        try:
            # UUID 기반 디렉토리 생성
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            # 메타데이터 저장
            metadata = {
                'session_uuid': self.session_uuid,
                'start_date': start_date,
                'end_date': end_date,
                'created_at': datetime.now().isoformat(),
                'data_sources': list(macro_data.keys()) + list(sector_data.keys()),
                'spy_rows': len(spy_data) if not spy_data.empty else 0,
                'macro_symbols': len(macro_data),
                'sector_symbols': len(sector_data)
            }
            
            with open(f"{session_dir}/metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # SPY 데이터 저장 (기술적 지표 포함)
            if not spy_data.empty:
                spy_data.to_csv(f"{session_dir}/spy_data.csv")
                self.logger.info(f"SPY 데이터 저장 완료: {len(spy_data)}개 행, {len(spy_data.columns)}개 컬럼")
            
            # 매크로 지표 저장
            for symbol, data in macro_data.items():
                if not data.empty:
                    data.to_csv(f"{session_dir}/{symbol.lower()}_data.csv")
                    self.logger.info(f"{symbol} 데이터 저장 완료: {len(data)}개 행")
            
            # 섹터 데이터 저장
            for symbol, data in sector_data.items():
                if not data.empty:
                    data.to_csv(f"{session_dir}/{symbol.lower()}_sector.csv")
                    self.logger.info(f"{symbol} 섹터 데이터 저장 완료: {len(data)}개 행")
            
            self.logger.info(f"매크로 데이터 저장 완료: {session_dir}")
            
        except Exception as e:
            self.logger.error(f"매크로 데이터 저장 중 오류: {e}")


class MacroSectorAnalyzer:
    """매크로 & 섹터 분석 시스템"""
    
    def __init__(self, data_dir: str = "data/macro", session_uuid: str = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.session_uuid = session_uuid or str(uuid.uuid4())
        self.macro_collector = GlobalMacroDataCollector(self.session_uuid)
        self.logger.info(f"MacroSectorAnalyzer 초기화 - Session UUID: {self.session_uuid}")
        
        # 섹터 분류
        self.sector_classification = {
            'XLK': {'name': 'Technology', 'type': SectorStrength.CYCLICAL},
            'XLF': {'name': 'Financials', 'type': SectorStrength.CYCLICAL},
            'XLE': {'name': 'Energy', 'type': SectorStrength.CYCLICAL},
            'XLV': {'name': 'Healthcare', 'type': SectorStrength.DEFENSIVE},
            'XLI': {'name': 'Industrials', 'type': SectorStrength.CYCLICAL},
            'XLP': {'name': 'Consumer Staples', 'type': SectorStrength.DEFENSIVE},
            'XLU': {'name': 'Utilities', 'type': SectorStrength.DEFENSIVE},
            'XLB': {'name': 'Materials', 'type': SectorStrength.CYCLICAL},
            'XLRE': {'name': 'Real Estate', 'type': SectorStrength.CYCLICAL}
        }
    
    def analyze_macro_environment(self, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """매크로 환경 분석"""
        analysis = {}
        
        try:
            # 1. VIX 기반 변동성 분석
            if '^vix_data' in macro_data:
                vix_data = macro_data['^vix_data']
                close_col = 'close' if 'close' in vix_data.columns else 'Close'
                vix = vix_data[close_col]
                analysis['vix_current'] = vix.iloc[-1]
                analysis['vix_ma_20'] = vix.rolling(20).mean().iloc[-1]
                analysis['vix_percentile'] = (vix.iloc[-1] / vix.rolling(252).max().iloc[-1]) * 100
                analysis['volatility_regime'] = 'high' if vix.iloc[-1] > 25 else 'normal'
            
            # 2. 국채 스프레드 분석 (경기침체 지표)
            if '^tnx_data' in macro_data and '^irx_data' in macro_data:
                tnx_data = macro_data['^tnx_data']
                irx_data = macro_data['^irx_data']
                tnx_close_col = 'close' if 'close' in tnx_data.columns else 'Close'
                irx_close_col = 'close' if 'close' in irx_data.columns else 'Close'
                tnx = tnx_data[tnx_close_col]
                irx = irx_data[irx_close_col]
                spread = tnx - irx
                analysis['yield_spread'] = spread.iloc[-1]
                analysis['spread_ma_20'] = spread.rolling(20).mean().iloc[-1]
                analysis['recession_risk'] = 'high' if spread.iloc[-1] < 0 else 'low'
            
            # 3. TIPS Spread 분석 (인플레이션 기대치)
            analysis = self._analyze_tips_spread(macro_data, analysis)
            
            # 4. 달러 강도 분석
            if 'uup_data' in macro_data:
                uup_data = macro_data['uup_data']
                close_col = 'close' if 'close' in uup_data.columns else 'Close'
                dxy = uup_data[close_col]
                analysis['dollar_strength'] = dxy.iloc[-1]
                analysis['dollar_ma_50'] = dxy.rolling(50).mean().iloc[-1]
                analysis['dollar_trend'] = 'strong' if dxy.iloc[-1] > dxy.rolling(50).mean().iloc[-1] else 'weak'
            
            # 5. 금 가격 분석 (안전자산 선호도)
            if 'gld_data' in macro_data:
                gld_data = macro_data['gld_data']
                close_col = 'close' if 'close' in gld_data.columns else 'Close'
                gold = gld_data[close_col]
                analysis['gold_price'] = gold.iloc[-1]
                analysis['gold_ma_50'] = gold.rolling(50).mean().iloc[-1]
                analysis['gold_trend'] = 'bullish' if gold.iloc[-1] > gold.rolling(50).mean().iloc[-1] else 'bearish'
            
            # 6. 국채 가격 분석
            if 'tlt_data' in macro_data:
                tlt_data = macro_data['tlt_data']
                close_col = 'close' if 'close' in tlt_data.columns else 'Close'
                tlt = tlt_data[close_col]
                analysis['bond_price'] = tlt.iloc[-1]
                analysis['bond_ma_50'] = tlt.rolling(50).mean().iloc[-1]
                analysis['bond_trend'] = 'bullish' if tlt.iloc[-1] > tlt.rolling(50).mean().iloc[-1] else 'bearish'
            
            # 7. 금리 환경 분석
            if '^tnx_data' in macro_data:
                tnx_data = macro_data['^tnx_data']
                close_col = 'close' if 'close' in tnx_data.columns else 'Close'
                tnx = tnx_data[close_col]
                current_rate = tnx.iloc[-1]
                rate_ma_50 = tnx.rolling(50).mean().iloc[-1]
                
                if current_rate > 5.0:
                    analysis['rate_environment'] = 'high_rates'
                elif current_rate > 3.0:
                    analysis['rate_environment'] = 'moderate_rates'
                else:
                    analysis['rate_environment'] = 'low_rates'
                
                analysis['rate_trend'] = 'increasing' if current_rate > rate_ma_50 else 'decreasing'
            else:
                analysis['rate_environment'] = 'unknown'
                analysis['rate_trend'] = 'unknown'
            
            # 8. 고도화된 성장 전망 분석 (여러 지표 종합)
            growth_indicators = []
            
            # VIX 기반 성장 전망
            if 'vix_current' in analysis:
                vix_level = analysis['vix_current']
                if vix_level < 15:
                    growth_indicators.append('positive')
                elif vix_level > 25:
                    growth_indicators.append('negative')
                else:
                    growth_indicators.append('neutral')
            
            # 국채 스프레드 기반 성장 전망
            if 'yield_spread' in analysis:
                spread = analysis['yield_spread']
                if spread > 1.0:
                    growth_indicators.append('positive')
                elif spread < 0:
                    growth_indicators.append('negative')
                else:
                    growth_indicators.append('neutral')
            
            # 달러 강도 기반 성장 전망
            if 'dollar_trend' in analysis:
                if analysis['dollar_trend'] == 'strong':
                    growth_indicators.append('negative')  # 강한 달러는 성장에 부정적
                else:
                    growth_indicators.append('positive')
            
            # 금 가격 기반 성장 전망 (안전자산 선호도)
            if 'gold_trend' in analysis:
                if analysis['gold_trend'] == 'bullish':
                    growth_indicators.append('negative')  # 금 상승 = 위험 회피
                else:
                    growth_indicators.append('positive')
            
            # 국채 가격 기반 성장 전망
            if 'bond_trend' in analysis:
                if analysis['bond_trend'] == 'bullish':
                    growth_indicators.append('negative')  # 국채 상승 = 위험 회피
                else:
                    growth_indicators.append('positive')
            
            # 추가 매크로 지표들 분석
            # QQQ (나스닥) 성과 기반 성장 전망
            if 'QQQ' in macro_data:
                qqq_data = macro_data['QQQ']
                close_col = 'close' if 'close' in qqq_data.columns else 'Close'
                qqq = qqq_data[close_col]
                qqq_return = qqq.pct_change(20).iloc[-1]  # 20일 수익률
                analysis['qqq_momentum'] = qqq_return
                if qqq_return > 0.05:  # 5% 이상 상승
                    growth_indicators.append('positive')
                elif qqq_return < -0.05:  # 5% 이상 하락
                    growth_indicators.append('negative')
                else:
                    growth_indicators.append('neutral')
            
            # IWM (소형주) 성과 기반 성장 전망
            if 'IWM' in macro_data:
                iwm_data = macro_data['IWM']
                close_col = 'close' if 'close' in iwm_data.columns else 'Close'
                iwm = iwm_data[close_col]
                iwm_return = iwm.pct_change(20).iloc[-1]  # 20일 수익률
                analysis['iwm_momentum'] = iwm_return
                if iwm_return > 0.05:  # 5% 이상 상승
                    growth_indicators.append('positive')
                elif iwm_return < -0.05:  # 5% 이상 하락
                    growth_indicators.append('negative')
                else:
                    growth_indicators.append('neutral')
            
            # 종합 성장 전망 판단 (가중치 적용)
            if growth_indicators:
                positive_count = growth_indicators.count('positive')
                negative_count = growth_indicators.count('negative')
                neutral_count = growth_indicators.count('neutral')
                
                # 신뢰도 계산 (더 많은 지표일수록 신뢰도 높음)
                total_indicators = len(growth_indicators)
                confidence_score = min(1.0, total_indicators / 5.0)  # 최대 5개 지표 기준
                analysis['growth_outlook_confidence'] = confidence_score
                analysis['growth_indicators_count'] = total_indicators
                
                if positive_count > negative_count:
                    analysis['growth_outlook'] = 'positive'
                elif negative_count > positive_count:
                    analysis['growth_outlook'] = 'negative'
                else:
                    analysis['growth_outlook'] = 'neutral'
            else:
                analysis['growth_outlook'] = 'neutral'  # unknown 대신 neutral로 변경
                analysis['growth_outlook_confidence'] = 0.0
                analysis['growth_indicators_count'] = 0
            
            # 9. 시장 체제 추가 분석
            # 리스크온/리스크오프 지표
            risk_on_score = 0
            risk_off_score = 0
            
            if 'vix_current' in analysis:
                vix_level = analysis['vix_current']
                if vix_level < 20:
                    risk_on_score += 1
                elif vix_level > 25:
                    risk_off_score += 1
            
            if 'gold_trend' in analysis:
                if analysis['gold_trend'] == 'bearish':
                    risk_on_score += 1
                else:
                    risk_off_score += 1
            
            if 'dollar_trend' in analysis:
                if analysis['dollar_trend'] == 'weak':
                    risk_on_score += 1
                else:
                    risk_off_score += 1
            
            if risk_on_score > risk_off_score:
                analysis['market_sentiment'] = 'risk_on'
            elif risk_off_score > risk_on_score:
                analysis['market_sentiment'] = 'risk_off'
            else:
                analysis['market_sentiment'] = 'neutral'
                
            analysis['risk_on_score'] = risk_on_score
            analysis['risk_off_score'] = risk_off_score
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"매크로 환경 분석 중 오류: {e}")
            return {}
    
    def _analyze_tips_spread(self, macro_data: Dict[str, pd.DataFrame], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """TIPS Spread 분석 (인플레이션 기대치)"""
        try:
            # TIPS ETF들의 상대적 성과를 기반으로 인플레이션 기대치 분석
            
            # 1. TIP vs TLT 비교 (TIPS vs 일반 국채)
            if 'TIP' in macro_data and 'TLT' in macro_data:
                tip_data = macro_data['TIP']
                tlt_data = macro_data['TLT']
                tip_close_col = 'close' if 'close' in tip_data.columns else 'Close'
                tlt_close_col = 'close' if 'close' in tlt_data.columns else 'Close'
                tip = tip_data[tip_close_col]
                tlt = tlt_data[tlt_close_col]
                tip_tlt_ratio = tip / tlt
                analysis['tips_spread_tip_tlt'] = tip_tlt_ratio.pct_change(20).iloc[-1]
                analysis['tips_spread_tip_tlt_ma_50'] = tip_tlt_ratio.pct_change(20).rolling(50).mean().iloc[-1]
            
            # 2. 단기 vs 장기 TIPS 비교
            if 'VTIP' in macro_data and 'LTPZ' in macro_data:
                vtip_data = macro_data['VTIP']
                ltpz_data = macro_data['LTPZ']
                vtip_close_col = 'close' if 'close' in vtip_data.columns else 'Close'
                ltpz_close_col = 'close' if 'close' in ltpz_data.columns else 'Close'
                vtip = vtip_data[vtip_close_col]
                ltpz = ltpz_data[ltpz_close_col]
                vtip_ltpz_ratio = vtip / ltpz
                analysis['tips_spread_short_long'] = vtip_ltpz_ratio.pct_change(20).iloc[-1]
                analysis['tips_spread_short_long_ma_50'] = vtip_ltpz_ratio.pct_change(20).rolling(50).mean().iloc[-1]
            
            # 3. TIPS vs 일반 국채 ETF 비교 (SCHP vs TLT)
            if 'SCHP' in macro_data and 'TLT' in macro_data:
                schp_data = macro_data['SCHP']
                tlt_data = macro_data['TLT']
                schp_close_col = 'close' if 'close' in schp_data.columns else 'Close'
                tlt_close_col = 'close' if 'close' in tlt_data.columns else 'Close'
                schp = schp_data[schp_close_col]
                tlt = tlt_data[tlt_close_col]
                schp_tlt_ratio = schp / tlt
                analysis['tips_spread_schp_tlt'] = schp_tlt_ratio.pct_change(20).iloc[-1]
                analysis['tips_spread_schp_tlt_ma_50'] = schp_tlt_ratio.pct_change(20).rolling(50).mean().iloc[-1]
            
            # 4. 종합 TIPS Spread 지표
            tips_spread_values = []
            for key in ['tips_spread_tip_tlt', 'tips_spread_short_long', 'tips_spread_schp_tlt']:
                if key in analysis:
                    tips_spread_values.append(analysis[key])
            
            if tips_spread_values:
                analysis['tips_spread_composite'] = np.mean(tips_spread_values)
                analysis['tips_spread_composite_ma_50'] = np.mean([analysis.get(f'{key}_ma_50', 0) for key in ['tips_spread_tip_tlt', 'tips_spread_short_long', 'tips_spread_schp_tlt']])
                
                # 인플레이션 기대치 판단
                if analysis['tips_spread_composite'] > 0.02:  # 2% 이상 상승
                    analysis['inflation_expectation'] = 'high'
                elif analysis['tips_spread_composite'] < -0.02:  # 2% 이상 하락
                    analysis['inflation_expectation'] = 'low'
                else:
                    analysis['inflation_expectation'] = 'stable'
                
                # 인플레이션 기대치 변화 추이
                analysis['inflation_trend'] = 'increasing' if analysis['tips_spread_composite'] > analysis['tips_spread_composite_ma_50'] else 'decreasing'
            
            self.logger.info(f"TIPS Spread 분석 완료: 인플레이션 기대치 = {analysis.get('inflation_expectation', 'unknown')}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"TIPS Spread 분석 중 오류: {e}")
            return analysis
    
    def classify_market_condition(self, macro_analysis: Dict[str, Any]) -> MarketCondition:
        """매크로 분석을 기반으로 시장 조건 분류 (고도화된 버전)"""
        scores = {
            MarketCondition.BULL_MARKET: 0,
            MarketCondition.BEAR_MARKET: 0,
            MarketCondition.SIDEWAYS_MARKET: 0,
            MarketCondition.VOLATILE_MARKET: 0,
            MarketCondition.RECESSION_FEAR: 0,
            MarketCondition.INFLATION_FEAR: 0
        }
        
        # 1. VIX 기반 변동성 점수 (가중치: 25%)
        if 'volatility_regime' in macro_analysis:
            if macro_analysis['volatility_regime'] == 'high':
                scores[MarketCondition.VOLATILE_MARKET] += 8
                scores[MarketCondition.BEAR_MARKET] += 3
            elif macro_analysis['volatility_regime'] == 'medium':
                scores[MarketCondition.VOLATILE_MARKET] += 4
                scores[MarketCondition.SIDEWAYS_MARKET] += 2
            elif macro_analysis['volatility_regime'] == 'low':
                scores[MarketCondition.BULL_MARKET] += 2
                scores[MarketCondition.SIDEWAYS_MARKET] += 1
        
        # VIX 변화율 기반 추가 점수
        if 'vix_change' in macro_analysis:
            vix_change = macro_analysis['vix_change']
            if vix_change > 0.2:  # 20% 이상 상승
                scores[MarketCondition.VOLATILE_MARKET] += 5
                scores[MarketCondition.BEAR_MARKET] += 2
            elif vix_change > 0.1:  # 10% 이상 상승
                scores[MarketCondition.VOLATILE_MARKET] += 3
            elif vix_change < -0.2:  # 20% 이상 하락
                scores[MarketCondition.BULL_MARKET] += 3
            elif vix_change < -0.1:  # 10% 이상 하락
                scores[MarketCondition.BULL_MARKET] += 1
        
        # 2. 국채 스프레드 기반 경기침체 점수 (가중치: 20%)
        if 'recession_risk' in macro_analysis:
            if macro_analysis['recession_risk'] == 'high':
                scores[MarketCondition.RECESSION_FEAR] += 10
                scores[MarketCondition.BEAR_MARKET] += 5
                scores[MarketCondition.VOLATILE_MARKET] += 3
            elif macro_analysis['recession_risk'] == 'medium':
                scores[MarketCondition.RECESSION_FEAR] += 5
                scores[MarketCondition.BEAR_MARKET] += 2
            elif macro_analysis['recession_risk'] == 'low':
                scores[MarketCondition.BULL_MARKET] += 2
        
        # 2-1. 2년-10년 국채 스프레드 기반 추가 점수
        if 'yield_curve_spread' in macro_analysis:
            spread = macro_analysis['yield_curve_spread']
            if spread < 0:  # 역수익률 곡선
                scores[MarketCondition.RECESSION_FEAR] += 8
                scores[MarketCondition.BEAR_MARKET] += 4
            elif spread < 0.5:  # 평평한 수익률 곡선
                scores[MarketCondition.RECESSION_FEAR] += 4
                scores[MarketCondition.SIDEWAYS_MARKET] += 2
            elif spread > 1.5:  # 가파른 수익률 곡선
                scores[MarketCondition.BULL_MARKET] += 3
        
        # 3. TIPS Spread 기반 인플레이션 점수 (가중치: 20%)
        if 'inflation_expectation' in macro_analysis:
            if macro_analysis['inflation_expectation'] == 'high':
                scores[MarketCondition.INFLATION_FEAR] += 10
                scores[MarketCondition.VOLATILE_MARKET] += 5
                scores[MarketCondition.BEAR_MARKET] += 3
            elif macro_analysis['inflation_expectation'] == 'medium':
                scores[MarketCondition.INFLATION_FEAR] += 5
                scores[MarketCondition.VOLATILE_MARKET] += 2
            elif macro_analysis['inflation_expectation'] == 'low':
                scores[MarketCondition.RECESSION_FEAR] += 4
                scores[MarketCondition.BULL_MARKET] += 2
        
        # 3-1. 인플레이션 추세 기반 추가 점수
        if 'inflation_trend' in macro_analysis:
            if macro_analysis['inflation_trend'] == 'increasing':
                scores[MarketCondition.INFLATION_FEAR] += 6
                scores[MarketCondition.VOLATILE_MARKET] += 3
            elif macro_analysis['inflation_trend'] == 'decreasing':
                scores[MarketCondition.RECESSION_FEAR] += 4
                scores[MarketCondition.BULL_MARKET] += 2
            elif macro_analysis['inflation_trend'] == 'stable':
                scores[MarketCondition.SIDEWAYS_MARKET] += 3
        
        # 4. 달러 강도 기반 점수 (가중치: 15%) - recession_risk와 연계
        if 'dollar_trend' in macro_analysis:
            recession_risk = macro_analysis.get('recession_risk', 'low')
            if macro_analysis['dollar_trend'] == 'strong':
                scores[MarketCondition.BULL_MARKET] += 4
                scores[MarketCondition.INFLATION_FEAR] += 2
            elif macro_analysis['dollar_trend'] == 'weak':
                if recession_risk == 'high':
                    scores[MarketCondition.BEAR_MARKET] += 4
                    scores[MarketCondition.RECESSION_FEAR] += 3
                elif recession_risk == 'medium':
                    scores[MarketCondition.BEAR_MARKET] += 2
                    scores[MarketCondition.RECESSION_FEAR] += 2
                else:  # low
                    scores[MarketCondition.BEAR_MARKET] += 1  # 약한 달러는 경기침체 신호가 아님
            elif macro_analysis['dollar_trend'] == 'sideways':
                scores[MarketCondition.SIDEWAYS_MARKET] += 3
        
        # 4-1. 달러 인덱스 변화율 기반 추가 점수
        if 'dollar_change' in macro_analysis:
            dollar_change = macro_analysis['dollar_change']
            if dollar_change > 0.05:  # 5% 이상 상승
                scores[MarketCondition.BULL_MARKET] += 3
            elif dollar_change < -0.05:  # 5% 이상 하락
                scores[MarketCondition.BEAR_MARKET] += 3
        
        # 5. 금 가격 기반 점수 (가중치: 10%) - recession_risk와 연계
        if 'gold_trend' in macro_analysis:
            recession_risk = macro_analysis.get('recession_risk', 'low')
            if macro_analysis['gold_trend'] == 'bullish':
                if recession_risk == 'high':
                    scores[MarketCondition.RECESSION_FEAR] += 6
                    scores[MarketCondition.VOLATILE_MARKET] += 3
                elif recession_risk == 'medium':
                    scores[MarketCondition.RECESSION_FEAR] += 3
                    scores[MarketCondition.VOLATILE_MARKET] += 2
                else:  # low
                    scores[MarketCondition.INFLATION_FEAR] += 3
                    scores[MarketCondition.VOLATILE_MARKET] += 2
            elif macro_analysis['gold_trend'] == 'bearish':
                scores[MarketCondition.BULL_MARKET] += 3
            elif macro_analysis['gold_trend'] == 'sideways':
                scores[MarketCondition.SIDEWAYS_MARKET] += 2
        
        # 5-1. 금 가격 변화율 기반 추가 점수 - recession_risk와 연계
        if 'gold_change' in macro_analysis:
            gold_change = macro_analysis['gold_change']
            recession_risk = macro_analysis.get('recession_risk', 'low')
            if gold_change > 0.1:  # 10% 이상 상승
                if recession_risk == 'high':
                    scores[MarketCondition.RECESSION_FEAR] += 4
                elif recession_risk == 'medium':
                    scores[MarketCondition.RECESSION_FEAR] += 2
                else:  # low
                    scores[MarketCondition.INFLATION_FEAR] += 2
                scores[MarketCondition.VOLATILE_MARKET] += 2
            elif gold_change < -0.1:  # 10% 이상 하락
                scores[MarketCondition.BULL_MARKET] += 2
        
        # 6. 국채 가격 기반 점수 (가중치: 10%) - recession_risk와 연계
        if 'bond_trend' in macro_analysis:
            recession_risk = macro_analysis.get('recession_risk', 'low')
            if macro_analysis['bond_trend'] == 'bullish':
                if recession_risk == 'high':
                    scores[MarketCondition.RECESSION_FEAR] += 6
                    scores[MarketCondition.VOLATILE_MARKET] += 2
                elif recession_risk == 'medium':
                    scores[MarketCondition.RECESSION_FEAR] += 3
                    scores[MarketCondition.VOLATILE_MARKET] += 1
                else:  # low
                    scores[MarketCondition.BULL_MARKET] += 2  # 안전자산 선호는 강세장 신호일 수도 있음
            elif macro_analysis['bond_trend'] == 'bearish':
                scores[MarketCondition.INFLATION_FEAR] += 4
                scores[MarketCondition.BULL_MARKET] += 2
            elif macro_analysis['bond_trend'] == 'sideways':
                scores[MarketCondition.SIDEWAYS_MARKET] += 2
        
        # 6-1. 10년 국채 수익률 기반 추가 점수
        if 'treasury_10y_yield' in macro_analysis:
            yield_10y = macro_analysis['treasury_10y_yield']
            if yield_10y > 4.0:  # 4% 이상
                scores[MarketCondition.INFLATION_FEAR] += 3
                scores[MarketCondition.BEAR_MARKET] += 2
            elif yield_10y < 2.0:  # 2% 미만
                scores[MarketCondition.RECESSION_FEAR] += 3
                scores[MarketCondition.BULL_MARKET] += 2
        
        # 7. 섹터 로테이션 기반 점수 (가중치: 5%)
        if 'sector_rotation' in macro_analysis:
            sector_rotation = macro_analysis['sector_rotation']
            if sector_rotation == 'defensive':
                scores[MarketCondition.RECESSION_FEAR] += 3
                scores[MarketCondition.BEAR_MARKET] += 2
            elif sector_rotation == 'cyclical':
                scores[MarketCondition.BULL_MARKET] += 3
            elif sector_rotation == 'mixed':
                scores[MarketCondition.SIDEWAYS_MARKET] += 2
        
        # 8. 거래량 기반 점수 (가중치: 5%)
        if 'volume_trend' in macro_analysis:
            if macro_analysis['volume_trend'] == 'high':
                scores[MarketCondition.VOLATILE_MARKET] += 3
            elif macro_analysis['volume_trend'] == 'low':
                scores[MarketCondition.SIDEWAYS_MARKET] += 2
        
        # 9. 시장 폭 기반 점수 (가중치: 5%)
        if 'market_breadth' in macro_analysis:
            market_breadth = macro_analysis['market_breadth']
            if market_breadth > 0.7:  # 70% 이상 상승
                scores[MarketCondition.BULL_MARKET] += 3
            elif market_breadth < 0.3:  # 30% 미만 상승
                scores[MarketCondition.BEAR_MARKET] += 3
            elif 0.3 <= market_breadth <= 0.7:
                scores[MarketCondition.SIDEWAYS_MARKET] += 2
        
        # 10. 신용 스프레드 기반 점수 (가중치: 5%)
        if 'credit_spread' in macro_analysis:
            credit_spread = macro_analysis['credit_spread']
            if credit_spread > 0.05:  # 5% 이상 (높은 신용 위험)
                scores[MarketCondition.RECESSION_FEAR] += 4
                scores[MarketCondition.BEAR_MARKET] += 2
            elif credit_spread < 0.02:  # 2% 미만 (낮은 신용 위험)
                scores[MarketCondition.BULL_MARKET] += 3
        
        # 최고 점수 시장 조건 반환
        return max(scores, key=scores.get)
    
    def analyze_sector_rotation(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, SectorStrength]:
        """섹터 로테이션 분석"""
        sector_analysis = {}
        
        try:
            # 섹터별 수익률 계산
            sector_returns = {}
            for symbol, data in sector_data.items():
                if not data.empty and 'close' in data.columns:
                    sector_returns[symbol] = data['close'].pct_change()
            
            if not sector_returns:
                return sector_analysis
            
            # 20일 상대 강도 계산
            sector_df = pd.DataFrame(sector_returns)
            for symbol in sector_df.columns:
                sector_df[f'{symbol}_rs'] = sector_df[symbol].rolling(20).mean()
            
            # 최근 상대 강도 순위
            rs_columns = [col for col in sector_df.columns if col.endswith('_rs')]
            latest_rs = sector_df[rs_columns].iloc[-1]
            ranked_sectors = latest_rs.rank(ascending=False)
            
            # 섹터 강도 분류
            for symbol in sector_data.keys():
                if symbol in ranked_sectors.index:
                    rank = ranked_sectors[symbol]
                    total_sectors = len(ranked_sectors)
                    
                    if rank <= total_sectors * 0.3:  # 상위 30%
                        sector_analysis[symbol] = SectorStrength.LEADING
                    elif rank >= total_sectors * 0.7:  # 하위 30%
                        sector_analysis[symbol] = SectorStrength.LAGGING
                    else:
                        # 섹터 타입에 따라 분류
                        sector_type = self.sector_classification.get(symbol, {}).get('type', SectorStrength.CYCLICAL)
                        sector_analysis[symbol] = sector_type
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"섹터 로테이션 분석 중 오류: {e}")
            return {}
    
    def generate_sector_recommendations(self, market_condition: MarketCondition, 
                                      sector_analysis: Dict[str, SectorStrength]) -> Dict[str, Any]:
        """시장 조건과 섹터 분석을 기반으로 추천 생성"""
        recommendations = {
            'overweight_sectors': [],
            'underweight_sectors': [],
            'neutral_sectors': [],
            'strategy': '',
            'risk_level': 'medium'
        }
        
        if market_condition == MarketCondition.BULL_MARKET:
            # 강세장: 순환적 섹터 중 선도 섹터 선호
            recommendations['strategy'] = 'Momentum following - 순환적 섹터 중 선도 섹터 집중'
            recommendations['risk_level'] = 'high'
            
            for symbol, strength in sector_analysis.items():
                sector_type = self.sector_classification.get(symbol, {}).get('type')
                if strength == SectorStrength.LEADING and sector_type == SectorStrength.CYCLICAL:
                    recommendations['overweight_sectors'].append(symbol)
                elif strength == SectorStrength.LAGGING:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        elif market_condition == MarketCondition.BEAR_MARKET:
            # 약세장: 방어적 섹터 선호
            recommendations['strategy'] = 'Defensive positioning - 방어적 섹터 집중'
            recommendations['risk_level'] = 'low'
            
            for symbol, strength in sector_analysis.items():
                sector_type = self.sector_classification.get(symbol, {}).get('type')
                if sector_type == SectorStrength.DEFENSIVE:
                    recommendations['overweight_sectors'].append(symbol)
                elif sector_type == SectorStrength.CYCLICAL:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        elif market_condition == MarketCondition.RECESSION_FEAR:
            # 경기침체 우려: 방어적 섹터 + 국채
            recommendations['strategy'] = 'Recession hedge - 방어적 섹터 + 국채 비중 확대'
            recommendations['risk_level'] = 'low'
            
            for symbol, strength in sector_analysis.items():
                sector_type = self.sector_classification.get(symbol, {}).get('type')
                if sector_type == SectorStrength.DEFENSIVE:
                    recommendations['overweight_sectors'].append(symbol)
                elif sector_type == SectorStrength.CYCLICAL:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        elif market_condition == MarketCondition.INFLATION_FEAR:
            # 인플레이션 우려: 실물자산 + 에너지 + 원자재
            recommendations['strategy'] = 'Inflation hedge - 실물자산 + 에너지 + 원자재 집중'
            recommendations['risk_level'] = 'medium'
            
            for symbol, strength in sector_analysis.items():
                sector_type = self.sector_classification.get(symbol, {}).get('type')
                if symbol in ['XLE', 'XLB']:  # 에너지, 원자재
                    recommendations['overweight_sectors'].append(symbol)
                elif symbol in ['XLF', 'XLK']:  # 금융, 기술 (인플레이션에 취약)
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        elif market_condition == MarketCondition.VOLATILE_MARKET:
            # 변동성 장: 분산 투자 + 현금 비중 확대
            recommendations['strategy'] = 'Diversification - 분산 투자 + 현금 비중 확대'
            recommendations['risk_level'] = 'medium'
            
            # 모든 섹터를 중립으로 설정하고 현금 비중 확대
            recommendations['neutral_sectors'] = list(sector_analysis.keys())
        
        else:  # SIDEWAYS_MARKET
            # 횡보장: 스윙 트레이딩
            recommendations['strategy'] = 'Swing trading - 섹터 로테이션 활용'
            recommendations['risk_level'] = 'medium'
            
            for symbol, strength in sector_analysis.items():
                if strength == SectorStrength.LEADING:
                    recommendations['overweight_sectors'].append(symbol)
                elif strength == SectorStrength.LAGGING:
                    recommendations['underweight_sectors'].append(symbol)
                else:
                    recommendations['neutral_sectors'].append(symbol)
        
        return recommendations
    
    def get_comprehensive_analysis(self, start_date: str = None, end_date: str = None, 
                                 spy_data: pd.DataFrame = None, macro_data: Dict[str, pd.DataFrame] = None, 
                                 sector_data: Dict[str, pd.DataFrame] = None) -> MacroAnalysis:
        """매크로 & 섹터 종합 분석"""
        try:
            # 날짜 설정
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                # 설정에서 데이터 수집 기간 가져오기
                days_back = self.macro_collector._get_days_back("macro_analysis")
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # 데이터 수집 (이미 제공된 데이터가 있으면 재사용)
            if spy_data is None or macro_data is None or sector_data is None:
                self.logger.info("매크로 & 섹터 데이터 수집 중...")
                spy_data = self.macro_collector.collect_spy_data(start_date, end_date)
                macro_data = self.macro_collector.collect_macro_indicators(start_date, end_date)
                sector_data = self.macro_collector.collect_sector_data(start_date, end_date)
            else:
                self.logger.info("제공된 매크로 & 섹터 데이터 재사용 중...")
            
            # 매크로 환경 분석
            macro_analysis = self.analyze_macro_environment(macro_data)
            
            # 시장 조건 분류
            market_condition = self.classify_market_condition(macro_analysis)
            
            # 섹터 로테이션 분석
            sector_analysis = self.analyze_sector_rotation(sector_data)
            
            # 추천 생성
            recommendations = self.generate_sector_recommendations(market_condition, sector_analysis)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(macro_analysis, sector_analysis)
            
            return MacroAnalysis(
                market_condition=market_condition,
                confidence=confidence,
                key_indicators=macro_analysis,
                sector_rotation=sector_analysis,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"매크로 & 섹터 분석 중 오류: {e}")
            return None
    
    def _calculate_confidence(self, macro_analysis: Dict[str, Any], 
                            sector_analysis: Dict[str, SectorStrength]) -> float:
        """매크로 분석 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 매크로 지표 개수에 따른 신뢰도 조정
        macro_indicators = len(macro_analysis)
        if macro_indicators >= 5:
            confidence += 0.2
        elif macro_indicators >= 3:
            confidence += 0.1
        
        # 섹터 데이터 개수에 따른 신뢰도 조정
        sector_count = len(sector_analysis)
        if sector_count >= 7:
            confidence += 0.2
        elif sector_count >= 5:
            confidence += 0.1
        
        # VIX 변동성에 따른 신뢰도 조정
        if 'vix_current' in macro_analysis:
            vix = macro_analysis['vix_current']
            if 15 <= vix <= 25:  # 정상 변동성
                confidence += 0.1
        
        return min(confidence, 1.0)  # 최대 1.0
    
    def save_analysis_results(self, analysis: MacroAnalysis, output_dir: str = "results/macro_sector_analysis"):
        """매크로 분석 결과 저장"""
        try:
            # UUID 기반 디렉토리 생성
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            # 분석 결과를 딕셔너리로 변환
            results = {
                'session_uuid': self.session_uuid,
                'market_condition': analysis.market_condition.value,
                'confidence': analysis.confidence,
                'key_indicators': analysis.key_indicators,
                'sector_rotation': {k: v.value for k, v in analysis.sector_rotation.items()},
                'recommendations': analysis.recommendations,
                'timestamp': analysis.timestamp.isoformat(),
                'created_at': datetime.now().isoformat()
            }
            
            # JSON 파일로 저장
            with open(f"{session_dir}/analysis_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"매크로 분석 결과 저장 완료: {session_dir}")
            
        except Exception as e:
            self.logger.error(f"매크로 분석 결과 저장 중 오류: {e}")


class HyperparamTuner:
    """하이퍼파라미터 튜닝 클래스"""
    
    def __init__(self, config_path: str = "config/config_macro.json", session_uuid: str = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.session_uuid = session_uuid or str(uuid.uuid4())
        self.collector = GlobalMacroDataCollector(self.session_uuid)
        self.logger.info(f"HyperparamTuner 초기화 - Session UUID: {self.session_uuid}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator 계산"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """CCI (Commodity Channel Index) 계산"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R 계산"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands 계산"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def _calculate_keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels 계산"""
        typical_price = (high + low + close) / 3
        atr = self._calculate_atr(high, low, close, period)
        
        middle = typical_price.rolling(window=period).mean()
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return upper, middle, lower
    
    def _calculate_donchian_channels(self, high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels 계산"""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    def _calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """VWAP (Volume Weighted Average Price) 계산"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return vwap
    
    def _calculate_money_flow_index(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index 계산"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
        return mfi
    
    def _calculate_pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 1) -> pd.Series:
        """Pivot Points 계산"""
        pivot = (high.rolling(window=period).max() + 
                low.rolling(window=period).min() + 
                close.rolling(window=period).mean()) / 3
        return pivot
    
    def _calculate_fibonacci_levels(self, high: pd.Series, low: pd.Series, level: float) -> pd.Series:
        """Fibonacci Retracement 레벨 계산"""
        swing_high = high.rolling(window=20).max()
        swing_low = low.rolling(window=20).min()
        
        fib_level = swing_high - (level * (swing_high - swing_low))
        return fib_level
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV (On-Balance Volume) 계산"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX (Average Directional Index) 계산"""
        # +DM, -DM 계산
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # TR 계산
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # +DI, -DI 계산
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean()
        
        # ADX 계산
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx, plus_di, minus_di
    
    def _calculate_supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """SuperTrend 계산"""
        atr = self._calculate_atr(high, low, close, period)
        
        # Basic Upper and Lower Bands
        basic_upper = (high + low) / 2 + (multiplier * atr)
        basic_lower = (high + low) / 2 - (multiplier * atr)
        
        # Final Upper and Lower Bands
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        
        for i in range(1, len(close)):
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
                
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # SuperTrend
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        for i in range(1, len(close)):
            if close.iloc[i] > final_upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < final_lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
        
        return supertrend
    
    def _calculate_technical_scores(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """기술적 지표 점수 계산 (분리된 메서드)"""
        regime_scores = pd.DataFrame(index=data.index)
        
        # 컬럼명 매핑 (대소문자 처리)
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 1. 트렌드 점수 (로그 수익률 지표 추가)
        if 'trend_weight' in params:
            trend_score = 0
            
            # 로그 수익률 계산 (주요 지표)
            log_returns = np.log(data[close_col] / data[close_col].shift(1))
            log_returns_ma = log_returns.rolling(window=params.get('log_returns_ma_period', 20)).mean()
            log_returns_std = log_returns.rolling(window=params.get('log_returns_ma_period', 20)).std()
            
            # 로그 수익률 기반 트렌드 점수
            log_return_trend = np.where(
                ~log_returns_ma.isna(),
                np.where(log_returns_ma > params.get('log_returns_ma_threshold', 0.0005), 0.5,
                         np.where(log_returns_ma < -params.get('log_returns_ma_threshold', 0.0005), -0.5, 0)),
                0
            )
            
            # 로그 수익률 변동성 점수
            log_return_volatility = np.where(
                ~log_returns_std.isna(),
                np.where(log_returns_std > log_returns_std.quantile(0.7), 0.3, 0),  # 높은 변동성
                0
            )
            
            # 기존 지표들
            if f'sma_{params.get("sma_short", 20)}' in data.columns and f'sma_{params.get("sma_long", 50)}' in data.columns:
                sma_short = data[f'sma_{params.get("sma_short", 20)}']
                sma_long = data[f'sma_{params.get("sma_long", 50)}']
                valid_mask = ~(sma_short.isna() | sma_long.isna())
                
                adx_strength = 0
                if 'adx' in data.columns:
                    adx = data['adx']
                    adx_threshold = params.get('adx_threshold', 25)
                    adx_strength = np.where(valid_mask & (adx > adx_threshold), 0.3, 0)
                
                supertrend_signal = 0
                if 'supertrend' in data.columns:
                    supertrend = data['supertrend']
                    supertrend_signal = np.where(valid_mask & (data[close_col] > supertrend), 0.2, -0.2)
                
                sma_trend = np.where(valid_mask, 
                    np.where(sma_short > sma_long, 0.5, -0.5), 0)
                
                # 종합 트렌드 점수 (로그 수익률 가중치 강화)
                trend_score = log_return_trend + log_return_volatility + sma_trend + adx_strength + supertrend_signal
            else:
                # SMA가 없는 경우 로그 수익률만 사용
                trend_score = log_return_trend + log_return_volatility
                
            regime_scores['trend_score'] = trend_score * params.get('trend_weight', 0.3)
        
        # 2. 모멘텀 점수
        if 'momentum_weight' in params:
            momentum_score = 0
            if 'rsi' in data.columns:
                rsi = data['rsi']
                valid_mask = ~rsi.isna()
                
                rsi_momentum = np.where(
                    valid_mask,
                    np.where(
                        (rsi > params.get('rsi_oversold', 30)) & (rsi < params.get('rsi_overbought', 70)),
                        0, np.where(rsi > params.get('rsi_overbought', 70), -1, 1)
                    ),
                    0
                )
                
                macd_momentum = 0
                if 'macd' in data.columns and 'macd_signal' in data.columns:
                    macd = data['macd']
                    macd_signal = data['macd_signal']
                    macd_momentum = np.where(valid_mask & (macd > macd_signal), 0.5, -0.5)
                
                stoch_momentum = 0
                if 'stoch_k' in data.columns:
                    stoch_k = data['stoch_k']
                    stoch_momentum = np.where(valid_mask & (stoch_k > 80), -0.3, 
                                            np.where(valid_mask & (stoch_k < 20), 0.3, 0))
                
                momentum_score = rsi_momentum + macd_momentum + stoch_momentum
            regime_scores['momentum_score'] = momentum_score * params.get('momentum_weight', 0.3)
        
        # 3. 변동성 점수
        if 'volatility_weight' in params:
            volatility_score = 0
            if 'atr' in data.columns and close_col in data.columns:
                atr_ratio = data['atr'] / data[close_col]
                valid_mask = ~(atr_ratio.isna())
                
                atr_volatility = np.where(valid_mask, np.where(atr_ratio > 0.02, 1, 0), 0)
                
                bb_volatility = 0
                if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                    bb_width = (data['bb_upper'] - data['bb_lower']) / data[close_col]
                    bb_volatility = np.where(valid_mask & (bb_width > 0.05), 0.5, 0)
                
                keltner_volatility = 0
                if 'keltner_upper' in data.columns and 'keltner_lower' in data.columns:
                    keltner_width = (data['keltner_upper'] - data['keltner_lower']) / data[close_col]
                    keltner_volatility = np.where(valid_mask & (keltner_width > 0.03), 0.3, 0)
                
                volatility_score = atr_volatility + bb_volatility + keltner_volatility
            regime_scores['volatility_score'] = volatility_score * params.get('volatility_weight', 0.2)
        
        # 4. 매크로 점수
        if 'macro_weight' in params:
            macro_score = 0
            # VIX 데이터 컬럼명 매핑 수정
            vix_col = None
            if '^VIX' in data.columns:
                vix_col = '^VIX'
            elif 'close' in data.columns and '^vix_data' in str(data.columns):  # VIX 데이터의 close 컬럼
                vix_col = 'close'
            elif 'close' in data.columns:  # 일반적인 close 컬럼 (VIX 데이터인 경우)
                vix_col = 'close'
            
            if vix_col:
                vix = data[vix_col]
                valid_mask = ~vix.isna()
                vix_score = np.where(valid_mask, np.where(vix > params.get('vix_threshold', 25), 1, 0), 0)
                
                vix_change = vix.pct_change()
                vix_momentum = np.where(valid_mask & (vix_change > 0.1), 0.5, 0)
                
                macro_score = vix_score + vix_momentum
            regime_scores['macro_score'] = macro_score * params.get('macro_weight', 0.1)
        
        # 5. 거래량 점수
        if 'volume_weight' in params:
            volume_score = 0
            # 거래량 데이터 컬럼명 매핑 수정
            volume_col = None
            volume_ma_col = None
            obv_col = None
            
            if 'volume' in data.columns:
                volume_col = 'volume'
            if 'average_volume' in data.columns:  # SPY 데이터의 average_volume 컬럼 사용
                volume_ma_col = 'average_volume'
            elif 'volume_ma' in data.columns:
                volume_ma_col = 'volume_ma'
            if 'obv' in data.columns:
                obv_col = 'obv'
            
            if volume_col and volume_ma_col:
                volume = data[volume_col]
                volume_ma = data[volume_ma_col]
                valid_mask = ~(volume.isna() | volume_ma.isna())
                
                volume_ratio = volume / volume_ma
                volume_score = np.where(valid_mask & (volume_ratio > 1.5), 0.5, 
                                      np.where(valid_mask & (volume_ratio < 0.5), -0.3, 0))
                
                # OBV 기반 추가 점수
                if obv_col:
                    obv = data[obv_col]
                    obv_change = obv.pct_change()
                    obv_score = np.where(valid_mask & (obv_change > 0.02), 0.3, 
                                       np.where(valid_mask & (obv_change < -0.02), -0.3, 0))
                    volume_score += obv_score
            elif volume_col:  # volume만 있는 경우
                volume = data[volume_col]
                valid_mask = ~volume.isna()
                
                # volume의 이동평균을 계산
                volume_ma = volume.rolling(window=params.get('volume_ma_period', 20)).mean()
                volume_ratio = volume / volume_ma
                volume_score = np.where(valid_mask & (volume_ratio > 1.5), 0.5, 
                                      np.where(valid_mask & (volume_ratio < 0.5), -0.3, 0))
                
                # OBV 기반 추가 점수
                if obv_col:
                    obv = data[obv_col]
                    obv_change = obv.pct_change()
                    obv_score = np.where(valid_mask & (obv_change > 0.02), 0.3, 
                                       np.where(valid_mask & (obv_change < -0.02), -0.3, 0))
                    volume_score += obv_score
                    
            regime_scores['volume_score'] = volume_score * params.get('volume_weight', 0.1)
        
        # 6. 지지/저항 점수
        if 'support_resistance_weight' in params:
            sr_score = 0
            # 지지/저항 데이터 컬럼명 매핑 수정
            pivot_col = None
            fibonacci_cols = []
            
            if 'pivot_point' in data.columns:
                pivot_col = 'pivot_point'
            elif 'pivot' in data.columns:
                pivot_col = 'pivot'
            
            # Fibonacci 레벨들 확인
            for level in ['0.236', '0.382', '0.500', '0.618', '0.786']:
                fib_col = f'fibonacci_{level}'
                if fib_col in data.columns:
                    fibonacci_cols.append(fib_col)
            
            # Pivot Points 기반 점수
            if pivot_col:
                pivot = data[pivot_col]
                valid_mask = ~pivot.isna()
                
                support_distance = (data[close_col] - pivot) / data[close_col]
                pivot_score = np.where(valid_mask & (abs(support_distance) < 0.01), 0.3, 0)
                sr_score += pivot_score
            
            # Fibonacci 레벨 기반 점수
            if fibonacci_cols:
                for fib_col in fibonacci_cols:
                    fib_level = data[fib_col]
                    valid_mask = ~fib_level.isna()
                    
                    # 가격이 Fibonacci 레벨 근처에 있는지 확인
                    fib_distance = (data[close_col] - fib_level) / data[close_col]
                    fib_score = np.where(valid_mask & (abs(fib_distance) < 0.005), 0.2, 0)  # 더 엄격한 임계값
                    sr_score += fib_score
            
            # Bollinger Bands 기반 지지/저항 점수 (추가)
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                bb_upper = data['bb_upper']
                bb_lower = data['bb_lower']
                valid_mask = ~(bb_upper.isna() | bb_lower.isna())
                
                # 가격이 Bollinger Band 경계 근처에 있는지 확인
                upper_distance = (bb_upper - data[close_col]) / data[close_col]
                lower_distance = (data[close_col] - bb_lower) / data[close_col]
                
                bb_score = np.where(valid_mask & (upper_distance < 0.01), 0.2, 0)  # 저항선 근처
                bb_score += np.where(valid_mask & (lower_distance < 0.01), 0.2, 0)  # 지지선 근처
                sr_score += bb_score
            
            # Keltner Channels 기반 지지/저항 점수 (추가)
            if 'keltner_upper' in data.columns and 'keltner_lower' in data.columns:
                keltner_upper = data['keltner_upper']
                keltner_lower = data['keltner_lower']
                valid_mask = ~(keltner_upper.isna() | keltner_lower.isna())
                
                # 가격이 Keltner Channel 경계 근처에 있는지 확인
                keltner_upper_distance = (keltner_upper - data[close_col]) / data[close_col]
                keltner_lower_distance = (data[close_col] - keltner_lower) / data[close_col]
                
                keltner_score = np.where(valid_mask & (keltner_upper_distance < 0.01), 0.15, 0)  # 저항선 근처
                keltner_score += np.where(valid_mask & (keltner_lower_distance < 0.01), 0.15, 0)  # 지지선 근처
                sr_score += keltner_score
                
            regime_scores['sr_score'] = sr_score * params.get('support_resistance_weight', 0.1)
        
        return regime_scores
    
    def _calculate_derived_features(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """파생 변수 계산 (고급 버전)"""
        result = data.copy()
        
        # 컬럼명 매핑 (대소문자 처리)
        close_col = 'close' if 'close' in data.columns else 'Close'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'
        volume_col = 'volume' if 'volume' in data.columns else 'Volume'
        
        # 1. 이동평균 (기존)
        sma_short = params.get('sma_short', 20)
        sma_medium = params.get('sma_medium', 50)
        sma_long = params.get('sma_long', 200)
        
        result[f'sma_{sma_short}'] = result[close_col].rolling(sma_short).mean()
        result[f'sma_{sma_medium}'] = result[close_col].rolling(sma_medium).mean()
        result[f'sma_{sma_long}'] = result[close_col].rolling(sma_long).mean()
        
        # EMA
        ema_short = params.get('ema_short', 12)
        ema_long = params.get('ema_long', 26)
        
        result[f'ema_{ema_short}'] = result[close_col].ewm(span=ema_short).mean()
        result[f'ema_{ema_long}'] = result[close_col].ewm(span=ema_long).mean()
        
        # 2. 모멘텀 지표 (기존 + 개선)
        # RSI
        rsi_period = params.get('rsi_period', 14)
        result['rsi'] = self._calculate_rsi(result[close_col], rsi_period)
        
        # MACD
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        
        macd, macd_signal_line, macd_histogram = self._calculate_macd(
            result[close_col], macd_fast, macd_slow, macd_signal
        )
        result['macd'] = macd
        result['macd_signal'] = macd_signal_line
        result['macd_histogram'] = macd_histogram
        
        # Stochastic
        stoch_k_period = params.get('stoch_k_period', 14)
        stoch_d_period = params.get('stoch_d_period', 3)
        stoch_k, stoch_d = self._calculate_stochastic(
            result[high_col], result[low_col], result[close_col], 
            stoch_k_period, stoch_d_period
        )
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d
        
        # CCI
        cci_period = params.get('cci_period', 20)
        result['cci'] = self._calculate_cci(result[high_col], result[low_col], result[close_col], cci_period)
        
        # Williams %R
        williams_r_period = params.get('williams_r_period', 14)
        result['williams_r'] = self._calculate_williams_r(result[high_col], result[low_col], result[close_col], williams_r_period)
        
        # 3. 변동성 지표 (기존 + 개선)
        # ATR
        atr_period = params.get('atr_period', 14)
        result['atr'] = self._calculate_atr(result[high_col], result[low_col], result[close_col], atr_period)
        
        # Bollinger Bands
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(result[close_col], bb_period, bb_std)
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        
        # Keltner Channels
        keltner_period = params.get('keltner_period', 20)
        keltner_multiplier = params.get('keltner_multiplier', 2.0)
        keltner_upper, keltner_middle, keltner_lower = self._calculate_keltner_channels(
            result[high_col], result[low_col], result[close_col], keltner_period, keltner_multiplier
        )
        result['keltner_upper'] = keltner_upper
        result['keltner_middle'] = keltner_middle
        result['keltner_lower'] = keltner_lower
        
        # Donchian Channels
        donchian_period = params.get('donchian_period', 20)
        donchian_upper, donchian_middle, donchian_lower = self._calculate_donchian_channels(
            result[high_col], result[low_col], donchian_period
        )
        result['donchian_upper'] = donchian_upper
        result['donchian_middle'] = donchian_middle
        result['donchian_lower'] = donchian_lower
        
        # 4. 거래량 지표 (새로 추가)
        # Volume MA
        volume_ma_period = params.get('volume_ma_period', 20)
        result['volume_ma'] = result[volume_col].rolling(volume_ma_period).mean()
        
        # OBV (On-Balance Volume)
        result['obv'] = self._calculate_obv(result[close_col], result[volume_col])
        
        # 5. 지지/저항 지표 (새로 추가)
        # Pivot Points
        pivot_period = params.get('pivot_period', 1)
        result['pivot_point'] = self._calculate_pivot_points(result[high_col], result[low_col], result[close_col], pivot_period)
        
        # Fibonacci Levels
        fibonacci_levels = params.get('fibonacci_levels', 5)
        result['fibonacci_0.236'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.236)
        result['fibonacci_0.382'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.382)
        result['fibonacci_0.500'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.500)
        result['fibonacci_0.618'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.618)
        result['fibonacci_0.786'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.786)
        
        # VWAP (Volume Weighted Average Price)
        vwap_period = params.get('vwap_period', 20)
        result['vwap'] = self._calculate_vwap(result[high_col], result[low_col], result[close_col], result[volume_col], vwap_period)
        
        # Money Flow Index
        money_flow_period = params.get('money_flow_period', 14)
        result['mfi'] = self._calculate_money_flow_index(result[high_col], result[low_col], result[close_col], result[volume_col], money_flow_period)
        
        # Pivot Points
        pivot_period = params.get('pivot_period', 1)
        result['pivot'] = self._calculate_pivot_points(result[high_col], result[low_col], result[close_col], pivot_period)
        
        # Fibonacci Retracement
        fibonacci_levels = params.get('fibonacci_levels', 5)
        result['fib_23_6'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.236)
        result['fib_38_2'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.382)
        result['fib_50_0'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.500)
        result['fib_61_8'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.618)
        
        # Money Flow Index
        money_flow_period = params.get('money_flow_period', 14)
        result['money_flow'] = self._calculate_money_flow_index(
            result[high_col], result[low_col], result[close_col], result[volume_col], money_flow_period
        )
        
        # 5. 트렌드 지표 (새로 추가)
        # ADX (Average Directional Index)
        adx_period = params.get('adx_period', 14)
        adx, plus_di, minus_di = self._calculate_adx(
            result[high_col], result[low_col], result[close_col], adx_period
        )
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        
        # SuperTrend
        supertrend_period = params.get('supertrend_period', 10)
        supertrend_multiplier = params.get('supertrend_multiplier', 3.0)
        result['supertrend'] = self._calculate_supertrend(
            result[high_col], result[low_col], result[close_col], supertrend_period, supertrend_multiplier
        )
        
        # 6. 지지/저항 지표 (새로 추가)
        # Pivot Points
        pivot_period = params.get('pivot_period', 1)
        result['pivot_point'] = self._calculate_pivot_points(
            result[high_col], result[low_col], result[close_col], pivot_period
        )
        
        # Fibonacci Retracement
        fib_levels = params.get('fibonacci_levels', 5)
        result['fib_38'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.382)
        result['fib_50'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.5)
        result['fib_61'] = self._calculate_fibonacci_levels(result[high_col], result[low_col], 0.618)
        
        # 7. 파생 지표
        # Returns
        result['returns'] = result[close_col].pct_change()
        
        # Volatility
        result['volatility'] = result['returns'].rolling(20).std()
        
        return result
    
    def _classify_market_regime(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """시장 상태 분류 (고급 버전 - SPY 기준 추세+수익률 강화)"""
        regime_scores = pd.DataFrame(index=data.index)
        
        # 컬럼명 매핑 (대소문자 처리)
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 1. SPY 기준 추세+수익률 점수 계산 (강화된 핵심 지표)
        if 'spy_trend_weight' in params:
            spy_trend_score = 0
            if f'sma_{params.get("sma_short", 20)}' in data.columns and f'sma_{params.get("sma_long", 50)}' in data.columns:
                sma_short = data[f'sma_{params.get("sma_short", 20)}']
                sma_long = data[f'sma_{params.get("sma_long", 50)}']
                # NaN 값 처리
                valid_mask = ~(sma_short.isna() | sma_long.isna())
                
                # 1-1. 기본 이동평균 비교 (가중치: 0.4)
                sma_trend = np.where(valid_mask, 
                    np.where(sma_short > sma_long, 1, -1), 0)
                
                # 1-2. ADX 기반 트렌드 강도 (가중치: 0.3)
                adx_strength = 0
                if 'adx' in data.columns:
                    adx = data['adx']
                    adx_threshold = params.get('adx_threshold', 25)
                    adx_strength = np.where(valid_mask & (adx > adx_threshold), 0.5, 0)
                
                # 1-3. SuperTrend 기반 트렌드 확인 (가중치: 0.2)
                supertrend_signal = 0
                if 'supertrend' in data.columns:
                    supertrend = data['supertrend']
                    supertrend_signal = np.where(valid_mask & (data[close_col] > supertrend), 0.3, -0.3)
                
                # 1-4. 수익률 기반 모멘텀 (가중치: 0.1) - 새로 추가
                returns_momentum = 0
                if close_col in data.columns:
                    # 5일, 10일, 20일 수익률 계산
                    returns_5d = data[close_col].pct_change(5)
                    returns_10d = data[close_col].pct_change(10)
                    returns_20d = data[close_col].pct_change(20)
                    
                    # 수익률 기반 점수 (단기 > 중기 > 장기 순으로 가중치)
                    returns_score = np.where(valid_mask,
                        np.where(returns_5d > 0.02, 0.4,  # 5일 수익률 > 2%
                        np.where(returns_5d < -0.02, -0.4, 0)) +  # 5일 수익률 < -2%
                        np.where(returns_10d > 0.03, 0.3,  # 10일 수익률 > 3%
                        np.where(returns_10d < -0.03, -0.3, 0)) +  # 10일 수익률 < -3%
                        np.where(returns_20d > 0.05, 0.2,  # 20일 수익률 > 5%
                        np.where(returns_20d < -0.05, -0.2, 0)), 0)  # 20일 수익률 < -5%
                    
                    returns_momentum = returns_score
                
                # 1-5. 가격 위치 기반 점수 (가중치: 0.1) - 새로 추가
                price_position_score = 0
                if 'bb_upper' in data.columns and 'bb_lower' in data.columns and 'bb_middle' in data.columns:
                    bb_upper = data['bb_upper']
                    bb_lower = data['bb_lower']
                    bb_middle = data['bb_middle']
                    
                    # 가격이 Bollinger Band 내에서의 위치
                    bb_position = (data[close_col] - bb_lower) / (bb_upper - bb_lower)
                    price_position_score = np.where(valid_mask & (bb_position > 0.8), 0.2,  # 상단 20%
                                          np.where(valid_mask & (bb_position < 0.2), -0.2, 0))  # 하단 20%
                
                # SPY 추세+수익률 종합 점수
                spy_trend_score = (sma_trend * 0.4 + adx_strength * 0.3 + 
                                 supertrend_signal * 0.2 + returns_momentum * 0.1 + 
                                 price_position_score * 0.1)
            
            regime_scores['spy_trend_score'] = spy_trend_score * params['spy_trend_weight']
        
        # 2. 모멘텀 점수 계산 (기존 유지, 가중치 조정)
        if 'momentum_weight' in params:
            momentum_score = 0
            if 'rsi' in data.columns:
                rsi = data['rsi']
                # NaN 값 처리
                valid_mask = ~rsi.isna()
                
                # RSI 기반 모멘텀 (가중치: 0.5)
                rsi_momentum = np.where(
                    valid_mask,
                    np.where(
                        (rsi > params.get('rsi_oversold', 30)) & (rsi < params.get('rsi_overbought', 70)),
                        0, np.where(rsi > params.get('rsi_overbought', 70), -1, 1)
                    ),
                    0
                )
                
                # MACD 기반 모멘텀 (가중치: 0.3)
                macd_momentum = 0
                if 'macd' in data.columns and 'macd_signal' in data.columns:
                    macd = data['macd']
                    macd_signal = data['macd_signal']
                    macd_momentum = np.where(valid_mask & (macd > macd_signal), 0.5, -0.5)
                
                # Stochastic 기반 모멘텀 (가중치: 0.2)
                stoch_momentum = 0
                if 'stoch_k' in data.columns:
                    stoch_k = data['stoch_k']
                    stoch_momentum = np.where(valid_mask & (stoch_k > 80), -0.3, 
                                            np.where(valid_mask & (stoch_k < 20), 0.3, 0))
                
                momentum_score = (rsi_momentum * 0.5 + macd_momentum * 0.3 + stoch_momentum * 0.2)
            regime_scores['momentum_score'] = momentum_score * params['momentum_weight']
        
        # 3. 변동성 점수 계산 (기존 유지)
        if 'volatility_weight' in params:
            volatility_score = 0
            if 'atr' in data.columns and close_col in data.columns:
                atr_ratio = data['atr'] / data[close_col]
                # NaN 값 처리
                valid_mask = ~(atr_ratio.isna())
                
                # ATR 기반 변동성
                atr_volatility = np.where(valid_mask, np.where(atr_ratio > 0.02, 1, 0), 0)
                
                # Bollinger Band 기반 변동성 추가
                bb_volatility = 0
                if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                    bb_width = (data['bb_upper'] - data['bb_lower']) / data[close_col]
                    bb_volatility = np.where(valid_mask & (bb_width > 0.05), 0.5, 0)
                
                # Keltner Channel 기반 변동성 추가
                keltner_volatility = 0
                if 'keltner_upper' in data.columns and 'keltner_lower' in data.columns:
                    keltner_width = (data['keltner_upper'] - data['keltner_lower']) / data[close_col]
                    keltner_volatility = np.where(valid_mask & (keltner_width > 0.03), 0.3, 0)
                
                volatility_score = atr_volatility + bb_volatility + keltner_volatility
            regime_scores['volatility_score'] = volatility_score * params['volatility_weight']
        
        # 4. 고도화된 매크로 점수 계산 (VIX + TIPS + 기타 지표)
        if 'macro_weight' in params:
            macro_score = 0
            
            # 4-1. VIX 기반 변동성 점수 (고도화)
            if '^VIX' in data.columns:
                vix = data['^VIX']
                # NaN 값 처리
                valid_mask = ~vix.isna()
                
                # VIX 레벨 기반 점수 (가중치: 0.4)
                vix_level_score = np.where(valid_mask, 
                    np.where(vix > params.get('vix_high_threshold', 30), 1,  # 높은 변동성
                    np.where(vix > params.get('vix_medium_threshold', 20), 0.5,  # 중간 변동성
                    np.where(vix < params.get('vix_low_threshold', 15), -0.3, 0))), 0)  # 낮은 변동성
                
                # VIX 변화율 기반 점수 (가중치: 0.3)
                vix_change = vix.pct_change()
                vix_momentum_score = np.where(valid_mask, 
                    np.where(vix_change > 0.15, 0.8,  # 급격한 상승
                    np.where(vix_change > 0.05, 0.4,  # 상승
                    np.where(vix_change < -0.15, -0.8,  # 급격한 하락
                    np.where(vix_change < -0.05, -0.4, 0)))), 0)  # 하락
                
                # VIX 이동평균 기반 점수 (가중치: 0.3)
                vix_ma_score = 0
                if len(vix) >= 20:
                    vix_ma_20 = vix.rolling(20).mean()
                    vix_ma_score = np.where(valid_mask & (vix > vix_ma_20 * 1.2), 0.5,  # VIX > MA20 * 1.2
                                  np.where(valid_mask & (vix < vix_ma_20 * 0.8), -0.3, 0))  # VIX < MA20 * 0.8
                
                vix_total_score = (vix_level_score * 0.4 + vix_momentum_score * 0.3 + vix_ma_score * 0.3)
                macro_score += vix_total_score
            
            # 4-2. TIPS Spread 기반 인플레이션 점수 (새로 추가)
            if 'TIPS_SPREAD' in data.columns:
                tips_spread = data['TIPS_SPREAD']
                valid_mask = ~tips_spread.isna()
                
                # TIPS Spread 레벨 기반 점수
                tips_level_score = np.where(valid_mask,
                    np.where(tips_spread > params.get('tips_high_threshold', 2.5), 0.8,  # 높은 인플레이션 기대
                    np.where(tips_spread > params.get('tips_medium_threshold', 2.0), 0.4,  # 중간 인플레이션 기대
                    np.where(tips_spread < params.get('tips_low_threshold', 1.5), -0.4, 0))), 0)  # 낮은 인플레이션 기대
                
                # TIPS Spread 변화율 기반 점수
                tips_change = tips_spread.pct_change()
                tips_momentum_score = np.where(valid_mask,
                    np.where(tips_change > 0.1, 0.5,  # 인플레이션 기대 상승
                    np.where(tips_change < -0.1, -0.5, 0)), 0)  # 인플레이션 기대 하락
                
                tips_total_score = tips_level_score + tips_momentum_score
                macro_score += tips_total_score
            
            # 4-3. 달러 인덱스 기반 점수 (새로 추가)
            if '^DXY' in data.columns:
                dxy = data['^DXY']
                valid_mask = ~dxy.isna()
                
                # 달러 강도 기반 점수
                dxy_ma_20 = dxy.rolling(20).mean()
                dxy_strength_score = np.where(valid_mask & (dxy > dxy_ma_20 * 1.05), 0.3,  # 강한 달러
                                    np.where(valid_mask & (dxy < dxy_ma_20 * 0.95), -0.3, 0))  # 약한 달러
                
                macro_score += dxy_strength_score
            
            # 4-4. 금 가격 기반 점수 (새로 추가)
            if 'GC=F' in data.columns:
                gold = data['GC=F']
                valid_mask = ~gold.isna()
                
                # 금 가격 추세 기반 점수
                gold_ma_20 = gold.rolling(20).mean()
                gold_trend_score = np.where(valid_mask & (gold > gold_ma_20 * 1.05), 0.2,  # 금 상승
                                  np.where(valid_mask & (gold < gold_ma_20 * 0.95), -0.2, 0))  # 금 하락
                
                macro_score += gold_trend_score
            
            # 4-5. 국채 수익률 기반 점수 (새로 추가)
            if '^TNX' in data.columns:
                treasury_10y = data['^TNX']
                valid_mask = ~treasury_10y.isna()
                
                # 10년 국채 수익률 기반 점수
                treasury_ma_20 = treasury_10y.rolling(20).mean()
                treasury_score = np.where(valid_mask & (treasury_10y > treasury_ma_20 * 1.1), 0.3,  # 금리 상승
                                np.where(valid_mask & (treasury_10y < treasury_ma_20 * 0.9), -0.3, 0))  # 금리 하락
                
                macro_score += treasury_score
            
            regime_scores['macro_score'] = macro_score * params['macro_weight']
        
        # 5. 거래량 점수 계산 (기존 유지)
        if 'volume_weight' in params:
            volume_score = 0
            # 거래량 데이터 컬럼명 매핑 수정
            volume_col = None
            volume_ma_col = None
            obv_col = None
            
            if 'volume' in data.columns:
                volume_col = 'volume'
            if 'volume_ma' in data.columns:
                volume_ma_col = 'volume_ma'
            if 'obv' in data.columns:
                obv_col = 'obv'
            
            if volume_col and volume_ma_col:
                volume = data[volume_col]
                volume_ma = data[volume_ma_col]
                valid_mask = ~(volume.isna() | volume_ma.isna())
                
                volume_ratio = volume / volume_ma
                volume_score = np.where(valid_mask & (volume_ratio > 1.5), 0.5, 
                                      np.where(valid_mask & (volume_ratio < 0.5), -0.3, 0))
                
                if obv_col:
                    obv = data[obv_col]
                    obv_change = obv.pct_change()
                    obv_score = np.where(valid_mask & (obv_change > 0.02), 0.3, 
                                       np.where(valid_mask & (obv_change < -0.02), -0.3, 0))
                    volume_score += obv_score
            regime_scores['volume_score'] = volume_score * params.get('volume_weight', 0.1)
        
        # 6. 지지/저항 점수 계산 (기존 유지)
        if 'support_resistance_weight' in params:
            sr_score = 0
            # 지지/저항 데이터 컬럼명 매핑 수정
            pivot_col = None
            
            if 'pivot_point' in data.columns:
                pivot_col = 'pivot_point'
            elif 'pivot' in data.columns:
                pivot_col = 'pivot'
            
            if pivot_col:
                pivot = data[pivot_col]
                valid_mask = ~pivot.isna()
                
                support_distance = (data[close_col] - pivot) / data[close_col]
                sr_score = np.where(valid_mask & (abs(support_distance) < 0.01), 0.3, 0)
            regime_scores['sr_score'] = sr_score * params.get('support_resistance_weight', 0.1)
        
        # 총점 계산 (SPY 추세+수익률 가중치 강화)
        total_score = regime_scores.sum(axis=1)
        
        # 시장 상태 분류 (동적 임계값 사용)
        trending_up_threshold = params.get('trending_up_threshold', 0.2)
        trending_down_threshold = params.get('trending_down_threshold', -0.2)
        volatile_threshold = params.get('volatile_threshold', 0.1)
        
        regime = pd.Series(index=data.index, dtype='object')
        regime = np.where(total_score > trending_up_threshold, MarketRegime.TRENDING_UP.value,
                 np.where(total_score < trending_down_threshold, MarketRegime.TRENDING_DOWN.value,
                 np.where(regime_scores['volatility_score'] > volatile_threshold, MarketRegime.VOLATILE.value,
                 MarketRegime.SIDEWAYS.value)))
        
        return pd.Series(regime, index=data.index)
    
    def _classify_market_regime_with_probabilities(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """시장 상태 분류 (확률 포함)"""
        # 기술적 지표 점수 계산 (분리된 메서드 사용)
        regime_scores = self._calculate_technical_scores(data, params)
        
        # 총점 계산
        total_score = regime_scores.sum(axis=1)
        
        # 각 시장 상태별 확률 계산
        trending_up_threshold = params.get('trending_up_threshold', 0.2)
        trending_down_threshold = params.get('trending_down_threshold', -0.2)
        volatile_threshold = params.get('volatile_threshold', 0.1)
        
        # Random Forest 기반 확률 계산
        def calculate_regime_probabilities_random_forest(scores):
            # 특성 매트릭스 구성 (Random Forest 입력용)
            features = np.column_stack([
                # 기본 특성들
                scores,  # 총점
                regime_scores['trend_score'],  # 트렌드 점수
                regime_scores['momentum_score'],  # 모멘텀 점수
                regime_scores['volatility_score'],  # 변동성 점수
                regime_scores.get('macro_score', np.zeros_like(scores)),  # 매크로 점수
                regime_scores.get('volume_score', np.zeros_like(scores)),  # 거래량 점수
                regime_scores.get('sr_score', np.zeros_like(scores)),  # 지지/저항 점수
                
                # 절댓값 특성들
                np.abs(scores),  # 총점 절댓값
                np.abs(regime_scores['trend_score']),  # 트렌드 절댓값
                np.abs(regime_scores['momentum_score']),  # 모멘텀 절댓값
                np.abs(regime_scores['volatility_score']),  # 변동성 절댓값
                
                # 상호작용 특성들
                regime_scores['trend_score'] * regime_scores['momentum_score'],  # 트렌드-모멘텀 상호작용
                regime_scores['trend_score'] * regime_scores['volatility_score'],  # 트렌드-변동성 상호작용
                regime_scores['momentum_score'] * regime_scores['volatility_score'],  # 모멘텀-변동성 상호작용
                regime_scores['volatility_score'] * np.abs(regime_scores['momentum_score']),  # 변동성-모멘텀 절댓값 상호작용
                
                # 비율 특성들
                np.where(regime_scores['volatility_score'] != 0, 
                        regime_scores['momentum_score'] / regime_scores['volatility_score'], 0),  # 모멘텀/변동성 비율
                np.where(regime_scores['momentum_score'] != 0, 
                        regime_scores['trend_score'] / np.abs(regime_scores['momentum_score']), 0),  # 트렌드/모멘텀 비율
                
                # 제곱 특성들 (비선형 관계 포착)
                scores ** 2,  # 총점 제곱
                regime_scores['trend_score'] ** 2,  # 트렌드 제곱
                regime_scores['momentum_score'] ** 2,  # 모멘텀 제곱
                regime_scores['volatility_score'] ** 2,  # 변동성 제곱
                
                # 범주형 특성들 (임계값 기반)
                (scores > trending_up_threshold).astype(float),  # 상승 추세 여부
                (scores < trending_down_threshold).astype(float),  # 하락 추세 여부
                (regime_scores['volatility_score'] > volatile_threshold).astype(float),  # 고변동성 여부
                (regime_scores['trend_score'] > 0).astype(float),  # 트렌드 양수 여부
                (regime_scores['momentum_score'] > 0).astype(float),  # 모멘텀 양수 여부
                
                # 복합 특성들
                np.where(scores > 0, scores * regime_scores['trend_score'], 0),  # 양수 총점 * 트렌드
                np.where(scores < 0, scores * regime_scores['trend_score'], 0),  # 음수 총점 * 트렌드
                np.where(regime_scores['momentum_score'] > 0, 
                        regime_scores['momentum_score'] * regime_scores['volatility_score'], 0),  # 양수 모멘텀 * 변동성
                np.where(regime_scores['momentum_score'] < 0, 
                        np.abs(regime_scores['momentum_score']) * regime_scores['volatility_score'], 0),  # 음수 모멘텀 * 변동성
                
                # 극값 특성들
                np.where(scores > 0.5, 1.0, 0.0),  # 강한 상승 신호
                np.where(scores < -0.5, 1.0, 0.0),  # 강한 하락 신호
                np.where(regime_scores['volatility_score'] > 0.7, 1.0, 0.0),  # 매우 높은 변동성
                np.where(np.abs(regime_scores['momentum_score']) > 0.6, 1.0, 0.0),  # 매우 높은 모멘텀
                
                # 균형 특성들
                np.abs(scores) * regime_scores['volatility_score'],  # 총점 절댓값 * 변동성
                regime_scores['trend_score'] * np.abs(regime_scores['momentum_score']),  # 트렌드 * 모멘텀 절댓값
                
                # 매크로 관련 특성들
                regime_scores.get('macro_score', np.zeros_like(scores)) * regime_scores['volatility_score'],  # 매크로 * 변동성
                regime_scores.get('volume_score', np.zeros_like(scores)) * np.abs(regime_scores['momentum_score']),  # 거래량 * 모멘텀 절댓값
            ])
            
            # Random Forest 스타일의 의사결정 트리 기반 확률 계산
            def tree_based_probability(features_row, regime_type):
                """의사결정 트리 기반 확률 계산"""
                # 특성 언패킹 (총 35개 특성)
                (score, trend, momentum, volatility, macro, volume, sr, 
                 abs_score, abs_trend, abs_momentum, abs_volatility,
                 trend_momentum, trend_volatility, momentum_volatility, vol_momentum_abs,
                 momentum_vol_ratio, trend_momentum_ratio,
                 score_sq, trend_sq, momentum_sq, volatility_sq,
                 is_trending_up, is_trending_down, is_high_vol, is_trend_pos, is_momentum_pos,
                 pos_score_trend, neg_score_trend, pos_momentum_vol, neg_momentum_vol,
                 strong_up, strong_down, very_high_vol, very_high_momentum,
                 abs_score_vol, trend_abs_momentum, macro_vol, volume_abs_momentum) = features_row
                
                if regime_type == 'trending_up':
                    # TRENDING_UP 조건들 (현실적인 조건들)
                    conditions = [
                        score > trending_up_threshold,  # 총점이 임계값보다 높음
                        trend > 0.15,  # 트렌드 점수가 양수 (임계값 더 낮춤)
                        momentum > 0.1,  # 모멘텀 점수가 양수 (임계값 더 낮춤)
                        trend_momentum > 0.02,  # 트렌드-모멘텀 상호작용이 양수 (임계값 더 낮춤)
                        volatility < 0.7,  # 변동성이 낮음 (임계값 더 높임)
                        is_trend_pos,  # 트렌드 양수 여부
                        is_momentum_pos,  # 모멘텀 양수 여부
                        pos_score_trend > 0.02,  # 양수 총점 * 트렌드 (임계값 더 낮춤)
                        momentum_vol_ratio > 0.2,  # 모멘텀/변동성 비율 (임계값 더 낮춤)
                        trend_abs_momentum > 0.02,  # 트렌드 * 모멘텀 절댓값 (임계값 더 낮춤)
                        strong_up,  # 강한 상승 신호
                        # 추가 조건: 약한 상승 신호도 포함
                        score > 0,  # 총점이 양수
                        trend > 0,  # 트렌드가 양수
                    ]
                    # 조건 만족 개수에 따른 확률 (가중치 적용)
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 6.0, 1.0)  # 분모를 7.0에서 6.0으로 낮춤
                
                elif regime_type == 'trending_down':
                    # TRENDING_DOWN 조건들 (현실적인 조건들)
                    conditions = [
                        score < trending_down_threshold,  # 총점이 임계값보다 낮음
                        trend < -0.15,  # 트렌드 점수가 음수 (임계값 더 낮춤)
                        momentum < -0.1,  # 모멘텀 점수가 음수 (임계값 더 낮춤)
                        trend_momentum < -0.02,  # 트렌드-모멘텀 상호작용이 음수 (임계값 더 낮춤)
                        volatility < 0.7,  # 변동성이 낮음 (임계값 더 높임)
                        not is_trend_pos,  # 트렌드 음수 여부
                        not is_momentum_pos,  # 모멘텀 음수 여부
                        neg_score_trend < -0.02,  # 음수 총점 * 트렌드 (임계값 더 낮춤)
                        momentum_vol_ratio < -0.2,  # 모멘텀/변동성 비율 (임계값 더 낮춤)
                        trend_abs_momentum < -0.02,  # 트렌드 * 모멘텀 절댓값 (임계값 더 낮춤)
                        strong_down,  # 강한 하락 신호
                        # 추가 조건: 약한 하락 신호도 포함
                        score < 0,  # 총점이 음수
                        trend < 0,  # 트렌드가 음수
                    ]
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 6.0, 1.0)  # 분모를 7.0에서 6.0으로 낮춤
                
                elif regime_type == 'volatile':
                    # VOLATILE 조건들 (더 현실적인 조건들)
                    conditions = [
                        volatility > volatile_threshold,  # 변동성 점수가 높음
                        abs_momentum > 0.3,  # 모멘텀 절댓값이 높음 (임계값 낮춤)
                        vol_momentum_abs > 0.1,  # 변동성-모멘텀 상호작용이 높음 (임계값 낮춤)
                        abs_score > 0.2,  # 총점 절댓값이 높음 (임계값 낮춤)
                        macro > 0.1,  # 매크로 점수가 높음 (임계값 낮춤)
                        is_high_vol,  # 고변동성 여부
                        very_high_vol,  # 매우 높은 변동성
                        very_high_momentum,  # 매우 높은 모멘텀
                        abs_score_vol > 0.1,  # 총점 절댓값 * 변동성 (임계값 낮춤)
                        macro_vol > 0.05,  # 매크로 * 변동성 (임계값 낮춤)
                        volume_abs_momentum > 0.05,  # 거래량 * 모멘텀 절댓값 (임계값 낮춤)
                    ]
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 6.0, 1.0)  # 분모를 8.0에서 6.0으로 낮춤
                
                else:  # sideways
                    # SIDEWAYS 조건들 (더 엄격한 조건들)
                    not_trending_up = not (score > trending_up_threshold and trend > 0.2)
                    not_trending_down = not (score < trending_down_threshold and trend < -0.2)
                    not_volatile = not (volatility > volatile_threshold and abs_momentum > 0.3)
                    
                    conditions = [
                        not_trending_up,
                        not_trending_down,
                        not_volatile,
                        abs_score < 0.2,  # 총점 절댓값이 낮음 (더 엄격하게)
                        volatility < volatile_threshold * 0.8,  # 변동성이 더 낮음
                        abs_momentum < 0.2,  # 모멘텀 절댓값이 낮음 (더 엄격하게)
                        abs_trend < 0.2,  # 트렌드 절댓값이 낮음 (더 엄격하게)
                        not very_high_vol,  # 매우 높은 변동성이 아님
                        not very_high_momentum,  # 매우 높은 모멘텀이 아님
                        not strong_up,  # 강한 상승 신호가 아님
                        not strong_down,  # 강한 하락 신호가 아님
                        abs_score_vol < 0.05,  # 총점 절댓값 * 변동성이 낮음 (더 엄격하게)
                        # 추가 조건: 모든 점수가 중간 범위에 있어야 함
                        abs_score > 0.05,  # 너무 낮은 점수도 제외
                        volatility > 0.01,  # 너무 낮은 변동성도 제외
                    ]
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 10.0, 1.0)  # 분모를 8.0에서 10.0으로 높임
            
            # 각 상태별 확률 계산
            probabilities = np.zeros((len(scores), 4))
            
            for i in range(len(scores)):
                features_row = features[i]
                probabilities[i, 0] = tree_based_probability(features_row, 'trending_up')
                probabilities[i, 1] = tree_based_probability(features_row, 'trending_down')
                probabilities[i, 2] = tree_based_probability(features_row, 'volatile')
                probabilities[i, 3] = tree_based_probability(features_row, 'sideways')
                
                # 정규화 (합이 1이 되도록)
                total_prob = np.sum(probabilities[i])
                if total_prob > 0:
                    probabilities[i] = probabilities[i] / total_prob
            
            return probabilities
        
        # 확률 계산
        regime_probs = calculate_regime_probabilities_random_forest(total_score)
        
        # 최종 분류 (더 민감한 임계값)
        regime = pd.Series(index=data.index, dtype='object')
        regime = np.where(regime_probs[:, 0] > 0.2, MarketRegime.TRENDING_UP.value,
                 np.where(regime_probs[:, 1] > 0.2, MarketRegime.TRENDING_DOWN.value,
                 np.where(regime_probs[:, 2] > 0.2, MarketRegime.VOLATILE.value,
                 MarketRegime.SIDEWAYS.value)))
        
        # 현재 시장 상태 (마지막 데이터 포인트)
        current_regime = regime[-1] if len(regime) > 0 else MarketRegime.UNCERTAIN.value
        
        # 신뢰도 계산 (확률의 최대값 기반)
        current_probs = regime_probs[-1] if len(regime_probs) > 0 else [0.25, 0.25, 0.25, 0.25]
        confidence = max(current_probs) if len(current_probs) > 0 else 0.5
        
        # 결과 반환
        return {
            'current_regime': current_regime,
            'confidence': confidence,
            'probabilities': {
                'TRENDING_UP': current_probs[0] if len(current_probs) > 0 else 0.25,
                'TRENDING_DOWN': current_probs[1] if len(current_probs) > 1 else 0.25,
                'VOLATILE': current_probs[2] if len(current_probs) > 2 else 0.25,
                'SIDEWAYS': current_probs[3] if len(current_probs) > 3 else 0.25
            },
            'regime_series': pd.Series(regime, index=data.index),
            'probabilities_series': {
                'trending_up': regime_probs[:, 0],
                'trending_down': regime_probs[:, 1],
                'volatile': regime_probs[:, 2],
                'sideways': regime_probs[:, 3]
            },
            'scores': regime_scores,
            'total_score': total_score
        }
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, regime: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """전략별 수익률 계산"""
        returns = pd.Series(index=data.index, dtype=float)
        position = pd.Series(index=data.index, dtype=float)
        
        # 컬럼명 매핑 (대소문자 처리)
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 첫 번째 포지션 설정
        first_regime = regime.iloc[0]
        if first_regime == MarketRegime.TRENDING_UP.value:
            position.iloc[0] = params.get('trending_boost', 1.0)
        elif first_regime == MarketRegime.TRENDING_DOWN.value:
            position.iloc[0] = -params.get('base_position', 0.5)
        elif first_regime == MarketRegime.SIDEWAYS.value:
            # RSI 기반 첫 번째 포지션
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[0]
                if rsi < params.get('rsi_oversold', 30):
                    position.iloc[0] = params.get('base_position', 0.8)
                elif rsi > params.get('rsi_overbought', 70):
                    position.iloc[0] = -params.get('base_position', 0.8)
                else:
                    position.iloc[0] = params.get('base_position', 0.8) * 0.3
            else:
                position.iloc[0] = params.get('base_position', 0.8) * 0.5
        elif first_regime == MarketRegime.VOLATILE.value:
            position.iloc[0] = params.get('volatile_reduction', 0.5) * params.get('base_position', 0.8)
        else:  # UNCERTAIN
            position.iloc[0] = 0
        
        for i in range(1, len(data)):
            current_regime = regime.iloc[i]
            prev_close = data[close_col].iloc[i-1]
            current_close = data[close_col].iloc[i]
            
            # 시장 상태별 전략
            if current_regime == MarketRegime.TRENDING_UP.value:
                # Buy & Hold 우선, 스윙 전략 보조
                position.iloc[i] = params.get('trending_boost', 1.0)
                
            elif current_regime == MarketRegime.TRENDING_DOWN.value:
                # 현금 보유 또는 역방향 전략
                position.iloc[i] = -params.get('base_position', 0.5)
                
            elif current_regime == MarketRegime.SIDEWAYS.value:
                # 스윙 전략 적극 활용
                if 'rsi' in data.columns:
                    rsi = data['rsi'].iloc[i]
                    if rsi < params.get('rsi_oversold', 30):
                        position.iloc[i] = params.get('base_position', 0.8)
                    elif rsi > params.get('rsi_overbought', 70):
                        position.iloc[i] = -params.get('base_position', 0.8)
                    else:
                        # RSI 중간 구간에서는 작은 포지션으로 스윙
                        position.iloc[i] = params.get('base_position', 0.8) * 0.3
                else:
                    # RSI가 없으면 기본 포지션
                    position.iloc[i] = params.get('base_position', 0.8) * 0.5
                    
            elif current_regime == MarketRegime.VOLATILE.value:
                # 포지션 크기 축소 + 단기 전략
                position.iloc[i] = params.get('volatile_reduction', 0.5) * params.get('base_position', 0.8)
            
            else:  # UNCERTAIN
                position.iloc[i] = 0
            
            # 수익률 계산
            price_return = (current_close - prev_close) / prev_close
            returns.iloc[i] = position.iloc[i] * price_return
        
        return returns
    
    def _calculate_performance_metrics(self, strategy_returns: pd.Series, buy_hold_returns: pd.Series) -> Dict[str, float]:
        """성과 지표 계산 (고도화된 버전)"""
        # 총 수익률 (단순 수익률)
        total_return = (1 + strategy_returns).prod() - 1
        buy_hold_total = (1 + buy_hold_returns).prod() - 1
        
        # 로그 수익률
        log_return = np.log((1 + strategy_returns).prod())
        buy_hold_log_return = np.log((1 + buy_hold_returns).prod())
        
        # 샤프 비율
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # 최대 낙폭
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 승률 (고도화된 계산)
        win_rate = (strategy_returns > 0).mean()
        
        # 승률 향상을 위한 추가 지표들
        # 1. 연속 승/패 분석
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for ret in strategy_returns:
            if ret > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        # 2. 평균 승/패 크기
        winning_returns = strategy_returns[strategy_returns > 0]
        losing_returns = strategy_returns[strategy_returns <= 0]
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        # 3. 수익 팩터 (Profit Factor)
        total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
        total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 4. 칼마 비율 (Calmar Ratio)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 5. 소르티노 비율 (Sortino Ratio) - 하방 변동성만 고려
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = strategy_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # 6. 승률 가중 수익률 (Win Rate Weighted Return)
        win_rate_weighted_return = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # 7. 거래 빈도 분석
        position_changes = strategy_returns.abs() > 0
        trade_frequency = position_changes.sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # 초과 수익률 (Buy & Hold 대비)
        excess_return = total_return - buy_hold_total
        
        return {
            'total_return': total_return,
            'log_return': log_return,
            'buy_hold_return': buy_hold_total,
            'buy_hold_log_return': buy_hold_log_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'win_rate_weighted_return': win_rate_weighted_return,
            'trade_frequency': trade_frequency,
            'total_trades': position_changes.sum()
        }
    
    def objective(self, trial: optuna.Trial, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame]) -> float:
        """Optuna 목적 함수"""
        # 하이퍼파라미터 샘플링
        params = {}
        
        # 지표 파라미터 샘플링
        indicators = self.config.get('market_regime_classification', {}).get('indicators', {})
        
        for category, indicators_dict in indicators.items():
            for indicator, config in indicators_dict.items():
                if config['type'] == 'int':
                    params[indicator] = trial.suggest_int(indicator, config['min'], config['max'])
                elif config['type'] == 'float':
                    params[indicator] = trial.suggest_float(indicator, config['min'], config['max'])
        
        # 분류 가중치 샘플링
        weights = self.config.get('market_regime_classification', {}).get('classification_weights', {})
        for weight, config in weights.items():
            params[weight] = trial.suggest_float(weight, config['min'], config['max'])
        
        # 전략 파라미터 샘플링
        strategy = self.config.get('trading_strategy', {})
        for category, strategy_dict in strategy.items():
            for param, config in strategy_dict.items():
                if config['type'] == 'int':
                    params[param] = trial.suggest_int(param, config['min'], config['max'])
                elif config['type'] == 'float':
                    params[param] = trial.suggest_float(param, config['min'], config['max'])
        
        try:
            # 파생 변수 계산
            data_with_features = self._calculate_derived_features(spy_data, params)
            
            # 매크로 데이터 병합 (컬럼명 대소문자 처리)
            if '^VIX' in macro_data:
                vix_df = macro_data['^VIX']
                # 컬럼명 확인 및 처리
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    self.logger.warning("VIX 데이터에서 close 컬럼을 찾을 수 없습니다.")
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 시장 상태 분류
            regime = self._classify_market_regime(data_with_features, params)
            
            # 전략 수익률 계산
            strategy_returns = self._calculate_strategy_returns(data_with_features, regime, params)
            
            # Buy & Hold 수익률 계산
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            buy_hold_returns = spy_data[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            # 목적 함수 선택 (승률 중심으로 개선)
            objective_metric = self.config.get('optimization', {}).get('objective', 'win_rate_optimized')
            
            # 목적 함수별 반환값 설정
            if objective_metric == 'win_rate_optimized':
                # 승률 최적화 (승률 + 수익률 + 안정성의 조합)
                win_rate = metrics.get('win_rate', 0)
                total_return = metrics.get('total_return', 0)
                profit_factor = metrics.get('profit_factor', 0)
                max_drawdown = abs(metrics.get('max_drawdown', 0))
                
                # 승률에 높은 가중치를 주되, 수익률과 안정성도 고려
                win_rate_score = win_rate * 100  # 승률을 0-100 스케일로
                return_score = total_return * 50  # 수익률에 적당한 가중치
                profit_factor_score = min(profit_factor * 10, 50)  # 수익 팩터 제한
                drawdown_penalty = max_drawdown * 100  # 낙폭 페널티
                
                # 최종 점수 = 승률 중심 + 수익률 보너스 + 안정성 보너스 - 위험 페널티
                final_score = win_rate_score + return_score + profit_factor_score - drawdown_penalty
                
                # 최소 승률 조건 (50% 미만이면 페널티)
                if win_rate < 0.5:
                    final_score *= 0.5
                
                return final_score
                
            elif objective_metric == 'total_return':
                return metrics.get('total_return', 0)
            elif objective_metric == 'log_return':
                return metrics.get('log_return', 0)
            elif objective_metric == 'excess_return':
                return metrics.get('excess_return', 0)
            elif objective_metric == 'sharpe_ratio':
                return metrics.get('sharpe_ratio', 0)
            elif objective_metric == 'win_rate':
                return metrics.get('win_rate', 0)
            elif objective_metric == 'profit_factor':
                return metrics.get('profit_factor', 0)
            else:
                # 기본값: 승률 최적화
                return metrics.get('win_rate', 0) * 100
            
        except Exception as e:
            self.logger.error(f"목적 함수 실행 중 오류: {e}")
            return -999  # 매우 낮은 값 반환
    
    def optimize_hyperparameters_with_data(self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame], n_trials: int = None) -> Dict[str, Any]:
        """하이퍼파라미터 최적화 (이미 로드된 데이터 사용)"""
        if n_trials is None:
            n_trials = self.config.get('optimization', {}).get('n_trials', 100)
        
        if spy_data.empty:
            raise ValueError("SPY 데이터가 비어있습니다")
        
        # Train/Test 분할
        train_test_split = self.config.get('optimization', {}).get('train_test_split', 0.8)
        split_idx = int(len(spy_data) * train_test_split)
        
        train_spy = spy_data.iloc[:split_idx]
        test_spy = spy_data.iloc[split_idx:]
        
        train_macro = {k: v.iloc[:split_idx] if not v.empty else v for k, v in macro_data.items()}
        test_macro = {k: v.iloc[split_idx:] if not v.empty else v for k, v in macro_data.items()}
        
        self.logger.info(f"Train 데이터: {len(train_spy)}개, Test 데이터: {len(test_spy)}개")
        
        # Optuna 스터디 생성 (시드 설정으로 일관성 향상)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        
        # 목적 함수 래퍼
        def objective_wrapper(trial):
            return self.objective(trial, train_spy, train_macro)
        
        # 최적화 실행
        self.logger.info(f"하이퍼파라미터 최적화 시작 (n_trials={n_trials})...")
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # 최적 파라미터
        best_params = study.best_params
        best_value = study.best_value
        
        # Test 데이터에서 성과 평가
        test_performance = self._evaluate_on_test_data(test_spy, test_macro, best_params)
        
        # 결과 저장
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'test_performance': test_performance,
            'n_trials': n_trials,
            'optimization_history': study.trials_dataframe().to_dict('records')
        }
        
        return results
    
    def optimize_hyperparameters(self, start_date: str = None, end_date: str = None, n_trials: int = None,
                               spy_data: pd.DataFrame = None, macro_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """하이퍼파라미터 최적화 (데이터 수집 포함)"""
        if n_trials is None:
            n_trials = self.config.get('optimization', {}).get('n_trials', 100)
        
        # 날짜 설정 (설정 파일 기반)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            # 설정에서 모델 훈련 기간 가져오기
            days_back = self.collector._get_days_back("model_training")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # 데이터 수집 (이미 제공된 데이터가 있으면 재사용)
        if spy_data is None or macro_data is None:
            self.logger.info(f"데이터 수집 중... (기간: {start_date} ~ {end_date})")
            spy_data = self.collector.collect_spy_data(start_date, end_date)
            macro_data = self.collector.collect_macro_indicators(start_date, end_date)
        else:
            self.logger.info("제공된 데이터 재사용 중...")
        
        if spy_data.empty:
            raise ValueError("SPY 데이터 수집 실패")
        
        # Train/Test 분할
        train_test_split = self.config.get('optimization', {}).get('train_test_split', 0.8)
        split_idx = int(len(spy_data) * train_test_split)
        
        train_spy = spy_data.iloc[:split_idx]
        test_spy = spy_data.iloc[split_idx:]
        
        train_macro = {k: v.iloc[:split_idx] if not v.empty else v for k, v in macro_data.items()}
        test_macro = {k: v.iloc[split_idx:] if not v.empty else v for k, v in macro_data.items()}
        
        self.logger.info(f"Train 데이터: {len(train_spy)}개, Test 데이터: {len(test_spy)}개")
        
        # Optuna 스터디 생성 (시드 설정으로 일관성 향상)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        
        # 목적 함수 래퍼
        def objective_wrapper(trial):
            return self.objective(trial, train_spy, train_macro)
        
        # 최적화 실행
        self.logger.info(f"하이퍼파라미터 최적화 시작 (n_trials={n_trials})...")
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # 최적 파라미터
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"최적화 완료 - Best {self.config.get('optimization', {}).get('objective', 'sharpe_ratio')}: {best_value:.4f}")
        
        # Test 데이터에서 성능 평가
        test_performance = self._evaluate_on_test_data(test_spy, test_macro, best_params)
        
        # 결과 구성
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'train_performance': study.best_value,
            'test_performance': test_performance,
            'study': study
        }
        
        # 결과 저장
        try:
            self.save_results(results)
        except Exception as e:
            self.logger.warning(f"최적화 결과 저장 실패: {e}")
        
        return results
    
    def _evaluate_on_test_data(self, test_spy: pd.DataFrame, test_macro: Dict[str, pd.DataFrame], 
                              best_params: Dict[str, Any]) -> Dict[str, float]:
        """Test 데이터에서 성능 평가"""
        try:
            # 파생 변수 계산
            data_with_features = self._calculate_derived_features(test_spy, best_params)
            
            # 매크로 데이터 병합 (컬럼명 대소문자 처리)
            if '^VIX' in test_macro:
                vix_df = test_macro['^VIX']
                # 컬럼명 확인 및 처리
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    self.logger.warning("VIX 데이터에서 close 컬럼을 찾을 수 없습니다.")
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 시장 상태 분류
            regime = self._classify_market_regime(data_with_features, best_params)
            
            # 전략 수익률 계산
            strategy_returns = self._calculate_strategy_returns(data_with_features, regime, best_params)
            
            # Buy & Hold 수익률 계산
            close_col = 'close' if 'close' in test_spy.columns else 'Close'
            buy_hold_returns = test_spy[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Test 데이터 평가 중 오류: {e}")
            return {}
    
class MarketRegimeValidator:
    """시장 상태 분류 검증 및 통계분석 클래스"""
    
    def __init__(self, session_uuid: str = None):
        self.session_uuid = session_uuid
        self.logger = logging.getLogger(__name__)
        
    def validate_classification_accuracy(self, actual_regimes: pd.Series, predicted_regimes: pd.Series) -> Dict[str, Any]:
        """시장 상태 분류 정확도 검증"""
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
            
            # 기본 정확도 계산
            accuracy = accuracy_score(actual_regimes, predicted_regimes)
            
            # 정밀도, 재현율, F1 점수 계산
            precision, recall, f1, support = precision_recall_fscore_support(
                actual_regimes, predicted_regimes, average='weighted'
            )
            
            # 혼동 행렬 계산
            cm = confusion_matrix(actual_regimes, predicted_regimes)
            
            # 분류 보고서 생성
            class_report = classification_report(actual_regimes, predicted_regimes, output_dict=True)
            
            # 시장 상태별 정확도 계산
            regime_accuracy = {}
            unique_regimes = actual_regimes.unique()
            for regime in unique_regimes:
                mask = actual_regimes == regime
                if mask.sum() > 0:
                    regime_accuracy[regime] = (actual_regimes[mask] == predicted_regimes[mask]).mean()
            
            # 연속성 분석 (시장 상태 변화의 일관성)
            actual_changes = actual_regimes.diff().fillna(0)
            predicted_changes = predicted_regimes.diff().fillna(0)
            change_accuracy = (actual_changes == predicted_changes).mean()
            
            # 지연 분석 (예측이 얼마나 빨리 변화를 감지하는지)
            lag_analysis = self._analyze_prediction_lag(actual_regimes, predicted_regimes)
            
            return {
                'overall_accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'regime_accuracy': regime_accuracy,
                'change_accuracy': change_accuracy,
                'lag_analysis': lag_analysis,
                'regime_distribution': {
                    'actual': actual_regimes.value_counts().to_dict(),
                    'predicted': predicted_regimes.value_counts().to_dict()
                }
            }
            
        except Exception as e:
            self.logger.error(f"분류 정확도 검증 중 오류: {e}")
            return {'error': str(e)}
    
    def _analyze_prediction_lag(self, actual: pd.Series, predicted: pd.Series, max_lag: int = 5) -> Dict[str, Any]:
        """예측 지연 분석"""
        lag_analysis = {}
        
        for lag in range(1, max_lag + 1):
            # 실제 변화를 lag일 후에 예측이 감지하는지 확인
            actual_changes = actual.diff().fillna(0)
            predicted_changes = predicted.diff().fillna(0)
            
            # 지연된 예측과 실제 변화의 상관관계
            lagged_prediction = predicted_changes.shift(-lag)
            correlation = actual_changes.corr(lagged_prediction)
            
            # 지연된 예측의 정확도
            accuracy = (actual_changes == lagged_prediction).mean()
            
            lag_analysis[f'lag_{lag}'] = {
                'correlation': correlation,
                'accuracy': accuracy
            }
        
        return lag_analysis
    
    def analyze_strategy_performance(self, strategy_returns: pd.Series, benchmark_returns: pd.Series, 
                                   regime_series: pd.Series) -> Dict[str, Any]:
        """전략 성과 분석 (시장 상태별)"""
        try:
            import numpy as np
            from scipy import stats
            
            # 전체 성과 지표
            total_return = (1 + strategy_returns).prod() - 1
            benchmark_return = (1 + benchmark_returns).prod() - 1
            excess_return = total_return - benchmark_return
            
            # 변동성 계산
            strategy_vol = strategy_returns.std() * np.sqrt(252)
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            
            # 샤프 비율 계산
            risk_free_rate = 0.02  # 연 2% 가정
            strategy_sharpe = (strategy_returns.mean() * 252 - risk_free_rate) / strategy_vol
            benchmark_sharpe = (benchmark_returns.mean() * 252 - risk_free_rate) / benchmark_vol
            
            # 최대 낙폭 계산
            strategy_cumulative = (1 + strategy_returns).cumprod()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            
            strategy_drawdown = (strategy_cumulative / strategy_cumulative.expanding().max() - 1).min()
            benchmark_drawdown = (benchmark_cumulative / benchmark_cumulative.expanding().max() - 1).min()
            
            # 승률 계산
            strategy_win_rate = (strategy_returns > 0).mean()
            benchmark_win_rate = (benchmark_returns > 0).mean()
            
            # 시장 상태별 성과 분석
            regime_performance = {}
            unique_regimes = regime_series.unique()
            
            for regime in unique_regimes:
                mask = regime_series == regime
                if mask.sum() > 0:
                    regime_strategy_returns = strategy_returns[mask]
                    regime_benchmark_returns = benchmark_returns[mask]
                    
                    regime_total_return = (1 + regime_strategy_returns).prod() - 1
                    regime_benchmark_return = (1 + regime_benchmark_returns).prod() - 1
                    regime_excess_return = regime_total_return - regime_benchmark_return
                    
                    regime_vol = regime_strategy_returns.std() * np.sqrt(252)
                    regime_sharpe = (regime_strategy_returns.mean() * 252 - risk_free_rate) / regime_vol
                    regime_win_rate = (regime_strategy_returns > 0).mean()
                    
                    # 정보 비율 계산
                    regime_information_ratio = regime_excess_return / (regime_strategy_returns - regime_benchmark_returns).std()
                    
                    regime_performance[regime] = {
                        'total_return': regime_total_return,
                        'benchmark_return': regime_benchmark_return,
                        'excess_return': regime_excess_return,
                        'volatility': regime_vol,
                        'sharpe_ratio': regime_sharpe,
                        'win_rate': regime_win_rate,
                        'information_ratio': regime_information_ratio,
                        'days_count': mask.sum(),
                        'avg_daily_return': regime_strategy_returns.mean(),
                        'max_daily_return': regime_strategy_returns.max(),
                        'min_daily_return': regime_strategy_returns.min()
                    }
            
            # 통계적 유의성 검정
            t_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns)
            
            # VaR (Value at Risk) 계산
            strategy_var_95 = np.percentile(strategy_returns, 5)
            benchmark_var_95 = np.percentile(benchmark_returns, 5)
            
            # CVaR (Conditional Value at Risk) 계산
            strategy_cvar_95 = strategy_returns[strategy_returns <= strategy_var_95].mean()
            benchmark_cvar_95 = benchmark_returns[benchmark_returns <= benchmark_var_95].mean()
            
            return {
                'overall_performance': {
                    'total_return': total_return,
                    'benchmark_return': benchmark_return,
                    'excess_return': excess_return,
                    'volatility': strategy_vol,
                    'benchmark_volatility': benchmark_vol,
                    'sharpe_ratio': strategy_sharpe,
                    'benchmark_sharpe': benchmark_sharpe,
                    'max_drawdown': strategy_drawdown,
                    'benchmark_drawdown': benchmark_drawdown,
                    'win_rate': strategy_win_rate,
                    'benchmark_win_rate': benchmark_win_rate,
                    'information_ratio': excess_return / (strategy_returns - benchmark_returns).std(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'var_95': strategy_var_95,
                    'benchmark_var_95': benchmark_var_95,
                    'cvar_95': strategy_cvar_95,
                    'benchmark_cvar_95': benchmark_cvar_95
                },
                'regime_performance': regime_performance,
                'regime_effectiveness': self._calculate_regime_effectiveness(regime_performance)
            }
            
        except Exception as e:
            self.logger.error(f"전략 성과 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_effectiveness(self, regime_performance: Dict[str, Any]) -> Dict[str, Any]:
        """시장 상태별 전략 효과성 분석"""
        effectiveness = {}
        
        # 각 시장 상태에서의 상대적 성과 계산
        for regime, performance in regime_performance.items():
            excess_return = performance['excess_return']
            sharpe_ratio = performance['sharpe_ratio']
            information_ratio = performance['information_ratio']
            
            # 효과성 점수 (가중 평균)
            effectiveness_score = (
                excess_return * 0.4 + 
                sharpe_ratio * 0.3 + 
                information_ratio * 0.3
            )
            
            effectiveness[regime] = {
                'effectiveness_score': effectiveness_score,
                'excess_return_contribution': excess_return * 0.4,
                'sharpe_contribution': sharpe_ratio * 0.3,
                'information_ratio_contribution': information_ratio * 0.3,
                'performance_rank': None  # 나중에 계산
            }
        
        # 성과 순위 계산
        scores = [(regime, data['effectiveness_score']) for regime, data in effectiveness.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (regime, _) in enumerate(scores, 1):
            effectiveness[regime]['performance_rank'] = rank
        
        return effectiveness
    
    def generate_validation_report(self, validation_results: Dict[str, Any], 
                                 performance_results: Dict[str, Any]) -> str:
        """검증 결과 종합 보고서 생성"""
        try:
            report = []
            report.append("=" * 80)
            report.append("📊 시장 상태 분류 및 전략 성과 검증 보고서")
            report.append("=" * 80)
            report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"세션 UUID: {self.session_uuid}")
            report.append("")
            
            # 1. 분류 정확도 분석
            if 'overall_accuracy' in validation_results:
                report.append("🎯 시장 상태 분류 정확도 분석")
                report.append("-" * 50)
                report.append(f"전체 정확도: {validation_results['overall_accuracy']:.3f} ({validation_results['overall_accuracy']*100:.1f}%)")
                report.append(f"정밀도: {validation_results['precision']:.3f}")
                report.append(f"재현율: {validation_results['recall']:.3f}")
                report.append(f"F1 점수: {validation_results['f1_score']:.3f}")
                report.append(f"상태 변화 정확도: {validation_results['change_accuracy']:.3f}")
                report.append("")
                
                # 시장 상태별 정확도
                if 'regime_accuracy' in validation_results:
                    report.append("📈 시장 상태별 분류 정확도:")
                    for regime, accuracy in validation_results['regime_accuracy'].items():
                        report.append(f"  {regime}: {accuracy:.3f} ({accuracy*100:.1f}%)")
                    report.append("")
                
                # 지연 분석
                if 'lag_analysis' in validation_results:
                    report.append("⏱️ 예측 지연 분석:")
                    for lag, data in validation_results['lag_analysis'].items():
                        report.append(f"  {lag}일 지연: 상관관계={data['correlation']:.3f}, 정확도={data['accuracy']:.3f}")
                    report.append("")
            
            # 2. 전략 성과 분석
            if 'overall_performance' in performance_results:
                perf = performance_results['overall_performance']
                report.append("💰 전략 성과 분석")
                report.append("-" * 50)
                report.append(f"총 수익률: {perf['total_return']:.3f} ({perf['total_return']*100:.1f}%)")
                report.append(f"벤치마크 수익률: {perf['benchmark_return']:.3f} ({perf['benchmark_return']*100:.1f}%)")
                report.append(f"초과 수익률: {perf['excess_return']:.3f} ({perf['excess_return']*100:.1f}%)")
                report.append(f"변동성: {perf['volatility']:.3f} ({perf['volatility']*100:.1f}%)")
                report.append(f"샤프 비율: {perf['sharpe_ratio']:.3f}")
                report.append(f"최대 낙폭: {perf['max_drawdown']:.3f} ({perf['max_drawdown']*100:.1f}%)")
                report.append(f"승률: {perf['win_rate']:.3f} ({perf['win_rate']*100:.1f}%)")
                report.append(f"정보 비율: {perf['information_ratio']:.3f}")
                report.append(f"VaR (95%): {perf['var_95']:.3f} ({perf['var_95']*100:.1f}%)")
                report.append(f"CVaR (95%): {perf['cvar_95']:.3f} ({perf['cvar_95']*100:.1f}%)")
                report.append(f"통계적 유의성 (p-value): {perf['p_value']:.4f}")
                report.append("")
                
                # 시장 상태별 성과
                if 'regime_performance' in performance_results:
                    report.append("📊 시장 상태별 성과:")
                    for regime, regime_perf in performance_results['regime_performance'].items():
                        report.append(f"  {regime}:")
                        report.append(f"    수익률: {regime_perf['total_return']:.3f} ({regime_perf['total_return']*100:.1f}%)")
                        report.append(f"    초과 수익률: {regime_perf['excess_return']:.3f} ({regime_perf['excess_return']*100:.1f}%)")
                        report.append(f"    샤프 비율: {regime_perf['sharpe_ratio']:.3f}")
                        report.append(f"    승률: {regime_perf['win_rate']:.3f} ({regime_perf['win_rate']*100:.1f}%)")
                        report.append(f"    정보 비율: {regime_perf['information_ratio']:.3f}")
                        report.append(f"    거래일수: {regime_perf['days_count']}일")
                    report.append("")
                
                # 전략 효과성
                if 'regime_effectiveness' in performance_results:
                    report.append("🏆 전략 효과성 순위:")
                    effectiveness = performance_results['regime_effectiveness']
                    sorted_effectiveness = sorted(effectiveness.items(), 
                                                key=lambda x: x[1]['performance_rank'])
                    for regime, data in sorted_effectiveness:
                        report.append(f"  {data['performance_rank']}. {regime}: {data['effectiveness_score']:.3f}")
                    report.append("")
            
            # 3. 결론 및 권장사항
            report.append("📋 결론 및 권장사항")
            report.append("-" * 50)
            
            if 'overall_accuracy' in validation_results and validation_results['overall_accuracy'] > 0.7:
                report.append("✅ 분류 정확도가 양호합니다 (70% 이상)")
            else:
                report.append("⚠️ 분류 정확도 개선이 필요합니다")
            
            if 'overall_performance' in performance_results:
                perf = performance_results['overall_performance']
                if perf['excess_return'] > 0.05:
                    report.append("✅ 전략이 벤치마크를 상당히 상회합니다")
                elif perf['excess_return'] > 0:
                    report.append("✅ 전략이 벤치마크를 상회합니다")
                else:
                    report.append("⚠️ 전략 성과 개선이 필요합니다")
                
                if perf['p_value'] < 0.05:
                    report.append("✅ 통계적으로 유의한 성과입니다")
                else:
                    report.append("⚠️ 통계적 유의성이 부족합니다")
            
            report.append("")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"검증 보고서 생성 중 오류: {e}")
            return f"보고서 생성 중 오류 발생: {str(e)}"
    
    def save_validation_results(self, validation_results: Dict[str, Any], 
                              performance_results: Dict[str, Any],
                              output_dir: str = "results/validation") -> str:
        """검증 결과 저장"""
        try:
            import os
            import json
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 타임스탬프 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON 파일로 결과 저장
            results_file = os.path.join(output_dir, f"validation_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'validation_results': validation_results,
                    'performance_results': performance_results,
                    'timestamp': datetime.now().isoformat(),
                    'session_uuid': self.session_uuid
                }, f, indent=2, ensure_ascii=False)
            
            # 보고서 파일 생성
            report_file = os.path.join(output_dir, f"validation_report_{timestamp}.txt")
            report_content = self.generate_validation_report(validation_results, performance_results)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return results_file
            
        except Exception as e:
            self.logger.error(f"검증 결과 저장 중 오류: {e}")
            return ""



    def _classify_market_regime(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """시장 상태 분류 (고급 버전 - SPY 기준 추세+수익률 강화)"""
        regime_scores = pd.DataFrame(index=data.index)
        
        # 컬럼명 매핑 (대소문자 처리)
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 1. SPY 기준 추세+수익률 점수 계산 (강화된 핵심 지표)
        if 'spy_trend_weight' in params:
            spy_trend_score = 0
            if f'sma_{params.get("sma_short", 20)}' in data.columns and f'sma_{params.get("sma_long", 50)}' in data.columns:
                sma_short = data[f'sma_{params.get("sma_short", 20)}']
                sma_long = data[f'sma_{params.get("sma_long", 50)}']
                # NaN 값 처리
                valid_mask = ~(sma_short.isna() | sma_long.isna())
                
                # 1-1. 기본 이동평균 비교 (가중치: 0.4)
                sma_trend = np.where(valid_mask, 
                    np.where(sma_short > sma_long, 1, -1), 0)
                
                # 1-2. ADX 기반 트렌드 강도 (가중치: 0.3)
                adx_strength = 0
                if 'adx' in data.columns:
                    adx = data['adx']
                    adx_threshold = params.get('adx_threshold', 25)
                    adx_strength = np.where(valid_mask & (adx > adx_threshold), 0.5, 0)
                
                # 1-3. SuperTrend 기반 트렌드 확인 (가중치: 0.2)
                supertrend_signal = 0
                if 'supertrend' in data.columns:
                    supertrend = data['supertrend']
                    supertrend_signal = np.where(valid_mask & (data[close_col] > supertrend), 0.3, -0.3)
                
                # 1-4. 수익률 기반 모멘텀 (가중치: 0.1) - 새로 추가
                returns_momentum = 0
                if close_col in data.columns:
                    # 5일, 10일, 20일 수익률 계산
                    returns_5d = data[close_col].pct_change(5)
                    returns_10d = data[close_col].pct_change(10)
                    returns_20d = data[close_col].pct_change(20)
                    
                    # 수익률 기반 점수 (단기 > 중기 > 장기 순으로 가중치)
                    returns_score = np.where(valid_mask,
                        np.where(returns_5d > 0.02, 0.4,  # 5일 수익률 > 2%
                        np.where(returns_5d < -0.02, -0.4, 0)) +  # 5일 수익률 < -2%
                        np.where(returns_10d > 0.03, 0.3,  # 10일 수익률 > 3%
                        np.where(returns_10d < -0.03, -0.3, 0)) +  # 10일 수익률 < -3%
                        np.where(returns_20d > 0.05, 0.2,  # 20일 수익률 > 5%
                        np.where(returns_20d < -0.05, -0.2, 0)), 0)  # 20일 수익률 < -5%
                    
                    returns_momentum = returns_score
                
                # 1-5. 가격 위치 기반 점수 (가중치: 0.1) - 새로 추가
                price_position_score = 0
                if 'bb_upper' in data.columns and 'bb_lower' in data.columns and 'bb_middle' in data.columns:
                    bb_upper = data['bb_upper']
                    bb_lower = data['bb_lower']
                    bb_middle = data['bb_middle']
                    
                    # 가격이 Bollinger Band 내에서의 위치
                    bb_position = (data[close_col] - bb_lower) / (bb_upper - bb_lower)
                    price_position_score = np.where(valid_mask & (bb_position > 0.8), 0.2,  # 상단 20%
                                          np.where(valid_mask & (bb_position < 0.2), -0.2, 0))  # 하단 20%
                
                # SPY 추세+수익률 종합 점수
                spy_trend_score = (sma_trend * 0.4 + adx_strength * 0.3 + 
                                 supertrend_signal * 0.2 + returns_momentum * 0.1 + 
                                 price_position_score * 0.1)
            
            regime_scores['spy_trend_score'] = spy_trend_score * params['spy_trend_weight']
        
        # 2. 모멘텀 점수 계산 (기존 유지, 가중치 조정)
        if 'momentum_weight' in params:
            momentum_score = 0
            if 'rsi' in data.columns:
                rsi = data['rsi']
                # NaN 값 처리
                valid_mask = ~rsi.isna()
                
                # RSI 기반 모멘텀 (가중치: 0.5)
                rsi_momentum = np.where(
                    valid_mask,
                    np.where(
                        (rsi > params.get('rsi_oversold', 30)) & (rsi < params.get('rsi_overbought', 70)),
                        0, np.where(rsi > params.get('rsi_overbought', 70), -1, 1)
                    ),
                    0
                )
                
                # MACD 기반 모멘텀 (가중치: 0.3)
                macd_momentum = 0
                if 'macd' in data.columns and 'macd_signal' in data.columns:
                    macd = data['macd']
                    macd_signal = data['macd_signal']
                    macd_momentum = np.where(valid_mask & (macd > macd_signal), 0.5, -0.5)
                
                # Stochastic 기반 모멘텀 (가중치: 0.2)
                stoch_momentum = 0
                if 'stoch_k' in data.columns:
                    stoch_k = data['stoch_k']
                    stoch_momentum = np.where(valid_mask & (stoch_k > 80), -0.3, 
                                            np.where(valid_mask & (stoch_k < 20), 0.3, 0))
                
                momentum_score = (rsi_momentum * 0.5 + macd_momentum * 0.3 + stoch_momentum * 0.2)
            regime_scores['momentum_score'] = momentum_score * params['momentum_weight']
        
        # 3. 변동성 점수 계산 (기존 유지)
        if 'volatility_weight' in params:
            volatility_score = 0
            if 'atr' in data.columns and close_col in data.columns:
                atr_ratio = data['atr'] / data[close_col]
                # NaN 값 처리
                valid_mask = ~(atr_ratio.isna())
                
                # ATR 기반 변동성
                atr_volatility = np.where(valid_mask, np.where(atr_ratio > 0.02, 1, 0), 0)
                
                # Bollinger Band 기반 변동성 추가
                bb_volatility = 0
                if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                    bb_width = (data['bb_upper'] - data['bb_lower']) / data[close_col]
                    bb_volatility = np.where(valid_mask & (bb_width > 0.05), 0.5, 0)
                
                # Keltner Channel 기반 변동성 추가
                keltner_volatility = 0
                if 'keltner_upper' in data.columns and 'keltner_lower' in data.columns:
                    keltner_width = (data['keltner_upper'] - data['keltner_lower']) / data[close_col]
                    keltner_volatility = np.where(valid_mask & (keltner_width > 0.03), 0.3, 0)
                
                volatility_score = atr_volatility + bb_volatility + keltner_volatility
            regime_scores['volatility_score'] = volatility_score * params['volatility_weight']
        
        # 4. 고도화된 매크로 점수 계산 (VIX + TIPS + 기타 지표)
        if 'macro_weight' in params:
            macro_score = 0
            
            # 4-1. VIX 기반 변동성 점수 (고도화)
            if '^VIX' in data.columns:
                vix = data['^VIX']
                # NaN 값 처리
                valid_mask = ~vix.isna()
                
                # VIX 레벨 기반 점수 (가중치: 0.4)
                vix_level_score = np.where(valid_mask, 
                    np.where(vix > params.get('vix_high_threshold', 30), 1,  # 높은 변동성
                    np.where(vix > params.get('vix_medium_threshold', 20), 0.5,  # 중간 변동성
                    np.where(vix < params.get('vix_low_threshold', 15), -0.3, 0))), 0)  # 낮은 변동성
                
                # VIX 변화율 기반 점수 (가중치: 0.3)
                vix_change = vix.pct_change()
                vix_momentum_score = np.where(valid_mask, 
                    np.where(vix_change > 0.15, 0.8,  # 급격한 상승
                    np.where(vix_change > 0.05, 0.4,  # 상승
                    np.where(vix_change < -0.15, -0.8,  # 급격한 하락
                    np.where(vix_change < -0.05, -0.4, 0)))), 0)  # 하락
                
                # VIX 이동평균 기반 점수 (가중치: 0.3)
                vix_ma_score = 0
                if len(vix) >= 20:
                    vix_ma_20 = vix.rolling(20).mean()
                    vix_ma_score = np.where(valid_mask & (vix > vix_ma_20 * 1.2), 0.5,  # VIX > MA20 * 1.2
                                  np.where(valid_mask & (vix < vix_ma_20 * 0.8), -0.3, 0))  # VIX < MA20 * 0.8
                
                vix_total_score = (vix_level_score * 0.4 + vix_momentum_score * 0.3 + vix_ma_score * 0.3)
                macro_score += vix_total_score
            
            # 4-2. TIPS Spread 기반 인플레이션 점수 (새로 추가)
            if 'TIPS_SPREAD' in data.columns:
                tips_spread = data['TIPS_SPREAD']
                valid_mask = ~tips_spread.isna()
                
                # TIPS Spread 레벨 기반 점수
                tips_level_score = np.where(valid_mask,
                    np.where(tips_spread > params.get('tips_high_threshold', 2.5), 0.8,  # 높은 인플레이션 기대
                    np.where(tips_spread > params.get('tips_medium_threshold', 2.0), 0.4,  # 중간 인플레이션 기대
                    np.where(tips_spread < params.get('tips_low_threshold', 1.5), -0.4, 0))), 0)  # 낮은 인플레이션 기대
                
                # TIPS Spread 변화율 기반 점수
                tips_change = tips_spread.pct_change()
                tips_momentum_score = np.where(valid_mask,
                    np.where(tips_change > 0.1, 0.5,  # 인플레이션 기대 상승
                    np.where(tips_change < -0.1, -0.5, 0)), 0)  # 인플레이션 기대 하락
                
                tips_total_score = tips_level_score + tips_momentum_score
                macro_score += tips_total_score
            
            # 4-3. 달러 인덱스 기반 점수 (새로 추가)
            if '^DXY' in data.columns:
                dxy = data['^DXY']
                valid_mask = ~dxy.isna()
                
                # 달러 강도 기반 점수
                dxy_ma_20 = dxy.rolling(20).mean()
                dxy_strength_score = np.where(valid_mask & (dxy > dxy_ma_20 * 1.05), 0.3,  # 강한 달러
                                    np.where(valid_mask & (dxy < dxy_ma_20 * 0.95), -0.3, 0))  # 약한 달러
                
                macro_score += dxy_strength_score
            
            # 4-4. 금 가격 기반 점수 (새로 추가)
            if 'GC=F' in data.columns:
                gold = data['GC=F']
                valid_mask = ~gold.isna()
                
                # 금 가격 추세 기반 점수
                gold_ma_20 = gold.rolling(20).mean()
                gold_trend_score = np.where(valid_mask & (gold > gold_ma_20 * 1.05), 0.2,  # 금 상승
                                  np.where(valid_mask & (gold < gold_ma_20 * 0.95), -0.2, 0))  # 금 하락
                
                macro_score += gold_trend_score
            
            # 4-5. 국채 수익률 기반 점수 (새로 추가)
            if '^TNX' in data.columns:
                treasury_10y = data['^TNX']
                valid_mask = ~treasury_10y.isna()
                
                # 10년 국채 수익률 기반 점수
                treasury_ma_20 = treasury_10y.rolling(20).mean()
                treasury_score = np.where(valid_mask & (treasury_10y > treasury_ma_20 * 1.1), 0.3,  # 금리 상승
                                np.where(valid_mask & (treasury_10y < treasury_ma_20 * 0.9), -0.3, 0))  # 금리 하락
                
                macro_score += treasury_score
            
            regime_scores['macro_score'] = macro_score * params['macro_weight']
        
        # 5. 거래량 점수 계산 (기존 유지)
        if 'volume_weight' in params:
            volume_score = 0
            # 거래량 데이터 컬럼명 매핑 수정
            volume_col = None
            volume_ma_col = None
            obv_col = None
            
            if 'volume' in data.columns:
                volume_col = 'volume'
            if 'volume_ma' in data.columns:
                volume_ma_col = 'volume_ma'
            if 'obv' in data.columns:
                obv_col = 'obv'
            
            if volume_col and volume_ma_col:
                volume = data[volume_col]
                volume_ma = data[volume_ma_col]
                valid_mask = ~(volume.isna() | volume_ma.isna())
                
                volume_ratio = volume / volume_ma
                volume_score = np.where(valid_mask & (volume_ratio > 1.5), 0.5, 
                                      np.where(valid_mask & (volume_ratio < 0.5), -0.3, 0))
                
                if obv_col:
                    obv = data[obv_col]
                    obv_change = obv.pct_change()
                    obv_score = np.where(valid_mask & (obv_change > 0.02), 0.3, 
                                       np.where(valid_mask & (obv_change < -0.02), -0.3, 0))
                    volume_score += obv_score
            regime_scores['volume_score'] = volume_score * params.get('volume_weight', 0.1)
        
        # 6. 지지/저항 점수 계산 (기존 유지)
        if 'support_resistance_weight' in params:
            sr_score = 0
            # 지지/저항 데이터 컬럼명 매핑 수정
            pivot_col = None
            
            if 'pivot_point' in data.columns:
                pivot_col = 'pivot_point'
            elif 'pivot' in data.columns:
                pivot_col = 'pivot'
            
            if pivot_col:
                pivot = data[pivot_col]
                valid_mask = ~pivot.isna()
                
                support_distance = (data[close_col] - pivot) / data[close_col]
                sr_score = np.where(valid_mask & (abs(support_distance) < 0.01), 0.3, 0)
            regime_scores['sr_score'] = sr_score * params.get('support_resistance_weight', 0.1)
        
        # 총점 계산 (SPY 추세+수익률 가중치 강화)
        total_score = regime_scores.sum(axis=1)
        
        # 시장 상태 분류 (동적 임계값 사용)
        trending_up_threshold = params.get('trending_up_threshold', 0.2)
        trending_down_threshold = params.get('trending_down_threshold', -0.2)
        volatile_threshold = params.get('volatile_threshold', 0.1)
        
        regime = pd.Series(index=data.index, dtype='object')
        regime = np.where(total_score > trending_up_threshold, MarketRegime.TRENDING_UP.value,
                 np.where(total_score < trending_down_threshold, MarketRegime.TRENDING_DOWN.value,
                 np.where(regime_scores['volatility_score'] > volatile_threshold, MarketRegime.VOLATILE.value,
                 MarketRegime.SIDEWAYS.value)))
        
        return pd.Series(regime, index=data.index)
    
    def _classify_market_regime_with_probabilities(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """시장 상태 분류 (확률 포함)"""
        regime_scores = pd.DataFrame(index=data.index)
        
        # 컬럼명 매핑 (대소문자 처리)
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 각 지표별 점수 계산 (기존과 동일)
        # 1. 트렌드 점수 (로그 수익률 지표 추가)
        if 'trend_weight' in params:
            trend_score = 0
            
            # 로그 수익률 계산 (주요 지표)
            log_returns = np.log(data[close_col] / data[close_col].shift(1))
            log_returns_ma = log_returns.rolling(window=params.get('log_returns_ma_period', 20)).mean()
            log_returns_std = log_returns.rolling(window=params.get('log_returns_ma_period', 20)).std()
            
            # 로그 수익률 기반 트렌드 점수
            log_return_trend = np.where(
                ~log_returns_ma.isna(),
                np.where(log_returns_ma > 0, 0.5, -0.5),  # 평균 수익률 방향
                0
            )
            
            # 로그 수익률 변동성 점수
            log_return_volatility = np.where(
                ~log_returns_std.isna(),
                np.where(log_returns_std > log_returns_std.quantile(0.7), 0.3, 0),  # 높은 변동성
                0
            )
            
            # 기존 지표들
            if f'sma_{params.get("sma_short", 20)}' in data.columns and f'sma_{params.get("sma_long", 50)}' in data.columns:
                sma_short = data[f'sma_{params.get("sma_short", 20)}']
                sma_long = data[f'sma_{params.get("sma_long", 50)}']
                valid_mask = ~(sma_short.isna() | sma_long.isna())
                
                adx_strength = 0
                if 'adx' in data.columns:
                    adx = data['adx']
                    adx_threshold = params.get('adx_threshold', 25)
                    adx_strength = np.where(valid_mask & (adx > adx_threshold), 0.3, 0)
                
                supertrend_signal = 0
                if 'supertrend' in data.columns:
                    supertrend = data['supertrend']
                    supertrend_signal = np.where(valid_mask & (data[close_col] > supertrend), 0.2, -0.2)
                
                sma_trend = np.where(valid_mask, 
                    np.where(sma_short > sma_long, 0.5, -0.5), 0)
                
                # 종합 트렌드 점수 (로그 수익률 가중치 강화)
                trend_score = log_return_trend + log_return_volatility + sma_trend + adx_strength + supertrend_signal
            else:
                # SMA가 없는 경우 로그 수익률만 사용
                trend_score = log_return_trend + log_return_volatility
                
            regime_scores['trend_score'] = trend_score * params['trend_weight']
        
        # 2. 모멘텀 점수
        if 'momentum_weight' in params:
            momentum_score = 0
            if 'rsi' in data.columns:
                rsi = data['rsi']
                valid_mask = ~rsi.isna()
                
                rsi_momentum = np.where(
                    valid_mask,
                    np.where(
                        (rsi > params.get('rsi_oversold', 30)) & (rsi < params.get('rsi_overbought', 70)),
                        0, np.where(rsi > params.get('rsi_overbought', 70), -1, 1)
                    ),
                    0
                )
                
                macd_momentum = 0
                if 'macd' in data.columns and 'macd_signal' in data.columns:
                    macd = data['macd']
                    macd_signal = data['macd_signal']
                    macd_momentum = np.where(valid_mask & (macd > macd_signal), 0.5, -0.5)
                
                stoch_momentum = 0
                if 'stoch_k' in data.columns:
                    stoch_k = data['stoch_k']
                    stoch_momentum = np.where(valid_mask & (stoch_k > 80), -0.3, 
                                            np.where(valid_mask & (stoch_k < 20), 0.3, 0))
                
                momentum_score = rsi_momentum + macd_momentum + stoch_momentum
            regime_scores['momentum_score'] = momentum_score * params['momentum_weight']
        
        # 3. 변동성 점수
        if 'volatility_weight' in params:
            volatility_score = 0
            if 'atr' in data.columns and close_col in data.columns:
                atr_ratio = data['atr'] / data[close_col]
                valid_mask = ~(atr_ratio.isna())
                
                atr_volatility = np.where(valid_mask, np.where(atr_ratio > 0.02, 1, 0), 0)
                
                bb_volatility = 0
                if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                    bb_width = (data['bb_upper'] - data['bb_lower']) / data[close_col]
                    bb_volatility = np.where(valid_mask & (bb_width > 0.05), 0.5, 0)
                
                keltner_volatility = 0
                if 'keltner_upper' in data.columns and 'keltner_lower' in data.columns:
                    keltner_width = (data['keltner_upper'] - data['keltner_lower']) / data[close_col]
                    keltner_volatility = np.where(valid_mask & (keltner_width > 0.03), 0.3, 0)
                
                volatility_score = atr_volatility + bb_volatility + keltner_volatility
            regime_scores['volatility_score'] = volatility_score * params['volatility_weight']
        
        # 4. 매크로 점수
        if 'macro_weight' in params:
            macro_score = 0
            # VIX 데이터 컬럼명 매핑 수정
            vix_col = None
            if '^VIX' in data.columns:
                vix_col = '^VIX'
            elif 'close' in data.columns and '^vix_data' in str(data.columns):  # VIX 데이터의 close 컬럼
                vix_col = 'close'
            elif 'close' in data.columns:  # 일반적인 close 컬럼 (VIX 데이터인 경우)
                vix_col = 'close'
            
            if vix_col:
                vix = data[vix_col]
                valid_mask = ~vix.isna()
                vix_score = np.where(valid_mask, np.where(vix > params.get('vix_threshold', 25), 1, 0), 0)
                
                vix_change = vix.pct_change()
                vix_momentum = np.where(valid_mask & (vix_change > 0.1), 0.5, 0)
                
                macro_score = vix_score + vix_momentum
            regime_scores['macro_score'] = macro_score * params['macro_weight']
        
        # 5. 거래량 점수
        if 'volume_weight' in params:
            volume_score = 0
            # 거래량 데이터 컬럼명 매핑 수정
            volume_col = None
            volume_ma_col = None
            obv_col = None
            
            if 'volume' in data.columns:
                volume_col = 'volume'
            if 'volume_ma' in data.columns:
                volume_ma_col = 'volume_ma'
            if 'obv' in data.columns:
                obv_col = 'obv'
            
            if volume_col and volume_ma_col:
                volume = data[volume_col]
                volume_ma = data[volume_ma_col]
                valid_mask = ~(volume.isna() | volume_ma.isna())
                
                volume_ratio = volume / volume_ma
                volume_score = np.where(valid_mask & (volume_ratio > 1.5), 0.5, 
                                      np.where(valid_mask & (volume_ratio < 0.5), -0.3, 0))
                
                if obv_col:
                    obv = data[obv_col]
                    obv_change = obv.pct_change()
                    obv_score = np.where(valid_mask & (obv_change > 0.02), 0.3, 
                                       np.where(valid_mask & (obv_change < -0.02), -0.3, 0))
                    volume_score += obv_score
            regime_scores['volume_score'] = volume_score * params.get('volume_weight', 0.1)
        
        # 6. 지지/저항 점수
        if 'support_resistance_weight' in params:
            sr_score = 0
            # 지지/저항 데이터 컬럼명 매핑 수정
            pivot_col = None
            fibonacci_cols = []
            
            if 'pivot_point' in data.columns:
                pivot_col = 'pivot_point'
            elif 'pivot' in data.columns:
                pivot_col = 'pivot'
            
            # Fibonacci 레벨들 확인
            for level in ['0.236', '0.382', '0.500', '0.618', '0.786']:
                fib_col = f'fibonacci_{level}'
                if fib_col in data.columns:
                    fibonacci_cols.append(fib_col)
            
            # Pivot Points 기반 점수
            if pivot_col:
                pivot = data[pivot_col]
                valid_mask = ~pivot.isna()
                
                support_distance = (data[close_col] - pivot) / data[close_col]
                pivot_score = np.where(valid_mask & (abs(support_distance) < 0.01), 0.3, 0)
                sr_score += pivot_score
            
            # Fibonacci 레벨 기반 점수
            if fibonacci_cols:
                for fib_col in fibonacci_cols:
                    fib_level = data[fib_col]
                    valid_mask = ~fib_level.isna()
                    
                    # 가격이 Fibonacci 레벨 근처에 있는지 확인
                    fib_distance = (data[close_col] - fib_level) / data[close_col]
                    fib_score = np.where(valid_mask & (abs(fib_distance) < 0.005), 0.2, 0)  # 더 엄격한 임계값
                    sr_score += fib_score
            
            # Bollinger Bands 기반 지지/저항 점수 (추가)
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                bb_upper = data['bb_upper']
                bb_lower = data['bb_lower']
                valid_mask = ~(bb_upper.isna() | bb_lower.isna())
                
                # 가격이 Bollinger Band 경계 근처에 있는지 확인
                upper_distance = (bb_upper - data[close_col]) / data[close_col]
                lower_distance = (data[close_col] - bb_lower) / data[close_col]
                
                bb_score = np.where(valid_mask & (upper_distance < 0.01), 0.2, 0)  # 저항선 근처
                bb_score += np.where(valid_mask & (lower_distance < 0.01), 0.2, 0)  # 지지선 근처
                sr_score += bb_score
            
            # Keltner Channels 기반 지지/저항 점수 (추가)
            if 'keltner_upper' in data.columns and 'keltner_lower' in data.columns:
                keltner_upper = data['keltner_upper']
                keltner_lower = data['keltner_lower']
                valid_mask = ~(keltner_upper.isna() | keltner_lower.isna())
                
                # 가격이 Keltner Channel 경계 근처에 있는지 확인
                keltner_upper_distance = (keltner_upper - data[close_col]) / data[close_col]
                keltner_lower_distance = (data[close_col] - keltner_lower) / data[close_col]
                
                keltner_score = np.where(valid_mask & (keltner_upper_distance < 0.01), 0.15, 0)  # 저항선 근처
                keltner_score += np.where(valid_mask & (keltner_lower_distance < 0.01), 0.15, 0)  # 지지선 근처
                sr_score += keltner_score
                
            regime_scores['sr_score'] = sr_score * params.get('support_resistance_weight', 0.1)
        
        # 총점 계산
        total_score = regime_scores.sum(axis=1)
        
        # 각 시장 상태별 확률 계산
        trending_up_threshold = params.get('trending_up_threshold', 0.2)
        trending_down_threshold = params.get('trending_down_threshold', -0.2)
        volatile_threshold = params.get('volatile_threshold', 0.1)
        
        # Random Forest 기반 확률 계산
        def calculate_regime_probabilities_random_forest(scores):
            # 특성 매트릭스 구성 (Random Forest 입력용)
            features = np.column_stack([
                # 기본 특성들
                scores,  # 총점
                regime_scores['trend_score'],  # 트렌드 점수
                regime_scores['momentum_score'],  # 모멘텀 점수
                regime_scores['volatility_score'],  # 변동성 점수
                regime_scores.get('macro_score', np.zeros_like(scores)),  # 매크로 점수
                regime_scores.get('volume_score', np.zeros_like(scores)),  # 거래량 점수
                regime_scores.get('sr_score', np.zeros_like(scores)),  # 지지/저항 점수
                
                # 절댓값 특성들
                np.abs(scores),  # 총점 절댓값
                np.abs(regime_scores['trend_score']),  # 트렌드 절댓값
                np.abs(regime_scores['momentum_score']),  # 모멘텀 절댓값
                np.abs(regime_scores['volatility_score']),  # 변동성 절댓값
                
                # 상호작용 특성들
                regime_scores['trend_score'] * regime_scores['momentum_score'],  # 트렌드-모멘텀 상호작용
                regime_scores['trend_score'] * regime_scores['volatility_score'],  # 트렌드-변동성 상호작용
                regime_scores['momentum_score'] * regime_scores['volatility_score'],  # 모멘텀-변동성 상호작용
                regime_scores['volatility_score'] * np.abs(regime_scores['momentum_score']),  # 변동성-모멘텀 절댓값 상호작용
                
                # 비율 특성들
                np.where(regime_scores['volatility_score'] != 0, 
                        regime_scores['momentum_score'] / regime_scores['volatility_score'], 0),  # 모멘텀/변동성 비율
                np.where(regime_scores['momentum_score'] != 0, 
                        regime_scores['trend_score'] / np.abs(regime_scores['momentum_score']), 0),  # 트렌드/모멘텀 비율
                
                # 제곱 특성들 (비선형 관계 포착)
                scores ** 2,  # 총점 제곱
                regime_scores['trend_score'] ** 2,  # 트렌드 제곱
                regime_scores['momentum_score'] ** 2,  # 모멘텀 제곱
                regime_scores['volatility_score'] ** 2,  # 변동성 제곱
                
                # 범주형 특성들 (임계값 기반)
                (scores > trending_up_threshold).astype(float),  # 상승 추세 여부
                (scores < trending_down_threshold).astype(float),  # 하락 추세 여부
                (regime_scores['volatility_score'] > volatile_threshold).astype(float),  # 고변동성 여부
                (regime_scores['trend_score'] > 0).astype(float),  # 트렌드 양수 여부
                (regime_scores['momentum_score'] > 0).astype(float),  # 모멘텀 양수 여부
                
                # 복합 특성들
                np.where(scores > 0, scores * regime_scores['trend_score'], 0),  # 양수 총점 * 트렌드
                np.where(scores < 0, scores * regime_scores['trend_score'], 0),  # 음수 총점 * 트렌드
                np.where(regime_scores['momentum_score'] > 0, 
                        regime_scores['momentum_score'] * regime_scores['volatility_score'], 0),  # 양수 모멘텀 * 변동성
                np.where(regime_scores['momentum_score'] < 0, 
                        np.abs(regime_scores['momentum_score']) * regime_scores['volatility_score'], 0),  # 음수 모멘텀 * 변동성
                
                # 극값 특성들
                np.where(scores > 0.5, 1.0, 0.0),  # 강한 상승 신호
                np.where(scores < -0.5, 1.0, 0.0),  # 강한 하락 신호
                np.where(regime_scores['volatility_score'] > 0.7, 1.0, 0.0),  # 매우 높은 변동성
                np.where(np.abs(regime_scores['momentum_score']) > 0.6, 1.0, 0.0),  # 매우 높은 모멘텀
                
                # 균형 특성들
                np.abs(scores) * regime_scores['volatility_score'],  # 총점 절댓값 * 변동성
                regime_scores['trend_score'] * np.abs(regime_scores['momentum_score']),  # 트렌드 * 모멘텀 절댓값
                
                # 매크로 관련 특성들
                regime_scores.get('macro_score', np.zeros_like(scores)) * regime_scores['volatility_score'],  # 매크로 * 변동성
                regime_scores.get('volume_score', np.zeros_like(scores)) * np.abs(regime_scores['momentum_score']),  # 거래량 * 모멘텀 절댓값
            ])
            
            # Random Forest 스타일의 의사결정 트리 기반 확률 계산
            def tree_based_probability(features_row, regime_type):
                """의사결정 트리 기반 확률 계산"""
                # 특성 언패킹 (총 35개 특성)
                (score, trend, momentum, volatility, macro, volume, sr, 
                 abs_score, abs_trend, abs_momentum, abs_volatility,
                 trend_momentum, trend_volatility, momentum_volatility, vol_momentum_abs,
                 momentum_vol_ratio, trend_momentum_ratio,
                 score_sq, trend_sq, momentum_sq, volatility_sq,
                 is_trending_up, is_trending_down, is_high_vol, is_trend_pos, is_momentum_pos,
                 pos_score_trend, neg_score_trend, pos_momentum_vol, neg_momentum_vol,
                 strong_up, strong_down, very_high_vol, very_high_momentum,
                 abs_score_vol, trend_abs_momentum, macro_vol, volume_abs_momentum) = features_row
                
                if regime_type == 'trending_up':
                    # TRENDING_UP 조건들 (현실적인 조건들)
                    conditions = [
                        score > trending_up_threshold,  # 총점이 임계값보다 높음
                        trend > 0.15,  # 트렌드 점수가 양수 (임계값 더 낮춤)
                        momentum > 0.1,  # 모멘텀 점수가 양수 (임계값 더 낮춤)
                        trend_momentum > 0.02,  # 트렌드-모멘텀 상호작용이 양수 (임계값 더 낮춤)
                        volatility < 0.7,  # 변동성이 낮음 (임계값 더 높임)
                        is_trend_pos,  # 트렌드 양수 여부
                        is_momentum_pos,  # 모멘텀 양수 여부
                        pos_score_trend > 0.02,  # 양수 총점 * 트렌드 (임계값 더 낮춤)
                        momentum_vol_ratio > 0.2,  # 모멘텀/변동성 비율 (임계값 더 낮춤)
                        trend_abs_momentum > 0.02,  # 트렌드 * 모멘텀 절댓값 (임계값 더 낮춤)
                        strong_up,  # 강한 상승 신호
                        # 추가 조건: 약한 상승 신호도 포함
                        score > 0,  # 총점이 양수
                        trend > 0,  # 트렌드가 양수
                    ]
                    # 조건 만족 개수에 따른 확률 (가중치 적용)
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 6.0, 1.0)  # 분모를 7.0에서 6.0으로 낮춤
                
                elif regime_type == 'trending_down':
                    # TRENDING_DOWN 조건들 (현실적인 조건들)
                    conditions = [
                        score < trending_down_threshold,  # 총점이 임계값보다 낮음
                        trend < -0.15,  # 트렌드 점수가 음수 (임계값 더 낮춤)
                        momentum < -0.1,  # 모멘텀 점수가 음수 (임계값 더 낮춤)
                        trend_momentum < -0.02,  # 트렌드-모멘텀 상호작용이 음수 (임계값 더 낮춤)
                        volatility < 0.7,  # 변동성이 낮음 (임계값 더 높임)
                        not is_trend_pos,  # 트렌드 음수 여부
                        not is_momentum_pos,  # 모멘텀 음수 여부
                        neg_score_trend < -0.02,  # 음수 총점 * 트렌드 (임계값 더 낮춤)
                        momentum_vol_ratio < -0.2,  # 모멘텀/변동성 비율 (임계값 더 낮춤)
                        trend_abs_momentum < -0.02,  # 트렌드 * 모멘텀 절댓값 (임계값 더 낮춤)
                        strong_down,  # 강한 하락 신호
                        # 추가 조건: 약한 하락 신호도 포함
                        score < 0,  # 총점이 음수
                        trend < 0,  # 트렌드가 음수
                    ]
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 6.0, 1.0)  # 분모를 7.0에서 6.0으로 낮춤
                
                elif regime_type == 'volatile':
                    # VOLATILE 조건들 (더 현실적인 조건들)
                    conditions = [
                        volatility > volatile_threshold,  # 변동성 점수가 높음
                        abs_momentum > 0.3,  # 모멘텀 절댓값이 높음 (임계값 낮춤)
                        vol_momentum_abs > 0.1,  # 변동성-모멘텀 상호작용이 높음 (임계값 낮춤)
                        abs_score > 0.2,  # 총점 절댓값이 높음 (임계값 낮춤)
                        macro > 0.1,  # 매크로 점수가 높음 (임계값 낮춤)
                        is_high_vol,  # 고변동성 여부
                        very_high_vol,  # 매우 높은 변동성
                        very_high_momentum,  # 매우 높은 모멘텀
                        abs_score_vol > 0.1,  # 총점 절댓값 * 변동성 (임계값 낮춤)
                        macro_vol > 0.05,  # 매크로 * 변동성 (임계값 낮춤)
                        volume_abs_momentum > 0.05,  # 거래량 * 모멘텀 절댓값 (임계값 낮춤)
                    ]
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 6.0, 1.0)  # 분모를 8.0에서 6.0으로 낮춤
                
                else:  # sideways
                    # SIDEWAYS 조건들 (더 엄격한 조건들)
                    not_trending_up = not (score > trending_up_threshold and trend > 0.2)
                    not_trending_down = not (score < trending_down_threshold and trend < -0.2)
                    not_volatile = not (volatility > volatile_threshold and abs_momentum > 0.3)
                    
                    conditions = [
                        not_trending_up,
                        not_trending_down,
                        not_volatile,
                        abs_score < 0.2,  # 총점 절댓값이 낮음 (더 엄격하게)
                        volatility < volatile_threshold * 0.8,  # 변동성이 더 낮음
                        abs_momentum < 0.2,  # 모멘텀 절댓값이 낮음 (더 엄격하게)
                        abs_trend < 0.2,  # 트렌드 절댓값이 낮음 (더 엄격하게)
                        not very_high_vol,  # 매우 높은 변동성이 아님
                        not very_high_momentum,  # 매우 높은 모멘텀이 아님
                        not strong_up,  # 강한 상승 신호가 아님
                        not strong_down,  # 강한 하락 신호가 아님
                        abs_score_vol < 0.05,  # 총점 절댓값 * 변동성이 낮음 (더 엄격하게)
                        # 추가 조건: 모든 점수가 중간 범위에 있어야 함
                        abs_score > 0.05,  # 너무 낮은 점수도 제외
                        volatility > 0.01,  # 너무 낮은 변동성도 제외
                    ]
                    base_conditions = sum(conditions[:5])  # 기본 조건
                    bonus_conditions = sum(conditions[5:])  # 보너스 조건
                    satisfied_conditions = base_conditions + (bonus_conditions * 0.5)
                    return min(satisfied_conditions / 10.0, 1.0)  # 분모를 8.0에서 10.0으로 높임
            
            # 각 상태별 확률 계산
            probabilities = np.zeros((len(scores), 4))
            
            for i in range(len(scores)):
                features_row = features[i]
                probabilities[i, 0] = tree_based_probability(features_row, 'trending_up')
                probabilities[i, 1] = tree_based_probability(features_row, 'trending_down')
                probabilities[i, 2] = tree_based_probability(features_row, 'volatile')
                probabilities[i, 3] = tree_based_probability(features_row, 'sideways')
                
                # 정규화 (합이 1이 되도록)
                total_prob = np.sum(probabilities[i])
                if total_prob > 0:
                    probabilities[i] = probabilities[i] / total_prob
            
            return probabilities
        
        # 확률 계산
        regime_probs = calculate_regime_probabilities_random_forest(total_score)
        
        # 최종 분류 (더 민감한 임계값)
        regime = pd.Series(index=data.index, dtype='object')
        regime = np.where(regime_probs[:, 0] > 0.2, MarketRegime.TRENDING_UP.value,
                 np.where(regime_probs[:, 1] > 0.2, MarketRegime.TRENDING_DOWN.value,
                 np.where(regime_probs[:, 2] > 0.2, MarketRegime.VOLATILE.value,
                 MarketRegime.SIDEWAYS.value)))
        
        # 현재 시장 상태 (마지막 데이터 포인트)
        current_regime = regime[-1] if len(regime) > 0 else MarketRegime.UNCERTAIN.value
        
        # 신뢰도 계산 (확률의 최대값 기반)
        current_probs = regime_probs[-1] if len(regime_probs) > 0 else [0.25, 0.25, 0.25, 0.25]
        confidence = max(current_probs) if len(current_probs) > 0 else 0.5
        
        # 결과 반환
        return {
            'current_regime': current_regime,
            'confidence': confidence,
            'probabilities': {
                'TRENDING_UP': current_probs[0] if len(current_probs) > 0 else 0.25,
                'TRENDING_DOWN': current_probs[1] if len(current_probs) > 1 else 0.25,
                'VOLATILE': current_probs[2] if len(current_probs) > 2 else 0.25,
                'SIDEWAYS': current_probs[3] if len(current_probs) > 3 else 0.25
            },
            'regime_series': pd.Series(regime, index=data.index),
            'probabilities_series': {
                'trending_up': regime_probs[:, 0],
                'trending_down': regime_probs[:, 1],
                'volatile': regime_probs[:, 2],
                'sideways': regime_probs[:, 3]
            },
            'scores': regime_scores,
            'total_score': total_score
        }
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, regime: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """전략별 수익률 계산"""
        returns = pd.Series(index=data.index, dtype=float)
        position = pd.Series(index=data.index, dtype=float)
        
        # 컬럼명 매핑 (대소문자 처리)
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # 첫 번째 포지션 설정
        first_regime = regime.iloc[0]
        if first_regime == MarketRegime.TRENDING_UP.value:
            position.iloc[0] = params.get('trending_boost', 1.0)
        elif first_regime == MarketRegime.TRENDING_DOWN.value:
            position.iloc[0] = -params.get('base_position', 0.5)
        elif first_regime == MarketRegime.SIDEWAYS.value:
            # RSI 기반 첫 번째 포지션
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[0]
                if rsi < params.get('rsi_oversold', 30):
                    position.iloc[0] = params.get('base_position', 0.8)
                elif rsi > params.get('rsi_overbought', 70):
                    position.iloc[0] = -params.get('base_position', 0.8)
                else:
                    position.iloc[0] = params.get('base_position', 0.8) * 0.3
            else:
                position.iloc[0] = params.get('base_position', 0.8) * 0.5
        elif first_regime == MarketRegime.VOLATILE.value:
            position.iloc[0] = params.get('volatile_reduction', 0.5) * params.get('base_position', 0.8)
        else:  # UNCERTAIN
            position.iloc[0] = 0
        
        for i in range(1, len(data)):
            current_regime = regime.iloc[i]
            prev_close = data[close_col].iloc[i-1]
            current_close = data[close_col].iloc[i]
            
            # 시장 상태별 전략
            if current_regime == MarketRegime.TRENDING_UP.value:
                # Buy & Hold 우선, 스윙 전략 보조
                position.iloc[i] = params.get('trending_boost', 1.0)
                
            elif current_regime == MarketRegime.TRENDING_DOWN.value:
                # 현금 보유 또는 역방향 전략
                position.iloc[i] = -params.get('base_position', 0.5)
                
            elif current_regime == MarketRegime.SIDEWAYS.value:
                # 스윙 전략 적극 활용
                if 'rsi' in data.columns:
                    rsi = data['rsi'].iloc[i]
                    if rsi < params.get('rsi_oversold', 30):
                        position.iloc[i] = params.get('base_position', 0.8)
                    elif rsi > params.get('rsi_overbought', 70):
                        position.iloc[i] = -params.get('base_position', 0.8)
                    else:
                        # RSI 중간 구간에서는 작은 포지션으로 스윙
                        position.iloc[i] = params.get('base_position', 0.8) * 0.3
                else:
                    # RSI가 없으면 기본 포지션
                    position.iloc[i] = params.get('base_position', 0.8) * 0.5
                    
            elif current_regime == MarketRegime.VOLATILE.value:
                # 포지션 크기 축소 + 단기 전략
                position.iloc[i] = params.get('volatile_reduction', 0.5) * params.get('base_position', 0.8)
            
            else:  # UNCERTAIN
                position.iloc[i] = 0
            
            # 수익률 계산
            price_return = (current_close - prev_close) / prev_close
            returns.iloc[i] = position.iloc[i] * price_return
        
        return returns
    
    def _calculate_performance_metrics(self, strategy_returns: pd.Series, buy_hold_returns: pd.Series) -> Dict[str, float]:
        """성과 지표 계산 (고도화된 버전)"""
        # 총 수익률 (단순 수익률)
        total_return = (1 + strategy_returns).prod() - 1
        buy_hold_total = (1 + buy_hold_returns).prod() - 1
        
        # 로그 수익률
        log_return = np.log((1 + strategy_returns).prod())
        buy_hold_log_return = np.log((1 + buy_hold_returns).prod())
        
        # 샤프 비율
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # 최대 낙폭
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 승률 (고도화된 계산)
        win_rate = (strategy_returns > 0).mean()
        
        # 승률 향상을 위한 추가 지표들
        # 1. 연속 승/패 분석
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for ret in strategy_returns:
            if ret > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        # 2. 평균 승/패 크기
        winning_returns = strategy_returns[strategy_returns > 0]
        losing_returns = strategy_returns[strategy_returns <= 0]
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        # 3. 수익 팩터 (Profit Factor)
        total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
        total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 4. 칼마 비율 (Calmar Ratio)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 5. 소르티노 비율 (Sortino Ratio) - 하방 변동성만 고려
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = strategy_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # 6. 승률 가중 수익률 (Win Rate Weighted Return)
        win_rate_weighted_return = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # 7. 거래 빈도 분석
        position_changes = strategy_returns.abs() > 0
        trade_frequency = position_changes.sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # 초과 수익률 (Buy & Hold 대비)
        excess_return = total_return - buy_hold_total
        
        return {
            'total_return': total_return,
            'log_return': log_return,
            'buy_hold_return': buy_hold_total,
            'buy_hold_log_return': buy_hold_log_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'win_rate_weighted_return': win_rate_weighted_return,
            'trade_frequency': trade_frequency,
            'total_trades': position_changes.sum()
        }
    
    def objective(self, trial: optuna.Trial, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame]) -> float:
        """Optuna 목적 함수"""
        # 하이퍼파라미터 샘플링
        params = {}
        
        # 지표 파라미터 샘플링
        indicators = self.config.get('market_regime_classification', {}).get('indicators', {})
        
        for category, indicators_dict in indicators.items():
            for indicator, config in indicators_dict.items():
                if config['type'] == 'int':
                    params[indicator] = trial.suggest_int(indicator, config['min'], config['max'])
                elif config['type'] == 'float':
                    params[indicator] = trial.suggest_float(indicator, config['min'], config['max'])
        
        # 분류 가중치 샘플링
        weights = self.config.get('market_regime_classification', {}).get('classification_weights', {})
        for weight, config in weights.items():
            params[weight] = trial.suggest_float(weight, config['min'], config['max'])
        
        # 전략 파라미터 샘플링
        strategy = self.config.get('trading_strategy', {})
        for category, strategy_dict in strategy.items():
            for param, config in strategy_dict.items():
                if config['type'] == 'int':
                    params[param] = trial.suggest_int(param, config['min'], config['max'])
                elif config['type'] == 'float':
                    params[param] = trial.suggest_float(param, config['min'], config['max'])
        
        try:
            # 파생 변수 계산
            data_with_features = self._calculate_derived_features(spy_data, params)
            
            # 매크로 데이터 병합 (컬럼명 대소문자 처리)
            if '^VIX' in macro_data:
                vix_df = macro_data['^VIX']
                # 컬럼명 확인 및 처리
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    self.logger.warning("VIX 데이터에서 close 컬럼을 찾을 수 없습니다.")
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 시장 상태 분류
            regime = self._classify_market_regime(data_with_features, params)
            
            # 전략 수익률 계산
            strategy_returns = self._calculate_strategy_returns(data_with_features, regime, params)
            
            # Buy & Hold 수익률 계산
            close_col = 'close' if 'close' in spy_data.columns else 'Close'
            buy_hold_returns = spy_data[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            # 목적 함수 선택 (승률 중심으로 개선)
            objective_metric = self.config.get('optimization', {}).get('objective', 'win_rate_optimized')
            
            # 목적 함수별 반환값 설정
            if objective_metric == 'win_rate_optimized':
                # 승률 최적화 (승률 + 수익률 + 안정성의 조합)
                win_rate = metrics.get('win_rate', 0)
                total_return = metrics.get('total_return', 0)
                profit_factor = metrics.get('profit_factor', 0)
                max_drawdown = abs(metrics.get('max_drawdown', 0))
                
                # 승률에 높은 가중치를 주되, 수익률과 안정성도 고려
                win_rate_score = win_rate * 100  # 승률을 0-100 스케일로
                return_score = total_return * 50  # 수익률에 적당한 가중치
                profit_factor_score = min(profit_factor * 10, 50)  # 수익 팩터 제한
                drawdown_penalty = max_drawdown * 100  # 낙폭 페널티
                
                # 최종 점수 = 승률 중심 + 수익률 보너스 + 안정성 보너스 - 위험 페널티
                final_score = win_rate_score + return_score + profit_factor_score - drawdown_penalty
                
                # 최소 승률 조건 (50% 미만이면 페널티)
                if win_rate < 0.5:
                    final_score *= 0.5
                
                return final_score
                
            elif objective_metric == 'total_return':
                return metrics.get('total_return', 0)
            elif objective_metric == 'log_return':
                return metrics.get('log_return', 0)
            elif objective_metric == 'excess_return':
                return metrics.get('excess_return', 0)
            elif objective_metric == 'sharpe_ratio':
                return metrics.get('sharpe_ratio', 0)
            elif objective_metric == 'win_rate':
                return metrics.get('win_rate', 0)
            elif objective_metric == 'profit_factor':
                return metrics.get('profit_factor', 0)
            else:
                # 기본값: 승률 최적화
                return metrics.get('win_rate', 0) * 100
            
        except Exception as e:
            self.logger.error(f"목적 함수 실행 중 오류: {e}")
            return -999  # 매우 낮은 값 반환
    
    def optimize_hyperparameters_with_data(self, spy_data: pd.DataFrame, macro_data: Dict[str, pd.DataFrame], n_trials: int = None) -> Dict[str, Any]:
        """하이퍼파라미터 최적화 (이미 로드된 데이터 사용)"""
        if n_trials is None:
            n_trials = self.config.get('optimization', {}).get('n_trials', 100)
        
        if spy_data.empty:
            raise ValueError("SPY 데이터가 비어있습니다")
        
        # Train/Test 분할
        train_test_split = self.config.get('optimization', {}).get('train_test_split', 0.8)
        split_idx = int(len(spy_data) * train_test_split)
        
        train_spy = spy_data.iloc[:split_idx]
        test_spy = spy_data.iloc[split_idx:]
        
        train_macro = {k: v.iloc[:split_idx] if not v.empty else v for k, v in macro_data.items()}
        test_macro = {k: v.iloc[split_idx:] if not v.empty else v for k, v in macro_data.items()}
        
        self.logger.info(f"Train 데이터: {len(train_spy)}개, Test 데이터: {len(test_spy)}개")
        
        # Optuna 스터디 생성 (시드 설정으로 일관성 향상)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        
        # 목적 함수 래퍼
        def objective_wrapper(trial):
            return self.objective(trial, train_spy, train_macro)
        
        # 최적화 실행
        self.logger.info(f"하이퍼파라미터 최적화 시작 (n_trials={n_trials})...")
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # 최적 파라미터
        best_params = study.best_params
        best_value = study.best_value
        
        # Test 데이터에서 성과 평가
        test_performance = self._evaluate_on_test_data(test_spy, test_macro, best_params)
        
        # 결과 저장
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'test_performance': test_performance,
            'n_trials': n_trials,
            'optimization_history': study.trials_dataframe().to_dict('records')
        }
        
        return results
    
    def optimize_hyperparameters(self, start_date: str = None, end_date: str = None, n_trials: int = None,
                               spy_data: pd.DataFrame = None, macro_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """하이퍼파라미터 최적화 (데이터 수집 포함)"""
        if n_trials is None:
            n_trials = self.config.get('optimization', {}).get('n_trials', 100)
        
        # 날짜 설정 (설정 파일 기반)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            # 설정에서 모델 훈련 기간 가져오기
            days_back = self.collector._get_days_back("model_training")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # 데이터 수집 (이미 제공된 데이터가 있으면 재사용)
        if spy_data is None or macro_data is None:
            self.logger.info(f"데이터 수집 중... (기간: {start_date} ~ {end_date})")
            spy_data = self.collector.collect_spy_data(start_date, end_date)
            macro_data = self.collector.collect_macro_indicators(start_date, end_date)
        else:
            self.logger.info("제공된 데이터 재사용 중...")
        
        if spy_data.empty:
            raise ValueError("SPY 데이터 수집 실패")
        
        # Train/Test 분할
        train_test_split = self.config.get('optimization', {}).get('train_test_split', 0.8)
        split_idx = int(len(spy_data) * train_test_split)
        
        train_spy = spy_data.iloc[:split_idx]
        test_spy = spy_data.iloc[split_idx:]
        
        train_macro = {k: v.iloc[:split_idx] if not v.empty else v for k, v in macro_data.items()}
        test_macro = {k: v.iloc[split_idx:] if not v.empty else v for k, v in macro_data.items()}
        
        self.logger.info(f"Train 데이터: {len(train_spy)}개, Test 데이터: {len(test_spy)}개")
        
        # Optuna 스터디 생성 (시드 설정으로 일관성 향상)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        
        # 목적 함수 래퍼
        def objective_wrapper(trial):
            return self.objective(trial, train_spy, train_macro)
        
        # 최적화 실행
        self.logger.info(f"하이퍼파라미터 최적화 시작 (n_trials={n_trials})...")
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # 최적 파라미터
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"최적화 완료 - Best {self.config.get('optimization', {}).get('objective', 'sharpe_ratio')}: {best_value:.4f}")
        
        # Test 데이터에서 성능 평가
        test_performance = self._evaluate_on_test_data(test_spy, test_macro, best_params)
        
        # 결과 구성
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'train_performance': study.best_value,
            'test_performance': test_performance,
            'study': study
        }
        
        # 결과 저장
        try:
            self.save_results(results)
        except Exception as e:
            self.logger.warning(f"최적화 결과 저장 실패: {e}")
        
        return results
    
    def _evaluate_on_test_data(self, test_spy: pd.DataFrame, test_macro: Dict[str, pd.DataFrame], 
                              best_params: Dict[str, Any]) -> Dict[str, float]:
        """Test 데이터에서 성능 평가"""
        try:
            # 파생 변수 계산
            data_with_features = self._calculate_derived_features(test_spy, best_params)
            
            # 매크로 데이터 병합 (컬럼명 대소문자 처리)
            if '^VIX' in test_macro:
                vix_df = test_macro['^VIX']
                # 컬럼명 확인 및 처리
                if 'close' in vix_df.columns:
                    vix_data = vix_df[['close']].rename(columns={'close': '^VIX'})
                elif 'Close' in vix_df.columns:
                    vix_data = vix_df[['Close']].rename(columns={'Close': '^VIX'})
                else:
                    self.logger.warning("VIX 데이터에서 close 컬럼을 찾을 수 없습니다.")
                    vix_data = pd.DataFrame()
                
                if not vix_data.empty:
                    data_with_features = data_with_features.join(vix_data, how='left')
            
            # 시장 상태 분류
            regime = self._classify_market_regime(data_with_features, best_params)
            
            # 전략 수익률 계산
            strategy_returns = self._calculate_strategy_returns(data_with_features, regime, best_params)
            
            # Buy & Hold 수익률 계산
            close_col = 'close' if 'close' in test_spy.columns else 'Close'
            buy_hold_returns = test_spy[close_col].pct_change()
            
            # 성과 지표 계산
            metrics = self._calculate_performance_metrics(strategy_returns, buy_hold_returns)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Test 데이터 평가 중 오류: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results/macro_optimization"):
        """결과 저장 - 안전한 파일 저장 방식으로 개선"""
        import os
        import tempfile
        import shutil
        

        
        try:
            # UUID 기반 디렉토리 생성
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            # 메타데이터 저장
            metadata = {
                'session_uuid': self.session_uuid,
                'optimization_objective': self.config.get('optimization', {}).get('objective', 'sharpe_ratio'),
                'n_trials': len(results['study'].trials) if 'study' in results else 0,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_success = safe_json_dump(
                metadata, 
                f"{session_dir}/metadata.json", 
                "메타데이터",
                self.logger
            )
            
            # 최적 파라미터 저장
            best_params_success = safe_json_dump(
                results['best_params'], 
                f"{session_dir}/best_params.json", 
                "최적 파라미터",
                self.logger
            )
            
            # 성과 지표 저장
            performance_summary = {
                'session_uuid': self.session_uuid,
                'train_performance': results['train_performance'],
                'test_performance': results['test_performance'],
                'optimization_objective': self.config.get('optimization', {}).get('objective', 'sharpe_ratio'),
                'created_at': datetime.now().isoformat()
            }
            
            performance_success = safe_json_dump(
                performance_summary, 
                f"{session_dir}/performance_summary.json", 
                "성과 지표",
                self.logger
            )
            
            # Optuna 스터디 저장
            study_success = True
            if 'study' in results:
                try:
                    # Optuna 버전에 따라 다른 메서드 사용
                    if hasattr(results['study'], 'export_data'):
                        results['study'].export_data(f"{session_dir}/optuna_study.csv")
                        self.logger.info(f"✅ Optuna 스터디 CSV 저장 완료: {session_dir}/optuna_study.csv")
                    else:
                        # 대안: 스터디 정보를 JSON으로 저장
                        study_info = {
                            'session_uuid': self.session_uuid,
                            'best_params': results['study'].best_params,
                            'best_value': results['study'].best_value,
                            'n_trials': len(results['study'].trials),
                            'created_at': datetime.now().isoformat()
                        }
                        study_success = safe_json_dump(
                            study_info, 
                            f"{session_dir}/optuna_study.json", 
                            "Optuna 스터디 정보",
                            self.logger
                        )
                except Exception as e:
                    self.logger.warning(f"Optuna 스터디 저장 중 오류: {e}")
                    study_success = False
            
            # 저장 결과 요약
            total_files = 3 + (1 if 'study' in results else 0)
            successful_files = sum([metadata_success, best_params_success, performance_success, study_success])
            
            if successful_files == total_files:
                self.logger.info(f"✅ 모든 결과 저장 완료: {session_dir} ({successful_files}/{total_files} 파일)")
            else:
                self.logger.warning(f"⚠️ 일부 결과 저장 실패: {session_dir} ({successful_files}/{total_files} 파일 성공)")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류: {e}")

    def _save_results_fallback(self, results: Dict[str, Any], output_dir: str = "results/macro_optimization"):
        """폴백 저장 방식 - 간단한 JSON 저장"""
        try:
            import os
            import json
            from datetime import datetime
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 타임스탬프 기반 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 최적 파라미터만 저장 (study 객체 제외)
            fallback_results = {
                'session_uuid': self.session_uuid,
                'best_params': results.get('best_params', {}),
                'best_value': results.get('best_value', 0),
                'train_performance': results.get('train_performance', {}),
                'test_performance': results.get('test_performance', {}),
                'created_at': datetime.now().isoformat()
            }
            
            # JSON 파일로 저장
            file_path = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 폴백 방식으로 최적화 결과 저장 완료: {file_path}")
            
        except Exception as e:
            self.logger.error(f"폴백 저장도 실패: {e}")

    def save_results(self, results: Dict[str, Any], output_dir: str = "results/macro_optimization"):
        """HyperparamTuner의 결과 저장 메서드"""
        try:
            # UUID 기반 디렉토리 생성
            session_dir = f"{output_dir}/{self.session_uuid}"
            os.makedirs(session_dir, exist_ok=True)
            
            # 메타데이터 저장
            metadata = {
                'session_uuid': self.session_uuid,
                'optimization_objective': self.config.get('optimization', {}).get('objective', 'total_return'),
                'n_trials': results.get('n_trials', 0),
                'created_at': datetime.now().isoformat()
            }
            
            metadata_success = safe_json_dump(
                metadata, 
                f"{session_dir}/metadata.json", 
                "메타데이터",
                self.logger
            )
            
            # 최적 파라미터 저장
            best_params_success = safe_json_dump(
                results.get('best_params', {}), 
                f"{session_dir}/best_params.json", 
                "최적 파라미터",
                self.logger
            )
            
            # 성과 지표 저장
            performance_summary = {
                'session_uuid': self.session_uuid,
                'best_value': results.get('best_value', 0),
                'test_performance': results.get('test_performance', {}),
                'optimization_objective': self.config.get('optimization', {}).get('objective', 'total_return'),
                'created_at': datetime.now().isoformat()
            }
            
            performance_success = safe_json_dump(
                performance_summary, 
                f"{session_dir}/performance_summary.json", 
                "성과 지표",
                self.logger
            )
            
            # 최적화 히스토리 저장 (study 객체가 있다면)
            history_success = True
            if 'optimization_history' in results:
                history_success = safe_json_dump(
                    results['optimization_history'], 
                    f"{session_dir}/optimization_history.json", 
                    "최적화 히스토리",
                    self.logger
                )
            
            # 저장 결과 요약
            total_files = 3 + (1 if 'optimization_history' in results else 0)
            successful_files = sum([metadata_success, best_params_success, performance_success, history_success])
            
            if successful_files == total_files:
                self.logger.info(f"✅ 모든 결과 저장 완료: {session_dir} ({successful_files}/{total_files} 파일)")
            else:
                self.logger.warning(f"⚠️ 일부 결과 저장 실패: {session_dir} ({successful_files}/{total_files} 파일 성공)")
            
        except Exception as e:
            self.logger.error(f"HyperparamTuner 결과 저장 중 오류: {e}")
            # 폴백 저장 시도
            self._save_results_fallback(results, output_dir)

    def _save_results_fallback(self, results: Dict[str, Any], output_dir: str = "results/macro_optimization"):
        """HyperparamTuner의 폴백 저장 방식 - 간단한 JSON 저장"""
        try:
            import os
            import json
            from datetime import datetime
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 타임스탬프 기반 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 최적 파라미터만 저장 (study 객체 제외)
            fallback_results = {
                'session_uuid': self.session_uuid,
                'best_params': results.get('best_params', {}),
                'best_value': results.get('best_value', 0),
                'test_performance': results.get('test_performance', {}),
                'n_trials': results.get('n_trials', 0),
                'created_at': datetime.now().isoformat()
            }
            
            # JSON 파일로 저장
            file_path = os.path.join(output_dir, f"hyperparam_optimization_results_{timestamp}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ HyperparamTuner 폴백 방식으로 최적화 결과 저장 완료: {file_path}")
            
        except Exception as e:
            self.logger.error(f"HyperparamTuner 폴백 저장도 실패: {e}")


def main():
    """테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='글로벌 매크로 데이터 수집 및 하이퍼파라미터 튜닝')
    parser.add_argument('--mode', choices=['collect', 'optimize'], default='collect',
                       help='실행 모드: collect (데이터 수집), optimize (하이퍼파라미터 튜닝)')
    parser.add_argument('--start_date', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna 시도 횟수')
    
    args = parser.parse_args()
    
    # 기본 날짜 설정
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
        
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
    
    print(f"📊 매크로 분석 시작: {start_date} ~ {end_date}")
    
    if args.mode == 'collect':
        # 데이터 수집 모드
        collector = GlobalMacroDataCollector()
        
        # SPY 데이터 수집
        spy_data = collector.collect_spy_data(start_date, end_date)
        
        # 매크로 지표 수집
        macro_data = collector.collect_macro_indicators(start_date, end_date)
        
        # 섹터 데이터 수집
        sector_data = collector.collect_sector_data(start_date, end_date)
        
        # 메트릭 계산
        macro_metrics = collector.calculate_macro_metrics(macro_data)
        sector_rotation = collector.calculate_sector_rotation(sector_data)
        
        # 데이터 저장
        collector.save_macro_data(spy_data, macro_data, sector_data)
        
        print("✅ 매크로 데이터 수집 완료!")
        
    elif args.mode == 'optimize':
        # 하이퍼파라미터 튜닝 모드
        print(f"🔧 하이퍼파라미터 튜닝 시작 (n_trials={args.n_trials})...")
        
        tuner = HyperparamTuner()
        results = tuner.optimize_hyperparameters(start_date, end_date, args.n_trials)
        
        # 결과 출력
        print("\n📈 최적화 결과:")
        print(f"최적 샤프 비율: {results['best_value']:.4f}")
        print(f"최적 파라미터: {results['best_params']}")
        
        if results['test_performance']:
            print(f"\n🧪 Test 성과:")
            for metric, value in results['test_performance'].items():
                print(f"  {metric}: {value:.4f}")
        
        # 결과 저장
        tuner.save_results(results)
        print("✅ 하이퍼파라미터 튜닝 완료!")


if __name__ == "__main__":
    main()
