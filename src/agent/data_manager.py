#!/usr/bin/env python3
"""
중앙화된 데이터 관리자
모든 데이터 다운로드와 캐시 관리를 담당
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from actions.y_finance import YahooFinanceDataCollector
from actions.global_macro import GlobalMacroDataCollector
from agent.helper import Logger, load_config


class DataCacheManager:
    """데이터 캐시 관리자"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_info_file = self.cache_dir / "cache_info.json"
        self.cache_info = self._load_cache_info()
        
    def _load_cache_info(self) -> Dict:
        """캐시 정보 로드"""
        if self.cache_info_file.exists():
            with open(self.cache_info_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_info(self):
        """캐시 정보 저장"""
        with open(self.cache_info_file, 'w') as f:
            json.dump(self.cache_info, f, indent=2)
    
    def get_cache_key(self, data_type: str, params: Dict) -> str:
        """캐시 키 생성"""
        # 파라미터를 정렬하여 일관된 키 생성
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{data_type}:{sorted_params}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def is_cache_valid(self, cache_key: str, max_age_days: int = 1) -> bool:
        """캐시 유효성 검사"""
        if cache_key not in self.cache_info:
            return False
        
        cache_time = datetime.fromisoformat(self.cache_info[cache_key]['timestamp'])
        age = datetime.now() - cache_time
        
        return age.days < max_age_days
    
    def get_cache_path(self, cache_key: str) -> Optional[Path]:
        """캐시 파일 경로 반환"""
        if cache_key in self.cache_info:
            path = Path(self.cache_info[cache_key]['path'])
            if path.exists():
                return path
        return None
    
    def save_cache(self, cache_key: str, file_path: str, metadata: Dict = None):
        """캐시 정보 저장"""
        self.cache_info[cache_key] = {
            'path': str(file_path),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_cache_info()


class UnifiedDataManager:
    """통합 데이터 관리자"""
    
    def __init__(
        self,
        config_path: str = "config/config_default.json",
        time_horizon: str = "swing",
        use_cached_data: bool = False,
        cache_days: int = 1,
        uuid: Optional[str] = None
    ):
        self.config_path = config_path
        self.time_horizon = time_horizon
        self.use_cached_data = use_cached_data
        self.cache_days = cache_days
        self.uuid = uuid
        
        # 설정 로드
        self.config = load_config(config_path)
        
        # 로거 설정
        self.logger = Logger()
        log_dir = f"log/{time_horizon}"
        self.logger.set_log_dir(log_dir)
        self.logger.setup_logger(strategy="data_manager", mode="download", uuid=uuid)
        
        # 데이터 디렉토리 설정
        self.base_data_dir = Path("data")
        self.time_horizon_dir = self.base_data_dir / time_horizon
        self.macro_dir = self.base_data_dir / "macro"
        self.trader_dir = self.base_data_dir / "trader"
        
        # 디렉토리 생성
        for dir_path in [self.time_horizon_dir, self.macro_dir, self.trader_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 캐시 관리자
        self.cache_manager = DataCacheManager()
        
        # 데이터 수집기
        self.yahoo_collector = YahooFinanceDataCollector()
        self.macro_collector = GlobalMacroDataCollector()
        
    def download_stock_data(
        self,
        symbols: List[str],
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 60,
        target_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """주식 데이터 다운로드 (중복 방지)"""
        
        if target_dir is None:
            target_dir = str(self.time_horizon_dir)
        
        downloaded_files = {}
        
        for symbol in symbols:
            try:
                # 캐시 키 생성
                cache_params = {
                    'symbol': symbol,
                    'interval': interval,
                    'start_date': start_date,
                    'end_date': end_date,
                    'lookback_days': lookback_days
                }
                cache_key = self.cache_manager.get_cache_key('stock', cache_params)
                
                # 캐시 확인
                if self.use_cached_data and self.cache_manager.is_cache_valid(cache_key, self.cache_days):
                    cached_path = self.cache_manager.get_cache_path(cache_key)
                    if cached_path:
                        # 캐시 사용 - 로그 생략
                        downloaded_files[symbol] = str(cached_path)
                        continue
                
                # 새로운 데이터 다운로드
                # 데이터 다운로드 시작
                
                # 종목 정보 가져오기
                info = self.yahoo_collector.get_stock_info(symbol)
                # 종목 정보 확인됨
                
                # 데이터 수집
                df = self.yahoo_collector.get_candle_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    days_back=lookback_days
                )
                
                if df is not None and not df.empty:
                    # 기술적 지표 계산
                    from actions.calculate_index import TechnicalIndicators, StrategyParams
                    params = StrategyParams()
                    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, params)
                    
                    # datetime 컬럼 보장
                    if 'datetime' not in df_with_indicators.columns:
                        df_with_indicators = df_with_indicators.reset_index()
                    
                    # 파일 저장
                    filepath = self.yahoo_collector.save_to_csv(
                        df=df_with_indicators,
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date or "auto",
                        end_date=end_date or "auto",
                        output_dir=target_dir,
                        uuid=self.uuid
                    )
                    
                    # 저장 완료
                    downloaded_files[symbol] = filepath
                    
                    # 캐시 정보 저장
                    self.cache_manager.save_cache(cache_key, filepath, {
                        'symbol': symbol,
                        'rows': len(df_with_indicators),
                        'columns': len(df_with_indicators.columns)
                    })
                    
                else:
                    self.logger.log_info_error(f"  ❌ {symbol} 데이터 수집 실패")
                    
            except Exception as e:
                self.logger.log_info_error(f"  ❌ {symbol} 처리 중 오류: {e}")
                continue
        
        return downloaded_files
    
    def download_macro_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 252
    ) -> Dict[str, str]:
        """매크로 데이터 다운로드 (중복 방지)"""
        
        # 기본 매크로 심볼
        if symbols is None:
            symbols = ['SPY', 'VIX', 'DXY', '^TNX', '^TYX', 'GLD', 'BTC-USD']
        
        downloaded_files = {}
        
        # 캐시 키 생성
        cache_params = {
            'symbols': sorted(symbols),
            'start_date': start_date,
            'end_date': end_date,
            'lookback_days': lookback_days
        }
        cache_key = self.cache_manager.get_cache_key('macro', cache_params)
        
        # 캐시 확인
        if self.use_cached_data and self.cache_manager.is_cache_valid(cache_key, self.cache_days):
            # 매크로 데이터는 여러 파일로 저장되므로 디렉토리 확인
            macro_files = list(self.macro_dir.glob("*.csv"))
            if macro_files:
                # 매크로 데이터 캐시 사용 (로그 생략)
                for file in macro_files:
                    symbol = file.stem.split('_')[0]
                    downloaded_files[symbol] = str(file)
                return downloaded_files
        
        # 새로운 데이터 다운로드
        self.logger.log_info("📈 매크로 데이터 다운로드...")
        
        # global_macro.py의 collect 모드 실행
        import subprocess
        result = subprocess.run(
            [sys.executable, "src/actions/global_macro.py", "--mode", "collect"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 매크로 데이터 다운로드 완료
            
            # 다운로드된 파일 확인
            macro_files = list(self.macro_dir.glob("*.csv"))
            for file in macro_files:
                symbol = file.stem.split('_')[0]
                downloaded_files[symbol] = str(file)
            
            # 캐시 정보 저장
            self.cache_manager.save_cache(cache_key, str(self.macro_dir), {
                'files': len(macro_files),
                'symbols': symbols
            })
        else:
            self.logger.log_info_error(f"❌ 매크로 데이터 다운로드 실패: {result.stderr}")
        
        return downloaded_files
    
    def ensure_data_available(
        self,
        data_type: str,
        symbols: List[str],
        **kwargs
    ) -> bool:
        """데이터 가용성 확인 및 필요시 다운로드"""
        
        if data_type == "stock":
            target_dir = kwargs.get('target_dir', str(self.time_horizon_dir))
            
            # 이미 존재하는 파일 확인 (캐시 만료 검사 포함)
            existing_files = []
            missing_symbols = []
            
            for symbol in symbols:
                pattern = f"{symbol}_*.csv"
                files = list(Path(target_dir).glob(pattern))
                
                if files and self.use_cached_data:
                    # 가장 최근 파일 확인
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    file_age_days = (datetime.now().timestamp() - latest_file.stat().st_mtime) / (24 * 3600)
                    
                    if file_age_days <= self.cache_days:
                        existing_files.append(symbol)
                        # 캐시 사용 (로그 생략)
                    else:
                        missing_symbols.append(symbol)
                        # 캐시 만료
                else:
                    missing_symbols.append(symbol)
            
            if existing_files:
                # 기존 데이터 사용 (로그 생략)
                pass
            
            if missing_symbols:
                self.logger.log_info(f"📊 다운로드 대상: {', '.join(missing_symbols)}")
                downloaded = self.download_stock_data(
                    missing_symbols,
                    interval=kwargs.get('interval', '1d'),
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    lookback_days=kwargs.get('lookback_days', 60),
                    target_dir=target_dir
                )
                return len(downloaded) == len(missing_symbols)
            
            return True
            
        elif data_type == "macro":
            # 매크로 데이터 확인
            macro_files = list(self.macro_dir.glob("*.csv"))
            
            if macro_files and self.use_cached_data:
                # 기존 매크로 데이터 사용 (로그 생략)
                return True
            else:
                downloaded = self.download_macro_data(symbols=symbols, **kwargs)
                return len(downloaded) > 0
        
        return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """현재 데이터 상태 요약"""
        summary = {
            'time_horizon': self.time_horizon,
            'use_cached_data': self.use_cached_data,
            'directories': {
                'time_horizon': str(self.time_horizon_dir),
                'macro': str(self.macro_dir),
                'trader': str(self.trader_dir)
            },
            'files': {}
        }
        
        # 각 디렉토리의 파일 수 계산
        for name, dir_path in summary['directories'].items():
            csv_files = list(Path(dir_path).glob("*.csv"))
            summary['files'][name] = {
                'count': len(csv_files),
                'size_mb': sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
            }
        
        return summary


def main():
    """테스트 및 독립 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="통합 데이터 관리자")
    parser.add_argument("--config", default="config/config_default.json", help="설정 파일")
    parser.add_argument("--time-horizon", default="swing", help="시간 지평")
    parser.add_argument("--use-cached-data", action="store_true", help="캐시 데이터 사용")
    parser.add_argument("--cache-days", type=int, default=1, help="캐시 유효 기간(일)")
    parser.add_argument("--symbols", nargs="+", help="다운로드할 심볼들")
    parser.add_argument("--data-type", choices=["stock", "macro", "all"], default="all", help="데이터 유형")
    parser.add_argument("--lookback-days", type=int, default=700, help="과거 데이터 일수")
    
    args = parser.parse_args()
    
    # 데이터 관리자 초기화
    manager = UnifiedDataManager(
        config_path=args.config,
        time_horizon=args.time_horizon,
        use_cached_data=args.use_cached_data,
        cache_days=args.cache_days
    )
    
    # 데이터 다운로드
    if args.data_type in ["stock", "all"] and args.symbols:
        manager.logger.log_info("\n📊 주식 데이터 다운로드...")
        manager.ensure_data_available("stock", args.symbols, lookback_days=args.lookback_days)
    
    if args.data_type in ["macro", "all"]:
        manager.logger.log_info("\n📈 매크로 데이터 다운로드...")
        manager.ensure_data_available("macro", [])
    
    # 요약 출력
    manager.logger.log_info("\n📋 데이터 상태 요약:")
    summary = manager.get_data_summary()
    manager.logger.log_info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()