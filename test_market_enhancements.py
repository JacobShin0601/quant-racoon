#!/usr/bin/env python3
"""
Market Enhancements 실용 테스트 스크립트

간단한 시장 분석 고도화 기능들을 테스트하고 데모를 보여주는 스크립트입니다.

실행 방법:
    python test_market_enhancements.py
    python test_market_enhancements.py --demo
    python test_market_enhancements.py --full-test
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import json

def load_real_market_data():
    """실제 마켓 데이터 로드"""
    print("📊 실제 마켓 데이터 로드 중...")
    
    data_dir = "data/macro"
    
    # 사용 가능한 CSV 파일들 확인
    csv_files = {
        'SPY': f"{data_dir}/spy_data.csv",
        '^VIX': f"{data_dir}/^vix_data.csv", 
        '^TNX': f"{data_dir}/^tnx_data.csv",
        'XLF': f"{data_dir}/xlf_sector.csv",
        'XRT': f"{data_dir}/xlb_sector.csv",  # 대용으로 XLB 사용
        'QQQ': f"{data_dir}/qqq_data.csv",
        'IWM': f"{data_dir}/iwm_data.csv"
    }
    
    # SPY 데이터 로드
    try:
        spy_data = pd.read_csv(csv_files['SPY'])
        # datetime 컬럼을 인덱스로 설정
        spy_data['datetime'] = pd.to_datetime(spy_data['datetime'])
        spy_data.set_index('datetime', inplace=True)
        # 최근 100일 데이터만 사용
        spy_data = spy_data.tail(100)
        print(f"✅ SPY 데이터 로드 완료: {len(spy_data)}일")
    except Exception as e:
        print(f"❌ SPY 데이터 로드 실패: {e}")
        return None, None
    
    # 매크로 데이터 로드
    macro_data = {}
    
    for symbol, file_path in csv_files.items():
        if symbol == 'SPY':
            continue
            
        try:
            df = pd.read_csv(file_path)
            # datetime 컬럼을 인덱스로 설정
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.tail(100)  # 최근 100일
            macro_data[symbol] = df
            print(f"✅ {symbol} 데이터 로드 완료: {len(df)}일")
        except Exception as e:
            print(f"⚠️ {symbol} 데이터 로드 실패: {e}")
    
    print(f"📈 총 로드된 데이터: SPY + {len(macro_data)}개 지표")
    return spy_data, macro_data

def show_data_summary():
    """실제 로드된 데이터 요약 정보 표시"""
    print("\n📋 로드된 데이터 요약")
    print("=" * 50)
    
    spy_data, macro_data = load_real_market_data()
    
    if spy_data is None or macro_data is None:
        print("❌ 데이터 로드 실패")
        return
    
    # SPY 데이터 요약
    print(f"📈 SPY 데이터:")
    print(f"   • 기간: {spy_data.index[0].date()} ~ {spy_data.index[-1].date()}")
    print(f"   • 데이터 수: {len(spy_data)}일")
    print(f"   • 현재 가격: ${spy_data['close'].iloc[-1]:.2f}")
    print(f"   • 가격 변화: {((spy_data['close'].iloc[-1]/spy_data['close'].iloc[0]-1)*100):+.1f}%")
    
    # 기술적 지표 확인
    available_indicators = []
    for indicator in ['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_short', 'ema_long']:
        if indicator in spy_data.columns:
            available_indicators.append(indicator.upper())
    
    if available_indicators:
        print(f"   • 기술적 지표: {', '.join(available_indicators)}")
    
    # 매크로 데이터 요약
    print(f"\n🌍 매크로 데이터:")
    for symbol, data in macro_data.items():
        latest_price = data['close'].iloc[-1]
        price_change = ((data['close'].iloc[-1]/data['close'].iloc[0]-1)*100)
        print(f"   • {symbol}: ${latest_price:.2f} ({price_change:+.1f}%)")
    
    print(f"\n✅ 총 {len(macro_data)+1}개 데이터셋 로드 완료")

def test_regime_classification():
    """간단한 Market Regime 분류 테스트"""
    print("\n🎯 Market Regime 분류 테스트")
    print("=" * 50)
    
    spy_data, macro_data = load_real_market_data()
    
    if spy_data is None or macro_data is None:
        print("❌ 데이터 로드 실패 - 테스트를 건너뜁니다.")
        return "UNKNOWN", "데이터 없음"
    
    # 간단한 regime 분류 로직
    def classify_regime(prices, vix_values):
        """간단한 regime 분류"""
        returns = prices.pct_change().dropna()
        
        # 최근 20일 수익률과 변동성 분석
        recent_returns = returns.tail(20)
        recent_vix = vix_values.tail(20)
        
        avg_return = recent_returns.mean()
        volatility = recent_returns.std()
        avg_vix = recent_vix.mean()
        
        # 분류 로직
        if avg_vix > 25:
            return "VOLATILE", f"높은 VIX ({avg_vix:.1f})"
        elif avg_return > 0.01:
            return "TRENDING_UP", f"상승 트렌드 ({avg_return:.1%})"
        elif avg_return < -0.01:
            return "TRENDING_DOWN", f"하락 트렌드 ({avg_return:.1%})"
        elif volatility < 0.015:
            return "SIDEWAYS", f"낮은 변동성 ({volatility:.1%})"
        else:
            return "UNCERTAIN", f"불확실한 상황"
    
    # Regime 분류 실행
    regime, reason = classify_regime(spy_data['close'], macro_data['^VIX']['close'])
    
    print(f"📈 현재 Market Regime: {regime}")
    print(f"🔍 분류 근거: {reason}")
    
    # 최근 성과 분석 및 기술적 지표 활용
    recent_return = spy_data['close'].pct_change(20).iloc[-1]
    recent_volatility = spy_data['close'].pct_change().tail(20).std()
    current_vix = macro_data['^VIX']['close'].iloc[-1]
    
    # 기술적 지표들 (이미 CSV에 계산되어 있음)
    current_rsi = spy_data['rsi'].iloc[-1] if 'rsi' in spy_data.columns else None
    current_macd = spy_data['macd'].iloc[-1] if 'macd' in spy_data.columns else None
    current_bb_position = None
    if all(col in spy_data.columns for col in ['bb_upper', 'bb_lower', 'close']):
        bb_upper = spy_data['bb_upper'].iloc[-1]
        bb_lower = spy_data['bb_lower'].iloc[-1]
        current_price = spy_data['close'].iloc[-1]
        current_bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
    
    print(f"📊 최근 20일 수익률: {recent_return:.1%}")
    print(f"📈 현재 변동성: {recent_volatility:.1%}")
    print(f"😰 현재 VIX: {current_vix:.1f}")
    
    if current_rsi is not None:
        rsi_signal = "과매수" if current_rsi > 70 else "과매도" if current_rsi < 30 else "중립"
        print(f"🎯 RSI: {current_rsi:.1f} ({rsi_signal})")
    
    if current_macd is not None:
        macd_signal = "상승" if current_macd > 0 else "하락"
        print(f"📈 MACD: {current_macd:.4f} ({macd_signal})")
    
    if current_bb_position is not None:
        bb_signal = "상단 근처" if current_bb_position > 0.8 else "하단 근처" if current_bb_position < 0.2 else "중간"
        print(f"📊 볼린저 밴드: {current_bb_position:.1%} ({bb_signal})")
    
    return regime, reason

def test_statistical_arbitrage():
    """통계적 차익거래 신호 테스트"""
    print("\n🔄 Statistical Arbitrage 신호 테스트")
    print("=" * 50)
    
    spy_data, macro_data = load_real_market_data()
    
    if spy_data is None or macro_data is None:
        print("❌ 데이터 로드 실패 - 테스트를 건너뜁니다.")
        return 0.0, "⚪ UNKNOWN"
    
    # Key Metrics 가중치 (Keybot the Quant 방식) - 실제 데이터에 맞게 조정
    key_metrics = {
        'XRT': {'weight': 0.32, 'threshold': 0.032},  # 소매업 (XLB로 대체)
        'XLF': {'weight': 0.27, 'threshold': 0.027},  # 금융업  
        '^VIX': {'weight': 0.41, 'threshold': 0.015}   # 변동성
    }
    
    signals = {}
    overall_signal = 0.0
    
    for metric, config in key_metrics.items():
        # 실제 CSV 파일명에 맞게 매핑
        if metric == 'XRT':
            symbol = 'XRT'  # XLB로 대체됨
        else:
            symbol = metric
        
        if symbol in macro_data:
            data = macro_data[symbol]['close']
            
            # 5일 수익률 계산
            returns = data.pct_change(5).iloc[-1]
            
            # 신호 계산
            if returns > config['threshold']:
                signal = 1.0  # 강세
                direction = "🟢 강세"
            elif returns < -config['threshold']:
                signal = -1.0  # 약세  
                direction = "🔴 약세"
            else:
                signal = 0.0  # 중립
                direction = "⚪ 중립"
            
            signals[metric] = {
                'signal': signal,
                'return': returns,
                'direction': direction,
                'weight': config['weight']
            }
            
            overall_signal += signal * config['weight']
            
            print(f"{metric:>3}: {direction} (수익률: {returns:+.1%}, 가중치: {config['weight']:.0%})")
    
    # 전체 신호 방향 결정
    if overall_signal > 0.1:
        market_bias = "🚀 BULLISH"
    elif overall_signal < -0.1:
        market_bias = "📉 BEARISH"
    else:
        market_bias = "⚖️ NEUTRAL"
    
    print(f"\n🎯 종합 신호: {market_bias} (점수: {overall_signal:+.3f})")
    print(f"🎪 신호 강도: {abs(overall_signal):.1%}")
    
    return overall_signal, market_bias

def test_confidence_calculation():
    """신뢰도 계산 테스트"""
    print("\n📊 다층 신뢰도 계산 테스트")
    print("=" * 50)
    
    # 가상의 각 구성요소별 신뢰도
    components = {
        'technical': np.random.uniform(0.4, 0.9),      # 기술적 분석
        'macro': np.random.uniform(0.3, 0.8),          # 매크로 환경
        'statistical_arb': np.random.uniform(0.5, 0.9), # 통계적 차익거래
        'rlmf_feedback': np.random.uniform(0.4, 0.7),   # RLMF 피드백
        'cross_validation': np.random.uniform(0.3, 0.8)  # 교차 검증
    }
    
    # 가중치
    weights = {
        'technical': 0.25,
        'macro': 0.20,
        'statistical_arb': 0.25,
        'rlmf_feedback': 0.20,
        'cross_validation': 0.10
    }
    
    # 가중 신뢰도 계산
    weighted_confidences = {}
    total_confidence = 0
    
    print("구성요소별 신뢰도:")
    for component, confidence in components.items():
        weighted_conf = confidence * weights[component]
        weighted_confidences[component] = weighted_conf
        total_confidence += weighted_conf
        
        print(f"  {component:>15}: {confidence:.1%} × {weights[component]:.0%} = {weighted_conf:.3f}")
    
    # 일관성 점수 계산
    confidence_values = list(weighted_confidences.values())
    consistency = 1.0 - min(np.std(confidence_values) / 0.2, 1.0)
    
    # 최종 조정된 신뢰도
    adjusted_confidence = total_confidence * (0.5 + 0.5 * consistency)
    
    print(f"\n📈 종합 신뢰도: {total_confidence:.1%}")
    print(f"🔗 일관성 점수: {consistency:.1%}")
    print(f"⚖️ 최종 조정 신뢰도: {adjusted_confidence:.1%}")
    
    if adjusted_confidence > 0.7:
        quality = "🟢 높음"
    elif adjusted_confidence > 0.5:
        quality = "🟡 보통"
    else:
        quality = "🔴 낮음"
    
    print(f"🎯 신뢰도 평가: {quality}")
    
    return adjusted_confidence

def test_regime_switching_detection():
    """Regime Switching 감지 테스트"""
    print("\n⚡ Regime Switching 감지 테스트")
    print("=" * 50)
    
    spy_data, macro_data = load_real_market_data()
    
    if spy_data is None or macro_data is None:
        print("❌ 데이터 로드 실패 - 테스트를 건너뜁니다.")
        return False, 0.0
    
    # 간단한 regime shift 감지 로직
    def detect_regime_shift(prices, window_size=30):
        """단순한 regime shift 감지"""
        returns = prices.pct_change().dropna()
        
        if len(returns) < window_size * 2:
            return False, 0.0
        
        # 현재 window와 이전 window 비교
        current_window = returns.tail(window_size)
        previous_window = returns.tail(window_size * 2).head(window_size)
        
        # 변동성 변화
        vol_current = current_window.std()
        vol_previous = previous_window.std()
        vol_change = abs(vol_current - vol_previous) / vol_previous
        
        # 트렌드 변화
        trend_current = current_window.mean()
        trend_previous = previous_window.mean()
        trend_change = abs(trend_current - trend_previous)
        
        # 종합 변화 점수
        shift_score = vol_change * 0.6 + trend_change * 100 * 0.4
        
        return shift_score > 0.3, shift_score
    
    # Regime shift 감지 실행
    shift_detected, shift_score = detect_regime_shift(spy_data['close'])
    
    print(f"🔍 Regime Shift 감지: {'🚨 YES' if shift_detected else '✅ NO'}")
    print(f"📊 변화 점수: {shift_score:.3f}")
    print(f"🎯 감지 임계값: 0.300")
    
    # 추가 분석
    recent_volatility = spy_data['close'].pct_change().tail(30).std()
    previous_volatility = spy_data['close'].pct_change().tail(60).head(30).std()
    
    print(f"📈 최근 30일 변동성: {recent_volatility:.1%}")
    print(f"📊 이전 30일 변동성: {previous_volatility:.1%}")
    print(f"🔄 변동성 변화: {((recent_volatility/previous_volatility-1)*100):+.1f}%")
    
    return shift_detected, shift_score

def generate_trading_recommendations(regime, signal, confidence):
    """거래 추천사항 생성"""
    print("\n💡 거래 추천사항")
    print("=" * 50)
    
    recommendations = []
    
    # Regime 기반 추천
    if regime == "TRENDING_UP":
        recommendations.append("📈 상승 추세 - 모멘텀 전략 활용")
        position_size = 0.8
    elif regime == "TRENDING_DOWN":
        recommendations.append("📉 하락 추세 - 방어적 포지셔닝")
        position_size = 0.3
    elif regime == "SIDEWAYS":
        recommendations.append("↔️ 횡보 - 스윙 트레이딩 전략")
        position_size = 0.6
    elif regime == "VOLATILE":
        recommendations.append("🌪️ 변동성 높음 - 포지션 크기 축소")
        position_size = 0.4
    else:
        recommendations.append("❓ 불확실 - 관망 또는 최소 포지션")
        position_size = 0.2
    
    # Statistical Arbitrage 신호 기반 추천
    if abs(signal) > 0.2:
        direction = "매수 비중 증가" if signal > 0 else "매수 비중 감소"
        recommendations.append(f"🔄 StatArb 신호 - {direction}")
    
    # 신뢰도 기반 조정
    confidence_adjusted_size = position_size * confidence
    
    print(f"🎯 추천 포지션 크기: {confidence_adjusted_size:.1%}")
    print(f"   • 기본 크기: {position_size:.1%}")
    print(f"   • 신뢰도 조정: {confidence:.1%}")
    
    print("\n📋 구체적 추천사항:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # 위험 관리 추천
    print("\n⚠️ 위험 관리:")
    if confidence < 0.5:
        print("   • 낮은 신뢰도 - 손절매 기준 강화")
    if regime == "VOLATILE":
        print("   • 높은 변동성 - 분할 매수/매도 고려")
    if abs(signal) < 0.1:
        print("   • 약한 신호 - 추가 확인 지표 필요")
    
    return confidence_adjusted_size

def demo_mode():
    """데모 모드 - 전체 워크플로우 시연"""
    print("🎪 Market Enhancements 데모 모드")
    print("=" * 80)
    
    # 1. Regime 분류
    regime, reason = test_regime_classification()
    
    # 2. Statistical Arbitrage
    signal, market_bias = test_statistical_arbitrage()
    
    # 3. 신뢰도 계산
    confidence = test_confidence_calculation()
    
    # 4. Regime Switching 감지
    shift_detected, shift_score = test_regime_switching_detection()
    
    # 5. 종합 추천
    position_size = generate_trading_recommendations(regime, signal, confidence)
    
    # 6. 종합 요약
    print("\n" + "=" * 80)
    print("📊 종합 분석 요약")
    print("=" * 80)
    
    print(f"🎯 Market Regime: {regime}")
    print(f"🔄 Statistical Signal: {market_bias}")
    print(f"📊 종합 신뢰도: {confidence:.1%}")
    print(f"⚡ Regime Shift: {'감지됨' if shift_detected else '안정'}")
    print(f"💰 추천 포지션: {position_size:.1%}")
    
    # JSON 형태로도 출력
    summary = {
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "signal": signal,
        "confidence": confidence,
        "regime_shift_detected": shift_detected,
        "recommended_position": position_size
    }
    
    print(f"\n📋 JSON 출력:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Market Enhancements 테스트")
    parser.add_argument('--demo', action='store_true', help='데모 모드 실행')
    parser.add_argument('--full-test', action='store_true', help='전체 테스트 실행')
    parser.add_argument('--data-summary', action='store_true', help='데이터 요약 정보만 표시')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mode()
    elif args.full_test:
        print("🧪 전체 테스트 모드")
        print("=" * 80)
        
        show_data_summary()
        test_regime_classification()
        test_statistical_arbitrage()
        test_confidence_calculation()
        test_regime_switching_detection()
        
        print("\n✅ 모든 테스트 완료!")
    elif args.data_summary:
        show_data_summary()
    else:
        print("🚀 Market Enhancements 기본 테스트")
        print("=" * 80)
        print("사용법:")
        print("  python test_market_enhancements.py --demo         (데모 모드)")
        print("  python test_market_enhancements.py --full-test    (전체 테스트)")
        print("  python test_market_enhancements.py --data-summary (데이터 요약)")
        print("")
        
        # 기본적으로 데이터 요약과 regime 분류 실행
        show_data_summary()
        test_regime_classification()

if __name__ == "__main__":
    main() 