"""
Market Sensor 고도화 컴포넌트 패키지

이 패키지는 arXiv:2406.15508 논문과 Keybot the Quant 전략을 기반으로 한 
고도화된 시장 분석 시스템 컴포넌트들을 포함합니다.

Components:
- RLMFRegimeAdaptation: RLMF 기반 동적 적응 시스템
- MultiLayerConfidenceSystem: 다층 신뢰도 계산 시스템  
- DynamicRegimeSwitchingDetector: 동적 regime switching 감지
- LLMPrivilegedInformationSystem: LLM 특권 정보 활용 시스템
"""

from .rlmf_adaptation import RLMFRegimeAdaptation
from .confidence_system import MultiLayerConfidenceSystem
from .regime_detection import DynamicRegimeSwitchingDetector
from .llm_insights import LLMPrivilegedInformationSystem

__version__ = "1.0.0"
__author__ = "Quant Team"

__all__ = [
    "RLMFRegimeAdaptation",
    "MultiLayerConfidenceSystem", 
    "DynamicRegimeSwitchingDetector",
    "LLMPrivilegedInformationSystem"
] 