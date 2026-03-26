"""
운동 동작 분석기 팩토리.

현재: 스켈레톤만 구현 (비전 모델 미탑재)
확장: 운동별 VisionAnalyzer 완성 시 register()로 한 줄 등록

사용 예:
    # 스쿼트 비전 모델 완성 후
    from squat_analyzer import SquatTDMAnalyzer
    FormAnalyzerFactory.register("squat", SquatTDMAnalyzer)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.rep_buffer import SetRecord

logger = logging.getLogger(__name__)


class FormAnalyzer(ABC):
    """모든 비전 기반 동작 분석기가 구현해야 하는 인터페이스."""

    @abstractmethod
    def analyze(self, sets: List["SetRecord"]) -> dict:
        """
        Returns:
            {
                "error_type":   str,    # 감지된 주요 오류
                "confidence":   float,  # 분석 신뢰도 (0~1)
                "summary":      str,    # LLM 프롬프트에 삽입할 요약 텍스트
            }
        """


class FormAnalyzerFactory:
    """운동명 → FormAnalyzer 매핑 레지스트리."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, exercise: str, analyzer_cls: type) -> None:
        cls._registry[exercise] = analyzer_cls
        logger.info(f"[FormAnalyzer] 등록: {exercise} → {analyzer_cls.__name__}")

    @classmethod
    def has_vision_model(cls, exercise: str) -> bool:
        return exercise in cls._registry

    @classmethod
    def analyze(cls, exercise: str, sets: List["SetRecord"]) -> Optional[dict]:
        """비전 모델이 없으면 None 반환 (정상 흐름)."""
        if exercise not in cls._registry:
            return None
        try:
            return cls._registry[exercise]().analyze(sets)
        except Exception as e:
            logger.error(f"[FormAnalyzer] {exercise} 분석 실패: {e}")
            return None


# ── 향후 비전 모델 등록 예시 (주석 해제하여 활성화) ──────────────────────
# from squat_analyzer import SquatTDMAnalyzer
# FormAnalyzerFactory.register("squat", SquatTDMAnalyzer)
