"""
WebSocket 연결 1개당 독립적인 세션 상태를 관리합니다.

phase 흐름:
  detecting → confirming → selecting → setup → counting → finishing
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from core.camera_guide import CameraGuide
from core.pose_counter import PoseCounter
from core.rep_buffer import RepBuffer
from detection.equipment_detector import EquipmentDetector
from exercise_config import EXERCISE_CONFIG, EXERCISE_MODEL_MAP
from analysis.form_analyzer import FormAnalyzerFactory
from models.pose_rac import PoseRAC

logger = logging.getLogger(__name__)


class ExerciseSession:

    DETECT_INTERVAL = 5  # YOLO bbox 갱신 주기 (프레임 수)

    def __init__(self, pose_model: PoseRAC, config: dict, index2action: dict[int, str]):
        self._index2action = index2action

        self.phase: str            = "detecting"
        self.equipment: str | None = None
        self.exercise: str | None  = None

        self.detector   = EquipmentDetector()
        self.guide      = CameraGuide()
        self.counter    = PoseCounter(pose_model, config, self.guide)
        self.rep_buffer = RepBuffer()

        self._frame_idx              = 0
        self._last_frame: np.ndarray | None = None
        self.is_processing: bool     = False
        self._prev_count: int        = 0

    # ── 운동 선택 ────────────────────────────────────────────────────────────
    def select_exercise(self, exercise: str) -> bool:
        if exercise not in EXERCISE_CONFIG:
            return False

        model_exercise = EXERCISE_MODEL_MAP.get(exercise, exercise)
        action_index   = next(
            (i for i, name in self._index2action.items() if name == model_exercise), -1
        )
        if action_index == -1:
            return False

        self.exercise = exercise
        self.counter.set_action(action_index, model_exercise)
        self.guide.set_exercise(exercise)
        self.phase = "setup"
        return True

    # ── 프레임 처리 ──────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> dict:
        result = self.counter.process_frame(frame, self.phase)

        if result.get("type") == "counting_ready":
            self.phase = "counting"

        # counting 구간에서 렙 데이터 누적
        if self.phase == "counting" and result.get("type") == "rep_update":
            score  = result.get("score", 0.5)
            is_bad = result.get("paused", False)
            self.rep_buffer.on_frame(score, is_bad)

            count = result.get("count", 0)
            if count > self._prev_count:
                self.rep_buffer.on_rep_complete()
                self._prev_count = count

        return result

    # ── 운동 완료 → 피드백 생성 ──────────────────────────────────────────────
    async def generate_feedback(self, feedback_gen) -> dict:
        """
        비전 모델 추론(있을 경우) + GPT 피드백 생성.
        finish_workout 메시지 수신 시 호출.
        """
        sets = self.rep_buffer.get_sets()

        # 1. 비전 모델 추론 (비동기 executor로 CPU-bound 처리)
        vision_result: Optional[dict] = None
        if self.exercise and FormAnalyzerFactory.has_vision_model(self.exercise):
            loop = asyncio.get_event_loop()
            vision_result = await loop.run_in_executor(
                None, FormAnalyzerFactory.analyze, self.exercise, sets
            )

        # 2. GPT 피드백 생성
        result = await feedback_gen.generate(self.exercise or "", sets, vision_result)
        return result

    # ── 리셋 ─────────────────────────────────────────────────────────────────
    def reset_counting(self) -> None:
        """세트 구분: 카운트 리셋 + 새 세트 시작."""
        self.counter.reset()
        self.rep_buffer.new_set()
        self._prev_count = 0

    def full_reset(self) -> None:
        self.phase       = "detecting"
        self.equipment   = None
        self.exercise    = None
        self._frame_idx  = 0
        self._last_frame = None
        self._prev_count = 0
        self.detector.reset()
        self.guide.reset()
        self.counter.full_reset()
        self.rep_buffer.full_reset()

    def close(self) -> None:
        self.counter.close()
