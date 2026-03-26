"""
렙 단위 운동 데이터 수집기.

흐름:
  counting 중 → on_frame() 매 프레임 호출
  렙 완료 시  → on_rep_complete() 호출
  세트 구분   → new_set() 호출 (↺ 초기화)
  전체 리셋   → full_reset() 호출
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RepRecord:
    rep_index:       int          # 세트 내 렙 번호
    start_time:      float        # unix timestamp
    end_time:        float
    duration:        float        # 렙 소요 시간 (초)
    score_seq:       List[float]  # 프레임별 classify_prob
    avg_score:       float
    min_score:       float
    max_score:       float
    score_std:       float        # 점수 분산 → 동작 안정성
    bad_frame_count: int          # 자세 이탈 프레임 수


@dataclass
class SetRecord:
    set_index: int
    reps: List[RepRecord] = field(default_factory=list)

    @property
    def total_reps(self) -> int:
        return len(self.reps)

    @property
    def avg_score(self) -> float:
        if not self.reps:
            return 0.0
        return float(np.mean([r.avg_score for r in self.reps]))

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.reps)


class RepBuffer:
    """세션 전체의 렙/세트 데이터를 누적합니다."""

    def __init__(self) -> None:
        self._scores: List[float] = []
        self._bad_frames: int     = 0
        self._rep_start: float    = 0.0
        self._sets: List[SetRecord] = [SetRecord(set_index=1)]
        self._rep_index: int      = 0   # 현재 세트 내 렙 번호

    # ── 프로퍼티 ──────────────────────────────────────────────────────────
    @property
    def current_set(self) -> SetRecord:
        return self._sets[-1]

    @property
    def total_reps(self) -> int:
        return sum(s.total_reps for s in self._sets)

    # ── 프레임 단위 기록 ──────────────────────────────────────────────────
    def on_frame(self, score: float, is_bad: bool = False) -> None:
        if not self._scores:
            self._rep_start = time.time()
        self._scores.append(score)
        if is_bad:
            self._bad_frames += 1

    # ── 렙 완료 ───────────────────────────────────────────────────────────
    def on_rep_complete(self) -> None:
        if not self._scores:
            return

        now    = time.time()
        scores = self._scores.copy()
        self._rep_index += 1

        record = RepRecord(
            rep_index       = self._rep_index,
            start_time      = self._rep_start,
            end_time        = now,
            duration        = now - self._rep_start,
            score_seq       = scores,
            avg_score       = float(np.mean(scores)),
            min_score       = float(np.min(scores)),
            max_score       = float(np.max(scores)),
            score_std       = float(np.std(scores)),
            bad_frame_count = self._bad_frames,
        )
        self.current_set.reps.append(record)
        self._scores      = []
        self._bad_frames  = 0

    # ── 세트 구분 (↺ 초기화) ─────────────────────────────────────────────
    def new_set(self) -> None:
        self._sets.append(SetRecord(set_index=len(self._sets) + 1))
        self._rep_index  = 0
        self._scores     = []
        self._bad_frames = 0

    # ── 조회 ──────────────────────────────────────────────────────────────
    def get_sets(self) -> List[SetRecord]:
        """렙이 1개 이상인 세트만 반환"""
        return [s for s in self._sets if s.total_reps > 0]

    # ── 전체 리셋 ─────────────────────────────────────────────────────────
    def full_reset(self) -> None:
        self._scores     = []
        self._bad_frames = 0
        self._rep_start  = 0.0
        self._sets       = [SetRecord(set_index=1)]
        self._rep_index  = 0
