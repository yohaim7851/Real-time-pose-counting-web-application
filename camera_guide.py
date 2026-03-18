from __future__ import annotations

import logging
from exercise_config import EXERCISE_CONFIG

logger = logging.getLogger(__name__)

VISIBILITY_THRESHOLD = 0.6   # 키포인트 가시성 기준
REQUIRED_STABLE_FRAMES = 15  # 연속 안정 프레임 수
DECAY_RATE = 2               # 나쁜 프레임 발생 시 안정 카운터 감소량

# 랜드마크 인덱스 → 관절 그룹 (양쪽 합산)
JOINT_GROUP: dict[int, str] = {
    11: "shoulder", 12: "shoulder",
    13: "elbow",    14: "elbow",
    15: "wrist",    16: "wrist",
    23: "hip",      24: "hip",
    25: "knee",     26: "knee",
    27: "ankle",    28: "ankle",
}


class CameraGuide:
    """
    선택된 운동에 필요한 핵심 키포인트의 가시성을 실시간으로 체크합니다.
    REQUIRED_STABLE_FRAMES 연속으로 조건을 만족하면 ready=True 를 반환합니다.
    """

    def __init__(self):
        self.exercise: str | None = None
        self.stable_count: int = 0

    def set_exercise(self, exercise: str):
        self.exercise = exercise
        self.stable_count = 0
        logger.info(f"[CameraGuide] 운동 설정: {exercise}")

    def check_frame(self, landmarks) -> dict:
        """
        landmarks: MediaPipe results.pose_landmarks.landmark
        반환값:
          ready         (bool)  - 카운팅 시작 가능 여부
          instruction   (str)   - 사용자 안내 메시지
          visible_ratio (float) - 필수 키포인트 가시성 비율 0~1
          stable_ratio  (float) - 안정화 진행률 0~1
        """
        if not self.exercise or self.exercise not in EXERCISE_CONFIG:
            return {
                "ready": False,
                "instruction": "운동을 선택해주세요",
                "visible_ratio": 0.0,
                "stable_ratio": 0.0,
            }

        config = EXERCISE_CONFIG[self.exercise]
        required_kps = config["keypoints"]
        min_visible = config["min_visible"]

        visible_count = sum(
            1 for idx in required_kps
            if landmarks[idx].visibility >= VISIBILITY_THRESHOLD
        )
        visible_ratio = round(visible_count / len(required_kps), 2)
        is_good_frame = visible_count >= min_visible

        # 안정 카운터 업데이트: 좋은 프레임이면 증가, 나쁜 프레임이면 감소
        if is_good_frame:
            self.stable_count = min(self.stable_count + 1, REQUIRED_STABLE_FRAMES)
        else:
            self.stable_count = max(0, self.stable_count - DECAY_RATE)

        stable_ratio = round(self.stable_count / REQUIRED_STABLE_FRAMES, 2)
        ready = self.stable_count >= REQUIRED_STABLE_FRAMES

        # 안 보이는 관절 그룹 목록 (중복 제거)
        missing_joints: list[str] = sorted({
            JOINT_GROUP[i]
            for i in required_kps
            if landmarks[i].visibility < VISIBILITY_THRESHOLD and i in JOINT_GROUP
        })

        instruction = ""
        if not ready:
            if visible_ratio < 0.5:
                instruction = config["guide_message"]
            else:
                instruction = "자세를 유지해주세요..."

        return {
            "ready": ready,
            "instruction": instruction,
            "visible_ratio": visible_ratio,
            "stable_ratio": stable_ratio,
            "missing_joints": missing_joints,
        }

    def check_during_counting(self, landmarks) -> bool:
        """
        카운팅 중 프레임 품질 체크.
        카운터 상태는 변경하지 않고 현재 프레임이 유효한지만 반환합니다.
        """
        if not self.exercise or self.exercise not in EXERCISE_CONFIG:
            return False

        config = EXERCISE_CONFIG[self.exercise]
        required_kps = config["keypoints"]
        min_visible = config["min_visible"]

        visible_count = sum(
            1 for idx in required_kps
            if landmarks[idx].visibility >= VISIBILITY_THRESHOLD
        )
        return visible_count >= min_visible

    def reset_stable_count(self):
        self.stable_count = 0

    def reset(self):
        self.exercise = None
        self.stable_count = 0
