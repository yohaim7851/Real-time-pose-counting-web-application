"""
MediaPipe 포즈 감지 + RepCount-Using-Skeleton-Information 추론 + 반복 횟수 카운팅

입력 피처 (104차원):
  - 33 랜드마크 × 3 (x, y, z) → 정규화 → 99
  - 5 평균 관절 각도 (elbow, shoulder, hip, knee, ankle) → 정규화 → 5
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from core.camera_guide import CameraGuide
from models.pose_rac import ActionTrigger, PoseRAC
from utils import encode_frame

logger = logging.getLogger(__name__)

_PL = mp_pose.PoseLandmark


class PoseCounter:
    """
    한 운동 세션의 포즈 추론과 반복 횟수 카운팅을 담당합니다.

    사용 흐름:
      1. set_action(action_index, action_name)  — 운동 선택 후 호출
      2. process_frame(frame, phase) 반복 호출
      3. reset() / full_reset() / close()
    """

    MAX_BAD_FRAMES = 20

    def __init__(self, pose_model: PoseRAC, config: dict, guide: CameraGuide):
        self._model  = pose_model
        self._config = config
        self._guide  = guide

        self.pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.action_index: int = -1
        self._action_name: str = ""
        self.trigger1: ActionTrigger | None = None
        self.trigger2: ActionTrigger | None = None

        self.rep_count       = 0
        self.classify_prob   = 0.5
        self.curr_pose       = "holder"
        self.init_pose       = "pose_holder"
        self.bad_frame_count = 0
        self.counting_paused = False

    # ── 운동 설정 ────────────────────────────────────────────────────────────
    def set_action(self, action_index: int, action_name: str) -> None:
        self.action_index = action_index
        self._action_name = action_name
        cfg = self._config["Action_trigger"]
        self.trigger1 = ActionTrigger(action_name, cfg["enter_threshold"], cfg["exit_threshold"])
        self.trigger2 = ActionTrigger(action_name, cfg["enter_threshold"], cfg["exit_threshold"])

    # ── 프레임 처리 ──────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray, phase: str) -> dict:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.pose_tracker.process(frame_rgb)

        out           = frame_rgb.copy()
        pose_detected = results.pose_landmarks is not None
        landmarks     = None

        if pose_detected:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(
                out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(217, 83, 79), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(8, 255, 200), thickness=2, circle_radius=2),
            )

        frame_b64 = encode_frame(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        base = {"frame": frame_b64, "pose_detected": pose_detected, "phase": phase}

        if phase == "setup":
            return self._handle_setup(base, landmarks, pose_detected)
        if phase == "counting":
            h, w = frame.shape[:2]
            return self._handle_counting(base, landmarks, pose_detected, w, h)
        return {**base, "type": "frame_only"}

    def _handle_setup(self, base: dict, landmarks, pose_detected: bool) -> dict:
        if not pose_detected:
            return {
                **base, "type": "camera_guide", "ready": False,
                "instruction": "카메라에 몸 전체가 보이도록 위치를 조정해주세요",
                "visible_ratio": 0.0, "stable_ratio": 0.0,
            }

        guide    = self._guide.check_frame(landmarks)
        response = {**base, "type": "camera_guide", **guide}
        if guide["ready"]:
            response.update({"type": "counting_ready", "phase": "counting"})
        return response

    def _handle_counting(
        self, base: dict, landmarks, pose_detected: bool, frame_w: int, frame_h: int
    ) -> dict:
        if not pose_detected or landmarks is None:
            self.bad_frame_count += 1
            if self.bad_frame_count >= self.MAX_BAD_FRAMES:
                self.counting_paused = True
        else:
            if self._guide.check_during_counting(landmarks):
                self.bad_frame_count = max(0, self.bad_frame_count - 1)
                self.counting_paused = False
            else:
                self.bad_frame_count += 1
                if self.bad_frame_count >= self.MAX_BAD_FRAMES:
                    self.counting_paused = True

            if not self.counting_paused:
                self._run_inference(landmarks, frame_w, frame_h)

        return {
            **base, "type": "rep_update",
            "count":  self.rep_count,
            "score":  round(self.classify_prob, 3),
            "paused": self.counting_paused,
        }

    # ── RepCount-Using-Skeleton 추론 ─────────────────────────────────────────
    def _run_inference(self, landmarks, frame_w: int, frame_h: int) -> None:
        momentum = self._config["Action_trigger"]["momentum"]
        features = self._compute_features(landmarks, frame_w, frame_h)

        with torch.no_grad():
            x   = torch.from_numpy(features).float().unsqueeze(0)  # (1, 104)
            out = torch.sigmoid(self._model(x))                    # (1, num_classes)
            raw = float(out[0][self.action_index].cpu())  # 선택된 운동 확률만 사용

        # 단순 EMA
        self.classify_prob = raw * (1.0 - momentum) + momentum * self.classify_prob

        s1 = self.trigger1(self.classify_prob)
        s2 = self.trigger2(1.0 - self.classify_prob)

        if self.init_pose == "pose_holder":
            if s1:   self.init_pose = "salient1"
            elif s2: self.init_pose = "salient2"

        if self.init_pose == "salient1":
            if self.curr_pose == "salient1" and s2: self.rep_count += 1
        else:
            if self.curr_pose == "salient2" and s1: self.rep_count += 1

        if s1:   self.curr_pose = "salient1"
        elif s2: self.curr_pose = "salient2"

    # ── 피처 추출 (104차원) ──────────────────────────────────────────────────
    @staticmethod
    def _compute_features(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
        """
        33 랜드마크 → 104차원 피처
          - 99: x,y,z 정규화 (per-frame min-max)
          -  5: 평균 관절 각도 정규화 (5개 중 min-max)
        훈련 전처리(`pre_test_angles.py`)와 동일한 방식.
        """
        # ── 픽셀 스케일 좌표 ────────────────────────────────────────────────
        pts = np.array(
            [[lm.x * frame_w, lm.y * frame_h, lm.z * frame_w]
             for lm in landmarks],
            dtype=np.float32,
        )  # (33, 3)

        # ── xyz 정규화 ───────────────────────────────────────────────────────
        norm_pts = pts.copy()
        for axis in range(3):
            mn = norm_pts[:, axis].min()
            mx = norm_pts[:, axis].max()
            norm_pts[:, axis] = (norm_pts[:, axis] - mn) / (mx - mn + 1e-8)

        # ── 5 평균 관절 각도 계산 ────────────────────────────────────────────
        def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            v1, v2 = a - b, c - b
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

        p = pts
        L = _PL

        def avg(l_val: float, r_val: float) -> float:
            return (l_val + r_val) / 2.0

        elbow_angle    = avg(
            _angle(p[L.LEFT_SHOULDER.value],  p[L.LEFT_ELBOW.value],  p[L.LEFT_WRIST.value]),
            _angle(p[L.RIGHT_SHOULDER.value], p[L.RIGHT_ELBOW.value], p[L.RIGHT_WRIST.value]),
        )
        shoulder_angle = avg(
            _angle(p[L.LEFT_HIP.value],  p[L.LEFT_SHOULDER.value],  p[L.LEFT_ELBOW.value]),
            _angle(p[L.RIGHT_HIP.value], p[L.RIGHT_SHOULDER.value], p[L.RIGHT_ELBOW.value]),
        )
        hip_angle      = avg(
            _angle(p[L.LEFT_SHOULDER.value],  p[L.LEFT_HIP.value],  p[L.LEFT_KNEE.value]),
            _angle(p[L.RIGHT_SHOULDER.value], p[L.RIGHT_HIP.value], p[L.RIGHT_KNEE.value]),
        )
        knee_angle     = avg(
            _angle(p[L.LEFT_HIP.value],  p[L.LEFT_KNEE.value],  p[L.LEFT_ANKLE.value]),
            _angle(p[L.RIGHT_HIP.value], p[L.RIGHT_KNEE.value], p[L.RIGHT_ANKLE.value]),
        )
        ankle_angle    = avg(
            _angle(p[L.LEFT_KNEE.value],  p[L.LEFT_ANKLE.value],  p[L.LEFT_HEEL.value]),
            _angle(p[L.RIGHT_KNEE.value], p[L.RIGHT_ANKLE.value], p[L.RIGHT_HEEL.value]),
        )

        angles = np.array(
            [elbow_angle, shoulder_angle, hip_angle, knee_angle, ankle_angle],
            dtype=np.float32,
        )

        # ── 각도 정규화 (5개 값의 min-max) ──────────────────────────────────
        a_min, a_max = angles.min(), angles.max()
        if a_max - a_min > 1e-8:
            angles = (angles - a_min) / (a_max - a_min)
        else:
            angles = np.zeros_like(angles)

        return np.concatenate([norm_pts.flatten(), angles])  # (99 + 5 = 104,)

    # ── 리셋 ─────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """카운팅 상태만 초기화 (운동 선택은 유지)."""
        self.rep_count       = 0
        self.classify_prob   = 0.5
        self.curr_pose       = "holder"
        self.init_pose       = "pose_holder"
        self.bad_frame_count = 0
        self.counting_paused = False
        self._guide.reset_stable_count()

        if self._action_name:
            cfg = self._config["Action_trigger"]
            self.trigger1 = ActionTrigger(self._action_name, cfg["enter_threshold"], cfg["exit_threshold"])
            self.trigger2 = ActionTrigger(self._action_name, cfg["enter_threshold"], cfg["exit_threshold"])

    def full_reset(self) -> None:
        """운동 선택 포함 전체 초기화."""
        self.action_index = -1
        self._action_name = ""
        self.trigger1     = None
        self.trigger2     = None
        self.reset()

    def close(self) -> None:
        self.pose_tracker.close()
