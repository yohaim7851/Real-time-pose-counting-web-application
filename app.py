from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections import deque

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from camera_guide import CameraGuide
from equipment_detector import EquipmentDetector
from exercise_config import EXERCISE_CONFIG
from model import Action_trigger, PoseRAC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 앱 초기화
# ---------------------------------------------------------------------------
app = FastAPI(title="GymAI Rep Counter", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# 전역 모델 (한 번만 로드)
# ---------------------------------------------------------------------------
_model: PoseRAC | None = None
_config: dict | None = None
_index2action: dict[int, str] = {}


def _load_model():
    global _model, _config, _index2action

    with open("RepCount_pose_config.yaml", "r") as fd:
        _config = yaml.load(fd, Loader=yaml.FullLoader)

    # CSV에서 전체 라벨 로드
    label_pd = pd.read_csv(_config["dataset"]["csv_label_path"])
    all_labels = {int(row["label"]): row["action"] for _, row in label_pd.iterrows()}

    # 체크포인트에서 실제 num_classes 읽기 (CSV와 불일치 방지)
    selected_path = None
    selected_state = None
    for weight_path in ["best_weights_PoseRAC.pth", "new_weights.pth"]:
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location="cpu")
            state = weights.get("state_dict", weights)
            selected_path = weight_path
            selected_state = state
            break

    if selected_state is None:
        raise FileNotFoundError("가중치 파일을 찾을 수 없습니다.")

    num_classes = selected_state["fc1.weight"].shape[0]
    logger.info(f"체크포인트 클래스 수: {num_classes} (파일: {selected_path})")

    # num_classes 범위 내 라벨만 사용
    _index2action = {k: v for k, v in all_labels.items() if k < num_classes}

    _model = PoseRAC(
        None, None, None, None,
        dim=_config["PoseRAC"]["dim"],
        heads=_config["PoseRAC"]["heads"],
        enc_layer=_config["PoseRAC"]["enc_layer"],
        learning_rate=_config["PoseRAC"]["learning_rate"],
        seed=_config["PoseRAC"]["seed"],
        num_classes=num_classes,
        alpha=_config["PoseRAC"]["alpha"],
    )

    _model.load_state_dict(selected_state)
    _model.eval()
    logger.info(f"모델 로드 완료. 운동 목록: {list(_index2action.values())}")


@app.on_event("startup")
async def startup_event():
    _load_model()


# ---------------------------------------------------------------------------
# 세션 클래스 (WebSocket 연결 1개 = 세션 1개)
# ---------------------------------------------------------------------------
class ExerciseSession:
    """
    WebSocket 연결 1개당 독립적인 상태를 관리합니다.

    phase 흐름:
      detecting → selecting → setup → counting
    """

    # ── 장비 감지: 2프레임마다 1회 GPT-4o mini 호출
    DETECT_INTERVAL_FRAMES = 2

    # ── 카운팅 중 연속 불량 프레임 허용 수 (이 이상이면 카운팅 일시정지)
    MAX_BAD_FRAMES = 10

    def __init__(self):
        self.phase: str = "detecting"
        self.equipment: str | None = None
        self.exercise: str | None = None
        self.action_index: int = -1

        # 장비 감지
        self.detector = EquipmentDetector(required_votes=3)
        self.detect_frame_idx: int = 0

        # 카메라 가이드
        self.guide = CameraGuide()

        # 포즈 트래커
        self.pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_buffer: deque = deque(maxlen=30)

        # 카운팅 상태
        self.rep_count: int = 0
        self.classify_prob: float = 0.5
        self.curr_pose: str = "holder"
        self.init_pose: str = "pose_holder"
        self.trigger1: Action_trigger | None = None
        self.trigger2: Action_trigger | None = None

        # 임시 게이트
        self.bad_frame_count: int = 0
        self.counting_paused: bool = False

    # ── 운동 선택 ──────────────────────────────────────────────────────────
    def select_exercise(self, exercise: str) -> bool:
        if exercise not in EXERCISE_CONFIG:
            return False

        self.exercise = exercise
        self.action_index = next(
            (idx for idx, name in _index2action.items() if name == exercise), -1
        )
        if self.action_index == -1:
            logger.warning(f"[Session] 모델에 {exercise} 클래스 없음")
            return False

        cfg = _config["Action_trigger"]
        self.trigger1 = Action_trigger(exercise, cfg["enter_threshold"], cfg["exit_threshold"])
        self.trigger2 = Action_trigger(exercise, cfg["enter_threshold"], cfg["exit_threshold"])
        self.guide.set_exercise(exercise)
        self.phase = "setup"
        logger.info(f"[Session] 운동 선택: {exercise} (index={self.action_index})")
        return True

    # ── 프레임 처리 ────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        현재 phase에 따라 프레임을 처리하고 클라이언트로 보낼 dict를 반환합니다.
        setup / counting 단계에서만 MediaPipe + PoseRAC를 실행합니다.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_tracker.process(frame_rgb)

        output_frame = frame_rgb.copy()
        pose_detected = results.pose_landmarks is not None
        landmarks = None

        if pose_detected:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(
                output_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(217, 83, 79), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(8, 255, 200), thickness=2, circle_radius=2),
            )
            kp = [v for lm in landmarks for v in (lm.x, lm.y, lm.z)]
            self.pose_buffer.append(kp)

        frame_b64 = self._encode_frame(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        base = {"frame": frame_b64, "pose_detected": pose_detected, "phase": self.phase}

        if self.phase == "setup":
            return self._handle_setup(base, landmarks, pose_detected)
        if self.phase == "counting":
            return self._handle_counting(base, landmarks, pose_detected)

        # selecting 단계: 프레임만 전달
        return {**base, "type": "frame_only"}

    def _handle_setup(self, base: dict, landmarks, pose_detected: bool) -> dict:
        if not pose_detected:
            return {
                **base,
                "type": "camera_guide",
                "ready": False,
                "instruction": "카메라에 몸 전체가 보이도록 위치를 조정해주세요",
                "visible_ratio": 0.0,
                "stable_ratio": 0.0,
            }

        guide = self.guide.check_frame(landmarks)
        response = {**base, "type": "camera_guide", **guide}

        if guide["ready"]:
            self.phase = "counting"
            response["type"] = "counting_ready"
            response["phase"] = "counting"

        return response

    def _handle_counting(self, base: dict, landmarks, pose_detected: bool) -> dict:
        if not pose_detected or landmarks is None:
            self.bad_frame_count += 1
            if self.bad_frame_count >= self.MAX_BAD_FRAMES:
                self.counting_paused = True
            return {
                **base,
                "type": "rep_update",
                "count": self.rep_count,
                "score": round(self.classify_prob, 3),
                "paused": self.counting_paused,
            }

        # 카운팅 중 키포인트 품질 게이트
        frame_ok = self.guide.check_during_counting(landmarks)
        if frame_ok:
            self.bad_frame_count = max(0, self.bad_frame_count - 1)
            self.counting_paused = False
        else:
            self.bad_frame_count += 1
            if self.bad_frame_count >= self.MAX_BAD_FRAMES:
                self.counting_paused = True

        # 일시정지 중이면 PoseRAC 추론 건너뜀 (카운트 유지)
        if not self.counting_paused and len(self.pose_buffer) >= 10:
            self._run_inference()

        return {
            **base,
            "type": "rep_update",
            "count": self.rep_count,
            "score": round(self.classify_prob, 3),
            "paused": self.counting_paused,
        }

    def _run_inference(self):
        momentum = _config["Action_trigger"]["momentum"]

        poses = np.array(list(self.pose_buffer)).reshape(-1, 33, 3)
        poses = self._normalize(poses)

        with torch.no_grad():
            tensor = torch.from_numpy(poses).float()
            outputs = torch.sigmoid(_model(tensor))
            latest = outputs[-1]

        raw = float(latest[self.action_index].detach().cpu().numpy())
        # 빠른 상승(피크 추적) + 느린 하강(노이즈 억제)
        if raw > self.classify_prob:
            self.classify_prob = raw * (1.0 - momentum * 0.3) + self.classify_prob * (momentum * 0.3)
        else:
            self.classify_prob = raw * (1.0 - momentum) + momentum * self.classify_prob

        s1 = self.trigger1(self.classify_prob)
        s2 = self.trigger2(1.0 - self.classify_prob)

        if self.init_pose == "pose_holder":
            if s1:
                self.init_pose = "salient1"
            elif s2:
                self.init_pose = "salient2"

        if self.init_pose == "salient1":
            if self.curr_pose == "salient1" and s2:
                self.rep_count += 1
        else:
            if self.curr_pose == "salient2" and s1:
                self.rep_count += 1

        if s1:
            self.curr_pose = "salient1"
        elif s2:
            self.curr_pose = "salient2"

    # ── 유틸 ───────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(landmarks: np.ndarray) -> np.ndarray:
        """landmarks: (N, 33, 3) → (N, 99) normalized"""
        eps = 1e-8
        for axis in range(3):
            col = landmarks[:, :, axis]
            mn = np.expand_dims(col.min(axis=1), 1)
            mx = np.expand_dims(col.max(axis=1), 1)
            landmarks[:, :, axis] = (col - mn) / (mx - mn + eps)
        return landmarks.reshape(len(landmarks), -1)

    @staticmethod
    def _encode_frame(frame: np.ndarray, quality: int = 70) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    def reset_counting(self):
        self.rep_count = 0
        self.classify_prob = 0.5
        self.curr_pose = "holder"
        self.init_pose = "pose_holder"
        self.bad_frame_count = 0
        self.counting_paused = False
        self.guide.reset_stable_count()
        if self.exercise:
            cfg = _config["Action_trigger"]
            self.trigger1 = Action_trigger(self.exercise, cfg["enter_threshold"], cfg["exit_threshold"])
            self.trigger2 = Action_trigger(self.exercise, cfg["enter_threshold"], cfg["exit_threshold"])

    def full_reset(self):
        self.phase = "detecting"
        self.equipment = None
        self.exercise = None
        self.action_index = -1
        self.detector.reset()
        self.detect_frame_idx = 0
        self.guide.reset()
        self.pose_buffer.clear()
        self.reset_counting()

    def close(self):
        self.pose_tracker.close()


# ---------------------------------------------------------------------------
# HTTP 엔드포인트
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse("static/real_time.html")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "exercises": list(_index2action.values()),
    }


# ---------------------------------------------------------------------------
# WebSocket 엔드포인트
# ---------------------------------------------------------------------------
@app.websocket("/ws/exercise")
async def websocket_exercise(websocket: WebSocket):
    await websocket.accept()
    session = ExerciseSession()
    logger.info("[WS] 새 연결")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── 프레임 수신 ──────────────────────────────────────────────
            if msg_type == "frame":
                img_bytes = base64.b64decode(msg["data"])
                arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # detecting 단계: 2프레임마다 GPT-4o mini 호출
                if session.phase == "detecting":
                    session.detect_frame_idx += 1
                    progress = session.detector.progress

                    if session.detect_frame_idx % session.DETECT_INTERVAL_FRAMES == 0:
                        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        frame_b64 = base64.b64encode(buf).decode("utf-8")

                        result = await session.detector.detect(frame_b64)
                        if result:
                            session.equipment = result["equipment"]
                            session.phase = "selecting"
                            await websocket.send_text(json.dumps({
                                "type": "equipment_detected",
                                "phase": "selecting",
                                "equipment": result["equipment"],
                                "exercises": result["exercises"],
                            }))
                            continue

                    await websocket.send_text(json.dumps({
                        "type": "detecting",
                        "phase": "detecting",
                        "progress": round(progress, 2),
                    }))

                # setup / counting 단계: 매 프레임 처리
                elif session.phase in ("setup", "counting"):
                    response = session.process_frame(frame)
                    await websocket.send_text(json.dumps(response))

                # selecting 단계: 프레임 처리 없음 (클라이언트가 로컬 캔버스 사용)

            # ── 운동 선택 ────────────────────────────────────────────────
            elif msg_type == "select_exercise":
                exercise = msg.get("exercise", "")
                if session.select_exercise(exercise):
                    await websocket.send_text(json.dumps({
                        "type": "exercise_selected",
                        "exercise": exercise,
                        "phase": "setup",
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"지원하지 않는 운동: {exercise}",
                    }))

            # ── 카운트 리셋 ──────────────────────────────────────────────
            elif msg_type == "reset":
                session.reset_counting()
                await websocket.send_text(json.dumps({
                    "type": "reset_confirmed",
                    "phase": session.phase,
                    "count": 0,
                }))

            # ── 전체 재시작 ──────────────────────────────────────────────
            elif msg_type == "restart":
                session.full_reset()
                await websocket.send_text(json.dumps({
                    "type": "restarted",
                    "phase": "detecting",
                }))

            # ── 강제 시작 (관절 감지 생략) ──────────────────────────────
            elif msg_type == "force_start":
                if session.phase == "setup" and session.exercise:
                    session.phase = "counting"
                    session.pose_buffer.clear()
                    await websocket.send_text(json.dumps({
                        "type": "counting_ready",
                        "phase": "counting",
                    }))

    except WebSocketDisconnect:
        logger.info("[WS] 연결 종료")
    except Exception as e:
        logger.error(f"[WS] 오류: {e}", exc_info=True)
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
