from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import base64
import json
import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from exercise_config import EQUIPMENT_EXERCISE_MAP
from llm.feedback_generator import FeedbackGenerator
from llm.equipment_identifier import EquipmentIdentifier
from models.pose_rac import PoseRAC
from core.session import ExerciseSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 앱 초기화
# ---------------------------------------------------------------------------
app = FastAPI(title="GymAI Rep Counter", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# 전역 모델
# ---------------------------------------------------------------------------
_pose_model: PoseRAC | None         = None
_config: dict | None                = None
_index2action: dict[int, str]       = {}
_llm: EquipmentIdentifier | None     = None
_feedback_gen: FeedbackGenerator | None = None


def _load_models():
    global _pose_model, _config, _index2action, _llm, _feedback_gen

    # 새 모델(RepCount-Using-Skeleton-Information) 경로
    _new_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "RepCount-Using-Skeleton-Information",
    )

    with open("RepCount_pose_config.yaml") as f:
        _config = yaml.safe_load(f)

    _csv_path = os.path.join(_new_model_dir, "all_action.csv")
    if not os.path.exists(_csv_path):
        _csv_path = _config["dataset"]["csv_label_path"]
    label_pd   = pd.read_csv(_csv_path)
    logger.info(f"액션 레이블 CSV: {_csv_path}")
    all_labels = {int(r["label"]): r["action"] for _, r in label_pd.iterrows()}

    state = None
    for path in [
        "new_best_weights.pth",
        os.path.join(_new_model_dir, "best_weights.pth"),
        "best_weights.pth",
        "best_weights_PoseRAC.pth",
        "new_weights.pth",
    ]:
        if os.path.exists(path):
            ckpt  = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            logger.info(f"가중치 로드: {path}")
            break

    if state is None:
        raise FileNotFoundError("PoseRAC 가중치 파일을 찾을 수 없습니다.")

    num_classes   = state["fc1.weight"].shape[0]
    _index2action = {k: v for k, v in all_labels.items() if k < num_classes}

    _pose_model = PoseRAC(
        dim=_config["PoseRAC"]["dim"],
        heads=_config["PoseRAC"]["heads"],
        enc_layer=_config["PoseRAC"]["enc_layer"],
        learning_rate=_config["PoseRAC"]["learning_rate"],
        seed=_config["PoseRAC"]["seed"],
        num_classes=num_classes,
        alpha=_config["PoseRAC"]["alpha"],
    )
    _pose_model.load_state_dict(state)
    _pose_model.eval()
    logger.info(f"PoseRAC 로드 완료 ({num_classes}개 동작)")

    _llm          = EquipmentIdentifier()
    _feedback_gen = FeedbackGenerator()


@app.on_event("startup")
async def startup_event():
    _load_models()
    logger.info("서버 시작 완료")


# ---------------------------------------------------------------------------
# HTTP 엔드포인트
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse("static/real_time.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/health")
async def health():
    return {
        "status":     "healthy",
        "pose_model": _pose_model is not None,
        "llm_ready":  _llm is not None and _llm.is_ready,
        "exercises":  list(_index2action.values()),
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/exercise")
async def websocket_exercise(websocket: WebSocket):
    await websocket.accept()
    session     = ExerciseSession(_pose_model, _config, _index2action)
    client_info = str(websocket.client)
    logger.info(f"[WS] 연결: {client_info}")

    async def send(data: dict):
        await websocket.send_text(json.dumps(data))

    try:
        while True:
            msg      = json.loads(await websocket.receive_text())
            msg_type = msg.get("type")

            # ── 프레임 수신 ──────────────────────────────────────────────
            if msg_type == "frame":
                img_bytes = base64.b64decode(msg["data"])
                frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                if session.phase == "detecting":
                    session._last_frame  = frame
                    session._frame_idx  += 1

                    if session._frame_idx % session.DETECT_INTERVAL == 0:
                        loop   = asyncio.get_event_loop()
                        bboxes = await loop.run_in_executor(None, session.detector.detect, frame)
                    else:
                        bboxes = session.detector.last_bboxes

                    await send({"type": "detecting", "phase": "detecting", "bboxes": bboxes})

                elif session.phase in ("setup", "counting"):
                    if session.is_processing:
                        continue  # 이전 프레임 처리 중 → 스킵
                    session.is_processing = True
                    loop = asyncio.get_event_loop()
                    try:
                        result = await loop.run_in_executor(None, session.process_frame, frame)
                        await send(result)
                    finally:
                        session.is_processing = False

            # ── 촬영 → LLM 분류 ──────────────────────────────────────────
            elif msg_type == "capture":
                if session.phase != "detecting" or session._last_frame is None:
                    await send({"type": "error", "message": "캡처할 프레임이 없습니다."})
                    continue

                await send({"type": "analyzing", "phase": "detecting"})

                result            = await _llm.identify(session._last_frame)
                session.equipment = result["equipment"]
                session.phase     = "confirming"

                await send({
                    "type":         "equipment_detected",
                    "phase":        "confirming",
                    "equipment":    result["equipment"],
                    "display_name": result["display_name"],
                    "confidence":   result["confidence"],
                    "reason":       result["reason"],
                    "exercises":    result["exercises"],
                    "bboxes":       session.detector.last_bboxes,
                })

            # ── 장비 확인 / 재시도 ────────────────────────────────────────
            elif msg_type == "confirm_equipment":
                if session.phase == "confirming" and session.equipment:
                    session.phase = "selecting"
                    await send({
                        "type":      "equipment_confirmed",
                        "phase":     "selecting",
                        "equipment": session.equipment,
                        "exercises": EQUIPMENT_EXERCISE_MAP.get(
                            session.equipment, EQUIPMENT_EXERCISE_MAP["unknown"]),
                    })

            elif msg_type == "deny_equipment":
                session.detector.reset()
                session.equipment   = None
                session._last_frame = None
                session.phase       = "detecting"
                session._frame_idx  = 0
                await send({"type": "restarted", "phase": "detecting"})

            # ── 운동 선택 ─────────────────────────────────────────────────
            elif msg_type == "select_exercise":
                exercise = msg.get("exercise", "")
                if session.select_exercise(exercise):
                    await send({"type": "exercise_selected", "exercise": exercise, "phase": "setup"})
                else:
                    await send({"type": "error", "message": f"지원하지 않는 운동: {exercise}"})

            # ── 카운트 리셋 ───────────────────────────────────────────────
            elif msg_type == "reset":
                session.reset_counting()
                await send({"type": "reset_confirmed", "phase": session.phase, "count": 0})

            # ── 전체 재시작 ───────────────────────────────────────────────
            elif msg_type == "restart":
                session.full_reset()
                await send({"type": "restarted", "phase": "detecting"})

            # ── 강제 시작 ─────────────────────────────────────────────────
            elif msg_type == "force_start":
                if session.phase == "setup" and session.exercise:
                    session.phase = "counting"
                    session.counter.pose_buffer.clear()
                    await send({"type": "counting_ready", "phase": "counting"})

            # ── 운동 완료 → 피드백 생성 ───────────────────────────────────
            elif msg_type == "finish_workout":
                if session.phase != "counting":
                    await send({"type": "error", "message": "카운팅 중이 아닙니다."})
                    continue

                session.phase = "finishing"
                await send({"type": "feedback_loading", "phase": "finishing"})

                try:
                    result = await session.generate_feedback(_feedback_gen)
                    await send({"type": "feedback_ready", "phase": "finishing", **result})
                except Exception as e:
                    logger.error(f"[WS] 피드백 생성 오류: {e}", exc_info=True)
                    await send({
                        "type":    "feedback_error",
                        "phase":   "finishing",
                        "message": "피드백 생성 중 오류가 발생했습니다.",
                    })

    except WebSocketDisconnect:
        logger.info(f"[WS] 연결 종료: {client_info}")
    except Exception as e:
        logger.error(f"[WS] 오류: {e}", exc_info=True)
    finally:
        session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
