from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

BBOX_CONFIDENCE_THRESHOLD = 0.25

DETECTION_CLASSES = [
    # 바벨 / 자유 중량
    "barbell", "dumbbell", "kettlebell", "weight plate", "ez curl bar",
    # 벤치 / 랙
    "bench press rack", "flat bench", "incline bench", "barbell bench",
    "squat rack", "power rack", "half rack",
    "smith machine",
    # 철봉 / 딥바
    "pull-up bar", "chin-up bar", "overhead bar",
    "dip bar", "parallel bars",
    # 케이블 / 풀리 머신
    "cable crossover machine", "cable machine", "functional trainer",
    "lat pulldown machine", "high pulley machine",
    # 레그 머신
    "leg press machine", "leg curl machine", "leg extension machine",
    # 체스트 / 숄더 머신
    "pec deck machine", "chest fly machine", "butterfly machine",
    "shoulder press machine", "lateral raise machine",
    # 유산소 기구
    "treadmill", "stationary bike", "spin bike",
    "rowing machine", "elliptical machine", "stair climber",
    # 복근 / 기타
    "ab machine", "captain's chair", "leg raise tower",
    "gym equipment", "weight rack",
    # 사람
    "person",
]


class EquipmentDetector:
    """
    YOLO-World 으로 바운딩 박스만 표시합니다.
    장비 분류는 LLM이 담당하므로, 감지된 객체의 bbox 위치만 반환합니다.
    """

    MODEL_PATH = "yolov8s-worldv2.pt"

    def __init__(self):
        self._model = None
        self.last_bboxes: list[dict] = []
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.MODEL_PATH):
            logger.warning(f"[Detector] '{self.MODEL_PATH}' 없음 → bbox 비활성")
            return
        try:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(self.MODEL_PATH)
            self._model.set_classes(DETECTION_CLASSES)
            logger.info(f"[Detector] YOLO-World 로드 완료: {self.MODEL_PATH} ({len(DETECTION_CLASSES)}개 클래스)")
        except Exception as e:
            logger.error(f"[Detector] 모델 로드 실패: {e}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        바운딩 박스 목록만 반환합니다.
        각 항목: { label, confidence, bbox:[x1n,y1n,x2n,y2n] }
        bbox 좌표는 0-1 정규화값
        """
        h, w = frame.shape[:2]
        bboxes: list[dict] = []

        if self._model is not None:
            results = self._model.predict(frame, verbose=False, conf=BBOX_CONFIDENCE_THRESHOLD)
            for r in results:
                for box in r.boxes:
                    cls_name: str = r.names[int(box.cls)]
                    conf: float   = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bboxes.append({
                        "label":      cls_name,
                        "confidence": round(conf, 3),
                        "bbox":       [x1/w, y1/h, x2/w, y2/h],
                    })

        if bboxes:
            bboxes = [max(bboxes, key=lambda b: b["confidence"])]
            logger.info(f"[Detector] 감지: {bboxes[0]['label']}({bboxes[0]['confidence']:.2f})")

        self.last_bboxes = bboxes
        return bboxes

    def reset(self):
        self.last_bboxes = []

    @property
    def is_ready(self) -> bool:
        return self._model is not None
