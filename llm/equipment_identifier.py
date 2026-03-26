"""
OpenAI Vision API로 헬스 기구를 분류하는 모듈.
촬영 버튼 클릭 시 단일 프레임을 LLM에 전송해 장비를 식별합니다.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re

import cv2
import numpy as np
from openai import AsyncOpenAI

from exercise_config import EQUIPMENT_EXERCISE_MAP, EQUIPMENT_DISPLAY_NAME

logger = logging.getLogger(__name__)

_PROMPT = """You are a gym equipment expert.
Analyze this image and identify the primary gym equipment visible.

Available equipment categories:
- bench_press: Barbell bench press rack, flat/incline/decline bench with barbell
- squat_rack: Squat rack, power rack, half rack, free-standing squat stand
- smith_machine: Smith machine (barbell on fixed vertical rail/guide rod)
- pull_up_bar: Pull-up bar, chin-up bar, horizontal overhead bar
- dip_bar: Dip bars, parallel bars, V-shape dip station
- lat_machine: Lat pulldown machine, high cable pulley with seat
- cable_machine: Cable crossover machine, functional trainer, low/mid cable pulley station
- leg_machine: Leg press, leg curl, leg extension machine
- pec_deck: Pec deck fly machine, chest fly machine, butterfly machine
- shoulder_machine: Shoulder press machine, lateral raise machine, rear delt machine
- dumbbell: Dumbbells, barbells, EZ curl bar, free weights on rack
- kettlebell: Kettlebells
- cardio_machine: Treadmill, stationary bike, spin bike, rowing machine, elliptical, stair climber
- abs_station: Ab crunch machine, leg raise tower, captain's chair, abdominal bench
- floor: No equipment visible, just floor, exercise mat, or open gym space

Respond ONLY with valid JSON in this exact format:
{
  "equipment": "<one of the category keys above>",
  "display_name": "<equipment name in Korean>",
  "confidence": <0.0 to 1.0>,
  "reason": "<brief reason in Korean, 1 sentence>"
}

If no gym equipment is clearly visible, use "floor".
"""

_VALID_KEYS = set(EQUIPMENT_EXERCISE_MAP.keys())


class EquipmentIdentifier:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            self._client = AsyncOpenAI(api_key=api_key)
            logger.info("OpenAI LLM 클라이언트 초기화 완료")
        else:
            self._client = None
            logger.warning("OPENAI_API_KEY 없음 → LLM 분류 비활성")

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    async def identify(self, frame_bgr: np.ndarray) -> dict:
        """
        프레임을 분석해 장비 정보를 반환합니다.
        반환: { equipment, display_name, confidence, reason, exercises }
        """
        if not self.is_ready:
            return self._fallback("OPENAI_API_KEY가 설정되지 않았습니다.")

        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }],
            )

            raw = response.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            result = json.loads(m.group() if m else raw)

            equipment = result.get("equipment", "unknown")
            if equipment not in _VALID_KEYS:
                equipment = "unknown"

            return {
                "equipment":    equipment,
                "display_name": result.get("display_name", EQUIPMENT_DISPLAY_NAME.get(equipment, equipment)),
                "confidence":   float(result.get("confidence", 0.8)),
                "reason":       result.get("reason", ""),
                "exercises":    EQUIPMENT_EXERCISE_MAP.get(equipment, EQUIPMENT_EXERCISE_MAP["unknown"]),
            }

        except Exception as e:
            logger.error(f"[LLM] 분류 실패: {e}")
            return self._fallback(f"분류 오류: {e}")

    def _fallback(self, reason: str) -> dict:
        return {
            "equipment":    "unknown",
            "display_name": "기구 미감지",
            "confidence":   0.0,
            "reason":       reason,
            "exercises":    EQUIPMENT_EXERCISE_MAP["unknown"],
        }
