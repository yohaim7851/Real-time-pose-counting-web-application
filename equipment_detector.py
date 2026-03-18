from __future__ import annotations

import logging
from collections import Counter
from exercise_config import EQUIPMENT_EXERCISE_MAP

logger = logging.getLogger(__name__)

VALID_EQUIPMENT = ["bench_press", "pull_up_bar", "dumbbell", "squat_rack", "floor", "unknown"]

PROMPT = """Look at this gym image and identify the primary fitness equipment visible.
Reply with ONLY one of these exact words (no punctuation, no explanation):

bench_press   - bench press rack or bench press machine
pull_up_bar   - pull-up bar or chin-up bar
dumbbell      - dumbbells or barbells (held or on rack)
squat_rack    - squat rack or power rack
floor         - exercise mat, open floor, or no major equipment
unknown       - cannot determine or other equipment not listed

Reply with only the single word."""


class EquipmentDetector:
    def __init__(self, required_votes: int = 3):
        self.required_votes = required_votes
        self.history: list[str] = []
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. pip install openai")
        return self._client

    async def detect(self, frame_base64: str) -> dict | None:
        """
        프레임을 분석해 장비를 감지합니다.
        required_votes 회 일치 시 확정 결과를 반환, 그 전까지는 None 반환.
        """
        try:
            equipment = await self._call_api(frame_base64)
            self.history.append(equipment)
            logger.info(f"[Equipment] 감지: {equipment} ({len(self.history)}/{self.required_votes})")

            if len(self.history) >= self.required_votes:
                confirmed = Counter(self.history).most_common(1)[0][0]
                return {
                    "equipment": confirmed,
                    "exercises": EQUIPMENT_EXERCISE_MAP.get(confirmed, EQUIPMENT_EXERCISE_MAP["unknown"]),
                }
            return None

        except Exception as e:
            logger.error(f"[Equipment] 감지 오류: {e}")
            self.history.append("unknown")
            if len(self.history) >= self.required_votes:
                return {
                    "equipment": "unknown",
                    "exercises": EQUIPMENT_EXERCISE_MAP["unknown"],
                }
            return None

    async def _call_api(self, frame_base64: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().lower()
        return result if result in VALID_EQUIPMENT else "unknown"

    def reset(self):
        self.history = []

    @property
    def progress(self) -> float:
        return min(len(self.history) / self.required_votes, 1.0)
