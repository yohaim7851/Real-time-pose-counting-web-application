"""
운동 세션 데이터를 기반으로 GPT-4o-mini 피드백을 생성합니다.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, TYPE_CHECKING

from openai import AsyncOpenAI

from exercise_config import EXERCISE_CONFIG
from core.rep_buffer import SetRecord, RepRecord

logger = logging.getLogger(__name__)

MIN_REPS_FOR_FEEDBACK = 3


# ── 수치 → 행동 언어 변환 ────────────────────────────────────────────────────

def _rep_tags(rep: RepRecord) -> list[str]:
    """렙 하나의 원시 지표를 사람이 이해할 수 있는 태그 목록으로 변환."""
    tags: list[str] = []

    # 자세 안정성
    if rep.avg_score < 0.40:
        tags.append("자세 많이 흔들림")
    elif rep.avg_score < 0.55:
        tags.append("자세 불안정")

    # 최저 구간
    if rep.min_score < 0.35:
        tags.append("특정 구간 자세 무너짐")
    elif rep.min_score < 0.45:
        tags.append("일부 구간 자세 흔들림")

    # 동작 일관성
    if rep.score_std > 0.12:
        tags.append("동작 일관성 부족")
    elif rep.score_std > 0.08:
        tags.append("동작 다소 불규칙")

    # 속도
    if rep.duration < 1.5:
        tags.append("매우 빠름 (강한 반동 의심)")
    elif rep.duration < 2.5:
        tags.append("속도 빠름 (반동 의심)")
    elif rep.duration > 10.0:
        tags.append("동작 매우 느림")

    # 카메라 이탈
    if rep.bad_frame_count > 10:
        tags.append("카메라 이탈 빈번")
    elif rep.bad_frame_count > 4:
        tags.append("카메라 이탈 있음")

    return tags if tags else ["양호"]


def _has_fatigue_trend(reps: list[RepRecord]) -> bool:
    """세트 후반부 평균 점수가 전반부보다 0.08 이상 낮으면 피로 징후로 판단."""
    if len(reps) < 4:
        return False
    mid = len(reps) // 2
    first  = sum(r.avg_score for r in reps[:mid]) / mid
    second = sum(r.avg_score for r in reps[mid:]) / (len(reps) - mid)
    return (first - second) > 0.08


def _build_sets_text(sets: list[SetRecord]) -> str:
    """세트/렙별 데이터를 행동 언어로 변환한 텍스트 블록 생성."""
    lines: list[str] = []
    for s in sets:
        if not s.reps:
            continue
        lines.append(f"[{s.set_index}세트] {s.total_reps}회")
        for r in s.reps:
            tags = _rep_tags(r)
            lines.append(f"  {r.rep_index}번째 동작: {' | '.join(tags)}")
        if _has_fatigue_trend(s.reps):
            lines.append("  ⚠ 세트 후반부로 갈수록 자세가 흔들리는 경향")
    return "\n".join(lines)


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────
def build_feedback_prompt(
    exercise: str,
    sets: List[SetRecord],
    vision_result: Optional[dict] = None,
) -> str:
    exercise_name = EXERCISE_CONFIG.get(exercise, {}).get("display_name", exercise)
    total_reps    = sum(s.total_reps for s in sets)
    total_time    = sum(s.total_duration for s in sets)
    sets_text     = _build_sets_text(sets)

    vision_text = ""
    if vision_result:
        vision_text = f"\n[동작 분류 결과]\n{vision_result.get('summary', '')}\n"

    return f"""당신은 친근한 헬스 트레이너입니다. 아래 운동 관찰 내용을 바탕으로 피드백을 한국어로 작성해주세요.

[운동 정보]
종목: {exercise_name} | 총 {total_reps}회 | {len(sets)}세트 | 총 {total_time:.0f}초
{vision_text}
[렙별 관찰 내용]
{sets_text}

[작성 규칙]
- 점수, 숫자, 내부 지표는 절대 언급하지 마세요.
- "몇 번째 동작"처럼 렙 순서를 자연스럽게 언급해도 됩니다.
- 사용자가 실제로 무엇을 개선해야 하는지 동작 언어로 설명하세요.

[출력 형식]
1. 잘한 점 1가지
2. 개선이 필요한 핵심 사항 2가지 (구체적인 동작 설명 포함)
3. 다음 운동을 위한 한 줄 조언

간결하고 실용적으로 작성해주세요."""


# ── 피드백 생성기 ─────────────────────────────────────────────────────────────
class FeedbackGenerator:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            self._client = AsyncOpenAI(api_key=api_key)
            logger.info("FeedbackGenerator 초기화 완료")
        else:
            self._client = None
            logger.warning("OPENAI_API_KEY 없음 → 피드백 생성 비활성")

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    async def generate(
        self,
        exercise: str,
        sets: List[SetRecord],
        vision_result: Optional[dict] = None,
    ) -> dict:
        """
        Returns:
            {
                "success":    bool,
                "feedback":   str,    # 성공 시
                "message":    str,    # 실패 시
                "total_reps": int,
                "sets":       int,
            }
        """
        total_reps = sum(s.total_reps for s in sets)
        base = {"total_reps": total_reps, "sets": len(sets)}

        if total_reps < MIN_REPS_FOR_FEEDBACK:
            return {
                **base,
                "success": False,
                "message": f"피드백을 생성하려면 최소 {MIN_REPS_FOR_FEEDBACK}회 이상 운동해야 합니다.",
            }

        if not self.is_ready:
            return {
                **base,
                "success": False,
                "message": "피드백 서비스를 사용할 수 없습니다. (API 키 미설정)",
            }

        try:
            prompt   = build_feedback_prompt(exercise, sets, vision_result)
            response = await self._client.chat.completions.create(
                model      = "gpt-4o-mini",
                max_tokens = 600,
                messages   = [{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            logger.info(f"[Feedback] 생성 완료 ({exercise}, {total_reps}회)")
            return {**base, "success": True, "feedback": text}

        except Exception as e:
            logger.error(f"[Feedback] 생성 실패: {e}")
            return {
                **base,
                "success": False,
                "message": "피드백 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            }
