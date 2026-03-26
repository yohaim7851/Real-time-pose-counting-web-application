"""
llm — OpenAI API 연동

EquipmentIdentifier : Vision API 기구 식별 (GPT-4o-mini)
FeedbackGenerator   : 운동 데이터 기반 피드백 생성 (GPT-4o-mini)
"""
from llm.equipment_identifier import EquipmentIdentifier
from llm.feedback_generator import FeedbackGenerator

__all__ = ["EquipmentIdentifier", "FeedbackGenerator"]
