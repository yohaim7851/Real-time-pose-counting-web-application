"""
analysis — 운동 동작 분석 (비전 모델 확장용)

FormAnalyzerFactory : 운동별 동작 분석기 레지스트리
FormAnalyzer        : 분석기 추상 인터페이스

확장 방법:
    from analysis.form_analyzer import FormAnalyzerFactory
    FormAnalyzerFactory.register("squat", SquatTDMAnalyzer)
"""
from analysis.form_analyzer import FormAnalyzerFactory, FormAnalyzer

__all__ = ["FormAnalyzerFactory", "FormAnalyzer"]
