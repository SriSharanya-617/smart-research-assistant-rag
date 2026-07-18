"""
Evaluation and Benchmarking package for the Smart Research Assistant.
Exposes evaluation engines, runner queues, collectors, and report generators.
"""

from src.evaluation.base import EvaluationEngine, MetricsCollector
from src.evaluation.runner import EvaluationRunner
from src.evaluation.reports import ReportGenerator

__all__ = [
    "EvaluationEngine",
    "MetricsCollector",
    "EvaluationRunner",
    "ReportGenerator"
]
