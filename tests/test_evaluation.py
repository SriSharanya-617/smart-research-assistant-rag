"""
Unit tests for the Evaluation & Benchmarking Module (Phase 8).
Verifies CSV/JSON parses, heuristics proxies, logs, and report exports.
"""

import os
import json
import pytest
from unittest.mock import MagicMock
from src.evaluation.base import EvaluationEngine, MetricsCollector
from src.evaluation.runner import EvaluationRunner
from src.evaluation.reports import ReportGenerator
from src.llm.rag_pipeline import RAGPipeline

# ==========================================
# 1. HEURISTIC PROXY METRIC TESTS
# ==========================================

def test_heuristic_quality_calculation():
    engine = EvaluationEngine()
    
    query = "What is Retrieval-Augmented Generation?"
    # High vocabulary overlap context
    context = ["Retrieval-Augmented Generation (RAG) merges LLMs with external data."]
    response = "Retrieval-Augmented Generation RAG merges LLMs with external data."
    
    scores = engine.evaluate_response(query, response, context)
    assert scores["faithfulness"] > 0.8
    assert scores["answer_relevance"] >= 0.6
    assert scores["context_precision"] == 1.0


def test_jaccard_empty_inputs():
    engine = EvaluationEngine()
    # Check boundaries for empty context / query / response
    scores = engine.evaluate_response("", "", [])
    assert scores["faithfulness"] == 0.0
    assert scores["answer_relevance"] == 1.0  # Empty query divided by empty response yields fallback bounds


# ==========================================
# 2. METRICS COLLECTOR TESTS
# ==========================================

def test_metrics_collector_aggregates():
    collector = MetricsCollector()
    
    # Run 1
    collector.record(
        rag_metadata={
            "latency_seconds": 1.5,
            "retrieved_count": 2,
            "confidence_estimate": "High",
            "cache_hit": False,
            "retrieval_inspector": {"latency_seconds": 0.5, "statistics": {"average_score": 0.85}}
        },
        quality_metrics={"faithfulness": 0.9, "answer_relevance": 0.8}
    )
    
    # Run 2 (Cache Hit)
    collector.record(
        rag_metadata={
            "latency_seconds": 0.2,
            "retrieved_count": 2,
            "confidence_estimate": "High",
            "cache_hit": True,
            "retrieval_inspector": {"latency_seconds": 0.0, "statistics": {"average_score": 0.85}}
        },
        quality_metrics={"faithfulness": 1.0, "answer_relevance": 0.9}
    )

    summary = collector.compute_aggregates()
    
    assert summary["total_queries"] == 2
    assert summary["cache_hit_percentage"] == 50.0
    assert summary["average_end_to_end_latency"] == 0.85  # (1.5 + 0.2)/2
    assert summary["average_retrieval_latency"] == 0.25   # (0.5 + 0.0)/2
    assert summary["confidence_distribution"]["High"] == 2
    assert summary["quality"]["faithfulness"] == 0.95


# ==========================================
# 3. REPORT EXPORTS TESTS
# ==========================================

def test_reports_formats():
    result = {
        "dataset_name": "test_suite",
        "timestamp": "2026-07-18T20:00:00",
        "summary": {
            "total_queries": 1,
            "retrieval_success_rate": 100.0,
            "average_end_to_end_latency": 0.5,
            "average_retrieval_latency": 0.1,
            "average_generation_latency": 0.4,
            "cache_hit_percentage": 0.0,
            "llm_failure_percentage": 0.0,
            "average_retrieved_chunks": 1.0,
            "average_citations": 1.0,
            "average_similarity_score": 0.9,
            "confidence_distribution": {"High": 1, "Medium": 0, "Low": 0},
            "quality": {"faithfulness": 0.95, "answer_relevance": 0.9, "context_precision": 1.0, "context_recall": 1.0}
        },
        "queries": [
            {
                "question": "Sample query",
                "generated_answer": "Sample answer",
                "quality_scores": {"faithfulness": 0.95, "answer_relevance": 0.9, "context_precision": 1.0, "context_recall": 1.0}
            }
        ]
    }

    # MD
    md = ReportGenerator.generate_markdown_report(result)
    assert "# RAG System Evaluation & Benchmark Report" in md
    assert "test_suite" in md

    # CSV
    csv_str = ReportGenerator.generate_csv_report(result)
    assert "Question,Generated Answer" in csv_str

    # HTML
    html = ReportGenerator.generate_html_report(result)
    assert "<html>" in html
    assert "Smart RAG Benchmark Report" in html


# ==========================================
# 4. DATASET PARSE & FAILURE TESTS
# ==========================================

def test_dataset_formats_and_duplicates(tmp_path):
    # Setup mock pipeline
    mock_pipeline = MagicMock(spec=RAGPipeline)
    runner = EvaluationRunner(mock_pipeline)

    # 1. JSON parse
    json_path = tmp_path / "dataset.json"
    json_data = [
        {"question": "What is RAG?", "expected_answer": "RAG merges data.", "source": "rag.pdf"},
        {"question": "What is RAG?", "expected_answer": "Duplicate question.", "source": "rag.pdf"} # Duplicate
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
        
    records = runner.load_dataset_from_file(str(json_path))
    # Deduplicates duplicate question
    assert len(records) == 1
    assert records[0]["question"] == "What is RAG?"

    # 2. Empty dataset
    with pytest.raises(ValueError):
        runner.run_evaluation([])


def test_evaluation_partial_failures(tmp_path):
    # Setup pipeline to mock crash on query
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.answer_question.side_effect = Exception("API connection timeout.")
    
    runner = EvaluationRunner(mock_pipeline)
    dataset = [{"question": "Will this fail?"}]
    
    # Run evaluation
    results = runner.run_evaluation(dataset, "failure_suite")
    
    assert results["summary"]["llm_failure_percentage"] == 100.0
    assert len(results["queries"]) == 1
    assert "API connection timeout" in results["queries"][0]["error"]
