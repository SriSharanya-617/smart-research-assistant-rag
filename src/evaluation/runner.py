"""
Evaluation Runner module.
Handles dataset imports and manages evaluation history logs.
"""

import os
import csv
import json
import time
import datetime
from typing import Dict, Any, List, Optional
from src.llm.rag_pipeline import RAGPipeline
from src.evaluation.base import EvaluationEngine, MetricsCollector
from src.logger import setup_logger

logger = setup_logger("evaluation_runner")

HISTORY_LOG_FILE = os.path.join("data", "evaluation_history.json")

class EvaluationRunner:
    """
    Parses datasets, executes runs over the RAG pipeline, and logs historical summaries.
    """
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.eval_engine = EvaluationEngine()
        self.metrics_collector = MetricsCollector()

    def load_dataset_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parses test dataset records (CSV, JSON, TXT).
        Each record has: question, expected_answer (optional), and source (optional).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")

        records = []
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = row.get("question") or row.get("query")
                    if q:
                        records.append({
                            "question": q.strip(),
                            "expected_answer": (row.get("expected_answer") or row.get("answer") or "").strip(),
                            "source": (row.get("source") or "").strip()
                        })
                        
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle lists of dicts
                if isinstance(data, list):
                    for item in data:
                        q = item.get("question") or item.get("query")
                        if q:
                            records.append({
                                "question": q.strip(),
                                "expected_answer": (item.get("expected_answer") or item.get("answer") or "").strip(),
                                "source": (item.get("source") or "").strip()
                            })
                            
        elif ext in [".txt", ".log"]:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    q = line.strip()
                    if q:
                        # For plain TXT, each line is treated as a separate question query
                        records.append({
                            "question": q,
                            "expected_answer": "",
                            "source": ""
                        })
        else:
            raise ValueError(f"Unsupported dataset file extension: {ext}")

        # Check for duplicates
        seen = set()
        unique_records = []
        for r in records:
            if r["question"] not in seen:
                seen.add(r["question"])
                unique_records.append(r)
            else:
                logger.warning(f"Skipping duplicate question query: '{r['question']}'")

        return unique_records

    def run_evaluation(
        self,
        dataset: List[Dict[str, str]],
        dataset_name: str = "custom_dataset"
    ) -> Dict[str, Any]:
        """
        Executes sequence of test questions, logs execution variables, and updates metrics.
        """
        if not dataset:
            raise ValueError("Cannot run evaluation: dataset record list is empty.")

        self.metrics_collector.reset()
        logger.info(f"Running evaluation benchmark suite '{dataset_name}' containing {len(dataset)} items.")
        
        detail_records = []

        for idx, item in enumerate(dataset):
            q = item["question"]
            expected = item.get("expected_answer")
            logger.info(f"[{idx+1}/{len(dataset)}] Evaluating query: '{q}'")
            
            try:
                # Execute pipeline
                answer = self.rag_pipeline.answer_question(q)
                
                # Fetch RAG metadata
                metadata = self.rag_pipeline.get_last_run_metadata()
                
                # Extract retrieved text segments to evaluate quality
                chunks_data = metadata.get("retrieval_inspector", {}).get("retrieved_chunks", []) if metadata.get("retrieval_inspector") else []
                retrieved_contexts = [chk.get("content", "") for chk in chunks_data]
                
                # Compute quality indexes
                quality = self.eval_engine.evaluate_response(
                    query=q,
                    response=answer,
                    retrieved_contexts=retrieved_contexts,
                    expected_answer=expected
                )
                
                # Record metrics
                self.metrics_collector.record(
                    rag_metadata=metadata,
                    quality_metrics=quality,
                    is_failure=False
                )
                
                # Append details
                detail_records.append({
                    "question": q,
                    "generated_answer": answer,
                    "expected_answer": expected or "",
                    "quality_scores": quality,
                    "metadata": metadata
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate question '{q}': {e}")
                # Record failure
                self.metrics_collector.record(
                    rag_metadata={},
                    quality_metrics={},
                    is_failure=True
                )
                detail_records.append({
                    "question": q,
                    "generated_answer": "",
                    "expected_answer": expected or "",
                    "error": str(e),
                    "quality_scores": {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0, "context_recall": 0.0},
                    "metadata": {}
                })

        # Calculate summaries
        aggregates = self.metrics_collector.compute_aggregates()
        
        # Save to history file
        self._append_to_history(dataset_name, len(dataset), aggregates)

        return {
            "dataset_name": dataset_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "summary": aggregates,
            "queries": detail_records
        }

    def _append_to_history(self, name: str, size: int, summary: Dict[str, Any]) -> None:
        """
        Saves historical run details to local JSON log.
        """
        history_list = []
        if os.path.exists(HISTORY_LOG_FILE):
            try:
                with open(HISTORY_LOG_FILE, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read evaluation history log: {e}")

        # Create new history record
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_name": name,
            "number_of_queries": size,
            "average_latency": summary["average_end_to_end_latency"],
            "average_confidence": summary["confidence_distribution"],
            "average_similarity": summary["average_similarity_score"],
            "retrieval_success_rate": summary["retrieval_success_rate"],
            "cache_hit_ratio": summary["cache_hit_percentage"],
            "quality_faithfulness": summary["quality"]["faithfulness"],
            "quality_relevance": summary["quality"]["answer_relevance"]
        }
        
        history_list.append(record)
        os.makedirs(os.path.dirname(HISTORY_LOG_FILE), exist_ok=True)
        
        try:
            with open(HISTORY_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(history_list, f, indent=2)
            logger.info("Evaluation run written to historical summary log.")
        except Exception as e:
            logger.error(f"Failed to write evaluation history log: {e}")

    @staticmethod
    def get_evaluation_history() -> List[Dict[str, Any]]:
        """
        Fetches full historical runs log.
        """
        if not os.path.exists(HISTORY_LOG_FILE):
            return []
        try:
            with open(HISTORY_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load evaluation logs: {e}")
            return []
