"""
Report Generator module.
Compiles evaluation summaries and exports reports in CSV, JSON, Markdown, and HTML.
"""

import json
import csv
import io
import datetime
from typing import Dict, Any, List

class ReportGenerator:
    """
    Compiles evaluation benchmark summaries and formats reports for download/display.
    """
    @staticmethod
    def generate_markdown_report(result: Dict[str, Any]) -> str:
        """
        Creates a clean Markdown evaluation report.
        """
        summary = result["summary"]
        quality = summary["quality"]
        conf = summary["confidence_distribution"]
        
        md = f"""# RAG System Evaluation & Benchmark Report
**Dataset:** `{result['dataset_name']}`
**Timestamp:** `{result['timestamp']}`
**Total Queries Evaluated:** `{summary['total_queries']}`

---

## 📈 Operational Performance Summary

| Metric Name | Value | Description |
| :--- | :---: | :--- |
| **Retrieval Success Rate** | `{summary['retrieval_success_rate']:.1f}%` | Percentage of queries returning 1+ context chunks |
| **Average E2E Latency** | `{summary['average_end_to_end_latency']:.3f}s` | Total latency from query submission to output |
| **Average Retrieval Latency** | `{summary['average_retrieval_latency']:.3f}s` | Latency spent searching vector store |
| **Average Generation Latency** | `{summary['average_generation_latency']:.3f}s` | Latency spent in LLM text generation |
| **Cache Hit Ratio** | `{summary['cache_hit_percentage']:.1f}%` | Percentage of queries answered from retrieval cache |
| **LLM Failure Percentage** | `{summary['llm_failure_percentage']:.1f}%` | Percentage of queries that raised provider exceptions |
| **Average Retrieved Chunks** | `{summary['average_retrieved_chunks']:.1f}` | Average context chunks fetched per search query |
| **Average Citation Count** | `{summary['average_citations']:.1f}` | Average unique references appended to answer |
| **Average Similarity Score** | `{summary['average_similarity_score']:.4f}` | Mean score (distance/cosine) of returned chunks |

---

## 📊 Quality Overlap Indicators (Heuristic Proxies)

- **Faithfulness (Groundedness):** `{quality['faithfulness'] * 100:.1f}%` *(Measures alignment of answer terms with source terms to identify hallucinations)*
- **Answer Relevancy:** `{quality['answer_relevance'] * 100:.1f}%` *(Measures direct correspondence of query terms inside answer)*
- **Context Precision:** `{quality['context_precision'] * 100:.1f}%` *(Proportion of retrieved chunks matching query words)*
- **Context Recall:** `{quality['context_recall'] * 100:.1f}%` *(Matches expected answers against retrieved context chunks)*

---

## 🛡️ Retrieval Confidence Mappings

- **High Confidence:** `{conf.get('High', 0)}` queries
- **Medium Confidence:** `{conf.get('Medium', 0)}` queries
- **Low Confidence:** `{conf.get('Low', 0)}` queries

---

## 📁 Detailed Query List
"""
        # Append query details
        for i, query_item in enumerate(result["queries"]):
            md += f"\n### Query #{i+1}: *\"{query_item['question']}\"*\n"
            if "error" in query_item:
                md += f"❌ **Error occurred during generation:** `{query_item['error']}`\n"
            else:
                md += f"- **Generated Answer:** {query_item['generated_answer']}\n"
                md += f"- **Expected Answer:** {query_item.get('expected_answer') or 'N/A'}\n"
                qs = query_item["quality_scores"]
                md += f"- **Scores:** Faithfulness: `{qs['faithfulness']:.2f}`, Relevancy: `{qs['answer_relevance']:.2f}`, Context Precision: `{qs['context_precision']:.2f}`\n"
                
        return md

    @staticmethod
    def generate_csv_report(result: Dict[str, Any]) -> str:
        """
        Compiles CSV details for spreadsheets.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(["Question", "Generated Answer", "Expected Answer", "Faithfulness", "Answer Relevance", "Context Precision", "Context Recall", "End-to-End Latency (s)", "Retrieved Chunks"])
        
        # Write rows
        for item in result["queries"]:
            qs = item.get("quality_scores", {})
            meta = item.get("metadata", {})
            writer.writerow([
                item["question"],
                item.get("generated_answer", ""),
                item.get("expected_answer", ""),
                qs.get("faithfulness", 0.0),
                qs.get("answer_relevance", 0.0),
                qs.get("context_precision", 0.0),
                qs.get("context_recall", 0.0),
                meta.get("latency_seconds", 0.0),
                meta.get("retrieved_count", 0)
            ])
            
        return output.getvalue()

    @staticmethod
    def generate_html_report(result: Dict[str, Any]) -> str:
        """
        Compiles an HTML page with responsive layouts.
        """
        summary = result["summary"]
        quality = summary["quality"]
        conf = summary["confidence_distribution"]
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Smart RAG Benchmark Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #F8FAFC; color: #1E293B; }}
        h1, h2 {{ color: #0F172A; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; text-align: center; }}
        .metric-val {{ font-size: 1.8rem; font-weight: bold; color: #0284C7; }}
        .metric-lbl {{ font-size: 0.8rem; color: #64748B; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        th, td {{ padding: 12px; border: 1px solid #E2E8F0; text-align: left; }}
        th {{ background-color: #F1F5F9; color: #475569; }}
        tr:nth-child(even) {{ background-color: #FFFFFF; }}
        .query-box {{ background: white; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
        .error {{ border-left: 4px solid #EF4444; }}
    </style>
</head>
<body>
    <h1>Smart Research Assistant - Benchmark Report</h1>
    <p><strong>Dataset Name:</strong> {result['dataset_name']} | <strong>Timestamp:</strong> {result['timestamp']}</p>
    
    <h2>Operational Performance</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-val">{summary['retrieval_success_rate']:.1f}%</div>
            <div class="metric-lbl">Retrieval Success</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{summary['average_end_to_end_latency']:.3f}s</div>
            <div class="metric-lbl">Avg E2E Latency</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{summary['cache_hit_percentage']:.1f}%</div>
            <div class="metric-lbl">Cache Hit Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{summary['average_similarity_score']:.4f}</div>
            <div class="metric-lbl">Avg Cosine Score</div>
        </div>
    </div>
    
    <h2>Quality Indicators (Heuristics)</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-val">{quality['faithfulness'] * 100:.1f}%</div>
            <div class="metric-lbl">Faithfulness</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{quality['answer_relevance'] * 100:.1f}%</div>
            <div class="metric-lbl">Answer Relevance</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{quality['context_precision'] * 100:.1f}%</div>
            <div class="metric-lbl">Context Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{quality['context_recall'] * 100:.1f}%</div>
            <div class="metric-lbl">Context Recall</div>
        </div>
    </div>

    <h2>Detailed Queries Run</h2>
"""
        for item in result["queries"]:
            if "error" in item:
                html += f"""
                <div class="query-box error">
                    <strong>Question:</strong> {item['question']}<br/>
                    <span style="color:#EF4444;"><strong>Error:</strong> {item['error']}</span>
                </div>
                """
            else:
                qs = item["quality_scores"]
                html += f"""
                <div class="query-box">
                    <strong>Question:</strong> {item['question']}<br/>
                    <strong>Answer:</strong> {item['generated_answer']}<br/>
                    <strong>Expected Answer:</strong> {item.get('expected_answer') or 'N/A'}<br/>
                    <small style="color:#64748B;">Faithfulness: {qs['faithfulness']:.2f} | Relevance: {qs['answer_relevance']:.2f}</small>
                </div>
                """
                
        html += """</body></html>"""
        return html
