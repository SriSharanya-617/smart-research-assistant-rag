#!/usr/bin/env python
"""
Benchmark tool comparing BAAI/bge-small-en-v1.5 and all-MiniLM-L6-v2.
Measures dimension sizes, model loading times, generation latency, and document throughput.
"""

import time
import sys
from src.embeddings.factory import EmbeddingFactory

# Sample texts representing typical research papers / query lines
TEST_TEXTS = [
    "Retrieval-Augmented Generation (RAG) is an architectural pattern for optimizing LLM outputs.",
    "This system utilizes external knowledge bases to retrieve factual chunks before generation.",
    "Decoupling the storage layer from the model interface ensures high modularity.",
    "ChromaDB and FAISS are both supported vector store libraries in this project.",
    "Default embeddings are generated via Sentence Transformers on the local device.",
    "Pydantic Settings parses and validates configurations from environment variables.",
    "Evaluation metrics include faithfulness, answer relevance, and context recall.",
    "Streamlit orchestrates the user interface and runs interactive web dashboards.",
    "Automatic device detection checks if CUDA is available, falling back to CPU.",
    "SHA-256 hashing detects duplicate documents and skips processing to prevent overlaps."
]

def run_benchmark():
    print("=" * 70)
    print("      SMART RESEARCH ASSISTANT - EMBEDDING MODELS BENCHMARK")
    print("=" * 70)
    print(f"Device: Auto-detected")
    print(f"Test Corpus Size: {len(TEST_TEXTS)} documents")
    print("-" * 70)

    candidate_models = [
        ("sentence-transformers", "BAAI/bge-small-en-v1.5"),
        ("sentence-transformers", "all-MiniLM-L6-v2")
    ]

    for provider, model_name in candidate_models:
        print(f"\n[*] Evaluating: {model_name} ({provider})")
        
        try:
            # 1. Measure Loading Time
            start_load = time.time()
            # Calling get_embeddings instantiates the provider wrapper (lazy)
            embedder = EmbeddingFactory.get_embeddings(provider=provider, model_name=model_name)
            # Force lazy load by retrieving dimension
            dimension = embedder.get_dimension()
            load_time = time.time() - start_load
            
            print(f"    [+] Model Load Time: {load_time:.3f} seconds")
            print(f"    [+] Vector Dimension: {dimension}")
            
            # 2. Warm-up Run
            _ = embedder.embed_query("Warm-up text.")
            
            # 3. Measure Inferences (Single Queries)
            start_single = time.time()
            for text in TEST_TEXTS:
                _ = embedder.embed_query(text)
            single_duration = time.time() - start_single
            single_avg = single_duration / len(TEST_TEXTS)
            
            print(f"    [+] Single Query Latency: {single_duration:.3f}s total (Avg: {single_avg*1000:.2f}ms/doc)")
            
            # 4. Measure Batch Inference
            start_batch = time.time()
            _ = embedder.embed_documents(TEST_TEXTS)
            batch_duration = time.time() - start_batch
            batch_avg = batch_duration / len(TEST_TEXTS)
            throughput = len(TEST_TEXTS) / max(0.0001, batch_duration)
            
            print(f"    [+] Batch Latency (size={len(TEST_TEXTS)}): {batch_duration:.3f}s total (Avg: {batch_avg*1000:.2f}ms/doc)")
            print(f"    [+] Batch Throughput: {throughput:.2f} docs/sec")
            
            # 5. Extract Provider Stats
            stats = embedder.get_statistics()
            print(f"    [+] Active Device: {stats.get('device_used')}")
            
        except Exception as e:
            print(f"    [-] Benchmark failed for {model_name}: {e}")
            
    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
