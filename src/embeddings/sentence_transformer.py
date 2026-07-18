"""
Sentence Transformers embedding provider implementing BaseEmbeddingProvider.
Supports lazy loading, hardware autodetect, fallback options, and batch stats tracking.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.exceptions import ModelLoadError, EmbeddingGenerationError
from src.logger import setup_logger

logger = setup_logger("sentence_transformer_provider")

class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Sentence Transformers wrapper.
    Lazy loads model resources on first embedding call.
    """
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_folder: Optional[str] = None
    ):
        """
        Args:
            model_name: Sentence-Transformers model ID.
            device: 'cpu', 'cuda', or None for auto-detect.
            batch_size: Configured batch size.
            cache_folder: Custom model cache download folder.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_folder = cache_folder
        self.device = self._detect_device(device)

        # Lazy-loaded attributes
        self._model = None
        self._dimension: Optional[int] = None
        self._model_loading_time: float = 0.0

        # Stats counters
        self._total_embedded_documents: int = 0
        self._total_generation_time: float = 0.0

    def _detect_device(self, requested_device: Optional[str]) -> str:
        """
        Detects hardware option. Auto-falls back to CPU on error.
        """
        try:
            import torch
            available_cuda = torch.cuda.is_available()
        except ImportError:
            available_cuda = False

        if requested_device is None:
            device = "cuda" if available_cuda else "cpu"
            logger.info(f"Auto-selected device: {device}")
            return device

        req_clean = requested_device.lower().strip()
        if req_clean == "cuda":
            if not available_cuda:
                logger.warning("CUDA device was requested but PyTorch CUDA support is not available. Falling back to CPU.")
                return "cpu"
            return "cuda"
        
        return req_clean

    def _load_model(self) -> None:
        """
        Lazy loader method. Instantiates Model resources once.
        """
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Initiating lazy load for model '{self.model_name}' on device '{self.device}'.")
        start = time.time()
        try:
            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
                cache_folder=self.cache_folder
            )
            self._model_loading_time = time.time() - start
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully in {self._model_loading_time:.3f}s. Vector dimension={self._dimension}")
        except Exception as e:
            self._model = None
            logger.error(f"Failed to load sentence-transformers model '{self.model_name}': {e}")
            raise ModelLoadError(
                f"Failed to download or load model '{self.model_name}'. Check model name/connection. Details: {e}"
            )

    def embed_documents(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        """
        Converts text list into vector lists.
        """
        if not texts:
            return []

        # Ensure model is initialized
        self._load_model()

        start_time = time.time()
        try:
            logger.info(f"Generating embeddings for {len(texts)} documents (batch size={self.batch_size}).")
            
            # Run model inference
            # encode returns a numpy array
            embeddings = self._model.encode(
                sentences=texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings
            )
            
            generation_time = time.time() - start_time
            self._total_embedded_documents += len(texts)
            self._total_generation_time += generation_time
            
            avg_time = (generation_time / len(texts)) if texts else 0.0
            logger.info(
                f"Generated embeddings. Generation time={generation_time:.3f}s. "
                f"Average time per document={avg_time:.4f}s."
            )
            
            return embeddings.tolist()
            
        except RuntimeError as re:
            # Catch out-of-memory or PyTorch specific execution failures
            if "out of memory" in str(re).lower():
                logger.error("PyTorch CUDA Out-Of-Memory error detected during embedding.")
                raise EmbeddingGenerationError("GPU memory limit exceeded during embedding batch pass.")
            logger.error(f"PyTorch runtime execution failed: {re}")
            raise EmbeddingGenerationError(f"Embedding execution failed: {re}")
        except Exception as e:
            logger.error(f"Unexpected embedding generation failure: {e}")
            raise EmbeddingGenerationError(f"Unexpected generation failure: {e}")

    def embed_query(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        """
        Converts a single search query into vector.
        """
        res = self.embed_documents([text], normalize_embeddings=normalize_embeddings)
        return res[0]

    def get_dimension(self) -> int:
        """
        Returns dimension of the model.
        """
        self._load_model()
        return self._dimension or 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Exposes usage and performance metrics.
        """
        avg_time = 0.0
        if self._total_embedded_documents > 0:
            avg_time = self._total_generation_time / self._total_embedded_documents

        return {
            "embedding_dimension": self.get_dimension(),
            "number_of_embedded_documents": self._total_embedded_documents,
            "processing_time": self._total_generation_time,
            "batch_size_used": self.batch_size,
            "model_loading_time": self._model_loading_time,
            "avg_embedding_time_per_document": avg_time,
            "device_used": self.device,
            "model_name": self.model_name
        }
