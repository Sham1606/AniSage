"""
Embedding model abstraction for Phase 2.

Two backends:
  - SentenceTransformerModel  : local, free, no API key, 384-dim (default)
  - OpenAIEmbeddingModel      : cloud, requires OPENAI_API_KEY, 1536-dim

Both expose the same interface:
    model = SentenceTransformerModel()
    vectors = model.embed(["text1", "text2"])  # np.ndarray shape (N, dim)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

# ── Base class ────────────────────────────────────────────────────────────────

class EmbeddingModel(ABC):
    """Abstract base — all embedding backends implement this interface."""

    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        Returns np.ndarray of shape (len(texts), dim), dtype float32, L2-normalised.
        """
        ...

    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns 1-D array of shape (dim,)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.name}, dim={self.dim})"


# ── Sentence Transformers (local, default) ────────────────────────────────────

class SentenceTransformerModel(EmbeddingModel):
    """
    Local embedding model via HuggingFace sentence-transformers.

    Default: all-MiniLM-L6-v2
      - 384 dimensions
      - ~80 MB download on first run (cached in ~/.cache/huggingface)
      - ~14k docs in ~2 minutes on CPU

    Other good options:
      - all-mpnet-base-v2       (768 dim, higher quality, slower)
      - paraphrase-MiniLM-L3-v2 (384 dim, fastest, slightly lower quality)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run:\n"
                "  pip install sentence-transformers"
            )

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        sentence-transformers returns normalised vectors by default
        when normalize_embeddings=True.
        """
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ── OpenAI Embeddings (cloud, optional) ───────────────────────────────────────

class OpenAIEmbeddingModel(EmbeddingModel):
    """
    OpenAI embedding via text-embedding-3-small.

    - 1536 dimensions
    - Requires OPENAI_API_KEY env var
    - Costs ~$0.02 per 1M tokens (~$0.05 for full 14k dataset)
    - Higher quality than MiniLM for nuanced queries

    Usage:
        model = OpenAIEmbeddingModel()   # reads OPENAI_API_KEY from env
        # OR
        model = OpenAIEmbeddingModel(api_key="sk-...")
    """

    DIM = 1536

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Run:\n"
                "  pip install openai"
            )

        self.name = model_name
        self.dim = self.DIM
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set. Either pass api_key= or set the env var."
            )
        self._client = OpenAI(api_key=key)  # type: ignore

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts via OpenAI API.
        OpenAI text-embedding-3-small returns normalised vectors.
        Max batch size: 2048 inputs (we chunk internally if needed).
        """
        # OpenAI has a 2048-input batch limit
        BATCH_LIMIT = 2048
        all_vectors: list[np.ndarray] = []

        for i in range(0, len(texts), BATCH_LIMIT):
            chunk = texts[i : i + BATCH_LIMIT]
            # Replace newlines — OpenAI recommendation for best quality
            chunk = [t.replace("\n", " ") for t in chunk]
            response = self._client.embeddings.create(
                input=chunk,
                model=self.name,
            )
            batch_vectors = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            # L2-normalise (OpenAI vectors are already unit-norm, but be explicit)
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            batch_vectors = batch_vectors / np.maximum(norms, 1e-10)
            all_vectors.append(batch_vectors)

        return np.vstack(all_vectors)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def estimate_cost(self, texts: list[str]) -> float:
        """
        Rough cost estimate in USD for embedding all texts.
        text-embedding-3-small: $0.02 per 1M tokens
        Approximation: 1 token ≈ 4 chars
        """
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars / 4
        cost = (estimated_tokens / 1_000_000) * 0.02
        return round(cost, 4)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedding_model(
    model_type: str = "sentence-transformer",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EmbeddingModel:
    """
    Factory function — returns the right model based on model_type.

    Args:
        model_type:  "sentence-transformer" | "openai"
        model_name:  Override the default model name
        api_key:     OpenAI API key (only for openai type)
    """
    if model_type == "sentence-transformer":
        name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerModel(model_name=name)
    elif model_type == "openai":
        name = model_name or "text-embedding-3-small"
        return OpenAIEmbeddingModel(model_name=name, api_key=api_key)
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Choose 'sentence-transformer' or 'openai'."
        )