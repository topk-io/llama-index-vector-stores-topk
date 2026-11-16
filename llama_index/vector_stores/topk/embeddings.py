"""Embedding helpers for the TopK vector store."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

MultiVector = List[List[float]]
"""A multi-vector (matrix): one row per token, all rows of the same width."""


class TopKServerSideEmbedding(BaseEmbedding):
    """
    Placeholder embedding model for TopK's server-side semantic search.

    TopK embeds documents and queries on the server, so no embedding model
    (and no embedding API key) is needed on the client. `VectorStoreIndex`
    still asks for an `embed_model`; pass this one so that LlamaIndex does not
    fall back to its default (OpenAI) model. It makes no network calls and
    returns empty vectors, which `TopKVectorStore` ignores.

    Examples:
        ```python
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.topk import (
            TopKServerSideEmbedding,
            TopKVectorStore,
        )

        vector_store = TopKVectorStore(collection_name="docs")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=TopKServerSideEmbedding(),
        )
        ```

    """

    model_name: str = "topk-server-side"

    @classmethod
    def class_name(cls) -> str:
        return "TopKServerSideEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        return []

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return []

    def _get_text_embedding(self, text: str) -> Embedding:
        return []

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return [[] for _ in texts]


class BaseMultiVectorEmbedding(ABC):
    """
    Interface for late-interaction (multi-vector) models such as ColBERT.

    LlamaIndex has no built-in interface for multi-vector embeddings. Implement
    this one to plug any late-interaction model into `TopKVectorStore`, which
    indexes the vectors with a MaxSim multi-vector index.
    """

    @abstractmethod
    def get_text_embedding_batch(self, texts: List[str]) -> List[MultiVector]:
        """Embed documents into multi-vectors (one row per token)."""

    @abstractmethod
    def get_query_embedding(self, query: str) -> MultiVector:
        """Embed a query into a multi-vector (one row per token)."""

    async def aget_text_embedding_batch(self, texts: List[str]) -> List[MultiVector]:
        """Asynchronously embed documents. Runs the sync method in a thread."""
        return await asyncio.to_thread(self.get_text_embedding_batch, texts)

    async def aget_query_embedding(self, query: str) -> MultiVector:
        """Asynchronously embed a query. Runs the sync method in a thread."""
        return await asyncio.to_thread(self.get_query_embedding, query)


class FastEmbedMultiVectorEmbedding(BaseMultiVectorEmbedding):
    """
    Late-interaction embeddings computed locally with FastEmbed.

    The default model is `answerdotai/answerai-colbert-small-v1` (96
    dimensions). Requires `pip install fastembed`.

    Examples:
        ```python
        from llama_index.vector_stores.topk import FastEmbedMultiVectorEmbedding

        colbert = FastEmbedMultiVectorEmbedding()
        matrix = colbert.get_query_embedding("memory safe systems programming")
        ```

    """

    def __init__(
        self,
        model_name: str = "answerdotai/answerai-colbert-small-v1",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        try:
            from fastembed import LateInteractionTextEmbedding
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "FastEmbedMultiVectorEmbedding requires `fastembed`. Install it "
                "with `pip install fastembed`."
            ) from e
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = LateInteractionTextEmbedding(model_name=model_name, **kwargs)

    @staticmethod
    def _rows(matrix: Any) -> MultiVector:
        return [[float(x) for x in row] for row in matrix]

    def get_text_embedding_batch(self, texts: List[str]) -> List[MultiVector]:
        return [
            self._rows(m) for m in self._model.embed(texts, batch_size=self.batch_size)
        ]

    def get_query_embedding(self, query: str) -> MultiVector:
        return self._rows(next(iter(self._model.query_embed(query))))


def normalize_multi_vector(matrix: Any) -> MultiVector:
    """Coerce a multi-vector (list of rows or 2-D array) into a list of rows."""
    rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    out = [[float(x) for x in row] for row in rows]
    if not out:
        raise ValueError("Multi-vector embeddings must have at least one row.")
    return out


def normalize_sparse(vector: Any) -> dict:
    """Coerce a sparse vector into a `{index: value}` dict."""
    if isinstance(vector, dict):
        return {int(k): float(v) for k, v in vector.items()}
    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)
    if indices is not None and values is not None:
        return {int(i): float(v) for i, v in zip(indices, values, strict=True)}
    raise TypeError(
        "Sparse embeddings must be dicts of {index: value} or objects with "
        f"`indices` and `values`, got {type(vector).__name__}."
    )


__all__ = [
    "BaseMultiVectorEmbedding",
    "FastEmbedMultiVectorEmbedding",
    "MultiVector",
    "TopKServerSideEmbedding",
]
