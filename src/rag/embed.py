"""Embed chunks and queries with a local sentence-transformers model.

Embedding runs on this machine: no API cost, and it works offline once the model has
been downloaded (the first use fetches it from Hugging Face and caches it). The model
is set by EMBEDDING_MODEL; its output size must match the `vector(...)` column in
src/storage/schema.sql.
"""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that turns text into unit-length vectors."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """A sentence-transformers model, loaded on first use.

    Vectors are L2-normalised, so cosine similarity is a dot product and pgvector's
    cosine distance (`<=>`) ranks them correctly.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: "SentenceTransformer | None" = None

    def _load(self) -> "SentenceTransformer":
        if self._model is None:
            # Imported lazily: loading torch takes seconds and isn't needed to import this module.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._load().encode(list(texts), normalize_embeddings=True), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def load_embedder() -> SentenceTransformerEmbedder:
    """The embedder named by EMBEDDING_MODEL (environment or .env), or the default."""
    load_dotenv()
    return SentenceTransformerEmbedder(os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL)
