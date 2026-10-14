"""Test doubles shared by the storage and RAG tests. No network, no API calls."""

import hashlib
import json
import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec

import numpy as np
from anthropic.resources.messages import Messages

DIMENSION = 384


def message(payload: dict[str, Any] | str, stop_reason: str = "end_turn") -> SimpleNamespace:
    """A stand-in for the SDK's Message, carrying a JSON reply."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        model="claude-haiku-4-5-20251001",
    )


def mock_client(*outcomes: Any) -> MagicMock:
    """A client whose messages.create (autospecced from the SDK) returns each outcome in turn."""
    client = MagicMock()
    client.messages = create_autospec(Messages, instance=True)
    client.messages.create.side_effect = list(outcomes)
    return client


class HashEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words are close."""

    dimension = DIMENSION

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), DIMENSION), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.casefold()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSION] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
