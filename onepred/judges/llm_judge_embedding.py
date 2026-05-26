"""Embedding-based similarity scoring using sentence-transformers.

Computes cosine similarity between prediction and ground truth embeddings
as a complementary metric to LLM-based judging.
"""

import os
import numpy as np

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda:0")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=EMBEDDING_DEVICE)
    return _model


def llm_judge_score(prediction_text: str, ground_truth: str, **kwargs) -> float:
    """Compute cosine similarity between prediction and ground truth embeddings.

    Returns a float in [0, 1].
    """
    if not prediction_text.strip() or not ground_truth.strip():
        return 0.0

    model = _get_model()
    embeddings = model.encode([prediction_text, ground_truth], normalize_embeddings=True)
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, similarity)
