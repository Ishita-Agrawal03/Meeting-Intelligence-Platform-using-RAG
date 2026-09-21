from functools import lru_cache
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match the model above


@lru_cache(maxsize=1)
def _get_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _validate_dim(vectors, expected_dim: int):
    """Catches a real, silent failure mode: if EMBEDDING_MODEL_NAME
    ever changes to a model with a different output size, or a bad
    encode() call returns something malformed, this fails loudly here
    instead of producing a cryptic FAISS dimension-mismatch error much
    later, far from the actual cause."""
    actual_dim = vectors.shape[-1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding model produced {actual_dim}-dim vectors, "
            f"expected {expected_dim}. Did EMBEDDING_MODEL_NAME change "
            f"without updating EMBEDDING_DIM?"
        )


def get_embedding(text: str):
    model = _get_model()
    vector = model.encode(text)
    vector = vector.astype("float32")
    _validate_dim(vector, EMBEDDING_DIM)
    return vector


def get_embeddings(texts: list[str]):
    model = _get_model()
    vectors = model.encode(texts)
    vectors = vectors.astype("float32")
    _validate_dim(vectors, EMBEDDING_DIM)
    return vectors