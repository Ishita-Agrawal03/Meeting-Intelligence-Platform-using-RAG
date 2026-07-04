from functools import lru_cache
from sentence_transformers import SentenceTransformer

# PINNED. Changing this makes every existing embedding incompatible
# with new ones — you'd have to re-embed every chunk already stored.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match the model above


@lru_cache(maxsize=1)
def _get_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embedding(text: str):
    """Returns a 1D float32 numpy array of length EMBEDDING_DIM."""
    model = _get_model()
    vector = model.encode(text)
    return vector.astype("float32")


def get_embeddings(texts: list[str]):
    """Returns a 2D float32 numpy array, shape (len(texts), EMBEDDING_DIM)."""
    model = _get_model()
    vectors = model.encode(texts)
    return vectors.astype("float32")