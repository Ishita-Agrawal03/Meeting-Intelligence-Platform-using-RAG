"""
FAISS wrapper used by the live app.

  - IndexIDMap so the id we pass in when adding IS the chunk's
    SQLite id — no separate translation table needed.
  - Saved to disk after every write, so vectors survive a restart.
  - search() and add() both take the lock — a real race existed
    before this fix: add() was protected, search() was not, so a
    concurrent upload and chat request could read the index mid-write.
"""
import threading
from pathlib import Path
import faiss
import numpy as np

from app.services.embeddings import EMBEDDING_DIM

INDEX_PATH = Path("vector_store") / "index.faiss"
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


class FaissStore:
    def __init__(self, index_path: Path = INDEX_PATH, dim: int = EMBEDDING_DIM):
        self.index_path = index_path
        self.dim = dim
        self._lock = threading.RLock()
        self.index = self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            # A stale index built with a different embedding model
            # (different dimension) would otherwise load silently and
            # fail much later, confusingly, inside a search() or add()
            # call far from the real cause. Fail loudly here instead.
            if index.d != self.dim:
                raise ValueError(
                    f"Loaded FAISS index has dimension {index.d}, but the "
                    f"configured embedding dimension is {self.dim}. The "
                    f"index on disk was likely built with a different "
                    f"embedding model. Delete {self.index_path} and "
                    f"re-upload to rebuild it."
                )
            return index
        base = faiss.IndexFlatL2(self.dim)
        return faiss.IndexIDMap(base)

    def save(self):
        with self._lock:
            faiss.write_index(self.index, str(self.index_path))

    def add(self, ids: list[int], vectors: np.ndarray):
        assert len(ids) == vectors.shape[0], "ids and vectors must align 1:1"
        ids_arr = np.array(ids, dtype="int64")
        with self._lock:
            self.index.add_with_ids(vectors, ids_arr)
            self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Returns list of (chunk_id, distance), best match first."""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            query_vector = query_vector.reshape(1, -1)
            distances, ids = self.index.search(query_vector, top_k)

        results = []
        for chunk_id, dist in zip(ids[0], distances[0]):
            if chunk_id == -1:
                continue
            results.append((int(chunk_id), float(dist)))
        return results


_store: FaissStore | None = None


def get_faiss_store() -> FaissStore:
    global _store
    if _store is None:
        _store = FaissStore()
    return _store