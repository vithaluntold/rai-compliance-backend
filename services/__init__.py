"""Services package for AUDRIC backend."""

from services.vector_store import FAISS_AVAILABLE, VectorStore, get_vector_store

__all__ = [
    "get_vector_store",
    "VectorStore",
    "FAISS_AVAILABLE",
]
