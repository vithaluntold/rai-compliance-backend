"""Services package for AUDRIC backend."""

# Document chunker has been replaced by NLP pipeline in nlp_tools/
from services.vector_store import FAISS_AVAILABLE, VectorStore, get_vector_store

__all__ = [
    "get_vector_store",
    "VectorStore",
    "FAISS_AVAILABLE",
]
