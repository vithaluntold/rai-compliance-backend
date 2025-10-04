import asyncio
import json
import logging
import os
import random
import string
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss  # Import FAISS at the module level
import numpy as np
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# Azure OpenAI configuration for embeddings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv(
    "AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15"
)


def generate_document_id() -> str:
    """Generate a document ID in the format RAI-DDMMYYYY-XXXXX-XXXXX."""
    now = datetime.now()
    date_part = now.strftime("%d%m%Y")

    # Generate two random 5-character alphanumeric strings
    def random_string(length=5):
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

    return f"RAI-{date_part}-{random_string()}-{random_string()}"


# Initialize FAISS flag with default
FAISS_AVAILABLE = False

# Global vector store instance with proper typing
_vector_store: Optional["VectorStore"] = None

# Optimize worker count for vector operations
NUM_WORKERS = min(32, (os.cpu_count() or 1) * 4)
CHUNK_SIZE = 20  # Increased chunk size for vector operations
MAX_RETRIES = 3  # Number of retries for API calls


# Simple similarity computation for fallback when FAISS isn't available
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)


try:
    # FAISS already imported at module level - check availability
    FAISS_AVAILABLE = True
    logger.info("FAISS CPU support enabled")
except ImportError:
    logger.warning("FAISS not available, falling back to numpy")
except Exception as e:
    logger.error(f"Error initializing FAISS: {str(e)}")

# Use the provided Azure OpenAI embedding endpoint and key
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")


class VectorStore:
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        embedding_deployment: str,
        embedding_api_version: str,
    ):
        if not api_key or not azure_endpoint or not embedding_deployment:
            raise ValueError("Azure OpenAI embedding configuration missing")
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.deployment_id = embedding_deployment
        self.api_version = embedding_api_version
        self.ai_client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=embedding_api_version,
        )
        self.index_cache: Dict[str, Tuple[faiss.Index, List[Dict[str, Any]]]] = {}
        self.index_dir = Path("vector_indices")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = Path("exported_dataset")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = 1536  # text-embedding-ada-002 dimension
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        logger.info(f"VectorStore (Azure) initialized with {NUM_WORKERS} workers")

    def _process_chunk(
        self, chunk: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Process a single chunk with retries."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.ai_client.embeddings.create(
                    model=self.deployment_id, input=chunk["text"]
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return embedding, chunk
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(
                        f"Failed to process chunk after {MAX_RETRIES} attempts: "
                        f"{str(e)}"
                    )
                    return None
                time.sleep(1)  # Wait before retry

    async def create_index(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> bool:
        """Create an index for a document using parallel processing."""
        try:
            logger.info(f"Starting parallel index creation for document {document_id}")
            embeddings = []
            valid_chunks = []
            loop = asyncio.get_running_loop()
            for i in range(0, len(chunks), CHUNK_SIZE):
                batch = chunks[i : i + CHUNK_SIZE]
                tasks = [
                    loop.run_in_executor(None, self._process_chunk, chunk)
                    for chunk in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, tuple) and not isinstance(result, Exception):
                        embedding, chunk = result
                        embeddings.append(embedding)
                        valid_chunks.append(chunk)
            if not embeddings:
                raise ValueError("No valid embeddings generated")
            embeddings_array = np.vstack(embeddings)
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)  # type: ignore[call-arg]
            index_path = self.index_dir / f"{document_id}_index.faiss"
            chunks_path = self.index_dir / f"{document_id}_chunks.json"
            try:
                faiss.write_index(index, str(index_path))
            except Exception as save_error:
                logger.error(f"Error saving index: {str(save_error)}")
                raise
            chunks_with_timestamp = [
                {**chunk, "processed_at": datetime.now().isoformat()}
                for chunk in valid_chunks
            ]
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks_with_timestamp, f, ensure_ascii=False, indent=2)
            self.index_cache[document_id] = (index, valid_chunks)
            logger.info(f"Index created successfully for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load_index(
        self, document_id: str
    ) -> Optional[Tuple[faiss.Index, List[Dict[str, Any]]]]:
        """Load a saved index and its chunks."""
        try:
            index_path = self.index_dir / f"{document_id}_index.faiss"
            chunks_path = self.index_dir / f"{document_id}_chunks.json"

            if not index_path.exists() or not chunks_path.exists():
                logger.warning(f"Index files not found for document {document_id}")
                return None

            # Load FAISS index
            index = faiss.read_index(str(index_path))

            # Load chunks
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            return index, chunks

        except Exception as e:
            logger.error(f"Error loading index for document {document_id}: {str(e)}")
            return None

    def index_exists(self, document_id: str) -> bool:
        """Check if an index exists for a document."""
        index_path = self.index_dir / f"{document_id}_index.faiss"
        chunks_path = self.index_dir / f"{document_id}_chunks.json"
        return index_path.exists() and chunks_path.exists()

    def search(
        self, query: str, document_id: str, top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for relevant chunks in a document's vector index."""
        try:
            # Load the index if not in cache
            if document_id not in self.index_cache:
                loaded = self.load_index(document_id)
                if not loaded:
                    logger.warning(f"Index files not found for document {document_id}")
                    return []
                self.index_cache[document_id] = loaded

            index, chunks = self.index_cache[document_id]

            # Get query embedding
            query_response = self.ai_client.embeddings.create(
                model=self.deployment_id, input=query
            )
            query_embedding = np.array(
                [query_response.data[0].embedding], dtype=np.float32
            )

            # Search the index
            distances, indices = index.search(query_embedding, top_k)  # type: ignore[call-arg]

            # Get the matching chunks with page information
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(chunks):  # Ensure index is valid
                    chunk = chunks[idx]
                    results.append(
                        {
                            "text": chunk["text"],
                            "score": float(
                                1.0 / (1.0 + distance)
                            ),  # Convert distance to similarity score
                            "page_number": chunk.get("page_no", chunk.get("page", 0)),  # Extract page number
                            "chunk_index": chunk.get("chunk_index", idx),
                            "chunk_type": chunk.get("chunk_type", "content"),
                            "metadata": chunk.get("metadata", {}),
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def delete_index(self, document_id: str) -> None:
        """Delete index files for a document."""
        index_path = self.index_dir / f"{document_id}_index.faiss"
        chunks_path = self.index_dir / f"{document_id}_chunks.json"
        embeddings_path = self.index_dir / f"{document_id}_embeddings.npy"
        export_path = self.export_dir / f"{document_id}_dataset.json"

        for path in [index_path, chunks_path, embeddings_path, export_path]:
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"Deleted {path}")
                except Exception as e:
                    logger.error(f"Error deleting {path}: {str(e)}")

        logger.info(f"Deleted index for document {document_id}")

    def get_chunks(self, document_id: str) -> list:
        """Return all chunks for a given document_id from cache or file."""
        # Try cache first
        if document_id in self.index_cache:
            _, chunks = self.index_cache[document_id]
            return chunks
        # Try loading from file
        chunks_path = self.index_dir / f"{document_id}_chunks.json"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"No chunks found for document_id: {document_id}")
        return []


def init_vector_store(
    api_key: str, api_base: str, deployment_id: str, api_version: str
) -> "VectorStore":
    """Initialize a new VectorStore instance.

    Args:
        api_key: Azure OpenAI API key
        api_base: Azure OpenAI API base URL
        deployment_id: Azure OpenAI deployment ID
        api_version: Azure OpenAI API version

    Returns:
        VectorStore: A new vector store instance
    """
    global _vector_store
    _vector_store = VectorStore(api_key, api_base, deployment_id, api_version)
    return _vector_store


def get_vector_store() -> "VectorStore":
    """Get or create a singleton instance of VectorStore.

    Returns:
        VectorStore: The singleton vector store instance
    """
    global _vector_store
    if _vector_store is None:
        api_key = AZURE_OPENAI_API_KEY
        api_base = AZURE_OPENAI_ENDPOINT
        deployment_id = AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        api_version = AZURE_OPENAI_EMBEDDING_API_VERSION
        if not api_key or not api_base or not deployment_id or not api_version:
            raise ValueError("Azure OpenAI configuration missing")
        _vector_store = VectorStore(api_key, api_base, deployment_id, api_version)
        logger.info("Vector store initialized with Azure configuration")
    return _vector_store


# Export the functions and classes
__all__ = [
    "init_vector_store",
    "get_vector_store",
    "VectorStore",
    "FAISS_AVAILABLE",
    "generate_document_id",
]
