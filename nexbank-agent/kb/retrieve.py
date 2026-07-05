import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

from kb.cache import InMemoryLRUCache

# Define Pydantic models for structured output
class ChunkResult(BaseModel):
    id: str
    title: str
    content: str
    category: str
    score: float


class RetrievalResult(BaseModel):
    chunks: List[ChunkResult]
    cache_hit: bool
    latency_ms: float


# Cached resources
_collection = None
_model = None
_cache = InMemoryLRUCache(maxsize=100)


def get_resources():
    """Initializes and caches the ChromaDB collection and sentence transformer model."""
    global _collection, _model
    if _collection is None or _model is None:
        chroma_dir = Path(__file__).parent / "chroma_store"
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_collection("nexbank_policy_kb")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _collection, _model


def retrieve(
    query: str,
    top_k: int = 3,
    product_type: Optional[str] = None,
    region: Optional[str] = None
) -> RetrievalResult:
    """Retrieves document chunks matching the query using embedding similarity.
    
    Checks in-memory LRU cache first. Filters metadata if parameters are specified.
    """
    # Start timer
    start_time = time.perf_counter()

    # Define unique cache key based on query, top_k, and metadata filters
    cache_key = f"{query}||top_k={top_k}||prod={product_type}||reg={region}"

    # Check cache first
    cached_val = _cache.get(cache_key)
    if cached_val is not None:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalResult(
            chunks=[ChunkResult(**chunk) for chunk in cached_val],
            cache_hit=True,
            latency_ms=latency_ms
        )

    # Cache miss: run similarity search in vector store
    collection, model = get_resources()

    # Embed query string
    query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0].tolist()

    # Build metadata filters (where clause)
    where_clause = {}
    filters = []
    if product_type is not None:
        filters.append({"product_type": product_type})
    if region is not None:
        filters.append({"region": region})

    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}
    else:
        where_clause = None

    # Query ChromaDB collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause
    )

    # Process query results
    chunks = []
    if results and "ids" in results and results["ids"]:
        # Chroma returns lists nested inside another list for batch queries
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        # Distances are cosine distances. Similarity score = 1.0 - cosine_distance.
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

        for i in range(len(ids)):
            score = float(1.0 - distances[i])
            meta = metadatas[i]
            chunks.append(
                ChunkResult(
                    id=ids[i],
                    title=meta.get("title", ""),
                    content=documents[i],
                    category=meta.get("category", ""),
                    score=score
                )
            )

    # Store in cache (store dict representation for easy serialization/copying)
    cache_value = [chunk.model_dump() for chunk in chunks]
    _cache.set(cache_key, cache_value)

    # End timer and compute latency
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return RetrievalResult(
        chunks=chunks,
        cache_hit=False,
        latency_ms=latency_ms
    )
