import string
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional

class BaseQueryCache(ABC):
    @abstractmethod
    def get(self, query: str) -> Optional[Any]:
        """Retrieves a cached value for the given query. Returns None if cache miss."""
        pass

    @abstractmethod
    def set(self, query: str, value: Any) -> None:
        """Saves a value in the cache for the given query."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all entries in the cache."""
        pass


class InMemoryLRUCache(BaseQueryCache):
    """InMemoryLRUCache implements a local Least-Recently-Used cache for query results.
    
    NOTE: In a production environment, this cache layer should be swapped out
    for a persistent and distributed key-value store such as Redis.
    """
    def __init__(self, maxsize: int = 100) -> None:
        self.maxsize = maxsize
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def _normalize(self, query: str) -> str:
        """Normalizes query string: lowercase, stripped, punctuation removed."""
        # Convert to lowercase and strip whitespaces
        q = query.lower().strip()
        # Remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        q_clean = q.translate(translator)
        # Collapse multiple spaces into one
        return " ".join(q_clean.split())

    def get(self, query: str) -> Optional[Any]:
        norm_q = self._normalize(query)
        if norm_q in self.cache:
            # Move accessed key to the end to maintain LRU order
            val = self.cache.pop(norm_q)
            self.cache[norm_q] = val
            return val
        return None

    def set(self, query: str, value: Any) -> None:
        norm_q = self._normalize(query)
        if norm_q in self.cache:
            self.cache.pop(norm_q)
        elif len(self.cache) >= self.maxsize:
            # Pop the oldest (first) item (least recently used)
            self.cache.popitem(last=False)
        self.cache[norm_q] = value

    def clear(self) -> None:
        self.cache.clear()
