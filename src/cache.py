from typing import Dict, Any, Optional
import os
import json
import hashlib


class Cache:
    def __init__(self, cache_dir: str, cache_file: str):
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_file)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_key(self, query: str, k: int, mode: str) -> str:
        """Creates a unique SHA-256 hash based on the
            query text, retrieval count and the search mode."""
        query = query.strip().lower()
        key = f"query:{query}|k:{k}|mode:{mode}"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def _read_cache(self) -> Optional[Dict[str, Any]]:
        """Reads the cache file from disk."""
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if cache is None:
                    return None
                if isinstance(cache, dict):
                    return dict(cache)
                return None
        except (json.JSONDecodeError, IOError):
            return None

    def get(self, query: str, k: int, mode: str) -> Optional[Dict[str, Any]]:
        """Gets the response of the query fron the cache file."""
        cache_key = self._generate_key(query, k, mode)
        cache = self._read_cache()
        if cache is None:
            return None
        return cache.get(cache_key, None)

    def set(self, query: str, k: int, mode: str, response: str) -> None:
        """Saves the query response to disk."""
        cache_key = self._generate_key(query, k, mode)
        cache = self._read_cache()
        if cache is None:
            cache = {}

        cache[cache_key] = {
            "query": query,
            "k": k,
            "response": response
        }

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)

    def flush(self) -> bool:
        """Completely removes the cache file when data changes."""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            return True
        return False
