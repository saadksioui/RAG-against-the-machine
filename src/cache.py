from typing import Dict, Any
import os
import json
import hashlib

class Cache:
    def __init__(self, cache_dir: str, cache_file: str):
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_file)

    def generate_key(self, query: str, k: int, mode: str) -> str:
        """Creates a unique SHA-256 hash based on the query text, retrieval count and the search mode."""
        query = query.strip().lower()
        key = f"query:{query}|k:{k}|mode:{mode}"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def read_cache(self) -> Dict[str, Any]:
        """Reads the cache file from disk."""
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def write_cache(self, cache_data: Dict[str, Any]):
        """Writes the updated cache object back to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=4)