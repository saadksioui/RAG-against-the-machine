from pathlib import Path

CHUNK_PATH = Path("data/processed/chunks/chunks.json")
BM25_PATH = "data/processed/bm25_index"
SYNC_STATE = CHUNK_PATH.parent / 'sync_state.json'
