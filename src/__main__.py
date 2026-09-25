import fire
import re
from tqdm import tqdm
from pathlib import Path
from src.chunker import TextChunker, PythonChunker
from src.constants import CHUNK_PATH
import chromadb
import json

class RagCLI:
    """Command-Line Interface for the RAG against the machine pipeline."""

    def __init__(self):
        self.dataset_path = Path("data/raw")
        self.bm25_index = None
        self.all_chunks = []

    def _tokenizer(self, text: str):
        text = text.lower()

        tokens = re.split(r'[^a-z0-9]+', text)
        return [t for t in tokens if t]

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        if not self.dataset_path.exists():
            raise Exception(f"Error: This directory doesn't exist!")
        files = [item for item in self.dataset_path.rglob("*")
                 if item.is_file() and item.suffix.lower()
                 in ['.md', '.txt', '.py']]

        for file in tqdm(files, desc="Chunking"):
            if file.suffix.lower() == '.py':
                chunks = PythonChunker(max_chunk_size).chunk(file)
            elif file.suffix.lower() in ['.md', '.txt']:
                chunks = TextChunker(max_chunk_size).chunk(file)

            

    def _bm25_search():
        pass

    def _semantic_search():
        pass

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        pass

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run search over a whole dataset and write results to JSON."""
        pass

    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context."""
        pass

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Generate answers for a dataset, producing a final JSON file."""
        pass

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """Report recall@k against a ground-truth dataset."""
        pass


if __name__ == "__main__":
    fire.Fire(RagCLI)