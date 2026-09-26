import fire
import re
from tqdm import tqdm
from pathlib import Path
from src.chunker import TextChunker, PythonChunker
from src.constants import CHUNK_PATH, BM25_PATH
import chromadb
import bm25s
import json

class RagCLI:
    """Command-Line Interface for the RAG against the machine pipeline."""

    def __init__(self):
        self.dataset_path = Path("data/raw")
        self.all_chunks = []

    def _tokenizer(self, text: str):
        text = text.lower()

        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = text.replace('_', ' ').replace('.', ' ').replace('-', ' ')
        return tokens

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        if not self.dataset_path.exists():
            raise Exception(f"Error: This directory doesn't exist!")
        files = [item for item in self.dataset_path.rglob("*")
                 if item.is_file() and item.suffix.lower()
                 in ['.md', '.txt', '.py']]

        processed_chunks = []
        documents = []
        metadatas = []
        ids = []
        global_counter = 0
        for file in tqdm(files, desc="Chunking"):
            if file.suffix.lower() == '.py':
                chunks = PythonChunker(max_chunk_size).chunk(file)
            elif file.suffix.lower() in ['.md', '.txt']:
                chunks = TextChunker(max_chunk_size).chunk(file)
            for chunk in chunks:
                idx = f'doc_{global_counter}'
                metadata = {
                    'file_path': chunk.file_path,
                    'first_character_index': chunk.first_character_index,
                    'last_character_index': chunk.last_character_index
                }

                self.all_chunks.append({
                    'id': idx,
                    'content': chunk.content,
                    'metadata': metadata
                })
                processed_chunks.append(self._tokenizer(chunk.content))
                documents.append(chunk.content)
                metadatas.append(metadata)
                ids.append(idx)
                global_counter += 1

        corpus_tokens = bm25s.tokenize(processed_chunks)
        retriver = bm25s.BM25()
        retriver.index(corpus_tokens)
        retriver.save(BM25_PATH)

        CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNK_PATH, 'w', encoding='utf-8') as file:
            json.dump(self.all_chunks, file, indent=2)


    def _bm25_search(self, query: str, k: int) -> list[dict]:
        if not self.all_chunks:
            try:
                with open(CHUNK_PATH, 'r') as file:
                    self.all_chunks = json.load(file)
            except FileNotFoundError:
                print(f"Error: Run `uv run python -m src index` First!!!")
                exit(1)
        retriever = bm25s.BM25.load(BM25_PATH, load_corpus=False)
        query_text = self._tokenizer(query)
        query_tokens = bm25s.tokenize([query_text])
        results, scores = retriever.retrieve(query_tokens, corpus=None, k=k)
        winners = [self.all_chunks[r] for r in results[0]]
        return winners

    def _semantic_search():
        pass

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        results = self._bm25_search(query, k)
        for i, item in enumerate(results, start=1):
            print(f"Result {i}:")
            print(f"    File: {item['metadata']['file_path']}")
            print(f"    Start: {item['metadata']['first_character_index']}")
            print(f"    End: {item['metadata']['last_character_index']}")
            print(f"    Content: {item['content']}")

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