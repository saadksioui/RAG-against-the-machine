import chromadb
import hashlib
import bm25s
import fire
import json
import re
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.llm_model import LLMModel
from src.cache import Cache
from src.chunker import TextChunker, PythonChunker
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
    RagDataset,
    AnsweredQuestion
)
from src.constants import CHUNK_PATH, BM25_PATH, SYNC_STATE


class RagCLI:
    """Command-Line Interface for the RAG against the machine pipeline."""

    def __init__(self) -> None:
        self.dataset_path = Path("data/raw")
        self._model: Optional[LLMModel] = None
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        self.collection = self.client.get_or_create_collection("rag_42")
        self.cache = Cache("data/processed/cache", "llm_answers_cache.json")
        self.all_chunks: List[Dict[str, Any]] = []

    @property
    def model(self) -> LLMModel:
        if self._model is None:
            self._model = LLMModel("Qwen/Qwen3-0.6B")
        return self._model

    def _tokenizer(self, text: str) -> str:
        text = text.lower()

        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = text.replace('_', ' ').replace('.', ' ').replace('-', ' ')
        return tokens

    def _get_file_hash(self, filepath: Path) -> str:
        hash_obj = hashlib.sha256()
        with open(filepath, "rb") as file:
            while chunk := file.read(4000):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        if not self.dataset_path.exists():
            raise Exception("Error: This directory "
                            f"`{self.dataset_path}` doesn't exist!")

        sync_state = {}
        if SYNC_STATE.exists():
            try:
                with open(SYNC_STATE, 'r', encoding='utf-8') as f:
                    sync_state = json.load(f)
            except json.JSONDecodeError:
                sync_state = {}

        files = [item for item in self.dataset_path.rglob("*")
                 if item.is_file() and item.suffix.lower()
                 in ['.md', '.txt', '.py']]

        current_files = {}
        new_files = []
        modified_files = []
        unchanged_files = []
        for file_path in files:
            file = str(file_path)
            file_hash = self._get_file_hash(file_path)
            current_files[file] = file_hash
            if file not in sync_state:
                new_files.append(file)
            elif sync_state[file] != file_hash:
                modified_files.append(file)
            elif sync_state[file] == file_hash:
                unchanged_files.append(file)

        deleted_files = set(sync_state.keys()) - set(current_files.keys())

        if not new_files and not modified_files and not deleted_files:
            print("No changes detected. Index is up to date!")
            return

        self.cache.flush()

        print(
            f"Sync stats: {len(new_files)} new,"
            f"{len(modified_files)} modified,"
            f"{len(deleted_files)} deleted."
        )

        files_to_remove = list(deleted_files) + modified_files
        for file in files_to_remove:
            self.collection.delete(where={'file_path': file})

        if CHUNK_PATH.exists():
            with open(CHUNK_PATH, 'r', encoding='utf-8') as f:
                self.all_chunks = json.load(f)

        if self.all_chunks:
            self.all_chunks = [
                chunk for chunk in self.all_chunks
                if chunk['metadata']['file_path'] not in files_to_remove
            ]
        process_files = new_files + modified_files

        if process_files:
            documents = []
            metadatas = []
            ids = []
            for file_str in tqdm(process_files, desc="Chunking"):
                file_path = Path(file_str)
                if file_path.suffix.lower() == '.py':
                    chunks = PythonChunker(max_chunk_size).chunk(file_str)
                elif file_path.suffix.lower() in ['.md', '.txt']:
                    chunks = TextChunker(max_chunk_size).chunk(file_str)
                for chunk in chunks:
                    idx = f"{chunk.file_path}_{chunk.first_character_index}"
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
                    documents.append(chunk.content)
                    metadatas.append(metadata)
                    ids.append(idx)

            all_contents = [chunk['content'] for chunk in self.all_chunks]
            processed_chunks = [
                self._tokenizer(chunk) for chunk in all_contents]
            corpus_tokens = bm25s.tokenize(processed_chunks)
            retriver = bm25s.BM25()
            retriver.index(corpus_tokens)
            retriver.save(BM25_PATH)

            batch_size = 5000
            total_chunks = len(ids)
            for i in tqdm(range(0, total_chunks, batch_size), desc="Indexing"):
                self.collection.add(
                    documents=documents[i: i + batch_size],
                    metadatas=metadatas[i: i + batch_size],
                    ids=ids[i: i + batch_size]
                )

        print("Semantic & Lexical indexing complete!")
        CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNK_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.all_chunks, f, indent=2)

        with open(SYNC_STATE, 'w', encoding='utf-8') as f:
            json.dump(current_files, f, indent=2)

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        if not self.all_chunks:
            try:
                with open(CHUNK_PATH, 'r') as file:
                    self.all_chunks = json.load(file)
            except FileNotFoundError:
                print("Error: Run `uv run "
                      "python -m src index` First!!!")
                exit(1)
        retriever = bm25s.BM25.load(BM25_PATH, load_corpus=False)
        query_text = self._tokenizer(query)
        query_tokens = bm25s.tokenize([query_text])
        results, _ = retriever.retrieve(query_tokens, corpus=None, k=k)
        winners = [self.all_chunks[r] for r in results[0]]
        return winners

    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=k)
        outputs = [
            {
                "id": doc_id,
                "content": document,
                "metadata": metadata,
                "distance": distance,
            }
            for doc_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        return outputs

    def _hybrid_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        bm25_results = self._bm25_search(query, k)
        semantic_results = self._semantic_search(query, k)
        rrf_scores: Dict[str, float] = {}
        metadata_lookup: Dict[str, Dict[str, Any]] = {}
        content_lookup: Dict[str, str] = {}
        for idx, item in enumerate(bm25_results):
            doc_id = item['id']
            metadata_lookup[doc_id] = item['metadata']
            content_lookup[doc_id] = item['content']
            rank = idx + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        for idx, item in enumerate(semantic_results):
            doc_id = item['id']
            metadata_lookup[doc_id] = item['metadata']
            content_lookup[doc_id] = item['content']
            rank = idx + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        sorted_ids = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True)
        top_k_results = []
        for doc_id, score in sorted_ids[:k]:
            top_k_results.append({
                "id": doc_id,
                "rrf_score": score,
                "content": content_lookup[doc_id],
                "metadata": metadata_lookup[doc_id]
            })
        return top_k_results

    def search(self, query: str,
               k: int, search_mode: str = 'bm25') -> List[str]:
        """Return the top-k sources for a single query."""
        if search_mode == 'bm25':
            results = self._bm25_search(query, k)
        elif search_mode == 'semantic':
            results = self._semantic_search(query, k)
        elif search_mode == 'hybrid':
            results = self._hybrid_search(query, k)

        output: List[str] = []
        for src in results:
            start = src['metadata']['first_character_index']
            end = src['metadata']['last_character_index']
            file_path = src['metadata']['file_path']
            output.append(f"{file_path} [{start}: {end}]")
        return output

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str, search_mode: str = 'bm25') -> None:
        """Run search over a whole dataset and write results to JSON."""
        try:
            with open(dataset_path, 'r') as file:
                questions = json.load(file)['rag_questions']
        except FileNotFoundError:
            print(f"This path `{dataset_path}` is incorrect!!")
            exit(1)
        dataset_name = Path(dataset_path).stem
        all_results = []

        for question in tqdm(questions):
            query = question['question']
            if search_mode == 'bm25':
                results = self._bm25_search(query, k)
            elif search_mode == 'semantic':
                results = self._semantic_search(query, k)
            elif search_mode == 'hybrid':
                results = self._hybrid_search(query, k)
            all_results.append(
                MinimalSearchResults(
                    question_id=question['question_id'],
                    question=question['question'],
                    retrieved_sources=[MinimalSource(
                        file_path=res['metadata']['file_path'],
                        first_character_index=res['metadata']
                        ['first_character_index'],
                        last_character_index=res['metadata']
                        ['last_character_index']
                    ) for res in results]
                )
            )
        output = StudentSearchResults(search_results=all_results, k=k)
        save_dir_path = Path(save_directory)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        save_dir_path = save_dir_path / f"{dataset_name}.json"
        with open(save_dir_path, 'w') as file:
            file.write(output.model_dump_json(indent=2))

    def answer(self, query: str,
               k: int, search_mode: str = "bm25") -> str | None:
        """Answer a single query using the retrieved context."""
        cache_result = self.cache.get(query, k, search_mode)
        if cache_result is not None:
            if isinstance(cache_result['response'], str):
                return cache_result['response']
            return None

        if search_mode == 'bm25':
            results = self._bm25_search(query, k)
        elif search_mode == 'semantic':
            results = self._semantic_search(query, k)
        elif search_mode == 'hybrid':
            results = self._hybrid_search(query, k)
        output = self.model.generate(query, results, 256)
        self.cache.set(query, k, search_mode, output)

        return output

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str, search_mode: str = "bm25") -> None:
        """Generate answers for a dataset, producing a final JSON file."""
        dataset_name = Path(student_search_results_path).name
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(student_search_results_path, 'r') as file:
            student_search_results = StudentSearchResults.model_validate_json(
                file.read())

        save_dir_path = output_dir / dataset_name
        results = student_search_results.search_results
        k = student_search_results.k
        final_answers = []

        for item in tqdm(results):
            question_id = item.question_id
            question = item.question
            retrieved_sources = item.retrieved_sources
            cache_result = self.cache.get(question, k, search_mode)

            if cache_result:
                answer = cache_result['response']
            else:
                if search_mode == 'bm25':
                    chunks = self._bm25_search(question, k)
                elif search_mode == 'semantic':
                    chunks = self._semantic_search(question, k)
                elif search_mode == 'hybrid':
                    chunks = self._hybrid_search(question, k)
                answer = self.model.generate(question, chunks, 256)
                self.cache.set(question, k, search_mode, answer)

            res = MinimalAnswer(
                question_id=question_id,
                question=question,
                retrieved_sources=retrieved_sources,
                answer=answer
            )
            final_answers.append(res)
        result = StudentSearchResultsAndAnswer(
            search_results=final_answers, k=k)
        with open(save_dir_path, 'w') as file:
            file.write(result.model_dump_json(indent=2))

    def _calculate_iou(self, start1: int, end1: int,
                       start2: int, end2: int) -> float:
        overlap = max(0, min(end1, end2) - max(start1, start2))
        union = (end1 - start1) + (end2 - start2) - overlap
        return overlap / union if union > 0 else 0.0

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Report recall@k against a ground-truth dataset."""
        try:
            with open(student_search_results_path) as file:
                student_search = StudentSearchResults.model_validate_json(
                    file.read()).search_results
        except FileNotFoundError:
            print("Error: run search_dataset first!")
            exit(1)
        try:
            with open(dataset_path) as file:
                dataset_results = RagDataset.model_validate_json(
                    file.read()).rag_questions
        except FileNotFoundError:
            print("Error")
            exit(1)

        student_dict = {
            item.question_id: item
            for item in student_search
        }
        total_recall = 0.0
        num_ques = len(dataset_results)
        for res in dataset_results:
            if not isinstance(res, AnsweredQuestion):
                continue
            student_res = student_dict.get(res.question_id)
            if not student_res:
                continue
            matched_count = 0
            for source in res.sources:
                source_is_found = False
                for retrieved in student_res.retrieved_sources:
                    if source.file_path == retrieved.file_path:
                        start1, end1 = (source.first_character_index,
                                        source.last_character_index)
                        start2, end2 = (retrieved.first_character_index,
                                        retrieved.last_character_index)
                        iou = self._calculate_iou(start1, end1, start2, end2)
                        if iou >= 0.05:
                            source_is_found = True
                            break
                if source_is_found:
                    matched_count += 1
            total_recall += matched_count / len(res.sources)

        average_recall = total_recall / num_ques
        print(f"Local Recall@k: {average_recall * 100:.2f}%")


if __name__ == "__main__":
    fire.Fire(RagCLI)
