import chromadb
import bm25s
import fire
import json
import re
from tqdm import tqdm
from pathlib import Path
from src.llm_model import LLM_Model
from src.chunker import TextChunker, PythonChunker
from src.models import MinimalSource, MinimalSearchResults, StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer, RagDataset
from src.constants import CHUNK_PATH, BM25_PATH

class RagCLI:
    """Command-Line Interface for the RAG against the machine pipeline."""

    def __init__(self):
        self.dataset_path = Path("data/raw")
        self.model = LLM_Model("Qwen/Qwen3-0.6B")
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
        results, _ = retriever.retrieve(query_tokens, corpus=None, k=k)
        winners = [self.all_chunks[r] for r in results[0]]
        return winners

    def _semantic_search():
        pass

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        results = self._bm25_search(query, k)
        for item in results:
            start = item['metadata']['first_character_index']
            end = item['metadata']['last_character_index']
            file_path = item['metadata']['file_path']
            print(f"{file_path} [{start}:{end}]")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run search over a whole dataset and write results to JSON."""
        try:
            with open(dataset_path, 'r') as file:
                questions = json.load(file)['rag_questions']
        except FileNotFoundError:
            print(f"This path `{dataset_path}` is incorrect!!")
            exit(1)
        dataset_name = Path(dataset_path).stem
        save_directory = Path(save_directory) / dataset_name
        all_results = []

        for question in tqdm(questions):
            results = self._bm25_search(question['question'], k)
            all_results.append(
                MinimalSearchResults(
                    question_id=question['question_id'],
                    question=question['question'],
                    retrieved_sources=[MinimalSource(
                        file_path=res['metadata']['file_path'],
                        first_character_index=res['metadata']['first_character_index'],
                        last_character_index=res['metadata']['last_character_index']
                    ) for res in results]
                )
            )
        output = StudentSearchResults(search_results=all_results, k=k)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_directory = save_directory / 'student_results.json'
        with open(save_directory, 'w') as file:
            file.write(output.model_dump_json(indent=2))
    
    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context."""
        results = self._bm25_search(query, k)
        output = self.model.generate(query, results, 256)
        
        print(f"\nFINAL ANSWER:\n{output}")

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Generate answers for a dataset, producing a final JSON file."""
        dataset_name = Path(student_search_results_path).parent.name
        output_dir = Path(save_directory) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(student_search_results_path, 'r') as file:
            student_search_results = StudentSearchResults.model_validate_json(file.read())

        save_directory = output_dir / 'student_results.json'
        results = student_search_results.search_results
        k = student_search_results.k
        final_answers = []

        for item in tqdm(results):
            question_id = item.question_id
            question = item.question
            retrieved_sources = item.retrieved_sources
            chunks = self._bm25_search(question, k)
            answer = self.model.generate(question, chunks, 256)
            res = MinimalAnswer(
                question_id = question_id,
                question = question,
                retrieved_sources = retrieved_sources,
                answer = answer
            )
            final_answers.append(res)
        result = StudentSearchResultsAndAnswer(search_results=final_answers, k=k)
        with open(save_directory, 'w') as file:
            file.write(result.model_dump_json(indent=2))

    def _calculate_iou(self, start1: int, end1: int, start2: int, end2: int) -> float:
        overlap = max(0, min(end1, end2) - max(start1, start2))
        union = (end1 - start1) + (end2 - start2) - overlap
        return overlap / union if union > 0 else 0.0

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """Report recall@k against a ground-truth dataset."""
        try:
            with open(student_search_results_path) as file:
                student_search_results = StudentSearchResults.model_validate_json(file.read()).search_results
        except FileNotFoundError:
            print("Error: run search_dataset first!")
            exit(1)
        try:
            with open(dataset_path) as file:
                dataset_results = RagDataset.model_validate_json(file.read()).rag_questions
        except FileNotFoundError:
            print("Error")
            exit(1)

        student_dict = {
            item.question_id: item 
            for item in student_search_results
        }
        total_recall = 0.0
        num_ques = len(dataset_results)
        for res in dataset_results:
            student_res = student_dict.get(res.question_id)
            if not student_res:
                continue
            matched_count = 0
            for source in res.sources:
                source_is_found = False
                for retrieved in student_res.retrieved_sources:
                    if source.file_path == retrieved.file_path:
                        start1, end1 = source.first_character_index, source.last_character_index
                        start2, end2 = retrieved.first_character_index, retrieved.last_character_index
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