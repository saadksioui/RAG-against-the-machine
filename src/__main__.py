import fire


class RagCLI:
    """Command-Line Interface for the RAG against the machine pipeline."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
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
    # Passing the class strictly limits the CLI to the methods defined inside it.
    fire.Fire(RagCLI)