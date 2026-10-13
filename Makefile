.PHONY: install run debug clean lint

install:
	uv sync

run:
	@echo "Indexing...."
	uv run python -m src index --max_chunk_size=2000

	@echo "Search code dataset"
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/

	@echo "Search docs dataset"
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/

	@echo "Evaluate docs retrieval"
	uv run python -m src evaluate --student_search_results_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
	
	@echo "Evaluate code retrieval"
	uv run python -m src evaluate --student_search_results_path data/output/search_results/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

debug:
	uv run python -m pdb src

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .mypy_cache

lint:
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
