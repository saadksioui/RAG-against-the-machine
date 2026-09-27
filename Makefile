.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src index --max_chunk_size=2000

debug:
	uv run python -m pdb src

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .mypy_cache

lint:
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
