from pathlib import Path
from src.chunker import TextChunker, PythonChunker

def get_files(dataset_path):
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise Exception(f"Error: This directory doesn't exist!")
    files = [item for item in path_obj.rglob("*") if item.is_file() and item.suffix.lower() in ['.md', '.txt', '.py']]
    file_paths = {
        '.md': [str(item) for item in files if item.suffix.lower() == '.md'],
        '.txt': [str(item) for item in files if item.suffix.lower() == '.txt'],
        '.py': [str(item) for item in files if item.suffix.lower() == '.py']
    }
    return file_paths



files = get_files("data/raw/")
chunker = PythonChunker()
chunks = chunker.chunk(files['.py'][0])

print(chunks)