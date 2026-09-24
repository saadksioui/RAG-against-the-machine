from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


@dataclass
class Chunk:
    file_path: str
    content: str
    first_character_index: int
    last_character_index: int


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_characters = max_chunk_size
        self.overlap = int(max_chunk_size * 10 / 100)

    @abstractmethod
    def chunk(self, file_path: str) -> List[Chunk]:
        """Reads a file and returns a list of validated Chunk chunks."""
        pass



class PythonChunker(BaseChunker):
    def chunk(self, file_path: str) -> List[Chunk]:
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Error: File {file_path} not found!")
        extension = path_obj.suffix.lower()
        if extension != ".py":
            raise ValueError(f"Error: Only `.py` files are allowed")

        with open(file_path, 'r') as file:
            content = file.read()

        docs = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_characters,
            chunk_overlap=self.overlap,
            keep_separator=True
        )
        results = docs.split_text(content)
        start_from = 0
        chunks = []

        for item in results:
            idx = content.find(item, start_from)
            if idx == -1:
                idx = content.find(item)
            start = idx
            end = start + len(item)
            chunks.append(
                Chunk(
                    file_path=str(path_obj),
                    content=item,
                    first_character_index=start,
                    last_character_index=end
                )
            )
            start_from = max(start+1, end-self.overlap)
        return chunks


class TextChunker(BaseChunker):
    def chunk(self, file_path: str) -> List[Chunk]:
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Error: File {file_path} not found!")

        extension = path_obj.suffix.lower()
        with open(file_path, 'r') as file:
            content = file.read()

        if extension == ".md":
            docs = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.max_characters,
                chunk_overlap=self.overlap,
                keep_separator=True
            )
            results = docs.split_text(content)
        elif extension == ".txt":
            docs = RecursiveCharacterTextSplitter(
                chunk_size=self.max_characters,
                chunk_overlap=self.overlap,
                keep_separator=True
            )
            results = docs.split_text(content)
        else:
            raise ValueError(f"Error: Only `.md` and `.txt` files are allowed")
        start_from = 0
        chunks = []

        for item in results:
            idx = content.find(item, start_from)
            if idx == -1:
                idx = content.find(item)
            start = idx
            end = start + len(item)
            chunks.append(
                Chunk(
                    file_path=str(path_obj),
                    content=item,
                    first_character_index=start,
                    last_character_index=end
                )
            )
            start_from = max(start+1, end-self.overlap)
        return chunks