from abc import ABC, abstractmethod
from typing import List
from src.models import MinimalSource


class BaseChunker(ABC):
    def __init__(self, max_chunk_size: int = 2000):
        self.max_characters = max_chunk_size

    @abstractmethod
    def chunk(self, file_path: str) -> List[MinimalSource]:
        """Reads a file and returns a list of validated MinimalSource chunks."""
        pass


class PythonChunker(BaseChunker):
    def chunk(self, file_path: str) -> List[MinimalSource]:
        pass


class TextChunker(BaseChunker):
    def chunk(self, file_path: str) -> List[MinimalSource]:
        pass