from abc import ABC, abstractmethod
from typing import List
from src.models import MinimalSource


class BaseChunker(ABC):
    def __init__(self):
        self.base_directory = "data/raw/vllm-0.10.1"
        self.max_characters = 2000

    @abstractmethod
    def chunk(file_path: str) -> List[MinimalSource]:
        pass


class PythonChunker(BaseChunker):
    def chunk(file_path: str) -> List[MinimalSource]:
        pass


class TextChunker(BaseChunker):
    def chunk(file_path: str) -> List[MinimalSource]:
        pass