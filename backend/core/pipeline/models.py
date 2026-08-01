from dataclasses import dataclass

@dataclass
class TextBlock:
    index: int
    text: str


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    char_start: int
    char_end: int
    source: str
    token_count: int = 0