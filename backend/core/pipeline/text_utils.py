from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Callable, List, Sequence

from .models import TextBlock, TextChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_NUMERIC_CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[-,;]\s*\d+)*\s*\]")
_WIKI_HEADING_PATTERN = re.compile(r"^\s*={2,}\s*(.*?)\s*={2,}\s*$")
_WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
_WORD_PATTERN = re.compile(r"\S+")


def parse_text(file_path: str | Path) -> List[TextBlock]:
    """
    Parse a UTF-8 text file into paragraph-level TextBlock objects.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Text file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file but received: {path}")

    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if not content.strip():
        logging.warning(
            "The text file is empty: %s",
            path,
        )
        return []

    raw_blocks = re.split(
        r"\n\s*\n",
        content.strip(),
    )

    blocks: List[TextBlock] = []

    for raw_block in raw_blocks:
        # Join wrapped lines inside one paragraph.
        text = " ".join(line.strip() for line in raw_block.splitlines() if line.strip())

        text = text.strip()

        if not text:
            continue

        blocks.append(
            TextBlock(
                index=len(blocks),
                text=text,
            )
        )

    logging.info(
        "Parsed %d text blocks from '%s'.",
        len(blocks),
        path.name,
    )

    return blocks


def clean_text(
    blocks: Sequence[TextBlock],
    remove_html_tags: bool = True,
    remove_numeric_citations: bool = True,
    minimum_length: int = 3,
) -> List[TextBlock]:
    """
    Clean text blocks while preserving potentially useful semantic content.
    """
    if minimum_length < 1:
        raise ValueError("minimum_length must be at least 1.")

    cleaned_blocks: List[TextBlock] = []

    for block in blocks:
        text = block.text

        if not text:
            continue

        # Preserve heading content while removing heading markers.
        heading_match = _WIKI_HEADING_PATTERN.fullmatch(text)

        if heading_match:
            text = heading_match.group(1)

        if remove_html_tags:
            text = _HTML_TAG_PATTERN.sub(
                " ",
                text,
            )

        if remove_numeric_citations:
            text = _NUMERIC_CITATION_PATTERN.sub(
                " ",
                text,
            )

        # Normalize horizontal whitespace but preserve meaningful content.
        text = _WHITESPACE_PATTERN.sub(
            " ",
            text,
        )

        text = text.strip()

        if len(text) < minimum_length:
            continue

        cleaned_blocks.append(
            TextBlock(
                index=block.index,
                text=text,
            )
        )

    logging.info(
        "Text cleaning completed: %d/%d blocks retained.",
        len(cleaned_blocks),
        len(blocks),
    )

    return cleaned_blocks


def _create_token_counter(
    encoding_name: str,
) -> Callable[[str], int]:
    """
    Create a token-counting function.
    """
    try:
        import tiktoken
    except ImportError:
        logging.warning(
            "tiktoken is not installed. " "Using whitespace token counting as fallback."
        )

        return lambda text: len(text.split())

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        logging.warning(
            "Unknown tiktoken encoding '%s'. "
            "Using whitespace token counting as fallback.",
            encoding_name,
        )

        return lambda text: len(text.split())

    # encode_ordinary treats special-token-like strings as ordinary text.
    return lambda text: len(encoding.encode_ordinary(text))


def _find_chunk_end(
    text: str,
    word_matches: Sequence[re.Match],
    start_word: int,
    chunk_size: int,
    count_tokens: Callable[[str], int],
) -> int:
    """
    Find the largest word boundary whose text does not exceed chunk_size.
    """
    total_words = len(word_matches)

    # Every whitespace-separated unit normally requires at least one token.
    # Limiting the search range avoids encoding very large portions of the
    # document during binary search.
    search_end = min(
        total_words,
        start_word + chunk_size,
    )

    start_char = word_matches[start_word].start()

    low = start_word + 1
    high = search_end
    best_end = start_word

    while low <= high:
        candidate_end = (low + high) // 2

        end_char = word_matches[candidate_end - 1].end()

        candidate_text = text[start_char:end_char]

        token_count = count_tokens(candidate_text)

        if token_count <= chunk_size:
            best_end = candidate_end
            low = candidate_end + 1
        else:
            high = candidate_end - 1

    return best_end


def _find_overlap_start(
    text: str,
    word_matches: Sequence[re.Match],
    current_start: int,
    current_end: int,
    overlap: int,
    count_tokens: Callable[[str], int],
) -> int:
    """
    Find the earliest word boundary producing an overlap no larger than
    the requested token count.
    """
    if overlap <= 0:
        return current_end

    chunk_end_char = word_matches[current_end - 1].end()

    next_start = current_end

    # Walk backward until adding another word would exceed the overlap.
    for candidate_start in range(
        current_end - 1,
        current_start,
        -1,
    ):
        overlap_start_char = word_matches[candidate_start].start()

        overlap_text = text[overlap_start_char:chunk_end_char]

        overlap_tokens = count_tokens(overlap_text)

        if overlap_tokens <= overlap:
            next_start = candidate_start
        else:
            break

    return next_start


def chunk_blocks(
    segments: Sequence[TextBlock],
    chunk_size: int,
    overlap: int,
    source_name: str,
    encoding_name: str = "cl100k_base",
) -> List[TextChunk]:
    """
    Chunk cleaned text blocks using token-aware, word-boundary windows.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if not source_name or not source_name.strip():
        raise ValueError("source_name cannot be empty.")

    valid_segments = [
        segment for segment in segments if segment.text and segment.text.strip()
    ]

    if not valid_segments:
        logging.warning(
            "No valid segments were provided for '%s'.",
            source_name,
        )
        return []

    logging.info(
        "Starting chunking for source '%s': " "chunk_size=%d, overlap=%d, encoding=%s",
        source_name,
        chunk_size,
        overlap,
        encoding_name,
    )

    count_tokens = _create_token_counter(encoding_name)

    # Preserve paragraph boundaries rather than flattening everything into
    # one continuous sentence.
    all_text = "\n\n".join(segment.text.strip() for segment in valid_segments)

    word_matches = list(_WORD_PATTERN.finditer(all_text))

    if not word_matches:
        return []

    chunks: List[TextChunk] = []

    start_word = 0
    chunk_id = 0
    total_words = len(word_matches)

    while start_word < total_words:
        end_word = _find_chunk_end(
            text=all_text,
            word_matches=word_matches,
            start_word=start_word,
            chunk_size=chunk_size,
            count_tokens=count_tokens,
        )

        # A single extremely long unbroken item, such as a URL or encoded
        # string, can exceed chunk_size by itself.
        if end_word == start_word:
            end_word = start_word + 1

            oversized_start = word_matches[start_word].start()

            oversized_end = word_matches[start_word].end()

            oversized_text = all_text[oversized_start:oversized_end]

            logging.warning(
                "A single text unit in source '%s' exceeds " "chunk_size: %d tokens.",
                source_name,
                count_tokens(oversized_text),
            )

        char_start = word_matches[start_word].start()

        char_end = word_matches[end_word - 1].end()

        chunk_text = all_text[char_start:char_end].strip()

        token_count = count_tokens(chunk_text)

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                source=source_name,
                token_count=token_count,
            )
        )

        chunk_id += 1

        if end_word >= total_words:
            break

        next_start = _find_overlap_start(
            text=all_text,
            word_matches=word_matches,
            current_start=start_word,
            current_end=end_word,
            overlap=overlap,
            count_tokens=count_tokens,
        )

        # Defensive forward-progress check.
        if next_start <= start_word:
            next_start = start_word + 1

        start_word = next_start

    logging.info(
        "Chunking completed for '%s': %d chunks created.",
        source_name,
        len(chunks),
    )

    return chunks