"""Core .doc conversion logic -- extract and chunk.

Provides ``extract_text()`` to pull raw text from legacy Word documents
using mammoth, and ``chunk_text()`` to split the result into token-sized
chunks suitable for embedding.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

import mammoth  # type: ignore[import-untyped]

from ingestkit_doc.config import DocProcessorConfig

logger = logging.getLogger("ingestkit_doc")


class ExtractResult(BaseModel):
    """Internal model for the output of ``extract_text()``."""

    text: str
    word_count: int
    mammoth_messages: list[str] = []


def extract_text(file_path: str) -> ExtractResult:
    """Extract raw text from a .doc file using mammoth.

    Opens the file in binary mode and calls
    ``mammoth.extract_raw_text()``.  Any messages returned by mammoth
    are collected and returned in the result.

    Parameters
    ----------
    file_path:
        Filesystem path to the .doc file.

    Returns
    -------
    ExtractResult
        The extracted text, word count, and any mammoth messages.

    Raises
    ------
    Exception
        If mammoth cannot extract text from the file.
    """
    with open(file_path, "rb") as fh:
        result = mammoth.extract_raw_text(fh)

    text: str = result.value
    messages = [str(m) for m in result.messages]

    if messages:
        for msg in messages:
            logger.warning("ingestkit_doc | mammoth message: %s", msg)

    word_count = len(text.split()) if text.strip() else 0

    return ExtractResult(
        text=text,
        word_count=word_count,
        mammoth_messages=messages,
    )


def chunk_text(text: str, config: DocProcessorConfig) -> list[str]:
    """Split extracted text into token-sized chunks with overlap.

    Uses word-count token estimation (``len(text.split())``), consistent
    with ingestkit-json and ingestkit-pdf chunkers.  Splits on paragraph
    boundaries (``\\n\\n``) first, then on single newlines for paragraphs
    that exceed ``chunk_size_tokens``.

    Parameters
    ----------
    text:
        The raw text extracted from the .doc file.
    config:
        Configuration for ``chunk_size_tokens`` and ``chunk_overlap_tokens``.

    Returns
    -------
    list[str]
        A list of text chunks ready for embedding.
    """
    if not text or not text.strip():
        return []

    chunk_size = config.chunk_size_tokens
    overlap = config.chunk_overlap_tokens

    # Split on paragraph boundaries first
    paragraphs = text.split("\n\n")

    # Further split paragraphs that exceed chunk_size on single newlines
    lines: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = len(para.split())
        if para_tokens > chunk_size:
            # Split on single newlines
            sub_lines = para.split("\n")
            for sub in sub_lines:
                sub = sub.strip()
                if sub:
                    lines.append(sub)
        else:
            lines.append(para)

    if not lines:
        return []

    # Build chunks from lines using the same algorithm as ingestkit-json
    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(line.split())

        # If a single line exceeds chunk_size, flush current and emit it alone
        if line_tokens > chunk_size:
            if current_lines:
                chunks.append("\n\n".join(current_lines))
            chunks.append(line)
            current_lines = []
            current_tokens = 0
            continue

        # If adding this line would exceed chunk_size, flush current chunk
        if current_tokens + line_tokens > chunk_size and current_lines:
            chunks.append("\n\n".join(current_lines))

            # Compute overlap: take trailing lines up to overlap token count
            overlap_lines: list[str] = []
            overlap_tokens = 0
            for prev_line in reversed(current_lines):
                prev_tokens = len(prev_line.split())
                if overlap_tokens + prev_tokens > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_tokens += prev_tokens

            current_lines = overlap_lines
            current_tokens = overlap_tokens

        current_lines.append(line)
        current_tokens += line_tokens

    # Flush remaining
    if current_lines:
        chunks.append("\n\n".join(current_lines))

    return chunks
