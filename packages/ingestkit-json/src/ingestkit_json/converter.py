"""Core JSON conversion logic -- flatten and chunk.

Provides ``flatten_json()`` to recursively convert parsed JSON into
dot-notation ``key_path: value`` lines, and ``chunk_text()`` to split
those lines into token-sized chunks suitable for embedding.
"""

from __future__ import annotations

from typing import Any

from ingestkit_json.config import JSONProcessorConfig
from ingestkit_json.models import FlattenResult


class _LimitExceeded(Exception):
    """Internal signal that a flattening limit was reached."""


def flatten_json(data: Any, config: JSONProcessorConfig) -> FlattenResult:
    """Recursively flatten parsed JSON into ``key_path: value`` lines.

    Objects use dot notation (configurable via ``config.path_separator``),
    arrays use index notation (``[0]``), root arrays start with ``[i]``,
    and root scalars produce a single line with the value.

    Parameters
    ----------
    data:
        The parsed JSON data (any valid JSON type).
    config:
        Configuration controlling depth limits, key limits, null handling,
        path separator, and index notation.

    Returns
    -------
    FlattenResult
        The flattened lines, stats, and whether the output was truncated.
    """
    # Root scalar
    if not isinstance(data, (dict, list)):
        return FlattenResult(
            lines=[_format_value(data)],
            total_keys=1,
            max_depth=0,
            truncated=False,
        )

    lines: list[str] = []
    state = {"key_count": 0, "max_depth": 0}
    truncated = False

    try:
        _flatten_recursive(
            data=data,
            prefix="",
            depth=0,
            config=config,
            lines=lines,
            state=state,
        )
    except _LimitExceeded:
        truncated = True

    return FlattenResult(
        lines=lines,
        total_keys=state["key_count"],
        max_depth=state["max_depth"],
        truncated=truncated,
    )


def _flatten_recursive(
    data: Any,
    prefix: str,
    depth: int,
    config: JSONProcessorConfig,
    lines: list[str],
    state: dict[str, int],
) -> None:
    """Recursive helper for ``flatten_json``.

    Mutates *lines* in place and tracks key_count/max_depth in *state*.
    Raises ``_LimitExceeded`` when depth or key limits are hit.
    """
    if depth > state["max_depth"]:
        state["max_depth"] = depth

    if depth > config.max_nesting_depth:
        raise _LimitExceeded("max_nesting_depth exceeded")

    sep = config.path_separator

    if isinstance(data, dict):
        if not data:
            # Empty object -- emit nothing
            return
        for key, value in data.items():
            child_prefix = f"{prefix}{sep}{key}" if prefix else key
            if isinstance(value, (dict, list)):
                _flatten_recursive(value, child_prefix, depth + 1, config, lines, state)
            else:
                if value is None and not config.include_null_values:
                    continue
                state["key_count"] += 1
                if state["key_count"] > config.max_keys:
                    raise _LimitExceeded("max_keys exceeded")
                lines.append(f"{child_prefix}: {_format_value(value)}")

    elif isinstance(data, list):
        if not data:
            # Empty array -- emit nothing
            return
        for i, item in enumerate(data):
            if config.array_index_notation:
                child_prefix = f"{prefix}[{i}]"
            else:
                child_prefix = prefix
            if isinstance(item, (dict, list)):
                _flatten_recursive(item, child_prefix, depth + 1, config, lines, state)
            else:
                if item is None and not config.include_null_values:
                    continue
                state["key_count"] += 1
                if state["key_count"] > config.max_keys:
                    raise _LimitExceeded("max_keys exceeded")
                lines.append(f"{child_prefix}: {_format_value(item)}")


def _format_value(value: Any) -> str:
    """Convert a Python value to its string representation for output."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def chunk_text(lines: list[str], config: JSONProcessorConfig) -> list[str]:
    """Split flattened lines into token-sized chunks with overlap.

    Token estimation uses word count (``len(text.split())``), consistent
    with the PDF chunker.  Lines are never split mid-line.

    Parameters
    ----------
    lines:
        The flattened ``key_path: value`` lines.
    config:
        Configuration for ``chunk_size_tokens`` and ``chunk_overlap_tokens``.

    Returns
    -------
    list[str]
        A list of text chunks ready for embedding.
    """
    if not lines:
        return []

    chunk_size = config.chunk_size_tokens
    overlap = config.chunk_overlap_tokens
    chunks: list[str] = []

    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(line.split())

        # If a single line exceeds chunk_size, flush current and emit it alone
        if line_tokens > chunk_size:
            if current_lines:
                chunks.append("\n".join(current_lines))
            chunks.append(line)
            current_lines = []
            current_tokens = 0
            continue

        # If adding this line would exceed chunk_size, flush current chunk
        if current_tokens + line_tokens > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))

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
        chunks.append("\n".join(current_lines))

    return chunks
