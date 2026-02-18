"""Core XML conversion logic -- extract and chunk.

Provides ``extract_xml()`` to recursively walk an XML element tree and
produce text lines, and ``chunk_text()`` to split those lines into
token-sized chunks suitable for embedding.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ingestkit_xml.config import XMLProcessorConfig
from ingestkit_xml.models import ExtractResult


class _LimitExceeded(Exception):
    """Internal signal that an element count limit was reached."""


def extract_xml(
    root: ET.Element,
    config: XMLProcessorConfig,
) -> ExtractResult:
    """Extract text content from an XML element tree.

    Recursively walks the tree, collecting text content from elements
    and (optionally) their attributes and tail text.

    Parameters
    ----------
    root:
        The root element of the parsed XML tree.
    config:
        Configuration controlling namespace stripping, attribute inclusion,
        depth/element limits, and indentation.

    Returns
    -------
    ExtractResult
        The extracted lines, statistics, and metadata.
    """
    # Collect unique namespaces
    namespaces = _collect_namespaces(root)

    lines: list[str] = []
    state = {"element_count": 0, "max_depth": 0}
    truncated = False

    try:
        _extract_recursive(root, depth=0, config=config, lines=lines, state=state)
    except _LimitExceeded:
        truncated = True

    root_tag = _strip_namespace(root.tag) if config.strip_namespaces else root.tag

    return ExtractResult(
        lines=lines,
        total_elements=state["element_count"],
        max_depth=state["max_depth"],
        namespaces=namespaces,
        root_tag=root_tag,
        truncated=truncated,
        fallback_used=False,
    )


def _extract_recursive(
    element: ET.Element,
    depth: int,
    config: XMLProcessorConfig,
    lines: list[str],
    state: dict[str, int],
) -> None:
    """Recursive helper for ``extract_xml``.

    Mutates *lines* in place and tracks element_count/max_depth in *state*.
    Raises ``_LimitExceeded`` when element count limit is hit.
    """
    if depth > state["max_depth"]:
        state["max_depth"] = depth

    state["element_count"] += 1
    if state["element_count"] > config.max_elements:
        raise _LimitExceeded("max_elements exceeded")

    # Build tag label
    tag = _strip_namespace(element.tag) if config.strip_namespaces else element.tag

    # Build attribute string
    attr_str = ""
    if config.include_attributes and element.attrib:
        filtered = {
            k: v
            for k, v in element.attrib.items()
            if not any(
                _strip_namespace(k).startswith(prefix)
                for prefix in config.skip_attribute_prefixes
            )
        }
        if filtered:
            parts = [f"{_strip_namespace(k)}={v}" for k, v in filtered.items()]
            attr_str = f" [{', '.join(parts)}]"

    # Emit element text
    text = element.text.strip() if element.text else ""
    if text:
        indent = "  " * depth if config.indent_text else ""
        lines.append(f"{indent}{tag}{attr_str}: {text}")
    elif attr_str:
        indent = "  " * depth if config.indent_text else ""
        lines.append(f"{indent}{tag}{attr_str}")

    # Recurse into children
    for child in element:
        _extract_recursive(child, depth + 1, config, lines, state)

    # Emit tail text
    if config.include_tail_text:
        tail = element.tail.strip() if element.tail else ""
        if tail:
            indent = "  " * depth if config.indent_text else ""
            lines.append(f"{indent}{tail}")


def _strip_namespace(tag: str) -> str:
    """Remove namespace URI prefix from a tag or attribute name.

    If the tag is ``{http://example.com}localname``, returns ``localname``.
    """
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _collect_namespaces(root: ET.Element) -> list[str]:
    """Collect unique namespace URIs from the element tree."""
    namespaces: set[str] = set()
    for element in root.iter():
        if element.tag.startswith("{"):
            uri = element.tag.split("}", 1)[0][1:]
            namespaces.add(uri)
        for attr_name in element.attrib:
            if attr_name.startswith("{"):
                uri = attr_name.split("}", 1)[0][1:]
                namespaces.add(uri)
    return sorted(namespaces)


def chunk_text(lines: list[str], config: XMLProcessorConfig) -> list[str]:
    """Split extracted lines into token-sized chunks with overlap.

    Token estimation uses word count (``len(text.split())``), consistent
    with the JSON and PDF chunkers.  Lines are never split mid-line.

    Parameters
    ----------
    lines:
        The extracted text lines from ``extract_xml()``.
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
