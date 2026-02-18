"""HTML tag stripping utility using stdlib ``html.parser``.

Provides :func:`strip_html_tags` which converts HTML to plain text by:
- Removing all tags
- Converting ``<br>`` and ``<p>``/``</p>`` to newlines
- Suppressing ``<script>`` and ``<style>`` content
- Decoding HTML entities
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class HTMLTagStripper(HTMLParser):
    """HTMLParser subclass that collects plain text from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._suppress = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self._suppress = True
        elif tag_lower == "br":
            self._pieces.append("\n")
        elif tag_lower == "p":
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self._suppress = False
        elif tag_lower == "p":
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._suppress:
            self._pieces.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self._suppress:
            char = html.unescape(f"&{name};")
            self._pieces.append(char)

    def handle_charref(self, name: str) -> None:
        if not self._suppress:
            char = html.unescape(f"&#{name};")
            self._pieces.append(char)

    def get_text(self) -> str:
        """Return the accumulated plain text."""
        return "".join(self._pieces)


def strip_html_tags(html_content: str) -> str:
    """Strip HTML tags and return plain text.

    Parameters
    ----------
    html_content:
        Raw HTML string.

    Returns
    -------
    str
        Cleaned plain text with normalized whitespace.
    """
    if not html_content:
        return ""

    stripper = HTMLTagStripper()
    stripper.feed(html_content)
    text = stripper.get_text()

    # Normalize whitespace: collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
