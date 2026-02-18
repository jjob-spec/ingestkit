"""Shared data structures for email converters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmailContent:
    """Shared return type from both EML and MSG converters.

    Contains the extracted headers, body text, and attachment metadata
    from a single email file.
    """

    from_address: str | None = None
    to_address: str | None = None
    cc_address: str | None = None
    date: str | None = None
    subject: str | None = None
    message_id: str | None = None
    body_text: str = ""
    body_source: str = "plain"  # "plain" or "html_converted"
    attachment_names: list[str] = field(default_factory=list)
    raw_headers: dict[str, str] = field(default_factory=dict)
