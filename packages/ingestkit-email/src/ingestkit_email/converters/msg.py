"""MSG converter using the optional ``extract-msg`` library.

Parses Outlook .msg files and extracts headers, body text, and
attachment filenames.  Requires ``extract-msg>=0.48.0`` (MIT).
"""

from __future__ import annotations

import logging

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.converters.base import EmailContent
from ingestkit_email.html_strip import strip_html_tags

logger = logging.getLogger("ingestkit_email")


class MSGConverter:
    """Convert .msg files to :class:`EmailContent`."""

    def convert(
        self, file_path: str, config: EmailProcessorConfig
    ) -> EmailContent:
        """Parse an MSG file and extract its content.

        Parameters
        ----------
        file_path:
            Path to the ``.msg`` file.
        config:
            Processing configuration.

        Returns
        -------
        EmailContent
            Extracted headers, body, and attachment names.

        Raises
        ------
        ImportError
            If ``extract-msg`` is not installed.
        """
        try:
            import extract_msg  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "extract-msg is required to process .msg files. "
                "Install it with: pip install extract-msg"
            )

        msg = extract_msg.Message(file_path)
        try:
            # Extract headers
            from_addr = getattr(msg, "sender", None) or None
            to_addr = getattr(msg, "to", None) or None
            cc_addr = getattr(msg, "cc", None) or None
            date_str = str(msg.date) if getattr(msg, "date", None) else None
            subject = getattr(msg, "subject", None) or None
            message_id = getattr(msg, "messageId", None) or None

            # Body: prefer plain text, fallback to HTML
            body_text = ""
            body_source = "plain"

            plain_body = getattr(msg, "body", None)
            html_body = getattr(msg, "htmlBody", None)

            if config.prefer_plain_text and plain_body:
                body_text = plain_body
                body_source = "plain"
            elif html_body:
                if isinstance(html_body, bytes):
                    html_body = html_body.decode("utf-8", errors="replace")
                body_text = strip_html_tags(html_body)
                body_source = "html_converted"
            elif plain_body:
                body_text = plain_body
                body_source = "plain"

            # Attachment names
            attachment_names: list[str] = []
            attachments = getattr(msg, "attachments", []) or []
            for att in attachments:
                name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None)
                if name:
                    attachment_names.append(name)

            # Raw headers (limited for MSG)
            raw_headers: dict[str, str] = {}
            if from_addr:
                raw_headers["From"] = from_addr
            if to_addr:
                raw_headers["To"] = to_addr
            if date_str:
                raw_headers["Date"] = date_str
            if subject:
                raw_headers["Subject"] = subject

            return EmailContent(
                from_address=from_addr,
                to_address=to_addr,
                cc_address=cc_addr,
                date=date_str,
                subject=subject,
                message_id=message_id,
                body_text=body_text,
                body_source=body_source,
                attachment_names=attachment_names,
                raw_headers=raw_headers,
            )
        finally:
            msg.close()
