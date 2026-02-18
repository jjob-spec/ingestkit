"""EML converter using Python stdlib ``email`` module.

Parses RFC 5322 email files (.eml) and extracts headers, body text,
and attachment filenames.  Zero external dependencies.
"""

from __future__ import annotations

import email
import email.policy
import logging

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.converters.base import EmailContent
from ingestkit_email.html_strip import strip_html_tags

logger = logging.getLogger("ingestkit_email")


class EMLConverter:
    """Convert .eml files to :class:`EmailContent`."""

    def convert(
        self, file_path: str, config: EmailProcessorConfig
    ) -> EmailContent:
        """Parse an EML file and extract its content.

        Parameters
        ----------
        file_path:
            Path to the ``.eml`` file.
        config:
            Processing configuration.

        Returns
        -------
        EmailContent
            Extracted headers, body, and attachment names.
        """
        with open(file_path, "rb") as f:
            data = f.read()

        msg = email.message_from_bytes(data, policy=email.policy.default)

        # Extract headers
        from_addr = str(msg.get("From", "")) or None
        to_addr = str(msg.get("To", "")) or None
        cc_addr = str(msg.get("Cc", "")) or None
        date_str = str(msg.get("Date", "")) or None
        subject = str(msg.get("Subject", "")) or None
        message_id = str(msg.get("Message-ID", "")) or None

        # Collect raw headers
        raw_headers: dict[str, str] = {}
        for key in msg.keys():
            raw_headers[key] = str(msg[key])

        # Walk parts and collect body content
        plain_parts: list[str] = []
        html_parts: list[str] = []
        attachment_names: list[str] = []

        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            # Skip multipart containers
            if part.get_content_maintype() == "multipart":
                continue

            # Check for attachments
            if "attachment" in disposition:
                filename = part.get_filename()
                if filename:
                    attachment_names.append(filename)
                continue

            if content_type == "text/plain":
                payload = part.get_content()
                if isinstance(payload, str):
                    plain_parts.append(payload)
            elif content_type == "text/html":
                payload = part.get_content()
                if isinstance(payload, str):
                    html_parts.append(payload)
            else:
                # Binary or other content type -- treat as attachment
                filename = part.get_filename()
                if filename:
                    attachment_names.append(filename)

        # Select body: prefer plain text if configured and available
        body_text = ""
        body_source = "plain"

        if config.prefer_plain_text and plain_parts:
            body_text = "\n".join(plain_parts)
            body_source = "plain"
        elif html_parts:
            body_text = strip_html_tags("\n".join(html_parts))
            body_source = "html_converted"
        elif plain_parts:
            body_text = "\n".join(plain_parts)
            body_source = "plain"

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
