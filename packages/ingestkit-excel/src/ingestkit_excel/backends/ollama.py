"""Ollama backends for the LLMBackend and EmbeddingBackend protocols.

Provides concrete implementations that communicate with a local Ollama server
via its HTTP API.  Requires ``httpx`` as an optional dependency.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ingestkit_excel.config import ExcelProcessorConfig

logger = logging.getLogger("ingestkit_excel")


class OllamaLLM:
    """Ollama-backed LLM.

    Satisfies :class:`~ingestkit_core.protocols.LLMBackend` via structural
    subtyping (no inheritance required).

    Parameters
    ----------
    base_url:
        Ollama server base URL (e.g. ``"http://localhost:11434"``).
    config:
        Pipeline configuration providing timeout and retry settings.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        config: ExcelProcessorConfig | None = None,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "httpx is required for OllamaLLM. "
                "Install it with: pip install 'ingestkit-excel[ollama]'"
            ) from exc

        from ingestkit_excel.config import ExcelProcessorConfig

        self._base_url = base_url.rstrip("/")
        self._config = config or ExcelProcessorConfig()

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a POST request to Ollama and return the JSON response.

        Retries on connection/timeout errors using the configured retry
        settings.
        """
        import httpx

        url = f"{self._base_url}{endpoint}"
        effective_timeout = timeout or self._config.backend_timeout_seconds

        last_exc: Exception | None = None
        max_attempts = 1 + self._config.backend_max_retries

        for attempt in range(max_attempts):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    timeout=effective_timeout,
                )
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "Ollama request timed out (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        max_attempts,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "Ollama request failed with HTTP %d (attempt %d/%d), retrying in %.1fs",
                        exc.response.status_code,
                        attempt + 1,
                        max_attempts,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "Ollama connection failed (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        max_attempts,
                        sleep_time,
                    )
                    time.sleep(sleep_time)

        # All retries exhausted
        if isinstance(last_exc, httpx.TimeoutException):
            raise TimeoutError(
                f"Ollama request timed out after {max_attempts} attempts: {last_exc}"
            ) from last_exc

        raise ConnectionError(
            f"Ollama connection failed after {max_attempts} attempts: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def classify(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout: float | None = None,
    ) -> dict:
        """Send a classification prompt and return the parsed JSON response.

        Posts to ``/api/generate`` with ``stream=False`` and ``format="json"``.
        Retries once if the response is not valid JSON.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }

        last_exc: Exception | None = None
        for attempt in range(2):  # 1 original + 1 retry for malformed JSON
            try:
                data = self._post("/api/generate", payload, timeout=timeout)
                response_text = data.get("response", "")
                return json.loads(response_text)
            except json.JSONDecodeError as exc:
                last_exc = exc
                logger.warning(
                    "Ollama classify returned malformed JSON (attempt %d/2): %s",
                    attempt + 1,
                    exc,
                )
                if attempt == 0:
                    # Retry with a hint appended to the prompt
                    payload = dict(payload)
                    payload["prompt"] = (
                        prompt
                        + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                        "Respond with valid JSON only."
                    )

        raise last_exc  # type: ignore[misc]

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str:
        """Send a generation prompt and return the raw text response."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        data = self._post("/api/generate", payload, timeout=timeout)
        return data.get("response", "")


class OllamaEmbedding:
    """Ollama-backed embedding model.

    Satisfies :class:`~ingestkit_core.protocols.EmbeddingBackend` via
    structural subtyping (no inheritance required).

    Parameters
    ----------
    base_url:
        Ollama server base URL.
    model:
        Embedding model name (e.g. ``"nomic-embed-text"``).
    embedding_dimension:
        Expected vector dimensionality.
    config:
        Pipeline configuration providing timeout settings.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        embedding_dimension: int = 768,
        config: ExcelProcessorConfig | None = None,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "httpx is required for OllamaEmbedding. "
                "Install it with: pip install 'ingestkit-excel[ollama]'"
            ) from exc

        from ingestkit_excel.config import ExcelProcessorConfig

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = embedding_dimension
        self._config = config or ExcelProcessorConfig()

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def embed(
        self, texts: list[str], timeout: float | None = None
    ) -> list[list[float]]:
        """Embed a batch of texts and return their vector representations.

        Posts to ``/api/embed`` with the configured model.
        """
        import httpx

        if not texts:
            return []

        url = f"{self._base_url}/api/embed"
        effective_timeout = timeout or self._config.backend_timeout_seconds
        payload = {
            "model": self._model,
            "input": texts,
        }

        last_exc: Exception | None = None
        max_attempts = 1 + self._config.backend_max_retries

        for attempt in range(max_attempts):
            try:
                response = httpx.post(url, json=payload, timeout=effective_timeout)
                response.raise_for_status()
                data = response.json()
                return data.get("embeddings", [])
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "Ollama embed timed out (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        max_attempts,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
            except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "Ollama embed failed (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        max_attempts,
                        sleep_time,
                    )
                    time.sleep(sleep_time)

        if isinstance(last_exc, httpx.TimeoutException):
            raise TimeoutError(
                f"Ollama embed timed out after {max_attempts} attempts: {last_exc}"
            ) from last_exc

        raise ConnectionError(
            f"Ollama embed connection failed after {max_attempts} attempts: {last_exc}"
        ) from last_exc

    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimension
