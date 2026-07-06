"""Shared response cache for LLM API calls."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")


class ResponseCache:
    """Simple disk-backed JSON cache shared by all LLM clients.

    Args:
        cache_dir: Directory to store cached responses.
        ttl: Time-to-live for cache entries in seconds (default 1 hour).
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        ttl: float = 3600.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(prompt: str, model: str, params: dict[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": prompt, "model": model, "params": params},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(
        self,
        prompt: str,
        model: str,
        params: dict[str, Any] | None = None,
    ) -> Optional[dict[str, Any]]:
        """Look up a cached response. Returns ``None`` on miss or expiry."""
        key = self._make_key(prompt, model, params or {})
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - data.get("timestamp", 0) > self.ttl:
            path.unlink(missing_ok=True)
            return None
        logger.debug("Cache hit for key %s", key[:12])
        return data.get("response")

    def put(
        self,
        prompt: str,
        model: str,
        response: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Store a response in the cache."""
        key = self._make_key(prompt, model, params or {})
        path = self._path_for(key)
        data = {"timestamp": time.time(), "response": response}
        path.write_text(json.dumps(data, indent=2))
        logger.debug("Cached response under key %s", key[:12])

    def clear(self) -> int:
        """Remove all cached entries. Returns the number of entries removed."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink(missing_ok=True)
            count += 1
        logger.info("Cleared %d cache entries", count)
        return count
