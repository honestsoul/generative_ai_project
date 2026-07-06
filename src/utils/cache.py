"""Response caching to reduce redundant API calls and costs.

Security note: Cache files may contain sensitive data from API responses.
Ensure the cache directory has appropriate file permissions and is
listed in .gitignore.
"""

import os
import json
import hashlib
import time
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


class ResponseCache:
    """File-based response cache with TTL expiration."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate a deterministic cache key from request parameters."""
        content = json.dumps(
            {"prompt": prompt, "model": model, **kwargs},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str, **kwargs) -> str | None:
        """Retrieve a cached response if it exists and is not expired."""
        key = self._make_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                entry = json.load(f)

            if time.time() - entry.get("timestamp", 0) > self.ttl_seconds:
                cache_file.unlink(missing_ok=True)
                return None

            return entry.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, prompt: str, model: str, response: str, **kwargs) -> None:
        """Store a response in the cache."""
        key = self._make_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        entry = {
            "timestamp": time.time(),
            "model": model,
            "response": response,
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(entry, f)
            # Restrict file permissions (owner read/write only)
            os.chmod(cache_file, 0o600)
        except OSError as e:
            logger.warning(f"Failed to write cache: {e}")

    def clear(self) -> int:
        """Remove all cached entries. Returns the number of entries removed."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == "sample.json":
                continue
            cache_file.unlink(missing_ok=True)
            count += 1
        return count
