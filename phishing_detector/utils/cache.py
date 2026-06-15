"""
PhishGuard AI - Cache Manager
Disk-based + in-memory LRU cache for analysis results and model outputs.
"""
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

from utils.logger import log


class CacheManager:
    """
    Two-level cache:
      L1 — in-memory LRU (instant lookup)
      L2 — disk cache via diskcache (survives restarts)
    """

    def __init__(self, cache_dir: str, max_size_mb: int = 256, ttl: int = 3600):
        self.ttl = ttl
        self._memory: dict[str, tuple[Any, float]] = {}
        self._max_memory = 1000  # max in-memory entries

        if HAS_DISKCACHE:
            self._disk = diskcache.Cache(
                cache_dir,
                size_limit=max_size_mb * 1024 * 1024,
            )
            log.debug(f"Disk cache initialized at {cache_dir}")
        else:
            self._disk = None
            log.warning("diskcache not available — using memory-only cache")

    @staticmethod
    def make_key(content: str, analysis_type: str = "") -> str:
        """Stable cache key from content hash."""
        raw = f"{analysis_type}::{content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        """Retrieve from L1 then L2."""
        # L1
        if key in self._memory:
            value, expires_at = self._memory[key]
            if time.time() < expires_at:
                return value
            del self._memory[key]

        # L2
        if self._disk is not None:
            try:
                value = self._disk.get(key)
                if value is not None:
                    self._memory[key] = (value, time.time() + self.ttl)
                    return value
            except Exception:
                pass

        return None

    def set(self, key: str, value: Any) -> None:
        """Store in L1 and L2."""
        expires_at = time.time() + self.ttl

        # Evict oldest if L1 full
        if len(self._memory) >= self._max_memory:
            oldest = min(self._memory, key=lambda k: self._memory[k][1])
            del self._memory[oldest]

        self._memory[key] = (value, expires_at)

        if self._disk is not None:
            try:
                self._disk.set(key, value, expire=self.ttl)
            except Exception:
                pass

    def clear(self) -> None:
        self._memory.clear()
        if self._disk is not None:
            try:
                self._disk.clear()
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "memory_entries": len(self._memory),
            "disk_available": self._disk is not None,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    global _cache
    if _cache is None:
        from config.settings import MODELS_DIR, CACHE_TTL_SECONDS, CACHE_MAX_SIZE_MB
        cache_dir = str(MODELS_DIR.parent / "cache_store")
        _cache = CacheManager(cache_dir, CACHE_MAX_SIZE_MB, CACHE_TTL_SECONDS)
    return _cache
