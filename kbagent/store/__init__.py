"""Store factory: config picks the backend, the rest of the system only
sees the KnowledgeStore interface."""
from __future__ import annotations

from .base import KnowledgeStore
from .sqlite_store import SQLiteStore


def build_store(config: dict) -> KnowledgeStore:
    backend = (config or {}).get("backend", "sqlite")
    if backend == "sqlite":
        return SQLiteStore(path=config.get("path", "data/kb.sqlite3"))
    if backend == "pgvector":
        from .pgvector_store import PgVectorStore
        return PgVectorStore(dsn=config["dsn"], dim=config.get("dim", 1024))
    raise ValueError(f"Unknown store backend: {backend}")


__all__ = ["KnowledgeStore", "SQLiteStore", "build_store"]
