"""The one interface every knowledge store implements.

Deferring the polyglot-persistence decision is deliberate: start on one
store (SQLite here, pgvector for production) behind this interface, and add
Elasticsearch / Redis / a graph DB later as a new adapter, a config change
rather than a rewrite. Hybrid retrieval (keyword + vector) and the link
graph both live behind these six methods.
"""
from __future__ import annotations

from typing import Protocol

from ..schema import KnowledgeItem, Link, SearchResult


class KnowledgeStore(Protocol):
    def upsert(self, item: KnowledgeItem) -> None:
        """Insert or replace an item by id."""

    def get(self, item_id: str) -> KnowledgeItem | None: ...

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 12,
        types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: keyword + vector, optionally filtered by type."""

    def link(self, link: Link) -> None:
        """Record a typed edge between two items."""

    def neighbors(self, item_id: str, relation: str | None = None) -> list[KnowledgeItem]:
        """Traverse the link graph out of an item."""

    def count(self) -> int: ...
