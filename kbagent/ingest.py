"""Ingestion pipeline: source -> normalize -> enrich -> embed -> store + link.

Manual capture (the daily habit) and automatic pulls (ServiceNow, git) both
land here as KnowledgeItems. The enrich step is where an LLM would extract
entities, tags, a summary, and candidate links; it is pluggable and defaults
to a cheap heuristic so the pipeline runs offline.
"""
from __future__ import annotations

import re
from typing import Callable

from .embeddings import Embedder
from .schema import KnowledgeItem, Link
from .store.base import KnowledgeStore

# An enricher takes a raw item and returns it with summary/tags/entities filled.
Enricher = Callable[[KnowledgeItem], KnowledgeItem]

_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is",
         "are", "was", "with", "this", "that", "it", "we", "i", "as", "at",
         "be", "by", "from", "has", "have"}


def heuristic_enricher(item: KnowledgeItem) -> KnowledgeItem:
    """No-LLM fallback: first sentence as summary, capitalized tokens as
    entities, frequent words as tags. Swap for an LLM enricher in production."""
    if not item.summary:
        first = re.split(r"(?<=[.!?])\s", item.body.strip(), maxsplit=1)[0]
        item.summary = first[:200]
    if not item.entities:
        item.entities = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", item.body)))[:10]
    if not item.tags:
        words = [w for w in re.findall(r"[a-z]{4,}", item.body.lower()) if w not in _STOP]
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        item.tags = [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]]
    return item


class Ingestor:
    def __init__(self, store: KnowledgeStore, embedder: Embedder,
                 enricher: Enricher = heuristic_enricher) -> None:
        self.store = store
        self.embedder = embedder
        self.enricher = enricher

    def ingest(self, item: KnowledgeItem, links: list[Link] | None = None) -> KnowledgeItem:
        item = self.enricher(item)
        item.embedding = self.embedder.embed(item.search_text())
        self.store.upsert(item)
        for link in links or []:
            self.store.link(link)
        return item

    def ingest_many(self, items: list[KnowledgeItem]) -> list[KnowledgeItem]:
        return [self.ingest(i) for i in items]
