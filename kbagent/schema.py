"""Core data model.

Everything that enters the knowledge base is normalized into a single
`KnowledgeItem`, regardless of source (a daily note, a ServiceNow ticket,
a git commit, a doc). One schema in the middle is what lets many sources
and many stores plug in behind one interface.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ItemType(str, Enum):
    """The kinds of knowledge the base holds.

    Extend this freely; the store does not care about the values, it only
    filters on them. These mirror the categories you listed.
    """

    NOTE = "note"                # daily brain-dump / brainstorm
    DOC = "doc"                  # product / design documentation
    PROCEDURE = "procedure"      # runbook, how-to, standard operating procedure
    CODE = "code"                # code snippet or file summary
    BUG = "bug"                  # bug ticket
    ISSUE = "issue"              # incident / event / issue
    RELEASE = "release"          # release note / update
    CUSTOMER = "customer"        # customer response / business impact
    CONTEXT = "context"          # conversation / session context


@dataclass
class KnowledgeItem:
    """One atomic unit of knowledge.

    `id` is deterministic from source + external_id (or body hash) so that
    re-ingesting the same source updates in place instead of duplicating.
    """

    type: ItemType
    title: str
    body: str
    source: str = "manual"                       # where it came from: manual, servicenow, git, ...
    external_id: str | None = None               # id in the source system (ticket number, sha)
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)   # people, systems, components mentioned
    summary: str | None = None                   # one-line, filled by the enrich step
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = ItemType(self.type)
        if not self.id:
            self.id = self.compute_id()

    def compute_id(self) -> str:
        basis = f"{self.source}:{self.external_id}" if self.external_id else self.body
        digest = hashlib.sha1(f"{self.type.value}:{basis}".encode()).hexdigest()
        return f"{self.type.value}_{digest[:16]}"

    def search_text(self) -> str:
        """The text used for keyword search and embedding."""
        parts = [self.title, self.body]
        if self.summary:
            parts.append(self.summary)
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.entities:
            parts.append(" ".join(self.entities))
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KnowledgeItem":
        d = dict(d)
        for k in ("tags", "entities", "metadata"):
            if isinstance(d.get(k), str):
                try:
                    d[k] = json.loads(d[k])
                except (ValueError, TypeError):
                    d[k] = [] if k != "metadata" else {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Link:
    """A typed edge between two items.

    This is the layer that turns plain RAG into something that understands
    your work: a BUG links to the RELEASE that fixed it, which links to the
    CUSTOMER response. Traversal beats guessing.
    """

    from_id: str
    to_id: str
    relation: str          # e.g. "fixed_by", "caused", "relates_to", "shipped_in"

    def id(self) -> str:
        return hashlib.sha1(f"{self.from_id}|{self.relation}|{self.to_id}".encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    item: KnowledgeItem
    score: float
