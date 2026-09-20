"""Zero-setup store: SQLite + FTS5 keyword search + in-Python cosine on
stored embeddings for the vector half of hybrid retrieval.

This is the default so the system runs on a laptop with nothing installed.
The pgvector adapter has the identical interface for production.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..embeddings import cosine
from ..schema import ItemType, KnowledgeItem, Link, SearchResult


class SQLiteStore:
    def __init__(self, path: str = "data/kb.sqlite3") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT,
                body TEXT,
                source TEXT,
                external_id TEXT,
                timestamp REAL,
                tags TEXT,
                entities TEXT,
                summary TEXT,
                metadata TEXT,
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS links (
                id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS items_fts
                USING fts5(id UNINDEXED, text);
            """
        )
        c.commit()

    # -- writes -------------------------------------------------------------
    def upsert(self, item: KnowledgeItem) -> None:
        c = self.conn
        c.execute(
            """INSERT OR REPLACE INTO items
               (id,type,title,body,source,external_id,timestamp,tags,entities,
                summary,metadata,embedding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item.id, item.type.value, item.title, item.body, item.source,
                item.external_id, item.timestamp,
                json.dumps(item.tags), json.dumps(item.entities),
                item.summary, json.dumps(item.metadata),
                json.dumps(item.embedding) if item.embedding else None,
            ),
        )
        c.execute("DELETE FROM items_fts WHERE id = ?", (item.id,))
        c.execute("INSERT INTO items_fts (id, text) VALUES (?, ?)",
                  (item.id, item.search_text()))
        c.commit()

    def link(self, link: Link) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO links (id,from_id,to_id,relation) VALUES (?,?,?,?)",
            (link.id(), link.from_id, link.to_id, link.relation),
        )
        self.conn.commit()

    # -- reads --------------------------------------------------------------
    def _row_to_item(self, row: sqlite3.Row) -> KnowledgeItem:
        return KnowledgeItem(
            id=row["id"], type=ItemType(row["type"]), title=row["title"],
            body=row["body"], source=row["source"], external_id=row["external_id"],
            timestamp=row["timestamp"], tags=json.loads(row["tags"] or "[]"),
            entities=json.loads(row["entities"] or "[]"), summary=row["summary"],
            metadata=json.loads(row["metadata"] or "{}"),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        )

    def get(self, item_id: str) -> KnowledgeItem | None:
        row = self.conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def search(self, query, query_embedding=None, top_k=12, types=None):
        # Keyword half via FTS5.
        keyword_scores: dict[str, float] = {}
        try:
            fts_q = " OR ".join(t for t in query.replace('"', " ").split() if t) or query
            rows = self.conn.execute(
                "SELECT id, bm25(items_fts) AS rank FROM items_fts "
                "WHERE items_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_q, top_k * 4),
            ).fetchall()
            for i, r in enumerate(rows):
                keyword_scores[r["id"]] = 1.0 / (i + 1)   # rank-normalized
        except sqlite3.OperationalError:
            pass

        # Vector half: cosine against stored embeddings.
        vector_scores: dict[str, float] = {}
        if query_embedding:
            for row in self.conn.execute("SELECT id, embedding FROM items WHERE embedding IS NOT NULL"):
                emb = json.loads(row["embedding"])
                vector_scores[row["id"]] = cosine(query_embedding, emb)

        # Reciprocal-style fusion of the two halves.
        ids = set(keyword_scores) | set(vector_scores)
        fused = {
            i: 0.5 * keyword_scores.get(i, 0.0) + 0.5 * vector_scores.get(i, 0.0)
            for i in ids
        }
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        results: list[SearchResult] = []
        for item_id, score in ordered:
            item = self.get(item_id)
            if item is None:
                continue
            if types and item.type.value not in types:
                continue
            results.append(SearchResult(item=item, score=score))
            if len(results) >= top_k:
                break
        return results

    def neighbors(self, item_id, relation=None):
        if relation:
            rows = self.conn.execute(
                "SELECT to_id FROM links WHERE from_id = ? AND relation = ?",
                (item_id, relation)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT to_id FROM links WHERE from_id = ?", (item_id,)).fetchall()
        out = [self.get(r["to_id"]) for r in rows]
        return [i for i in out if i]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
