"""Production store: Postgres + pgvector.

Same interface as SQLiteStore. Postgres covers hybrid search (BM25-style
full-text + dense vectors) and the relational link graph in one engine, so
this is the single store to grow into before reaching for Elasticsearch or
a dedicated vector DB. psycopg is imported lazily so the offline path never
needs it installed.
"""
from __future__ import annotations

import json

from ..schema import ItemType, KnowledgeItem, Link, SearchResult

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    title TEXT,
    body TEXT,
    source TEXT,
    external_id TEXT,
    timestamp DOUBLE PRECISION,
    tags JSONB,
    entities JSONB,
    summary TEXT,
    metadata JSONB,
    tsv tsvector,
    embedding vector(%(dim)s)
);
CREATE INDEX IF NOT EXISTS idx_items_tsv ON items USING gin(tsv);
CREATE INDEX IF NOT EXISTS idx_items_type ON items(type);
CREATE TABLE IF NOT EXISTS links (
    id TEXT PRIMARY KEY,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_id);
"""


class PgVectorStore:
    def __init__(self, dsn: str, dim: int = 1024) -> None:
        import psycopg
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute(_SCHEMA % {"dim": dim})

    def upsert(self, item: KnowledgeItem) -> None:
        emb = item.embedding or [0.0] * self.dim
        self.conn.execute(
            """INSERT INTO items
               (id,type,title,body,source,external_id,timestamp,tags,entities,
                summary,metadata,tsv,embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                       to_tsvector('english', %s), %s)
               ON CONFLICT (id) DO UPDATE SET
                 title=EXCLUDED.title, body=EXCLUDED.body, summary=EXCLUDED.summary,
                 tags=EXCLUDED.tags, entities=EXCLUDED.entities,
                 metadata=EXCLUDED.metadata, tsv=EXCLUDED.tsv,
                 embedding=EXCLUDED.embedding""",
            (item.id, item.type.value, item.title, item.body, item.source,
             item.external_id, item.timestamp, json.dumps(item.tags),
             json.dumps(item.entities), item.summary, json.dumps(item.metadata),
             item.search_text(), str(emb)),
        )

    def link(self, link: Link) -> None:
        self.conn.execute(
            """INSERT INTO links (id,from_id,to_id,relation) VALUES (%s,%s,%s,%s)
               ON CONFLICT (id) DO NOTHING""",
            (link.id(), link.from_id, link.to_id, link.relation),
        )

    def _row_to_item(self, r) -> KnowledgeItem:
        return KnowledgeItem(
            id=r[0], type=ItemType(r[1]), title=r[2], body=r[3], source=r[4],
            external_id=r[5], timestamp=r[6], tags=r[7] or [], entities=r[8] or [],
            summary=r[9], metadata=r[10] or {},
        )

    _COLS = "id,type,title,body,source,external_id,timestamp,tags,entities,summary,metadata"

    def get(self, item_id: str) -> KnowledgeItem | None:
        r = self.conn.execute(
            f"SELECT {self._COLS} FROM items WHERE id=%s", (item_id,)).fetchone()
        return self._row_to_item(r) if r else None

    def search(self, query, query_embedding=None, top_k=12, types=None):
        type_filter = ""
        params: list = []
        if types:
            type_filter = "WHERE type = ANY(%s)"
            params.append(list(types))

        # Fuse full-text rank and vector distance in SQL.
        emb = str(query_embedding) if query_embedding else str([0.0] * self.dim)
        sql = f"""
            SELECT {self._COLS},
                   0.5 * ts_rank(tsv, plainto_tsquery('english', %s))
                 + 0.5 * (1 - (embedding <=> %s)) AS score
            FROM items {type_filter}
            ORDER BY score DESC
            LIMIT %s
        """
        rows = self.conn.execute(sql, [query, emb, *params, top_k]).fetchall()
        return [SearchResult(item=self._row_to_item(r[:-1]), score=r[-1]) for r in rows]

    def neighbors(self, item_id, relation=None):
        if relation:
            rows = self.conn.execute(
                "SELECT to_id FROM links WHERE from_id=%s AND relation=%s",
                (item_id, relation)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT to_id FROM links WHERE from_id=%s", (item_id,)).fetchall()
        out = [self.get(r[0]) for r in rows]
        return [i for i in out if i]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
