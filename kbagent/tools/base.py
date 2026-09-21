"""Knowledge-base tools, exposed to the agent as an in-process MCP server."""
from __future__ import annotations

from typing import Any

from ..embeddings import Embedder
from ..ingest import Ingestor
from ..schema import ItemType, KnowledgeItem, Link
from ..store.base import KnowledgeStore


def build_kb_server(store: KnowledgeStore, embedder: Embedder):
    from claude_code_sdk import tool, create_sdk_mcp_server

    ingestor = Ingestor(store, embedder)

    @tool(
        "search",
        "Search the knowledge base (bugs, docs, procedures, releases, "
        "customer responses, past notes). Returns the most relevant items.",
        {"query": str, "top_k": int, "types": str},
    )
    async def search(args: dict[str, Any]) -> dict[str, Any]:
        q = args["query"]
        top_k = int(args.get("top_k") or 8)
        types = [t.strip() for t in (args.get("types") or "").split(",") if t.strip()] or None
        emb = embedder.embed(q)
        results = store.search(q, query_embedding=emb, top_k=top_k, types=types)
        if not results:
            return {"content": [{"type": "text", "text": "No matching items."}]}
        lines = []
        for r in results:
            it = r.item
            lines.append(
                f"[{it.type.value}] {it.title} (id={it.id}, score={r.score:.3f})\n"
                f"  {it.summary or it.body[:400]}"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @tool(
        "get",
        "Fetch one knowledge item in full by its id, and its linked items.",
        {"id": str},
    )
    async def get(args: dict[str, Any]) -> dict[str, Any]:
        item = store.get(args["id"])
        if not item:
            return {"content": [{"type": "text", "text": f"No item with id {args['id']}."}]}
        neigh = store.neighbors(item.id)
        text = f"[{item.type.value}] {item.title}\n\n{item.body}\n"
        if item.tags:
            text += f"\ntags: {', '.join(item.tags)}"