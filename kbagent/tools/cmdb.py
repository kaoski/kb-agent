"""CMDB Connector tools — scoped to source='cmdb-connector', type='code'."""
from __future__ import annotations

from typing import Any

from ..embeddings import Embedder
from ..store.base import KnowledgeStore


def build_cmdb_server(store: KnowledgeStore, embedder: Embedder):
    from claude_code_sdk import tool, create_sdk_mcp_server

    @tool(
        "search_cmdb",
        "Search the CMDB Connector source code. "
        "Use for IRE pipelines, CMDB population scripts, and related components.",
        {"query": str, "top_k": int},
    )
    async def search_cmdb(args: dict[str, Any]) -> dict[str, Any]:
        q = args["query"]
        top_k = int(args.get("top_k") or 6)
        emb = embedder.embed(q)
        results = store.search(q, query_embedding=emb, top_k=top_k, types=["code"])
        results = [r for r in results if r.item.source == "cmdb-connector"]
        if not results:
            return {"content": [{"type": "text", "text": "No CMDB matches."}]}
        lines = []
        for r in results:
            it = r.item
            lines.append(
                f"[score={r.score:.3f}] {it.title} (id={it.id})\n"
                f"  {it.summary or it.body[:300]}"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @tool(
        "get_cmdb",
        "Fetch a single CMDB Connector file in full by its id. "
        "Always call this after search_cmdb to read the complete source.",
        {"id": str},
    )
    async def get_cmdb(args: dict[str, Any]) -> dict[str, Any]:
        item = store.get(args["id"])
        if not item or item.source != "cmdb-connector":
            return {"content": [{"type": "text", "text": "CMDB item not found."}]}
        text = f"{item.title}\n\n{item.body}"
        return {"content": [{"type": "text", "text": text}]}

    server = create_sdk_mcp_server(name="cmdb", version="0.1.0",
                                   tools=[search_cmdb, get_cmdb])
    tool_names = ["mcp__cmdb__search_cmdb", "mcp__cmdb__get_cmdb"]
    return server, tool_names