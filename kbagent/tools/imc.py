"""IMC Connector tools — scoped to source='imc-connector', type='code'."""
from __future__ import annotations

from typing import Any

from ..embeddings import Embedder
from ..store.base import KnowledgeStore


def build_imc_server(store: KnowledgeStore, embedder: Embedder):
    from claude_code_sdk import tool, create_sdk_mcp_server

    @tool(
        "search_imc",
        "Search the IMC Connector source code. "
        "Use for Script Includes, Business Rules, Client Scripts, UI Policies, "
        "Scheduled Scripts, Processors, and UI Actions in the IMC connector.",
        {"query": str, "top_k": int},
    )
    async def search_imc(args: dict[str, Any]) -> dict[str, Any]:
        q = args["query"]
        top_k = int(args.get("top_k") or 6)
        emb = embedder.embed(q)
        results = store.search(q, query_embedding=emb, top_k=top_k, types=["code"])
        results = [r for r in results if r.item.source == "imc-connector"]
        if not results:
            return {"content": [{"type": "text", "text": "No IMC matches."}]}
        lines = []
        for r in results:
            it = r.item
            lines.append(
                f"[score={r.score:.3f}] {it.title} (id={it.id})\n"
                f"  {it.summary or it.body[:300]}"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @tool(
        "get_imc",
        "Fetch a single IMC Connector file in full by its id. "
        "Always call this after search_imc to read the complete source.",
        {"id": str},
    )
    async def get_imc(args: dict[str, Any]) -> dict[str, Any]:
        item = store.get(args["id"])
        if not item or item.source != "imc-connector":
            return {"content": [{"type": "text", "text": "IMC item not found."}]}
        text = f"{item.title}\n\n{item.body}"
        return {"content": [{"type": "text", "text": text}]}

    server = create_sdk_mcp_server(name="imc", version="0.1.0",
                                   tools=[search_imc, get_imc])
    tool_names = ["mcp__imc__search_imc", "mcp__imc__get_imc"]
    return server, tool_names