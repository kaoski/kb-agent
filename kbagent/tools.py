"""Knowledge-base tools, exposed to the agent as an in-process MCP server.

`@tool` + `create_sdk_mcp_server` is the Agent SDK's clean way to give an
agent custom capabilities. These functions close over a live store and
embedder, so the agent searches and writes to the same KB the ingestion
pipeline fills. Which tools an agent may call is set per-agent in its spec
(`allowed_tools`), so "customize by tools" is a config edit.
"""
from __future__ import annotations

from typing import Any

from .embeddings import Embedder
from .ingest import Ingestor
from .schema import ItemType, KnowledgeItem, Link
from .store.base import KnowledgeStore


def build_kb_server(store: KnowledgeStore, embedder: Embedder):
    """Return an McpSdkServerConfig plus the list of fully-qualified tool
    names (for allowed_tools). Imports the SDK lazily so non-agent paths
    (ingest, search from the CLI) never need it installed."""
    from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

    ingestor = Ingestor(store, embedder)

    @tool(
        "search",
        "Search the knowledge base (bugs, docs, procedures, releases, "
        "customer responses, past notes). Returns the most relevant items.",
        {"query": str, "top_k": int, "types": str},
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
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
                f"  {it.summary or it.body[:160]}"
            )
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @tool(
        "get",
        "Fetch one knowledge item in full by its id, and its linked items.",
        {"id": str},
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    async def get(args: dict[str, Any]) -> dict[str, Any]:
        item = store.get(args["id"])
        if not item:
            return {"content": [{"type": "text", "text": f"No item with id {args['id']}."}]}
        neigh = store.neighbors(item.id)
        text = f"[{item.type.value}] {item.title}\n\n{item.body}\n"
        if item.tags:
            text += f"\ntags: {', '.join(item.tags)}"
        if neigh:
            text += "\n\nlinked:\n" + "\n".join(f"  - {n.type.value}: {n.title} (id={n.id})"
                                                 for n in neigh)
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "write_note",
        "Save a new note, decision, or piece of knowledge to the base so it "
        "persists for future sessions. Use type: note, doc, procedure, bug, "
        "issue, release, customer, or context.",
        {"title": str, "body": str, "type": str, "tags": str},
        annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False),
    )
    async def write_note(args: dict[str, Any]) -> dict[str, Any]:
        try:
            itype = ItemType(args.get("type") or "note")
        except ValueError:
            itype = ItemType.NOTE
        tags = [t.strip() for t in (args.get("tags") or "").split(",") if t.strip()]
        item = KnowledgeItem(type=itype, title=args["title"], body=args["body"],
                             source="agent", tags=tags)
        item = ingestor.ingest(item)
        return {"content": [{"type": "text", "text": f"Saved (id={item.id})."}]}

    @tool(
        "link",
        "Record a typed relationship between two items, e.g. relation "
        "'fixed_by', 'caused', 'shipped_in', 'relates_to'.",
        {"from_id": str, "to_id": str, "relation": str},
        annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False),
    )
    async def link(args: dict[str, Any]) -> dict[str, Any]:
        store.link(Link(from_id=args["from_id"], to_id=args["to_id"],
                        relation=args["relation"]))
        return {"content": [{"type": "text", "text": "Linked."}]}

    server = create_sdk_mcp_server(name="kb", version="0.1.0",
                                   tools=[search, get, write_note, link])
    tool_names = [f"mcp__kb__{n}" for n in ("search", "get", "write_note", "link")]
    return server, tool_names
