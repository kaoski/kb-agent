"""Command-line entry point.

  kb add        -- capture a daily note (the daily-input habit)
  kb ingest-file -- ingest a text/markdown file as one item
  kb search     -- retrieve from the KB (no LLM, offline)
  kb ask        -- run an agent against the KB (needs an API key)
  kb runs       -- show the token/iteration audit
  kb export     -- export an agent spec so it can be cloned
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import AgentSpec, Environment
from .embeddings import build_embedder
from .ingest import Ingestor
from .observability import RunRecorder
from .schema import ItemType, KnowledgeItem
from .store import build_store


def _store_and_embedder(env_name: str):
    env = Environment.load(env_name)
    return build_store(env.store), build_embedder(env.embedder)


def cmd_add(args):
    store, embedder = _store_and_embedder(args.env)
    body = args.body or sys.stdin.read()
    item = KnowledgeItem(type=ItemType(args.type), title=args.title, body=body,
                         source="manual",
                         tags=[t for t in (args.tags or "").split(",") if t])
    item = Ingestor(store, embedder).ingest(item)
    print(f"Saved [{item.type.value}] {item.title}  id={item.id}")


def cmd_ingest_file(args):
    store, embedder = _store_and_embedder(args.env)
    path = Path(args.path)
    item = KnowledgeItem(type=ItemType(args.type), title=args.title or path.stem,
                         body=path.read_text(), source="file",
                         external_id=str(path))
    item = Ingestor(store, embedder).ingest(item)
    print(f"Ingested {path} -> id={item.id}")


def cmd_search(args):
    store, embedder = _store_and_embedder(args.env)
    emb = embedder.embed(args.query)
    types = [t for t in (args.types or "").split(",") if t] or None
    results = store.search(args.query, query_embedding=emb, top_k=args.top_k, types=types)
    if not results:
        print("No matches.")
        return
    for r in results:
        it = r.item
        print(f"[{r.score:.3f}] ({it.type.value}) {it.title}  id={it.id}")
        print(f"        {it.summary or it.body[:120]}")


def cmd_ask(args):
    from .runtime import Agent
    spec = AgentSpec.load(args.agent)
    agent = Agent(spec)
    out = asyncio.run(agent.run(args.prompt))
    print(out["result"])
    r = out["run"]
    print(f"\n--- {r.iterations} iters | {r.total_tokens} tokens "
          f"| ${r.cost_usd:.4f} | {r.duration_s:.1f}s | {r.outcome} ---", file=sys.stderr)


def cmd_runs(args):
    rec = RunRecorder()
    if args.stats and args.agent:
        for row in rec.prompt_stats(args.agent):
            print(f"prompt v{row['prompt_version']}: n={row['n']} "
                  f"avg_iters={row['avg_iters']:.1f} avg_tokens={row['avg_tokens']:.0f} "
                  f"avg_cost=${row['avg_cost']:.4f} ok={row['successes']}/{row['n']}")
        return
    for row in rec.recent(limit=args.limit, agent=args.agent):
        print(f"{row['agent']:16} v{row['prompt_version']} {row['model']:20} "
              f"iters={row['iterations']} tok={row['input_tokens']+row['output_tokens']} "
              f"${row['cost_usd']:.4f} {row['outcome']}")


def cmd_export(args):
    spec = AgentSpec.load(args.agent)
    spec.export(Path(args.out))
    print(f"Exported {args.agent} -> {args.out} (self-contained; clone-ready)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kb", description="Knowledge-base agent")
    p.add_argument("--env", default="local", help="environment config name")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="capture a note")
    a.add_argument("title")
    a.add_argument("body", nargs="?", help="body text, or omit to read stdin")
    a.add_argument("--type", default="note")
    a.add_argument("--tags", default="")
    a.set_defaults(func=cmd_add)

    f = sub.add_parser("ingest-file")
    f.add_argument("path")
    f.add_argument("--title", default="")
    f.add_argument("--type", default="doc")
    f.set_defaults(func=cmd_ingest_file)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=8, dest="top_k")
    s.add_argument("--types", default="")
    s.set_defaults(func=cmd_search)

    k = sub.add_parser("ask", help="run an agent against the KB")
    k.add_argument("agent")
    k.add_argument("prompt")
    k.set_defaults(func=cmd_ask)

    r = sub.add_parser("runs", help="token/iteration audit")
    r.add_argument("--agent", default=None)
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--stats", action="store_true", help="per-prompt-version averages")
    r.set_defaults(func=cmd_runs)

    e = sub.add_parser("export", help="export an agent spec for cloning")
    e.add_argument("agent")
    e.add_argument("out")
    e.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
