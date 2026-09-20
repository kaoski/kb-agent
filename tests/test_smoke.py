"""Offline smoke test: proves ingest + hybrid retrieval + linking work with
no API key and no external services. This is the walking skeleton's spine."""
import os
import tempfile

from kbagent.embeddings import build_embedder
from kbagent.ingest import Ingestor
from kbagent.observability import RunRecorder, RunRecord
from kbagent.schema import ItemType, KnowledgeItem, Link
from kbagent.store import build_store


def test_ingest_search_link_audit():
    tmp = tempfile.mkdtemp()
    store = build_store({"backend": "sqlite", "path": f"{tmp}/kb.sqlite3"})
    embedder = build_embedder({"backend": "hash", "dim": 128})
    ing = Ingestor(store, embedder)

    bug = ing.ingest(KnowledgeItem(
        type=ItemType.BUG, title="CMDB IRE dedup regression",
        body="Identification and Reconciliation Engine created duplicate CI "
             "records after the coalesce change. GlideRecord query returned "
             "stale sys_ids.", source="servicenow", external_id="INC0012345"))
    rel = ing.ingest(KnowledgeItem(
        type=ItemType.RELEASE, title="V6 hotfix 6.2.1",
        body="Fixed the CMDB IRE duplicate CI issue by restoring the prior "
             "coalesce rule.", source="git", external_id="v6.2.1"))
    ing.ingest(KnowledgeItem(
        type=ItemType.CUSTOMER, title="Acme feedback on 6.2.1",
        body="Customer confirmed duplicates stopped after upgrade. Happy.",
        source="manual"))

    store.link(Link(from_id=bug.id, to_id=rel.id, relation="fixed_by"))

    # Keyword + vector retrieval finds the bug.
    emb = embedder.embed("duplicate CI records IRE")
    results = store.search("duplicate CI records IRE", query_embedding=emb, top_k=5)
    assert results, "search returned nothing"
    assert any(r.item.id == bug.id for r in results)

    # Type filter works.
    rel_only = store.search("coalesce", query_embedding=embedder.embed("coalesce"),
                            top_k=5, types=["release"])
    assert all(r.item.type == ItemType.RELEASE for r in rel_only)

    # Link traversal works.
    neigh = store.neighbors(bug.id, relation="fixed_by")
    assert neigh and neigh[0].id == rel.id

    # Audit recorder works.
    rec = RunRecorder(path=f"{tmp}/runs.sqlite3")
    rec.record(RunRecord(agent="kb_assistant", prompt_version=1,
                         model="claude-sonnet-4-5", account_profile="personal",
                         input_tokens=1200, output_tokens=300, iterations=3,
                         cost_usd=0.006))
    stats = rec.prompt_stats("kb_assistant")
    assert stats and stats[0]["n"] == 1

    print("OK:", store.count(), "items,", len(results), "hits")


if __name__ == "__main__":
    test_ingest_search_link_audit()
