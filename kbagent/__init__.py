"""kbagent: a generic, config-driven knowledge-base agent on the Claude Agent SDK.

Customize an agent by prompt, tools, and model (its spec). Audit tokens and
iterations per run (observability). Swap the store (sqlite -> pgvector ->
elastic/redis) behind one interface. Export a spec to clone an agent.
"""
from .schema import KnowledgeItem, ItemType, Link, SearchResult
from .config import AgentSpec, Environment, Credentials, CredentialsProfile
from .embeddings import build_embedder
from .store import build_store
from .ingest import Ingestor
from .observability import RunRecorder, RunRecord

__all__ = [
    "KnowledgeItem", "ItemType", "Link", "SearchResult",
    "AgentSpec", "Environment", "Credentials", "CredentialsProfile",
    "build_embedder", "build_store", "Ingestor", "RunRecorder", "RunRecord",
]
__version__ = "0.1.0"
