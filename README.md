# kb-agent

A generic, config-driven knowledge-base agent built on the Claude Agent SDK.
It maintains a knowledge base of your product, code, docs, procedures, bug
tickets, past issues, release notes, customer responses, and brainstorms, and
answers from it. One runtime; every agent is customized by **prompt, tools,
and model** in a YAML spec. Every run is **audited** for tokens and iterations
so you can optimize prompts. The store is **pluggable** (SQLite now, pgvector
for production, Elastic/Redis later behind the same interface). A spec can be
**exported to clone** an identical agent.

## Design in one line

An agent is a versioned spec, not a class. So:

| Your requirement            | How it's met                                             |
|-----------------------------|----------------------------------------------------------|
| Customize by prompt         | `prompt.system_ref` + `version` in the agent YAML        |
| Customize by tools          | `tools:` list in the agent YAML (`allowed_tools`)        |
| Customize by model          | `model.primary` / `model.fallback` in the agent YAML     |
| Audit tokens & iterations   | `RunRecorder` writes one row per run; `kb runs --stats`  |
| Pluggable database          | `KnowledgeStore` interface; `store.backend` in env YAML  |
| Export & clone an agent     | `kb export <agent> out.yaml` (prompt inlined)            |
| Plug into a multi-agent sys | typed `io_contract`; `Agent.run(payload)` is a node      |
| Pick which account to use   | credentials profiles; `credentials_profile` per env      |

## Layout

```
kbagent/
  schema.py          KnowledgeItem, Link  (one normalized model for all sources)
  config.py          AgentSpec, Environment, Credentials (account selection)
  embeddings.py      HashEmbedder (offline) + LiteLLMEmbedder
  store/             KnowledgeStore interface + SQLite and pgvector adapters
  ingest.py          source -> normalize -> enrich -> embed -> store + link
  tools.py           KB tools exposed to the agent as an in-process MCP server
  observability.py   per-run token/iteration/cost audit
  runtime.py         wires a spec + account into the Agent SDK loop
  cli.py             kb add / ingest-file / search / ask / runs / export
config/
  agents/            one YAML per agent
  environments/      one YAML per environment (store + account + observability)
  prompts/           versioned system prompts
  credentials.example.yaml
```

## Quick start (offline, no key)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # only PyYAML needed for the offline path

# capture knowledge
python -m kbagent.cli add "IRE dedup fix" "Restored coalesce rule in 6.2.1" --type release

# retrieve (hybrid keyword + vector, runs locally)
python -m kbagent.cli search "coalesce duplicate CI"

# smoke test the whole spine
python tests/test_smoke.py
```

## Running an agent (needs an account)

```bash
pip install claude-agent-sdk
cp config/credentials.example.yaml config/credentials.yaml   # edit profiles
export ANTHROPIC_API_KEY_PERSONAL=sk-ant-...

python -m kbagent.cli ask kb_assistant "What did we ship for the CMDB IRE regression?"
python -m kbagent.cli runs --agent kb_assistant --stats     # prompt-optimization view
```

## Choosing which account to use

Credentials never live in an agent spec. An agent names an environment, the
environment names a credentials **profile**, the profile resolves to an API
key from an env var at runtime:

```
agent (kb_assistant.yaml)  ->  environment: local
environment (local.yaml)   ->  credentials_profile: personal
credentials.yaml           ->  personal: { api_key_env: ANTHROPIC_API_KEY_PERSONAL }
```

Pin heavy or production agents to a higher-tier account, and run prompt
experiments on another, by pointing their environments at different profiles.
A profile may also carry a `workspace` (to bill a workspace within an account)
or a `base_url` (to route through a gateway such as LiteLLM).

## Growing it

- **Real embeddings:** set `embedder.backend: litellm` in the env.
- **Production store:** set `store.backend: pgvector` and a DSN. Same interface.
- **Tracing:** point observability at Langfuse (OpenTelemetry) behind `record()`.
- **More sources:** add ingestion adapters (ServiceNow, git) that emit
  `KnowledgeItem`s into the same `Ingestor`.
- **Multi-agent:** give a supervisor agent the sub-agents as tools; each honors
  its `io_contract`.
```
