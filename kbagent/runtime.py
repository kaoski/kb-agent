"""The generic agent runtime.

Reads an AgentSpec + its Environment, wires the KB tools and the selected
account's credentials into ClaudeCodeOptions, runs the Agent SDK loop, and
records tokens/iterations/cost from the final ResultMessage. One runtime,
customized entirely by config: prompt, tools, model, and account are all
fields.

Because it takes typed input and returns typed output (per the spec's
io_contract), an instance is directly usable as a node in a multi-agent
system: an orchestrator calls `.run(payload)` on any agent.
"""
from __future__ import annotations

import time

from .config import AgentSpec, Credentials, Environment
from .embeddings import build_embedder
from .observability import RunRecord, RunRecorder
from .store import build_store
from .tools import build_kb_server, build_imc_server, build_cmdb_server



# Registry: tool-name prefix -> builder function.
# Only servers whose prefix appears in the agent spec's tools list are mounted.
_SERVER_REGISTRY = {
    "mcp__kb__":   ("kb",   build_kb_server),
    "mcp__imc__":  ("imc",  build_imc_server),
    "mcp__cmdb__": ("cmdb", build_cmdb_server),
}


class Agent:
    def __init__(self, spec: AgentSpec,
                 recorder: RunRecorder | None = None) -> None:
        self.spec = spec
        self.env = Environment.load(spec.environment)
        self.creds = Credentials.load().get(self.env.credentials_profile)
        self.store = build_store(self.env.store)
        self.embedder = build_embedder(self.env.embedder)
        self.recorder = recorder or RunRecorder(
            path=self.env.observability.get("path", "data/runs.sqlite3"))

    def _options(self):
        from claude_code_sdk import ClaudeCodeOptions

        servers: dict = {}
        tool_names: list[str] = []

        # Mount only the servers whose prefix appears in spec.tools.
        # If spec.tools is empty, fall back to mounting all KB tools.
        spec_tools = self.spec.tools or []
        for prefix, (server_name, builder) in _SERVER_REGISTRY.items():
            if not spec_tools or any(t.startswith(prefix) for t in spec_tools):
                srv, names = builder(self.store, self.embedder)
                servers[server_name] = srv
                tool_names += names

        # Connector tools (ServiceNow, Engine) — added when the environment
        # declares endpoints and the spec requests them.
        if self.env.connectors and any(
            t.startswith("mcp__snow__") for t in spec_tools
        ):
            from .connectors import build_connector_server
            conn_server, conn_names = build_connector_server(self.env.connectors)
            servers["snow"] = conn_server
            tool_names += conn_names

        # Filter to only what the spec explicitly allows.
        allowed = (
            [t for t in tool_names if t in spec_tools]
            if spec_tools else tool_names
        )

        ClaudeCodeOptions(
            system_prompt=self.spec.prompt.text(),
            model=self.spec.model.primary,
            mcp_servers=servers,
            allowed_tools=allowed,
            max_turns=self.spec.policy.max_iterations,
            permission_mode=self.spec.policy.permission_mode,
            env=self.creds.as_env(),
        )

    async def run(self, prompt: str) -> dict:
        """Run one task. Returns the result text plus the audit record."""
        from claude_code_sdk import query, AssistantMessage, TextBlock, ResultMessage

        options = self._options()
        started = time.time()
        text_out: list[str] = []
        rec = RunRecord(
            agent=self.spec.name, prompt_version=self.spec.prompt.version,
            model=self.spec.model.primary,
            account_profile=self.env.credentials_profile, prompt=prompt,
        )

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_out.append(block.text)
            elif isinstance(message, ResultMessage):
                usage = getattr(message, "usage", None) or {}
                rec.input_tokens = _u(usage, "input_tokens")
                rec.output_tokens = _u(usage, "output_tokens")
                rec.iterations = getattr(message, "num_turns", 0) or 0
                rec.cost_usd = getattr(message, "total_cost_usd", 0.0) or 0.0
                subtype = getattr(message, "subtype", "success")
                rec.outcome = "success" if subtype == "success" else subtype

        rec.result = "\n".join(text_out)
        rec.duration_s = time.time() - started
        self.recorder.record(rec)
        return {"result": rec.result, "run": rec}


def _short(qualified: str) -> str:
    return qualified.split("__")[-1]


def _u(usage, key: str) -> int:
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)