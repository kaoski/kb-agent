"""The generic agent runtime.

Reads an AgentSpec + its Environment, wires the KB tools and the selected
account's credentials into ClaudeAgentOptions, runs the Agent SDK loop, and
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
from .tools import build_kb_server


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
        from claude_agent_sdk import ClaudeAgentOptions

        server, kb_tool_names = build_kb_server(self.store, self.embedder)
        # An agent may narrow the KB tools it uses via its spec; default to all.
        allowed = [t for t in kb_tool_names
                   if not self.spec.tools or _short(t) in self.spec.tools] or kb_tool_names

        return ClaudeAgentOptions(
            system_prompt=self.spec.prompt.text(),
            model=self.spec.model.primary,
            fallback_model=self.spec.model.fallback,
            mcp_servers={"kb": server},
            allowed_tools=allowed,
            max_turns=self.spec.policy.max_iterations,
            max_budget_usd=self.spec.policy.max_budget_usd,
            permission_mode=self.spec.policy.permission_mode,
            # Account selection: the profile's key/workspace/base-url override
            # the SDK subprocess environment for this run only.
            env=self.creds.as_env(),
        )

    async def run(self, prompt: str) -> dict:
        """Run one task. Returns the result text plus the audit record."""
        from claude_agent_sdk import query, AssistantMessage, TextBlock, ResultMessage

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
                # The audit numbers come off the terminal ResultMessage.
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
    # mcp__kb__search -> search
    return qualified.split("__")[-1]


def _u(usage, key: str) -> int:
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)
