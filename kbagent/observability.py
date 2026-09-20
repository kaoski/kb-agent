"""Run recorder: the audit spine.

Every agent run writes one row here with tokens in/out, iteration count
(agentic turns), estimated cost, latency, and outcome. This is what turns
"optimize the prompts" into something concrete: find the prompt versions
with high iterations or token blow-ups and A/B them.

The default backend is a local SQLite table so it works offline. A Langfuse
backend can be dropped in behind the same `record()` call for OpenTelemetry
tracing without changing callers.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    agent: str
    prompt_version: int
    model: str
    account_profile: str
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0            # agentic turns / tool round-trips
    cost_usd: float = 0.0
    duration_s: float = 0.0
    outcome: str = "success"       # success | error | budget_exceeded | max_turns
    prompt: str = ""
    result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RunRecorder:
    def __init__(self, path: str = "data/runs.sqlite3") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, ts REAL, agent TEXT, prompt_version INTEGER,
                model TEXT, account_profile TEXT, input_tokens INTEGER,
                output_tokens INTEGER, iterations INTEGER, cost_usd REAL,
                duration_s REAL, outcome TEXT, prompt TEXT, result TEXT, metadata TEXT
            )""")
        self.conn.commit()

    def record(self, r: RunRecord) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO runs VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r.run_id, r.ts, r.agent, r.prompt_version, r.model, r.account_profile,
             r.input_tokens, r.output_tokens, r.iterations, r.cost_usd, r.duration_s,
             r.outcome, r.prompt, r.result, json.dumps(r.metadata)),
        )
        self.conn.commit()

    def recent(self, limit: int = 20, agent: str | None = None) -> list[dict]:
        if agent:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE agent=? ORDER BY ts DESC LIMIT ?",
                (agent, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def prompt_stats(self, agent: str) -> list[dict]:
        """Per-prompt-version averages: the table you read to pick which
        prompt to keep."""
        rows = self.conn.execute(
            """SELECT prompt_version, COUNT(*) n,
                      AVG(iterations) avg_iters,
                      AVG(input_tokens+output_tokens) avg_tokens,
                      AVG(cost_usd) avg_cost,
                      SUM(outcome='success') successes
               FROM runs WHERE agent=? GROUP BY prompt_version
               ORDER BY prompt_version""",
            (agent,)).fetchall()
        return [dict(r) for r in rows]
