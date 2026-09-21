"""Declarative configuration layer.

The whole design rests on one idea: an agent is a versioned spec (a YAML
file), not a Python class. Customizing by prompt, tool, or model is editing
fields. Exporting and cloning is copying the file. Plugging into a
multi-agent system is honoring the io_contract.

Three config objects, resolved in order:
  AgentSpec       -> what the agent is (prompt, tools, model, policy)
  Environment     -> where it runs (which store, which credentials profile)
  CredentialsProfile -> which account/key/workspace to bill and rate-limit against

Credentials are never written into an agent spec. The spec names an
environment, the environment names a profile, the profile resolves to an
API key at runtime from an env var. That is how "which account to use"
stays a one-line switch and never leaks a secret into version control.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(os.environ.get("KB_CONFIG_ROOT", "config"))


# --------------------------------------------------------------------------- #
# Credentials / accounts
# --------------------------------------------------------------------------- #
@dataclass
class CredentialsProfile:
    """One Anthropic account (or workspace within one)."""

    name: str
    api_key_env: str = "ANTHROPIC_API_KEY"   # name of the env var holding the key
    workspace: str | None = None             # anthropic-workspace-id, if this is a workspace
    base_url: str | None = None              # for a gateway / proxy in front of the API

    def resolve_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Profile '{self.name}' expects the API key in ${self.api_key_env}, "
                f"but that variable is not set."
            )
        return key

    def as_env(self) -> dict[str, str]:
        """Environment overrides handed to the Agent SDK for this run.

        The SDK subprocess reads ANTHROPIC_API_KEY (and base URL / workspace)
        from its environment, so selecting an account is just building this
        dict per run.
        """
        env = {"ANTHROPIC_API_KEY": self.resolve_key()}
        if self.base_url:
            env["ANTHROPIC_BASE_URL"] = self.base_url
        if self.workspace:
            env["ANTHROPIC_WORKSPACE_ID"] = self.workspace
        return env


@dataclass
class Credentials:
    profiles: dict[str, CredentialsProfile] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Credentials":
        path = path or CONFIG_ROOT / "credentials.yaml"
        if not path.exists():
            # Fall back to a single default profile off ANTHROPIC_API_KEY.
            return cls(profiles={"default": CredentialsProfile(name="default")})
        raw = yaml.safe_load(path.read_text()) or {}
        profiles = {
            name: CredentialsProfile(name=name, **spec)
            for name, spec in (raw.get("profiles") or {}).items()
        }
        return cls(profiles=profiles)

    def get(self, name: str) -> CredentialsProfile:
        if name not in self.profiles:
            raise KeyError(f"No credentials profile named '{name}'. "
                           f"Known: {', '.join(self.profiles) or '(none)'}")
        return self.profiles[name]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

@dataclass
class Environment:
    """Where an agent runs: which store backs it, which account bills it."""

    name: str
    credentials_profile: str = "default"
    store: dict[str, Any] = field(default_factory=lambda: {"backend": "sqlite"})
    embedder: dict[str, Any] = field(default_factory=lambda: {"backend": "hash"})
    observability: dict[str, Any] = field(default_factory=lambda: {"backend": "sqlite"})
    # Optional domain-connector endpoints (ServiceNow). When
    # present, the runtime exposes the connector tools to the agent.
    connectors: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str) -> "Environment":
        path = CONFIG_ROOT / "environments" / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No environment config at {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        raw.pop("name", None)
        return cls(name=name, **raw)


# --------------------------------------------------------------------------- #
# Agent spec
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    primary: str = "claude-sonnet-4-5"
    fallback: str | None = None


@dataclass
class PromptConfig:
    system_ref: str | None = None      # path (relative to config/) to a prompt file
    inline: str | None = None          # or an inline system prompt
    version: int = 1

    def text(self) -> str:
        if self.inline:
            return self.inline
        if self.system_ref:
            return (CONFIG_ROOT / self.system_ref).read_text()
        return "You are a helpful knowledge-base assistant."


@dataclass
class PolicyConfig:
    max_iterations: int = 8            # maps to Agent SDK max_turns
    max_budget_usd: float | None = None
    permission_mode: str = "default"


@dataclass
class IOContract:
    """Typed input/output. This is what makes an agent pluggable into a
    multi-agent system: an orchestrator can call any agent that declares
    what it takes and returns."""

    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    name: str
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    tools: list[str] = field(default_factory=list)     # tool names this agent may use
    retrieval: dict[str, Any] = field(default_factory=dict)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    io_contract: IOContract = field(default_factory=IOContract)
    environment: str = "local"

    @classmethod
    def load(cls, name: str) -> "AgentSpec":
        path = CONFIG_ROOT / "agents" / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No agent spec at {path}")
        return cls.from_dict(yaml.safe_load(path.read_text()) or {})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AgentSpec":
        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            model=ModelConfig(**(raw.get("model") or {})),
            prompt=PromptConfig(**(raw.get("prompt") or {})),
            tools=raw.get("tools") or [],
            retrieval=raw.get("retrieval") or {},
            policy=PolicyConfig(**(raw.get("policy") or {})),
            io_contract=IOContract(**(raw.get("io_contract") or {})),
            environment=raw.get("environment", "local"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": {"primary": self.model.primary, "fallback": self.model.fallback},
            "prompt": {"system_ref": self.prompt.system_ref,
                       "inline": self.prompt.inline,
                       "version": self.prompt.version},
            "tools": self.tools,
            "retrieval": self.retrieval,
            "policy": {"max_iterations": self.policy.max_iterations,
                       "max_budget_usd": self.policy.max_budget_usd,
                       "permission_mode": self.policy.permission_mode},
            "io_contract": {"input": self.io_contract.input,
                            "output": self.io_contract.output},
            "environment": self.environment,
        }

    def export(self, path: Path) -> None:
        """Write a self-contained spec so another operator can clone this
        agent and get identical behavior (prompt text is inlined so the
        clone carries no external references)."""
        d = self.to_dict()
        d["prompt"]["inline"] = self.prompt.text()
        d["prompt"]["system_ref"] = None
        path.write_text(yaml.safe_dump(d, sort_keys=False))
