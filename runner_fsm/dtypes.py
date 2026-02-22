from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class CmdResult:
    cmd: str
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False

@dataclass(frozen=True)
class StageResult:
    ok: bool
    results: list[CmdResult]
    failed_index: int | None = None

@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    failed_stage: str | None
    bootstrap: StageResult | None = None
    auth: StageResult | None = None
    tests: StageResult | None = None
    deploy_setup: StageResult | None = None
    deploy_health: StageResult | None = None
    rollout: StageResult | None = None
    evaluation: StageResult | None = None
    benchmark: StageResult | None = None
    metrics_path: str | None = None
    metrics: dict[str, Any] | None = None
    metrics_errors: list[str] | None = None

@dataclass(frozen=True)
class AgentResult:
    assistant_text: str
    raw: Any | None = None
    tool_trace: list[dict[str, Any]] | None = None

class AgentClient(Protocol):
    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        ...
    def close(self) -> None:
        ...
