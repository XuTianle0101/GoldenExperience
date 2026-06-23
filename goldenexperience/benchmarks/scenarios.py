"""Benchmark scenario descriptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    prompt_tokens: int
    decode_tokens: int
    shared_prefix_ratio: float
    batch_size: int


LONG_CONTEXT_QA = Scenario(
    name="long_context_qa",
    prompt_tokens=32768,
    decode_tokens=256,
    shared_prefix_ratio=0.15,
    batch_size=1,
)

MULTI_TURN_CHAT = Scenario(
    name="multi_turn_chat",
    prompt_tokens=8192,
    decode_tokens=512,
    shared_prefix_ratio=0.70,
    batch_size=8,
)

RAG_PREFIX_SHARING = Scenario(
    name="rag_prefix_sharing",
    prompt_tokens=16384,
    decode_tokens=256,
    shared_prefix_ratio=0.85,
    batch_size=16,
)

AGENT_WORKFLOW = Scenario(
    name="agent_workflow",
    prompt_tokens=12288,
    decode_tokens=1024,
    shared_prefix_ratio=0.60,
    batch_size=4,
)

