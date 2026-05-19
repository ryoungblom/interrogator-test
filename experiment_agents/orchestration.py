"""Non-streaming wrappers around streaming.py. Batch harness uses these to
just get an EpisodeResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marketplace import Marketplace, QueryView

from .streaming import (
    stream_solo,
    stream_competitive_no_verifier,
    stream_competitive_with_verifier,
)


@dataclass
class EpisodeResult:
    condition: str
    query_id: str
    recommended_product_id: str
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    trace_events: list[dict[str, Any]] = field(default_factory=list)


async def _drain_stream(
    condition: str,
    query: QueryView,
    stream,
) -> EpisodeResult:
    trace: list[dict[str, Any]] = []
    agent_outputs: dict[str, Any] = {}
    winner: str | None = None
    decision_info: dict[str, Any] | None = None

    async for evt in stream:
        trace.append(evt)
        etype = evt.get("type")
        if etype == "agent_finish":
            agent_outputs[evt["agent"]] = evt["final_output"]
        elif etype == "decision":
            winner = evt["winner_product_id"]
            decision_info = {
                "method": evt.get("method"),
                "chose_agent": evt.get("chose_agent"),
                "reasoning": evt.get("reasoning"),
            }

    if decision_info is not None:
        agent_outputs["decision"] = decision_info
    if winner is None:
        raise RuntimeError(
            f"Stream for condition '{condition}' did not produce a decision. "
            f"Last events: {trace[-3:]}"
        )

    return EpisodeResult(
        condition=condition,
        query_id=query.query_id,
        recommended_product_id=winner,
        agent_outputs=agent_outputs,
        trace_events=trace,
    )


async def run_solo(
    marketplace: Marketplace,
    query: QueryView,
    model: str = "gpt-4o-mini",
    persona: str = "neutral",
) -> EpisodeResult:
    return await _drain_stream(
        "solo", query,
        stream_solo(marketplace, query, model=model, persona=persona),
    )


async def run_competitive_no_verifier(
    marketplace: Marketplace,
    query: QueryView,
    model: str = "gpt-4o-mini",
    persona: str = "neutral",
) -> EpisodeResult:
    return await _drain_stream(
        "competitive_no_verifier", query,
        stream_competitive_no_verifier(
            marketplace, query, model=model, persona=persona
        ),
    )


async def run_competitive_with_verifier(
    marketplace: Marketplace,
    query: QueryView,
    model: str = "gpt-4o-mini",
    persona: str = "neutral",
    interrogator_persona: str = "honest",
    interrogator_model: str | None = None,
) -> EpisodeResult:
    return await _drain_stream(
        "competitive_with_verifier", query,
        stream_competitive_with_verifier(
            marketplace, query,
            model=model,
            persona=persona,
            interrogator_persona=interrogator_persona,
            interrogator_model=interrogator_model,
        ),
    )


CONDITION_DISPATCHERS = {
    "solo": run_solo,
    "competitive_no_verifier": run_competitive_no_verifier,
    "competitive_with_verifier": run_competitive_with_verifier,
}
