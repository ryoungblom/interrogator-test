"""Agent factories. Persona text comes from configs/prompts.py."""

from __future__ import annotations

from agents import Agent

from configs import DEFAULT_MODEL, get_persona

from .context import EpisodeContext
from .schemas import Recommendation, InterrogatorDecision
from .tools import RESEARCH_TOOLS


def make_solo_agent(
    model: str = DEFAULT_MODEL,
    persona: str = "neutral",
) -> Agent[EpisodeContext]:
    persona_cfg = get_persona(persona)
    return Agent[EpisodeContext](
        name=f"SoloRecommender_{persona}",
        instructions=persona_cfg["solo_system"],
        model=model,
        tools=RESEARCH_TOOLS,
        output_type=Recommendation,
    )


def make_research_agent(
    name: str,
    model: str = DEFAULT_MODEL,
    persona: str = "neutral",
) -> Agent[EpisodeContext]:
    persona_cfg = get_persona(persona)
    return Agent[EpisodeContext](
        name=name,
        instructions=persona_cfg["research_system"],
        model=model,
        tools=RESEARCH_TOOLS,
        output_type=Recommendation,
    )


def make_interrogator_agent(
    model: str = DEFAULT_MODEL,
    persona: str = "honest",
) -> Agent[EpisodeContext]:
    persona_cfg = get_persona(persona)
    return Agent[EpisodeContext](
        name=f"Interrogator_{persona}",
        instructions=persona_cfg["interrogator_system"],
        model=model,
        tools=RESEARCH_TOOLS,
        output_type=InterrogatorDecision,
    )
