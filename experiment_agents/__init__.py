"""experiment_agents: agents, tools, and orchestration."""
from .context import EpisodeContext
from .schemas import Recommendation, InterrogatorDecision

from .orchestration import (
    EpisodeResult,
    run_solo,
    run_competitive_no_verifier,
    run_competitive_with_verifier,
    CONDITION_DISPATCHERS,
)

from .streaming import (
    stream_solo,
    stream_competitive_no_verifier,
    stream_competitive_with_verifier,
    STREAM_DISPATCHERS,
)

__all__ = [
    "EpisodeContext",
    "EpisodeResult",
    "Recommendation",
    "InterrogatorDecision",
    "run_solo",
    "run_competitive_no_verifier",
    "run_competitive_with_verifier",
    "CONDITION_DISPATCHERS",
    "stream_solo",
    "stream_competitive_no_verifier",
    "stream_competitive_with_verifier",
    "STREAM_DISPATCHERS",
]
