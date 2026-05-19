"""EpisodeContext: shared state for an episode. No ground truth — that's
loaded by the harness during scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marketplace import Marketplace, QueryView


@dataclass
class EpisodeContext:
    """Passed to RunContextWrapper. Tools access it via ctx.context; the
    agent itself only sees tool outputs."""
    marketplace: Marketplace
    query: QueryView
    #harness fills this in after the run, for logging.
    trace_events: list[dict[str, Any]] = field(default_factory=list)

    def log(self, event_type: str, **payload: Any) -> None:
        evt = {"event_type": event_type, **payload}
        self.trace_events.append(evt)
