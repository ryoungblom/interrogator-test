"""Structured-output schemas for agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    """One product recommendation from a research or solo agent."""
    product_id: str = Field(
        description="product_id of the recommended product, exactly as "
                    "returned by list_products_in_set."
    )
    reasoning: str = Field(
        description="Argument for this recommendation. Cite specific reviews, "
                    "articles, or product features. Note source-bias concerns."
    )


class InterrogatorDecision(BaseModel):
    """Final decision from the interrogator."""
    chosen_product_id: str = Field(
        description="The final recommended product_id."
    )
    chose_agent: str = Field(
        description="'agent_a', 'agent_b', or 'neither' (overruled both)."
    )
    reasoning: str = Field(
        description="Why this product was chosen, including verification work "
                    "and reasons one agent's argument was weaker."
    )
