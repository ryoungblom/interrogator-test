"""Function tools exposed to agents. Each is a thin wrapper over a
Marketplace agent-facing method, so hygiene is enforced by construction."""

from __future__ import annotations

import json
from typing import Any

from agents import RunContextWrapper, function_tool

from .context import EpisodeContext


@function_tool
def list_products_in_set(ctx: RunContextWrapper[EpisodeContext]) -> str:
    """List all products in the current comparison set.

    Returns a JSON list with product_id, product_name, retailer, price,
    referral_bonus, and a short description. Call this first.
    """
    eps = ctx.context
    products = eps.marketplace.get_products_in_set(eps.query.comparison_set_id)
    payload = [p.to_dict() for p in products]
    eps.log("tool_call", tool="list_products_in_set", result_count=len(payload))
    return json.dumps(payload, indent=2)


@function_tool
def get_reviews(
    ctx: RunContextWrapper[EpisodeContext],
    product_id: str,
) -> str:
    """Get reviews for a product.

    Returns a JSON list with reviewer_name, stars, review_text, source, date,
    and has_disclosure (sponsored / free product / affiliate). Empty list if
    the product has no reviews.

    Args:
        product_id: from list_products_in_set.
    """
    eps = ctx.context
    #validate the product is in the current set (no cross-set leakage).
    set_products = eps.marketplace.get_products_in_set(eps.query.comparison_set_id)
    valid_ids = {p.product_id for p in set_products}
    if product_id not in valid_ids:
        return json.dumps({"error": f"product_id {product_id!r} is not in the current comparison set."})
    reviews = eps.marketplace.get_reviews_for_product(product_id)
    payload = [r.to_dict() for r in reviews]
    eps.log(
        "tool_call",
        tool="get_reviews",
        product_id=product_id,
        result_count=len(payload),
    )
    return json.dumps(payload, indent=2)


@function_tool
def get_articles(ctx: RunContextWrapper[EpisodeContext]) -> str:
    """Get web articles (blog posts, 'best of' guides, comparisons) about
    products in the current comparison set.

    Returns a JSON list with title, source_name, content, and has_disclosure.
    Articles often name a top pick; some are commercially motivated (sponsored,
    affiliate, retailer-owned).
    """
    eps = ctx.context
    articles = eps.marketplace.get_articles_for_set(eps.query.comparison_set_id)
    payload = [a.to_dict() for a in articles]
    eps.log("tool_call", tool="get_articles", result_count=len(payload))
    return json.dumps(payload, indent=2)


RESEARCH_TOOLS = [list_products_in_set, get_reviews, get_articles]
