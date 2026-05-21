"""Function tools exposed to agents. Each is a thin wrapper over a
Marketplace agent-facing method, so hygiene is enforced by construction."""

from __future__ import annotations

import json
from typing import Any

from agents import RunContextWrapper, function_tool

from .context import EpisodeContext


def _valid_product_ids(ctx: RunContextWrapper[EpisodeContext]) -> set[str]:
    eps = ctx.context
    set_products = eps.marketplace.get_products_in_set(eps.query.comparison_set_id)
    return {p.product_id for p in set_products}


@function_tool
def list_products_in_set(ctx: RunContextWrapper[EpisodeContext]) -> str:
    """List all products in the current comparison set.

    Returns a JSON list with product_id, product_name, price (canonical),
    referral_bonus, and short_description. Call this first.
    """
    eps = ctx.context
    products = eps.marketplace.get_products_in_set(eps.query.comparison_set_id)
    payload = [p.to_dict() for p in products]
    eps.log("tool_call", tool="list_products_in_set", result_count=len(payload))
    return json.dumps(payload, indent=2)


@function_tool
def get_listings_for_product(
    ctx: RunContextWrapper[EpisodeContext],
    product_id: str,
) -> str:
    """Per-retailer listings for a product: price, star aggregates, review counts.

    Args:
        product_id: from list_products_in_set.
    """
    eps = ctx.context
    if product_id not in _valid_product_ids(ctx):
        return json.dumps(
            {"error": f"product_id {product_id!r} is not in the current comparison set."}
        )
    listings = eps.marketplace.get_listings_for_product(product_id)
    payload = [lv.to_dict() for lv in listings]
    eps.log(
        "tool_call",
        tool="get_listings_for_product",
        product_id=product_id,
        result_count=len(payload),
    )
    return json.dumps(payload, indent=2)


@function_tool
def get_reviews(
    ctx: RunContextWrapper[EpisodeContext],
    product_id: str,
    retailer: str | None = None,
) -> str:
    """Written reviews for a product (optionally one retailer).

    Returns reviewer_name, stars, review_text, source, date, has_disclosure,
    listing_id, retailer. Empty list if none. Omit retailer to get all listings.

    Args:
        product_id: from list_products_in_set.
        retailer: optional retailer name to scope reviews.
    """
    eps = ctx.context
    if product_id not in _valid_product_ids(ctx):
        return json.dumps(
            {"error": f"product_id {product_id!r} is not in the current comparison set."}
        )
    reviews = eps.marketplace.get_reviews_for_product(product_id, retailer=retailer)
    payload = [r.to_dict() for r in reviews]
    eps.log(
        "tool_call",
        tool="get_reviews",
        product_id=product_id,
        retailer=retailer,
        result_count=len(payload),
    )
    return json.dumps(payload, indent=2)


@function_tool
def get_comparison_articles(ctx: RunContextWrapper[EpisodeContext]) -> str:
    """Comparison guides and 'best of' articles for the current set.

    Returns title, source_name, content, has_disclosure, top pick if any.
    """
    eps = ctx.context
    articles = eps.marketplace.get_comparison_articles_for_set(
        eps.query.comparison_set_id
    )
    payload = [a.to_dict() for a in articles]
    eps.log("tool_call", tool="get_comparison_articles", result_count=len(payload))
    return json.dumps(payload, indent=2)


@function_tool
def get_product_articles(
    ctx: RunContextWrapper[EpisodeContext],
    product_id: str,
) -> str:
    """Single-product deep-dive articles (blogs, reviews roundups).

    Args:
        product_id: from list_products_in_set.
    """
    eps = ctx.context
    if product_id not in _valid_product_ids(ctx):
        return json.dumps(
            {"error": f"product_id {product_id!r} is not in the current comparison set."}
        )
    articles = eps.marketplace.get_product_articles_for_product(product_id)
    payload = [a.to_dict() for a in articles]
    eps.log(
        "tool_call",
        tool="get_product_articles",
        product_id=product_id,
        result_count=len(payload),
    )
    return json.dumps(payload, indent=2)


RESEARCH_TOOLS = [
    list_products_in_set,
    get_listings_for_product,
    get_reviews,
    get_comparison_articles,
    get_product_articles,
]
