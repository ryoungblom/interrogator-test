"""Marketplace data layer (v3 schema). Loads CSVs and enforces column hygiene:
agent-facing methods return only agent-visible views; ground-truth lookup
lives in separately-named methods only the harness calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


#v3 products.csv: canonical product row (no retailer on product).
#referral_bonus visible for capture experiments.
AGENT_VISIBLE_PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "subcategory",
    "canonical_price",
    "referral_bonus",
    "short_description",
]

#per-(product, retailer) listing: star aggregates + listing price.
#referral_bonus on listing is hidden (same value as product in v3).
AGENT_VISIBLE_LISTING_COLUMNS = [
    "listing_id",
    "product_id",
    "retailer",
    "price",
    "num_star_raters",
    "mean_star_rating",
    "num_written_reviews",
]

#written reviews attach to listings.
AGENT_VISIBLE_REVIEW_COLUMNS = [
    "review_id",
    "listing_id",
    "product_id",
    "retailer",
    "reviewer_name",
    "stars",
    "review_text",
    "source",
    "date",
    "has_disclosure",
]

AGENT_VISIBLE_COMPARISON_ARTICLE_COLUMNS = [
    "article_id",
    "comparison_set_id",
    "title",
    "source_name",
    "content",
    "date",
    "top_recommendation_product_name",
    "has_disclosure",
]

AGENT_VISIBLE_PRODUCT_ARTICLE_COLUMNS = [
    "article_id",
    "product_id",
    "title",
    "source_name",
    "content",
    "date",
    "top_recommendation_product_name",
    "has_disclosure",
]

#backward-compat aliases (v2 names).
AGENT_VISIBLE_ARTICLE_COLUMNS = AGENT_VISIBLE_COMPARISON_ARTICLE_COLUMNS


def _product_row_to_view(row: dict[str, Any]) -> "ProductView":
    return ProductView(
        product_id=row["product_id"],
        product_name=row["product_name"],
        category=row["category"],
        subcategory=row["subcategory"],
        price=float(row["canonical_price"]),
        referral_bonus=float(row["referral_bonus"]),
        short_description=row["short_description"],
    )


def _listing_row_to_view(row: dict[str, Any]) -> "ListingView":
    return ListingView(
        listing_id=row["listing_id"],
        product_id=row["product_id"],
        retailer=row["retailer"],
        price=float(row["price"]),
        num_star_raters=int(row["num_star_raters"]),
        mean_star_rating=float(row["mean_star_rating"]),
        num_written_reviews=int(row["num_written_reviews"]),
    )


def _review_row_to_view(row: dict[str, Any]) -> "ReviewView":
    return ReviewView(
        review_id=row["review_id"],
        listing_id=row["listing_id"],
        product_id=row["product_id"],
        retailer=row["retailer"],
        reviewer_name=row["reviewer_name"],
        stars=int(row["stars"]),
        review_text=row["review_text"],
        source=row["source"],
        date=row["date"],
        has_disclosure=bool(row["has_disclosure"]),
    )


def _comparison_article_row_to_view(row: dict[str, Any]) -> "ArticleView":
    top_rec = row["top_recommendation_product_name"]
    if pd.isna(top_rec):
        top_rec = None
    return ArticleView(
        article_id=row["article_id"],
        comparison_set_id=row["comparison_set_id"],
        title=row["title"],
        source_name=row["source_name"],
        content=row["content"],
        date=row["date"],
        top_recommendation_product_name=top_rec,
        has_disclosure=bool(row["has_disclosure"]),
    )


def _product_article_row_to_view(row: dict[str, Any]) -> "ProductArticleView":
    top_rec = row["top_recommendation_product_name"]
    if pd.isna(top_rec):
        top_rec = None
    return ProductArticleView(
        article_id=row["article_id"],
        product_id=row["product_id"],
        title=row["title"],
        source_name=row["source_name"],
        content=row["content"],
        date=row["date"],
        top_recommendation_product_name=top_rec,
        has_disclosure=bool(row["has_disclosure"]),
    )


@dataclass
class ProductView:
    """Canonical product in a comparison set. Agents recommend product_id."""
    product_id: str
    product_name: str
    category: str
    subcategory: str
    price: float              #canonical_price from products.csv
    referral_bonus: float
    short_description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "category": self.category,
            "subcategory": self.subcategory,
            "price": self.price,
            "referral_bonus": self.referral_bonus,
            "short_description": self.short_description,
        }


@dataclass
class ListingView:
    """One retailer listing for a product: star aggregates + listing price."""
    listing_id: str
    product_id: str
    retailer: str
    price: float
    num_star_raters: int
    mean_star_rating: float
    num_written_reviews: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "product_id": self.product_id,
            "retailer": self.retailer,
            "price": self.price,
            "num_star_raters": self.num_star_raters,
            "mean_star_rating": self.mean_star_rating,
            "num_written_reviews": self.num_written_reviews,
        }


@dataclass
class ReviewView:
    review_id: str
    listing_id: str
    product_id: str
    retailer: str
    reviewer_name: str
    stars: int
    review_text: str
    source: str
    date: str
    has_disclosure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "listing_id": self.listing_id,
            "product_id": self.product_id,
            "retailer": self.retailer,
            "reviewer_name": self.reviewer_name,
            "stars": self.stars,
            "review_text": self.review_text,
            "source": self.source,
            "date": self.date,
            "has_disclosure": self.has_disclosure,
        }


@dataclass
class ArticleView:
    """Comparison-style article (best-of guides, multi-product)."""
    article_id: str
    comparison_set_id: str
    title: str
    source_name: str
    content: str
    date: str
    top_recommendation_product_name: str | None
    has_disclosure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "comparison_set_id": self.comparison_set_id,
            "title": self.title,
            "source_name": self.source_name,
            "content": self.content,
            "date": self.date,
            "top_recommendation_product_name": self.top_recommendation_product_name,
            "has_disclosure": self.has_disclosure,
        }


@dataclass
class ProductArticleView:
    """Single-product deep-dive article."""
    article_id: str
    product_id: str
    title: str
    source_name: str
    content: str
    date: str
    top_recommendation_product_name: str | None
    has_disclosure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "product_id": self.product_id,
            "title": self.title,
            "source_name": self.source_name,
            "content": self.content,
            "date": self.date,
            "top_recommendation_product_name": self.top_recommendation_product_name,
            "has_disclosure": self.has_disclosure,
        }


@dataclass
class QueryView:
    query_id: str
    comparison_set_id: str
    query_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "comparison_set_id": self.comparison_set_id,
            "query_text": self.query_text,
        }


@dataclass
class GroundTruth:
    """Harness-only. Never expose to agents."""
    query_id: str
    comparison_set_id: str
    consumer_type: str
    weight_price: float
    weight_quality: float
    weight_aesthetics: float
    welfare_optimal_product_id: str
    highest_fee_product_id: str
    has_conflict_for_query: bool
    max_welfare: float
    runner_up_welfare: float
    welfare_gap: float


class Marketplace:
    """Agent-facing vs harness-facing interfaces (see method groupings below)."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}. "
                "Generate the dataset first using generate_marketplace.py."
            )

        self._products = pd.read_csv(self.data_dir / "products.csv")
        self._sets = pd.read_csv(self.data_dir / "comparison_sets.csv")
        self._listings = pd.read_csv(self.data_dir / "product_listings.csv")
        self._reviews = pd.read_csv(self.data_dir / "reviews.csv")
        self._comparison_articles = pd.read_csv(self.data_dir / "web_articles.csv")
        self._product_articles = pd.read_csv(
            self.data_dir / "product_articles.csv"
        )
        self._queries = pd.read_csv(self.data_dir / "customer_queries.csv")

        self._products_by_set = {
            sid: g for sid, g in self._products.groupby("comparison_set_id")
        }
        self._listings_by_product = {
            pid: g for pid, g in self._listings.groupby("product_id")
        }
        self._reviews_by_listing = {
            lid: g for lid, g in self._reviews.groupby("listing_id")
        }
        self._comparison_articles_by_set = {
            sid: g for sid, g in self._comparison_articles.groupby("comparison_set_id")
        }
        self._product_articles_by_product = {
            pid: g for pid, g in self._product_articles.groupby("product_id")
        }
        self._query_index = self._queries.set_index("query_id")

        #product_id -> set of listing_ids (for validation).
        self._listing_ids_by_product = {
            pid: set(g["listing_id"].tolist())
            for pid, g in self._listings.groupby("product_id")
        }

    def _product_ids_in_set(self, comparison_set_id: str) -> set[str]:
        if comparison_set_id not in self._products_by_set:
            raise KeyError(f"Unknown comparison_set_id: {comparison_set_id}")
        return set(self._products_by_set[comparison_set_id]["product_id"].tolist())

    def _assert_product_in_set(self, product_id: str, comparison_set_id: str) -> None:
        if product_id not in self._product_ids_in_set(comparison_set_id):
            raise KeyError(
                f"product_id {product_id!r} is not in comparison set "
                f"{comparison_set_id!r}"
            )

    #agent-facing methods.

    def get_query(self, query_id: str) -> QueryView:
        if query_id not in self._query_index.index:
            raise KeyError(f"Unknown query_id: {query_id}")
        row = self._query_index.loc[query_id]
        return QueryView(
            query_id=query_id,
            comparison_set_id=row["comparison_set_id"],
            query_text=row["query_text"],
        )

    def get_products_in_set(self, comparison_set_id: str) -> list[ProductView]:
        if comparison_set_id not in self._products_by_set:
            raise KeyError(f"Unknown comparison_set_id: {comparison_set_id}")
        df = self._products_by_set[comparison_set_id]
        cols = [c for c in AGENT_VISIBLE_PRODUCT_COLUMNS if c in df.columns]
        return [_product_row_to_view(row) for row in df[cols].to_dict(orient="records")]

    def get_listings_for_product(self, product_id: str) -> list[ListingView]:
        """Per-retailer listings with star aggregates. Empty if none."""
        if product_id not in self._listings_by_product:
            return []
        df = self._listings_by_product[product_id]
        return [_listing_row_to_view(row) for row in df.to_dict(orient="records")]

    def get_reviews_for_product(
        self,
        product_id: str,
        retailer: str | None = None,
    ) -> list[ReviewView]:
        """Written reviews for a product. If retailer is set, only that listing's
        reviews; otherwise all listings for the product (may be large)."""
        if product_id not in self._listings_by_product:
            return []
        listings = self._listings_by_product[product_id]
        if retailer is not None:
            match = listings[listings["retailer"] == retailer]
            if match.empty:
                return []
            listing_ids = match["listing_id"].tolist()
        else:
            listing_ids = listings["listing_id"].tolist()

        out: list[ReviewView] = []
        for lid in listing_ids:
            if lid not in self._reviews_by_listing:
                continue
            df = self._reviews_by_listing[lid][AGENT_VISIBLE_REVIEW_COLUMNS]
            for row in df.to_dict(orient="records"):
                out.append(_review_row_to_view(row))
        return out

    def get_comparison_articles_for_set(
        self, comparison_set_id: str
    ) -> list[ArticleView]:
        if comparison_set_id not in self._comparison_articles_by_set:
            return []
        df = self._comparison_articles_by_set[comparison_set_id][
            AGENT_VISIBLE_COMPARISON_ARTICLE_COLUMNS
        ]
        return [
            _comparison_article_row_to_view(row)
            for row in df.to_dict(orient="records")
        ]

    def get_product_articles_for_product(
        self, product_id: str
    ) -> list[ProductArticleView]:
        if product_id not in self._product_articles_by_product:
            return []
        df = self._product_articles_by_product[product_id][
            AGENT_VISIBLE_PRODUCT_ARTICLE_COLUMNS
        ]
        return [
            _product_article_row_to_view(row)
            for row in df.to_dict(orient="records")
        ]

    def get_articles_for_set(self, comparison_set_id: str) -> list[ArticleView]:
        """Alias for get_comparison_articles_for_set (v2 compatibility)."""
        return self.get_comparison_articles_for_set(comparison_set_id)

    #harness-facing methods.

    def get_ground_truth(self, query_id: str) -> GroundTruth:
        if query_id not in self._query_index.index:
            raise KeyError(f"Unknown query_id: {query_id}")
        row = self._query_index.loc[query_id]
        return GroundTruth(
            query_id=query_id,
            comparison_set_id=row["comparison_set_id"],
            consumer_type=row["consumer_type"],
            weight_price=float(row["weight_price"]),
            weight_quality=float(row["weight_quality"]),
            weight_aesthetics=float(row["weight_aesthetics"]),
            welfare_optimal_product_id=row["welfare_optimal_product_id"],
            highest_fee_product_id=row["highest_fee_product_id"],
            has_conflict_for_query=bool(row["has_conflict_for_query"]),
            max_welfare=float(row["max_welfare"]),
            runner_up_welfare=float(row["runner_up_welfare"]),
            welfare_gap=float(row["welfare_gap"]),
        )

    def get_product_full_record(self, product_id: str) -> dict[str, Any]:
        """All product columns including ground truth. Harness only."""
        match = self._products[self._products["product_id"] == product_id]
        if match.empty:
            raise KeyError(f"Unknown product_id: {product_id}")
        return match.iloc[0].to_dict()

    def list_query_ids(
        self,
        split: str | None = None,
        only_conflicts: bool = False,
        consumer_type: str | None = None,
    ) -> list[str]:
        df = self._queries
        if only_conflicts:
            df = df[df["has_conflict_for_query"] == True]  #noqa: E712
        if consumer_type is not None:
            df = df[df["consumer_type"] == consumer_type]
        if split is not None:
            sets_filt = self._sets[self._sets["split"] == split]
            valid_set_ids = set(sets_filt["comparison_set_id"])
            df = df[df["comparison_set_id"].isin(valid_set_ids)]
        return df["query_id"].tolist()

    def summary_stats(self) -> dict[str, Any]:
        return {
            "n_products": int(len(self._products)),
            "n_sets": int(len(self._sets)),
            "n_listings": int(len(self._listings)),
            "n_reviews": int(len(self._reviews)),
            "n_comparison_articles": int(len(self._comparison_articles)),
            "n_product_articles": int(len(self._product_articles)),
            "n_articles": int(len(self._comparison_articles)),
            "n_queries": int(len(self._queries)),
            "n_train_sets": int((self._sets["split"] == "train").sum()),
            "n_eval_sets": int((self._sets["split"] == "eval").sum()),
            "n_conflict_queries": int(
                (self._queries["has_conflict_for_query"] == True).sum()  #noqa: E712
            ),
        }
