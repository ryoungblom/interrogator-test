"""Marketplace data layer. Loads CSVs and enforces column hygiene:
agent-facing methods return only agent-visible views, ground-truth lookup
lives in a separately-named method only the harness calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


#agent-visible columns. Anything not listed is hidden.
#referral_bonus is deliberately visible: the experiment tests whether
#competition resists capture by referral fees, which requires the
#agent to see them.
AGENT_VISIBLE_PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "subcategory",
    "retailer",
    "price",
    "referral_bonus",
    "short_description",
]

#excludes bias_type, reviewer_disposition, bot_sophistication.
#has_disclosure is visible (real consumers can see disclosures).
AGENT_VISIBLE_REVIEW_COLUMNS = [
    "review_id",
    "product_id",
    "reviewer_name",
    "stars",
    "review_text",
    "source",
    "date",
    "has_disclosure",
]

#excludes bias_type and sponsoring_product_name.
AGENT_VISIBLE_ARTICLE_COLUMNS = [
    "article_id",
    "comparison_set_id",
    "title",
    "source_name",
    "content",
    "date",
    "top_recommendation_product_name",
    "has_disclosure",
]


@dataclass
class ProductView:
    product_id: str
    product_name: str
    category: str
    subcategory: str
    retailer: str
    price: float
    referral_bonus: float
    short_description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "category": self.category,
            "subcategory": self.subcategory,
            "retailer": self.retailer,
            "price": self.price,
            "referral_bonus": self.referral_bonus,
            "short_description": self.short_description,
        }


@dataclass
class ReviewView:
    review_id: str
    product_id: str
    reviewer_name: str
    stars: int
    review_text: str
    source: str
    date: str
    has_disclosure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "product_id": self.product_id,
            "reviewer_name": self.reviewer_name,
            "stars": self.stars,
            "review_text": self.review_text,
            "source": self.source,
            "date": self.date,
            "has_disclosure": self.has_disclosure,
        }


@dataclass
class ArticleView:
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
class QueryView:
    """Agent-visible portion of a query. Only the natural-language text;
    weights, consumer_type, and ground-truth labels are hidden."""
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
    """Two interfaces:
      - agent-facing: returns agent-visible columns only.
      - harness-facing (get_ground_truth, get_product_full_record):
        returns full data. Don't call from agent code."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}. "
                "Generate the dataset first using generate_marketplace.py."
            )

        self._products = pd.read_csv(self.data_dir / "products.csv")
        self._sets = pd.read_csv(self.data_dir / "comparison_sets.csv")
        self._reviews = pd.read_csv(self.data_dir / "reviews.csv")
        self._articles = pd.read_csv(self.data_dir / "web_articles.csv")
        self._queries = pd.read_csv(self.data_dir / "customer_queries.csv")

        self._products_by_set = {
            sid: g for sid, g in self._products.groupby("comparison_set_id")
        }
        self._reviews_by_product = {
            pid: g for pid, g in self._reviews.groupby("product_id")
        }
        self._articles_by_set = {
            sid: g for sid, g in self._articles.groupby("comparison_set_id")
        }
        self._query_index = self._queries.set_index("query_id")

    #agent-facing methods. Agent-visible columns only.

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
        df = self._products_by_set[comparison_set_id][AGENT_VISIBLE_PRODUCT_COLUMNS]
        return [ProductView(**row) for row in df.to_dict(orient="records")]

    def get_reviews_for_product(self, product_id: str) -> list[ReviewView]:
        """Empty list if no reviews (some products have none, by design)."""
        if product_id not in self._reviews_by_product:
            return []
        df = self._reviews_by_product[product_id][AGENT_VISIBLE_REVIEW_COLUMNS]
        return [
            ReviewView(**{**row, "has_disclosure": bool(row["has_disclosure"])})
            for row in df.to_dict(orient="records")
        ]

    def get_articles_for_set(self, comparison_set_id: str) -> list[ArticleView]:
        if comparison_set_id not in self._articles_by_set:
            return []
        df = self._articles_by_set[comparison_set_id][AGENT_VISIBLE_ARTICLE_COLUMNS]
        records = df.to_dict(orient="records")
        out = []
        for r in records:
            top_rec = r["top_recommendation_product_name"]
            if pd.isna(top_rec):
                top_rec = None
            out.append(ArticleView(
                article_id=r["article_id"],
                comparison_set_id=r["comparison_set_id"],
                title=r["title"],
                source_name=r["source_name"],
                content=r["content"],
                date=r["date"],
                top_recommendation_product_name=top_rec,
                has_disclosure=bool(r["has_disclosure"]),
            ))
        return out

    #harness-facing methods. Return ground truth. Don't call from agent code.

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
        """All columns including ground truth. Don't call from agent code."""
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
        """Optional filters: split ('train'/'eval'/None), only_conflicts,
        consumer_type ('balanced', 'price_sensitive', 'quality_focused',
        'aesthetics_focused'). Invalid consumer_type returns []."""
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
        """Native ints so the dict is stdlib-json serializable."""
        return {
            "n_products": int(len(self._products)),
            "n_sets": int(len(self._sets)),
            "n_reviews": int(len(self._reviews)),
            "n_articles": int(len(self._articles)),
            "n_queries": int(len(self._queries)),
            "n_train_sets": int((self._sets["split"] == "train").sum()),
            "n_eval_sets": int((self._sets["split"] == "eval").sum()),
            "n_conflict_queries": int(
                (self._queries["has_conflict_for_query"] == True).sum()  #noqa: E712
            ),
        }
