"""
Batch experiment runner.

Runs N episodes for one condition × persona, writes JSONL, prints summary.
Example:
    python -m harness.run --condition solo --persona commission \\
        --n 20 --split eval --only-conflicts --out solo_commission.jsonl

Use harness.analyze to compare multiple result files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from marketplace import Marketplace
from metrics import score_episode
from experiment_agents import CONDITION_DISPATCHERS
from configs import (
    DEFAULT_ALLOW_REPEAT,
    DEFAULT_CONCURRENCY,
    DEFAULT_CONDITION,
    DEFAULT_CONSUMER_TYPE,
    DEFAULT_DATA_DIR,
    DEFAULT_INTERROGATOR_MODEL,
    DEFAULT_INTERROGATOR_PERSONA,
    DEFAULT_MODEL,
    DEFAULT_N,
    DEFAULT_ONLY_CONFLICTS,
    DEFAULT_OUT,
    DEFAULT_PERSONA,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    list_personas,
)

from .report import print_run_summary


async def run_one(
    marketplace: Marketplace,
    query_id: str,
    condition: str,
    model: str,
    persona: str,
    interrogator_persona: str = "honest",
    interrogator_model: str | None = None,
) -> dict[str, Any]:
    """Run one episode, return a scored record."""
    if condition not in CONDITION_DISPATCHERS:
        raise ValueError(
            f"Unknown condition: {condition}. "
            f"Choices: {list(CONDITION_DISPATCHERS)}"
        )
    dispatch = CONDITION_DISPATCHERS[condition]
    query = marketplace.get_query(query_id)
    t_start = time.time()

    if condition == "competitive_with_verifier":
        episode = await dispatch(
            marketplace, query,
            model=model, persona=persona,
            interrogator_persona=interrogator_persona,
            interrogator_model=interrogator_model,
        )
    else:
        episode = await dispatch(marketplace, query, model=model, persona=persona)
    t_end = time.time()

    score = score_episode(query_id, episode.recommended_product_id, marketplace)

    return {
        "query_id": query_id,
        "condition": condition,
        "persona": persona,
        "interrogator_persona": (
            interrogator_persona if condition == "competitive_with_verifier" else None
        ),
        "model": model,
        "interrogator_model": interrogator_model,
        "recommended_product_id": episode.recommended_product_id,
        "wall_time_seconds": t_end - t_start,
        "score": asdict(score),
        "agent_outputs": episode.agent_outputs,
        "trace_events": episode.trace_events,
    }


async def run_batch(
    marketplace: Marketplace,
    query_ids: list[str],
    condition: str,
    model: str,
    persona: str,
    output_path: Path,
    interrogator_persona: str = "honest",
    interrogator_model: str | None = None,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """One JSON record per line; also returns all records in memory."""
    sem = asyncio.Semaphore(concurrency)
    all_records: list[dict[str, Any]] = []

    async def _bounded(qid: str) -> dict[str, Any]:
        async with sem:
            try:
                return await run_one(
                    marketplace, qid, condition, model, persona,
                    interrogator_persona=interrogator_persona,
                    interrogator_model=interrogator_model,
                )
            except Exception as e:
                return {
                    "query_id": qid,
                    "condition": condition,
                    "persona": persona,
                    "error": f"{type(e).__name__}: {e}",
                }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_total = len(query_ids)
    with output_path.open("a", encoding="utf-8") as f:
        tasks = [asyncio.create_task(_bounded(qid)) for qid in query_ids]
        for fut in asyncio.as_completed(tasks):
            record = await fut
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_done += 1
            all_records.append(record)
            if "error" in record:
                tag = "FAILED"
            elif record.get("score", {}).get("is_correct"):
                tag = "✓"
            else:
                tag = "✗"
            print(f"[{n_done:>3}/{n_total}] {tag} "
                  f"{record['query_id']} "
                  f"(condition={condition}, persona={persona})")
    return all_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--condition", default=DEFAULT_CONDITION,
                    choices=list(CONDITION_DISPATCHERS.keys()),
                    help=f"Experimental condition. Default: {DEFAULT_CONDITION}.")
    ap.add_argument("--persona", default=DEFAULT_PERSONA,
                    choices=list_personas(),
                    help=f"Persona for the research/solo agent(s). "
                         f"Default: {DEFAULT_PERSONA}.")
    ap.add_argument("--interrogator-persona", default=DEFAULT_INTERROGATOR_PERSONA,
                    choices=list_personas(),
                    help=f"Persona for the interrogator "
                         f"(competitive_with_verifier only). "
                         f"Default: {DEFAULT_INTERROGATOR_PERSONA}.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"OpenAI model for research/solo agent(s). "
                         f"Default: {DEFAULT_MODEL}.")
    ap.add_argument("--interrogator-model", default=DEFAULT_INTERROGATOR_MODEL,
                    help="Optional separate model for the interrogator.")
    ap.add_argument("--split", choices=["train", "eval", "all"],
                    default=DEFAULT_SPLIT,
                    help=f"Default: {DEFAULT_SPLIT}.")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"Number of episodes. Default: {DEFAULT_N}.")
    ap.add_argument("--only-conflicts", action=argparse.BooleanOptionalAction,
                    default=DEFAULT_ONLY_CONFLICTS,
                    help=f"Restrict to queries where welfare-optimal differs "
                         f"from highest-fee. Use --no-only-conflicts to "
                         f"disable. Default: {DEFAULT_ONLY_CONFLICTS}.")
    ap.add_argument("--consumer-type",
                    choices=["all", "balanced", "price_sensitive",
                             "quality_focused", "aesthetics_focused"],
                    default=DEFAULT_CONSUMER_TYPE,
                    help=f"'all' = no filter (random query from full pool). "
                         f"Otherwise restrict to one consumer type. "
                         f"Default: {DEFAULT_CONSUMER_TYPE}.")
    ap.add_argument("--allow-repeat", action=argparse.BooleanOptionalAction,
                    default=DEFAULT_ALLOW_REPEAT,
                    help=f"Sample with replacement. Statistically valid: "
                         f"each model call is independent (agent has no "
                         f"cross-episode memory). Use --no-allow-repeat to "
                         f"disable. Default: {DEFAULT_ALLOW_REPEAT}.")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"Default: {DEFAULT_CONCURRENCY}.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"JSONL output path. Appends if it exists. "
                         f"Default: {DEFAULT_OUT}.")
    args = ap.parse_args()

    marketplace = Marketplace(args.data_dir)
    split = None if args.split == "all" else args.split
    consumer_type = (
        None if args.consumer_type == "all" else args.consumer_type
    )
    all_qids = marketplace.list_query_ids(
        split=split,
        only_conflicts=args.only_conflicts,
        consumer_type=consumer_type,
    )
    if not all_qids:
        raise SystemExit("No queries match the requested filters.")

    import random
    rng = random.Random(args.seed)
    if args.allow_repeat:
        qids = [rng.choice(all_qids) for _ in range(args.n)]
    else:
        rng.shuffle(all_qids)
        if args.n > len(all_qids):
            print(f"WARNING: requested --n {args.n} but only "
                  f"{len(all_qids)} unique queries match the filters. "
                  f"Running {len(all_qids)}. Pass --allow-repeat to "
                  f"sample with replacement up to --n.")
        qids = all_qids[: args.n]

    print(f"Running {len(qids)} episodes")
    print(f"  condition:       {args.condition}")
    print(f"  persona:         {args.persona}")
    if args.condition == "competitive_with_verifier":
        print(f"  interrogator_persona: {args.interrogator_persona}")
        if args.interrogator_model:
            print(f"  interrogator_model:   {args.interrogator_model}")
    print(f"  model:           {args.model}")
    print(f"  split:           {args.split}")
    print(f"  consumer_type:   {args.consumer_type}")
    print(f"  only_conflicts:  {args.only_conflicts}")
    print(f"  allow_repeat:    {args.allow_repeat}")
    print(f"  concurrency:     {args.concurrency}")
    print(f"  seed:            {args.seed}")
    print(f"  unique queries available after filtering: {len(all_qids)}")
    print(f"  output:          {args.out}")
    print()

    records = asyncio.run(run_batch(
        marketplace=marketplace,
        query_ids=qids,
        condition=args.condition,
        model=args.model,
        persona=args.persona,
        output_path=Path(args.out),
        interrogator_persona=args.interrogator_persona,
        interrogator_model=args.interrogator_model,
        concurrency=args.concurrency,
    ))

    print_run_summary(records, marketplace)


if __name__ == "__main__":
    main()
