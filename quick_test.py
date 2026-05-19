"""One-episode smoke test for the pipeline.

    python quick_test.py
    python quick_test.py --condition competitive_with_verifier --persona commission
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from experiment_agents import CONDITION_DISPATCHERS
    from configs import DEFAULT_MODEL, list_personas

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="./marketplace_dataset")
    ap.add_argument("--condition", default="solo",
                    choices=list(CONDITION_DISPATCHERS.keys()))
    ap.add_argument("--persona", default="neutral", choices=list_personas())
    ap.add_argument("--interrogator-persona", default="honest",
                    choices=list_personas())
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY is not set. See .env.example.")
        sys.exit(1)
    print("✓ OPENAI_API_KEY is set")

    from marketplace import Marketplace
    try:
        marketplace = Marketplace(args.data_dir)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        sys.exit(1)
    print(f"✓ Marketplace loaded: {marketplace.summary_stats()}")

    qids = marketplace.list_query_ids(split="eval", only_conflicts=True)
    if not qids:
        qids = marketplace.list_query_ids()
    if not qids:
        print("✗ No queries available.")
        sys.exit(1)
    qid = qids[0]
    query = marketplace.get_query(qid)
    print(f"✓ Selected query: {qid}")
    print(f"\nQuery text: {query.query_text}\n")
    print(f"Running condition={args.condition} persona={args.persona} "
          f"model={args.model}...")

    dispatch = CONDITION_DISPATCHERS[args.condition]
    if args.condition == "competitive_with_verifier":
        coro = dispatch(marketplace, query, model=args.model,
                        persona=args.persona,
                        interrogator_persona=args.interrogator_persona)
    else:
        coro = dispatch(marketplace, query, model=args.model,
                        persona=args.persona)
    try:
        result = asyncio.run(coro)
    except RuntimeError as e:
        #typically: agent crashed before producing a recommendation
        #(e.g., transient API error that survived all retries in streaming.py).
        print()
        print("✗ The episode failed to produce a recommendation.")
        print(f"  Reason: {e}")
        print()
        print("  Usually a transient upstream API error. Try again, or switch")
        print("  to a different model with --model. Streaming already retries")
        print("  up to 3 times; if still failing, OpenAI is probably degraded.")
        sys.exit(2)
    except Exception as e:
        print()
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        raise

    print(f"\n=== AGENT OUTPUTS ===")
    print(json.dumps(result.agent_outputs, indent=2)[:2000])

    from metrics import score_episode
    score = score_episode(qid, result.recommended_product_id, marketplace)
    print(f"\n=== SCORE ===")
    print(f"Recommended:        {result.recommended_product_id}")
    print(f"Welfare-optimal:    {score.welfare_optimal_product_id}")
    print(f"Highest-fee:        {score.highest_fee_product_id}")
    print(f"Is correct:         {score.is_correct}")
    print(f"Is captured:        {score.is_captured}")
    print(f"Welfare regret:     {score.welfare_regret:.3f}")
    print(f"\n✓ Smoke test completed.")


if __name__ == "__main__":
    main()
