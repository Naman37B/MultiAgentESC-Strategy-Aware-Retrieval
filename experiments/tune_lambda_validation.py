"""
tune_lambda_validation.py -- Validation-only selection of `lambda_diversity`.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from strategy_aware_retrieval import split_conversations, build_quadruple_safe, StrategyAwareRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/ESConv.json")
    ap.add_argument("--cache-path", default="embeddings_train.txt")
    ap.add_argument("--model-path", default="all-roberta-large-v1")
    ap.add_argument("--confusable-pairs", default="results/confirmed_confusable_pairs.json")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.15, 0.3, 0.45, 0.6])
    ap.add_argument("--max-val-posts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/validation_lambda_sweep.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from main import get_embeddings, set_seed

    set_seed(args.seed)
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    train_data, val_data, _ = split_conversations(dataset, seed=args.seed)

    confirmed_pairs = []
    if args.confusable_pairs and os.path.exists(args.confusable_pairs):
        with open(args.confusable_pairs, "r", encoding="utf-8") as f:
            confirmed_pairs = [tuple(p) for p in json.load(f).get("confirmed_confusable_pairs", [])]

    model = SentenceTransformer(args.model_path)
    quadruple = build_quadruple_safe(model, train_data, args.cache_path, get_embeddings, model_name=args.model_path)
    retriever = StrategyAwareRetriever(quadruple, model)

    val_posts = []
    for sample in val_data:
        dialog = sample.get("dialog", [])
        for i in range(1, len(dialog)):
            if dialog[i].get("speaker") == "supporter" and dialog[i - 1].get("speaker") == "seeker":
                val_posts.append(dialog[i - 1]["content"].strip())
                if len(val_posts) >= args.max_val_posts:
                    break
        if len(val_posts) >= args.max_val_posts:
            break

    sweep_results = []
    base_sets = []
    for post in val_posts:
        pairs_0, _ = retriever.rerank(post, n=10, lambda_diversity=0.0, confusable_pairs=None)
        base_sets.append({p[0] for p in pairs_0})

    for lam in args.lambdas:
        distinct_counts = []
        pair_surface_hits = 0
        overlap_ratios = []
        strat_counter = Counter()

        for idx, post in enumerate(val_posts):
            pairs, strats = retriever.rerank(post, n=10, lambda_diversity=lam, confusable_pairs=confirmed_pairs)
            present = set(strats)
            distinct_counts.append(len(present))
            strat_counter.update(strats)
            overlap = len({p[0] for p in pairs} & base_sets[idx]) / 10.0
            overlap_ratios.append(overlap)
            if confirmed_pairs:
                pair_surface_hits += sum(1 for a, b in confirmed_pairs if a in present and b in present) / len(confirmed_pairs)

        total_s = sum(strat_counter.values()) or 1
        entropy = -sum((c / total_s) * math.log2(c / total_s) for c in strat_counter.values() if c > 0) / math.log2(8)
        mean_distinct = sum(distinct_counts) / len(distinct_counts) if distinct_counts else 0.0
        mean_retention = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0
        co_surface_rate = pair_surface_hits / len(val_posts) if val_posts else 0.0
        val_score = 0.35 * mean_retention + 0.40 * (mean_distinct / 8.0) + 0.25 * co_surface_rate - 1.5 * max(0.0, 0.75 - mean_retention)

        sweep_results.append({
            "lambda_diversity": lam,
            "mean_distinct_strategies_top10": round(mean_distinct, 4),
            "normalized_strategy_entropy": round(entropy, 4),
            "confusable_pair_cosurfacing_rate": round(co_surface_rate, 4),
            "rrf_relevance_retention": round(mean_retention, 4),
            "validation_composite_score": round(val_score, 4),
        })

    best = max(sweep_results, key=lambda r: r["validation_composite_score"])
    out_data = {
        "split": "val",
        "n_validation_posts": len(val_posts),
        "selected_lambda_diversity": best["lambda_diversity"],
        "sweep": sweep_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(json.dumps(out_data, indent=2))
    print(f"Selected lambda_diversity on validation: {best['lambda_diversity']} (saved to {args.out})")


if __name__ == "__main__":
    main()
