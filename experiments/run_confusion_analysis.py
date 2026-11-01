"""
run_confusion_analysis.py -- Validation-split confusion analysis.

Takes the flat_results from a baseline validation run (never test) and produces
the frozen confusable-pairs file (`results/confirmed_confusable_pairs.json`).
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
from strategy_aware_retrieval import confusion_matrix_confusable_pairs  # noqa: E402
from eval_metrics import confusion_matrix, none_rate, composite_rate, _is_composite_turn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-results", required=True,
                    help="Results JSON from run_baseline.py --arm baseline --split val")
    ap.add_argument("--top-m", type=int, default=5)
    ap.add_argument("--out", default="results/confirmed_confusable_pairs.json")
    ap.add_argument("--allow-non-validation", action="store_true",
                    help="Override the val+baseline guard for diagnostic inspection only.")
    args = ap.parse_args()

    with open(args.baseline_results, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_config = data.get("config", {})
    if not args.allow_non_validation:
        if run_config.get("split") != "val":
            raise ValueError(
                f"{args.baseline_results} was run with --split={run_config.get('split')!r}, "
                "not 'val'. Confusable pairs must be discovered on validation, never on test."
            )
        if run_config.get("arm") != "baseline":
            raise ValueError(
                f"{args.baseline_results} was run with --arm={run_config.get('arm')!r}, "
                "not 'baseline'. Confusable pairs must come from the unmodified baseline."
            )

    flat = data["flat_results"]
    nr = none_rate(flat)
    cr = composite_rate(flat)
    print(f"None-rate on this run: {nr['none_count']}/{nr['total']} = {nr['none_rate']:.1%}")
    print(f"Composite-gold rate on this run: {cr['composite_count']}/{cr['total']} = {cr['composite_rate']:.1%}")

    single_label_rows = [r for r in flat if not _is_composite_turn(r)]
    gold = [r["gold_strategy"] for r in single_label_rows]
    pred = [r["predicted_strategy"] for r in single_label_rows]
    pairs = confusion_matrix_confusable_pairs(gold, pred, top_m=args.top_m)

    print(f"\nTop {args.top_m} confused strategy pairs (gold vs. predicted mismatches):")
    for (a, b), count in pairs:
        print(f"  {a}  <->  {b}   :  {count} mismatches")

    cm = confusion_matrix(flat)
    out = {
        "source_run": args.baseline_results,
        "none_rate": nr,
        "composite_rate": cr,
        "n_eligible_deliberation_turns": sum(sum(row.values()) for row in cm.values()),
        "confirmed_confusable_pairs": [list(p) for p, _c in pairs],
        "pair_counts": {f"{a} vs {b}": c for (a, b), c in pairs},
        "full_confusion_matrix": cm,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote frozen confusable pairs to {args.out}")


if __name__ == "__main__":
    main()
