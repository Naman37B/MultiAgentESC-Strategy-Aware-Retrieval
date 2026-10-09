"""
run_confusion_analysis.py -- Phase 2.

Pure data processing on a baseline run's output -- no GPU/model needed,
safe to run anywhere once you have a results JSON from run_baseline.py.

Takes the flat_results from a VALIDATION run (never test -- see
INTEGRATION_NOTES.md's data-split discipline section) and produces the
frozen confusable-pairs list the rest of the project depends on.

Usage:
    python experiments/run_confusion_analysis.py \
        --baseline-results results/baseline_val_150.json \
        --top-m 5 \
        --out results/confirmed_confusable_pairs.json
"""

import argparse
import json
import sys

sys.path.insert(0, "src")
from strategy_aware_retrieval import confusion_matrix_confusable_pairs  # noqa: E402
from eval_metrics import confusion_matrix, none_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-results", required=True,
                     help="A results JSON from run_baseline.py --arm baseline --split val")
    ap.add_argument("--top-m", type=int, default=5)
    ap.add_argument("--out", default="results/confirmed_confusable_pairs.json")
    ap.add_argument("--allow-non-validation", action="store_true",
                     help="Override the val+baseline guard below. Only for deliberate "
                          "diagnostic use -- never for the real frozen confusable-pairs run.")
    args = ap.parse_args()

    with open(args.baseline_results) as f:
        data = json.load(f)

    # Confirmed gap from review: the docs said "run this on validation only"
    # but nothing enforced it. run_baseline.py's output always records its
    # own config, so check it rather than just document the rule.
    run_config = data.get("config", {})
    if not args.allow_non_validation:
        if run_config.get("split") != "val":
            raise ValueError(
                f"{args.baseline_results} was run with --split={run_config.get('split')!r}, "
                "not 'val'. Confusable pairs must be discovered on validation, never on "
                "test -- that's leakage. Pass --allow-non-validation if you really mean to "
                "override this for a deliberate diagnostic, not for the frozen pairs file."
            )
        if run_config.get("arm") != "baseline":
            raise ValueError(
                f"{args.baseline_results} was run with --arm={run_config.get('arm')!r}, "
                "not 'baseline'. Confusable pairs must come from the UNMODIFIED baseline's "
                "own errors, not from a run that already has retrieval changes applied."
            )

    flat = data["flat_results"]

    nr = none_rate(flat)
    print(f"None-rate on this run: {nr['none_count']}/{nr['total']} = {nr['none_rate']:.1%}")
    print("(turns that never entered Strategy Deliberation -- excluded below, reported separately)")

    # Confirmed gap: eval_metrics.py's Macro-F1/confusion-matrix functions
    # were fixed to exclude composite-gold turns ("StrategyA and
    # StrategyB"), but confusion_matrix_confusable_pairs() only sees two
    # plain lists and has no way to know which rows were composite -- so
    # without filtering here first, a composite turn could still enter the
    # FROZEN confusable-pairs list that the evidence arm depends on, even
    # though it's correctly excluded everywhere else by now.
    single_label_rows = [r for r in flat if not r.get("is_composite_gold", False)]
    n_excluded = len(flat) - len(single_label_rows)
    print(f"Composite-gold turns excluded from confusion analysis: {n_excluded}/{len(flat)}")

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
        "confirmed_confusable_pairs": [list(p) for p, _c in pairs],
        "pair_counts": {f"{a} vs {b}": c for (a, b), c in pairs},
        "full_confusion_matrix": cm,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote frozen confusable pairs to {args.out}")
    print("These are now FROZEN -- do not regenerate this file from test-set results.")


if __name__ == "__main__":
    main()
