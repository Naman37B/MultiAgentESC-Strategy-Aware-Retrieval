"""Paired A-vs-D correctness test and conversation-cluster Macro-F1 bootstrap.

Example:
  python experiments/paired_significance.py \
      --baseline results/final_test_A_baseline_300.json \
      --modified results/final_test_D_evidence_300.json \
      --bootstrap-reps 50000 --seed 2026 \
      --out results/paired_significance_300.json

Clusters are conversations, not individual turns. Each bootstrap replicate
samples the 195 conversation clusters with replacement and recomputes
8-class Macro-F1 for both configurations on the resampled turns.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LABELS = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def load_flat(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("flat_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no non-empty flat_results list")
    return rows


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    denom = 2.0 * tp + fp + fn
    scores = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(scores.mean())


def exact_mcnemar_p(d_only_correct: int, a_only_correct: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = d_only_correct + a_only_correct
    if n == 0:
        return 1.0
    k = min(d_only_correct, a_only_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--modified", required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a_rows = load_flat(Path(args.baseline))
    d_rows = load_flat(Path(args.modified))
    a_ids = [row.get("turn_id") for row in a_rows]
    d_ids = [row.get("turn_id") for row in d_rows]
    if a_ids != d_ids:
        raise ValueError("Baseline and modified files must contain identical turn IDs in identical order.")

    conversations = sorted({row.get("conversation_idx") for row in a_rows})
    if any(row.get("conversation_idx") is None for row in a_rows):
        raise ValueError("Every turn must contain conversation_idx for cluster resampling.")
    conv_to_idx = {conv: i for i, conv in enumerate(conversations)}
    cm_a = np.zeros((len(conversations), len(LABELS), len(LABELS)), dtype=np.int32)
    cm_d = np.zeros_like(cm_a)
    gold_correct_a = []
    gold_correct_d = []

    for a, d in zip(a_rows, d_rows):
        gold = a.get("gold_strategy")
        pred_a = a.get("predicted_strategy")
        pred_d = d.get("predicted_strategy")
        if gold not in LABEL_TO_INDEX or pred_a not in LABEL_TO_INDEX or pred_d not in LABEL_TO_INDEX:
            raise ValueError(f"Unexpected or non-canonical strategy label in turn {a.get('turn_id')!r}.")
        cluster = conv_to_idx[a["conversation_idx"]]
        y = LABEL_TO_INDEX[gold]
        cm_a[cluster, y, LABEL_TO_INDEX[pred_a]] += 1
        cm_d[cluster, y, LABEL_TO_INDEX[pred_d]] += 1
        gold_correct_a.append(pred_a == gold)
        gold_correct_d.append(pred_d == gold)

    correct_a = np.asarray(gold_correct_a, dtype=bool)
    correct_d = np.asarray(gold_correct_d, dtype=bool)
    d_only = int(np.sum(~correct_a & correct_d))
    a_only = int(np.sum(correct_a & ~correct_d))
    macro_a = macro_f1_from_confusion(cm_a.sum(axis=0))
    macro_d = macro_f1_from_confusion(cm_d.sum(axis=0))

    reps = int(args.bootstrap_reps)
    if reps < 1000:
        raise ValueError("Use at least 1,000 bootstrap replicates; the submitted report uses 50,000.")
    rng = np.random.default_rng(args.seed)
    n_clusters = len(conversations)
    cluster_counts = np.zeros((reps, n_clusters), dtype=np.int16)
    for i in range(reps):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        cluster_counts[i] = np.bincount(draw, minlength=n_clusters)

    boot_a = np.einsum("bc,cij->bij", cluster_counts, cm_a, optimize=True)
    boot_d = np.einsum("bc,cij->bij", cluster_counts, cm_d, optimize=True)
    diffs = np.empty(reps, dtype=float)
    for i in range(reps):
        diffs[i] = macro_f1_from_confusion(boot_d[i]) - macro_f1_from_confusion(boot_a[i])
    ci_low, ci_high = np.quantile(diffs, [0.025, 0.975])

    result = {
        "comparison": "D - A",
        "n_turns": len(a_rows),
        "n_conversation_clusters": n_clusters,
        "baseline_macro_f1": macro_a,
        "modified_macro_f1": macro_d,
        "macro_f1_difference": macro_d - macro_a,
        "baseline_correct": int(correct_a.sum()),
        "modified_correct": int(correct_d.sum()),
        "modified_only_correct": d_only,
        "baseline_only_correct": a_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(d_only, a_only),
        "bootstrap": {
            "method": "conversation-cluster percentile bootstrap",
            "replicates": reps,
            "seed": args.seed,
            "confidence_level": 0.95,
            "difference_ci_low": float(ci_low),
            "difference_ci_high": float(ci_high),
            "fraction_bootstrap_differences_above_zero": float(np.mean(diffs > 0)),
        },
    }

    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
