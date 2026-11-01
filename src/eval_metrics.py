"""
eval_metrics.py

Strategy-selection, strategy-distribution, JS divergence, confused-pair
accuracy, and retrieval-latency metrics for the MultiAgentESC ablation study.
Pure Python data processing with zero GPU or network dependencies.
"""

import json
import math
from collections import Counter

STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]
_STRATEGY_SET = set(STRATEGIES)
_LOWER_TO_CANON = {s.lower(): s for s in STRATEGIES}


def _canon(label):
    if not isinstance(label, str):
        return label
    low = label.strip().lower()
    return _LOWER_TO_CANON.get(low, label.strip())


def _is_composite_turn(p):
    """
    Determines whether a turn has a merged multi-strategy gold label
    (e.g., "Question and Providing Suggestions") rather than one of the 8
    canonical single-strategy labels.

    Note: "Affirmation and Reassurance" is a single canonical strategy that
    happens to contain the substring " and ". Checking `gold in _STRATEGY_SET`
    first guarantees "Affirmation and Reassurance" is never misclassified as
    composite, even when reading legacy result files that set
    `is_composite_gold = (" and " in gold_strategy)`.
    """
    gold = _canon(p.get("gold_strategy"))
    if gold in _STRATEGY_SET or gold == "None":
        return False
    if "is_composite_gold" in p:
        return bool(p["is_composite_gold"])
    return bool(gold and " and " in gold)


def none_rate(predictions):
    """
    predicted_strategy == "None" turns skipped Strategy Deliberation via
    MultiAgentESC's early/simple-turn zero-shot path (or failed strategy
    extraction). Reported separately and excluded from deliberation Macro-F1.
    """
    total = len(predictions)
    none_count = sum(1 for p in predictions if p.get("predicted_strategy") == "None")
    return {
        "none_count": none_count,
        "total": total,
        "none_rate": none_count / total if total else 0.0,
    }


def composite_rate(predictions):
    """
    Fraction of evaluated turns where two consecutive supporter utterances
    were merged into a composite gold label ("StrategyA and StrategyB").
    """
    total = len(predictions)
    composite_count = sum(1 for p in predictions if _is_composite_turn(p))
    return {
        "composite_count": composite_count,
        "total": total,
        "composite_rate": composite_count / total if total else 0.0,
    }


def _exclude_none(predictions):
    """
    Returns the eligible single-label deliberation turns:
      - predicted_strategy != "None"
      - gold_strategy != "None"
      - not a composite-gold turn
    Both Macro-F1 and test-gold JS divergence operate over this exact same
    set of eligible turns so numerator and denominator populations match.
    """
    return [
        p for p in predictions
        if p.get("predicted_strategy", "None") != "None"
        and p.get("gold_strategy", "None") != "None"
        and not _is_composite_turn(p)
    ]


def confusion_matrix(predictions, labels=None):
    """rows = gold, cols = predicted over eligible single-label deliberation turns."""
    labels = labels or STRATEGIES
    rows = _exclude_none(predictions)
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for r in rows:
        g, p = _canon(r["gold_strategy"]), _canon(r["predicted_strategy"])
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def per_strategy_prf(predictions, labels=None):
    labels = labels or STRATEGIES
    rows = _exclude_none(predictions)
    tp, fp, fn = Counter(), Counter(), Counter()
    for r in rows:
        g, p = _canon(r["gold_strategy"]), _canon(r["predicted_strategy"])
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    result = {}
    for s in labels:
        precision = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) else 0.0
        recall = tp[s] / (tp[s] + fn[s]) if (tp[s] + fn[s]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[s] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp[s] + fn[s],
            "predicted_count": tp[s] + fp[s],
        }
    return result


def macro_f1(predictions, labels=None):
    """
    Unweighted mean of per-strategy F1 across strategies with positive gold
    support on the eligible single-label deliberation turns.
    """
    prf = per_strategy_prf(predictions, labels)
    f1s = [v["f1"] for v in prf.values() if v["support"] > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


def strategy_distribution(predictions, key="predicted_strategy", labels=None, eligible_only=True):
    """
    Normalized probability distribution over the 8 canonical strategies.

    Crucial population discipline (`eligible_only=True`, the default):
    When evaluating paired `(gold_strategy, predicted_strategy)` turn records,
    both `strategy_distribution(predictions, "predicted_strategy")` and
    `strategy_distribution(predictions, "gold_strategy")` filter out turns
    where EITHER `predicted_strategy == "None"` OR `gold_strategy == "None"`
    OR the turn has a composite gold label. This ensures that predicted and
    gold distributions represent the exact same set of eligible turns when
    computing JS divergence.

    If `eligible_only=False`, only filters `p[key] != "None"` and non-composite
    turns (useful for inspecting the unconditioned gold distribution across all
    single-label turns including early zero-shot turns).
    """
    labels = labels or STRATEGIES
    rows = []
    for p in predictions:
        if _is_composite_turn(p):
            continue
        if p.get(key) == "None" or p.get(key) is None:
            continue
        if eligible_only:
            # If both keys exist on the record, require both to be non-"None"
            # so predicted_strategy and gold_strategy distributions use the
            # exact same eligible turns.
            if "predicted_strategy" in p and p["predicted_strategy"] == "None":
                continue
            if "gold_strategy" in p and p["gold_strategy"] == "None":
                continue
        rows.append(p)

    counts = Counter(_canon(r[key]) for r in rows if _canon(r.get(key)) in labels)
    total = sum(counts.values()) or 1
    return {s: counts.get(s, 0) / total for s in labels}


def compute_esconv_reference_distribution(dataset_or_path="dataset/ESConv.json",
                                          split="all", seed=42, labels=None):
    """
    Computes the reference ESConv strategy distribution specified in the
    project proposal ("Jensen-Shannon divergence from the reference ESConv
    strategy distribution").

    Follows main.py / run_baseline.py's turn-extraction convention over the
    ESConv corpus, counting all single-label supporter turns (excluding merged
    consecutive supporter turns).

    Args:
        dataset_or_path: Loaded list of ESConv conversation dicts or path to ESConv.json.
        split: "all" (all 1,300 ESConv conversations), "train" (910 train conversations),
               "val" (195 validation conversations), or "test" (195 test conversations).
        seed: Random seed for split_conversations when split != "all".
    Returns:
        dict with keys:
          - "distribution": {strategy: probability, ...} summing to 1.0
          - "counts": {strategy: count, ...}
          - "n_single_label_turns": int
          - "n_composite_turns_excluded": int
          - "n_conversations": int
          - "split": split name
    """
    labels = labels or STRATEGIES
    if isinstance(dataset_or_path, str):
        with open(dataset_or_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
    else:
        conversations = list(dataset_or_path)

    if split != "all":
        from strategy_aware_retrieval import split_conversations
        train_d, val_d, test_d = split_conversations(conversations, seed=seed)
        conversations = {"train": train_d, "val": val_d, "test": test_d}[split]

    counts = Counter()
    n_composite = 0
    for sample in conversations:
        dialog = sample.get("dialog", [])
        count = 0
        while count < len(dialog):
            if count != 0 and dialog[count].get("speaker") == "supporter":
                if (count < len(dialog) - 1 and dialog[count + 1].get("speaker") != "supporter") or (count == len(dialog) - 1):
                    strat = dialog[count].get("annotation", {}).get("strategy")
                    if strat in labels:
                        counts[strat] += 1
                    count += 1
                else:
                    n_composite += 1
                    count += 2
            else:
                count += 1

    total = sum(counts.values()) or 1
    dist = {s: counts.get(s, 0) / total for s in labels}
    return {
        "split": split,
        "n_conversations": len(conversations),
        "n_single_label_turns": sum(counts.values()),
        "n_composite_turns_excluded": n_composite,
        "counts": {s: counts.get(s, 0) for s in labels},
        "distribution": dist,
    }


def jensen_shannon_divergence(dist_p, dist_q, labels=None):
    """
    Base-2 Jensen-Shannon divergence (in bits, bounded in [0, 1]) between two
    normalized strategy distributions over the same label set.
    """
    labels = labels or STRATEGIES
    p = [dist_p.get(s, 0.0) for s in labels]
    q = [dist_q.get(s, 0.0) for s in labels]
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]

    def _kl(a, b):
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def confusable_pair_accuracy(predictions, pair):
    """
    Accuracy restricted to eligible single-label turns whose gold strategy is
    one of `pair = (strategy_a, strategy_b)` (frozen from validation baseline).
    Also records per-member support and how often the two strategies were
    directly confused with each other.
    """
    a, b = pair
    rows = [p for p in _exclude_none(predictions) if _canon(p["gold_strategy"]) in (a, b)]
    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "correct": 0,
            "mutual_confusions": 0,
            "support_a": 0,
            "support_b": 0,
        }
    correct = sum(1 for r in rows if _canon(r["predicted_strategy"]) == _canon(r["gold_strategy"]))
    mutual = sum(
        1 for r in rows
        if (_canon(r["gold_strategy"]) == a and _canon(r["predicted_strategy"]) == b)
        or (_canon(r["gold_strategy"]) == b and _canon(r["predicted_strategy"]) == a)
    )
    sup_a = sum(1 for r in rows if _canon(r["gold_strategy"]) == a)
    sup_b = sum(1 for r in rows if _canon(r["gold_strategy"]) == b)
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "correct": correct,
        "mutual_confusions": mutual,
        "support_a": sup_a,
        "support_b": sup_b,
    }


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _summarize_seconds_list(vals_sec, unit_scale=1.0):
    if not vals_sec:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    scaled = sorted(v * unit_scale for v in vals_sec)
    n = len(scaled)
    mean_val = sum(scaled) / n
    median_val = _percentile(scaled, 50)
    p95_val = _percentile(scaled, 95)
    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "p95": p95_val,
        "min": scaled[0],
        "max": scaled[-1],
    }


def retrieval_latency_summary(predictions):
    """
    Summarizes both retrieval-only latency (in milliseconds, measured inside
    PatchedStrategyDeliberation) and end-to-end turn latency (in seconds).
    """
    retrieval_secs = [
        p["retrieval_latency_seconds"]
        for p in predictions
        if p.get("retrieval_latency_seconds") is not None
        and p.get("entered_deliberation", True)
    ]
    delib_turn_secs = [
        p["latency_seconds"]
        for p in predictions
        if p.get("latency_seconds") is not None
        and p.get("entered_deliberation", True)
    ]
    all_turn_secs = [
        p["latency_seconds"]
        for p in predictions
        if p.get("latency_seconds") is not None
    ]
    return {
        "retrieval_ms": _summarize_seconds_list(retrieval_secs, unit_scale=1000.0),
        "deliberation_turn_s": _summarize_seconds_list(delib_turn_secs, unit_scale=1.0),
        "all_turn_s": _summarize_seconds_list(all_turn_secs, unit_scale=1.0),
    }


def full_report(predictions, reference_distribution=None, confusable_pairs=None):
    """
    Builds the complete evaluation report for one experimental arm.
    """
    eligible_rows = _exclude_none(predictions)
    single_label_rows = [
        p for p in predictions
        if p.get("gold_strategy", "None") != "None" and not _is_composite_turn(p)
    ]

    pred_dist = strategy_distribution(predictions, "predicted_strategy", eligible_only=True)
    gold_dist_eligible = strategy_distribution(predictions, "gold_strategy", eligible_only=True)
    gold_dist_all_single = strategy_distribution(predictions, "gold_strategy", eligible_only=False)

    report = {
        "n_total_turns": len(predictions),
        "n_single_label_turns": len(single_label_rows),
        "n_eligible_turns": len(eligible_rows),
        "none": none_rate(predictions),
        "composite": composite_rate(predictions),
        "macro_f1": macro_f1(predictions),
        "per_strategy": per_strategy_prf(predictions),
        "confusion_matrix": confusion_matrix(predictions),
        "predicted_distribution": pred_dist,
        "gold_distribution": gold_dist_eligible,
        "gold_distribution_all_single_label": gold_dist_all_single,
        "js_divergence_vs_test_gold": jensen_shannon_divergence(pred_dist, gold_dist_eligible),
        "latency": retrieval_latency_summary(predictions),
    }

    if reference_distribution is not None:
        ref_dist = (
            reference_distribution["distribution"]
            if isinstance(reference_distribution, dict) and "distribution" in reference_distribution
            else reference_distribution
        )
        js_ref = jensen_shannon_divergence(pred_dist, ref_dist)
        report["js_divergence_vs_esconv_reference"] = js_ref
        report["js_divergence_vs_global_reference"] = js_ref

    if confusable_pairs:
        report["confusable_pair_accuracy"] = {
            f"{a} vs {b}": confusable_pair_accuracy(predictions, (a, b))
            for a, b in confusable_pairs
        }
    return report
