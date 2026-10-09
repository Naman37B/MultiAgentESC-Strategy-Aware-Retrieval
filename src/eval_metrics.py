"""
eval_metrics.py

All strategy-selection and distribution metrics for the ablation report.
Pure data processing -- no model, no GPU, no network -- so this module is
fully unit-tested in tests/test_eval_metrics.py without needing any of the
infrastructure the rest of the project needs.

Every function takes a list of per-turn result dicts shaped like:
    {"turn_id": ..., "gold_strategy": ..., "predicted_strategy": ...}
matching what experiments/run_baseline.py (and run_ablation.py) write out.
"""

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


def none_rate(predictions):
    """
    predicted_strategy == "None" turns never entered Strategy Deliberation
    at all (main.py's early/simple-turn zero-shot path). Report this
    fraction separately -- never fold it into Macro-F1 as either a hit or
    a miss.
    """
    total = len(predictions)
    none_count = sum(1 for p in predictions if p["predicted_strategy"] == "None")
    return {"none_count": none_count, "total": total, "none_rate": none_count / total if total else 0.0}


def composite_rate(predictions):
    """
    Two consecutive supporter utterances get merged by MultiAgentESC's own
    logic into one turn with gold_strategy "StrategyA and StrategyB"
    (run_baseline.py sets is_composite_gold accordingly). There's no valid
    single-label ground truth for these turns -- don't silently keep only
    the first label and score the model's one prediction against it, that
    mis-scores genuinely correct predictions that match the second label.
    Exclude them from single-label metrics (see _exclude_none_and_composite)
    and report this rate alongside none_rate, not instead of it.
    """
    total = len(predictions)
    composite_count = sum(1 for p in predictions if p.get("is_composite_gold"))
    return {"composite_count": composite_count, "total": total,
            "composite_rate": composite_count / total if total else 0.0}


def _exclude_none(predictions):
    """Excludes None AND composite-gold turns -- kept this name since every
    caller already uses it; see composite_rate() for why composite turns
    are excluded the same way None turns are."""
    return [
        p for p in predictions
        if p["predicted_strategy"] != "None" and p["gold_strategy"] != "None"
        and not p.get("is_composite_gold")
    ]


def confusion_matrix(predictions, labels=None):
    """rows = gold, cols = predicted. None-excluded (see none_rate())."""
    labels = labels or STRATEGIES
    rows = _exclude_none(predictions)
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for r in rows:
        g, p = r["gold_strategy"], r["predicted_strategy"]
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def per_strategy_prf(predictions, labels=None):
    labels = labels or STRATEGIES
    rows = _exclude_none(predictions)
    tp, fp, fn = Counter(), Counter(), Counter()
    for r in rows:
        g, p = r["gold_strategy"], r["predicted_strategy"]
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
        result[s] = {"precision": precision, "recall": recall, "f1": f1, "support": tp[s] + fn[s]}
    return result


def macro_f1(predictions, labels=None):
    """
    Strategy-selection Macro-F1 over the strategy actually saved as
    pred_strategy after MultiAgentESC's full process (debate/vote/judge/
    self-reflection) completes. Excludes "None" turns -- see none_rate()
    for why, and report that number alongside this one, never instead of it.
    """
    prf = per_strategy_prf(predictions, labels)
    f1s = [v["f1"] for v in prf.values() if v["support"] > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


def strategy_distribution(predictions, key="predicted_strategy", labels=None):
    """
    Confirmed real bug in an earlier version: this excluded "None" but not
    composite-gold turns. A composite gold_strategy like "StrategyA and
    StrategyB" matches none of the 8 canonical labels, so it contributed to
    the denominator (`total`) without ever being counted toward any label
    -- the returned distribution silently summed to less than 1.0 (verified:
    exactly 0.6667 for a 3-row case with 1 composite turn), which makes any
    Jensen-Shannon divergence computed from it not actually comparing two
    normalized distributions. Excluded the same way None is now.
    """
    labels = labels or STRATEGIES
    rows = [p for p in predictions if p[key] != "None" and not p.get("is_composite_gold", False)]
    counts = Counter(r[key] for r in rows)
    total = sum(counts.values()) or 1
    return {s: counts.get(s, 0) / total for s in labels}


def jensen_shannon_divergence(dist_p, dist_q, labels=None):
    """
    JS divergence between two distributions over the same label set.

    Use this as: predicted-strategy distribution on the TEST set vs. the
    TEST set's OWN gold-strategy distribution. Do NOT use the global
    18,376-turn ESConv distribution as dist_q here -- a test subset can
    have a different class balance than the full corpus purely by chance,
    which would make the model look more or less "biased" than it
    actually is. The global distribution is fine as descriptive context
    in the report, never as this metric's denominator.
    """
    labels = labels or STRATEGIES
    p = [dist_p.get(s, 0.0) for s in labels]
    q = [dist_q.get(s, 0.0) for s in labels]
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

    def _kl(a, b):
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def confusable_pair_accuracy(predictions, pair):
    """
    Accuracy restricted to turns whose gold strategy is one of `pair`
    (a frozen, validation-discovered confusable pair -- see
    confusion_matrix_confusable_pairs() in strategy_aware_retrieval.py).
    """
    a, b = pair
    rows = [p for p in _exclude_none(predictions) if p["gold_strategy"] in (a, b)]
    if not rows:
        return {"n": 0, "accuracy": None}
    correct = sum(1 for r in rows if r["predicted_strategy"] == r["gold_strategy"])
    return {"n": len(rows), "accuracy": correct / len(rows)}


def full_report(predictions, reference_distribution=None, confusable_pairs=None):
    """
    One call that builds everything run_ablation.py and analyze_results.py
    need for a single experimental arm's results.json.
    """
    report = {
        "none": none_rate(predictions),
        "composite": composite_rate(predictions),
        "macro_f1": macro_f1(predictions),
        "per_strategy": per_strategy_prf(predictions),
        "confusion_matrix": confusion_matrix(predictions),
        "predicted_distribution": strategy_distribution(predictions, "predicted_strategy"),
        "gold_distribution": strategy_distribution(predictions, "gold_strategy"),
    }
    report["js_divergence_vs_test_gold"] = jensen_shannon_divergence(
        report["predicted_distribution"], report["gold_distribution"]
    )
    if reference_distribution is not None:
        report["js_divergence_vs_global_reference"] = jensen_shannon_divergence(
            report["predicted_distribution"], reference_distribution
        )
    if confusable_pairs:
        report["confusable_pair_accuracy"] = {
            f"{a} vs {b}": confusable_pair_accuracy(predictions, (a, b)) for a, b in confusable_pairs
        }
    return report
