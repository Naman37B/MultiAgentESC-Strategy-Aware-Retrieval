"""
Unit tests for eval_metrics.py. Runs anywhere, no GPU/model/network needed
-- this is the one part of the project verifiable without your machine's
hardware, so it's worth actually running (`python -m pytest tests/` or
`python tests/test_eval_metrics.py`) before you trust it in the real
ablation.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eval_metrics import (  # noqa: E402
    none_rate, composite_rate, macro_f1, per_strategy_prf, confusion_matrix,
    strategy_distribution, jensen_shannon_divergence, confusable_pair_accuracy,
)


def test_none_rate():
    preds = [
        {"gold_strategy": "Question", "predicted_strategy": "Question"},
        {"gold_strategy": "Others", "predicted_strategy": "None"},
        {"gold_strategy": "Information", "predicted_strategy": "None"},
        {"gold_strategy": "Information", "predicted_strategy": "Information"},
    ]
    r = none_rate(preds)
    assert r == {"none_count": 2, "total": 4, "none_rate": 0.5}, r


def test_macro_f1_perfect():
    preds = [
        {"gold_strategy": s, "predicted_strategy": s}
        for s in ["Question", "Question", "Information", "Others"]
    ]
    assert macro_f1(preds) == 1.0


def test_composite_gold_excluded_not_truncated():
    preds = [
        # composite gold, model predicted the SECOND label correctly --
        # the old behavior (take first label) would wrongly score this a miss
        {"gold_strategy": "Question and Providing Suggestions",
         "predicted_strategy": "Providing Suggestions", "is_composite_gold": True},
        {"gold_strategy": "Information", "predicted_strategy": "Information", "is_composite_gold": False},
    ]
    cr = composite_rate(preds)
    assert cr == {"composite_count": 1, "total": 2, "composite_rate": 0.5}, cr
    # macro_f1 should be computed ONLY over the clean Information turn -> perfect score,
    # not degraded or wrongly penalized by the composite turn either direction
    assert macro_f1(preds) == 1.0


def test_macro_f1_excludes_none():
    preds = [
        {"gold_strategy": "Question", "predicted_strategy": "Question"},
        {"gold_strategy": "None", "predicted_strategy": "None"},
    ]
    # with None excluded, this is a single perfect prediction -> 1.0,
    # not degraded by the None turn either direction
    assert macro_f1(preds) == 1.0


def test_known_confusion_case():
    # 2 Question turns: 1 correct, 1 predicted as Restatement
    # 2 Restatement turns: both correct
    preds = [
        {"gold_strategy": "Question", "predicted_strategy": "Question"},
        {"gold_strategy": "Question", "predicted_strategy": "Restatement or Paraphrasing"},
        {"gold_strategy": "Restatement or Paraphrasing", "predicted_strategy": "Restatement or Paraphrasing"},
        {"gold_strategy": "Restatement or Paraphrasing", "predicted_strategy": "Restatement or Paraphrasing"},
    ]
    prf = per_strategy_prf(preds)
    # Question: tp=1, fn=1 (the one predicted as Restatement), fp=0 -> precision=1.0, recall=0.5
    assert prf["Question"]["precision"] == 1.0
    assert prf["Question"]["recall"] == 0.5
    # Restatement: tp=2, fp=1 (the misclassified Question turn), fn=0 -> precision=2/3, recall=1.0
    assert abs(prf["Restatement or Paraphrasing"]["precision"] - 2 / 3) < 1e-9
    assert prf["Restatement or Paraphrasing"]["recall"] == 1.0

    cm = confusion_matrix(preds)
    assert cm["Question"]["Restatement or Paraphrasing"] == 1
    assert cm["Question"]["Question"] == 1

    pair_acc = confusable_pair_accuracy(preds, ("Question", "Restatement or Paraphrasing"))
    assert pair_acc == {"n": 4, "accuracy": 0.75}


def test_jensen_shannon_identical_distributions_is_zero():
    dist = {"Question": 0.5, "Others": 0.5}
    jsd = jensen_shannon_divergence(dist, dist, labels=["Question", "Others"])
    assert abs(jsd) < 1e-9, jsd


def test_jensen_shannon_disjoint_distributions_is_one_bit():
    # maximally different distributions over 2 labels -> JSD = 1 bit (log2-based)
    p = {"Question": 1.0, "Others": 0.0}
    q = {"Question": 0.0, "Others": 1.0}
    jsd = jensen_shannon_divergence(p, q, labels=["Question", "Others"])
    assert abs(jsd - 1.0) < 1e-9, jsd


def test_strategy_distribution_sums_to_one():
    preds = [
        {"predicted_strategy": "Question"}, {"predicted_strategy": "Question"},
        {"predicted_strategy": "Others"}, {"predicted_strategy": "None"},
    ]
    dist = strategy_distribution(preds)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert abs(dist["Question"] - 2 / 3) < 1e-9  # None excluded from denominator


def test_strategy_distribution_excludes_composite_sums_to_one():
    # the exact case that previously summed to 0.6667 instead of 1.0
    preds = [
        {"gold_strategy": "Question and Providing Suggestions",
         "predicted_strategy": "Question", "is_composite_gold": True},
        {"gold_strategy": "Information", "predicted_strategy": "Information", "is_composite_gold": False},
        {"gold_strategy": "Question", "predicted_strategy": "Question", "is_composite_gold": False},
    ]
    gold_dist = strategy_distribution(preds, key="gold_strategy")
    pred_dist = strategy_distribution(preds, key="predicted_strategy")
    assert abs(sum(gold_dist.values()) - 1.0) < 1e-9, sum(gold_dist.values())
    assert abs(sum(pred_dist.values()) - 1.0) < 1e-9, sum(pred_dist.values())


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
