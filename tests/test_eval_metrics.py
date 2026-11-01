"""
Unit tests for eval_metrics.py, strategy_aware_retrieval.py, integration_patch.py,
and the pipeline fixes in main.py / multiagent.py.
"""

import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_metrics import (  # noqa: E402
    STRATEGIES,
    none_rate,
    composite_rate,
    macro_f1,
    per_strategy_prf,
    confusion_matrix,
    strategy_distribution,
    jensen_shannon_divergence,
    confusable_pair_accuracy,
    retrieval_latency_summary,
    compute_esconv_reference_distribution,
    full_report,
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
        {"gold_strategy": "Question and Providing Suggestions",
         "predicted_strategy": "Providing Suggestions", "is_composite_gold": True},
        {"gold_strategy": "Information", "predicted_strategy": "Information", "is_composite_gold": False},
    ]
    cr = composite_rate(preds)
    assert cr == {"composite_count": 1, "total": 2, "composite_rate": 0.5}, cr
    assert macro_f1(preds) == 1.0


def test_affirmation_and_reassurance_never_treated_as_composite():
    """
    Regression test for Critical Bug #1: "Affirmation and Reassurance" contains
    the substring " and " but is a single canonical ESConv strategy and must
    never be excluded as a composite turn, even if a legacy file set
    is_composite_gold=True.
    """
    preds = [
        {"gold_strategy": "Affirmation and Reassurance",
         "predicted_strategy": "Affirmation and Reassurance",
         "is_composite_gold": True},
        {"gold_strategy": "Question", "predicted_strategy": "Question", "is_composite_gold": False},
    ]
    cr = composite_rate(preds)
    assert cr["composite_count"] == 0, cr
    assert macro_f1(preds) == 1.0
    gold_dist = strategy_distribution(preds, "gold_strategy")
    assert abs(gold_dist["Affirmation and Reassurance"] - 0.5) < 1e-9, gold_dist


def test_macro_f1_excludes_none():
    preds = [
        {"gold_strategy": "Question", "predicted_strategy": "Question"},
        {"gold_strategy": "None", "predicted_strategy": "None"},
    ]
    assert macro_f1(preds) == 1.0


def test_known_confusion_case():
    preds = [
        {"gold_strategy": "Question", "predicted_strategy": "Question"},
        {"gold_strategy": "Question", "predicted_strategy": "Restatement or Paraphrasing"},
        {"gold_strategy": "Restatement or Paraphrasing", "predicted_strategy": "Restatement or Paraphrasing"},
        {"gold_strategy": "Restatement or Paraphrasing", "predicted_strategy": "Restatement or Paraphrasing"},
    ]
    prf = per_strategy_prf(preds)
    assert prf["Question"]["precision"] == 1.0
    assert prf["Question"]["recall"] == 0.5
    assert abs(prf["Restatement or Paraphrasing"]["precision"] - 2 / 3) < 1e-9
    assert prf["Restatement or Paraphrasing"]["recall"] == 1.0

    cm = confusion_matrix(preds)
    assert cm["Question"]["Restatement or Paraphrasing"] == 1
    assert cm["Question"]["Question"] == 1

    pair_acc = confusable_pair_accuracy(preds, ("Question", "Restatement or Paraphrasing"))
    assert pair_acc["n"] == 4
    assert pair_acc["accuracy"] == 0.75
    assert pair_acc["mutual_confusions"] == 1


def test_jensen_shannon_identical_distributions_is_zero():
    dist = {"Question": 0.5, "Others": 0.5}
    jsd = jensen_shannon_divergence(dist, dist, labels=["Question", "Others"])
    assert abs(jsd) < 1e-9, jsd


def test_jensen_shannon_disjoint_distributions_is_one_bit():
    p = {"Question": 1.0, "Others": 0.0}
    q = {"Question": 0.0, "Others": 1.0}
    jsd = jensen_shannon_divergence(p, q, labels=["Question", "Others"])
    assert abs(jsd - 1.0) < 1e-9, jsd


def test_js_divergence_uses_same_eligible_turns_for_pred_and_gold():
    """
    Verifies Fix #1: When early/simple turns have predicted_strategy == 'None'
    and a non-None gold_strategy, both predicted_distribution and
    gold_distribution must filter to the exact same eligible turns so a
    perfect predictor on deliberation turns gets JS divergence == 0.0.
    """
    preds = [
        {"gold_strategy": "Question", "predicted_strategy": "None", "is_composite_gold": False},
        {"gold_strategy": "Question", "predicted_strategy": "None", "is_composite_gold": False},
        {"gold_strategy": "Others", "predicted_strategy": "Others", "is_composite_gold": False},
        {"gold_strategy": "Information", "predicted_strategy": "Information", "is_composite_gold": False},
    ]
    rep = full_report(preds)
    assert rep["n_eligible_turns"] == 2
    assert abs(rep["js_divergence_vs_test_gold"]) < 1e-9, rep["js_divergence_vs_test_gold"]
    # Unconditioned all-single-label gold distribution is also preserved separately
    assert abs(rep["gold_distribution_all_single_label"]["Question"] - 0.5) < 1e-9


def test_strategy_distribution_sums_to_one():
    preds = [
        {"predicted_strategy": "Question"}, {"predicted_strategy": "Question"},
        {"predicted_strategy": "Others"}, {"predicted_strategy": "None"},
    ]
    dist = strategy_distribution(preds)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert abs(dist["Question"] - 2 / 3) < 1e-9


def test_strategy_distribution_excludes_composite_sums_to_one():
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


def test_retrieval_latency_summary():
    preds = [
        {"entered_deliberation": True, "retrieval_latency_seconds": 0.010, "latency_seconds": 5.0},
        {"entered_deliberation": True, "retrieval_latency_seconds": 0.020, "latency_seconds": 7.0},
        {"entered_deliberation": True, "retrieval_latency_seconds": 0.030, "latency_seconds": 9.0},
        {"entered_deliberation": False, "retrieval_latency_seconds": None, "latency_seconds": 1.0},
    ]
    lat = retrieval_latency_summary(preds)
    assert lat["retrieval_ms"]["n"] == 3
    assert abs(lat["retrieval_ms"]["mean"] - 20.0) < 1e-6
    assert abs(lat["retrieval_ms"]["median"] - 20.0) < 1e-6
    assert lat["all_turn_s"]["n"] == 4


def test_esconv_reference_distribution():
    fake_convs = [
        {
            "dialog": [
                {"speaker": "seeker", "content": "hi"},
                {"speaker": "supporter", "content": "how can I help?", "annotation": {"strategy": "Question"}},
                {"speaker": "seeker", "content": "I feel sad"},
                {"speaker": "supporter", "content": "you are strong", "annotation": {"strategy": "Affirmation and Reassurance"}},
            ]
        }
    ]
    ref = compute_esconv_reference_distribution(fake_convs, split="all")
    assert ref["n_single_label_turns"] == 2
    assert abs(ref["distribution"]["Question"] - 0.5) < 1e-9
    assert abs(ref["distribution"]["Affirmation and Reassurance"] - 0.5) < 1e-9


def test_self_disclosure_and_markdown_parsing_in_multiagent():
    """
    Regression test for Critical Bug #2 and 7B Markdown formatting: ensures
    multiagent.py parses all 8 strategies (including 'Self-disclosure') across
    bracketed, unbracketed, and Markdown-bolded 7B outputs, and normalizes
    case variations in eval_metrics.py.
    """
    import types
    if "sentence_transformers" not in sys.modules:
        st_mod = types.ModuleType("sentence_transformers")
        st_mod.util = types.ModuleType("sentence_transformers.util")
        sys.modules["sentence_transformers"] = st_mod
        sys.modules["sentence_transformers.util"] = st_mod.util

    from multiagent import normalize_strategy_name, _parse_response_and_strategy, vote

    for s in STRATEGIES:
        assert normalize_strategy_name(f"[{s}]") == s
        assert normalize_strategy_name(f"**{s.lower()}**") == s
        strat, resp = _parse_response_and_strategy(
            f"**Response:** [{s.lower()}] Here is a supportive reply.\n**Reasoning:** Because it helps."
        )
        assert strat == s, (s, strat)
        assert resp == "Here is a supportive reply.", resp

    # Vote should handle Markdown **Reasoning:** and case differences
    v_strats, v_resps = vote([
        "**Response:** [Self-Disclosure] I felt the same way last year.\n**Reasoning:** Builds empathy.",
        "Response: Self-disclosure - I also struggled with this.\nReasoning: Shared experience.",
    ])
    assert v_strats == ["Self-disclosure"], v_strats
    assert len(v_resps) == 1

    # eval_metrics should canonicalize case variations automatically
    preds = [
        {"gold_strategy": "Reflection of feelings", "predicted_strategy": "Reflection of Feelings"},
        {"gold_strategy": "Providing Suggestions", "predicted_strategy": "providing suggestions"},
    ]
    assert macro_f1(preds) == 1.0


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
