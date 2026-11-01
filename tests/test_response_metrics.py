import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from response_metrics import (  # noqa: E402
    LLM_JUDGE_DIMENSIONS,
    distinct_n,
    sentence_bleu,
    corpus_bleu,
    unigram_f1,
    rouge_l,
    filter_valid_response_pairs,
    automatic_response_report,
    make_gpt4o_judge_prompt,
    parse_llm_judge_scores,
    aggregate_llm_judge_scores,
    evaluate_responses_with_llm_judge,
)


def test_distinct_n_all_unique():
    responses = ["aa bb cc", "dd ee ff", "gg hh ii"]
    assert distinct_n(responses, 1) == 1.0


def test_distinct_n_all_repeated():
    responses = ["aa aa aa", "aa aa aa"]
    assert abs(distinct_n(responses, 1) - 1 / 6) < 1e-9


def test_bleu_identical_is_one():
    assert abs(sentence_bleu("i am so sorry to hear that", "i am so sorry to hear that", 3) - 1.0) < 1e-9


def test_bleu_disjoint_is_zero():
    assert sentence_bleu("completely different words here", "nothing matches at all today", 2) == 0.0


def test_bleu_brevity_penalty_applies():
    short_but_correct = sentence_bleu("i am so", "i am so sorry to hear that you feel this way", 1)
    full_match = sentence_bleu("i am so sorry to hear that you feel this way",
                               "i am so sorry to hear that you feel this way", 1)
    assert short_but_correct < full_match
    assert full_match == 1.0


def test_unigram_f1_identical_is_one():
    assert unigram_f1("that sounds really hard", "that sounds really hard") == 1.0


def test_unigram_f1_no_overlap_is_zero():
    assert unigram_f1("aaa bbb ccc", "ddd eee fff") == 0.0


def test_unigram_f1_known_value():
    f1 = unigram_f1("a a b", "a b b c")
    assert abs(f1 - 0.5714285714) < 1e-6, f1


def test_rouge_l_identical_is_one():
    assert abs(rouge_l("this is a test sentence", "this is a test sentence") - 1.0) < 1e-9


def test_rouge_l_no_common_subsequence_is_zero():
    assert rouge_l("aaa bbb ccc", "ddd eee fff") == 0.0


def test_corpus_bleu_is_not_sentence_averaging():
    candidates = ["that sounds really hard", "i hear you"]
    references = ["that sounds really hard to deal with", "i hear you and i am here"]

    naive_average = sum(sentence_bleu(c, r, 2) for c, r in zip(candidates, references)) / len(candidates)
    true_corpus = corpus_bleu(candidates, references, 2)

    assert true_corpus != naive_average
    assert true_corpus > 0.0


def test_automatic_response_report_shape():
    report = automatic_response_report(
        ["i am sorry to hear that", "that must be really hard"],
        ["i am sorry to hear that", "that sounds really difficult"],
    )
    expected_keys = {"n_evaluated", "distinct_1", "distinct_2", "bleu_1", "bleu_2", "bleu_3", "f1", "rouge_l"}
    assert set(report.keys()) == expected_keys
    assert report["n_evaluated"] == 2
    assert all(0.0 <= v <= 1.0 for k, v in report.items() if k != "n_evaluated"), report


def test_filter_valid_response_pairs_excludes_none_strings():
    records = [
        {"response": "None", "reference": "valid ref", "entered_deliberation": True},
        {"response": "valid reply", "reference": "valid ref", "entered_deliberation": True},
        {"response": "early reply", "reference": "early ref", "entered_deliberation": False},
    ]
    c_all, r_all = filter_valid_response_pairs(records, deliberation_only=False)
    assert c_all == ["valid reply", "early reply"]
    c_delib, r_delib = filter_valid_response_pairs(records, deliberation_only=True)
    assert c_delib == ["valid reply"]


def test_llm_judge_prompt_and_parsing():
    prompt = make_gpt4o_judge_prompt(
        context_or_situation="User: I feel completely overwhelmed by work.",
        response="It sounds like your workload has been exhausting lately.",
        situation="Burnout at work",
    )
    for dim in ["Fluency", "Identification", "Comforting", "Suggestion", "Overall"]:
        assert dim in prompt

    # Valid JSON parsing
    scores_json = parse_llm_judge_scores(
        'Here is the evaluation:\n{"fluency": 5, "identification": 4, "comforting": 4, "suggestion": 3, "overall": 4, "reasoning": "Empathetic."}'
    )
    assert scores_json is not None
    assert scores_json["fluency"] == 5.0
    assert scores_json["overall"] == 4.0

    # Valid key-value text parsing
    scores_kv = parse_llm_judge_scores(
        "Fluency: 4\nIdentification: 4\nComforting: 5\nSuggestion: 3\nOverall: 4"
    )
    assert scores_kv is not None
    assert scores_kv["comforting"] == 5.0

    # Out-of-range score rejected
    assert parse_llm_judge_scores('{"fluency": 6, "identification": 4, "comforting": 4, "suggestion": 3, "overall": 4}') is None


def test_evaluate_responses_with_llm_judge_end_to_end():
    records = [
        {
            "turn_id": "t1",
            "context": "User: I failed my test.",
            "response": "I understand how disappointing that feels.",
            "reference": "That must be really frustrating.",
            "entered_deliberation": True,
        },
        {
            "turn_id": "t2",
            "context": "User: Hi",
            "response": "Hello",
            "reference": "Hi there",
            "entered_deliberation": False,
        },
    ]

    def fake_judge(prompt_text):
        return json.dumps({
            "fluency": 5,
            "identification": 4,
            "comforting": 4,
            "suggestion": 3,
            "overall": 4,
            "reasoning": "Clear and supportive.",
        })

    res = evaluate_responses_with_llm_judge(records, judge_fn=fake_judge, deliberation_only=True)
    assert res["aggregate"]["n_scored"] == 1
    assert res["aggregate"]["fluency"] == 5.0
    assert res["aggregate"]["overall"] == 4.0


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
