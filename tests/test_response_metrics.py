import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from response_metrics import (  # noqa: E402
    distinct_n, sentence_bleu, corpus_bleu, unigram_f1, rouge_l, automatic_response_report,
)


def test_distinct_n_all_unique():
    # 3 responses, 3 words each, all distinct unigrams -> distinct-1 = 1.0
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
    # candidate is a short, fully-correct PREFIX of a much longer reference
    # -> perfect precision but should be penalized by brevity
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
    # candidate: {a,a,b}, reference: {a,b,b,c} -> overlap(a:1,b:1)=2
    # precision = 2/3, recall = 2/4=0.5, F1 = 2*(2/3*0.5)/(2/3+0.5) = 0.5714...
    f1 = unigram_f1("a a b", "a b b c")
    assert abs(f1 - 0.5714285714) < 1e-6, f1


def test_rouge_l_identical_is_one():
    assert abs(rouge_l("this is a test sentence", "this is a test sentence") - 1.0) < 1e-9


def test_rouge_l_no_common_subsequence_is_zero():
    assert rouge_l("aaa bbb ccc", "ddd eee fff") == 0.0


def test_corpus_bleu_is_not_sentence_averaging():
    """
    Proves corpus_bleu aggregates counts first rather than averaging
    sentence_bleu() per pair -- an earlier version did the latter, which a
    review correctly flagged as a different, non-standard quantity.
    One short candidate here scores sentence_bleu = 0.0 (misses a 2-gram
    entirely), which would drag a naive average down to 0.0 for the whole
    pair; true corpus aggregation dilutes that single sentence's shortfall
    into the corpus-wide n-gram totals instead, so it should NOT be 0.0.
    """
    candidates = ["that sounds really hard", "i hear you"]
    references = ["that sounds really hard to deal with", "i hear you and i am here"]

    naive_average = sum(sentence_bleu(c, r, 2) for c, r in zip(candidates, references)) / len(candidates)
    true_corpus = corpus_bleu(candidates, references, 2)

    assert true_corpus != naive_average, "corpus_bleu must not equal a sentence-level average"
    assert true_corpus > 0.0, "true corpus aggregation should not zero out from one short sentence"


def test_automatic_response_report_shape():
    report = automatic_response_report(
        ["i am sorry to hear that", "that must be really hard"],
        ["i am sorry to hear that", "that sounds really difficult"],
    )
    expected_keys = {"distinct_1", "distinct_2", "bleu_1", "bleu_2", "bleu_3", "f1", "rouge_l"}
    assert set(report.keys()) == expected_keys
    assert all(0.0 <= v <= 1.0 for v in report.values()), report


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
