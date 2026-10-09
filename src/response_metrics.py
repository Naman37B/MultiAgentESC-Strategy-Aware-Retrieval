"""
response_metrics.py

Automatic response-quality metrics implemented according to the metric
NAMES MultiAgentESC's own paper reports using (Xu et al., EMNLP 2025,
Automatic Evaluation section), checked directly against that section
before writing any of this:

    Distinct-1, Distinct-2, BLEU-1, BLEU-2, BLEU-3, F1, ROUGE-L

That list of metric names is confirmed from the paper itself. The paper's
Automatic Evaluation section does not specify every tokenization/
aggregation detail of its own internal evaluator, so treat the numbers
below as "computed according to these standard metric definitions," not
as verified-identical to the authors' own implementation -- call them that
in the report unless you separately confirm exact numerical agreement. The paper also reports human
evaluation and GPT-4o-judge evaluation (Fluency / Identification /
Comforting / Suggestion / Overall, per the review that flagged this gap)
-- that specific five-dimension breakdown was NOT independently confirmed
here against the primary source the way the automatic metrics were, so
it's deliberately not hardcoded into scoring logic below. See
make_gpt4o_judge_prompt()'s docstring for what to do about that before
using it for real.

Pure Python, no new dependencies (no nltk/sacrebleu/rouge-score), so it
installs with nothing beyond what's already in requirements.txt and runs
anywhere -- verified with hand-computable cases in
tests/test_response_metrics.py.
"""

import math
from collections import Counter


def _tokenize(text):
    return text.lower().split()


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(responses, n):
    """
    Distinct-n: ratio of unique n-grams to total n-grams across a set of
    generated responses. Higher = less repetitive. Computed over the whole
    batch (matching how Distinct-1/2 is standardly reported in this
    literature), not averaged per-response.
    """
    all_ngrams = []
    for r in responses:
        all_ngrams.extend(_ngrams(_tokenize(r), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _clipped_match_counts(candidate_tokens, reference_tokens, n):
    """Returns (clipped_matches, total_candidate_ngrams) as raw counts, not
    a ratio -- lets corpus_bleu() sum counts across sentences BEFORE
    dividing, which is what makes it a true corpus-level BLEU rather than
    an average of per-sentence scores (see corpus_bleu()'s docstring)."""
    cand_ngrams = Counter(_ngrams(candidate_tokens, n))
    ref_ngrams = Counter(_ngrams(reference_tokens, n))
    clipped = sum(min(c, ref_ngrams[g]) for g, c in cand_ngrams.items())
    return clipped, sum(cand_ngrams.values())


def sentence_bleu(candidate, reference, max_n):
    """
    BLEU-1/2/3 for a single candidate/reference pair (max_n=1, 2, or 3),
    with the standard brevity penalty. Implemented directly rather than
    importing nltk, so this has zero new dependencies -- verified against
    a hand-computable case (identical candidate/reference -> BLEU = 1.0,
    disjoint vocabularies -> BLEU = 0.0) in the test suite.

    This is a per-sentence score. For a whole-dataset number, use
    corpus_bleu() below, NOT an average of this function's output over many
    pairs -- those are different, standardly-distinguished quantities (an
    earlier version of corpus_bleu() in this file made exactly that
    mistake, averaging sentence_bleu() instead of aggregating counts first).
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        clipped, total = _clipped_match_counts(cand_tokens, ref_tokens, n)
        precisions.append(clipped / total if total else 0.0)
    if any(p == 0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return bp * geo_mean


def corpus_bleu(candidates, references, max_n):
    """
    True corpus-level BLEU: clipped n-gram matches and totals are summed
    across every candidate/reference pair FIRST, then one precision per
    n-gram order is computed from those totals, then one geometric mean and
    one brevity penalty for the whole corpus -- the standard form (matching
    e.g. sacrebleu/nltk's corpus_bleu), not an average of per-sentence
    scores. An earlier version of this function did average sentence_bleu()
    per pair, which is a different, non-standard quantity -- short
    responses with a single zero-count n-gram order sentence-score to
    exactly 0.0 under BLEU's definition, which drags a per-sentence average
    down in a way true corpus aggregation doesn't, since one short
    sentence's shortfall gets diluted into the corpus-wide totals instead.

    Note scope: this reproduces BLEU-1/2/3 as the metric is standardly
    defined, matching the metric NAMES MultiAgentESC's paper reports it
    used. The paper's Automatic Evaluation section doesn't specify every
    tokenization/aggregation detail of their own internal evaluator, so
    this isn't verified to be numerically identical to their specific
    implementation -- call these "automatic response metrics implemented
    according to the metrics MultiAgentESC reports using," not an exact
    reproduction of their own tool's output, unless that's separately checked.
    """
    if not candidates:
        return 0.0
    total_clipped = [0] * max_n
    total_counts = [0] * max_n
    cand_len_sum, ref_len_sum = 0, 0
    for cand, ref in zip(candidates, references):
        cand_tokens, ref_tokens = _tokenize(cand), _tokenize(ref)
        cand_len_sum += len(cand_tokens)
        ref_len_sum += len(ref_tokens)
        for i, n in enumerate(range(1, max_n + 1)):
            clipped, total = _clipped_match_counts(cand_tokens, ref_tokens, n)
            total_clipped[i] += clipped
            total_counts[i] += total
    precisions = [c / t if t else 0.0 for c, t in zip(total_clipped, total_counts)]
    if any(p == 0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bp = 1.0 if cand_len_sum > ref_len_sum else math.exp(1 - ref_len_sum / max(cand_len_sum, 1))
    return bp * geo_mean


def unigram_f1(candidate, reference):
    """
    Response-level F1: unigram precision/recall/F1 between candidate and
    reference, the same style of overlap-F1 used throughout the ESC/QA
    literature for this "F1" metric (distinct from the strategy-selection
    Macro-F1 in eval_metrics.py -- different thing, same name, which is
    exactly why that one is called "Strategy-selection Macro-F1" in the
    report, not just "F1").
    """
    cand = Counter(_tokenize(candidate))
    ref = Counter(_tokenize(reference))
    overlap = sum((cand & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate, reference, beta=1.2):
    """ROUGE-L: LCS-based F-measure, the standard beta=1.2 weighting."""
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand_tokens), lcs / len(ref_tokens)
    return ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)


def automatic_response_report(candidates, references):
    """
    One call for everything above, matched to the paper's own metric list
    (D-1, D-2, B-1, B-2, B-3, F1, R-L). Use candidates/references from the
    SAME arm and SAME test turns you're computing strategy-selection
    metrics for, so the two reports describe the same evaluation run.
    """
    return {
        "distinct_1": distinct_n(candidates, 1),
        "distinct_2": distinct_n(candidates, 2),
        "bleu_1": corpus_bleu(candidates, references, 1),
        "bleu_2": corpus_bleu(candidates, references, 2),
        "bleu_3": corpus_bleu(candidates, references, 3),
        "f1": sum(unigram_f1(c, r) for c, r in zip(candidates, references)) / len(candidates) if candidates else 0.0,
        "rouge_l": sum(rouge_l(c, r) for c, r in zip(candidates, references)) / len(candidates) if candidates else 0.0,
    }


def make_gpt4o_judge_prompt(situation, response):
    """
    Scaffolding only, deliberately not scored automatically here. The paper
    reports GPT-4o-judge evaluation on named dimensions (reported
    elsewhere as Fluency / Identification / Comforting / Suggestion /
    Overall) and human evaluation -- neither was independently verified
    against the paper's own appendix/figures the way the seven automatic
    metrics above were, so this project isn't hardcoding that rubric as
    fact. Before using this for real: open the actual paper's appendix
    (it includes the literal GPT-4o evaluation prompt it used) and copy
    the real rubric and prompt wording from there, rather than trusting
    this placeholder or any secondhand description of it, including the
    one that prompted adding this function.
    """
    return (
        f"Situation: {situation}\n"
        f"Response: {response}\n\n"
        "TODO: replace with the actual rubric and prompt text from the "
        "MultiAgentESC paper's appendix before using this for real scoring."
    )
