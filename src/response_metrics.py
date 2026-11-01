"""
response_metrics.py

Automatic and LLM-judged response-quality evaluation for MultiAgentESC:
  1. Automatic metrics matching Xu et al. (EMNLP 2025, Section 4.4):
     Distinct-1, Distinct-2, BLEU-1, BLEU-2, BLEU-3, Unigram F1, and ROUGE-L.
  2. LLM-as-a-Judge evaluation matching Xu et al. (EMNLP 2025, Section 5.2,
     Section 5.3, and Appendix C): 5-dimension evaluation on a 1-5 scale across
     Fluency, Identification, Comforting, Suggestion, and Overall.
"""

import json
import math
import os
import re
from collections import Counter


LLM_JUDGE_DIMENSIONS = (
    "fluency",
    "identification",
    "comforting",
    "suggestion",
    "overall",
)


def _tokenize(text):
    return text.lower().split()


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(responses, n):
    """
    Distinct-n: ratio of unique n-grams to total n-grams across a corpus of
    generated responses.
    """
    all_ngrams = []
    for r in responses:
        all_ngrams.extend(_ngrams(_tokenize(r), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _clipped_match_counts(candidate_tokens, reference_tokens, n):
    cand_ngrams = Counter(_ngrams(candidate_tokens, n))
    ref_ngrams = Counter(_ngrams(reference_tokens, n))
    clipped = sum(min(c, ref_ngrams[g]) for g, c in cand_ngrams.items())
    return clipped, sum(cand_ngrams.values())


def sentence_bleu(candidate, reference, max_n):
    """Per-sentence BLEU-1/2/3 with standard brevity penalty."""
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
    True corpus-level BLEU-1/2/3: aggregates clipped n-gram match counts and
    totals across the entire corpus before computing precision and brevity penalty.
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
    """Response-level unigram overlap F1 between candidate and reference."""
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
    """ROUGE-L LCS-based F-measure with standard beta=1.2."""
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand_tokens), lcs / len(ref_tokens)
    return ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)


def filter_valid_response_pairs(records, deliberation_only=False):
    """
    Extracts valid (candidate, reference) pairs from turn records while
    excluding failed parses where response == "None" or empty.
    """
    candidates, references = [], []
    for r in records:
        if deliberation_only and not r.get("entered_deliberation", True):
            continue
        cand = (r.get("response") or r.get("generated_response") or "").strip()
        ref = (r.get("reference") or r.get("gold_reference") or "").strip()
        if not cand or cand == "None" or not ref or ref == "None":
            continue
        candidates.append(cand)
        references.append(ref)
    return candidates, references


def automatic_response_report(candidates, references):
    """
    Computes all 7 automatic response-quality metrics reported in MultiAgentESC
    (Xu et al., EMNLP 2025, Table 1): Distinct-1/2, BLEU-1/2/3, F1, ROUGE-L.
    """
    return {
        "n_evaluated": len(candidates),
        "distinct_1": distinct_n(candidates, 1),
        "distinct_2": distinct_n(candidates, 2),
        "bleu_1": corpus_bleu(candidates, references, 1),
        "bleu_2": corpus_bleu(candidates, references, 2),
        "bleu_3": corpus_bleu(candidates, references, 3),
        "f1": sum(unigram_f1(c, r) for c, r in zip(candidates, references)) / len(candidates) if candidates else 0.0,
        "rouge_l": sum(rouge_l(c, r) for c, r in zip(candidates, references)) / len(candidates) if candidates else 0.0,
    }


def make_gpt4o_judge_prompt(context_or_situation, response, situation="", gold_reference=None):
    """
    Constructs the 5-dimension LLM-as-a-Judge evaluation prompt following
    MultiAgentESC (Xu et al., EMNLP 2025, Section 5.2, Section 5.3, Appendix C).

    Evaluates an emotional support response on a 1 to 5 integer scale across:
      1. Fluency: coherence, grammaticality, and naturalness of the response.
      2. Identification: effectiveness in identifying and exploring the user's
         emotional problem and underlying situation.
      3. Comforting: effectiveness in comforting the user, validating their
         feelings, and conveying warmth/empathy.
      4. Suggestion: helpfulness, appropriateness, and constructiveness of
         suggestions or coping guidance offered for the user's situation.
      5. Overall: overall effectiveness of the emotional support provided.
    """
    situation_block = f"### User Situation\n{situation.strip()}\n\n" if situation and situation.strip() else ""
    ref_block = (
        f"### Reference Supporter Response (for context only)\n{gold_reference.strip()}\n\n"
        if gold_reference and gold_reference.strip()
        else ""
    )
    return (
        "### Instruction\n"
        "You are an expert psychological counseling evaluator. You will be provided "
        "with an ongoing emotional support conversation between a help-seeker ('User') "
        "and a supporter ('Assistant'), followed by a candidate response from the Assistant.\n\n"
        "Please rate the candidate response on a scale from 1 (lowest) to 5 (highest) "
        "across the following five dimensions (following Liu et al., 2021 and Xu et al., 2025):\n"
        "1. Fluency (1-5): Is the response fluent, grammatical, coherent, and smooth in the dialogue context?\n"
        "2. Identification (1-5): How effectively does the response identify, acknowledge, or explore the user's emotional problem and situation?\n"
        "3. Comforting (1-5): How effectively does the response comfort the user, validate their feelings, and express empathy?\n"
        "4. Suggestion (1-5): How helpful, appropriate, and constructive are the suggestions or perspectives provided?\n"
        "5. Overall (1-5): Overall, how effective is this response at providing emotional support to the user?\n\n"
        f"{situation_block}"
        f"### Dialogue Context\n{context_or_situation.strip()}\n\n"
        f"{ref_block}"
        f"### Candidate Assistant Response\n{response.strip()}\n\n"
        "### Output Format\n"
        "Respond with a valid JSON object containing integer or float scores in [1, 5] "
        "for all five dimensions and a brief rationale:\n"
        '{"fluency": <1-5>, "identification": <1-5>, "comforting": <1-5>, '
        '"suggestion": <1-5>, "overall": <1-5>, "reasoning": "<short explanation>"}'
    )


def parse_llm_judge_scores(raw_output):
    """
    Parses and validates a judge LLM's response into a normalized dict with
    keys ('fluency', 'identification', 'comforting', 'suggestion', 'overall')
    in [1.0, 5.0]. Supports both JSON objects and key-value lines.
    Returns None if any required dimension is missing or out of range [1, 5].
    """
    if not raw_output or not isinstance(raw_output, str):
        return None

    text = raw_output.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    parsed_scores = {}

    # 1. Try JSON block extraction first
    json_matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for jm in json_matches:
        try:
            obj = json.loads(jm)
            if isinstance(obj, dict):
                lower_obj = {str(k).strip().lower(): v for k, v in obj.items()}
                if all(dim in lower_obj for dim in LLM_JUDGE_DIMENSIONS):
                    for dim in LLM_JUDGE_DIMENSIONS:
                        val = float(lower_obj[dim])
                        if not (1.0 <= val <= 5.0):
                            return None
                        parsed_scores[dim] = val
                    if "reasoning" in lower_obj and isinstance(lower_obj["reasoning"], str):
                        parsed_scores["reasoning"] = lower_obj["reasoning"].strip()
                    return parsed_scores
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    # 2. Fallback: parse "Fluency: 4", "Identification: 3.5", etc.
    for dim in LLM_JUDGE_DIMENSIONS:
        pattern = rf"\b{dim}\b\s*[:=\-]\s*([1-5](?:\.\d+)?)"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return None
        val = float(m.group(1))
        if not (1.0 <= val <= 5.0):
            return None
        parsed_scores[dim] = val

    return parsed_scores


def aggregate_llm_judge_scores(scored_turns):
    """
    Computes mean and standard deviation across validated per-turn judge scores.
    """
    valid = [
        t["scores"] if "scores" in t and isinstance(t["scores"], dict) else t
        for t in scored_turns
        if t and isinstance(t, dict)
    ]
    valid = [
        d for d in valid
        if all(dim in d and d[dim] is not None and 1.0 <= float(d[dim]) <= 5.0 for dim in LLM_JUDGE_DIMENSIONS)
    ]
    n = len(valid)
    if n == 0:
        return {
            "n_scored": 0,
            **{dim: None for dim in LLM_JUDGE_DIMENSIONS},
            **{f"{dim}_std": None for dim in LLM_JUDGE_DIMENSIONS},
        }

    summary = {"n_scored": n}
    for dim in LLM_JUDGE_DIMENSIONS:
        vals = [float(d[dim]) for d in valid]
        mean_val = sum(vals) / n
        variance = sum((v - mean_val) ** 2 for v in vals) / n
        summary[dim] = mean_val
        summary[f"{dim}_std"] = math.sqrt(variance)
    return summary


def evaluate_responses_with_llm_judge(
    records,
    judge_fn=None,
    config_list=None,
    cache_path=None,
    deliberation_only=True,
    max_turns=None,
    max_retries=2,
):
    """
    Evaluates turn records using an LLM judge (either a passed-in `judge_fn(prompt)`
    callable or an AutoGen ConversableAgent configured with `config_list`).

    Saves per-turn results incrementally to `cache_path` (if provided) and
    returns `{"aggregate": ..., "per_turn_scores": [...]}`.
    """
    if judge_fn is None:
        if not config_list:
            raise ValueError("Provide either judge_fn(prompt) or config_list for LLM-judge evaluation.")
        try:
            import autogen
        except ImportError:
            import integration_patch as autogen
        agent = autogen.ConversableAgent(
            name="ESC_LLM_Judge",
            system_message="You are an expert psychological counseling evaluator. Always output valid JSON.",
            llm_config={
                "config_list": config_list,
                "cache_seed": 2024,
                "temperature": 0.0,
                "max_tokens": 250,
            },
            human_input_mode="NEVER",
        )

        def _default_judge_fn(prompt_text):
            reply = agent.generate_reply(messages=[{"content": prompt_text, "role": "user"}])
            return reply if isinstance(reply, str) else str(reply)

        judge_fn = _default_judge_fn

    cached_by_turn = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for item in existing.get("per_turn_scores", []):
                tid = item.get("turn_id")
                if tid is not None and item.get("scores"):
                    cached_by_turn[str(tid)] = item
        except Exception:
            cached_by_turn = {}

    per_turn_scores = []
    evaluated_count = 0

    for idx, rec in enumerate(records):
        if deliberation_only and not rec.get("entered_deliberation", True):
            continue
        response = (rec.get("response") or rec.get("generated_response") or "").strip()
        context = (rec.get("context") or rec.get("situation") or "").strip()
        if not response or response == "None" or not context:
            continue

        if max_turns is not None and evaluated_count >= max_turns:
            break
        evaluated_count += 1

        turn_id = str(rec.get("turn_id", f"turn_{idx}"))
        if turn_id in cached_by_turn:
            per_turn_scores.append(cached_by_turn[turn_id])
            continue

        prompt = make_gpt4o_judge_prompt(
            context_or_situation=context,
            response=response,
            situation=rec.get("situation", ""),
            gold_reference=rec.get("reference") or rec.get("gold_reference"),
        )

        scores = None
        raw_reply = ""
        for _ in range(max_retries + 1):
            raw_reply = judge_fn(prompt)
            scores = parse_llm_judge_scores(raw_reply)
            if scores is not None:
                break

        entry = {
            "turn_id": turn_id,
            "gold_strategy": rec.get("gold_strategy") or rec.get("strategy"),
            "predicted_strategy": rec.get("predicted_strategy") or rec.get("pred_strategy"),
            "scores": scores,
            "raw_judge_output": raw_reply,
        }
        per_turn_scores.append(entry)

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "aggregate": aggregate_llm_judge_scores(per_turn_scores),
                        "per_turn_scores": per_turn_scores,
                    },
                    f,
                    indent=2,
                )

    result = {
        "n_eligible_candidates": evaluated_count,
        "aggregate": aggregate_llm_judge_scores(per_turn_scores),
        "per_turn_scores": per_turn_scores,
    }
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result
