"""
export_case_studies.py -- Qualitative side-by-side conversation case-study generator.

Directly satisfies the project proposal requirement:
  "Final deliverables: modified codebase, evaluation results, ablation analysis,
   and representative conversations showing changes in retrieved evidence and
   strategy selection."

Produces both a structured JSON (`results/case_studies.json`) and a formatted
Markdown document (`results/case_studies.md`) comparing:
  - Dialogue context, seeker post, and extracted psychological state (emotion, cause, intention)
  - Gold strategy and reference supporter response
  - Baseline (Arm A) retrieved cases, candidate strategies, final strategy, and generated response
  - Modified (Arm D) reranked retrieved cases, injected contrastive strategy cues,
    candidate strategies, final strategy, and generated response

Usage:
    python experiments/export_case_studies.py \
        --baseline results/test_A_baseline.json \
        --modified results/test_D_evidence.json \
        --confusable-pairs results/confirmed_confusable_pairs.json \
        --max-cases 8 \
        --out-json results/case_studies.json \
        --out-md results/case_studies.md
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
from eval_metrics import _is_composite_turn  # noqa: E402
from strategy_aware_retrieval import ensure_deliberation_calibrated  # noqa: E402


def _index_turns(data):
    """Builds a dict mapping turn_id -> merged turn record from flat_results + full_results."""
    flat_list = data.get("flat_results", [])
    full_list = data.get("full_results", [])
    indexed = {}
    for idx, flat in enumerate(flat_list):
        tid = str(flat.get("turn_id", f"turn_{idx}"))
        merged = dict(flat)
        if idx < len(full_list) and isinstance(full_list[idx], dict):
            for k, v in full_list[idx].items():
                if k not in merged or merged[k] in (None, "", []):
                    merged[k] = v
        indexed[tid] = merged
    return indexed


def _format_retrieved_cases(turn_rec, top_k=5):
    examples = turn_rec.get("retrieved_examples") or []
    if examples:
        formatted = []
        for item in examples[:top_k]:
            rank = item.get("rank", len(formatted) + 1)
            strat = item.get("strategy", "Unknown")
            post = item.get("post", "").strip()
            resp = item.get("response", "").strip()
            formatted.append({
                "rank": rank,
                "strategy": strat,
                "seeker_post": post,
                "supporter_response": resp,
            })
        return formatted

    # Fallback if reading a legacy file that only stored retrieved_strategies
    strats = turn_rec.get("retrieved_strategies") or []
    return [
        {
            "rank": i + 1,
            "strategy": s,
            "seeker_post": "(legacy run: re-run with updated run_baseline.py to log full retrieved text)",
            "supporter_response": "",
        }
        for i, s in enumerate(strats[:top_k])
    ]


def build_case_studies(base_data, mod_data, confusable_pairs=None, max_cases=10, top_k_examples=5):
    base_turns = _index_turns(base_data)
    mod_turns = _index_turns(mod_data)

    pair_set = set()
    if confusable_pairs:
        for a, b in confusable_pairs:
            pair_set.add(tuple(sorted((a, b))))

    candidates = []
    for tid, b_rec in base_turns.items():
        if tid not in mod_turns:
            continue
        m_rec = mod_turns[tid]
        if _is_composite_turn(b_rec):
            continue
        if not b_rec.get("entered_deliberation", True) or not m_rec.get("entered_deliberation", True):
            continue

        gold = b_rec.get("gold_strategy") or b_rec.get("strategy")
        b_pred = b_rec.get("predicted_strategy") or b_rec.get("pred_strategy")
        m_pred = m_rec.get("predicted_strategy") or m_rec.get("pred_strategy")
        if not gold or gold == "None" or b_pred == "None" or m_pred == "None":
            continue

        cues = m_rec.get("contrastive_cues") or []
        is_confusable_error = tuple(sorted((gold, b_pred))) in pair_set if gold != b_pred else False

        # Priority scoring so the most informative examples appear first:
        #   Priority 1: Baseline misclassified a frozen confusable pair AND Modified got gold right with contrastive cues
        #   Priority 2: Baseline misclassified a frozen confusable pair AND Modified got gold right
        #   Priority 3: Baseline wrong -> Modified right (with cues)
        #   Priority 4: Baseline wrong -> Modified right
        #   Priority 5: Strategy changed when contrastive cues were injected
        if b_pred != gold and m_pred == gold and is_confusable_error and cues:
            category = "confusable_pair_corrected_with_contrastive_cue"
            priority = 1
        elif b_pred != gold and m_pred == gold and is_confusable_error:
            category = "confusable_pair_corrected_by_reranking"
            priority = 2
        elif b_pred != gold and m_pred == gold and cues:
            category = "baseline_error_corrected_with_contrastive_cue"
            priority = 3
        elif b_pred != gold and m_pred == gold:
            category = "baseline_error_corrected"
            priority = 4
        elif b_pred != m_pred and cues:
            category = "strategy_shifted_under_contrastive_cue"
            priority = 5
        elif b_pred != m_pred:
            category = "strategy_shifted"
            priority = 6
        else:
            continue

        candidates.append((
            priority,
            {
                "turn_id": tid,
                "category": category,
                "situation": m_rec.get("situation") or b_rec.get("situation") or "",
                "dialogue_context": m_rec.get("context") or b_rec.get("context") or "",
                "seeker_post": m_rec.get("post") or b_rec.get("post") or "",
                "dialogue_analysis": {
                    "emotion": m_rec.get("emotion") or b_rec.get("emotion") or "",
                    "cause": m_rec.get("cause") or b_rec.get("cause") or "",
                    "intention": m_rec.get("intention") or b_rec.get("intention") or "",
                },
                "gold": {
                    "strategy": gold,
                    "reference_response": m_rec.get("gold_reference") or m_rec.get("reference") or b_rec.get("reference") or "",
                },
                "baseline_arm": {
                    "retrieved_strategy_distribution": b_rec.get("retrieved_strategies", []),
                    "retrieved_cases_top_k": _format_retrieved_cases(b_rec, top_k=top_k_examples),
                    "deliberation_candidates": b_rec.get("deliberation_candidates", []),
                    "final_strategy": b_pred,
                    "generated_response": b_rec.get("generated_response") or b_rec.get("response") or "",
                    "retrieval_latency_ms": (
                        round(b_rec["retrieval_latency_seconds"] * 1000.0, 3)
                        if b_rec.get("retrieval_latency_seconds") is not None else None
                    ),
                },
                "modified_arm": {
                    "retrieved_strategy_distribution": m_rec.get("retrieved_strategies", []),
                    "retrieved_cases_top_k": _format_retrieved_cases(m_rec, top_k=top_k_examples),
                    "contrastive_cues_injected": cues,
                    "deliberation_candidates": m_rec.get("deliberation_candidates", []),
                    "final_strategy": m_pred,
                    "generated_response": m_rec.get("generated_response") or m_rec.get("response") or "",
                    "retrieval_latency_ms": (
                        round(m_rec["retrieval_latency_seconds"] * 1000.0, 3)
                        if m_rec.get("retrieval_latency_seconds") is not None else None
                    ),
                },
            },
        ))

    candidates.sort(key=lambda item: (item[0], item[1]["turn_id"]))
    return [c[1] for c in candidates[:max_cases]]


def render_case_studies_markdown(cases):
    lines = [
        "# Representative Conversation Case Studies (Baseline vs. Strategy-Aware Retrieval)",
        "",
        f"Total representative turns exported: **{len(cases)}**",
        "",
    ]
    if not cases:
        lines.append("_No differing deliberation turns found between the provided result files._")
        return "\n".join(lines) + "\n"

    for idx, c in enumerate(cases, 1):
        lines.extend([
            f"## Case Study {idx}: `{c['turn_id']}` ({c['category']})",
            "",
            f"- **Situation:** {c['situation'] or 'N/A'}",
            f"- **Dialogue Context:** {c['dialogue_context'] or 'N/A'}",
            f"- **Last Seeker Utterance (`post`):** {c['seeker_post'] or 'N/A'}",
            f"- **Gold Strategy:** `{c['gold']['strategy']}`",
            f"- **Gold Reference Supporter Response:** {c['gold']['reference_response']}",
            "",
            "### Side-by-Side Outcome Comparison",
            "",
            "| Dimension | Arm A: Semantic Baseline | Arm D: Full Strategy-Aware System | Gold Ground Truth |",
            "|---|---|---|---|",
            f"| **Retrieved Strategy Labels (Top-10)** | {', '.join(c['baseline_arm']['retrieved_strategy_distribution']) or 'N/A'} | {', '.join(c['modified_arm']['retrieved_strategy_distribution']) or 'N/A'} | — |",
            f"| **Deliberation Candidate Strategies** | {', '.join(c['baseline_arm']['deliberation_candidates']) or 'N/A'} | {', '.join(c['modified_arm']['deliberation_candidates']) or 'N/A'} | — |",
            f"| **Final Selected Strategy** | `{c['baseline_arm']['final_strategy']}` | `{c['modified_arm']['final_strategy']}` | `{c['gold']['strategy']}` |",
            f"| **Generated Supporter Response** | {c['baseline_arm']['generated_response']} | {c['modified_arm']['generated_response']} | {c['gold']['reference_response']} |",
            "",
            "### Baseline Retrieved Examples (Top Cases)",
            "",
        ])
        for ex in c["baseline_arm"]["retrieved_cases_top_k"]:
            lines.append(
                f"{ex['rank']}. **[{ex['strategy']}]** Seeker: *{ex['seeker_post']}* -> Supporter: *{ex['supporter_response']}*"
            )
        lines.extend(["", "### Modified Retrieved Examples & Contrastive Evidence", ""])
        for ex in c["modified_arm"]["retrieved_cases_top_k"]:
            lines.append(
                f"{ex['rank']}. **[{ex['strategy']}]** Seeker: *{ex['seeker_post']}* -> Supporter: *{ex['supporter_response']}*"
            )
        lines.extend(["", "**Injected Contrastive Strategy Cues:**", ""])
        if c["modified_arm"]["contrastive_cues_injected"]:
            for cue in c["modified_arm"]["contrastive_cues_injected"]:
                lines.append(f"> {cue}")
        else:
            lines.append("_No contrastive cue triggered on this turn._")
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline results JSON (e.g. results/test_A_baseline.json)")
    ap.add_argument("--modified", default=None, help="Modified results JSON (e.g. results/test_D_evidence.json)")
    ap.add_argument("--evidence", default=None, help="Alias for --modified")
    ap.add_argument("--rerank", default=None, help="Optional Arm C rerank results JSON")
    ap.add_argument("--confusable-pairs", default=None,
                    help="Optional confirmed_confusable_pairs.json to prioritize confusable-pair recoveries")
    ap.add_argument("--max-cases", type=int, default=8)
    ap.add_argument("--top-k-examples", type=int, default=5)
    ap.add_argument("--out-json", default="results/case_studies.json")
    ap.add_argument("--out-md", default="results/case_studies.md")
    args = ap.parse_args()

    mod_path = args.modified or args.evidence
    if not mod_path:
        raise ValueError("Please provide --modified (or --evidence) pointing to results/test_D_evidence.json")

    confusable_pairs = None
    if args.confusable_pairs and os.path.exists(args.confusable_pairs):
        with open(args.confusable_pairs, "r", encoding="utf-8") as f:
            confusable_pairs = [tuple(p) for p in json.load(f).get("confirmed_confusable_pairs", [])]

    with open(args.baseline, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    if args.rerank and os.path.exists(args.rerank):
        with open(args.rerank, "r", encoding="utf-8") as rf:
            r_data = json.load(rf)
        ensure_deliberation_calibrated(args.rerank, r_data, confusable_pairs=confusable_pairs)
    with open(mod_path, "r", encoding="utf-8") as f:
        mod_data = json.load(f)
    mod_data = ensure_deliberation_calibrated(mod_path, mod_data, confusable_pairs=confusable_pairs)

    cases = build_case_studies(
        base_data, mod_data, confusable_pairs=confusable_pairs,
        max_cases=args.max_cases, top_k_examples=args.top_k_examples,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"n_cases": len(cases), "cases": cases}, f, indent=2)

    md_text = render_case_studies_markdown(cases)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Exported {len(cases)} representative case studies to {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        os.makedirs("results", exist_ok=True)
        base_sample = {
            "flat_results": [
                {
                    "turn_id": "conv007_job_loss_8",
                    "situation": "Lost job unexpectedly due to company downsizing.",
                    "post": "I feel so ashamed telling my family that I got laid off again.",
                    "gold_strategy": "Affirmation and Reassurance",
                    "predicted_strategy": "Reflection of feelings",
                    "deliberation_candidates": ["Reflection of feelings", "Question"],
                    "is_composite_gold": False,
                    "entered_deliberation": True,
                    "retrieved_strategies": ["Reflection of feelings", "Reflection of feelings", "Question"],
                    "retrieved_examples": [
                        {
                            "rank": 1,
                            "strategy": "Reflection of feelings",
                            "post": "I lost my job today and feel terrible.",
                            "response": "It sounds like you are feeling deeply hurt and overwhelmed by this sudden loss.",
                        }
                    ],
                    "contrastive_cues": [],
                    "retrieval_latency_seconds": 0.011,
                    "latency_seconds": 8.4,
                    "generated_response": "It sounds like you are feeling deep shame and anxiety about sharing this news.",
                    "gold_reference": "Being laid off during downsizing is not a reflection of your worth or hard work.",
                }
            ],
            "full_results": [],
        }
        mod_sample = {
            "flat_results": [
                {
                    "turn_id": "conv007_job_loss_8",
                    "situation": "Lost job unexpectedly due to company downsizing.",
                    "post": "I feel so ashamed telling my family that I got laid off again.",
                    "gold_strategy": "Affirmation and Reassurance",
                    "predicted_strategy": "Affirmation and Reassurance",
                    "deliberation_candidates": ["Affirmation and Reassurance", "Reflection of feelings", "Providing Suggestions"],
                    "is_composite_gold": False,
                    "entered_deliberation": True,
                    "retrieved_strategies": ["Reflection of feelings", "Affirmation and Reassurance", "Providing Suggestions"],
                    "retrieved_examples": [
                        {
                            "rank": 1,
                            "strategy": "Reflection of feelings",
                            "post": "I lost my job today and feel terrible.",
                            "response": "It sounds like you are feeling deeply hurt and overwhelmed by this sudden loss.",
                        },
                        {
                            "rank": 2,
                            "strategy": "Affirmation and Reassurance",
                            "post": "I am embarrassed to tell my partner I was let go.",
                            "response": "A company downsizing does not diminish your skills or dedication; your family cares about you.",
                        },
                    ],
                    "contrastive_cues": [
                        "Note: 'Affirmation and Reassurance' and 'Reflection of feelings' can look similar. "
                        "'Affirmation and Reassurance' means: Affirm the user's strengths, motivation, and capabilities "
                        "and provide reassurance and encouragement. 'Reflection of feelings' means: Articulate and describe "
                        "the user's feelings. Choose based on which function the situation actually calls for."
                    ],
                    "retrieval_latency_seconds": 0.016,
                    "latency_seconds": 8.9,
                    "generated_response": "A downsizing is not your fault, and you have shown resilience before—your family will want to support you.",
                    "gold_reference": "Being laid off during downsizing is not a reflection of your worth or hard work.",
                }
            ],
            "full_results": [],
        }
        pairs_sample = {"confirmed_confusable_pairs": [["Affirmation and Reassurance", "Reflection of feelings"]]}
        with open("results/_selftest_case_A.json", "w", encoding="utf-8") as f:
            json.dump(base_sample, f)
        with open("results/_selftest_case_D.json", "w", encoding="utf-8") as f:
            json.dump(mod_sample, f)
        with open("results/_selftest_pairs.json", "w", encoding="utf-8") as f:
            json.dump(pairs_sample, f)
        sys.argv = [
            "export_case_studies.py",
            "--baseline", "results/_selftest_case_A.json",
            "--modified", "results/_selftest_case_D.json",
            "--confusable-pairs", "results/_selftest_pairs.json",
            "--out-json", "results/_selftest_case_studies.json",
            "--out-md", "results/_selftest_case_studies.md",
        ]
        print("No args given -- running self-test with synthetic case study data.\n")
    main()
