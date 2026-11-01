"""
run_llm_judge.py -- LLM-as-a-Judge evaluation runner across ablation arms.

Scores generated supporter responses on the 5 dimensions from MultiAgentESC
(Xu et al., EMNLP 2025, Section 5.2, Section 5.3, and Appendix C):
  Fluency, Identification, Comforting, Suggestion, and Overall (1-5 scale).

Updates each result JSON in-place (under the `"llm_judge"` key) and also saves
per-response judge outputs and aggregate scores to `--out-dir` so
`experiments/analyze_results.py` automatically includes them in the unified
comparison table and plots.

Usage:
    python experiments/run_llm_judge.py \
        --results results/test_A_baseline.json results/test_B_rrf.json \
                  results/test_C_rerank.json results/test_D_evidence.json \
        --llm-name qwen2.5:7b-instruct
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
from response_metrics import (  # noqa: E402
    LLM_JUDGE_DIMENSIONS,
    evaluate_responses_with_llm_judge,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="One or more experiment result JSON files from run_baseline.py")
    ap.add_argument("--llm-name", default="qwen2.5:7b-instruct",
                    help="Judge model name in OAI_CONFIG_LIST (e.g., qwen2.5:7b-instruct or gpt-4o)")
    ap.add_argument("--config-file", default="OAI_CONFIG_LIST")
    ap.add_argument("--all-turns", action="store_true",
                    help="Score all turns including early zero-shot turns (default: deliberation turns only)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Optional cap on number of judged turns per arm")
    ap.add_argument("--mock-judge", action="store_true",
                    help="Run with a deterministic mock judge for pipeline verification without an LLM server")
    args = ap.parse_args()

    judge_fn = None
    config_list = None
    if args.mock_judge:
        def _mock_judge(prompt_text):
            # Deterministic heuristic for offline pipeline testing
            words = len(prompt_text.split())
            base = 3 + (words % 2)
            return json.dumps({
                "fluency": min(5, base + 1),
                "identification": base,
                "comforting": base,
                "suggestion": base,
                "overall": base,
                "reasoning": "Mock evaluation score for offline verification.",
            })
        judge_fn = _mock_judge
    else:
        try:
            import autogen
        except ImportError:
            import integration_patch as autogen
        config_list = autogen.config_list_from_json(
            env_or_file=args.config_file,
            file_location=".",
            filter_dict={"model": [args.llm_name]},
        )
        if not config_list:
            raise RuntimeError(
                f"Model {args.llm_name!r} not found in {args.config_file}. "
                "Add the judge model configuration before running."
            )

    summary_rows = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("full_results") or data.get("flat_results") or []
        cache_path = path + ".judge_scores.json"
        judge_res = evaluate_responses_with_llm_judge(
            records,
            judge_fn=judge_fn,
            config_list=config_list,
            cache_path=cache_path,
            deliberation_only=not args.all_turns,
            max_turns=args.max_turns,
        )
        data["llm_judge"] = judge_res
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        agg = judge_res["aggregate"]
        summary_rows.append({
            "file": path,
            "n_scored": agg["n_scored"],
            **{dim: round(agg[dim], 4) if agg[dim] is not None else None for dim in LLM_JUDGE_DIMENSIONS},
        })
        print(f"Scored {agg['n_scored']} turns for {path} -> saved to {path} and {cache_path}")

    print("\nLLM Judge Summary:")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        os.makedirs("results", exist_ok=True)
        sample_file = "results/_selftest_judge.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump({
                "full_results": [
                    {
                        "turn_id": "conv001_1",
                        "situation": "Feeling overwhelmed by exams.",
                        "context": "User: I cannot sleep because of my upcoming finals.",
                        "response": "It sounds like the exam pressure is really weighing on you right now.",
                        "reference": "I hear how stressful those finals are for you.",
                        "strategy": "Reflection of feelings",
                        "pred_strategy": "Reflection of feelings",
                        "entered_deliberation": True,
                    }
                ],
                "flat_results": [],
            }, f, indent=2)
        sys.argv = ["run_llm_judge.py", "--results", sample_file, "--mock-judge"]
        print("No args given -- running self-test with mock judge.\n")
    main()
