
"""Run the final validation + four-arm test pipeline for MultiAgentESC.

Run from the project root:
    python experiments/run_final_experiments.py --skip-judge

The script:
- Uses the existing 300-turn validation and test selections.
- Reuses the validation-selected lambda.
- Derives confusable pairs from validation only.
- Runs A/B/C/D on exactly the same selected test IDs.
- Resumes matching partial checkpoints after interruption.
- Exports case studies, plots, comparison table and report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n> " + subprocess.list2cmdline(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def selection_ids(path: Path, expected_split: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing selection file: {path}")
    selection = load_json(path)
    ids = selection.get("turn_ids")
    if selection.get("split") != expected_split:
        raise ValueError(f"{path} must have split={expected_split!r}.")
    if not isinstance(ids, list) or len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError(f"{path} must contain exactly 300 unique turn IDs.")
    return set(ids)


def is_complete(path: Path, expected_ids: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        data = load_json(path)
        flat = data.get("flat_results", [])
        ids = [r.get("turn_id") for r in flat]
        check = data.get("selection_validation") or {}
        return (
            len(ids) == 300
            and len(set(ids)) == 300
            and set(ids) == expected_ids
            and check.get("valid") is True
            and check.get("expected_turns") == 300
            and check.get("observed_turns") == 300
        )
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return False


def run_arm(
    arm: str,
    out_path: Path,
    selection_path: Path,
    expected_ids: set[str],
    pairs_path: Path | None,
    lambda_value: float,
    args: argparse.Namespace,
) -> None:
    if is_complete(out_path, expected_ids):
        print(f"\n[SKIP] {out_path.name}: exact 300-turn result already exists.")
        return

    cmd = [
        sys.executable, "experiments/run_baseline.py",
        "--split", "test",
        "--n-conversations", str(args.n_conversations),
        "--max-turns", "300",
        "--turn-ids-file", str(selection_path.relative_to(ROOT)),
        "--arm", arm,
        "--llm-name", args.llm_name,
        "--model-path", args.model_path,
        "--lambda-diversity", str(lambda_value),
        "--seed", str(args.seed),
        "--out", str(out_path.relative_to(ROOT)),
    ]

    if arm in ("rerank", "evidence"):
        if pairs_path is None:
            raise ValueError(f"{arm} requires validation-derived confusable pairs.")
        cmd.extend([
            "--confusable-pairs-file",
            str(pairs_path.relative_to(ROOT)),
        ])

    # New output paths are used. A matching partial checkpoint resumes safely.
    if not out_path.exists():
        cmd.append("--fresh")

    run(cmd, args.dry_run)

    if not args.dry_run and not is_complete(out_path, expected_ids):
        raise RuntimeError(
            f"{out_path} did not finish with exactly the 300 requested IDs. "
            "The partial checkpoint is preserved; rerun this same command to resume."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-name", default="qwen2.5:7b-instruct")
    parser.add_argument("--model-path", default="all-roberta-large-v1")
    parser.add_argument("--n-conversations", type=int, default=195)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-diversity", type=float, default=None)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)

    required = [
        ROOT / "experiments/run_baseline.py",
        ROOT / "experiments/run_confusion_analysis.py",
        ROOT / "experiments/run_llm_judge.py",
        ROOT / "experiments/export_case_studies.py",
        ROOT / "experiments/analyze_results.py",
        ROOT / "dataset/ESConv.json",
        ROOT / "OAI_CONFIG_LIST",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing project files: " + ", ".join(missing))

    val_selection = ROOT / "results/val_turn_ids.json"
    test_selection = ROOT / "results/test_turn_ids.json"
    val_ids = selection_ids(val_selection, "val")
    test_ids = selection_ids(test_selection, "test")

    if args.lambda_diversity is None:
        sweep_path = ROOT / "results/validation_lambda_sweep.json"
        if not sweep_path.is_file():
            raise FileNotFoundError("Missing results/validation_lambda_sweep.json")
        sweep = load_json(sweep_path)
        if sweep.get("split") != "val":
            raise ValueError("Lambda sweep must be based on validation data.")
        lambda_value = float(sweep["selected_lambda_diversity"])
    else:
        lambda_value = float(args.lambda_diversity)

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(f"Invalid lambda value: {lambda_value}")

    results = ROOT / "results"
    val_out = results / "final_val_baseline_300.json"
    pairs_out = results / "final_confirmed_confusable_pairs_300.json"

    test_files = [
        ("baseline", results / "final_test_A_baseline_300.json"),
        ("rrf", results / "final_test_B_rrf_300.json"),
        ("rerank", results / "final_test_C_rerank_300.json"),
        ("evidence", results / "final_test_D_evidence_300.json"),
    ]

    print("\nFINAL EXPERIMENT SETTINGS")
    print(f"Model: {args.llm_name}")
    print(f"Embedding model: {args.model_path}")
    print(f"Validation turns: {len(val_ids)}")
    print(f"Test turns: {len(test_ids)}")
    print(f"Conversations per split: 195")
    print(f"Lambda from validation: {lambda_value}")
    print("Arms: A baseline, B RRF, C reranking, D evidence")

    # 1. Validation baseline.
    if not is_complete(val_out, val_ids):
        cmd = [
            sys.executable, "experiments/run_baseline.py",
            "--split", "val",
            "--n-conversations", str(args.n_conversations),
            "--max-turns", "300",
            "--turn-ids-file", "results/val_turn_ids.json",
            "--arm", "baseline",
            "--llm-name", args.llm_name,
            "--model-path", args.model_path,
            "--seed", str(args.seed),
            "--out", "results/final_val_baseline_300.json",
        ]
        if not val_out.exists():
            cmd.append("--fresh")
        run(cmd, args.dry_run)

        if not args.dry_run and not is_complete(val_out, val_ids):
            raise RuntimeError("Validation baseline did not cover all 300 selected IDs.")
    else:
        print("\n[SKIP] Validation baseline is already complete.")

    # 2. Discover confusable strategy pairs from validation only.
    run([
        sys.executable, "experiments/run_confusion_analysis.py",
        "--baseline-results", "results/final_val_baseline_300.json",
        "--top-m", "5",
        "--out", "results/final_confirmed_confusable_pairs_300.json",
    ], args.dry_run)

    if not args.dry_run and not pairs_out.is_file():
        raise RuntimeError("Confusable-pairs analysis did not create its output.")

    # 3. Run all four test arms on identical selected IDs.
    for arm, out_path in test_files:
        run_arm(
            arm=arm,
            out_path=out_path,
            selection_path=test_selection,
            expected_ids=test_ids,
            pairs_path=pairs_out if arm in ("rerank", "evidence") else None,
            lambda_value=lambda_value,
            args=args,
        )

    result_paths = [path for _, path in test_files]

    # 4. Optional LLM judge. Run separately to save time initially.
    if not args.skip_judge:
        run([
            sys.executable, "experiments/run_llm_judge.py",
            "--results",
            *[str(p.relative_to(ROOT)) for p in result_paths],
            "--llm-name", args.llm_name,
            "--max-turns", "300",
        ], args.dry_run)

    # 5. Export representative cases.
    run([
        sys.executable, "experiments/export_case_studies.py",
        "--baseline", "results/final_test_A_baseline_300.json",
        "--modified", "results/final_test_D_evidence_300.json",
        "--confusable-pairs", "results/final_confirmed_confusable_pairs_300.json",
        "--max-cases", "10",
        "--out-json", "results/final_case_studies_300.json",
        "--out-md", "results/final_case_studies_300.md",
    ], args.dry_run)

    # 6. Generate unified quantitative report and plots.
    run([
        sys.executable, "experiments/analyze_results.py",
        "--results", *[str(p.relative_to(ROOT)) for p in result_paths],
        "--labels", "A: Baseline", "B: +RRF", "C: +Rerank", "D: +Evidence",
        "--confusable-pairs", "results/final_confirmed_confusable_pairs_300.json",
        "--lambda-sweep", "results/validation_lambda_sweep.json",
        "--out-dir", "figures/final_300",
        "--table-out", "results/final_comparison_table_300.json",
        "--report-md", "results/final_evaluation_report_300.md",
    ], args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No experiments were launched.")
    else:
        print("\nFINAL PIPELINE COMPLETE")
        print("Report: results/final_evaluation_report_300.md")
        print("Table:  results/final_comparison_table_300.json")
        print("Cases:  results/final_case_studies_300.md")
        print("Figures: figures/final_300/")


if __name__ == "__main__":
    main()
