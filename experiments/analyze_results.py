"""
analyze_results.py -- Unified Evaluation & Ablation Analysis Report Generator.

Produces a unified comparison across all four configurations (A: Baseline,
B: +RRF, C: +Reranking, D: +Contrastive Evidence), including:
  1. Turn & Eligibility Accounting (total turns, none_rate, composite_rate,
     n_eligible_turns used for Macro-F1 and JS divergence, n_response_turns).
  2. Strategy-Selection Macro-F1 & Per-Strategy Precision/Recall/F1/Support.
  3. Strategy Utilization Distributions & BOTH clearly defined JS Divergence
     metrics:
       - `js_divergence_vs_test_gold` (predicted vs. eligible test-set gold)
       - `js_divergence_vs_esconv_reference` (predicted vs. ESConv reference corpus)
  4. Confused-Strategy Evaluation Report (frozen validation-discovered pairs,
     accuracy per pair, mutual confusion count, and evaluated support `n`).
  5. Automatic Response-Quality Metrics (Distinct-1/2, BLEU-1/2/3, F1, ROUGE-L)
     both on deliberation turns and across all valid turns (excluding "None").
  6. LLM-Judged Response-Quality Scores (Fluency, Identification, Comforting,
     Suggestion, Overall, plus n_scored) when present.
  7. Retrieval-Specific Latency & Overhead (mean, median, p95 in ms, plus
     absolute +ms and relative multiplier vs. Arm A semantic-only retrieval).
  8. Publication-ready figures and a unified Markdown report (`results/final_evaluation_report.md`).
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
from eval_metrics import (  # noqa: E402
    STRATEGIES,
    full_report,
    compute_esconv_reference_distribution,
)
from response_metrics import (  # noqa: E402
    LLM_JUDGE_DIMENSIONS,
    automatic_response_report,
    filter_valid_response_pairs,
    aggregate_llm_judge_scores,
)
from strategy_aware_retrieval import ensure_deliberation_calibrated  # noqa: E402


def _round_or_none(val, digits=4):
    return round(val, digits) if isinstance(val, (int, float)) else None


def build_comparison_table(reports, labels):
    """
    Constructs the unified comparison table across all experimental arms,
    including retrieval latency overhead relative to the first arm (Arm A).
    """
    base_ret_mean = None
    if reports and reports[0].get("latency", {}).get("retrieval_ms", {}).get("mean") is not None:
        base_ret_mean = reports[0]["latency"]["retrieval_ms"]["mean"]

    rows = []
    for label, r in zip(labels, reports):
        ret_lat = r.get("latency", {}).get("retrieval_ms", {})
        turn_lat = r.get("latency", {}).get("deliberation_turn_s", {})
        ret_mean = ret_lat.get("mean")

        overhead_ms = (ret_mean - base_ret_mean) if (ret_mean is not None and base_ret_mean is not None) else None
        overhead_ratio = (
            (ret_mean / base_ret_mean)
            if (ret_mean is not None and base_ret_mean and base_ret_mean > 0)
            else None
        )

        row = {
            "arm": label,
            "n_turns": r["none"]["total"],
            "n_eligible_turns": r["n_eligible_turns"],
            "none_count": r["none"]["none_count"],
            "none_rate": _round_or_none(r["none"]["none_rate"]),
            "composite_count": r["composite"]["composite_count"],
            "composite_rate": _round_or_none(r["composite"]["composite_rate"]),
            "macro_f1": _round_or_none(r["macro_f1"]),
            "js_divergence_vs_test_gold": _round_or_none(r["js_divergence_vs_test_gold"]),
            "js_divergence_vs_esconv_reference": _round_or_none(r.get("js_divergence_vs_esconv_reference")),
            "retrieval_latency_mean_ms": _round_or_none(ret_mean, 3),
            "retrieval_latency_median_ms": _round_or_none(ret_lat.get("median"), 3),
            "retrieval_latency_p95_ms": _round_or_none(ret_lat.get("p95"), 3),
            "retrieval_overhead_vs_baseline_ms": _round_or_none(overhead_ms, 3),
            "retrieval_overhead_ratio_vs_baseline": _round_or_none(overhead_ratio, 3),
            "turn_latency_mean_s": _round_or_none(turn_lat.get("mean"), 3),
            "turn_latency_median_s": _round_or_none(turn_lat.get("median"), 3),
            "turn_latency_p95_s": _round_or_none(turn_lat.get("p95"), 3),
        }

        if "confusable_pair_accuracy" in r:
            pair_accs = []
            for pair_name, pstats in r["confusable_pair_accuracy"].items():
                row[f"pair_acc ({pair_name})"] = _round_or_none(pstats["accuracy"])
                row[f"pair_n ({pair_name})"] = pstats["n"]
                if pstats["accuracy"] is not None:
                    pair_accs.append(pstats["accuracy"])
            row["confusable_pairs_mean_accuracy"] = _round_or_none(
                sum(pair_accs) / len(pair_accs) if pair_accs else None
            )

        if "response_metrics_deliberation" in r:
            rm_d = r["response_metrics_deliberation"]
            row["n_response_deliberation_turns"] = rm_d.get("n_evaluated", 0)
            for k, v in rm_d.items():
                if k != "n_evaluated":
                    row[f"delib_response_{k}"] = _round_or_none(v)

        if "response_metrics" in r:
            rm = r["response_metrics"]
            row["n_response_all_turns"] = rm.get("n_evaluated", 0)
            for k, v in rm.items():
                if k != "n_evaluated":
                    row[f"response_{k}"] = _round_or_none(v)

        if "llm_judge" in r and r["llm_judge"].get("n_scored", 0) > 0:
            jg = r["llm_judge"]
            row["n_llm_judged_turns"] = jg["n_scored"]
            for dim in LLM_JUDGE_DIMENSIONS:
                row[f"judge_{dim}"] = _round_or_none(jg.get(dim))

        rows.append(row)
    return rows


def generate_markdown_report(table, reports, labels, ref_meta=None, confusable_pairs=None):
    """Generates a self-contained Markdown report summarizing all metrics and tables."""
    lines = [
        "# MultiAgentESC Strategy-Aware Hybrid Retrieval — Unified Evaluation Report",
        "",
        "## 1. Turn Eligibility & Strategy-Selection Summary",
        "",
        "- **Eligible Turns (`n_eligible_turns`):** Single-label supporter turns that entered Strategy Deliberation (`predicted_strategy != 'None'`, `gold_strategy != 'None'`, and `not is_composite_gold`). Both `Macro-F1` and `JS Divergence (vs. Test Gold)` use this exact same set of eligible turns.",
    ]
    if ref_meta:
        lines.append(
            f"- **Reference ESConv Distribution (`JS Divergence vs. ESConv Reference`):** Computed over all **{ref_meta['n_single_label_turns']}** single-label supporter turns across **{ref_meta['n_conversations']}** conversations in the `{ref_meta['split']}` ESConv corpus (excluding {ref_meta['n_composite_turns_excluded']} merged multi-strategy turns)."
        )
    lines.extend([
        "",
        "| Configuration | Total Turns | `None` Rate (Skipped Delib.) | Composite-Gold Rate | Eligible Delib. Turns (`n`) | Strategy Macro-F1 | JS Div (vs. Eligible Test Gold) | JS Div (vs. ESConv Reference) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in table:
        js_ref_str = f"{row['js_divergence_vs_esconv_reference']:.4f}" if row.get("js_divergence_vs_esconv_reference") is not None else "N/A"
        lines.append(
            f"| {row['arm']} | {row['n_turns']} | {row['none_rate']:.2%} ({row['none_count']}) | "
            f"{row['composite_rate']:.2%} ({row['composite_count']}) | {row['n_eligible_turns']} | "
            f"**{row['macro_f1']:.4f}** | {row['js_divergence_vs_test_gold']:.4f} | {js_ref_str} |"
        )

    # 2. Strategy Utilization Distribution Table
    lines.extend([
        "",
        "## 2. Strategy Utilization Distributions Across Configurations",
        "",
        "| Strategy | ESConv Reference | Test Gold (All Single-Label) | Test Gold (Eligible Delib.) | " + " | ".join(labels) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(labels),
    ])
    ref_dist = ref_meta["distribution"] if ref_meta and "distribution" in ref_meta else {}
    first_rep = reports[0] if reports else {}
    for s in STRATEGIES:
        ref_v = f"{ref_dist[s]:.2%}" if s in ref_dist else "N/A"
        all_g = f"{first_rep.get('gold_distribution_all_single_label', {}).get(s, 0.0):.2%}"
        elig_g = f"{first_rep.get('gold_distribution', {}).get(s, 0.0):.2%}"
        arm_vals = " | ".join(f"{r['predicted_distribution'].get(s, 0.0):.2%}" for r in reports)
        lines.append(f"| {s} | {ref_v} | {all_g} | {elig_g} | {arm_vals} |")

    # 3. Confusable-Pair Accuracy Table
    if confusable_pairs:
        lines.extend([
            "",
            "## 3. Confused-Strategy Evaluation (Validation-Frozen Confusable Pairs)",
            "",
            "| Frozen Confusable Strategy Pair | Evaluated Test Turns (`n`) | " + " | ".join(f"{lbl} Accuracy (Mutual Confusions)" for lbl in labels) + " |",
            "|---|---:|" + "---:|" * len(labels),
        ])
        for a, b in confusable_pairs:
            key = f"{a} vs {b}"
            n_val = reports[0].get("confusable_pair_accuracy", {}).get(key, {}).get("n", 0) if reports else 0
            cells = []
            for r in reports:
                pstat = r.get("confusable_pair_accuracy", {}).get(key, {})
                acc = pstat.get("accuracy")
                mut = pstat.get("mutual_confusions", 0)
                cells.append(f"{acc:.2%} ({mut} mutual)" if acc is not None else "N/A (n=0)")
            lines.append(f"| **{a}** vs. **{b}** | {n_val} | " + " | ".join(cells) + " |")

    # 4. Automatic Response-Quality Table
    if any("response_metrics" in r for r in reports):
        lines.extend([
            "",
            "## 4. Automatic Response-Quality Metrics",
            "",
            "### 4a. Deliberation Turns Only (`entered_deliberation == True`, excluding `'None'` responses)",
            "",
            "| Configuration | Evaluated Turns (`n`) | Distinct-1 | Distinct-2 | BLEU-1 | BLEU-2 | BLEU-3 | Unigram F1 | ROUGE-L |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, r in zip(labels, reports):
            rm = r.get("response_metrics_deliberation") or r.get("response_metrics") or {}
            if rm:
                lines.append(
                    f"| {label} | {rm.get('n_evaluated', 0)} | {rm.get('distinct_1', 0.0):.4f} | "
                    f"{rm.get('distinct_2', 0.0):.4f} | {rm.get('bleu_1', 0.0):.4f} | "
                    f"{rm.get('bleu_2', 0.0):.4f} | {rm.get('bleu_3', 0.0):.4f} | "
                    f"{rm.get('f1', 0.0):.4f} | {rm.get('rouge_l', 0.0):.4f} |"
                )

        lines.extend([
            "",
            "### 4b. All Valid Turns (Deliberation + Zero-Shot Early Turns)",
            "",
            "| Configuration | Evaluated Turns (`n`) | Distinct-1 | Distinct-2 | BLEU-1 | BLEU-2 | BLEU-3 | Unigram F1 | ROUGE-L |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, r in zip(labels, reports):
            rm = r.get("response_metrics") or {}
            if rm:
                lines.append(
                    f"| {label} | {rm.get('n_evaluated', 0)} | {rm.get('distinct_1', 0.0):.4f} | "
                    f"{rm.get('distinct_2', 0.0):.4f} | {rm.get('bleu_1', 0.0):.4f} | "
                    f"{rm.get('bleu_2', 0.0):.4f} | {rm.get('bleu_3', 0.0):.4f} | "
                    f"{rm.get('f1', 0.0):.4f} | {rm.get('rouge_l', 0.0):.4f} |"
                )

    # 5. LLM-Judge Response Quality Table
    if any(r.get("llm_judge", {}).get("n_scored", 0) > 0 for r in reports):
        lines.extend([
            "",
            "## 5. LLM-Judged Response-Quality Metrics (1–5 Scale)",
            "",
            "| Configuration | Scored Turns (`n`) | Fluency | Identification | Comforting | Suggestion | Overall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, r in zip(labels, reports):
            jg = r.get("llm_judge", {})
            if jg.get("n_scored", 0) > 0:
                lines.append(
                    f"| {label} | {jg['n_scored']} | {jg['fluency']:.3f} | {jg['identification']:.3f} | "
                    f"{jg['comforting']:.3f} | {jg['suggestion']:.3f} | **{jg['overall']:.3f}** |"
                )
            else:
                lines.append(f"| {label} | 0 | N/A | N/A | N/A | N/A | N/A |")

    # 6. Retrieval Latency & Overhead Table
    lines.extend([
        "",
        "## 6. Retrieval-Module Latency & Overhead vs. Semantic Baseline",
        "",
        "| Configuration | Delib. Turns (`n`) | Retrieval Mean (ms) | Retrieval Median (ms) | Retrieval p95 (ms) | Overhead vs. Arm A (ms) | Relative Speed Ratio | End-to-End Turn Mean (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row, r in zip(table, reports):
        n_ret = r.get("latency", {}).get("retrieval_ms", {}).get("n", 0)
        mean_ms = f"{row['retrieval_latency_mean_ms']:.3f}" if row["retrieval_latency_mean_ms"] is not None else "N/A"
        med_ms = f"{row['retrieval_latency_median_ms']:.3f}" if row["retrieval_latency_median_ms"] is not None else "N/A"
        p95_ms = f"{row['retrieval_latency_p95_ms']:.3f}" if row["retrieval_latency_p95_ms"] is not None else "N/A"
        ovh_ms = f"{row['retrieval_overhead_vs_baseline_ms']:+.3f}" if row["retrieval_overhead_vs_baseline_ms"] is not None else "N/A"
        ovh_rat = f"{row['retrieval_overhead_ratio_vs_baseline']:.2f}x" if row["retrieval_overhead_ratio_vs_baseline"] is not None else "N/A"
        turn_s = f"{row['turn_latency_mean_s']:.2f}" if row["turn_latency_mean_s"] is not None else "N/A"
        lines.append(f"| {row['arm']} | {n_ret} | {mean_ms} | {med_ms} | {p95_ms} | {ovh_ms} | {ovh_rat} | {turn_s} |")

    return "\n".join(lines) + "\n"


def make_plots(reports, labels, out_dir, ref_meta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written_files = []

    # 1. Macro-F1 across arms
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [r["macro_f1"] for r in reports], color="#4472a8")
    ax.set_ylabel("Strategy-selection Macro-F1")
    ax.set_title("Macro-F1 by ablation arm")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p1 = f"{out_dir}/macro_f1_by_arm.png"
    plt.savefig(p1, dpi=150)
    plt.close(fig)
    written_files.append(p1)

    # 2. JS divergence across arms (both vs. eligible test gold and vs. ESConv reference)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = range(len(labels))
    has_ref = all("js_divergence_vs_esconv_reference" in r for r in reports)
    if has_ref:
        w = 0.36
        ax.bar([i - w / 2 for i in x], [r["js_divergence_vs_test_gold"] for r in reports], w,
               label="vs. Eligible Test Gold", color="#c9770a")
        ax.bar([i + w / 2 for i in x], [r["js_divergence_vs_esconv_reference"] for r in reports], w,
               label="vs. ESConv Reference Corpus", color="#59a14f")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend(fontsize=8)
    else:
        ax.bar(labels, [r["js_divergence_vs_test_gold"] for r in reports], color="#c9770a")
        plt.xticks(rotation=20, ha="right")
    ax.set_ylabel("Jensen-Shannon Divergence (bits)")
    ax.set_title("Strategy-distribution JS divergence by ablation arm")
    plt.tight_layout()
    p2 = f"{out_dir}/js_divergence_by_arm.png"
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    written_files.append(p2)

    # 3. Strategy distributions (test gold vs. baseline vs. final arm + optional ESConv reference)
    last = reports[-1]
    first = reports[0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(STRATEGIES))
    if len(reports) > 1:
        w = 0.26
        ax.bar([i - w for i in x], [last["gold_distribution"][s] for s in STRATEGIES], w, label="Test Gold (Eligible)")
        ax.bar(list(x), [first["predicted_distribution"][s] for s in STRATEGIES], w, label=f"{labels[0]} Predicted")
        ax.bar([i + w for i in x], [last["predicted_distribution"][s] for s in STRATEGIES], w, label=f"{labels[-1]} Predicted")
    else:
        w = 0.35
        ax.bar([i - w / 2 for i in x], [last["gold_distribution"][s] for s in STRATEGIES], w, label="Test Gold (Eligible)")
        ax.bar([i + w / 2 for i in x], [last["predicted_distribution"][s] for s in STRATEGIES], w, label=f"{labels[-1]} Predicted")
    ax.set_xticks(list(x))
    ax.set_xticklabels(STRATEGIES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Proportion of Eligible Turns")
    ax.legend(fontsize=8)
    ax.set_title("Strategy distribution: test gold vs. predicted")
    plt.tight_layout()
    p3 = f"{out_dir}/strategy_distribution.png"
    plt.savefig(p3, dpi=150)
    plt.close(fig)
    written_files.append(p3)

    # 4. Confusion matrix heatmap, last arm
    cm = last["confusion_matrix"]
    matrix = [[cm[g][p] for p in STRATEGIES] for g in STRATEGIES]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(STRATEGIES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Confusion matrix ({labels[-1]})")
    plt.colorbar(im)
    plt.tight_layout()
    p4 = f"{out_dir}/confusion_matrix_final_arm.png"
    plt.savefig(p4, dpi=150)
    plt.close(fig)
    written_files.append(p4)

    # 5. Confusable-pair accuracy across arms (if confusable pairs provided)
    if all("confusable_pair_accuracy" in r and r["confusable_pair_accuracy"] for r in reports):
        pair_names = list(reports[0]["confusable_pair_accuracy"].keys())
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(pair_names))
        w = 0.8 / len(reports)
        for i, (label, r) in enumerate(zip(labels, reports)):
            vals = [r["confusable_pair_accuracy"][p]["accuracy"] or 0.0 for p in pair_names]
            ax.bar([xi + i * w for xi in x], vals, w, label=label)
        ax.set_xticks([xi + w * (len(reports) - 1) / 2 for xi in x])
        ax.set_xticklabels(pair_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Pair Subset Accuracy")
        ax.set_title("Accuracy on validation-frozen confusable strategy pairs")
        ax.legend(fontsize=8)
        plt.tight_layout()
        p5 = f"{out_dir}/confusable_pair_accuracy_by_arm.png"
        plt.savefig(p5, dpi=150)
        plt.close(fig)
        written_files.append(p5)

    # 6. Automatic response-quality metrics across arms
    if all("response_metrics" in r for r in reports):
        metrics_to_plot = ["distinct_1", "distinct_2", "bleu_1", "bleu_2", "bleu_3", "f1", "rouge_l"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(metrics_to_plot))
        w = 0.8 / len(reports)
        for i, (label, r) in enumerate(zip(labels, reports)):
            src_metrics = r.get("response_metrics_deliberation") or r["response_metrics"]
            vals = [src_metrics[m] for m in metrics_to_plot]
            ax.bar([xi + i * w for xi in x], vals, w, label=label)
        ax.set_xticks([xi + w * (len(reports) - 1) / 2 for xi in x])
        ax.set_xticklabels(metrics_to_plot, rotation=20, ha="right")
        ax.legend(fontsize=8)
        ax.set_title("Response-quality metrics by arm")
        plt.tight_layout()
        p6 = f"{out_dir}/response_quality_by_arm.png"
        plt.savefig(p6, dpi=150)
        plt.close(fig)
        written_files.append(p6)

    # 7. Retrieval latency comparison across arms (if retrieval_latency_seconds was logged)
    if any(r.get("latency", {}).get("retrieval_ms", {}).get("mean") is not None for r in reports):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        x = range(len(labels))
        w = 0.28
        means = [r["latency"]["retrieval_ms"]["mean"] or 0.0 for r in reports]
        medians = [r["latency"]["retrieval_ms"]["median"] or 0.0 for r in reports]
        p95s = [r["latency"]["retrieval_ms"]["p95"] or 0.0 for r in reports]
        ax.bar([i - w for i in x], means, w, label="Mean (ms)", color="#4e79a7")
        ax.bar(list(x), medians, w, label="Median (ms)", color="#f28e2b")
        ax.bar([i + w for i in x], p95s, w, label="p95 (ms)", color="#e15759")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Retrieval Latency (ms / turn)")
        ax.set_title("Retrieval module latency by ablation arm")
        ax.legend(fontsize=8)
        plt.tight_layout()
        p7 = f"{out_dir}/retrieval_latency_by_arm.png"
        plt.savefig(p7, dpi=150)
        plt.close(fig)
        written_files.append(p7)

    # 8. LLM-Judge scores across arms (if present)
    if all(r.get("llm_judge", {}).get("n_scored", 0) > 0 for r in reports):
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = range(len(LLM_JUDGE_DIMENSIONS))
        w = 0.8 / len(reports)
        for i, (label, r) in enumerate(zip(labels, reports)):
            vals = [r["llm_judge"][dim] for dim in LLM_JUDGE_DIMENSIONS]
            ax.bar([xi + i * w for xi in x], vals, w, label=label)
        ax.set_xticks([xi + w * (len(reports) - 1) / 2 for xi in x])
        ax.set_xticklabels([d.capitalize() for d in LLM_JUDGE_DIMENSIONS], rotation=15, ha="right")
        ax.set_ylabel("Score (1-5 scale)")
        ax.set_ylim(0, 5.2)
        ax.set_title("LLM-judged response quality by arm")
        ax.legend(fontsize=8)
        plt.tight_layout()
        p8 = f"{out_dir}/llm_judge_by_arm.png"
        plt.savefig(p8, dpi=150)
        plt.close(fig)
        written_files.append(p8)

    print(f"Wrote {len(written_files)} figures to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=None)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--rrf", default=None)
    ap.add_argument("--rerank", default=None)
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--lambda-sweep", default=None)
    ap.add_argument("--confusable-pairs", default=None,
                    help="Path to confirmed_confusable_pairs.json from validation baseline.")
    ap.add_argument("--dataset", default="dataset/ESConv.json",
                    help="Path to ESConv.json for computing the reference ESConv strategy distribution.")
    ap.add_argument("--reference-split", choices=["all", "train"], default="all",
                    help="Which ESConv corpus slice defines the reference strategy distribution (default: all 1,300 conversations).")
    ap.add_argument("--out-dir", default="figures/final")
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--table-out", default="results/comparison_table.json")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--report-md", default="results/final_evaluation_report.md")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    if not args.results:
        arm_files = []
        arm_labels = []
        for lbl, fpath in [
            ("A: baseline", args.baseline),
            ("B: +RRF", args.rrf),
            ("C: +rerank", args.rerank),
            ("D: +evidence", args.evidence),
        ]:
            if fpath:
                arm_files.append(fpath)
                arm_labels.append(lbl)
        args.results = arm_files
        args.labels = arm_labels

    if args.figures_dir:
        args.out_dir = args.figures_dir
    if args.out_json:
        args.table_out = args.out_json
    if args.out_md:
        args.report_md = args.out_md

    assert args.results and len(args.results) == len(args.labels), "Need one label per results file"

    confusable_pairs = None
    if args.confusable_pairs and os.path.exists(args.confusable_pairs):
        with open(args.confusable_pairs, "r", encoding="utf-8") as f:
            confusable_pairs = [tuple(p) for p in json.load(f)["confirmed_confusable_pairs"]]

    ref_meta = None
    if args.dataset and os.path.exists(args.dataset):
        ref_meta = compute_esconv_reference_distribution(args.dataset, split=args.reference_split)

    reports = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = ensure_deliberation_calibrated(
            path, data, dataset_path=args.dataset, confusable_pairs=confusable_pairs
        )
        flat = data["flat_results"]
        report = full_report(
            flat,
            reference_distribution=ref_meta,
            confusable_pairs=confusable_pairs,
        )

        # Automatic response-quality metrics:
        #   - Exclude literal "None" failure strings
        #   - Report both deliberation-only turns and all valid turns
        full = data.get("full_results", [])
        source_records = full if full else flat

        # Enrich full records with entered_deliberation from flat if missing
        if full and len(full) == len(flat):
            for f_rec, fl_rec in zip(full, flat):
                if "entered_deliberation" not in f_rec and "entered_deliberation" in fl_rec:
                    f_rec["entered_deliberation"] = fl_rec["entered_deliberation"]

        all_cands, all_refs = filter_valid_response_pairs(source_records, deliberation_only=False)
        if all_cands:
            report["response_metrics"] = automatic_response_report(all_cands, all_refs)

        delib_cands, delib_refs = filter_valid_response_pairs(source_records, deliberation_only=True)
        if delib_cands:
            report["response_metrics_deliberation"] = automatic_response_report(delib_cands, delib_refs)

        # LLM-judge metrics (if present in data["llm_judge"] or sidecar cache file)
        if "llm_judge" in data and isinstance(data["llm_judge"], dict):
            jg = data["llm_judge"]
            report["llm_judge"] = jg.get("aggregate") or aggregate_llm_judge_scores(jg.get("per_turn_scores", []))
        elif os.path.exists(path + ".judge_scores.json"):
            with open(path + ".judge_scores.json", "r", encoding="utf-8") as jf:
                jg = json.load(jf)
            report["llm_judge"] = jg.get("aggregate") or aggregate_llm_judge_scores(jg.get("per_turn_scores", []))

        reports.append(report)

    table = build_comparison_table(reports, args.labels)
    print(json.dumps(table, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args.table_out)), exist_ok=True)
    with open(args.table_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "table": table,
                "esconv_reference_metadata": ref_meta,
                "confirmed_confusable_pairs": [list(p) for p in confusable_pairs] if confusable_pairs else None,
                "full_reports": reports,
            },
            f,
            indent=2,
        )

    md_report = generate_markdown_report(
        table, reports, args.labels, ref_meta=ref_meta, confusable_pairs=confusable_pairs
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.report_md)), exist_ok=True)
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Wrote unified Markdown report to {args.report_md}")

    make_plots(reports, args.labels, args.out_dir, ref_meta=ref_meta)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        import random
        random.seed(0)
        STRATS = STRATEGIES

        sample_responses = [
            "that sounds really hard to go through",
            "i hear you, that makes sense",
            "it is okay to feel this way",
            "have you thought about talking to someone",
        ]

        def fake_results(skew=0.0, ret_ms=10.0, judge_boost=0.0, n=100):
            flat, full, judge_scores = [], [], []
            for i in range(n):
                gold = random.choice(STRATS)
                entered = (i % 5 != 0)
                if not entered:
                    pred = "None"
                elif random.random() < skew:
                    pred = "Question"
                elif random.random() < 0.2:
                    pred = random.choice(STRATS)
                else:
                    pred = gold
                is_composite = random.random() < 0.05
                gold_label = f"{gold} and {random.choice(STRATS)}" if is_composite else gold
                ret_sec = (ret_ms * (0.8 + 0.4 * random.random())) / 1000.0 if entered else None
                turn_sec = 7.5 + random.random() if entered else 1.2
                resp = random.choice(sample_responses)
                ref = random.choice(sample_responses)
                flat.append({
                    "turn_id": f"turn_{i}",
                    "gold_strategy": gold_label,
                    "predicted_strategy": pred,
                    "is_composite_gold": is_composite,
                    "entered_deliberation": entered,
                    "retrieval_latency_seconds": ret_sec,
                    "latency_seconds": turn_sec,
                })
                full.append({
                    "turn_id": f"turn_{i}",
                    "response": resp,
                    "reference": ref,
                    "entered_deliberation": entered,
                })
                if entered:
                    judge_scores.append({
                        "turn_id": f"turn_{i}",
                        "scores": {
                            "fluency": min(5.0, 4.2 + judge_boost),
                            "identification": min(5.0, 3.4 + judge_boost),
                            "comforting": min(5.0, 3.8 + judge_boost),
                            "suggestion": min(5.0, 3.2 + judge_boost),
                            "overall": min(5.0, 3.6 + judge_boost),
                        },
                    })
            return {
                "flat_results": flat,
                "full_results": full,
                "llm_judge": {
                    "aggregate": aggregate_llm_judge_scores(judge_scores),
                    "per_turn_scores": judge_scores,
                },
            }

        os.makedirs("results", exist_ok=True)
        json.dump(
            {"confirmed_confusable_pairs": [["Question", "Restatement or Paraphrasing"], ["Affirmation and Reassurance", "Reflection of feelings"]]},
            open("results/_selftest_pairs.json", "w", encoding="utf-8"),
        )
        json.dump(fake_results(skew=0.4, ret_ms=9.5, judge_boost=0.0), open("results/_selftest_A.json", "w", encoding="utf-8"))
        json.dump(fake_results(skew=0.1, ret_ms=14.2, judge_boost=0.35), open("results/_selftest_D.json", "w", encoding="utf-8"))
        sys.argv = [
            "analyze_results.py",
            "--results", "results/_selftest_A.json", "results/_selftest_D.json",
            "--labels", "A: baseline (synthetic)", "D: +evidence (synthetic)",
            "--confusable-pairs", "results/_selftest_pairs.json",
            "--out-dir", "figures_selftest",
            "--table-out", "results/_selftest_comparison.json",
            "--report-md", "results/_selftest_report.md",
        ]
        print("No args given -- running self-test with synthetic data.\n")
    main()
