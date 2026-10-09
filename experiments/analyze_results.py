"""
analyze_results.py -- Phase 8.

Loads the four ablation-arm result files, computes every metric via
eval_metrics.py (never hand-typed), and writes a comparison table plus the
suggested figures. Pure post-processing -- no GPU/model needed, fully
testable with synthetic result files (see the __main__ self-test below).

Usage:
    python experiments/analyze_results.py \
        --results results/test_A_baseline.json results/test_B_rrf.json \
                   results/test_C_rerank.json results/test_D_evidence.json \
        --labels "A: baseline" "B: +RRF" "C: +rerank" "D: +evidence" \
        --confusable-pairs results/confirmed_confusable_pairs.json \
        --out-dir figures/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")
from eval_metrics import full_report, STRATEGIES  # noqa: E402
from response_metrics import automatic_response_report  # noqa: E402


def build_comparison_table(reports, labels):
    rows = []
    for label, r in zip(labels, reports):
        row = {
            "arm": label,
            "macro_f1": round(r["macro_f1"], 4),
            "none_rate": round(r["none"]["none_rate"], 4),
            "composite_rate": round(r["composite"]["composite_rate"], 4),
            "js_divergence_vs_test_gold": round(r["js_divergence_vs_test_gold"], 4),
            "n_turns": r["none"]["total"],
        }
        if "response_metrics" in r:
            row.update({f"response_{k}": round(v, 4) for k, v in r["response_metrics"].items()})
        rows.append(row)
    return rows


def make_plots(reports, labels, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 1. Macro-F1 across arms
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [r["macro_f1"] for r in reports], color="#4472a8")
    ax.set_ylabel("Strategy-selection Macro-F1")
    ax.set_title("Macro-F1 by ablation arm")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/macro_f1_by_arm.png", dpi=150)
    plt.close(fig)

    # 2. JS divergence across arms
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [r["js_divergence_vs_test_gold"] for r in reports], color="#c9770a")
    ax.set_ylabel("JS divergence (predicted vs. test-set gold)")
    ax.set_title("Strategy-distribution bias by ablation arm")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/js_divergence_by_arm.png", dpi=150)
    plt.close(fig)

    # 3. Strategy distributions (predicted, last arm) vs test gold
    last = reports[-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(STRATEGIES))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [last["gold_distribution"][s] for s in STRATEGIES], w, label="test gold")
    ax.bar([i + w / 2 for i in x], [last["predicted_distribution"][s] for s in STRATEGIES], w, label=f"{labels[-1]} predicted")
    ax.set_xticks(list(x))
    ax.set_xticklabels(STRATEGIES, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_title("Strategy distribution: test gold vs. predicted")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/strategy_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Confusion matrix heatmap, last arm
    cm = last["confusion_matrix"]
    matrix = [[cm[g][p] for p in STRATEGIES] for g in STRATEGIES]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(STRATEGIES))); ax.set_xticklabels(STRATEGIES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(STRATEGIES))); ax.set_yticklabels(STRATEGIES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
    ax.set_title(f"Confusion matrix ({labels[-1]})")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix_final_arm.png", dpi=150)
    plt.close(fig)

    # 5. Response-quality metrics across arms, if available (see main() --
    # needs full_results' response/reference fields, not just flat_results)
    if all("response_metrics" in r for r in reports):
        metrics_to_plot = ["distinct_1", "distinct_2", "bleu_1", "bleu_2", "bleu_3", "f1", "rouge_l"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(metrics_to_plot))
        w = 0.8 / len(reports)
        for i, (label, r) in enumerate(zip(labels, reports)):
            vals = [r["response_metrics"][m] for m in metrics_to_plot]
            ax.bar([xi + i * w for xi in x], vals, w, label=label)
        ax.set_xticks([xi + w * (len(reports) - 1) / 2 for xi in x])
        ax.set_xticklabels(metrics_to_plot, rotation=20, ha="right")
        ax.legend(fontsize=8)
        ax.set_title("Response-quality metrics by arm")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/response_quality_by_arm.png", dpi=150)
        plt.close(fig)
        print(f"Wrote 5 figures to {out_dir}/ (including response-quality comparison)")
    else:
        print(f"Wrote 4 figures to {out_dir}/")
        print("(response-quality plot skipped -- pass results files that include "
              "full_results with response/reference fields to get it)")

    print("Latency: present in full_results' latency_seconds per turn, not yet "
          "aggregated into a plot here -- straightforward to add alongside the "
          "response-quality figure above using the same full_results data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--confusable-pairs", default=None)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--table-out", default="results/comparison_table.json")
    args = ap.parse_args()
    assert len(args.results) == len(args.labels), "need one label per results file"

    confusable_pairs = None
    if args.confusable_pairs:
        with open(args.confusable_pairs) as f:
            confusable_pairs = [tuple(p) for p in json.load(f)["confirmed_confusable_pairs"]]

    reports = []
    for path in args.results:
        with open(path) as f:
            data = json.load(f)
        flat = data["flat_results"]
        report = full_report(flat, confusable_pairs=confusable_pairs)

        # Confirmed gap: response_metrics.py existed and was tested, but
        # nothing called it from here -- the response/reference text lives
        # in full_results, not flat_results, since flat_results only has
        # the strategy-selection fields eval_metrics.py needs.
        full = data.get("full_results", [])
        candidates = [r["response"] for r in full if r.get("response") and r.get("reference")]
        references = [r["reference"] for r in full if r.get("response") and r.get("reference")]
        if candidates:
            report["response_metrics"] = automatic_response_report(candidates, references)

        reports.append(report)

    table = build_comparison_table(reports, args.labels)
    print(json.dumps(table, indent=2))
    with open(args.table_out, "w") as f:
        json.dump({"table": table, "full_reports": reports}, f, indent=2)
    make_plots(reports, args.labels, args.out_dir)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # self-test with synthetic data -- no GPU/model needed, proves this
        # script's own logic (not the experiment's results) is correct
        import random
        random.seed(0)
        STRATS = STRATEGIES

        sample_responses = ["that sounds really hard to go through", "i hear you, that makes sense",
                             "it is okay to feel this way", "have you thought about talking to someone"]

        def fake_results(skew=0.0, n=100):
            flat, full = [], []
            for i in range(n):
                gold = random.choice(STRATS)
                if random.random() < skew:
                    pred = "Question"  # simulate a biased arm
                elif random.random() < 0.2:
                    pred = random.choice(STRATS)
                else:
                    pred = gold
                is_composite = random.random() < 0.05
                gold_label = f"{gold} and {random.choice(STRATS)}" if is_composite else gold
                flat.append({"turn_id": i, "gold_strategy": gold_label, "predicted_strategy": pred,
                             "is_composite_gold": is_composite})
                resp = random.choice(sample_responses)
                full.append({"response": resp, "reference": random.choice(sample_responses)})
            return flat, full

        os.makedirs("results", exist_ok=True)
        flat_a, full_a = fake_results(skew=0.4)
        flat_d, full_d = fake_results(skew=0.1)
        json.dump({"flat_results": flat_a, "full_results": full_a}, open("results/_selftest_A.json", "w"))
        json.dump({"flat_results": flat_d, "full_results": full_d}, open("results/_selftest_D.json", "w"))
        sys.argv = [
            "analyze_results.py", "--results", "results/_selftest_A.json", "results/_selftest_D.json",
            "--labels", "A: baseline (synthetic)", "D: +evidence (synthetic)",
            "--out-dir", "figures_selftest",
        ]
        print("No args given -- running self-test with synthetic data.\n")
    main()
