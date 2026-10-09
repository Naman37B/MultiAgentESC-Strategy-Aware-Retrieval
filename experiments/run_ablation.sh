#!/bin/bash
# run_ablation.sh -- Phase 7: the final four-arm comparison on TEST.
#
# Deliberately just calls run_baseline.py four times rather than
# duplicating its (already fairly intricate) orchestration logic in a
# second script -- fewer places for the real logic to drift out of sync.
#
# Run this only after:
#   1. Phase 1 smoke test passed
#   2. Phase 2 (run_confusion_analysis.py) has produced
#      results/confirmed_confusable_pairs.json from a VALIDATION run
#   3. lambda_diversity has been chosen from a validation sweep (see
#      INTEGRATION_NOTES.md's suggested 0.0-0.8 sweep) and hardcoded below
#
# Usage: bash experiments/run_ablation.sh <n_conversations> [max_turns] [llm_name]
#   e.g. bash experiments/run_ablation.sh 195 "" qwen2.5:7b-instruct
#   e.g. bash experiments/run_ablation.sh 195 1000 qwen2.5:7b-instruct   # also cap at 1000 turns

set -e

# Confirmed stale from a rename in run_baseline.py that didn't get
# propagated here on an earlier pass: this takes conversations (matching
# the test split's own size) and an optional hard turn cap, not the old
# --n-turns name run_baseline.py no longer accepts.
N_CONVERSATIONS="${1:-195}"
MAX_TURNS="${2:-}"
LLM_NAME="${3:-qwen2.5:7b-instruct}"
TURN_CAP_ARGS=()
if [ -n "$MAX_TURNS" ]; then
  TURN_CAP_ARGS=(--max-turns "$MAX_TURNS")
fi
LAMBDA=0.3   # <-- set this to whatever your validation sweep selected, not a guess
PAIRS_FILE="results/confirmed_confusable_pairs.json"

if [ ! -f "$PAIRS_FILE" ]; then
  echo "ERROR: $PAIRS_FILE not found. Run run_confusion_analysis.py on a validation run first."
  exit 1
fi

echo "=== Arm A: semantic-only baseline ==="
python experiments/run_baseline.py --split test --n-conversations "$N_CONVERSATIONS" "${TURN_CAP_ARGS[@]}" --arm baseline \
  --llm-name "$LLM_NAME" --out results/test_A_baseline.json

echo "=== Arm B: semantic + BM25/RRF ==="
python experiments/run_baseline.py --split test --n-conversations "$N_CONVERSATIONS" "${TURN_CAP_ARGS[@]}" --arm rrf \
  --llm-name "$LLM_NAME" --out results/test_B_rrf.json

echo "=== Arm C: + soft strategy-diversity reranking ==="
python experiments/run_baseline.py --split test --n-conversations "$N_CONVERSATIONS" "${TURN_CAP_ARGS[@]}" --arm rerank \
  --lambda-diversity "$LAMBDA" --llm-name "$LLM_NAME" --out results/test_C_rerank.json

echo "=== Arm D: + contrastive strategy evidence ==="
python experiments/run_baseline.py --split test --n-conversations "$N_CONVERSATIONS" "${TURN_CAP_ARGS[@]}" --arm evidence \
  --lambda-diversity "$LAMBDA" --confusable-pairs-file "$PAIRS_FILE" \
  --llm-name "$LLM_NAME" --out results/test_D_evidence.json

echo ""
echo "All four arms done. Next: python experiments/analyze_results.py"
