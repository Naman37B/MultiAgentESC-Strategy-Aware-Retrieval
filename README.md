
# Strategy-Aware Hybrid Retrieval for MultiAgentESC

An experimental extension of MultiAgentESC for emotional support conversations. This project investigates whether combining semantic retrieval, lexical retrieval, strategy-aware reranking, and contrastive strategy evidence improves support-strategy selection.

## Project Overview

MultiAgentESC consists of three main stages:

1. **Dialogue Analysis** — extracts relevant information about the user's emotional state, situation, and intention.
2. **Strategy Deliberation** — retrieves similar historical examples and selects appropriate support strategies.
3. **Response Generation** — generates and evaluates candidate support responses.

Our modification focuses on retrieval inside Strategy Deliberation. The goal is to help the system distinguish analogous strategies and reduce biased strategy utilization.

## What We Changed

We evaluate four configurations:

| Arm | Configuration |
|---|---|
| A — Baseline | Semantic retrieval using sentence embeddings and cosine similarity |
| B — RRF | Semantic retrieval combined with BM25 lexical retrieval using Reciprocal Rank Fusion |
| C — Reranking | RRF candidate retrieval followed by soft strategy-diversity reranking |
| D — Full system | Reranking plus contrastive strategy evidence for selected confusable strategy pairs |

The rest of the MultiAgentESC pipeline is kept unchanged for the main comparison so that we can study the effect of the retrieval modification.

## Implementation

Key files:

- `src/strategy_aware_retrieval.py` — semantic retrieval, BM25, RRF, strategy-aware reranking, and contrastive evidence.
- `src/integration_patch.py` — integrates the retrieval configurations into Strategy Deliberation.
- `experiments/run_baseline.py` — experiment runner supporting all four configurations.
- `experiments/run_confusion_analysis.py` — validation-based analysis of confused strategies.
- `experiments/run_ablation.sh` — runs the four experiment configurations.
- `experiments/analyze_results.py` — analyzes and compares experiment outputs.
- `src/eval_metrics.py` — strategy-selection evaluation metrics.
- `src/response_metrics.py` — response-quality metrics.
- `tests/` — automated metric tests.
- `INTEGRATION_NOTES.md` — implementation and experimental notes.

## Dataset and Experimental Setup

We use ESConv, a dataset of emotional-support conversations with annotated supporter strategies.

The conversation-level split uses seed 42:

| Split | Conversations |
|---|---:|
| Train | 910 |
| Validation | 195 |
| Test | 195 |
| Total | 1,300 |

The retrieval corpus is constructed from the training split only. Validation is used for identifying confusable strategy pairs and selecting parameters. The held-out test split is reserved for the final comparison.

The reported comparison uses the same local Qwen2.5-7B model across all four configurations.

**Dataset:** Obtain ESConv from its original source rather than committing a separate copy of the dataset to this repository.

## Results

The final held-out comparison evaluated up to 300 turns per configuration.

| Configuration | Strategy Macro-F1 | JS Divergence |
|---|---:|---:|
| A — Baseline | 0.0588 | 0.4819 |
| B — RRF | 0.0580 | 0.4812 |
| C — Reranking | 0.0684 | 0.5382 |
| D — Full system | 0.0771 | 0.5405 |

The full system improved Macro-F1 by approximately 31.1% relative to the baseline. However, JS divergence increased, so the results do **not** demonstrate reduced strategy-utilization bias. The findings suggest that better retrieval alone does not guarantee better strategy decisions by the language model.

These results are specific to the evaluated setup and should not be interpreted as an exact reproduction of the original paper's model-specific results.

## Setup

1. Clone this repository and install the dependencies listed in `requirements.txt`.
2. Obtain ESConv from its original source and place the dataset at `dataset/ESConv.json`, as expected by the experiment runner.
3. Configure a locally hosted, OpenAI-API-compatible model using your local configuration. Keep real credentials out of GitHub.
4. Run the validation analysis and generate the confirmed confusable-pairs file before executing the full evidence configuration.
5. Run the ablation experiment with a model name matching your local model configuration.

The ablation script expects `results/confirmed_confusable_pairs.json` to exist before it runs.

Example, after completing the validation analysis:

```bash
bash experiments/run_ablation.sh 195 300 qwen2.5:7b
```

See `INTEGRATION_NOTES.md` and the experiment scripts for configuration details.

## Acknowledgements

This project builds on the released implementation of MultiAgentESC:

- Original repository: https://github.com/MindIntLab-HFUT/MultiAgentESC
- ESConv dataset: https://github.com/thu-coai/Emotional-Support-Conversation

Original paper:

Xu et al., *MultiAgentESC: A LLM-based Multi-agent Collaboration Framework for Emotional Support Conversation*, EMNLP 2025.

Please refer to the original project for its authorship and citation details.
