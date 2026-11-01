# MultiAgentESC Strategy-Aware Hybrid Retrieval — Unified Evaluation Report

## 1. Turn Eligibility & Strategy-Selection Summary

- **Eligible Turns (`n_eligible_turns`):** Single-label supporter turns that entered Strategy Deliberation (`predicted_strategy != 'None'`, `gold_strategy != 'None'`, and `not is_composite_gold`). Both `Macro-F1` and `JS Divergence (vs. Test Gold)` use this exact same set of eligible turns.
- **Reference ESConv Distribution (`JS Divergence vs. ESConv Reference`):** Computed over all **12261** single-label supporter turns across **1300** conversations in the `all` ESConv corpus (excluding 2716 merged multi-strategy turns).

| Configuration | Total Turns | `None` Rate (Skipped Delib.) | Composite-Gold Rate | Eligible Delib. Turns (`n`) | Strategy Macro-F1 | JS Div (vs. Eligible Test Gold) | JS Div (vs. ESConv Reference) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: Baseline | 300 | 0.00% (0) | 0.00% (0) | 300 | **0.1519** | 0.2159 | 0.1866 |
| B: +RRF | 300 | 0.00% (0) | 0.00% (0) | 300 | **0.1393** | 0.0429 | 0.0344 |
| C: +Rerank | 300 | 0.00% (0) | 0.00% (0) | 300 | **0.1430** | 0.0405 | 0.0312 |
| D: +Evidence | 300 | 0.00% (0) | 0.00% (0) | 300 | **0.1655** | 0.0286 | 0.0198 |

## 2. Strategy Utilization Distributions Across Configurations

| Strategy | ESConv Reference | Test Gold (All Single-Label) | Test Gold (Eligible Delib.) | A: Baseline | B: +RRF | C: +Rerank | D: +Evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Question | 19.96% | 19.67% | 19.67% | 4.33% | 35.67% | 35.00% | 30.67% |
| Restatement or Paraphrasing | 6.81% | 4.67% | 4.67% | 2.33% | 4.00% | 4.33% | 4.67% |
| Reflection of feelings | 7.83% | 6.67% | 6.67% | 15.00% | 6.67% | 7.00% | 8.67% |
| Self-disclosure | 8.66% | 11.67% | 11.67% | 0.33% | 5.00% | 5.00% | 6.33% |
| Affirmation and Reassurance | 15.74% | 13.33% | 13.33% | 34.00% | 11.67% | 11.67% | 12.33% |
| Providing Suggestions | 16.48% | 17.00% | 17.00% | 36.67% | 19.33% | 18.00% | 14.67% |
| Information | 6.66% | 8.67% | 8.67% | 0.33% | 2.33% | 2.33% | 2.67% |
| Others | 17.86% | 18.33% | 18.33% | 7.00% | 15.33% | 16.67% | 20.00% |

## 3. Confused-Strategy Evaluation (Validation-Frozen Confusable Pairs)

| Frozen Confusable Strategy Pair | Evaluated Test Turns (`n`) | A: Baseline Accuracy (Mutual Confusions) | B: +RRF Accuracy (Mutual Confusions) | C: +Rerank Accuracy (Mutual Confusions) | D: +Evidence Accuracy (Mutual Confusions) |
|---|---:|---:|---:|---:|---:|
| **Affirmation and Reassurance** vs. **Providing Suggestions** | 91 | 49.45% (32 mutual) | 19.78% (11 mutual) | 19.78% (9 mutual) | 20.88% (10 mutual) |
| **Affirmation and Reassurance** vs. **Others** | 95 | 23.16% (29 mutual) | 14.74% (12 mutual) | 18.95% (10 mutual) | 27.37% (9 mutual) |
| **Affirmation and Reassurance** vs. **Reflection of feelings** | 60 | 31.67% (11 mutual) | 10.00% (3 mutual) | 10.00% (3 mutual) | 10.00% (8 mutual) |
| **Affirmation and Reassurance** vs. **Question** | 99 | 21.21% (19 mutual) | 36.36% (23 mutual) | 35.35% (26 mutual) | 32.32% (25 mutual) |
| **Others** vs. **Providing Suggestions** | 106 | 36.79% (16 mutual) | 20.75% (25 mutual) | 24.53% (23 mutual) | 33.02% (16 mutual) |

## 4. Automatic Response-Quality Metrics

### 4a. Deliberation Turns Only (`entered_deliberation == True`, excluding `'None'` responses)

| Configuration | Evaluated Turns (`n`) | Distinct-1 | Distinct-2 | BLEU-1 | BLEU-2 | BLEU-3 | Unigram F1 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Baseline | 300 | 0.2825 | 0.6770 | 0.0547 | 0.0112 | 0.0040 | 0.0818 | 0.0727 |
| B: +RRF | 300 | 0.2876 | 0.6914 | 0.0678 | 0.0185 | 0.0066 | 0.0940 | 0.0785 |
| C: +Rerank | 300 | 0.2851 | 0.6846 | 0.0636 | 0.0163 | 0.0065 | 0.0905 | 0.0766 |
| D: +Evidence | 300 | 0.2812 | 0.6789 | 0.0644 | 0.0173 | 0.0067 | 0.0925 | 0.0791 |

### 4b. All Valid Turns (Deliberation + Zero-Shot Early Turns)

| Configuration | Evaluated Turns (`n`) | Distinct-1 | Distinct-2 | BLEU-1 | BLEU-2 | BLEU-3 | Unigram F1 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Baseline | 300 | 0.2825 | 0.6770 | 0.0547 | 0.0112 | 0.0040 | 0.0818 | 0.0727 |
| B: +RRF | 300 | 0.2876 | 0.6914 | 0.0678 | 0.0185 | 0.0066 | 0.0940 | 0.0785 |
| C: +Rerank | 300 | 0.2851 | 0.6846 | 0.0636 | 0.0163 | 0.0065 | 0.0905 | 0.0766 |
| D: +Evidence | 300 | 0.2812 | 0.6789 | 0.0644 | 0.0173 | 0.0067 | 0.0925 | 0.0791 |

## 6. Retrieval-Module Latency & Overhead vs. Semantic Baseline

| Configuration | Delib. Turns (`n`) | Retrieval Mean (ms) | Retrieval Median (ms) | Retrieval p95 (ms) | Overhead vs. Arm A (ms) | Relative Speed Ratio | End-to-End Turn Mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: Baseline | 300 | 1349.267 | 1297.072 | 1753.938 | +0.000 | 1.00x | 8.44 |
| B: +RRF | 300 | 1147.163 | 1070.467 | 1460.057 | -202.103 | 0.85x | 5.69 |
| C: +Rerank | 300 | 1280.940 | 1387.494 | 1730.158 | -68.327 | 0.95x | 4.95 |
| D: +Evidence | 300 | 1255.424 | 1324.472 | 1641.942 | -93.843 | 0.93x | 4.59 |
