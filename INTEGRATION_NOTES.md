# Integrating strategy_aware_retrieval_v2.py into MultiAgentESC

v2 replaces v1 after a code review caught four real issues in it. Use v2;
keep v1 only as a reference for what changed and why.

## The one function this replaces

In `multiagent.py`, `get_strategy()` currently does cosine-similarity-only
top-10 retrieval, then calls `select_strategy_by_group(...)`. Everything
before that call is what we're replacing — `select_strategy_by_group`
itself (the actual multi-agent deliberation) is untouched.

## Correct integration (fixes issue #4: v1 never actually injected the cue)

```python
from strategy_aware_retrieval_v2 import StrategyAwareRetriever, split_conversations

# Earlier reviewer catch, confirmed real: this snippet previously built the
# retriever straight from `get_quadruple(model, dataset)` -- the FULL
# dataset -- which contradicts the train-only rule below and would leak
# validation/test conversations into the retrieval corpus. Split first:
train_data, val_data, test_data = split_conversations(dataset, seed=42)
quadruple = build_quadruple_safe(model, train_data, "embeddings_train.txt", get_embeddings)
retriever = StrategyAwareRetriever(quadruple, model)
```

**Use `build_quadruple_safe()`, not `main.py`'s own `get_quadruple()`, for this.**
Checked the real function directly: `get_quadruple()` does `dataset[100:]`
internally — it skips the first 100 items of whatever list it's given — and
caches to a hardcoded `"./embeddings.txt"` that it will NOT rebuild if that
file already exists, no matter what dataset is passed in. Pass it
`train_data` expecting the full split to be indexed and you'd silently get
only 810 of 910 conversations, or a stale cache from an earlier run, with
no error either way. `build_quadruple_safe()` (now in `strategy_aware_retrieval_v2.py`)
fixes both: no internal slice, and an explicit `cache_path` so train/val/test
or different experiment runs can't collide or silently reuse each other's
embeddings.

```python

# CONFIRMED_CONFUSABLE_PAIRS must come from confusion_matrix_confusable_pairs()
# run on the VALIDATION split (see Phase 2 below), then frozen as a constant
# before touching the test set -- never from retrieval_cooccurrence_pairs(),
# which is a cheap preliminary signal only, not evidence of real confusion.
CONFIRMED_CONFUSABLE_PAIRS = [...]  # fill in after Phase 2

def get_strategy(emo_and_reason, cau_and_reason, int_and_reason, context, post,
                  quadruple, model, config_list, n=10):
    examples, pairs = retriever.build_examples_with_evidence(
        post, CONFIRMED_CONFUSABLE_PAIRS, n=n, lambda_diversity=0.3  # tune on validation
    )
    strategy = select_strategy_by_group(emo_and_reason, cau_and_reason, int_and_reason, context, examples, config_list)
    return strategy, pairs
```

`build_examples_with_evidence()` is step 1+2+3 in one call — hybrid RRF
retrieval, soft strategy-aware rerank, and the contrastive cue — so there's
one clean line to comment in/out for the ablation, not three.

## What changed from v1, and why (credit: code review caught all four)

1. **Confusable pairs**: v1's `find_confusable_pairs()` measured retrieval
   co-occurrence (which strategies show up together in the top-10 for similar
   posts) — that's not the same as the model confusing them. v2 adds
   `confusion_matrix_confusable_pairs(gold, predicted)`, which needs real
   baseline output (gold ESConv label vs. what `main.py` actually saved as
   `pred_strategy`) and is the methodologically correct match for what the
   proposal promised ("frequently confused **during baseline analysis**").
   v1's function is kept as `retrieval_cooccurrence_pairs()`, relabeled
   clearly as a cheap, LLM-free preliminary signal only.

2. **Reranking**: v1's `max_per_strategy = n//3` was a hard quota — it would
   discard highly relevant results from one strategy even when that strategy
   was genuinely the right answer for most of the top candidates. v2 replaces
   it with a single-parameter soft MMR-style rerank (`lambda_diversity`,
   0 = pure relevance, higher = more contrast-seeking) — tune this one number
   on validation rather than guessing. (The review's sketch used two separate
   parameters — relevance + λ·contrast − μ·redundancy. I simplified to one
   λ for something a 3-person team can actually tune in the time available;
   the standalone redundancy term is a reasonable extension if time allows.)

3. **Score fusion**: v1 added semantic similarity to `0.5 × normalized BM25`
   — an arbitrary constant combining two incomparable scales. v2 uses
   Reciprocal Rank Fusion (`rrf_k=60`, the standard TREC default, not
   something to tune per-run), which only looks at rank position in each
   list, sidestepping the scale problem entirely.

4. **The cue was never wired in**: v1 defined `contrastive_cue()` but the
   integration patch only called `rerank()` — the cue text never reached the
   prompt. v2's `build_examples_with_evidence()` actually appends the cue to
   `examples` when a confirmed confusable pair is present in the retrieved
   set. Verified this actually happens (not just by reading the code): ran it
   against real ESConv posts and confirmed the cue text shows up at the end
   of the generated `examples` string.

All four verified against real ESConv data in this sandbox (stand-in
embedding function, same caveat as before — no huggingface.co access here,
real model needed on your machine for the actual numbers).

## Data-split discipline (fix #5 from the review — this is the important one)

`split_conversations(data, val_frac=0.15, test_frac=0.15, seed=42)` is in
v2. Use it like this, and don't deviate:

- **Train**: build the retrieval corpus (the embeddings/BM25 index) from this only.
- **Validation**: run the *unmodified* baseline here first, build the real
  confusion matrix with `confusion_matrix_confusable_pairs()`, and tune
  `lambda_diversity` here. Freeze both before looking at test.
- **Test**: untouched until the final baseline-vs-ours comparison. If you
  peek at test to pick confusable pairs or tune lambda, the final numbers
  are leaked and not trustworthy — this is the single easiest mistake to
  make by accident, so it's worth being explicit about it with the whole team.

## Two more experimental-rigor notes from the review, both correct

- **Same backbone, both arms.** Baseline and modified must run on the exact
  same local LLM (e.g., both on Qwen2.5-7B-Instruct). Never compare a 7B
  baseline against a 14B modified run — that confounds the retrieval change
  with a backbone change. A larger-backbone sensitivity check is a fine
  *additional* experiment on the college GPU, just not the main comparison.
- **JS-divergence population.** Compute it as (this system's predicted-strategy
  distribution on the test set) vs. (the test set's *own* gold-strategy
  distribution) — not against the global 18,376-turn distribution. Use the
  global distribution as descriptive context in the writeup, not as the
  metric's denominator, since a test subset can have a different class
  balance than the full corpus by chance.

## Defining "the predicted strategy" for Macro-F1 (the review's point #7)

`main.py`'s baseline doesn't always produce one clean label: simple/early
turns (`count <= 5` or not "complex") get `pred_strategy = "None"` via a
zero-shot single-agent path that skips Strategy Deliberation entirely; complex
turns go through debate/reflect/vote/judge before landing on one final
strategy. For Macro-F1: use the strategy actually saved as `pred_strategy`
after that process completes, exclude `"None"` turns from the Macro-F1
calculation itself, and report what fraction of turns were `"None"` as a
separate number — don't fold it into accuracy either way.

## Recommended phase order (don't start by integrating everything at once)

1. **Baseline smoke test** — official MultiAgentESC + Qwen2.5-7B + ESConv, run
   end-to-end unmodified on ~20–50 validation turns first, purely to confirm
   the pipeline runs without crashing. This sample is too small to find real
   confusable pairs from (not enough turns per strategy to say anything with
   confidence) — treat it as "does this work," not as data.
2. **Confusion matrix** — once the smoke test passes, rerun the baseline on a
   substantially larger slice of validation before trusting the result —
   as a practical starting point, something like 150-250 turns balances
   getting enough turns per strategy against how long sequential multi-agent
   LLM calls actually take on a single local 7B model; scale up if time and
   compute allow. Then run `confusion_matrix_confusable_pairs()` on that. This,
   not a guess, tells you what `CONFIRMED_CONFUSABLE_PAIRS` should actually contain.
3. **Hybrid retrieval (RRF)** — compare retrieval quality before involving the LLM at all.
4. **Soft strategy-aware rerank** — tune `lambda_diversity` on validation.
5. **Contrastive evidence** — using only the pairs Phase 2 actually found.
6. **Full ablation on test** — semantic-only → +BM25/RRF → +rerank → +evidence.
