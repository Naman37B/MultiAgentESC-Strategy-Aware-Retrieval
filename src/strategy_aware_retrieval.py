"""
strategy_aware_retrieval_v2.py

Second pass, fixing four issues a code review of v1 caught:

  1. confusable pairs should come from a real gold-vs-predicted confusion
     matrix on the baseline, not retrieval co-occurrence (co-occurrence is
     kept, relabeled clearly as a cheap preliminary signal only).
  2. the hard `max_per_strategy` cap is replaced with a soft, single-
     parameter MMR-style rerank, so strongly relevant results from one
     strategy are never discarded just to force balance.
  3. the arbitrary `semantic + 0.5*bm25` score is replaced with Reciprocal
     Rank Fusion (RRF), which doesn't require comparing two incomparable
     scales.
  4. the contrastive cue is now actually injected into the examples string
     that reaches Strategy Deliberation, via build_examples_with_evidence()
     -- v1 defined contrastive_cue() but never called it from the
     integration patch.

Also adds split_conversations() for a clean train/validation/test split,
since confusable-pair discovery and lambda tuning must happen on
validation, never on the held-out test set.
"""

import heapq
import json
import random
from collections import Counter
from itertools import combinations

# NOTE: numpy/rank_bm25/sentence_transformers are imported lazily, inside
# StrategyAwareRetriever specifically, NOT at module level. split_conversations()
# and confusion_matrix_confusable_pairs() are pure data processing and are used
# by scripts (like run_confusion_analysis.py) that have no reason to need the
# embedding/retrieval stack installed -- don't force that coupling.


def parse_quadruple_line(line):
    import numpy as np
    post, response, strat, emb = line.strip().split("__SEP__")
    return post, response, strat, np.array(json.loads(emb), dtype=np.float32)


def _dataset_signature(dataset, model_name):
    """
    Cheap fingerprint of (conversation count, a stable sample of content,
    model name) -- enough to catch "this cache doesn't match these inputs"
    without hashing the entire dataset on every call.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(str(model_name).encode())
    h.update(str(len(dataset)).encode())
    for sample in dataset[::max(1, len(dataset) // 20)]:  # ~20 evenly-spaced samples
        h.update(sample.get("situation", "")[:200].encode(errors="ignore"))
    return h.hexdigest()[:16]


def build_quadruple_safe(model, dataset, cache_path, get_embeddings_fn, model_name="unspecified"):
    """
    Safe replacement for main.py's own get_quadruple(), fixing issues
    confirmed directly from the repo and from review:

      1. The original does `for sample in tqdm(dataset[100:])` internally
         -- skips the first 100 items of WHATEVER list it's given, matching
         the paper's own eval-set convention. Passing it train_data would
         silently drop 100 real conversations. This wrapper indexes the
         FULL list passed in, no internal slice -- split with
         split_conversations() before calling this.
      2. The original caches to a hardcoded "./embeddings.txt" and skips
         rebuilding if that path exists, regardless of what dataset was
         passed. This wrapper takes an explicit cache_path, so different
         splits/experiments can't collide.
      3. (confirmed gap from review) Path-existence alone can't tell a
         cache built for THIS dataset/model apart from one built for a
         different split, seed, or model, and a process killed mid-write
         would leave a partial file that still "exists" and would be
         silently treated as complete. Fixed two ways: a manifest file
         (cache_path + ".manifest.json") records a signature of the
         dataset+model that produced the cache, checked before reuse; and
         the cache is built in a temp file and atomically renamed into
         place only once writing fully completes, so an interrupted build
         never leaves a convincing partial cache at the real path --
         os.replace() either fully lands or only the temp file is left
         dangling, never a half-written file at cache_path itself.

    get_embeddings_fn: pass in the get_embeddings function already defined
    in main.py (main.py isn't set up to be imported as a module as-is, so
    this takes it as an argument rather than importing it directly).
    model_name: a string identifying the embedding model, purely for the
    manifest signature -- doesn't need to be exact, just stable across runs
    that should be treated as "the same cache."
    """
    import os
    import json as _json
    from tqdm import tqdm

    manifest_path = cache_path + ".manifest.json"
    signature = _dataset_signature(dataset, model_name)

    if os.path.exists(cache_path) and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = _json.load(f)
        if manifest.get("signature") == signature and manifest.get("complete"):
            with open(cache_path, "r") as f:
                return f.readlines()
        # signature mismatch or incomplete manifest -- don't trust this cache, rebuild

    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as txt:
        for sample in tqdm(dataset):  # full list -- no [100:] slice
            targets = []
            dialog = sample["dialog"]
            for count in range(len(dialog) - 1):
                if dialog[count]["speaker"] == "seeker" and dialog[count + 1]["speaker"] == "supporter":
                    targets.append((
                        dialog[count]["content"].strip(),
                        dialog[count + 1]["content"].strip(),
                        dialog[count + 1]["annotation"]["strategy"],
                    ))
            embeddings = get_embeddings_fn(model, targets)
            for triple, embedding in zip(targets, embeddings):
                post, response, strat = triple
                line = f"{post}__SEP__{response}__SEP__{strat}__SEP__{embedding}".replace("\n", "\\n") + "\n"
                txt.write(line)

    os.replace(tmp_path, cache_path)  # atomic on the same filesystem
    with open(manifest_path, "w") as f:
        _json.dump({"signature": signature, "complete": True, "n_conversations": len(dataset)}, f)

    with open(cache_path, "r") as f:
        return f.readlines()


def split_conversations(conversations, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Train: build the retrieval corpus from this.
    Validation: run the baseline here to build the real confusion matrix,
      and tune lambda_diversity / the RRF constant here. Freeze both
      before looking at test.
    Test: untouched until the final baseline-vs-ours comparison.
    """
    idx = list(range(len(conversations)))
    random.Random(seed).shuffle(idx)
    n_val = int(len(idx) * val_frac)
    n_test = int(len(idx) * test_frac)
    val_idx = set(idx[:n_val])
    test_idx = set(idx[n_val:n_val + n_test])
    train_idx = set(idx[n_val + n_test:])
    return (
        [conversations[i] for i in sorted(train_idx)],
        [conversations[i] for i in sorted(val_idx)],
        [conversations[i] for i in sorted(test_idx)],
    )


def confusion_matrix_confusable_pairs(gold_strategies, predicted_strategies, top_m=5, exclude_none=True):
    """
    The methodologically correct version of 'frequently confused during
    baseline analysis': run the UNMODIFIED MultiAgentESC baseline over a
    validation set, then compare pred_strategy (what main.py actually
    saved) against the ESConv gold label, turn by turn. Only needs two
    parallel lists -- doesn't touch retrieval.

    Excludes "None" predictions by default (the early/simple-turn
    zero-shot branch in main.py, which never goes through Strategy
    Deliberation at all) since those aren't strategy-selection errors.
    Report what fraction of turns were "None" separately, don't fold it
    into this.
    """
    confusion = Counter()
    for gold, pred in zip(gold_strategies, predicted_strategies):
        if exclude_none and (gold == "None" or pred == "None"):
            continue
        if gold != pred:
            confusion[tuple(sorted((gold, pred)))] += 1
    return confusion.most_common(top_m)


def retrieval_cooccurrence_pairs(retriever, sample_posts, n=10, top_m=5):
    """
    NOT a substitute for confusion_matrix_confusable_pairs(). This only
    tells you which strategies cluster together in the embedding
    neighborhood of similar posts -- useful as a cheap, LLM-free first
    guess before you've run the baseline, but it is not evidence the
    model actually confuses them. Use confusion_matrix_confusable_pairs()
    as the real, frozen answer once baseline predictions exist.
    """
    co_occurrence = Counter()
    for post in sample_posts:
        idx, _ = retriever._semantic_topk(post, n)
        strats_here = set(retriever.strategies[i] for i in idx)
        for a, b in combinations(sorted(strats_here), 2):
            co_occurrence[(a, b)] += 1
    return co_occurrence.most_common(top_m)


class StrategyAwareRetriever:
    """
    Needs rank_bm25 and sentence_transformers installed (imported here, not
    at module level -- see the note at the top of this file for why).
    """

    def __init__(self, quadruple_lines, sbert_model):
        from rank_bm25 import BM25Okapi

        self.model = sbert_model
        parsed = [parse_quadruple_line(q) for q in quadruple_lines]
        self.posts = [p[0] for p in parsed]
        self.responses = [p[1] for p in parsed]
        self.strategies = [p[2] for p in parsed]
        self.embeddings = [p[3] for p in parsed]
        self._bm25 = BM25Okapi([p.lower().split() for p in self.posts])

    def _semantic_topk(self, post, k):
        from sentence_transformers import util

        post_embedding = self.model.encode(post)
        sims = util.pytorch_cos_sim(post_embedding, self.embeddings)[0].tolist()
        idx = heapq.nlargest(k, range(len(sims)), key=lambda i: sims[i])
        return idx, sims

    def _bm25_topk(self, post, k):
        scores = self._bm25.get_scores(post.lower().split())
        idx = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
        return idx, scores

    # ---------- fix #3: Reciprocal Rank Fusion, not a hand-picked weight ---
    def candidate_pool(self, post, k_each=30, rrf_k=60):
        """
        RRF: score(doc) = sum over each ranking it appears in of
        1 / (rrf_k + rank_in_that_ranking). Standard, parameter-light way
        to combine rankings on incomparable scales (cosine similarity and
        BM25 scores aren't directly comparable) -- no weight to hand-pick.
        rrf_k=60 is the conventional default from the TREC rank-fusion
        literature; treat it as fixed, not something to tune per-run.
        """
        sem_idx, _ = self._semantic_topk(post, k_each)
        bm25_idx, _ = self._bm25_topk(post, k_each)
        scores = {}
        for rank, idx in enumerate(sem_idx):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_idx):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        return scores

    # ---------- fix #2: soft MMR-style rerank, not a hard per-strategy cap -
    def rerank(self, post, n=10, lambda_diversity=0.3):
        """
        lambda_diversity=0: pure relevance according to the RRF ranking.
        Higher lambda_diversity: progressively favors strategy diversity by
        penalizing repeated strategy labels among already-selected examples.
        This is a soft strategy-diversity preference and does not impose a
        fixed per-strategy quota, unlike v1's hard cap -- it has no way to
        know which candidate is actually "correct" at retrieval time, it
        only trades relevance against repeated strategy labels. Tune
        lambda_diversity on the VALIDATION split only.
        """
        scores = self.candidate_pool(post)
        if not scores:
            return [], []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        max_score = ranked[0][1] or 1.0

        selected, selected_strategies = [], []
        remaining = ranked
        while remaining and len(selected) < n:
            best_idx, best_val = None, float("-inf")
            for idx, score in remaining:
                relevance = score / max_score
                redundancy = (
                    selected_strategies.count(self.strategies[idx]) / len(selected_strategies)
                    if selected_strategies else 0.0
                )
                mmr = (1 - lambda_diversity) * relevance - lambda_diversity * redundancy
                if mmr > best_val:
                    best_val, best_idx = mmr, idx
            selected.append(best_idx)
            selected_strategies.append(self.strategies[best_idx])
            remaining = [(i, s) for i, s in remaining if i != best_idx]

        pairs = [(self.posts[i], f"[{self.strategies[i]}] {self.responses[i]}") for i in selected]
        return pairs, selected_strategies

    # ---------- contrastive cue text (unchanged logic from v1) -----------
    @staticmethod
    def contrastive_cue(strategy_a, strategy_b):
        from strategy import strategy as STRATEGY_DEFINITIONS  # MultiAgentESC's own strategy.py

        def_a = STRATEGY_DEFINITIONS.get(strategy_a)
        def_b = STRATEGY_DEFINITIONS.get(strategy_b)
        if not def_a or not def_b:
            return ""
        return (
            f"Note: '{strategy_a}' and '{strategy_b}' can look similar. "
            f"'{strategy_a}' means: {def_a} "
            f"'{strategy_b}' means: {def_b} "
            f"Choose based on which function the situation actually calls for."
        )

    # ---------- fix #4: actually wire the cue into what get_strategy() uses
    def build_examples_with_evidence(self, post, confirmed_confusable_pairs, n=10, lambda_diversity=0.3):
        """
        This -- not rerank() alone -- is what should replace the body of
        get_strategy() in multiagent.py. confirmed_confusable_pairs must
        come from confusion_matrix_confusable_pairs() run once on
        validation and frozen; don't pass retrieval_cooccurrence_pairs()
        output here for the real experiment.
        """
        pairs, strategies_present = self.rerank(post, n=n, lambda_diversity=lambda_diversity)
        examples = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])
        present = set(strategies_present)
        cues = [
            self.contrastive_cue(a, b)
            for a, b in confirmed_confusable_pairs
            if a in present and b in present and self.contrastive_cue(a, b)
        ]
        if cues:
            examples = examples + "\n\n" + "\n".join(dict.fromkeys(cues))  # dedupe, keep order
        return examples, pairs
