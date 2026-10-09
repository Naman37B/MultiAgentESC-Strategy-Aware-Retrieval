"""
integration_patch.py

Concrete replacement for get_strategy() in MultiAgentESC's multiagent.py.
Import this module from your copy of multiagent.py (or copy the function
body in directly) -- it's written to match the real function's exact
signature, confirmed from the cloned repo, so it's a drop-in, not a sketch.

Original (multiagent.py):
    def get_strategy(emo_and_reason, cau_and_reason, int_and_reason, context,
                      post, quadruple, model, config_list, n=10):
        post_embedding = model.encode(post)
        can_embeddings = [...]
        similarities = util.pytorch_cos_sim(post_embedding, can_embeddings)[0].tolist()
        top_indices = heapq.nlargest(n, range(len(similarities)), key=lambda i: similarities[i])
        pairs = [...]
        examples = "\n\n".join([f"{pair[0]}\n{pair[1]}" for pair in pairs])
        strategy = select_strategy_by_group(emo_and_reason, cau_and_reason, int_and_reason, context, examples, config_list)
        return strategy, pairs

This patch only changes how `examples` gets built. `select_strategy_by_group`
-- the actual multi-agent deliberation -- is untouched, which is what makes
the four-arm ablation clean: every arm calls the same deliberation function,
only what's upstream of it changes.
"""

from strategy_aware_retrieval import StrategyAwareRetriever


class PatchedStrategyDeliberation:
    """
    One object, built once per experimental arm, holding the retriever and
    this arm's configuration. Call .get_strategy(...) with the exact same
    arguments the original free function took (plus select_strategy_by_group
    passed in, since multiagent.py isn't set up to be imported as-is).

    arm="baseline"  -> semantic-only top-n, byte-for-byte the original behavior
    arm="rrf"       -> semantic + BM25 -> RRF, no rerank, no evidence
    arm="rerank"    -> + soft strategy-diversity rerank
    arm="evidence"  -> + contrastive evidence (needs confirmed_confusable_pairs)
    """

    def __init__(self, quadruple_lines, sbert_model, arm,
                 lambda_diversity=0.3, confirmed_confusable_pairs=None):
        assert arm in ("baseline", "rrf", "rerank", "evidence"), arm
        self.retriever = StrategyAwareRetriever(quadruple_lines, sbert_model)
        self.arm = arm
        self.lambda_diversity = lambda_diversity
        self.confirmed_confusable_pairs = confirmed_confusable_pairs or []
        if arm == "evidence" and not self.confirmed_confusable_pairs:
            raise ValueError(
                "arm='evidence' needs confirmed_confusable_pairs from "
                "confusion_matrix_confusable_pairs() run on validation -- "
                "pass [] explicitly only if you really mean 'no pairs frozen yet'."
            )

    def _build_examples(self, post, n):
        if self.arm == "baseline":
            idx, _ = self.retriever._semantic_topk(post, n)
            pairs = [
                (self.retriever.posts[i], f"[{self.retriever.strategies[i]}] {self.retriever.responses[i]}")
                for i in idx
            ]
            examples = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])
            return examples, pairs

        if self.arm == "rrf":
            # RRF-fused pool, plain top-n by fused score, no diversity term
            scores = self.retriever.candidate_pool(post)
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
            pairs = [
                (self.retriever.posts[i], f"[{self.retriever.strategies[i]}] {self.retriever.responses[i]}")
                for i, _ in ranked
            ]
            examples = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])
            return examples, pairs

        if self.arm == "rerank":
            pairs, _ = self.retriever.rerank(post, n=n, lambda_diversity=self.lambda_diversity)
            examples = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])
            return examples, pairs

        # arm == "evidence"
        return self.retriever.build_examples_with_evidence(
            post, self.confirmed_confusable_pairs, n=n, lambda_diversity=self.lambda_diversity
        )

    def get_strategy(self, emo_and_reason, cau_and_reason, int_and_reason, context,
                      post, select_strategy_by_group_fn, config_list, n=10):
        examples, pairs = self._build_examples(post, n)
        strategy = select_strategy_by_group_fn(
            emo_and_reason, cau_and_reason, int_and_reason, context, examples, config_list
        )
        return strategy, pairs
