"""
run_baseline.py -- Phase 1 & the data-gathering half of Phase 2.

This is a MINIMALLY modified copy of MultiAgentESC's own main.py __main__
loop -- re-read in full from the cloned repo rather than reconstructed
from memory, after an earlier draft of this script got several function
signatures wrong by guessing instead of checking. Only four things differ
from the original:

  1. `dataset` is your train/val/test split (split_conversations), not
     dataset[:100] -- the paper's own convention, which this project's
     split replaces.
  2. `get_quadruple()` -> `build_quadruple_safe()`, built from TRAIN only,
     with an explicit cache path (see strategy_aware_retrieval.py's
     docstring for exactly why the original get_quadruple() is unsafe
     once you're using a custom split).
  3. `get_strategy(...)` -> `deliberation.get_strategy(...)` -- the one
     call this project's retrieval/reranking/evidence module replaces.
     Everything downstream of it (clean_strategy, the per-strategy
     example filtering, response_with_strategy, debate/reflect/vote/judge,
     self_reflection) is untouched, byte-for-byte the original logic.
  4. Each turn also gets a flattened record (turn_id/gold_strategy/
     predicted_strategy/entered_deliberation/retrieved_strategies/
     latency) appended to `flat_results`, in the shape eval_metrics.py
     expects. The original's own richer `save` dict (emotion/cause/
     intention/ori_response/etc.) is kept too, under `full_results`, in
     case you want it for error analysis later.

This script needs your GPU + a served local LLM to run -- it cannot be
executed in a sandbox with neither, which is why it's a careful adaptation
of proven original logic rather than something tested end-to-end here.
What WAS tested here: eval_metrics.py (7/7 unit tests passing) and
strategy_aware_retrieval.py (smoke-tested against real ESConv data).
Read this script against your own multiagent.py before trusting it, the
same way you'd review any new code before running it.

Usage (run from inside your MultiAgentESC clone, with this project's src/
directory also on PYTHONPATH, e.g. `PYTHONPATH=path/to/strategy_aware_esc/src`):

    python experiments/run_baseline.py \
        --split val --n-conversations 50 --arm baseline \
        --llm-name qwen2.5:7b-instruct \
        --out results/smoke_test.json

--n-conversations caps how many conversations get pulled from the split;
each can contain several evaluated turns, so turn count will run higher
than this number. For direct control over actual turn count (and
therefore runtime), add --max-turns, e.g. --max-turns 50 for something
closer to "exactly 50 evaluated turns, stop there."

Bump --n-conversations (or --max-turns) up for the run the real confusion
matrix should be built from, once the smoke test passes -- see
INTEGRATION_NOTES.md for why initial smoke tests and real analysis runs
need different sample sizes.
"""

import argparse
import json
import sys
import time

sys.path.insert(0, "src")

from strategy_aware_retrieval import split_conversations, build_quadruple_safe  # noqa: E402
from integration_patch import PatchedStrategyDeliberation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/ESConv.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                     help="Which split to EVALUATE over. The retrieval corpus is always "
                          "built from train regardless of this flag.")
    ap.add_argument("--n-conversations", type=int, default=50,
                     help="How many CONVERSATIONS to pull from the split (each can contain "
                          "many evaluated turns -- use --max-turns to cap actual turn count "
                          "directly if you need a predictable runtime budget).")
    ap.add_argument("--max-turns", type=int, default=None,
                     help="Hard cap on evaluated (seeker, supporter) turns, independent of "
                          "--n-conversations. Stops as soon as this many turns are logged, "
                          "even mid-conversation. Leave unset to run every turn in the "
                          "--n-conversations conversations pulled.")
    ap.add_argument("--arm", choices=["baseline", "rrf", "rerank", "evidence"], default="baseline")
    ap.add_argument("--lambda-diversity", type=float, default=0.3)
    ap.add_argument("--confusable-pairs-file", default=None,
                     help="JSON [[strategyA, strategyB], ...]; required for --arm evidence.")
    ap.add_argument("--llm-name", default="qwen2.5:7b-instruct")
    ap.add_argument("--model-path", default="all-roberta-large-v1")
    ap.add_argument("--cache-path", default="embeddings_train.txt")
    ap.add_argument("--cache-path-root", default="",
                     help="Passed straight through to multiagent.py's own per-call caching.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/run.json")
    args = ap.parse_args()

    # --- everything below this line uses the original main.py's own
    # imports, unmodified -- run this script from inside your
    # MultiAgentESC clone ---
    import autogen
    from sentence_transformers import SentenceTransformer
    from prompt import get_prompt
    from multiagent import (
        is_complex, single_agent_response, get_emotion, get_cause, get_intention,
        select_strategy_by_group, response_with_strategy, debate, reflect, vote,
        judge, self_reflection,
    )
    from main import get_embeddings, clean_strategy, json2natural, set_seed

    set_seed(args.seed)
    with open(args.dataset, "r") as f:
        dataset = json.load(f)
    train_data, val_data, test_data = split_conversations(dataset, seed=args.seed)
    # NOTE: this slices CONVERSATIONS, confirmed and fixed from an earlier
    # version of this script where the flag was called --n-turns but
    # actually sliced conversations -- each of which can contain many
    # evaluated turns, so that name overstated the runtime control it
    # actually gave you. Use --max-turns for an actual turn-level budget.
    samples = {"train": train_data, "val": val_data, "test": test_data}[args.split][: args.n_conversations]

    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST", file_location=".",
        filter_dict={"model": [args.llm_name]},
    )

    model = SentenceTransformer(args.model_path)
    quadruple = build_quadruple_safe(model, train_data, args.cache_path, get_embeddings, model_name=args.model_path)

    confirmed_pairs = []
    if args.confusable_pairs_file:
        with open(args.confusable_pairs_file) as f:
            pair_data = json.load(f)
        # Reproduced and confirmed: run_confusion_analysis.py writes a DICT
        # with a "confirmed_confusable_pairs" key, not a bare list. Reading
        # json.load(f) directly as a list silently iterated over the dict's
        # string KEYS instead, turning e.g. "confirmed_confusable_pairs"
        # into a tuple of its individual characters.
        if not isinstance(pair_data, dict) or "confirmed_confusable_pairs" not in pair_data:
            raise ValueError(
                f"{args.confusable_pairs_file} doesn't look like a "
                "run_confusion_analysis.py output file (expected a dict "
                "with a 'confirmed_confusable_pairs' key)."
            )
        confirmed_pairs = [tuple(p) for p in pair_data["confirmed_confusable_pairs"] if len(p) == 2]
    deliberation = PatchedStrategyDeliberation(
        quadruple, model, arm=args.arm,
        lambda_diversity=args.lambda_diversity,
        confirmed_confusable_pairs=confirmed_pairs,
    )

    full_results = []
    flat_results = []

    for sample in samples:
        if args.max_turns is not None and len(flat_results) >= args.max_turns:
            break
        dialog = sample["dialog"]
        count = 0
        history = []

        while True:
            save = {}
            if count == len(dialog):
                break
            if args.max_turns is not None and len(flat_results) >= args.max_turns:
                break

            if count != 0 and dialog[count]["speaker"] == "supporter":
                if (count < len(dialog) - 1 and dialog[count + 1]["speaker"] != "supporter") or (count == len(dialog) - 1):
                    save["strategy"] = dialog[count]["annotation"]["strategy"]
                    save["reference"] = dialog[count]["content"].strip()
                    context = json2natural(history)
                    save["context"] = context
                    post = history[-1]["content"]
                    history.append({"content": dialog[count]["content"].strip(),
                                     "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"})
                    count += 1
                elif count < len(dialog) - 1 and dialog[count + 1]["speaker"] == "supporter":
                    save["strategy"] = f"{dialog[count]['annotation']['strategy']} and {dialog[count + 1]['annotation']['strategy']}"
                    save["reference"] = dialog[count]["content"].strip() + " " + dialog[count + 1]["content"].strip()
                    context = json2natural(history)
                    save["context"] = context
                    post = history[-1]["content"]
                    history.append({"content": dialog[count]["content"].strip(),
                                     "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"})
                    history.append({"content": dialog[count + 1]["content"].strip(),
                                     "role": "user" if dialog[count + 1]["speaker"] == "seeker" else "assistant"})
                    count += 2

                t0 = time.time()
                retrieved_strategies = []
                entered_deliberation = count > 5 and is_complex(
                    get_prompt("behavior_control").format(context=context),
                    config_list=config_list, cache_path_root=args.cache_path_root,
                )

                if not entered_deliberation:
                    response = single_agent_response(
                        get_prompt("zero_shot").format(context=context),
                        config_list=config_list, cache_path_root=args.cache_path_root,
                    )
                    save["response"] = response
                    save["pred_strategy"] = "None"
                else:
                    emotion, emo_and_reason = get_emotion(
                        get_prompt("get_emotion").format(context=context), config_list, args.cache_path_root)
                    cause, cau_and_reason = get_cause(
                        get_prompt("get_cause").format(emo_and_reason=emo_and_reason, context=context),
                        config_list, args.cache_path_root)
                    intention, int_and_reason = get_intention(
                        get_prompt("get_intention").format(emo_and_reason=emo_and_reason, cau_and_reason=cau_and_reason, context=context),
                        config_list, args.cache_path_root)

                    # the one call this project replaces:
                    pred_strategy, pairs = deliberation.get_strategy(
                        emo_and_reason, cau_and_reason, int_and_reason, context, post,
                        select_strategy_by_group, config_list, n=10,
                    )
                    retrieved_strategies = [p[1].split("]", 1)[0].strip("[").strip() for p in pairs]
                    pred_strategy = clean_strategy(pred_strategy)
                    save["emotion"], save["cause"], save["intention"] = emotion, cause, intention

                    if len(pred_strategy) == 0:
                        response = single_agent_response(
                            get_prompt("zero_shot").format(context=context),
                            config_list=config_list, cache_path_root=args.cache_path_root,
                        )
                        save["response"] = response
                        save["pred_strategy"] = "None"
                    elif len(pred_strategy) == 1:
                        examples = "\n\n".join(
                            f"{p[0]}\n{p[1]}" for p in pairs
                            if p[1].split("]", 1)[0].strip("[").strip() == pred_strategy[0]
                        ).strip()
                        response = response_with_strategy(
                            context, emo_and_reason, cau_and_reason, int_and_reason,
                            pred_strategy[0], examples, config_list, args.cache_path_root)
                        pred_strategy = pred_strategy[0]
                        save["ori_response"], save["pred_strategy"] = response, pred_strategy
                    else:
                        responses = []
                        for strat in pred_strategy:
                            examples = "\n\n".join(
                                f"{p[0]}\n{p[1]}" for p in pairs
                                if p[1].split("]", 1)[0].strip("[").strip() == strat
                            ).strip()
                            resp = response_with_strategy(
                                context, emo_and_reason, cau_and_reason, int_and_reason,
                                strat, examples, config_list, args.cache_path_root)
                            responses.append(f"[{strat}] {resp}")

                        debate_history = debate(context, emo_and_reason, cau_and_reason, int_and_reason, responses, config_list)
                        reflection_result = reflect(context, emo_and_reason, cau_and_reason, int_and_reason, debate_history, responses, config_list)
                        strats, responses = vote(reflection_result)

                        if len(strats) == 1 and strats[0] != "None":
                            pred_strategy, response = strats[0].strip(), responses[0].strip()
                            save["ori_response"], save["pred_strategy"] = response, pred_strategy
                        else:
                            pred_strategy, response = judge(context, strats, responses, config_list, args.cache_path_root)
                            save["ori_response"], save["pred_strategy"] = response, pred_strategy

                    _, response = self_reflection(context, pred_strategy, response, config_list, args.cache_path_root)
                    save["response"] = response

                latency = time.time() - t0
                full_results.append(save)
                # Confirmed real issue in an earlier version: silently taking
                # save["strategy"].split(" and ")[0] discarded the second
                # label on composite turns (two consecutive supporter
                # utterances get merged into "StrategyA and StrategyB" by
                # MultiAgentESC's own logic) and compared the model's ONE
                # prediction against only the first of two legitimate gold
                # labels -- a correct prediction matching the second label
                # would get scored as a flat error. Store the raw label and
                # a flag instead; eval_metrics.py excludes composite turns
                # from single-label Macro-F1/confusion-matrix and reports
                # them as their own separate rate, rather than mangling them.
                is_composite = " and " in save["strategy"]
                flat_results.append({
                    "turn_id": f"{sample.get('situation', 'conv')[:24]}_{count}",
                    "gold_strategy": save["strategy"],  # may be "StrategyA and StrategyB"
                    "predicted_strategy": save["pred_strategy"],
                    "is_composite_gold": is_composite,
                    "entered_deliberation": entered_deliberation,
                    "retrieved_strategies": retrieved_strategies,
                    "latency_seconds": latency,
                })

            else:
                history.append({"content": dialog[count]["content"].strip(),
                                 "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"})
                count += 1

    out = {
        "config": vars(args),
        "n_train": len(train_data), "n_val": len(val_data), "n_test": len(test_data),
        "n_evaluated_turns": len(flat_results),
        "flat_results": flat_results,  # feed this to eval_metrics.py
        "full_results": full_results,  # original main.py's own richer per-turn fields, for error analysis
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(flat_results)} turns to {args.out}")


if __name__ == "__main__":
    main()
