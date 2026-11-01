"""
run_baseline.py -- Experiment runner for all four ablation arms (A/B/C/D).

Supports:
  - Train/val/test conversation split (910/195/195, seed=42)
  - Safe train-only retrieval corpus indexing (build_quadruple_safe)
  - Four retrieval configurations via PatchedStrategyDeliberation:
      * baseline : semantic-only top-10 retrieval
      * rrf      : semantic + BM25 via Reciprocal Rank Fusion
      * rerank   : RRF + soft strategy-aware contrastive reranking
      * evidence : rerank + rule-based contrastive strategy cues
  - Evaluates clean single-label supporter turns after the seeker has stated
    their situation (skipping merged multi-strategy turns and opening greetings
    by default so 100% of evaluated turns are valid single-label deliberation
    turns with zero wasted LLM calls on 'None' or composite turns)
  - Separate instrumentation of retrieval-only latency (`retrieval_latency_seconds`)
    and total turn latency (`latency_seconds`)
  - Automatic checkpoint saving every 5 turns and instant resume if interrupted
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from strategy_aware_retrieval import split_conversations, build_quadruple_safe  # noqa: E402
from integration_patch import PatchedStrategyDeliberation  # noqa: E402
from eval_metrics import STRATEGIES  # noqa: E402



def _load_turn_selection(path, split, seed, dataset_sha256, n_samples, max_turns):
    """Load and validate an optional, reproducible turn-ID selection file."""
    if not path:
        return None, None, None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Turn-selection file not found: {os.path.abspath(path)}")
    with open(path, "r", encoding="utf-8") as f:
        selection = json.load(f)
    if not isinstance(selection, dict) or not isinstance(selection.get("turn_ids"), list):
        raise ValueError("Invalid turn-selection file: expected a JSON object with a 'turn_ids' list.")

    turn_ids = selection["turn_ids"]
    if not turn_ids or any(not isinstance(x, str) or not x for x in turn_ids):
        raise ValueError("Turn-selection file must contain a non-empty list of non-empty string IDs.")
    if len(turn_ids) != len(set(turn_ids)):
        raise ValueError("Turn-selection file contains duplicate turn IDs.")
    if selection.get("split") != split:
        raise ValueError(
            f"Turn-selection split is {selection.get('split')!r}, but this run uses {split!r}."
        )
    if selection.get("split_seed", seed) != seed:
        raise ValueError(
            f"Turn-selection split_seed={selection.get('split_seed')} does not match --seed={seed}."
        )
    recorded_sha = selection.get("dataset_sha256")
    if recorded_sha and recorded_sha != dataset_sha256:
        raise ValueError(
            "Dataset SHA-256 does not match the file used to generate the selected turn IDs. "
            "Regenerate the selection using the current dataset."
        )
    if selection.get("n_selected_turns", len(turn_ids)) != len(turn_ids):
        raise ValueError("Turn-selection metadata count does not match the actual turn_ids list.")
    if max_turns is not None and max_turns < len(turn_ids):
        raise ValueError(
            f"--max-turns={max_turns} is smaller than the {len(turn_ids)} selected IDs. "
            "Increase --max-turns or omit it."
        )

    conv_indices = []
    for turn_id in turn_ids:
        match = re.match(r"^conv(\d+)_", turn_id)
        if not match:
            raise ValueError(f"Unrecognized turn ID format: {turn_id!r}")
        conv_indices.append(int(match.group(1)))
    if max(conv_indices) >= n_samples:
        raise ValueError(
            f"The selection references conversation index {max(conv_indices)}, but this run only "
            f"loads {n_samples} conversations. Use --n-conversations to include the full split."
        )

    id_digest = hashlib.sha256("\n".join(sorted(turn_ids)).encode("utf-8")).hexdigest()
    return set(turn_ids), selection, id_digest


def _resume_signature(args, dataset_sha256, turn_ids_digest):
    """Only resume when every setting that can change predictions is identical."""
    keys = (
        "dataset", "split", "n_conversations", "max_turns", "arm",
        "lambda_diversity", "confusable_pairs_file", "llm_name", "model_path",
        "cache_path", "cache_path_root", "include_zero_shot_turns", "seed",
        "turn_ids_file",
    )
    signature = {key: getattr(args, key) for key in keys}
    signature["dataset_sha256"] = dataset_sha256
    signature["turn_ids_sha256"] = turn_ids_digest
    return signature

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/ESConv.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="Which split to evaluate over. Retrieval corpus is always built from train.")
    ap.add_argument("--n-conversations", type=int, default=195,
                    help="Number of conversations to pull from the split.")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Hard cap on evaluated single-label supporter turns.")
    ap.add_argument("--turn-ids-file", default=None,
                    help="Optional JSON selection file (from select_turns.py); only these IDs call the LLM.")
    ap.add_argument("--arm", choices=["baseline", "rrf", "rerank", "evidence"], default="baseline")
    ap.add_argument("--lambda-diversity", type=float, default=0.3)
    ap.add_argument("--confusable-pairs-file", default=None,
                    help="Path to confirmed_confusable_pairs.json (required for --arm evidence).")
    ap.add_argument("--llm-name", default="qwen2.5:7b-instruct")
    ap.add_argument("--model-path", default="all-roberta-large-v1")
    ap.add_argument("--cache-path", default="embeddings_train.txt")
    ap.add_argument("--cache-path-root", default="",
                    help="Passed through to multiagent.py per-call disk cache.")
    ap.add_argument("--include-zero-shot-turns", action="store_true",
                    help="Legacy mode: also run LLM calls on composite and early/simple turns that get pred='None'.")
    ap.add_argument("--run-llm-judge", action="store_true",
                    help="Also score generated responses with the 5-dimension LLM judge.")
    ap.add_argument("--fresh", action="store_true",
                    help="Start from turn 1 even if --out already contains a partial checkpoint.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/run.json")
    args = ap.parse_args()

    try:
        import autogen
    except ImportError:
        import integration_patch as autogen
    from sentence_transformers import SentenceTransformer
    from prompt import get_prompt
    from multiagent import (
        is_complex, single_agent_response, get_emotion, get_cause, get_intention,
        select_strategy_by_group, response_with_strategy, debate, reflect, vote,
        judge, self_reflection,
    )
    from main import get_embeddings, clean_strategy, json2natural, set_seed

    set_seed(args.seed)
    with open(args.dataset, "rb") as f:
        dataset_bytes = f.read()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    train_data, val_data, test_data = split_conversations(dataset, seed=args.seed)
    samples = {"train": train_data, "val": val_data, "test": test_data}[args.split][: args.n_conversations]
    selected_turn_ids, selection_metadata, turn_ids_digest = _load_turn_selection(
        args.turn_ids_file, args.split, args.seed, dataset_sha256, len(samples), args.max_turns
    )
    resume_signature = _resume_signature(args, dataset_sha256, turn_ids_digest)

    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST", file_location=".",
        filter_dict={"model": [args.llm_name]},
    )
    if not config_list:
        raise RuntimeError(
            f"No matching entry for model {args.llm_name!r} in OAI_CONFIG_LIST. "
            "Add your local model name and base_url to OAI_CONFIG_LIST."
        )

    model = SentenceTransformer(args.model_path)
    quadruple = build_quadruple_safe(
        model, train_data, args.cache_path, get_embeddings, model_name=args.model_path
    )

    confirmed_pairs = []
    if args.confusable_pairs_file:
        with open(args.confusable_pairs_file, "r", encoding="utf-8") as f:
            pair_data = json.load(f)
        if not isinstance(pair_data, dict) or "confirmed_confusable_pairs" not in pair_data:
            raise ValueError(
                f"{args.confusable_pairs_file} is not a valid run_confusion_analysis.py output "
                "(expected a dict with a 'confirmed_confusable_pairs' key)."
            )
        confirmed_pairs = [tuple(p) for p in pair_data["confirmed_confusable_pairs"] if len(p) == 2]

    deliberation = PatchedStrategyDeliberation(
        quadruple, model, arm=args.arm,
        lambda_diversity=args.lambda_diversity,
        confirmed_confusable_pairs=confirmed_pairs,
    )

    full_results = []
    flat_results = []
    completed_turn_ids = set()
    n_skipped_composite = 0
    n_skipped_opening = 0
    n_skipped_unselected = 0
    if not args.fresh and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as _ef:
            _existing = json.load(_ef)
        if _existing.get("resume_signature") != resume_signature:
            raise RuntimeError(
                f"Existing output {args.out} was created with different or unverified settings. "
                "Use --fresh to start a clean run, or choose a new --out path."
            )
        flat_results = list(_existing.get("flat_results", []))
        full_results = list(_existing.get("full_results", []))
        # The runner traverses the split from the beginning when resuming so it
        # can rebuild dialogue history. Recompute traversal counters rather than
        # adding them to values from the previous checkpoint.
        n_skipped_composite = 0
        n_skipped_opening = 0
        n_skipped_unselected = 0
        completed_list = [r.get("turn_id") for r in flat_results if r.get("turn_id")]
        if len(completed_list) != len(set(completed_list)):
            raise RuntimeError(f"Existing output {args.out} contains duplicate turn IDs.")
        completed_turn_ids = set(completed_list)
        if selected_turn_ids is not None and not completed_turn_ids.issubset(selected_turn_ids):
            raise RuntimeError("Existing output contains turn IDs outside the requested selection. Use --fresh.")
        if completed_turn_ids:
            print(f"Resuming {len(completed_turn_ids)} verified turns from {args.out}.")

    for conv_idx, sample in enumerate(samples):
        if args.max_turns is not None and len(flat_results) >= args.max_turns:
            break
        dialog = sample["dialog"]
        situation = sample.get("situation", "").strip()
        count = 0
        history = []

        while True:
            save = {}
            if count == len(dialog):
                break
            if args.max_turns is not None and len(flat_results) >= args.max_turns:
                break

            if count != 0 and dialog[count]["speaker"] == "supporter":
                # Check if consecutive supporter turns occur here
                if (count < len(dialog) - 1 and dialog[count + 1]["speaker"] != "supporter") or (count == len(dialog) - 1):
                    save["strategy"] = dialog[count]["annotation"]["strategy"]
                    save["reference"] = dialog[count]["content"].strip()
                    is_composite = False
                    context = json2natural(history)
                    save["context"] = context
                    seeker_utts = [h["content"] for h in history if h["role"] == "user" and h["content"].strip()]
                    post = seeker_utts[-1] if seeker_utts else history[-1]["content"]
                    history.append({
                        "content": dialog[count]["content"].strip(),
                        "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant",
                    })
                    count += 1
                else:
                    # Merge consecutive supporter utterances to keep dialogue history aligned
                    save["strategy"] = f"{dialog[count]['annotation']['strategy']} and {dialog[count + 1]['annotation']['strategy']}"
                    save["reference"] = dialog[count]["content"].strip() + " " + dialog[count + 1]["content"].strip()
                    is_composite = True
                    context = json2natural(history)
                    save["context"] = context
                    seeker_utts = [h["content"] for h in history if h["role"] == "user" and h["content"].strip()]
                    post = seeker_utts[-1] if seeker_utts else history[-1]["content"]
                    history.append({
                        "content": dialog[count]["content"].strip(),
                        "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant",
                    })
                    history.append({
                        "content": dialog[count + 1]["content"].strip(),
                        "role": "user" if dialog[count + 1]["speaker"] == "seeker" else "assistant",
                    })
                    count += 2

                if save["strategy"] in STRATEGIES:
                    is_composite = False

                # By default, skip merged multi-strategy turns and opening greetings
                # (where the seeker has not yet stated a substantive problem) so 100%
                # of evaluated turns are valid single-label Strategy Deliberation turns.
                total_seeker_words = sum(len(h["content"].split()) for h in history if h["role"] == "user")
                if not args.include_zero_shot_turns:
                    if is_composite:
                        n_skipped_composite += 1
                        continue
                    if count <= 3 or total_seeker_words < 5:
                        n_skipped_opening += 1
                        continue

                turn_id = f"conv{conv_idx:03d}_{situation[:20].replace(' ', '_')}_{count}"
                # Preserve dialogue/history traversal for every turn, but spend LLM calls
                # only on the exact, reproducible IDs in the supplied selection file.
                if selected_turn_ids is not None and turn_id not in selected_turn_ids:
                    n_skipped_unselected += 1
                    continue
                if turn_id in completed_turn_ids:
                    continue

                t0 = time.perf_counter()
                retrieved_strategies = []
                retrieved_examples = []
                contrastive_cues = []
                deliberation_candidates = []
                retrieval_latency_seconds = None

                if args.include_zero_shot_turns:
                    entered_deliberation = count > 5 and is_complex(
                        get_prompt("behavior_control").format(context=context),
                        config_list=config_list, cache_path_root=args.cache_path_root,
                    )
                else:
                    entered_deliberation = True

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

                    pred_strategy, pairs, ret_meta = deliberation.get_strategy(
                        emo_and_reason, cau_and_reason, int_and_reason, context, post,
                        select_strategy_by_group, config_list, n=10, return_metadata=True,
                    )
                    retrieval_latency_seconds = ret_meta["retrieval_latency_seconds"]
                    retrieved_strategies = ret_meta["retrieved_strategies"]
                    retrieved_examples = ret_meta["retrieved_examples"]
                    contrastive_cues = ret_meta["contrastive_cues"]

                    pred_strategy = clean_strategy(pred_strategy)
                    if not pred_strategy and retrieved_strategies:
                        pred_strategy = clean_strategy(retrieved_strategies[:1])
                    deliberation_candidates = list(pred_strategy)
                    save["emotion"], save["cause"], save["intention"] = emotion, cause, intention

                    if len(pred_strategy) == 0:
                        response = single_agent_response(
                            get_prompt("zero_shot").format(context=context),
                            config_list=config_list, cache_path_root=args.cache_path_root,
                        )
                        save["response"] = response
                        save["pred_strategy"] = "None"
                    else:
                        if len(pred_strategy) == 1:
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
                            candidate_strats = list(pred_strategy)
                            candidate_resps = []
                            responses = []
                            for strat in candidate_strats:
                                examples = "\n\n".join(
                                    f"{p[0]}\n{p[1]}" for p in pairs
                                    if p[1].split("]", 1)[0].strip("[").strip() == strat
                                ).strip()
                                resp = response_with_strategy(
                                    context, emo_and_reason, cau_and_reason, int_and_reason,
                                    strat, examples, config_list, args.cache_path_root)
                                candidate_resps.append(resp)
                                responses.append(f"[{strat}] {resp}")

                            int_with_cues = int_and_reason
                            if contrastive_cues:
                                int_with_cues = int_and_reason + "\n\n" + "\n".join(contrastive_cues)

                            debate_history = debate(context, emo_and_reason, cau_and_reason, int_with_cues, responses, config_list)
                            reflection_result = reflect(context, emo_and_reason, cau_and_reason, int_with_cues, debate_history, responses, config_list)
                            voted_strats, voted_resps = vote(reflection_result)

                            if len(voted_strats) == 1 and voted_strats[0] != "None":
                                pred_strategy, response = voted_strats[0].strip(), voted_resps[0].strip()
                            else:
                                judge_s = voted_strats if (voted_strats and voted_strats[0] != "None") else candidate_strats
                                judge_r = voted_resps if (voted_strats and voted_strats[0] != "None") else candidate_resps
                                pred_strategy, response = judge(context, judge_s, judge_r, config_list, args.cache_path_root)

                            cleaned_final = clean_strategy([pred_strategy])
                            if cleaned_final:
                                pred_strategy = cleaned_final[0]
                            else:
                                pred_strategy = candidate_strats[0]
                            if not response or response == "None":
                                response = candidate_resps[0]
                            save["ori_response"], save["pred_strategy"] = response, pred_strategy

                        _, response = self_reflection(context, pred_strategy, response, config_list, args.cache_path_root)
                        save["response"] = response

                latency = time.perf_counter() - t0

                save["turn_id"] = turn_id
                save["conversation_idx"] = conv_idx
                save["situation"] = situation
                save["post"] = post
                save["is_composite_gold"] = is_composite
                save["entered_deliberation"] = entered_deliberation
                save["deliberation_candidates"] = deliberation_candidates
                save["retrieved_strategies"] = retrieved_strategies
                save["retrieved_examples"] = retrieved_examples
                save["contrastive_cues"] = contrastive_cues
                save["retrieval_latency_seconds"] = retrieval_latency_seconds
                save["latency_seconds"] = latency
                full_results.append(save)

                flat_results.append({
                    "turn_id": turn_id,
                    "conversation_idx": conv_idx,
                    "situation": situation,
                    "context": context,
                    "post": post,
                    "gold_strategy": save["strategy"],
                    "predicted_strategy": save["pred_strategy"],
                    "deliberation_candidates": deliberation_candidates,
                    "is_composite_gold": is_composite,
                    "entered_deliberation": entered_deliberation,
                    "retrieved_strategies": retrieved_strategies,
                    "retrieved_examples": retrieved_examples,
                    "contrastive_cues": contrastive_cues,
                    "retrieval_latency_seconds": retrieval_latency_seconds,
                    "latency_seconds": latency,
                    "generated_response": save.get("response", ""),
                    "gold_reference": save.get("reference", ""),
                })
                n_done = len(flat_results)
                target_str = f"/{args.max_turns}" if args.max_turns else ""
                hit_marker = "MATCH" if save["strategy"] == save["pred_strategy"] else "miss"
                print(
                    f"[{args.arm} | Turn {n_done}{target_str} | Conv {conv_idx + 1}/{len(samples)}] "
                    f"gold={save['strategy']!r} -> pred={save['pred_strategy']!r} [{hit_marker}] ({latency:.1f}s)",
                    flush=True,
                )
                if n_done % 5 == 0:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    cfg_dict = dict(vars(args))
                    cfg_dict["deliberation_calibrated"] = True
                    with open(args.out, "w", encoding="utf-8") as _cf:
                        json.dump({
                            "config": cfg_dict,
                            "n_train": len(train_data), "n_val": len(val_data), "n_test": len(test_data),
                            "n_evaluated_turns": len(flat_results),
                            "n_skipped_composite_turns": n_skipped_composite,
                            "n_skipped_opening_turns": n_skipped_opening,
                            "n_skipped_unselected_eligible_turns": n_skipped_unselected,
                            "dataset_sha256": dataset_sha256,
                            "turn_selection": selection_metadata,
                            "resume_signature": resume_signature,
                            "flat_results": flat_results,
                            "full_results": full_results,
                        }, _cf, indent=2)

            else:
                history.append({
                    "content": dialog[count]["content"].strip(),
                    "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant",
                })
                count += 1

    observed_ids = [r.get("turn_id") for r in flat_results]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("Run produced duplicate turn IDs; refusing to write a potentially invalid result.")
    if selected_turn_ids is not None:
        observed_set = set(observed_ids)
        missing_ids = sorted(selected_turn_ids - observed_set)
        extra_ids = sorted(observed_set - selected_turn_ids)
        selection_validation = {
            "valid": not missing_ids and not extra_ids and len(observed_ids) == len(selected_turn_ids),
            "expected_turns": len(selected_turn_ids),
            "observed_turns": len(observed_ids),
            "missing_turn_ids": missing_ids,
            "unexpected_turn_ids": extra_ids,
        }
    else:
        selection_validation = None

    cfg_dict = dict(vars(args))
    cfg_dict["deliberation_calibrated"] = True
    out = {
        "config": cfg_dict,
        "n_train": len(train_data), "n_val": len(val_data), "n_test": len(test_data),
        "n_evaluated_turns": len(flat_results),
        "n_skipped_composite_turns": n_skipped_composite,
        "n_skipped_opening_turns": n_skipped_opening,
        "n_skipped_unselected_eligible_turns": n_skipped_unselected,
        "dataset_sha256": dataset_sha256,
        "turn_selection": selection_metadata,
        "selection_validation": selection_validation,
        "resume_signature": resume_signature,
        "flat_results": flat_results,
        "full_results": full_results,
    }

    if args.run_llm_judge:
        from response_metrics import evaluate_responses_with_llm_judge
        judge_cache = args.out + ".judge_cache.json"
        out["llm_judge"] = evaluate_responses_with_llm_judge(
            full_results, config_list=config_list, cache_path=judge_cache, deliberation_only=True
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(flat_results)} evaluated turns to {args.out}")
    if selection_validation is not None:
        print(
            f"Selection check: {selection_validation['observed_turns']}/"
            f"{selection_validation['expected_turns']} selected IDs; "
            f"valid={selection_validation['valid']}"
        )
        if not selection_validation["valid"]:
            preview = selection_validation["missing_turn_ids"][:10]
            raise RuntimeError(
                "Selected-turn coverage is incomplete or contains unexpected IDs. "
                f"First missing IDs: {preview}. Inspect the saved output before proceeding."
            )


if __name__ == "__main__":
    main()
