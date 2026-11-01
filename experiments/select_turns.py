"""Create a reproducible, conversation-stratified set of eligible ESConv turns.

This script mirrors the turn eligibility and ID construction used by the
project's experiments/run_baseline.py. It does not call an LLM and does not
modify experiment result files.

Example:
  python experiments/select_turns.py --split test --n-turns 300 \
      --out results/test_turn_ids.json
  python experiments/select_turns.py --split val --n-turns 300 \
      --out results/val_turn_ids.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CANONICAL_STRATEGIES = {
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
}


def split_conversations(conversations: list[dict[str, Any]], seed: int = 42):
    """Match src/strategy_aware_retrieval.py::split_conversations exactly."""
    indices = list(range(len(conversations)))
    random.Random(seed).shuffle(indices)
    n_val = int(len(indices) * 0.15)
    n_test = int(len(indices) * 0.15)
    val_idx = set(indices[:n_val])
    test_idx = set(indices[n_val:n_val + n_test])
    train_idx = set(indices[n_val + n_test:])
    return (
        [conversations[i] for i in sorted(train_idx)],
        [conversations[i] for i in sorted(val_idx)],
        [conversations[i] for i in sorted(test_idx)],
    )


def find_eligible_turns(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror the current runner's single-label/opening-turn filters."""
    eligible: list[dict[str, Any]] = []

    for conv_idx, sample in enumerate(samples):
        dialog = sample.get("dialog", [])
        situation = sample.get("situation", "").strip()
        count = 0
        history: list[dict[str, str]] = []

        while count < len(dialog):
            if count != 0 and dialog[count].get("speaker") == "supporter":
                is_last_single = (
                    (count < len(dialog) - 1 and dialog[count + 1].get("speaker") != "supporter")
                    or count == len(dialog) - 1
                )

                if is_last_single:
                    gold_strategy = dialog[count]["annotation"]["strategy"]
                    is_composite = False
                    count += 1
                else:
                    gold_strategy = (
                        f"{dialog[count]['annotation']['strategy']} and "
                        f"{dialog[count + 1]['annotation']['strategy']}"
                    )
                    is_composite = True
                    count += 2

                # Match the runner: a concatenated label that happens to be
                # canonical is treated as a single-label record.
                if gold_strategy in CANONICAL_STRATEGIES:
                    is_composite = False

                if is_composite:
                    continue

                total_seeker_words = sum(
                    len(item["content"].split())
                    for item in history
                    if item["role"] == "user"
                )
                if count <= 3 or total_seeker_words < 5:
                    continue

                seeker_utterances = [
                    item["content"] for item in history
                    if item["role"] == "user" and item["content"].strip()
                ]
                post = seeker_utterances[-1] if seeker_utterances else (
                    history[-1]["content"] if history else ""
                )
                turn_id = f"conv{conv_idx:03d}_{situation[:20].replace(' ', '_')}_{count}"
                eligible.append({
                    "turn_id": turn_id,
                    "conversation_idx": conv_idx,
                    "dialog_count_after_unit": count,
                    "gold_strategy": gold_strategy,
                    "post": post,
                })
            else:
                history.append({
                    "content": dialog[count].get("content", "").strip(),
                    "role": "user" if dialog[count].get("speaker") == "seeker" else "assistant",
                })
                count += 1

    return eligible


def select_stratified(
    eligible: list[dict[str, Any]], n_turns: int, sample_seed: int
) -> list[dict[str, Any]]:
    """Take one unit from every available conversation, then round-robin extras.

    For the current ESConv split, all 195 test conversations have at least two
    eligible units. At n_turns=300 this yields 2 units from 105 conversations
    and 1 from the other 90, covering all 195 conversations.
    """
    by_conv: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_conv[row["conversation_idx"]].append(row)

    conv_ids = sorted(by_conv)
    if n_turns < len(conv_ids):
        raise ValueError(
            f"Requested {n_turns} turns, but there are {len(conv_ids)} conversations with "
            "eligible turns. To cover all such conversations, n-turns must be at least "
            f"{len(conv_ids)}."
        )
    if n_turns > len(eligible):
        raise ValueError(f"Requested {n_turns} turns, but only {len(eligible)} are eligible.")

    rng = random.Random(sample_seed)
    shuffled_by_conv: dict[int, list[dict[str, Any]]] = {}
    for conv_id in conv_ids:
        items = list(by_conv[conv_id])
        rng.shuffle(items)
        shuffled_by_conv[conv_id] = items

    selected: list[dict[str, Any]] = [shuffled_by_conv[c][0] for c in conv_ids]
    remaining = n_turns - len(selected)

    # Add second, third, ... units in seeded conversation order, rotating
    # rounds so no early conversation gets preferentially overrepresented.
    round_number = 1
    while remaining > 0:
        candidates = [c for c in conv_ids if len(shuffled_by_conv[c]) > round_number]
        if not candidates:
            raise ValueError("Not enough eligible turns to satisfy the requested sample size.")
        rng.shuffle(candidates)
        for conv_id in candidates:
            if remaining == 0:
                break
            selected.append(shuffled_by_conv[conv_id][round_number])
            remaining -= 1
        round_number += 1

    # Restore the order in which the runner traverses conversations/turns.
    order_lookup = {row["turn_id"]: i for i, row in enumerate(eligible)}
    selected.sort(key=lambda row: order_lookup[row["turn_id"]])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/ESConv.json")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--n-turns", type=int, default=300)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path.resolve()}")

    raw_bytes = dataset_path.read_bytes()
    dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    conversations = json.loads(raw_bytes.decode("utf-8"))
    train_data, val_data, test_data = split_conversations(conversations, seed=args.split_seed)
    split_data = {"val": val_data, "test": test_data}[args.split]

    eligible = find_eligible_turns(split_data)
    selected = select_stratified(eligible, args.n_turns, args.sample_seed)
    per_conversation = Counter(row["conversation_idx"] for row in selected)

    output = {
        "schema_version": 1,
        "split": args.split,
        "split_seed": args.split_seed,
        "sample_seed": args.sample_seed,
        "dataset_path_at_generation": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "n_train_conversations": len(train_data),
        "n_validation_conversations": len(val_data),
        "n_test_conversations": len(test_data),
        "n_split_conversations": len(split_data),
        "n_eligible_turns_in_split": len(eligible),
        "n_selected_turns": len(selected),
        "n_conversations_covered": len(per_conversation),
        "min_selected_per_covered_conversation": min(per_conversation.values()),
        "max_selected_per_covered_conversation": max(per_conversation.values()),
        "selected_per_conversation": {str(k): per_conversation[k] for k in sorted(per_conversation)},
        "turn_ids": [row["turn_id"] for row in selected],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Split: {args.split}")
    print(f"Split conversations: {len(split_data)}")
    print(f"Eligible single-label turns: {len(eligible)}")
    print(f"Selected turns: {len(selected)}")
    print(f"Conversations covered: {len(per_conversation)}")
    print(
        "Selected per covered conversation: "
        f"min={min(per_conversation.values())}, max={max(per_conversation.values())}"
    )
    print(f"Saved selection: {out_path.resolve()}")
    print(f"Dataset SHA-256: {dataset_sha256}")


if __name__ == "__main__":
    main()
