"""
integration_patch.py

Integrates Strategy-Aware Hybrid Retrieval into MultiAgentESC's Strategy
Deliberation stage while keeping `select_strategy_by_group` and the downstream
Response Generation stage untouched across all four ablation arms:
  - arm="baseline" : semantic-only top-n retrieval (original behavior)
  - arm="rrf"      : semantic + BM25 combined via Reciprocal Rank Fusion (RRF)
  - arm="rerank"   : RRF candidate pool + soft strategy-aware MMR reranking
  - arm="evidence" : RRF + reranking + contrastive strategy evidence cues

Also instruments retrieval-only execution latency (`retrieval_latency_seconds`),
records detailed retrieved cases and contrastive cues per turn, and provides
a built-in AutoGen v0.2 compatibility layer for Python 3.13 / Windows.
"""

import atexit
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from strategy_aware_retrieval import StrategyAwareRetriever


def _parse_pair_tuple(pair):
    """Splits a (post, '[Strategy] response') pair into structured fields."""
    post_text, bracketed_resp = pair
    if bracketed_resp.startswith("[") and "]" in bracketed_resp:
        strat, resp = bracketed_resp.split("]", 1)
        strat = strat.lstrip("[").strip()
        resp = resp.strip()
    else:
        strat, resp = "Unknown", bracketed_resp.strip()
    return {
        "post": post_text,
        "strategy": strat,
        "response": resp,
        "formatted": f"{post_text}\n{bracketed_resp}",
    }


class PatchedStrategyDeliberation:
    """
    Holds the retriever and configuration for a single ablation arm and exposes
    .get_strategy(...) with full retrieval latency and evidence instrumentation.
    """

    def __init__(self, quadruple_lines, sbert_model, arm,
                 lambda_diversity=0.3, confirmed_confusable_pairs=None):
        assert arm in ("baseline", "rrf", "rerank", "evidence"), arm
        self.retriever = StrategyAwareRetriever(quadruple_lines, sbert_model)
        self.arm = arm
        self.lambda_diversity = lambda_diversity
        self.confirmed_confusable_pairs = confirmed_confusable_pairs or []
        self.last_retrieval_metadata = None
        if arm == "evidence" and not self.confirmed_confusable_pairs:
            raise ValueError(
                "arm='evidence' requires confirmed_confusable_pairs discovered "
                "by confusion_matrix_confusable_pairs() on the validation split."
            )

    def _build_examples(self, post, n):
        self.retriever.last_cues = []
        if self.arm == "baseline":
            idx, _ = self.retriever._semantic_topk(post, n)
            pairs = [
                (self.retriever.posts[i], f"[{self.retriever.strategies[i]}] {self.retriever.responses[i]}")
                for i in idx
            ]
            examples = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])
            return examples, pairs

        if self.arm == "rrf":
            scores = self.retriever.candidate_pool(post)
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
            pairs = [
                (self.retriever.posts[i], f"[{self.retriever.strategies[i]}] {self.retriever.responses[i]}")
                for i, _ in ranked
            ]
            examples = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])
            return examples, pairs

        if self.arm == "rerank":
            pairs, _ = self.retriever.rerank(
                post, n=n, lambda_diversity=self.lambda_diversity,
                confusable_pairs=self.confirmed_confusable_pairs or None,
            )
            examples = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])
            return examples, pairs

        # arm == "evidence"
        return self.retriever.build_examples_with_evidence(
            post, self.confirmed_confusable_pairs, n=n, lambda_diversity=self.lambda_diversity
        )

    def retrieve_with_metadata(self, post, n=10):
        """
        Executes the retrieval/reranking/evidence step, measures retrieval-only
        wall-clock latency via time.perf_counter(), and returns structured
        metadata for logging and case-study generation.
        """
        t0 = time.perf_counter()
        examples, pairs = self._build_examples(post, n)
        retrieval_latency_seconds = time.perf_counter() - t0

        structured_examples = []
        retrieved_strategies = []
        for rank_idx, pair in enumerate(pairs):
            parsed = _parse_pair_tuple(pair)
            parsed["rank"] = rank_idx + 1
            structured_examples.append(parsed)
            retrieved_strategies.append(parsed["strategy"])

        cues = list(getattr(self.retriever, "last_cues", []))
        metadata = {
            "arm": self.arm,
            "retrieval_latency_seconds": retrieval_latency_seconds,
            "retrieved_strategies": retrieved_strategies,
            "retrieved_examples": structured_examples,
            "contrastive_cues": cues,
            "examples_prompt_block": examples,
        }
        self.last_retrieval_metadata = metadata
        return examples, pairs, metadata

    def get_strategy(self, emo_and_reason, cau_and_reason, int_and_reason, context,
                     post, select_strategy_by_group_fn, config_list, n=10,
                     return_metadata=False):
        examples, pairs, metadata = self.retrieve_with_metadata(post, n=n)
        strategy = select_strategy_by_group_fn(
            emo_and_reason, cau_and_reason, int_and_reason, context, examples, config_list
        )
        if self.arm in ("rrf", "rerank", "evidence"):
            strategy = self.retriever.deliberate_strategy(
                post=post,
                context=context,
                llm_strategy=strategy,
                arm=self.arm,
                confirmed_confusable_pairs=self.confirmed_confusable_pairs,
            )
        if return_metadata:
            return strategy, pairs, metadata
        return strategy, pairs


# ---------------------------------------------------------------------------
# Built-in AutoGen v0.2 Compatibility Layer (for Python 3.13 / Windows)
# ---------------------------------------------------------------------------

_GLOBAL_CACHE_STORES = {}
_GLOBAL_DIRTY_COUNTS = {}


def _flush_all_caches():
    for cache_file, store in list(_GLOBAL_CACHE_STORES.items()):
        if _GLOBAL_DIRTY_COUNTS.get(cache_file, 0) > 0:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
                tmp_f = cache_file + ".tmp"
                with open(tmp_f, "w", encoding="utf-8") as f:
                    json.dump(store, f)
                os.replace(tmp_f, cache_file)
                _GLOBAL_DIRTY_COUNTS[cache_file] = 0
            except Exception:
                pass


atexit.register(_flush_all_caches)


def config_list_from_json(env_or_file, file_location=".", filter_dict=None):
    raw = os.environ.get(env_or_file)
    if raw:
        try:
            configs = json.loads(raw)
        except json.JSONDecodeError:
            with open(raw, "r", encoding="utf-8") as f:
                configs = json.load(f)
    else:
        candidates = [
            os.path.join(file_location, env_or_file),
            env_or_file,
        ]
        config_path = next((p for p in candidates if os.path.exists(p)), None)
        if not config_path:
            raise FileNotFoundError(
                f"Could not find {env_or_file!r} in {file_location!r}. "
                "Create OAI_CONFIG_LIST from OAI_CONFIG_LIST.example."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)

    if filter_dict:
        allowed_models = set(filter_dict.get("model", []))
        if allowed_models:
            filtered = [c for c in configs if c.get("model") in allowed_models]
            if filtered:
                return filtered
            if configs:
                requested_model = list(allowed_models)[0]
                cloned = dict(configs[0])
                cloned["model"] = requested_model
                if cloned.get("base_url") == "http://0.0.0.0:4000":
                    cloned["base_url"] = "http://127.0.0.1:11434/v1"
                return [cloned]
    return configs


class _DiskCacheContext:
    def __init__(self, cache_path_root=""):
        self.root = cache_path_root or ".cache"
        self.cache_file = os.path.abspath(os.path.join(self.root, "autogen_compat_cache.json"))

    def __enter__(self):
        if self.cache_file not in _GLOBAL_CACHE_STORES:
            os.makedirs(self.root, exist_ok=True)
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        _GLOBAL_CACHE_STORES[self.cache_file] = json.load(f)
                except Exception:
                    _GLOBAL_CACHE_STORES[self.cache_file] = {}
            else:
                _GLOBAL_CACHE_STORES[self.cache_file] = {}
            _GLOBAL_DIRTY_COUNTS[self.cache_file] = 0
        return self

    def get(self, key):
        return _GLOBAL_CACHE_STORES.get(self.cache_file, {}).get(key)

    def set(self, key, value):
        store = _GLOBAL_CACHE_STORES.setdefault(self.cache_file, {})
        store[key] = value
        _GLOBAL_DIRTY_COUNTS[self.cache_file] = _GLOBAL_DIRTY_COUNTS.get(self.cache_file, 0) + 1
        if _GLOBAL_DIRTY_COUNTS[self.cache_file] >= 10:
            _flush_all_caches()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class Cache:
    @staticmethod
    def disk(cache_path_root=""):
        return _DiskCacheContext(cache_path_root=cache_path_root)


def _call_openai_compatible(config, messages, temperature=0.0, max_tokens=400):
    model = config.get("model", "qwen2.5:7b-instruct")
    base_url = (config.get("base_url") or "http://127.0.0.1:11434/v1").rstrip("/")
    if base_url == "http://0.0.0.0:4000":
        base_url = "http://127.0.0.1:11434/v1"
    api_key = config.get("api_key") or "ollama"

    # Try candidate URLs (if localhost fails on Windows IPv6 ::1, automatically retry 127.0.0.1)
    candidate_urls = [base_url]
    if "localhost" in base_url:
        candidate_urls.append(base_url.replace("localhost", "127.0.0.1"))

    last_err = None
    for b_url in candidate_urls:
        try:
            import openai
            client = openai.OpenAI(base_url=b_url, api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            pass
        except Exception as e:
            last_err = e
            continue

        # Standard-library HTTP fallback if openai package is not installed
        url = f"{b_url}/chat/completions" if b_url.endswith("/v1") or "/v1/" in b_url else f"{b_url}/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"] or ""
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Failed to connect to local LLM server at {base_url} for model {model!r}. "
        f"Ensure Ollama is running (`ollama serve` and `ollama pull {model}`). "
        f"Underlying error: {last_err}"
    ) from last_err


class ConversableAgent:
    def __init__(self, name, system_message="", llm_config=None, human_input_mode="NEVER", description=""):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config or {}
        self.human_input_mode = human_input_mode
        self.description = description

    def generate_reply(self, messages=None, cache=None):
        messages = messages or []
        config_list = self.llm_config.get("config_list") or [{}]
        config = config_list[0]
        temperature = self.llm_config.get("temperature", 0.0)
        max_tokens = self.llm_config.get("max_tokens", 400)
        cache_seed = self.llm_config.get("cache_seed", 2024)

        full_messages = []
        if self.system_message:
            full_messages.append({"role": "system", "content": self.system_message})
        full_messages.extend(messages)

        # Automatically use default disk cache when cache_seed is set
        active_cache = cache
        if active_cache is None and cache_seed is not None:
            active_cache = Cache.disk("")
            active_cache.__enter__()

        cache_key = None
        if active_cache is not None and hasattr(active_cache, "get"):
            raw_key = json.dumps({
                "model": config.get("model"),
                "seed": cache_seed,
                "temp": temperature,
                "max_tokens": max_tokens,
                "messages": full_messages,
            }, sort_keys=True)
            cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            cached_val = active_cache.get(cache_key)
            if cached_val is not None:
                return cached_val

        reply = _call_openai_compatible(
            config, full_messages, temperature=temperature, max_tokens=max_tokens
        )
        if active_cache is not None and cache_key is not None and hasattr(active_cache, "set"):
            active_cache.set(cache_key, reply)
        return reply

    def initiate_chat(self, manager, message):
        """
        Runs the round-robin group discussion initialized by `self` (Admin)
        with `message` and returns an object with `.chat_history` matching
        AutoGen's `ChatResult.chat_history`.
        """
        groupchat = manager.groupchat
        chat_history = [{"role": "user", "name": self.name, "content": message}]
        speaking_agents = [a for a in groupchat.agents if a is not self]
        max_speaker_turns = max(0, groupchat.max_round - 1)

        for turn_idx in range(max_speaker_turns):
            if not speaking_agents:
                break
            speaker = speaking_agents[turn_idx % len(speaking_agents)]
            if len(chat_history) == 1:
                combined_prompt = message
            else:
                prior_turns = "\n\n".join(
                    f"[{entry['name']}]:\n{entry['content']}"
                    for entry in chat_history[1:]
                )
                combined_prompt = (
                    f"{message}\n\n"
                    f"### Other Agents' Responses So Far\n{prior_turns}\n\n"
                    f"Now provide your own response as {speaker.name}. "
                    f"Follow the exact output format specified in the instruction above."
                )
            reply = speaker.generate_reply(messages=[{"role": "user", "content": combined_prompt}])
            chat_history.append({
                "role": "assistant",
                "name": speaker.name,
                "content": reply,
            })

        return SimpleNamespace(chat_history=chat_history)


class GroupChat:
    def __init__(self, agents, messages=None, max_round=4, speaker_selection_method="round_robin"):
        self.agents = agents
        self.messages = messages if messages is not None else []
        self.max_round = max_round
        self.speaker_selection_method = speaker_selection_method


class GroupChatManager:
    def __init__(self, groupchat, llm_config=None):
        self.groupchat = groupchat
        self.llm_config = llm_config or {}
