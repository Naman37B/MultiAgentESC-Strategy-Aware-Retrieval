import json
import heapq
import os
import re
import sys
import numpy as np
from sentence_transformers import util
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    import autogen
    from autogen import Cache
except ImportError:
    import integration_patch as autogen
    from integration_patch import Cache


CANONICAL_STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]

_CANONICAL_LOWER_MAP = {s.lower(): s for s in CANONICAL_STRATEGIES}
_SORTED_CANONICAL = sorted(CANONICAL_STRATEGIES, key=len, reverse=True)

CONCISE_COUNSELOR_SYSTEM_MSG = (
    "You are a psychological counseling expert. Follow the requested format strictly "
    "and keep your Reasoning to one concise sentence."
)


def normalize_strategy_name(raw_name):
    """
    Maps any case/punctuation/Markdown variation of an ESConv strategy name
    (e.g. '[Self-Disclosure]', '**Providing suggestions**', 'Reflection of Feelings')
    to its exact canonical string in CANONICAL_STRATEGIES, or None if no match.
    """
    if not raw_name or not isinstance(raw_name, str):
        return None
    cleaned = re.sub(r"[\[\]\*_`'\"]", " ", raw_name).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in _CANONICAL_LOWER_MAP:
        return _CANONICAL_LOWER_MAP[cleaned]
    for canon in _SORTED_CANONICAL:
        if re.search(rf"\b{re.escape(canon.lower())}\b", cleaned):
            return canon
    return None


def _parse_response_and_strategy(raw_text, fallback_strategy=None):
    """
    Robustly extracts (canonical_strategy, response_text) from an agent reply
    formatted like:
        Response: [strategy] [response]
        Reasoning: [reasoning]
    Handles Markdown bolding (**Response:**, **Reasoning:**), hyphenated
    strategy names ([Self-disclosure]), missing brackets, and case differences.
    """
    if not raw_text or not isinstance(raw_text, str):
        return (normalize_strategy_name(fallback_strategy) or "None"), "None"

    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    before_reasoning = re.split(r"(?im)^\s*[#\*\-\d\.]*\s*reasoning\s*\**\s*:", text)[0].strip()
    if not before_reasoning:
        before_reasoning = text

    m_resp_line = re.search(r"(?is)\bresponse\s*\**\s*:\s*(.*)", before_reasoning)
    resp_body = m_resp_line.group(1).strip() if m_resp_line else before_reasoning

    # 1. Try explicit bracketed [Strategy] at the start of resp_body
    m_bracket = re.match(r"^\**\s*\[([a-zA-Z \-]+)\]\**\s*(.*)", resp_body, flags=re.DOTALL)
    if m_bracket:
        strat = normalize_strategy_name(m_bracket.group(1))
        rest = m_bracket.group(2).strip()
        if rest.startswith("[") and rest.endswith("]"):
            rest = rest[1:-1].strip()
        if strat and rest:
            return strat, rest

    # 2. Try matching any canonical strategy name at the start of resp_body
    for canon in _SORTED_CANONICAL:
        pat = rf"^\**\s*\[?{re.escape(canon)}\]?\**\s*[:\-\]]?\s*(.*)"
        m_canon = re.match(pat, resp_body, flags=re.IGNORECASE | re.DOTALL)
        if m_canon:
            rest = m_canon.group(1).strip()
            if rest.startswith("[") and rest.endswith("]"):
                rest = rest[1:-1].strip()
            if rest:
                return canon, rest

    # 3. Search anywhere in before_reasoning for [Strategy]
    for m_any in re.finditer(r"\[([a-zA-Z \-]+)\]", before_reasoning):
        strat = normalize_strategy_name(m_any.group(1))
        if strat:
            after_bracket = before_reasoning[m_any.end():].strip()
            if after_bracket.startswith("[") and after_bracket.endswith("]"):
                after_bracket = after_bracket[1:-1].strip()
            if after_bracket:
                return strat, after_bracket

    # 4. Fallback to fallback_strategy if response text is non-empty
    canon_fallback = normalize_strategy_name(fallback_strategy)
    cleaned_resp = re.sub(r"^\**\s*\[?[a-zA-Z \-]+\]?\**\s*", "", resp_body).strip()
    if canon_fallback and (cleaned_resp or resp_body):
        return canon_fallback, (cleaned_resp or resp_body)

    return "None", "None"


def is_complex(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 60,
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    flag = True if "yes" in str(response).lower() else False
    return flag


def single_agent_response(prompt, config_list, cache_path_root):
    """When multi-agent is not needed, generate responses through zero shot."""
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 80,
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    text = str(raw_response).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    m = re.search(r"(?is)\bresponse\s*\**\s*:\s*(.*)", text)
    if m and m.group(1).strip():
        resp = m.group(1).strip()
        if resp.startswith("[") and resp.endswith("]"):
            resp = resp[1:-1].strip()
        return resp
    return text if text else "None"


def get_emotion(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Emotion Perception Agent',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 100,
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    try:
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        emotion = re.findall(r'Emotion\s*\**\s*:(.*)', response.split("\n")[0])[0].strip()
    except Exception:
        emotion = "Negative"

    return emotion, response


def get_cause(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Cause Perception Agent',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 100,
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )
    try:
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        cause = re.findall(r'Event\s*\**\s*:(.*)', response.split("\n")[0])[0].strip()
    except Exception:
        cause = "Not mention"

    return cause, response


def get_intention(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Intention Perception Agent',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 100,
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )
    try:
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        intention = re.findall(r'Intention\s*\**\s*:(.*)', response.split("\n")[0])[0].strip()
    except Exception:
        intention = "Not mention"

    return intention, response


def select_strategy_by_group(emo_and_reason, cau_and_reason, int_and_reason, context, examples, config_list, agent_num=3):
    # Extract the candidate strategies actually present in the retrieved examples
    ex_strats_in_order = [
        normalize_strategy_name(m)
        for m in re.findall(r'\[([a-zA-Z \-]+)\]', examples)
    ]
    ex_strats_in_order = [s for s in ex_strats_in_order if s]
    ex_counts = Counter(ex_strats_in_order)
    available_strategies = list(dict.fromkeys(ex_strats_in_order)) or list(CANONICAL_STRATEGIES)
    available_str = ", ".join(f"[{s}]" for s in available_strategies)

    prompt = f"""### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information, dialogue context, and retrieved examples below, select the single most appropriate strategy for the 'Assistant' to respond to the User's last utterance, and briefly explain why in one sentence.
The following are retrieved examples of strategies, presented in the format of <post\n[strategy] response>, followed by any contrastive strategy notes:

### Examples
{examples}

### Candidate Strategies Present in Examples
{available_str}

Your answer must include the following elements:
Strategy: The single best strategy selected from {available_str}. Pay close attention to whether the user is asking for information/suggestions, sharing feelings that need reflection/reassurance, or needing a clarifying question.
Reasoning: One concise sentence explaining why this strategy fits best.

Your answer must follow this format: 
Strategy: [strategy]
Reasoning: [reasoning]
"""

    admin = autogen.ConversableAgent(
        name="Admin",
        system_message="Initialize the group discussion about strategy selection.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 120,
        },
        human_input_mode='NEVER',
    )

    agents = []
    for i in range(agent_num):
        agent = autogen.ConversableAgent(
            name=f'agent_{i}',
            system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
            llm_config={
                "config_list": config_list,
                "cache_seed": 2024,
                "temperature": 0.0,
                "max_tokens": 120,
            },
            human_input_mode='NEVER',
            description=f"agent_{i} participate in a group discussion about strategy selection."
        )
        agents.append(agent)

    groupchat = autogen.GroupChat(
        agents=[admin] + agents,
        messages=[],
        max_round=agent_num + 1,
        speaker_selection_method="round_robin"
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 120,
        }
    )

    discussion_results = admin.initiate_chat(
        manager,
        message=prompt,
    )

    chat_history = [history["content"] for history in discussion_results.chat_history[1:]]

    raw_votes = []
    for msg in chat_history:
        strat_lines = re.findall(r"(?im)^\s*[#\*\-\d\.]*\s*strategy\s*\**\s*:\s*(.+)$", msg)
        chosen = None
        for sline in strat_lines:
            canon = normalize_strategy_name(sline)
            if canon:
                chosen = canon
                break
        if not chosen:
            chosen = normalize_strategy_name(msg)
        if chosen:
            raw_votes.append(chosen)

    # Enforce "The strategy must appear in the examples" when examples are present
    available_set = set(available_strategies)
    valid_votes = [v for v in raw_votes if v in available_set]
    if not valid_votes:
        valid_votes = raw_votes if raw_votes else [s for s, _ in ex_counts.most_common(2)]

    vote_counter = Counter(valid_votes)
    if not vote_counter:
        return available_strategies[:1]

    # If >=2 agents agree on a strategy (majority consensus in Stage 2) or only 1 strategy was voted,
    # return that single consensus winner so Stage 3 directly generates and refines the response
    # without overwriting Stage 2's decision.
    top_strat, top_count = vote_counter.most_common(1)[0]
    if top_count >= 2 or len(vote_counter) == 1:
        return [top_strat]

    # Otherwise (1-1-1 split), return the top 2 strategies ranked by retrieval support in examples
    ranked_split = sorted(
        vote_counter.keys(),
        key=lambda s: (vote_counter[s], ex_counts.get(s, 0), -available_strategies.index(s) if s in available_strategies else -99),
        reverse=True,
    )
    return ranked_split[:2]


def get_strategy(emo_and_reason, cau_and_reason, int_and_reason, context, post, quadruple, model, config_list, n=10):
    post_embedding = model.encode(post)
    can_embeddings = [np.array(json.loads(q.split('__SEP__')[3]), dtype=np.float32) for q in quadruple]
    similarities = util.pytorch_cos_sim(post_embedding, can_embeddings)[0].tolist()
    top_indices = heapq.nlargest(n, range(len(similarities)), key=lambda i: similarities[i])
    pairs = []
    for i in top_indices:
        post_i, response, strategy, _ = quadruple[i].split('__SEP__')
        pairs.append((post_i, f"[{strategy}] {response}"))

    examples = "\n\n".join([f"{pair[0]}\n{pair[1]}" for pair in pairs])
    strategy = select_strategy_by_group(emo_and_reason, cau_and_reason, int_and_reason, context, examples, config_list)
    return strategy, pairs


def response_with_strategy(context, emo_and_reason, cau_and_reason, int_and_reason, strategy, examples, config_list, cache_path_root):
    """Generate response conditioned on a single selected strategy."""
    prompt = f"""You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}


Please generate a response from the Assistant's perspective using the {strategy} strategy.
The following are examples of this strategy, all presented in the format of <post\n[strategy] response>.

### Examples
{examples}

Your answer must be fewer than 30 words and must follow this format:
Response: [{strategy}] [response]
"""

    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 80,
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    _, parsed_resp = _parse_response_and_strategy(raw_response, fallback_strategy=strategy)
    return parsed_resp


def debate(context, emo_and_reason, cau_and_reason, int_and_reason, responses, config_list):
    responses_template = "\n\n".join(responses)

    prompt = f"""### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information and dialogue context, please select the most appropriate response from the following options and explain why in one sentence.

### Response
{responses_template}

Your answer must follow this format: 
Response: [strategy] [response]
Reasoning: [one-sentence reasoning]"""

    admin = autogen.ConversableAgent(
        name="Admin",
        system_message="Initialize the group discussion about which response is the most appropriate.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 120,
        },
        human_input_mode='NEVER',
    )

    agents = []
    for i in range(len(responses)):
        agent = autogen.ConversableAgent(
            name=f'agent_{i}',
            system_message=f"""You are a psychologist participating in a group discussion about which response is most appropriate. You initially support "{responses[i]}", but should choose whichever option best fits the user's actual need. Keep Reasoning to one sentence.""",
            llm_config={
                "config_list": config_list,
                "cache_seed": 2024,
                "temperature": 0.0,
                "max_tokens": 120,
            },
            human_input_mode='NEVER',
            description=f"agent_{i} participate in a group discussion about which response is the most appropriate."
        )
        agents.append(agent)

    groupchat = autogen.GroupChat(
        agents=[admin] + agents,
        messages=[],
        max_round=len(responses) + 1,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 120,
        }
    )

    discussion_results = admin.initiate_chat(
        manager,
        message=prompt,
    )

    return [history["content"] for history in discussion_results.chat_history[1:]]


def reflect(context, emo_and_reason, cau_and_reason, int_and_reason, debate_history, responses, config_list):
    discussion_content = "\n\n".join(debate_history)

    prompt = f"""### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

### Discussion content
{discussion_content}

Analyze the viewpoints above and select the single most appropriate response and strategy.

Your answer must follow this format: 
Response: [strategy] [response]
Reasoning: [one-sentence reasoning]"""

    admin = autogen.ConversableAgent(
        name="Admin",
        system_message="Initialize the group discussion about which response is the most appropriate.",
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 120,
        },
        human_input_mode='NEVER',
    )

    agents = []
    for i in range(len(responses)):
        agent = autogen.ConversableAgent(
            name=f'agent_{i}',
            system_message=f"""You are a psychologist reflecting on the group discussion to choose the most appropriate response. Keep Reasoning to one sentence.""",
            llm_config={
                "config_list": config_list,
                "cache_seed": 2024,
                "temperature": 0.0,
                "max_tokens": 120,
            },
            human_input_mode='NEVER',
            description=f"agent_{i} participate in a group discussion about which response is the most appropriate."
        )
        agents.append(agent)

    groupchat = autogen.GroupChat(
        agents=[admin] + agents,
        messages=[],
        max_round=len(responses) + 1,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 120,
        }
    )

    discussion_results = admin.initiate_chat(
        manager,
        message=prompt,
    )

    return [history["content"] for history in discussion_results.chat_history[1:]]


def vote(results):
    count = {}
    strat2response = {}
    for result in results:
        strategy, resp = _parse_response_and_strategy(result)
        if strategy != "None" and resp != "None":
            count[strategy] = count.get(strategy, 0) + 1
            if strategy not in strat2response:
                strat2response[strategy] = resp
    if len(count) == 0:
        return ["None"], ["None"]
    max_count = max(count.values())
    max_strat = [key for key, value in count.items() if value == max_count]
    responses = [strat2response[strat] for strat in max_strat]
    return max_strat, responses


def judge(context, strategies, responses, config_list, cache_path_root):
    valid_pairs = [
        (normalize_strategy_name(s) or s, r)
        for s, r in zip(strategies, responses)
        if s and s != "None" and r and r != "None"
    ]
    template = [f"[{s}] {r}" for s, r in valid_pairs] if valid_pairs else [
        f"[{s}] {r}" for s, r in zip(strategies, responses)
    ]
    template_responses = "\n\n".join(template)

    prompt = f"""You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

The following are responses generated by the therapist using different strategies, all presented in the format of <[strategy] response>. Please select the most appropriate response and explain why in one sentence.

### Examples
{template_responses}

Your answer must follow this format: 
Response: [strategy] [response]
Reasoning: [one-sentence reasoning]"""

    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 120,
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_reply = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    fallback_s = valid_pairs[0][0] if valid_pairs else None
    strategy, response = _parse_response_and_strategy(raw_reply, fallback_strategy=fallback_s)
    if (strategy == "None" or response == "None") and valid_pairs:
        return valid_pairs[0][0], valid_pairs[0][1]
    return strategy, response


def self_reflection(context, pred_strategy, response, config_list, cache_path_root):
    prompt = f"""You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

The following is a response generated by the therapist using {pred_strategy} strategy, presented in the format of <[strategy] response>. If the response is consistent with the conversation and aligns with the {pred_strategy} strategy, return it as is; otherwise provide a refined version under 30 words.

### Response
[{pred_strategy}] {response}

Your answer must follow this format: 
Response: [{pred_strategy}] [original or refined response]
Reasoning: [one-sentence reasoning]"""

    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message=CONCISE_COUNSELOR_SYSTEM_MSG,
        llm_config={
            "config_list": config_list,
            "cache_seed": 2024,
            "temperature": 0.0,
            "max_tokens": 100,
        },
        human_input_mode='NEVER'
    )

    orig_response = response
    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_reply = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    strategy, refined = _parse_response_and_strategy(raw_reply, fallback_strategy=pred_strategy)
    if not refined or refined == "None":
        refined = orig_response
    canon_strat = normalize_strategy_name(pred_strategy) or normalize_strategy_name(strategy) or str(pred_strategy)
    return canon_strat.strip(), refined.strip()
