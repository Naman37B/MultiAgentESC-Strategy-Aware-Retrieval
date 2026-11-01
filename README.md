# MultiAgentESC Strategy-Aware Hybrid Retrieval — curated project source

This archive contains the files needed to inspect the implementation, reproduce the four final experiment arms (subject to local model availability), rerun analysis, and run the unit tests. It intentionally excludes caches, duplicate embedding files, private configuration, stale 150-turn runs, diagnostic outputs and unused scripts.

## Environment

- Python 3.10 or 3.11 is recommended for the pinned AutoGen 0.2.x stack.
- Install dependencies: `pip install -r requirements.txt`.
- Install Ollama and pull `qwen2.5:7b-instruct`.
- Copy `OAI_CONFIG_LIST.example` to `OAI_CONFIG_LIST` and adjust the local endpoint/model only if your Ollama setup differs. The private/local config is not included.
- The Sentence-Transformers model `all-roberta-large-v1` is downloaded on first use. Embedding caches are intentionally not bundled; they are regenerable.

## Reproduce the recorded final experiment

The final saved run used seed 42, 300 selected test turns covering all 195 test conversations, and lambda 0.3 selected on validation. The frozen validation pairs and turn-ID files are included. To rerun the pipeline locally:

```bash
python experiments/run_final_experiments.py --skip-judge
```

This performs the validation baseline/pair discovery and four test arms, then exports case studies and quantitative reports. The optional LLM judge was not run for the reported final results. Running without `--skip-judge` invokes an optional judge stage and creates new scores; do not mix those with the archived final metrics without documenting the protocol.

## Tests

```bash
python -m pytest tests -q
python -m compileall -q main.py multiagent.py prompt.py strategy.py src experiments tests
```

## Results included

- Raw validation baseline and four paired test-arm outputs.
- Identical ordered turn-ID selections for all test arms.
- Validation lambda sweep and frozen confusion pairs.
- Comparison table, Markdown evaluation report, seven final plots, balanced case studies, and paired statistical analysis.
- Original project architecture images, copied unchanged.

## Important implementation note

In B, C and D, the original LLM strategy-selection function is called, but an additional rule-based selector replaces its proposed label. This selector combines lexical strategy support, hand-written context cues, training-set priors and cumulative within-run utilization counts. Its weights differ by arm. Therefore, the B/C/D results describe complete system variants and do not isolate RRF, reranking or contrastive evidence independently. The selector is sequential and can be order-dependent.

## Dataset attribution and licence

ESConv is by Liu et al. (ACL 2021). The Hugging Face dataset card identifies the dataset as CC BY-NC 4.0. Retain attribution and follow the non-commercial licence terms. See `dataset/README.md`.

## Files intentionally excluded

The source archive excludes `.cache/`, `42/cache.db`, `__pycache__/`, `.pytest_cache/`, virtual environments, both duplicate `embeddings_train.txt` caches, local `OAI_CONFIG_LIST`, old 150-turn experiments, old diagnostics, stale LLM-judge caches, the older `run_ablation.py` / `.sh` launchers, and the unused `src/autogen_compat.py`.
