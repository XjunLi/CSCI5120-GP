# Self-evolving NAS agents

Three controllers adapt different parts of the search system. Each has a frozen counterpart under the same evaluator, shared warm start and final-candidate budget.

| Layer | Method / ablation | What changes during a run |
| --- | --- | --- |
| Harness | `harness` / `harness-frozen` | A versioned search program: workflow, memory selection, parent selection, exploration, mutation and prompt instruction |
| Multi-agent | `multiagent` / `multiagent-frozen` | Agent genomes and prompts, population membership, inherited memories and communication shortcuts |
| Policy weights | `weight-ttt` / `weight-frozen` | Parameters of an autoregressive NAS policy or LoRA adapters inside a causal language model |

These are separate controllers, so each pair measures adaptation at one layer. Evolving a candidate CNN architecture alone does not constitute any of these three mechanisms. The benchmark supplies previously measured CNN performance; the search never trains the candidate CNNs.

## Run all six conditions on CPU

Install the small policy dependency and use a normalized benchmark table:

```bash
python -m pip install -e '.[adaptation]'

python scripts/run_evolution.py \
  --backend table --data /path/to/nats_cifar10_hp200.json \
  --output results/self_evolution \
  --controller-config configs/evolution.json \
  --methods harness harness-frozen multiagent multiagent-frozen weight-ttt weight-frozen \
  --seeds 0 1 2 --budget 50 --warm-start 10
```

For the original official archive, install `.[adaptation,nats]` and replace the data arguments with `--backend nats --data data/NATS-tss-v1_0-3ffb9-simple`. Data preparation is described in the [README](../README.md#prepare-official-data).

The default proposer backend is `local`: explicit CPU search policies for harness and multi-agent experiments, and a NumPy policy for weight adaptation. This path requires neither an LLM nor an API key. Its outputs should be labelled **local-policy results**. Use separate output directories when changing algorithms, model backends or settings.

## 1. Harness evolution

Implementation: [`harness_evolution.py`](../src/nas_agent_bench/harness_evolution.py).

The harness is a validated declarative program. Its workflow can select a parent, cross over two candidates, mutate edges, and repair disconnected cells; other fields select memory and control exploration. The local backend executes these operators. The model backend applies memory selection directly and supplies the remaining program as instructions to the model. No generated Python code is executed.

A candidate program comes from a local mutation or a model-proposed JSON patch. Incumbent and challenger receive equal numbers of evaluation attempts in interleaved order. With the default `evaluation_window=4`, one trial costs **eight ordinary attempts**: four per program. Promotion requires a higher mean validation reward, after duplicate penalties, by more than `promotion_margin`. Rejection retains the incumbent. Versions, patches, credits and promotion/rejection decisions are recorded.

The resource archive retains the best observed validation result in each FLOPs/parameter bin. An `archive` memory policy can draw evidence from several resource niches. The bins organize memory; they do not impose a hard resource constraint.

`harness-frozen` preserves the initial program while continuing to collect observations. In model mode, adaptive runs also spend calls on harness patches; compare those calls and time alongside the equal evaluation budget. Online promotion is a selection rule, not an independent statistical test that a program generalizes to new tasks.

## 2. Multi-agent evolution

Implementation: [`multiagent_evolution.py`](../src/nas_agent_bench/multiagent_evolution.py).

Four initial strategies—exploration, exploitation, recombination and diversity—have separate genomes and bounded memories. Every search round has a proposer and a distinct peer critic:

1. The proposer creates one draft.
2. The peer critiques it and can revise it once.
3. The evaluator scores only the final candidate.
4. Both participants receive joint credit; scored elite observations are sent to connected peers.

The local critic checks repetition, connectivity and evidence in its archive. In model mode, the two stages make exactly **two model calls per attempt**, including failed calls. Internal drafts and revisions are logged separately from benchmark queries.

After each agent has proposed three evaluated candidates, the population is ranked. A stronger agent replaces a weaker one with a mutated descendant. Offspring inherit memory and mutate a role, exploration rate, mutation radius or critic revision rate; prompts reflect the new genome. Communication shortcuts can change while a ring backbone remains. Replacement lineage and old/new genomes and routes are saved.

`multiagent-frozen` retains critique, memory updates and communication but fixes genomes and routes. Joint final-candidate rewards do not estimate the critic's marginal contribution: the draft is deliberately not evaluated separately.

## 3. Weight adaptation inspired by TTT-Discover

Implementation: [`weight_evolution.py`](../src/nas_agent_bench/weight_evolution.py).

[Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175) trains a model using experience from the current discovery problem; its [official implementation](https://github.com/test-time-training/discover) provides the full TTT-Discover system. This repository adapts that principle to NAS with a small, auditable on-policy learner.

Weights remain fixed while a batch of four proposals is sampled and evaluated. Validation rewards determine an exponential weighting over the batch. A bounded temperature gives higher-reward samples greater weight, then a policy-gradient update changes their future sampling probabilities. A KL term anchors the policy to its initial distribution and an entropy term supports exploration. The implementation uses a batch-mean baseline; it does not reproduce the paper's complete estimator or search stack.

Two backends are available:

| Backend | Trainable object | Dependencies |
| --- | --- | --- |
| `numpy` | Autoregressive edge-operation policy with 405 active parameters | NumPy; CPU |
| `hf_lora` | Low-rank adapters on selected linear layers of a local causal LM | PyTorch, Transformers; current implementation uses CPU |

The LoRA backend uses a fixed task prompt and six constrained operation tokens. Prompt and sampling support are identical in frozen and adaptive conditions; validation feedback enters the gradient update. It changes actual model parameters. The NumPy backend changes a small search policy, not an LLM.

Warm-start candidates do not enter the gradient batch because the policy did not sample them. Repeated proposals still consume attempts; each completed policy sample can contribute its own reward once. Every run starts from fresh parameters and optimizer state. Events record update losses, gradient norms, parameter hashes and weight changes. `weight-frozen` samples from the same initial policy while skipping optimizer updates.

### Run actual language-model weight updates

Use a local Hugging Face checkpoint containing full trainable model files and its tokenizer, such as a locally downloaded Qwen2.5-0.5B-Instruct checkpoint. A GGUF inference file is not a trainable checkpoint for this backend.

```bash
python -m pip install -e '.[llm]'

python scripts/run_evolution.py \
  --backend table --data /path/to/nats_cifar10_hp200.json \
  --output results/self_evolution_lora \
  --controller-config configs/ttt_lora.json \
  --methods weight-ttt weight-frozen \
  --weight-backend hf_lora \
  --weight-model-path /path/to/Qwen2.5-0.5B-Instruct \
  --seeds 0 1 2 --budget 50 --warm-start 10
```

The model loader uses local files and disables custom remote code. The default adapter targets are `q_proj` and `v_proj`, rank 4. Change `target_modules` in the configuration for another architecture.

## Use an LLM for harness and multi-agent experiments

The same local HF checkpoint can serve as a frozen text proposer:

```bash
python scripts/run_evolution.py \
  --backend table --data /path/to/nats_cifar10_hp200.json \
  --output results/self_evolution_llm \
  --controller-config configs/evolution.json \
  --methods harness harness-frozen multiagent multiagent-frozen \
  --proposer-backend hf --model-path /path/to/Qwen2.5-0.5B-Instruct \
  --seeds 0 1 2 --budget 50 --warm-start 10
```

Alternatively, use `--proposer-backend command --adapter-command 'python your_adapter.py'`. The command receives one JSON payload on stdin and returns one JSON object on stdout. It must support architecture proposals (`ops`), harness edits (`patch`, when `task` is `propose_harness_patch`) and peer critique (`critique`, optionally revised `ops`, when `stage` is `critic`). Optional `model` and numeric `usage` fields support accounting. Logs belong on stderr. A legacy adapter that only returns `ops` cannot implement harness edits.

A ready-made [OpenAI-compatible HTTP adapter](../adapters/evolution/openai_compatible.py) supports all three tasks. Set `NAS_LLM_API_KEY` in the environment when authentication is required, and pass:

```bash
--proposer-backend command \
--model-label your-model-version \
--adapter-command 'python adapters/evolution/openai_compatible.py --base-url http://localhost:8000/v1 --model your-model'
```

The endpoint must implement `/chat/completions`. Add the adapter's `--json-mode` only if the chosen server supports it. No hosted API calls are made by default.

## Checkpoints and result interpretation

Each run writes:

| File | Contents |
| --- | --- |
| `<method>_seed<seed>.jsonl` | Every budgeted attempt, validation result, controller metadata and timing |
| `<method>_seed<seed>_events.jsonl` | Program trials, agent communication/replacement or weight updates |
| `checkpoints/<method>_seed<seed>.json` | Authoritative atomic checkpoint: history, controller, RNG, adapter and optimizer state |
| `config.json` and run configurations | Settings, data provenance and source hashes |
| `summary.json`, `summary.csv` | Run-level outcomes |
| `locked_winners.json` | Winners fixed by validation after all requested searches finish |

Add `--stop-after 25` to an initial command to checkpoint at absolute attempt 25. Repeat the same command with `--resume` and without `--stop-after` to finish the original budget. Resume requires identical algorithms, data, model settings and source code. It does not allocate another 50 attempts.

Only after every requested run completes, score the locked winners:

```bash
python scripts/finalize_results.py results/self_evolution \
  --backend table --data /path/to/nats_cifar10_hp200.json

python scripts/summarize_evolution.py results/self_evolution --plot
```

Controllers receive only allowlisted validation observations. Held-out test scores cannot enter program selection, agent credit or gradient updates. Test scoring remains a separate command after winner selection. The JSON boundary controls information passed to adapters; a custom subprocess adapter is not an operating-system sandbox.

Report each adaptive condition against its frozen counterpart across the same seeds. Include best validation/test results, unique evaluations, invalid/duplicate attempts, internal model calls, adaptation time and actual program/agent/weight changes. Equal final-candidate budgets do not imply equal model compute. Existing pilot results in [`results/REPORT.md`](../results/REPORT.md) remain the earlier frozen-LLM study.

Run the software checks with:

```bash
python -m unittest discover -s tests -v
```
