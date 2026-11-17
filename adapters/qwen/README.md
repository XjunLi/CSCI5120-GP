# Optional CPU model adapter

This adapter runs the official `Qwen/Qwen2.5-0.5B-Instruct` checkpoint on CPU. A real loading/generation smoke check succeeded in the development environment. **The main reported NAS experiments use interactive ChatGPT sessions, not this checkpoint.** No NAS score is claimed for the local model.

The adapter uses ordinary unconstrained sampling and does not repair invalid model outputs or replace them with random candidates. It preserves the raw response and actual token usage. Malformed outputs count against the search budget.

## Prepare the model

In a dedicated environment, install the CPU PyTorch wheel, then the remaining packages:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.57.1 huggingface-hub==0.36.2 safetensors==0.8.0
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775', local_dir='models/Qwen2.5-0.5B-Instruct')"
```

The model revision and file checksums from the development download are in `model_provenance.json`. Weights are excluded from Git.

## Run

From the repository root, with the NAS data prepared:

```bash
python adapters/qwen/run_with_qwen.py -- \
  python scripts/run_benchmarks.py \
    --backend nats --data data/NATS-tss-v1_0-3ffb9-simple \
    --output results/my_qwen \
    --methods llm llm-no-feedback --seeds 0 1 2 \
    --budget 50 --warm-start 10 \
    --model-label Qwen2.5-0.5B-Instruct \
    --adapter-command 'python adapters/qwen/qwen_adapter.py'
```

The wrapper keeps a single local model process alive during the entire command, then cleans it up. By default it uses the active Python interpreter, the checkpoint under `models/`, port 8712, and four CPU threads. Override `--model-python`, `--model-path`, or `--port` before `--` when needed. If overriding the port, also pass the matching `--url` to `qwen_adapter.py`.

Default generation settings are temperature 0.7, top-p 0.9, and at most 128 new tokens. The generation seed is derived from the run seed and attempt number, and included with every response. The server offers no tools, table access, or filesystem access to the model.

Source: [official model card and checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775).
