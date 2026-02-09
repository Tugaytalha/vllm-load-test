# vllm-load-test

Async step-sweep load testing tool for vLLM OpenAI-compatible servers with:

- Streaming client metrics (TTFT, e2e, inter-token gap stats)
- vLLM `/metrics` scraping (Prometheus text parsing)
- GPU telemetry via `nvidia-smi`
- Per-step summaries and markdown report output

## Requirements

- Python 3.10+
- `nvidia-smi` available on PATH for GPU telemetry
- vLLM server exposing:
  - `POST /v1/chat/completions`
  - `GET /metrics`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

```bash
python vllm_loadtest.py \
  --base-url http://localhost:8801 \
  --steps 1,2,4 \
  --step-duration-s 60 \
  --warmup-s 10 \
  --prompt-text "Hello" \
  --max-output-tokens 128
```

## Input Modes

- `fixed-prompt`:
  - Requires `--prompt-text`
- `prompt-file`:
  - Requires `--prompt-file` JSONL
  - Each row can be plain text, or JSON with `prompt`, `text`, or `messages`
- `synthetic`:
  - Requires `--input-size`
  - `--input-size` can be fixed integer or distribution spec

## Distribution Syntax

For `--max-output-tokens` and `--input-size`:

- Fixed: `128`
- Normal: `normal:<mean>:<std>:<min>:<max>`
- Lognormal: `lognormal:<mean>:<std>:<min>:<max>`

Examples:

```bash
--max-output-tokens normal:256:64:32:512
--input-size lognormal:6.0:0.5:128:4096
```

## Useful Flags

- Target:
  - `--base-url`
  - `--api-key`
  - `--model`
- Sweep:
  - `--steps`
  - `--step-duration-s`
  - `--warmup-s`
  - `--ramp-s`
  - `--cooldown-s`
- Sampling:
  - `--input-mode`
  - `--prompt-text`
  - `--prompt-file`
  - `--input-size`
  - `--max-output-tokens`
- Control:
  - `--seed`
  - `--timeout-s`
  - `--max-inflight`
  - `--poll-interval-s`
  - `--gpu-poll-interval-s`

## Outputs

Each run writes to:

```text
<output-dir>/<run-name>_<timestamp>/
```

Files:

- `config.json`
- `requests.jsonl`
- `vllm_metrics.csv`
- `gpu_metrics.csv`
- `step_summary.json`
- `vllm_step_deltas.json`
- `summary.md`

## Notes

- Client throughput is not inferred from streamed text; server token throughput comes from vLLM counter deltas.
- The tool requests streaming usage metadata, but remains functional if usage data is not present.
- Metric name aliases are handled for vLLM variations (for example `*_total` counters and TPOT/ITL naming differences).
