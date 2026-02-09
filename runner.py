from __future__ import annotations

import asyncio
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from loadgen import (
    NumericSampler,
    PromptProvider,
    RequestRecord,
    RequestSettings,
    worker_loop,
)
from metrics_gpu import GPUMetricsScraper
from metrics_vllm import VLLMMetricsScraper
from report import compute_step_summary, write_step_summary_json, write_summary_markdown


@dataclass
class RunConfig:
    base_url: str = "http://localhost:8801"
    api_key: Optional[str] = None
    model: Optional[str] = None
    steps: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    step_duration_s: int = 120
    warmup_s: int = 30
    ramp_s: int = 10
    cooldown_s: int = 10
    stream: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens_spec: str = "128"
    input_mode: str = "fixed-prompt"
    prompt_text: Optional[str] = None
    prompt_file: Optional[Path] = None
    input_size_spec: Optional[str] = None
    output_dir: Path = Path("runs")
    run_name: Optional[str] = None
    seed: int = 42
    poll_interval_s: float = 1.0
    gpu_poll_interval_s: float = 1.0
    timeout_s: float = 300.0
    max_inflight: Optional[int] = None


class AsyncJSONLWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._file = output_path.open("w", encoding="utf-8", buffering=1)
        self._lock = asyncio.Lock()

    async def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        async with self._lock:
            self._file.write(line + "\n")

    def close(self) -> None:
        self._file.close()


def _ensure_output_dir(base_output_dir: Path, run_name: Optional[str]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    normalized_run_name = (run_name or "run").strip().replace(" ", "_")
    output_dir = base_output_dir / f"{normalized_run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def _wait_for_workers(tasks: list[asyncio.Task[None]], timeout_s: float) -> None:
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_s))
    if pending:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    if done:
        await asyncio.gather(*done, return_exceptions=True)


def _warmup_concurrency(steps: list[int]) -> int:
    if not steps:
        return 1
    return max(1, min(min(steps), 2))


async def _run_warmup(
    *,
    warmup_s: int,
    concurrency: int,
    prompt_provider: PromptProvider,
    output_sampler: NumericSampler,
    client: httpx.AsyncClient,
    request_settings: RequestSettings,
    request_writer: AsyncJSONLWriter,
    seed: int,
    max_inflight: Optional[int],
) -> None:
    if warmup_s <= 0:
        return

    phase_state = {"value": "warmup"}
    stop_new_requests_event = asyncio.Event()
    inflight_semaphore = (
        asyncio.Semaphore(max_inflight) if max_inflight and max_inflight > 0 else None
    )

    async def on_request_done(record: RequestRecord) -> None:
        await request_writer.write(record.to_dict())

    worker_tasks: list[asyncio.Task[None]] = []
    for worker_id in range(concurrency):
        worker_seed = seed + (worker_id * 1009) + 17
        worker_rng = random.Random(worker_seed)
        worker_tasks.append(
            asyncio.create_task(
                worker_loop(
                    worker_id=worker_id,
                    step_name="warmup",
                    step_concurrency=concurrency,
                    stop_new_requests_event=stop_new_requests_event,
                    current_phase_fn=lambda: phase_state["value"],
                    prompt_provider=prompt_provider,
                    output_sampler=output_sampler,
                    rng=worker_rng,
                    client=client,
                    request_settings=request_settings,
                    on_request_done=on_request_done,
                    inflight_semaphore=inflight_semaphore,
                )
            )
        )

    await asyncio.sleep(float(warmup_s))
    stop_new_requests_event.set()
    phase_state["value"] = "warmup_cooldown"
    drain_timeout_s = max(5.0, float(request_settings.timeout_s))
    await _wait_for_workers(worker_tasks, timeout_s=drain_timeout_s)


async def _run_step(
    *,
    step_index: int,
    concurrency: int,
    step_duration_s: int,
    ramp_s: int,
    cooldown_s: int,
    prompt_provider: PromptProvider,
    output_sampler: NumericSampler,
    client: httpx.AsyncClient,
    request_settings: RequestSettings,
    request_writer: AsyncJSONLWriter,
    seed: int,
    max_inflight: Optional[int],
    vllm_scraper: VLLMMetricsScraper,
    gpu_scraper: GPUMetricsScraper,
) -> dict[str, Any]:
    step_name = f"step_{step_index}_c{concurrency}"
    phase_state = {"value": "ramp"}
    stop_new_requests_event = asyncio.Event()
    inflight_semaphore = (
        asyncio.Semaphore(max_inflight) if max_inflight and max_inflight > 0 else None
    )

    measured_records: list[RequestRecord] = []
    measured_lock = asyncio.Lock()

    async def on_request_done(record: RequestRecord) -> None:
        await request_writer.write(record.to_dict())
        if record.phase == "measure":
            async with measured_lock:
                measured_records.append(record)

    worker_tasks: list[asyncio.Task[None]] = []
    for worker_id in range(concurrency):
        worker_seed = seed + (step_index * 100_003) + (worker_id * 971)
        worker_rng = random.Random(worker_seed)
        worker_tasks.append(
            asyncio.create_task(
                worker_loop(
                    worker_id=worker_id,
                    step_name=step_name,
                    step_concurrency=concurrency,
                    stop_new_requests_event=stop_new_requests_event,
                    current_phase_fn=lambda: phase_state["value"],
                    prompt_provider=prompt_provider,
                    output_sampler=output_sampler,
                    rng=worker_rng,
                    client=client,
                    request_settings=request_settings,
                    on_request_done=on_request_done,
                    inflight_semaphore=inflight_semaphore,
                )
            )
        )

    if ramp_s > 0:
        await asyncio.sleep(float(ramp_s))

    phase_state["value"] = "measure"
    measurement_start_snapshot = await vllm_scraper.snapshot(
        source=f"{step_name}_measure_start"
    )
    await gpu_scraper.snapshot(source=f"{step_name}_measure_start")

    await asyncio.sleep(float(step_duration_s))

    measurement_end_snapshot = await vllm_scraper.snapshot(source=f"{step_name}_measure_end")
    await gpu_scraper.snapshot(source=f"{step_name}_measure_end")

    phase_state["value"] = "cooldown"
    stop_new_requests_event.set()
    if cooldown_s > 0:
        await asyncio.sleep(float(cooldown_s))
    drain_timeout_s = max(1.0, float(request_settings.timeout_s))
    await _wait_for_workers(worker_tasks, timeout_s=drain_timeout_s)

    measurement_start_unix_ms = measurement_start_snapshot.timestamp_unix_ms
    measurement_end_unix_ms = measurement_end_snapshot.timestamp_unix_ms
    vllm_rows_in_window = await vllm_scraper.rows_between(
        measurement_start_unix_ms, measurement_end_unix_ms
    )
    gpu_rows_in_window = await gpu_scraper.rows_between(
        measurement_start_unix_ms, measurement_end_unix_ms
    )

    async with measured_lock:
        measured_records_copy = list(measured_records)

    return compute_step_summary(
        step_index=step_index,
        step_name=step_name,
        concurrency=concurrency,
        step_duration_s=float(step_duration_s),
        measurement_start_unix_ms=measurement_start_unix_ms,
        measurement_end_unix_ms=measurement_end_unix_ms,
        measured_records=measured_records_copy,
        measurement_start_snapshot=measurement_start_snapshot,
        measurement_end_snapshot=measurement_end_snapshot,
        vllm_rows_in_window=vllm_rows_in_window,
        gpu_rows_in_window=gpu_rows_in_window,
    )


def _resolved_config_dict(config: RunConfig, output_dir: Path) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["prompt_file"] = str(config.prompt_file) if config.prompt_file else None
    payload["resolved_run_dir"] = str(output_dir)
    payload["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


async def run_load_test(config: RunConfig) -> Path:
    output_dir = _ensure_output_dir(config.output_dir, config.run_name)

    requests_path = output_dir / "requests.jsonl"
    config_path = output_dir / "config.json"
    vllm_metrics_path = output_dir / "vllm_metrics.csv"
    gpu_metrics_path = output_dir / "gpu_metrics.csv"
    step_summary_path = output_dir / "step_summary.json"
    vllm_step_deltas_path = output_dir / "vllm_step_deltas.json"
    summary_md_path = output_dir / "summary.md"

    input_sampler = (
        NumericSampler(config.input_size_spec, "input-size")
        if config.input_size_spec
        else None
    )
    output_sampler = NumericSampler(config.max_output_tokens_spec, "max-output-tokens")
    prompt_provider = PromptProvider(
        mode=config.input_mode,
        prompt_text=config.prompt_text,
        prompt_file=config.prompt_file,
        input_sampler=input_sampler,
    )

    request_settings = RequestSettings(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        stream=config.stream,
        temperature=config.temperature,
        top_p=config.top_p,
        timeout_s=float(config.timeout_s),
    )

    resolved_config = _resolved_config_dict(config, output_dir)
    _write_json(config_path, resolved_config)

    max_connections = max(max(config.steps) * 4, 64) if config.steps else 64
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max(max_connections // 2, 32),
    )

    request_writer = AsyncJSONLWriter(requests_path)
    vllm_scraper = VLLMMetricsScraper(
        base_url=config.base_url,
        poll_interval_s=float(config.poll_interval_s),
        request_timeout_s=min(10.0, max(2.0, float(config.poll_interval_s) * 2.0)),
    )
    gpu_scraper = GPUMetricsScraper(poll_interval_s=float(config.gpu_poll_interval_s))

    step_summaries: list[dict[str, Any]] = []
    try:
        await vllm_scraper.start()
        await gpu_scraper.start()

        async with httpx.AsyncClient(limits=limits) as client:
            if config.warmup_s > 0:
                await _run_warmup(
                    warmup_s=config.warmup_s,
                    concurrency=_warmup_concurrency(config.steps),
                    prompt_provider=prompt_provider,
                    output_sampler=output_sampler,
                    client=client,
                    request_settings=request_settings,
                    request_writer=request_writer,
                    seed=config.seed,
                    max_inflight=config.max_inflight,
                )

            for step_index, concurrency in enumerate(config.steps, start=1):
                summary = await _run_step(
                    step_index=step_index,
                    concurrency=concurrency,
                    step_duration_s=config.step_duration_s,
                    ramp_s=config.ramp_s,
                    cooldown_s=config.cooldown_s,
                    prompt_provider=prompt_provider,
                    output_sampler=output_sampler,
                    client=client,
                    request_settings=request_settings,
                    request_writer=request_writer,
                    seed=config.seed,
                    max_inflight=config.max_inflight,
                    vllm_scraper=vllm_scraper,
                    gpu_scraper=gpu_scraper,
                )
                step_summaries.append(summary)
    finally:
        request_writer.close()
        await vllm_scraper.stop()
        await gpu_scraper.stop()

    vllm_rows = await vllm_scraper.rows()
    gpu_rows = await gpu_scraper.rows()
    _write_csv(vllm_metrics_path, vllm_rows)
    _write_csv(gpu_metrics_path, gpu_rows)

    write_step_summary_json(step_summary_path, step_summaries)
    _write_json(
        vllm_step_deltas_path,
        [
            {
                "step_index": item["step_index"],
                "concurrency": item["concurrency"],
                "server_prompt_tokens_delta": item["server_prompt_tokens_delta"],
                "server_generation_tokens_delta": item["server_generation_tokens_delta"],
                "server_throughput_tok_s": item["server_throughput_tok_s"],
                "server_per_user_tok_s": item["server_per_user_tok_s"],
            }
            for item in step_summaries
        ],
    )
    write_summary_markdown(
        output_path=summary_md_path,
        run_name=config.run_name or "run",
        resolved_config=resolved_config,
        step_summaries=step_summaries,
    )
    return output_dir
