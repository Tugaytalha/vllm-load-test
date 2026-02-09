from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loadgen import RequestRecord, percentile
from metrics_vllm import VLLMMetricsSnapshot


def _counter_delta(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None:
        return None
    delta = end - start
    if delta < 0:
        return None
    return float(delta)


def _stat_triplet(values: list[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    return float(min(values)), float(statistics.fmean(values)), float(max(values))


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_ok_records(records: list[RequestRecord]) -> list[RequestRecord]:
    return [record for record in records if record.status == "ok"]


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def compute_step_summary(
    *,
    step_index: int,
    step_name: str,
    concurrency: int,
    step_duration_s: float,
    measurement_start_unix_ms: int,
    measurement_end_unix_ms: int,
    measured_records: list[RequestRecord],
    measurement_start_snapshot: VLLMMetricsSnapshot,
    measurement_end_snapshot: VLLMMetricsSnapshot,
    vllm_rows_in_window: list[dict[str, Any]],
    gpu_rows_in_window: list[dict[str, Any]],
) -> dict[str, Any]:
    req_count = len(measured_records)
    ok_records = _extract_ok_records(measured_records)
    ok_count = len(ok_records)
    timeout_count = sum(1 for record in measured_records if record.status == "timeout")
    error_count = req_count - ok_count
    error_rate = float(error_count / req_count) if req_count else 0.0

    ttft_values = [
        float(record.ttft_ms)
        for record in ok_records
        if record.ttft_ms is not None and record.ttft_ms >= 0
    ]
    e2e_values = [
        float(record.e2e_ms)
        for record in ok_records
        if record.e2e_ms is not None and record.e2e_ms >= 0
    ]
    client_itl_values = [
        float(record.itl_mean_ms)
        for record in ok_records
        if record.itl_mean_ms is not None and record.itl_mean_ms >= 0
    ]

    prompt_delta = _counter_delta(
        measurement_start_snapshot.prompt_tokens,
        measurement_end_snapshot.prompt_tokens,
    )
    generation_delta = _counter_delta(
        measurement_start_snapshot.generation_tokens,
        measurement_end_snapshot.generation_tokens,
    )

    throughput_tok_s = (
        float(generation_delta / step_duration_s)
        if generation_delta is not None and step_duration_s > 0
        else None
    )
    per_user_tok_s = (
        float(throughput_tok_s / concurrency)
        if throughput_tok_s is not None and concurrency > 0
        else None
    )

    queue_waiting = [
        float(value)
        for value in (_to_float(row.get("num_requests_waiting")) for row in vllm_rows_in_window)
        if value is not None
    ]
    queue_running = [
        float(value)
        for value in (_to_float(row.get("num_requests_running")) for row in vllm_rows_in_window)
        if value is not None
    ]
    queue_swapped = [
        float(value)
        for value in (_to_float(row.get("num_requests_swapped")) for row in vllm_rows_in_window)
        if value is not None
    ]

    waiting_min, waiting_mean, waiting_max = _stat_triplet(queue_waiting)
    running_min, running_mean, running_max = _stat_triplet(queue_running)
    swapped_min, swapped_mean, swapped_max = _stat_triplet(queue_swapped)

    itl_count_delta = _counter_delta(
        measurement_start_snapshot.itl_count,
        measurement_end_snapshot.itl_count,
    )
    itl_sum_delta = _counter_delta(
        measurement_start_snapshot.itl_sum,
        measurement_end_snapshot.itl_sum,
    )
    server_mean_itl_seconds = None
    server_mean_itl_ms = None
    server_estimated_generation_tok_s = None
    if (
        itl_count_delta is not None
        and itl_sum_delta is not None
        and itl_count_delta > 0
        and itl_sum_delta >= 0
    ):
        server_mean_itl_seconds = float(itl_sum_delta / itl_count_delta)
        server_mean_itl_ms = float(server_mean_itl_seconds * 1000.0)
        if server_mean_itl_seconds > 0:
            server_estimated_generation_tok_s = float(1.0 / server_mean_itl_seconds)

    gpu_memory_used = [
        float(value)
        for value in (_to_float(row.get("memory_used_mib")) for row in gpu_rows_in_window)
        if value is not None
    ]
    gpu_utilization = [
        float(value)
        for value in (_to_float(row.get("utilization_gpu_pct")) for row in gpu_rows_in_window)
        if value is not None
    ]
    gpu_mem_utilization = [
        float(value)
        for value in (_to_float(row.get("utilization_memory_pct")) for row in gpu_rows_in_window)
        if value is not None
    ]
    peak_memory_used_mib = max(gpu_memory_used) if gpu_memory_used else None
    peak_gpu_utilization_pct = max(gpu_utilization) if gpu_utilization else None
    peak_gpu_memory_utilization_pct = (
        max(gpu_mem_utilization) if gpu_mem_utilization else None
    )

    return {
        "step_index": step_index,
        "step_name": step_name,
        "concurrency": concurrency,
        "measurement_start_unix_ms": measurement_start_unix_ms,
        "measurement_end_unix_ms": measurement_end_unix_ms,
        "step_duration_s": step_duration_s,
        "req_count": req_count,
        "ok_count": ok_count,
        "error_count": error_count,
        "timeout_count": timeout_count,
        "error_rate": error_rate,
        "client_ttft_ms_p50": percentile(ttft_values, 50.0),
        "client_ttft_ms_p90": percentile(ttft_values, 90.0),
        "client_ttft_ms_p99": percentile(ttft_values, 99.0),
        "client_e2e_ms_p50": percentile(e2e_values, 50.0),
        "client_e2e_ms_p90": percentile(e2e_values, 90.0),
        "client_e2e_ms_p99": percentile(e2e_values, 99.0),
        "client_itl_ms_mean_p50": percentile(client_itl_values, 50.0),
        "client_itl_ms_mean_p90": percentile(client_itl_values, 90.0),
        "server_prompt_tokens_delta": prompt_delta,
        "server_generation_tokens_delta": generation_delta,
        "server_throughput_tok_s": throughput_tok_s,
        "server_per_user_tok_s": per_user_tok_s,
        "server_mean_itl_ms": server_mean_itl_ms,
        "server_estimated_generation_tok_s": server_estimated_generation_tok_s,
        "queue_waiting_min": waiting_min,
        "queue_waiting_mean": waiting_mean,
        "queue_waiting_max": waiting_max,
        "queue_running_min": running_min,
        "queue_running_mean": running_mean,
        "queue_running_max": running_max,
        "queue_swapped_min": swapped_min,
        "queue_swapped_mean": swapped_mean,
        "queue_swapped_max": swapped_max,
        "gpu_peak_memory_used_mib": peak_memory_used_mib,
        "gpu_peak_utilization_gpu_pct": peak_gpu_utilization_pct,
        "gpu_peak_utilization_memory_pct": peak_gpu_memory_utilization_pct,
        "measurement_start_snapshot": measurement_start_snapshot.to_row(),
        "measurement_end_snapshot": measurement_end_snapshot.to_row(),
    }


def write_step_summary_json(output_path: Path, step_summaries: list[dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(step_summaries, indent=2), encoding="utf-8")


def write_summary_markdown(
    output_path: Path,
    run_name: str,
    resolved_config: dict[str, Any],
    step_summaries: list[dict[str, Any]],
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append(f"# vLLM Load Test Summary - {run_name}")
    lines.append("")
    lines.append(f"Generated at (UTC): `{generated_at}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(resolved_config, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Step Results")
    lines.append("")
    lines.append(
        "| Concurrency | Req | Error % | TTFT p50 ms | TTFT p90 ms | TTFT p99 ms | "
        "E2E p50 ms | E2E p90 ms | E2E p99 ms | Throughput tok/s | Per-user tok/s | "
        "Peak VRAM MiB | Peak GPU util % |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for summary in step_summaries:
        lines.append(
            "| "
            f"{summary['concurrency']} | "
            f"{summary['req_count']} | "
            f"{_fmt(summary['error_rate'] * 100.0)} | "
            f"{_fmt(summary['client_ttft_ms_p50'])} | "
            f"{_fmt(summary['client_ttft_ms_p90'])} | "
            f"{_fmt(summary['client_ttft_ms_p99'])} | "
            f"{_fmt(summary['client_e2e_ms_p50'])} | "
            f"{_fmt(summary['client_e2e_ms_p90'])} | "
            f"{_fmt(summary['client_e2e_ms_p99'])} | "
            f"{_fmt(summary['server_throughput_tok_s'])} | "
            f"{_fmt(summary['server_per_user_tok_s'])} | "
            f"{_fmt(summary['gpu_peak_memory_used_mib'])} | "
            f"{_fmt(summary['gpu_peak_utilization_gpu_pct'])} |"
        )

    lines.append("")
    lines.append("## Queue Stats")
    lines.append("")
    lines.append(
        "| Concurrency | Waiting min/mean/max | Running min/mean/max | Swapped min/mean/max |"
    )
    lines.append("|---:|---:|---:|---:|")
    for summary in step_summaries:
        lines.append(
            "| "
            f"{summary['concurrency']} | "
            f"{_fmt(summary['queue_waiting_min'])}/{_fmt(summary['queue_waiting_mean'])}/{_fmt(summary['queue_waiting_max'])} | "
            f"{_fmt(summary['queue_running_min'])}/{_fmt(summary['queue_running_mean'])}/{_fmt(summary['queue_running_max'])} | "
            f"{_fmt(summary['queue_swapped_min'])}/{_fmt(summary['queue_swapped_mean'])}/{_fmt(summary['queue_swapped_max'])} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
