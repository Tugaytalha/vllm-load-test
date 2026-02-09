from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import httpx
from prometheus_client.parser import text_string_to_metric_families


PROMPT_TOKEN_ALIASES = [
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens",
]
GENERATION_TOKEN_ALIASES = [
    "vllm:generation_tokens_total",
    "vllm:generation_tokens",
]
RUNNING_ALIASES = ["vllm:num_requests_running"]
WAITING_ALIASES = ["vllm:num_requests_waiting"]
SWAPPED_ALIASES = ["vllm:num_requests_swapped"]

TTFT_BASE_ALIASES = ["vllm:time_to_first_token_seconds"]
ITL_BASE_ALIASES = [
    "vllm:inter_token_latency_seconds",
    "vllm:time_per_output_token_seconds",
]
E2E_BASE_ALIASES = ["vllm:e2e_request_latency_seconds"]


def _le_sort_key(le_value: str) -> float:
    if le_value in {"+Inf", "Inf", "inf"}:
        return math.inf
    try:
        return float(le_value)
    except ValueError:
        return math.inf


def _sorted_buckets(buckets: dict[str, float]) -> dict[str, float]:
    ordered = sorted(buckets.items(), key=lambda item: _le_sort_key(item[0]))
    return {key: value for key, value in ordered}


def _pick_first(values: dict[str, float], aliases: list[str]) -> Optional[float]:
    for alias in aliases:
        if alias in values:
            return values[alias]
    return None


def _pick_histogram(
    values: dict[str, float],
    buckets: dict[str, dict[str, float]],
    base_aliases: list[str],
) -> tuple[Optional[float], Optional[float], dict[str, float]]:
    for base in base_aliases:
        count_name = f"{base}_count"
        sum_name = f"{base}_sum"
        bucket_name = f"{base}_bucket"
        if count_name in values or sum_name in values or bucket_name in buckets:
            return (
                values.get(count_name),
                values.get(sum_name),
                _sorted_buckets(buckets.get(bucket_name, {})),
            )
    return None, None, {}


@dataclass
class VLLMMetricsSnapshot:
    timestamp_unix_ms: int
    source: str
    scrape_ok: bool
    scrape_error: Optional[str]
    prompt_tokens: Optional[float]
    generation_tokens: Optional[float]
    num_requests_running: Optional[float]
    num_requests_waiting: Optional[float]
    num_requests_swapped: Optional[float]
    ttft_count: Optional[float]
    ttft_sum: Optional[float]
    ttft_buckets: dict[str, float] = field(default_factory=dict)
    itl_count: Optional[float] = None
    itl_sum: Optional[float] = None
    itl_buckets: dict[str, float] = field(default_factory=dict)
    e2e_count: Optional[float] = None
    e2e_sum: Optional[float] = None
    e2e_buckets: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["ttft_buckets"] = json.dumps(self.ttft_buckets, sort_keys=False)
        row["itl_buckets"] = json.dumps(self.itl_buckets, sort_keys=False)
        row["e2e_buckets"] = json.dumps(self.e2e_buckets, sort_keys=False)
        return row


def parse_prometheus_metrics(text: str, source: str, timestamp_unix_ms: int) -> VLLMMetricsSnapshot:
    values: dict[str, float] = {}
    buckets: dict[str, dict[str, float]] = {}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name
            value = float(sample.value)
            if name.endswith("_bucket"):
                le = sample.labels.get("le")
                if le is None:
                    continue
                if name not in buckets:
                    buckets[name] = {}
                buckets[name][le] = buckets[name].get(le, 0.0) + value
            else:
                values[name] = values.get(name, 0.0) + value

    ttft_count, ttft_sum, ttft_buckets = _pick_histogram(
        values=values,
        buckets=buckets,
        base_aliases=TTFT_BASE_ALIASES,
    )
    itl_count, itl_sum, itl_buckets = _pick_histogram(
        values=values,
        buckets=buckets,
        base_aliases=ITL_BASE_ALIASES,
    )
    e2e_count, e2e_sum, e2e_buckets = _pick_histogram(
        values=values,
        buckets=buckets,
        base_aliases=E2E_BASE_ALIASES,
    )

    return VLLMMetricsSnapshot(
        timestamp_unix_ms=timestamp_unix_ms,
        source=source,
        scrape_ok=True,
        scrape_error=None,
        prompt_tokens=_pick_first(values, PROMPT_TOKEN_ALIASES),
        generation_tokens=_pick_first(values, GENERATION_TOKEN_ALIASES),
        num_requests_running=_pick_first(values, RUNNING_ALIASES),
        num_requests_waiting=_pick_first(values, WAITING_ALIASES),
        num_requests_swapped=_pick_first(values, SWAPPED_ALIASES),
        ttft_count=ttft_count,
        ttft_sum=ttft_sum,
        ttft_buckets=ttft_buckets,
        itl_count=itl_count,
        itl_sum=itl_sum,
        itl_buckets=itl_buckets,
        e2e_count=e2e_count,
        e2e_sum=e2e_sum,
        e2e_buckets=e2e_buckets,
    )


class VLLMMetricsScraper:
    def __init__(
        self,
        base_url: str,
        poll_interval_s: float = 1.0,
        request_timeout_s: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.request_timeout_s = request_timeout_s
        self._client: Optional[httpx.AsyncClient] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._rows: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._client = httpx.AsyncClient(timeout=self.request_timeout_s)
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def snapshot(self, source: str = "snapshot") -> VLLMMetricsSnapshot:
        snapshot = await self._scrape_once(source=source)
        async with self._lock:
            self._rows.append(snapshot.to_row())
        return snapshot

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_started = time.monotonic()
            snapshot = await self._scrape_once(source="poll")
            async with self._lock:
                self._rows.append(snapshot.to_row())
            elapsed = time.monotonic() - loop_started
            sleep_for = max(0.0, self.poll_interval_s - elapsed)
            if sleep_for <= 0.0:
                continue
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                pass

    async def _scrape_once(self, source: str) -> VLLMMetricsSnapshot:
        timestamp_unix_ms = int(time.time() * 1000)
        metrics_url = f"{self.base_url}/metrics"

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout_s)

        try:
            response = await self._client.get(metrics_url, timeout=self.request_timeout_s)
            if response.status_code != 200:
                return VLLMMetricsSnapshot(
                    timestamp_unix_ms=timestamp_unix_ms,
                    source=source,
                    scrape_ok=False,
                    scrape_error=f"HTTP {response.status_code}",
                    prompt_tokens=None,
                    generation_tokens=None,
                    num_requests_running=None,
                    num_requests_waiting=None,
                    num_requests_swapped=None,
                    ttft_count=None,
                    ttft_sum=None,
                    itl_count=None,
                    itl_sum=None,
                    e2e_count=None,
                    e2e_sum=None,
                )
            return parse_prometheus_metrics(
                text=response.text,
                source=source,
                timestamp_unix_ms=timestamp_unix_ms,
            )
        except Exception as exc:  # noqa: BLE001
            return VLLMMetricsSnapshot(
                timestamp_unix_ms=timestamp_unix_ms,
                source=source,
                scrape_ok=False,
                scrape_error=str(exc),
                prompt_tokens=None,
                generation_tokens=None,
                num_requests_running=None,
                num_requests_waiting=None,
                num_requests_swapped=None,
                ttft_count=None,
                ttft_sum=None,
                itl_count=None,
                itl_sum=None,
                e2e_count=None,
                e2e_sum=None,
            )

    async def rows(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._rows)

    async def rows_between(self, start_unix_ms: int, end_unix_ms: int) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                row
                for row in self._rows
                if start_unix_ms <= int(row["timestamp_unix_ms"]) <= end_unix_ms
            ]
