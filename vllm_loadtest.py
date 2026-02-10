from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(values)
    index = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    fraction = index - low
    return float((ordered[low] * (1.0 - fraction)) + (ordered[high] * fraction))


class NumericSampler:
    def __init__(self, spec: str, label: str) -> None:
        self.spec = spec.strip()
        self.label = label
        self._sampler = self._build_sampler(self.spec)

    def _build_sampler(self, spec: str) -> Callable[[random.Random], int]:
        if ":" not in spec:
            value = int(spec)
            if value <= 0:
                raise ValueError(f"{self.label} must be > 0, got {value}")
            return lambda _rng: value

        parts = spec.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"Invalid {self.label} distribution: {spec}. "
                "Expected normal:<mean>:<std>:<min>:<max> or "
                "lognormal:<mean>:<std>:<min>:<max>."
            )
        dist = parts[0].lower()
        mean = float(parts[1])
        std = float(parts[2])
        lower = int(float(parts[3]))
        upper = int(float(parts[4]))
        if std < 0:
            raise ValueError(f"{self.label} std must be >= 0, got {std}")
        if lower <= 0 or upper <= 0 or lower > upper:
            raise ValueError(
                f"{self.label} bounds must satisfy 0 < min <= max, got {lower}, {upper}"
            )

        if dist == "normal":
            return lambda rng: int(min(max(round(rng.gauss(mean, std)), lower), upper))
        if dist == "lognormal":
            return lambda rng: int(
                min(max(round(rng.lognormvariate(mean, std)), lower), upper)
            )
        raise ValueError(f"Unsupported {self.label} distribution kind: {dist}")

    def sample(self, rng: random.Random) -> int:
        value = int(self._sampler(rng))
        return max(1, value)


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def _extract_prompt_from_jsonl_row(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""

    prompt = obj.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    text = obj.get("text")
    if isinstance(text, str) and text.strip():
        return text

    messages = obj.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "")
            content = _flatten_message_content(message.get("content"))
            if not content:
                continue
            parts.append(f"{role}: {content}".strip())
        return "\n".join(parts)
    return ""


def load_prompts_from_jsonl(prompt_file: Path) -> list[str]:
    prompts: list[str] = []
    with prompt_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                prompts.append(value)
                continue
            prompt = _extract_prompt_from_jsonl_row(parsed)
            if prompt:
                prompts.append(prompt)
            else:
                raise ValueError(
                    f"Unsupported prompt row at line {line_number} in {prompt_file}"
                )
    if not prompts:
        raise ValueError(f"No usable prompts found in {prompt_file}")
    return prompts


def build_synthetic_prompt(target_chars: int) -> str:
    target_chars = max(32, target_chars)
    base_block = (
        "Synthetic load-test prompt block for repeatable benchmarking. "
        "Summarize performance considerations for streaming language model inference. "
    )
    repeats = math.ceil(target_chars / len(base_block))
    body = (base_block * repeats)[:target_chars]
    return f"{body}\nAnswer in concise bullet points."


class PromptProvider:
    def __init__(
        self,
        mode: str,
        prompt_text: Optional[str],
        prompt_file: Optional[Path],
        input_sampler: Optional[NumericSampler],
    ) -> None:
        self.mode = mode
        self.prompt_text = prompt_text
        self.prompt_file = prompt_file
        self.input_sampler = input_sampler
        self._prompts: list[str] = []

        if mode == "fixed-prompt":
            if not prompt_text:
                raise ValueError("--prompt-text is required when --input-mode=fixed-prompt")
            self._prompts = [prompt_text]
        elif mode == "prompt-file":
            if not prompt_file:
                raise ValueError("--prompt-file is required when --input-mode=prompt-file")
            self._prompts = load_prompts_from_jsonl(prompt_file)
        elif mode == "synthetic":
            if not input_sampler:
                raise ValueError(
                    "--input-size is required when --input-mode=synthetic"
                )
        else:
            raise ValueError(f"Unsupported input mode: {mode}")

    def sample(self, rng: random.Random) -> str:
        if self.mode == "fixed-prompt":
            return self._prompts[0]
        if self.mode == "prompt-file":
            return rng.choice(self._prompts)
        if self.mode == "synthetic":
            assert self.input_sampler is not None
            return build_synthetic_prompt(self.input_sampler.sample(rng))
        raise RuntimeError(f"Unsupported input mode at runtime: {self.mode}")


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_content(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    delta = first_choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            return "".join(text_parts)

    text = first_choice.get("text")
    if isinstance(text, str):
        return text
    return ""


def _compute_itl_stats(content_timestamps_ms: list[int]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(content_timestamps_ms) < 2:
        return None, None, None
    gaps = [
        float(next_ts - current_ts)
        for current_ts, next_ts in zip(content_timestamps_ms, content_timestamps_ms[1:])
    ]
    return (
        float(statistics.fmean(gaps)),
        percentile(gaps, 50.0),
        percentile(gaps, 95.0),
    )


@dataclass
class RequestSettings:
    base_url: str
    api_key: Optional[str]
    model: Optional[str]
    stream: bool
    temperature: float
    top_p: float
    timeout_s: float


@dataclass
class RequestRecord:
    request_id: str
    step_name: str
    step_concurrency: int
    phase: str
    worker_id: int
    start_time_unix_ms: int
    first_token_time_unix_ms: Optional[int]
    end_time_unix_ms: int
    ttft_ms: Optional[float]
    e2e_ms: float
    status: str
    error: Optional[str]
    http_status: Optional[int]
    sse_chunks: int
    content_chunks: int
    itl_mean_ms: Optional[float]
    itl_p50_ms: Optional[float]
    itl_p95_ms: Optional[float]
    bytes_received: int
    prompt_chars: int
    max_output_tokens: int
    usage_prompt_tokens: Optional[int]
    usage_completion_tokens: Optional[int]
    usage_total_tokens: Optional[int]
    saw_done: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_payload(
    settings: RequestSettings,
    prompt: str,
    max_output_tokens: int,
    include_usage: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "stream": settings.stream,
        "max_tokens": max_output_tokens,
    }
    if settings.model:
        payload["model"] = settings.model
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _headers(api_key: Optional[str]) -> dict[str, str]:
    base = {"Content-Type": "application/json"}
    if api_key:
        base["Authorization"] = f"Bearer {api_key}"
    return base


def _should_retry_without_usage(http_status: Optional[int], error_text: Optional[str]) -> bool:
    if http_status is None or http_status >= 500:
        return False
    if not error_text:
        return False
    lowered = error_text.lower()
    if "stream_options" in lowered:
        return True
    if "include_usage" in lowered:
        return True
    if "extra_forbidden" in lowered:
        return True
    return False


async def _stream_once(
    client: httpx.AsyncClient,
    settings: RequestSettings,
    prompt: str,
    max_output_tokens: int,
    include_usage: bool,
    request_id: str,
    step_name: str,
    step_concurrency: int,
    phase: str,
    worker_id: int,
    start_time_ms: int,
) -> RequestRecord:
    first_token_time_ms: Optional[int] = None
    end_time_ms = start_time_ms
    sse_chunks = 0
    content_chunks = 0
    bytes_received = 0
    content_timestamps: list[int] = []
    usage_prompt_tokens: Optional[int] = None
    usage_completion_tokens: Optional[int] = None
    usage_total_tokens: Optional[int] = None
    saw_done = False
    status = "ok"
    error_text: Optional[str] = None
    http_status: Optional[int] = None

    url = f"{settings.base_url.rstrip('/')}/v1/chat/completions"
    payload = _build_payload(
        settings=settings,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        include_usage=include_usage,
    )

    try:
        async with client.stream(
            "POST",
            url,
            headers=_headers(settings.api_key),
            json=payload,
            timeout=settings.timeout_s,
        ) as response:
            http_status = int(response.status_code)
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                error_text = body[:2000]
                status = "error"
            else:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    bytes_received += len(line.encode("utf-8"))
                    if not line.startswith("data:"):
                        continue
                    payload_text = line[5:].strip()
                    if not payload_text:
                        continue
                    if payload_text == "[DONE]":
                        saw_done = True
                        break

                    sse_chunks += 1
                    try:
                        chunk = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        usage_prompt_tokens = _safe_int(usage.get("prompt_tokens"))
                        usage_completion_tokens = _safe_int(
                            usage.get("completion_tokens")
                        )
                        usage_total_tokens = _safe_int(usage.get("total_tokens"))

                    content = _extract_content(chunk)
                    if content:
                        timestamp_ms = now_unix_ms()
                        content_timestamps.append(timestamp_ms)
                        content_chunks += 1
                        if first_token_time_ms is None:
                            first_token_time_ms = timestamp_ms
    except httpx.TimeoutException as exc:
        status = "timeout"
        error_text = str(exc)
    except httpx.HTTPError as exc:
        status = "error"
        error_text = str(exc)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_text = str(exc)

    end_time_ms = now_unix_ms()
    ttft_ms: Optional[float] = None
    if first_token_time_ms is not None:
        ttft_ms = float(first_token_time_ms - start_time_ms)

    itl_mean_ms, itl_p50_ms, itl_p95_ms = _compute_itl_stats(content_timestamps)
    return RequestRecord(
        request_id=request_id,
        step_name=step_name,
        step_concurrency=step_concurrency,
        phase=phase,
        worker_id=worker_id,
        start_time_unix_ms=start_time_ms,
        first_token_time_unix_ms=first_token_time_ms,
        end_time_unix_ms=end_time_ms,
        ttft_ms=ttft_ms,
        e2e_ms=float(end_time_ms - start_time_ms),
        status=status,
        error=error_text,
        http_status=http_status,
        sse_chunks=sse_chunks,
        content_chunks=content_chunks,
        itl_mean_ms=itl_mean_ms,
        itl_p50_ms=itl_p50_ms,
        itl_p95_ms=itl_p95_ms,
        bytes_received=bytes_received,
        prompt_chars=len(prompt),
        max_output_tokens=max_output_tokens,
        usage_prompt_tokens=usage_prompt_tokens,
        usage_completion_tokens=usage_completion_tokens,
        usage_total_tokens=usage_total_tokens,
        saw_done=saw_done,
    )


async def execute_streaming_request(
    client: httpx.AsyncClient,
    settings: RequestSettings,
    prompt: str,
    max_output_tokens: int,
    step_name: str,
    step_concurrency: int,
    phase: str,
    worker_id: int,
) -> RequestRecord:
    request_id = str(uuid.uuid4())
    start_time_ms = now_unix_ms()

    first_attempt = await _stream_once(
        client=client,
        settings=settings,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        include_usage=True,
        request_id=request_id,
        step_name=step_name,
        step_concurrency=step_concurrency,
        phase=phase,
        worker_id=worker_id,
        start_time_ms=start_time_ms,
    )
    if _should_retry_without_usage(first_attempt.http_status, first_attempt.error):
        second_attempt = await _stream_once(
            client=client,
            settings=settings,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            include_usage=False,
            request_id=request_id,
            step_name=step_name,
            step_concurrency=step_concurrency,
            phase=phase,
            worker_id=worker_id,
            start_time_ms=start_time_ms,
        )
        if second_attempt.status == "ok":
            return second_attempt
    return first_attempt


async def worker_loop(
    worker_id: int,
    step_name: str,
    step_concurrency: int,
    stop_new_requests_event: asyncio.Event,
    current_phase_fn: Callable[[], str],
    prompt_provider: PromptProvider,
    output_sampler: NumericSampler,
    rng: random.Random,
    client: httpx.AsyncClient,
    request_settings: RequestSettings,
    on_request_done: Callable[[RequestRecord], Awaitable[None]],
    inflight_semaphore: Optional[asyncio.Semaphore] = None,
) -> None:
    while not stop_new_requests_event.is_set():
        phase = current_phase_fn()
        prompt = prompt_provider.sample(rng)
        max_output_tokens = output_sampler.sample(rng)

        try:
            if inflight_semaphore is not None:
                async with inflight_semaphore:
                    record = await execute_streaming_request(
                        client=client,
                        settings=request_settings,
                        prompt=prompt,
                        max_output_tokens=max_output_tokens,
                        step_name=step_name,
                        step_concurrency=step_concurrency,
                        phase=phase,
                        worker_id=worker_id,
                    )
            else:
                record = await execute_streaming_request(
                    client=client,
                    settings=request_settings,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    step_name=step_name,
                    step_concurrency=step_concurrency,
                    phase=phase,
                    worker_id=worker_id,
                )
            await on_request_done(record)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            fallback_record = RequestRecord(
                request_id=str(uuid.uuid4()),
                step_name=step_name,
                step_concurrency=step_concurrency,
                phase=phase,
                worker_id=worker_id,
                start_time_unix_ms=now_unix_ms(),
                first_token_time_unix_ms=None,
                end_time_unix_ms=now_unix_ms(),
                ttft_ms=None,
                e2e_ms=0.0,
                status="error",
                error=str(exc),
                http_status=None,
                sse_chunks=0,
                content_chunks=0,
                itl_mean_ms=None,
                itl_p50_ms=None,
                itl_p95_ms=None,
                bytes_received=0,
                prompt_chars=len(prompt),
                max_output_tokens=max_output_tokens,
                usage_prompt_tokens=None,
                usage_completion_tokens=None,
                usage_total_tokens=None,
                saw_done=False,
            )
            await on_request_done(fallback_record)

# ---- metrics_vllm.py ----
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import httpx

try:
    from prometheus_client.parser import text_string_to_metric_families
except ImportError:  # pragma: no cover - exercised in environments without deps installed.
    text_string_to_metric_families = None  # type: ignore[assignment]


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
    if text_string_to_metric_families is None:
        raise RuntimeError(
            "prometheus-client is required for /metrics parsing. "
            "Install dependencies with: pip install -r requirements.txt"
        )

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
            error_text = str(exc) or exc.__class__.__name__
            return VLLMMetricsSnapshot(
                timestamp_unix_ms=timestamp_unix_ms,
                source=source,
                scrape_ok=False,
                scrape_error=error_text,
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

# ---- metrics_gpu.py ----
import asyncio
import csv
import io
import time
from typing import Any, Optional


GPU_QUERY_FIELDS = [
    "timestamp",
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
]


def _to_float(value: str) -> Optional[float]:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.upper() in {"N/A", "NA"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _error_row(timestamp_unix_ms: int, source: str, error: str) -> dict[str, Any]:
    return {
        "timestamp_unix_ms": timestamp_unix_ms,
        "source": source,
        "scrape_ok": False,
        "scrape_error": error,
        "query_timestamp": None,
        "gpu_index": None,
        "gpu_uuid": None,
        "gpu_name": None,
        "utilization_gpu_pct": None,
        "utilization_memory_pct": None,
        "memory_used_mib": None,
        "memory_total_mib": None,
        "temperature_gpu_c": None,
        "power_draw_w": None,
    }


class GPUMetricsScraper:
    def __init__(self, poll_interval_s: float = 1.0) -> None:
        self.poll_interval_s = poll_interval_s
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None
        self._rows: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._disabled = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            rows = await self._scrape_once(source="poll")
            async with self._lock:
                self._rows.extend(rows)
            elapsed = time.monotonic() - started
            sleep_for = max(0.0, self.poll_interval_s - elapsed)
            if sleep_for <= 0:
                continue
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                pass

    async def snapshot(self, source: str = "snapshot") -> list[dict[str, Any]]:
        rows = await self._scrape_once(source=source)
        async with self._lock:
            self._rows.extend(rows)
        return rows

    async def _scrape_once(self, source: str) -> list[dict[str, Any]]:
        timestamp_unix_ms = int(time.time() * 1000)
        if self._disabled:
            return [_error_row(timestamp_unix_ms, source, "nvidia-smi unavailable")]

        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        except FileNotFoundError:
            self._disabled = True
            return [_error_row(timestamp_unix_ms, source, "nvidia-smi command not found")]
        except Exception as exc:  # noqa: BLE001
            return [_error_row(timestamp_unix_ms, source, str(exc))]

        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            return [
                _error_row(
                    timestamp_unix_ms,
                    source,
                    f"nvidia-smi exited with {process.returncode}: {stderr_text}",
                )
            ]

        rows: list[dict[str, Any]] = []
        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            return [_error_row(timestamp_unix_ms, source, "nvidia-smi returned empty output")]

        reader = csv.reader(io.StringIO(output))
        for parsed in reader:
            if not parsed:
                continue
            parsed = [value.strip() for value in parsed]
            if len(parsed) < len(GPU_QUERY_FIELDS):
                rows.append(
                    _error_row(
                        timestamp_unix_ms,
                        source,
                        f"Unexpected nvidia-smi row with {len(parsed)} fields",
                    )
                )
                continue

            if len(parsed) > len(GPU_QUERY_FIELDS):
                extra = len(parsed) - len(GPU_QUERY_FIELDS)
                merged_name = ",".join(parsed[3 : 3 + extra + 1]).strip()
                parsed = parsed[:3] + [merged_name] + parsed[3 + extra + 1 :]

            record = {
                "timestamp_unix_ms": timestamp_unix_ms,
                "source": source,
                "scrape_ok": True,
                "scrape_error": None,
                "query_timestamp": parsed[0],
                "gpu_index": parsed[1],
                "gpu_uuid": parsed[2],
                "gpu_name": parsed[3],
                "utilization_gpu_pct": _to_float(parsed[4]),
                "utilization_memory_pct": _to_float(parsed[5]),
                "memory_used_mib": _to_float(parsed[6]),
                "memory_total_mib": _to_float(parsed[7]),
                "temperature_gpu_c": _to_float(parsed[8]),
                "power_draw_w": _to_float(parsed[9]),
            }
            rows.append(record)

        if not rows:
            rows.append(_error_row(timestamp_unix_ms, source, "No GPU rows parsed"))
        return rows

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

# ---- report.py ----
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional



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

# ---- runner.py ----
import asyncio
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

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

# ---- CLI ----


def _parse_steps(value: str) -> list[int]:
    if not value.strip():
        raise argparse.ArgumentTypeError("--steps cannot be empty")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    steps: list[int] = []
    for part in parts:
        try:
            parsed = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid concurrency step '{part}'. Expected comma-separated integers."
            ) from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                f"Concurrency steps must be > 0, got {parsed}."
            )
        steps.append(parsed)
    if not steps:
        raise argparse.ArgumentTypeError("--steps cannot be empty")
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step-sweep load test tool for vLLM with streaming and telemetry."
    )

    parser.add_argument("--base-url", default="http://localhost:8801")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)

    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default=_parse_steps("1,2,4,8"),
        help="Comma-separated concurrency steps, e.g. 1,2,4,8,16",
    )
    parser.add_argument("--step-duration-s", type=int, default=120)
    parser.add_argument("--warmup-s", type=int, default=30)
    parser.add_argument("--ramp-s", type=int, default=10)
    parser.add_argument("--cooldown-s", type=int, default=10)

    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-output-tokens",
        dest="max_output_tokens_spec",
        default="128",
        help="Fixed integer or normal/lognormal distribution spec.",
    )

    parser.add_argument(
        "--input-mode",
        choices=["fixed-prompt", "prompt-file", "synthetic"],
        default="fixed-prompt",
    )
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--input-size",
        dest="input_size_spec",
        default=None,
        help="Fixed integer or normal/lognormal distribution spec.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--gpu-poll-interval-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-inflight", type=int, default=None)

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.step_duration_s <= 0:
        parser.error("--step-duration-s must be > 0")
    if args.warmup_s < 0 or args.ramp_s < 0 or args.cooldown_s < 0:
        parser.error("--warmup-s, --ramp-s, and --cooldown-s must be >= 0")
    if args.poll_interval_s <= 0 or args.gpu_poll_interval_s <= 0:
        parser.error("--poll-interval-s and --gpu-poll-interval-s must be > 0")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be > 0")
    if args.max_inflight is not None and args.max_inflight <= 0:
        parser.error("--max-inflight must be > 0 when set")

    if args.input_mode == "fixed-prompt" and not args.prompt_text:
        parser.error("--prompt-text is required when --input-mode=fixed-prompt")
    if args.input_mode == "prompt-file":
        if args.prompt_file is None:
            parser.error("--prompt-file is required when --input-mode=prompt-file")
        if not args.prompt_file.exists():
            parser.error(f"--prompt-file not found: {args.prompt_file}")
    if args.input_mode == "synthetic" and not args.input_size_spec:
        parser.error("--input-size is required when --input-mode=synthetic")


async def _run_from_args(args: argparse.Namespace) -> Path:
    config = RunConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        steps=args.steps,
        step_duration_s=args.step_duration_s,
        warmup_s=args.warmup_s,
        ramp_s=args.ramp_s,
        cooldown_s=args.cooldown_s,
        stream=bool(args.stream),
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens_spec=args.max_output_tokens_spec,
        input_mode=args.input_mode,
        prompt_text=args.prompt_text,
        prompt_file=args.prompt_file,
        input_size_spec=args.input_size_spec,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        poll_interval_s=args.poll_interval_s,
        gpu_poll_interval_s=args.gpu_poll_interval_s,
        timeout_s=args.timeout_s,
        max_inflight=args.max_inflight,
    )
    return await run_load_test(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    output_dir = asyncio.run(_run_from_args(args))
    print(f"Run complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
