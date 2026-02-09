from __future__ import annotations

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
