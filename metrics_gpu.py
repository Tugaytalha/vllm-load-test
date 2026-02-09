from __future__ import annotations

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
