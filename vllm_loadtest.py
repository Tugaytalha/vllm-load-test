from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from runner import RunConfig, run_load_test


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
