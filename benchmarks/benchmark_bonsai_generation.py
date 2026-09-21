# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Bonsai generation with and without the fused GEMM path."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def peak_memory_allocated_mb(cuda: object) -> float | None:
    try:
        if not getattr(cuda, "is_available")():
            return None
        return getattr(cuda, "max_memory_allocated")() / (1024 * 1024)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def read_gpu_peak_memory_mb() -> float | None:
    try:
        import torch

        return peak_memory_allocated_mb(torch.cuda)
    except (AttributeError, ImportError, OSError, RuntimeError):
        return None


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentiles require at least one value")

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight


def count_generated_tokens(outputs: Sequence[object]) -> int:
    return sum(
        len(sample.token_ids)
        for request_output in outputs
        for sample in request_output.outputs
    )


def measure_batches(
    generate: Callable[[], Sequence[object]],
    repetitions: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[int], list[float], Sequence[object]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    batch_seconds: list[float] = []
    generated_token_counts: list[int] = []
    measured_outputs: Sequence[object] = ()
    for _ in range(repetitions):
        generation_start = clock()
        measured_outputs = generate()
        batch_seconds.append(clock() - generation_start)
        generated_token_counts.append(count_generated_tokens(measured_outputs))
    return generated_token_counts, batch_seconds, measured_outputs


def summarize_batch_stats(
    generated_token_counts: Sequence[int],
    batch_seconds: Sequence[float],
    batch_size: int,
) -> dict[str, int | float]:
    if not generated_token_counts or len(generated_token_counts) != len(batch_seconds):
        raise ValueError(
            "token counts and batch timings must have equal nonzero length"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    generated_tokens = sum(generated_token_counts)
    elapsed_seconds = sum(batch_seconds)
    aggregate_generated_tokens_per_second = (
        generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    per_request_generated_tokens_per_second = (
        aggregate_generated_tokens_per_second / batch_size
    )
    return {
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed_seconds,
        "aggregate_generated_tokens_per_second": (
            aggregate_generated_tokens_per_second
        ),
        "generated_tokens_per_second": per_request_generated_tokens_per_second,
        "per_request_generated_tokens_per_second": (
            per_request_generated_tokens_per_second
        ),
        "p50_batch_seconds": _percentile(batch_seconds, 0.50),
        "p95_batch_seconds": _percentile(batch_seconds, 0.95),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model path or Hugging Face ID.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("baseline", "fused"),
        help="Select the Bonsai GEMM implementation.",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt to generate from.",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Number of identical prompts in each measured batch.",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=5,
        help="Number of full batches to measure.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable torch.compile and CUDA graph execution when enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    if args.mode == "baseline":
        os.environ["BONSAI_FUSED_GEMM"] = "0"
    else:
        os.environ.pop("BONSAI_FUSED_GEMM", None)

    from vllm import LLM, SamplingParams

    prompts = [args.prompt] * args.batch_size
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0, ignore_eos=True
    )

    load_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.batch_size,
        max_model_len=2048,
        quantization="bonsai_ternary",
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
    )
    load_seconds = time.perf_counter() - load_start

    warmup_start = time.perf_counter()
    llm.generate(prompts, sampling_params)
    warmup_seconds = time.perf_counter() - warmup_start

    generated_token_counts, batch_seconds, measured_outputs = measure_batches(
        generate=lambda: llm.generate(prompts, sampling_params),
        repetitions=args.repetitions,
    )
    gpu_peak_memory_mb = read_gpu_peak_memory_mb()

    summary = summarize_batch_stats(
        generated_token_counts=generated_token_counts,
        batch_seconds=batch_seconds,
        batch_size=args.batch_size,
    )
    output = measured_outputs[0]
    prompt_tokens = len(output.prompt_token_ids)
    generated_text = output.outputs[0].text
    generated_tokens = summary["generated_tokens"]
    elapsed_seconds = summary["elapsed_seconds"]
    aggregate_generated_tokens_per_second = summary[
        "aggregate_generated_tokens_per_second"
    ]
    per_request_generated_tokens_per_second = summary[
        "per_request_generated_tokens_per_second"
    ]
    generated_tokens_per_second = summary["generated_tokens_per_second"]
    p50_batch_seconds = summary["p50_batch_seconds"]
    p95_batch_seconds = summary["p95_batch_seconds"]
    gpu_memory_report = (
        str(gpu_peak_memory_mb) if gpu_peak_memory_mb is not None else "unavailable"
    )
    total_seconds = time.perf_counter() - total_start

    print(f"mode={args.mode}")
    print(f"enforce_eager={args.enforce_eager}")
    print(f"batch_size={args.batch_size}")
    print(f"repetitions={args.repetitions}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"generated_tokens={generated_tokens}")
    print(f"generated_text={generated_text!r}")
    print(f"elapsed_seconds={elapsed_seconds:.6f}")
    print(
        "aggregate_generated_tokens_per_second="
        f"{aggregate_generated_tokens_per_second:.6f}"
    )
    print(
        "per_request_generated_tokens_per_second="
        f"{per_request_generated_tokens_per_second:.6f}"
    )
    print(f"generated_tokens_per_second={generated_tokens_per_second:.6f}")
    print(f"p50_batch_seconds={p50_batch_seconds:.6f}")
    print(f"p95_batch_seconds={p95_batch_seconds:.6f}")
    print(f"gpu_memory_used_mb={gpu_memory_report}")
    print(f"load_seconds={load_seconds:.6f}")
    print(f"warmup_seconds={warmup_seconds:.6f}")
    print(f"total_seconds={total_seconds:.6f}")


if __name__ == "__main__":
    main()
