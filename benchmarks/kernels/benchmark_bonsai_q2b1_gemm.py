# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the baseline and fused Bonsai Q2b1 GEMM paths."""

from __future__ import annotations

import argparse
import math
import os
import statistics
from collections.abc import Callable
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BASELINE_ROW_CHUNK = 16384
_BF16_ATOL = 0.1
_BF16_RTOL = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="Number of activation rows (must be in [1, 64]).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5120,
        help="Number of output rows.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5120,
        help="Number of input columns (must be divisible by 128).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of steady-state timing iterations.",
    )
    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fused-kernel autotuning (use --no-autotune to disable).",
    )
    return parser.parse_args()


def validate_dimensions(args: argparse.Namespace) -> None:
    if args.m <= 0:
        raise ValueError("--m must be positive")
    if args.m > 64:
        raise ValueError("--m must be <= 64 for q2b1_gemm")
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.k % 128 != 0:
        raise ValueError("--k must be divisible by 128")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")


def require_supported_cuda():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to check CUDA availability"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA device is available")
    if torch.version.cuda is None:
        raise RuntimeError("a CUDA-enabled PyTorch build is required")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        raise RuntimeError(
            "SM80 or newer is required for the Bonsai Q2b1 GEMM "
            f"(found SM{capability[0]}{capability[1]})"
        )
    return device


def baseline_gemm(x, packed, scales, decode_trits):
    n = packed.shape[0]
    k = packed.shape[1] * 4
    output = x.new_empty((x.shape[0], n))
    for row_start in range(0, n, _BASELINE_ROW_CHUNK):
        row_end = min(row_start + _BASELINE_ROW_CHUNK, n)
        trits = decode_trits(packed[row_start:row_end].reshape(-1)).reshape(
            row_end - row_start, k
        )
        grouped_scales = scales[row_start:row_end].float().repeat_interleave(
            128, dim=1
        )
        weights = trits.float()
        weights.mul_(grouped_scales)
        output[:, row_start:row_end] = x @ weights.t()
    return output


def measure_first_call(
    fn: Callable[[], object], device
) -> tuple[float, object]:
    import torch

    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record(stream)
    output = fn()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end), output


def measure_steady_state(
    fn: Callable[[], object], iters: int, device
) -> tuple[float, float]:
    import torch

    stream = torch.cuda.current_stream(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    torch.cuda.synchronize(device)
    for start, end in zip(starts, ends):
        start.record(stream)
        fn()
        end.record(stream)
    ends[-1].synchronize()

    timings_ms = [
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ]
    return statistics.median(timings_ms), statistics.fmean(timings_ms)


def main() -> None:
    args = parse_args()
    try:
        validate_dimensions(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    if args.autotune:
        os.environ.pop("BONSAI_AUTOTUNE", None)
    else:
        os.environ["BONSAI_AUTOTUNE"] = "0"

    try:
        import torch
    except ModuleNotFoundError:
        raise SystemExit(
            "error: PyTorch is required to run the CUDA benchmark"
        ) from None

    try:
        device = require_supported_cuda()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None

    from vllm.model_executor.layers.quantization import bonsai_decode

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    generator = torch.Generator(device=device).manual_seed(0)
    packed = torch.randint(
        0,
        256,
        (args.n, args.k // 4),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    scales = torch.rand(
        (args.n, args.k // 128),
        dtype=torch.float16,
        device=device,
        generator=generator,
    ) + 0.5
    x = torch.randn(
        (args.m, args.k),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    def run_baseline():
        return baseline_gemm(x, packed, scales, bonsai_decode.decode_trits)

    def run_fused():
        return bonsai_decode.q2b1_gemm_autotuned(x, packed, scales)

    extension_first_ms = None
    if args.autotune:
        # Load the extension before measuring autotune so JIT compilation does
        # not get conflated with the configuration-selection cost.
        extension_first_ms, _ = measure_first_call(
            lambda: bonsai_decode.q2b1_gemm(
                x, packed, scales, config_id=0
            ),
            device,
        )
        bonsai_decode._AUTOTUNE_CACHE.clear()

    warmup_iters = min(5, args.iters)
    for _ in range(warmup_iters):
        run_baseline()
    torch.cuda.synchronize(device)

    first_fused_ms, first_fused_output = measure_first_call(run_fused, device)
    for _ in range(warmup_iters):
        run_fused()
    torch.cuda.synchronize(device)

    baseline_ms, baseline_avg_ms = measure_steady_state(
        run_baseline, args.iters, device
    )
    fused_ms, fused_avg_ms = measure_steady_state(
        run_fused, args.iters, device
    )
    selected_config = (
        next(iter(bonsai_decode._AUTOTUNE_CACHE.values()))
        if args.autotune
        else 0
    )

    baseline_output = run_baseline()
    torch.cuda.synchronize(device)
    error = (first_fused_output.float() - baseline_output).abs()
    allowed_error = _BF16_ATOL + _BF16_RTOL * baseline_output.abs()
    max_error = error.max().item()
    max_error_ratio = (error / allowed_error).max().item()

    output_count = args.m * args.n
    baseline_outputs_per_sec = output_count / (baseline_ms * 1e-3)
    fused_outputs_per_sec = output_count / (fused_ms * 1e-3)
    capability = torch.cuda.get_device_capability(device)
    machine = torch.cuda.get_device_name(device)
    print(
        f"machine={machine} device={device} "
        f"capability=SM{capability[0]}{capability[1]} "
        f"cuda={torch.version.cuda}"
    )
    print(
        f"shape=(m={args.m}, n={args.n}, k={args.k}) "
        f"iters={args.iters} autotune={args.autotune}"
    )
    print(f"baseline_ms={baseline_ms:.6f} baseline_avg_ms={baseline_avg_ms:.6f}")
    print(f"fused_ms={fused_ms:.6f} fused_avg_ms={fused_avg_ms:.6f}")
    print(f"speedup={baseline_ms / fused_ms:.3f}x")
    print(f"fused_first_call_ms={first_fused_ms:.6f}")
    if extension_first_ms is not None:
        print(f"extension_first_call_ms={extension_first_ms:.6f}")
    print(f"selected_config_id={selected_config}")
    print(
        f"steady_output_count={output_count} "
        f"baseline_effective_outputs_per_sec={baseline_outputs_per_sec:.3f} "
        f"fused_effective_outputs_per_sec={fused_outputs_per_sec:.3f}"
    )
    print(f"max_numerical_error={max_error:.6g}")
    print(f"max_error_ratio={max_error_ratio:.6g}")
    if not math.isfinite(max_error) or not math.isfinite(max_error_ratio):
        raise SystemExit(
            f"error: numerical error is non-finite (max={max_error!r}, "
            f"ratio={max_error_ratio!r}); "
            "benchmark result is invalid"
        )
    if max_error_ratio > 1.0:
        raise SystemExit(
            f"error: max_error_ratio={max_error_ratio:.6g} exceeds "
            f"the BF16 tolerance (atol={_BF16_ATOL}, rtol={_BF16_RTOL})"
        )


if __name__ == "__main__":
    main()
