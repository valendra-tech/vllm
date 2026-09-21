# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import SimpleNamespace

import pytest

from benchmarks import benchmark_bonsai_generation as benchmark


def _set_args(monkeypatch, *args):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_bonsai_generation.py",
            "--model",
            "model",
            "--mode",
            "fused",
            *args,
        ],
    )


def test_parse_args_accepts_batch_controls_and_defaults_to_compile_mode(monkeypatch):
    _set_args(monkeypatch, "--batch-size", "3", "--repetitions", "4")

    args = benchmark.parse_args()

    assert args.batch_size == 3
    assert args.repetitions == 4
    assert args.enforce_eager is False
    assert not hasattr(args, "warmup")


def test_parse_args_rejects_warmup_toggle(monkeypatch):
    _set_args(monkeypatch, "--no-warmup")

    with pytest.raises(SystemExit) as exc_info:
        benchmark.parse_args()

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--enforce-eager", True), ("--no-enforce-eager", False)],
)
def test_parse_args_accepts_both_eager_mode_flags(monkeypatch, flag, expected):
    _set_args(monkeypatch, flag)

    assert benchmark.parse_args().enforce_eager is expected


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--batch-size", "0"),
        ("--repetitions", "-2"),
        ("--batch-size", "not-an-int"),
        ("--repetitions", "bad"),
    ],
)
def test_parse_args_rejects_invalid_batch_controls(monkeypatch, capsys, option, value):
    _set_args(monkeypatch, option, value)

    with pytest.raises(SystemExit) as exc_info:
        benchmark.parse_args()

    assert exc_info.value.code == 2
    assert "positive" in capsys.readouterr().err


def test_peak_memory_allocated_mb_converts_torch_peak_bytes_to_mb():
    assert hasattr(benchmark, "peak_memory_allocated_mb")
    fake_torch_cuda = SimpleNamespace(
        is_available=lambda: True,
        max_memory_allocated=lambda: 3 * 1024 * 1024 + 512 * 1024,
    )

    assert benchmark.peak_memory_allocated_mb(fake_torch_cuda) == pytest.approx(3.5)


def test_peak_memory_allocated_mb_returns_unavailable_when_cuda_is_unusable():
    assert hasattr(benchmark, "peak_memory_allocated_mb")
    fake_torch_cuda = SimpleNamespace(is_available=lambda: False)

    assert benchmark.peak_memory_allocated_mb(fake_torch_cuda) is None


def test_peak_memory_allocated_mb_returns_unavailable_when_peak_query_fails():
    assert hasattr(benchmark, "peak_memory_allocated_mb")

    def max_memory_allocated():
        raise RuntimeError("CUDA unavailable")

    fake_torch_cuda = SimpleNamespace(
        is_available=lambda: True,
        max_memory_allocated=max_memory_allocated,
    )

    assert benchmark.peak_memory_allocated_mb(fake_torch_cuda) is None


def test_summarize_batch_stats_uses_actual_aggregate_tokens():
    summary = benchmark.summarize_batch_stats(
        generated_token_counts=[7, 11],
        batch_seconds=[1.0, 3.0],
        batch_size=2,
    )

    assert summary["generated_tokens"] == 18
    assert summary["elapsed_seconds"] == pytest.approx(4.0)
    assert summary["aggregate_generated_tokens_per_second"] == pytest.approx(4.5)
    assert summary["per_request_generated_tokens_per_second"] == pytest.approx(2.25)


def test_summarize_batch_stats_reports_singleton_p50_and_p95():
    summary = benchmark.summarize_batch_stats(
        generated_token_counts=[5],
        batch_seconds=[2.5],
        batch_size=1,
    )

    assert summary["p50_batch_seconds"] == pytest.approx(2.5)
    assert summary["p95_batch_seconds"] == pytest.approx(2.5)


def test_summarize_batch_stats_rejects_empty_or_mismatched_batches():
    for generated_token_counts, batch_seconds in (
        ([], []),
        ([], [1.0]),
        ([1], [1.0, 2.0]),
        ([1, 2], [1.0]),
    ):
        with pytest.raises(ValueError, match="equal nonzero length"):
            benchmark.summarize_batch_stats(
                generated_token_counts=generated_token_counts,
                batch_seconds=batch_seconds,
                batch_size=1,
            )


def test_measure_batches_aggregates_completion_tokens_excluding_warmup():
    def request_output(token_ids):
        return SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=token_ids, text="output")]
        )

    warmup_outputs = [request_output([99]), request_output([100])]
    measured_batches = [
        [request_output([1, 2]), request_output([3, 4])],
        [request_output([5, 6, 7]), request_output([8, 9, 10])],
    ]
    outputs = iter([warmup_outputs, *measured_batches])

    def generate():
        return next(outputs)

    generate()
    clock_values = iter([10.0, 11.0, 20.0, 22.0])

    assert hasattr(benchmark, "measure_batches")
    token_counts, batch_seconds, last_outputs = benchmark.measure_batches(
        generate=generate,
        repetitions=2,
        clock=lambda: next(clock_values),
    )

    assert token_counts == [4, 6]
    assert batch_seconds == [1.0, 2.0]
    assert last_outputs is measured_batches[-1]
