import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))


def _require_supported_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    if torch.version.cuda is None:
        pytest.skip("CUDA build required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80+ required")


def _q2b1_reference(x, packed, scales):
    packed_cpu = packed.detach().cpu().numpy()
    shifts = 2 * np.arange(4, dtype=np.uint8)
    codes = (packed_cpu[..., None] >> shifts) & np.uint8(3)
    trits = np.array([0, 1, -1, 0], dtype=np.float32)[codes]
    trits = trits.reshape(packed.shape[0], -1)
    scales_cpu = scales.detach().cpu().float()
    weights = torch.from_numpy(trits) * scales_cpu.repeat_interleave(128, dim=1)
    expected = x.detach().cpu() @ weights.t()
    return expected.to(device=x.device)


def test_decode_lut_cpu_fallback():
    from vllm.model_executor.layers.quantization.bonsai_decode import decode_trits

    packed = torch.tensor([0b11_10_01_00, 0b00_01_10_11], dtype=torch.uint8)
    expected = torch.tensor([[0, 1, -1, 0], [0, -1, 1, 0]], dtype=torch.int8)
    torch.testing.assert_close(decode_trits(packed), expected)


def test_decode_lut_matches_numpy():
    _require_supported_cuda()
    from prism_pq2 import q2b1_to_trits
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (1 << 20,), dtype=torch.uint8, device="cuda")
    from vllm.model_executor.layers.quantization.bonsai_decode import decode_trits
    trits = decode_trits(packed)  # (n, 4) int8 on cuda
    lut = q2b1_to_trits()
    expected = lut[packed.cpu().numpy().astype(int)]
    assert trits.dtype == torch.int8
    assert trits.shape == (packed.numel(), 4)
    np.testing.assert_array_equal(trits.cpu().numpy(), expected)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 128, 1024),
        (8, 257, 2048),
        (32, 512, 1024),
        (33, 512, 1024),
        (64, 512, 1024),
    ],
)
def test_q2b1_gemm_matches_reference(m, n, k):
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization.bonsai_decode import q2b1_gemm

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(11 + m + n + k)
        packed_cpu = torch.randint(0, 256, (n, k // 4), dtype=torch.uint8)
        packed = packed_cpu.cuda()
        scales = (
            torch.rand(n, k // 128, device="cuda", dtype=torch.float16) + 0.5
        ).contiguous()
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)

        got = q2b1_gemm(x, packed, scales, config_id=0)
        expected = _q2b1_reference(x, packed, scales)

        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(
            got.float(), expected.float(), rtol=1e-2, atol=1e-1
        )


def test_q2b1_gemm_rejects_m_above_64():
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization.bonsai_decode import q2b1_gemm

    packed = torch.zeros((1, 32), dtype=torch.uint8, device="cuda")
    scales = torch.ones((1, 1), dtype=torch.float16, device="cuda")
    x = torch.zeros((65, 128), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match=r"M must be <= 64 for q2b1_gemm"):
        q2b1_gemm(x, packed, scales, config_id=0)


def test_q2b1_gemm_lut_code_mapping():
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization.bonsai_decode import q2b1_gemm

    packed = torch.zeros((1, 32), dtype=torch.uint8, device="cuda")
    # Low-to-high slots are 00, 01, 10, 11 -> [0, +1, -1, 0].
    packed[0, 0] = 0b11_10_01_00
    scales = torch.ones((1, 1), dtype=torch.float16, device="cuda")
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0] + [0.0] * 124],
        dtype=torch.float32,
        device="cuda",
    )

    got = q2b1_gemm(x, packed, scales, config_id=0)
    assert got.dtype == torch.bfloat16
    assert got.item() == -1.0


def test_q2b1_gemm_autotune_cache_matches_reference(monkeypatch):
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization import bonsai_decode

    m, n, k = 1, 128, 1024
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(29)
        packed = torch.randint(
            0, 256, (n, k // 4), dtype=torch.uint8, device="cuda"
        )
        scales = (
            torch.rand(n, k // 128, device="cuda", dtype=torch.float16) + 0.5
        ).contiguous()
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)

    timing_calls = 0
    original_benchmark = bonsai_decode._benchmark_gemm_config
    cache = bonsai_decode._AUTOTUNE_CACHE
    previous_cache = dict(cache)

    def count_benchmark(*args, **kwargs):
        nonlocal timing_calls
        timing_calls += 1
        return original_benchmark(*args, **kwargs)

    monkeypatch.setattr(
        bonsai_decode,
        "_benchmark_gemm_config",
        count_benchmark,
        raising=True,
    )
    try:
        cache.clear()
        monkeypatch.delenv("BONSAI_AUTOTUNE", raising=False)
        first = bonsai_decode.q2b1_gemm_autotuned(x, packed, scales)
        timing_calls_after_first = timing_calls
        second = bonsai_decode.q2b1_gemm_autotuned(x, packed, scales)

        expected = _q2b1_reference(x, packed, scales)
        assert first.dtype == torch.bfloat16
        assert second.dtype == torch.bfloat16
        torch.testing.assert_close(
            first.float(), expected.float(), rtol=1e-2, atol=1e-1
        )
        torch.testing.assert_close(
            second.float(), expected.float(), rtol=1e-2, atol=1e-1
        )
        # The bounded autotuner has three candidate configurations.
        assert timing_calls_after_first == 3
        assert timing_calls == timing_calls_after_first
        assert len(cache) == 1
        selected_config = next(iter(cache.values()))
        assert selected_config in (0, 1, 2)
    finally:
        cache.clear()
        cache.update(previous_cache)


def test_q2b1_gemm_autotune_disabled(monkeypatch):
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization import bonsai_decode

    m, n, k = 1, 128, 1024
    packed = torch.zeros((n, k // 4), dtype=torch.uint8, device="cuda")
    scales = torch.ones((n, k // 128), dtype=torch.float16, device="cuda")
    x = torch.zeros((m, k), dtype=torch.float32, device="cuda")
    selected_configs = []

    def capture_config(x, packed, scales, config_id=0):
        selected_configs.append(config_id)
        return torch.empty(
            (x.shape[0], packed.shape[0]), dtype=torch.bfloat16, device=x.device
        )

    monkeypatch.setattr(
        bonsai_decode, "q2b1_gemm", capture_config, raising=True
    )

    def fail_benchmark(*args, **kwargs):
        raise AssertionError(
            "BONSAI_AUTOTUNE=0 must not benchmark candidate configurations"
        )

    monkeypatch.setattr(
        bonsai_decode,
        "_benchmark_gemm_config",
        fail_benchmark,
        raising=True,
    )
    cache = bonsai_decode._AUTOTUNE_CACHE
    previous_cache = dict(cache)
    try:
        cache.clear()
        monkeypatch.setenv("BONSAI_AUTOTUNE", "0")
        got = bonsai_decode.q2b1_gemm_autotuned(x, packed, scales)

        assert got.dtype == torch.bfloat16
        assert selected_configs == [0]
        assert len(cache) == 0
    finally:
        cache.clear()
        cache.update(previous_cache)


def test_q2b1_gemm_skips_autotune_while_compiling(monkeypatch):
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization import bonsai_decode

    m, n, k = 1, 128, 1024
    packed = torch.zeros((n, k // 4), dtype=torch.uint8, device="cuda")
    scales = torch.ones((n, k // 128), dtype=torch.float16, device="cuda")
    x = torch.zeros((m, k), dtype=torch.float32, device="cuda")
    selected_configs = []

    def capture_config(x, packed, scales, config_id=0):
        selected_configs.append(config_id)
        return torch.empty(
            (x.shape[0], packed.shape[0]), dtype=torch.bfloat16, device=x.device
        )

    monkeypatch.setattr(
        bonsai_decode, "q2b1_gemm", capture_config, raising=True
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def fail_benchmark(*args, **kwargs):
        raise AssertionError(
            "torch.compile must not benchmark candidate configurations"
        )

    monkeypatch.setattr(
        bonsai_decode,
        "_benchmark_gemm_config",
        fail_benchmark,
        raising=True,
    )

    got = bonsai_decode.q2b1_gemm_autotuned(x, packed, scales)

    assert got.dtype == torch.bfloat16
    assert selected_configs == [0]


def test_q2b1_gemm_autotuned_uses_compile_cache(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    packed = torch.zeros((7, 32), dtype=torch.uint8)
    scales = torch.ones((7, 1), dtype=torch.float16)
    x = torch.zeros((2, 128), dtype=torch.float32)
    selected_configs = []

    def capture_config(x, packed, scales, config_id=0):
        selected_configs.append(config_id)
        return torch.empty((x.shape[0], packed.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setattr(
        bonsai_decode, "_validate_q2b1_gemm_inputs", lambda *args: None
    )
    monkeypatch.setattr(bonsai_decode, "q2b1_gemm", capture_config)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    cache = bonsai_decode._COMPILE_AUTOTUNE_CACHE
    previous_cache = dict(cache)
    cache_key = (
        x.device.index,
        x.shape[0],
        packed.shape[0],
        x.shape[1],
        x.dtype,
    )
    try:
        cache.clear()
        cache[cache_key] = 2
        got = bonsai_decode.q2b1_gemm_autotuned(x, packed, scales)
    finally:
        cache.clear()
        cache.update(previous_cache)

    assert got.dtype == torch.bfloat16
    assert selected_configs == [2]


def test_prewarm_q2b1_autotune_reuses_shapes_and_honors_disable(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    packed = torch.zeros((7, 32), dtype=torch.uint8)
    scales = torch.ones((7, 1), dtype=torch.float16)
    eager_calls = []
    selected_by_m = {1: 1, 2: 2, 4: 0}

    def fake_autotuned(x, received_packed, received_scales):
        m = x.shape[0]
        eager_calls.append((m, x.shape[1], x.dtype))
        eager_key = (
            x.device.index,
            (8, 0),
            m,
            received_packed.shape[0],
            x.shape[1],
            x.dtype,
        )
        bonsai_decode._AUTOTUNE_CACHE[eager_key] = selected_by_m[m]
        return torch.empty((m, received_packed.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setattr(
        bonsai_decode, "q2b1_gemm_autotuned", fake_autotuned
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (8, 0)
    )

    eager_cache = bonsai_decode._AUTOTUNE_CACHE
    compile_cache = bonsai_decode._COMPILE_AUTOTUNE_CACHE
    previous_eager_cache = dict(eager_cache)
    previous_compile_cache = dict(compile_cache)
    try:
        eager_cache.clear()
        compile_cache.clear()
        monkeypatch.delenv("BONSAI_AUTOTUNE", raising=False)

        bonsai_decode.prewarm_q2b1_autotune(packed, scales, (1, 2, 1))
        assert eager_calls == [
            (1, 128, torch.float32),
            (2, 128, torch.float32),
        ]
        assert compile_cache[
            (None, 1, 7, 128, torch.float32)
        ] == 1
        assert compile_cache[
            (None, 2, 7, 128, torch.float32)
        ] == 2

        bonsai_decode.prewarm_q2b1_autotune(packed, scales, (1, 2, 4))
        assert eager_calls == [
            (1, 128, torch.float32),
            (2, 128, torch.float32),
            (4, 128, torch.float32),
        ]
        assert compile_cache[
            (None, 4, 7, 128, torch.float32)
        ] == 0

        monkeypatch.setenv("BONSAI_AUTOTUNE", "0")
        bonsai_decode.prewarm_q2b1_autotune(packed, scales, (8,))
        assert eager_calls == [
            (1, 128, torch.float32),
            (2, 128, torch.float32),
            (4, 128, torch.float32),
        ]
        assert (None, 8, 7, 128, torch.float32) not in compile_cache
    finally:
        eager_cache.clear()
        eager_cache.update(previous_eager_cache)
        compile_cache.clear()
        compile_cache.update(previous_compile_cache)


def test_q2b1_gemm_compile_cold_state_does_not_load_extension(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    x = torch.zeros((1, 128), dtype=torch.float32)
    packed = torch.zeros((2, 32), dtype=torch.uint8)
    scales = torch.ones((2, 1), dtype=torch.float16)

    def fail(*args, **kwargs):
        raise AssertionError("compile path touched a cold-state helper")

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(
        bonsai_decode, "_validate_q2b1_gemm_inputs", fail, raising=True
    )
    monkeypatch.setattr(bonsai_decode, "_load_ext", fail, raising=True)
    monkeypatch.setattr(torch.cuda, "device", fail, raising=True)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", fail, raising=True
    )

    with pytest.raises(
        RuntimeError,
        match="q2b1_gemm extension must be preloaded before CUDA graph capture",
    ):
        bonsai_decode.q2b1_gemm(x, packed, scales, config_id=1)


def test_decode_trits_compile_cold_state_does_not_initialize_state(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    class Packed:
        device = torch.device("cuda:1")
        is_cuda = True

        def contiguous(self):
            return self

    packed = Packed()

    def fail(*args, **kwargs):
        raise AssertionError("compile path touched a cold-state helper")

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(bonsai_decode, "_LUT_CACHE", {}, raising=True)
    monkeypatch.setattr(bonsai_decode, "_load_ext", fail, raising=True)
    monkeypatch.setattr(bonsai_decode, "_make_lut", fail, raising=True)
    monkeypatch.setattr(torch.cuda, "device", fail, raising=True)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", fail, raising=True
    )

    with pytest.raises(
        RuntimeError,
        match="q2b1_gemm extension must be preloaded before CUDA graph capture",
    ):
        bonsai_decode.decode_trits(packed)


def test_decode_trits_compile_cached_unavailable_uses_cpu_fallback(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    packed = torch.tensor(
        [0b11_10_01_00, 0b00_01_10_11], dtype=torch.uint8
    )
    expected = torch.tensor(
        [[0, 1, -1, 0], [0, -1, 1, 0]], dtype=torch.int8
    )
    unavailable = bonsai_decode.BonsaiQ2b1UnavailableError(
        "cached extension load failure"
    )
    calls = []

    def fail(name):
        def callback(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"cached failure used {name}")

        return callback

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(
        bonsai_decode, "_ext_error", unavailable, raising=True
    )
    monkeypatch.setattr(bonsai_decode, "_LUT_CACHE", {}, raising=True)
    monkeypatch.setattr(
        bonsai_decode, "_decode_lut_op", fail("custom op"), raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_load_ext", fail("extension load"), raising=True
    )

    got = bonsai_decode.decode_trits(packed)

    assert got.shape == (2, 4)
    assert got.dtype == torch.int8
    torch.testing.assert_close(got, expected)
    assert calls == []


def test_q2b1_gemm_compile_cached_unavailable_uses_fallback(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    x = torch.ones((2, 128), dtype=torch.float32)
    packed = torch.stack(
        [
            torch.full((32,), 0x55, dtype=torch.uint8),
            torch.full((32,), 0xAA, dtype=torch.uint8),
        ]
    )
    scales = torch.tensor([[0.5], [2.0]], dtype=torch.float16)
    expected = torch.tensor(
        [[64.0, -256.0], [64.0, -256.0]], dtype=torch.bfloat16
    )
    unavailable = bonsai_decode.BonsaiQ2b1UnavailableError(
        "cached extension load failure"
    )
    calls = []

    def fail(name):
        def callback(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"cached failure used {name}")

        return callback

    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(
        bonsai_decode, "_ext_error", unavailable, raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_q2b1_gemm_op", fail("custom op"), raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_load_ext", fail("extension load"), raising=True
    )

    def run(value):
        return bonsai_decode.q2b1_gemm(value, packed, scales, config_id=0)

    compiled = torch.compile(run, fullgraph=True, backend="eager")
    got = compiled(x)

    assert got.shape == (2, 2)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected)
    assert calls == []


def test_q2b1_gemm_autotuned_compile_cached_unavailable_skips_cache(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    x = torch.ones((2, 128), dtype=torch.float32)
    packed = torch.stack(
        [
            torch.full((32,), 0x55, dtype=torch.uint8),
            torch.full((32,), 0xAA, dtype=torch.uint8),
        ]
    )
    scales = torch.tensor([[0.5], [2.0]], dtype=torch.float16)
    expected = torch.tensor(
        [[64.0, -256.0], [64.0, -256.0]], dtype=torch.bfloat16
    )
    unavailable = bonsai_decode.BonsaiQ2b1UnavailableError(
        "cached extension load failure"
    )
    calls = []

    def fail(name):
        def callback(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"cached failure used {name}")

        return callback

    class ForbiddenCompileCache:
        def get(self, *args, **kwargs):
            raise AssertionError("cached extension failure read compile cache")

    monkeypatch.delenv("BONSAI_AUTOTUNE", raising=False)
    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(
        bonsai_decode, "_ext_error", unavailable, raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_COMPILE_AUTOTUNE_CACHE", ForbiddenCompileCache(),
        raising=True,
    )
    monkeypatch.setattr(
        bonsai_decode, "_q2b1_gemm_op", fail("custom op"), raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_load_ext", fail("extension load"), raising=True
    )

    def run(value):
        return bonsai_decode.q2b1_gemm_autotuned(value, packed, scales)

    compiled = torch.compile(run, fullgraph=True, backend="eager")
    got = compiled(x)

    assert got.shape == (2, 2)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected)
    assert calls == []


def test_q2b1_gemm_autotuned_compile_disables_cached_config(monkeypatch):
    from vllm.model_executor.layers.quantization import bonsai_decode

    packed = torch.zeros((7, 32), dtype=torch.uint8)
    scales = torch.ones((7, 1), dtype=torch.float16)
    x = torch.zeros((2, 128), dtype=torch.float32)
    selected_configs = []

    def capture_config(x, packed, scales, config_id=0):
        selected_configs.append(config_id)
        return torch.empty((x.shape[0], packed.shape[0]), dtype=torch.bfloat16)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        bonsai_decode, "_validate_q2b1_gemm_inputs", lambda *args: None
    )
    monkeypatch.setattr(bonsai_decode, "q2b1_gemm", capture_config)
    monkeypatch.setenv("BONSAI_AUTOTUNE", "0")

    cache = bonsai_decode._COMPILE_AUTOTUNE_CACHE
    previous_cache = dict(cache)
    cache_key = (
        x.device.index,
        x.shape[0],
        packed.shape[0],
        x.shape[1],
        x.dtype,
    )
    try:
        cache.clear()
        cache[cache_key] = 2
        got = bonsai_decode.q2b1_gemm_autotuned(x, packed, scales)
    finally:
        cache.clear()
        cache.update(previous_cache)

    assert got.dtype == torch.bfloat16
    assert selected_configs == [0]


def test_prewarm_capture_check_uses_packed_device(monkeypatch):
    from contextlib import contextmanager

    from vllm.model_executor.layers.quantization import bonsai_decode

    class Packed:
        device = torch.device("cuda:1")
        is_cuda = True
        shape = (7, 32)

    packed = Packed()
    active_devices = []

    @contextmanager
    def device_scope(device):
        active_devices.append(device)
        try:
            yield
        finally:
            active_devices.pop()

    def capture_check():
        assert active_devices == [packed.device]
        return True

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.delenv("BONSAI_AUTOTUNE", raising=False)
    monkeypatch.setattr(torch.cuda, "device", device_scope)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", capture_check
    )

    def fail_autotune(*args, **kwargs):
        raise AssertionError("prewarm must stop during capture")

    monkeypatch.setattr(
        bonsai_decode, "q2b1_gemm_autotuned", fail_autotune
    )
    bonsai_decode.prewarm_q2b1_autotune(packed, object(), (1,))

    assert active_devices == []


def test_prewarm_waits_for_inflight_shape(monkeypatch):
    from threading import Event, Thread

    from vllm.model_executor.layers.quantization import bonsai_decode

    packed = torch.zeros((7, 32), dtype=torch.uint8)
    scales = torch.ones((7, 1), dtype=torch.float16)
    first_started = Event()
    release_first = Event()
    second_done = Event()
    calls = []
    errors = []

    def fake_autotuned(x, received_packed, received_scales):
        calls.append((x.shape[0], x.shape[1]))
        first_started.set()
        release_first.wait(timeout=2)
        eager_key = (
            x.device.index,
            (8, 0),
            x.shape[0],
            received_packed.shape[0],
            x.shape[1],
            x.dtype,
        )
        bonsai_decode._AUTOTUNE_CACHE[eager_key] = 1

    def run_prewarm():
        try:
            bonsai_decode.prewarm_q2b1_autotune(packed, scales, (1,))
        except Exception as exc:
            errors.append(exc)

    def run_second_prewarm():
        try:
            bonsai_decode.prewarm_q2b1_autotune(packed, scales, (1,))
        except Exception as exc:
            errors.append(exc)
        finally:
            second_done.set()

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.delenv("BONSAI_AUTOTUNE", raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (8, 0)
    )
    monkeypatch.setattr(
        bonsai_decode, "q2b1_gemm_autotuned", fake_autotuned
    )

    eager_cache = bonsai_decode._AUTOTUNE_CACHE
    compile_cache = bonsai_decode._COMPILE_AUTOTUNE_CACHE
    inflight = bonsai_decode._PREWARM_INFLIGHT
    previous_eager_cache = dict(eager_cache)
    previous_compile_cache = dict(compile_cache)
    previous_inflight = set(inflight)
    first = Thread(target=run_prewarm)
    second = Thread(target=run_second_prewarm)
    second_started = False
    try:
        eager_cache.clear()
        compile_cache.clear()
        inflight.clear()
        first.start()
        assert first_started.wait(timeout=2)
        second.start()
        second_started = True
        assert not second_done.wait(timeout=0.1)
    finally:
        release_first.set()
        first.join(timeout=2)
        if second_started:
            second.join(timeout=2)
        eager_cache.clear()
        eager_cache.update(previous_eager_cache)
        compile_cache.clear()
        compile_cache.update(previous_compile_cache)
        inflight.clear()
        inflight.update(previous_inflight)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == [(1, 128)]


def test_decode_lut_is_compile_safe():
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization import bonsai_decode

    bonsai_decode._load_ext()
    packed = torch.tensor(
        [0b11_10_01_00, 0b00_01_10_11], dtype=torch.uint8, device="cuda"
    )
    bonsai_decode._make_lut(packed.device)
    compiled = torch.compile(
        lambda value: bonsai_decode.decode_trits(value),
        fullgraph=True,
        backend="eager",
    )

    got = compiled(packed)
    expected = torch.tensor(
        [[0, 1, -1, 0], [0, -1, 1, 0]], dtype=torch.int8, device="cuda"
    )
    torch.testing.assert_close(got, expected)


def test_q2b1_gemm_is_compile_safe():
    _require_supported_cuda()

    from vllm.model_executor.layers.quantization import bonsai_decode

    bonsai_decode._load_ext()
    packed = torch.zeros((2, 32), dtype=torch.uint8, device="cuda")
    scales = torch.ones((2, 1), dtype=torch.float16, device="cuda")
    x = torch.ones((1, 128), dtype=torch.float32, device="cuda")
    compiled = torch.compile(
        lambda value: bonsai_decode.q2b1_gemm(
            value, packed, scales, config_id=0
        ),
        fullgraph=True,
        backend="eager",
    )

    got = compiled(x)

    assert got.shape == (1, 2)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, torch.zeros_like(got))


def test_extension_load_failure_is_cached(monkeypatch):
    from torch.utils import cpp_extension

    from vllm.model_executor.layers.quantization import bonsai_decode

    attempts = 0

    def fail_load(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("nvcc unavailable")

    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(bonsai_decode, "_ext_error", None, raising=True)
    monkeypatch.setattr(cpp_extension, "load", fail_load, raising=True)

    with pytest.raises(
        bonsai_decode.BonsaiQ2b1UnavailableError,
        match="failed to load the Bonsai Q2b1 extension",
    ):
        bonsai_decode._load_ext()
    with pytest.raises(bonsai_decode.BonsaiQ2b1UnavailableError):
        bonsai_decode._load_ext()
    assert attempts == 1


@pytest.mark.parametrize(
    "m64_value, expected_m_values",
    [
        (None, (1, 2, 4, 8, 16, 32)),
        ("0", (1, 2, 4, 8, 16, 32)),
        ("1", (1, 2, 4, 8, 16, 32, 64)),
    ],
)
def test_process_weights_does_not_probe_another_device_capture_state(
    monkeypatch, m64_value, expected_m_values
):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import bonsai_ternary

    class TensorStub:
        device = torch.device("cuda:1")
        is_cuda = True
        shape = (7, 32)

        def contiguous(self):
            return self

    class LayerStub:
        def __init__(self):
            packed = TensorStub()
            scale = TensorStub()
            self.weight = SimpleNamespace(data=packed)
            self.weight_scale = SimpleNamespace(data=scale)
            self._bonsai_signs = SimpleNamespace(device=packed.device)
            self._buffers = {"_bonsai_signs": self._bonsai_signs}

        def register_parameter(self, name, value):
            setattr(self, name, value)

    layer = LayerStub()
    method = object.__new__(bonsai_ternary.BonsaiTernaryLinearMethodVLLM)
    prewarm_calls = []

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: (_ for _ in ()).throw(
            AssertionError("process path queried an unscoped CUDA stream")
        ),
    )
    monkeypatch.setattr(bonsai_ternary, "_load_ext", lambda: object())
    monkeypatch.setattr(bonsai_ternary, "_make_lut", lambda device: None)
    monkeypatch.setattr(
        bonsai_ternary,
        "prewarm_q2b1_autotune",
        lambda packed, scale, m_values: prewarm_calls.append(m_values),
    )
    monkeypatch.setenv("BONSAI_FUSED_GEMM", "1")
    if m64_value is None:
        monkeypatch.delenv("BONSAI_FUSED_GEMM_M64", raising=False)
    else:
        monkeypatch.setenv("BONSAI_FUSED_GEMM_M64", m64_value)

    method.process_weights_after_loading(layer)
    assert prewarm_calls == [expected_m_values]


def test_prewarm_failure_notifies_waiter_and_allows_retry(monkeypatch):
    from threading import Event, Thread, current_thread

    from vllm.model_executor.layers.quantization import bonsai_decode

    packed = torch.zeros((7, 32), dtype=torch.uint8)
    scales = torch.ones((7, 1), dtype=torch.float16)
    owner_started = Event()
    waiter_started = Event()
    release_owner = Event()
    waiter_done = Event()
    waiter_autotune_called = Event()
    calls = []
    owner_errors = []
    waiter_errors = []

    def fake_autotuned(x, received_packed, received_scales):
        calls.append((current_thread().name, x.shape[0], x.shape[1]))
        if current_thread().name == "waiter":
            waiter_autotune_called.set()
        if len(calls) == 1:
            owner_started.set()
            release_owner.wait(timeout=2)
            raise RuntimeError("autotune failed")

    def run_owner():
        try:
            bonsai_decode.prewarm_q2b1_autotune(packed, scales, (1,))
        except Exception as exc:
            owner_errors.append(exc)

    def run_waiter():
        waiter_started.set()
        try:
            bonsai_decode.prewarm_q2b1_autotune(packed, scales, (1,))
        except Exception as exc:
            waiter_errors.append(exc)
        finally:
            waiter_done.set()

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.delenv("BONSAI_AUTOTUNE", raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (8, 0)
    )
    monkeypatch.setattr(
        bonsai_decode, "q2b1_gemm_autotuned", fake_autotuned
    )

    eager_cache = bonsai_decode._AUTOTUNE_CACHE
    compile_cache = bonsai_decode._COMPILE_AUTOTUNE_CACHE
    inflight = bonsai_decode._PREWARM_INFLIGHT
    previous_eager_cache = dict(eager_cache)
    previous_compile_cache = dict(compile_cache)
    previous_inflight = set(inflight)
    first = Thread(target=run_owner, name="owner")
    second = Thread(target=run_waiter, name="waiter")
    first_started = False
    second_started = False
    try:
        eager_cache.clear()
        compile_cache.clear()
        inflight.clear()
        first.start()
        first_started = True
        assert owner_started.wait(timeout=2)
        second.start()
        second_started = True
        assert waiter_started.wait(timeout=2)
        assert not waiter_done.wait(timeout=0.1)
        release_owner.set()
        assert waiter_autotune_called.wait(timeout=2)
        assert waiter_done.wait(timeout=2)
        first.join(timeout=2)
        second.join(timeout=2)

        assert not first.is_alive()
        assert not second.is_alive()
        assert len(owner_errors) == 1
        assert str(owner_errors[0]) == "autotune failed"
        assert waiter_errors == []
        assert not inflight
        assert calls == [("owner", 1, 128), ("waiter", 1, 128)]

        calls_before_retry = len(calls)
        bonsai_decode.prewarm_q2b1_autotune(packed, scales, (1,))
        assert len(calls) == calls_before_retry + 1
        assert calls[-1][1:] == (1, 128)
        assert not inflight
    finally:
        release_owner.set()
        if first_started:
            first.join(timeout=2)
        if second_started:
            second.join(timeout=2)
        eager_cache.clear()
        eager_cache.update(previous_eager_cache)
        compile_cache.clear()
        compile_cache.update(previous_compile_cache)
        inflight.clear()
        inflight.update(previous_inflight)
