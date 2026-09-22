"""Q2b1 packed ternary bytes -> trits via a 256x4 LUT CUDA kernel.

Loads the LUT into shared memory once per block; each thread decodes its byte
into 4 int8 trits. Extension is compiled once via torch.utils.cpp_extension.load
and cached by torch's extension cache.
"""

import os
import threading

import torch

_ext = None
_ext_error = None
_EXT_LOCK = threading.Lock()
_LUT_CACHE = {}
_LUT_LOCK = threading.Lock()
_GEMM_CONFIGS = ((64, 2), (128, 4), (256, 8))
_AUTOTUNE_CACHE = {}
_AUTOTUNE_CONDITION = threading.Condition()
_AUTOTUNE_INFLIGHT = set()
_COMPILE_AUTOTUNE_CACHE = {}
_PREWARM_INFLIGHT = set()
_CAPTURE_ERROR = "q2b1_gemm extension must be preloaded before CUDA graph capture"


class BonsaiQ2b1UnavailableError(RuntimeError):
    """The optional Bonsai Q2b1 CUDA extension could not be loaded."""


def _is_current_stream_capturing() -> bool:
    """Avoid querying CUDA capture state while torch.compile is tracing."""
    if torch.compiler.is_compiling():
        return False
    return torch.cuda.is_current_stream_capturing()


def _compile_autotune_cache_key(x: torch.Tensor, packed: torch.Tensor):
    return (
        x.device.index,
        x.shape[0],
        packed.shape[0],
        x.shape[1],
        x.dtype,
    )


def _autotune_cache_key(x: torch.Tensor, packed: torch.Tensor):
    return (
        x.device.index,
        tuple(torch.cuda.get_device_capability(x.device)),
        x.shape[0],
        packed.shape[0],
        x.shape[1],
        x.dtype,
    )


def _load_ext():
    global _ext, _ext_error
    if _ext_error is not None:
        raise _ext_error
    if _ext is None:
        with _EXT_LOCK:
            if _ext_error is not None:
                raise _ext_error
            if _ext is None:
                src = os.path.join(os.path.dirname(__file__), "bonsai_decode.cu")
                try:
                    from torch.utils.cpp_extension import load

                    _ext = load(name="bonsai_decode_lut", sources=[src], verbose=False)
                except (ImportError, OSError, RuntimeError) as exc:
                    _ext_error = BonsaiQ2b1UnavailableError(
                        "failed to load the Bonsai Q2b1 extension"
                    )
                    raise _ext_error from exc
    return _ext


@torch.library.custom_op("bonsai::decode_lut", mutates_args=())
def _decode_lut_op(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return _load_ext().bonsai_decode_lut(packed, lut)


@_decode_lut_op.register_fake
def _(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return torch.empty((*packed.shape, 4), dtype=torch.int8, device=packed.device)


@torch.library.custom_op("bonsai::q2b1_gemm", mutates_args=())
def _q2b1_gemm_op(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    config_id: int,
) -> torch.Tensor:
    return _load_ext().bonsai_q2b1_gemm(x, packed, scales, config_id)


@_q2b1_gemm_op.register_fake
def _(
    x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, config_id: int
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], packed.shape[0]), dtype=torch.bfloat16, device=x.device
    )


def _make_lut(device):
    # 00->0, 01->+1, 10->-1, 11->0
    device = torch.device(device)
    lut = _LUT_CACHE.get(device)
    if lut is None:
        with _LUT_LOCK:
            lut = _LUT_CACHE.get(device)
            if lut is None:
                lut = torch.zeros((256, 4), dtype=torch.int8)
                for byte in range(256):
                    for b in range(4):
                        code = (byte >> (2 * b)) & 3
                        lut[byte, b] = {0: 0, 1: 1, 2: -1, 3: 0}[code]
                lut = lut.to(device)
                _LUT_CACHE[device] = lut
    return lut


def _decode_trits_fallback(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.contiguous().reshape(-1)
    codes = torch.stack([(packed >> shift) & 0x03 for shift in (0, 2, 4, 6)], dim=-1)
    lut = torch.tensor([0, 1, -1, 0], dtype=torch.int8, device=packed.device)
    return lut[codes.long()]


def decode_trits(packed: torch.Tensor) -> torch.Tensor:
    """(N,) uint8 -> (N, 4) int8 trits on the same device."""
    if isinstance(_ext_error, BonsaiQ2b1UnavailableError):
        return _decode_trits_fallback(packed)

    if torch.compiler.is_compiling():
        lut = _LUT_CACHE.get(packed.device)
        if _ext is None or lut is None:
            raise RuntimeError(_CAPTURE_ERROR)
        return _decode_lut_op(packed.contiguous(), lut)

    if not packed.is_cuda:
        packed = packed.contiguous().reshape(-1)
        lut = _make_lut(packed.device)
        return lut[packed.long()]

    with torch.cuda.device(packed.device):
        if _is_current_stream_capturing() and (
            _ext is None or packed.device not in _LUT_CACHE
        ):
            raise RuntimeError(_CAPTURE_ERROR)
        _load_ext()
        lut = _make_lut(packed.device)
        return _decode_lut_op(packed.contiguous(), lut)


def _q2b1_gemm_fallback(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    trits = _decode_trits_fallback(packed.reshape(-1)).reshape(
        packed.shape[0], packed.shape[1] * 4
    )
    weights = trits.float() * scales.float().repeat_interleave(128, dim=-1)
    return (x.float() @ weights.transpose(0, 1)).to(torch.bfloat16)


def _validate_q2b1_gemm_inputs(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    if not x.is_cuda or not packed.is_cuda or not scales.is_cuda:
        raise ValueError("x, packed, and scales must be CUDA tensors")
    if x.device != packed.device or x.device != scales.device:
        raise ValueError("x, packed, and scales must be on the same device")
    if x.ndim != 2 or packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("x, packed, and scales must be 2D tensors")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("x must have dtype float32 or bfloat16")
    if packed.dtype != torch.uint8:
        raise ValueError("packed must have dtype uint8")
    if scales.dtype != torch.float16:
        raise ValueError("scales must have dtype float16")

    m, k = x.shape
    n = packed.shape[0]
    if m > 64:
        raise ValueError("M must be <= 64 for q2b1_gemm")
    if k % 128 != 0:
        raise ValueError("K must be divisible by 128")
    if packed.shape[1] != k // 4:
        raise ValueError("packed must have shape (N, K // 4)")
    if scales.shape != (n, k // 128):
        raise ValueError("scales must have shape (N, K // 128)")
    if torch.version.cuda is None:
        raise ValueError("q2b1_gemm requires a CUDA-enabled PyTorch build")
    capability = torch.cuda.get_device_capability(x.device)
    if capability[0] < 8:
        raise ValueError(
            "q2b1_gemm requires CUDA compute capability >= 8.0 "
            f"(got {capability[0]}.{capability[1]})"
        )


def q2b1_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    config_id: int = 0,
) -> torch.Tensor:
    """Compute ``x @ W.T`` directly from packed Q2b1 weights."""
    if torch.compiler.is_compiling():
        if isinstance(_ext_error, BonsaiQ2b1UnavailableError):
            return _q2b1_gemm_fallback(x, packed, scales)
        if _ext is None:
            raise RuntimeError(_CAPTURE_ERROR)
        return _q2b1_gemm_op(
            x.contiguous(),
            packed.contiguous(),
            scales.contiguous(),
            config_id,
        )

    _validate_q2b1_gemm_inputs(x, packed, scales)

    with torch.cuda.device(x.device):
        if _ext is None and _is_current_stream_capturing():
            raise RuntimeError(_CAPTURE_ERROR)
        _load_ext()
        return _q2b1_gemm_op(
            x.contiguous(),
            packed.contiguous(),
            scales.contiguous(),
            config_id,
        )


def _benchmark_gemm_config(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    config_id: int,
) -> float:
    """Measure one GEMM configuration on the input device's current stream."""
    _validate_q2b1_gemm_inputs(x, packed, scales)
    with torch.cuda.device(x.device):
        if _is_current_stream_capturing():
            raise RuntimeError("cannot benchmark q2b1_gemm during CUDA graph capture")

        stream = torch.cuda.current_stream()
        for _ in range(2):
            q2b1_gemm(x, packed, scales, config_id=config_id)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        for start, end in zip(start_events, end_events):
            start.record(stream)
            q2b1_gemm(x, packed, scales, config_id=config_id)
            end.record(stream)
        end_events[-1].synchronize()
        timings = [
            start.elapsed_time(end) for start, end in zip(start_events, end_events)
        ]

    return float(sorted(timings)[len(timings) // 2])


def q2b1_gemm_autotuned(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run Q2b1 GEMM with a process-local shape/device autotuned configuration."""
    if torch.compiler.is_compiling():
        if isinstance(_ext_error, BonsaiQ2b1UnavailableError):
            return _q2b1_gemm_fallback(x, packed, scales)
        if os.getenv("BONSAI_AUTOTUNE") == "0":
            config_id = 0
        else:
            config_id = _COMPILE_AUTOTUNE_CACHE.get(
                _compile_autotune_cache_key(x, packed), 0
            )
        return q2b1_gemm(x, packed, scales, config_id=config_id)

    _validate_q2b1_gemm_inputs(x, packed, scales)

    with torch.cuda.device(x.device):
        if _ext_error is not None:
            raise _ext_error
        if _is_current_stream_capturing():
            if _ext is None:
                raise RuntimeError(_CAPTURE_ERROR)
            return q2b1_gemm(x, packed, scales, config_id=0)
        if os.getenv("BONSAI_AUTOTUNE") == "0":
            return q2b1_gemm(x, packed, scales, config_id=0)

        cache_key = _autotune_cache_key(x, packed)

    config_id = _AUTOTUNE_CACHE.get(cache_key)
    if config_id is None:
        owner = False
        while config_id is None:
            with _AUTOTUNE_CONDITION:
                config_id = _AUTOTUNE_CACHE.get(cache_key)
                if config_id is not None:
                    break
                if cache_key not in _AUTOTUNE_INFLIGHT:
                    _AUTOTUNE_INFLIGHT.add(cache_key)
                    owner = True
                    break
                _AUTOTUNE_CONDITION.wait()

        if owner:
            try:
                timings = [
                    _benchmark_gemm_config(x, packed, scales, candidate_id)
                    for candidate_id in range(len(_GEMM_CONFIGS))
                ]
                config_id = min(range(len(timings)), key=timings.__getitem__)
            except Exception:
                with _AUTOTUNE_CONDITION:
                    _AUTOTUNE_INFLIGHT.discard(cache_key)
                    _AUTOTUNE_CONDITION.notify_all()
                raise

            with _AUTOTUNE_CONDITION:
                _AUTOTUNE_CACHE[cache_key] = config_id
                _AUTOTUNE_INFLIGHT.discard(cache_key)
                _AUTOTUNE_CONDITION.notify_all()

    return q2b1_gemm(x, packed, scales, config_id=config_id)


def prewarm_q2b1_autotune(
    packed: torch.Tensor,
    scales: torch.Tensor,
    m_values,
) -> None:
    """Precompute Q2b1 configurations for the requested activation rows."""
    if os.getenv("BONSAI_AUTOTUNE") == "0" or torch.compiler.is_compiling():
        return
    if packed.is_cuda:
        with torch.cuda.device(packed.device):
            if torch.cuda.is_current_stream_capturing():
                return

    k = packed.shape[1] * 4
    for m in m_values:
        compile_cache_key = (packed.device.index, m, packed.shape[0], k, torch.float32)
        owner = False
        with _AUTOTUNE_CONDITION:
            while True:
                if compile_cache_key in _COMPILE_AUTOTUNE_CACHE:
                    break
                if compile_cache_key not in _PREWARM_INFLIGHT:
                    _PREWARM_INFLIGHT.add(compile_cache_key)
                    owner = True
                    break
                _AUTOTUNE_CONDITION.wait()
        if not owner:
            continue

        try:
            x = torch.zeros((m, k), device=packed.device, dtype=torch.float32)
            q2b1_gemm_autotuned(x, packed, scales)
            config_id = _AUTOTUNE_CACHE.get(_autotune_cache_key(x, packed))
            if config_id is not None:
                with _AUTOTUNE_CONDITION:
                    _COMPILE_AUTOTUNE_CACHE[compile_cache_key] = int(config_id)
        finally:
            with _AUTOTUNE_CONDITION:
                _PREWARM_INFLIGHT.discard(compile_cache_key)
                _AUTOTUNE_CONDITION.notify_all()
