"""Fused signs + fast Walsh-Hadamard transform for block=1024 (Bonsai rotation).

Math per row of 1024:  y = H @ (signs * x), where H is the normalized
(1/sqrt(1024)) Sylvester WHT matrix.  Implementation applies the signs
(1/sqrt(1024) folded in) and then runs 10 FWHT butterfly stages over
strides h = 1, 2, 4, ..., 512.

Two implementations are provided:

- ``fwht_signs``: fully vectorized torch path (default; also kept as
  ``_fwht_signs_torch`` reference fallback).  Each butterfly stage is
  expressed as a strided (M, n/(2h), 2, h) view so no python loop over
  elements is needed.
- ``fwht_signs_triton``: single Triton kernel launch, fully
  register-resident: the whole 1024-element row is kept as one tl
  vector and the 10 butterfly stages are done with ``tl.gather`` on an
  XOR-partner index pattern (no global ping-pong, no per-stage HBM
  round trips, no scratch buffers).  Signs are folded into the initial
  load; the 1/sqrt(1024) = 1/32 normalization is applied on the final
  store.
"""

import torch

# Triton is optional at import time; the torch path is always available.
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _fwht_signs_kernel(
        x_ptr,  # (M, 1024) input, any float dtype
        signs_ptr,  # (1024,) float32 (+1 or -1)
        y_ptr,  # (M, 1024) float32 output
        BLOCK: tl.constexpr,  # 1024
        NUM_STAGES: tl.constexpr,  # 10
    ):
        """One program per row: register-resident FWHT via tl.gather.

        Stage s operates on stride h = 2**s (h = 1, 2, ..., 512).  For
        indices i with the bit cleared (bit = (i // h) % 2 == 0):
        a' = a + b;  the partner index i + h gets b' = a - b (note:
        b - a would be the wrong convention).  Standard decimated FWHT
        recurrence, executed in registers with an XOR-partner gather.
        """
        pid = tl.program_id(0)
        idx = tl.arange(0, BLOCK)

        # Load input with signs folded in.
        v = tl.load(x_ptr + pid * BLOCK + idx).to(tl.float32)
        s = tl.load(signs_ptr + idx)
        v = v * s

        # 10 in-register butterfly stages via XOR-partner gather.
        for stage in range(NUM_STAGES):
            h = 1 << stage
            bit0 = (idx // h) % 2 == 0
            partner = tl.where(bit0, idx + h, idx - h)
            p = tl.gather(v, partner, axis=0)
            v = tl.where(bit0, v + p, p - v)

        # Apply the 1/sqrt(BLOCK) scale on the final store.
        tl.store(y_ptr + pid * BLOCK + idx, v * (1.0 / 32.0))

    def fwht_signs_triton(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Y = H @ (signs * x) via a single Triton kernel launch.

        x: (..., 1024) float32/bf16 on cuda.  signs: (1024,) float (+/-1).
        Returns (..., 1024) float32.
        """
        assert x.shape[-1] == 1024, "only block=1024 is supported"
        M = x.numel() // 1024
        # Rotation blocks are often slices of a wider activation matrix. The
        # Triton pointer arithmetic assumes a packed row stride.
        x2 = x.reshape(M, 1024).contiguous()
        y = torch.empty(M, 1024, dtype=torch.float32, device=x.device)
        _fwht_signs_kernel[(M,)](
            x2,
            signs.float().contiguous(),
            y,
            BLOCK=1024,
            NUM_STAGES=10,
            num_warps=4,
        )
        return y.view(x.shape)


def _fwht_signs_torch(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Reference fallback: fully vectorized torch FWHT with folded signs.

    x: (..., 1024) float32/bf16 (any device).  signs: (1024,) float.
    Returns y = H @ (signs * x) as float32 with x's shape.
    """
    original_shape = x.shape
    n = original_shape[-1]
    assert n == 1024, "only block=1024 is supported"
    M = x.numel() // n
    x = x.reshape(M, n).float() * signs.to(x.device).float()
    h = 1
    while h < n:
        x = x.view(M, n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.empty_like(x)
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        h *= 2
    return (x.view(M, n) * (1.0 / (n**0.5))).reshape(original_shape)


def fwht_signs(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Y = H @ (signs * x) for block=1024, float32 output.

    x: (..., 1024) float32/bf16 cuda (works on cpu too).  signs: (1024,).
    """
    if x.is_cuda and _HAS_TRITON:
        return fwht_signs_triton(x, signs)
    return _fwht_signs_torch(x, signs)


def fwht_signs_quant_fp8(
    x: torch.Tensor, signs: torch.Tensor, group: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """FWHT with signs followed by per-group amax fp8 (e4m3) quantization.

    x: (..., 1024); signs: (1024,).  Returns (q, amax) where q is
    float8_e4m3fn of x's shape and amax is float32 of shape
    (..., n/group) holding each group's absolute max (dequant:
    q.float() * amax).
    """
    y = fwht_signs(x, signs)
    n = y.shape[-1]
    M = y.numel() // n
    assert n % group == 0, "last dim must be divisible by group"
    groups = y.reshape(M, n // group, group)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-12)
    q = (groups / amax.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.reshape(y.shape), amax.reshape(*y.shape[:-1], n // group)


if __name__ == "__main__":
    # Micro-benchmark: 4096 rows x 1024, bf16 input, effective GB/s.
    assert torch.cuda.is_available(), "cuda required for benchmark"
    torch.manual_seed(0)
    x = torch.randn(4096, 1024, dtype=torch.bfloat16, device="cuda")
    s = torch.tensor([-1.0, 1.0] * 512, device="cuda")

    def _bench(fn, iters=200):
        for _ in range(20):
            fn(x, s)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn(x, s)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / iters
        bytes_moved = 4096 * 1024 * 2 + 4096 * 1024 * 4  # bf16 read + fp32 write
        return ms, bytes_moved / (ms * 1e-3) / 1e9

    ms, gbps = _bench(fwht_signs)
    print(f"torch fwht_signs:      {ms:8.3f} ms  {gbps:7.1f} GB/s")
    ms, gbps = _bench(_fwht_signs_torch)
    print(f"torch _fwht_signs_torch: {ms:6.3f} ms  {gbps:7.1f} GB/s")
    if _HAS_TRITON:
        ms, gbps = _bench(fwht_signs_triton)
        print(f"triton in-register:    {ms:8.3f} ms  {gbps:7.1f} GB/s")
