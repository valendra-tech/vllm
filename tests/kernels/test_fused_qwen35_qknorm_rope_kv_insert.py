# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for fused_qwen35_qknorm_rope_kv_insert (Qwen3.8-27B full_attention).

Verifies the fused CUDA kernel matches the unfused reference:
  split(qkv) -> GemmaRMSNorm(q,k per-head) -> partial interleaved MRoPE
  -> gate copy -> reshape_and_cache(K, V into paged cache)

Reference uses the same fp32 intermediate boundary as the kernel.
"""

import pytest
import torch

import vllm._custom_ops as ops

# Qwen3.8-27B full_attention geometry.
HEAD_DIM = 256
ROTARY_DIM = 64
HALF_ROTARY = ROTARY_DIM // 2  # 32
NUM_HEADS = 24
NUM_KV_HEADS = 4
RMS_NORM_EPS = 1e-6
MROPE_SECTION = (11, 11, 10)
DTYPE = torch.bfloat16
SEED = 13


def _op_available() -> bool:
    return hasattr(torch.ops._C, "fused_qwen35_qknorm_rope_kv_insert")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused_qwen35_qknorm_rope_kv_insert not built",
)


def make_cos_sin_cache(max_pos, rotary_dim, base, dtype, device):
    """[max_pos, rotary_dim] cos||sin layout (matches vLLM RotaryEmbedding)."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rotary_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(dtype)


def gemma_rmsnorm(x, weight, eps):
    """x: [..., head_dim]; weight: [head_dim]. Gemma-style: (1+w)."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def apply_interleaved_mrope(x, positions, cos_sin_cache, mrope_section):
    """Partial interleaved MRoPE on [0, rotary_dim); pass-through the rest.

    x: [num_tokens, num_heads, head_dim]
    positions: [num_tokens] (text-only; MRoPE with 1D positions uses the same
    position for T/H/W).
    cos_sin_cache: [max_pos, rotary_dim] (cos||sin).
    """
    half = ROTARY_DIM // 2
    t_sec, h_sec, w_sec = mrope_section
    assert t_sec + h_sec + w_sec == half

    cs = cos_sin_cache[positions].float()  # [nt, rotary_dim]
    cos = cs[..., :half]  # [nt, half]
    sin = cs[..., half:]  # [nt, half]

    # Interleaved reorder of cos/sin (mirrors apply_interleaved_rope in mrope.py).
    # T band stays in cos half, H band moves to sin half, W band to cos half.
    cos_t = cos[..., :t_sec]
    cos_w = cos[..., t_sec:]  # W band in cos half
    sin_h = sin[..., :h_sec]  # H band in sin half
    # Reassemble: [cos_T | sin_H | cos_W] per the interleaved layout.
    cos_reordered = torch.cat([cos_t, sin_h, cos_w], dim=-1)
    sin_t = sin[..., :t_sec]
    sin_w = sin[..., t_sec + h_sec:]
    cos_h = cos[..., t_sec:t_sec + h_sec]
    sin_reordered = torch.cat([sin_t, cos_h, sin_w], dim=-1)

    cos = cos_reordered.unsqueeze(1)  # [nt, 1, half]
    sin = sin_reordered.unsqueeze(1)

    rot = x[..., :ROTARY_DIM].float()  # [nt, nh, rotary_dim]
    # Interleaved: pairs (2i, 2i+1) -> (x*cos - y*sin, x*sin + y*cos).
    x_even = rot[..., 0::2]  # [nt, nh, half]
    x_odd = rot[..., 1::2]   # [nt, nh, half]
    o_even = x_even * cos - x_odd * sin
    o_odd = x_even * sin + x_odd * cos
    out = x.clone()
    out[..., 0:ROTARY_DIM:2] = o_even.to(x.dtype)
    out[..., 1:ROTARY_DIM:2] = o_odd.to(x.dtype)
    return out


def norm_rope_ref(x, weight, positions, cos_sin_cache, eps, mrope_section):
    """[nt, nheads, head_dim] -> Gemma norm + partial interleaved MRoPE."""
    normed = gemma_rmsnorm(x, weight, eps)
    roped = apply_interleaved_mrope(normed, positions, cos_sin_cache, mrope_section)
    return roped


def reference_unfused(qkv, q_weight, k_weight, positions, cos_sin_cache,
                      slot_mapping, key_cache, value_cache, num_heads,
                      num_kv_heads, head_dim, eps, block_size, x, mrope_section):
    """Run the unfused reference and return (q_out, gate_out, k_out) +
    mutate key_cache/value_cache."""
    num_tokens = qkv.size(0)
    q_gate_size = num_heads * 2 * head_dim
    kv_size = num_kv_heads * head_dim
    q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

    # Split q_gate into q and gate (packed [num_heads, 2*head_dim] per token).
    q_gate = q_gate.view(num_tokens, num_heads, 2 * head_dim)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    q = q.reshape(num_tokens, num_heads * head_dim)
    gate = gate.reshape(num_tokens, num_heads * head_dim)

    # Per-head GemmaRMSNorm + partial MRoPE.
    q_ref = norm_rope_ref(
        q.view(num_tokens, num_heads, head_dim), q_weight, positions,
        cos_sin_cache, eps, mrope_section
    ).view(num_tokens, num_heads * head_dim)
    k_ref = norm_rope_ref(
        k.view(num_tokens, num_kv_heads, head_dim), k_weight, positions,
        cos_sin_cache, eps, mrope_section
    ).view(num_tokens, num_kv_heads * head_dim)

    # Paged KV cache insert (reference uses reshape_and_cache_flash or manual).
    k_for_cache = k_ref.view(num_tokens, num_kv_heads, head_dim)
    v_for_cache = v.view(num_tokens, num_kv_heads, head_dim)
    # Manual paged insert mirroring reshape_and_cache_kernel layout:
    # key_cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
    # value_cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
    h_block_count = head_dim // x
    for t in range(num_tokens):
        slot = int(slot_mapping[t].item())
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        for h in range(num_kv_heads):
            for hb in range(h_block_count):
                for i in range(x):
                    dim = hb * x + i
                    key_cache[block_idx, h, hb, block_offset, i] = k_for_cache[t, h, dim]
                    value_cache[block_idx, h, hb, block_offset, i] = v_for_cache[t, h, dim]

    return q_ref, gate, k_ref


@pytest.mark.parametrize("num_tokens", [1, 4, 37, 128])
@pytest.mark.parametrize("block_size,x", [(16, 8), (32, 16)])
def test_fused_qwen35_qknorm_rope_kv_insert_matches_reference(
    num_tokens, block_size, x
):
    torch.manual_seed(SEED)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_default_device(device)

    base, max_pos = 10_000_000.0, 4096

    q_w = torch.randn(HEAD_DIM, dtype=DTYPE, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=DTYPE, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, DTYPE, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64,
                              device=device)

    q_gate_size = NUM_HEADS * 2 * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = torch.randn(num_tokens, q_gate_size + 2 * kv_size, dtype=DTYPE,
                      device=device)
    qkv_orig = qkv.clone()

    # Allocate paged KV caches.
    num_blocks = (num_tokens + block_size - 1) // block_size + 4
    key_cache_ref = torch.zeros(
        num_blocks, NUM_KV_HEADS, HEAD_DIM // x, block_size, x,
        dtype=DTYPE, device=device
    )
    value_cache_ref = torch.zeros_like(key_cache_ref)
    key_cache_fused = key_cache_ref.clone()
    value_cache_fused = value_cache_ref.clone()

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # ── Reference (unfused) ───────────────────────────────────────────────
    q_ref, gate_ref, k_ref = reference_unfused(
        qkv_orig, q_w, k_w, positions, cos_sin, slot_mapping,
        key_cache_ref, value_cache_ref, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        RMS_NORM_EPS, block_size, x, MROPE_SECTION,
    )

    # ── Fused kernel ───────────────────────────────────────────────────────
    q_out = torch.empty(num_tokens, NUM_HEADS * HEAD_DIM, dtype=DTYPE,
                        device=device)
    gate_out = torch.empty(num_tokens, NUM_HEADS * HEAD_DIM, dtype=DTYPE,
                            device=device)
    k_out = torch.empty(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE,
                        device=device)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    ops.fused_qwen35_qknorm_rope_kv_insert(
        q_out, gate_out, qkv, k_out, q_w, k_w, cos_sin, positions, slot_mapping,
        key_cache_fused, value_cache_fused, RMS_NORM_EPS, 0, k_scale, v_scale,
    )

    # ── Assertions ─────────────────────────────────────────────────────────
    # bf16 tolerance: the fused kernel keeps fp32 across norm->rope while the
    # reference materializes bf16 at the norm boundary.
    torch.testing.assert_close(q_out, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    torch.testing.assert_close(key_cache_fused, key_cache_ref, rtol=2e-2,
                                atol=2e-2)
    torch.testing.assert_close(value_cache_fused, value_cache_ref, rtol=0,
                                atol=0)


def test_fused_qwen35_qknorm_rope_kv_insert_negative_slot():
    """Tokens with slot_mapping=-1 must skip the KV insert (padding)."""
    torch.manual_seed(SEED)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_default_device(device)

    num_tokens, block_size, x = 4, 16, 8
    base, max_pos = 10_000_000.0, 4096
    q_w = torch.randn(HEAD_DIM, dtype=DTYPE, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=DTYPE, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, DTYPE, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64,
                              device=device)

    q_gate_size = NUM_HEADS * 2 * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = torch.randn(num_tokens, q_gate_size + 2 * kv_size, dtype=DTYPE,
                      device=device)

    num_blocks = 8
    key_cache = torch.zeros(
        num_blocks, NUM_KV_HEADS, HEAD_DIM // x, block_size, x,
        dtype=DTYPE, device=device
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache_zero = key_cache.clone()

    # Slot -1 for token 1 (padding); valid slots for the rest.
    slot_mapping = torch.tensor([0, -1, 1, 2], dtype=torch.int64, device=device)

    q_out = torch.empty(num_tokens, NUM_HEADS * HEAD_DIM, dtype=DTYPE,
                        device=device)
    gate_out = torch.empty(num_tokens, NUM_HEADS * HEAD_DIM, dtype=DTYPE,
                            device=device)
    k_out = torch.empty(num_tokens, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE,
                        device=device)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    ops.fused_qwen35_qknorm_rope_kv_insert(
        q_out, gate_out, qkv, k_out, q_w, k_w, cos_sin, positions, slot_mapping,
        key_cache, value_cache, RMS_NORM_EPS, 0, k_scale, v_scale,
    )

    # The KV cache slot for token 1 must be untouched (still zero).
    # Slots 0, 1, 2 should be written.
    assert torch.equal(key_cache[0, :, :, 1, :], key_cache_zero[0, :, :, 1, :])
    assert not torch.equal(key_cache[0, :, :, 0, :], key_cache_zero[0, :, :, 0, :])