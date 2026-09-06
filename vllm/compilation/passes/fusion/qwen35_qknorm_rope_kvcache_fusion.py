# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inductor pass: fuse Qwen3.5 QK-norm + partial MRoPE + gate copy + KV-cache
insert into a single CUDA kernel.

Targets Qwen3.5-family full_attention layers with attn_output_gate=True
(e.g. Qwen3.8-27B). The unfused sequence in Qwen3NextAttention._project_qkv_gate
is:

    q_gate, k, v = split(qkv, [q_size*2, kv_size, kv_size], -1)
    q, gate = chunk(q_gate.view(nt, num_heads, 2*head_dim), 2, dim=-1)
    q = q.reshape(nt, num_heads*head_dim)
    gate = gate.reshape(nt, num_heads*head_dim)
    q = q_norm(q.view(nt, num_heads, head_dim)).view(nt, -1)
    k = k_norm(k.view(nt, num_kv_heads, head_dim)).view(nt, -1)
    q, k = rotary_emb(positions, q, k)
    # K, V -> reshape_and_cache

The fused replacement calls the custom op which dispatches to the CUDA
kernel fused_qwen35_qknorm_rope_kv_insert (registered in _custom_ops.py):

    fused_qwen35_qknorm_rope_and_unified_kv_cache_update(
        q_out, gate_out, qkv, k_out, q_weight, k_weight, positions,
        rms_norm_eps, cos_sin_cache, layer_name)

CUDA-only (sm80+). Distinct from the ROCm AITER qk_norm_rope_kvcache_fusion
because Qwen3.5 uses GemmaRMSNorm + partial interleaved MRoPE + gate, which
the AITER kernel does not cover.
"""

from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_i64

logger = init_logger(__name__)

P = ParamSpec("P")

# Qwen3.8-27B full_attention geometry (head_dim=256, rotary_dim=64).
SUPPORTED_HEAD_DIMS: tuple[int, ...] = (256,)


# ---------------------------------------------------------------------------
# Custom op: fused Qwen3.5 QK-norm + RoPE + gate + KV cache update
# ---------------------------------------------------------------------------


def fused_qwen35_qknorm_rope_and_unified_kv_cache_update_impl(
    q_out: torch.Tensor,
    gate_out: torch.Tensor,
    qkv: torch.Tensor,
    k_out: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    layer_name: str = "",
) -> torch.Tensor:
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_qwen35_qknorm_rope_kvcache_update(
            attn_layer,
            qkv,
            q_out,
            gate_out,
            k_out,
            positions,
            q_weight,
            k_weight,
            rms_norm_eps,
            cos_sin_cache,
            kv_cache,
            layer_slot_mapping,
        )
    else:
        # Profiling/dummy run: zero the outputs (consumed by attention).
        q_out.zero_()
        gate_out.zero_()
        k_out.zero_()

    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


def fused_qwen35_qknorm_rope_and_unified_kv_cache_update_fake(
    q_out: torch.Tensor,
    gate_out: torch.Tensor,
    qkv: torch.Tensor,
    k_out: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    layer_name: str = "",
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="fused_qwen35_qknorm_rope_and_unified_kv_cache_update",
    op_func=fused_qwen35_qknorm_rope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "gate_out", "k_out"],
    fake_impl=fused_qwen35_qknorm_rope_and_unified_kv_cache_update_fake,
)


FUSED_OP = (
    torch.ops.vllm.fused_qwen35_qknorm_rope_and_unified_kv_cache_update.default
)


# ---------------------------------------------------------------------------
# Pattern: Qwen3.5 QK-norm + partial MRoPE + gate copy + KV cache update
# ---------------------------------------------------------------------------


class Qwen35QkNormRopeKvCachePattern:
    """Match the unfused Qwen3NextAttention._project_qkv_gate sequence and
    replace with the fused op.

    Unfused (conceptually):
      q_gate, k, v = split(qkv, [q_size*2, kv_size, kv_size], -1)
      q_gate = q_gate.view(nt, num_heads, 2*head_dim)
      q, gate = chunk(q_gate, 2, dim=-1)
      q = q.reshape(nt, num_heads*head_dim)
      gate = gate.reshape(nt, num_heads*head_dim)
      q = q_norm(q.view(nt, num_heads, head_dim)).view(nt, -1)
      k = k_norm(k.view(nt, num_kv_heads, head_dim)).view(nt, -1)
      q, k = rotary_emb(positions, q, k, cos_sin_cache)
      unified_kv_cache_update(k, v, layer_name)

    Fused replacement:
      q_out = empty(...)
      gate_out = empty(...)
      k_out = empty(...)
      dummy = fused_qwen35_qknorm_rope_and_unified_kv_cache_update(
          q_out, gate_out, qkv, k_out, positions, q_weight, k_weight,
          eps, cos_sin_cache, layer_name)
      v = split(qkv, ...)[2]
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        rope_flashinfer: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.q_gate_size = num_heads * 2 * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_flashinfer=rope_flashinfer,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        # qkv layout: [q_gate | k | v] = [num_heads*2*head_dim | kv_size | kv_size]
        qkv = empty_bf16(T, self.q_gate_size + 2 * self.kv_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        cos_sin_cache = empty_bf16(L, self.head_dim)
        return [qkv, positions, q_weight, k_weight, cos_sin_cache]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # Split qkv -> q_gate, k, v
            try:
                q_gate, k, v = qkv.split(
                    [self.q_gate_size, self.kv_size, self.kv_size], dim=-1
                )
            except ValueError as e:
                raise RuntimeError from e

            # Split q_gate -> q, gate
            qg = q_gate.view(*q_gate.shape[:-1], self.num_heads, 2 * self.head_dim)
            q_h, gate_h = torch.chunk(qg, 2, dim=-1)
            q = q_h.reshape(*q_gate.shape[:-1], self.q_size)
            gate = gate_h.reshape(*q_gate.shape[:-1], self.q_size)

            # Q path: view -> RMS -> view back
            q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            q_normed = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
            q_flat = q_normed.view(q.shape)

            # K path: view -> RMS -> view back
            k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
            k_normed = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
            k_flat = k_normed.view(k.shape)

            # RoPE: apply to flattened q/k
            q_rope, k_rope = self.rope_matcher(
                positions, q_flat, k_flat, cos_sin_cache
            )
            return q_rope, gate, k_rope, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            T = qkv.size(0)
            q_out = torch.empty(
                T, self.q_size, dtype=qkv.dtype, device=qkv.device
            )
            gate_out = torch.empty(
                T, self.q_size, dtype=qkv.dtype, device=qkv.device
            )
            k_out = torch.empty(
                T, self.kv_size, dtype=qkv.dtype, device=qkv.device
            )
            auto_functionalized(
                FUSED_OP,
                q_out=q_out,
                gate_out=gate_out,
                qkv=qkv,
                k_out=k_out,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                rms_norm_eps=self.eps,
                cos_sin_cache=cos_sin_cache,
                layer_name="",
            )
            # V is split from qkv (untouched by the fused op).
            v = qkv.split([self.q_gate_size, self.kv_size, self.kv_size], dim=-1)[2]
            return q_out, gate_out, k_out, v

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


class Qwen35QkNormRopeKvCacheFusionPass(VllmPatternMatcherPass):
    """Fuse Qwen3.5 Q/K GemmaRMSNorm + partial MRoPE + gate copy + KV-cache
    insert into a single CUDA kernel when the custom op is available and the
    attention backend exposes do_qwen35_qknorm_rope_kvcache_update.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qwen35_qknorm_rope_kvcache_fusion_pass"
        )
        self._attention_geometries: tuple[tuple[int, int, int], ...] = ()

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "Qwen3.5 QK-Norm+RoPE+KVCache fusion not enabled: unsupported "
                "dtype %s",
                dtype,
            )
            return

        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            config, Attention
        )
        if len(attn_layers) == 0:
            logger.warning_once(
                "Qwen3.5 QK-Norm+RoPE+KVCache fusion enabled, but no Attention "
                "layers were discovered."
            )
            return

        for layer in attn_layers.values():
            if layer.head_size not in SUPPORTED_HEAD_DIMS:
                logger.warning_once(
                    "Qwen3.5 QK-Norm+RoPE+KVCache fusion not enabled: "
                    "layer head_size=%d is not supported (supported: %s).",
                    layer.head_size,
                    SUPPORTED_HEAD_DIMS,
                )
                return

        self._attention_geometries = tuple(
            sorted(
                {
                    (layer.head_size, layer.num_heads, layer.num_kv_heads)
                    for layer in attn_layers.values()
                }
            )
        )

        rope_flashinfer_options = (
            [False, True] if RotaryEmbedding.enabled() else [False]
        )
        for head_dim, num_heads, num_kv_heads in self._attention_geometries:
            for epsilon in [1e-5, 1e-6]:
                for neox in [True, False]:
                    for rope_flashinfer in rope_flashinfer_options:
                        Qwen35QkNormRopeKvCachePattern(
                            head_dim=head_dim,
                            num_heads=num_heads,
                            num_kv_heads=num_kv_heads,
                            eps=epsilon,
                            is_neox=neox,
                            rope_flashinfer=rope_flashinfer,
                        ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused Qwen3.5 QK-Norm+RoPE+KVCache on %s sites",
                     self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self, Qwen35QkNormRopeKvCachePattern, repr(self._attention_geometries)
        )