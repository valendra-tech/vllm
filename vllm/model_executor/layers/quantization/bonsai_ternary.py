"""Bonsai ternary (Q2b1) Linear method with fp8 / int4 / int8 targets.

Task 7 first-pass correctness implementation.  Weights arrive as the packed
safetensors layout (Q2b1 uint8 + per-(out, group) fp16 scales):

- ``packed``: uint8 (N, K//4).  Per output row, per 128-input group: 32
  bytes, weight j (within group) -> byte j//4, bits (j%4)*2 LSB-first;
  code 00->0, 01->+1, 10->-1, 11->0.
- ``scale``: fp16 (N, K//128), one scale per (output row, input group).

After loading we decode trits with the CUDA LUT kernel and keep resident
weights per target:

- "fp8":  (N, K) float8_e4m3fn with fp32 per-128-group amax scales.  apply()
  dequantizes to bf16 once (cached) and uses plain torch.matmul.  A
  grouped-scale fp8 GEMM integration lands in Task 9; this path exists to
  lock down numerics.
- "int4"/"int8": exact int8 trits + fp32 grouped scales, dequantized to
  bf16 for a W4A16-style dequant matmul.  Marlin int4 packing lands in
  Task 9.
- "nvfp4": NotImplementedError for now.

Hadamard rotation: y = (x*signs @ H^T) @ W^T where H is the normalized
1024-block WHT and signs is a per-input-position (+/-1) vector.  For K > 1024
the signs+FWHT is applied per 1024-block (each block of x uses its own slice
of signs), matching the block-diagonal rotation of the checkpoint format.
"""
from typing import Dict, List, Optional

import torch

from .bonsai_decode import decode_trits
from .hadamard_fwht import fwht_signs, fwht_signs_quant_fp8

_HADAMARD_BLOCK = 1024
_SCALE_GROUP = 128


class BonsaiTernaryConfig:
    """Target selection for ternary Linear layers."""

    def __init__(self, ternary_target: str = "auto"):
        self.ternary_target = ternary_target

    def resolve(self) -> str:
        if self.ternary_target != "auto":
            return self.ternary_target
        if torch.cuda.get_device_capability()[0] >= 12:
            return "nvfp4"
        return "int4"


class BonsaiTernaryLinearMethod:
    """Hadamard-rotated ternary Linear: y = (x*signs @ H^T) @ W^T."""

    def __init__(self, config: BonsaiTernaryConfig, in_features: int,
                 out_features: int):
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self._target: Optional[str] = None
        self.signs: Optional[torch.Tensor] = None
        self.w_dq_bf16: Optional[torch.Tensor] = None
        self.w_target: Optional[torch.Tensor] = None
        self.w_scales: Optional[torch.Tensor] = None
        self._n_weight_bytes = 0

    @property
    def target(self) -> str:
        assert self._target is not None, (
            "process_weights_after_loading has not been called")
        return self._target

    def weight_bytes(self) -> int:
        """Resident weight tensor bytes (for memory reporting)."""
        return self._n_weight_bytes

    def process_weights_after_loading(
        self,
        weights: Dict[str, torch.Tensor],
        signs: torch.Tensor,
        device,
    ) -> None:
        target = self.config.resolve()
        if target == "nvfp4":
            raise NotImplementedError(
                "nvfp4 ternary target lands in a later task")
        K, N = self.in_features, self.out_features
        assert K % _HADAMARD_BLOCK == 0, (
            "in_features must be a multiple of the 1024 Hadamard block")
        assert K % _SCALE_GROUP == 0, "in_features must be divisible by 128"

        packed = weights["packed"].to(device)
        scale = weights["scale"].to(device)
        assert packed.shape == (N, K // 4) and packed.dtype == torch.uint8, (
            f"packed weights must be uint8 (N, K//4), got {packed.shape} "
            f"{packed.dtype}")
        assert scale.shape == (N, K // 128) and scale.dtype == torch.float16, (
            f"scales must be fp16 (N, K//128), got {scale.shape} {scale.dtype}")

        # decode_trits consumes flat uint8 bytes -> (n, 4) int8 trits
        # (LSB-first, 4 slots per byte); reshape back to (N, K//4, 4) then
        # (N, K) to undo the packing.
        trits = decode_trits(packed.reshape(-1)).reshape(N, K // 4,
                                                         4).reshape(N, K)

        self.signs = signs.to(device=device, dtype=torch.float32).flatten()
        assert self.signs.shape == (K, )

        if target == "fp8":
            w_f = trits.float() * scale.float().repeat_interleave(
                _SCALE_GROUP, dim=-1)
            groups = w_f.reshape(N, K // _SCALE_GROUP, _SCALE_GROUP)
            amax = groups.abs().amax(dim=-1).clamp(min=1e-12)
            w_target = (groups / amax.unsqueeze(-1)).reshape(N, K).to(
                torch.float8_e4m3fn)
            self.w_target = w_target
            self.w_scales = amax  # fp32 (N, K//128)
            # First-pass correctness implementation: cache the dequantized
            # bf16 weights so apply() can use a plain matmul.  A
            # grouped-scale fp8 GEMM (no bf16 dequant resident) is Task 9.
            self.w_dq_bf16 = (w_target.float().view(
                N, K // _SCALE_GROUP, _SCALE_GROUP) *
                amax.unsqueeze(-1)).view(N, K).to(torch.bfloat16)
            self._n_weight_bytes = (w_target.numel() *
                                    w_target.element_size() +
                                    amax.numel() * amax.element_size())
        else:
            # int4/int8: exact int8 trits + fp32 grouped scales.  Marlin
            # int4 packing lands in Task 9; the int4 target currently uses
            # the exact trits with a W4A16-style dequant matmul.
            self.w_target = trits.to(torch.int8)
            self.w_scales = scale.float()  # fp32 (N, K//128)
            self.w_dq_bf16 = (self.w_target.float().view(
                N, K // _SCALE_GROUP, _SCALE_GROUP) *
                self.w_scales.unsqueeze(-1)).view(N, K).to(torch.bfloat16)
            self._n_weight_bytes = (self.w_target.numel() *
                                    self.w_target.element_size() +
                                    self.w_scales.numel() *
                                    self.w_scales.element_size())
        self._target = target

    def _hadamard_blocks(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: (M, K) fp32 -> per-1024-block H @ (signs * x_b), concatenated.

        Each block b uses its own signs slice, matching the block-diagonal
        Hadamard rotation of the checkpoint format.  Returns (M, K) fp32.
        """
        nb = self.in_features // _HADAMARD_BLOCK
        outs = [
            fwht_signs(x[:, b * _HADAMARD_BLOCK:(b + 1) * _HADAMARD_BLOCK],
                       self.signs[b * _HADAMARD_BLOCK:(b + 1) *
                                  _HADAMARD_BLOCK])
            for b in range(nb)
        ]
        return outs if nb > 1 else [outs[0]]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        assert self._target is not None, (
            "process_weights_after_loading has not been called")
        assert x.dim() == 2 and x.shape[1] == self.in_features
        M, K = x.shape

        if self._target == "fp8":
            # fwht_signs_quant_fp8 folds signs + Hadamard itself and handles
            # (M, 1024) rows, so run it directly on each raw x block (do NOT
            # pre-fold signs) and concatenate.
            qs, amaxes = [], []
            for b in range(K // _HADAMARD_BLOCK):
                b0 = b * _HADAMARD_BLOCK
                q, amax = fwht_signs_quant_fp8(
                    x.float()[:, b0:b0 + _HADAMARD_BLOCK],
                    self.signs[b0:b0 + _HADAMARD_BLOCK])
                qs.append(q)
                amaxes.append(amax)
            q = torch.cat(qs, dim=-1)          # (M, K) fp8
            amax = torch.cat(amaxes, dim=-1)   # (M, K//128) fp32
            x_dq = (q.float().view(M, K // _SCALE_GROUP, _SCALE_GROUP) *
                    amax.unsqueeze(-1)).view(M, K).to(torch.bfloat16)
            return torch.matmul(x_dq, self.w_dq_bf16.t())
        # W4A16-style dequant matmul: xh stays fp32 out of the FWHT, the
        # bf16 dequantized weights are upcast for the matmul (fp32
        # accumulate; avoids an extra bf16 rounding of the activations).
        xh = torch.cat(self._hadamard_blocks(x.float()), dim=1)
        return torch.matmul(xh, self.w_dq_bf16.t().float())
