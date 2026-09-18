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


# ---------------------------------------------------------------------------
# vLLM QuantizationConfig wrapper
# ---------------------------------------------------------------------------
import torch.nn as nn

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .base_config import QuantizationConfig, QuantizeMethodBase


class BonsaiTernaryLinearMethodVLLM(LinearMethodBase):
    """vLLM linear method for Bonsai ternary (Q2b1) checkpoint layers.

    create_weights registers weight_packed (uint8, (out, in//4)) and
    weight_scale (fp16, (out, in//128)) params whose checkpoint names are
    <name>.weight and <name>.weight.scale.  apply performs the
    block-1024 signs+FWHT activation rotation, dequantizes the ternary
    weights to bf16 (cached) and uses a plain fp32-accumulate matmul
    (first-pass correctness implementation; kernel GEMM in Task 9).
    """

    def __init__(self, quant_config: "BonsaiTernaryQuantConfig"):
        self.quant_config = quant_config
        self._target_config: Optional[BonsaiTernaryConfig] = None
        self._in_features: Optional[int] = None
        self._out_features: Optional[int] = None
        self._core: Optional[BonsaiTernaryLinearMethod] = None
        self._signs_by_k: Dict[int, torch.Tensor] = {}
        self._gdn_v_grouped = bool(
            getattr(quant_config, "gdn_v_grouped", False))
        self._w_dq_cache: Dict[str, torch.Tensor] = {}
        self._signs_map = quant_config.hadamard_signs

    @staticmethod
    def _signs_from_config(hadamard_signs, k: int, device) -> torch.Tensor:
        assert hadamard_signs is not None, (
            "bonsai_ternary checkpoint is missing hadamard_signs")
        key = str(k)
        assert key in hadamard_signs, (
            f"hadamard_signs has no entry for width {k}; available: "
            f"{list(hadamard_signs.keys())}")
        s = torch.tensor(hadamard_signs[key], dtype=torch.float32,
                         device=device)
        assert s.numel() == k
        return s

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        K = input_size_per_partition
        N = sum(output_partition_sizes)
        self._in_features = K
        self._out_features = N
        self._target_config = BonsaiTernaryConfig(
            ternary_target=getattr(self.quant_config, "ternary_target",
                                   "int4"))

        weight_loader = extra_weight_attrs.pop("weight_loader")

        packed = ModelWeightParameter(
            data=torch.empty(N, K // 4, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", packed)

        scale = ModelWeightParameter(
            data=torch.empty(N, K // 128, dtype=torch.float16),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

        self._core = BonsaiTernaryLinearMethod(self._target_config, K, N)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        pass

    def _get_signs(self, K: int, device) -> torch.Tensor:
        signs = self._signs_by_k.get(K)
        if signs is None or signs.device != device:
            signs = self._signs_from_config(
                self.quant_config.hadamard_signs, K, device)
            self._signs_by_k[K] = signs
        return signs

    def _rotate_gdn_v(self, xh: torch.Tensor) -> torch.Tensor:
        # x: (M, 6144); permute per-128 blocks so the Hadamard sees the
        # grouped-V layout: (M, 128, 16, 3) -> (M, 128, 3, 16) -> (M, 6144)
        return xh.reshape(xh.shape[0], _HADAMARD_BLOCK, 16, 3).transpose(
            2, 3).reshape(xh.shape)

    @staticmethod
    def _fwht_plain(x: torch.Tensor) -> torch.Tensor:
        """Unnormalized WHT butterflies on the last dim (any size), signs folded
        by the caller.  Used for the embedding inverse rotation."""
        n = x.shape[-1]
        M = x.numel() // n
        x = x.reshape(M, n).float()
        h = 1
        while h < n:
            x = x.view(M, n // (2 * h), 2, h)
            a, b = x[..., 0, :], x[..., 1, :]
            x = torch.empty_like(x)
            x[..., 0, :] = a + b
            x[..., 1, :] = a - b
            h *= 2
        return x.view(M, n)

    @staticmethod
    def _rotate_embed(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Embedding inverse rotation: y = signs * (H @ row) per 1024 block.

        Checkpoint stores token_embd rows pre-rotated as
        (H (signs*row)) * signs; to recover the raw row:
        raw = signs * fwht((row * signs), ones).  signs: (K,).
        """
        M, K = x.shape
        y = torch.empty(M, K, dtype=torch.float32, device=x.device)
        nb = K // _HADAMARD_BLOCK
        ones = torch.ones(_HADAMARD_BLOCK, device=x.device)
        for b in range(nb):
            b0 = b * _HADAMARD_BLOCK
            xb = x[:, b0:b0 + _HADAMARD_BLOCK] * signs[b0:b0 + _HADAMARD_BLOCK]
            y[:, b0:b0 + _HADAMARD_BLOCK] = fwht_signs(xb, ones) * signs[
                b0:b0 + _HADAMARD_BLOCK]
        return y

    def embedding(
        self,
        layer: nn.Module,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Embedding lookup with the inverse Hadamard rotation applied."""
        dequant = _dequant_ternary(layer.weight_packed.data,
                                   layer.weight_scale.data)
        out = torch.nn.functional.embedding(input_.long(), dequant)
        signs = self._get_signs(dequant.shape[1], out.device)
        return self._rotate_embed(out.float(), signs).to(out.dtype)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = self._in_features
        N = self._out_features

        signs = self._get_signs(K, x.device)
        xf = x.float()
        prefix = getattr(layer, "prefix", "")
        if self._gdn_v_grouped and ".linear_attn.out_proj" in prefix:
            xf = self._rotate_gdn_v(xf)
        xh = torch.cat(
            [
                fwht_signs(
                    xf[:, b * _HADAMARD_BLOCK:(b + 1) * _HADAMARD_BLOCK],
                    signs[b * _HADAMARD_BLOCK:(b + 1) * _HADAMARD_BLOCK],
                )
                for b in range(K // _HADAMARD_BLOCK)
            ],
            dim=1,
        )
        packed = layer.weight_packed.data
        scale = layer.weight_scale.data
        if "packed" not in self._w_dq_cache or                 self._w_dq_cache["packed"].data_ptr() != packed.data_ptr():
            trits = decode_trits(packed.reshape(-1)).reshape(N, K)
            w_dq = (trits.float().view(N, K // _SCALE_GROUP, _SCALE_GROUP) *
                    scale.float().unsqueeze(-1)).view(N, K).to(torch.bfloat16)
            self._w_dq_cache["packed"] = packed
            self._w_dq = w_dq
        y = torch.matmul(xh.to(torch.bfloat16), self._w_dq.t())
        if bias is not None:
            y = y + bias
        return y


class BonsaiTernaryQuantConfig(QuantizationConfig):
    """vLLM config wrapper for the bonsai_ternary checkpoint format.

    Stores the ternary target plus the Hadamard metadata the quant method
    needs at runtime (signs per width, folded/inverse weight names, GDN
    grouped-V flag).
    """

    def __init__(
        self,
        ternary_target: str = "auto",
        hadamard_signs: Optional[Dict[str, List[int]]] = None,
        hadamard_folded: Optional[List[str]] = None,
        hadamard_inverse: Optional[List[str]] = None,
        gdn_v_grouped: bool = False,
    ):
        super().__init__()
        self.ternary_target = ternary_target
        self.hadamard_signs = hadamard_signs or {}
        self.hadamard_folded = hadamard_folded or []
        self.hadamard_inverse = hadamard_inverse or []
        self.gdn_v_grouped = gdn_v_grouped

    def get_name(self) -> str:
        return "bonsai_ternary"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict) -> "BonsaiTernaryQuantConfig":
        return cls(
            ternary_target=config.get("ternary_target", "auto"),
            hadamard_signs=config.get("hadamard_signs", {}),
            hadamard_folded=config.get("hadamard_folded", []),
            hadamard_inverse=config.get("hadamard_inverse", []),
            gdn_v_grouped=bool(config.get("gdn_v_grouped", False)),
        )

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        # Non-ternary GDN pieces stay unquantized.
        if any(
            p in prefix for p in (
                ".linear_attn.in_proj_a",
                ".linear_attn.in_proj_b",
                ".linear_attn.conv1d",
            )
        ):
            return None
        if isinstance(layer, (LinearBase, ParallelLMHead,
                              VocabParallelEmbedding)):
            return BonsaiTernaryLinearMethodVLLM(self)
        return None


# ---------------------------------------------------------------------------
# vLLM integration: QuantizationConfig + LinearMethodBase
# ---------------------------------------------------------------------------
from torch.nn import Parameter

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs


class BonsaiTernaryQuantConfig(QuantizationConfig):
    """vLLM config for packed ternary (Q2b1) Bonsai checkpoints.

    The checkpoint's quantization_config dict carries the Hadamard
    metadata produced by the converter: hadamard_signs maps an input
    width to the explicit +/-1 sign vector, hadamard_folded lists the
    (HF-named) weights whose activations receive H @ (signs * x) and
    hadamard_inverse lists latent lookup tables that receive
    signs * (H @ row) after the row gather.
    """

    def __init__(
        self,
        ternary_target: str = "auto",
        hadamard_signs: Optional[Dict[str, List[int]]] = None,
        hadamard_folded: Optional[List[str]] = None,
        hadamard_inverse: Optional[List[str]] = None,
        gdn_v_grouped: bool = False,
    ):
        super().__init__()
        self.ternary_target = ternary_target
        self.hadamard_signs = {
            int(k): torch.tensor(v, dtype=torch.float32)
            for k, v in (hadamard_signs or {}).items()
        }
        self.hadamard_folded = {
            n[: -len(".weight")] if n.endswith(".weight") else n
            for n in (hadamard_folded or [])
        }
        self.hadamard_inverse = {
            n[: -len(".weight")] if n.endswith(".weight") else n
            for n in (hadamard_inverse or [])
        }
        self.gdn_v_grouped = bool(gdn_v_grouped)

    @classmethod
    def get_name(cls) -> str:
        return "bonsai_ternary"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "BonsaiTernaryQuantConfig":
        return cls(
            ternary_target=config.get("ternary_target", "auto"),
            hadamard_signs=config.get("hadamard_signs"),
            hadamard_folded=config.get("hadamard_folded"),
            hadamard_inverse=config.get("hadamard_inverse"),
            gdn_v_grouped=config.get("gdn_v_grouped", False),
        )

    def get_quant_method(self, layer, prefix: str):
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if not isinstance(layer, (LinearBase, VocabParallelEmbedding)):
            return None
        if prefix in self.hadamard_folded or prefix in self.hadamard_inverse:
            return BonsaiTernaryLinearMethodVLLM(self, prefix)
        # Non-ternary GDN pieces (in_proj_ba, conv1d) keep full precision.
        return UnquantizedLinearMethod()


class BonsaiTernaryLinearMethodVLLM(LinearMethodBase):
    """Packed ternary linear with the offline-folded Hadamard rotation.

    Checkpoint tensors: <prefix>.weight (uint8 Q2b1, (N, K//4)) and
    <prefix>.weight_scale (fp16, (N, K//128)).  After loading, the
    exact trits are repacked to 2-per-byte nibbles plus fp32 grouped scales
    (resident weights stay at ~0.5 B/param; never expanded to FP16).
    """

    def __init__(self, quant_config: BonsaiTernaryQuantConfig, prefix: str):
        self.quant_config = quant_config
        self.prefix = prefix
        self.is_inverse = prefix in quant_config.hadamard_inverse
        self.is_gdn_out = (
            quant_config.gdn_v_grouped
            and prefix.endswith(".linear_attn.out_proj")
        )
        self.signs: Optional[torch.Tensor] = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        in_features = input_size_per_partition
        out_features = sum(output_partition_sizes)
        signs = self.quant_config.hadamard_signs.get(in_features)
        if signs is None:
            raise ValueError(
                f"no Hadamard signs for input width {in_features} ({self.prefix})"
            )
        self.signs = signs

        is_vocab = isinstance(layer, VocabParallelEmbedding)
        weight_loader = extra_weight_attrs.get("weight_loader")
        base_attrs: Dict[str, object] = {"input_dim": 1, "output_dim": 0}
        # Fused linear loaders need their shard-aware loader; vocab layers at
        # TP=1 load the full tensor with the default loader (their own loader
        # assumes the unquantized shapes).
        if weight_loader is not None and not is_vocab:
            base_attrs["weight_loader"] = weight_loader

        packed = Parameter(
            torch.empty(out_features, in_features // 4, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(packed, dict(base_attrs))
        layer.register_parameter("weight", packed)

        scale = Parameter(
            torch.empty(out_features, in_features // 128, dtype=torch.float16),
            requires_grad=False,
        )
        set_weight_attrs(scale, dict(base_attrs))
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        packed = layer.weight.data
        scale = layer.weight_scale.data
        out_features, k4 = packed.shape
        in_features = k4 * 4
        trits = decode_trits(packed.reshape(-1)).reshape(
            out_features, k4, 4
        ).reshape(out_features, in_features)
        # {-1,0,+1} -> {0,1,2}, two per byte (low nibble = even index).
        t = trits.to(torch.uint8) + 1
        nibbles = (t[:, 0::2] | (t[:, 1::2] << 4)).contiguous()
        layer.register_parameter("weight", None)
        layer.register_parameter("weight_scale", None)
        layer._bonsai_nibbles = nibbles
        layer._bonsai_scales = scale.float().contiguous()

    def _dequant(self, layer: torch.nn.Module) -> torch.Tensor:
        nib = layer._bonsai_nibbles
        scales = layer._bonsai_scales
        low = (nib & 0x0F).to(torch.int8)
        high = ((nib >> 4) & 0x0F).to(torch.int8)
        n, k2 = nib.shape
        t = torch.stack((low, high), dim=-1).reshape(n, k2 * 2)
        trits = t.float() - 1.0
        w = trits * scales.repeat_interleave(_SCALE_GROUP, dim=-1)
        return w.to(torch.bfloat16)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Forward activation transform: y = H @ (signs * x) per block."""
        assert self.signs is not None
        k = x.shape[-1]
        if self.is_gdn_out:
            # GDN head regroup [hd=128, nk=16, rep=3] -> [hd, rep, nk]
            # (mirrors llama.cpp prism.hadamard gdn_v_grouped).
            assert k == 6144, f"gdn_v_grouped expects inner size 6144, got {k}"
            m = x.shape[0]
            x = x.view(m, 128, 16, 3).permute(0, 1, 3, 2).contiguous().view(m, k)
        x = x.float()
        nb = k // _HADAMARD_BLOCK
        outs = []
        for b in range(nb):
            s = self.signs[b * _HADAMARD_BLOCK:(b + 1) * _HADAMARD_BLOCK].to(
                x.device
            )
            outs.append(
                fwht_signs(x[:, b * _HADAMARD_BLOCK:(b + 1) * _HADAMARD_BLOCK], s)
            )
        return torch.cat(outs, dim=-1) if nb > 1 else outs[0]

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xh = self._rotate(x).to(torch.bfloat16)
        w = self._dequant(layer)
        out = xh @ w.t()
        if bias is not None:
            out = out + bias
        return out

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        idx = input_.long()
        nib = layer._bonsai_nibbles
        scales = layer._bonsai_scales
        rows = nib[idx]
        low = (rows & 0x0F).to(torch.int8)
        high = ((rows >> 4) & 0x0F).to(torch.int8)
        m, k2 = rows.shape
        t = torch.stack((low, high), dim=-1).reshape(m, k2 * 2)
        trits = t.float() - 1.0
        w = trits * scales[idx].repeat_interleave(_SCALE_GROUP, dim=-1)
        if self.is_inverse:
            return self._inverse_rotate(w.to(torch.bfloat16))
        return w.to(torch.bfloat16)

    def _inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Latent lookup restore: x = signs * (H @ row) per block."""
        assert self.signs is not None
        k = x.shape[-1]
        nb = k // _HADAMARD_BLOCK
        ones = torch.ones(_HADAMARD_BLOCK, device=x.device, dtype=torch.float32)
        outs = []
        for b in range(nb):
            h = fwht_signs(x[:, b * _HADAMARD_BLOCK:(b + 1) * _HADAMARD_BLOCK], ones)
            s = self.signs[b * _HADAMARD_BLOCK:(b + 1) * _HADAMARD_BLOCK].to(
                x.device
            )
            outs.append(h * s)
        y = torch.cat(outs, dim=-1) if nb > 1 else outs[0]
        return y.to(x.dtype)
