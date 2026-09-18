# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bonsai 2: packed-ternary Qwen3.5-family hybrid-attention model.

The language backbone is architecturally Qwen3.5 (GDN linear attention +
full attention hybrid); the only differences from a stock Qwen3.5 text
checkpoint are:

* Weights are stored as packed ternary (Q2b1 uint8 + fp16 group-128
  scales) with the Hadamard rotation folded offline.  The
  ``bonsai_ternary`` quantization method applies the matching activation
  transform and never expands weights beyond the packed target format.
* The token embedding table is stored in the rotated (latent) basis and
  must be restored with ``signs * (H @ row)`` right after the lookup
  (``hadamard_inverse`` in the quantization config).
"""

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.utils import maybe_prefix


class Bonsai2ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        quant_config = vllm_config.quant_config
        if quant_config is None or quant_config.get_name() != "bonsai_ternary":
            return

        # The stock Qwen3.5 text model creates ``embed_tokens`` without a
        # quant config; Bonsai checkpoints store it as a latent ternary
        # table, so recreate it quantized (the ``bonsai_ternary`` method
        # implements ``embedding`` with the inverse Hadamard restore).
        cfg = vllm_config.model_config.hf_text_config
        self.model.embed_tokens = VocabParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model.embed_tokens"),
        )
