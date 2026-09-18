# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for Bonsai2ForCausalLM on a tiny synthetic model.

Builds a 2-layer full-attention Bonsai2 checkpoint (packed Q2b1 ternary
linears + ternary latent embedding + Hadamard metadata), a 512-token
byte-level BPE tokenizer, and runs one generation through the vLLM engine.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

HIDDEN = 1024
INTER = 2048
VOCAB = 512
BLOCK = 1024
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 128


def _pack_q2b1(trits: torch.Tensor) -> torch.Tensor:
    """(N, K) int {-1,0,1} -> (N, K//4) uint8, LSB-first, code 2/0/1."""
    codes = torch.tensor([2, 0, 1])[(trits + 1)]
    n, k = trits.shape
    packed = torch.zeros(n, k // 4, dtype=torch.uint8)
    for j in range(4):
        packed |= codes[:, j::4].to(torch.uint8) << (2 * j)
    return packed


def _ternary(out_f: int, in_f: int, gen: torch.Generator):
    trits = torch.randint(-1, 2, (out_f, in_f), generator=gen)
    return {
        "weight": _pack_q2b1(trits),
        "weight_scale": (torch.rand(out_f, in_f // 128, generator=gen) * 0.5 + 0.75).to(
            torch.float16
        ),
    }


def _make_tiny_model(tmp_path):
    gen = torch.Generator().manual_seed(0)
    signs = {
        str(HIDDEN): (torch.randint(0, 2, (HIDDEN,), generator=gen) * 2 - 1).tolist(),
        str(INTER): (torch.randint(0, 2, (INTER,), generator=gen) * 2 - 1).tolist(),
    }
    ternary_specs = {
        "model.embed_tokens": (VOCAB, HIDDEN),
        "lm_head": (VOCAB, HIDDEN),
    }
    for n in range(2):
        p = f"model.layers.{n}"
        ternary_specs.update(
            {
                f"{p}.self_attn.q_proj": (HEADS * HEAD_DIM * 2, HIDDEN),
                f"{p}.self_attn.k_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
                f"{p}.self_attn.v_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
                f"{p}.self_attn.o_proj": (HIDDEN, HEADS * HEAD_DIM),
                f"{p}.mlp.gate_proj": (INTER, HIDDEN),
                f"{p}.mlp.up_proj": (INTER, HIDDEN),
                f"{p}.mlp.down_proj": (HIDDEN, INTER),
            }
        )

    tensors = {}
    for name, (out_f, in_f) in ternary_specs.items():
        w = _ternary(out_f, in_f, gen)
        tensors[f"{name}.weight"] = w["weight"]
        tensors[f"{name}.weight_scale"] = w["weight_scale"]

    for n in range(2):
        p = f"model.layers.{n}"
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(HIDDEN)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(HIDDEN)
        tensors[f"{p}.self_attn.q_norm.weight"] = torch.ones(HEAD_DIM)
        tensors[f"{p}.self_attn.k_norm.weight"] = torch.ones(HEAD_DIM)
    tensors["model.norm.weight"] = torch.ones(HIDDEN)

    folded = [f"{n}.weight" for n in ternary_specs if n != "model.embed_tokens"]
    inverse = ["model.embed_tokens.weight"]

    config = {
        "architectures": ["Bonsai2ForCausalLM"],
        "model_type": "qwen3_5_text",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": 2,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 2048,
        "full_attention_interval": 2,
        "layer_types": ["full_attention", "full_attention"],
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "use_cache": True,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [6, 5, 5],
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "bonsai_ternary",
            "ternary_target": "int4",
            "hadamard_block": BLOCK,
            "hadamard_signs": signs,
            "hadamard_folded": folded,
            "hadamard_inverse": inverse,
            "gdn_v_grouped": False,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(tmp_path / "model.safetensors"))

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["hello world this is a tiny test corpus " * 50], vocab_size=VOCAB)
    tok.save(str(tmp_path / "tokenizer.json"))
    return tmp_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_tiny_bonsai2_generates(tmp_path):
    path = _make_tiny_model(tmp_path)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(path),
        enforce_eager=True,
        max_model_len=256,
        quantization="bonsai_ternary",
        dtype="bfloat16",
    )
    out = llm.generate(
        ["hello world"], SamplingParams(max_tokens=8, temperature=0)
    )
    assert out and len(out[0].outputs[0].token_ids) > 0
