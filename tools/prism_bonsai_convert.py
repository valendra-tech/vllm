"""Convert a Prism Bonsai2 PQ2_0 GGUF into HF safetensors (ternary Q2b1 + scales).

Validated layout facts (cross-checked against llama.cpp and the
prism.hadamard metadata of this GGUF):
- GGUF matmul weights store (ne0=in_features, ne1=out_features); HF wants (out, in).
- PQ2_0 blocks are output-major: block b = o*G + g with G = ne0/128 input
  groups per output row; fp16 scale d[b] belongs to (output row o, input
  group g) and is preserved 1:1 in the HF `(out, in)` layout.
- Export ternarized W = Hadamard-fold(signs * W0) with an error-compensated
  threshold (d is a dequantization scale, not the decision boundary), so the
  converter preserves trits and scales byte-exactly instead of re-quantizing.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader_min import GgufMinReader
from prism_pq2 import PQ2_CODE_TO_TRIT, TRIT_TO_Q2B1_CODE

GGML_F32 = 0
GGML_BF16 = 30
GGML_PQ2_0 = 142

_LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_COMMON_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_FULL_ATTN_SUFFIX_MAP = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}
_LINEAR_ATTN_SUFFIX_MAP = {
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
}
_GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

_GEMMA_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    "model.norm.weight",
)

_GDN_HEAD_SOURCE_FOR_TARGET = np.arange(48).reshape(3, 16).T.reshape(-1)
_GDN_QKV_WIDTH = 4096
_GDN_VALUE_WIDTH = 6144
_GDN_VALUE_GROUP = 128


def _reorder_gdn_heads(values):
    """Convert GGUF rep-major GDN heads to vLLM's grouped-head order."""
    if values.shape[0] != len(_GDN_HEAD_SOURCE_FOR_TARGET):
        return values
    return np.ascontiguousarray(values[_GDN_HEAD_SOURCE_FOR_TARGET])


def _reorder_gdn_value_channels(values):
    """Reorder the value part of a [q, k, v] channel tensor."""
    value = values[_GDN_QKV_WIDTH : _GDN_QKV_WIDTH + _GDN_VALUE_WIDTH]
    value = value.reshape(3, 16, _GDN_VALUE_GROUP, -1).transpose(1, 0, 2, 3)
    value = value.reshape(-1, *values.shape[1:])
    return np.concatenate((values[:_GDN_QKV_WIDTH], value))


def _reorder_gdn_ternary_output(trits, scales, hf_name):
    if hf_name.endswith("linear_attn.in_proj_qkv.weight"):
        start = _GDN_QKV_WIDTH
    elif hf_name.endswith("linear_attn.in_proj_z.weight"):
        start = 0
    else:
        return trits, scales

    source = _GDN_HEAD_SOURCE_FOR_TARGET
    rows = (
        start + source[:, None] * _GDN_VALUE_GROUP + np.arange(_GDN_VALUE_GROUP)
    ).reshape(-1)
    reordered_trits = trits.copy()
    reordered_scales = scales.copy()
    reordered_trits[start : start + _GDN_VALUE_WIDTH] = trits[rows]
    reordered_scales[start : start + _GDN_VALUE_WIDTH] = scales[rows]
    return reordered_trits, reordered_scales


def build_name_map(reader):
    interval = int(reader.kv.get("qwen35.full_attention_interval", 4))
    mapping = dict(_GLOBAL_MAP)
    for info in reader.tensors:
        m = _LAYER_RE.match(info.name)
        if not m:
            if info.name in _GLOBAL_MAP:
                continue
            raise ValueError(f"unrecognized tensor name: {info.name}")
        n, suffix = int(m.group(1)), m.group(2)
        layer_type = "full_attention" if n % interval == interval - 1 else "linear_attention"
        suffix_map = _COMMON_SUFFIX_MAP
        if suffix in _FULL_ATTN_SUFFIX_MAP or suffix in _LINEAR_ATTN_SUFFIX_MAP:
            suffix_map = _FULL_ATTN_SUFFIX_MAP if layer_type == "full_attention" else _LINEAR_ATTN_SUFFIX_MAP
            if suffix not in suffix_map:
                raise ValueError(f"tensor {info.name} not valid for {layer_type} layer {n}")
        if suffix not in suffix_map:
            raise ValueError(f"unrecognized layer tensor name: {info.name}")
        mapping[info.name] = f"model.layers.{n}.{suffix_map[suffix]}"
    return mapping


def pack_q2b1_codes(trits):
    """Pack int8 trits {-1,0,1} shaped (n, m), m%4==0, into Q2b1 bytes (n, m//4).

    Q2b1 slot, LSB-first: weight j -> byte j//4, bits (j%4)*2; -1->0b10, 0->0b00, +1->0b01.
    """
    trits = np.asarray(trits, dtype=np.int8)
    n, m = trits.shape
    if m % 4 != 0:
        raise ValueError(f"input width {m} not a multiple of 4")
    codes = TRIT_TO_Q2B1_CODE[trits + 1]                      # (n, m) uint8 in {0,1,2}
    bits = codes.reshape(n, m // 4, 4) << np.uint8([0, 2, 4, 6])
    packed = bits[..., 0] | bits[..., 1] | bits[..., 2] | bits[..., 3]
    return np.ascontiguousarray(packed, dtype=np.uint8)


def decode_pq2_tensor_trits(reader, info):
    """Decode to standard HF layout: trits (out, in), scales (out, in//128).

    GGML quantization groups the 128 input values in each output row. Blocks
    are ordered output-major: block index ``b = out * G + input_group``.
    """
    ne0, ne1 = (int(x) for x in info.dims)
    if ne0 % 128 != 0:
        raise ValueError(f"{info.name}: ne0 {ne0} not a multiple of 128")
    n_groups = ne0 // 128
    n_blocks = ne1 * n_groups
    blocks = np.frombuffer(reader.tensor_data(info), np.uint8).reshape(n_blocks, 34)
    d = blocks[:, :2].copy().view(np.float16).reshape(ne1, n_groups)
    qs = blocks[:, 2:]
    shifts = np.uint8(2 * np.arange(4, dtype=np.uint8))
    codes = ((qs.reshape(n_blocks, 32, 1) >> shifts) & np.uint8(3)).reshape(n_blocks, 128)
    if (codes == 3).any():
        raise ValueError(f"PQ2_0 code 3 (+2) found in {info.name}: not a ternary tensor")
    trits = PQ2_CODE_TO_TRIT[codes]  # (n_blocks, 128) int8
    return np.ascontiguousarray(trits.reshape(ne1, ne0)), d


def encode_ternary_natural(trits, d):
    """Encode standard (out, in) trits and scales as Q2b1 tensors."""
    qs = pack_q2b1_codes(trits)
    scale = np.ascontiguousarray(d, dtype=np.float16)
    return qs, scale


def convert_ternary(reader, info, hf_name, out):
    trits, d = decode_pq2_tensor_trits(reader, info)
    trits, d = _reorder_gdn_ternary_output(trits, d, hf_name)
    qs, scale = encode_ternary_natural(trits, d)
    out[hf_name] = torch.from_numpy(qs)
    out[hf_name + "_scale"] = torch.from_numpy(scale)


def _fuse_gdn_qkvz_tensors(tensors):
    """Fuse the separate GGUF qkv and gate tensors for Qwen3.5's module."""
    qkv_suffix = ".linear_attn.in_proj_qkv.weight"
    for qkv_name in list(tensors):
        if not qkv_name.endswith(qkv_suffix):
            continue
        prefix = qkv_name[: -len(qkv_suffix)]
        z_name = prefix + ".linear_attn.in_proj_z.weight"
        qkv_scale_name = qkv_name + "_scale"
        z_scale_name = z_name + "_scale"
        if z_name not in tensors or z_scale_name not in tensors:
            raise ValueError(f"missing GDN gate tensor for {qkv_name}")
        fused_name = prefix + ".linear_attn.in_proj_qkvz.weight"
        tensors[fused_name] = torch.cat((tensors[qkv_name], tensors[z_name]), dim=0)
        tensors[fused_name + "_scale"] = torch.cat(
            (tensors[qkv_scale_name], tensors[z_scale_name]), dim=0
        )
        del tensors[qkv_name], tensors[z_name]
        del tensors[qkv_scale_name], tensors[z_scale_name]


def convert_f32(reader, info, hf_name, out):
    shape = tuple(reversed(info.dims)) if len(info.dims) == 2 else info.dims
    arr = np.frombuffer(reader.tensor_data(info), np.float32).copy().reshape(shape)
    if arr.ndim == 2:
        if hf_name.endswith("linear_attn.conv1d.weight"):
            # Checkpoint layout is (conv_dim, 1, kernel); the GDN module's
            # depthwise Conv1d expects the singleton channel dim.
            arr = _reorder_gdn_value_channels(arr)
            arr = arr[:, None, :]
    if hf_name.endswith(_GEMMA_NORM_SUFFIXES):
        # GGUF/llama.cpp stores Qwen3.5's effective RMSNorm weight, while
        # vLLM's GemmaRMSNorm stores the zero-centered value and adds 1.
        arr -= np.float32(1.0)
    if hf_name.endswith(".linear_attn.A_log"):
        # llama.cpp stores the already-expanded decay coefficient -exp(A_log),
        # while vLLM's GDN kernel applies -exp() itself.
        if np.any(arr >= 0):
            raise ValueError(f"expected negative expanded A_log values in {hf_name}")
        arr = np.log(-arr)
    if hf_name.endswith(".linear_attn.A_log") or hf_name.endswith(
        ".linear_attn.dt_bias"
    ):
        arr = _reorder_gdn_heads(arr)
    # The model dtype is bf16; F32 norm/conv/scalar weights are downcast so
    # vLLM's RMSNorm keeps the activation dtype consistent end-to-end.
    out[hf_name] = torch.from_numpy(arr).to(torch.bfloat16)


def convert_bf16(reader, info, hf_name, out):
    u16 = np.frombuffer(reader.tensor_data(info), np.uint16)
    shape = tuple(reversed(info.dims)) if len(info.dims) == 2 else info.dims
    t = torch.from_numpy(np.ascontiguousarray(u16)).view(torch.bfloat16).reshape(shape)
    if hf_name.endswith(
        (".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")
    ):
        t = t[torch.from_numpy(_GDN_HEAD_SOURCE_FOR_TARGET).to(t.device)]
    out[hf_name] = t


def convert_tensor(reader, info, hf_name, out):
    if info.ggml_type == GGML_PQ2_0:
        convert_ternary(reader, info, hf_name, out)
    elif info.ggml_type == GGML_F32:
        convert_f32(reader, info, hf_name, out)
    elif info.ggml_type == GGML_BF16:
        convert_bf16(reader, info, hf_name, out)
    else:
        raise ValueError(f"unsupported ggml type {info.ggml_type} for {info.name}")


def build_config(reader):
    kv = reader.kv
    widths = reader.get_i32s("prism.hadamard.sign_widths")
    values = reader.get_i32s("prism.hadamard.sign_values")
    signs, pos = {}, 0
    for w in widths:
        signs[str(w)] = values[pos:pos + w]
        pos += w
    if pos != len(values):
        raise ValueError("hadamard sign_values length does not match sign_widths")
    name_map = build_name_map(reader)
    folded = [name_map[n] for n in reader.get_strs("prism.hadamard.weight_names")]
    inverse = [name_map[n] for n in reader.get_strs("prism.hadamard.inverse_weight_names")]
    n_layers = reader.get_i32("qwen35.block_count")
    interval = reader.get_i32("qwen35.full_attention_interval")
    if n_layers % interval != 0:
        raise ValueError("block_count not divisible by full_attention_interval")
    group = ["linear_attention"] * (interval - 1) + ["full_attention"]
    layer_types = group * (n_layers // interval)
    sections = reader.get_i32s("qwen35.rope.dimension_sections")
    ssm_state = reader.get_i32("qwen35.ssm.state_size")
    vocab_size = [t for t in reader.tensors if t.name == "token_embd.weight"][0].dims[1]
    return {
        "architectures": ["Bonsai2ForCausalLM"],
        "model_type": "qwen3_5_text",
        "quantization_config": {
            "quant_method": "bonsai_ternary",
            "ternary_target": "auto",
            "hadamard_block": reader.get_i32("prism.hadamard.block_size"),
            "hadamard_version": reader.get_i32("prism.hadamard.version"),
            "hadamard_transform": reader.get_str("prism.hadamard.transform"),
            "hadamard_axis": reader.get_str("prism.hadamard.axis"),
            "hadamard_sign_mode": reader.get_str("prism.hadamard.sign_mode"),
            "hadamard_signs": signs,
            "hadamard_folded": folded,
            "hadamard_inverse": inverse,
            "gdn_v_grouped": bool(reader.kv.get("prism.hadamard.gdn_v_grouped", 0)),
        },
        "hidden_size": reader.get_i32("qwen35.embedding_length"),
        "intermediate_size": reader.get_i32("qwen35.feed_forward_length"),
        "num_hidden_layers": n_layers,
        "num_attention_heads": reader.get_i32("qwen35.attention.head_count"),
        "num_key_value_heads": reader.get_i32("qwen35.attention.head_count_kv"),
        "head_dim": reader.get_i32("qwen35.attention.key_length"),
        "vocab_size": int(vocab_size),
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": reader.get_i32("qwen35.context_length"),
        "full_attention_interval": interval,
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
            "mrope_section": sections[:3],
            "partial_rotary_factor": 0.25,
            "rope_theta": reader.get_f32("qwen35.rope.freq_base"),
        },
        "linear_num_key_heads": reader.get_i32("qwen35.ssm.group_count"),
        "linear_num_value_heads": reader.get_i32("qwen35.ssm.time_step_rank"),
        "linear_key_head_dim": ssm_state,
        "linear_value_head_dim": ssm_state,
        "linear_conv_kernel_dim": reader.get_i32("qwen35.ssm.conv_kernel"),
        "mamba_ssm_dtype": "float32",
        "layer_types": layer_types,
        "torch_dtype": "bfloat16",
        "bos_token_id": reader.get_i32("tokenizer.ggml.bos_token_id"),
        "eos_token_id": reader.get_i32("tokenizer.ggml.eos_token_id"),
        "pad_token_id": reader.get_i32("tokenizer.ggml.padding_token_id"),
    }


GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "bos_token_id": 248044,
}


def export_tokenizer(reader, out_dir):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("tokenizer export failed: transformers is unavailable") from exc
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError("tokenizer export failed: could not load Qwen/Qwen3.8-27B") from exc
    ggml_tokens = reader.get_strs("tokenizer.ggml.tokens")
    spot_ids = [0, 1, 42, 248044, 248046]
    spot_ok = all(tok.convert_ids_to_tokens(i) == ggml_tokens[i] for i in spot_ids)
    added = 0
    if len(tok) < len(ggml_tokens) and spot_ok:
        extra_tokens = ggml_tokens[len(tok):]
        added = tok.add_tokens(extra_tokens, special_tokens=True)
        for attr, token_id in (
            ("bos_token", GENERATION_CONFIG["bos_token_id"]),
            ("eos_token", GENERATION_CONFIG["eos_token_id"]),
            ("pad_token", GENERATION_CONFIG["pad_token_id"]),
        ):
            setattr(tok, attr, ggml_tokens[token_id])

    spot_ok = all(tok.convert_ids_to_tokens(i) == ggml_tokens[i] for i in spot_ids)
    if len(tok) != len(ggml_tokens) or not spot_ok:
        raise RuntimeError(
            f"tokenizer export failed: vocabulary mismatch "
            f"len(tok)={len(tok)} vs ggml={len(ggml_tokens)}, "
            f"spot-check ids {spot_ids} ok={spot_ok}"
        )

    try:
        tok.save_pretrained(out_dir)
    except Exception as exc:
        raise RuntimeError(
            f"tokenizer export failed: could not save tokenizer to {out_dir}"
        ) from exc

    required_artifacts = ("tokenizer.json", "tokenizer_config.json")
    try:
        missing_artifacts = [
            name
            for name in required_artifacts
            if not os.path.isfile(os.path.join(out_dir, name))
        ]
    except Exception as exc:
        raise RuntimeError(
            f"tokenizer export failed: could not verify saved artifacts in {out_dir}"
        ) from exc
    if missing_artifacts:
        raise RuntimeError(
            f"tokenizer export failed: missing required artifacts in {out_dir}: "
            f"{', '.join(missing_artifacts)}"
        )

    print(
        f"[tokenizer] verified (len={len(tok)}, added={added}, "
        f"spot-check ok) and saved to {out_dir}"
    )


def main():
    ap = argparse.ArgumentParser(description="Convert Prism Bonsai2 PQ2_0 GGUF to HF safetensors")
    ap.add_argument("gguf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    reader = GgufMinReader(args.gguf)
    name_map = build_name_map(reader)
    print(f"[init] {len(reader.tensors)} tensors, {len(name_map)} mapped names")

    tensors = {}
    for i, info in enumerate(reader.tensors):
        hf_name = name_map[info.name]
        convert_tensor(reader, info, hf_name, tensors)
        if (i + 1) % 50 == 0 or i + 1 == len(reader.tensors):
            print(f"[convert] {i + 1}/{len(reader.tensors)} ({info.name} -> {hf_name}) "
                  f"elapsed {time.time() - t_start:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    st_path = os.path.join(args.out, "model.safetensors")
    from safetensors.torch import save_file
    save_file(tensors, st_path, metadata={"format": "pt"})
    del tensors
    st_bytes = os.path.getsize(st_path)
    print(f"[save] {st_path}: {st_bytes} bytes, wall {time.time() - t_start:.1f}s")

    cfg = build_config(reader)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GENERATION_CONFIG, f, indent=2)
    print(f"[save] config.json + generation_config.json written ({len(cfg['quantization_config']['hadamard_folded'])} folded names)")

    export_tokenizer(reader, args.out)
    print(f"[done] model.safetensors {st_bytes} bytes in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
