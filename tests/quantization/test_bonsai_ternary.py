import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from prism_pq2 import decode_pq2_block, encode_pq2_block, pq2_to_q2b1_bytes, q2b1_to_trits, hadamard_matrix

def test_decode_pq2_block_golden():
    # codes (0,1,2,1) = (-1,0,+1,0) packed LSB-first in byte0
    qs = np.full(32, 0x55, dtype=np.uint8)  # all remaining codes = 1 -> 0
    qs[0] = 0b01_10_01_00
    d = np.float16(0.5)
    w = decode_pq2_block(qs, d)
    assert w.shape == (128,)
    assert w[0] == -0.5 and w[1] == 0.0 and w[2] == 0.5 and w[3] == 0.0
    assert np.all(w[4:] == 0.0)  # remaining codes all 1 -> 0

def test_roundtrip_pq2():
    rng = np.random.default_rng(0)
    t = rng.choice([-1.0, 0.0, 1.0], size=128).astype(np.float32)
    qs, d = encode_pq2_block(t)
    assert d[0] == np.float16(1.0)
    w = decode_pq2_block(qs, d)
    assert np.array_equal(w, t)

def test_pq2_to_q2b1_bytes():
    # PQ2 codes (0,1,2,1) -> trits (-1,0,+1,0) -> Q2b1 codes (2,0,1,0) -> byte 0b00_01_00_10 = 0x12
    qs = np.array([0b01_10_01_00], dtype=np.uint8)
    out = pq2_to_q2b1_bytes(qs)
    assert out[0] == 0x12

def test_q2b1_lut_matches_candle():
    lut = q2b1_to_trits()
    assert lut[0b00][0] == 0 and lut[0b01][0] == 1 and lut[0b10][0] == -1 and lut[0b11][0] == 0
    assert lut.shape == (256, 4) and lut.dtype == np.int8

def test_hadamard_matrix_matches_fork():
    H = hadamard_matrix(1024)
    assert H.shape == (1024, 1024)
    s = 1.0 / np.sqrt(1024.0)
    i, j = 5, 3
    par = i & j
    par ^= par >> 16; par ^= par >> 8; par ^= par >> 4; par ^= par >> 2; par ^= par >> 1
    assert H[i, j] == (-s if par & 1 else s)
    assert np.allclose(H @ H.T, np.eye(1024), atol=1e-5)

GGUF_PATH = "/dev/shm/bonsai-pq2.gguf"

def test_gguf_reader_metadata_and_dir():
    import os
    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present on this machine")
    from gguf_reader_min import GgufMinReader
    r = GgufMinReader(GGUF_PATH)
    assert r.get_str("general.architecture") == "qwen35"
    assert r.get_i32("qwen35.block_count") == 64
    assert r.get_i32("prism.hadamard.block_size") == 1024
    assert r.get_str("prism.hadamard.transform") == "normalized-sylvester-walsh-hadamard"
    assert r.get_str("prism.hadamard.axis") == "input-last-dimension"
    assert r.get_str("prism.hadamard.sign_mode") == "explicit"
    assert r.get_strs("prism.hadamard.inverse_weight_names") == ["token_embd.weight"]
    assert r.get_i32s("prism.hadamard.sign_widths") == [5120, 6144, 17408]
    assert len(r.get_i32s("prism.hadamard.sign_values")) == 5120 + 6144 + 17408
    assert len(r.tensors) == 851
    by = {t.name: t for t in r.tensors}
    assert by["output.weight"].dims == (5120, 248320) and by["output.weight"].ggml_type == 142
    assert by["blk.0.attn_qkv.weight"].dims == (5120, 10240)
    assert by["blk.0.ssm_out.weight"].dims == (6144, 5120)
    assert by["output_norm.weight"].ggml_type == 0


def test_q2b1_pack_roundtrip():
    rng = np.random.default_rng(1)
    trits = rng.choice([-1, 0, 1], size=(64, 128)).astype(np.int8)
    from prism_bonsai_convert import pack_q2b1_codes
    packed = pack_q2b1_codes(trits)
    assert packed.shape == (64, 32) and packed.dtype == np.uint8
    lut = q2b1_to_trits()
    dec = lut[packed.reshape(-1)].reshape(64, 128)
    assert np.array_equal(dec, trits)


def test_q2b1_pack_bit_positions():
    # LSB-first: weight j -> byte j//4, bits (j%4)*2; -1->0b10, 0->0b00, +1->0b01
    trits = np.zeros((1, 128), dtype=np.int8)
    trits[0, 0] = -1
    trits[0, 1] = 1
    trits[0, 4] = 1
    trits[0, 5] = -1
    from prism_bonsai_convert import pack_q2b1_codes
    packed = pack_q2b1_codes(trits)
    assert packed[0, 0] == 0b01_10  # weights 0,1 = codes 2,1
    assert packed[0, 1] == 0b10_01  # weights 4,5 = codes 1,2


def test_converter_name_map_full_coverage():
    import os
    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present")
    from prism_bonsai_convert import build_name_map
    from gguf_reader_min import GgufMinReader
    r = GgufMinReader(GGUF_PATH)
    mapping = build_name_map(r)
    unmapped = [t.name for t in r.tensors if t.name not in mapping]
    assert unmapped == [], f"unmapped: {unmapped[:10]}"
    dst = [mapping[t.name] for t in r.tensors]
    assert len(dst) == len(set(dst)), "duplicate HF names"


def test_converter_decode_transpose_reencode_bitexact():
    import os
    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present")
    import numpy as np
    from prism_bonsai_convert import decode_pq2_tensor_trits
    from gguf_reader_min import GgufMinReader
    r = GgufMinReader(GGUF_PATH)
    by = {t.name: t for t in r.tensors}
    # blk.0.ssm_out.weight: GGUF dims (ne0=6144 in, ne1=5120 out), PQ2_0
    info = by["blk.0.ssm_out.weight"]
    n_in, n_out = 6144, 5120
    trits, d = decode_pq2_tensor_trits(r, info)
    assert trits.shape == (n_out, n_in)
    assert d.shape == (n_out, n_in // 128)
    assert trits.dtype == np.int8 and d.dtype == np.float16
    assert set(np.unique(trits)) <= {-1, 0, 1}
    # reference decode straight from raw bytes (independent of helper layout)
    from prism_pq2 import decode_pq2_block
    raw = r.tensor_data(info)
    blocks = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    dd = blocks[:, :2].copy().view(np.float16).reshape(-1)
    qs = blocks[:, 2:]
    w = decode_pq2_block(qs, dd)  # (nb, 128) per block, block b = o*G + g
    ref = w.reshape(n_out, n_in // 128, 128)
    np.testing.assert_array_equal(trits.astype(np.float32).reshape(n_out, n_in // 128, 128) * d.astype(np.float32)[:, :, None], ref)
    # scales: one per (output row, input group), preserved from the file
    np.testing.assert_array_equal(d.reshape(-1), dd)


def test_converter_reencode_q2b1_semantics():
    import os
    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present")
    import numpy as np
    from prism_bonsai_convert import decode_pq2_tensor_trits, encode_ternary_hf
    from gguf_reader_min import GgufMinReader
    from prism_pq2 import q2b1_to_trits
    r = GgufMinReader(GGUF_PATH)
    by = {t.name: t for t in r.tensors}
    info = by["blk.0.ssm_out.weight"]
    n_in, n_out = 6144, 5120
    trits, d = decode_pq2_tensor_trits(r, info)
    qs, sc = encode_ternary_hf(trits, d)
    assert qs.shape == (n_out, n_in // 4) and qs.dtype == np.uint8
    assert sc.shape == (n_out, n_in // 128) and sc.dtype == np.float16
    lut = q2b1_to_trits()
    dec = lut[qs.reshape(-1)].reshape(n_out, n_in)
    assert np.array_equal(dec, trits)
    assert np.array_equal(sc, d)


def test_converter_build_config_structure():
    import os
    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present")
    from prism_bonsai_convert import build_config, build_name_map
    from gguf_reader_min import GgufMinReader
    r = GgufMinReader(GGUF_PATH)
    cfg = build_config(r)
    assert cfg["architectures"] == ["Bonsai2ForCausalLM"]
    assert cfg["model_type"] == "bonsai2"
    assert cfg["quant_method"] == "bonsai_ternary" and cfg["ternary_target"] == "auto"
    assert cfg["hidden_size"] == 5120 and cfg["intermediate_size"] == 17408
    assert cfg["num_hidden_layers"] == 64 and cfg["vocab_size"] == 248320
    assert cfg["num_attention_heads"] == 24 and cfg["num_key_value_heads"] == 4
    assert cfg["head_dim"] == 256 and cfg["full_attention_interval"] == 4
    assert cfg["rope_sections"] == [11, 11, 10] and cfg["rotary_dim"] == 64
    assert cfg["hadamard_sign_mode"] == "explicit" and cfg["hadamard_block"] == 1024
    signs = cfg["hadamard_signs"]
    assert set(signs) == {"5120", "6144", "17408"}
    assert len(signs["5120"]) == 5120 and len(signs["6144"]) == 6144 and len(signs["17408"]) == 17408
    assert set(np.unique(np.array(signs["5120"]))) <= {-1, 1}
    lt = cfg["layer_types"]
    assert len(lt) == 64
    assert lt[0] == "linear_attention" and lt[2] == "linear_attention"
    assert lt[3] == "full_attention" and lt[63] == "full_attention"
    assert lt[4] == "linear_attention"
    lac = cfg["linear_attn_config"]
    assert lac["num_key_heads"] == 16 and lac["num_value_heads"] == 48
    assert lac["key_head_dim"] == 128 and lac["value_head_dim"] == 128
    assert lac["state_size"] == 128 and lac["group_count"] == 16
    assert lac["time_step_rank"] == 48 and lac["inner_size"] == 6144 and lac["conv_kernel"] == 4
    name_map = build_name_map(r)
    folded = cfg["hadamard_folded"]
    assert len(folded) == 401
    assert "lm_head.weight" in folded
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in folded
    assert set(folded) <= set(name_map.values())
    assert cfg["hadamard_inverse"] == ["model.embed_tokens.weight"]
    assert cfg["gdn_v_grouped"] is True
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["attn_output_gate"] is True and cfg["output_gate_type"] == "swish"