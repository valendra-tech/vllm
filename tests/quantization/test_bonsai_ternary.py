import numpy as np
import torch
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


def test_converter_decode_standard_layout():
    """PQ2_0 groups run along ne0 with output-major block order."""
    import os

    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present")
    import numpy as np
    from gguf_reader_min import GgufMinReader
    from prism_pq2 import PQ2_CODE_TO_TRIT
    from prism_bonsai_convert import decode_pq2_tensor_trits

    r = GgufMinReader(GGUF_PATH)
    by = {t.name: t for t in r.tensors}
    info = by["blk.0.ffn_down.weight"]  # (ne0=17408, ne1=5120)
    trits, d = decode_pq2_tensor_trits(r, info)
    assert trits.shape == (5120, 17408)
    assert d.shape == (5120, 136)
    assert trits.dtype == np.int8 and d.dtype == np.float16
    assert set(np.unique(trits)) <= {-1, 0, 1}
    # Independent raw parse: block b = output*G + input_group covers
    # ne0[input_group*128:(input_group+1)*128] for one output row.
    raw = r.tensor_data(info)
    blocks = np.frombuffer(raw, np.uint8).reshape(-1, 34)
    G = 17408 // 128
    for b in (0, 1, 39, 40, 41, 1000):
        output_i, g = divmod(b, G)
        dref = np.frombuffer(blocks[b, :2].tobytes(), np.float16)[0]
        codes = (
            (blocks[b, 2:].reshape(32, 1) >> np.uint8(2 * np.arange(4))) & 3
        ).reshape(128)
        tref = PQ2_CODE_TO_TRIT[codes]
        assert d[output_i, g] == dref
        np.testing.assert_array_equal(
            trits[output_i, g * 128:(g + 1) * 128], tref
        )


def test_converter_reencode_q2b1_semantics():
    import os

    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present")
    import numpy as np
    from gguf_reader_min import GgufMinReader
    from prism_pq2 import q2b1_to_trits
    from prism_bonsai_convert import decode_pq2_tensor_trits, encode_ternary_natural

    r = GgufMinReader(GGUF_PATH)
    by = {t.name: t for t in r.tensors}
    info = by["blk.0.ssm_out.weight"]
    ne0, ne1 = 6144, 5120
    trits, d = decode_pq2_tensor_trits(r, info)
    qs, sc = encode_ternary_natural(trits, d)
    assert qs.shape == (ne1, ne0 // 4) and qs.dtype == np.uint8
    assert sc.shape == (ne1, ne0 // 128) and sc.dtype == np.float16
    lut = q2b1_to_trits()
    dec = lut[qs.reshape(-1)].reshape(ne1, ne0)
    assert np.array_equal(dec, trits)
    assert np.array_equal(sc, d)


def test_converter_reorders_gdn_ternary_output_rows(monkeypatch):
    import prism_bonsai_convert as converter

    source_for_target = np.arange(48).reshape(3, 16).T.reshape(-1)
    for name in (
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
    ):
        rows_count = 10240 if name.endswith("in_proj_qkv.weight") else 6144
        trits = np.zeros((rows_count, 128), dtype=np.int8)
        trits[:, 0] = np.arange(rows_count, dtype=np.int8) % 3 - 1
        scales = np.arange(rows_count, dtype=np.float16).reshape(rows_count, 1)
        monkeypatch.setattr(
            converter,
            "decode_pq2_tensor_trits",
            lambda _reader, _info, trits=trits, scales=scales: (trits, scales),
        )
        expected_trits = trits.copy()
        expected_scales = scales.copy()
        start = 4096 if rows_count == 10240 else 0
        rows = (start + source_for_target[:, None] * 128 + np.arange(128)).reshape(-1)
        expected_trits[start:] = trits[rows]
        expected_scales[start:] = scales[rows]
        expected_qs, expected_scales = converter.encode_ternary_natural(
            expected_trits, expected_scales
        )
        out = {}
        converter.convert_ternary(None, None, name, out)
        np.testing.assert_array_equal(out[name].numpy(), expected_qs)
        np.testing.assert_array_equal(out[name + "_scale"].numpy(), expected_scales)


def test_converter_keeps_qwen35_gdn_projection_tensors_separate(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    import prism_bonsai_convert as converter

    qkv = "model.layers.0.linear_attn.in_proj_qkv.weight"
    z = "model.layers.0.linear_attn.in_proj_z.weight"
    reader = SimpleNamespace(
        tensors=[
            SimpleNamespace(name="blk.0.attn_qkv.weight"),
            SimpleNamespace(name="blk.0.attn_gate.weight"),
        ]
    )
    saved = {}

    monkeypatch.setattr(converter, "GgufMinReader", lambda _path: reader)
    monkeypatch.setattr(
        converter,
        "build_name_map",
        lambda _reader: {
            "blk.0.attn_qkv.weight": qkv,
            "blk.0.attn_gate.weight": z,
        },
    )

    def fake_convert(_reader, _info, hf_name, output):
        output[hf_name] = torch.zeros((2, 3))
        output[hf_name + "_scale"] = torch.zeros((2, 1))

    monkeypatch.setattr(converter, "convert_tensor", fake_convert)
    monkeypatch.setattr(
        converter,
        "build_config",
        lambda _reader: {"quantization_config": {"hadamard_folded": []}},
    )
    monkeypatch.setattr(converter, "export_tokenizer", lambda _reader, _out: None)

    def fake_save_file(tensors, path, metadata):
        saved.update(tensors)
        open(path, "wb").close()

    monkeypatch.setattr("safetensors.torch.save_file", fake_save_file)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prism_bonsai_convert.py", "input.gguf", "--out", str(tmp_path)],
    )

    converter.main()

    assert set(saved) == {qkv, qkv + "_scale", z, z + "_scale"}


def test_vllm_weight_metadata_tracks_input_packing():
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    class Layer(torch.nn.Module):
        pass

    config = BonsaiTernaryQuantConfig(hadamard_signs={128: [1] * 128})
    method = BonsaiTernaryLinearMethodVLLM(
        config, "model.layers.0.linear_attn.in_proj_qkvz"
    )
    layer = Layer()
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[8, 8],
        input_size=128,
        output_size=16,
        params_dtype=torch.bfloat16,
    )

    assert layer.weight.shape == (16, 32)
    assert layer.weight.packed_dim == 1
    assert layer.weight_scale.shape == (16, 1)
    assert layer.weight_scale.packed_dim == 1


def test_vllm_weight_metadata_slices_global_signs_for_tp():
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    class Layer(torch.nn.Module):
        pass

    loader = object()
    config = BonsaiTernaryQuantConfig(hadamard_signs={128: list(range(128))})
    method = BonsaiTernaryLinearMethodVLLM(config, "model.layers.0.proj")
    layer = Layer()
    layer.tp_rank = 1
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[8],
        input_size=128,
        output_size=8,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )

    torch.testing.assert_close(
        layer._bonsai_signs, torch.arange(64, 128, dtype=torch.float32)
    )
    assert layer.weight.weight_loader is loader
    assert layer.weight_scale.weight_loader is loader


def test_process_weights_after_loading_raw_storage(monkeypatch):
    import vllm.model_executor.layers.quantization.bonsai_ternary as bonsai_ternary
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    K, N = 1024, 3
    config = BonsaiTernaryQuantConfig(
        ternary_target="int4", hadamard_signs={K: [1] * K}
    )
    method = BonsaiTernaryLinearMethodVLLM(
        config, "model.layers.0.linear_attn.in_proj_qkvz"
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        expected_packed = torch.randint(
            0, 256, (N, K // 4), dtype=torch.uint8
        )
        expected_scales = torch.rand(N, K // 128, dtype=torch.float16)
    layer.weight.data.copy_(expected_packed)
    layer.weight_scale.data.copy_(expected_scales)

    def fail_decode(_packed):
        raise AssertionError(
            "process_weights_after_loading must not call decode_trits"
        )

    monkeypatch.setattr(
        bonsai_ternary, "decode_trits", fail_decode, raising=True
    )
    method.process_weights_after_loading(layer)

    assert layer._bonsai_packed.shape == (N, K // 4)
    assert layer._bonsai_packed.dtype == torch.uint8
    assert torch.equal(layer._bonsai_packed, expected_packed)
    assert layer._bonsai_scales.shape == (N, K // 128)
    assert layer._bonsai_scales.dtype == torch.float16
    assert torch.equal(layer._bonsai_scales, expected_scales)
    assert not hasattr(layer, "_bonsai_nibbles")
    assert layer._parameters.get("weight") is None
    assert layer._parameters.get("weight_scale") is None


def test_process_weights_preloads_cuda_decode_state(monkeypatch):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("CUDA required")

    import vllm.model_executor.layers.quantization.bonsai_ternary as bonsai_ternary
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    K, N = 1024, 3
    config = BonsaiTernaryQuantConfig(
        ternary_target="int4", hadamard_signs={K: [1] * K}
    )
    method = BonsaiTernaryLinearMethodVLLM(
        config, "model.layers.0.linear_attn.in_proj_qkvz"
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
    )
    layer = layer.cuda()
    calls = []
    monkeypatch.setattr(
        bonsai_ternary, "_load_ext", lambda: calls.append("extension"), raising=True
    )
    monkeypatch.setattr(
        bonsai_ternary,
        "_make_lut",
        lambda device: calls.append(("lut", device)),
        raising=True,
    )
    device = layer.weight.device
    monkeypatch.setenv("BONSAI_FUSED_GEMM", "1")

    method.process_weights_after_loading(layer)

    assert calls == ["extension", ("lut", device)]


def test_process_weights_skips_fused_setup_when_globally_disabled(monkeypatch):
    from types import SimpleNamespace

    import vllm.model_executor.layers.quantization.bonsai_ternary as bonsai_ternary

    class TensorStub:
        device = torch.device("cuda:1")
        is_cuda = True
        shape = (3, 256)

        def contiguous(self):
            return self

    class LayerStub:
        def __init__(self):
            packed = TensorStub()
            scale = TensorStub()
            self.packed = packed
            self.scale = scale
            self.weight = SimpleNamespace(data=packed)
            self.weight_scale = SimpleNamespace(data=scale)
            self._bonsai_signs = SimpleNamespace(device=packed.device)
            self._buffers = {"_bonsai_signs": self._bonsai_signs}
            self._parameters = {"weight": packed, "weight_scale": scale}

        def register_parameter(self, name, value):
            self._parameters[name] = value
            setattr(self, name, value)

    layer = LayerStub()
    method = object.__new__(bonsai_ternary.BonsaiTernaryLinearMethodVLLM)
    calls = []

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        bonsai_ternary,
        "_load_ext",
        lambda: calls.append("extension") or object(),
    )
    monkeypatch.setattr(
        bonsai_ternary,
        "_make_lut",
        lambda device: calls.append(("lut", device)),
    )
    monkeypatch.setattr(
        bonsai_ternary,
        "prewarm_q2b1_autotune",
        lambda *args: calls.append("prewarm"),
    )
    monkeypatch.setenv("BONSAI_FUSED_GEMM", "0")
    monkeypatch.delenv("BONSAI_FUSED_GEMM_M64", raising=False)

    result = method.process_weights_after_loading(layer)

    assert result is None
    assert calls == ["extension", ("lut", layer._bonsai_packed.device)]
    assert layer._bonsai_packed is layer.packed
    assert layer._bonsai_scales is layer.scale
    assert layer._parameters.get("weight") is None
    assert layer._parameters.get("weight_scale") is None


@pytest.mark.parametrize("error_type", (ImportError, OSError, RuntimeError))
def test_bonsai_decode_wraps_extension_load_errors(monkeypatch, error_type):
    from torch.utils import cpp_extension
    import vllm.model_executor.layers.quantization.bonsai_decode as bonsai_decode

    failure = error_type("extension load failed")

    def fail_load(*args, **kwargs):
        raise failure

    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(bonsai_decode, "_ext_error", None, raising=True)
    monkeypatch.setattr(cpp_extension, "load", fail_load, raising=True)

    with pytest.raises(bonsai_decode.BonsaiQ2b1UnavailableError) as exc_info:
        bonsai_decode._load_ext()
    assert exc_info.value.__cause__ is failure


def test_bonsai_decode_preserves_non_load_errors(monkeypatch):
    from torch.utils import cpp_extension
    import vllm.model_executor.layers.quantization.bonsai_decode as bonsai_decode

    failure = ValueError("invalid extension input")

    def fail_load(*args, **kwargs):
        raise failure

    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(bonsai_decode, "_ext_error", None, raising=True)
    monkeypatch.setattr(cpp_extension, "load", fail_load, raising=True)

    with pytest.raises(ValueError, match="invalid extension input"):
        bonsai_decode._load_ext()


def test_dequant_rows_uses_bit_shift_when_decoder_unavailable(monkeypatch):
    import vllm.model_executor.layers.quantization.bonsai_ternary as bonsai_ternary
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiQ2b1UnavailableError,
        BonsaiTernaryLinearMethodVLLM,
    )

    packed = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    scales = torch.tensor([[0.5]], dtype=torch.float16)

    def unavailable(_packed):
        raise BonsaiQ2b1UnavailableError("decoder unavailable")

    monkeypatch.setattr(
        bonsai_ternary, "decode_trits", unavailable, raising=True
    )
    got = BonsaiTernaryLinearMethodVLLM._dequant_rows(packed, scales)

    codes = torch.stack(
        [(packed >> shift) & 0x03 for shift in (0, 2, 4, 6)], dim=-1
    )
    lut = torch.tensor([0, 1, -1, 0], dtype=torch.float32)
    expected = (
        lut[codes.long()].reshape(1, 128)
        * scales.float().repeat_interleave(128, dim=-1)
    ).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected)


def test_dequant_rows_uses_bit_shift_for_cpu_storage(monkeypatch):
    import vllm.model_executor.layers.quantization.bonsai_ternary as bonsai_ternary
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
    )

    packed = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    scales = torch.tensor([[0.5]], dtype=torch.float16)

    def fail_decode(_packed):
        raise AssertionError("CPU storage must not invoke decode_trits")

    monkeypatch.setattr(
        bonsai_ternary, "decode_trits", fail_decode, raising=True
    )
    got = BonsaiTernaryLinearMethodVLLM._dequant_rows(packed, scales)

    codes = torch.stack(
        [(packed >> shift) & 0x03 for shift in (0, 2, 4, 6)], dim=-1
    )
    lut = torch.tensor([0, 1, -1, 0], dtype=torch.float32)
    expected = (
        lut[codes.long()].reshape(1, 128)
        * scales.float().repeat_interleave(128, dim=-1)
    ).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected)


def test_embedding_preserves_indexing_errors_and_negative_indices():
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    config = BonsaiTernaryQuantConfig(ternary_target="int4")
    method = BonsaiTernaryLinearMethodVLLM(config, "embed_tokens")
    layer = torch.nn.Module()
    layer._bonsai_packed = torch.tensor(
        [[0] * 32, [0x55] * 32, [0xAA] * 32], dtype=torch.uint8
    )
    layer._bonsai_scales = torch.ones((3, 1), dtype=torch.float16)

    negative = method.embedding(layer, torch.tensor([-1]))
    last = method.embedding(layer, torch.tensor([2]))
    torch.testing.assert_close(negative, last)
    empty = method.embedding(layer, torch.empty(0, dtype=torch.long))
    assert empty.shape == (0, 128)
    with pytest.raises(IndexError):
        method.embedding(layer, torch.tensor([3]))


def test_vllm_apply_fused_dispatch_and_env_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    if torch.version.cuda is None:
        pytest.skip("CUDA build required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80+ required")
    import vllm.model_executor.layers.quantization.bonsai_decode as bonsai_decode
    import vllm.model_executor.layers.quantization.bonsai_ternary as bonsai_ternary
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiQ2b1UnavailableError,
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    monkeypatch.setattr(
        bonsai_decode, "_ext", bonsai_decode._ext, raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_ext_error", bonsai_decode._ext_error, raising=True
    )
    monkeypatch.setattr(
        bonsai_decode,
        "_LUT_CACHE",
        dict(bonsai_decode._LUT_CACHE),
        raising=True,
    )
    try:
        bonsai_ternary._load_ext()
    except BonsaiQ2b1UnavailableError as exc:
        pytest.skip(f"Bonsai extension unavailable: {exc}")

    K, N, M = 1024, 3, 2
    config = BonsaiTernaryQuantConfig(
        ternary_target="int4", hadamard_signs={K: [1] * K}
    )
    method = BonsaiTernaryLinearMethodVLLM(
        config, "model.layers.0.linear_attn.in_proj_qkvz"
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
    )
    layer.cuda()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(24)
        packed = torch.randint(
            0, 256, (N, K // 4), device="cuda", dtype=torch.uint8
        )
        scales = (
            torch.rand(N, K // 128, device="cuda") + 0.5
        ).to(torch.float16)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    layer.weight.data.copy_(packed)
    layer.weight_scale.data.copy_(scales)
    monkeypatch.setenv("BONSAI_AUTOTUNE", "0")
    monkeypatch.setenv("BONSAI_FUSED_GEMM", "1")
    method.process_weights_after_loading(layer)

    calls = []

    real_fused = bonsai_ternary.q2b1_gemm_autotuned

    def fused(xh, received_packed, received_scales):
        calls.append((xh, received_packed, received_scales))
        assert xh.is_cuda and xh.is_contiguous()
        assert received_packed.is_cuda and received_packed.is_contiguous()
        assert received_scales.is_cuda and received_scales.is_contiguous()
        return real_fused(xh, received_packed, received_scales)

    monkeypatch.setattr(
        bonsai_ternary, "q2b1_gemm_autotuned", fused, raising=True
    )
    bias = torch.arange(N, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("BONSAI_FUSED_GEMM", "1")
    fused = method.apply(layer, x, bias)
    assert len(calls) == 1
    assert calls[0][1] is layer._bonsai_packed
    assert calls[0][2] is layer._bonsai_scales
    codes = torch.stack(
        [(packed >> shift) & 0x03 for shift in (0, 2, 4, 6)], dim=-1
    )
    trits = torch.tensor(
        [0, 1, -1, 0], dtype=torch.float32, device="cuda"
    )[codes.long()].reshape(N, K)
    weights = trits * scales.float().repeat_interleave(128, dim=-1)
    expected = (
        method._rotate(x, layer._bonsai_signs).float() @ weights.t()
    ).to(torch.bfloat16) + bias
    torch.testing.assert_close(
        fused,
        expected,
    )

    def unavailable_fused(xh, received_packed, received_scales):
        raise BonsaiQ2b1UnavailableError("GEMM extension unavailable")

    monkeypatch.setattr(
        bonsai_ternary,
        "q2b1_gemm_autotuned",
        unavailable_fused,
        raising=True,
    )
    unavailable = method.apply(layer, x, bias)
    assert len(calls) == 1

    codes = torch.stack(
        [(layer._bonsai_packed >> shift) & 0x03 for shift in (0, 2, 4, 6)],
        dim=-1,
    )
    trits = torch.tensor(
        [0, 1, -1, 0], dtype=torch.float32, device="cuda"
    )[codes.long()].reshape(N, K)
    weights = trits * layer._bonsai_scales.float().repeat_interleave(
        128, dim=-1
    )
    expected = (
        method._rotate(x, layer._bonsai_signs).float() @ weights.t()
    ).to(torch.bfloat16) + bias
    torch.testing.assert_close(unavailable, expected)

    def fail_runtime(xh, received_packed, received_scales):
        raise RuntimeError("kernel execution failed")

    monkeypatch.setattr(
        bonsai_ternary, "q2b1_gemm_autotuned", fail_runtime, raising=True
    )
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        method.apply(layer, x, bias)

    monkeypatch.setenv("BONSAI_FUSED_GEMM", "0")
    fallback = method.apply(layer, x, bias)
    assert len(calls) == 1
    torch.testing.assert_close(fallback, expected)


@pytest.mark.parametrize("fused_gemm", ("1", "0"))
def test_vllm_apply_compile_cached_unavailable_uses_dequant_fallback(
    monkeypatch, fused_gemm
):
    """Simulate compiler state; this is not real torch.compile coverage."""
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    if torch.version.cuda is None:
        pytest.skip("CUDA build required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80+ required")

    import vllm.model_executor.layers.quantization.bonsai_decode as bonsai_decode
    from vllm.model_executor.layers.quantization import bonsai_ternary
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    K, M, N = 128, 64, 2
    method = BonsaiTernaryLinearMethodVLLM(
        BonsaiTernaryQuantConfig(
            ternary_target="int4", hadamard_signs={K: [1] * K}
        ),
        "model.layers.0.linear_attn.in_proj_qkvz",
    )
    layer = torch.nn.Module()
    layer.register_buffer(
        "_bonsai_signs", torch.ones(K, device="cuda"), persistent=False
    )
    layer._bonsai_packed = torch.stack(
        [
            torch.full((K // 4,), 0x55, dtype=torch.uint8, device="cuda"),
            torch.full((K // 4,), 0xAA, dtype=torch.uint8, device="cuda"),
        ]
    )
    layer._bonsai_scales = torch.tensor(
        [[0.5], [2.0]], dtype=torch.float16, device="cuda"
    )
    x = torch.tensor(
        [[1.0] * K] * M, dtype=torch.bfloat16, device="cuda"
    )
    expected = torch.tensor(
        [[64.0, -256.0]] * M,
        dtype=torch.bfloat16,
        device="cuda",
    )

    unavailable = bonsai_decode.BonsaiQ2b1UnavailableError(
        "cached extension load failure"
    )
    custom_op_calls = []
    extension_load_calls = []
    fused_dispatch_calls = []

    def fail_custom_op(*args, **kwargs):
        custom_op_calls.append((args, kwargs))
        raise AssertionError("cached extension failure must use dequant fallback")

    def fail_extension_load(*args, **kwargs):
        extension_load_calls.append((args, kwargs))
        raise AssertionError("cached extension failure must not load extension")

    real_fused = bonsai_ternary.q2b1_gemm_autotuned

    def record_fused(xh, received_packed, received_scales):
        fused_dispatch_calls.append(
            (xh, received_packed, received_scales)
        )
        return real_fused(xh, received_packed, received_scales)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setenv("BONSAI_FUSED_GEMM", fused_gemm)
    monkeypatch.setenv("BONSAI_FUSED_GEMM_M64", "1")
    monkeypatch.setattr(bonsai_decode, "_ext", None, raising=True)
    monkeypatch.setattr(
        bonsai_decode, "_ext_error", unavailable, raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_load_ext", fail_extension_load, raising=True
    )
    monkeypatch.setattr(
        bonsai_ternary, "_load_ext", fail_extension_load, raising=True
    )
    monkeypatch.setattr(
        bonsai_ternary, "q2b1_gemm_autotuned", record_fused, raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_q2b1_gemm_op", fail_custom_op, raising=True
    )
    monkeypatch.setattr(
        bonsai_decode, "_decode_lut_op", fail_custom_op, raising=True
    )
    monkeypatch.setattr(method, "_rotate", lambda value, signs: value.float())

    got = method.apply(layer, x)

    assert got.shape == (M, N)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected)
    assert custom_op_calls == []
    assert extension_load_calls == []
    if fused_gemm == "1":
        assert len(fused_dispatch_calls) == 1
        assert fused_dispatch_calls[0][0].shape == x.shape
        assert fused_dispatch_calls[0][1] is layer._bonsai_packed
        assert fused_dispatch_calls[0][2] is layer._bonsai_scales
    else:
        assert fused_dispatch_calls == []


@pytest.mark.parametrize(
    "M, m64_value, expect_fused",
    [
        (32, None, True),
        (32, "0", True),
        (33, None, False),
        (64, None, False),
        (33, "0", False),
        (64, "0", False),
        (33, "1", True),
        (64, "1", True),
        (65, None, False),
        (65, "0", False),
        (65, "1", False),
    ],
)
def test_vllm_apply_fused_dispatch_boundaries(
    monkeypatch, M, m64_value, expect_fused
):
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    if torch.version.cuda is None:
        pytest.skip("CUDA build required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80+ required")

    import vllm.model_executor.layers.quantization.bonsai_ternary as bonsai_ternary
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )

    K, N = 1024, 3
    config = BonsaiTernaryQuantConfig(
        ternary_target="int4", hadamard_signs={K: [1] * K}
    )
    method = BonsaiTernaryLinearMethodVLLM(
        config, "model.layers.0.linear_attn.in_proj_qkvz"
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
    )
    layer._bonsai_signs = method.signs.to("cuda")
    layer._bonsai_packed = torch.zeros(
        (N, K // 4), dtype=torch.uint8, device="cuda"
    )
    layer._bonsai_scales = torch.ones(
        (N, K // 128), dtype=torch.float16, device="cuda"
    )
    fused_calls = []
    fallback_calls = []

    def record_fused(xh, packed, scales):
        fused_calls.append((xh.shape[0], packed, scales))
        return torch.zeros(
            (xh.shape[0], packed.shape[0]), dtype=torch.bfloat16, device=xh.device
        )

    def record_fallback(packed, scales):
        fallback_calls.append((packed, scales))
        return torch.zeros(
            (packed.shape[0], packed.shape[1] * 4),
            dtype=torch.bfloat16,
            device=packed.device,
        )

    monkeypatch.setattr(
        bonsai_ternary, "q2b1_gemm_autotuned", record_fused, raising=True
    )
    monkeypatch.setattr(method, "_rotate", lambda x, signs: x.float())
    monkeypatch.setattr(method, "_dequant_rows", record_fallback)
    monkeypatch.setenv("BONSAI_FUSED_GEMM", "1")
    if m64_value is None:
        monkeypatch.delenv("BONSAI_FUSED_GEMM_M64", raising=False)
    else:
        monkeypatch.setenv("BONSAI_FUSED_GEMM_M64", m64_value)

    x = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")
    got = method.apply(layer, x)

    assert len(fused_calls) == int(expect_fused)
    assert len(fallback_calls) == int(not expect_fused)
    assert got.shape == (M, N)


def test_gdn_out_projection_keeps_vllm_head_order():
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryLinearMethodVLLM,
        BonsaiTernaryQuantConfig,
    )
    from vllm.model_executor.layers.quantization.hadamard_fwht import fwht_signs

    signs = [1] * 6144
    config = BonsaiTernaryQuantConfig(hadamard_signs={"6144": signs})
    method = BonsaiTernaryLinearMethodVLLM(
        config, "model.layers.0.linear_attn.out_proj"
    )
    method.signs = torch.tensor(signs, dtype=torch.float32)
    x = torch.arange(6144, dtype=torch.float32).reshape(1, -1)
    expected = torch.cat(
        [fwht_signs(x[:, start : start + 1024], torch.ones(1024))
         for start in range(0, 6144, 1024)],
        dim=-1,
    )

    torch.testing.assert_close(method._rotate(x), expected)


def test_converter_unbiases_gemma_norm_weights():
    from types import SimpleNamespace
    from prism_bonsai_convert import convert_f32

    class Reader:
        def tensor_data(self, _info):
            return np.asarray([1.25, 0.5, 2.0], dtype=np.float32).tobytes()

    info = SimpleNamespace(dims=(3,))
    out = {}
    convert_f32(Reader(), info, "model.layers.0.input_layernorm.weight", out)

    torch.testing.assert_close(
        out["model.layers.0.input_layernorm.weight"].float(),
        torch.tensor([0.25, -0.5, 1.0]),
    )


def test_converter_restores_gdn_a_log():
    from types import SimpleNamespace
    from prism_bonsai_convert import convert_f32

    class Reader:
        def tensor_data(self, _info):
            return np.asarray(
                [-1.0, -np.exp(2.0), -np.exp(-3.0)], dtype=np.float32
            ).tobytes()

    info = SimpleNamespace(dims=(3,))
    out = {}
    convert_f32(Reader(), info, "model.layers.0.linear_attn.A_log", out)

    torch.testing.assert_close(
        out["model.layers.0.linear_attn.A_log"].float(),
        torch.tensor([0.0, 2.0, -3.0]),
        rtol=0.01,
        atol=0.01,
    )


def test_converter_preserves_gguf_bf16_matrix_row_order():
    from types import SimpleNamespace
    from prism_bonsai_convert import convert_bf16

    class Reader:
        def tensor_data(self, _info):
            values = torch.arange(12, dtype=torch.float32).to(torch.bfloat16)
            return values.view(torch.uint16).numpy().tobytes()

    out = {}
    convert_bf16(
        Reader(),
        SimpleNamespace(dims=(3, 4)),
        "generic.weight",
        out,
    )

    torch.testing.assert_close(
        out["generic.weight"].float(),
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
    )


def test_converter_preserves_gguf_f32_matrix_row_order():
    from types import SimpleNamespace
    from prism_bonsai_convert import convert_f32

    class Reader:
        def tensor_data(self, _info):
            return np.arange(12, dtype=np.float32).tobytes()

    out = {}
    convert_f32(Reader(), SimpleNamespace(dims=(3, 4)), "generic.weight", out)

    torch.testing.assert_close(
        out["generic.weight"].float(),
        torch.arange(12, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16).float(),
    )


def test_converter_reorders_gdn_gate_head_order():
    from types import SimpleNamespace
    from prism_bonsai_convert import convert_bf16

    class Reader:
        def tensor_data(self, _info):
            values = torch.arange(48 * 4, dtype=torch.float32).to(torch.bfloat16)
            return values.view(torch.uint16).numpy().tobytes()

    out = {}
    convert_bf16(
        Reader(),
        SimpleNamespace(dims=(4, 48)),
        "model.layers.0.linear_attn.in_proj_a.weight",
        out,
    )

    source_for_target = np.arange(48).reshape(3, 16).T.reshape(-1)
    expected = torch.arange(48 * 4, dtype=torch.float32).reshape(48, 4)[source_for_target]
    torch.testing.assert_close(out["model.layers.0.linear_attn.in_proj_a.weight"].float(), expected.to(torch.bfloat16).float())


def test_converter_reorders_gdn_scalar_head_order():
    from types import SimpleNamespace
    from prism_bonsai_convert import convert_f32

    class Reader:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def tensor_data(self, _info):
            return self.values.tobytes()

    info = SimpleNamespace(dims=(48,))
    out = {}
    convert_f32(
        Reader(np.arange(48)),
        info,
        "model.layers.0.linear_attn.dt_bias",
        out,
    )
    torch.testing.assert_close(
        out["model.layers.0.linear_attn.dt_bias"].float(),
        torch.arange(48, dtype=torch.float32)[np.arange(48).reshape(3, 16).T.reshape(-1)],
    )


def test_converter_reorders_gdn_conv_channel_order():
    from types import SimpleNamespace
    from prism_bonsai_convert import convert_f32

    class Reader:
        def tensor_data(self, _info):
            return np.arange(4 * 10240, dtype=np.float32).tobytes()

    out = {}
    convert_f32(
        Reader(),
        SimpleNamespace(dims=(4, 10240)),
        "model.layers.0.linear_attn.conv1d.weight",
        out,
    )
    source = torch.arange(4 * 10240, dtype=torch.float32).reshape(10240, 4)
    source_for_target = np.arange(48).reshape(3, 16).T.reshape(-1)
    expected = source.clone()
    rows = (4096 + source_for_target[:, None] * 128 + np.arange(128)).reshape(-1)
    expected[4096:] = source[rows]
    torch.testing.assert_close(
        out["model.layers.0.linear_attn.conv1d.weight"].float().reshape(10240, 4),
        expected.to(torch.bfloat16).float(),
    )


def test_converter_build_config_structure():
    import os
    if not os.path.exists(GGUF_PATH):
        pytest.skip("GGUF not present")
    from prism_bonsai_convert import build_config, build_name_map
    from gguf_reader_min import GgufMinReader
    r = GgufMinReader(GGUF_PATH)
    cfg = build_config(r)
    assert cfg["architectures"] == ["Bonsai2ForCausalLM"]
    assert cfg["model_type"] == "qwen3_5_text"
    qc = cfg["quantization_config"]
    assert qc["quant_method"] == "bonsai_ternary" and qc["ternary_target"] == "auto"
    assert cfg["hidden_size"] == 5120 and cfg["intermediate_size"] == 17408
    assert cfg["num_hidden_layers"] == 64 and cfg["vocab_size"] == 248320
    assert cfg["num_attention_heads"] == 24 and cfg["num_key_value_heads"] == 4
    assert cfg["head_dim"] == 256 and cfg["full_attention_interval"] == 4
    rp = cfg["rope_parameters"]
    assert rp["mrope_section"] == [11, 11, 10]
    assert rp["rope_theta"] == 10000000.0
    assert cfg["partial_rotary_factor"] == 0.25
    assert qc["hadamard_sign_mode"] == "explicit" and qc["hadamard_block"] == 1024
    assert qc["hadamard_transform"] == "normalized-sylvester-walsh-hadamard"
    assert qc["hadamard_axis"] == "input-last-dimension"
    signs = qc["hadamard_signs"]
    assert set(signs) == {"5120", "6144", "17408"}
    assert len(signs["5120"]) == 5120 and len(signs["6144"]) == 6144 and len(signs["17408"]) == 17408
    assert set(np.unique(np.array(signs["5120"]))) <= {-1, 1}
    lt = cfg["layer_types"]
    assert len(lt) == 64
    assert lt[0] == "linear_attention" and lt[2] == "linear_attention"
    assert lt[3] == "full_attention" and lt[63] == "full_attention"
    assert lt[4] == "linear_attention"
    assert cfg["linear_num_key_heads"] == 16 and cfg["linear_num_value_heads"] == 48
    assert cfg["linear_key_head_dim"] == 128 and cfg["linear_value_head_dim"] == 128
    assert cfg["linear_conv_kernel_dim"] == 4
    name_map = build_name_map(r)
    folded = qc["hadamard_folded"]
    assert len(folded) == 401
    assert "lm_head.weight" in folded
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in folded
    assert set(folded) <= set(name_map.values())
    assert qc["hadamard_inverse"] == ["model.embed_tokens.weight"]
    assert qc["gdn_v_grouped"] is True
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["attn_output_gate"] is True and cfg["output_gate_type"] == "swish"

# ---------------------------------------------------------------------------
# BonsaiTernaryLinearMethod (Task 7)
# ---------------------------------------------------------------------------

def _pack_q2b1(trits: torch.Tensor) -> torch.Tensor:
    # trits (N, K) int {-1,0,1} -> (N, K//4) uint8 LSB-first, per byte 4 slots
    codes = torch.tensor([2, 0, 1], device=trits.device)[(trits + 1)]
    N, K = trits.shape
    packed = torch.zeros(N, K // 4, dtype=torch.uint8, device=trits.device)
    for j in range(4):
        packed |= codes[:, j::4].to(torch.uint8) << (2 * j)
    return packed


def test_bonsai_ternary_linear_fp8_end_to_end():
    import torch
    from prism_pq2 import hadamard_matrix
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryConfig, BonsaiTernaryLinearMethod)
    torch.manual_seed(0)
    K, N, M = 1024, 512, 3
    trits = torch.randint(-1, 2, (N, K), device="cuda")
    packed = _pack_q2b1(trits)
    scale = (torch.rand(N, K // 128, device="cuda") + 0.5).to(torch.float16)
    signs = torch.tensor([-1.0, 1.0] * 512, device="cuda")
    lm = BonsaiTernaryLinearMethod(BonsaiTernaryConfig(ternary_target="fp8"), K, N)
    lm.process_weights_after_loading({"packed": packed, "scale": scale}, signs, "cuda:0")
    # memory check (small shapes): fp8 weights + fp32 grouped scales
    assert lm.weight_bytes() == K * N * 1 + N * (K // 128) * 4
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out = lm.apply(x)
    H = torch.from_numpy(hadamard_matrix(1024)).to("cuda")
    w_dq = (trits.float() * scale.float().repeat_interleave(128, dim=1)).double()
    xh = ((x.float() * signs) @ H.T).double()
    ref = xh @ w_dq.T
    # fp8 e4m3 grouped quantization of the activations has ~4% relative
    # noise per element; the resulting output noise (~0.026 * ref std, i.e.
    # ~0.7 abs for this shape) is independent of |ref|, so atol must cover
    # it. Measured need up to 2.4 across seeds -> 2.5.
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.15, atol=2.5)


def test_bonsai_ternary_linear_int4_exactish():
    import torch
    from prism_pq2 import hadamard_matrix
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    from vllm.model_executor.layers.quantization.bonsai_ternary import (
        BonsaiTernaryConfig, BonsaiTernaryLinearMethod)
    torch.manual_seed(1)
    K, N, M = 2048, 256, 2
    trits = torch.randint(-1, 2, (N, K), device="cuda")
    packed = _pack_q2b1(trits)
    scale = (torch.rand(N, K // 128, device="cuda") + 0.5).to(torch.float16)
    signs = torch.tensor([-1.0, 1.0] * 1024, device="cuda")
    lm = BonsaiTernaryLinearMethod(BonsaiTernaryConfig(ternary_target="int4"), K, N)
    lm.process_weights_after_loading({"packed": packed, "scale": scale}, signs, "cuda:0")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out = lm.apply(x)
    H = torch.from_numpy(hadamard_matrix(1024)).to("cuda")
    # block-hadamard: H + signs applied per 1024-block
    xh = torch.empty_like(x, dtype=torch.float64)
    for b in range(K // 1024):
        xb = x.double()[:, b*1024:(b+1)*1024] * signs[b*1024:(b+1)*1024].double()
        xh[:, b*1024:(b+1)*1024] = xb @ torch.from_numpy(hadamard_matrix(1024)).double().to("cuda").T
    w_dq = trits.double() * scale.double().repeat_interleave(128, dim=1)
    ref = xh @ w_dq.T
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.2)


def test_bonsai_ternary_auto_blackwell():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    from vllm.model_executor.layers.quantization.bonsai_ternary import BonsaiTernaryConfig
    cfg = BonsaiTernaryConfig(ternary_target="auto")
    if torch.cuda.get_device_capability()[0] >= 12:
        assert cfg.resolve() == "nvfp4"
    else:
        assert cfg.resolve() == "int4"
    assert BonsaiTernaryConfig(ternary_target="int4").resolve() == "int4"
