# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from itertools import product

import numpy as np
import pytest

from tools.gguf_reader_min import GgufMinReader, TensorInfo
from tools.prism_pq2 import (
    PQ2_CODE_TO_TRIT,
    Q2B1_CODE_TO_TRIT,
    TRIT_TO_Q2B1_CODE,
    decode_pq2_block,
    encode_pq2_block,
    hadamard_matrix,
    pq2_to_q2b1_bytes,
    q2b1_to_trits,
)

GGML_F32 = 0
GGML_F16 = 1
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_BF16 = 30
GGML_PQ2_0 = 142


def _pack_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata(key, value_type, value):
    return _pack_string(key) + struct.pack("<I", value_type) + value


def _pack_array(value_type, values):
    if value_type == 8:
        payload = b"".join(_pack_string(value) for value in values)
    else:
        format_by_type = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            10: "Q",
            11: "q",
            12: "d",
        }
        payload = struct.pack(f"<{len(values)}{format_by_type[value_type]}", *values)
    return struct.pack("<IQ", value_type, len(values)) + payload


def _make_gguf(metadata, tensors=(), alignment=32):
    metadata_blob = b"".join(metadata)
    tensor_directory = bytearray()
    max_end = 0
    for name, dims, tensor_type, tensor_offset, tensor_payload in tensors:
        tensor_directory += _pack_string(name)
        tensor_directory += struct.pack("<I", len(dims))
        tensor_directory += struct.pack(f"<{len(dims)}Q", *dims)
        tensor_directory += struct.pack("<IQ", tensor_type, tensor_offset)
        max_end = max(max_end, tensor_offset + len(tensor_payload))

    prefix = (
        struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
        + metadata_blob
        + tensor_directory
    )
    data_start = (len(prefix) + alignment - 1) // alignment * alignment
    data = bytearray(max_end)
    for _, _, _, tensor_offset, tensor_payload in tensors:
        end = tensor_offset + len(tensor_payload)
        data[tensor_offset:end] = tensor_payload
    return prefix + b"\x00" * (data_start - len(prefix)) + bytes(data)


def _pack_2bit_codes(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    packed = np.bitwise_or.reduce(codes.reshape(-1, 4) << shifts, axis=1)
    return packed.reshape(codes.shape[:-1] + (codes.shape[-1] // 4,))


def test_decode_pq2_block_maps_all_codes_to_scaled_float32():
    codes = np.tile(np.arange(4, dtype=np.uint8), 32)
    qs = _pack_2bit_codes(codes)

    decoded = decode_pq2_block(qs, np.float16(0.5))

    expected = np.tile(
        np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32), 32
    ) * np.float32(0.5)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)


def test_decoder_accepts_code_3_as_plus_2_while_ternary_repacker_rejects_code_3():
    # The decoder accepts code 3 as +2, while the ternary repacker rejects code 3.
    np.testing.assert_array_equal(PQ2_CODE_TO_TRIT, [-1, 0, 1, 2])
    np.testing.assert_array_equal(TRIT_TO_Q2B1_CODE, [2, 0, 1])

    qs = _pack_2bit_codes(np.tile(np.arange(4, dtype=np.uint8), 32))
    decoded = decode_pq2_block(qs, np.float16(0.5))
    expected = np.tile(
        np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32), 32
    ) * np.float32(0.5)
    np.testing.assert_array_equal(decoded, expected)

    with pytest.raises(ValueError, match="code 3"):
        pq2_to_q2b1_bytes(np.array([0b11], dtype=np.uint8))


def test_pq2_decode_and_repack_cover_all_byte_values_across_multiple_blocks():
    all_byte_values = np.arange(256, dtype=np.uint8).reshape(8, 32)
    scales = np.linspace(0.25, 2.0, 8, dtype=np.float16)

    decoded = decode_pq2_block(all_byte_values, scales)

    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes = (all_byte_values[:, :, None] >> shifts) & np.uint8(3)
    expected = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)[codes]
    expected = expected.reshape(8, 128) * scales.astype(np.float32)[:, None]
    assert decoded.shape == (8, 128)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)

    valid_code_rows = np.array(list(product(range(3), repeat=4)), dtype=np.uint8)
    valid_code_rows = np.resize(valid_code_rows, (128, 4))
    repack_input = _pack_2bit_codes(valid_code_rows).reshape(4, 32)
    expected_q2_codes = np.array([2, 0, 1], dtype=np.uint8)[valid_code_rows]
    expected_repacked = _pack_2bit_codes(expected_q2_codes).reshape(-1)

    repacked = pq2_to_q2b1_bytes(repack_input)

    assert repacked.shape == (128,)
    np.testing.assert_array_equal(repacked, expected_repacked)


def test_q2b1_to_trits_has_expected_lut_shape_dtype_and_values():
    np.testing.assert_array_equal(Q2B1_CODE_TO_TRIT, [0, 1, -1, 0])
    lut = q2b1_to_trits()

    expected = np.empty((256, 4), dtype=np.int8)
    code_to_trit = np.array([0, 1, -1, 0], dtype=np.int8)
    for byte in range(256):
        expected[byte] = code_to_trit[(byte >> np.array([0, 2, 4, 6])) & 3]

    assert lut.shape == (256, 4)
    assert lut.dtype == np.int8
    np.testing.assert_array_equal(lut, expected)


def test_encode_pq2_block_has_independent_scale_shape_dtype_and_packed_bytes():
    source = np.zeros((2, 128), dtype=np.float32)
    source[0, :4] = [-2.0, 0.0, 2.0, -2.0]
    source[1, :4] = [3.0, -3.0, 0.0, 3.0]

    # These expected bytes are intentionally independent of _pack_2bit_codes.
    expected_qs = np.array(
        [
            [0x24] + [0x55] * 31,
            [0x92] + [0x55] * 31,
        ],
        dtype=np.uint8,
    )
    expected_scales = np.array([2.0, 3.0], dtype=np.float16)

    qs, scales = encode_pq2_block(source)

    assert qs.shape == (2, 32)
    assert qs.dtype == np.uint8
    assert scales.shape == (2,)
    assert scales.dtype == np.float16
    np.testing.assert_array_equal(qs, expected_qs)
    np.testing.assert_array_equal(scales, expected_scales)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_encode_pq2_block_rejects_non_finite_values(bad_value):
    source = np.zeros(128, dtype=np.float32)
    source[0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        encode_pq2_block(source)


def test_encode_pq2_block_rejects_shape_with_non_block_first_dimension():
    source = np.zeros((2, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="128"):
        encode_pq2_block(source)


@pytest.mark.parametrize(
    "source",
    [
        np.empty(0, dtype=np.float32),
        np.empty((0, 128), dtype=np.float32),
    ],
    ids=["flat", "batched"],
)
def test_encode_pq2_block_rejects_empty_input(source):
    with pytest.raises(ValueError, match="non-empty"):
        encode_pq2_block(source)


def test_decode_pq2_block_and_repack_support_empty_inputs():
    decoded = decode_pq2_block(
        np.empty((0, 32), dtype=np.uint8),
        np.empty(0, dtype=np.float16),
    )
    repacked = pq2_to_q2b1_bytes(np.empty(0, dtype=np.uint8))

    assert decoded.shape == (0, 128)
    assert decoded.dtype == np.float32
    assert repacked.shape == (0,)
    assert repacked.dtype == np.uint8


@pytest.mark.parametrize("block_count", [1, 3, 7])
def test_pq2_codec_handles_varied_multi_block_inputs(block_count):
    trits = np.resize(np.array([-1, 0, 1], dtype=np.int8), block_count * 128).reshape(
        block_count, 128
    )
    block_scales = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=np.float32)[
        :block_count
    ]
    source = trits.astype(np.float32) * block_scales[:, None]

    qs, scales = encode_pq2_block(source)
    decoded = decode_pq2_block(qs, scales)
    expected_q2b1 = _pack_2bit_codes(
        np.array([2, 0, 1], dtype=np.uint8)[trits + 1]
    ).reshape(-1)

    assert qs.shape == (block_count, 32)
    assert scales.shape == (block_count,)
    np.testing.assert_array_equal(decoded, source)
    np.testing.assert_array_equal(pq2_to_q2b1_bytes(qs), expected_q2b1)


@pytest.mark.parametrize(
    ("block_size", "selected_signs"),
    [
        (1, ((0, 0, 1),)),
        (
            4,
            (
                (0, 0, 1),
                (1, 1, -1),
                (2, 1, 1),
                (3, 2, -1),
            ),
        ),
        (
            1024,
            (
                (0, 0, 1),
                (1, 1, -1),
                (5, 3, -1),
                (7, 3, 1),
                (1023, 1023, 1),
                (1023, 511, -1),
            ),
        ),
    ],
)
def test_hadamard_matrix_sizes_are_orthogonal_with_selected_signs(
    block_size, selected_signs
):
    matrix = hadamard_matrix(block_size)

    assert matrix.shape == (block_size, block_size)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(
        matrix @ matrix.T,
        np.eye(block_size, dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    scale = np.float32(1.0 / np.sqrt(block_size))
    for row, column, sign in selected_signs:
        assert matrix[row, column] == np.float32(sign) * scale


def test_gguf_reader_parses_typed_metadata_and_ordered_relative_tensor_data(tmp_path):
    metadata = [
        _metadata("general.alignment", 4, struct.pack("<I", 64)),
        _metadata("test.u8", 0, struct.pack("<B", 0xA5)),
        _metadata("test.i8", 1, struct.pack("<b", -101)),
        _metadata("test.u16", 2, struct.pack("<H", 0xBEEF)),
        _metadata("test.i16", 3, struct.pack("<h", -12345)),
        _metadata("test.u32", 4, struct.pack("<I", 0xDEADBEEF)),
        _metadata("test.i32", 5, struct.pack("<i", -123456)),
        _metadata("test.f32", 6, struct.pack("<f", 1.25)),
        _metadata("test.bool", 7, struct.pack("<?", True)),
        _metadata("test.u64", 10, struct.pack("<Q", 2**63 + 17)),
        _metadata("test.i64", 11, struct.pack("<q", -(2**62) + 9)),
        _metadata("test.f64", 12, struct.pack("<d", 3.141592653589793)),
        _metadata("test.string", 8, _pack_string("bonsai")),
        _metadata("test.u8s", 9, _pack_array(0, [0, 127, 255])),
        _metadata("test.i8s", 9, _pack_array(1, [-128, -1, 127])),
        _metadata("test.u16s", 9, _pack_array(2, [0, 1024, 65535])),
        _metadata("test.i16s", 9, _pack_array(3, [-32768, 0, 32767])),
        _metadata("test.i32s", 9, _pack_array(5, [-2, 0, 7])),
        _metadata("test.strings", 9, _pack_array(8, ["first", "second"])),
        _metadata("test.prefix_marker", 7, struct.pack("<?", True)),
    ]
    f32_payload = struct.pack("<2f", 1.25, -2.5)
    pq2_256_payload = (
        struct.pack("<e", -1.25)
        + bytes(range(32, 64))
        + struct.pack("<e", 0.75)
        + bytes(range(64, 96))
    )
    pq2_128_payload = struct.pack("<e", 0.5) + bytes(range(32))
    tensors = [
        ("bias.f32", (2,), GGML_F32, 256, f32_payload),
        ("weight.pq2.256", (128, 2), GGML_PQ2_0, 0, pq2_256_payload),
        ("weight.pq2.128", (128,), GGML_PQ2_0, 128, pq2_128_payload),
    ]
    path = tmp_path / "typed-multi-tensor.gguf"
    path.write_bytes(_make_gguf(metadata, tensors, alignment=64))

    reader = GgufMinReader(path)

    alignment = reader.get_i32("general.alignment")
    assert alignment == 64
    assert alignment != 32
    assert reader.get("general.alignment") == alignment
    assert reader.get("test.u8") == 0xA5
    assert reader.get("test.i8") == -101
    assert reader.get("test.u16") == 0xBEEF
    assert reader.get("test.i16") == -12345
    assert reader.get("test.u32") == 0xDEADBEEF
    assert reader.get("test.i32") == -123456
    assert reader.get("test.f32") == pytest.approx(1.25)
    assert reader.get("test.bool") is True
    assert reader.get("test.u64") == 2**63 + 17
    assert reader.get("test.i64") == -(2**62) + 9
    assert reader.get("test.f64") == pytest.approx(3.141592653589793)
    assert reader.get_str("test.string") == "bonsai"
    assert reader.get("test.u8s") == [0, 127, 255]
    assert reader.get("test.i8s") == [-128, -1, 127]
    assert reader.get("test.u16s") == [0, 1024, 65535]
    assert reader.get("test.i16s") == [-32768, 0, 32767]
    assert reader.get_i32s("test.i32s") == [-2, 0, 7]
    assert reader.get_strs("test.strings") == ["first", "second"]
    assert reader.get("test.prefix_marker") is True

    assert len(reader.tensors) == 3
    assert [tensor.name for tensor in reader.tensors] == [
        "bias.f32",
        "weight.pq2.256",
        "weight.pq2.128",
    ]
    assert [tensor.offset for tensor in reader.tensors] == [256, 0, 128]
    assert all(tensor.offset % alignment == 0 for tensor in reader.tensors)
    assert sorted(tensor.offset for tensor in reader.tensors) == [0, 128, 256]

    raw = path.read_bytes()
    data_start = 832
    # The 771-byte directory prefix rounds to 832 at 64 bytes, not 800 at 32.
    assert data_start % 64 == 0
    assert data_start != 800
    assert raw[data_start : data_start + len(pq2_256_payload)] == pq2_256_payload
    assert (
        raw[data_start + 128 : data_start + 128 + len(pq2_128_payload)]
        == pq2_128_payload
    )
    assert raw[data_start + 256 : data_start + 256 + len(f32_payload)] == f32_payload

    f32_tensor, pq2_256_tensor, pq2_128_tensor = reader.tensors
    assert f32_tensor.dims == (2,)
    assert f32_tensor.ggml_type == GGML_F32
    assert reader.tensor_nbytes(f32_tensor) == 8
    assert reader.tensor_data(f32_tensor) == f32_payload

    assert pq2_256_tensor.dims == (128, 2)
    assert pq2_256_tensor.ggml_type == GGML_PQ2_0
    assert int(np.prod(pq2_256_tensor.dims)) == 256
    assert reader.tensor_nbytes(pq2_256_tensor) == 2 * 34
    assert reader.tensor_data(pq2_256_tensor) == pq2_256_payload

    assert pq2_128_tensor.dims == (128,)
    assert pq2_128_tensor.ggml_type == GGML_PQ2_0
    assert reader.tensor_nbytes(pq2_128_tensor) == 34
    assert reader.tensor_data(pq2_128_tensor) == pq2_128_payload


@pytest.mark.parametrize(
    ("dims", "ggml_type", "expected_nbytes"),
    [
        ((3, 2), GGML_F32, 24),
        ((3, 2), GGML_F16, 12),
        ((3, 2), GGML_BF16, 12),
        ((128,), GGML_PQ2_0, 34),
        ((128, 2), GGML_PQ2_0, 68),
        ((256, 3), GGML_PQ2_0, 204),
    ],
)
def test_tensor_nbytes_uses_product_and_first_dimension_block_rules(
    dims, ggml_type, expected_nbytes
):
    reader = object.__new__(GgufMinReader)
    tensor = TensorInfo("tensor", dims, ggml_type, 0)

    assert reader.tensor_nbytes(tensor) == expected_nbytes


def test_tensor_nbytes_rejects_pq2_first_dimension_not_block_aligned():
    reader = object.__new__(GgufMinReader)
    tensor = TensorInfo("invalid.pq2", (16, 16), GGML_PQ2_0, 0)

    with pytest.raises(ValueError, match="128"):
        reader.tensor_nbytes(tensor)


@pytest.mark.parametrize("ggml_type", [GGML_Q5_0, GGML_Q5_1])
def test_tensor_nbytes_rejects_unsupported_q5_types(ggml_type):
    reader = object.__new__(GgufMinReader)
    tensor = TensorInfo("unsupported.q5", (32,), ggml_type, 0)

    with pytest.raises(ValueError, match=f"unsupported ggml type {ggml_type}"):
        reader.tensor_nbytes(tensor)


def test_gguf_reader_accepts_alignment_24(tmp_path):
    metadata = [
        _metadata("general.alignment", 4, struct.pack("<I", 24)),
    ]
    path = tmp_path / "alignment-24.gguf"
    path.write_bytes(_make_gguf(metadata, alignment=24))

    reader = GgufMinReader(path)

    assert reader.get_i32("general.alignment") == 24


def test_gguf_reader_rejects_alignment_not_multiple_of_eight(tmp_path):
    metadata = [
        _metadata("general.alignment", 4, struct.pack("<I", 10)),
    ]
    path = tmp_path / "invalid-alignment.gguf"
    path.write_bytes(_make_gguf(metadata, alignment=32))

    with pytest.raises(ValueError, match="alignment"):
        GgufMinReader(path)


def test_gguf_reader_rejects_tensor_offset_not_aligned(tmp_path):
    metadata = [
        _metadata("general.alignment", 4, struct.pack("<I", 64)),
    ]
    path = tmp_path / "misaligned-tensor.gguf"
    path.write_bytes(
        _make_gguf(
            metadata,
            [("misaligned", (1,), GGML_F32, 4, struct.pack("<f", 1.0))],
            alignment=64,
        )
    )

    with pytest.raises(ValueError, match="offset"):
        GgufMinReader(path)


def test_gguf_reader_rejects_non_canonical_raw_bool_metadata(tmp_path):
    metadata = [
        _metadata("test.bool", 7, b"\x02"),
    ]
    path = tmp_path / "invalid-bool.gguf"
    path.write_bytes(_make_gguf(metadata))

    with pytest.raises(ValueError, match="bool"):
        GgufMinReader(path)


def test_gguf_reader_rejects_invalid_magic(tmp_path):
    path = tmp_path / "invalid-magic.gguf"
    path.write_bytes(struct.pack("<4sIQQ", b"NOPE", 3, 0, 0))

    with pytest.raises(AssertionError, match="GGUF"):
        GgufMinReader(path)


def test_gguf_reader_rejects_truncated_header(tmp_path):
    path = tmp_path / "truncated-header.gguf"
    path.write_bytes(b"GGUF")

    with pytest.raises(AssertionError, match="short read"):
        GgufMinReader(path)


def test_gguf_reader_rejects_truncated_tensor_data(tmp_path):
    path = tmp_path / "truncated-tensor.gguf"
    path.write_bytes(
        _make_gguf(
            [],
            [("truncated", (2,), GGML_F32, 0, struct.pack("<f", 1.0))],
        )
    )
    reader = GgufMinReader(path)

    with pytest.raises(AssertionError, match="short read"):
        reader.tensor_data(reader.tensors[0])


def test_gguf_reader_defaults_to_32_byte_alignment_when_omitted(tmp_path):
    tensor_bytes = struct.pack("<f", 3.25)
    path = tmp_path / "default-alignment.gguf"
    path.write_bytes(
        _make_gguf(
            [],
            [("abcdefghij", (1,), GGML_F32, 0, tensor_bytes)],
            alignment=32,
        )
    )

    reader = GgufMinReader(path)

    assert len(reader.tensors) == 1
    assert reader.tensor_data(reader.tensors[0]) == tensor_bytes
