"""Dependency-light reader for the GGUF header and tensor directory."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

# GGUF metadata scalar types: u8, i8, u16, i16, u32, i32, f32, bool,
# u64, i64, and f64. Strings and arrays are handled separately.
TYPE_FMT = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}

GGML_TENSOR_BYTES = {
    0: (1, 4),
    1: (1, 2),
    30: (1, 2),
    142: (128, 34),
}


@dataclass
class TensorInfo:
    """Entry in the GGUF tensor directory."""

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int


class GgufMinReader:
    """Read GGUF v3 metadata and raw tensor payloads."""

    def __init__(self, path):
        self.f = open(path, "rb")  # noqa: SIM115 - seekable reader lifetime
        header = self._read(24)
        magic, self.version, tensor_count, metadata_count = struct.unpack(
            "<4sIQQ", header
        )
        if magic != b"GGUF":
            raise AssertionError("not a GGUF file")

        self.kv = {}
        for _ in range(metadata_count):
            key = self._read_string()
            value_type = struct.unpack("<I", self._read(4))[0]
            self.kv[key] = self._read_value(value_type)

        alignment = self._validate_alignment(self.kv.get("general.alignment", 32))
        self.tensors = []
        for _ in range(tensor_count):
            name = self._read_string()
            dimensions = struct.unpack("<I", self._read(4))[0]
            dims = struct.unpack(
                f"<{dimensions}Q",
                self._read(8 * dimensions),
            )
            ggml_type = struct.unpack("<I", self._read(4))[0]
            offset = struct.unpack("<Q", self._read(8))[0]
            if offset % alignment:
                raise ValueError(
                    f"tensor {name} offset {offset} is not aligned to {alignment}"
                )
            self.tensors.append(TensorInfo(name, dims, ggml_type, offset))

        directory_end = self.f.tell()
        self._data_start = ((directory_end + alignment - 1) // alignment) * alignment

    def _read(self, size: int) -> bytes:
        data = self.f.read(size)
        if len(data) != size:
            raise AssertionError(f"short read: expected {size} bytes, got {len(data)}")
        return data

    def _read_string(self) -> str:
        length = struct.unpack("<Q", self._read(8))[0]
        return self._read(length).decode("utf-8")

    def _read_value(self, value_type: int):
        if value_type == 8:
            return self._read_string()
        if value_type == 9:
            element_type = struct.unpack("<I", self._read(4))[0]
            count = struct.unpack("<Q", self._read(8))[0]
            return [self._read_value(element_type) for _ in range(count)]
        if value_type == 7:
            raw = self._read(1)
            if raw not in (b"\x00", b"\x01"):
                raise ValueError("GGUF bool metadata must be encoded as 0 or 1")
            return raw == b"\x01"
        try:
            format_string, size = TYPE_FMT[value_type]
        except KeyError as exc:
            raise ValueError(f"unsupported GGUF metadata type {value_type}") from exc
        return struct.unpack(format_string, self._read(size))[0]

    @staticmethod
    def _validate_alignment(alignment: int) -> int:
        if (
            type(alignment) is not int
            or not 8 <= alignment <= 0xFFFFFFFF
            or alignment % 8
        ):
            raise ValueError(
                f"invalid GGUF alignment: {alignment!r}; "
                "expected a positive uint32 multiple of 8"
            )
        return alignment

    def tensor_nbytes(self, info: TensorInfo) -> int:
        """Return the exact number of bytes occupied by a tensor."""
        try:
            block_size, type_size = GGML_TENSOR_BYTES[info.ggml_type]
        except KeyError as exc:
            raise ValueError(
                f"unsupported ggml type {info.ggml_type} for {info.name}"
            ) from exc

        if info.ggml_type == 142:
            if not info.dims or info.dims[0] % block_size:
                raise ValueError(
                    f"PQ2 tensor {info.name} first dimension must be a multiple "
                    f"of {block_size}"
                )
            block_count = (info.dims[0] // block_size) * math.prod(info.dims[1:])
        else:
            elements = math.prod(info.dims)
            if elements % block_size:
                raise ValueError(
                    f"tensor {info.name} has {elements} elements; "
                    f"expected a multiple of {block_size}"
                )
            block_count = elements // block_size
        return block_count * type_size

    def tensor_data(self, info: TensorInfo) -> bytes:
        """Read one tensor payload and reject truncated files."""
        self.f.seek(self._data_start + info.offset)
        return self._read(self.tensor_nbytes(info))

    def get(self, key):
        return self.kv[key]

    def get_str(self, key) -> str:
        return str(self.kv[key])

    def get_i32(self, key) -> int:
        return int(self.kv[key])

    def get_f32(self, key) -> float:
        return float(self.kv[key])

    def get_strs(self, key) -> list[str]:
        return [str(value) for value in self.kv[key]]

    def get_i32s(self, key) -> list[int]:
        return [int(value) for value in self.kv[key]]
