"""Minimal raw GGUF reader that tolerates unknown ggml tensor type ids (e.g. 142=PQ2_0)."""
import struct
import os
import numpy as np

# GGUF metadata value types (GGUF v2/v3 spec): 0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32,
# 6=f32, 7=bool(1B), 8=string, 9=array, 10=u64, 11=i64, 12=f64
TYPE_FMT = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
            5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

# ggml tensor type -> bytes per element (or per 128 weights for block types we know)
GGML_TENSOR_BYTES = {0: 4, 1: 2, 6: 2, 7: 4, 30: 2, 142: None}  # 142 handled as blocks

class TensorInfo:
    __slots__ = ("name", "dims", "ggml_type", "offset")
    def __init__(self, name, dims, ggml_type, offset):
        self.name, self.dims, self.ggml_type, self.offset = name, dims, ggml_type, offset
    def __repr__(self):
        return f"TensorInfo({self.name!r}, dims={self.dims}, type={self.ggml_type}, off={self.offset})"

class GgufMinReader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.fsize = os.path.getsize(path)
        head = self.f.read(24)
        assert head[:4] == b"GGUF", "not a GGUF file"
        _, n_t, n_kv = struct.unpack_from("<IQI", head, 4)
        self.kv = {}
        for _ in range(n_kv):
            key = self._str()
            t = struct.unpack("<I", self._read(4))[0]
            self.kv[key] = self._value(t)
        self.tensors = []
        for _ in range(n_t):
            name = self._str()
            nd = struct.unpack("<I", self._read(4))[0]
            dims = struct.unpack(f"<{nd}Q", self._read(8 * nd))
            gt = struct.unpack("<I", self._read(4))[0]
            off = struct.unpack("<Q", self._read(8))[0]
            self.tensors.append(TensorInfo(name, dims, gt, off))
        pos = self.f.tell()
        self._data_start = (pos + 31) // 32 * 32  # default GGUF alignment
        # sanity: padding between dir end and data start should be small (< 64B)
        assert self._data_start - pos < 4096, f"data start too far: {self._data_start - pos}"

    def _read(self, n):
        b = self.f.read(n)
        assert len(b) == n, "short read"
        return b

    def _str(self):
        n = struct.unpack("<Q", self._read(8))[0]
        return self._read(n).decode("utf-8")

    def _value(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            at = struct.unpack("<I", self._read(4))[0]
            cnt = struct.unpack("<Q", self._read(8))[0]
            return [self._value(at) for _ in range(cnt)]
        fmt, sz = TYPE_FMT[t]
        return struct.unpack(fmt, self._read(sz))[0]

    def tensor_nbytes(self, info: TensorInfo) -> int:
        n = 1
        for d in info.dims: n *= d
        if info.ggml_type == 142:
            return n // 128 * 34
        b = GGML_TENSOR_BYTES.get(info.ggml_type)
        if b is None:
            raise ValueError(f"unsupported ggml type {info.ggml_type} for {info.name}")
        return n * b

    def tensor_data(self, info: TensorInfo) -> bytes:
        self.f.seek(self._data_start + info.offset)
        return self.f.read(self.tensor_nbytes(info))

    def get(self, key): return self.kv[key]
    def get_str(self, key): return str(self.kv[key])
    def get_i32(self, key): return int(self.kv[key])
    def get_f32(self, key): return float(self.kv[key])
    def get_strs(self, key): return [str(x) for x in self.kv[key]]
    def get_i32s(self, key): return [int(x) for x in self.kv[key]]
