"""Q2b1 packed ternary bytes -> trits via a 256x4 LUT CUDA kernel.

Loads the LUT into shared memory once per block; each thread decodes its byte
into 4 int8 trits. Extension is compiled once via torch.utils.cpp_extension.load
and cached by torch's extension cache.
"""
import os
import torch

_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        src = os.path.join(os.path.dirname(__file__), "bonsai_decode.cu")
        from torch.utils.cpp_extension import load
        _ext = load(name="bonsai_decode_lut", sources=[src], verbose=False)
    return _ext

def _make_lut(device):
    # 00->0, 01->+1, 10->-1, 11->0
    lut = torch.zeros((256, 4), dtype=torch.int8)
    for byte in range(256):
        for b in range(4):
            code = (byte >> (2 * b)) & 3
            lut[byte, b] = {0: 0, 1: 1, 2: -1, 3: 0}[code]
    return lut.to(device)

def decode_trits(packed: torch.Tensor) -> torch.Tensor:
    """(N,) uint8 cuda -> (N, 4) int8 trits on the same device."""
    ext = _load_ext()
    lut = _make_lut(packed.device)
    return ext.bonsai_decode_lut(packed.contiguous(), lut)
