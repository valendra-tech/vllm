"""NumPy reference codecs for PQ2_0 and Q2b1 ternary weights."""

from __future__ import annotations

import numpy as np

PQ2_CODE_TO_TRIT = np.array([-1, 0, 1, 2], dtype=np.int8)
Q2B1_CODE_TO_TRIT = np.array([0, 1, -1, 0], dtype=np.int8)
TRIT_TO_Q2B1_CODE = np.array([2, 0, 1], dtype=np.uint8)

_CODE_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)
_BYTES_PER_BLOCK = 32
_VALUES_PER_BLOCK = 128


def decode_pq2_block(qs, scale) -> np.ndarray:
    """Decode one or more 32-byte PQ2 blocks using fp16 scales."""
    raw = np.asarray(qs, dtype=np.uint8)
    if raw.ndim == 1:
        if raw.shape != (_BYTES_PER_BLOCK,):
            raise ValueError("one PQ2 block must contain exactly 32 bytes")
        single = True
        blocks = raw.reshape(1, _BYTES_PER_BLOCK)
    elif raw.ndim == 2 and raw.shape[1] == _BYTES_PER_BLOCK:
        single = False
        blocks = raw
    else:
        raise ValueError("PQ2 blocks must have shape (32,) or (n, 32)")

    block_count = blocks.shape[0]
    codes = (blocks[:, :, None] >> _CODE_SHIFTS) & np.uint8(3)
    codes = codes.reshape(block_count, _VALUES_PER_BLOCK)
    trits = PQ2_CODE_TO_TRIT[codes].astype(np.float32)

    scales = np.asarray(scale, dtype=np.float16)
    if scales.ndim == 0:
        block_scales = np.full(block_count, scales, dtype=np.float32)
    elif scales.ndim == 1 and scales.shape == (block_count,):
        block_scales = scales.astype(np.float32)
    else:
        raise ValueError("scales must be a scalar or one value per PQ2 block")

    decoded = trits * block_scales[:, None]
    return decoded[0] if single else decoded


def encode_pq2_block(values):
    """Ternary-quantize values into LSB-first PQ2 blocks and fp16 scales."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim == 1:
        valid_shape = source.size > 0 and source.size % _VALUES_PER_BLOCK == 0
    elif source.ndim == 2:
        valid_shape = source.shape[0] > 0 and source.shape[1] == _VALUES_PER_BLOCK
    else:
        valid_shape = False
    if not valid_shape:
        raise ValueError(
            "values must be a non-empty 1D multiple of 128 elements or "
            "have shape (n, 128)"
        )
    if not np.all(np.isfinite(source)):
        raise ValueError("values must be finite")

    single = source.ndim == 1 and source.size == _VALUES_PER_BLOCK
    blocks = source.reshape(-1, _VALUES_PER_BLOCK)
    max_abs = np.max(np.abs(blocks), axis=1)
    if np.any(max_abs > np.float32(np.finfo(np.float16).max)):
        raise ValueError("scales must be representable as finite fp16 values")
    scales = max_abs.astype(np.float16)
    denominator = np.where(max_abs == 0, np.float32(1.0), max_abs)
    rounded = np.rint(blocks / denominator[:, None])
    ternary = np.clip(rounded, -1, 1).astype(np.int8)
    codes = (ternary + 1).astype(np.uint8)

    packed_codes = codes.reshape(-1, _BYTES_PER_BLOCK, 4)
    packed = np.bitwise_or.reduce(
        packed_codes << _CODE_SHIFTS,
        axis=2,
    ).astype(np.uint8)
    return (packed[0] if single else packed), scales


def pq2_to_q2b1_bytes(qs) -> np.ndarray:
    """Repack a flat ternary PQ2 byte stream as Q2b1 bytes."""
    raw = np.asarray(qs, dtype=np.uint8)
    if raw.ndim == 1:
        flat = raw
    elif raw.ndim == 2 and raw.shape[1] == _BYTES_PER_BLOCK:
        flat = raw.reshape(-1)
    else:
        raise ValueError("PQ2 bytes must have shape (32,) or (n, 32)")
    if flat.size == 0:
        return flat.copy()

    codes = (flat[:, None] >> _CODE_SHIFTS) & np.uint8(3)
    if np.any(codes == 3):
        raise ValueError("PQ2 code 3 (+2) cannot be represented in Q2b1")

    q2b1_codes = TRIT_TO_Q2B1_CODE[PQ2_CODE_TO_TRIT[codes] + 1]
    return np.bitwise_or.reduce(
        q2b1_codes << _CODE_SHIFTS,
        axis=-1,
    ).astype(np.uint8)


def q2b1_to_trits() -> np.ndarray:
    """Return a fresh byte-to-four-trit Q2b1 lookup table."""
    byte_values = np.arange(256, dtype=np.uint16)[:, None]
    codes = (byte_values >> _CODE_SHIFTS) & np.uint16(3)
    return Q2B1_CODE_TO_TRIT[codes].astype(np.int8, copy=True)


def hadamard_matrix(block: int) -> np.ndarray:
    """Return a normalized natural-order Sylvester Hadamard matrix."""
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise ValueError("Hadamard block size must be a positive power of two")
    block = int(block)
    if block < 1 or block & (block - 1):
        raise ValueError("Hadamard block size must be a positive power of two")

    indices = np.arange(block, dtype=np.uint64)
    parity = indices[:, None] & indices[None, :]
    for shift in (32, 16, 8, 4, 2, 1):
        parity ^= parity >> np.uint64(shift)
    signs = np.where((parity & 1) == 0, 1.0, -1.0).astype(np.float32)
    return signs * np.float32(1.0 / np.sqrt(block))
