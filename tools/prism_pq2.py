"""Reference codec for Prism PQ2_0 ternary packing and Q2b1 repack.

PQ2_0 block (128 weights): fp16 scale d (amax), 32 bytes of 2-bit slots,
weight j -> byte j//4, bits (j%4)*2 (LSB-first). Code q: 0=-1, 1=0, 2=+1, 3=+2.
Q2b1 (candle PR 2683): 2-bit slot per weight, LSB-first: 00->0, 01->+1, 10->-1, 11->0.
"""
import numpy as np

PQ2_CODE_TO_TRIT = np.array([-1, 0, 1, 2], dtype=np.int8)      # q - 1
Q2B1_CODE_TO_TRIT = np.array([0, 1, -1, 0], dtype=np.int8)     # candle LUT
TRIT_TO_Q2B1_CODE = np.array([2, 0, 1], dtype=np.uint8)        # -1->2, 0->0, +1->1

def decode_pq2_block(qs: np.ndarray, d) -> np.ndarray:
    qs = np.asarray(qs, dtype=np.uint8)
    single = qs.ndim == 1
    qs = qs.reshape(-1, 32)
    nb = qs.shape[0]
    shifts = np.uint8(2 * np.arange(4, dtype=np.uint8))
    codes = (qs.reshape(nb, 32, 1) >> shifts) & np.uint8(3)   # (nb,32,4)
    codes = codes.reshape(nb, 128)
    if (codes == 3).any():
        raise ValueError("PQ2_0 code 3 (+2) found: not a ternary tensor")
    trits = PQ2_CODE_TO_TRIT[codes].astype(np.float32)
    d = np.atleast_1d(np.asarray(d, dtype=np.float16)).astype(np.float32).reshape(nb)
    out = trits * d[:, None]
    return out[0] if single else out

def encode_pq2_block(t: np.ndarray):
    t = np.asarray(t, dtype=np.float32)
    single = t.ndim == 1
    t = t.reshape(-1, 128)
    amax = np.abs(t).max(axis=-1)
    d = amax.astype(np.float16)
    out_qs = np.zeros((t.shape[0], 32), dtype=np.uint8)
    for j in range(128):
        w = t[:, j]
        q = np.rint(w / np.where(amax > 0, amax, 1.0)).astype(np.int32) + 1
        q = np.clip(q, 0, 3).astype(np.uint8)
        out_qs[:, j // 4] |= (q << np.uint8((j % 4) * 2))
    return (out_qs[0] if single else out_qs), d

def pq2_to_q2b1_bytes(qs: np.ndarray) -> np.ndarray:
    """Repack PQ2_0 slot bytes (32 B per 128 weights) into Q2b1 bytes (32 B per 128 weights)."""
    qs = np.asarray(qs, dtype=np.uint8).reshape(-1)
    shifts = np.uint8(2 * np.arange(4, dtype=np.uint8))
    codes = (qs.reshape(-1, 1) >> shifts) & np.uint8(3)       # (m,4)
    if (codes == 3).any():
        raise ValueError("PQ2_0 code 3 (+2) found: not a ternary tensor")
    trits = PQ2_CODE_TO_TRIT[codes]
    q2 = TRIT_TO_Q2B1_CODE[trits + 1]
    return (q2[:, 0] | (q2[:, 1] << np.uint8(2)) | (q2[:, 2] << np.uint8(4)) | (q2[:, 3] << np.uint8(6))).astype(np.uint8)

def q2b1_to_trits() -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.int8)
    for byte in range(256):
        for b in range(4):
            lut[byte, b] = Q2B1_CODE_TO_TRIT[(byte >> (2 * b)) & 3]
    return lut

def hadamard_matrix(block: int) -> np.ndarray:
    idx = np.arange(block, dtype=np.uint32)
    parity = idx[:, None] & idx[None, :]
    parity ^= parity >> np.uint32(16); parity ^= parity >> np.uint32(8)
    parity ^= parity >> np.uint32(4);  parity ^= parity >> np.uint32(2); parity ^= parity >> np.uint32(1)
    s = np.float32(1.0 / np.sqrt(block))
    return np.where((parity & 1).astype(bool), -s, s).astype(np.float32)
