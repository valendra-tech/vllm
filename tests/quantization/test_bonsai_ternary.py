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
