import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

def test_decode_lut_matches_numpy():
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    from prism_pq2 import q2b1_to_trits
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (1 << 20,), dtype=torch.uint8, device="cuda")
    from vllm.model_executor.layers.quantization.bonsai_decode import decode_trits
    trits = decode_trits(packed)  # (n, 4) int8 on cuda
    lut = q2b1_to_trits()
    expected = lut[packed.cpu().numpy().astype(int)]
    assert trits.shape == (packed.numel(), 4)
    np.testing.assert_array_equal(trits.cpu().numpy(), expected)
