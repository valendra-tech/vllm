import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

@pytest.mark.parametrize("rows", [1, 7, 33])
def test_fwht_signs_matches_reference(rows):
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    from prism_pq2 import hadamard_matrix
    torch.manual_seed(0)
    x = torch.randn(rows, 1024, dtype=torch.float32, device="cuda")
    signs = torch.tensor([-1.0, 1.0] * 512, device="cuda")
    from vllm.model_executor.layers.quantization.hadamard_fwht import fwht_signs
    y = fwht_signs(x, signs)
    H = torch.from_numpy(hadamard_matrix(1024)).to("cuda")
    ref = (x * signs) @ H.T
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)

def test_fwht_signs_bf16():
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    from prism_pq2 import hadamard_matrix
    torch.manual_seed(1)
    x = torch.randn(4, 1024, dtype=torch.bfloat16, device="cuda")
    signs = torch.tensor([-1.0, 1.0] * 512, device="cuda")
    from vllm.model_executor.layers.quantization.hadamard_fwht import fwht_signs
    y = fwht_signs(x, signs)
    H = torch.from_numpy(hadamard_matrix(1024)).to("cuda")
    ref = (x.float() * signs) @ H.T
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)

def test_fwht_fp8_epilogue_shapes_and_error():
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    torch.manual_seed(0)
    x = torch.randn(8, 1024, dtype=torch.float32, device="cuda")
    signs = torch.tensor([-1.0, 1.0] * 512, device="cuda")
    from prism_pq2 import hadamard_matrix
    from vllm.model_executor.layers.quantization.hadamard_fwht import fwht_signs_quant_fp8
    q, amax = fwht_signs_quant_fp8(x, signs, group=128)
    assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
    assert amax.dtype == torch.float32 and amax.shape == (8, 8)
    # dequantized within fp8 tolerance of the true FWHT
    H = torch.from_numpy(hadamard_matrix(1024)).to("cuda")
    ref = (x * signs) @ H.T
    deq = q.float().view(8, 8, 128) * amax.view(8, 8, 1)
    err = (deq.view(8, 1024) - ref).abs().max()
    assert err < 0.5, f"fp8 epilogue error too large: {err}"
