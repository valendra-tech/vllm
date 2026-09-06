# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test for fused sigmoid_and_mul activation kernel.

Verifies out = sigmoid(input[:d]) * input[d:] matches the unfused reference.
"""

import pytest
import torch

from vllm.model_executor.layers.activation import SigmoidAndMul
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="sigmoid_and_mul CUDA kernel requires CUDA/ROCm",
)


@pytest.mark.parametrize("d", [128, 256, 512, 4096])
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
def test_sigmoid_and_mul_matches_reference(num_tokens, d):
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(num_tokens, 2 * d, dtype=torch.bfloat16, device=device)

    fn = SigmoidAndMul()
    out = fn(x)

    # Reference: sigmoid(x[:d]) * x[d:]
    ref = torch.sigmoid(x[:, :d].float()) * x[:, d:].float()
    ref = ref.to(torch.bfloat16)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_sigmoid_and_mul_native():
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float32)
    fn = SigmoidAndMul()
    out = fn.forward_native(x)
    ref = torch.sigmoid(x[:, :32]) * x[:, 32:]
    torch.testing.assert_close(out, ref, rtol=0, atol=0)