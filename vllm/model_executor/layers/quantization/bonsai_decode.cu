#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void bonsai_decode_lut_kernel(const uint8_t* __restrict__ packed,
                                         const int8_t*  __restrict__ lut,
                                         int8_t* __restrict__ out,
                                         int64_t n) {
  __shared__ int8_t smem_lut[256][4];
  const int tid = threadIdx.x;
  for (int i = tid; i < 256 * 4; i += blockDim.x) {
    smem_lut[i / 4][i % 4] = lut[i];
  }
  __syncthreads();
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
  if (idx < n) {
    const uint8_t b = packed[idx];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[idx * 4 + j] = smem_lut[b][j];
    }
  }
}

static torch::Tensor bonsai_decode_lut(torch::Tensor packed, torch::Tensor lut) {
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.is_cuda());
  TORCH_CHECK(lut.scalar_type() == at::kChar && lut.numel() == 256 * 4);
  auto out = at::empty({packed.numel(), 4},
                       packed.options().dtype(at::kChar));
  const int64_t n = packed.numel();
  if (n == 0) return out;
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  bonsai_decode_lut_kernel<<<blocks, threads, 0, stream>>>(
      packed.data_ptr<uint8_t>(), lut.data_ptr<int8_t>(), out.data_ptr<int8_t>(), n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bonsai_decode_lut", &bonsai_decode_lut, "Q2b1 bytes -> trits via 256x4 LUT");
}
