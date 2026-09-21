#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
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
  TORCH_CHECK(lut.scalar_type() == at::kChar, "lut must have dtype int8");
  TORCH_CHECK(lut.is_cuda(), "lut must be a CUDA tensor");
  TORCH_CHECK(lut.device() == packed.device(),
              "lut and packed must be on the same CUDA device");
  TORCH_CHECK(lut.is_contiguous(), "lut must be contiguous");
  TORCH_CHECK(lut.dim() == 2 && lut.size(0) == 256 && lut.size(1) == 4 &&
                  lut.numel() == 256 * 4,
              "lut must have shape (256, 4)");
  auto out = at::empty({packed.numel(), 4},
                       packed.options().dtype(at::kChar));
  const int64_t n = packed.numel();
  if (n == 0) return out;
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  c10::cuda::CUDAGuard device_guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  bonsai_decode_lut_kernel<<<blocks, threads, 0, stream>>>(
      packed.data_ptr<uint8_t>(), lut.data_ptr<int8_t>(), out.data_ptr<int8_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <typename scalar_t, int TILE_N, int WARPS>
__global__ void bonsai_q2b1_gemm_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const at::Half* __restrict__ scales,
    at::BFloat16* __restrict__ out,
    int64_t m,
    int64_t n,
    int64_t k) {
  __shared__ int8_t smem_lut[256][4];

  const int tid = threadIdx.x;
  for (int i = tid; i < 256 * 4; i += blockDim.x) {
    const int byte = i / 4;
    const int slot = i % 4;
    const int code = (byte >> (2 * slot)) & 3;
    smem_lut[byte][slot] = static_cast<int8_t>(
        code == 1 ? 1 : (code == 2 ? -1 : 0));
  }
  __syncthreads();

  const int lane = tid % 32;
  const int warp = tid / 32;
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t packed_stride = k / 4;
  const int64_t scale_stride = k / 128;

  for (int tile_row = warp; tile_row < TILE_N; tile_row += WARPS) {
    const int64_t output_row =
        static_cast<int64_t>(blockIdx.x) * TILE_N + tile_row;
    float acc = 0.0f;
    if (row < m && output_row < n) {
      for (int64_t group = 0; group < scale_stride; ++group) {
        float scale = 0.0f;
        if (lane == 0) {
          scale = static_cast<float>(
              scales[output_row * scale_stride + group]);
        }
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const int64_t byte_idx = group * 32 + lane;
        const uint8_t byte = packed[output_row * packed_stride + byte_idx];
        const int64_t k_idx = byte_idx * 4;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
          acc += static_cast<float>(x[row * k + k_idx + slot]) *
                 static_cast<float>(smem_lut[byte][slot]) * scale;
        }
      }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0 && row < m && output_row < n) {
      out[row * n + output_row] = static_cast<at::BFloat16>(acc);
    }
  }
}

template <typename scalar_t, int TILE_N, int WARPS>
static void launch_bonsai_q2b1_gemm_variant(torch::Tensor x,
                                            torch::Tensor packed,
                                            torch::Tensor scales,
                                            torch::Tensor out) {
  const int64_t m = x.size(0);
  const int64_t n = packed.size(0);
  const int64_t k = x.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned int>((n + TILE_N - 1) / TILE_N),
                  static_cast<unsigned int>(m));
  const dim3 block(WARPS * 32);

  bonsai_q2b1_gemm_kernel<scalar_t, TILE_N, WARPS>
      <<<grid, block, 0, stream>>>(
          x.data_ptr<scalar_t>(), packed.data_ptr<uint8_t>(),
          scales.data_ptr<at::Half>(), out.data_ptr<at::BFloat16>(), m, n, k);
}

template <typename scalar_t>
static void launch_bonsai_q2b1_gemm(torch::Tensor x,
                                     torch::Tensor packed,
                                     torch::Tensor scales,
                                     torch::Tensor out,
                                     int64_t config_id) {
  c10::cuda::CUDAGuard device_guard(x.device());

  switch (config_id) {
    case 0:
      launch_bonsai_q2b1_gemm_variant<scalar_t, 64, 2>(
          x, packed, scales, out);
      break;
    case 1:
      launch_bonsai_q2b1_gemm_variant<scalar_t, 128, 4>(
          x, packed, scales, out);
      break;
    case 2:
      launch_bonsai_q2b1_gemm_variant<scalar_t, 256, 8>(
          x, packed, scales, out);
      break;
    default:
      TORCH_CHECK(false, "config_id must be 0, 1, or 2");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static torch::Tensor bonsai_q2b1_gemm(torch::Tensor x,
                                      torch::Tensor packed,
                                      torch::Tensor scales,
                                      int64_t config_id) {
  TORCH_CHECK(config_id >= 0 && config_id <= 2,
              "config_id must be 0, 1, or 2");
  TORCH_CHECK(x.is_cuda() && packed.is_cuda() && scales.is_cuda(),
              "x, packed, and scales must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && packed.dim() == 2 && scales.dim() == 2,
              "x, packed, and scales must be 2D tensors");
  TORCH_CHECK(x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kBFloat16,
              "x must have dtype float32 or bfloat16");
  TORCH_CHECK(packed.scalar_type() == at::kByte,
              "packed must have dtype uint8");
  TORCH_CHECK(scales.scalar_type() == at::kHalf,
              "scales must have dtype float16");
  TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() &&
                  scales.is_contiguous(),
              "x, packed, and scales must be contiguous");
  TORCH_CHECK(x.device() == packed.device() && x.device() == scales.device(),
              "x, packed, and scales must be on the same CUDA device");

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = packed.size(0);
  TORCH_CHECK(m <= 64, "M must be <= 64 for q2b1_gemm");
  TORCH_CHECK(k % 128 == 0, "K must be divisible by 128");
  TORCH_CHECK(packed.size(1) == k / 4,
              "packed must have shape (N, K // 4)");
  TORCH_CHECK(scales.size(0) == n && scales.size(1) == k / 128,
              "scales must have shape (N, K // 128)");

  auto out = at::empty({m, n}, x.options().dtype(at::kBFloat16));
  if (m == 0 || n == 0) return out;

  AT_DISPATCH_SWITCH(
      x.scalar_type(),
      "bonsai_q2b1_gemm",
      AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
        launch_bonsai_q2b1_gemm<float>(x, packed, scales, out, config_id);
      })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        launch_bonsai_q2b1_gemm<at::BFloat16>(x, packed, scales, out,
                                              config_id);
      }));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bonsai_decode_lut", &bonsai_decode_lut, "Q2b1 bytes -> trits via 256x4 LUT");
  m.def("bonsai_q2b1_gemm", &bonsai_q2b1_gemm,
        "Fused Q2b1 GEMM with a shared LUT");
}
