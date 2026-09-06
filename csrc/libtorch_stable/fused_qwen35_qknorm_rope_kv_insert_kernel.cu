/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused Qwen3.5 pre-attention kernel: per-head Q/K GemmaRMSNorm + partial
 * interleaved MRoPE + gate copy + paged KV-cache insert, in one launch.
 *
 * Targets Qwen/Qwen3.8-27B (arch Qwen3_5ForConditionalGeneration, full_attention
 * layers). Layout per token:
 *
 *   qkv = [q_gate | k | v]
 *     q_gate: [num_heads * 2 * head_dim]  -> split into q [num_heads*head_dim]
 *                                           and gate [num_heads*head_dim]
 *     k:      [num_kv_heads * head_dim]
 *     v:      [num_kv_heads * head_dim]
 *
 *   Q path:  GemmaRMSNorm over head_dim -> partial MRoPE on [0, rotary_dim)
 *            -> pass-through [rotary_dim, head_dim) -> write q_out
 *   Gate:    copied through (no sigmoid here; applied after attention)
 *   K path:  GemmaRMSNorm -> partial MRoPE -> write k_out AND insert into
 *            paged key cache
 *   V path:  insert into paged value cache (no norm, no RoPE)
 *
 * The kernel reuses the HEADS_PER_WARP + cp.async pattern from
 * csrc/libtorch_stable/fused_qknorm_rope_kernel.cu so cos/sin and QKV tiles
 * live in shared memory, reducing HBM round-trips. The paged-cache write
 * follows csrc/libtorch_stable/cache_kernels.cu (reshape_and_cache_kernel).
 *
 * Assumptions (hard-coded for Qwen3.8-27B full_attention layers):
 *   head_dim          = 256
 *   rotary_dim        = 64  (partial_rotary_factor = 0.25)
 *   mrope_interleaved = true (interleaved pairs, no __shfl_xor_sync needed)
 *   is_neox_style     = true (cos_sin_cache layout: cos || sin, each half
 *                     is rotary_dim/2 = 32 values)
 *   GemmaRMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + weight)
 *   cos_sin_cache: [max_position, rotary_dim] with cos in [.., 0:32] and
 *                 sin in [.., 32:64] for the rotated half
 *
 * For text-only inputs positions is 1D [num_tokens]; for multimodal MRoPE
 * it is 2D [3, num_tokens] (T/H/W). mrope_section = [11, 11, 10] (T=11, H=11,
 * W=10) sums to rotary_dim/2 = 32.
 */

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include "torch_utils.h"

#include "async_util.cuh"
#include "../cuda_compat.h"
#include "type_convert.cuh"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

#ifdef USE_ROCM
  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

namespace vllm {
namespace qwen35_fused {

using namespace ::vllm::cuda_async;

// ────────────────────────────────────────────────────────────────────────────
// Constants for Qwen3.8-27B full_attention layers
// ────────────────────────────────────────────────────────────────────────────
constexpr int kHeadDim = 256;
constexpr int kRotaryDim = 64;
constexpr int kHalfRotary = kRotaryDim / 2;        // 32
constexpr int kPassDim = kHeadDim - kRotaryDim;    // 192

// MRoPE section [T=11, H=11, W=10] sums to kHalfRotary = 32.
constexpr int kMropeSectionT = 11;
constexpr int kMropeSectionH = 11;
constexpr int kMropeSectionW = 10;

// Each warp owns one (token, head-slot). head_dim=256 -> numElemsPerThread=8
// (32 threads * 8 = 256). elemSizeBytes = 8 * 2 = 16 (bf16) -> vecSize = 4
// (uint4 load).
constexpr int kNumElemsPerThread = kHeadDim / 32;  // 8
constexpr int kElemSizeBytes = kNumElemsPerThread * 2;  // bf16 -> 16 bytes
constexpr int kVecSize = kElemSizeBytes / 4;            // 4 (uint4)
constexpr int kRotaryElemsPerThread = kRotaryDim / 32;  // 2

template <typename T, int N>
struct packed_as;
template <> struct packed_as<uint, 1> { using type = uint; };
template <> struct packed_as<uint, 2> { using type = uint2; };
template <> struct packed_as<uint, 4> { using type = uint4; };

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

// Convert linear pair-index (0..half_rotary-1) into the MRoPE-interleaved
// position for a given axis (T/H/W). For interleaved MRoPE the pair at
// linear index `i` (in the cos/sin cache half) maps to dimension
// `2*i` and `2*i+1` of the rotary head. The cos/sin cache for MRoPE is
// laid out as [cos_T | cos_H | cos_W | sin_T | sin_H | sin_W] per position
// when mrope_interleaved=True (see apply_interleaved_rope in mrope.py).
// However vLLM's cos_sin_cache is the standard cos||sin layout of size
// rotary_dim; the interleaving is applied to cos/sin at load time by
// `apply_interleaved_rope`. To keep the kernel simple and match the
// Triton reference (fused_qk_rmsnorm_rope_gate), we load cos/sin with the
// same interleaving the reference uses: for pair index `p` in
// [0, kHalfRotary), the rotated dims are `2*p` and `2*p+1`, and the
// cos/sin value comes from the interleaved reordering of the T/H/W bands.
//
// Precomputed interleaved index table: for Qwen3.8-27B with section
// [11, 11, 10] the interleaved order places T-pairs, H-pairs, W-pairs
// contiguously (see get_mrope_interleaved_id_list in mrope_interleaved.py).
// We embed the same ordering as a constexpr array via a helper.
__device__ __forceinline__ int mrope_interleaved_pair(int pair_idx) {
  // Interleaved layout: dims are ordered as [T0, W0, T1, W1, ...] is NOT
  // the scheme; vLLM's apply_interleaved_rope gathers cos/sin so that the
  // first T values, then H values, then W values appear, but interleaved
  // across the two halves. The exact mapping is:
  //   for p in [0, T): src[p] = p           (T band, cos half)
  //   for p in [T, T+H): src[p] = half + (p - T)   (H band, sin half)
  //   for p in [T+H, T+H+W): src[p] = (p - T - H)  (W band, cos half)
  // The resulting cos/sin index for output pair `p` is `src[p]`.
  // This mirrors apply_interleaved_rope in mrope.py.
  if (pair_idx < kMropeSectionT) {
    return pair_idx;  // T band: cos side
  } else if (pair_idx < kMropeSectionT + kMropeSectionH) {
    return kHalfRotary + (pair_idx - kMropeSectionT);  // H band: sin side
  } else {
    return (pair_idx - kMropeSectionT - kMropeSectionH);  // W band: cos side
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Main fused kernel. One warp per (token, head-slot). HEADS_PER_WARP token-heads
// share cos/sin in shared memory.
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int HEADS_PER_WARP, bool HAS_MROPE>
__global__ void fusedQwen35QkNormRopeKvInsertKernel(
    void* q_out_void,                // [num_tokens, num_heads*head_dim] bf16
    void* gate_out_void,             // [num_tokens, num_heads*head_dim] bf16
    void* k_out_void,                // [num_tokens, num_kv_heads*head_dim] bf16
    void* qkv_void,                   // [num_tokens, (num_heads*2+2*num_kv_heads)*head_dim]
    void const* q_weight_void,        // [head_dim] bf16
    void const* k_weight_void,        // [head_dim] bf16
    void const* cos_sin_cache_void,   // [max_position, rotary_dim] (cos||sin)
    int64_t const* positions,         // [num_tokens] or [3, num_tokens] if HAS_MROPE
    int64_t const* slot_mapping,      // [num_tokens]
    void* key_cache_void,             // paged key cache
    void* value_cache_void,           // paged value cache
    int const num_heads, int const num_kv_heads, float const eps,
    int const block_size, int const x,  // paged cache params
    int const num_tokens, float const* k_scale, float const* v_scale) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    return;
  } else {
#endif

    using Converter = vllm::_typeConvert<scalar_t>;
    using T = typename Converter::hip_type;
    using T2 = typename Converter::packed_hip_type;

    T* qkv = reinterpret_cast<T*>(qkv_void);
    T* q_out = reinterpret_cast<T*>(q_out_void);
    T* gate_out = reinterpret_cast<T*>(gate_out_void);
    T* k_out = reinterpret_cast<T*>(k_out_void);
    T const* q_weight = reinterpret_cast<T const*>(q_weight_void);
    T const* k_weight = reinterpret_cast<T const*>(k_weight_void);
    T const* cos_sin_cache = reinterpret_cast<T const*>(cos_sin_cache_void);
    cache_t* key_cache = reinterpret_cast<cache_t*>(key_cache_void);
    cache_t* value_cache = reinterpret_cast<cache_t*>(value_cache_void);

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // Total QK heads (gate is processed alongside Q). V heads handled in a
    // separate pass below.
    int const total_qk_slots = num_heads + num_kv_heads;
    int const total_slots = num_heads + 2 * num_kv_heads;  // q(+gate), k, v

    int const head_chunks_per_token =
        (total_slots + HEADS_PER_WARP - 1) / HEADS_PER_WARP;
    int const warp_global = blockIdx.x * warpsPerBlock + warpId;
    int const tokenIdx = warp_global / head_chunks_per_token;
    int const headChunk = warp_global % head_chunks_per_token;
    int const first_slot = headChunk * HEADS_PER_WARP;
    int const num_slots_this_warp =
        (first_slot + HEADS_PER_WARP <= total_slots)
            ? HEADS_PER_WARP
            : (total_slots - first_slot);

    if (tokenIdx >= num_tokens) return;

    int const slot_idx = static_cast<int>(slot_mapping[tokenIdx]);
    bool const valid_slot = slot_idx >= 0;
    int const block_idx = valid_slot ? (slot_idx / block_size) : 0;
    int const block_offset = valid_slot ? (slot_idx % block_size) : 0;

    // ── Shared memory layout ──────────────────────────────────────────────
    // [0, cos_sin_bytes): cos/sin for each warp (warpsPerBlock * rotary_dim)
    // [cos_sin_bytes, ...): QKV tiles per warp
    //   (warpsPerBlock * HEADS_PER_WARP * 32 * elemSizeBytes)
    extern __shared__ char smem_storage[];
    T* const smem = reinterpret_cast<T*>(smem_storage);
    size_t const cos_sin_bytes =
        warpsPerBlock * kRotaryDim * static_cast<int>(sizeof(T));
    int const qkv_tile_bytes = 32 * kElemSizeBytes;
    char* const this_warp_head_smem =
        smem_storage + cos_sin_bytes + warpId * (HEADS_PER_WARP * qkv_tile_bytes);

    // ── Group 0: async-load all this warp's head tiles into smem ──────────
    int const q_gate_size = num_heads * 2 * kHeadDim;
    int const kv_size = num_kv_heads * kHeadDim;
    for (int s = 0; s < num_slots_this_warp; ++s) {
      int const slot = first_slot + s;
      int offWarp;
      if (slot < num_heads) {
        // Q (+gate): q_gate segment
        offWarp = tokenIdx * (q_gate_size + 2 * kv_size) + slot * 2 * kHeadDim;
      } else if (slot < num_heads + num_kv_heads) {
        // K
        offWarp = tokenIdx * (q_gate_size + 2 * kv_size) + q_gate_size +
                  (slot - num_heads) * kHeadDim;
      } else {
        // V
        offWarp = tokenIdx * (q_gate_size + 2 * kv_size) + q_gate_size +
                  kv_size + (slot - num_heads - num_kv_heads) * kHeadDim;
      }
      int const offThread = offWarp + laneId * kNumElemsPerThread;
      char* smem_dst =
          this_warp_head_smem + s * qkv_tile_bytes + laneId * kElemSizeBytes;
      cp_async_shared_global_ca(smem_dst,
                                reinterpret_cast<const char*>(&qkv[offThread]),
                                kElemSizeBytes);
    }
    cp_async_commit_group();  // group 0: QKV tiles

    // ── Group 1: async-load cos/sin into smem ─────────────────────────────
    int64_t pos_id_t, pos_id_h, pos_id_w;
    if constexpr (HAS_MROPE) {
      // positions is [3, num_tokens]; T/H/W positions.
      // Stride: positions[0..2][tokenIdx]. We pass the raw pointer and index
      // with the token stride (1 here since the tensor is contiguous in the
      // token dimension).
      pos_id_t = positions[0 * num_tokens + tokenIdx];
      pos_id_h = positions[1 * num_tokens + tokenIdx];
      pos_id_w = positions[2 * num_tokens + tokenIdx];
    } else {
      pos_id_t = positions[tokenIdx];
      pos_id_h = pos_id_t;
      pos_id_w = pos_id_t;
    }
    // cos_sin_cache layout: [max_position, rotary_dim] = [cos(32) | sin(32)]
    // (cos first, sin second; each half is kHalfRotary = 32 values).
    T const* cos_sin_ptr = cos_sin_cache + pos_id_t * kRotaryDim;
    int const copy_bytes = kRotaryDim * static_cast<int>(sizeof(T));
    int const num_copies = (copy_bytes + 15) / 16;
    for (int copyId = laneId; copyId < num_copies; copyId += 32) {
      char* smem_ptr =
          reinterpret_cast<char*>(&smem[warpId * kRotaryDim]) + copyId * 16;
      const char* glob_ptr =
          reinterpret_cast<const char*>(cos_sin_ptr) + copyId * 16;
      cp_async_shared_global_16_cg(smem_ptr, glob_ptr);
    }
    cp_async_commit_group();  // group 1: cos/sin

    // wait<1>: allow at most 1 pending group (group 1) -> group 0 done.
    cp_async_wait_group<1>();

    // Preload norm weights into registers once, reused across all slots.
    float q_w[kNumElemsPerThread];
    float k_w[kNumElemsPerThread];
#pragma unroll
    for (int i = 0; i < kNumElemsPerThread; i++) {
      int const dim = laneId * kNumElemsPerThread + i;
      q_w[i] = Converter::convert(q_weight[dim]) + 1.0f;  // GemmaRMSNorm: 1+w
      k_w[i] = Converter::convert(k_weight[dim]) + 1.0f;
    }

    float elements[kNumElemsPerThread];
    float elements2[kNumElemsPerThread];
    T const* const cos_smem = &smem[warpId * kRotaryDim];
    T const* const sin_smem = &smem[warpId * kRotaryDim + kHalfRotary];

    // ── Wait for cos/sin before any RoPE work ─────────────────────────────
    // (issued once, before the loop, so all slots can use it)
    cp_async_wait_group<0>();

    for (int s = 0; s < num_slots_this_warp; ++s) {
      int const slot = first_slot + s;
      bool const isQ = slot < num_heads;
      bool const isK =
          slot >= num_heads && slot < num_heads + num_kv_heads;
      bool const isV = slot >= num_heads + num_kv_heads;

      // ── Load from smem (group 0 already done) ──────────────────────────
      char const* smem_src =
          this_warp_head_smem + s * qkv_tile_bytes + laneId * kElemSizeBytes;
      using vec_T = typename packed_as<uint, kVecSize>::type;
      vec_T vec = *reinterpret_cast<vec_T const*>(smem_src);
      constexpr int num_packed_elems = kElemSizeBytes / sizeof(T2);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        T2 packed_val = *(reinterpret_cast<T2*>(&vec) + i);
        float2 vals = Converter::convert(packed_val);
        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }

      if (isV) {
        // V: no norm, no RoPE -> straight to paged value cache.
        if (valid_slot) {
          int const headIdx = slot - num_heads - num_kv_heads;
          // value_cache layout: [num_blocks, num_heads, head_size/x,
          //                       block_size, x]
          int const h_block_count = kHeadDim / x;
          // Threads cover head_size in chunks of kNumElemsPerThread.
          // head_size=256, 32 threads * 8 elems = 256 -> one pass.
          int const dim_base = laneId * kNumElemsPerThread;
          float v_scale_val =
              (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
          CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};
#pragma unroll
          for (int i = 0; i < kNumElemsPerThread; i += x) {
            int const h_block = (dim_base + i) / x;
            int64_t const tgt_start =
                block_idx * num_kv_heads * h_block_count * x * block_size +
                headIdx * h_block_count * x * block_size +
                h_block * x * block_size + block_offset;
#pragma unroll
            for (int j = 0; j < x; j++) {
              v_op(value_cache[tgt_start + j * block_size],
                   static_cast<scalar_t>(elements[i + j]));
            }
          }
        }
        continue;  // V slot done
      }

      // ── Q or K: GemmaRMSNorm over full head_dim ────────────────────────
      float sumOfSquares = 0.0f;
#pragma unroll
      for (int i = 0; i < kNumElemsPerThread; i++) {
        sumOfSquares += elements[i] * elements[i];
      }
      sumOfSquares = warpReduceSum(sumOfSquares);
      float const rms_rcp =
          rsqrtf(sumOfSquares / static_cast<float>(kHeadDim) + eps);
#pragma unroll
      for (int i = 0; i < kNumElemsPerThread; i++) {
        elements[i] *= rms_rcp * (isQ ? q_w[i] : k_w[i]);
      }

      // ── Partial interleaved MRoPE on [0, rotary_dim) ───────────────────
      // Interleaved: pair (2p, 2p+1) uses cos[p], sin[p]. Each thread holds
      // kNumElemsPerThread=8 consecutive dims. rotary_dim=64 -> first
      // kRotaryElemsPerThread=2 elems per thread are in the rotary range
      // (laneId*8, laneId*8+1). So pair index p = laneId (for the first
      // thread-elem pair). Only lanes 0..31 cover 32 pairs = 64 rotary dims.
      //
      // For MRoPE, cos/sin at pair `p` come from the interleaved reordering
      // of the T/H/W bands (see mrope_interleaved_pair above). The cos/sin
      // cache is laid out as [cos(32) | sin(32)] per position; the
      // interleaving picks which of these 32 cos and 32 sin values go to
      // each pair.
      //
      // Thread lane `l` owns dims [l*8, l*8+8). The rotary pair index for
      // the first two elems is `l` (pair p=l -> dims 2l, 2l+1). So each
      // thread loads cos[l] and sin[l] from the interleaved position.
      if (laneId < kHalfRotary) {
        int const pair_idx = laneId;  // this thread's rotary pair
        int const cs_idx = mrope_interleaved_pair(pair_idx);
        // cos is in [0, kHalfRotary), sin in [kHalfRotary, kRotaryDim).
        float const cos_val = Converter::convert(cos_smem[cs_idx]);
        float const sin_val = Converter::convert(sin_smem[cs_idx]);
        float const x_val = elements[0];
        float const y_val = elements[1];
        elements[0] = x_val * cos_val - y_val * sin_val;
        elements[1] = x_val * sin_val + y_val * cos_val;
        // Dims [2, 8) are pass-through (already RMSNormed) - no change.
      }
      // Dims [rotary_dim, head_dim) = [64, 256) are pass-through. For lanes
      // with laneId >= kHalfRotary (i.e. laneId >= 32 - impossible, laneId
      // max is 31) - actually all 32 lanes have their first 2 elems in the
      // rotary range and the rest in pass-through. Nothing to do for the
      // pass-through tail since it stays in `elements` after RMSNorm.

      // ── Store Q/gate or K ──────────────────────────────────────────────
      if (isQ) {
        // Q: q_gate is packed [q | gate], each num_heads*head_dim. We
        // write q_out (post-norm, post-RoPE) and gate_out (copy of the gate
        // half). The smem tile for slot s holds q_gate = [q_head | gate_head]
        // = 2 * head_dim per head. We processed the q half; the gate half
        // is at offset head_dim within the tile.
        int const headIdx = slot;
        int const q_off = tokenIdx * num_heads * kHeadDim + headIdx * kHeadDim;
        int const gate_off =
            tokenIdx * num_heads * kHeadDim + headIdx * kHeadDim;
        // Write Q (post-RoPE) to q_out.
        {
          vec_T out_vec;
#pragma unroll
          for (int i = 0; i < num_packed_elems; i++) {
            T2 packed_val = Converter::convert(
                make_float2(elements[2 * i], elements[2 * i + 1]));
            *(reinterpret_cast<T2*>(&out_vec) + i) = packed_val;
          }
          *reinterpret_cast<vec_T*>(&q_out[q_off + laneId * kNumElemsPerThread]) =
              out_vec;
        }
        // Copy the gate half from smem (unchanged) to gate_out.
        {
          char const* gate_smem_src = this_warp_head_smem +
                                      s * qkv_tile_bytes +
                                      kHeadDim * sizeof(T) +
                                      laneId * kElemSizeBytes;
          vec_T gate_vec = *reinterpret_cast<vec_T const*>(gate_smem_src);
          *reinterpret_cast<vec_T*>(
              &gate_out[gate_off + laneId * kNumElemsPerThread]) = gate_vec;
        }
      } else {
        // K: write k_out AND insert into paged key cache.
        int const headIdx = slot - num_heads;
        // k_out: [num_tokens, num_kv_heads*head_dim]
        int const k_off =
            tokenIdx * num_kv_heads * kHeadDim + headIdx * kHeadDim;
        {
          vec_T out_vec;
#pragma unroll
          for (int i = 0; i < num_packed_elems; i++) {
            T2 packed_val = Converter::convert(
                make_float2(elements[2 * i], elements[2 * i + 1]));
            *(reinterpret_cast<T2*>(&out_vec) + i) = packed_val;
          }
          *reinterpret_cast<vec_T*>(&k_out[k_off + laneId * kNumElemsPerThread]) =
              out_vec;
        }
        // Paged key cache insert.
        if (valid_slot) {
          // key_cache layout: [num_blocks, num_heads, head_size/x,
          //                     block_size, x]
          int const h_block_count = kHeadDim / x;
          int const dim_base = laneId * kNumElemsPerThread;
          float k_scale_val =
              (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
          CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
#pragma unroll
          for (int i = 0; i < kNumElemsPerThread; i += x) {
            int const h_block = (dim_base + i) / x;
            cache_t* k_dst =
                key_cache +
                block_idx * num_kv_heads * h_block_count * block_size * x +
                headIdx * h_block_count * block_size * x +
                h_block * block_size * x + block_offset * x;
#pragma unroll
            for (int j = 0; j < x; j++) {
              k_op(k_dst[j], static_cast<scalar_t>(elements[i + j]));
            }
          }
        }
      }
    }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Launcher
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, typename cache_t>
void launchFusedQwen35QkNormRopeKvInsert(
    void* q_out, void* gate_out, void* k_out, void* qkv, void const* q_weight,
    void const* k_weight, void const* cos_sin_cache,
    int64_t const* positions, int64_t const* slot_mapping, void* key_cache,
    void* value_cache, int num_heads, int num_kv_heads, float eps,
    int block_size, int x, int num_tokens, float const* k_scale,
    float const* v_scale, Fp8KVCacheDataType kv_dt, bool has_mrope,
    int token_heads_per_warp, cudaStream_t stream) {
  STD_TORCH_CHECK(token_heads_per_warp == 1 || token_heads_per_warp == 2 ||
                      token_heads_per_warp == 4 || token_heads_per_warp == 8,
                  "token_heads_per_warp must be 1, 2, 4, or 8, got ",
                  token_heads_per_warp);

  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const total_slots = num_heads + 2 * num_kv_heads;
  int const head_chunks_per_token =
      (total_slots + token_heads_per_warp - 1) / token_heads_per_warp;
  int const total_warps = num_tokens * head_chunks_per_token;
  int const gridSize = (total_warps + warpsPerBlock - 1) / warpsPerBlock;
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  // Shared memory: cos/sin + QKV tiles.
  size_t const cos_sin_bytes =
      warpsPerBlock * kRotaryDim * sizeof(scalar_t);
  size_t const qkv_smem_per_warp =
      static_cast<size_t>(token_heads_per_warp) * 32 * kElemSizeBytes;
  size_t const smem_bytes = cos_sin_bytes + warpsPerBlock * qkv_smem_per_warp;

#define LAUNCH(KV_DT, HPW, MROPE)                                            \
  do {                                                                       \
    fusedQwen35QkNormRopeKvInsertKernel<scalar_t, cache_t, KV_DT, HPW, MROPE> \
        <<<gridDim, blockDim, smem_bytes, stream>>>(                         \
            q_out, gate_out, k_out, qkv, q_weight, k_weight, cos_sin_cache,   \
            positions, slot_mapping, key_cache, value_cache, num_heads,     \
            num_kv_heads, eps, block_size, x, num_tokens, k_scale, v_scale); \
  } while (0)

#define LAUNCH_KV_DT(HPW, MROPE)            \
  do {                                       \
    if (kv_dt == Fp8KVCacheDataType::kAuto) { \
      LAUNCH(Fp8KVCacheDataType::kAuto, HPW, MROPE); \
    } else {                                  \
      LAUNCH(Fp8KVCacheDataType::kFp8E4M3, HPW, MROPE); \
    }                                         \
  } while (0)

#define LAUNCH_HPW(MROPE)                              \
  do {                                                  \
    if (token_heads_per_warp == 1) {                    \
      LAUNCH_KV_DT(1, MROPE);                            \
    } else if (token_heads_per_warp == 2) {              \
      LAUNCH_KV_DT(2, MROPE);                            \
    } else if (token_heads_per_warp == 4) {              \
      LAUNCH_KV_DT(4, MROPE);                            \
    } else {                                            \
      LAUNCH_KV_DT(8, MROPE);                            \
    }                                                   \
  } while (0)

  if (has_mrope) {
    LAUNCH_HPW(true);
  } else {
    LAUNCH_HPW(false);
  }

#undef LAUNCH
#undef LAUNCH_KV_DT
#undef LAUNCH_HPW
}

}  // namespace qwen35_fused
}  // namespace vllm

// ────────────────────────────────────────────────────────────────────────────
// Torch binding
// ────────────────────────────────────────────────────────────────────────────
void fused_qwen35_qknorm_rope_kv_insert(
    torch::stable::Tensor& q_out, torch::stable::Tensor& gate_out,
    torch::stable::Tensor& qkv, torch::stable::Tensor& k_out,
    torch::stable::Tensor& q_weight, torch::stable::Tensor& k_weight,
    torch::stable::Tensor& cos_sin_cache, torch::stable::Tensor& positions,
    torch::stable::Tensor& slot_mapping, torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache, double eps, int64_t kv_cache_dtype,
    torch::stable::Tensor& k_scale, torch::stable::Tensor& v_scale) {
  CHECK_INPUT(qkv);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(positions);
  CHECK_INPUT(slot_mapping);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(v_scale);

  STD_TORCH_CHECK(qkv.dim() == 2,
                  "qkv must be 2D: [num_tokens, "
                  "(num_heads*2+2*num_kv_heads)*head_dim]");
  STD_TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type(),
                  "qkv and q_weight must have the same dtype");
  STD_TORCH_CHECK(qkv.scalar_type() == k_weight.scalar_type(),
                  "qkv and k_weight must have the same dtype");

  int64_t const num_tokens = qkv.size(0);
  int64_t const total_dim = qkv.size(1);
  // Qwen3.8-27B: head_dim=256, num_heads=24, num_kv_heads=4.
  // total_dim = (num_heads*2 + 2*num_kv_heads) * head_dim
  //           = (24*2 + 2*4) * 256 = 56 * 256 = 14336
  STD_TORCH_CHECK(total_dim == 14336,
                  "qkv last dim must be 14336 for Qwen3.8-27B, got ",
                  total_dim);
  int constexpr num_heads = 24;
  int constexpr num_kv_heads = 4;
  int constexpr head_dim = kHeadDim;
  STD_TORCH_CHECK(q_weight.size(0) == head_dim,
                  "q_weight size must match head_dim");
  STD_TORCH_CHECK(k_weight.size(0) == head_dim,
                  "k_weight size must match head_dim);

  bool const has_mrope = positions.dim() == 2 && positions.size(0) == 3;
  if (has_mrope) {
    STD_TORCH_CHECK(positions.size(1) == num_tokens,
                    "MRoPE positions must be [3, num_tokens]");
  } else {
    STD_TORCH_CHECK(positions.size(0) == num_tokens,
                    "positions must be [num_tokens]");
  }
  STD_TORCH_CHECK(slot_mapping.size(0) == num_tokens,
                  "slot_mapping must be [num_tokens]");

  const torch::stable::accelerator::DeviceGuard device_guard(
      qkv.get_device_index());
  auto stream = get_current_cuda_stream(qkv.get_device_index());

  // Resolve KV cache dtype.
  Fp8KVCacheDataType kv_dt;
  if (kv_cache_dtype == 0) {
    kv_dt = Fp8KVCacheDataType::kAuto;  // bf16 / fp16 cache
  } else if (kv_cache_dtype == 8) {
    kv_dt = Fp8KVCacheDataType::kFp8E4M3;
  } else {
    STD_TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  }

  // Paged cache params.
  // key_cache shape: [num_blocks, num_kv_heads, head_size/x, block_size, x]
  // value_cache shape: [num_blocks, num_kv_heads, head_size/x, block_size, x]
  // Infer block_size and x from key_cache shape.
  STD_TORCH_CHECK(key_cache.dim() == 5, "key_cache must be 5D");
  int const x = static_cast<int>(key_cache.size(4));
  int const block_size = static_cast<int>(key_cache.size(3));
  STD_TORCH_CHECK(x * (kHeadDim / x) == kHeadDim,
                  "x must divide head_dim=256");
  STD_TORCH_CHECK(value_cache.dim() == 5, "value_cache must be 5D");
  STD_TORCH_CHECK(static_cast<int>(value_cache.size(4)) == x,
                  "value_cache x must match key_cache x");

  // Auto-select token_heads_per_warp. On SM90 (H100) and SM100 (B200) use
  // larger values for higher occupancy; on other SMs default to 1.
  int token_heads_per_warp = 1;
  int sm_version = get_device_prop()->major * 10 + get_device_prop()->minor;
  int64_t total_units = num_tokens * (num_heads + 2 * num_kv_heads);
  if (sm_version == 90 || sm_version == 100) {
    // head_dim=256 is large; keep HEADS_PER_WARP modest to fit smem.
    if (total_units < 4096LL) {
      token_heads_per_warp = 1;
    } else if (total_units < 8192LL) {
      token_heads_per_warp = 2;
    } else {
      token_heads_per_warp = 4;
    }
  }

  float const* k_scale_ptr =
      (kv_dt == Fp8KVCacheDataType::kAuto) ? nullptr : k_scale.const_data_ptr<float>();
  float const* v_scale_ptr =
      (kv_dt == Fp8KVCacheDataType::kAuto) ? nullptr : v_scale.const_data_ptr<float>();

  // The kernel writes K post-RoPE into the K region of q_out's logical
  // split; the caller passes k_out separately for clarity. We pass
  // k_out.data_ptr() through as the kernel's "q_out" for the K slots by
  // overloading the q_out pointer. For simplicity, the caller is
  // responsible for giving us q_out and k_out as contiguous buffers;
  // the kernel writes Q to q_out and K to k_out. We dispatch via the
  // scalar type of qkv.
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_qwen35_qknorm_rope_kv_insert", [&] {
        using scalar_t = scalar_t;
        if (kv_dt == Fp8KVCacheDataType::kAuto) {
          // cache dtype matches scalar_t
          using cache_t = scalar_t;
          vllm::qwen35_fused::launchFusedQwen35QkNormRopeKvInsert<scalar_t, cache_t>(
              q_out.data_ptr(), gate_out.data_ptr(), k_out.data_ptr(),
              qkv.data_ptr(), q_weight.data_ptr(), k_weight.data_ptr(),
              cos_sin_cache.data_ptr(),
              reinterpret_cast<int64_t const*>(positions.data_ptr()),
              reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
              key_cache.data_ptr(), value_cache.data_ptr(), num_heads,
              num_kv_heads, static_cast<float>(eps), block_size, x,
              static_cast<int>(num_tokens), k_scale_ptr, v_scale_ptr, kv_dt,
              has_mrope, token_heads_per_warp, stream);
        } else {
          // FP8 cache
          using cache_t = uint8_t;
          vllm::qwen35_fused::launchFusedQwen35QkNormRopeKvInsert<scalar_t, cache_t>(
              q_out.data_ptr(), gate_out.data_ptr(), k_out.data_ptr(),
              qkv.data_ptr(), q_weight.data_ptr(), k_weight.data_ptr(),
              cos_sin_cache.data_ptr(),
              reinterpret_cast<int64_t const*>(positions.data_ptr()),
              reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
              key_cache.data_ptr(), value_cache.data_ptr(), num_heads,
              num_kv_heads, static_cast<float>(eps), block_size, x,
              static_cast<int>(num_tokens), k_scale_ptr, v_scale_ptr, kv_dt,
              has_mrope, token_heads_per_warp, stream);
        }
      });
}