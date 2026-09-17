// SPDX-License-Identifier: Apache-2.0
// In-place 4-bit GPTQ packing transform for the RX 7900 XTX appliance.

#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

namespace {
constexpr int kThreads = 256;

__global__ void shuffle_4bit_kernel(uint32_t* qweight, int64_t words, int64_t columns) {
  const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) {
    return;
  }
  for (int64_t word = 0; word < words; ++word) {
    uint32_t packed = qweight[word * columns + column];
    uint32_t shuffled = 0;
#pragma unroll
    for (int bit = 0; bit < 4; ++bit) {
      const uint32_t even = packed & 0x0f;
      const uint32_t odd = (packed & 0xf0) >> 4;
      packed >>= 8;
      shuffled |= even << (bit * 4);
      shuffled |= odd << (bit * 4 + 16);
    }
    qweight[word * columns + column] = shuffled;
  }
}
}  // namespace

void gptq_shuffle_rdna3(torch::Tensor qweight, torch::Tensor g_idx) {
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA/HIP tensor");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
  TORCH_CHECK(qweight.dim() == 2, "qweight must have shape [K/8, N]");
  TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
  TORCH_CHECK(
      g_idx.numel() == 0,
      "7900xtx-qwen38-27b supports only identity g_idx; act-order GPTQ is outside the target contract");

  const at::cuda::OptionalCUDAGuard device_guard(qweight.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto columns = qweight.size(1);
  const dim3 block(kThreads);
  const dim3 grid((columns + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      shuffle_4bit_kernel,
      grid,
      block,
      0,
      stream,
      static_cast<uint32_t*>(qweight.data_ptr()),
      qweight.size(0),
      columns);
}
