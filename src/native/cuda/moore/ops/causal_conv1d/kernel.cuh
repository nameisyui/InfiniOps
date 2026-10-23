#ifndef INFINI_OPS_MOORE_CAUSAL_CONV1D_KERNEL_CUH_
#define INFINI_OPS_MOORE_CAUSAL_CONV1D_KERNEL_CUH_

#include <musa_runtime.h>

#include "tensor.h"

namespace infini::ops {

template <int kKernelSize>
__global__ void CausalConv1dKernel(const float* qkv, const float* conv_state,
                                   const float* weight, const float* bias,
                                   float* final_conv_state, float* out,
                                   Tensor::Size seq_len,
                                   Tensor::Size channels) {
  const auto i =
      static_cast<Tensor::Size>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= channels) return;

  // Load the whole history before any write, including when states alias.
  float s0 = conv_state[i * (kKernelSize - 1)];
  float s1 = conv_state[i * (kKernelSize - 1) + 1];
  float s2 = conv_state[i * (kKernelSize - 1) + 2];
  const float w0 = weight[i * kKernelSize];
  const float w1 = weight[i * kKernelSize + 1];
  const float w2 = weight[i * kKernelSize + 2];
  const float w3 = weight[i * kKernelSize + 3];
  const float bias_value = bias ? bias[i] : 0.0f;
  for (Tensor::Size t = 0; t < seq_len; ++t) {
    const auto offset = t * channels + i;
    const float x = qkv[offset];
    out[offset] = s0 * w0 + s1 * w1 + s2 * w2 + x * w3 + bias_value;
    s0 = s1;
    s1 = s2;
    s2 = x;
  }
  final_conv_state[i * (kKernelSize - 1)] = s0;
  final_conv_state[i * (kKernelSize - 1) + 1] = s1;
  final_conv_state[i * (kKernelSize - 1) + 2] = s2;
}

}  // namespace infini::ops

#endif
