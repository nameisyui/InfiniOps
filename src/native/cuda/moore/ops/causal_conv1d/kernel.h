#ifndef INFINI_OPS_MOORE_CAUSAL_CONV1D_KERNEL_H_
#define INFINI_OPS_MOORE_CAUSAL_CONV1D_KERNEL_H_

#include <cassert>
#include <cstdint>

#include "base/causal_conv1d.h"
#include "native/cuda/moore/ops/causal_conv1d/kernel.cuh"
#include "native/cuda/moore/runtime_.h"

namespace infini::ops {

template <>
class Operator<CausalConv1d, Device::Type::kMoore> : public CausalConv1d {
 public:
  using CausalConv1d::CausalConv1d;

  static bool supports(const Tensor qkv, const Tensor weight, Tensor conv_state,
                       const std::optional<Tensor> bias,
                       const std::optional<Tensor> cu_seqlens,
                       const std::optional<Tensor> initial_state_indices,
                       const std::optional<Tensor> final_state_indices,
                       Tensor out, std::optional<Tensor> final_conv_state) {
    if (cu_seqlens || initial_state_indices || final_state_indices ||
        !final_conv_state || qkv.shape()[0] != 1 || !qkv.numel() ||
        weight.shape()[2] != 4) {
      return false;
    }
    const Views views{qkv,
                      weight,
                      conv_state,
                      bias,
                      cu_seqlens,
                      initial_state_indices,
                      final_state_indices,
                      out,
                      final_conv_state};
    for (const auto& view : views) {
      if (!view) continue;
      if (view->dtype() != DataType::kFloat32) return false;
      size_t stride = 1;
      for (size_t axis = view->ndim(); axis-- > 0;) {
        if (view->size(axis) > 1 &&
            view->stride(axis) != static_cast<ptrdiff_t>(stride)) {
          return false;
        }
        stride *= view->size(axis);
      }
    }
    const auto overlaps = [](const Tensor a, const Tensor b) {
      const auto begin_a = reinterpret_cast<uintptr_t>(a.data());
      const auto begin_b = reinterpret_cast<uintptr_t>(b.data());
      return begin_a < begin_b + b.numel() * sizeof(float) &&
             begin_b < begin_a + a.numel() * sizeof(float);
    };
    for (const auto& input :
         {views[0], views[1], views[2], views[3], views[8]}) {
      if (input && overlaps(out, *input)) return false;
    }
    for (const auto& input : {views[0], views[1], views[3]}) {
      if (input && overlaps(*final_conv_state, *input)) return false;
    }
    return final_conv_state->data() == conv_state.data() ||
           !overlaps(*final_conv_state, conv_state);
  }

  void operator()(const Tensor qkv, const Tensor weight, Tensor conv_state,
                  const std::optional<Tensor> bias,
                  const std::optional<Tensor> cu_seqlens,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_conv_state) const override {
    constexpr int kBlockSize = 128;
    const auto stream =
        static_cast<Runtime<Device::Type::kMoore>::Stream>(stream_);
    CausalConv1dKernel<4>
        <<<(channels_ + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            static_cast<const float*>(qkv.data()),
            static_cast<const float*>(conv_state.data()),
            static_cast<const float*>(weight.data()),
            bias ? static_cast<const float*>(bias->data()) : nullptr,
            static_cast<float*>(final_conv_state->data()),
            static_cast<float*>(out.data()), tokens_, channels_);
    const auto status = musaGetLastError();
    assert(status == musaSuccess && "CausalConv1d MUSA launch failed");
    (void)status;
  }
};

}  // namespace infini::ops

#endif
