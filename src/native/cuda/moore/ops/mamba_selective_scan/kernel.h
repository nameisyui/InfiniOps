#ifndef INFINI_OPS_MOORE_MAMBA_SELECTIVE_SCAN_KERNEL_H_
#define INFINI_OPS_MOORE_MAMBA_SELECTIVE_SCAN_KERNEL_H_

#include <cassert>

// clang-format off
#include "native/cuda/moore/polyfills.cuh"
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/mamba_selective_scan/kernel.h"

namespace infini::ops {

struct MambaScanMooreBackend : Runtime<Device::Type::kMoore> {
  static void CheckLaunch() {
    const auto status = musaGetLastError();
    assert(status == musaSuccess && "MambaSelectiveScan MUSA launch failed");
    (void)status;
  }
};

template <>
class Operator<MambaSelectiveScan, Device::Type::kMoore>
    : public CudaMambaSelectiveScan<MambaScanMooreBackend> {
 public:
  using CudaMambaSelectiveScan<MambaScanMooreBackend>::CudaMambaSelectiveScan;
};

}  // namespace infini::ops

#endif
