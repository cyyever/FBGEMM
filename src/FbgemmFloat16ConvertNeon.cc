/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__aarch64__)

#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmConvert.h"

#include <algorithm>
#include <bit>

namespace fbgemm {

// Scalar loop using __fp16; Clang auto-vectorizes to LDP/FCVTN/STP at -O3,
// which is faster than hand-written NEON intrinsics (vld1q_f32_x4 lowers to
// LD1 4-reg, with worse throughput than paired LDP on Apple Silicon).

void FloatToFloat16_neon(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  if (do_clip) {
    constexpr float FP16_MAX = 65504.f;
    for (size_t i = 0; i < size; ++i) {
      const float v = std::max(-FP16_MAX, std::min(src[i], FP16_MAX));
      dst[i] = std::bit_cast<float16>(static_cast<__fp16>(v));
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      dst[i] = std::bit_cast<float16>(static_cast<__fp16>(src[i]));
    }
  }
}

void Float16ToFloat_neon(const float16* src, float* dst, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    dst[i] = std::bit_cast<__fp16>(src[i]);
  }
}

} // namespace fbgemm

#endif // __aarch64__
