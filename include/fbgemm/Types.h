/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <concepts>
#include <cstdint>

namespace fbgemm {

struct float16 {
  uint16_t val;
};

struct bfloat16 {
  uint16_t val;
};

static_assert(sizeof(float16) == 2);
static_assert(sizeof(bfloat16) == 2);

// Half-precision float concept
template <typename T>
concept FbgemmHalfType = std::same_as<T, float16> || std::same_as<T, bfloat16>;

constexpr int64_t round_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit * unit;
}

constexpr int64_t div_up(int64_t val, int64_t unit) {
  return (val + unit - 1) / unit;
}

} // namespace fbgemm
