/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_common.h"

at::Tensor
fp8fp8bf16_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // Secret kernel that seems good with small M but large N and K.

  int N = WQ.size(0);
  int K = WQ.size(1);

  if ((K % 256 == 0) && (N % 4 == 0)) {
    return f8f8bf16_rowwise_wrapper<
        64,
        16,
        16,
        256,
        16,
        16,
        1,
        1,
        S<16, 4, 1>,
        S<16, 4, 1>,
        S<1, 16, 1, 4>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        16,
        16>(XQ, WQ, x_scale, w_scale, Y, 1);
  } else if ((K % 16 == 0) && (N % 4 == 0)) {
    return f8f8bf16_rowwise_wrapper<
        64,
        16,
        16,
        256,
        16,
        16,
        1,
        1,
        S<16, 4, 1>,
        S<16, 4, 1>,
        S<1, 16, 1, 4>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        16,
        16>(XQ, WQ, x_scale, w_scale, Y, 1);
  } else if ((K % 8 == 0) && (N % 4 == 0)) {
    return f8f8bf16_rowwise_wrapper<
        64,
        16,
        16,
        256,
        16,
        16,
        1,
        1,
        S<16, 4, 1>,
        S<16, 4, 1>,
        S<1, 16, 1, 4>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        8,
        8>(XQ, WQ, x_scale, w_scale, Y, 1);
  } else if ((K % 2 == 0) && (N % 4 == 0)) {
    return f8f8bf16_rowwise_wrapper<
        64,
        16,
        16,
        256,
        16,
        16,
        1,
        1,
        S<16, 4, 1>,
        S<16, 4, 1>,
        S<1, 16, 1, 4>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        2,
        2>(XQ, WQ, x_scale, w_scale, Y, 1);
  } else {
    return f8f8bf16_rowwise_wrapper<
        64,
        16,
        16,
        256,
        16,
        16,
        1,
        1,
        S<16, 4, 1>,
        S<16, 4, 1>,
        S<1, 16, 1, 4>,
        S<1, 1, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        1,
        1>(XQ, WQ, x_scale, w_scale, Y, 1);
  }
}
