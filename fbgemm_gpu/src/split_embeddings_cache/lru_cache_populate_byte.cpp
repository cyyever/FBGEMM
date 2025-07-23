/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

DLL_PUBLIC void lru_cache_populate_byte_cpu(
    const Tensor& weights,
    const Tensor& cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    const Tensor& cache_index_table_map,
    const Tensor& weights_offsets,
    const Tensor& weights_tys,
    const Tensor& D_offsets,
    const Tensor& linear_cache_indices,
    const Tensor& lxu_cache_state,
    const Tensor& lxu_cache_weights,
    int64_t time_stamp,
    const Tensor& lru_state,
    int64_t row_alignment,
    bool gather_cache_stats,
    const std::optional<Tensor>& uvm_cache_stats) {
  return;
}

DLL_PUBLIC void direct_mapped_lru_cache_populate_byte_cpu(
    const Tensor& weights,
    const Tensor& cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    const Tensor& cache_index_table_map,
    const Tensor& weights_offsets,
    const Tensor& weights_tys,
    const Tensor& D_offsets,
    const Tensor& linear_cache_indices,
    const Tensor& lxu_cache_state,
    const Tensor& lxu_cache_weights,
    int64_t time_stamp,
    const Tensor& lru_state,
    const Tensor& lxu_cache_miss_timestamp,
    int64_t row_alignment,
    bool gather_cache_stats,
    const std::optional<Tensor>& uvm_cache_stats) {
  return;
}

} // namespace fbgemm_gpu
