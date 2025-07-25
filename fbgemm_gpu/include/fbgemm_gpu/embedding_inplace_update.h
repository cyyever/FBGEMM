/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <torch/torch.h>

#include "fbgemm_gpu/embedding_common.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

/**
 * Embedding tables inplace updates with absolute values (idempotent guarantee)
 *
 * dev_weights: the loaded tables on device in TBE format
 * uvm_weights: the loaded tables on UVM in TBE format
 * weights_placements: placements for each table
 * weights_offsets: physical offsets for each table
 * weights_tys: weight types for each table
 * D_offsets: table dimensions
 * update_weights: new update weights tensor in TBE format
 * update_table_idx: table indices for every new row
 * update_row_idx: row indices for every new row
 * update_offsets: offsets of new update weights
 * row_alignment: alignment byte for embedding row
 * lxu_cache_weights: the loaded cache weights
 * lxu_cache_locations: the loaded cache location info
 *
 * it's guaranteed from upper service level that each row of table will
 * only receive one update at a time.
 *
 * This function has embedding update parameters (update_weights,
 * update_table_idx, updata_offsets) and delta embedding weights
 * on the CUDA devices.
 */
void embedding_inplace_update_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor update_weights,
    Tensor update_table_idx,
    Tensor update_row_idx,
    Tensor update_offsets,
    const int64_t row_alignment,
    std::optional<Tensor> lxu_cache_weights = std::nullopt,
    std::optional<Tensor> lxu_cache_locations = std::nullopt);

void embedding_inplace_update_single_placement_cuda(
    Tensor& dev_weights,
    Tensor& uvm_weights,
    const PlacementType& weights_placement,
    const Tensor& weights_offsets,
    const Tensor& weights_tys,
    const Tensor& D_offsets,
    const Tensor& update_weights,
    const Tensor& update_table_idx,
    const Tensor& update_row_idx,
    const Tensor& update_offsets,
    const int64_t row_alignment,
    std::optional<Tensor> lxu_cache_weights = std::nullopt,
    std::optional<Tensor> lxu_cache_locations = std::nullopt);

void embedding_inplace_update_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor update_weights,
    Tensor update_table_idx,
    Tensor update_row_idx,
    Tensor update_offsets,
    const int64_t row_alignment,
    std::optional<Tensor> lxu_cache_weights =
        std::nullopt, // Not used, to match cache interface for CUDA op
    std::optional<Tensor> lxu_cache_locations =
        std::nullopt // Not used, to match cache interface for CUDA op
);

void dram_kv_embedding_inplace_update_cpu(
    torch::jit::Module* tbe_module,
    std::string tbe_module_update_func_name,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor update_weights,
    Tensor update_table_idx,
    Tensor update_row_idx,
    Tensor update_offsets,
    const int64_t row_alignment);

/**
 * Index remapping function that returns the remapped indices.
 *
 * Args:
 *    update_row_indices: row indices for every new row
 *    update_table_indices: table indices for every new row
 *    index_remappings: concated index remapping for every embedding table
 *    index_remappings_offsets: offset for each embedding table
 *
 * Returns:
 *    remapped indices for each new row.
 */
Tensor pruned_array_lookup_from_row_idx_cuda(
    const Tensor& update_row_indices,
    const Tensor& update_table_indices,
    const Tensor& index_remappings,
    const Tensor& index_remappings_offsets);

Tensor pruned_array_lookup_from_row_idx_cpu(
    const Tensor& update_row_indices,
    const Tensor& update_table_indices,
    const Tensor& index_remappings,
    const Tensor& index_remappings_offsets);

} // namespace fbgemm_gpu
