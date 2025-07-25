/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{% set wdesc =  "weighted" if weighted else "unweighted" %}

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>

#include "fbgemm_gpu/utils/cpu_utils.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm/FbgemmEmbedding.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

#if defined(__x86_64__) || defined(__i386__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#include <emmintrin.h>
#endif
#include <cstring>

using namespace fbgemm_gpu;

namespace {

using Tensor = at::Tensor;

inline uint32_t pruned_hash_function(uint32_t h) {
    // MurmorHash3 32-bit mixing function.
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

inline uint64_t pruned_hash_function(uint64_t k) {
  // MurmorHash3 64-bit mixing function.
  k ^= k >> 33;
  k *= (0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= (0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

} // namespace

void pruned_hashmap_insert_{{ wdesc }}_cpu(
    Tensor indices,
    Tensor dense_indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets) {
    TENSOR_ON_CPU(indices);
    TENSOR_ON_CPU(dense_indices);
    TENSOR_ON_CPU(offsets);
    TENSOR_ON_CPU(hash_table);
    TENSOR_ON_CPU(hash_table_offsets);
    TENSORS_HAVE_SAME_SCALAR_TYPE(indices, offsets);

    const int32_t T = hash_table_offsets.size(0) - 1;
    const int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);

    AT_DISPATCH_INDEX_TYPES(hash_table.scalar_type(), "pruned_hashmap_insert_{{ wdesc }}_cpu_0", [&] {
        using hash_t = index_t;

        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "pruned_hashmap_insert_{{ wdesc }}_cpu_1", [&] {
            using uidx_t =
                std::conditional_t<std::is_same_v<index_t, int64_t>, uint64_t, uint32_t>;

            const auto* indices_acc = indices.data_ptr<index_t>();
            const auto* dense_indices_acc = dense_indices.data_ptr<index_t>();
            const auto* offsets_acc = offsets.data_ptr<index_t>();

            auto hash_table_acc = hash_table.accessor<hash_t, 2>();
            const auto hash_table_offsets_acc = hash_table_offsets.accessor<int64_t, 1>();

            for (const auto t : c10::irange(T)) {
                const auto table_start = hash_table_offsets_acc[t];
                const auto table_end = hash_table_offsets_acc[t + 1];
                if (table_start == table_end) {
                    continue;
                }
                const auto capacity = table_end - table_start;

                for (const auto b : c10::irange(B)) {
                    const auto indices_start = offsets_acc[t * B + b];
                    const auto indices_end = offsets_acc[t * B + b + 1];
                    const auto L = indices_end - indices_start;

                    for (const auto l : c10::irange(L)) {
                        const auto idx = indices_acc[indices_start + l];
                        const auto dense_idx = dense_indices_acc[indices_start + l];
                        if (dense_idx == -1) {
                            // -1 means this row has been pruned, do not insert it.
                            continue;
                        }

                        auto slot = pruned_hash_function(static_cast<uidx_t>(idx)) % capacity;
                        while (true) {
                            const auto ht_idx = table_start + static_cast<int64_t>(slot);
                            const auto slot_sparse_idx = hash_table_acc[ht_idx][0];

                            // Empty slot
                            if (slot_sparse_idx == -1) {
                                hash_table_acc[ht_idx][0] = static_cast<hash_t>(idx);
                                hash_table_acc[ht_idx][1] = static_cast<hash_t>(dense_idx);
                                break;
                            }

                            // Already exists (shouldn't happen in practice)
                            if (slot_sparse_idx == idx) {
                                hash_table_acc[ht_idx][1] = static_cast<hash_t>(dense_idx);
                                break;
                            }

                            // Linear probe
                            slot = (slot + 1) % capacity;
                        }
                    }
                }
            }
        });
    });

    return;
}

{% for nobag in [True, False] %}
{% if not nobag or not weighted %}
Tensor int_nbit_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    {% if not nobag %}
    Tensor D_offsets,
    int64_t total_D,
    {% else %}
    const int64_t D,
    {% endif %}
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t row_alignment,
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    int64_t output_dtype,
    int64_t fp8_exponent_bits,
    int64_t fp8_exponent_bias,
    bool scale_bias_last
) {
    TENSOR_ON_CPU(dev_weights);
    TENSOR_ON_CPU(uvm_weights);
    TENSOR_ON_CPU(weights_placements);
    TENSOR_ON_CPU(weights_offsets);
    TENSOR_ON_CPU(weights_tys);
    {% if not nobag %}
    TENSOR_ON_CPU(D_offsets);
    {% endif %}
    TENSOR_ON_CPU(indices);
    TENSOR_ON_CPU(offsets);
    {% if weighted %}
    TENSOR_EMPTY_OR_ON_CPU(indice_weights);
    {% endif %}

    {% if not nobag %}
    const int32_t T = D_offsets.numel() - 1;
    {% else %}
    const int32_t total_L = indices.numel();
    const int32_t T = weights_offsets.numel();
    {% endif %}
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);
    {% if not nobag %}
    TORCH_CHECK(total_D > 0);
    {% else %}
    TORCH_CHECK(D > 0);
    {% endif %}
    bool pinned_memory = false;
    if (at::Context::hasCUDA() && at::getNumGPUs() > 0) {
      pinned_memory = true;
    }

    Tensor output;
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 || o_dtype == SparseType::INT8 || o_dtype == SparseType::BF16 || o_dtype == SparseType::INT4);
    bool output_is_bf16 = o_dtype == SparseType::BF16;
    bool output_is_int8 = o_dtype == SparseType::INT8;
    bool output_is_int4 = o_dtype == SparseType::INT4;
    {% if not nobag %}
    const int kINT8QparamsBytes = 8;
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
      total_adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)).pinned_memory(pinned_memory));
    {% else %}
    constexpr int kINT8QparamsBytes = 4; // no bag int8 output aligns with fbgemm weights storage size and layout
    constexpr int kINT4QparamsElems = 8; // scale + bias takes 4 bytes which are 8 int4 elements
    int64_t adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
      adjusted_D += kINT8QparamsBytes;
    } else if (o_dtype == SparseType::INT4) {
      adjusted_D += kINT4QparamsElems;
    }
    output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)).pinned_memory(pinned_memory));

    {% endif %}


    if (B == 0) {
        return output;
    }

    const int32_t* weights_placements_ptr = weights_placements.data_ptr<int32_t>();
    const uint8_t* weights_acc;

    const auto* weights_tys_acc = weights_tys.data_ptr<uint8_t>();

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "intn_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", [&] {
        {% if weighted %}
        const float* indice_weights_acc = indice_weights.data_ptr<float>();
        {% endif %}

        using float16 = uint16_t;
        using bfloat16 = uint16_t;
        using int8 = uint8_t;
        using base_fbgemm_out_t = typename std::conditional<
            std::is_same<output_t, at::Half>::value,
            float16,
            std::conditional<std::is_same<output_t, at::BFloat16>::value, bfloat16, std::conditional<std::is_same<output_t, float>::value, float, int8>::type> ::type >::type;
        using other_fbgemm_out_t = typename std::conditional<
            std::is_same<output_t, at::Half>::value,
            float16,
            std::conditional<std::is_same<output_t, at::BFloat16>::value, bfloat16, float>::type> ::type;
        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "int_nbit_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_", [&] {
            const auto* indices_acc = indices.data_ptr<index_t>();
            const auto* offsets_acc = offsets.data_ptr<index_t>();
            const auto* weights_offsets_acc = weights_offsets.data_ptr<int64_t>();

            auto* output_acc = output.data_ptr<output_t>();

            for (const auto t : c10::irange(T)) {
                {% if not nobag %}
                const auto* D_offsets_acc = D_offsets.data_ptr<int32_t>();
                const int32_t D_start = D_offsets_acc[t];
                const int32_t D_end = D_offsets_acc[t + 1];
                const int32_t D = D_end - D_start;
                {% else %}
                const int32_t elems_D = (o_dtype == SparseType::INT4) ? at::divup(adjusted_D, 2) : adjusted_D;
                const int32_t D_start = offsets_acc[t * B] * elems_D;
                {% endif %}

                const auto placement = static_cast<PlacementType>(weights_placements_ptr[t]);
                TORCH_CHECK(placement != PlacementType::DEVICE);
                const auto& weight_tensor = (placement == PlacementType::HOST) ? dev_weights : uvm_weights;
                weights_acc = weight_tensor.data_ptr<uint8_t>();
                const uint8_t* weights = &weights_acc[weights_offsets_acc[t]];
                const auto weight_ty = static_cast<SparseType>(weights_tys_acc[t]);
                if (output_is_int8) {
                    TORCH_CHECK(weight_ty == SparseType::INT8, "int8 output are only supported for int8 weights");
                }
                const int32_t scale_bias_size = (weight_ty == SparseType::INT8 && scale_bias_last) ? 8 : 4;
                // default to 1 byte alignment for CPU TBE
                const int32_t D_bytes = nbit::padded_row_size_in_bytes(D, weight_ty, row_alignment, scale_bias_size);

                int tt;
                for (tt = t + 1; tt < T && weights_offsets_acc[tt] == weights_offsets_acc[t]; ++tt);
                const size_t num_rows = ((tt == T ? weight_tensor.numel() : weights_offsets_acc[tt]) - weights_offsets_acc[t]) / D_bytes;
                const index_t* offsets_begin_ptr = offsets_acc + t * B;

                bool success = true;
                const bool has_weight = {{ "true" if weighted else "false" }};
                const bool normalize_by_lengths = static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN;

                const index_t index_size = offsets_acc[(t + 1) * B] - *offsets_begin_ptr;
                const int32_t output_stride = {{ "total_D" if not nobag else "adjusted_D" }};

                {% if nobag %}
                // Create virtual offsets for the nobag case. Lengths are all ones.
                const auto offsets_nobag = at::arange(*offsets_begin_ptr, offsets_acc[(t + 1) * B] + 1, offsets.options());
                const index_t* offsets_nobag_ptr = offsets_nobag.data_ptr<index_t>();
                TORCH_CHECK(offsets_nobag.numel() == index_size + 1);
                TORCH_CHECK(offsets_nobag_ptr[index_size] - offsets_nobag_ptr[0] == index_size);
                {% endif %}

                const float* indice_weights_ptr = nullptr;
                // int8/int4 output only enabled for nobag case
                const bool nobag_op = {{ "false" if not nobag else "output_is_int8 || output_is_int4" }};
                {% if weighted %}
                indice_weights_ptr = indice_weights_acc + *offsets_begin_ptr;
                {% endif %}

                {% macro generate_and_exec_kernel(weight_type, use_base, use_nbit, use_fp8) %}
                {% set has_asmjit = use_base or use_nbit %}
                {% set kernel_name = "GenerateEmbeddingSpMDMWithStrides"
                    if use_base else ("GenerateEmbeddingSpMDMNBitWithStrides"
                    if use_nbit else "GenerateEmbeddingSpMDMFP8WithStrides")
                 %}
                using fbgemm_out_t = {{ "base_fbgemm_out_t" if use_base or use_nbit else "other_fbgemm_out_t" }};
                {% if use_nbit %}
                const int output_bit_rate = output_is_int4 ? 4 : sizeof(fbgemm_out_t) * 8;
                {% endif %}
                // TODO: merge nobag int8 path with normal asmjit dispatch
                {% if nobag %}
                    const index_t* offset_ptr = (output_is_int8)? offsets_begin_ptr: offsets_nobag_ptr;
                {% else %}
                    const index_t* offset_ptr = offsets_begin_ptr;
                {% endif %}
                const auto kernel = fbgemm::{{ kernel_name }}<
                    {% if use_base %}
                    {{ weight_type }},
                    {% endif %}
                    index_t,
                    index_t,
                    {% if has_asmjit %}
                    fbgemm_out_t,
                    /*THREAD_LOCAL=*/true
                    {% else %}
                    fbgemm_out_t
                    {% endif %}
                >(
                    {% if use_nbit %}
                    /*input_bit_rate=*/bit_rate,
                    {% endif %}
                    D,
                    {% if has_asmjit %}
                    has_weight,
                    {% endif %}
                    normalize_by_lengths,
                    {% if has_asmjit %}
                    /*prefetch=*/16,
                    {% endif %}
                    /*is_weight_positional=*/false,
                    /*use_offsets=*/true,
                    /*output_stride=*/output_stride,
                    /*input_stride=*/D_bytes / sizeof({{ weight_type }}),
                    {% if use_fp8 %}
                    /*exponent_bits=*/fp8_exponent_bits,
                    /*exponent_bias=*/fp8_exponent_bias,
                    {% endif %}
                    {% if has_asmjit %}
                    /*scale_bias_last=*/scale_bias_last,
                    {% endif %}
                    {% if use_base %}
                    /*no_bag=*/nobag_op,
                    {% endif %}
                    /*is_bf16_out=*/output_is_bf16
                    {% if use_nbit %}
                    ,/*no_bag=*/nobag_op,
                    /*output_bit_rate=*/output_bit_rate
                    {% endif %}
                );
                success = kernel(
                    {{ "B" if not nobag else "index_size"}},
                    index_size,
                    num_rows,
                    reinterpret_cast<const {{ weight_type }}*>(weights),
                    indices_acc + *offsets_begin_ptr,
                    offset_ptr,
                    indice_weights_ptr,
                    reinterpret_cast<fbgemm_out_t*>(output_acc + D_start));
                {% endmacro %}

                if (weight_ty == SparseType::FP32) {
                    {{ generate_and_exec_kernel("float", True, False, False) }}
                } else if (weight_ty == SparseType::FP16) {
                    {{ generate_and_exec_kernel("float16", True, False, False) }}
                } else if (weight_ty == SparseType::INT8) {
                    {{ generate_and_exec_kernel("uint8_t", True, False, False) }}
                } else if (weight_ty == SparseType::FP8) {
                    assert(fp8_exponent_bits > 0 && fp8_exponent_bias > 0);
                    {{ generate_and_exec_kernel("uint8_t", False, False, True) }}
                } else if (weight_ty == SparseType::INT4 || weight_ty == SparseType::INT2) {
                    int bit_rate;
                    switch (weight_ty) {
                        case SparseType::INT4 :
                          bit_rate = 4;
                          break;
                        case SparseType::INT2 :
                          bit_rate = 2;
                          break;
                        default:
                          throw std::logic_error(
                              "Unsupported SparseType: " + std::to_string(static_cast<int>(weight_ty)));
                    }
                    {{ generate_and_exec_kernel("uint8_t", False, True, False) }}
                } else {
                    throw std::logic_error(
                        "Unsupported SparseType: " + std::to_string(static_cast<int>(weight_ty)));
                }
                if (!success) {
                    fbgemm_gpu::report_embedding_error(
                        t,
                        B,
                        0,
                        B,
                        offsets_acc,
                        indices_acc,
                        num_rows,
                        /*allow_minus_one=*/true);
                }
            }
            return;
        });
    });
    return output;
}
{% endif %} // if not nobag or not weighted
{% endfor %} // for nobag in [True, False]

Tensor pruned_hashmap_lookup_{{ wdesc }}_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets) {
    TENSOR_ON_CPU(indices);
    TENSOR_ON_CPU(offsets);
    TENSOR_ON_CPU(hash_table);
    TENSOR_ON_CPU(hash_table_offsets);
    TENSORS_HAVE_SAME_SCALAR_TYPE(indices, offsets);

    int32_t T = hash_table_offsets.size(0) - 1;
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);

    auto dense_indices = empty_like(indices);

    AT_DISPATCH_INDEX_TYPES(hash_table.scalar_type(), "pruned_hashmap_lookup_{{ wdesc }}_cpu_0", [&] {
        using hash_t = index_t;

        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "pruned_hashmap_lookup_{{ wdesc }}_cpu_1", [&] {
            using utdx_t =
                std::conditional_t<std::is_same_v<index_t, int64_t>, uint64_t, uint32_t>;

            const auto* indices_acc = indices.data_ptr<index_t>();
            auto* dense_indices_acc = dense_indices.data_ptr<index_t>();
            const auto* offsets_acc = offsets.data_ptr<index_t>();

            const auto hash_table_acc = hash_table.accessor<hash_t, 2>();
            const auto hash_table_offsets_acc = hash_table_offsets.accessor<int64_t, 1>();

            for (const auto t : c10::irange(T)) {
                const auto table_start = hash_table_offsets_acc[t];
                const auto table_end = hash_table_offsets_acc[t + 1];
                const auto capacity = table_end - table_start;

                for (const auto b : c10::irange(B)) {
                    const auto indices_start = offsets_acc[t * B + b];
                    const auto indices_end = offsets_acc[t * B + b + 1];
                    const auto L = indices_end - indices_start;

                    if (table_start == table_end) {
                        for (const auto l : c10::irange(L)) {
                            dense_indices_acc[indices_start + l] = indices_acc[indices_start + l];
                        }

                    } else {
                        for (const auto l : c10::irange(L)) {
                            const auto idx = indices_acc[indices_start + l];
                            auto slot = pruned_hash_function(static_cast<utdx_t>(idx)) % capacity;

                            while (true) {
                                const auto ht_idx = table_start + static_cast<int64_t>(slot);
                                const auto slot_sparse_idx = hash_table_acc[ht_idx][0];

                                // Empty slot
                                if (slot_sparse_idx == -1) {
                                    dense_indices_acc[indices_start + l] = -1;
                                    break;
                                }
                                // Already exists
                                if (slot_sparse_idx == idx) {
                                    dense_indices_acc[indices_start + l] = static_cast<index_t>(hash_table_acc[ht_idx][1]);
                                    break;
                                }

                                // Linear probe
                                slot = (slot + 1) % capacity;
                            }
                        }
                    }
                }
            }
        });
    });

    return dense_indices;
}

{% if not weighted %}

Tensor pruned_array_lookup_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets) {
    TENSOR_ON_CPU(indices);
    TENSOR_ON_CPU(offsets);
    TENSOR_ON_CPU(index_remappings);
    TENSOR_ON_CPU(index_remappings_offsets);
    TENSORS_HAVE_SAME_SCALAR_TYPE(indices, offsets);

    int32_t T = index_remappings_offsets.size(0) - 1;
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);

    auto dense_indices = empty_like(indices);

    AT_DISPATCH_INDEX_TYPES(index_remappings.scalar_type(), "pruned_array_lookup_cpu_0", [&] {
        using remap_t = index_t;

        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "pruned_array_lookup_cpu_1", [&] {
            const auto* indices_acc = indices.data_ptr<index_t>();
            auto* dense_indices_acc = dense_indices.data_ptr<index_t>();
            const auto* offsets_acc = offsets.data_ptr<index_t>();

            const auto index_remappings_acc = index_remappings.data_ptr<remap_t>();
            const auto index_remappings_offsets_acc = index_remappings_offsets.data_ptr<int64_t>();

            at::parallel_for(0, T, 1, [&](int64_t begin, int64_t end) {
            for (const auto t : c10::irange(begin, end)) {
                const auto index_remappings_start = index_remappings_offsets_acc[t];
                const auto index_remappings_end = index_remappings_offsets_acc[t + 1];
                const auto capacity = index_remappings_end - index_remappings_start;

                const auto indices_start = offsets_acc[t * B];
                const auto indices_end = offsets_acc[(t + 1) * B];

                if (capacity > 0) {
                    for (const auto i : c10::irange(indices_start, indices_end)) {
                        auto idx = indices_acc[i];
                        dense_indices_acc[i] = static_cast<index_t>(index_remappings_acc[index_remappings_start + idx]);
                    }
                } else {
                    std::memcpy(
                        dense_indices_acc + indices_start,
                        indices_acc + indices_start,
                        (indices_end - indices_start) * sizeof(index_t));
                }
            }
            });
        });
    });

    return dense_indices;
}

{% endif %}
// clang-format on
