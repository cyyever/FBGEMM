# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional

import torch

import triton
import triton.language as tl

from fbgemm_gpu.experimental.gemm.triton_gemm import utils


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
            },
            num_stages=num_stages,
            num_warps=num_warps,
            num_ctas=num_ctas,
        )
        for block_size_m in [64, 128]
        for block_size_n in [128, 256]
        for block_size_k in [128, 256]
        for num_stages in [3, 4]
        for num_warps in [4, 8]
        for num_ctas in [1]
    ],
    key=["G", "M_BUCKET", "N", "K"],
)
@triton.jit
def _kernel_grouped_gemm(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_offsets,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    c_desc_ptr = workspace + tidx * TMA_SIZE

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        M_end_offset = tl.load(m_offsets + g)
        m_size = M_end_offset - M_start_offset

        if m_size > 0:
            N_start_offset = g * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            # pyre-ignore
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=c_ptr + M_start_offset * N,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, n_size],
                element_ty=c_ptr.dtype.element_ty,
            )
            # pyre-ignore
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                for k_offset in range(0, K, BLOCK_SIZE_K):
                    a = tl._experimental_descriptor_load(
                        a_desc_ptr,
                        [m_offset, k_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_K],
                        dtype,
                    )
                    b = tl._experimental_descriptor_load(
                        b_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        dtype,
                    )
                    accumulator += tl.dot(a, b.T)

                m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(c_ptr.dtype.element_ty),
                    [m_offset, n_offset],
                )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
            },
            num_stages=num_stages,
            num_warps=num_warps,
            num_ctas=num_ctas,
        )
        for block_size_m in [64, 128]
        for block_size_n in [128, 256]
        for block_size_k in [128, 256]
        for num_stages in [3, 4]
        for num_warps in [4, 8]
        for num_ctas in [1]
    ],
    key=["G", "M_BUCKET", "N", "K"],
)
@triton.jit
def _kernel_grouped_gemm_fp8_rowwise(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    c_ptr,
    workspace,
    m_offsets,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    dtype = tl.float8e4nv
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    c_desc_ptr = workspace + tidx * TMA_SIZE

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        M_end_offset = tl.load(m_offsets + g)
        m_size = M_end_offset - M_start_offset

        if m_size > 0:
            N_start_offset = g * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            # pyre-ignore
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=c_ptr + M_start_offset * N,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, n_size],
                element_ty=c_ptr.dtype.element_ty,
            )
            # pyre-ignore
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                for k_offset in range(0, K, BLOCK_SIZE_K):
                    a = tl._experimental_descriptor_load(
                        a_desc_ptr,
                        [m_offset, k_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_K],
                        dtype,
                    )
                    b = tl._experimental_descriptor_load(
                        b_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        dtype,
                    )
                    accumulator += tl.dot(a, b.T)

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                a_scale = tl.load(
                    a_scale_ptr + M_start_offset + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )
                b_scale = tl.load(
                    b_scale_ptr + N_start_offset + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )
                c = accumulator.to(tl.float32) * a_scale * b_scale

                m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    c.to(c_ptr.dtype.element_ty),
                    [m_offset, n_offset],
                )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


_ON_DEVICE_TMA_WORKSPACE = {}


def _grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_offsets: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not utils.HAS_TMA_DESC:
        raise NotImplementedError("Grouped GEMM without TMA is not supported yet")

    G = m_offsets.shape[0]

    # TODO(shikaili): G=1 could produce NaNs results with on-device TMA store. Need to debug.
    if G == 1:
        raise NotImplementedError("Grouped GEMM with NUM_GROUPS=1 is not supported yet")

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_offsets.is_contiguous()

    M, K = x.shape
    N = w.shape[0] // G
    assert K == w.shape[1]

    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    desc_helper = utils.TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("x")
    desc_helper.init_tma_descriptor("w")

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    global _ON_DEVICE_TMA_WORKSPACE
    if x.device not in _ON_DEVICE_TMA_WORKSPACE:
        _ON_DEVICE_TMA_WORKSPACE[x.device] = torch.empty(
            NUM_SMS * utils.TmaAutoTuneHelper.TMA_SIZE,
            device=x.device,
            dtype=torch.uint8,
        )
    workspace = _ON_DEVICE_TMA_WORKSPACE[x.device]

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "x",
            x.data_ptr(),
            M,
            K,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            x.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "w",
            w.data_ptr(),
            N * G,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w.element_size(),
        )

        return (NUM_SMS,)

    desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
    desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    M_BUCKET = triton.next_power_of_2(M)
    if x_scale is not None and w_scale is not None:
        assert x_scale.is_contiguous()
        assert w_scale.is_contiguous()
        _kernel_grouped_gemm_fp8_rowwise[grid](
            desc_x,
            x_scale,
            desc_w,
            w_scale,
            y,
            workspace,
            m_offsets,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
        )
    else:
        assert x_scale is None
        assert w_scale is None
        _kernel_grouped_gemm[grid](
            desc_x,
            desc_w,
            y,
            workspace,
            m_offsets,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
        )

    return y


def grouped_gemm(
    x: torch.Tensor, w: torch.Tensor, m_offsets: torch.Tensor
) -> torch.Tensor:
    return _grouped_gemm(x, w, m_offsets)


def grouped_gemm_fp8_rowwise(
    x: torch.Tensor,
    w: torch.Tensor,
    m_offsets: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    return _grouped_gemm(x, w, m_offsets, x_scale, w_scale)
