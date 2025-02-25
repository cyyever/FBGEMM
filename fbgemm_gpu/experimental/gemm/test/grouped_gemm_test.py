# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Tuple

import torch

if torch.cuda.is_available():
    from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import quantize_fp8_row
    from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
        grouped_gemm,
        grouped_gemm_fp8_rowwise,
    )
    from fbgemm_gpu.experimental.gemm.triton_gemm.utils import HAS_TMA_DESC


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9
    or not HAS_TMA_DESC,
    "Skip when H100 or TMA is not available",
)
class TestGroupedGEMM(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_grouped_gemm_fp8_rowwise(self) -> None:
        def _test_grouped_gemm_fp8_rowwise(
            shape: Tuple[int, int, int, int],
            device: torch.device,
        ) -> None:
            G, M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
            m_offsets, _ = torch.sort(
                torch.randint(low=0, high=M, size=[G], device=device, dtype=torch.int32)
            )
            m_offsets[G - 1] = M

            a_fp8, a_scale = quantize_fp8_row(a)
            b_fp8, b_scale = quantize_fp8_row(b)

            result = grouped_gemm_fp8_rowwise(
                a_fp8,
                b_fp8,
                m_offsets,
                a_scale,
                b_scale,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
            # Running baseline with quantization to exclude quantization error from the test as it has nothing to do with the correctness of the kernel implementation.
            for g in range(G):
                m_start = 0 if g == 0 else m_offsets[g - 1]
                m_end = m_offsets[g]
                n_start = g * N
                n_end = (g + 1) * N

                expected_result[m_start:m_end, :] = (
                    a_fp8[m_start:m_end, :].to(torch.float32)
                    @ b_fp8[n_start:n_end, :].to(torch.float32).T
                    * a_scale[m_start:m_end][:, None]
                    * b_scale[n_start:n_end][None, :]
                ).to(torch.bfloat16)

            torch.testing.assert_close(result, expected_result, atol=2e-2, rtol=1.6e-2)

        _test_grouped_gemm_fp8_rowwise((16, 512, 256, 256), torch.device("cuda"))
        _test_grouped_gemm_fp8_rowwise((8, 512, 256, 256), torch.device("cuda"))
        _test_grouped_gemm_fp8_rowwise((4, 512, 256, 256), torch.device("cuda"))
        _test_grouped_gemm_fp8_rowwise((2, 512, 256, 256), torch.device("cuda"))
        # TODO(shikaili): G=1 could produce NaNs results with on-device TMA store. Need to debug.
        # _test_grouped_gemm_fp8_rowwise((1, 512, 256, 256), torch.device("cuda"))

    def test_grouped_gemm_bf16(self) -> None:
        def _test_grouped_gemm_bf16(
            shape: Tuple[int, int, int, int],
            device: torch.device,
        ) -> None:
            G, M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
            m_offsets, _ = torch.sort(
                torch.randint(low=0, high=M, size=[G], device=device, dtype=torch.int32)
            )
            m_offsets[G - 1] = M

            result = grouped_gemm(
                a,
                b,
                m_offsets,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
            for g in range(G):
                m_start = 0 if g == 0 else m_offsets[g - 1]
                m_end = m_offsets[g]
                expected_result[m_start:m_end, :] = (
                    a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
                )

            torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        _test_grouped_gemm_bf16((16, 512, 256, 256), torch.device("cuda"))
        _test_grouped_gemm_bf16((8, 512, 256, 256), torch.device("cuda"))
        _test_grouped_gemm_bf16((4, 512, 256, 256), torch.device("cuda"))
        _test_grouped_gemm_bf16((2, 512, 256, 256), torch.device("cuda"))
        # TODO(shikaili): G=1 could produce NaNs results with on-device TMA store. Need to debug.
        # _test_grouped_gemm_bf16((1, 512, 256, 256), torch.device("cuda"))
