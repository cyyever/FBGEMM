# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import pandas as pd

import torch
import triton  # @manual=//triton:triton

E4M3_MAX_POS: float = torch.finfo(torch.float8_e4m3fnuz).max
EPS = 1e-12


def set_amd_env_vars():
    print("Setting environment variables for AMD GPU performance")
    os.environ["DISABLE_ADDMM_HIP_LT"] = "0"
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "0"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "hipblas_tuning_pt_llama.csv"
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "30"
    os.environ["PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS"] = "30"


class BaselineMatmul(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class CKMatmul(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.fbgemm.bf16_gemm(a, b)


@torch.no_grad()
def evaluate_impl(M, N, K, baseline_func, ck_func):
    print(f"Evaluating {M=}, {N=}, {K=}")
    A = torch.randn(M, K).to(dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K).to(dtype=torch.bfloat16, device="cuda")

    # Check accuracy.
    out_ref = baseline_func(A.to(torch.float32), B.t().to(torch.float32))

    baseline_out = baseline_func(A, B.t())
    baseline_sim = torch.mean(torch.abs(baseline_out - out_ref) ** 2)
    print(f"Baseline accuracy: {baseline_sim}")

    ck_out = ck_func(A, B)
    ck_sim = torch.mean(torch.abs(ck_out - out_ref) ** 2)
    print(f"CK accuracy: {ck_sim}")

    # Benchmark runtimes.
    ms_baseline = triton.testing.do_bench(lambda: baseline_func(A, B.t()))
    print(f"Baseline runtime: {ms_baseline} ms")

    ms_ck = triton.testing.do_bench(lambda: ck_func(A, B))
    print(f"CK runtime: {ms_ck} ms")

    return baseline_sim, ck_sim, ms_baseline, ms_ck


def main(args):
    if args.enable_amd_env_vars:
        set_amd_env_vars()

    with torch.no_grad():
        ck_mod = CKMatmul()
        baseline_mod = BaselineMatmul()
        if args.torch_compile_mode:
            ck_mod = torch.compile(
                ck_mod,
                dynamic=False,
                backend="inductor",
                mode=args.torch_compile_mode,
            )
            baseline_mod = torch.compile(
                baseline_mod,
                dynamic=False,
                backend="inductor",
                mode=args.torch_compile_mode,
            )
        # Create a list of results.
        benchmark_results = []

        # Test over a bunch of shapes.
        M = [2048, 2304, 13312, 16032, 16384]
        N = [2304, 4096, 8192, 13312]
        K = [2048, 2304, 6656, 13312, 16384]

        for m in M:
            for n in N:
                for k in K:
                    baseline_sim, ck_sim, ms_baseline, ms_ck = evaluate_impl(
                        m, n, k, baseline_mod, ck_mod
                    )
                    benchmark_results.append(
                        {
                            "M": m,
                            "N": n,
                            "K": k,
                            "baseline_sim": baseline_sim,
                            "ck_sim": ck_sim,
                            "ms_baseline": ms_baseline,
                            "ms_ck": ms_ck,
                        }
                    )
        if args.export_csv:
            benchmark_results_df = pd.DataFrame(benchmark_results)
            benchmark_results_df.to_csv("ck_bench.csv", index=False)


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to a CSV file.",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="",
        help="Torch compile mode to use.",
    )
    parser.add_argument(
        "--enable_amd_env_vars",
        default=False,
        action="store_true",
        help="Enable a set of environment variables for AMD GPU performance",
    )

    args = parser.parse_args()
    main(args)
