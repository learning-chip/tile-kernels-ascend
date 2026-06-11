import pytest
import torch

from tile_kernels_ascend.torch.mhc import (
    expand_to_mhc_ref,
    sinkhorn_normalize_ref,
    mhc_pre_apply_mix_ref,
    mhc_post_ref,
    mhc_pre_norm_fn_ref,
)


NUM_TOKENS = 256
HIDDEN = 512
MHC_MULT = 4
NUM_HEADS = 4


def _has_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()


@pytest.mark.benchmark
def test_bench_expand_to_mhc(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        hidden = torch.randn(1, NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)

        def run():
            expand_to_mhc_ref(hidden, MHC_MULT)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='expand_to_mhc_ref',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'mhc_mult': MHC_MULT, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_sinkhorn_normalize(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(1, NUM_TOKENS, NUM_HEADS, dtype=torch.float32, device=device)

        def run():
            sinkhorn_normalize_ref(x)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='sinkhorn_normalize_ref',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'num_heads': NUM_HEADS, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_mhc_pre_apply_mix(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(1, NUM_TOKENS, HIDDEN, MHC_MULT, dtype=torch.float32, device=device)
        mix = torch.randn(1, NUM_TOKENS, HIDDEN, MHC_MULT, dtype=torch.float32, device=device)

        def run():
            mhc_pre_apply_mix_ref(x, mix)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='mhc_pre_apply_mix_ref',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'mhc_mult': MHC_MULT, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_mhc_post(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(1, NUM_TOKENS, MHC_MULT, HIDDEN, dtype=torch.bfloat16, device=device)
        residual = torch.randn(1, NUM_TOKENS, MHC_MULT, HIDDEN, dtype=torch.bfloat16, device=device)
        post_layer_mix = torch.randn(1, NUM_TOKENS, HIDDEN, 1, dtype=torch.float32, device=device)
        comb_res_mix = torch.randn(1, NUM_TOKENS, MHC_MULT, MHC_MULT, dtype=torch.float32, device=device)

        def run():
            mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='mhc_post_ref',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'mhc_mult': MHC_MULT, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_mhc_pre_norm_fn(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        rms_group_size = HIDDEN
        residual = torch.randn(1, NUM_TOKENS, MHC_MULT, HIDDEN, dtype=torch.float32, device=device)
        mhc_fn = torch.randn(MHC_MULT, 1, rms_group_size, dtype=torch.float32, device=device)
        mhc_norm_weight = torch.randn(MHC_MULT, 1, rms_group_size, dtype=torch.float32, device=device)

        def run():
            mhc_pre_norm_fn_ref(residual, mhc_fn, mhc_norm_weight, 1e-6)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='mhc_pre_norm_fn_ref',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'mhc_mult': MHC_MULT, 'device': device},
            time_us=time_us,
        )
