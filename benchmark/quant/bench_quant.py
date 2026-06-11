import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast, cast_back
from tile_kernels_ascend.torch.quant.swiglu import swiglu_forward


NUM_TOKENS = 256
HIDDEN = 512
BLOCK_SIZE = (32, 32)


def _has_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()


@pytest.mark.benchmark
def test_bench_cast_e4m3(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)

        def run():
            cast(x, 'e4m3', BLOCK_SIZE)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='cast',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'fmt': 'e4m3', 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_cast_e2m1(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)

        def run():
            cast(x, 'e2m1', BLOCK_SIZE)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='cast',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'fmt': 'e2m1', 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_cast_back(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        out_weight, dq_sf = cast(x, 'e4m3', BLOCK_SIZE)

        def run():
            cast_back((out_weight, dq_sf), 'bf16', BLOCK_SIZE)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='cast_back',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'fmt': 'bf16', 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_swiglu_forward(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(NUM_TOKENS, HIDDEN * 2, dtype=torch.bfloat16, device=device)

        def run():
            swiglu_forward(x)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='swiglu_forward',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'device': device},
            time_us=time_us,
        )
