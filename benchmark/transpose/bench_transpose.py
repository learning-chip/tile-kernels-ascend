import pytest
import torch

from tile_kernels_ascend.torch.transpose import transpose_ref, batched_transpose_ref


NUM_TOKENS = 512
HIDDEN = 512
NUM_EXPERTS = 4


def _has_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()


@pytest.mark.benchmark
def test_bench_transpose(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)

        def run():
            transpose_ref(x)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='transpose_ref',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_batched_transpose(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        x = torch.randn(NUM_EXPERTS, NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)

        def run():
            batched_transpose_ref(x)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='batched_transpose_ref',
            operation='fwd',
            params={'num_experts': NUM_EXPERTS, 'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'device': device},
            time_us=time_us,
        )
