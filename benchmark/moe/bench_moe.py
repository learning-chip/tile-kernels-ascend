import pytest
import torch

from tile_kernels_ascend.torch.moe import top2_sum_gate, aux_fi
from tile_kernels_ascend.torch.moe.expand_to_fused import expand_to_fused
from tile_kernels_ascend.torch.moe.reduce_fused import reduce_fused
from tile_kernels_ascend.torch.moe import get_fused_mapping


NUM_TOKENS = 256
HIDDEN = 512
NUM_TOPK = 2
NUM_EXPERTS = 64
NUM_GROUPS = 8
NUM_TOPK_GROUPS = 2
NUM_EP_RANKS = 1
NUM_TP_RANKS = 1
ALIGNMENT = 16


def _has_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()


def _make_top2_sum_gate_inputs(device):
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.bfloat16, device=device)
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device=device)
    return logits, bias


@pytest.mark.benchmark
def test_bench_top2_sum_gate(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        logits, bias = _make_top2_sum_gate_inputs(device)

        def run():
            top2_sum_gate(
                logits, bias, NUM_TOPK, NUM_TOPK_GROUPS, NUM_GROUPS,
                False, 0, 1.0, 0, NUM_EP_RANKS, 0, NUM_TP_RANKS, 'softmax',
            )

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='top2_sum_gate',
            operation='fwd',
            params={
                'num_tokens': NUM_TOKENS, 'num_experts': NUM_EXPERTS,
                'num_topk': NUM_TOPK, 'device': device,
            },
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_aux_fi(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device=device)

        def run():
            aux_fi(topk_idx, NUM_EXPERTS, NUM_TOPK)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='aux_fi',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'num_experts': NUM_EXPERTS, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_expand_to_fused(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device=device)
        num_expanded = NUM_TOKENS * NUM_TOPK
        token_topk_to_pos, pos_to_expert, _, aligned_expanded = get_fused_mapping(
            num_expanded, NUM_TOKENS, NUM_TOPK, topk_idx, ALIGNMENT,
        )
        x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)

        def run():
            expand_to_fused(x, token_topk_to_pos, pos_to_expert)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='expand_to_fused',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'num_topk': NUM_TOPK, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_reduce_fused(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        topk_idx = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int64, device=device)
        num_expanded = NUM_TOKENS * NUM_TOPK
        token_topk_to_pos, pos_to_expert, _, aligned_expanded = get_fused_mapping(
            num_expanded, NUM_TOKENS, NUM_TOPK, topk_idx, ALIGNMENT,
        )
        x_expanded = torch.randn(aligned_expanded, HIDDEN, dtype=torch.bfloat16, device=device)
        topk_weights = torch.randn(NUM_TOKENS, NUM_TOPK, dtype=torch.float32, device=device)

        def run():
            reduce_fused(x_expanded, topk_weights, token_topk_to_pos)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='reduce_fused',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'hidden': HIDDEN, 'num_topk': NUM_TOPK, 'device': device},
            time_us=time_us,
        )
