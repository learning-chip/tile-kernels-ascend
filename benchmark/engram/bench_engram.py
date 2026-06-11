import pytest
import torch

from tile_kernels_ascend.torch.engram import engram_hash_ref, engram_gate_ref, make_offsets


NUM_TOKENS = 128
HIDDEN = 512
MAX_NGRAM = 3
NUM_LAYERS = 2
TABLES_PER_LAYER = 8
VOCAB_SIZE = 1024


def _has_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()


def _make_hash_inputs(device):
    ngram_token_ids = torch.randint(0, 10000, (NUM_TOKENS, MAX_NGRAM), dtype=torch.int32, device=device)
    multipliers = torch.randint(1, 10000, (NUM_LAYERS, MAX_NGRAM), dtype=torch.int32, device=device)
    vocab_sizes = torch.ones(NUM_LAYERS, MAX_NGRAM - 1, dtype=torch.int32, device=device) * TABLES_PER_LAYER * VOCAB_SIZE
    offsets = make_offsets(vocab_sizes)
    return ngram_token_ids, multipliers, vocab_sizes, offsets


@pytest.mark.benchmark
def test_bench_engram_hash(benchmark_timer, benchmark_record):
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        ngram, mult, vocab, offsets = _make_hash_inputs(device)

        def run():
            engram_hash_ref(ngram, mult, vocab, offsets)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='engram_hash_ref',
            operation='fwd',
            params={'num_tokens': NUM_TOKENS, 'ngram': MAX_NGRAM, 'layers': NUM_LAYERS, 'device': device},
            time_us=time_us,
        )


@pytest.mark.benchmark
def test_bench_engram_gate(benchmark_timer, benchmark_record):
    num_tokens, hidden = NUM_TOKENS, HIDDEN
    clamp_value, eps = 1e-6, 1e-6
    for device in (['cpu', 'npu'] if _has_npu() else ['cpu']):
        hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        k = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        v = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        weight_hidden = torch.randn(hidden, dtype=torch.bfloat16, device=device)
        weight_embed = torch.randn(hidden, dtype=torch.bfloat16, device=device)

        def run():
            engram_gate_ref(hidden_states, k, v, weight_hidden, weight_embed, clamp_value, eps)

        time_us = benchmark_timer(run)
        benchmark_record(
            kernel='engram_gate_ref',
            operation='fwd',
            params={'num_tokens': num_tokens, 'hidden': hidden, 'device': device},
            time_us=time_us,
        )
