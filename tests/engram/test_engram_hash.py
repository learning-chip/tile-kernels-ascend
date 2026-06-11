import pytest
import torch

from tile_kernels_ascend.torch.engram import engram_hash_ref, make_offsets
from tile_kernels_ascend.torch.utils import make_param_id


NUM_TOKENS = 256
MAX_NGRAM = 3
NUM_LAYERS = 2
TABLES_PER_LAYER = 8
VOCAB_SIZE = 1024


@pytest.fixture
def engram_hash_inputs():
    ngram_token_ids = torch.randint(0, 10000, (NUM_TOKENS, MAX_NGRAM), dtype=torch.int32)
    multipliers = torch.randint(1, 10000, (NUM_LAYERS, MAX_NGRAM), dtype=torch.int32)
    vocab_sizes = torch.ones(NUM_LAYERS, MAX_NGRAM - 1, dtype=torch.int32) * TABLES_PER_LAYER * VOCAB_SIZE
    offsets = make_offsets(vocab_sizes)
    return ngram_token_ids, multipliers, vocab_sizes, offsets


@pytest.mark.skipif(not hasattr(torch, 'npu') or not torch.npu.is_available(), reason='NPU not available')
def test_engram_hash_npu_vs_cpu(engram_hash_inputs):
    ngram_token_ids, multipliers, vocab_sizes, offsets = engram_hash_inputs

    golden = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets)

    npu_ngram = ngram_token_ids.to('npu')
    npu_mult = multipliers.to('npu')
    npu_vocab = vocab_sizes.to('npu')
    npu_offsets = offsets.to('npu')

    result = engram_hash_ref(npu_ngram, npu_mult, npu_vocab, npu_offsets).cpu()

    param_id = make_param_id({
        'num_tokens': NUM_TOKENS,
        'ngram': MAX_NGRAM,
        'layers': NUM_LAYERS,
        'tables': TABLES_PER_LAYER,
    })
    torch.testing.assert_close(result, golden, rtol=0, atol=0)
