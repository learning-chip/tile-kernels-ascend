import pytest
import torch
from tile_kernels_ascend.torch.transpose import transpose_ref, batched_transpose_ref
from tile_kernels_ascend.torch.utils import make_param_id

def _has_npu():
    return hasattr(torch, 'npu') and torch.npu.is_available()

PARAMS_2D = [
    {'num_tokens': 128, 'hidden': 512, 'dtype': torch.bfloat16},
    {'num_tokens': 128, 'hidden': 512, 'dtype': torch.float32},
]

PARAMS_3D = [
    {'num_tokens': 128, 'hidden': 512, 'num_experts': 4, 'dtype': torch.bfloat16},
    {'num_tokens': 128, 'hidden': 512, 'num_experts': 4, 'dtype': torch.float32},
]


@pytest.mark.parametrize('params', PARAMS_2D, ids=make_param_id)
@pytest.mark.skipif(not _has_npu(), reason='NPU not available')
def test_transpose(params):
    num_tokens, hidden, dtype = params['num_tokens'], params['hidden'], params['dtype']
    if dtype == torch.bfloat16:
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    else:
        x = torch.randn(num_tokens, hidden, dtype=torch.float32)
    
    ref = transpose_ref(x)
    x_npu = x.npu()
    out_npu = transpose_ref(x_npu).cpu()
    torch.testing.assert_close(out_npu, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('params', PARAMS_3D, ids=make_param_id)
@pytest.mark.skipif(not _has_npu(), reason='NPU not available')
def test_batched_transpose(params):
    n, h, e, dtype = params['num_tokens'], params['hidden'], params['num_experts'], params['dtype']
    x = torch.randn(e, n, h, dtype=dtype)
    ref = batched_transpose_ref(x)
    x_npu = x.npu()
    out_npu = batched_transpose_ref(x_npu).cpu()
    torch.testing.assert_close(out_npu, ref, rtol=1e-3, atol=1e-3)
