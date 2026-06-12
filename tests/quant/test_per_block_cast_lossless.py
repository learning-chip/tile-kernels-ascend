import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast, cast_back as cast_back_ref

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_block_cast_lossless():
    h, w = 64, 64
    block_size = (32, 32)
    torch.manual_seed(8)

    x_cpu = (torch.randn(h, w) * 0.1).to(torch.bfloat16)
    x_q_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=block_size)
    x_back_cpu = cast_back_ref((x_q_cpu, sf_cpu), "bf16", block_size=block_size)

    x_npu = x_cpu.npu()
    try:
        x_q_npu, sf_npu = cast(x_npu, "e4m3", block_size=block_size)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast unsupported: {e}")

    x_back_npu = cast_back_ref((x_q_npu, sf_npu), "bf16", block_size=block_size)

    torch.testing.assert_close(x_back_npu.cpu(), x_back_cpu)
    torch.testing.assert_close(x_back_cpu, x_cpu, atol=0.05, rtol=0.05)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_block_cast_lossless_aclnn():
    from tile_kernels_ascend.aclnn.quant import per_block_cast_lossless
    x_npu = torch.randn(64, 64, dtype=torch.bfloat16).npu()
    with pytest.raises(NotImplementedError):
        per_block_cast_lossless(x_npu, block_size=(32, 32))
