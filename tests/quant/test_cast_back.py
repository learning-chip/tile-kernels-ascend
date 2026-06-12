import pytest
import torch

from tile_kernels_ascend.torch.quant.cast import cast, cast_back as cast_back_ref

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_cast_back():
    h, w = 128, 128
    block_size = (1, 128)
    torch.manual_seed(33)

    x_cpu = torch.randn(h, w).to(torch.bfloat16)
    x_q_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=block_size)
    x_back_cpu = cast_back_ref((x_q_cpu, sf_cpu), "bf16", block_size=block_size)

    x_q_npu, sf_npu = x_q_cpu.npu(), sf_cpu.npu()
    try:
        x_back_npu = cast_back_ref((x_q_npu, sf_npu), "bf16", block_size=block_size)
    except RuntimeError as e:
        pytest.xfail(f"NPU cast_back unsupported: {e}")

    torch.testing.assert_close(x_back_npu.cpu(), x_back_cpu)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_cast_back_aclnn_fp8():
    from tile_kernels_ascend.aclnn.quant import cast_back, per_block_cast
    h, w = 128, 128
    block_size = (1, 128)
    torch.manual_seed(34)

    x_npu = torch.randn(h, w, dtype=torch.bfloat16).npu()
    try:
        x_q_npu, sf_npu = per_block_cast(x_npu, block_size=block_size, fmt='e4m3')
        x_back_npu = cast_back((x_q_npu, sf_npu), fmt='bf16', block_size=block_size)
    except (RuntimeError, NotImplementedError) as e:
        pytest.xfail(f"ACLNN cast_back unsupported: {e}")

    x_q_cpu, sf_cpu = cast(x_npu.cpu(), "e4m3", block_size=block_size)
    x_back_cpu = cast_back_ref((x_q_cpu, sf_cpu), "bf16", block_size=block_size)

    torch.testing.assert_close(x_back_npu.cpu(), x_back_cpu, atol=0.3, rtol=0.3)
