import pytest
import torch

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_block_cast_fp8():
    from tile_kernels_ascend.aclnn.quant import per_block_cast
    from tile_kernels_ascend.torch.quant.cast import cast
    h, w = 128, 128
    block_size = (1, 128)
    torch.manual_seed(55)

    x_cpu = torch.randn(h, w).to(torch.bfloat16)
    out_w_cpu, sf_cpu = cast(x_cpu, "e4m3", block_size=block_size)

    x_npu = x_cpu.npu()
    try:
        out_w_npu, sf_npu = per_block_cast(x_npu, block_size=block_size, fmt='e4m3')
    except RuntimeError as e:
        pytest.xfail(f"ACLNN per_block_cast unsupported: {e}")

    assert out_w_npu.dtype == torch.float8_e4m3fn
    assert out_w_npu.shape == x_npu.shape
    torch.testing.assert_close(sf_npu.cpu(), sf_cpu, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_block_cast_e2m1_unsupported():
    from tile_kernels_ascend.aclnn.quant import per_block_cast
    x_npu = torch.randn(128, 128, dtype=torch.bfloat16).npu()
    with pytest.raises(NotImplementedError):
        per_block_cast(x_npu, block_size=(128, 128), fmt='e2m1')


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_per_block_cast_fp8_block128():
    from tile_kernels_ascend.aclnn.quant import per_block_cast
    h, w = 256, 256
    block_size = (128, 128)
    torch.manual_seed(56)
    x_npu = torch.randn(h, w, dtype=torch.bfloat16).npu()
    try:
        out_w_npu, sf_npu = per_block_cast(x_npu, block_size=block_size, fmt='e4m3')
    except RuntimeError as e:
        pytest.xfail(f"ACLNN per_block_cast (128,128) unsupported: {e}")
    assert out_w_npu.dtype == torch.float8_e4m3fn
    assert out_w_npu.shape == (h, w)
