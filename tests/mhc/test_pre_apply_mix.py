import pytest
import torch
from tile_kernels_ascend.torch.mhc import mhc_pre_apply_mix_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"


def _make_inputs(n0, n1, mhc_mult, h):
    x = torch.randn(n0, n1, mhc_mult, h, dtype=torch.float32)
    mix = torch.randn(n0, n1, mhc_mult, h, dtype=torch.float32).softmax(dim=-2)
    return x, mix


class TestPreApplyMix:
    n0, n1, mhc_mult, h = 1, 256, 4, 128

    def test_forward(self):
        x, mix = _make_inputs(self.n0, self.n1, self.mhc_mult, self.h)
        with torch.no_grad():
            golden = mhc_pre_apply_mix_ref(x.clone(), mix.clone())
            x_npu = x.clone().to(device)
            mix_npu = mix.clone().to(device)
            npu_out = mhc_pre_apply_mix_ref(x_npu, mix_npu)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)
        assert npu_out.shape == (self.n0, self.n1, self.h)

    def test_backward(self):
        x_base, mix_base = _make_inputs(self.n0, self.n1, self.mhc_mult, self.h)

        x_ref = x_base.clone().requires_grad_(True)
        mix_ref = mix_base.clone().requires_grad_(True)
        out_ref = mhc_pre_apply_mix_ref(x_ref, mix_ref)
        loss_ref = out_ref.float().sum()
        loss_ref.backward()

        x_npu = x_base.clone().to(device).requires_grad_(True)
        mix_npu = mix_base.clone().to(device).requires_grad_(True)
        out_npu = mhc_pre_apply_mix_ref(x_npu, mix_npu)
        loss_npu = out_npu.float().sum()
        loss_npu.backward()

        torch.testing.assert_close(x_npu.grad.cpu(), x_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(mix_npu.grad.cpu(), mix_ref.grad, rtol=1e-2, atol=1e-2)
