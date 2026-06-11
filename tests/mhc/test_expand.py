import pytest
import torch
from tile_kernels_ascend.torch.mhc import expand_to_mhc_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"


def _make_inputs(n0, n1, h, mhc_mult):
    hidden = torch.randn(n0, n1, h, dtype=torch.bfloat16)
    return hidden, mhc_mult


class TestExpandComprehensive:
    n0, n1, h, mhc_mult = 1, 256, 128, 4

    def test_forward(self):
        hidden, mult = _make_inputs(self.n0, self.n1, self.h, self.mhc_mult)
        with torch.no_grad():
            golden = expand_to_mhc_ref(hidden, mult)
            npu_hidden = hidden.to(device)
            npu_out = expand_to_mhc_ref(npu_hidden, mult)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)
        assert npu_out.shape == (self.n0, self.n1, self.mhc_mult, self.h)

    def test_backward(self):
        hidden_cpu, mult = _make_inputs(self.n0, self.n1, self.h, self.mhc_mult)

        hidden_ref = hidden_cpu.float().clone().requires_grad_(True)
        out_ref = expand_to_mhc_ref(hidden_ref, mult)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        grad_ref = hidden_ref.grad.clone()

        hidden_npu = hidden_cpu.float().to(device).clone().requires_grad_(True)
        out_npu = expand_to_mhc_ref(hidden_npu, mult)
        loss_npu = out_npu.sum()
        loss_npu.backward()
        grad_npu = hidden_npu.grad.cpu().clone()

        torch.testing.assert_close(grad_npu, grad_ref, rtol=1e-2, atol=1e-2)
