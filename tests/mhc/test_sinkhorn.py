import pytest
import torch
from tile_kernels_ascend.torch.mhc import sinkhorn_normalize_ref

npu_available = hasattr(torch, "npu") and torch.npu.is_available()
device = "npu" if npu_available else "cpu"


def _make_inputs(n0, n1, mhc_mult):
    x = torch.randn(n0, n1, mhc_mult, mhc_mult, dtype=torch.float32)
    return x


class TestSinkhornNormalize:
    n0, n1, mhc_mult = 1, 256, 4

    def test_forward(self):
        x = _make_inputs(self.n0, self.n1, self.mhc_mult)
        with torch.no_grad():
            golden = sinkhorn_normalize_ref(x.clone(), repeat=3)
            x_npu = x.clone().to(device)
            npu_out = sinkhorn_normalize_ref(x_npu, repeat=3)
        torch.testing.assert_close(npu_out.cpu(), golden, rtol=1e-2, atol=1e-2)

    def test_backward(self):
        x_base = _make_inputs(self.n0, self.n1, self.mhc_mult)

        x_ref = x_base.clone().requires_grad_(True)
        out_ref = sinkhorn_normalize_ref(x_ref, repeat=3)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        grad_ref = x_ref.grad.clone()

        x_npu = x_base.clone().to(device).requires_grad_(True)
        out_npu = sinkhorn_normalize_ref(x_npu, repeat=3)
        loss_npu = out_npu.sum()
        loss_npu.backward()
        grad_npu = x_npu.grad.cpu().clone()

        torch.testing.assert_close(grad_npu, grad_ref, rtol=1e-2, atol=1e-2)
