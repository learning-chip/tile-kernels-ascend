import torch

try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    torch_npu = None
    NPU_AVAILABLE = False


def _try_npu(fn, *args, **kwargs):
    if not NPU_AVAILABLE or not hasattr(torch_npu, fn):
        raise NotImplementedError(f"ACLNN backend: torch_npu.{fn} not available")
    try:
        return getattr(torch_npu, fn)(*args, **kwargs)
    except Exception as e:
        raise NotImplementedError(f"ACLNN backend: torch_npu.{fn} failed: {e}") from e


def expand_to_mhc(hidden: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    return hidden.unsqueeze(-2).expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1]).contiguous()


def sinkhorn_normalize(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    try:
        return _try_npu('npu_mhc_sinkhorn', x, eps=eps, num_iters=repeat, out_flag=0)
    except NotImplementedError:
        raise


def mhc_head_compute_mix(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
    return torch.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


def mhc_pre_split_mixes(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = input_mixes.shape[:2]
    mhc_scale = torch.cat([
        mhc_scale[0].expand(mhc_mult),
        mhc_scale[1].expand(mhc_mult),
        mhc_scale[2].expand(mhc_mult * mhc_mult),
    ])
    input_mixes = input_mixes * mhc_scale + mhc_base
    pre_layer_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (input_mixes[:, :, mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult :].view(a, b, mhc_mult, mhc_mult)
    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return (x * mix).sum(-2).bfloat16()


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    try:
        return _try_npu('npu_mhc_post', x, comb_res_mix, residual, post_layer_mix)
    except NotImplementedError:
        raise


def mhc_pre_norm_fn(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight,
    mhc_norm_eps: float,
) -> torch.Tensor:
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.flatten(2, 3).float()
    assert mhc_fn.dtype == residual.dtype == torch.float
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum(
        'mbk,nbk->mbn',
        residual.view(-1, 1, rms_group_size),
        mhc_fn.view(mhc_mult, 1, rms_group_size),
    )
    sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
    mixes = (mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()).sum(-2)
    return mixes.view(*residual.shape[:2], -1)


def mhc_pre_big_fuse(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    gamma,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    try:
        return _try_npu('npu_mhc_pre', x, phi, alpha, bias, gamma=gamma, norm_eps=norm_eps, hc_eps=hc_eps, out_flag=1)
    except NotImplementedError:
        raise


def mhc_multilayer_recompute(
    residual: torch.Tensor,
    pre_mixes: list,
    post_mixes: list,
    comb_mixes: list,
) -> torch.Tensor:
    for mix, post_mix, comb in zip(pre_mixes, post_mixes, comb_mixes):
        x = mhc_pre_apply_mix(residual, mix)
        residual = mhc_post(x, residual, post_mix, comb)
    return residual
