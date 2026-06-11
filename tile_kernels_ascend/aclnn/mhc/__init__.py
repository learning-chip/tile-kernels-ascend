def _not_implemented(family, name):
    def fn(*args, **kwargs):
        raise NotImplementedError(f"ACLNN backend {family}:{name} not yet implemented")
    fn.__name__ = name
    return fn


_FAMILY = "mhc"

expand_to_mhc = _not_implemented(_FAMILY, "expand_to_mhc")
sinkhorn_normalize = _not_implemented(_FAMILY, "sinkhorn_normalize")
mhc_head_compute_mix = _not_implemented(_FAMILY, "mhc_head_compute_mix")
mhc_pre_split_mixes = _not_implemented(_FAMILY, "mhc_pre_split_mixes")
mhc_pre_apply_mix = _not_implemented(_FAMILY, "mhc_pre_apply_mix")
mhc_post = _not_implemented(_FAMILY, "mhc_post")
mhc_pre_norm_fn = _not_implemented(_FAMILY, "mhc_pre_norm_fn")
mhc_pre_big_fuse = _not_implemented(_FAMILY, "mhc_pre_big_fuse")
mhc_multilayer_recompute = _not_implemented(_FAMILY, "mhc_multilayer_recompute")
