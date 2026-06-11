def _not_implemented(family, name):
    def fn(*args, **kwargs):
        raise NotImplementedError(f"ACLNN backend {family}:{name} not yet implemented")
    fn.__name__ = name
    return fn


_FAMILY = "quant"

per_token_cast = _not_implemented(_FAMILY, "per_token_cast")
per_channel_cast = _not_implemented(_FAMILY, "per_channel_cast")
per_channel_cast_and_transpose = _not_implemented(_FAMILY, "per_channel_cast_and_transpose")
per_channel_cast_fused = _not_implemented(_FAMILY, "per_channel_cast_fused")
per_block_cast = _not_implemented(_FAMILY, "per_block_cast")
per_block_cast_lossless = _not_implemented(_FAMILY, "per_block_cast_lossless")
cast_back = _not_implemented(_FAMILY, "cast_back")
cast_back_e5m6 = _not_implemented(_FAMILY, "cast_back_e5m6")
swiglu_forward_and_per_channel_cast_and_transpose = _not_implemented(_FAMILY, "swiglu_forward_and_per_channel_cast_and_transpose")
swiglu_forward_and_per_token_cast = _not_implemented(_FAMILY, "swiglu_forward_and_per_token_cast")
swiglu_backward_and_per_token_cast = _not_implemented(_FAMILY, "swiglu_backward_and_per_token_cast")
per_token_cast_to_e5m6 = _not_implemented(_FAMILY, "per_token_cast_to_e5m6")
