def _not_implemented(family, name):
    def fn(*args, **kwargs):
        raise NotImplementedError(f"PTO backend {family}:{name} not yet implemented")
    fn.__name__ = name
    return fn


_FAMILY = "engram"

engram_hash = _not_implemented(_FAMILY, "engram_hash")
engram_gate_fwd = _not_implemented(_FAMILY, "engram_gate_fwd")
engram_gate_bwd = _not_implemented(_FAMILY, "engram_gate_bwd")
grad_w_reduce = _not_implemented(_FAMILY, "grad_w_reduce")
fused_weight = _not_implemented(_FAMILY, "fused_weight")
