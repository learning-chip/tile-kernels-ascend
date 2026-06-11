def _not_implemented(family, name):
    def fn(*args, **kwargs):
        raise NotImplementedError(f"ACLNN backend {family}:{name} not yet implemented")
    fn.__name__ = name
    return fn


_FAMILY = "transpose"

transpose = _not_implemented(_FAMILY, "transpose")
batched_transpose = _not_implemented(_FAMILY, "batched_transpose")
