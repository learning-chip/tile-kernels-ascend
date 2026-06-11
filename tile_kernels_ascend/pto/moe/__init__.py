def _not_implemented(family, name):
    def fn(*args, **kwargs):
        raise NotImplementedError(f"PTO backend {family}:{name} not yet implemented")
    fn.__name__ = name
    return fn


_FAMILY = "moe"

aux_fi = _not_implemented(_FAMILY, "aux_fi")
group_count = _not_implemented(_FAMILY, "group_count")
mask_indices_by_tp = _not_implemented(_FAMILY, "mask_indices_by_tp")
normalize_weight = _not_implemented(_FAMILY, "normalize_weight")
inplace_unique_group_indices = _not_implemented(_FAMILY, "inplace_unique_group_indices")
get_fused_mapping = _not_implemented(_FAMILY, "get_fused_mapping")
topk_gate = _not_implemented(_FAMILY, "topk_gate")
top2_sum_gate = _not_implemented(_FAMILY, "top2_sum_gate")
topk_sum_and_topk_group_idx = _not_implemented(_FAMILY, "topk_sum_and_topk_group_idx")
expand_to_fused = _not_implemented(_FAMILY, "expand_to_fused")
expand_to_fused_with_sf = _not_implemented(_FAMILY, "expand_to_fused_with_sf")
reduce_fused = _not_implemented(_FAMILY, "reduce_fused")
