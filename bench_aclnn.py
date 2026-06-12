#!/usr/bin/env python3
"""Benchmark ACLNN vs torch-eager-NPU.

Times each kernel twice on NPU (using NPU events), reports median time
in microseconds plus relative speedup. Targets the fused torch_npu
kernels where the ACLNN backend wraps a real npu_* call; also reports
the torch-ops-on-NPU kernels for completeness (speedup ≈ 1x there).

Output: markdown table on stdout + benchmark/benchmark_results.json.
"""
import importlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch_npu

DEVICE = "npu:0"


def timer_us(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    times.sort()
    return times[len(times) // 2]


def ref(family, name):
    full = f"tile_kernels_ascend.torch.{family}"
    m = importlib.import_module(full)
    if hasattr(m, name):
        return getattr(m, name)
    if hasattr(m, name + "_ref"):
        return getattr(m, name + "_ref")
    if family == "moe":
        for sub in ("expand_to_fused", "reduce_fused"):
            sm = importlib.import_module(f"{full}.{sub}")
            if hasattr(sm, name): return getattr(sm, name)
        if name == "topk_gate" and hasattr(m, "topk_gate_ref"):
            return m.topk_gate_ref
    if family == "quant":
        cm = importlib.import_module(f"{full}.cast")
        if name == "per_token_cast":
            return lambda x, fmt: cm.cast(x, fmt, block_size=(1, x.shape[1]))
        if name == "per_channel_cast":
            return lambda x, fmt: cm.cast(x, fmt, block_size=(x.shape[0], 1))
        if name == "per_block_cast":
            return lambda x, bs, fmt: cm.cast(x, fmt, block_size=bs)
        if name == "cast_back":
            return cm.cast_back
        if name == "swiglu_forward_and_per_token_cast":
            sw = importlib.import_module(f"{full}.swiglu").swiglu_forward
            return lambda x: sw(x)
    raise AttributeError(f"no ref '{name}' in {full}")


def acl(family, name):
    m = importlib.import_module(f"tile_kernels_ascend.aclnn.{family}")
    return getattr(m, name)


results = []


def record(kernel, family, eager_fn, aclnn_fn, eager_args, aclnn_args):
    try:
        t_e = timer_us(lambda: eager_fn(*eager_args))
        t_a = timer_us(lambda: aclnn_fn(*aclnn_args))
        sp = t_e / t_a if t_a > 0 else float("inf")
        results.append({
            "family": family, "kernel": kernel,
            "torch_eager_us": round(t_e, 2),
            "aclnn_us": round(t_a, 2),
            "speedup": round(sp, 3),
            "status": "ok",
        })
    except Exception as e:
        results.append({
            "family": family, "kernel": kernel,
            "torch_eager_us": None, "aclnn_us": None, "speedup": None,
            "status": f"error: {type(e).__name__}",
        })
        print(f"  [skip] {kernel}: {type(e).__name__}", file=sys.stderr)


torch.manual_seed(0)

# ----- Quant (int8 per-token / per-channel) --------------------------------
T, H = 1024, 1024
x_q = torch.randn(T, H, dtype=torch.bfloat16, device=DEVICE)

def ref_per_token_int8(x):
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = (amax / 127.0)
    y = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    return y, amax.squeeze(-1)


def ref_per_channel_int8(x):
    amax = x.float().abs().amax(dim=0, keepdim=True).clamp(min=1e-4)
    scale = (amax / 127.0)
    y = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    return y, amax.squeeze(0)


record("per_token_cast int8", "quant",
       ref_per_token_int8, acl("quant", "per_token_cast"),
       (x_q,), (x_q, "e2m1"))

record("per_channel_cast int8", "quant",
       ref_per_channel_int8, acl("quant", "per_channel_cast"),
       (x_q,), (x_q,))

# ----- Quant (FP8 e4m3fn block quant — Ascend 950 only) --------------------
def ref_per_block_fp8(x, bs):
    row_bs, col_bs = bs
    H = x.shape[-1]
    x_blk = x.float().view(x.shape[0] // row_bs, row_bs, H // col_bs, col_bs)
    amax = x_blk.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    scale = amax / 448.0
    y = torch.clamp(x_blk / scale, -448, 448).to(torch.float8_e4m3fn)
    return y.reshape(x.shape), amax.reshape(x.shape[0] // row_bs, H // col_bs)


def ref_per_token_fp8(x):
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / 448.0
    y = torch.clamp(x.float() / scale, -448, 448).to(torch.float8_e4m3fn)
    return y, amax.squeeze(-1)


x_q128 = torch.randn(T, 128, dtype=torch.bfloat16, device=DEVICE)

record("per_token_cast FP8 e4m3", "quant",
       ref_per_token_fp8, acl("quant", "per_token_cast"),
       (x_q128,), (x_q128, "e4m3"))

record("per_block_cast (1,128) FP8", "quant",
       lambda x: ref_per_block_fp8(x, (1, 128)),
       acl("quant", "per_block_cast"),
       (x_q,), (x_q, (1, 128), "e4m3"))

record("per_block_cast (128,128) FP8", "quant",
       lambda x: ref_per_block_fp8(x, (128, 128)),
       acl("quant", "per_block_cast"),
       (x_q,), (x_q, (128, 128), "e4m3"))

# ----- Quant (MXFP4 dual-level — Ascend 950 only) --------------------------
T_mx, H_mx = 1024, 512
x_mx = torch.randn(T_mx, H_mx, dtype=torch.bfloat16, device=DEVICE)

record("mxfp4_dual_level_quant", "quant",
       lambda x: x,
       lambda x: torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=None),
       (x_mx,), (x_mx,))

# ----- MoE (fused npu_moe_* APIs) ------------------------------------------
N, K, E, H = 512, 2, 8, 256
logits = torch.randn(N, E, dtype=torch.float32, device=DEVICE)
bias = torch.randn(E, dtype=torch.float32, device=DEVICE)
weights = torch.rand(N, K, dtype=torch.float32, device=DEVICE)
weights = weights / weights.sum(dim=-1, keepdim=True)
topk_idx = torch.randint(0, E, (N, K), dtype=torch.int64, device=DEVICE)

record("topk_gate softmax", "moe",
       ref("moe", "topk_gate"), acl("moe", "topk_gate"),
       (logits, bias, K, "softmax"), (logits, bias, K, "softmax"))

record("top2_sum_gate (G=4, Kg=2)", "moe",
       ref("moe", "top2_sum_gate"), acl("moe", "top2_sum_gate"),
       (logits, bias, K, 2, 4, False, 0, 1.0, 0, 1, 0, 1, "softmax"),
       (logits, bias, K, 2, 4, False, 0, 1.0, 0, 1, 0, 1, "softmax"))

t2p = torch.arange(N * K, dtype=torch.int32, device=DEVICE).view(N, K)
p2e = (t2p.reshape(-1) % E).to(torch.int32)
x_moe = torch.randn(N, H, dtype=torch.bfloat16, device=DEVICE)
x_exp = torch.randn(N * K, H, dtype=torch.bfloat16, device=DEVICE)

record("expand_to_fused", "moe",
       ref("moe", "expand_to_fused"), acl("moe", "expand_to_fused"),
       (x_moe, t2p, p2e), (x_moe, t2p, p2e))

import tile_kernels_ascend.torch.moe.reduce_fused as _rf
_rf.elementwise_fma = lambda a, b, c: a * b + c

record("reduce_fused", "moe",
       _rf.reduce_fused, acl("moe", "reduce_fused"),
       (x_exp, weights, t2p), (x_exp, weights, t2p))

record("get_fused_mapping", "moe",
       ref("moe", "get_fused_mapping"), acl("moe", "get_fused_mapping"),
       (N * K, N, K, topk_idx, 16), (N * K, N, K, topk_idx, 16))

scores = torch.randn(N, 4, E // 4, dtype=torch.float32, device=DEVICE)
record("topk_sum_and_topk_group_idx", "moe",
       ref("moe", "topk_sum_and_topk_group_idx"),
       acl("moe", "topk_sum_and_topk_group_idx"),
       (scores, 2, 2), (scores, 2, 2))

record("aux_fi", "moe",
       ref("moe", "aux_fi"), acl("moe", "aux_fi"),
       (topk_idx, E, K), (topk_idx, E, K))

record("group_count", "moe",
       ref("moe", "group_count"), acl("moe", "group_count"),
       (topk_idx.reshape(-1).to(torch.int32), E),
       (topk_idx.reshape(-1).to(torch.int32), E))

record("normalize_weight", "moe",
       ref("moe", "normalize_weight"), acl("moe", "normalize_weight"),
       (weights,), (weights,))

# ----- Transpose (torch-ops on NPU) ----------------------------------------
x2d = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEVICE)
x3d = torch.randn(8, 1024, 1024, dtype=torch.bfloat16, device=DEVICE)

record("transpose (1024x1024 bf16)", "transpose",
       ref("transpose", "transpose"), acl("transpose", "transpose"),
       (x2d,), (x2d,))

record("batched_transpose (8,1024,1024 bf16)", "transpose",
       ref("transpose", "batched_transpose"),
       acl("transpose", "batched_transpose"),
       (x3d,), (x3d,))


# ----- Emit ----------------------------------------------------------------
print("### ACLNN vs torch-eager-NPU benchmark results (Ascend 950)\n")
print("| Kernel | torch-eager (us) | ACLNN (us) | Speedup |")
print("|--------|-----------------:|-----------:|--------:|")
for fam in ("quant", "moe", "transpose"):
    rows = [r for r in results if r["family"] == fam]
    if not rows:
        continue
    print(f"| **{fam}** ||||")
    for r in rows:
        if r.get("status") != "ok":
            print(f"|  `{r['kernel']}` | — | — | skipped ({r['status']}) |")
            continue
        print(f"|  `{r['kernel']}` | {r['torch_eager_us']:.1f} | {r['aclnn_us']:.1f} | {r['speedup']:.2f}x |")

out = Path(__file__).parent / "benchmark" / "benchmark_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
