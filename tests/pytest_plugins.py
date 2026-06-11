import hashlib
import json
import os
import time

import pytest
import torch

from tile_kernels_ascend.torch.utils import make_param_id, dtype_to_str


def pytest_addoption(parser):
    parser.addoption('--seed', type=int, default=0)
    parser.addoption('--run-benchmark', action='store_true', default=False, help='Run benchmark tests')
    parser.addoption('--benchmark-output', default=None, help='Path to write benchmark results as JSONL')


@pytest.fixture(autouse=True)
def seed(request):
    base = request.config.getoption('--seed')
    node_hash = int(hashlib.sha256(request.node.nodeid.encode()).hexdigest(), 16) % (2**31)
    s = base + node_hash
    torch.manual_seed(s)
    return s


def pytest_configure(config):
    config.addinivalue_line('markers', 'benchmark: marks tests as benchmark (skipped by default)')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-benchmark'):
        skip_benchmark = pytest.mark.skip(reason='use --run-benchmark to run')
        for item in items:
            if 'benchmark' in item.keywords:
                item.add_marker(skip_benchmark)


def make_param_key(params: dict) -> str:
    return ','.join(f'{k}={v}' for k, v in params.items() if v is not None)


def count_bytes(*tensors) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


@pytest.fixture
def benchmark_record(request):
    records = []

    def _record(kernel, operation, params, time_us, bandwidth_gbs=0, extras=None):
        rec = {
            'kernel': kernel,
            'operation': operation,
            'params': params,
            'time_us': time_us,
            'bandwidth_gbs': bandwidth_gbs,
        }
        if extras:
            rec.update(extras)
        records.append(rec)
        output_path = request.config.getoption('--benchmark-output')
        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

    return _record


@pytest.fixture
def benchmark_timer():
    def _timer(fn, warmup=5, rep=30, **kwargs):
        for _ in range(warmup):
            fn()
        if hasattr(torch, 'npu'):
            torch.npu.synchronize()
        times = []
        for _ in range(rep):
            if hasattr(torch, 'npu'):
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.npu.synchronize()
                times.append(start.elapsed_time(end) * 1000)
            else:
                start = time.perf_counter()
                fn()
                start_us = time.perf_counter()
                fn()
                end = time.perf_counter()
                times.append((end - start_us) * 1e6)
        median_us = sorted(times)[len(times) // 2]
        return median_us

    return _timer
