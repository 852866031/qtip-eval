"""Matvec decomposition -- FP16 cuBLAS matvec, EAGER timer.

Path A (FP16) timed with `torch.utils.benchmark.Timer`: each iteration is
a full `fn(); torch.cuda.synchronize()` cycle, so per-call Python dispatch,
torch.ops overhead, kernel launch, and device sync are all included.

Run:
    CUDA_VISIBLE_DEVICES=0 python decomp/run_fp16_eager.py

Writes: output/fp16_eager.json
"""
# -------- configuration (edit here) -----------------------------------------
LABEL = 'fp16_eager'
REPEATS = 200
WARMUP = 10
SHAPES_7B = [
    (4096,  1, 4096),    # q / k / v / o projections
    (4096,  1, 11008),   # down projection
    (11008, 1, 4096),    # gate / up projections
]
# ----------------------------------------------------------------------------

import json
import os

import torch
import torch.utils.benchmark as bench

DECOMP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DECOMP_DIR, 'output')


def eager_time(fn):
    """Time `fn()` eagerly: fn(); sync per iteration. Returns ms/call."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t = bench.Timer(
        stmt='fn(); torch.cuda.synchronize()',
        globals={'fn': fn, 'torch': torch},
        num_threads=1,
    )
    return t.timeit(REPEATS).mean * 1e3  # s -> ms


def bench_shape(m, n, k):
    W = torch.randn(m, k, device='cuda', dtype=torch.float16) / 16
    x = torch.randn(k, n, device='cuda', dtype=torch.float16) / 16
    out = torch.zeros(m, n, device='cuda', dtype=torch.float16)
    ms = eager_time(lambda: torch.matmul(W, x, out=out))
    return {
        'shape_mk': [m, k],
        'time_us': ms * 1000.0,
        'bytes_weight': W.nbytes,
        'effective_gbps': W.nbytes / 1e9 / (ms / 1e3),
    }


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    print(f'[decomp:{LABEL}] GPU={gpu}  REPEATS={REPEATS}  WARMUP={WARMUP}')

    rows = []
    for (m, n, k) in SHAPES_7B:
        r = bench_shape(m, n, k)
        print(f'[decomp:{LABEL}] ({m:>5}, {k:>5})  '
              f'{r["time_us"]:8.1f} us   {r["effective_gbps"]:8.1f} GB/s')
        rows.append(r)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{LABEL}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'label': LABEL,
            'method': 'FP16 cuBLAS matvec',
            'timer': 'eager (torch.utils.benchmark.Timer, fn; sync per iter)',
            'repeats': REPEATS, 'warmup': WARMUP, 'gpu': gpu,
            'rows': rows,
        }, f, indent=2)
    print(f'[decomp:{LABEL}] wrote {out_path}')


if __name__ == '__main__':
    main()
