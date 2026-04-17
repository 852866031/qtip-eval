"""Matvec decomposition -- QTIP fused decompress+matvec, EAGER timer.

Path C timed eagerly. Each timer iteration executes the fused kernel once
and then cudaDeviceSynchronize -- which is how a naive Python inference
loop would drive the kernel. See run_qtip_graph.py for the counterpart.

Run:
    CUDA_VISIBLE_DEVICES=0 python decomp/run_qtip_eager.py

Writes: output/qtip_eager.json
"""
# -------- configuration (edit here) -----------------------------------------
LABEL = 'qtip_eager'
BITS = (2, 3, 4)
REPEATS = 200
WARMUP = 10
SHAPES_7B = [
    (4096,  1, 4096),
    (4096,  1, 11008),
    (11008, 1, 4096),
]
# ----------------------------------------------------------------------------

import json
import os
import sys

import torch
import torch.utils.benchmark as bench

DECOMP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(DECOMP_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'qtip', 'qtip-kernels'))

import qtip_kernels  # noqa: F401  (torch must load first; already imported above)

from test_decompress_matvec import kernels, prepare_arguments  # noqa: E402

OUTPUT_DIR = os.path.join(DECOMP_DIR, 'output')


def eager_time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t = bench.Timer(
        stmt='fn(); torch.cuda.synchronize()',
        globals={'fn': fn, 'torch': torch},
        num_threads=1,
    )
    return t.timeit(REPEATS).mean * 1e3


def bench_shape(R, m, n, k):
    L, S, V = 16, 9, 1  # match the V=1-symbol kernel shipped by qtip-kernels
    out, compressed, x, codebook, _ = prepare_arguments(L, S, R, V, m, n, k)
    kernel = kernels[R][(m, n, k)]

    def run():
        out.zero_()
        kernel(out, compressed, x, codebook)

    ms = eager_time(run)
    return {
        'shape_mk': [m, k],
        'bits': R,
        'time_us': ms * 1000.0,
        'bytes_compressed': compressed.nbytes,
        'effective_gbps': compressed.nbytes / 1e9 / (ms / 1e3),
    }


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    print(f'[decomp:{LABEL}] GPU={gpu}  REPEATS={REPEATS}  WARMUP={WARMUP}')

    rows = []
    for (m, n, k) in SHAPES_7B:
        for R in BITS:
            r = bench_shape(R, m, n, k)
            print(f'[decomp:{LABEL}] ({m:>5}, {k:>5}) @ {R}b  '
                  f'{r["time_us"]:8.1f} us   {r["effective_gbps"]:8.1f} GB/s')
            rows.append(r)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{LABEL}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'label': LABEL,
            'method': 'QTIP fused decompress+matvec',
            'timer': 'eager (torch.utils.benchmark.Timer, fn; sync per iter)',
            'repeats': REPEATS, 'warmup': WARMUP, 'gpu': gpu,
            'rows': rows,
        }, f, indent=2)
    print(f'[decomp:{LABEL}] wrote {out_path}')


if __name__ == '__main__':
    main()
