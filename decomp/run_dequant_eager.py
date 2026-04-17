"""Matvec decomposition -- dequant + FP16 matmul (LOWER BOUND), EAGER timer.

Path B timed eagerly. See run_dequant_graph.py for the method explanation;
this file differs only in how the timer runs.

Run:
    CUDA_VISIBLE_DEVICES=0 python decomp/run_dequant_eager.py

Writes: output/dequant_eager.json
"""
# -------- configuration (edit here) -----------------------------------------
LABEL = 'dequant_eager'
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

import torch
import torch.utils.benchmark as bench

DECOMP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DECOMP_DIR, 'output')


def eager_time(fn):
    """Time `fn()` eagerly: fn(); sync per iteration."""
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
    # D2D copy of fp16 weight is an UPPER BOUND on memory traffic of any real
    # dequant kernel (which would read <= 2*m*k bytes and write 2*m*k bytes).
    # That makes this whole path a LOWER BOUND on dequant+matmul wall time.
    W_src = torch.randn(m, k, device='cuda', dtype=torch.float16) / 16
    W_dst = torch.empty_like(W_src)
    x = torch.randn(k, n, device='cuda', dtype=torch.float16) / 16
    out = torch.zeros(m, n, device='cuda', dtype=torch.float16)

    def run_total():
        W_dst.copy_(W_src)
        torch.matmul(W_dst, x, out=out)

    def run_dq():
        W_dst.copy_(W_src)

    ms_total = eager_time(run_total)
    ms_dq = eager_time(run_dq)

    bytes_compressed = m * k * R // 8
    return {
        'shape_mk': [m, k],
        'bits': R,
        'time_us': ms_total * 1000.0,
        'dequant_only_us': ms_dq * 1000.0,
        'bytes_compressed': bytes_compressed,
        'effective_gbps': bytes_compressed / 1e9 / (ms_total / 1e3),
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
                  f'{r["time_us"]:8.1f} us total   '
                  f'{r["dequant_only_us"]:8.1f} us dequant   '
                  f'{r["effective_gbps"]:8.1f} GB/s')
            rows.append(r)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{LABEL}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'label': LABEL,
            'method': 'dequant + fp16 matmul (LB)',
            'timer': 'eager (torch.utils.benchmark.Timer, fn; sync per iter)',
            'repeats': REPEATS, 'warmup': WARMUP, 'gpu': gpu,
            'rows': rows,
        }, f, indent=2)
    print(f'[decomp:{LABEL}] wrote {out_path}')


if __name__ == '__main__':
    main()
