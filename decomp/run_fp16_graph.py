"""Matvec decomposition -- FP16 cuBLAS matvec, CUDA GRAPH timer.

Path A (FP16) timed by capturing a single call into a CUDA graph and
replaying it `ITER` times inside a single CUDA-event range. An explicit
`torch.cuda.synchronize()` is issued BEFORE recording the end event so
the timer reflects real, completed GPU execution of all replays.

Run:
    CUDA_VISIBLE_DEVICES=0 python decomp/run_fp16_graph.py

Writes: output/fp16_graph.json
"""
# -------- configuration (edit here) -----------------------------------------
LABEL = 'fp16_graph'
ITER = 200       # timed graph replays (matches eager REPEATS for apples-to-apples mean)
WARMUP = 50      # untimed eager warmup before capture
SHAPES_7B = [
    (4096,  1, 4096),
    (4096,  1, 11008),
    (11008, 1, 4096),
]
# ----------------------------------------------------------------------------

import json
import os

import torch

DECOMP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DECOMP_DIR, 'output')


def graph_time(closure):
    """Capture closure as a CUDA graph and time ITER replays.

    Explicit host sync before stopping the timer guarantees we measure
    real execution of all queued replays, not just the time the host
    spent queueing them.
    """
    for _ in range(WARMUP):
        closure()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        closure()
    g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITER):
        g.replay()
    torch.cuda.synchronize()   # force all replays to complete before stopping timer
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITER  # ms/replay


def bench_shape(m, n, k):
    W = torch.randn(m, k, device='cuda', dtype=torch.float16) / 16
    x = torch.randn(k, n, device='cuda', dtype=torch.float16) / 16
    out = torch.zeros(m, n, device='cuda', dtype=torch.float16)
    ms = graph_time(lambda: torch.matmul(W, x, out=out))
    return {
        'shape_mk': [m, k],
        'time_us': ms * 1000.0,
        'bytes_weight': W.nbytes,
        'effective_gbps': W.nbytes / 1e9 / (ms / 1e3),
    }


def main():
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0)
    print(f'[decomp:{LABEL}] GPU={gpu}  ITER={ITER}  WARMUP={WARMUP}')

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
            'timer': 'CUDA graph replay + explicit host sync before end event',
            'iter': ITER, 'warmup': WARMUP, 'gpu': gpu,
            'rows': rows,
        }, f, indent=2)
    print(f'[decomp:{LABEL}] wrote {out_path}')


if __name__ == '__main__':
    main()
