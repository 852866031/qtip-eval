"""Matvec decomposition -- QTIP fused decompress+matvec, CUDA GRAPH timer.

Path C = the paper's fused CUDA kernel. Reads compressed weight, decodes
via the trellis LUT in registers, accumulates the matmul inline; no
intermediate fp16 weight ever touches HBM.

Timer is CUDA-graph replay with an explicit host sync before stopping.
This is the measurement regime that matches how a production inference
stack (torch.compile(max-autotune), vLLM, TensorRT-LLM, ...) actually
drives this kernel.

Run:
    CUDA_VISIBLE_DEVICES=0 python decomp/run_qtip_graph.py

Writes: output/qtip_graph.json
"""
# -------- configuration (edit here) -----------------------------------------
LABEL = 'qtip_graph'
BITS = (2, 3, 4)
ITER = 200       # matches eager REPEATS for apples-to-apples mean
WARMUP = 50
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

DECOMP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(DECOMP_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'qtip', 'qtip-kernels'))

import qtip_kernels  # noqa: F401

from test_decompress_matvec import kernels, prepare_arguments  # noqa: E402

OUTPUT_DIR = os.path.join(DECOMP_DIR, 'output')


def graph_time(closure):
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
    torch.cuda.synchronize()   # ensure all replays complete before stopping timer
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITER


def bench_shape(R, m, n, k):
    L, S, V = 16, 9, 1  # match the V=1-symbol kernel shipped by qtip-kernels
    out, compressed, x, codebook, _ = prepare_arguments(L, S, R, V, m, n, k)
    kernel = kernels[R][(m, n, k)]

    def run():
        out.zero_()
        kernel(out, compressed, x, codebook)

    ms = graph_time(run)
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
    print(f'[decomp:{LABEL}] GPU={gpu}  ITER={ITER}  WARMUP={WARMUP}')

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
            'timer': 'CUDA graph replay + explicit host sync before end event',
            'iter': ITER, 'warmup': WARMUP, 'gpu': gpu,
            'rows': rows,
        }, f, indent=2)
    print(f'[decomp:{LABEL}] wrote {out_path}')


if __name__ == '__main__':
    main()
