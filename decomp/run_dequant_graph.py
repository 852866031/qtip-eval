"""Matvec decomposition -- dequant + FP16 matmul (LOWER BOUND), CUDA GRAPH timer.

Path B = the two-step "decompress into fp16 buffer, then cuBLAS matvec"
path that any non-fused quantized inference stack has to take.

We approximate the dequant step with a device-to-device fp16 copy. A real
dequant kernel would read <= 2*m*k bytes of compressed source and write
2*m*k bytes of fp16 weight; the D2D copy reads 2*m*k and writes 2*m*k, so
the copy is an OVERestimate of the real kernel's memory traffic and this
whole path is therefore a LOWER BOUND on any real dequant+matmul.

Timer is CUDA-graph replay with an explicit host sync before stopping.

Run:
    CUDA_VISIBLE_DEVICES=0 python decomp/run_dequant_graph.py

Writes: output/dequant_graph.json
"""
# -------- configuration (edit here) -----------------------------------------
LABEL = 'dequant_graph'
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

import torch

DECOMP_DIR = os.path.dirname(os.path.abspath(__file__))
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
    W_src = torch.randn(m, k, device='cuda', dtype=torch.float16) / 16
    W_dst = torch.empty_like(W_src)
    x = torch.randn(k, n, device='cuda', dtype=torch.float16) / 16
    out = torch.zeros(m, n, device='cuda', dtype=torch.float16)

    def run_total():
        W_dst.copy_(W_src)
        torch.matmul(W_dst, x, out=out)

    def run_dq():
        W_dst.copy_(W_src)

    ms_total = graph_time(run_total)
    ms_dq = graph_time(run_dq)

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
    print(f'[decomp:{LABEL}] GPU={gpu}  ITER={ITER}  WARMUP={WARMUP}')

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
            'timer': 'CUDA graph replay + explicit host sync before end event',
            'iter': ITER, 'warmup': WARMUP, 'gpu': gpu,
            'rows': rows,
        }, f, indent=2)
    print(f'[decomp:{LABEL}] wrote {out_path}')


if __name__ == '__main__':
    main()
