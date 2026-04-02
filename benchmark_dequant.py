"""
Benchmark: dequantize-then-gemm inference path.

For each K=2/3/4, fully decodes the packed trellis into a fp16 weight matrix
(via decode_compressed), then does x @ hatW.T as a separate operation.
This is the batch>1 / training path — it materialises the full weight matrix.

Results saved to results/dequant/results.csv.
Run prepare_qtip.py first to generate data/.
"""
import csv
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qtip'))

import torch
import torch.utils.benchmark as bench

from lib.utils.kernel_check import has_kernel

# ── Config ────────────────────────────────────────────────────────────────────
M, N        = 4096, 4096
L, V        = 16, 2
TD_X, TD_Y  = 16, 16
TLUT_BITS   = 9
DECODE_MODE = 'quantlut_sym'
K_BITS      = [2, 3, 4]
REPEATS     = 200

DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'dequant')
WX_PATH     = os.path.join(DATA_DIR, 'W_x.pt')

def quant_path(K):
    return os.path.join(DATA_DIR, f'qtip_K{K}.pt')


def time_ms(fn, repeats=REPEATS):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = bench.Timer(
        stmt='fn(); torch.cuda.synchronize()',
        globals={'fn': fn, 'torch': torch},
        num_threads=1,
    )
    return t.timeit(repeats).mean * 1e3


def main():
    if not os.path.exists(WX_PATH):
        sys.exit(f"ERROR: {WX_PATH} not found. Run prepare_qtip.py first.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    wx     = torch.load(WX_PATH, weights_only=True)
    W_fp32 = wx['W'].float()
    x      = wx['x']

    fp16_bytes = W_fp32.numel() * 2
    ref_out    = W_fp32 @ x

    print(f"\nWeight matrix: {M}×{N}  |  FP16 = {fp16_bytes / 1e6:.1f} MB")
    print(f"Path: dequant+gemm  (decode_compressed → full fp16 hatW → x @ hatW.T)\n")

    W_fp16_g = W_fp32.half().cuda()
    x_fp16_g = x.half().cuda()
    fp16_ms  = time_ms(lambda: W_fp16_g @ x_fp16_g)
    fp16_bw  = (fp16_bytes + (M + N) * 2) / fp16_ms / 1e6

    rows = []  # for CSV

    hdr = (f"{'Config':<18} {'W MSE':>12} {'Out MSE':>12} "
           f"{'W bytes':>12} {'Ratio':>7} {'ms':>8} {'GB/s':>8}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'FP16 baseline':<18} {'—':>12} {'—':>12} "
          f"{fp16_bytes:>12,} {'1.00x':>7} {fp16_ms:>8.3f} {fp16_bw:>8.1f}")
    rows.append(dict(config='FP16 baseline', w_mse='', out_mse='',
                     w_bytes=fp16_bytes, ratio=1.0, ms=fp16_ms, gb_s=fp16_bw))

    for K in K_BITS:
        path = quant_path(K)
        if not os.path.exists(path):
            print(f"  [missing K={K}] run prepare_qtip.py first")
            continue

        d = torch.load(path, weights_only=True)
        hatW       = d['hatW']
        packed     = d['packed']
        lut        = d['lut']
        tlut       = d['tlut']
        use_kernel = bool(d['use_kernel'])

        w_mse   = (W_fp32 - hatW).pow(2).mean().item()
        out_mse = (ref_out - hatW @ x).pow(2).mean().item()

        trellis_bytes = packed.numel() * packed.element_size()
        quant_bytes   = (trellis_bytes
                         + d['SU'].numel() * 4
                         + d['SV'].numel() * 4
                         + (2**TLUT_BITS) * V * 2)
        ratio = fp16_bytes / quant_bytes

        packed_g = packed.cuda()
        lut_g    = lut.cuda()
        x_g      = x.half().cuda()

        if use_kernel:
            from lib.utils.kernel_decompress import decode_compressed
            def quant_fwd():
                hw = decode_compressed(L, TLUT_BITS, K, int(math.log2(V)),
                                       M, N, packed_g.view(-1), lut_g.T)
                return x_g @ hw.T
        else:
            from lib.codebook.bitshift import bitshift_codebook
            cb = bitshift_codebook(L=L, K=K, V=V, tlut_bits=TLUT_BITS,
                                   decode_mode=DECODE_MODE, tlut=tlut).cuda()
            def quant_fwd():
                trel = cb.unpack_trellis(packed_g, TD_X * TD_Y)
                hw = (cb.recons(trel).transpose(0, 1).transpose(1, 2)
                             .reshape(M // TD_X, N // TD_Y, TD_X, TD_Y)
                             .transpose(1, 2).reshape(M, N))
                return x_g @ hw.T

        q_ms = time_ms(quant_fwd)
        q_bw = (quant_bytes + (M + N) * 2) / q_ms / 1e6

        print(f"{'QTIP K='+str(K):<18} {w_mse:>12.6f} {out_mse:>12.6f} "
              f"{quant_bytes:>12,} {ratio:>6.2f}x {q_ms:>8.3f} {q_bw:>8.1f}")
        rows.append(dict(config=f'QTIP K={K}', w_mse=f'{w_mse:.6f}',
                         out_mse=f'{out_mse:.6f}', w_bytes=quant_bytes,
                         ratio=f'{ratio:.2f}', ms=f'{q_ms:.3f}', gb_s=f'{q_bw:.1f}'))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config','w_mse','out_mse',
                                          'w_bytes','ratio','ms','gb_s'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")

    print("\nNotes:")
    print("  W MSE    = mean((W - hatW)²)  in fp32 weight space")
    print("  Out MSE  = mean((Wx - hatW·x)²)  for random fp32 x")
    print("  Throughput: decode_compressed writes full fp16 hatW, then x @ hatW.T")
    print("  GB/s     = (quant weight bytes + x/y IO) / latency")


if __name__ == '__main__':
    main()
