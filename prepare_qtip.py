"""
Quantize a 4096×4096 FP16 weight matrix with QTIP at K=2/3/4 bits.
Always re-runs quantization (overwrites existing files).

Outputs:
  data/W_x.pt              — W (fp16), x (fp32)  [generated once if missing]
  data/qtip_K{K}.pt        — packed trellis, SU/SV, tlut, lut, hatW, Wscale
  data/quant_timings.csv   — per-step timing breakdown

Run plot_results.py to generate plots from the saved CSV.
"""
import csv
import sys
import os
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qtip'))

import torch

from lib.codebook.bitshift import bitshift_codebook

# Triton >= 3.x incompatibility: libdevice.pow(int,int) fails in the compiled
# update kernel. Disable dynamo for this one method; everything else stays JIT.
bitshift_codebook.update = torch._dynamo.disable(bitshift_codebook.update)

from lib.utils.math_utils import block_LDL, regularize_H
from lib.utils.matmul_had import matmul_hadU, matmul_hadUt
from lib.utils.kernel_check import has_kernel
from lib.algo import ldlq

# ── Config ───────────────────────────────────────────────────────────────────
M, N           = 4096, 4096
DEVICE         = 'cuda'
L, V           = 16, 2
TD_X, TD_Y     = 16, 16
TLUT_BITS      = 9
DECODE_MODE    = 'quantlut_sym'
SCALE_OVERRIDE = 0.9
SIGMA_REG      = 0.01
K_BITS         = [2, 3, 4]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WX_PATH  = os.path.join(DATA_DIR, 'W_x.pt')
CSV_PATH = os.path.join(DATA_DIR, 'quant_timings.csv')

BREAKDOWN_STEPS = ['preprocess', 'block_ldl', 'viterbi', 'postprocess']

def quant_path(K):
    return os.path.join(DATA_DIR, f'qtip_K{K}.pt')


def tick():
    """Return current time after syncing CUDA so GPU work is included."""
    torch.cuda.synchronize()
    return time.perf_counter()


def generate_wx():
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.manual_seed(0)
    W = torch.randn(M, N, dtype=torch.float16)
    torch.manual_seed(1)
    x = torch.randn(N, dtype=torch.float32)
    torch.save({'W': W, 'x': x}, WX_PATH)
    print(f"Generated W ({M}×{N} fp16) and x ({N} fp32) → {WX_PATH}")
    return W, x


def quantize(W_fp32, K):
    """
    Quantize W at K bits with per-step timing.
    Returns (artefacts_dict, timings_dict).

    Timing breakdown:
      preprocess  — codebook init, sign vectors, Hadamard rotation of W and H,
                    scale computation
      block_ldl   — Cholesky + block LDL decomposition of rotated Hessian
      viterbi     — LDLQ trellis search (dominant step)
      postprocess — rescale hatWr, pack trellis into kernel format,
                    reverse Hadamard to recover hatW in original weight space
    """
    args = types.SimpleNamespace(
        td_x=TD_X, td_y=TD_Y, L=L, K=K, V=V,
        tlut_bits=TLUT_BITS, decode_mode=DECODE_MODE,
        scale_override=SCALE_OVERRIDE,
    )
    m, n = W_fp32.shape
    timings = {}

    # ── preprocess ────────────────────────────────────────────────────────────
    t0 = tick()

    cb = bitshift_codebook(L=L, K=K, V=V, tlut_bits=TLUT_BITS,
                           decode_mode=DECODE_MODE).to(DEVICE).float()
    torch.manual_seed(42)
    SU = (torch.randn(n, device=DEVICE).sign() + 1e-5).sign().float()
    SV = (torch.randn(m, device=DEVICE).sign() + 1e-5).sign().float()

    W   = W_fp32.to(DEVICE).float()
    Wr  = matmul_hadUt(matmul_hadUt(W.T * SV).T * SU)
    HR  = regularize_H(torch.eye(n, device=DEVICE), SIGMA_REG)
    HRr = matmul_hadUt(matmul_hadUt(HR * SU).T * SU)

    cb_lut_rms = cb.lut.to(torch.float64).square().mean().sqrt().float()
    Wscale = Wr.square().mean().sqrt() / (cb_lut_rms * SCALE_OVERRIDE)
    Wr = Wr / Wscale

    timings['preprocess'] = tick() - t0

    # ── block_ldl ─────────────────────────────────────────────────────────────
    t0 = tick()

    result = block_LDL(HRr, TD_Y)
    if result is None:
        raise RuntimeError("block_LDL failed")
    LRr, _ = result
    idx = torch.arange(n, device=DEVICE)
    LRr[idx, idx] = 0

    timings['block_ldl'] = tick() - t0

    # ── viterbi (LDLQ) ────────────────────────────────────────────────────────
    t0 = tick()

    use_kernel = has_kernel(DECODE_MODE, L, K, V, TLUT_BITS, TD_X, TD_Y)
    hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, args, for_kernel=use_kernel)

    timings['viterbi'] = tick() - t0

    # ── postprocess ───────────────────────────────────────────────────────────
    t0 = tick()

    hatWr = hatWr * Wscale

    Qidxs_cpu = Qidxs.cpu()
    packed = cb.pack_trellis(
        Qidxs_cpu.reshape(m // TD_X, TD_X, n // TD_Y, TD_Y // V)
                 .transpose(1, 2)
                 .reshape(-1, TD_X * TD_Y // V)
    )
    if use_kernel:
        packed = (packed.view(torch.uint8)
                        .view(-1, 2).flip((-1,))
                        .reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8, K)
                        .permute(0, 2, 4, 3, 1, 5).flip((-1,))
                        .contiguous().flatten()
                        .view(torch.int16).reshape(packed.shape))
    else:
        packed = packed.view(torch.int16)

    C    = matmul_hadU(hatWr)
    hatW = (matmul_hadU((C * SU).T) * SV).T

    timings['postprocess'] = tick() - t0

    timings['total'] = sum(timings[s] for s in BREAKDOWN_STEPS)

    artefacts = dict(
        packed=packed.cpu(),
        SU=SU.cpu(), SV=SV.cpu(),
        tlut=cb.tlut.half().cpu(),
        lut=cb.lut.half().cpu(),
        Wscale=Wscale.cpu(),
        hatW=hatW.cpu(),
        use_kernel=use_kernel,
        quant_time_s=timings['total'],
    )
    return artefacts, timings


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(all_timings):
    """Write per-step timings for all K values to CSV."""
    fields = ['K', 'total_s'] + [f'{s}_s' for s in BREAKDOWN_STEPS]
    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for K in K_BITS:
            t = all_timings[K]
            w.writerow({
                'K': K,
                'total_s': f"{t['total']:.3f}",
                **{f'{s}_s': f"{t[s]:.3f}" for s in BREAKDOWN_STEPS},
            })
    print(f"Saved timings → {CSV_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(WX_PATH):
        generate_wx()
    else:
        print(f"Using existing {WX_PATH}")

    d = torch.load(WX_PATH, weights_only=True)
    W_fp32 = d['W'].float()

    all_timings = {}
    for K in K_BITS:
        print(f"\n── K={K} ──────────────────────────────────────────────────")
        artefacts, timings = quantize(W_fp32, K)
        all_timings[K] = timings

        torch.save(artefacts, quant_path(K))
        print(f"  preprocess : {timings['preprocess']:.2f}s")
        print(f"  block_ldl  : {timings['block_ldl']:.2f}s")
        print(f"  viterbi    : {timings['viterbi']:.2f}s")
        print(f"  postprocess: {timings['postprocess']:.2f}s")
        print(f"  total      : {timings['total']:.2f}s")
        print(f"  Saved → {quant_path(K)}")

    save_csv(all_timings)
    print("\nDone. Run benchmark_dequant.py / benchmark_fused.py, then plot_results.py.")


if __name__ == '__main__':
    main()
