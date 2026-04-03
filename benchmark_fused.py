"""
Benchmark: fused decode+matvec inference path.

Uses the decompress_matvec_qtip CUDA kernel, which decodes the packed trellis
and accumulates into the output vector in a single pass — the full fp16 weight
matrix is never written to memory. This is the real production path inside
BitshiftLinear.forward for batch=1 (single-token generation).

Full operation mirrors BitshiftLinear.forward (rcp=0, batch=1):
  x_rot = HadUt(x * SU) / scale
  out_rot = decompress_matvec_qtip(trellis, x_rot, tlut)   # fused
  out = HadU(out_rot) * (SV * Wscale * scale)

Results saved to results/fused/results.csv.
Run prepare_qtip.py first to generate data/.
"""
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qtip'))

import torch
import torch.utils.benchmark as bench

# Registers torch.ops.quip_lib.decompress_matvec_qtip_* for all known shapes
import lib.codebook  # noqa: F401
import qtip_kernels

from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda, get_hadK

# ── Config ────────────────────────────────────────────────────────────────────
M, N        = 4096, 4096
L, V        = 16, 2
TLUT_BITS   = 9
K_BITS      = [2, 3, 4]
REPEATS     = 200
SCALE       = 32   # fixed normalisation in BitshiftLinear

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WX_PATH  = os.path.join(DATA_DIR, 'W_x.pt')

def gpu_slug():
    """Return a filesystem-safe GPU name, e.g. 'RTX_5090'."""
    name = torch.cuda.get_device_name(0)
    name = name.replace('NVIDIA ', '').replace('GeForce ', '')
    return name.replace(' ', '_').replace('/', '_')

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


def time_breakdown(fused_op, packed_g, tlut_g, SU_g, SV_out_g, x_fp32_g,
                   had_left, K_left, had_right, K_right, repeats=REPEATS):
    """
    Time each step of fused_matvec individually.
    Each step is timed in isolation (other steps pre-computed) so the numbers
    are comparable and sum to roughly the total end-to-end latency.

    Steps:
      input_rot      — (x * SU) → HadUt → /SCALE
      fused_decode   — decompress_matvec_qtip (trellis decode + accumulate)
        packed_read  — proxy: read all packed bytes from HBM (memory floor)
        decode_overhead — derived: fused_decode - packed_read (compute cost)
      output_rot     — HadU(out)
      output_scale   — out * SV_out
    """
    # Pre-compute intermediate tensors so each step can be timed independently
    x_rot = matmul_hadUt_cuda(
        (x_fp32_g * SU_g).unsqueeze(0), had_left, K_left
    ) / SCALE
    out_fused = fused_op(packed_g, x_rot, tlut_g)
    out_rot   = matmul_hadU_cuda(out_fused, had_right, K_right)
    torch.cuda.synchronize()

    def step_input_rot():
        return matmul_hadUt_cuda(
            (x_fp32_g * SU_g).unsqueeze(0), had_left, K_left
        ) / SCALE

    def step_fused_decode():
        return fused_op(packed_g, x_rot, tlut_g)

    def step_output_rot():
        return matmul_hadU_cuda(out_fused, had_right, K_right)

    def step_output_scale():
        return out_rot * SV_out_g

    def step_packed_read():
        # Proxy for the pure HBM read cost of the packed trellis data.
        # Forces all packed bytes through the memory hierarchy (L2 → HBM)
        # without any decode compute. This is the theoretical minimum time
        # that the fused decode kernel could take if it were purely memory-bound.
        return packed_g.sum()

    fused_decode_ms = time_ms(step_fused_decode, repeats)
    packed_read_ms  = time_ms(step_packed_read,  repeats)

    return {
        'input_rot_ms':       time_ms(step_input_rot,   repeats),
        'fused_decode_ms':    fused_decode_ms,
        'packed_read_ms':     packed_read_ms,
        'decode_overhead_ms': max(0.0, fused_decode_ms - packed_read_ms),
        'output_rot_ms':      time_ms(step_output_rot,  repeats),
        'output_scale_ms':    time_ms(step_output_scale, repeats),
    }


def time_kernel_internal(K, packed_g, x_rot, tlut_g, repeats=REPEATS):
    """
    Call the clock64()-instrumented kernel variant to measure time spent in
    each phase inside the CUDA kernel itself (averaged over 128 blocks).

    Phases:
      codebook  — loading the LUT into shared memory
      loop      — the ki loop: trellis unpack + LUT lookup + MMA accumulate
      reduce    — warp-level reduction + write output
      total     — codebook + loop + reduce (full kernel wall time per SM)

    Returns dict of *_ms keys (converted from nanoseconds via globaltimer).
    """
    # globaltimer gives nanoseconds; no clock rate needed
    timing = torch.zeros(128 * 4, dtype=torch.int64, device='cuda')
    out    = torch.zeros((M, 1), dtype=torch.float32, device='cuda')

    timed_fn = getattr(qtip_kernels,
                       f"decompress_matvec_timed_16_9_{K}_1_{M}_1_{N}")

    # Warm up
    for _ in range(5):
        timing.zero_()
        out.zero_()
        timed_fn(out,
                 packed_g.reshape(-1).view(torch.int32),
                 x_rot.to(torch.float16).T,
                 tlut_g.reshape(-1),
                 timing)
    torch.cuda.synchronize()

    # Collect across repeats (accumulate, then average)
    accum = torch.zeros(4, dtype=torch.float64, device='cuda')
    for _ in range(repeats):
        timing.zero_()
        out.zero_()
        timed_fn(out,
                 packed_g.reshape(-1).view(torch.int32),
                 x_rot.to(torch.float16).T,
                 tlut_g.reshape(-1),
                 timing)
        torch.cuda.synchronize()
        # timing: [128*4] int64 → (128, 4); mean over blocks
        accum += timing.view(128, 4).double().mean(dim=0)

    means = (accum / repeats).cpu()  # [codebook, loop, reduce, total] in nanoseconds
    to_ms = lambda c: (c / 1e6).item()  # ns → ms
    return {
        'kernel_codebook_ms': to_ms(means[0]),
        'kernel_loop_ms':     to_ms(means[1]),
        'kernel_reduce_ms':   to_ms(means[2]),
        'kernel_total_ms':    to_ms(means[3]),
    }


def main():
    if not os.path.exists(WX_PATH):
        sys.exit(f"ERROR: {WX_PATH} not found. Run prepare_qtip.py first.")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', gpu_slug(), 'fused')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    wx     = torch.load(WX_PATH, weights_only=True)
    W_fp32 = wx['W'].float()
    x      = wx['x']   # fp32 cpu, shape (N,)

    fp16_bytes = W_fp32.numel() * 2
    ref_out    = W_fp32 @ x

    print(f"\nWeight matrix: {M}×{N}  |  FP16 = {fp16_bytes / 1e6:.1f} MB")
    print(f"Path: fused matvec  (decompress_matvec_qtip, batch=1, no weight materialisation)\n")

    # FP16 baseline
    W_fp16_g = W_fp32.half().cuda()
    x_fp16_g = x.half().cuda()
    fp16_ms  = time_ms(lambda: W_fp16_g @ x_fp16_g)
    fp16_bw  = (fp16_bytes + (M + N) * 2) / fp16_ms / 1e6

    # Hadamard params (power-of-2 dims → had=None, K_had=1)
    had_left,  K_left  = get_hadK(N)
    had_right, K_right = get_hadK(M)

    rows = []

    hdr = (f"{'Config':<18} {'W MSE':>12} {'Out MSE':>12} "
           f"{'W bytes':>12} {'Ratio':>7} {'ms':>8} {'GB/s':>8} "
           f"{'in_rot':>8} {'decode':>8} {'out_rot':>8} {'scale':>8} "
           f"{'k_cb':>8} {'k_loop':>8} {'k_red':>8} {'k_tot':>8}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'FP16 baseline':<18} {'—':>12} {'—':>12} "
          f"{fp16_bytes:>12,} {'1.00x':>7} {fp16_ms:>8.3f} {fp16_bw:>8.1f} "
          f"{'—':>8} {'—':>8} {'—':>8} {'—':>8} "
          f"{'—':>8} {'—':>8} {'—':>8} {'—':>8}")
    rows.append(dict(config='FP16 baseline', w_mse='', out_mse='',
                     w_bytes=fp16_bytes, ratio=1.0, ms=fp16_ms, gb_s=fp16_bw,
                     input_rot_ms='', fused_decode_ms='', packed_read_ms='',
                     decode_overhead_ms='', output_rot_ms='', output_scale_ms='',
                     kernel_codebook_ms='', kernel_loop_ms='',
                     kernel_reduce_ms='', kernel_total_ms=''))

    for K in K_BITS:
        path = quant_path(K)
        if not os.path.exists(path):
            print(f"  [missing K={K}] run prepare_qtip.py first")
            continue

        d = torch.load(path, weights_only=True)
        hatW       = d['hatW']
        packed     = d['packed']
        tlut       = d['tlut']
        SU         = d['SU']
        SV         = d['SV']
        Wscale     = float(d['Wscale'])
        use_kernel = bool(d['use_kernel'])

        if not use_kernel:
            print(f"{'QTIP K='+str(K):<18}  [fused kernel not available for this shape]")
            continue

        # Quality (identical to dequant path — same weights, different compute path)
        w_mse   = (W_fp32 - hatW).pow(2).mean().item()
        out_mse = (ref_out - hatW @ x).pow(2).mean().item()

        trellis_bytes = packed.numel() * packed.element_size()
        quant_bytes   = (trellis_bytes
                         + SU.numel() * 4
                         + SV.numel() * 4
                         + (2**TLUT_BITS) * V * 2)
        ratio = fp16_bytes / quant_bytes

        # GPU tensors
        packed_g  = packed.cuda()
        tlut_g    = tlut.half().cuda()
        SU_g      = SU.cuda()
        # Wscale absorbed into SV (matches unpack_quip / QuantizedLinear convention)
        SV_out_g  = (SV * Wscale * SCALE).cuda()
        x_fp32_g  = x.float().cuda()

        fused_op = getattr(torch.ops.quip_lib,
                           f"decompress_matvec_qtip_{M}_1_{N}_{K}")

        def fused_matvec():
            # Rotate input into weight's Hadamard space, normalise by SCALE
            # x must stay 2D (1, N) — the kernel calls x.T internally
            x_rot = matmul_hadUt_cuda(
                (x_fp32_g * SU_g).unsqueeze(0), had_left, K_left
            ) / SCALE                          # shape: (1, N)
            # Fused decode + accumulate; output shape: (1, M) fp32
            out = fused_op(packed_g, x_rot, tlut_g)
            # Unrotate and apply output scale (SV * Wscale * SCALE)
            out = matmul_hadU_cuda(out, had_right, K_right)
            return out * SV_out_g

        q_ms = time_ms(fused_matvec)
        q_bw = (quant_bytes + (M + N) * 2) / q_ms / 1e6

        bd = time_breakdown(fused_op, packed_g, tlut_g, SU_g, SV_out_g, x_fp32_g,
                            had_left, K_left, had_right, K_right)

        # Pre-compute x_rot for kernel-internal timing (same as time_breakdown uses)
        x_rot_for_timing = matmul_hadUt_cuda(
            (x_fp32_g * SU_g).unsqueeze(0), had_left, K_left
        ) / SCALE
        kd = time_kernel_internal(K, packed_g, x_rot_for_timing, tlut_g)

        print(f"{'QTIP K='+str(K):<18} {w_mse:>12.6f} {out_mse:>12.6f} "
              f"{quant_bytes:>12,} {ratio:>6.2f}x {q_ms:>8.3f} {q_bw:>8.1f} "
              f"{bd['input_rot_ms']:>8.3f} {bd['fused_decode_ms']:>8.3f} "
              f"(mem={bd['packed_read_ms']:.3f} cmp={bd['decode_overhead_ms']:.3f}) "
              f"{bd['output_rot_ms']:>8.3f} {bd['output_scale_ms']:>8.3f} "
              f"  kernel: cb={kd['kernel_codebook_ms']:.3f} "
              f"loop={kd['kernel_loop_ms']:.3f} "
              f"red={kd['kernel_reduce_ms']:.3f} "
              f"tot={kd['kernel_total_ms']:.3f}")
        rows.append(dict(config=f'QTIP K={K}', w_mse=f'{w_mse:.6f}',
                         out_mse=f'{out_mse:.6f}', w_bytes=quant_bytes,
                         ratio=f'{ratio:.2f}', ms=f'{q_ms:.3f}', gb_s=f'{q_bw:.1f}',
                         input_rot_ms=f"{bd['input_rot_ms']:.3f}",
                         fused_decode_ms=f"{bd['fused_decode_ms']:.3f}",
                         packed_read_ms=f"{bd['packed_read_ms']:.3f}",
                         decode_overhead_ms=f"{bd['decode_overhead_ms']:.3f}",
                         output_rot_ms=f"{bd['output_rot_ms']:.3f}",
                         output_scale_ms=f"{bd['output_scale_ms']:.3f}",
                         kernel_codebook_ms=f"{kd['kernel_codebook_ms']:.4f}",
                         kernel_loop_ms=f"{kd['kernel_loop_ms']:.4f}",
                         kernel_reduce_ms=f"{kd['kernel_reduce_ms']:.4f}",
                         kernel_total_ms=f"{kd['kernel_total_ms']:.4f}"))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'w_mse', 'out_mse',
                                          'w_bytes', 'ratio', 'ms', 'gb_s',
                                          'input_rot_ms', 'fused_decode_ms',
                                          'packed_read_ms', 'decode_overhead_ms',
                                          'output_rot_ms', 'output_scale_ms',
                                          'kernel_codebook_ms', 'kernel_loop_ms',
                                          'kernel_reduce_ms', 'kernel_total_ms'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")

    print("\nNotes:")
    print("  W MSE / Out MSE  — computed from hatW on CPU, same for both inference paths")
    print("  Fused matvec     — decompress_matvec_qtip never writes full fp16 hatW;")
    print("                     the real latency of batch=1 token generation per layer")
    print("  GB/s             — (quant weight bytes + x/y IO) / latency")
    print("  in_rot           — (x * SU) → HadUt → /SCALE  [input rotation]")
    print("  decode           — decompress_matvec_qtip  [trellis decode + accumulate]")
    print("  out_rot          — HadU(out)  [output rotation]")
    print("  scale            — out * (SV * Wscale * SCALE)  [output scaling]")
    print("  Breakdown steps are timed individually; they won't sum exactly to total")
    print("  kernel_* cols  — clock64() SM-cycle timing inside the CUDA kernel")
    print("    k_cb   = LUT load into shared memory (one-time per block)")
    print("    k_loop = ki loop: trellis unpack + LUT lookup + MMA accumulate")
    print("    k_red  = warp-level reduction + write output")
    print("    k_tot  = total kernel time from SM clock (should ≈ fused_decode_ms)")


if __name__ == '__main__':
    main()
