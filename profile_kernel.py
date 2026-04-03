"""
Minimal script for Nsight Compute profiling of the fused decode kernel.

Usage:
    ncu --kernel-name kernel_decompress_matvec \
        --launch-skip 5 --launch-count 1       \
        --set full -o profile_out              \
        python profile_kernel.py

Then open profile_out.ncu-rep in Nsight Compute GUI,
or inspect on command line:
    ncu --import profile_out.ncu-rep
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qtip'))

import torch
import lib.codebook  # noqa: registers torch.ops.quip_lib.*
from lib.utils.matmul_had import matmul_hadUt_cuda, get_hadK

K_BITS   = int(os.environ.get('K_BITS', '2'))   # override with K_BITS=4 python profile_kernel.py
M, N     = 4096, 4096
SCALE    = 32
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WARMUP   = 5

d = torch.load(os.path.join(DATA_DIR, f'qtip_K{K_BITS}.pt'), weights_only=True)
wx = torch.load(os.path.join(DATA_DIR, 'W_x.pt'), weights_only=True)

packed_g = d['packed'].cuda()
tlut_g   = d['tlut'].half().cuda()
SU_g     = d['SU'].cuda()
x_fp32_g = wx['x'].float().cuda()

had_left, K_left = get_hadK(N)
x_rot = matmul_hadUt_cuda((x_fp32_g * SU_g).unsqueeze(0), had_left, K_left) / SCALE

fused_op = getattr(torch.ops.quip_lib,
                   f"decompress_matvec_qtip_{M}_1_{N}_{K_BITS}")

# Warmup — ncu will skip these (--launch-skip 5)
for _ in range(WARMUP):
    fused_op(packed_g, x_rot, tlut_g)
torch.cuda.synchronize()

# The one launch ncu will profile (--launch-count 1)
out = fused_op(packed_g, x_rot, tlut_g)
torch.cuda.synchronize()

print(f"K={K_BITS}: output sum = {out.sum().item():.4f}")
