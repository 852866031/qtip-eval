"""
Generate all benchmark plots from data/ and results/ CSVs.

Quantization timing (from data/quant_timings.csv):
  results/quant_time.png            — stacked bar: per-step breakdown + compression ratio

Per-path plots (only data from that path):
  results/dequant/latency_bar.png   — latency bar + Out MSE subplot
  results/dequant/bandwidth_bar.png
  results/fused/latency_bar.png     — latency bar + Out MSE subplot
  results/fused/bandwidth_bar.png

Comparison plots (require both results CSVs):
  results/quality_vs_k.png          — W MSE and Out MSE vs K
  results/latency_comparison.png    — latency grouped bars: dequant vs fused vs FP16
  results/bandwidth_comparison.png  — GB/s grouped bars: dequant vs fused vs FP16
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, 'results')
DATA_DIR    = os.path.join(ROOT, 'data')

QUANT_TIMING_CSV = os.path.join(DATA_DIR, 'quant_timings.csv')


def discover_gpu_dirs():
    """Return sorted list of GPU slug names that have at least one results CSV."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    gpus = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        gpu_dir = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(gpu_dir):
            continue
        has_dequant = os.path.exists(os.path.join(gpu_dir, 'dequant', 'results.csv'))
        has_fused   = os.path.exists(os.path.join(gpu_dir, 'fused',   'results.csv'))
        if has_dequant or has_fused:
            gpus.append(entry)
    return gpus

# ── Style ─────────────────────────────────────────────────────────────────────
FP16_COLOR   = '#555555'
K_COLORS     = {2: '#4c72b0', 3: '#dd8452', 4: '#55a868'}
DEQUANT_COLOR = '#e07b54'
FUSED_COLOR   = '#5b9bd5'

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})


# ── Quant timing constants (must match prepare_qtip.py) ──────────────────────
BREAKDOWN_STEPS = ['preprocess', 'block_ldl', 'viterbi', 'postprocess']
STEP_COLORS = {
    'preprocess':  '#5b9bd5',
    'block_ldl':   '#ed7d31',
    'viterbi':     '#a9d18e',
    'postprocess': '#ffc000',
}
STEP_LABELS = {
    'preprocess':  'Preprocess\n(Hadamard rotation)',
    'block_ldl':   'Block LDL\n(Cholesky)',
    'viterbi':     'Viterbi\n(LDLQ trellis search)',
    'postprocess': 'Postprocess\n(pack + reverse rotation)',
}

# ── Fused inference breakdown constants (must match benchmark_fused.py) ───────
FUSED_STEPS = ['input_rot_ms', 'fused_decode_ms', 'output_rot_ms', 'output_scale_ms']
FUSED_STEP_COLORS = {
    'input_rot_ms':    '#aec6e8',
    'fused_decode_ms': '#2c6fad',
    'output_rot_ms':   '#6aaed6',
    'output_scale_ms': '#c6dbef',
}
FUSED_STEP_LABELS = {
    'input_rot_ms':    'Input rotation\n(x·SU → HadUt / scale)',
    'fused_decode_ms': 'Fused decode\n(trellis → accumulate)',
    'output_rot_ms':   'Output rotation\n(HadU)',
    'output_scale_ms': 'Output scaling\n(· SV·Wscale·scale)',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_quant_timings(path):
    """Return {K: {step: float, 'total': float}} or None if file missing."""
    if not os.path.exists(path):
        return None
    timings = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            K = int(row['K'])
            timings[K] = {s: float(row[f'{s}_s']) for s in BREAKDOWN_STEPS}
            timings[K]['total'] = float(row['total_s'])
    return timings


def load_csv(path):
    """Return (fp16_row, {K: row}) parsed from a results CSV.

    Breakdown columns (input_rot_ms etc.) are included when present; None otherwise.
    """
    if not os.path.exists(path):
        return None, None
    fp16 = None
    quant = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['config'] == 'FP16 baseline':
                fp16 = {
                    'ms':      float(row['ms']),
                    'gb_s':    float(row['gb_s']),
                    'w_bytes': int(row['w_bytes']),
                }
            elif row['config'].startswith('QTIP K='):
                K = int(row['config'].split('=')[1])
                entry = {
                    'ms':      float(row['ms']),
                    'gb_s':    float(row['gb_s']),
                    'w_bytes': int(row['w_bytes']),
                    'ratio':   float(row['ratio']),
                    'w_mse':   float(row['w_mse'])   if row['w_mse']   else None,
                    'out_mse': float(row['out_mse']) if row['out_mse'] else None,
                }
                for step in FUSED_STEPS:
                    entry[step] = float(row[step]) if row.get(step) else None
                quant[K] = entry
    return fp16, quant


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def k_labels(ks):
    return [f'K={k}' for k in ks]


# ── Quant timing plot ─────────────────────────────────────────────────────────

def plot_quant_breakdown(timings, quant, out_path):
    """Stacked quant-time breakdown (left) + compression ratio (right)."""
    ks     = sorted(timings.keys())
    labels = [f'K={k}' for k in ks]

    fig, (ax_time, ax_ratio) = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: stacked timing breakdown ───────────────────────────────────────
    bottoms = [0.0] * len(ks)
    for step in BREAKDOWN_STEPS:
        vals = [timings[k][step] for k in ks]
        bars = ax_time.bar(labels, vals, bottom=bottoms,
                           color=STEP_COLORS[step], label=STEP_LABELS[step],
                           edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 1.0:
                ax_time.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_y() + bar.get_height() / 2,
                             f'{v:.1f}s', ha='center', va='center',
                             fontsize=8, color='white', fontweight='bold')
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, k in enumerate(ks):
        ax_time.text(i, bottoms[i] + 0.3, f"{timings[k]['total']:.1f}s",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_time.set_ylabel('Time (s)')
    ax_time.set_title('Quantization time breakdown\n(4096×4096 FP16 weight, H=I)')
    ax_time.set_ylim(0, max(bottoms) * 1.2)
    ax_time.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                   fontsize=8, framealpha=0.8)
    ax_time.grid(axis='y', alpha=0.3)

    # ── right: compression ratio ──────────────────────────────────────────────
    ratios = [quant[k]['ratio'] for k in ks]
    bars = ax_ratio.bar(labels, ratios,
                        color=[K_COLORS[k] for k in ks], edgecolor='white')
    ax_ratio.bar_label(bars, fmt='%.2f×', padding=3, fontsize=10)
    ax_ratio.axhline(1.0, color=FP16_COLOR, linestyle='--',
                     linewidth=1, label='FP16 (1×)')
    ax_ratio.set_ylabel('Compression ratio  (FP16 bytes / quant bytes)')
    ax_ratio.set_title('Memory compression ratio\n(4096×4096 weight)')
    ax_ratio.set_ylim(0, max(ratios) * 1.3)
    ax_ratio.legend(fontsize=9)
    ax_ratio.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    savefig(fig, out_path)


# ── Per-path plots ────────────────────────────────────────────────────────────

def _has_breakdown(quant):
    """True if fused breakdown columns are present for at least one K."""
    return any(quant[k].get('fused_decode_ms') is not None for k in quant)


def plot_latency_bar(fp16, quant, out_path, title_suffix):
    ks = sorted(quant.keys())

    has_bd = _has_breakdown(quant)
    fig, (ax, ax_mse) = plt.subplots(1, 2, figsize=(11, 4))

    if has_bd:
        # ── Stacked breakdown bars for fused path ─────────────────────────────
        k_labels_list = k_labels(ks)
        # FP16 baseline as a plain bar at the left
        ax.bar(['FP16'], [fp16['ms']], color=FP16_COLOR, edgecolor='white',
               label='FP16 baseline')
        ax.text(0, fp16['ms'], f"{fp16['ms']:.3f}", ha='center', va='bottom',
                fontsize=8)

        # stacked segments for each K
        bottoms = [0.0] * len(ks)
        for step in FUSED_STEPS:
            vals = [quant[k][step] if quant[k][step] is not None else 0.0
                    for k in ks]
            bars = ax.bar(k_labels_list, vals, bottom=bottoms,
                          color=FUSED_STEP_COLORS[step],
                          label=FUSED_STEP_LABELS[step],
                          edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0.02:   # only annotate segments large enough to read
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f'{v:.3f}', ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        # total label above each stacked bar
        for i, k in enumerate(ks):
            ax.text(i + 1, bottoms[i], f"{quant[k]['ms']:.3f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=8, framealpha=0.8)
        ax.set_ylim(0, max(max(bottoms), fp16['ms']) * 1.25)

    else:
        # ── Plain bars for dequant path ───────────────────────────────────────
        labels = ['FP16'] + k_labels(ks)
        values = [fp16['ms']] + [quant[k]['ms'] for k in ks]
        colors = [FP16_COLOR] + [K_COLORS[k] for k in ks]
        bars = ax.bar(labels, values, color=colors, edgecolor='white')
        ax.bar_label(bars, fmt='%.3f ms', padding=3, fontsize=9)
        ax.set_ylim(0, max(values) * 1.3)

    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Inference latency — {title_suffix}\n(4096×4096 matvec, batch=1)')
    ax.grid(axis='y', alpha=0.3)

    # ── Out MSE subplot ───────────────────────────────────────────────────────
    mse_ks     = [k for k in ks if quant[k]['out_mse'] is not None]
    mse_vals   = [quant[k]['out_mse'] for k in mse_ks]

    if mse_vals:
        mbars = ax_mse.bar(k_labels(mse_ks), mse_vals,
                           color=[K_COLORS[k] for k in mse_ks], edgecolor='white')
        ax_mse.bar_label(mbars, fmt='%.4f', padding=3, fontsize=8)
        ax_mse.set_ylabel('Output MSE  [mean((Wx − hatW·x)²)]')
        ax_mse.set_title(f'Output error — {title_suffix}\n(fp32 x, N=4096)')
        ax_mse.set_ylim(0, max(mse_vals) * 1.35)
        ax_mse.grid(axis='y', alpha=0.3)
    else:
        ax_mse.set_visible(False)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_bandwidth_bar(fp16, quant, out_path, title_suffix):
    ks     = sorted(quant.keys())
    labels = ['FP16'] + k_labels(ks)
    values = [fp16['gb_s']] + [quant[k]['gb_s'] for k in ks]
    colors = [FP16_COLOR] + [K_COLORS[k] for k in ks]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor='white')
    ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=9)
    ax.set_ylabel('Effective bandwidth (GB/s)')
    ax.set_title(f'Effective memory bandwidth — {title_suffix}\n(4096×4096 matvec, batch=1)')
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig(fig, out_path)


# ── Comparison plots ──────────────────────────────────────────────────────────

def plot_quality_vs_k(dq_quant, out_path):
    ks      = sorted(dq_quant.keys())
    w_mse   = [dq_quant[k]['w_mse']   for k in ks]
    out_mse = [dq_quant[k]['out_mse'] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    bars1 = ax1.bar(k_labels(ks), w_mse,
                    color=[K_COLORS[k] for k in ks], edgecolor='white')
    ax1.bar_label(bars1, fmt='%.4f', padding=3, fontsize=8)
    ax1.set_ylabel('W MSE  [mean((W − hatW)²)]')
    ax1.set_title('Weight reconstruction error')
    ax1.set_ylim(0, max(w_mse) * 1.35)
    ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(k_labels(ks), out_mse,
                    color=[K_COLORS[k] for k in ks], edgecolor='white')
    ax2.bar_label(bars2, fmt='%.1f', padding=3, fontsize=8)
    ax2.set_ylabel('Out MSE  [mean((Wx − hatW·x)²)]')
    ax2.set_title('Output error (fp32 x, N=4096)')
    ax2.set_ylim(0, max(out_mse) * 1.35)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('QTIP quantization quality vs K bits', fontsize=12)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_latency_comparison(fp16_dq, dq_quant, fp16_fu, fu_quant, out_path):
    ks        = sorted(dq_quant.keys())
    x         = np.arange(len(ks))
    w         = 0.25
    has_bd    = _has_breakdown(fu_quant)

    fig, ax = plt.subplots(figsize=(9, 4))

    fp16_vals = [fp16_dq['ms']] * len(ks)
    dq_vals   = [dq_quant[k]['ms'] for k in ks]
    fu_vals   = [fu_quant[k]['ms'] for k in ks]

    b0 = ax.bar(x - w, fp16_vals, w, label='FP16 baseline',  color=FP16_COLOR)
    b1 = ax.bar(x,     dq_vals,   w, label='Dequant + gemm', color=DEQUANT_COLOR)

    for bars in (b0, b1):
        ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=7.5, rotation=45)

    if has_bd:
        # Stacked fused bars
        bottoms = np.zeros(len(ks))
        for step in FUSED_STEPS:
            vals = np.array([fu_quant[k][step] if fu_quant[k][step] is not None else 0.0
                             for k in ks])
            ax.bar(x + w, vals, w, bottom=bottoms,
                   color=FUSED_STEP_COLORS[step],
                   label=FUSED_STEP_LABELS[step],
                   edgecolor='white', linewidth=0.5)
            bottoms += vals
        # total labels above fused bars
        for i, k in enumerate(ks):
            ax.text(x[i] + w, fu_vals[i], f'{fu_vals[i]:.3f}',
                    ha='center', va='bottom', fontsize=7.5, rotation=45)
    else:
        b2 = ax.bar(x + w, fu_vals, w, label='Fused matvec', color=FUSED_COLOR)
        ax.bar_label(b2, fmt='%.3f', padding=2, fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(k_labels(ks))
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference latency comparison\n(4096×4096 matvec, batch=1)')
    ax.set_ylim(0, max(dq_vals + fu_vals + fp16_vals) * 1.55)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, framealpha=0.8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_bandwidth_comparison(fp16_dq, dq_quant, fp16_fu, fu_quant, out_path):
    ks = sorted(dq_quant.keys())
    x  = np.arange(len(ks))
    w  = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    fp16_vals = [fp16_dq['gb_s']] * len(ks)
    dq_vals   = [dq_quant[k]['gb_s'] for k in ks]
    fu_vals   = [fu_quant[k]['gb_s'] for k in ks]

    b0 = ax.bar(x - w, fp16_vals, w, label='FP16 baseline', color=FP16_COLOR)
    b1 = ax.bar(x,     dq_vals,   w, label='Dequant + gemm', color=DEQUANT_COLOR)
    b2 = ax.bar(x + w, fu_vals,   w, label='Fused matvec',   color=FUSED_COLOR)

    for bars in (b0, b1, b2):
        ax.bar_label(bars, fmt='%.0f', padding=2, fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(k_labels(ks))
    ax.set_ylabel('Effective bandwidth (GB/s)')
    ax.set_title('Effective memory bandwidth comparison\n(4096×4096 matvec, batch=1)')
    ax.set_ylim(0, max(fp16_vals) * 1.4)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_for_gpu(gpu, quant_timings):
    """Generate all plots for a single GPU slug."""
    gpu_dir     = os.path.join(RESULTS_DIR, gpu)
    dequant_dir = os.path.join(gpu_dir, 'dequant')
    fused_dir   = os.path.join(gpu_dir, 'fused')

    fp16_dq, dq_quant = load_csv(os.path.join(dequant_dir, 'results.csv'))
    fp16_fu, fu_quant = load_csv(os.path.join(fused_dir,   'results.csv'))

    have_dequant = fp16_dq is not None
    have_fused   = fp16_fu is not None
    have_both    = have_dequant and have_fused

    print(f"\n[{gpu}]")

    # ── Quantization timing breakdown ─────────────────────────────────────────
    if quant_timings:
        ref_quant_for_ratio = dq_quant if have_dequant else fu_quant
        plot_quant_breakdown(quant_timings, ref_quant_for_ratio,
                             os.path.join(gpu_dir, 'quant_time.png'))
    else:
        print("  [skip] quant_time.png — data/quant_timings.csv not found")

    # ── Per-path: dequant ─────────────────────────────────────────────────────
    if have_dequant:
        plot_latency_bar(fp16_dq, dq_quant,
                         os.path.join(dequant_dir, 'latency_bar.png'),
                         f'dequant + gemm  [{gpu}]')
        plot_bandwidth_bar(fp16_dq, dq_quant,
                           os.path.join(dequant_dir, 'bandwidth_bar.png'),
                           f'dequant + gemm  [{gpu}]')

    # ── Per-path: fused ───────────────────────────────────────────────────────
    if have_fused:
        plot_latency_bar(fp16_fu, fu_quant,
                         os.path.join(fused_dir, 'latency_bar.png'),
                         f'fused matvec  [{gpu}]')
        plot_bandwidth_bar(fp16_fu, fu_quant,
                           os.path.join(fused_dir, 'bandwidth_bar.png'),
                           f'fused matvec  [{gpu}]')

    # ── Comparison ────────────────────────────────────────────────────────────
    ref_quant = dq_quant if have_dequant else fu_quant
    plot_quality_vs_k(ref_quant,
                      os.path.join(gpu_dir, 'quality_vs_k.png'))

    if have_both:
        plot_latency_comparison(fp16_dq, dq_quant, fp16_fu, fu_quant,
                                os.path.join(gpu_dir, 'latency_comparison.png'))
        plot_bandwidth_comparison(fp16_dq, dq_quant, fp16_fu, fu_quant,
                                  os.path.join(gpu_dir, 'bandwidth_comparison.png'))
    else:
        print("  [skip] comparison plots require both dequant and fused CSVs")


def main():
    gpus = discover_gpu_dirs()
    if not gpus:
        sys.exit("No results found. Run benchmark_dequant.py and/or benchmark_fused.py first.")

    quant_timings = load_quant_timings(QUANT_TIMING_CSV)

    print("Generating plots...")
    for gpu in gpus:
        plot_for_gpu(gpu, quant_timings)

    print("\nDone.")


if __name__ == '__main__':
    main()
