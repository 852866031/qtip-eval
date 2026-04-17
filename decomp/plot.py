"""Plot the matvec decomposition across 3 methods x 2 timing modes.

Reads the six output/*.json files written by the six run_*.py scripts and
produces:

  - plot/matvec_time_<m>x<k>.png    per-shape grouped bar chart (log-y)
      with eager-vs-graph pairs for each of {FP16, dequant+matmul, QTIP}
  - plot/graph_speedup.png          ratio of eager/graph per method,
      showing how much each kernel gains from CUDA-graph capture
  - plot/matvec_bandwidth.png       effective BW per method (log-y)

No CLI arguments. Run after the six run_*.py scripts have produced JSONs:

    python decomp/plot.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

DECOMP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DECOMP_DIR, 'output')
PLOT_DIR = os.path.join(DECOMP_DIR, 'plot')

# (label, pretty name, color, hatch for graph variant)
LABELS = [
    ('fp16',    'FP16 cuBLAS',        '#808080'),
    ('dequant', 'Dequant + matmul',   '#d62728'),
    ('qtip',    'QTIP fused',         '#1f77b4'),
]


def load_one(label):
    path = os.path.join(OUTPUT_DIR, f'{label}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all():
    """Return nested dict: out[label][(m,k,bits)] = row."""
    out = {}
    for base, _, _ in LABELS:
        for mode in ('eager', 'graph'):
            key = f'{base}_{mode}'
            run = load_one(key)
            if run is None:
                continue
            d = {}
            for r in run['rows']:
                m, k = r['shape_mk']
                d[(m, k, r.get('bits'))] = r
            out[key] = d
    return out


def plot_time_per_shape(data, shapes, bits_list):
    for (m, k) in shapes:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        # one "group" per bit width (fp16 has no bits so we repeat it per group)
        groups = len(bits_list)
        bar_w = 0.12
        x = np.arange(groups)

        # method order: fp16, dequant, qtip; each has eager + graph
        colors = {'fp16': '#808080', 'dequant': '#d62728', 'qtip': '#1f77b4'}
        offsets = {
            'fp16_eager':    -2.5 * bar_w,
            'fp16_graph':    -1.5 * bar_w,
            'dequant_eager': -0.5 * bar_w,
            'dequant_graph':  0.5 * bar_w,
            'qtip_eager':     1.5 * bar_w,
            'qtip_graph':     2.5 * bar_w,
        }
        patterns = {'eager': '', 'graph': '//'}
        labels_shown = {}
        for key, off in offsets.items():
            base, mode = key.split('_')
            if key not in data:
                continue
            ys = []
            for b in bits_list:
                row = data[key].get((m, k, b)) if base != 'fp16' else data[key].get((m, k, None))
                ys.append(row['time_us'] if row else np.nan)
            legend_label = f'{base} {mode}'
            ax.bar(x + off, ys, bar_w,
                   color=colors[base], alpha=0.55 if mode == 'eager' else 1.0,
                   hatch=patterns[mode], edgecolor='black', linewidth=0.4,
                   label=legend_label)

        ax.set_yscale('log')
        ax.set_xticks(x, [f'{b}-bit' for b in bits_list])
        ax.set_ylabel('time per matvec (µs, log scale)')
        ax.set_title(f'Matvec latency, shape=({m}, {k})   (eager = faded, graph = hatched)')
        ax.grid(axis='y', which='both', alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        out = os.path.join(PLOT_DIR, f'matvec_time_{m}x{k}.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'[plot] wrote {out}')


def plot_graph_speedup(data, shapes, bits_list):
    """For each method, how much does graph capture speed it up?"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    method_colors = {'fp16': '#808080', 'dequant': '#d62728', 'qtip': '#1f77b4'}
    shape_hatches = ['', '//', '++']  # one per shape

    # Group by (method, bits); within a group, one bar per shape.
    groups_x = []  # (method, bits, label)
    for base in ('fp16', 'dequant', 'qtip'):
        if base == 'fp16':
            groups_x.append((base, None, 'FP16'))
        else:
            for b in bits_list:
                groups_x.append((base, b, f'{base} {b}b'))

    n_groups = len(groups_x)
    bar_w = 0.22
    xs = np.arange(n_groups)

    for si, (m, k) in enumerate(shapes):
        sp = []
        for (base, b, _) in groups_x:
            key_bits = None if base == 'fp16' else b
            e = data.get(f'{base}_eager', {}).get((m, k, key_bits))
            g = data.get(f'{base}_graph', {}).get((m, k, key_bits))
            if e and g and g['time_us'] > 0:
                sp.append(e['time_us'] / g['time_us'])
            else:
                sp.append(np.nan)
        colors = [method_colors[base] for (base, _, _) in groups_x]
        ax.bar(xs + (si - 1) * bar_w, sp, bar_w,
               color=colors, alpha=0.9,
               hatch=shape_hatches[si], edgecolor='black', linewidth=0.4,
               label=f'({m}, {k})')
        # annotate
        for xi, val in zip(xs + (si - 1) * bar_w, sp):
            if np.isfinite(val):
                ax.text(xi, val, f'{val:.1f}x', ha='center', va='bottom', fontsize=7)

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':')
    ax.set_yscale('log')
    ax.set_xticks(xs, [label for _, _, label in groups_x], fontsize=8)
    ax.set_ylabel('speedup from CUDA graph capture (eager / graph, log scale)')
    ax.set_title('How much does each kernel gain from CUDA-graph capture?')
    ax.grid(axis='y', which='both', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, title='shape (m, k)')
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, 'graph_speedup.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'[plot] wrote {out}')


def plot_bandwidth(data, shapes, bits_list):
    """Effective BW per method (GB/s, log scale).

    FP16's BW is against the fp16 weight bytes; the two quantized paths'
    BW is against the compressed weight bytes. Methods are not strictly
    comparable across rows, but within a method the bar heights show how
    close to the HBM roof the implementation gets.
    """
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.3 * len(shapes), 4.0),
                             sharey=True)
    if len(shapes) == 1:
        axes = [axes]
    method_colors = {'fp16': '#808080', 'dequant': '#d62728', 'qtip': '#1f77b4'}
    bar_w = 0.12

    for ax, (m, k) in zip(axes, shapes):
        xs = np.arange(len(bits_list))
        offsets = {
            'fp16_eager':    -2.5 * bar_w,
            'fp16_graph':    -1.5 * bar_w,
            'dequant_eager': -0.5 * bar_w,
            'dequant_graph':  0.5 * bar_w,
            'qtip_eager':     1.5 * bar_w,
            'qtip_graph':     2.5 * bar_w,
        }
        for key, off in offsets.items():
            base, mode = key.split('_')
            if key not in data:
                continue
            ys = []
            for b in bits_list:
                row = data[key].get((m, k, b)) if base != 'fp16' else data[key].get((m, k, None))
                ys.append(row['effective_gbps'] if row else np.nan)
            ax.bar(xs + off, ys, bar_w, color=method_colors[base],
                   alpha=0.55 if mode == 'eager' else 1.0,
                   hatch='' if mode == 'eager' else '//',
                   edgecolor='black', linewidth=0.4,
                   label=f'{base} {mode}')
        ax.set_yscale('log')
        ax.set_xticks(xs, [f'{b}-bit' for b in bits_list])
        ax.set_title(f'({m}, {k})')
        ax.grid(axis='y', which='both', alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('effective bandwidth (GB/s, log scale)')
    axes[-1].legend(fontsize=7, loc='lower right', ncol=1)
    fig.suptitle('Effective bandwidth per matvec (FP16 vs compressed-bytes basis)')
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, 'matvec_bandwidth.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'[plot] wrote {out}')


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = load_all()
    if not data:
        print(f'[plot] no JSONs in {OUTPUT_DIR} -- run the six run_*.py scripts first')
        return

    for key, d in data.items():
        ms_list = [r['time_us'] for r in d.values()]
        print(f'[plot] loaded {key}: n_rows={len(d)}  '
              f'min={min(ms_list):.1f} us  max={max(ms_list):.1f} us')

    # collect shapes and bit widths from whatever's loaded
    shapes = sorted({(m, k) for d in data.values() for (m, k, _) in d.keys()})
    bits_list = sorted({b for d in data.values() for (_, _, b) in d.keys() if b is not None})

    plot_time_per_shape(data, shapes, bits_list)
    plot_graph_speedup(data, shapes, bits_list)
    plot_bandwidth(data, shapes, bits_list)


if __name__ == '__main__':
    main()
