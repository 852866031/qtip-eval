"""Plot end-to-end decoding throughput across the four runs.

Reads output/{fp16,qtip4bit}_{eager,graph}.json and writes:

  - plot/throughput_bar.png      mean tok/s with per-trial dots, one bar
                                  per (method, mode) cell.
  - plot/throughput_trials.png   per-trial line plot for timing-noise
                                  sanity check.
  - plot/graph_speedup.png       mode ratio (graph / eager) for FP16 and
                                  QTIP, echoing the decomp story: how
                                  much does each path gain from
                                  torch.compile / CUDA-graph capture?

No CLI arguments. Run after the four run_*.py scripts have produced JSONs:

    python e2e/plot.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

E2E_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(E2E_DIR, 'output')
PLOT_DIR = os.path.join(E2E_DIR, 'plot')

DISPLAY = {
    'fp16_eager':     ('FP16 eager',       '#808080', ''  ),
    'fp16_graph':     ('FP16 graph',       '#808080', '//'),
    'qtip4bit_eager': ('QTIP-4bit eager',  '#1f77b4', ''  ),
    'qtip4bit_graph': ('QTIP-4bit graph',  '#1f77b4', '//'),
}


def load_runs():
    runs = {}
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.json'))):
        with open(p) as f:
            payload = json.load(f)
        runs[payload['label']] = payload
    return runs


def plot_bar(runs, out_path):
    keys = [k for k in DISPLAY if k in runs]
    labels = [DISPLAY[k][0] for k in keys]
    colors = [DISPLAY[k][1] for k in keys]
    hatches = [DISPLAY[k][2] for k in keys]
    trials = [np.asarray(runs[k]['trials_tok_s']) for k in keys]
    means = [t.mean() for t in trials]
    stds = [t.std(ddof=0) for t in trials]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    xs = np.arange(len(keys))
    alphas = [0.55 if '_eager' in k else 1.0 for k in keys]
    for x, m, s, c, h, a in zip(xs, means, stds, colors, hatches, alphas):
        ax.bar(x, m, color=c, alpha=a, hatch=h, edgecolor='black',
               linewidth=0.4, yerr=s, capsize=5, zorder=2)
    for x, t in zip(xs, trials):
        ax.scatter([x]*len(t), t, color='black', s=14, zorder=3)
    for x, m in zip(xs, means):
        ax.text(x, m, f' {m:.1f}', va='bottom', ha='center', fontsize=9)

    title_bits = []
    if 'fp16_eager' in runs and 'fp16_graph' in runs:
        ratio = np.mean(runs['fp16_graph']['trials_tok_s']) / np.mean(runs['fp16_eager']['trials_tok_s'])
        title_bits.append(f'FP16 graph/eager = {ratio:.2f}x')
    if 'qtip4bit_eager' in runs and 'qtip4bit_graph' in runs:
        ratio = np.mean(runs['qtip4bit_graph']['trials_tok_s']) / np.mean(runs['qtip4bit_eager']['trials_tok_s'])
        title_bits.append(f'QTIP graph/eager = {ratio:.2f}x')

    title = 'Llama-2-7B bs=1 decode (RTX 5090)'
    if title_bits:
        title += '\n' + '   '.join(title_bits)
    ax.set_title(title)
    ax.set_ylabel('throughput (tokens / sec)')
    ax.set_xticks(xs, labels, fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[plot] wrote {out_path}')


def plot_trials(runs, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for key, (pretty, color, hatch) in DISPLAY.items():
        if key not in runs:
            continue
        t = runs[key]['trials_tok_s']
        ls = '--' if '_eager' in key else '-'
        ax.plot(range(1, len(t) + 1), t, 'o' + ls, color=color, label=pretty,
                alpha=0.55 if '_eager' in key else 1.0)
    ax.set_xlabel('trial #')
    ax.set_ylabel('throughput (tokens / sec)')
    ax.set_title('Per-trial throughput (timing-noise sanity check)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[plot] wrote {out_path}')


def plot_graph_speedup(runs, out_path):
    """Ratio of graph/eager mean tok/s per method."""
    pairs = [('FP16', 'fp16_eager', 'fp16_graph', '#808080'),
             ('QTIP-4bit', 'qtip4bit_eager', 'qtip4bit_graph', '#1f77b4')]
    labels, ratios, colors = [], [], []
    for name, ek, gk, c in pairs:
        if ek in runs and gk in runs:
            e = np.mean(runs[ek]['trials_tok_s'])
            g = np.mean(runs[gk]['trials_tok_s'])
            labels.append(name)
            ratios.append(g / e)
            colors.append(c)
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs = np.arange(len(labels))
    ax.bar(xs, ratios, color=colors, edgecolor='black', linewidth=0.4, zorder=2)
    for x, r in zip(xs, ratios):
        ax.text(x, r, f'{r:.2f}x', ha='center', va='bottom', fontsize=10)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xticks(xs, labels)
    ax.set_ylabel('tok/s speedup (graph / eager)')
    ax.set_title('How much does each path gain from CUDA-graph capture?')
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[plot] wrote {out_path}')


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        print(f'[plot] no JSONs in {OUTPUT_DIR} -- run the four run_*.py scripts first')
        return
    for key in DISPLAY:
        if key in runs:
            r = runs[key]
            print(f'[plot] loaded {key}: best={r["best_tok_s"]:.2f}  '
                  f'mean={r["mean_tok_s"]:.2f} tok/s  '
                  f'(trials={[round(t,2) for t in r["trials_tok_s"]]})')
    plot_bar(runs, os.path.join(PLOT_DIR, 'throughput_bar.png'))
    plot_trials(runs, os.path.join(PLOT_DIR, 'throughput_trials.png'))
    plot_graph_speedup(runs, os.path.join(PLOT_DIR, 'graph_speedup.png'))


if __name__ == '__main__':
    main()
