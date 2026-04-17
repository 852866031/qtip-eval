"""Compare QTIP-4bit logits captured under eager vs graph inference.

Loads the two .pt files produced by the qtip4bit run scripts' greedy
logit-capture passes and reports:

  - Whether the generated token sequences agree (and where they first diverge)
  - Per-step logit error (max |Δ|, mean |Δ|, RMS reference)
  - Top-K agreement between eager and graph predictions
  - Decoded text from each for an eyeball check

Usage:
    python e2e/run_qtip4bit_eager.py
    python e2e/run_qtip4bit_graph.py
    python e2e/compare_qtip4bit_logits.py
"""
import os

import torch
from transformers import AutoTokenizer

E2E_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(E2E_DIR, 'output')

EAGER = os.path.join(OUT, 'qtip4bit_eager_logits.pt')
GRAPH = os.path.join(OUT, 'qtip4bit_graph_logits.pt')


def _fmt_tok_list(tok_ids, limit=20):
    if len(tok_ids) <= limit:
        return ' '.join(str(int(t)) for t in tok_ids)
    return (' '.join(str(int(t)) for t in tok_ids[:limit//2])
            + ' … '
            + ' '.join(str(int(t)) for t in tok_ids[-limit//2:]))


def main():
    for p in (EAGER, GRAPH):
        if not os.path.exists(p):
            raise SystemExit(f'missing: {p}. Run run_qtip4bit_eager.py and run_qtip4bit_graph.py first.')

    e = torch.load(EAGER, weights_only=False, map_location='cpu')
    g = torch.load(GRAPH, weights_only=False, map_location='cpu')

    # basic consistency
    for key in ('hf_path', 'prompt', 'max_new_tokens'):
        if e[key] != g[key]:
            print(f'[warn] {key} differs: eager={e[key]!r}  graph={g[key]!r}')

    el = e['logits'].float()     # (T, vocab)
    gl = g['logits'].float()
    et = e['tokens'].view(-1)
    gt = g['tokens'].view(-1)

    T, V = el.shape
    assert gl.shape == (T, V), f'shape mismatch: eager {el.shape} vs graph {gl.shape}'
    assert et.shape == gt.shape == (T,)

    print(f'prompt          : {e["prompt"]!r}')
    print(f'hf_path         : {e["hf_path"]}')
    print(f'tokens generated: {T}')
    print(f'vocab size      : {V}')
    print(f'eager mode      : {e["mode"]}')
    print(f'graph mode      : {g["mode"]}')
    print()

    # --- token sequence agreement ---
    mismatch_mask = et.ne(gt)
    n_mismatch = int(mismatch_mask.sum())
    first_div = int(mismatch_mask.nonzero()[0, 0]) if n_mismatch else None
    print(f'== token sequence ==')
    print(f'  exact matches   : {T - n_mismatch}/{T}  '
          f'({100.0 * (T - n_mismatch) / T:.2f}%)')
    if first_div is None:
        print(f'  first divergence: <none -- sequences are identical>')
    else:
        print(f'  first divergence: step {first_div}  '
              f'eager={int(et[first_div])}  graph={int(gt[first_div])}')

    # --- logit numeric diff ---
    diff = (el - gl).abs()
    print()
    print(f'== full logits (T x V = {T} x {V}) ==')
    print(f'  max |Δ|         : {diff.max().item():.4e}')
    print(f'  mean |Δ|        : {diff.mean().item():.4e}')
    print(f'  p50 |Δ|         : {diff.flatten().median().item():.4e}')
    print(f'  p99 |Δ|         : {torch.quantile(diff.flatten(), 0.99).item():.4e}')
    rms_ref = el.pow(2).mean().sqrt().item()
    print(f'  RMS(eager)      : {rms_ref:.4f}   (scale reference)')
    print(f'  max|Δ| / RMS    : {diff.max().item() / rms_ref:.4e}')

    # --- per-step max-abs and top-1 agreement ---
    per_step_max = diff.max(dim=-1).values                   # (T,)
    top1_e = el.argmax(dim=-1)
    top1_g = gl.argmax(dim=-1)
    top1_agree = int((top1_e == top1_g).sum())
    print()
    print(f'== per-step divergence ==')
    print(f'  top-1 argmax agreement: {top1_agree}/{T}  '
          f'({100.0 * top1_agree / T:.2f}%)')

    # top-k overlap at a few checkpoint steps
    K = 5
    checkpoints = sorted(set([0, max(0, T//4), max(0, T//2), max(0, 3*T//4), T - 1]))
    for i in checkpoints:
        te = set(torch.topk(el[i], K).indices.tolist())
        tg = set(torch.topk(gl[i], K).indices.tolist())
        overlap = len(te & tg)
        print(f'  step {i:3d}: max|Δ|={per_step_max[i].item():.3e}   '
              f'top-{K} overlap={overlap}/{K}   '
              f'argmax(e)={int(top1_e[i])} argmax(g)={int(top1_g[i])}')

    # --- decoded text eyeball ---
    try:
        tok = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        tok.pad_token = tok.eos_token
        prefix = tok(e['prompt'], return_tensors='pt')['input_ids'][0].tolist()
        print()
        print(f'== decoded text (greedy, first 80 tokens after prompt) ==')
        print(f'  eager: {tok.decode(prefix + et[:80].tolist())!r}')
        print(f'  graph: {tok.decode(prefix + gt[:80].tolist())!r}')
    except Exception as ex:
        print(f'\n[warn] could not decode text: {ex}')

    # --- token ids, raw ---
    print()
    print(f'== raw token ids (first / last 10) ==')
    print(f'  eager: {_fmt_tok_list(et.tolist())}')
    print(f'  graph: {_fmt_tok_list(gt.tolist())}')


if __name__ == '__main__':
    main()
