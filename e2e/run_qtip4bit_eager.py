"""End-to-end bs=1 decoding throughput, QTIP 4-bit Llama-2-7B -- EAGER (no torch.compile).

Runs the QTIP-patched Llama's decode loop purely eagerly. Every layer's
`BitshiftLinear.forward` is called from Python, which invokes the fused
`qtip_kernels.decompress_matvec_*` kernel once per linear. No torch.compile,
no CUDA-graph capture. This is the setup in which the fused kernel's
cold-launch overhead (1 block / SM, 64 KB shmem codebook preamble) is NOT
amortized across back-to-back calls -- we expect it to be much slower
than the graph variant.

Self-contained: no shared helpers, no CLI arguments.

Run:
    CUDA_VISIBLE_DEVICES=0 python e2e/run_qtip4bit_eager.py

Writes: output/qtip4bit_eager.json
"""
# -------- configuration (edit here) -----------------------------------------
HF_PATH = 'relaxml/Llama-2-7b-QTIP-4Bit'
LABEL = 'qtip4bit_eager'
MAX_NEW_TOKENS = 256
N_TRIALS = 3
PROMPT = 'This is a test of this large language model'
# ----------------------------------------------------------------------------

import json
import os
import sys
import time
from typing import Optional

E2E_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(E2E_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'qtip'))

import torch
from transformers import AutoTokenizer

from model.cache_utils import StaticCache  # patched cache required by patched Llama
from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)


def multinomial_sample_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float('Inf'), logits)
    return torch.nn.functional.softmax(logits, dim=-1)


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    return multinomial_sample_no_sync(probs), probs


@torch.no_grad()
def decode_one(model, cur_token, past_kv, cache_position):
    logits = model(cur_token, past_key_values=past_kv,
                   cache_position=cache_position)[0]
    return sample(logits, temperature=0.6, top_k=32)[0]


@torch.no_grad()
def decode_one_greedy(model, cur_token, past_kv, cache_position):
    """Forward + deterministic argmax; returns (last_logits, next_token).

    Used for the logit-capture pass so eager and graph runs can be diffed
    without stochastic sampling masking any kernel-level differences.
    """
    logits = model(cur_token, past_key_values=past_kv,
                   cache_position=cache_position)[0]
    last = logits[:, -1, :]
    next_token = torch.argmax(last, dim=-1, keepdim=True).to(dtype=torch.int)
    return last, next_token


@torch.no_grad()
def run_greedy_capture(model, tokenizer, text, max_new_tokens, past_kv, decode_greedy_fn):
    """Deterministic greedy generation; returns (logits [T, vocab], tokens [T])."""
    inputs = tokenizer(text, return_tensors='pt').to(0)
    _, seq_len = inputs['input_ids'].shape
    cache_position = torch.arange(seq_len, device=0)
    past_kv.reset()

    # Prefill: model returns full logits tensor; take last-token row.
    prefill_logits = model(**inputs, past_key_values=past_kv,
                           cache_position=cache_position)[0][:, -1, :]
    next_token = torch.argmax(prefill_logits, dim=-1, keepdim=True).to(dtype=torch.int)

    logits_acc = [prefill_logits.detach().to('cpu', torch.float32)]
    token_acc = [next_token.detach().cpu().view(-1)]

    cache_position = torch.tensor([seq_len + 1], device=0)
    for _ in range(1, max_new_tokens):
        last, next_token = decode_greedy_fn(model, next_token.clone(),
                                            past_kv, cache_position)
        logits_acc.append(last.detach().to('cpu', torch.float32))
        token_acc.append(next_token.detach().cpu().view(-1))
        cache_position += 1

    return torch.cat(logits_acc, dim=0), torch.cat(token_acc, dim=0)


@torch.no_grad()
def run_generate(model, tokenizer, text, max_new_tokens, past_kv):
    inputs = tokenizer(text, return_tensors='pt').to(0)
    _, seq_len = inputs['input_ids'].shape
    cache_position = torch.arange(seq_len, device=0)
    past_kv.reset()
    logits = model(**inputs, past_key_values=past_kv,
                   cache_position=cache_position)[0]
    next_token, _ = sample(logits, top_k=32)

    cache_position = torch.tensor([seq_len + 1], device=0)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1, max_new_tokens):
        with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one(model, next_token.clone(),
                                    past_kv, cache_position)
        cache_position += 1
    torch.cuda.synchronize()
    return (max_new_tokens - 1) / (time.time() - t0)


def main():
    torch.set_float32_matmul_precision('high')

    print(f'[e2e:{LABEL}] loading {HF_PATH}')
    t0 = time.time()
    model, model_str = model_from_hf_path(HF_PATH)
    load_s = time.time() - t0
    assert hasattr(model.config, 'quip_params'), \
        'expected a QTIP-quantized checkpoint for the 4-bit run'
    print(f'[e2e:{LABEL}] loaded in {load_s:.1f}s')

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token

    past_kv = StaticCache(model.config, 1, 2 * MAX_NEW_TOKENS,
                          device=0, dtype=model.dtype)

    # Eager warmup; NO torch.compile anywhere.
    run_generate(model, tokenizer, PROMPT, 8, past_kv)

    tps = []
    for i in range(N_TRIALS):
        t = run_generate(model, tokenizer, PROMPT, MAX_NEW_TOKENS, past_kv)
        print(f'[e2e:{LABEL}] trial {i}: {t:.2f} tok/s')
        tps.append(t)

    best = max(tps)
    mean = sum(tps) / len(tps)
    print(f'\n[e2e:{LABEL}] RESULT  best={best:.2f}  mean={mean:.2f} tok/s  '
          f'(n={len(tps)})')

    # Logit-capture run (eager greedy decode, fixed prompt, deterministic).
    print(f'[e2e:{LABEL}] capturing logits under greedy decoding ...')
    logits, tokens = run_greedy_capture(model, tokenizer, PROMPT,
                                        MAX_NEW_TOKENS, past_kv,
                                        decode_one_greedy)
    print(f'[e2e:{LABEL}] captured logits shape={tuple(logits.shape)}  '
          f'tokens shape={tuple(tokens.shape)}')
    print(f'[e2e:{LABEL}] sample decoded: {tokenizer.decode(tokens[:40].tolist())!r}')

    out_dir = os.path.join(E2E_DIR, 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{LABEL}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'label': LABEL,
            'mode': 'eager (no torch.compile, no CUDA graph)',
            'hf_path': HF_PATH,
            'max_new_tokens': MAX_NEW_TOKENS,
            'n_trials': N_TRIALS,
            'prompt': PROMPT,
            'trials_tok_s': tps,
            'best_tok_s': best,
            'mean_tok_s': mean,
            'load_seconds': load_s,
            'quantized': True,
            'gpu': torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f'[e2e:{LABEL}] wrote {out_path}')

    logits_path = os.path.join(out_dir, f'{LABEL}_logits.pt')
    torch.save({
        'label': LABEL,
        'mode': 'eager (no torch.compile, no CUDA graph); greedy sampling',
        'hf_path': HF_PATH,
        'prompt': PROMPT,
        'max_new_tokens': MAX_NEW_TOKENS,
        'logits': logits,       # [T, vocab] fp32 on CPU
        'tokens': tokens,       # [T] int on CPU
        'gpu': torch.cuda.get_device_name(0),
    }, logits_path)
    print(f'[e2e:{LABEL}] wrote {logits_path}')


if __name__ == '__main__':
    main()
