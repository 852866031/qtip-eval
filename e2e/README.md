# `e2e/` — end-to-end bs=1 decoding throughput

Reproduces the FP16 and QTIP-4bit rows of Table 4 from the QTIP paper on
Llama-2-7B, **and** cross-cuts the question of "what does CUDA-graph
capture actually do for each path" that the decomp experiments raised.
Four self-contained pipelines, each argument-free with all knobs in
module-level constants at the top of the file.

## Methods × modes

| | **eager** (no `torch.compile`, no CUDA graph) | **graph** (`torch.compile(max-autotune, fullgraph=True)`) |
|---|---|---|
| **FP16 Llama-2-7B**      | `run_fp16_eager.py`      | `run_fp16_graph.py`      |
| **QTIP-4bit Llama-2-7B** | `run_qtip4bit_eager.py`  | `run_qtip4bit_graph.py`  |

`torch.compile(mode='max-autotune', fullgraph=True)` is Inductor's
reduce-overhead path: kernel autotuning + CUDA graph capture of the
compiled decode step. This is the mode the QTIP paper's
`interactive_gen.py` uses, and what every production inference stack
(vLLM, SGLang, TensorRT-LLM, …) uses for bs=1 decode.

The `eager` variants deliberately skip it — every generated token is one
Python-driven forward pass per layer per call. That is what a naive
`model.generate(...)` loop looks like without framework-level graph
capture.

Why cross the two axes?
- **Methods** answer: "is QTIP-4bit end-to-end faster than FP16 for real
  token generation?"
- **Modes** answer: "does the answer to the methods question depend on
  whether you capture CUDA graphs?" (Spoiler: yes, significantly, in the
  same direction the matvec-level `decomp/` experiments show.)

## Layout

```
e2e/
├── README.md                this file
├── run_fp16_eager.py        FP16 Llama-2-7B, eager    -> output/fp16_eager.json
├── run_fp16_graph.py        FP16 Llama-2-7B, graph    -> output/fp16_graph.json
├── run_qtip4bit_eager.py    QTIP-4bit, eager          -> output/qtip4bit_eager.json
├── run_qtip4bit_graph.py    QTIP-4bit, graph          -> output/qtip4bit_graph.json
├── plot.py                  reads output/*.json, writes plot/*.png
├── output/                  per-run JSON (created on first run)
└── plot/                    PNG plots (created by plot.py)
```

The four `run_*.py` scripts are fully independent — each is a
self-contained pipeline (model load, warmup, optional compile, timed
trials, JSON dump). They duplicate their small plumbing intentionally so
any one can be read, copied, or diff'd alone.

## How to run

Requires the env set up per the project-root README (conda env `qtip-eval`,
PyTorch cu128, pinned transformers 4.45.2, built `qtip_kernels`, HF login
for the gated Llama-2 tokenizer).

```bash
conda activate qtip-eval
cd /home/jiaxuan/Documents/Projects/qtip-eval-1

CUDA_VISIBLE_DEVICES=0 python e2e/run_fp16_eager.py
CUDA_VISIBLE_DEVICES=0 python e2e/run_fp16_graph.py
CUDA_VISIBLE_DEVICES=0 python e2e/run_qtip4bit_eager.py
CUDA_VISIBLE_DEVICES=0 python e2e/run_qtip4bit_graph.py

python e2e/plot.py
```

Each run loads the model once (first run per model downloads ~3.5 GB for
QTIP-4bit, ~13 GB for FP16, both to `$HF_HOME`), does an 8-token eager
warmup, optionally compiles for the graph variants (16-token compile
trigger), then runs `N_TRIALS` generations of `MAX_NEW_TOKENS` each.

## Configuration (edit in code, not via CLI)

Top of each script:

| constant          | meaning                                 | default |
|-------------------|-----------------------------------------|---------|
| `HF_PATH`         | HF Hub path for the model               | fixed   |
| `LABEL`           | output filename stem + plot key         | fixed   |
| `MAX_NEW_TOKENS`  | tokens generated per trial              | 256     |
| `N_TRIALS`        | number of timed generations             | 3       |
| `PROMPT`          | seed text                               | fixed   |

## Output format

`output/<label>.json`:

```json
{
  "label": "qtip4bit_graph",
  "mode": "torch.compile(max-autotune, fullgraph=True) -- CUDA graph capture",
  "hf_path": "relaxml/Llama-2-7b-QTIP-4Bit",
  "max_new_tokens": 256,
  "n_trials": 3,
  "trials_tok_s": [197.8, 199.4, 199.4],
  "best_tok_s": 199.4,
  "mean_tok_s": 198.9,
  "load_seconds": 76.2,
  "quantized": true,
  "gpu": "NVIDIA GeForce RTX 5090"
}
```

## Plots

`plot.py` produces:

- **`plot/throughput_bar.png`** — mean tok/s, one bar per (method, mode)
  cell. Eager bars are faded, graph bars are hatched. Per-trial dots and
  an error bar are overlaid. Title lines show the graph/eager ratio for
  each method.
- **`plot/throughput_trials.png`** — per-trial line plot to eyeball
  timing noise across trials.
- **`plot/graph_speedup.png`** — graph/eager ratio per method. Echoes the
  decomp `graph_speedup.png`: how much does each end-to-end path actually
  gain from CUDA-graph capture?

## What to expect (RTX 5090)

| label              | mean tok/s (approx) |
|--------------------|--------------------:|
| `fp16_eager`       | ~45–55              |
| `fp16_graph`       | ~115–120            |
| `qtip4bit_eager`   | ~10–25              |
| `qtip4bit_graph`   | ~195–210            |

So the sign of the "is QTIP-4bit faster than FP16?" question **flips**
between the two modes:
- In eager: QTIP-4bit is **slower** than FP16 (the matvec-level
  ~100×-graph-sensitivity observed in `decomp/` shows up here as a
  catastrophic penalty for the per-layer fused kernel being called with a
  sync per linear per token).
- In graph: QTIP-4bit is ~1.7× **faster** than FP16 (paper's
  claim reproduces).

That is the headline result for this directory: **the QTIP paper's
throughput claim is real but mode-conditional**, and the relevant mode
(CUDA graphs via `torch.compile`) is what every production inference
stack already uses.
