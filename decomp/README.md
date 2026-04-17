# `decomp/` — matvec-level decomposition (where does QTIP's speedup come from?)

End-to-end tok/s (Table 4) tells you QTIP is fast, but it does not separate
*low-bit weights* from *fused-kernel engineering*, **and** it does not
surface the fact that the fused kernel's measured speed is extremely
sensitive to whether the caller uses CUDA graphs. This directory pins that
down by cross-cutting three computational paths with two timing methods.

## Methods × timing modes

| | **eager timer** (`torch.utils.benchmark.Timer`, sync per iter) | **CUDA-graph timer** (capture once + `g.replay()` × N, explicit host sync before stopping) |
|---|---|---|
| **FP16 cuBLAS matvec**     | `run_fp16_eager.py`    | `run_fp16_graph.py`    |
| **Dequant + matmul (LB)**  | `run_dequant_eager.py` | `run_dequant_graph.py` |
| **QTIP fused matvec**      | `run_qtip_eager.py`    | `run_qtip_graph.py`    |

Each of the six scripts is fully self-contained (no shared helpers) and
takes no CLI arguments. Edit the constants at the top of the file to
change knobs.

Why cross the two axes?

- **Methods** answer: "is the fused kernel structurally faster than
  dequant+matmul or FP16?"
- **Timing modes** answer: "is the fused kernel *actually* faster the way
  a real inference stack uses it, or only on paper?"

On this hardware the answer to (2) is surprising: FP16 and dequant+matmul
gain ~1–2× from CUDA graphs, but the QTIP fused kernel gains ~50–100×.
Which timing mode you use completely changes the qualitative conclusion.

## Layout

```
decomp/
├── README.md                this file
├── run_fp16_eager.py        Path A, eager timer
├── run_fp16_graph.py        Path A, CUDA-graph timer
├── run_dequant_eager.py     Path B, eager timer
├── run_dequant_graph.py     Path B, CUDA-graph timer
├── run_qtip_eager.py        Path C, eager timer
├── run_qtip_graph.py        Path C, CUDA-graph timer
├── plot.py                  reads output/*.json, writes plot/*.png
├── output/                  per-run JSON (created on first run)
└── plot/                    PNG plots (created by plot.py)
```

Each script sweeps three Llama-2-7B weight shapes:

| shape | weights it covers |
|-------|-------------------|
| (4096, 4096)   | q / k / v / o proj |
| (4096, 11008)  | down proj          |
| (11008, 4096)  | gate / up proj     |

The quantized paths (B and C) also sweep bit-widths `{2, 3, 4}` (HYB code's
available kernels). FP16 has no bit-width axis.

## How to run

```bash
conda activate qtip-eval
cd /home/jiaxuan/Documents/Projects/qtip-eval-1

CUDA_VISIBLE_DEVICES=0 python decomp/run_fp16_eager.py
CUDA_VISIBLE_DEVICES=0 python decomp/run_fp16_graph.py
CUDA_VISIBLE_DEVICES=0 python decomp/run_dequant_eager.py
CUDA_VISIBLE_DEVICES=0 python decomp/run_dequant_graph.py
CUDA_VISIBLE_DEVICES=0 python decomp/run_qtip_eager.py
CUDA_VISIBLE_DEVICES=0 python decomp/run_qtip_graph.py

python decomp/plot.py
```

Each script runs in ~10–30 seconds. No model download, no compile.

## Timing method details

**Eager timer** (all `*_eager.py`) uses `torch.utils.benchmark.Timer` with
`stmt='fn(); torch.cuda.synchronize()'`, `REPEATS=200`, `WARMUP=10`. Each
iteration pays full Python dispatch + torch.ops overhead + kernel launch
+ `cudaDeviceSynchronize`. This matches how a naive Python inference loop
or a microbenchmark without a graph-aware runtime would drive the op.

**Graph timer** (all `*_graph.py`) captures one call of the closure into a
`torch.cuda.CUDAGraph`, then replays it `ITER=200` times inside a single
CUDA-event range. An explicit `torch.cuda.synchronize()` is issued BEFORE
recording the end event so the measurement reflects completed GPU work,
not just host-side queueing time:

```python
start.record()
for _ in range(ITER):
    g.replay()
torch.cuda.synchronize()   # all replays must complete before we stop
end.record()
torch.cuda.synchronize()
return start.elapsed_time(end) / ITER
```

This matches how `torch.compile(mode='max-autotune')`, vLLM, SGLang, and
TensorRT-LLM all drive their decode step.

## Output format

`output/<label>.json`:

```json
{
  "label": "qtip_graph",
  "method": "QTIP fused decompress+matvec",
  "timer": "CUDA graph replay + explicit host sync before end event",
  "iter": 500,  "warmup": 50,
  "gpu": "NVIDIA GeForce RTX 5090",
  "rows": [
    {"shape_mk": [4096, 4096], "bits": 2, "time_us": 8.2,
     "bytes_compressed": 4194304, "effective_gbps": 511.4},
    ...
  ]
}
```

Eager JSONs have `"repeats"` and `"warmup"` instead of `"iter"` and a
different `"timer"` description. The row schema is otherwise identical,
which lets `plot.py` cross-compare without branching.

## Plots

`plot.py` produces:

- **`plot/matvec_time_<m>x<k>.png`** — per-shape grouped bars, log-y.
  Three methods × two timing modes = six bars per bit-width group. Eager
  bars are faded, graph bars are hatched.
- **`plot/graph_speedup.png`** — eager / graph ratio for each method and
  bit-width. Makes it visually obvious that `qtip` gets ~100× from graphs
  while `fp16` and `dequant` get ~1.5–2×.
- **`plot/matvec_bandwidth.png`** — effective BW (GB/s, log-y) per method,
  per shape, across bit-widths. BW for FP16 is against fp16 weight bytes;
  BW for quantized paths is against compressed weight bytes, so bars in
  different methods aren't directly comparable, but within a method the
  height tracks "how close to the HBM roof is this implementation".

## Expected pattern (RTX 5090)

Numbers below are approximate (4096 × 4096 matvec, µs per call):

| | eager | graph | graph speedup |
|---|---:|---:|---:|
| FP16 cuBLAS           |  ~22  |  ~17  |  ~1.3× |
| Dequant + matmul (LB) |  ~35  |  ~31  |  ~1.1× |
| QTIP fused (2-bit)    | ~950  |  ~8.2 |  ~115× |
| QTIP fused (4-bit)    | ~840  |  ~8.2 |  ~103× |

Reading the rows:

1. **At the matvec level, QTIP fused is *slower* than FP16 in eager mode.**
   The kernel is tuned for the case where the scheduler can amortize
   per-launch ramp-up across many back-to-back calls. Call it once, sync,
   call it once, sync — most of the wall clock is ramp-up, not compute.
2. **Under CUDA graphs, QTIP fused is faster than FP16.** In the graph
   regime the kernel approaches its memory-bandwidth roof and the
   compressed-weight-only HBM traffic argument finally pays off.
3. **Dequant+matmul is slower than FP16 in both modes.** Any "stage the
   fp16 weight through HBM" path pays 2× the memory traffic of FP16.
4. **The relevant mode for production** is the graph one, because every
   serious inference stack captures graphs. That's also why the paper's
   Table 4 numbers reproduce in `e2e/` (which uses `torch.compile`) but
   would not reproduce if you ran `BitshiftLinear.forward` in an eager
   Python loop.
