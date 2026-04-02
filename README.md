# qtip-eval

A self-contained benchmark for [QTIP](https://arxiv.org/abs/2406.11235) (NeurIPS 2024) that quantizes a single random 4096×4096 FP16 weight matrix at K=2/3/4 bits and measures both quantization quality and inference throughput.

No LLM weights needed — everything runs on synthetic data.

## What it measures

| Script | What it does | Output |
|---|---|---|
| `prepare_qtip.py` | Quantizes the matrix, records per-step timing | `data/quant_timings.csv`, `data/qtip_K{2,3,4}.pt` |
| `benchmark_dequant.py` | Dequantize → full fp16 weight → `x @ W.T` (batch>1 path) | `results/dequant/results.csv` |
| `benchmark_fused.py` | Fused decode+matvec kernel, no weight materialisation (batch=1 path) | `results/fused/results.csv` |
| `plot_results.py` | Generates all plots from the CSVs | `results/*.png`, `results/dequant/*.png`, `results/fused/*.png` |

## Setup

**1. Clone with the QTIP submodule**

```bash
git clone --recurse-submodules <this-repo>
cd qtip-eval
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

**2. Create the conda environment**

```bash
conda create -n qtip python=3.11 -y
conda activate qtip
```

**3. Install dependencies**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git --no-build-isolation
pip install matplotlib
pip install -r requirements.txt --no-build-isolation
```

> If `fast-hadamard-transform` fails to install via pip, build from source:
> ```bash
> pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git
> ```

**4. Build the QTIP inference kernels** (required for fused benchmark)

```bash
cd qtip/qtip-kernels
python setup.py install
cd ../..
```

> The kernels are written for Ampere/Ada GPUs. They will compile on other architectures but may not reach peak performance.

## Running

Run the scripts in order:

```bash
# Step 1 — quantize (always re-runs, overwrites existing artefacts)
python prepare_qtip.py

# Step 2 — benchmark inference paths
python benchmark_dequant.py
python benchmark_fused.py

# Step 3 — generate plots
python plot_results.py
```

You can run Steps 2 and 3 independently of each other; `plot_results.py` will skip any plot whose source CSV is missing.

## Configuration

All parameters are defined as constants at the top of each script. The defaults match the QTIP paper's HYB code configuration:

| Parameter | Value | Description |
|---|---|---|
| `M, N` | 4096, 4096 | Weight matrix dimensions |
| `L` | 16 | Trellis length (2^16 = 65536 states) |
| `K` | 2, 3, 4 | Bits per weight |
| `V` | 2 | Vector dimension |
| `td_x, td_y` | 16, 16 | Trellis tile dimensions |
| `tlut_bits` | 9 | Tunable LUT bits (512-entry LUT) |
| `decode_mode` | `quantlut_sym` | HYB code variant |
| `scale_override` | 0.9 | Scale factor (from QTIP example.sh) |
| `sigma_reg` | 0.01 | Hessian regularisation |

## Outputs

**Plots saved to `results/`:**

- `quant_time.png` — Stacked bar of per-step quantization time + compression ratio
- `quality_vs_k.png` — W MSE and Output MSE vs K bits
- `latency_comparison.png` — Inference latency: dequant vs fused vs FP16 baseline
- `bandwidth_comparison.png` — Effective memory bandwidth for each path

**Plots saved to `results/dequant/` and `results/fused/`:**

- `latency_bar.png` — Latency and Output MSE per path
- `bandwidth_bar.png` — Effective bandwidth per path

## Notes on results

**Quantization time is flat across K=2/3/4.** The dominant cost is the LDLQ error-propagation loop — a 4096×4096 matrix multiply iterated 32 times. The Viterbi search over the trellis (whose cost does scale with K) is negligible by comparison.

**Fused matvec may be slower than dequant+gemm on recent GPUs (Blackwell/Ada).** The fused kernel (`decompress_matvec_qtip`) is hand-written CUDA C++ tuned for Ampere-era hardware. `decode_compressed` (the dequant path) uses Triton with `@torch.compile`, which adapts to the target GPU. On a 5090 (sm_120), the Triton-compiled path outperforms the older CUDA kernel. This gap would narrow on Ampere/Ada where the fused kernel was designed.

## Repository structure

```
qtip-eval/
  prepare_qtip.py       — quantization + timing
  benchmark_dequant.py  — dequant+gemm benchmark
  benchmark_fused.py    — fused matvec benchmark
  plot_results.py       — plotting
  qtip/                 — QTIP source (submodule)
  data/                 — generated weight artefacts and timing CSV
  results/              — benchmark CSVs and plots
```

## Reference

> **QTIP: Quantization with Trellises and Incoherence Processing**
> Guo et al., NeurIPS 2024 Spotlight
> https://arxiv.org/abs/2406.11235
# qtip-eval
