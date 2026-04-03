# qtip-eval

A self-contained benchmark for [QTIP](https://arxiv.org/abs/2406.11235) (NeurIPS 2024 Spotlight)
that quantizes a single random 4096×4096 FP16 weight matrix at K=2/3/4 bits and measures both
quantization quality and inference throughput, including hardware-level profiling via Nsight Compute.

No LLM weights needed — everything runs on synthetic data.

See [`results/README.md`](results/README.md) for the full benchmark report and findings.

---

## Scripts

| Script | What it does | Output |
|---|---|---|
| `prepare_qtip.py` | Quantize the matrix, record per-step timing | `data/quant_timings.csv`, `data/qtip_K{2,3,4}.pt` |
| `benchmark_dequant.py` | Dequantize → full FP16 weight → cuBLAS gemv | `results/{gpu}/dequant/results.csv` |
| `benchmark_fused.py` | Fused decode+matvec kernel, full step-by-step timing breakdown | `results/{gpu}/fused/results.csv` |
| `plot_results.py` | Generate all plots from CSVs | `results/{gpu}/**/*.png` |
| `profile_kernel.py` | Minimal single-kernel launch script for Nsight Compute | (used by `profile_ncu.py`) |
| `profile_ncu.py` | Run `ncu` via sudo, parse stall metrics, plot and save | `results/{gpu}/fused/ncu_stall_K{k}.{png,csv}` |

Results are placed in `results/{gpu}/` where `{gpu}` is the GPU name (e.g. `RTX_5090`).

---

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

**4. Build the QTIP inference kernels** (required for fused benchmark and profiling)

```bash
cd qtip/qtip-kernels
python setup.py install
cd ../..
```

> The kernels are written and tuned for Ampere/Ada GPUs. They compile on other architectures
> but performance may differ.

---

## Running

### Core benchmark

Run in order:

```bash
# Step 1 — quantize (always re-runs, overwrites existing artefacts)
python prepare_qtip.py

# Step 2 — benchmark both inference paths
python benchmark_dequant.py
python benchmark_fused.py

# Step 3 — generate all plots
python plot_results.py
```

Steps 2 and 3 are independent of each other; `plot_results.py` skips any plot whose source CSV
is missing. Results go to `results/{gpu}/`.

### Nsight Compute profiling

Nsight Compute requires sudo to access hardware performance counters.

**Option A — Enable counters permanently (recommended for a personal workstation):**

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
# Then reboot, or reload the driver:
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

**Option B — Run with sudo each time:**

Use `profile_ncu.py`, which handles the sudo invocation automatically:

```bash
# Profile K=2 (default)
python profile_ncu.py

# Profile a specific bitrate
python profile_ncu.py --k 4

# Profile all three bitrates in sequence
python profile_ncu.py --all-k
```

The script finds `ncu` and `python` via `which`, passes them to `sudo -E`, captures the output,
parses the warp stall metrics, and saves:

```
results/{gpu}/fused/ncu_stall_K{k}.png   — stall breakdown bar chart
results/{gpu}/fused/ncu_stall_K{k}.csv   — raw metric values
results/{gpu}/fused/ncu_stall_summary.txt — annotated bottleneck summary
```

To run the ncu command manually (e.g. to collect a full `.ncu-rep` for the GUI):

```bash
sudo -E $(which ncu) \
    --kernel-name kernel_decompress_matvec \
    --launch-skip 5 --launch-count 1 \
    --set full -o profile_out \
    $(which python) profile_kernel.py

# View on command line (import the saved file)
ncu --import profile_out.ncu-rep

# Query specific stall metrics from a saved file
ncu --import profile_out.ncu-rep --csv \
    --metrics "regex:smsp__warp_issue_stalled.*per_warp_active.pct"
```

---

## Configuration

All parameters are constants at the top of each script. The defaults match the QTIP paper's HYB
code configuration:

| Parameter | Value | Description |
|---|---|---|
| `M, N` | 4096, 4096 | Weight matrix dimensions |
| `L` | 16 | Trellis shift-register bits (2^16 = 65536 states) |
| `K` | 2, 3, 4 | Bits per weight |
| `V` | 1 | VQ dimension |
| `tlut_bits` | 9 | Tunable LUT bits (512-entry codebook) |
| `decode_mode` | `quantlut_sym` | HYB code variant |
| `td_x, td_y` | 16, 16 | Trellis tile dimensions |
| `scale_override` | 0.9 | Scale factor |
| `sigma_reg` | 0.01 | Hessian regularisation |
| `REPEATS` | 200 | Timing repetitions per measurement |

---

## Outputs

**Per-GPU results in `results/{gpu}/`:**

| File | Description |
|---|---|
| `quant_time.png` | Stacked quantization time breakdown + compression ratio |
| `quality_vs_k.png` | Weight MSE and output MSE vs K bits |
| `latency_comparison.png` | Inference latency: dequant vs fused vs FP16 |
| `bandwidth_comparison.png` | Effective memory bandwidth per path |
| `dequant/latency_bar.png` | Dequant latency + output MSE subplot |
| `dequant/bandwidth_bar.png` | Dequant effective bandwidth |
| `fused/latency_bar.png` | Fused latency with 5-step breakdown (input_rot / packed_read / decode_overhead / output_rot / output_scale) |
| `fused/bandwidth_bar.png` | Fused effective bandwidth |
| `fused/kernel_breakdown.png` | Kernel-internal phase breakdown (LUT load / decode loop / reduction), scaled to wall-clock |
| `fused/ncu_stall_K{k}.png` | Nsight Compute warp stall breakdown by category |
| `fused/ncu_stall_K{k}.csv` | Raw stall metric values |
| `fused/ncu_stall_summary.txt` | Annotated bottleneck explanation |

**Fused CSV columns** (`results/{gpu}/fused/results.csv`):

| Column | Description |
|---|---|
| `ms` | End-to-end fused matvec latency |
| `input_rot_ms` | (x · SU) → HadUt / scale |
| `fused_decode_ms` | decompress_matvec_qtip kernel (wall-clock) |
| `packed_read_ms` | HBM read proxy: `packed.sum()` |
| `decode_overhead_ms` | fused_decode − packed_read (compute above HBM floor) |
| `output_rot_ms` | HadU(out) |
| `output_scale_ms` | out · (SV · Wscale · scale) |
| `kernel_codebook_ms` | LUT load phase (globaltimer, proportionally scaled) |
| `kernel_loop_ms` | Decode loop phase (globaltimer, proportionally scaled) |
| `kernel_reduce_ms` | Reduction phase (globaltimer, proportionally scaled) |

> **Note on kernel_* columns**: The CUDA `%globaltimer` register on Blackwell counts only active
> SM execution cycles, not memory stall cycles. Raw values are ~200× smaller than wall-clock time.
> The benchmark scales them proportionally to `fused_decode_ms` so they sum to the correct total,
> giving accurate phase *fractions* even though absolute values would be misleading.

---

## Repository Structure

```
qtip-eval/
  prepare_qtip.py        — quantization + timing
  benchmark_dequant.py   — dequant+gemv benchmark
  benchmark_fused.py     — fused matvec benchmark with step breakdown
  plot_results.py        — all plots
  profile_kernel.py      — minimal kernel launch for ncu
  profile_ncu.py         — ncu stall profiling + plotting
  qtip/                  — QTIP source (git submodule)
  data/                  — generated weight artefacts (gitignored)
  results/               — benchmark CSVs and plots (gitignored)
    README.md            — full benchmark report
    {gpu}/               — per-GPU results
      dequant/
      fused/
```

---

## Reference

> **QTIP: Quantization with Trellises and Incoherence Processing**
> Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa
> NeurIPS 2024 Spotlight
> https://arxiv.org/abs/2406.11235
