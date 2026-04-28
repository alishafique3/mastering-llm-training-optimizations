# PyTorch LLM Training Optimizations

A simple benchmark project showing how common PyTorch training optimizations can significantly improve throughput for training a causal language model.

Base model used: `Qwen/Qwen3-0.6B`.


## Installation

```bash
# Create conda env
conda create -n optim python=3.11 -y
conda activate optim

# Install uv (fast pip)
pip install uv

# Install dependencies
uv pip install transformers datasets accelerate vllm torch

# Run
python optim.py
```

## Optimization Steps

Applied incrementally — each one stacks on the previous:

| # | Optimization | Change |
|---|---|---|
| 0 | **Baseline** | FP32, eager attention, batch=2, no compile |
| 1 | **TF32** | `torch.set_float32_matmul_precision("high")` |
| 2 | **BF16** | `dtype=torch.bfloat16` |
| 3 | **SDPA / Flash Attention** | `attn_implementation="sdpa"` |
| 4 | **Fused AdamW** | `fused=True` |
| 5 | **torch.compile** | `model = torch.compile(model)` |
| 6 | **Larger batch size** | `batch_size=32` |
| 7 | **DataLoader speedups** | `num_workers=4`, `pin_memory=True`, `non_blocking=True` |

## Metrics (DCGM)

Monitor live with:

```bash
dcgmi dmon -e 203,1002,1003,1004,1006,1007,1008,1013,1014,1005,252,250,155,150,140 -d 2000
```

## Results
| Optimization        | Throughput (tok/sec) | GPU Memory (GB) | THMMA (%) | FP32A (%) |
|---------------------|----------------------|-----------------|-----------|-----------|
| Baseline (FP32)     | 3.0K                 | 23.7            | 0.1       | 68.4      |
| + TF32              | 8.7K                 | 23.7            | 26.9      | 2.5       |
| + BF16              | 11.3K                | 15.7            | 17.7      | 3.3       |
| + SDPA Attention    | 15.4K                | 9.8             | 22.5      | 3.3       |
| + Fused AdamW       | 16.7K                | 9.8             | 24.3      | 3.3       |
| + torch.compile     | 21.5K                | 8.5             | 32.6      | 6.5       |
| + Batch size 32     | 28.3K                | 42.4            | 43.6      | 8.2       |



