---
sidebar_position: 1
title: Installation
---

# Installation

## Requirements

- Python 3.12+
- PyTorch 2.0+ (unpinned — `uv sync`/`pip install` pulls the current latest release)
- CUDA 13.0 (default PyPI build; requires compute capability sm_75+ — Turing or newer). See [CUDA and GPU Compatibility](#cuda-and-gpu-compatibility) below if you're on an older GPU.

## Quick Install

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package and project manager, written in Rust. It is the recommended way to manage this project.

```bash
# Clone the repository
git clone https://github.com/phborba/pytorch_segmentation_models_trainer.git
cd pytorch_segmentation_models_trainer

# Install dependencies and create a virtual environment
uv sync
```

### Option 2: PyPI

```bash
pip install pytorch_segmentation_models_trainer
```

### Option 3: From Source (pip)

```bash
git clone https://github.com/phborba/pytorch_segmentation_models_trainer.git
cd pytorch_segmentation_models_trainer
pip install -e .
```

## CUDA and GPU Compatibility

`torch`/`torchvision` are unpinned in this project, so a plain install pulls the latest PyTorch release. On Linux, the default PyPI wheel bundles **CUDA 13.0**, which requires compute capability **sm_75 or newer** — Turing, Ampere, Ada, Hopper, or Blackwell GPUs (RTX 20-series and up, A100, H100, etc.).

**Volta-generation GPUs (Tesla V100, sm_70) are not supported by the default install.** CUDA 13.0 dropped offline compilation for Maxwell/Pascal/Volta architectures. If you're running on a V100 (or any sm_70 card), install the CUDA 12.6 build explicitly after the normal install:

```bash
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Check which architectures your installed build actually supports:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

`pyproject.toml` does not define a `cu126` extra for this — uv has no clean way to express "use the default index normally, but override it only for one hardware target" without breaking the default install for everyone else, so the V100 case is a manual, documented step rather than an automated flag.

## Verify Installation

Test your installation:

```python
import pytorch_segmentation_models_trainer
print("Installation successful!")

# Check available modes
from pytorch_segmentation_models_trainer.main import main
```

Or use the CLI:

```bash
pytorch-smt --help
```


## Optional Dependencies

### For Advanced Features

```bash
# For visualization and plotting
pip install matplotlib seaborn

# For additional image processing
pip install opencv-python-headless

# For COCO dataset support
pip install pycocotools

# For PostGIS database integration
pip install psycopg2-binary geopandas

# For advanced metrics
pip install scikit-learn
```

### Development Dependencies

```bash
pip install pytest black flake8 pre-commit
```

## Common Issues

### CUDA/GPU Issues

**Problem**: CUDA out of memory
```bash
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in your config:
```yaml
hyperparameters:
  batch_size: 1  # Reduce from higher value
```

**Problem**: No CUDA devices available
```bash
AssertionError: Torch not compiled with CUDA support
```

**Solution**: Install PyTorch with CUDA (see [CUDA and GPU Compatibility](#cuda-and-gpu-compatibility) above — use `cu126` instead of the default index if you're on a Tesla V100 or other Volta-generation GPU):
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'pytorch_scatter'`

**Solution**: Install pytorch-scatter for your CUDA version:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**Problem**: `ImportError: cannot import name 'instantiate'`

**Solution**: Update Hydra:
```bash
pip install --upgrade hydra-core
```

## Tips

1. **Use uv (Highly Recommended)**:
   ```bash
   uv sync
   source .venv/bin/activate
   ```

2. **Check your CUDA version**:
   ```bash
   nvidia-smi
   ```

3. **For M1/M2 Macs**: Install with MPS support:
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
   ```

## Next Steps

- [Quick Start Guide](/docs/getting-started/quickstart) - Train your first model
- [Configuration](/docs/getting-started/configuration) - Understanding config files
- [Examples](/docs/examples/basic-segmentation) - Working examples
