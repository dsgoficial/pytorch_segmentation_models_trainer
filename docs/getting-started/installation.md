# Installation

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+ (for GPU acceleration)

## 🔧 Quick Install

### Option 1: PyPI (Recommended)

```bash
pip install pytorch_segmentation_models_trainer
```

!!! warning "GPU Acceleration"
    If you want GPU acceleration and are not using Docker, install `pytorch_scatter` first:
    
    ```bash
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
    ```
    
    Replace `cu113` with your CUDA version (cu102, cu113, cu116, etc.)

### Option 2: From Source

```bash
git clone https://github.com/phborba/pytorch_segmentation_models_trainer.git
cd pytorch_segmentation_models_trainer
pip install -e .
```

### Option 3: Docker (Easiest for GPU)

We provide pre-built Docker images with all dependencies:

```bash
# Pull the latest image
docker pull phborba/pytorch_segmentation_models_trainer:latest

# Run interactively
docker run -it --gpus all \
    -v /path/to/your/data:/data \
    -v /path/to/your/configs:/configs \
    phborba/pytorch_segmentation_models_trainer:latest bash

# Run training directly
docker run --gpus all \
    -v /path/to/your/data:/data \
    -v /path/to/your/configs:/configs \
    phborba/pytorch_segmentation_models_trainer:latest \
    pytorch-smt --config-dir /configs --config-name my_config +mode=train
```

## 🔍 Verify Installation

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

## 🐳 Docker Compose Setup

For development with database support:

```yaml
version: '3'
services:
  db:
    image: postgis/postgis
    environment:
      POSTGRES_DB: test_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
  
  app:
    image: phborba/pytorch_segmentation_models_trainer:latest
    volumes:
      - .:/workspace
    depends_on:
      - db
```

Run with:
```bash
docker-compose up -d
```

## 📦 Optional Dependencies

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

## 🚨 Common Issues

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

**Solution**: Install PyTorch with CUDA:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'pytorch_scatter'`

**Solution**: Install pytorch-scatter for your CUDA version:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

**Problem**: `ImportError: cannot import name 'instantiate'`

**Solution**: Update Hydra:
```bash
pip install --upgrade hydra-core
```

## 💡 Tips

1. **Use virtual environments**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install pytorch_segmentation_models_trainer
   ```

2. **Check your CUDA version**:
   ```bash
   nvidia-smi
   ```

3. **For M1/M2 Macs**: Use the Docker option or install with MPS support:
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
   ```

## ✅ Next Steps

- [Quick Start Guide](quickstart.md) - Train your first model
- [Configuration](configuration.md) - Understanding config files  
- [Examples](../examples/basic-segmentation.md) - Working examples