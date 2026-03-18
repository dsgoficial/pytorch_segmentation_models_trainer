import os
import sys

# Windows: corrigir conflito de DLLs entre PyTorch (fbgemm.dll) e GDAL/rasterio.
# O conda coloca DLLs do GDAL no PATH que conflitam com dependencias do torch.
# Solucao: adicionar torch/lib ao DLL search path e importar torch ANTES de
# qualquer modulo que possa carregar GDAL (rasterio, fiona, etc).
if sys.platform == "win32":
    # Allow coexistence of libomp.dll (torch) and libiomp5md.dll (conda/MKL)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _torch_lib = os.path.join(
        sys.prefix, "Lib", "site-packages", "torch", "lib"
    )
    if os.path.isdir(_torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
        # Prioritize torch/lib on PATH to resolve dependencies first
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

    # Fix PROJ conflict: PostGIS installs an older proj.db that rasterio/GDAL
    # finds first via PATH. Force PROJ_DATA to rasterio's bundled version.
    if "PROJ_DATA" not in os.environ:
        _rasterio_proj = os.path.join(
            sys.prefix, "Lib", "site-packages", "rasterio", "proj_data"
        )
        if os.path.isdir(_rasterio_proj):
            os.environ["PROJ_DATA"] = _rasterio_proj

import torch  # noqa: E402, F401 — deve ser importado antes de rasterio/GDAL

from . import dataset_loader
from . import model_loader
from . import utils
from . import tools
from . import custom_callbacks

__all__ = ["dataset_loader", "model_loader", "tools", "utils"]
