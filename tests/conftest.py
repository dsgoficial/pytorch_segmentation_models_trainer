# conftest.py — resolve conflito fbgemm.dll vs GDAL no Windows
# Deve ser processado pelo pytest ANTES de importar qualquer test module.
import os
import sys

if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _torch_lib = os.path.join(
        sys.prefix, "Lib", "site-packages", "torch", "lib"
    )
    if os.path.isdir(_torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import pytorch_segmentation_models_trainer  # noqa: F401, E402
