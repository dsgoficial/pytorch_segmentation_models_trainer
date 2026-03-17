# Root-level conftest.py
# Provides collection-time skip for test files that require torch_scatter.
# Uses a subprocess check so that a broken/crashing torch_scatter installation
# (e.g. macOS + Python 3.12 ABI mismatch) does not crash the test runner.

import subprocess
import sys

def _torch_scatter_works() -> bool:
    """Return True only if torch_scatter can be imported in a subprocess without crashing."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch_scatter"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


_HAS_TORCH_SCATTER = _torch_scatter_works()

# Test files that require torch_scatter (directly or transitively via train.py)
_TORCH_SCATTER_TESTS = {
    "test_detection_model.py",
    "test_frame_field_model.py",
    "test_gradient_centralization.py",
    "test_inference.py",
    "test_inference_service.py",
    "test_mod_polygonrnn.py",
    "test_optimizers.py",
    "test_polygonizer.py",
    "test_polygonrnn_model.py",
    "test_polygonrnn_utils.py",
    "test_predict.py",
    "test_script.py",
    "test_skeletonize_tensor_tools.py",
    "test_train.py",
}


def pytest_ignore_collect(collection_path, config):
    """Skip test files that transitively require torch_scatter when unavailable or broken."""
    if not _HAS_TORCH_SCATTER and collection_path.name in _TORCH_SCATTER_TESTS:
        return True
    return False
