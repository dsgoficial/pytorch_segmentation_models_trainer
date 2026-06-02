# -*- coding: utf-8 -*-
"""Mask post-processing and semantic boundary refinement.

Provides :class:`DenseCRFPostprocessor` (via ``pydensecrf``) and
:class:`GraphCutsPostprocessor` (via ``pygco``) for refining raw class
probability maps into clean segmentation masks.  Both classes raise
``ImportError`` at instantiation time when their optional backend library is
absent, so misconfiguration is caught early.

:class:`PostprocessingPipeline` chains an ordered list of postprocessors,
each acting independently on the original probabilities, and returns the
output of the last one.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BasePostprocessor(ABC):
    """Abstract base class for mask postprocessors.

    All postprocessors share the same interface: they accept class
    probability maps and the original image, and return a label map.
    """

    @abstractmethod
    def refine(self, probabilities: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Refine a raw probability map into a clean label map.

        Args:
            probabilities: Class probability map of shape
                ``(n_classes, H, W)``, dtype ``float32``.
            image: Original image of shape ``(H, W, 3)``, dtype ``uint8``,
                used as the appearance signal for spatial/color terms.

        Returns:
            Predicted label map of shape ``(H, W)`` with integer class IDs.
        """


# ---------------------------------------------------------------------------
# Dense CRF
# ---------------------------------------------------------------------------


class DenseCRFPostprocessor(BasePostprocessor):
    """Refine segmentation masks with Dense Conditional Random Fields.

    Applies a fully-connected Dense CRF using ``pydensecrf``.  The unary
    potentials come from the softmax probabilities.  Bilateral (appearance +
    position) and Gaussian (smoothness) pairwise terms are added.

    Args:
        n_iterations: Number of mean-field inference iterations.
        bilateral_sxy: Spatial standard deviation for the bilateral term.
        bilateral_srgb: Color standard deviation for the bilateral term.
        bilateral_compat: Compatibility weight for the bilateral term.
        gaussian_sxy: Spatial standard deviation for the Gaussian term.
        gaussian_compat: Compatibility weight for the Gaussian term.
        **kwargs: Ignored extra Hydra arguments.

    Raises:
        ImportError: If ``pydensecrf`` is not installed.

    Example YAML:
        postprocessing:
          _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.DenseCRFPostprocessor
          n_iterations: 5
          bilateral_sxy: 80
          bilateral_srgb: 13
          bilateral_compat: 10
          gaussian_sxy: 3
          gaussian_compat: 3
    """

    def __init__(
        self,
        n_iterations: int = 5,
        bilateral_sxy: float = 80.0,
        bilateral_srgb: float = 13.0,
        bilateral_compat: float = 10.0,
        gaussian_sxy: float = 3.0,
        gaussian_compat: float = 3.0,
        **kwargs,
    ):
        try:
            import pydensecrf.densecrf  # noqa: F401
            import pydensecrf.utils  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pydensecrf is required for DenseCRFPostprocessor. "
                "Install with: pip install pytorch-segmentation-models-trainer[gpu-ml]"
            ) from exc
        self.n_iterations = n_iterations
        self.bilateral_sxy = bilateral_sxy
        self.bilateral_srgb = bilateral_srgb
        self.bilateral_compat = bilateral_compat
        self.gaussian_sxy = gaussian_sxy
        self.gaussian_compat = gaussian_compat

    def refine(self, probabilities: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply Dense CRF inference.

        Args:
            probabilities: ``(n_classes, H, W)`` float32 softmax outputs.
            image: ``(H, W, 3)`` uint8 image used for bilateral appearance.

        Returns:
            ``(H, W)`` integer label map.
        """
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax

        n_classes, H, W = probabilities.shape
        proba_f32 = np.ascontiguousarray(probabilities.astype(np.float32))
        img_uint8 = np.ascontiguousarray(image.astype(np.uint8))

        d = dcrf.DenseCRF2D(W, H, n_classes)
        U = unary_from_softmax(proba_f32)
        d.setUnaryEnergy(U)
        d.addPairwiseBilateral(
            sxy=self.bilateral_sxy,
            srgb=self.bilateral_srgb,
            rgbim=img_uint8,
            compat=self.bilateral_compat,
        )
        d.addPairwiseGaussian(sxy=self.gaussian_sxy, compat=self.gaussian_compat)
        Q = d.inference(self.n_iterations)
        return np.argmax(Q, axis=0).reshape(H, W)


# ---------------------------------------------------------------------------
# Graph Cuts
# ---------------------------------------------------------------------------


class GraphCutsPostprocessor(BasePostprocessor):
    """Refine segmentation masks with Min-Cut / Max-Flow graph cuts.

    Uses ``pygco`` to solve a discrete MRF energy minimisation.  Unary
    costs derive from ``-log(probability)``.  Pairwise costs apply a
    Potts smoothness model scaled by image edge strength.

    Args:
        unary_scale: Multiplicative scale applied to the unary costs before
            rounding to integer (required by ``pygco``).
        pairwise_weight: Weight applied to the Potts pairwise term.
        n_iter: Maximum number of graph-cut iterations (``-1`` = until
            convergence).
        algorithm: Graph-cut algorithm — ``"swap"`` or ``"expansion"``.
        **kwargs: Ignored extra Hydra arguments.

    Raises:
        ImportError: If ``pygco`` is not installed.

    Example YAML:
        postprocessing:
          _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.GraphCutsPostprocessor
          unary_scale: 10.0
          pairwise_weight: 1.0
    """

    def __init__(
        self,
        unary_scale: float = 10.0,
        pairwise_weight: float = 1.0,
        n_iter: int = -1,
        algorithm: str = "swap",
        **kwargs,
    ):
        try:
            import pygco  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pygco is required for GraphCutsPostprocessor. "
                "Install with: pip install pytorch-segmentation-models-trainer[gpu-ml]"
            ) from exc
        self.unary_scale = unary_scale
        self.pairwise_weight = pairwise_weight
        self.n_iter = n_iter
        self.algorithm = algorithm

    def refine(self, probabilities: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply graph-cut energy minimisation.

        Args:
            probabilities: ``(n_classes, H, W)`` float32 softmax outputs.
            image: ``(H, W, 3)`` uint8 image used for edge-weighted pairwise
                costs.

        Returns:
            ``(H, W)`` integer label map.
        """
        import pygco

        n_classes, H, W = probabilities.shape
        proba = np.clip(probabilities.astype(np.float64), 1e-10, 1.0)

        # Unary costs: -log(probability), scaled and rounded to int32
        unary = (-np.log(proba)).transpose(1, 2, 0)  # (H, W, n_classes)
        unary_int = (unary * self.unary_scale).astype(np.int32)

        # Pairwise (Potts): off-diagonal penalty
        pairwise = ((1 - np.eye(n_classes)) * self.pairwise_weight).astype(np.int32)

        # Edge weights from grey-scale gradient magnitude
        grey = image.mean(axis=-1) if image.ndim == 3 else image.astype(float)
        grey = grey / (grey.max() + 1e-10)
        cost_v = (np.abs(np.diff(grey, axis=0)) * self.unary_scale + 1).astype(np.int32)
        cost_h = (np.abs(np.diff(grey, axis=1)) * self.unary_scale + 1).astype(np.int32)

        labels = pygco.cut_grid_graph(
            unary_cost=unary_int,
            pairwise_cost=pairwise,
            cost_v=cost_v,
            cost_h=cost_h,
            n_iter=self.n_iter,
            algorithm=self.algorithm,
        )
        return labels.reshape(H, W)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PostprocessingPipeline(BasePostprocessor):
    """Chain multiple postprocessors and return the last one's output.

    Each postprocessor receives the **original** probability map and image.
    Only the output of the final postprocessor is returned.  This design
    allows comparing or combining independent refinement strategies via
    configuration without altering the shared probability input.

    Args:
        postprocessors: Ordered list of :class:`BasePostprocessor` instances.
        **kwargs: Ignored extra Hydra arguments.

    Example YAML:
        postprocessing:
          _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.PostprocessingPipeline
          postprocessors:
            - _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.DenseCRFPostprocessor
              n_iterations: 5
    """

    def __init__(self, postprocessors: List[BasePostprocessor], **kwargs):
        if not postprocessors:
            raise ValueError(
                "PostprocessingPipeline requires at least one postprocessor."
            )
        self.postprocessors = list(postprocessors)

    def refine(self, probabilities: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply each postprocessor in order; return the last result.

        Args:
            probabilities: ``(n_classes, H, W)`` float32 softmax outputs.
            image: ``(H, W, 3)`` uint8 image.

        Returns:
            ``(H, W)`` integer label map from the last postprocessor.
        """
        result = None
        for processor in self.postprocessors:
            result = processor.refine(probabilities, image)
        return result
