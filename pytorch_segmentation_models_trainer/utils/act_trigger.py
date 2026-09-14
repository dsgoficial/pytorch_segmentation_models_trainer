# -*- coding: utf-8 -*-
"""Adaptive Correction Trigger (ACT) — Liu et al., TGRS 2024 (AIO2)."""

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class ACTTracker:
    """Adaptive Correction Trigger (ACT) module, AIO2 paper section 2.3.

    Detects, without access to ground truth, the epoch at which noisy-label
    training should stop (warm-up) and online object-wise correction (O2C)
    should start — the middle of the "transition stage" between early
    learning and memorization (Liu, C. et al., "AIO2: Online Correction of
    Object Labels for Deep Learning with Incomplete Annotation in Remote
    Sensing Image Segmentation," IEEE TGRS 2024).

    Ported from the official reference implementation
    (https://github.com/zhu-xlab/AIO2, ``utils/early_learning_detection.py``),
    not re-derived from the paper text alone. Two details only visible in
    the code, not the paper:

    - The buffer ``z`` used to confirm a detected local minimum is not a
      free hyperparameter — it is ``mean(window_sizes)``, derived.
    - The published example (README) uses ``window_sizes=[10, 20, 30, 40]``
      with checkpoints saved every 5 epochs; used here as the default.

    Algorithm (per ``update()`` call, i.e. once per epoch):

    1. For each window size ``w`` in ``window_sizes``, fit a local linear
       regression over the last ``w`` epochs of accuracy history; its slope
       is the numerical gradient estimate for that window at this epoch.
    2. A window "detects" the end of the transition stage (``I_t``) the
       first time its slope stops being the historical minimum for that
       window *and* stays that way for more than ``buffer`` epochs (robust
       to noisy single-epoch fluctuations).
    3. Once every window has detected ``I_t``, fit an exponential-saturation
       curve to the accuracy history up to ``mean(I_t)``; use its analytic
       derivative and an adaptive threshold (mean slope from epoch 1 to
       ``I_t``) to find ``I_e``, the end of early learning.
    4. The trigger point is ``I_r = floor((I_e + I_t) / 2)``, the middle of
       the transition stage.

    Args:
        window_sizes: Sliding window sizes (in epochs) for the local
            gradient estimate. Default ``[10, 20, 30, 40]`` matches the
            official README's published example.
        **kwargs: Accepted for Hydra / ConfigStore compatibility.

    Example:
        tracker = ACTTracker(window_sizes=[10, 20, 30, 40])
        for epoch_iou in teacher_train_iou_per_epoch:
            i_r = tracker.update(epoch_iou)
            if i_r is not None:
                break  # trigger fired; i_r is the epoch to resume training from
    """

    def __init__(self, window_sizes: Optional[List[int]] = None, **kwargs) -> None:
        self.window_sizes: List[int] = (
            list(window_sizes) if window_sizes is not None else [10, 20, 30, 40]
        )
        if not self.window_sizes:
            raise ValueError("window_sizes must be non-empty")
        if any(w <= 0 for w in self.window_sizes):
            raise ValueError("window_sizes must all be positive")

        self.buffer: float = float(np.mean(self.window_sizes))  # z, derived
        self.history: List[float] = []
        self.triggered_at: Optional[int] = None

        self._ngs: Dict[int, List[float]] = {w: [] for w in self.window_sizes}
        self._detect_eps: np.ndarray = np.zeros(len(self.window_sizes))

    # ------------------------------------------------------------------
    # Curve fitting primitives (module-level functions in the reference
    # implementation; kept as static methods here for readability).
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b

    @staticmethod
    def _curve_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * (1 - np.exp(-b * x**c))

    @staticmethod
    def _curve_derivative(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * c * b * np.exp(-b * x**c) * x ** (c - 1)

    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> Optional[tuple]:
        try:
            popt, _ = curve_fit(
                self._linear_func,
                x,
                y,
                p0=(1, 0),
                method="trf",
                bounds=([0, -np.inf], [np.inf, np.inf]),
            )
        except RuntimeError:
            return None
        return tuple(popt)

    def _fit_curve(self, x: np.ndarray, y: np.ndarray) -> Optional[tuple]:
        try:
            popt, _ = curve_fit(
                self._curve_func,
                x,
                y,
                p0=(1, 0.5, 0.5),
                method="trf",
                bounds=([0, 0, 0], [1, np.inf, 1]),
            )
        except RuntimeError:
            return None
        return tuple(popt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, epoch_iou: float) -> Optional[int]:
        """Feed one epoch's training accuracy (teacher IoU) into the tracker.

        Args:
            epoch_iou: Training accuracy (e.g. mean IoU against the noisy
                label) for the epoch just completed.

        Returns:
            The detected resume epoch ``I_r`` the first time the trigger
            fires; ``None`` on every call before that. Once triggered,
            further calls are no-ops that keep returning the same value
            (``self.triggered_at``) — call sites should stop calling
            ``update`` once they observe a non-``None`` return, but this
            makes the tracker safe to call again regardless.
        """
        if self.triggered_at is not None:
            return self.triggered_at

        self.history.append(float(epoch_iou))
        data = self.history
        n_ep = len(data)

        for bi, ws in enumerate(self.window_sizes):
            if n_ep < ws:
                continue
            x0 = np.arange(n_ep - ws + 1, n_ep + 1)
            y0 = np.array(data[n_ep - ws : n_ep])
            fit = self._fit_linear(x0, y0)
            if fit is None:
                continue
            a, _b = fit
            self._ngs[ws].append(a)

            if min(self._ngs[ws]) < a:
                ind = int(np.argmin(self._ngs[ws])) + ws
                if n_ep - ind > self.buffer:
                    self._detect_eps[bi] = ind

        if (self._detect_eps > 0).sum() != len(self.window_sizes):
            return None

        dep = int(np.mean(self._detect_eps))

        x0 = np.arange(dep) + 1
        y0 = np.array(data[:dep])
        fit = self._fit_curve(x0, y0)
        if fit is None:
            return None
        a, b, c = fit

        yh = self._curve_func(x0, a, b, c)
        thr = (yh[-1] - yh[0]) / (x0[-1] - x0[0])
        yd = self._curve_derivative(x0, a, b, c)
        mid = int(np.sum(yd > thr))

        fdep = int((dep + mid) / 2)
        self.triggered_at = fdep
        return fdep
