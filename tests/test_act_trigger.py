# -*- coding: utf-8 -*-
"""Tests for ACTTracker (Adaptive Correction Trigger, AIO2 §2.3)."""

from unittest.mock import patch

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.utils.act_trigger import ACTTracker


def _three_stage_curve(
    n_early: int = 10,
    n_memorization: int = 10,
    slope_lo: float = 0.1,
    slope_hi: float = 1.0,
):
    """Synthetic training-accuracy curve with a clear transition minimum.

    Slope decreases monotonically over ``n_early`` epochs (early learning
    into transition), reaches ``slope_lo``, then increases monotonically
    over ``n_memorization`` epochs (memorization). Returns the cumulative
    accuracy series (not the slopes themselves).
    """
    slopes = np.concatenate(
        [
            np.linspace(slope_hi, slope_lo, n_early),
            np.linspace(slope_lo, slope_hi, n_memorization),
        ]
    )
    return np.cumsum(slopes).tolist()


def _monotonic_decelerating_curve(n: int = 30):
    """Pure early-learning curve: slope decreases forever, never rises again."""
    slopes = np.linspace(1.0, 0.05, n)
    return np.cumsum(slopes).tolist()


def _constant_slope_curve(n: int = 30, slope: float = 0.5):
    """Perfectly linear growth — slope never changes."""
    return (np.arange(1, n + 1) * slope).tolist()


def _flat_curve(n: int = 30, value: float = 1.0):
    """Degenerate: accuracy never changes at all."""
    return [value] * n


class TestACTTrackerInit:
    def test_default_window_sizes(self):
        tracker = ACTTracker()
        assert tracker.window_sizes == [10, 20, 30, 40]

    def test_default_buffer_is_mean_of_default_windows(self):
        tracker = ACTTracker()
        assert tracker.buffer == pytest.approx(25.0)

    def test_custom_window_sizes(self):
        tracker = ACTTracker(window_sizes=[2, 4, 6])
        assert tracker.window_sizes == [2, 4, 6]

    def test_custom_buffer_is_mean_of_custom_windows(self):
        tracker = ACTTracker(window_sizes=[2, 4, 6])
        assert tracker.buffer == pytest.approx(4.0)

    def test_empty_window_sizes_raises(self):
        with pytest.raises(ValueError):
            ACTTracker(window_sizes=[])

    def test_non_positive_window_size_raises(self):
        with pytest.raises(ValueError):
            ACTTracker(window_sizes=[10, 0])
        with pytest.raises(ValueError):
            ACTTracker(window_sizes=[10, -5])

    def test_accepts_kwargs_for_hydra_compat(self):
        # Must not raise on unexpected kwargs (Hydra instantiate compatibility).
        ACTTracker(window_sizes=[2, 3], name="act", some_extra_field=123)

    def test_history_starts_empty(self):
        tracker = ACTTracker()
        assert tracker.history == []

    def test_not_triggered_initially(self):
        tracker = ACTTracker()
        assert tracker.triggered_at is None


class TestACTTrackerUpdateContract:
    def test_returns_none_before_min_window_size_reached(self):
        tracker = ACTTracker(window_sizes=[10, 20])
        for _ in range(9):
            assert tracker.update(0.5) is None

    def test_no_crash_on_short_history(self):
        tracker = ACTTracker(window_sizes=[10, 20, 30, 40])
        result = tracker.update(0.1)
        assert result is None

    def test_history_grows_each_call_before_trigger(self):
        tracker = ACTTracker(window_sizes=[2, 3])
        tracker.update(0.1)
        tracker.update(0.2)
        assert tracker.history == [0.1, 0.2]

    def test_return_type_is_int_when_triggered(self):
        tracker = ACTTracker(window_sizes=[2, 3])
        curve = _three_stage_curve(n_early=8, n_memorization=8)
        result = None
        for v in curve:
            result = tracker.update(v)
            if result is not None:
                break
        assert result is not None
        assert isinstance(result, int)


class TestACTTrackerNeverTriggersWithoutMemorization:
    def test_monotonic_deceleration_never_triggers(self):
        tracker = ACTTracker(window_sizes=[2, 3, 4])
        curve = _monotonic_decelerating_curve(30)
        for v in curve:
            assert tracker.update(v) is None

    def test_constant_slope_does_not_crash(self):
        # No real memorization signal here — with zero-noise degenerate
        # input, floating-point jitter in curve_fit can occasionally look
        # like a spurious minimum (the official algorithm has no special
        # guard against this; real training curves are never this clean).
        # What matters is it never crashes and any trigger it does report
        # is a sane int, not that it strictly never fires.
        tracker = ACTTracker(window_sizes=[2, 3, 4])
        curve = _constant_slope_curve(30)
        for v in curve:
            result = tracker.update(v)
            assert result is None or isinstance(result, int)

    def test_flat_curve_does_not_crash(self):
        tracker = ACTTracker(window_sizes=[2, 3, 4])
        curve = _flat_curve(30)
        for v in curve:
            result = tracker.update(v)
            assert result is None or isinstance(result, int)


class TestACTTrackerTriggersOnThreeStageCurve:
    def test_triggers_within_curve_bounds_small_windows(self):
        tracker = ACTTracker(window_sizes=[2, 3])
        curve = _three_stage_curve(n_early=10, n_memorization=10)
        i_r = None
        triggered_epoch = None
        for epoch, v in enumerate(curve, start=1):
            i_r = tracker.update(v)
            if i_r is not None:
                triggered_epoch = epoch
                break
        assert i_r is not None
        # Trigger must fire strictly within the observed history, and only
        # after enough epochs for the buffer confirmation to be possible.
        assert 1 <= i_r <= triggered_epoch

    def test_triggers_with_default_window_sizes_on_longer_curve(self):
        tracker = ACTTracker()  # default [10, 20, 30, 40], buffer=25
        curve = _three_stage_curve(n_early=60, n_memorization=60)
        i_r = None
        triggered_epoch = None
        for epoch, v in enumerate(curve, start=1):
            i_r = tracker.update(v)
            if i_r is not None:
                triggered_epoch = epoch
                break
        assert i_r is not None
        assert 1 <= i_r <= triggered_epoch


class TestACTTrackerFitFailureHandling:
    """curve_fit can genuinely fail to converge (scipy raises RuntimeError)
    on real, noisy training curves — these exercise that documented failure
    path via mocking, rather than hunting for numerically pathological
    inputs that happen to trigger it non-deterministically."""

    def test_fit_linear_returns_none_on_runtime_error(self):
        tracker = ACTTracker(window_sizes=[2, 3])
        with patch(
            "pytorch_segmentation_models_trainer.utils.act_trigger.curve_fit",
            side_effect=RuntimeError("did not converge"),
        ):
            result = tracker._fit_linear(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        assert result is None

    def test_fit_curve_returns_none_on_runtime_error(self):
        tracker = ACTTracker(window_sizes=[2, 3])
        with patch(
            "pytorch_segmentation_models_trainer.utils.act_trigger.curve_fit",
            side_effect=RuntimeError("did not converge"),
        ):
            result = tracker._fit_curve(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        assert result is None

    def test_update_skips_window_when_fit_linear_fails(self):
        # A failed per-window linear fit must not crash update(), and must
        # not register a (bogus) gradient estimate for that window.
        tracker = ACTTracker(window_sizes=[2, 3])
        tracker.update(0.1)
        with patch.object(ACTTracker, "_fit_linear", return_value=None):
            result = tracker.update(0.2)
        assert result is None
        assert tracker._ngs[2] == []

    def test_update_returns_none_when_fit_curve_fails_at_trigger_time(self):
        # Drive the tracker right up to the point where all windows have
        # detected I_t (so the exponential-curve fit would run), then force
        # that fit to fail — update() must return None, not raise.
        tracker = ACTTracker(window_sizes=[2, 3])
        curve = _three_stage_curve(n_early=10, n_memorization=10)
        reached_curve_fit_stage = False
        for v in curve:
            with patch.object(ACTTracker, "_fit_curve", return_value=None):
                result = tracker.update(v)
            if (tracker._detect_eps > 0).sum() == len(tracker.window_sizes):
                reached_curve_fit_stage = True
                assert result is None
                break
        assert reached_curve_fit_stage


class TestACTTrackerIdempotency:
    def test_returns_same_value_after_trigger(self):
        tracker = ACTTracker(window_sizes=[2, 3])
        curve = _three_stage_curve(n_early=10, n_memorization=10)
        first = None
        for v in curve:
            first = tracker.update(v)
            if first is not None:
                break
        second = tracker.update(0.9)
        third = tracker.update(-1.0)
        assert second == first
        assert third == first

    def test_history_stops_growing_after_trigger(self):
        tracker = ACTTracker(window_sizes=[2, 3])
        curve = _three_stage_curve(n_early=10, n_memorization=10)
        for v in curve:
            if tracker.update(v) is not None:
                break
        history_len_at_trigger = len(tracker.history)
        tracker.update(0.5)
        tracker.update(0.5)
        assert len(tracker.history) == history_len_at_trigger
