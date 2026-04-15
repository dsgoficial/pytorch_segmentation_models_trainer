# -*- coding: utf-8 -*-
"""
Tests for BaseDomainAdaptationMethod and DomainAdaptationLossOutput.
"""
import pytest
import torch

from pytorch_segmentation_models_trainer.domain_adaptation.base_method import (
    BaseDomainAdaptationMethod,
    DomainAdaptationLossOutput,
)


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing the base contract
# ---------------------------------------------------------------------------


class _ConcreteMethod(BaseDomainAdaptationMethod):
    """Minimal implementation: returns a fixed zero loss."""

    def compute_da_loss(
        self,
        source_batch,
        target_batch,
        source_output,
        target_output,
        source_features,
        target_features,
        **kwargs
    ):
        loss = torch.tensor(0.0)
        return DomainAdaptationLossOutput(loss=loss, log_dict={"test_loss": 0.0})


class _MethodWithExtras(BaseDomainAdaptationMethod):
    """Method with an auxiliary network to test get_extra_parameter_groups."""

    requires_features = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import torch.nn as nn

        self.discriminator = nn.Linear(16, 1)

    def compute_da_loss(
        self,
        source_batch,
        target_batch,
        source_output,
        target_output,
        source_features,
        target_features,
        **kwargs
    ):
        loss = torch.tensor(0.5)
        return DomainAdaptationLossOutput(loss=loss, log_dict={"disc_loss": 0.5})

    def get_extra_parameter_groups(self):
        return [{"params": self.discriminator.parameters(), "lr": 1e-4}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDomainAdaptationLossOutput:
    def test_required_fields(self):
        loss = torch.tensor(1.5)
        out = DomainAdaptationLossOutput(loss=loss, log_dict={"x": 1.5})
        assert out.loss is loss
        assert out.log_dict == {"x": 1.5}
        assert out.extra == {}

    def test_extra_field(self):
        out = DomainAdaptationLossOutput(
            loss=torch.tensor(0.0),
            log_dict={},
            extra={"pseudo_labels": torch.zeros(4)},
        )
        assert "pseudo_labels" in out.extra


class TestBaseDomainAdaptationMethod:
    def test_abstract_compute_da_loss_raises(self):
        """Direct instantiation of base class raises NotImplementedError."""
        method = BaseDomainAdaptationMethod.__new__(BaseDomainAdaptationMethod)
        BaseDomainAdaptationMethod.__init__(method)
        with pytest.raises(NotImplementedError, match="compute_da_loss"):
            method.compute_da_loss({}, {}, None, None, {}, {})

    def test_concrete_subclass_works(self):
        method = _ConcreteMethod()
        out = method.compute_da_loss({}, {}, None, None, {}, {})
        assert isinstance(out, DomainAdaptationLossOutput)
        assert out.loss.item() == 0.0

    def test_default_flags(self):
        method = _ConcreteMethod()
        assert method.requires_features is False
        assert method.requires_target_labels is False

    def test_lambda_da_default(self):
        method = _ConcreteMethod()
        assert method.lambda_da == pytest.approx(1.0)

    def test_lambda_da_read_from_kwargs(self):
        method = _ConcreteMethod(lambda_da=0.3)
        assert method.lambda_da == pytest.approx(0.3)

    def test_lifecycle_hooks_are_noop(self):
        """Default hooks should not raise."""
        method = _ConcreteMethod()
        method.on_fit_start(None)
        method.on_train_epoch_start(None, 0)
        method.on_train_epoch_end(None, 0)

    def test_get_extra_parameter_groups_default_empty(self):
        method = _ConcreteMethod()
        assert method.get_extra_parameter_groups() == []

    def test_requires_features_flag(self):
        method = _MethodWithExtras()
        assert method.requires_features is True

    def test_extra_parameter_groups(self):
        method = _MethodWithExtras()
        groups = method.get_extra_parameter_groups()
        assert len(groups) == 1
        assert "params" in groups[0]
        assert groups[0]["lr"] == pytest.approx(1e-4)

    def test_method_is_nn_module(self):
        import torch.nn as nn

        method = _ConcreteMethod()
        assert isinstance(method, nn.Module)

    def test_discriminator_params_included(self):
        """Parameters of auxiliary nn.Module attrs appear in method.parameters()."""
        method = _MethodWithExtras()
        method_param_ids = {id(p) for p in method.parameters()}
        disc_param_ids = {id(p) for p in method.discriminator.parameters()}
        assert disc_param_ids.issubset(method_param_ids)

    def test_cfg_stored_as_omegaconf(self):
        from omegaconf import DictConfig

        method = _ConcreteMethod(lambda_da=0.5, my_param=42)
        assert isinstance(method.cfg, DictConfig)
        assert method.cfg.my_param == 42
