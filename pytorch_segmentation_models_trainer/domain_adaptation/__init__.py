# -*- coding: utf-8 -*-
"""
Domain Adaptation module for pytorch_segmentation_models_trainer.

This package provides the infrastructure for training segmentation models with
domain adaptation techniques. It follows the Open/Closed Principle: adding a
new DA method requires only creating a new subclass of
``BaseDomainAdaptationMethod`` — no framework modification needed.

Package structure:
    base_method.py       — ``DomainAdaptationLossOutput`` dataclass and
                           ``BaseDomainAdaptationMethod`` abstract base class.
    schedulers.py        — Lambda schedulers for controlling the DA loss weight
                           over training (``ConstantScheduler``, ``LinearScheduler``,
                           ``DANNScheduler``).
    feature_hooks.py     — ``FeatureExtractorHook`` for capturing intermediate
                           feature maps without modifying the model.
    methods/             — Concrete DA method implementations.
    callbacks/           — PL callbacks for monitoring DA training
                           (``DomainAdaptationMonitorCallback``).
"""
