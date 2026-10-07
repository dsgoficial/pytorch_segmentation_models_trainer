# -*- coding: utf-8 -*-
"""
Generalized few-shot semantic segmentation (GFSS) module.

A segmentation model trained on base classes is adapted with a few labelled
support tiles to also predict novel classes. Novel classes are declared as
*children* of a base *mother* class (``ClassHierarchy``): in standard GFSS
(DIaM, ClassTrans) the mother is the background; in category splitting the
mother is the base class that contained the novel class during base training.

Package structure:
    hierarchy.py   — ``ClassHierarchy``: mother → children mapping and the
                     new → old projections shared by every method.
"""
