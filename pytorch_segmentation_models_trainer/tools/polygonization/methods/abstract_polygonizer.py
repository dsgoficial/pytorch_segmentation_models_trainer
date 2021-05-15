# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-08
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer 
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ****
"""
import abc
from dataclasses import dataclass

@dataclass
class Polygonizer(abc.ABC):
    DEBUG: bool = False

    @abc.abstractclassmethod
    def compute(self, seg_batch, crossfield_batch, pre_computed=None):
        pass
    
    @abc.abstractclassmethod
    def shapely_postprocess(self, input_objects):
        pass
    
    @abc.abstractclassmethod
    def post_process(self, input_objects):
        pass