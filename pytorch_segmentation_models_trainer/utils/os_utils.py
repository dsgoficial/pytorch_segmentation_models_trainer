# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-02
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
 *                                                                         *
 ****
"""
import hashlib
import os
import shutil
from pathlib import Path

def remove_folder(folder):
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        return True
    except:
        return False

def create_folder(path_to_folder):
    if os.path.exists(path_to_folder):
        return path_to_folder
    os.makedirs(path_to_folder)
    return path_to_folder

def hash_file(filename):
    """"This function returns the SHA-1 hash
    of the file passed into it"""
    h = hashlib.sha1()
    with open(filename, 'rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h.update(chunk)
    return h.hexdigest()