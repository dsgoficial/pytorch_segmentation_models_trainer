#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-10-15
        git sha              : $Format:%H$
        copyright            : (C) 2025 by Philipe Borba - Cartographic Engineer
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

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def validate_experiment_config(exp_config: DictConfig, exp_idx: int) -> bool:
    """
    Valida configuração de um experimento.
    
    Args:
        exp_config: Config do experimento
        exp_idx: Índice do experimento
        
    Returns:
        True se válido, False caso contrário
    """
    errors = []
    warnings = []
    
    # Validar campos obrigatórios
    required_fields = ['name', 'predict_config', 'checkpoint_path', 'output_folder']
    for field in required_fields:
        if field not in exp_config:
            errors.append(f"Experiment {exp_idx}: Missing required field '{field}'")
    
    if errors:
        return False, errors, warnings
    
    # Validar predict_config existe
    if not os.path.exists(exp_config.predict_config):
        errors.append(
            f"Experiment {exp_idx} ({exp_config.name}): "
            f"predict_config not found: {exp_config.predict_config}"
        )
    
    # Validar checkpoint existe
    if not os.path.exists(exp_config.checkpoint_path):
        errors.append(
            f"Experiment {exp_idx} ({exp_config.name}): "
            f"checkpoint not found: {exp_config.checkpoint_path}"
        )
    
    # Warning se output_folder já existe
    if os.path.exists(exp_config.output_folder):
        pred_files = list(Path(exp_config.output_folder).glob("seg_*_output.tif"))
        if len(pred_files) > 0:
            warnings.append(
                f"Experiment {exp_idx} ({exp_config.name}): "
                f"output_folder already contains {len(pred_files)} predictions. "
                f"Will skip if skip_existing_predictions=true"
            )
    
    return len(errors) == 0, errors, warnings


def validate_dataset_config(dataset_config: DictConfig) -> tuple:
    """
    Valida configuração do dataset.
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Verificar se está usando CSV ou build_from_folders
    if dataset_config.build_csv_from_folders.enabled:
        # Validar pastas
        if not os.path.exists(dataset_config.build_csv_from_folders.images_folder):
            errors.append(
                f"Images folder not found: "
                f"{dataset_config.build_csv_from_folders.images_folder}"
            )
        
        if not os.path.exists(dataset_config.build_csv_from_folders.masks_folder):
            errors.append(
                f"Masks folder not found: "
                f"{dataset_config.build_csv_from_folders.masks_folder}"
            )
        
        # Verificar se há imagens nas pastas
        if os.path.exists(dataset_config.build_csv_from_folders.images_folder):
            images = list(
                Path(dataset_config.build_csv_from_folders.images_folder).glob(
                    dataset_config.build_csv_from_folders.image_pattern
                )
            )
            if len(images) == 0:
                errors.append(
                    f"No images found in images_folder with pattern "
                    f"{dataset_config.build_csv_from_folders.image_pattern}"
                )
            else:
                print(f"  ✓ Found {len(images)} images")
        
    else:
        # Validar CSV existe
        if not os.path.exists(dataset_config.input_csv_path):
            errors.append(
                f"Dataset CSV not found: {dataset_config.input_csv_path}"
            )
        else:
            # Verificar estrutura do CSV
            import pandas as pd
            try:
                df = pd.read_csv(dataset_config.input_csv_path)
                
                required_cols = ['image', 'mask']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    errors.append(
                        f"Dataset CSV missing columns: {missing_cols}"
                    )
                else:
                    print(f"  ✓ CSV has {len(df)} rows")
                    
                    # Verificar se arquivos existem (amostra)
                    sample_size = min(5, len(df))
                    missing_images = 0
                    missing_masks = 0
                    
                    for _, row in df.head(sample_size).iterrows():
                        if not os.path.exists(row['image']):
                            missing_images += 1
                        if not os.path.exists(row['mask']):
                            missing_masks += 1
                    
                    if missing_images > 0:
                        warnings.append(
                            f"{missing_images}/{sample_size} sampled images not found"
                        )
                    if missing_masks > 0:
                        warnings.append(
                            f"{missing_masks}/{sample_size} sampled masks not found"
                        )
                    
            except Exception as e:
                errors.append(f"Error reading CSV: {e}")
    
    return len(errors) == 0, errors, warnings


def validate_metrics_config(metrics_config: DictConfig) -> tuple:
    """
    Valida configuração de métricas.
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    if len(metrics_config.segmentation_metrics) == 0:
        warnings.append("No metrics configured. Only confusion matrix will be computed.")
    
    # Verificar se todas as métricas têm _target_
    for idx, metric_cfg in enumerate(metrics_config.segmentation_metrics):
        if '_target_' not in metric_cfg:
            errors.append(f"Metric {idx}: Missing '_target_' field")
    
    return len(errors) == 0, errors, warnings


def validate_output_config(output_config: DictConfig) -> tuple:
    """
    Valida configuração de output.
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Verificar se base_dir é acessível
    base_dir = Path(output_config.base_dir)
    
    if base_dir.exists() and not os.access(base_dir, os.W_OK):
        errors.append(f"Output base_dir is not writable: {base_dir}")
    
    # Tentar criar se não existe
    if not base_dir.exists():
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created output directory: {base_dir}")
            base_dir.rmdir()  # Remover dir de teste
        except Exception as e:
            errors.append(f"Cannot create output base_dir: {e}")
    
    return len(errors) == 0, errors, warnings


@hydra.main(
    version_base=None,
    config_path="configs/evaluation",
    config_name="pipeline_config"
)
def main(cfg: DictConfig):
    """
    Valida configuração do pipeline de avaliação.
    
    Este script verifica:
    - Experimentos: configs, checkpoints, output folders
    - Dataset: CSV ou pastas
    - Métricas: configuração válida
    - Output: diretórios acessíveis
    
    Usage:
        python validate_evaluation_config.py
        python validate_evaluation_config.py experiments[0].checkpoint_path=/new/path
    """
    print("="*80)
    print("VALIDATION: EVALUATION PIPELINE CONFIG")
    print("="*80)
    
    all_valid = True
    all_errors = []
    all_warnings = []
    
    # 1. Validar experimentos
    print("\n[1/4] Validating experiments...")
    for idx, exp in enumerate(cfg.experiments):
        print(f"\n  Experiment {idx + 1}: {exp.name}")
        is_valid, errors, warnings = validate_experiment_config(exp, idx)
        
        if not is_valid:
            all_valid = False
            all_errors.extend(errors)
            for error in errors:
                print(f"    ✗ ERROR: {error}")
        else:
            print(f"    ✓ Valid")
        
        all_warnings.extend(warnings)
        for warning in warnings:
            print(f"    ⚠ WARNING: {warning}")
    
    # 2. Validar dataset
    print("\n[2/4] Validating dataset...")
    is_valid, errors, warnings = validate_dataset_config(cfg.evaluation_dataset)
    
    if not is_valid:
        all_valid = False
        all_errors.extend(errors)
        for error in errors:
            print(f"  ✗ ERROR: {error}")
    else:
        print(f"  ✓ Valid")
    
    all_warnings.extend(warnings)
    for warning in warnings:
        print(f"  ⚠ WARNING: {warning}")
    
    # 3. Validar métricas
    print("\n[3/4] Validating metrics...")
    is_valid, errors, warnings = validate_metrics_config(cfg.metrics)
    
    if not is_valid:
        all_valid = False
        all_errors.extend(errors)
        for error in errors:
            print(f"  ✗ ERROR: {error}")
    else:
        print(f"  ✓ Valid ({len(cfg.metrics.segmentation_metrics)} metrics)")
    
    all_warnings.extend(warnings)
    for warning in warnings:
        print(f"  ⚠ WARNING: {warning}")
    
    # 4. Validar output
    print("\n[4/4] Validating output...")
    is_valid, errors, warnings = validate_output_config(cfg.output)
    
    if not is_valid:
        all_valid = False
        all_errors.extend(errors)
        for error in errors:
            print(f"  ✗ ERROR: {error}")
    else:
        print(f"  ✓ Valid")
    
    all_warnings.extend(warnings)
    for warning in warnings:
        print(f"  ⚠ WARNING: {warning}")
    
    # Resumo
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if all_valid:
        print("\n✓ Configuration is VALID!")
        if all_warnings:
            print(f"\n⚠ {len(all_warnings)} warning(s) found (see above)")
        print("\nYou can now run:")
        print("  python evaluate_experiments.py")
        return 0
    else:
        print(f"\n✗ Configuration is INVALID!")
        print(f"\n{len(all_errors)} error(s) found:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        
        if all_warnings:
            print(f"\n{len(all_warnings)} warning(s) found:")
            for i, warning in enumerate(all_warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\nPlease fix the errors and run validation again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())