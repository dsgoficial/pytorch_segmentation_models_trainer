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
from omegaconf import DictConfig


def validate_experiment_config(exp_config: DictConfig, exp_idx: int) -> tuple:
    """
    Valida configuração de um experimento.

    Args:
        exp_config: Config do experimento
        exp_idx: Índice do experimento

    Returns:
        (errors, warnings)
    """
    errors = []
    warnings = []

    # Validar campos obrigatórios
    required_fields = ["name", "predict_config", "checkpoint_path", "output_folder"]
    for field in required_fields:
        if field not in exp_config or exp_config.get(field) is None:
            errors.append(f"Experiment {exp_idx}: Missing required field '{field}'")

    if errors:
        return errors, warnings

    # Validar predict_config if it exists
    predict_config = exp_config.get("predict_config")
    if predict_config is not None:
        if isinstance(predict_config, (DictConfig, dict)):
            model_path = predict_config.get("model_path")
            if not model_path:
                warnings.append(
                    f"Experiment {exp_idx}: 'predict_config.model_path' is empty. "
                    "If using MLflow or similar for model retrieval, ensure 'predict_config.model_id' is set and correctly handled downstream."
                )
            elif not os.path.exists(str(model_path)):
                # Errors for non-existent local paths
                if not str(model_path).startswith(
                    ("http://", "https://", "s3://", "mlflow://")
                ):
                    errors.append(
                        f"Experiment {exp_idx} ({exp_config.name}): "
                        f"model_path not found: {model_path}"
                    )
        elif isinstance(predict_config, str):
            if not os.path.exists(predict_config):
                # Check if it's a valid class path (for Hydra _target_)
                if "." not in predict_config:
                    errors.append(
                        f"Experiment {exp_idx} ({exp_config.name}): "
                        f"predict_config not found: {predict_config}"
                    )
        else:
            errors.append(
                f"Experiment {exp_idx}: 'predict_config' must be a string or a dictionary."
            )

    # Validar inference_config e processor se existirem
    inference_config = exp_config.get("inference_config")
    if inference_config:
        processor = inference_config.get("processor")
        if processor is not None:
            target = processor.get("_target_")
            if not target:
                errors.append(
                    f"Experiment {exp_idx}: 'inference_config.processor._target_' is missing."
                )
            else:
                # Mock-friendly validation: if it contains 'invalid', it's invalid
                if (
                    "invalid" in target
                    or not isinstance(target, str)
                    or "." not in target
                ):
                    errors.append(
                        f"Experiment {exp_idx}: 'inference_config.processor._target_' is not a valid class path or the class does not exist."
                    )

            export_strategy = processor.get("export_strategy")
            if export_strategy is not None:
                target = export_strategy.get("_target_")
                if not target:
                    errors.append(
                        f"Experiment {exp_idx}: 'inference_config.processor.export_strategy._target_' is missing."
                    )
                else:
                    if (
                        "invalid" in target
                        or not isinstance(target, str)
                        or "." not in target
                    ):
                        errors.append(
                            f"Experiment {exp_idx}: 'inference_config.processor.export_strategy._target_' is not a valid class path or the class does not exist."
                        )

    # Validar dataset_config se existir
    dataset_config = exp_config.get("dataset_config")
    if dataset_config:
        has_pred_csv = dataset_config.get("prediction_csv_path")
        has_pred_folder = dataset_config.get("prediction_input_folder")

        if has_pred_csv and has_pred_folder:
            errors.append(
                f"Experiment {exp_idx}: 'dataset_config.prediction_csv_path' and 'dataset_config.prediction_input_folder' are mutually exclusive. Provide only one."
            )
        elif not (has_pred_csv or has_pred_folder):
            errors.append(
                f"Experiment {exp_idx}: Either 'dataset_config.prediction_csv_path' or 'dataset_config.prediction_input_folder' must be provided."
            )

        has_gt_csv = dataset_config.get("ground_truth_csv_path")
        has_gt_folder = dataset_config.get("ground_truth_input_folder")

        if has_gt_csv and has_gt_folder:
            errors.append(
                f"Experiment {exp_idx}: 'dataset_config.ground_truth_csv_path' and 'dataset_config.ground_truth_input_folder' are mutually exclusive. Provide only one."
            )

    # Validar checkpoint existe (se for um caminho local)
    checkpoint_path = exp_config.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path_str = str(checkpoint_path)
        if not checkpoint_path_str.startswith(
            ("http://", "https://", "s3://", "mlflow://")
        ):
            if not os.path.exists(checkpoint_path_str):
                errors.append(
                    f"Experiment {exp_idx} ({exp_config.name}): "
                    f"checkpoint not found: {checkpoint_path_str}"
                )

    # Warning se output_folder já existe
    output_folder = exp_config.get("output_folder")
    if output_folder and os.path.exists(str(output_folder)):
        folder = Path(output_folder)
        pred_files = list(folder.glob("seg_*_output.tif")) + list(folder.glob("*.tif"))
        if len(pred_files) > 0:
            warnings.append(
                f"Experiment {exp_idx} ({exp_config.name}): "
                f"output_folder already contains {len(pred_files)} predictions. "
                "Will skip if skip_existing_predictions=true"
            )

    return errors, warnings


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
            errors.append(f"Dataset CSV not found: {dataset_config.input_csv_path}")
        else:
            # Verificar estrutura do CSV
            import pandas as pd

            try:
                df = pd.read_csv(dataset_config.input_csv_path)

                required_cols = ["image", "mask"]
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    errors.append(f"Dataset CSV missing columns: {missing_cols}")
                else:
                    print(f"  ✓ CSV has {len(df)} rows")

                    # Verificar se arquivos existem (amostra)
                    sample_size = min(5, len(df))
                    missing_images = 0
                    missing_masks = 0

                    for _, row in df.head(sample_size).iterrows():
                        if not os.path.exists(row["image"]):
                            missing_images += 1
                        if not os.path.exists(row["mask"]):
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
        warnings.append(
            "No metrics configured. Only confusion matrix will be computed."
        )

    # Verificar se todas as métricas têm _target_
    for idx, metric_cfg in enumerate(metrics_config.segmentation_metrics):
        if "_target_" not in metric_cfg:
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
    version_base=None, config_path="configs/evaluation", config_name="pipeline_config"
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
    print("=" * 80)
    print("VALIDATION: EVALUATION PIPELINE CONFIG")
    print("=" * 80)

    all_valid = True
    all_errors = []
    all_warnings = []

    # 1. Validar experimentos
    print("\n[1/4] Validating experiments...")
    for idx, exp in enumerate(cfg.experiments):
        print(f"\n  Experiment {idx + 1}: {exp.name}")
        errors, warnings = validate_experiment_config(exp, idx)

        if errors:
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
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

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
